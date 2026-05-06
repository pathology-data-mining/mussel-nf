"""Configuration dataclasses and secrets/model-type helpers for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("mussel-dispatcher")


@dataclass
class WatcherConfig:
    type: str                          # "local", "s3", "tcga", or "databricks"
    # local
    path: Optional[str] = None
    recursive: bool = True
    stability_wait_seconds: int = 30
    min_file_size_mb: float = 10.0
    # s3
    bucket: Optional[str] = None
    prefix: str = ""
    min_file_size_bytes: int = 10_000_000
    aws_profile: Optional[str] = None
    endpoint_url: Optional[str] = None
    # shared
    poll_interval_seconds: int = 60
    extensions: list = field(default_factory=lambda: [".svs", ".tiff", ".tif", ".ndpi", ".scn"])
    # tcga watcher
    inventory_csv: str = ""
    status_csv: str = ""
    models: list = field(default_factory=list)
    local_slides_dir: str = ""
    s3_base: str = ""
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    project: str = ""
    slide_type: str = "DX"
    sample_type: str = "Primary Tumor"
    gdc_token_file: str = ""
    gdc_max_age_hours: float = 24.0
    download_enabled: bool = False  # if True, dispatcher downloads slides locally before NF run
    download_dir: str = ""          # local directory for downloaded slides (required if download_enabled)
    scripts_dir: str = ""
    wds_destinations: dict = field(default_factory=dict)
    wds_patch_encoders: dict = field(default_factory=dict)  # slide_model → patch_encoder_model
    wds_staging_dir: str = ""
    wds_s3_max_concurrency: int = 4
    secrets_env_file: str = ""
    nextflow_secrets: list = field(default_factory=list)
    databricks_volume_folder: str = ""
    databricks_volume_path: str = ""
    databricks_table: str = ""
    databricks_job_id: str = ""
    # databricks watcher
    warehouse_id: str = ""
    source_filter: list = field(default_factory=list)
    additional_where: str = ""


@dataclass
class Config:
    # Required
    nextflow_profiles: str
    outdir: str

    repo_dir: str = ""
    work_base_dir: str = ""
    dispatch_dir: str = ""
    state_dir: str = ""
    log_dir: str = ""

    max_concurrent_runs: int = 2
    batch_size: int = 20
    min_batch_size: int = 1
    max_wait_seconds: int = 300
    retry_failed: bool = True
    max_slide_retries: int = 5
    cleanup_work_dir: bool = False
    cleanup_downloads: bool = False
    cleanup_batch_csv: bool = False
    cleanup_logs_after_days: int = 0
    cleanup_results: bool = False
    nextflow_config: str = ""
    nextflow_params_file: str = ""
    nextflow_version: str = ""
    combined_manifest_path: Optional[str] = None
    post_batch_hooks: list = field(default_factory=list)
    watchers: list = field(default_factory=list)

    def resolved_combined_manifest_path(self) -> str:
        return self.combined_manifest_path or os.path.join(self.outdir, "manifest-combined.csv")

    @classmethod
    def load(cls, path: str) -> "Config":
        config_dir = os.path.dirname(os.path.abspath(path))

        with open(path) as f:
            raw = yaml.safe_load(f)

        def _resolve(key: str, default: str) -> str:
            val = raw.get(key, "") or default
            return val if os.path.isabs(val) else os.path.join(config_dir, val)

        raw["repo_dir"]      = _resolve("repo_dir",      "..")
        raw["work_base_dir"] = _resolve("work_base_dir", "work")
        raw["dispatch_dir"]  = _resolve("dispatch_dir",  "batches")
        raw["state_dir"]     = _resolve("state_dir",     "state")
        raw["log_dir"]       = _resolve("log_dir",       "logs")

        for key in ("outdir", "nextflow_config", "nextflow_params_file"):
            val = raw.get(key, "")
            if val and not os.path.isabs(val):
                raw[key] = os.path.join(config_dir, val)

        _WATCHER_PATH_FIELDS = (
            "path", "inventory_csv", "status_csv", "local_slides_dir",
            "wds_staging_dir", "scripts_dir", "gdc_token_file", "secrets_env_file",
        )

        def _resolve_watcher_path(val: str) -> str:
            return val if (not val or os.path.isabs(val)) else os.path.join(config_dir, val)

        watcher_cfgs = []
        for w in raw.pop("watchers", []):
            for f in _WATCHER_PATH_FIELDS:
                if f in w:
                    w[f] = _resolve_watcher_path(w[f])
            watcher_cfgs.append(WatcherConfig(**w))

        raw["watchers"] = watcher_cfgs
        cfg = cls(**raw)

        for w in cfg.watchers:
            if w.secrets_env_file and os.path.isfile(w.secrets_env_file):
                _load_secrets_env(w.secrets_env_file, w)
            if w.nextflow_secrets:
                _load_nf_secrets(w.nextflow_secrets, w)

        nf_models: list[str] | None = None
        for w in cfg.watchers:
            if not w.models:
                if nf_models is None:
                    nf_models = _read_nf_model_types(cfg.repo_dir)
                    if nf_models:
                        log.info("Auto-detected model_types from nextflow.config: %s",
                                 ", ".join(nf_models))
                w.models = nf_models

        cfg.post_batch_hooks = cfg._build_auto_hooks() + cfg.post_batch_hooks
        return cfg

    def _build_auto_hooks(self) -> list:
        """Generate WDS-append and Databricks-sync post-batch hooks from watcher config."""
        hooks = []
        db_hooks = []
        for w in self.watchers:
            if w.type == "databricks" and w.wds_destinations:
                for model, dest in w.wds_destinations.items():
                    args = [
                        "--pt-dir={outdir}/features/" + model,
                        "--h5-dir={outdir}/tiles",
                        "--wds-dest=" + dest,
                        "--model-type=" + model,
                        "--slide-ids-csv={batch_csv}",
                        "--project-id-column=oncotree_code",
                        "--manifest-csv={outdir}/wds_manifest.csv",
                    ]
                    if w.wds_staging_dir:
                        args.append("--staging-dir=" + w.wds_staging_dir)
                    if w.wds_s3_max_concurrency != 4:
                        args.append(f"--s3-max-concurrency={w.wds_s3_max_concurrency}")
                    if self.cleanup_results:
                        args.append("--delete-local")
                        patch_enc = w.wds_patch_encoders.get(model)
                        if patch_enc:
                            args.append("--also-delete-pt-dirs={outdir}/features/" + patch_enc)
                    hooks.append({"command": "python -m mussel_dispatcher.tcga.append_wds", "args": args})

            if w.type != "tcga":
                continue

            if w.wds_destinations:
                for model, dest in w.wds_destinations.items():
                    args = [
                        "--pt-dir={outdir}/features/" + model,
                        "--h5-dir={outdir}/tiles",
                        "--inventory=" + w.inventory_csv,
                        "--wds-dest=" + dest,
                        "--model-type=" + model,
                        "--slide-ids-csv={batch_csv}",
                        "--manifest-csv={outdir}/wds_manifest.csv",
                        "--status-csv=" + w.status_csv,
                    ]
                    if w.wds_staging_dir:
                        args.append("--staging-dir=" + w.wds_staging_dir)
                    if w.wds_s3_max_concurrency != 4:
                        args.append(f"--s3-max-concurrency={w.wds_s3_max_concurrency}")
                    if w.s3_endpoint:
                        args.append("--s3-endpoint=" + w.s3_endpoint)
                    if self.cleanup_results:
                        args.append("--delete-local")
                        patch_enc = w.wds_patch_encoders.get(model)
                        if patch_enc:
                            args.append("--also-delete-pt-dirs={outdir}/features/" + patch_enc)
                    hooks.append({"command": "python -m mussel_dispatcher.tcga.append_wds", "args": args})

            if w.databricks_volume_folder or w.databricks_volume_path:
                args = ["--inventory=" + w.inventory_csv, "--status=" + w.status_csv]
                if w.databricks_volume_folder:
                    args.append("--volume-folder=" + w.databricks_volume_folder)
                else:
                    args.append("--volume-path=" + w.databricks_volume_path)
                if w.databricks_table:
                    args.append("--table=" + w.databricks_table)
                if w.databricks_job_id:
                    args.append("--job-id=" + w.databricks_job_id)
                db_hooks.append({
                    "command": "python -m mussel_dispatcher.tcga.sync_databricks",
                    "args": args,
                })

        return hooks + db_hooks


def _load_secrets_env(path: str, watcher: WatcherConfig) -> None:
    """Parse a shell env file (KEY=value) and populate watcher S3 credentials."""
    _KEY_MAP = {
        "ECS_ACCESS_KEY": "s3_access_key", "AWS_ACCESS_KEY_ID": "s3_access_key",
        "ECS_SECRET_KEY": "s3_secret_key", "AWS_SECRET_ACCESS_KEY": "s3_secret_key",
    }
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip().lstrip("export").strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                attr = _KEY_MAP.get(k.strip())
                if attr and not getattr(watcher, attr):
                    setattr(watcher, attr, v.strip().strip('"').strip("'"))
        log.debug("Loaded secrets_env_file: %s", path)
    except OSError as exc:
        log.warning("Could not read secrets_env_file %s: %s", path, exc)


def _load_nf_secrets(secret_names: list[str], watcher: WatcherConfig) -> None:
    """Resolve Nextflow secrets by name and populate watcher S3 credentials."""
    _KEY_MAP = {
        "ECS_ACCESS_KEY": "s3_access_key", "AWS_ACCESS_KEY_ID": "s3_access_key",
        "ECS_SECRET_KEY": "s3_secret_key", "AWS_SECRET_ACCESS_KEY": "s3_secret_key",
    }
    for name in secret_names:
        attr = _KEY_MAP.get(name)
        if not attr or getattr(watcher, attr):
            continue
        try:
            result = subprocess.run(
                ["nextflow", "secrets", "get", name],
                capture_output=True, text=True, timeout=15,
            )
            value = result.stdout.strip()
            if result.returncode != 0 or not value:
                log.warning("nextflow secrets get %s failed: %s", name,
                            result.stderr.strip() or "empty output")
                continue
            setattr(watcher, attr, value)
            log.debug("Loaded %s from Nextflow secrets → %s", name, attr)
        except Exception as exc:
            log.warning("Could not load Nextflow secret %s: %s", name, exc)


def _read_nf_model_types(repo_dir: str) -> list[str]:
    """Parse model_types list from nextflow.config."""
    import re
    config_path = os.path.join(repo_dir, "nextflow.config")
    try:
        text = Path(config_path).read_text()
    except OSError:
        log.warning("Could not read %s — model_types not auto-detected", config_path)
        return []
    m = re.search(r"model_types\s*=\s*\[([^\]]*)\]", text)
    if not m:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
