"""Configuration dataclasses and secrets/model-type helpers for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
import re
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
    wds_destination: str = ""
    wds_destinations: dict = field(default_factory=dict)
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
    query: str = ""        # inline SQL (required unless query_file is set)
    query_file: str = ""   # path to a .sql file (alternative to inline query)


@dataclass
class StoragePolicyConfig:
    """Execution-storage cleanup and dispatch-capacity policy.

    A zero threshold disables that particular capacity check.  Paths are
    supplied by the local scheduler or remote target because those backends
    do not share a filesystem.
    """

    min_free_percent: float = 0.0
    min_free_gib: float = 0.0
    resume_free_percent: float = 0.0
    resume_free_gib: float = 0.0
    cleanup_interval_seconds: int = 3600
    cleanup_batch_size: int = 50
    cleanup_budget_seconds: float = 15.0


@dataclass
class RemoteTargetConfig:
    """Configuration for a non-shared remote Slurm execution target."""

    name: str
    backend: str = "remote_agent"
    dispatcher_host: str = ""
    login_host: str = ""
    remote_scratch_root: str = ""
    remote_repo_dir: str = ""
    remote_outdir: str = ""
    nextflow_profiles: str = ""
    nextflow_config: str = ""
    nextflow_params_file: str = ""
    nextflow_version: str = ""
    nextflow_cli_args: list = field(default_factory=list)
    # Target-specific, non-semantic tuning. Scientific parameters are loaded
    # from the canonical params file and materialized into each batch.
    parameter_overrides: dict = field(default_factory=dict)
    profile_name: str = ""
    release_stage_root: str = ""
    active_release_link: str = ""
    release_restart_units: list = field(default_factory=list)
    tower_endpoint: str = ""
    conda_env_path: str = ""
    driver_partition: str = "cpu"
    driver_time: str = "7-00:00:00"
    driver_cpus: int = 2
    driver_mem: str = "8G"
    driver_account: str = ""
    driver_qos: str = ""
    hook_partition: str = "cpu"
    hook_time: str = "2:00:00"
    hook_cpus: int = 2
    hook_mem: str = "16G"
    hook_account: str = ""
    hook_qos: str = ""
    hook_exclude: str = ""
    hook_use_slurm: bool = False
    remote_hook_path_replacements: dict = field(default_factory=dict)
    gpu_partition: str = "gpu"
    cpu_partition: str = "cpu"
    preempt_gpu_partition: str = "gpu_preempt"
    task_scratch_subdir: str = "task_scratch"
    poll_interval_seconds: int = 60
    idle_poll_max_interval_seconds: int = 300
    agent_summary_interval_seconds: int = 60
    finalization_poll_interval_seconds: int = 60
    finalization_lease_ttl_seconds: int = 300
    # A remote target may have a different safe post-success throughput than
    # the local dispatcher.  Zero inherits Config.max_concurrent_finalizations.
    max_concurrent_finalizations: int = 0
    storage_policy: StoragePolicyConfig = field(default_factory=StoragePolicyConfig)
    stale_timeout_seconds: int = 1800
    agent_heartbeat_timeout_seconds: int = 300
    secrets_env_file: str = ""
    env: dict = field(default_factory=dict)


@dataclass
class ExecutionPoolConfig:
    name: str
    backend: str = "local"
    target: str = ""
    max_concurrent_runs: int = 1


@dataclass
class PostgresConfig:
    host: str = ""
    port: int = 5432
    database: str = ""
    user: str = ""
    password_env: str = ""
    sslmode: str = "prefer"


@dataclass
class Config:
    # Required
    nextflow_profiles: str
    outdir: str

    config_path: str = ""
    repo_dir: str = ""
    work_base_dir: str = ""
    dispatch_dir: str = ""
    state_dir: str = ""
    log_dir: str = ""
    state_backend: str = "sqlite"

    max_concurrent_runs: int = 2
    # Post-success hooks have a separate lane from Nextflow execution. Keep
    # the compatibility default conservative; production configs can raise it
    # when WDS/DB locks and host resources permit.
    max_concurrent_finalizations: int = 1
    # Optional lane-specific overrides.  Target-level overrides take
    # precedence for remote lanes; the top-level value remains the fallback.
    finalization_lane_limits: dict = field(default_factory=dict)
    batch_size: int = 20
    batch_sizes: list = field(default_factory=list)  # if set, rotate through these sizes for natural stagger
    min_batch_size: int = 1
    max_wait_seconds: int = 300
    retry_failed: bool = True
    max_slide_retries: int = 5
    stuck_batch_timeout_hours: float = 4.0  # kill NF if log file silent for this long (0 = disabled)
    cleanup_work_dir: bool = False
    cleanup_downloads: bool = False
    cleanup_batch_csv: bool = False
    cleanup_logs_after_days: int = 0
    cleanup_results: bool = False
    # Shared run guard. By default this is derived from WDS destinations and
    # prevents two dispatcher configs from processing the same output cohort.
    cohort_lock_name: str = ""
    cohort_lock_dir: str = ""
    allow_shared_cohort: bool = False
    nextflow_config: str = ""
    nextflow_params_file: str = ""
    nextflow_version: str = ""
    # Local execution profile tuning. Remote targets use their own field.
    parameter_overrides: dict = field(default_factory=dict)
    profile_name: str = ""
    tower_endpoint: str = ""           # if set, pass -with-tower to NF (e.g. http://localhost:8050)
    dashboard_port: int = 8050         # port for the dashboard HTTP server
    tower_proxy_port: int = 8049       # port for the persistent Tower proxy (sits in front of dashboard)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    # Top-level S3/ECS credentials — inherited by all watchers that don't set their own.
    # Prefer nextflow_secrets (portable); secrets_env_file and inline keys are fallbacks.
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    secrets_env_file: str = ""
    nextflow_secrets: list = field(default_factory=list)
    combined_manifest_path: Optional[str] = None
    manifest_path: Optional[str] = None     # CSV with image_ids; slides NOT in it get priority=1
    manifest_id_column: int = 0             # 0-based column index containing the image_id
    post_batch_hooks: list = field(default_factory=list)
    watchers: list = field(default_factory=list)
    execution_backend: str = "local"
    default_dispatch_target: str = ""
    dispatch_targets: list = field(default_factory=list)
    execution_pools: list = field(default_factory=list)
    pool_selection_policy: str = "balanced"
    finalization_lease_ttl_seconds: int = 300
    finalization_heartbeat_interval_seconds: int = 60
    storage_policy: StoragePolicyConfig = field(default_factory=StoragePolicyConfig)
    # Shared post-batch hook safety controls.  Failures are classified before
    # retrying so a release/config storm pauses the lane instead of multiplying
    # identical Slurm jobs.
    hook_circuit_breaker_threshold: int = 3
    hook_circuit_breaker_window_seconds: int = 300
    hook_max_infra_retries: int = 3
    # After this many attempts of a required WDS hook (any failure category),
    # finalization stops waiting for it: coverage is reconciled from the WDS
    # manifest (missing slides requeued, bounded by max_slide_retries) and the
    # batch moves on. 0 disables the cap (category-based limits still apply).
    wds_hook_max_attempts: int = 6
    hook_preflight: bool = True
    # Optional PostgreSQL-coordinated release synchronization.  The local
    # tracked-clean checkout is authoritative; remote pools stay gated until
    # their release agent and running worker both report this revision.
    auto_sync_release: bool = False
    release_deployment_id: str = "default"
    release_sync_interval_seconds: int = 30
    release_origin_remote: str = "origin"
    release_require_origin: bool = True
    release_activation_timeout_seconds: int = 600
    release_auto_rollback: bool = True
    hook_executor_reconcile_interval_seconds: int = 60
    hook_executor_missing_grace_seconds: int = 900
    hook_executor_missing_polls: int = 3
    finalization_reconcile_interval_seconds: int = 60
    finalization_reconcile_failure_threshold: int = 3

    @property
    def postgres_host(self) -> str:
        return self.postgres.host

    @property
    def postgres_port(self) -> int:
        return self.postgres.port

    @property
    def postgres_database(self) -> str:
        return self.postgres.database

    @property
    def postgres_user(self) -> str:
        return self.postgres.user

    @property
    def postgres_password_env(self) -> str:
        return self.postgres.password_env

    @property
    def postgres_sslmode(self) -> str:
        return self.postgres.sslmode

    def state_db_path(self) -> str:
        return os.path.join(self.state_dir, "dispatcher.db")

    def resolved_combined_manifest_path(self) -> str:
        return self.combined_manifest_path or os.path.join(self.outdir, "manifest-combined.csv")

    def effective_publish_mode(self) -> tuple[str | None, str]:
        """Return the active Nextflow publish mode and the source used.

        Precedence mirrors how dispatcher-launched runs are configured:
        1. nextflow params file (`publish_mode: ...`)
        2. extra nextflow config (`publish_mode = ...` or `params.publish_mode = ...`)
        3. repo nextflow.config
        """
        if self.nextflow_params_file:
            mode = _read_nf_publish_mode_from_params(self.nextflow_params_file)
            if mode:
                return mode, self.nextflow_params_file

        if self.nextflow_config:
            mode = _read_nf_publish_mode_from_config(self.nextflow_config)
            if mode:
                return mode, self.nextflow_config

        repo_cfg = os.path.join(self.repo_dir, "nextflow.config")
        mode = _read_nf_publish_mode_from_config(repo_cfg)
        return mode, repo_cfg

    def publish_mode_safety_warnings(self) -> list[str]:
        mode, source = self.effective_publish_mode()
        if mode != "symlink":
            return []

        warnings: list[str] = []
        if self.cleanup_work_dir:
            warnings.append(
                "Unsafe publish-mode combination: publish_mode=symlink "
                f"(from {source}) with cleanup_work_dir=true will leave "
                "published outputs under outdir as broken symlinks after batch cleanup."
            )
        if self.cleanup_results:
            warnings.append(
                "Unsafe publish-mode combination: publish_mode=symlink "
                f"(from {source}) with cleanup_results=true will remove the "
                "published local feature/coord entries after WDS append."
            )
        return warnings

    @classmethod
    def load(cls, path: str) -> "Config":
        config_dir = os.path.dirname(os.path.abspath(path))

        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = raw or {}

        def _resolve(key: str, default: str) -> str:
            val = raw.get(key, "") or default
            return val if os.path.isabs(val) else os.path.join(config_dir, val)

        raw["config_path"]   = os.path.abspath(path)
        raw["repo_dir"]      = _resolve("repo_dir",      "..")
        raw["work_base_dir"] = _resolve("work_base_dir", "work")
        raw["dispatch_dir"]  = _resolve("dispatch_dir",  "batches")
        raw["state_dir"]     = _resolve("state_dir",     "state")
        raw["log_dir"]       = _resolve("log_dir",       "logs")
        if raw.get("cohort_lock_dir"):
            raw["cohort_lock_dir"] = _resolve("cohort_lock_dir", raw["cohort_lock_dir"])

        for key in ("outdir", "nextflow_config", "nextflow_params_file", "secrets_env_file"):
            val = raw.get(key, "")
            if val and not os.path.isabs(val):
                raw[key] = os.path.join(config_dir, val)

        _WATCHER_PATH_FIELDS = (
            "path", "inventory_csv", "status_csv", "local_slides_dir",
            "wds_staging_dir", "scripts_dir", "gdc_token_file", "secrets_env_file",
            "query_file",
        )

        def _resolve_watcher_path(val: str) -> str:
            return val if (not val or os.path.isabs(val)) else os.path.join(config_dir, val)

        target_cfgs = []
        for t in raw.pop("dispatch_targets", []):
            retired = {
                key: t.pop(key)
                for key in (
                    "transfer_host", "queue_dir", "status_dir",
                    "log_mirror_dir", "remote_shared_root",
                )
                if key in t
            }
            if retired:
                log.warning(
                    "Ignoring retired remote transport settings for target %s: %s",
                    t.get("name", ""), ", ".join(sorted(retired)),
                )
            for f in (
                "remote_scratch_root", "remote_repo_dir", "remote_outdir",
                "nextflow_config", "nextflow_params_file", "secrets_env_file",
            ):
                if f in t and t[f] and not os.path.isabs(t[f]) and f.startswith(("nextflow", "secrets")):
                    t[f] = os.path.join(config_dir, t[f])
            t["storage_policy"] = StoragePolicyConfig(**(t.get("storage_policy") or {}))
            target_cfgs.append(RemoteTargetConfig(**t))
        raw["dispatch_targets"] = target_cfgs

        watcher_cfgs = []
        for w in raw.pop("watchers", []):
            for f in _WATCHER_PATH_FIELDS:
                if f in w:
                    w[f] = _resolve_watcher_path(w[f])
            watcher_cfgs.append(WatcherConfig(**w))

        raw["watchers"] = watcher_cfgs
        raw["execution_pools"] = [ExecutionPoolConfig(**p) for p in raw.get("execution_pools", [])]
        raw["postgres"] = PostgresConfig(**(raw.get("postgres") or {}))
        raw["storage_policy"] = StoragePolicyConfig(**(raw.get("storage_policy") or {}))
        cfg = cls(**raw)

        if cfg.execution_backend != "local" and not cfg.default_dispatch_target and cfg.dispatch_targets:
            cfg.default_dispatch_target = cfg.dispatch_targets[0].name

        # Auto-derive watcher defaults that do not depend on model discovery.
        for w in cfg.watchers:
            # Auto-derive volume folder from table name so only databricks_table is required.
            if not w.databricks_volume_folder and not w.databricks_volume_path and w.databricks_table:
                parts = w.databricks_table.split(".")
                if len(parts) == 3:
                    catalog, schema, tbl = parts
                    w.databricks_volume_folder = f"/Volumes/{catalog}/{schema}/{tbl}_sync"
                    log.debug("Auto-derived databricks_volume_folder: %s", w.databricks_volume_folder)
            # Auto-detect SQL warehouse when a Databricks watcher has no warehouse_id.
            if w.type == "databricks" and not w.warehouse_id:
                from mussel_dispatcher.databricks_sync import resolve_warehouse_id
                w.warehouse_id = resolve_warehouse_id()
                if w.warehouse_id:
                    log.info("Auto-detected warehouse_id: %s", w.warehouse_id)
                else:
                    log.warning("Could not auto-detect warehouse_id; set it explicitly in the config")

        # Resolve top-level S3 credentials (secrets_env_file / nextflow_secrets).
        if cfg.secrets_env_file and os.path.isfile(cfg.secrets_env_file):
            _load_secrets_env(cfg.secrets_env_file, cfg)  # type: ignore[arg-type]
        if cfg.nextflow_secrets:
            _load_nf_secrets(cfg.nextflow_secrets, cfg)  # type: ignore[arg-type]

        # Propagate top-level S3 config to watchers that don't have their own.
        for w in cfg.watchers:
            if not w.s3_endpoint and cfg.s3_endpoint:
                w.s3_endpoint = cfg.s3_endpoint
            if not w.s3_access_key and cfg.s3_access_key:
                w.s3_access_key = cfg.s3_access_key
            if not w.s3_secret_key and cfg.s3_secret_key:
                w.s3_secret_key = cfg.s3_secret_key

        for w in cfg.watchers:
            if w.secrets_env_file and os.path.isfile(w.secrets_env_file):
                _load_secrets_env(w.secrets_env_file, w)
            if w.nextflow_secrets:
                _load_nf_secrets(w.nextflow_secrets, w)

        for target in cfg.dispatch_targets:
            if cfg.s3_access_key:
                target.env.setdefault("ECS_ACCESS_KEY", cfg.s3_access_key)
                target.env.setdefault("AWS_ACCESS_KEY_ID", cfg.s3_access_key)
            if cfg.s3_secret_key:
                target.env.setdefault("ECS_SECRET_KEY", cfg.s3_secret_key)
                target.env.setdefault("AWS_SECRET_ACCESS_KEY", cfg.s3_secret_key)
            if cfg.s3_endpoint:
                target.env.setdefault("ECS_ENDPOINT_URL", cfg.s3_endpoint)
                target.env.setdefault("S3_ENDPOINT_URL", cfg.s3_endpoint)
            if "HF_TOKEN" not in target.env:
                try:
                    result = subprocess.run(
                        ["nextflow", "secrets", "get", "HF_TOKEN"],
                        capture_output=True, text=True, timeout=15,
                    )
                    value = result.stdout.strip()
                    if result.returncode == 0 and value:
                        target.env["HF_TOKEN"] = value
                except Exception:
                    pass

        nf_models: list[str] | None = None
        for w in cfg.watchers:
            if not w.models:
                if nf_models is None:
                    nf_models = _read_nf_model_types(
                        cfg.repo_dir, nextflow_params_file=cfg.nextflow_params_file
                    )
                    if nf_models:
                        source = cfg.nextflow_params_file or os.path.join(cfg.repo_dir, "nextflow.config")
                        log.info("Auto-detected model_types from %s: %s",
                                 source,
                                 ", ".join(nf_models))
                w.models = nf_models
            _apply_wds_destination_defaults(w)
            if not w.wds_staging_dir and (w.wds_destination or w.wds_destinations):
                w.wds_staging_dir = os.path.join(cfg.state_dir, "wds-staging")

        cfg.post_batch_hooks = cfg._build_auto_hooks() + cfg.post_batch_hooks
        return cfg

    def get_dispatch_target(self, name: str | None = None) -> RemoteTargetConfig | None:
        target_name = name or self.default_dispatch_target
        if not target_name and len(self.dispatch_targets) == 1:
            return self.dispatch_targets[0]
        for target in self.dispatch_targets:
            if target.name == target_name:
                return target
        return None

    def effective_max_concurrent_runs(self) -> int:
        pools = self.resolved_execution_pools()
        if pools:
            return sum(max(0, int(pool.max_concurrent_runs or 0)) for pool in pools)
        return max(0, int(self.max_concurrent_runs or 0))

    def effective_max_concurrent_finalizations(
        self, *, lane: str = "local", target: str | None = None
    ) -> int:
        """Return the configured finalization limit for one execution lane."""
        if lane == "remote" and target:
            target_cfg = self.get_dispatch_target(target)
            target_limit = int(
                getattr(target_cfg, "max_concurrent_finalizations", 0) or 0
            ) if target_cfg is not None else 0
            if target_limit > 0:
                return target_limit
        lane_limit = (self.finalization_lane_limits or {}).get(lane)
        if lane_limit is not None and int(lane_limit or 0) > 0:
            return int(lane_limit)
        return max(1, int(self.max_concurrent_finalizations or 1))

    def resolved_execution_pools(self) -> list[ExecutionPoolConfig]:
        if self.execution_pools:
            return [ExecutionPoolConfig(**vars(pool)) if isinstance(pool, ExecutionPoolConfig) else ExecutionPoolConfig(**pool)
                    for pool in self.execution_pools]

        backend = (self.execution_backend or "local").strip() or "local"
        target = self.default_dispatch_target if backend == "remote_agent" else ""
        if backend == "remote_agent" and not target and self.dispatch_targets:
            target = self.dispatch_targets[0].name
        pool_name = target or ("local" if backend == "local" else backend)
        return [ExecutionPoolConfig(
            name=pool_name,
            backend=backend,
            target=target,
            max_concurrent_runs=max(0, int(self.max_concurrent_runs or 0)),
        )]

    def get_execution_pool(self, name: str | None = None) -> ExecutionPoolConfig | None:
        pool_name = name
        pools = self.resolved_execution_pools()
        if not pool_name and len(pools) == 1:
            return pools[0]
        for pool in pools:
            if pool.name == pool_name:
                return pool
        return None

    def infer_batch_pool_name(self, backend: str | None = None, target: str | None = None) -> str | None:
        backend_name = (backend or "local").strip() or "local"
        target_name = (target or "").strip()
        for pool in self.resolved_execution_pools():
            if pool.backend != backend_name:
                continue
            if backend_name == "remote_agent" and pool.target != target_name:
                continue
            return pool.name
        if backend_name == "remote_agent":
            return target_name or self.default_dispatch_target or "remote_agent"
        return "local"

    def wds_lock_args(self) -> list[str]:
        if (self.state_backend or "sqlite").strip().lower() == "postgres":
            return [
                "--lock-backend=postgres",
                "--lock-postgres-host=" + self.postgres_host,
                f"--lock-postgres-port={self.postgres_port}",
                "--lock-postgres-database=" + self.postgres_database,
                "--lock-postgres-user=" + self.postgres_user,
                "--lock-postgres-password-env=" + self.postgres_password_env,
                "--lock-postgres-sslmode=" + self.postgres_sslmode,
            ]
        return ["--lock-dir=" + self.resolved_cohort_lock_dir()]

    def cohort_lock_keys(self) -> list[str]:
        """Return normalized output-cohort keys this dispatcher mutates.

        Keys are based on WDS destination and model, because those define the
        shared output namespace. Multiple configs targeting the same key must
        not run concurrently unless explicitly allowed.
        """
        keys: set[str] = set()
        for w in self.watchers:
            for model, dest in (w.wds_destinations or {}).items():
                if dest and model:
                    keys.add(f"{dest.rstrip('/')}::{model}")
        if self.cohort_lock_name:
            keys.add(self.cohort_lock_name)
        return sorted(keys)

    def resolved_cohort_lock_dir(self) -> str:
        if self.cohort_lock_dir:
            return self.cohort_lock_dir

        # Configs that write the same cohort should share a lock directory: set
        # cohort_lock_dir explicitly when their state dirs have different parents.
        state = Path(self.state_dir).resolve()
        return str(state.parent / "locks")

    @staticmethod
    def lock_filename_for_key(key: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
        return (safe or "cohort") + ".lock"

    def _build_auto_hooks(self) -> list:
        """Generate WDS-append and Databricks-sync post-batch hooks from watcher config."""
        hooks = []
        db_hooks = []
        for w in self.watchers:
            if w.type == "databricks" and w.wds_destinations:
                destinations = list(w.wds_destinations.items())
                for idx, (model, dest) in enumerate(destinations):
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
                    args.extend(self.wds_lock_args())
                    # A WDS hook must fail loudly when a requested slide has
                    # no model output. Silent skips otherwise leave the slide
                    # marked SUCCEEDED while widening the dashboard gap.
                    args.append("--require-all-slide-ids")
                    if w.wds_s3_max_concurrency != 4:
                        args.append(f"--s3-max-concurrency={w.wds_s3_max_concurrency}")
                    if w.aws_profile:
                        args.append("--aws-profile=" + w.aws_profile)
                    if self.cleanup_results:
                        args.append("--delete-local")
                        if idx == len(destinations) - 1:
                            args.append("--delete-coords-local")
                    hooks.append({"command": "python -m mussel_dispatcher.wds", "args": args, "required": True})

            if w.type != "tcga":
                continue

            if w.wds_destinations:
                destinations = list(w.wds_destinations.items())
                for idx, (model, dest) in enumerate(destinations):
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
                    args.extend(self.wds_lock_args())
                    args.append("--require-all-slide-ids")
                    if w.wds_s3_max_concurrency != 4:
                        args.append(f"--s3-max-concurrency={w.wds_s3_max_concurrency}")
                    if w.aws_profile:
                        args.append("--aws-profile=" + w.aws_profile)
                    if w.s3_endpoint:
                        args.append("--s3-endpoint=" + w.s3_endpoint)
                    if self.cleanup_results:
                        args.append("--delete-local")
                        if idx == len(destinations) - 1:
                            args.append("--delete-coords-local")
                    hooks.append({"command": "python -m mussel_dispatcher.wds", "args": args, "required": True})

            if w.databricks_volume_folder or w.databricks_volume_path:
                extra_args = ["--inventory=" + w.inventory_csv, "--status=" + w.status_csv]
                db_hooks.append(
                    _build_db_sync_hook(w, "mussel_dispatcher.tcga.sync_databricks",
                                        extra_args, state_dir=self.state_dir,
                                        phase="db_sync")
                )

        return hooks + db_hooks


def _build_db_sync_hook(w: "WatcherConfig", module: str, extra_args: list[str],
                        *, state_dir: str = "", phase: str = "") -> dict:
    """Build a post-batch hook dict for a Databricks sync script.

    Appends the common volume-folder/path, table, job-id, and status-file args
    from *w* to *extra_args* and returns a hook dict.
    """
    args = list(extra_args)
    if w.databricks_volume_folder:
        args.append("--volume-folder=" + w.databricks_volume_folder)
    elif w.databricks_volume_path:
        args.append("--volume-path=" + w.databricks_volume_path)
    if w.databricks_table:
        args.append("--table=" + w.databricks_table)
    if w.databricks_job_id:
        args.append("--job-id=" + w.databricks_job_id)
    elif w.warehouse_id:
        # No pre-created job — merge directly via the SQL warehouse.
        args.append("--warehouse-id=" + w.warehouse_id)
    if state_dir and (w.databricks_job_id or w.warehouse_id):
        status_file = os.path.join(state_dir, "sync_status.json")
        args.append("--status-file=" + status_file)
    hook: dict = {"command": f"python -m {module}", "args": args, "required": True}
    if phase:
        hook["_phase"] = phase
    return hook


def _apply_wds_destination_defaults(w: WatcherConfig) -> None:
    """Expand a single WDS destination root across configured models.

    Preferred config is `wds_destination: s3://bucket/wds`, which applies to
    every model in `watcher.models`. For backward compatibility, if
    `wds_destinations` contains a single unique destination root, missing models
    also inherit that root automatically.
    """
    if not w.models:
        return

    default_dest = w.wds_destination
    if not default_dest and w.wds_destinations:
        unique_dests = {dest for dest in w.wds_destinations.values() if dest}
        if len(unique_dests) == 1:
            default_dest = next(iter(unique_dests))

    if not default_dest:
        return

    for model in w.models:
        w.wds_destinations.setdefault(model, default_dest)


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


def _load_nf_secrets(secret_names: list | dict, target: object) -> None:
    """Resolve Nextflow secrets and set attributes on *target*.

    *secret_names* can be:
    - A **dict** ``{attr: secret_name}`` — explicit mapping, e.g.
      ``{"s3_access_key": "ECS_ACCESS_KEY", "s3_secret_key": "ECS_SECRET_KEY"}``
    - A **list** of secret names — looked up via a built-in KEY_MAP for
      backward compatibility (ECS_ACCESS_KEY, AWS_ACCESS_KEY_ID, etc.)
    """
    if isinstance(secret_names, dict):
        mapping = secret_names  # {attr: secret_name}
    else:
        # Legacy list form: map well-known names to config attrs
        _KEY_MAP = {
            "ECS_ACCESS_KEY": "s3_access_key", "AWS_ACCESS_KEY_ID": "s3_access_key",
            "ECS_SECRET_KEY": "s3_secret_key", "AWS_SECRET_ACCESS_KEY": "s3_secret_key",
        }
        mapping = {_KEY_MAP[n]: n for n in secret_names if n in _KEY_MAP}

    for attr, secret_name in mapping.items():
        if getattr(target, attr, None):
            continue
        try:
            result = subprocess.run(
                ["nextflow", "secrets", "get", secret_name],
                capture_output=True, text=True, timeout=15,
            )
            value = result.stdout.strip()
            if result.returncode != 0 or not value:
                log.warning("nextflow secrets get %s failed: %s", secret_name,
                            result.stderr.strip() or "empty output")
                continue
            setattr(target, attr, value)
            log.debug("Loaded %s from Nextflow secret %s", attr, secret_name)
        except Exception as exc:
            log.warning("Could not load Nextflow secret %s: %s", secret_name, exc)


def load_runtime_env(cfg: Config) -> None:
    """Load configured env files into ``os.environ`` without overriding shell-set values.

    This is used by long-lived local processes (dispatcher, dashboard, sync hooks)
    that may need credentials such as ``POSTGRES_APP_PASSWORD`` before any remote
    target is selected or before Postgres state is initialized.
    """

    env_paths: list[str] = []
    if cfg.secrets_env_file:
        env_paths.append(cfg.secrets_env_file)
    for watcher in cfg.watchers:
        if watcher.secrets_env_file:
            env_paths.append(watcher.secrets_env_file)
    for target in cfg.dispatch_targets:
        if target.secrets_env_file:
            env_paths.append(target.secrets_env_file)

    seen: set[str] = set()
    for path in env_paths:
        norm = os.path.abspath(path)
        if norm in seen or not os.path.isfile(norm):
            continue
        seen.add(norm)
        try:
            with open(norm, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
        except OSError as exc:
            log.warning("Could not read runtime env file %s: %s", norm, exc)


def _read_nf_model_types(repo_dir: str, nextflow_params_file: str = "") -> list[str]:
    """Parse model_types from the active Nextflow config surface.

    Prefer `nextflow_params_file` when present, since it overrides the repo
    defaults actually used by the dispatcher's Nextflow runs. Fall back to the
    first `model_types = [...]` list found in `nextflow.config`.
    """
    import re
    if nextflow_params_file:
        try:
            with open(nextflow_params_file) as f:
                data = yaml.safe_load(f) or {}
            models = (data.get("featurize") or {}).get("model_types") or []
            if isinstance(models, list):
                return [str(m) for m in models]
        except OSError:
            log.warning("Could not read %s — falling back to nextflow.config", nextflow_params_file)

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


def _read_nf_publish_mode_from_params(params_file: str) -> str | None:
    try:
        with open(params_file) as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return None
    mode = data.get("publish_mode")
    return str(mode) if mode is not None else None


def _read_nf_publish_mode_from_config(config_path: str) -> str | None:
    try:
        text = Path(config_path).read_text()
    except OSError:
        return None
    matches = re.findall(
        r"(?:^|\b)(?:params\.)?publish_mode\s*=\s*['\"]([^'\"]+)['\"]",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else None
