#!/usr/bin/env python3
"""Cron-friendly orchestrator for TCGA feature extraction.

Chains the TCGA pipeline steps end-to-end:
  1. sync       — Sync slide inventory from GDC API (tcga_sync_inventory.py)
  2. status     — Update per-slide status from local results (tcga_update_status.py)
  3. prepare    — Prepare samples CSV; resolves paths local → S3 → needs_download
  4. download   — Download only the slides not already on disk/S3 (gdc-client)
  5. run        — Run mussel-nf nextflow pipeline
  6. append-wds — Append new features to WDS shards (tcga_append_wds.py)
  7. databricks — Sync inventory to Databricks (tcga_sync_databricks.py)

Each phase can be skipped individually with --skip-<phase>. In --initial-run
mode, phases 3–6 loop until no pending slides remain, processing --chunk-size
slides per iteration to keep disk usage bounded.

Usage
-----
    # Incremental run (cron):
    python tcga_run.py --config tcga_run_config.yaml

    # Initial full load (chunked, resumable):
    python tcga_run.py --config tcga_run_config.yaml --initial-run

    # Override chunk size or project:
    python tcga_run.py --config tcga_run_config.yaml --initial-run \\
        --chunk-size 200 --project TCGA-BRCA

    # Dry run (print commands, write nothing):
    python tcga_run.py --config tcga_run_config.yaml --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent.parent


def _load_nextflow_secret(name: str) -> str | None:
    """Read a single nextflow secret by name. Returns None if not found."""
    try:
        result = subprocess.run(
            ["nextflow", "secrets", "get", name],
            capture_output=True, text=True, check=False,
        )
        value = result.stdout.strip()
        return value if value else None
    except FileNotFoundError:
        return None


def _load_nextflow_config() -> dict[str, str]:
    """Return a flat dict from `nextflow config -flat` (key → unquoted value).

    Returns empty dict if nextflow is unavailable or the config cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["nextflow", "config", "-flat"],
            capture_output=True, text=True, check=False,
            cwd=str(REPO_ROOT),
        )
        cfg: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip("'\"")
        return cfg
    except FileNotFoundError:
        return {}


def _inject_ecs_credentials(slides_cfg: dict) -> None:
    """Ensure ECS credentials and endpoint are set in the process environment.

    Resolution order for credentials (first non-empty wins):
      1. Already present in environment
      2. Explicit keys in slides config (s3_access_key / s3_secret_key)
      3. Nextflow secrets store (ECS_ACCESS_KEY / ECS_SECRET_KEY)

    Resolution order for endpoint (first non-empty wins):
      1. Already present in environment (ECS_ENDPOINT_URL)
      2. slides config s3_endpoint
      3. aws.client.endpoint from `nextflow config -flat`
    """
    import os
    for env_var, cfg_key, secret_name in [
        ("ECS_ACCESS_KEY", "s3_access_key", "ECS_ACCESS_KEY"),
        ("ECS_SECRET_KEY", "s3_secret_key", "ECS_SECRET_KEY"),
    ]:
        if os.environ.get(env_var):
            continue
        value = slides_cfg.get(cfg_key) or _load_nextflow_secret(secret_name)
        if value:
            os.environ[env_var] = value
            log.debug("Set %s from %s", env_var,
                      "config" if slides_cfg.get(cfg_key) else "nextflow secrets")

    # Propagate the ECS endpoint URL so child scripts don't need --s3-endpoint
    if not os.environ.get("ECS_ENDPOINT_URL"):
        endpoint = slides_cfg.get("s3_endpoint") or _load_nextflow_config().get("aws.client.endpoint")
        if endpoint:
            os.environ["ECS_ENDPOINT_URL"] = endpoint
            log.debug("Set ECS_ENDPOINT_URL=%s from %s", endpoint,
                      "config" if slides_cfg.get("s3_endpoint") else "nextflow config")


def _run_script(script: str, args_list: list[str], dry_run: bool = False) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args_list]
    log.info("$ %s", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_nextflow_params(config: dict, samples_csv: str) -> tuple[dict, str, list[str]]:
    """Build the merged nextflow params dict from user config + TCGA overrides.

    Returns (merged_params, outdir, model_types_list).

    The resolution order (later entries win):
      1. nextflow.config defaults (not read here — nextflow handles them)
      2. User's nextflow.params_file  (base params YAML, e.g. filter_workflow.yaml)
      3. TCGA config overrides        (nextflow.outdir, nextflow.model_types, etc.)
      4. Mandatory TCGA overrides     (samples_csv, wds.enabled=false)
    """
    nf_cfg = config.get("nextflow", {})

    # Start from user's params file if provided
    merged: dict = {}
    params_file = nf_cfg.get("params_file")
    if params_file and Path(params_file).exists():
        with open(params_file) as f:
            merged = yaml.safe_load(f) or {}
        log.info("Loaded base params from %s", params_file)

    # Apply TCGA config overrides (nextflow sub-keys map directly to nextflow params)
    for key in ("outdir", "publish_mode", "publish_slide_prefix"):
        if nf_cfg.get(key) is not None:
            merged[key] = nf_cfg[key]

    # model_types override: TCGA config wins over params_file; params_file wins over default
    if nf_cfg.get("model_types"):
        model_types = nf_cfg["model_types"]
        if isinstance(model_types, str):
            model_types = [m.strip() for m in model_types.split(",")]
        merged.setdefault("featurize", {})
        merged["featurize"]["model_types"] = model_types
    elif "featurize" not in merged or "model_types" not in merged.get("featurize", {}):
        # Fall back to nextflow.config default (ctranspath, h0mini, uni2h)
        log.info("model_types not set in TCGA config or params_file — using nextflow.config default")

    # Mandatory TCGA overrides — always applied
    merged["samples_csv"] = samples_csv
    merged["outdir"] = nf_cfg.get("outdir", merged.get("outdir", "./tcga-results"))
    merged.setdefault("wds", {})
    merged["wds"]["enabled"] = False   # WDS sharding is handled by tcga_append_wds.py

    # Resolve outdir and model_types for callers
    outdir: str = merged["outdir"]
    raw_models = merged.get("featurize", {}).get("model_types", ["ctranspath"])
    if isinstance(raw_models, str):
        raw_models = [m.strip() for m in raw_models.split(",")]

    # Expand slide-level model aliases to their patch encoder names for output scanning
    # (nextflow outputs patch features under the patch encoder name)
    slide_to_patch: dict[str, str] = {
        "gigapath_slide": "gigapath",
        "titan_slide": "conch1_5",
        "prism_slide": "conch1_5",
        "feather_slide": "uni2h",
        "chief_slide": "ctranspath",
        "madeleine_slide": "conch1_5",
    }
    patch_models = list(dict.fromkeys(
        slide_to_patch.get(m, m) for m in raw_models
    ))

    return merged, outdir, patch_models


def _run_nextflow(config: dict, samples_csv: str, dry_run: bool = False) -> int:
    nf_cfg = config.get("nextflow", {})
    profile = nf_cfg.get("profile", "standard,apptainer")

    merged_params, outdir, _ = resolve_nextflow_params(config, samples_csv)

    # Write merged params to a temp file — single -params-file, no conflicting CLI overrides
    main_nf = SCRIPTS_DIR.parent.parent / "main.nf"
    tmp_params = Path(tempfile.mktemp(suffix="_tcga_params.yaml", dir=main_nf.parent))
    try:
        tmp_params.write_text(yaml.dump(merged_params, default_flow_style=False))
        log.debug("Merged nextflow params written to %s", tmp_params)

        cmd = [
            "nextflow", "run", str(main_nf),
            "-profile", profile,
            "-params-file", str(tmp_params),
            "-resume",
        ] + (nf_cfg.get("extra_args") or [])

        log.info("$ %s", " ".join(cmd))
        if dry_run:
            return 0
        return subprocess.run(cmd, check=False, cwd=str(main_nf.parent)).returncode
    finally:
        if tmp_params.exists() and not dry_run:
            tmp_params.unlink()


def _run_append_wds(config: dict, dry_run: bool = False, chunk_samples_csv: str | None = None) -> int:
    nf_cfg = config.get("nextflow", {})
    _, outdir, model_types = resolve_nextflow_params(config, "")
    wds_cfg = config.get("wds", {})
    wds_dest = wds_cfg.get("dest", "")
    staging_dir = wds_cfg.get("staging_dir", "")
    max_shard_bytes = wds_cfg.get("max_shard_bytes", 2 * 1024 ** 3)
    inventory_csv = config.get("inventory_csv", "tcga_inventory.csv")

    rc = 0
    for model in model_types:
        pt_dir = Path(outdir) / "features" / model / "pt"
        h5_dir = Path(outdir) / "features" / model / "tile_h5"
        script_args = [
            "--pt-dir", str(pt_dir),
            "--h5-dir", str(h5_dir),
            "--inventory", inventory_csv,
            "--wds-dest", wds_dest,
            "--model-type", model,
            "--max-shard-bytes", str(max_shard_bytes),
        ]
        if staging_dir:
            script_args += ["--staging-dir", staging_dir]
        if chunk_samples_csv and Path(chunk_samples_csv).exists():
            script_args += ["--slide-ids-csv", chunk_samples_csv]
        code = _run_script("tcga_append_wds.py", script_args, dry_run=dry_run)
        if code != 0:
            log.error("tcga_append_wds.py failed for model %s (exit %d)", model, code)
            rc = code
    return rc


def _run_databricks(config: dict, dry_run: bool = False) -> int:
    db_cfg = config.get("databricks", {})
    volume_path = db_cfg.get("volume_path")
    if not volume_path:
        log.info("No databricks.volume_path configured — skipping Databricks sync")
        return 0
    script_args = [
        "--status", config.get("status_csv", "tcga_status.csv"),
        "--inventory", config.get("inventory_csv", "tcga_inventory.csv"),
        "--volume-path", volume_path,
    ]
    if db_cfg.get("host"):
        script_args += ["--databricks-host", db_cfg["host"]]
    if db_cfg.get("job_id"):
        script_args += ["--job-id", str(db_cfg["job_id"])]
    return _run_script("tcga_sync_databricks.py", script_args, dry_run=dry_run)


def _delete_downloaded_slides(config: dict) -> None:
    local_dir = config.get("download", {}).get("local_dir")
    if not local_dir:
        return
    p = Path(local_dir)
    if not p.exists():
        return
    count = 0
    for f in list(p.rglob("*.svs")) + list(p.rglob("*.tif")) + list(p.rglob("*.tiff")):
        f.unlink()
        count += 1
    log.info("Deleted %d slide files from %s", count, local_dir)


def _append_run_log(log_path: Path, record: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_one_pass(config: dict, args: argparse.Namespace) -> int:
    """Execute a single pass of phases 1–6. Returns exit code.

    Exit code 2 means "no pending slides; loop should stop".
    """
    inventory_csv = config.get("inventory_csv", "tcga_inventory.csv")
    status_csv = config.get("status_csv", "tcga_status.csv")
    samples_csv = config.get("samples_csv", "samples_to_run.csv")
    samples_meta_csv = str(Path(samples_csv).with_suffix(".meta.csv"))

    slides_cfg = config.get("slides", {})
    gdc_cfg = config.get("gdc", {})
    download_cfg = config.get("download", {})

    # Ensure ECS credentials are in the environment for S3 operations in child scripts
    _inject_ecs_credentials(slides_cfg)

    # Resolve outdir and model_types from merged nextflow params (single source of truth)
    _, outdir, model_types = resolve_nextflow_params(config, samples_csv)
    first_model = model_types[0] if model_types else "ctranspath"

    effective_project = args.project or slides_cfg.get("project")

    # ------------------------------------------------------------------
    # Phase 1: sync inventory
    # ------------------------------------------------------------------
    if not args.skip_sync:
        sync_args = ["--output", inventory_csv]
        if effective_project:
            sync_args += ["--project", effective_project]
        gdc_max_age = gdc_cfg.get("max_age_hours", 24)
        sync_args += ["--max-age-hours", str(gdc_max_age)]
        if args.force_sync:
            sync_args.append("--force")
        rc = _run_script("tcga_sync_inventory.py", sync_args, dry_run=args.dry_run)
        if rc not in (0, 2):
            log.error("tcga_sync_inventory.py failed (exit %d)", rc)
            return rc
        if rc == 2 and not args.initial_run:
            log.info("No new slides in GDC inventory — nothing to do")
            return 2

    # ------------------------------------------------------------------
    # Phase 2: update status  (model_types auto-discovered from results dir)
    # ------------------------------------------------------------------
    if not args.skip_status:
        rc = _run_script("tcga_update_status.py", [
            "--inventory", inventory_csv,
            "--results-dir", outdir,
            "--output", status_csv,
        ], dry_run=args.dry_run)
        if rc != 0:
            log.error("tcga_update_status.py failed (exit %d)", rc)
            return rc

    # ------------------------------------------------------------------
    # Phase 3: prepare samples
    # ------------------------------------------------------------------
    if not args.skip_prepare:
        prepare_args = [
            "--inventory", inventory_csv,
            "--status", status_csv,
            "--output", samples_csv,
            "--model", first_model,
            "--skip-done",
        ]
        if slides_cfg.get("local_slides_dir"):
            prepare_args += ["--local-slides-dir", slides_cfg["local_slides_dir"]]
        if slides_cfg.get("s3_base"):
            prepare_args += ["--s3-base", slides_cfg["s3_base"]]
        if slides_cfg.get("s3_endpoint"):
            prepare_args += ["--s3-endpoint", slides_cfg["s3_endpoint"]]
        # Credentials: config > ECS_ACCESS_KEY env var (auto-detected in script)
        if slides_cfg.get("s3_access_key"):
            prepare_args += ["--s3-access-key", slides_cfg["s3_access_key"]]
        if slides_cfg.get("s3_secret_key"):
            prepare_args += ["--s3-secret-key", slides_cfg["s3_secret_key"]]
        # check_s3_exists defaults to True in config; use --no-check-s3-exists to disable
        if slides_cfg.get("check_s3_exists", True):
            prepare_args.append("--check-s3-exists")
        else:
            prepare_args.append("--no-check-s3-exists")
        if slides_cfg.get("slide_type") and slides_cfg["slide_type"] != "all":
            prepare_args += ["--slide-type", slides_cfg["slide_type"]]
        if effective_project:
            prepare_args += ["--project", effective_project]
        chunk_size = args.chunk_size or config.get("initial_run", {}).get("chunk_size")
        if chunk_size:
            prepare_args += ["--limit", str(chunk_size)]

        rc = _run_script("tcga_prepare_samples.py", prepare_args, dry_run=args.dry_run)
        if rc == 2:
            log.info("No pending slides — pass complete")
            return 2
        if rc != 0:
            log.error("tcga_prepare_samples.py failed (exit %d)", rc)
            return rc

    # ------------------------------------------------------------------
    # Phase 4: run Nextflow on all slides.
    # DOWNLOAD_SLIDE in main.nf handles GDC downloads for any slide with
    # needs_download=true; slides already on disk/S3 flow straight through.
    # ------------------------------------------------------------------
    if not args.skip_run:
        import pandas as pd
        n_total = n_ready = n_download = 0
        if Path(samples_csv).exists():
            samples_df = pd.read_csv(samples_csv, dtype=str).fillna("")
            n_total    = len(samples_df)
            n_download = int((samples_df.get("needs_download", pd.Series("false")) == "true").sum())
            n_ready    = n_total - n_download
        log.info(
            "Phase 4: running nextflow on %d slides (%d on disk/S3, %d need GDC download)",
            n_total, n_ready, n_download,
        )
        rc = _run_nextflow(config, samples_csv, dry_run=args.dry_run)
        if rc != 0:
            log.error("nextflow failed (exit %d)", rc)
            return rc

    # ------------------------------------------------------------------
    # Phase 6: append WDS
    # ------------------------------------------------------------------
    if not args.skip_append_wds:
        rc = _run_append_wds(config, dry_run=args.dry_run, chunk_samples_csv=samples_csv)
        if rc != 0:
            log.error("tcga_append_wds.py failed")
            return rc

    if args.delete_slides:
        _delete_downloaded_slides(config)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True,
                        help="Path to tcga_run_config.yaml")
    parser.add_argument("--initial-run", action="store_true",
                        help="Loop phases 3–6 until all pending slides are processed")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Slides per iteration in --initial-run mode (overrides config)")
    parser.add_argument("--project", default=None,
                        help="Override: process specific TCGA project(s), comma-separated")
    parser.add_argument("--delete-slides", action="store_true",
                        help="Delete downloaded SVS files after each chunk to reclaim disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing anything")

    # Phase skip flags
    for phase in ("sync", "status", "prepare", "run", "append-wds", "databricks"):
        parser.add_argument(f"--skip-{phase}", action="store_true",
                            dest=f"skip_{phase.replace('-', '_')}")

    parser.add_argument("--force-sync", action="store_true",
                        help="Force GDC inventory re-fetch even if cache is still fresh")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_path = Path(config.get("run_log", "tcga_run_log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.initial_run:
        ir_cfg = config.get("initial_run", {})
        if args.chunk_size is None:
            args.chunk_size = ir_cfg.get("chunk_size", 500)
        if not args.delete_slides:
            args.delete_slides = ir_cfg.get("delete_slides_after_chunk", False)
        if args.project is None:
            args.project = ir_cfg.get("project")

        chunk = 0
        while True:
            chunk += 1
            log.info("=== Initial run — chunk %d (chunk_size=%d) ===", chunk, args.chunk_size)
            start = datetime.now(timezone.utc)
            rc = run_one_pass(config, args)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            _append_run_log(log_path, {
                "ts": start.isoformat(),
                "mode": "initial",
                "chunk": chunk,
                "exit_code": rc,
                "elapsed_s": round(elapsed, 1),
            })
            if rc == 2:
                log.info("All slides processed — initial run complete after %d chunks", chunk)
                break
            if rc != 0:
                log.error("Chunk %d failed (exit %d) — stopping", chunk, rc)
                return rc
            # After the first chunk skip global sync/status (status is updated each iteration)
            args.skip_sync = True

        # Databricks sync once at the end of the initial run
        if not args.skip_databricks:
            _run_databricks(config, dry_run=args.dry_run)

    else:
        start = datetime.now(timezone.utc)
        rc = run_one_pass(config, args)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        _append_run_log(log_path, {
            "ts": start.isoformat(),
            "mode": "incremental",
            "exit_code": rc,
            "elapsed_s": round(elapsed, 1),
        })

        # Phase 7: Databricks (after successful incremental run)
        if rc in (0, 2) and not args.skip_databricks:
            _run_databricks(config, dry_run=args.dry_run)

        return 0 if rc in (0, 2) else rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
