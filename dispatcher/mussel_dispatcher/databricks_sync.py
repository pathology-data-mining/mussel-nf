"""Shared Databricks upload / job-trigger utilities for mussel-dispatcher sync scripts.

Used by both ``mussel_dispatcher.tcga.sync_databricks`` and
``mussel_dispatcher.impact.sync_databricks`` to avoid duplicating credential
resolution, HTTP upload, and argparse boilerplate.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Terminal lifecycle states for a Databricks job run.
_TERMINAL_STATES = {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}
_POLL_INTERVAL_S  = 5
_POLL_TIMEOUT_S   = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def resolve_credentials(host: str, token: str) -> tuple[str, str]:
    """Resolve host/token, falling back to env vars then ~/.databrickscfg.

    Resolution order:
    1. Non-empty values passed in (e.g. from CLI flags)
    2. ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` environment variables
    3. ``~/.databrickscfg`` ``[DEFAULT]`` section
    """
    host = host or os.environ.get("DATABRICKS_HOST", "")
    token = token or os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        host, token = _load_databrickscfg(host, token)
    return host, token


def _load_databrickscfg(host: str, token: str) -> tuple[str, str]:
    """Fill missing host/token from ~/.databrickscfg [DEFAULT] section."""
    cfg_path = Path.home() / ".databrickscfg"
    if not cfg_path.exists():
        return host, token
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    section = "DEFAULT"
    if not host:
        host = cfg.get(section, "host", fallback="")
    if not token:
        token = cfg.get(section, "token", fallback="")
    if host or token:
        log.info("Loaded Databricks credentials from %s", cfg_path)
    return host, token


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def upload_parquet(local_path: Path, volume_path: str, host: str, token: str) -> None:
    """PUT a local file to the Databricks Files API (UC volume)."""
    url = f"{host.rstrip('/')}/api/2.0/fs/files{volume_path}"
    headers = {"Authorization": f"Bearer {token}"}
    with open(local_path, "rb") as f:
        resp = requests.put(url, headers=headers, data=f, timeout=300)
    resp.raise_for_status()
    log.info("Uploaded to Databricks: %s", volume_path)


def trigger_job(job_id: str, host: str, token: str, params: dict | None = None) -> str:
    """POST to jobs/run-now; return run_id as str."""
    url = f"{host.rstrip('/')}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body: dict = {"job_id": int(job_id)}
    if params:
        body["notebook_params"] = params
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    run_id = str(resp.json()["run_id"])
    log.info("Triggered job %s → run_id %s", job_id, run_id)
    return run_id


def poll_job_run(
    run_id: str,
    host: str,
    token: str,
    *,
    timeout_s: float = _POLL_TIMEOUT_S,
    interval_s: float = _POLL_INTERVAL_S,
) -> tuple[bool, str]:
    """Poll a Databricks job run until it reaches a terminal state.

    Returns (success, message) where success is True iff result_state == SUCCESS.
    """
    url = f"{host.rstrip('/')}/api/2.1/jobs/runs/get"
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, headers=headers, params={"run_id": run_id}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("poll_job_run: request error (will retry): %s", exc)
            time.sleep(interval_s)
            continue

        state      = data.get("state", {})
        lifecycle  = state.get("life_cycle_state", "")
        result     = state.get("result_state", "")
        message    = state.get("state_message", "")

        if lifecycle not in _TERMINAL_STATES:
            log.debug("Job run %s: %s — waiting…", run_id, lifecycle)
            time.sleep(interval_s)
            continue

        succeeded = result == "SUCCESS"
        if not succeeded:
            # Try to get task-level error for a cleaner message
            for task in data.get("tasks", []):
                task_msg = task.get("state", {}).get("state_message", "")
                if task_msg:
                    message = task_msg
                    break
            log.error("Job run %s FAILED (%s): %s", run_id, result, message)
        else:
            log.info("Job run %s succeeded", run_id)
        return succeeded, message

    msg = f"Job run {run_id} did not finish within {timeout_s}s"
    log.error(msg)
    return False, msg


def write_sync_status(
    status_file: str,
    *,
    job_id: str,
    run_id: str,
    success: bool,
    message: str,
    table: str = "",
) -> None:
    """Write a JSON status file so the dashboard can surface sync health."""
    payload = {
        "job_id":     job_id,
        "run_id":     run_id,
        "status":     "SUCCESS" if success else "FAILED",
        "message":    message,
        "table":      table,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        Path(status_file).write_text(json.dumps(payload, indent=2))
        log.debug("Wrote sync status to %s", status_file)
    except OSError as exc:
        log.warning("Could not write sync status file %s: %s", status_file, exc)


# ---------------------------------------------------------------------------
# Shared argparse
# ---------------------------------------------------------------------------

def add_upload_args(parser: argparse.ArgumentParser) -> None:
    """Add shared Databricks upload/trigger arguments to *parser*.

    Added flags:
        --databricks-host, --token
        --volume-folder (preferred, timestamped)
        --volume-path   (legacy, single overwritten file)
        --table, --job-id
        --output-parquet (local copy)
        --verbose / -v
    """
    parser.add_argument(
        "--databricks-host", default=None,
        help="Databricks workspace URL (or set DATABRICKS_HOST env var)",
    )
    parser.add_argument(
        "--token", default=None,
        help="Databricks personal access token (or set DATABRICKS_TOKEN)",
    )
    parser.add_argument(
        "--volume-folder", default=None,
        help="UC volume folder to upload into; file is named "
             "<prefix><timestamp>.parquet",
    )
    parser.add_argument(
        "--volume-path", default=None,
        help="[Legacy] Full UC volume path for a single overwritten Parquet. "
             "Use --volume-folder for timestamped uploads.",
    )
    parser.add_argument(
        "--table", default=None,
        help="Target Delta table (passed as notebook_param 'target_table'). "
             "E.g. cdsi_prod.pathology_data_mining.tcga_slide_embeddings_v2",
    )
    parser.add_argument(
        "--job-id", default=None,
        help="Databricks job ID to trigger after upload (optional)",
    )
    parser.add_argument(
        "--status-file", default=None,
        help="Path to write a JSON sync-status file (read by the dashboard). "
             "If omitted no file is written.",
    )
    parser.add_argument(
        "--output-parquet", default=None,
        help="Also save the Parquet file locally at this path",
    )
    parser.add_argument("--verbose", "-v", action="store_true")


# ---------------------------------------------------------------------------
# High-level upload + trigger
# ---------------------------------------------------------------------------

def upload_and_trigger(
    export_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    filename_prefix: str,
) -> None:
    """Write *export_df* to a temp Parquet, upload to Databricks, optionally trigger job.

    Handles both ``--volume-folder`` (timestamped file name) and
    ``--volume-path`` (legacy single-file path).

    ``filename_prefix`` sets the file name stem, e.g. ``"tcga_inventory_"`` or
    ``"impact_inventory_"``.
    """
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        export_df.to_parquet(tmp_path, index=False)
        log.info("Parquet size: %.1f KB", tmp_path.stat().st_size / 1024)

        if args.output_parquet:
            import shutil
            shutil.copy(tmp_path, args.output_parquet)
            log.info("Saved local copy: %s", args.output_parquet)

        host, token = resolve_credentials(
            args.databricks_host or "",
            args.token or "",
        )
        if not host or not token:
            log.error(
                "Databricks credentials required: set DATABRICKS_HOST / DATABRICKS_TOKEN, "
                "use --databricks-host / --token, or configure ~/.databrickscfg"
            )
            raise SystemExit(1)

        if args.volume_folder:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            folder = args.volume_folder.rstrip("/")
            volume_path = f"{folder}/{filename_prefix}{ts}.parquet"
        else:
            volume_path = args.volume_path

        upload_parquet(tmp_path, volume_path, host, token)

        if args.job_id:
            job_params: dict = {}
            if args.volume_folder:
                job_params["volume_folder"] = args.volume_folder
            if args.table:
                job_params["target_table"] = args.table
            run_id = trigger_job(args.job_id, host, token, params=job_params or None)

            # Poll to completion so the dispatcher hook fails visibly when the MERGE fails.
            success, message = poll_job_run(run_id, host, token)

            if getattr(args, "status_file", None):
                write_sync_status(
                    args.status_file,
                    job_id=args.job_id,
                    run_id=run_id,
                    success=success,
                    message=message,
                    table=args.table or "",
                )

            if not success:
                raise SystemExit(f"Databricks job {args.job_id} run {run_id} FAILED: {message}")

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
