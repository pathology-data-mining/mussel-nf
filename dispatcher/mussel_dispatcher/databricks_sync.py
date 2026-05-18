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
_SQL_POLL_INTERVAL_S = 5
_SQL_TIMEOUT_S       = 600


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


def resolve_warehouse_id(host: str = "", token: str = "") -> str:
    """Auto-detect the best available SQL warehouse for this workspace.

    Prefers RUNNING warehouses, then SERVERLESS > PRO > CLASSIC by type.
    Returns the warehouse ID string, or ``""`` if none found or credentials
    are unavailable (non-fatal — caller should warn and continue).
    """
    host, token = resolve_credentials(host, token)
    if not host or not token:
        return ""
    try:
        resp = requests.get(
            f"{host.rstrip('/')}/api/2.0/sql/warehouses",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        warehouses = resp.json().get("warehouses", [])
    except Exception as exc:
        log.debug("resolve_warehouse_id: could not list warehouses: %s", exc)
        return ""

    active = [w for w in warehouses if w.get("state") != "DELETED"]
    if not active:
        return ""

    _state_rank = {"RUNNING": 0, "STARTING": 1, "STOPPING": 2, "STOPPED": 3}
    _type_rank  = {"SERVERLESS": 0, "PRO": 1, "CLASSIC": 2}
    active.sort(key=lambda w: (
        _state_rank.get(w.get("state", ""), 9),
        _type_rank.get(w.get("warehouse_type", ""), 9),
        w.get("name", ""),
    ))
    chosen = active[0]
    if len(active) > 1:
        log.info(
            "Multiple SQL warehouses available; auto-selected %r (%s). "
            "Set warehouse_id explicitly to suppress this message.",
            chosen.get("name"), chosen["id"],
        )
    else:
        log.debug("Auto-selected SQL warehouse %r (%s)", chosen.get("name"), chosen["id"])
    return chosen["id"]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def ensure_volume_exists(
    volume_folder: str, host: str, token: str, warehouse_id: str
) -> None:
    """CREATE VOLUME IF NOT EXISTS for the UC volume backing *volume_folder*.

    *volume_folder* must start with ``/Volumes/<catalog>/<schema>/<volume>``.
    The three-part name is extracted and used in the DDL statement so the
    volume is created as a managed volume if it does not already exist.
    """
    parts = [p for p in volume_folder.strip("/").split("/") if p]
    # parts[0] == "Volumes", parts[1..3] == catalog / schema / volume
    if len(parts) < 4 or parts[0].lower() != "volumes":
        log.warning("Cannot parse volume path for auto-create: %s", volume_folder)
        return
    fqn = ".".join(parts[1:4])
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "warehouse_id": warehouse_id,
        "statement": f"CREATE VOLUME IF NOT EXISTS {fqn}",
        "wait_timeout": "30s",
    }
    resp = requests.post(
        f"{host.rstrip('/')}/api/2.0/sql/statements",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    state = resp.json().get("status", {}).get("state", "")
    if state == "SUCCEEDED":
        log.info("Volume ready: %s", fqn)
    else:
        err = resp.json().get("status", {}).get("error", {})
        log.warning("CREATE VOLUME %s: %s", fqn, err.get("message", state))


def upload_parquet(local_path: Path, volume_path: str, host: str, token: str) -> None:
    """PUT a local file to the Databricks Files API (UC volume)."""
    url = f"{host.rstrip('/')}/api/2.0/fs/files{volume_path}"
    headers = {"Authorization": f"Bearer {token}"}
    with open(local_path, "rb") as f:
        resp = requests.put(url, headers=headers, data=f, timeout=300)
    resp.raise_for_status()
    log.info("Uploaded to Databricks: %s", volume_path)


def purge_volume_folder(folder: str, host: str, token: str) -> None:
    """Delete all parquet files in the UC volume folder before uploading a fresh one.

    This prevents the MERGE from reading stale parquets with incompatible schemas
    or duplicate rows from prior runs.
    """
    base = host.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    list_url = f"{base}/api/2.0/fs/directories{folder.rstrip('/')}"
    try:
        resp = requests.get(list_url, headers=headers, timeout=30)
        if resp.status_code == 404:
            return  # folder doesn't exist yet — nothing to purge
        resp.raise_for_status()
        files = resp.json().get("contents", [])
    except Exception as exc:
        log.warning("purge_volume_folder: could not list %s: %s", folder, exc)
        return

    for f in files:
        path = f.get("path", "")
        if path.endswith(".parquet"):
            del_url = f"{base}/api/2.0/fs/files{path}"
            try:
                r = requests.delete(del_url, headers=headers, timeout=30)
                r.raise_for_status()
                log.debug("Deleted stale parquet: %s", path)
            except Exception as exc:
                log.warning("purge_volume_folder: could not delete %s: %s", path, exc)
    log.info("Purged old parquets from %s", folder)


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
# Direct SQL warehouse MERGE (no pre-created job required)
# ---------------------------------------------------------------------------

def merge_via_warehouse(
    volume_folder: str,
    table: str,
    host: str,
    token: str,
    warehouse_id: str,
    *,
    poll_interval_s: int = _SQL_POLL_INTERVAL_S,
    timeout_s: int = _SQL_TIMEOUT_S,
) -> tuple[bool, str]:
    """MERGE parquet files from *volume_folder* into *table* via SQL warehouse.

    Creates the table from the Parquet schema if it does not yet exist, then
    runs a MERGE using dynamic column intersection so target-only columns are
    preserved on UPDATE and set to NULL on INSERT (same approach as the TCGA
    notebook fix for ``DELTA_MERGE_UNRESOLVED_EXPRESSION``).

    Returns ``(success, message)`` matching the ``poll_job_run`` contract.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = host.rstrip("/")

    def _exec(statement: str) -> tuple[bool, str]:
        """Submit a SQL statement and poll to completion. Returns (ok, message)."""
        payload = {
            "warehouse_id": warehouse_id,
            "statement": statement,
            "wait_timeout": "0s",  # async — poll below
        }
        resp = requests.post(f"{base}/api/2.0/sql/statements", headers=headers,
                             json=payload, timeout=30)
        resp.raise_for_status()
        stmt_id = resp.json()["statement_id"]

        deadline = time.monotonic() + timeout_s
        while True:
            r = requests.get(f"{base}/api/2.0/sql/statements/{stmt_id}",
                             headers=headers, timeout=30)
            r.raise_for_status()
            body   = r.json()
            state  = body.get("status", {}).get("state", "")
            if state == "SUCCEEDED":
                return True, f"Statement {stmt_id} succeeded"
            if state in ("FAILED", "CANCELED", "CLOSED"):
                err = body.get("status", {}).get("error", {})
                return False, err.get("message", f"Statement {stmt_id} {state}")
            if time.monotonic() > deadline:
                return False, f"Statement {stmt_id} timed out after {timeout_s}s (state={state})"
            time.sleep(poll_interval_s)

    def _fetch_one(statement: str) -> list[list]:
        """Run a query and return rows as list-of-lists."""
        payload = {
            "warehouse_id": warehouse_id,
            "statement": statement,
            "wait_timeout": "30s",
        }
        resp = requests.post(f"{base}/api/2.0/sql/statements", headers=headers,
                             json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        return [r for r in (body.get("result", {}).get("data_array") or [])]

    folder = volume_folder.rstrip("/")

    # 1. Create table from Parquet schema if it doesn't exist yet.
    ok, msg = _exec(
        f"CREATE TABLE IF NOT EXISTS {table} "
        f"USING DELTA AS SELECT * FROM parquet.`{folder}/` WHERE 1=0"
    )
    if not ok:
        return False, f"CREATE TABLE failed: {msg}"

    # 2. Get source columns from a sample row description.
    source_cols: list[str] = []
    try:
        rows = _fetch_one(f"SELECT * FROM parquet.`{folder}/` LIMIT 0")
        # Column names come from the schema, not data rows — use DESCRIBE instead.
        desc_rows = _fetch_one(
            f"DESCRIBE SELECT * FROM parquet.`{folder}/` LIMIT 0"
        )
        source_cols = [r[0] for r in desc_rows if r and r[0] and not r[0].startswith("#")]
    except Exception:
        pass

    target_cols: list[str] = []
    try:
        desc_rows = _fetch_one(f"DESCRIBE TABLE {table}")
        target_cols = [r[0] for r in desc_rows if r and r[0] and not r[0].startswith("#")]
    except Exception:
        pass

    if source_cols and target_cols:
        common = [c for c in source_cols if c in set(target_cols)]
        extra_target = [c for c in target_cols if c not in set(source_cols)]
        set_clause    = ", ".join(f"t.{c} = s.{c}" for c in common)
        insert_cols   = ", ".join(common + [c for c in extra_target])
        insert_vals   = ", ".join(
            [f"s.{c}" for c in common] + ["NULL" for _ in extra_target]
        )
        merge_sql = (
            f"MERGE INTO {table} t "
            f"USING (SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY slide_id, model ORDER BY completed_at DESC NULLS LAST) AS _rn FROM parquet.`{folder}/`) WHERE _rn = 1) s "
            f"ON t.slide_id = s.slide_id AND t.model = s.model "
            f"WHEN MATCHED THEN UPDATE SET {set_clause} "
            f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
        )
    else:
        # Fallback: optimistic wildcard MERGE (works when schemas match exactly).
        merge_sql = (
            f"MERGE INTO {table} t "
            f"USING (SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY slide_id, model ORDER BY completed_at DESC NULLS LAST) AS _rn FROM parquet.`{folder}/`) WHERE _rn = 1) s "
            f"ON t.slide_id = s.slide_id AND t.model = s.model "
            f"WHEN MATCHED THEN UPDATE SET * "
            f"WHEN NOT MATCHED THEN INSERT *"
        )

    log.info("Running warehouse MERGE into %s from %s", table, folder)
    return _exec(merge_sql)


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
             "E.g. your_catalog.your_schema.tcga_slide_embeddings_v2",
    )
    parser.add_argument(
        "--job-id", default=None,
        help="Databricks job ID to trigger after upload (optional; "
             "if omitted and --warehouse-id is set, MERGE runs directly via the warehouse)",
    )
    parser.add_argument(
        "--warehouse-id", default=None,
        help="Databricks SQL warehouse ID for direct MERGE (used when --job-id is not set)",
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
            # Auto-create the UC volume if it doesn't exist yet
            warehouse_id = getattr(args, "warehouse_id", None)
            if warehouse_id:
                ensure_volume_exists(args.volume_folder, host, token, warehouse_id)
            # Remove stale parquets so MERGE always sees exactly one source file
            purge_volume_folder(folder, host, token)
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

        elif args.table and getattr(args, "warehouse_id", None):
            # No pre-created job — run MERGE directly via the SQL warehouse.
            folder = args.volume_folder.rstrip("/") if args.volume_folder else None
            if not folder:
                log.warning("--warehouse-id set but no --volume-folder; skipping MERGE")
            else:
                success, message = merge_via_warehouse(
                    folder, args.table, host, token, args.warehouse_id
                )
                if getattr(args, "status_file", None):
                    write_sync_status(
                        args.status_file,
                        job_id="warehouse:" + args.warehouse_id,
                        run_id="direct",
                        success=success,
                        message=message,
                        table=args.table,
                    )
                if not success:
                    raise SystemExit(f"Warehouse MERGE into {args.table} FAILED: {message}")

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
