#!/usr/bin/env python3
"""Export the TCGA feature inventory to a Databricks Unity Catalog volume.

Reads tcga_status.csv and tcga_inventory.csv, builds a Parquet export, and
uploads it to a Databricks Unity Catalog volume via the Files API. Optionally
triggers a Databricks job to MERGE the Parquet into a Delta table.

The Parquet is uploaded to a timestamped path inside the volume folder so
that all historical snapshots are preserved and the Databricks notebook can
always pick up the latest file:

    <volume_folder>/tcga_inventory_<YYYYMMDDTHHMMSS>.parquet

Credentials are read from the DATABRICKS_HOST and DATABRICKS_TOKEN environment
variables, or passed as CLI flags.

Usage
-----
    python tcga_sync_databricks.py \\
        --status tcga_status.csv \\
        --inventory tcga_inventory.csv \\
        --volume-folder /Volumes/catalog/schema/tcga_dispatcher \\
        [--table cdsi_prod.pathology_data_mining.tcga_slide_embeddings_v2] \\
        [--job-id 12345]
"""

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

EXPORT_COLUMNS = [
    "slide_id", "file_id", "file_name",
    "project_id", "slide_type", "file_size",
    "model", "status", "pt_path", "h5_path", "last_updated",
]


def build_export(status_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Join status and inventory into a flat export DataFrame."""
    inv_cols = ["file_id", "file_name", "file_size"]
    # Add project_id / slide_type from inventory only if missing in status
    for col in ("project_id", "slide_type"):
        if col not in status_df.columns or status_df[col].eq("").all():
            inv_cols.append(col)

    merged = status_df.merge(
        inventory_df[list(dict.fromkeys(["file_id"] + inv_cols))],
        on="file_id",
        how="left",
        suffixes=("", "_inv"),
    )

    for col in ("project_id", "slide_type"):
        inv_col = f"{col}_inv"
        if inv_col in merged.columns:
            merged[col] = merged[col].fillna(merged[inv_col]).replace("", None)
            merged[col] = merged[col].fillna(merged[inv_col])
            merged = merged.drop(columns=[inv_col])

    return merged.reindex(columns=[c for c in EXPORT_COLUMNS if c in merged.columns])


def upload_parquet(local_path: Path, volume_path: str, host: str, token: str) -> None:
    """Upload a file to a Databricks Unity Catalog volume (Files API)."""
    url = f"{host.rstrip('/')}/api/2.0/fs/files{volume_path}"
    headers = {"Authorization": f"Bearer {token}"}
    with open(local_path, "rb") as f:
        resp = requests.put(url, headers=headers, data=f, timeout=300)
    resp.raise_for_status()
    log.info("Uploaded to Databricks: %s", volume_path)


def trigger_job(job_id: str, host: str, token: str, params: dict | None = None) -> str:
    """Trigger a Databricks job by ID and return the run_id."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--status", default="tcga_status.csv")
    parser.add_argument("--inventory", default="tcga_inventory.csv")
    parser.add_argument("--databricks-host", default=None,
                        help="Databricks workspace URL (or set DATABRICKS_HOST env var)")
    parser.add_argument("--token", default=None,
                        help="Databricks personal access token (or set DATABRICKS_TOKEN)")

    # Folder-based upload (timestamped files) — preferred
    parser.add_argument("--volume-folder", default=None,
                        help="UC volume folder to upload into; file will be named "
                             "tcga_inventory_<timestamp>.parquet")
    # Legacy: single overwritten file path
    parser.add_argument("--volume-path", default=None,
                        help="[Legacy] Full UC volume path for a single overwritten Parquet. "
                             "Use --volume-folder instead for timestamped uploads.")

    parser.add_argument("--table", default=None,
                        help="Target Delta table (passed as notebook_param 'target_table'). "
                             "E.g. cdsi_prod.pathology_data_mining.tcga_slide_embeddings_v2")
    parser.add_argument("--job-id", default=None,
                        help="Databricks job ID to trigger after upload (optional)")
    parser.add_argument("--output-parquet", default=None,
                        help="Also save the Parquet file locally at this path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if not args.volume_folder and not args.volume_path:
        parser.error("One of --volume-folder or --volume-path is required")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    host = args.databricks_host or os.environ.get("DATABRICKS_HOST", "")
    token = args.token or os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        log.error(
            "Databricks credentials required: set DATABRICKS_HOST / DATABRICKS_TOKEN "
            "or use --databricks-host / --token"
        )
        return 1

    status_df = pd.read_csv(args.status, dtype=str).fillna("")
    inventory_df = pd.read_csv(args.inventory, dtype=str).fillna("")
    log.info("Loaded %d status rows, %d inventory rows", len(status_df), len(inventory_df))

    export_df = build_export(status_df, inventory_df)
    log.info("Built export: %d rows", len(export_df))

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

        # Determine upload path — timestamped folder preferred
        if args.volume_folder:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            folder = args.volume_folder.rstrip("/")
            volume_path = f"{folder}/tcga_inventory_{ts}.parquet"
        else:
            volume_path = args.volume_path

        upload_parquet(tmp_path, volume_path, host, token)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


    if args.job_id:
        job_params: dict = {}
        if args.volume_folder:
            job_params["volume_folder"] = args.volume_folder
        if args.table:
            job_params["target_table"] = args.table
        trigger_job(args.job_id, host, token, params=job_params or None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
