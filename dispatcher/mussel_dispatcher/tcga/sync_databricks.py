#!/usr/bin/env python3
"""Export the TCGA feature inventory to a Databricks Unity Catalog volume.

Reads tcga_status.csv and tcga_inventory.csv, builds a Parquet export, and
uploads it to a Databricks Unity Catalog volume via the Files API. Optionally
triggers a Databricks job to MERGE the Parquet into a Delta table.

The Parquet is uploaded to a timestamped path inside the volume folder so
that all historical snapshots are preserved and the Databricks notebook can
always pick up the latest file:

    <volume_folder>/tcga_inventory_<YYYYMMDDTHHMMSS>.parquet

Credentials are resolved in order:
1. CLI flags ``--databricks-host`` / ``--token``
2. Environment variables ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN``
3. ``~/.databrickscfg`` ``[DEFAULT]`` section (``host`` and ``token`` keys)

Shared upload/trigger utilities live in
``mussel_dispatcher.databricks_sync``; this module only contains
TCGA-specific logic (``build_export``) and the CLI entry point.

Usage
-----
    python -m mussel_dispatcher.tcga.sync_databricks \\
        --status tcga_status.csv \\
        --inventory tcga_inventory.csv \\
        --volume-folder /Volumes/catalog/schema/tcga_dispatcher \\
        [--table cdsi_prod.pathology_data_mining.tcga_slide_embeddings_v2] \\
        [--job-id 12345]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from mussel_dispatcher.databricks_sync import add_upload_args, upload_and_trigger

log = logging.getLogger(__name__)

# Columns emitted first (from status), followed by all remaining inventory columns.
# Any inventory column not already in status is appended automatically so the
# export always carries the full clinical metadata.
STATUS_COLUMNS = [
    "slide_id", "file_id", "file_name",
    "project_id", "slide_type", "file_size",
    "model", "status", "failure_reason", "native_mpp", "mpp_is_fallback", "tiling_mpp", "wds_path", "last_updated",
]


def build_export(status_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Join status and inventory into a flat export DataFrame.

    All inventory columns are included; status columns take precedence for
    any overlapping fields (project_id, slide_type).
    """
    # Bring all inventory columns into the merge; status columns win on conflict.
    all_inv_cols = list(inventory_df.columns)

    merged = status_df.merge(
        inventory_df,
        on="file_id",
        how="left",
        suffixes=("", "_inv"),
    )

    # For columns present in both: fill blanks in status column from inventory.
    for col in ("project_id", "slide_type"):
        inv_col = f"{col}_inv"
        if inv_col in merged.columns:
            merged[col] = merged[col].replace("", None).fillna(merged[inv_col])
            merged = merged.drop(columns=[inv_col])

    # Drop any other _inv suffix duplicates (inventory columns already in status).
    dup_cols = [c for c in merged.columns if c.endswith("_inv")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # Put STATUS_COLUMNS first, then any extra inventory columns.
    leading = [c for c in STATUS_COLUMNS if c in merged.columns]
    extra = [c for c in merged.columns if c not in leading]
    result = merged[leading + extra]

    # Cast numeric columns: empty strings → NaN so parquet writes correct type.
    for col in ("native_mpp", "tiling_mpp", "file_size", "age_at_index",
                "percent_tumor_cells", "percent_stromal_cells",
                "percent_necrosis", "percent_normal_cells"):
        if col in result.columns:
            result = result.copy()
            result[col] = pd.to_numeric(result[col].replace("", None), errors="coerce")

    # Cast boolean columns: empty strings / None → pd.NA so parquet writes boolean.
    for col in ("mpp_is_fallback",):
        if col in result.columns:
            result = result.copy()
            result[col] = result[col].map(
                lambda v: True if v is True or str(v).lower() == "true"
                else (False if v is False or str(v).lower() == "false" else pd.NA)
            ).astype("boolean")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--status", default="tcga_status.csv")
    parser.add_argument("--inventory", default="tcga_inventory.csv")
    add_upload_args(parser)
    args = parser.parse_args(argv)

    if not args.volume_folder and not args.volume_path:
        parser.error("One of --volume-folder or --volume-path is required")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    status_df = pd.read_csv(args.status, dtype=str).fillna("")
    inventory_df = pd.read_csv(args.inventory, dtype=str).fillna("")
    log.info("Loaded %d status rows, %d inventory rows", len(status_df), len(inventory_df))

    export_df = build_export(status_df, inventory_df)
    log.info("Built export: %d rows", len(export_df))

    upload_and_trigger(export_df, args, filename_prefix="tcga_inventory_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
