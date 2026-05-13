#!/usr/bin/env python3
"""Export the IMPACT feature inventory to a Databricks Unity Catalog volume.

Reads the dispatcher SQLite database and the WDS manifest CSV, builds a
Parquet export (one row per slide × model), and uploads it to a Databricks
Unity Catalog volume. Optionally triggers a Databricks job to MERGE it into
a Delta table.

The schema mirrors the TCGA export but uses IMPACT-specific columns:
    slide_id       image_id from impact_matched_slides
    oncotree_code  cancer type code
    slide_path     original ECS S3 path
    model          feature model name (e.g. titan_slide)
    status         PENDING | SUCCEEDED | FAILED
    failure_reason non-empty when status == FAILED
    wds_path       ECS S3 path to the WDS shard (null when not yet complete)
    first_seen_at  ISO timestamp when the slide was first enqueued
    completed_at   ISO timestamp when the slide completed (or null)

Credentials resolved in order:
1. CLI flags --databricks-host / --token
2. Environment variables DATABRICKS_HOST / DATABRICKS_TOKEN
3. ~/.databrickscfg [DEFAULT] section

Usage
-----
    python -m mussel_dispatcher.impact.sync_databricks \\
        --db dispatcher.db \\
        --wds-manifest /path/to/wds_manifest.csv \\
        --model-types titan_slide,hoptimus1 \\
        --volume-folder /Volumes/catalog/schema/impact_dispatcher \\
        [--table cdsi_prod.pathology_data_mining.impact_slide_embeddings_v2] \\
        [--job-id 12345]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from mussel_dispatcher.databricks_sync import add_upload_args, upload_and_trigger
from mussel_dispatcher.state import StateStore

log = logging.getLogger(__name__)


def build_export(
    db_path: str,
    wds_manifest_path: str | None,
    model_types: list[str] | None,
) -> pd.DataFrame:
    """Build a flat export DataFrame from the state DB and WDS manifest.

    One row per (slide_id, model).  If *model_types* is empty or None,
    the distinct set of models already present in the WDS manifest (or a
    single synthetic placeholder ``"unknown"``) is used.

    Parameters
    ----------
    db_path:
        Path to the dispatcher SQLite database.
    wds_manifest_path:
        Path to ``wds_manifest.csv`` (columns: slide_id, model, wds_path).
        Missing or non-existent file is handled gracefully.
    model_types:
        List of model names to fan out to.  If None/empty, models are
        inferred from the WDS manifest; if the manifest is also missing,
        a single placeholder model ``"unknown"`` is used.
    """
    store = StateStore(db_path)
    slides = store.get_all_slides()

    # Build slide lookup: slide_id → row dict (last write wins for duplicates)
    slide_lookup: dict[str, dict] = {}
    for row in slides:
        sid = row.get("slide_id") or ""
        if sid:
            slide_lookup[sid] = row

    # Load WDS manifest
    wds_df: pd.DataFrame | None = None
    if wds_manifest_path:
        p = Path(wds_manifest_path)
        if p.exists():
            wds_df = pd.read_csv(p, dtype=str).fillna("")
        else:
            log.warning("WDS manifest not found: %s — wds_path will be null", wds_manifest_path)

    # Resolve model list
    if not model_types:
        if wds_df is not None and "model" in wds_df.columns and not wds_df.empty:
            model_types = sorted(wds_df["model"].unique().tolist())
        else:
            model_types = ["unknown"]

    # Build (slide_id, model) → wds_path index
    wds_index: dict[tuple[str, str], str] = {}
    if wds_df is not None and "slide_id" in wds_df.columns and "model" in wds_df.columns:
        for _, r in wds_df.iterrows():
            wds_index[(r["slide_id"], r["model"])] = r.get("wds_path", "")

    # Fan out: one row per (slide_id, model)
    rows = []
    for slide_id, slide in slide_lookup.items():
        db_status = slide.get("status") or "PENDING"
        if db_status == "SUCCEEDED":
            status = "SUCCEEDED"
        elif db_status == "FAILED":
            status = "FAILED"
        else:
            status = "PENDING"

        failure_reason = slide.get("error_msg") or ""
        if status != "FAILED":
            failure_reason = ""

        for model in model_types:
            wds_path = wds_index.get((slide_id, model)) or None
            rows.append({
                "slide_id": slide_id,
                "oncotree_code": slide.get("oncotree_code") or "",
                "slide_path": slide.get("slide_path") or "",
                "model": model,
                "status": status,
                "failure_reason": failure_reason,
                "wds_path": wds_path,
                "first_seen_at": slide.get("first_seen_at"),
                "completed_at": slide.get("completed_at"),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "slide_id", "oncotree_code", "slide_path", "model", "status",
        "failure_reason", "wds_path", "first_seen_at", "completed_at",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default="dispatcher.db",
                        help="Path to the dispatcher SQLite database")
    parser.add_argument("--wds-manifest", default=None,
                        help="Path to wds_manifest.csv (slide_id, model, wds_path)")
    parser.add_argument("--model-types", default=None,
                        help="Comma-separated model types, e.g. titan_slide,hoptimus1. "
                             "Inferred from wds_manifest if omitted.")
    add_upload_args(parser)
    args = parser.parse_args(argv)

    if not args.volume_folder and not args.volume_path:
        parser.error("One of --volume-folder or --volume-path is required")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    model_types = (
        [m.strip() for m in args.model_types.split(",") if m.strip()]
        if args.model_types
        else None
    )

    export_df = build_export(args.db, args.wds_manifest, model_types)
    log.info("Built export: %d rows", len(export_df))

    upload_and_trigger(export_df, args, filename_prefix="impact_inventory_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
