#!/usr/bin/env python3
"""Build or update a per-slide status CSV by scanning a local results directory.

Reads tcga_inventory.csv and scans the nextflow results directory for completed
feature outputs (.features.pt files). Outputs tcga_status.csv with one row per
(slide, model) combination.

Usage
-----
    python tcga_update_status.py \\
        --inventory tcga_inventory.csv \\
        --results-dir /data/tcga-results \\
        --model-types ctranspath,uni2h \\
        --output tcga_status.csv
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

STATUS_COLUMNS = [
    "file_id",
    "slide_id",
    "project_id",
    "slide_type",
    "model",
    "status",       # pending | done
    "pt_path",
    "h5_path",
    "last_updated",
]

_SLIDE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2})")


def _slide_id_from_filename(file_name: str) -> str:
    """Extract the full TCGA slide barcode from a filename or bare barcode.

    TCGA filenames embed a full slide barcode before the first dot:
        TCGA-BR-A44T-01Z-00-DX1.<uuid>.svs  →  TCGA-BR-A44T-01Z-00-DX1

    Using the full barcode (rather than the 4-part sample barcode) is essential
    because a single patient-sample can have multiple slides (DX1, DX2, …) that
    would otherwise collapse to the same identifier.

    Also handles bare barcodes passed directly (no dot → returned unchanged).
    """
    return file_name.split(".")[0]


def _find_pt_files(results_dir: Path, model: str) -> dict[str, Path]:
    """Return {slide_id: pt_path} for all .features.pt files for a model.

    The pipeline publishes flat to features/<model>/*.features.pt.
    Also checks the legacy features/<model>/pt/ subdirectory layout.
    """
    model_dir = results_dir / "features" / model
    if not model_dir.exists():
        return {}
    result = {}
    # Flat layout (current): features/<model>/*.features.pt
    for pt_file in model_dir.glob("*.features.pt"):
        result[pt_file.name.replace(".features.pt", "")] = pt_file
    # Legacy layout: features/<model>/pt/*.features.pt
    pt_subdir = model_dir / "pt"
    if pt_subdir.exists():
        for pt_file in pt_subdir.rglob("*.features.pt"):
            result[pt_file.name.replace(".features.pt", "")] = pt_file
    return result


def _find_h5_files(results_dir: Path, model: str) -> dict[str, Path]:
    """Return {slide_id: h5_path} for all slide-level .h5 files for a model.

    The pipeline publishes flat to features/<model>/*.features.h5.
    Also checks the legacy features/<model>/h5/ subdirectory layout (*.patch.h5).
    """
    model_dir = results_dir / "features" / model
    if not model_dir.exists():
        return {}
    result = {}
    # Flat layout (current): features/<model>/*.features.h5
    for h5_file in model_dir.glob("*.features.h5"):
        result[h5_file.name.replace(".features.h5", "")] = h5_file
    # Legacy layout: features/<model>/h5/*.patch.h5
    h5_subdir = model_dir / "h5"
    if h5_subdir.exists():
        for h5_file in h5_subdir.rglob("*.patch.h5"):
            result[h5_file.name.replace(".patch.h5", "")] = h5_file
    return result


def _discover_models(results_dir: Path) -> list[str]:
    """Auto-discover model types by scanning results/features/*/ directories."""
    features_dir = results_dir / "features"
    if not features_dir.exists():
        return []
    return sorted(
        d.name for d in features_dir.iterdir()
        if d.is_dir() and (
            any(d.glob("*.features.pt"))        # flat layout
            or (d / "pt").exists()              # legacy layout
        )
    )


def _load_wds_manifest(manifest_path: str | None) -> dict[tuple[str, str], str]:
    """Load WDS manifest CSV → {(slide_id, model): wds_path}."""
    if not manifest_path:
        return {}
    p = Path(manifest_path)
    if not p.exists():
        log.warning("WDS manifest not found: %s", manifest_path)
        return {}
    df = pd.read_csv(p, dtype=str).fillna("")
    mapping = {}
    for _, row in df.iterrows():
        mapping[(row["slide_id"], row["model"])] = row.get("wds_path", "")
    log.info("Loaded WDS manifest: %d entries from %s", len(mapping), manifest_path)
    return mapping


def build_status(
    inventory_df: pd.DataFrame,
    results_dir: Path,
    model_types: list[str],
    wds_manifest: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Scan results_dir and WDS manifest for completed outputs."""
    now = pd.Timestamp.now().isoformat()
    records = []

    for model in model_types:
        pt_map = _find_pt_files(results_dir, model)
        h5_map = _find_h5_files(results_dir, model)
        log.info("Model %-20s  %d .pt files,  %d .h5 files", model, len(pt_map), len(h5_map))

        for _, row in inventory_df.iterrows():
            slide_id = _slide_id_from_filename(row["file_name"])
            pt_path_local = pt_map.get(slide_id)
            h5_path_local = h5_map.get(slide_id)
            wds_path = (wds_manifest or {}).get((slide_id, model), "")

            if pt_path_local is not None:
                # Local file present (not yet cleaned up) — use it.
                status = "done"
                pt_path = str(pt_path_local)
                h5_path = str(h5_path_local) if h5_path_local else ""
            elif wds_path:
                # In WDS on S3 — mark done with S3 path.
                status = "done"
                pt_path = wds_path
                h5_path = wds_path
            else:
                status = "pending"
                pt_path = ""
                h5_path = ""

            records.append({
                "file_id": row["file_id"],
                "slide_id": slide_id,
                "project_id": row.get("project_id", ""),
                "slide_type": row.get("slide_type", ""),
                "model": model,
                "status": status,
                "pt_path": pt_path,
                "h5_path": h5_path,
                "last_updated": now,
            })

    return pd.DataFrame(records, columns=STATUS_COLUMNS)


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    done = (df["status"] == "done").sum()
    pending = (df["status"] == "pending").sum()
    print(f"\nStatus: {done}/{total} done, {pending} pending\n")

    pivot = (
        df.groupby(["project_id", "model", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    print(pivot.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--inventory", default="tcga_inventory.csv",
                        help="Path to tcga_inventory.csv")
    parser.add_argument("--slide-type", default=None,
                        help="Filter inventory by slide_type prefix (e.g. 'DX' matches DX1, DX2, …). "
                             "Defaults to no filter.")
    parser.add_argument("--results-dir", required=True,
                        help="Local nextflow results directory (contains features/<model>/pt/)")
    parser.add_argument("--model-types", default=None,
                        help="Comma-separated model types to check. "
                             "Defaults to auto-discovery from results/features/*/pt/ dirs.")
    parser.add_argument("--output", default="tcga_status.csv",
                        help="Output CSV path (default: tcga_status.csv)")
    parser.add_argument("--wds-manifest", default=None,
                        help="Path to wds_manifest.csv (slide_id,model,wds_path). "
                             "Slides in the manifest are marked 'done' with the S3 WDS "
                             "shard path, even if local .pt files have been cleaned up.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    inventory_df = pd.read_csv(args.inventory, dtype=str).fillna("")
    log.info("Loaded %d slides from %s", len(inventory_df), args.inventory)

    if args.slide_type and "slide_type" in inventory_df.columns:
        before = len(inventory_df)
        inventory_df = inventory_df[inventory_df["slide_type"].str.startswith(args.slide_type, na=False)]
        log.info("Filtered to slide_type '%s*': %d → %d slides", args.slide_type, before, len(inventory_df))

    results_dir = Path(args.results_dir)
    wds_manifest = _load_wds_manifest(args.wds_manifest)

    if args.model_types:
        model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    else:
        model_types = _discover_models(results_dir)
        if model_types:
            log.info("Auto-discovered models: %s", ", ".join(model_types))
        else:
            log.warning("No completed model outputs found in %s — status will show all as pending", results_dir)
    status_df = build_status(inventory_df, Path(args.results_dir), model_types, wds_manifest=wds_manifest)
    print_summary(status_df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(args.output, index=False)
    log.info("Wrote %s (%d rows)", args.output, len(status_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
