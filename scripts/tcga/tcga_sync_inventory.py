#!/usr/bin/env python3
"""Sync TCGA slide inventory from the GDC API.

Queries the GDC API for all TCGA slide images and writes (or updates)
tcga_inventory.csv. On subsequent runs it diffs against the existing file
and reports what changed.

Usage
-----
    python tcga_sync_inventory.py --output tcga_inventory.csv
    python tcga_sync_inventory.py --output tcga_inventory.csv --project TCGA-BRCA
    python tcga_sync_inventory.py --output tcga_inventory.csv --show-diff

Exit codes
----------
    0  -- success, at least one new / updated slide found
    1  -- error
    2  -- success, no changes detected (useful for cron skip logic)
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_FIELDS = ",".join([
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "updated_datetime",
    # Case / project
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.project.primary_site",
    "cases.project.disease_type",
    # Demographics
    "cases.demographic.gender",
    "cases.demographic.age_at_index",
    "cases.demographic.vital_status",
    "cases.demographic.race",
    "cases.demographic.ethnicity",
    # Diagnosis
    "cases.diagnoses.primary_diagnosis",
    "cases.diagnoses.morphology",
    "cases.diagnoses.ajcc_pathologic_stage",
    "cases.diagnoses.tumor_grade",
    # Sample / slide pathology
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "cases.samples.tumor_descriptor",
    "cases.samples.portions.slides.section_location",
    "cases.samples.portions.slides.percent_tumor_cells",
    "cases.samples.portions.slides.percent_stromal_cells",
    "cases.samples.portions.slides.percent_necrosis",
    "cases.samples.portions.slides.percent_normal_cells",
])

INVENTORY_COLUMNS = [
    # File identity
    "file_id",
    "file_name",
    "slide_type",
    "file_size",
    "md5sum",
    "updated_datetime",
    # Case / project
    "case_submitter_id",
    "project_id",
    "primary_site",
    "disease_type",
    # Demographics
    "gender",
    "age_at_index",
    "vital_status",
    "race",
    "ethnicity",
    # Diagnosis
    "primary_diagnosis",
    "morphology",
    "ajcc_pathologic_stage",
    "tumor_grade",
    # Sample
    "sample_type",
    "tissue_type",
    "tumor_descriptor",
    # Slide-level pathology estimates
    "section_location",
    "percent_tumor_cells",
    "percent_stromal_cells",
    "percent_necrosis",
    "percent_normal_cells",
]

_SLIDE_TYPE_RE = re.compile(r"-([A-Z]{2}\d+)\.")


def _slide_type(file_name: str) -> str:
    """Extract slide type (DX1, DX2, BS1, TS1 …) from a TCGA filename.

    Example: TCGA-BR-A44T-01Z-00-DX1.1A2B3C4D.svs → 'DX1'
    """
    m = _SLIDE_TYPE_RE.search(file_name)
    return m.group(1) if m else ""


def _fetch_page(filters: dict, from_: int, size: int, retries: int = 3) -> dict:
    params = {
        "filters": json.dumps(filters),
        "fields": GDC_FIELDS,
        "size": size,
        "from": from_,
        "format": "json",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning("GDC API error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _first(lst: list, key: str, default="") -> str:
    """Return the first non-null value of *key* from a list of dicts."""
    for item in lst or []:
        v = item.get(key)
        if v is not None:
            return v
    return default


def _parse_hit(hit: dict) -> dict:
    cases = hit.get("cases") or [{}]
    case = cases[0]
    project = case.get("project") or {}
    demographic = case.get("demographic") or {}
    diagnoses = case.get("diagnoses") or [{}]
    samples = case.get("samples") or [{}]
    sample = samples[0]
    portions = sample.get("portions") or [{}]
    slides = (portions[0] or {}).get("slides") or [{}]
    slide_path = slides[0] or {}

    return {
        # File identity
        "file_id":        hit.get("file_id", ""),
        "file_name":      hit.get("file_name", ""),
        "slide_type":     _slide_type(hit.get("file_name", "")),
        "file_size":      hit.get("file_size", 0),
        "md5sum":         hit.get("md5sum", ""),
        "updated_datetime": hit.get("updated_datetime", ""),
        # Case / project
        "case_submitter_id": case.get("submitter_id", ""),
        "project_id":     project.get("project_id", ""),
        "primary_site":   project.get("primary_site", ""),
        "disease_type":   project.get("disease_type", ""),
        # Demographics
        "gender":         demographic.get("gender", ""),
        "age_at_index":   demographic.get("age_at_index", ""),
        "vital_status":   demographic.get("vital_status", ""),
        "race":           demographic.get("race", ""),
        "ethnicity":      demographic.get("ethnicity", ""),
        # Diagnosis (first entry that has the field)
        "primary_diagnosis":    _first(diagnoses, "primary_diagnosis"),
        "morphology":           _first(diagnoses, "morphology"),
        "ajcc_pathologic_stage": _first(diagnoses, "ajcc_pathologic_stage"),
        "tumor_grade":          _first(diagnoses, "tumor_grade"),
        # Sample
        "sample_type":     sample.get("sample_type", ""),
        "tissue_type":     sample.get("tissue_type", ""),
        "tumor_descriptor": sample.get("tumor_descriptor", ""),
        # Slide-level pathology estimates
        "section_location":       slide_path.get("section_location", ""),
        "percent_tumor_cells":    slide_path.get("percent_tumor_cells", ""),
        "percent_stromal_cells":  slide_path.get("percent_stromal_cells", ""),
        "percent_necrosis":       slide_path.get("percent_necrosis", ""),
        "percent_normal_cells":   slide_path.get("percent_normal_cells", ""),
    }


def fetch_inventory(project_filter: str | None = None, page_size: int = 500) -> pd.DataFrame:
    """Fetch the full TCGA slide inventory from the GDC API."""
    filters: dict = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "cases.project.program.name", "value": "TCGA"}},
        ],
    }
    if project_filter:
        filters["content"].append(
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project_filter}}
        )

    records: list[dict] = []
    from_ = 0
    total: int | None = None

    while True:
        data = _fetch_page(filters, from_=from_, size=page_size)
        hits = data["data"]["hits"]
        if total is None:
            total = data["data"]["pagination"]["total"]
            log.info("GDC: %d slides to fetch", total)
        for hit in hits:
            records.append(_parse_hit(hit))
        from_ += len(hits)
        log.info("  fetched %d / %d", from_, total)
        if not hits or from_ >= total:
            break

    df = pd.DataFrame(records, columns=INVENTORY_COLUMNS)
    df = df.drop_duplicates("file_id").sort_values("file_id").reset_index(drop=True)
    return df


def diff_inventory(
    old: pd.DataFrame, new: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (added, removed, updated) DataFrames."""
    old_ids = set(old["file_id"])
    new_ids = set(new["file_id"])
    added = new[new["file_id"].isin(new_ids - old_ids)]
    removed = old[old["file_id"].isin(old_ids - new_ids)]
    common_ids = old_ids & new_ids
    old_common = old[old["file_id"].isin(common_ids)].set_index("file_id")
    new_common = new[new["file_id"].isin(common_ids)].set_index("file_id")
    updated_ids = new_common.index[
        new_common["updated_datetime"] != old_common["updated_datetime"]
    ]
    updated = new[new["file_id"].isin(updated_ids)]
    return added, removed, updated


def _inventory_age_hours(path: Path) -> float | None:
    """Return age of *path* in hours, or None if it doesn't exist."""
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    return (now - mtime).total_seconds() / 3600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="tcga_inventory.csv",
                        help="Output CSV path (default: tcga_inventory.csv)")
    parser.add_argument("--project", default=None,
                        help="Filter to a single GDC project, e.g. TCGA-BRCA")
    parser.add_argument("--page-size", type=int, default=500,
                        help="GDC API page size (default: 500)")
    parser.add_argument("--show-diff", action="store_true",
                        help="Print file_id/name of newly added slides")
    parser.add_argument("--max-age-hours", type=float, default=24.0,
                        help="Skip API fetch if inventory CSV is younger than this many hours "
                             "(default: 24). Set to 0 to always fetch.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-fetch even if the cache is still fresh")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    output_path = Path(args.output)

    # ------------------------------------------------------------------
    # Cache check: skip API if inventory is fresh enough
    # ------------------------------------------------------------------
    if not args.force and args.max_age_hours > 0:
        age = _inventory_age_hours(output_path)
        if age is not None and age < args.max_age_hours:
            log.info(
                "Inventory is %.1f h old (limit %.1f h) — skipping GDC fetch. "
                "Use --force to override.",
                age, args.max_age_hours,
            )
            return 2

    log.info("Fetching TCGA slide inventory from GDC API…")
    new_df = fetch_inventory(project_filter=args.project, page_size=args.page_size)
    log.info("GDC returned %d slides", len(new_df))

    has_changes = True
    if output_path.exists():
        old_df = pd.read_csv(output_path, dtype=str).fillna("")
        added, removed, updated = diff_inventory(old_df, new_df)
        n_add, n_rem, n_upd = len(added), len(removed), len(updated)
        print(f"Diff: +{n_add} new, -{n_rem} removed, ~{n_upd} updated (total {len(new_df)})")
        if args.show_diff and n_add:
            print("\nAdded:")
            print(added[["file_id", "file_name", "project_id"]].to_string(index=False))
        has_changes = n_add > 0 or n_rem > 0 or n_upd > 0
    else:
        print(f"No existing inventory — writing {len(new_df)} slides")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_path, index=False)
    log.info("Wrote %s", output_path)

    return 0 if has_changes else 2


if __name__ == "__main__":
    sys.exit(main())
