#!/usr/bin/env python3
"""Build a canonical WDS manifest from live WDS index snapshots.

Reads per-model ``wds_index.json`` files from the dispatcher staging root and
writes a one-row-per-(slide_id, model) CSV with columns:

    slide_id,model,wds_path

This is the authoritative source for a cohort's WDS contents because the
append-only ``wds_manifest.csv`` only reflects historical hook appends and can
lag or omit live index state.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_model_rows(index_root: Path, wds_prefix: str, model: str) -> list[dict[str, str]]:
    index_path = index_root / model / "wds_index.json"
    payload = json.loads(index_path.read_text())
    rows: list[dict[str, str]] = []
    for slide_id, info in sorted(payload.items(), key=lambda item: str(item[0])):
        if str(slide_id).startswith("_") or not isinstance(info, dict):
            continue
        shard_file = str(info.get("shard_file") or "").strip()
        if not shard_file:
            continue
        rows.append(
            {
                "slide_id": str(slide_id).strip(),
                "model": model,
                "wds_path": f"{wds_prefix.rstrip('/')}/{model}/{shard_file}",
            }
        )
    return rows


def build_manifest(
    *,
    index_root: Path,
    wds_prefix: str,
    models: list[str],
    output_manifest: Path,
    report_json: Path,
) -> dict[str, object]:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["slide_id", "model", "wds_path"]
    model_counts: dict[str, int] = {}
    unique_slides: set[str] = set()
    total_rows = 0

    with output_manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            rows = _load_model_rows(index_root, wds_prefix, model)
            model_counts[model] = len(rows)
            total_rows += len(rows)
            unique_slides.update(row["slide_id"] for row in rows)
            writer.writerows(rows)

    report = {
        "index_root": str(index_root),
        "wds_prefix": wds_prefix,
        "models": models,
        "model_counts": model_counts,
        "total_rows": total_rows,
        "unique_slide_ids": len(unique_slides),
        "output_manifest": str(output_manifest),
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--wds-prefix", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model list")
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()

    report = build_manifest(
        index_root=args.index_root,
        wds_prefix=args.wds_prefix,
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        output_manifest=args.output_manifest,
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
