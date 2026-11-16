#!/usr/bin/env python3
"""Inventory tessellation H5 files by their recorded segmentation model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import h5py


def inspect_h5(path: Path) -> tuple[str, str]:
    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                return "", "missing coords dataset"
            value = handle["coords"].attrs.get("seg_model")
    except Exception as exc:
        return "", f"unreadable H5: {exc}"
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return (str(value).strip().lower() if value is not None else ""), ""


def audit(root: Path, output: Path, expected: str) -> dict[str, int]:
    expected = expected.strip().lower()
    counts: Counter[str] = Counter()
    rows = []
    for path in sorted(root.rglob("*.patch.h5")):
        actual, error = inspect_h5(path)
        status = "error" if error else ("matching" if actual == expected else "mismatch")
        counts[status] += 1
        counts[f"model:{actual or '<missing>'}"] += 1
        if status != "matching":
            rows.append(
                {
                    "slide_id": path.name.removesuffix(".patch.h5"),
                    "h5_path": str(path),
                    "actual_seg_model": actual,
                    "expected_seg_model": expected,
                    "status": status,
                    "error": error,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slide_id",
                "h5_path",
                "actual_seg_model",
                "expected_seg_model",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    counts["total"] = sum(counts[key] for key in ("matching", "mismatch", "error"))
    counts["reported"] = len(rows)
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-seg-model", default="neural")
    args = parser.parse_args()

    if not args.h5_root.is_dir():
        parser.error(f"H5 root is not a directory: {args.h5_root}")
    summary = audit(args.h5_root, args.output, args.expected_seg_model)
    print(json.dumps(summary, sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
