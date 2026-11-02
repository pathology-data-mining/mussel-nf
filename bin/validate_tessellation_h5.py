#!/usr/bin/env python3
"""Reject tessellation outputs whose recorded backend differs from the request."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def validate(path: Path, expected_seg_model: str) -> str | None:
    expected = expected_seg_model.strip().lower()
    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                return "missing coords dataset"
            actual = handle["coords"].attrs.get("seg_model")
    except Exception as exc:
        return f"unreadable H5: {exc}"

    if isinstance(actual, bytes):
        actual = actual.decode("utf-8", "replace")
    normalized = str(actual).strip().lower() if actual is not None else ""
    if normalized != expected:
        return f"seg_model={normalized or '<missing>'}, expected {expected}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-seg-model", required=True)
    parser.add_argument("h5", nargs="+")
    args = parser.parse_args()

    failures = []
    for raw_path in args.h5:
        path = Path(raw_path)
        error = validate(path, args.expected_seg_model)
        if error:
            failures.append(f"{path}: {error}")
    if failures:
        print("Invalid tessellation metadata:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
