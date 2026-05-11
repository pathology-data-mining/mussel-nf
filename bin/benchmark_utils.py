#!/usr/bin/env python3
"""Shared utilities for mussel-nf benchmark summarisation scripts.

Importable from any script in bin/ via:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from benchmark_utils import _safe, extract_metrics
"""

import math


def _safe(v):
    """Replace NaN/Inf with None for JSON-safe serialisation."""
    return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v


def extract_metrics(data, splits=("val", "test"), metrics=("auroc",)):
    """Extract mean/std/CI values from a benchmark JSON data structure.

    Parameters
    ----------
    data    : dict   – parsed results.json
    splits  : tuple  – which split keys to look up (e.g. "val", "test")
    metrics : tuple  – metric keys within each split (e.g. "auroc", "tile_auc_roc")

    Returns
    -------
    dict with flat keys: {split}_{metric}_mean, {split}_{metric}_std,
    and (for test) test_{metric}_ci95_lo / test_{metric}_ci95_hi when present.
    """
    row = {}
    for split in splits:
        for metric in metrics:
            if metric in data.get(split, {}):
                row[f"{split}_{metric}_mean"] = _safe(data[split][metric].get("mean"))
                row[f"{split}_{metric}_std"]  = _safe(data[split][metric].get("std"))
                if split == "test" and "bootstrap_ci_95" in data[split].get(metric, {}):
                    ci = data[split][metric]["bootstrap_ci_95"]
                    row[f"test_{metric}_ci95_lo"] = _safe(ci[0])
                    row[f"test_{metric}_ci95_hi"] = _safe(ci[1])
    return row
