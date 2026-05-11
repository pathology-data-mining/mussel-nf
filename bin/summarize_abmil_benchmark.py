#!/usr/bin/env python3
"""Summarise ABMIL benchmark results across model types.

Usage:
    summarize_abmil_benchmark.py model1:results1.json model2:results2.json ...

Writes:
    abmil_benchmark_summary.csv
    abmil_benchmark_summary.json
"""

import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark_utils import _safe, extract_metrics  # noqa: E402

pairs_str = " ".join(sys.argv[1:])

rows = []
combined = {}
for token in pairs_str.split():
    model_name, json_path = token.split(":", 1)
    data = json.loads(pathlib.Path(json_path).read_text())
    combined[model_name] = data

    row = {"model": model_name}
    row.update(extract_metrics(data, splits=("val", "test"), metrics=("auroc",)))
    rows.append(row)

df = pd.DataFrame(rows).sort_values("test_auroc_mean", ascending=False)
df.to_csv("abmil_benchmark_summary.csv", index=False)

pathlib.Path("abmil_benchmark_summary.json").write_text(json.dumps(combined, indent=2))
