"""Summarise ABMIL benchmark results across model/precision variants."""

import json
import pathlib
import sys

import pandas as pd

from .utils import extract_metrics


def summarize(pairs):
    """Build summary CSV and JSON from model:json_path pairs.

    Parameters
    ----------
    pairs : list of (model_name, json_path) tuples

    Writes
    ------
    abmil_benchmark_summary.csv
    abmil_benchmark_summary.json
    """
    rows = []
    combined = {}
    for model_name, json_path in pairs:
        data = json.loads(pathlib.Path(json_path).read_text())
        combined[model_name] = data
        row = {"model": model_name}
        row.update(extract_metrics(data, splits=("val", "test"), metrics=("auroc",)))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("test_auroc_mean", ascending=False)
    df.to_csv("abmil_benchmark_summary.csv", index=False)
    pathlib.Path("abmil_benchmark_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"Written: abmil_benchmark_summary.csv, abmil_benchmark_summary.json ({len(rows)} models)")


def main(argv=None):
    """CLI entry point: model1:results1.json model2:results2.json ..."""
    args = (argv if argv is not None else sys.argv[1:])
    if not args:
        print(f"Usage: summarize_abmil_benchmark.py model:results.json ...", file=sys.stderr)
        sys.exit(1)

    pairs = []
    for token in args:
        model_name, json_path = token.split(":", 1)
        pairs.append((model_name, json_path))

    summarize(pairs)
