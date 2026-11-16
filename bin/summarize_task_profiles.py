#!/usr/bin/env python3
"""Summarize task profiles collected with params.profile_dir (lib/TaskProfile.groovy).

For each profiled task:
- wall time of the task script, from its start/end (inside the container,
  around Mussel: Python start-up and imports included);
- GPU utilization (mean, share of samples below 5%) and memory from nvidia-smi;
- Mussel's stage timings (timings_path, mussel.utils.timings; needs a Mussel
  with that option, otherwise these columns are empty):
    - startup_s: script start to the first Mussel stage (Python, imports);
    - model_load_s;
    - median seconds per slide for tessellate / patch_features / slide_encode;
    - load_wait_share, model_share: of patch-feature time, the share spent
      waiting for tiles from the DataLoader, and in the model;
    - tiles_per_s;
    - gpu_mem_peak_mib, as PyTorch allocated it;
    - after_last_stage_s: the last stage's end to the task's end (writing
      .pt files, cleanup, interpreter exit);
- from the Nextflow trace (--trace), joined by task hash: %cpu, peak RSS,
  bytes read/written (wchar shows model re-downloads), and outside_script_s:
  the trace's realtime minus the script's wall time (container start-up,
  input staging, output unstaging).

Prints a per-process summary. --out-csv writes one row per task, and
--out-stages-csv one row per Mussel stage.

Usage: summarize_task_profiles.py --profile-dir DIR [--trace trace.tsv] [--out-csv tasks.csv] [--out-stages-csv stages.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

SIZE = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
SLIDE_STAGES = ("tessellate", "patch_features", "slide_encode")


def parse_bytes(text: str) -> float | None:
    m = re.match(r"\s*([\d.]+)\s*(B|KB|MB|GB|TB)", text or "")
    return float(m.group(1)) * SIZE[m.group(2)] if m else None


def parse_duration(text: str) -> float | None:
    parts = re.findall(r"([\d.]+)(ms|s|m|h|d)", text or "")
    if not parts:
        return None
    return sum(float(v) * {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400}[u] for v, u in parts)


def read_float(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def gpu_stats(path: Path) -> dict:
    util, mem = [], []
    try:
        for row in csv.reader(path.open()):
            if len(row) >= 5:
                try:
                    util.append(float(row[3])); mem.append(float(row[4]))
                except ValueError:
                    continue
    except OSError:
        return {}
    if not util:
        return {}
    return {"gpu_util_mean": round(statistics.fmean(util), 1),
            "gpu_idle_share": round(sum(u < 5 for u in util) / len(util), 3),
            "gpu_mem_max_mib": max(mem), "gpu_samples": len(util)}


def read_stages(path: Path) -> list[dict]:
    """Mussel's timings file: one JSON object per stage; unreadable lines are skipped."""
    stages = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and {"stage", "start", "end"} <= record.keys():
            stages.append(record)
    return stages


def stage_stats(stages: list[dict], start: float | None, end: float | None) -> dict:
    if not stages:
        return {}
    out = {"stages": len(stages)}
    if start is not None:
        out["startup_s"] = round(min(s["start"] for s in stages) - start, 1)
    if end is not None:
        out["after_last_stage_s"] = round(end - max(s["end"] for s in stages), 1)
    out["model_load_s"] = round(sum(s.get("seconds", 0) for s in stages if s["stage"] == "model_load"), 1)
    for name in SLIDE_STAGES:
        secs = [s.get("seconds", 0) for s in stages if s["stage"] == name]
        if secs:
            out[f"{name}_n"] = len(secs)
            out[f"{name}_s_median"] = round(statistics.median(secs), 1)
    patch = [s for s in stages if s["stage"] == "patch_features"]
    total = sum(s.get("seconds", 0) for s in patch)
    if total > 0:
        out["load_wait_share"] = round(sum(s.get("load_wait_s", 0) for s in patch) / total, 3)
        out["model_share"] = round(sum(s.get("model_s", 0) for s in patch) / total, 3)
        out["tiles_per_s"] = round(sum(s.get("tiles_done", 0) for s in patch) / total, 1)
    peaks = [s["gpu_mem_peak_mib"] for s in stages if isinstance(s.get("gpu_mem_peak_mib"), (int, float))]
    if peaks:
        out["gpu_mem_peak_mib"] = max(peaks)
    errors = [s for s in stages if s.get("error")]
    if errors:
        out["stage_errors"] = len(errors)
    return out


def read_trace(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    """Trace rows by abbreviated hash ("ab/cdef12" -> "abcdef12"), a profile dir name's prefix."""
    rows = {}
    with path.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("hash"):
                rows[r["hash"].replace("/", "")] = r
    return rows


HASH_PREFIX = len("abcdef12")


def summarize(profile_dir: Path, trace: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """(one row per task, one row per Mussel stage)."""
    tasks, all_stages = [], []
    for d in sorted(p for p in profile_dir.iterdir() if p.is_dir()):
        meta = {}
        try:
            meta = json.loads((d / "task.json").read_text())
        except (OSError, ValueError):
            pass
        start, end = read_float(d / ".task_profile_start"), read_float(d / ".task_profile_end")
        row = {"task_dir": d.name, "process": meta.get("process", "?").split(":")[-1], "tag": meta.get("tag", "")}
        if start is not None and end is not None:
            row["wall_s"] = round(end - start, 1)
        row.update(gpu_stats(d / ".task_profile_gpu.csv"))
        stages = read_stages(d / ".task_timings.jsonl")
        row.update(stage_stats(stages, start, end))
        for s in stages:
            all_stages.append({"task_dir": d.name, "process": row["process"],
                               **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items()}})
        tr = trace.get(d.name[:HASH_PREFIX])
        if tr:
            row.update({"realtime_s": parse_duration(tr.get("realtime")), "cpu_pct": tr.get("%cpu"),
                        "peak_rss": tr.get("peak_rss"), "rchar_gb": round((parse_bytes(tr.get("rchar")) or 0) / 1e9, 2),
                        "wchar_gb": round((parse_bytes(tr.get("wchar")) or 0) / 1e9, 2)})
            if row["realtime_s"] is not None and "wall_s" in row:
                row["outside_script_s"] = round(row["realtime_s"] - row["wall_s"], 1)
        tasks.append(row)
    return tasks, all_stages


def print_summary(tasks: list[dict]) -> None:
    cols = ["wall_s", "outside_script_s", "gpu_util_mean", "gpu_idle_share", "startup_s", "model_load_s",
            "tessellate_s_median", "patch_features_s_median", "slide_encode_s_median",
            "load_wait_share", "tiles_per_s", "after_last_stage_s", "rchar_gb", "wchar_gb"]
    by = {}
    for t in tasks:
        by.setdefault(t["process"], []).append(t)
    width = max(len(c) for c in cols)
    print(f"{'process':28s} {'tasks':>5s} " + " ".join(f"{c:>{width}s}" for c in cols))
    for proc, ts in sorted(by.items()):
        vals = []
        for c in cols:
            v = [t[c] for t in ts if isinstance(t.get(c), (int, float))]
            vals.append(f"{statistics.median(v):{width}.2f}" if v else f"{'-':>{width}s}")
        print(f"{proc:28s} {len(ts):5d} " + " ".join(vals))
    print("(medians over tasks; times in s; stage columns need a Mussel with timings_path)")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", required=True, type=Path)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--out-stages-csv", type=Path)
    args = ap.parse_args(argv)
    tasks, stages = summarize(args.profile_dir, read_trace(args.trace))
    if not tasks:
        print(f"no task profiles in {args.profile_dir}", file=sys.stderr)
        return 1
    print_summary(tasks)
    if args.out_csv:
        write_csv(args.out_csv, tasks)
    if args.out_stages_csv:
        write_csv(args.out_stages_csv, stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
