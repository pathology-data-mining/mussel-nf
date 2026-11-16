"""bin/summarize_task_profiles.py on a synthetic profile directory and trace."""
import csv
import importlib.util
import json
from pathlib import Path

BIN = Path(__file__).parent.parent / "bin" / "summarize_task_profiles.py"
spec = importlib.util.spec_from_file_location("summarize_task_profiles", BIN)
stp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stp)

T0 = 1_790_000_000.0


def make_task(root, name, process, gpu_util, stages, wall=100):
    """A profile dir as TaskProfile's EXIT trap leaves it; stages are (stage, start, end, fields) relative to T0."""
    d = root / name
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"process": process, "tag": "batch"}))
    (d / ".task_profile_start").write_text(f"{T0}\n")
    (d / ".task_profile_end").write_text(f"{T0 + wall}\n")
    with (d / ".task_profile_gpu.csv").open("w") as fh:
        for u in gpu_util:
            fh.write(f"2026/10/11 12:00:00.000, 0, NVIDIA H100, {u}, 30000\n")
    if stages is not None:
        with (d / ".task_timings.jsonl").open("w") as fh:
            for stage, a, b, fields in stages:
                fh.write(json.dumps({"stage": stage, **fields, "start": T0 + a, "end": T0 + b, "seconds": b - a}) + "\n")
    return d


PATCH_TASK = [
    ("model_load", 12, 40, {"model": "HOPTIMUS1"}),
    ("patch_features", 40, 70, {"model": "HOPTIMUS1", "slide": "a.svs", "tiles": 3000, "tiles_done": 3000,
                                "load_wait_s": 6.0, "model_s": 21.0, "write_s": 1.0, "gpu_mem_peak_mib": 9000.0}),
    ("patch_features", 70, 90, {"model": "HOPTIMUS1", "slide": "b.svs", "tiles": 2000, "tiles_done": 2000,
                                "load_wait_s": 4.0, "model_s": 15.0, "write_s": 0.5, "gpu_mem_peak_mib": 9500.0}),
]


def test_summary_stages_gpu_and_trace(tmp_path):
    make_task(tmp_path / "prof", "ab12345678", "MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH_PATCH",
              [0, 0, 80, 90, 100, 0], PATCH_TASK)
    trace = tmp_path / "trace.tsv"
    with trace.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["hash", "name", "realtime", "%cpu", "peak_rss", "rchar", "wchar"])
        w.writerow(["ab/123456", "FEATURIZE_BATCH_PATCH (1)", "1m 52s", "384.5%", "20 GB", "538 GB", "9.3 GB"])
        w.writerow(["ab/999999", "FEATURIZE_BATCH_PATCH (2)", "5s", "1%", "1 GB", "1 GB", "1 GB"])
    (row,), stages = stp.summarize(tmp_path / "prof", stp.read_trace(trace))
    assert row["process"] == "FEATURIZE_BATCH_PATCH" and row["wall_s"] == 100.0
    assert row["startup_s"] == 12.0            # python and imports before Mussel's first stage
    assert row["model_load_s"] == 28.0
    assert row["patch_features_n"] == 2 and row["patch_features_s_median"] == 25.0
    assert row["load_wait_share"] == 0.2 and row["model_share"] == 0.72   # 10/50, 36/50
    assert row["tiles_per_s"] == 100.0         # 5000 tiles in 50 s
    assert row["gpu_mem_peak_mib"] == 9500.0
    assert row["after_last_stage_s"] == 10.0
    assert row["gpu_util_mean"] == 45.0 and row["gpu_idle_share"] == 0.5
    assert row["realtime_s"] == 112.0 and row["rchar_gb"] == 538.0 and row["wchar_gb"] == 9.3
    assert row["outside_script_s"] == 12.0     # container start-up and staging
    assert [s["stage"] for s in stages] == ["model_load", "patch_features", "patch_features"]
    assert stages[1]["slide"] == "a.svs" and stages[1]["task_dir"] == "ab12345678"


def test_cli_prints_and_writes_csvs(tmp_path, capsys):
    for i in range(2):
        make_task(tmp_path / "prof", f"cd{i}0000000", "MUSSEL:TESSELLATE_PATCHES:TESSELLATE", [50],
                  [("model_load", 1, 5, {"model": "segmentation"}),
                   ("tessellate", 5, 30, {"slide": "x.svs", "patch_sizes": [224, 512], "tiles": {"224": 9, "512": 2}})])
    out, stages_out = tmp_path / "tasks.csv", tmp_path / "stages.csv"
    assert stp.main(["--profile-dir", str(tmp_path / "prof"), "--out-csv", str(out),
                     "--out-stages-csv", str(stages_out)]) == 0
    assert "TESSELLATE" in capsys.readouterr().out
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2 and rows[0]["tessellate_s_median"] == "25"
    stage_rows = list(csv.DictReader(stages_out.open()))
    assert len(stage_rows) == 4 and json.loads(stage_rows[1]["tiles"]) == {"224": 9, "512": 2}


def test_without_mussel_timings_only_task_columns(tmp_path):
    """An image whose Mussel has no timings_path: wall time and GPU only."""
    make_task(tmp_path / "prof", "ee00000000", "MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH_PATCH", [10], None)
    (row,), stages = stp.summarize(tmp_path / "prof", {})
    assert row["wall_s"] == 100.0 and row["gpu_util_mean"] == 10.0
    assert "startup_s" not in row and "model_load_s" not in row and stages == []


def test_bad_lines_and_failed_stages(tmp_path):
    d = make_task(tmp_path / "prof", "ff00000000", "MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH_SLIDE", [10],
                  [("slide_encode", 2, 3, {"model": "TITAN_SLIDE", "slide": "a", "tiles": 9000, "error": "OutOfMemoryError"})])
    with (d / ".task_timings.jsonl").open("a") as fh:
        fh.write('{"stage": "slide_enc')  # cut off when the task was killed
    (row,), _ = stp.summarize(tmp_path / "prof", {})
    assert row["slide_encode_n"] == 1 and row["stage_errors"] == 1


def test_missing_files_are_tolerated(tmp_path):
    (tmp_path / "prof" / "ef00000000").mkdir(parents=True)
    (row,), _ = stp.summarize(tmp_path / "prof", {})
    assert row["process"] == "?" and "wall_s" not in row
