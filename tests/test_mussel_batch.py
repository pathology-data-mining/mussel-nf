"""Unit tests for bin/mussel_batch.py (Mussel >= 1.6.0 partial batch outcomes)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "mussel_batch.py"
spec = importlib.util.spec_from_file_location("mussel_batch", SCRIPT)
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

MARKER = "failed (the others' outputs were written)"


def _h5(path: Path, rows: int, coords: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=np.zeros((rows, 4), dtype="float32"))
        if coords:
            f.create_dataset("coords", data=np.zeros((rows, 2), dtype="int64"))


def _pt(path: Path, rows: int):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(rows, 4), path)


def _finish(tmp_path, monkeypatch, rc, log, slide_ids, expects, scans=(".",)):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mussel.log").write_text(log)
    argv = ["finish", "--rc", str(rc), "--log", "mussel.log", "--slide-ids", ",".join(slide_ids),
            "--report", "failed_slides.tsv"]
    for e in expects:
        argv += ["--expect", e]
    for s in scans:
        argv += ["--scan", s]
    return mb.main(argv)


def _report(tmp_path):
    p = tmp_path / "failed_slides.tsv"
    return p.read_text().splitlines()[1:] if p.exists() else None


def test_success_without_gaps_writes_no_report(tmp_path, monkeypatch):
    for s in ("A", "B"):
        _h5(tmp_path / f"{s}.features.h5", 3)
        _pt(tmp_path / f"{s}.features.pt", 3)
    assert _finish(tmp_path, monkeypatch, 0, "", ["A", "B"], ["m={sid}.features.pt"]) == 0
    assert _report(tmp_path) is None


def test_partial_failure_is_a_success_with_a_report(tmp_path, monkeypatch):
    _h5(tmp_path / "A.features.h5", 3)
    _pt(tmp_path / "A.features.pt", 3)
    log = f"... RuntimeError: 1 of 3 slide(s) {MARKER}: B: CUDA out of memory\n"
    # C: no output and not named (e.g. skipped without tissue).
    rc = _finish(tmp_path, monkeypatch, 1, log, ["A", "B", "C"], ["m={sid}.features.pt"])
    assert rc == 0
    assert _report(tmp_path) == ["B\tm\tfailed", "C\tm\tmissing"]


def test_slide_id_prefix_is_not_taken_for_a_failure(tmp_path, monkeypatch):
    _pt(tmp_path / "S1.features.pt", 1)
    log = f"RuntimeError: 1 of 2 slide(s) {MARKER}: S10: boom\n"
    assert _finish(tmp_path, monkeypatch, 1, log, ["S1", "S10"], ["m={sid}.features.pt"]) == 0
    assert _report(tmp_path) == ["S10\tm\tfailed"]


def test_slide_id_inside_a_longer_failed_id_is_not_taken_for_a_failure(tmp_path, monkeypatch):
    """Numeric slide IDs: "1001234" must not match inside "11001234: ..."."""
    _pt(tmp_path / "A.features.pt", 1)
    log = f"RuntimeError: 1 of 3 slide(s) {MARKER}: 11001234: boom\n"
    rc = _finish(tmp_path, monkeypatch, 1, log, ["A", "1001234", "11001234"], ["m={sid}.features.pt"])
    assert rc == 0
    assert _report(tmp_path) == ["1001234\tm\tmissing", "11001234\tm\tfailed"]


@pytest.mark.parametrize("log, rc", [
    (f"RuntimeError: 2 of 2 slide(s) {MARKER}: A: x; B: y\n", 1),  # marker, but no outputs
    ("RuntimeError: none of the 2 slide(s) had tiles, so no output files were written\n", 1),
    ("Killed\n", 137),
])
def test_real_failures_keep_their_status(tmp_path, monkeypatch, log, rc):
    if rc == 137:
        _pt(tmp_path / "A.features.pt", 1)  # outputs, but no partial-failure message
    assert _finish(tmp_path, monkeypatch, rc, log, ["A", "B"], ["m={sid}.features.pt"]) == rc


def test_multi_model_needs_outputs_for_every_model(tmp_path, monkeypatch):
    for s in ("A", "B"):
        _pt(tmp_path / "optimus" / f"{s}.features.pt", 2)
    (tmp_path / "hoptimus1").mkdir()
    log = f"RuntimeError: 2 of 4 (slide, model) pair(s) {MARKER}: A (HOPTIMUS1): oom; B (HOPTIMUS1): oom\n"
    expects = ["optimus=optimus/{sid}.features.pt", "hoptimus1=hoptimus1/{sid}.features.pt"]
    assert _finish(tmp_path, monkeypatch, 1, log, ["A", "B"], expects, ["optimus", "hoptimus1"]) == 1

    _pt(tmp_path / "hoptimus1" / "A.features.pt", 2)
    log = f"RuntimeError: 1 of 4 (slide, model) pair(s) {MARKER}: B (HOPTIMUS1): oom\n"
    assert _finish(tmp_path, monkeypatch, 1, log, ["A", "B"], expects, ["optimus", "hoptimus1"]) == 0
    assert _report(tmp_path) == ["B\thoptimus1\tfailed"]


def test_empty_outputs_are_moved_aside(tmp_path, monkeypatch):
    # Identity: Mussel writes an empty (0, D) .features.h5 and an empty .pt.
    _h5(tmp_path / "A.features.h5", 3)
    _pt(tmp_path / "A.features.pt", 3)
    _h5(tmp_path / "E.features.h5", 0)
    _pt(tmp_path / "E.features.pt", 0)
    assert _finish(tmp_path, monkeypatch, 0, "", ["A", "E"], ["m={sid}.features.pt"]) == 0
    names = sorted(p.name for p in tmp_path.iterdir() if ".features." in p.name)
    assert names == ["A.features.h5", "A.features.pt", "E.features.h5.empty", "E.features.pt.empty"]
    assert _report(tmp_path) == ["E\tm\tno_tiles"]


def test_empty_outputs_in_pt_and_h5_dirs(tmp_path, monkeypatch):
    # One-step layout: pt/ and h5/; a .pt pairs with the .h5 of the same name. The
    # .pt is not a tensor here: it is judged by its .h5 alone.
    _h5(tmp_path / "h5" / "E.patch_features.h5", 0)
    (tmp_path / "pt").mkdir()
    (tmp_path / "pt" / "E.patch_features.pt").write_text("not loaded")
    _pt(tmp_path / "pt" / "A.features.pt", 1)
    _pt(tmp_path / "pt" / "Z.features.pt", 0)  # no .h5: loaded
    rc = _finish(tmp_path, monkeypatch, 0, "", ["A", "E"], ["s=pt/{sid}.features.pt"], ["pt", "h5"])
    assert rc == 0
    assert (tmp_path / "h5" / "E.patch_features.h5.empty").exists()
    assert (tmp_path / "pt" / "E.patch_features.pt.empty").exists()
    assert (tmp_path / "pt" / "Z.features.pt.empty").exists()
    assert (tmp_path / "pt" / "A.features.pt").exists()


def test_unreadable_outputs_are_kept(tmp_path, monkeypatch):
    (tmp_path / "A.features.h5").write_text("")
    (tmp_path / "A.features.pt").write_text("")
    assert _finish(tmp_path, monkeypatch, 0, "", ["A"], ["m={sid}.features.pt"]) == 0
    assert (tmp_path / "A.features.h5").exists() and (tmp_path / "A.features.pt").exists()


def test_run_copies_output_to_the_log_and_keeps_the_status(tmp_path):
    log = tmp_path / "cmd.log"
    cmd = [sys.executable, str(SCRIPT), "run", "--log", str(log), "--",
           "bash", "-c", "echo out; echo err >&2; exit 3"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 3
    assert result.stdout == "out\n" and result.stderr == "err\n"
    assert sorted(log.read_text().splitlines()) == ["err", "out"]


def test_run_reports_a_signal_as_the_shell_does(tmp_path):
    cmd = [sys.executable, str(SCRIPT), "run", "--log", str(tmp_path / "l"), "--", "bash", "-c", "kill -9 $$"]
    assert subprocess.run(cmd).returncode == 137
