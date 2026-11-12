"""Regressions for audit follow-ups: nf_pid write failures, launcher exit 2."""
from __future__ import annotations

import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

from mussel_dispatcher import runner as runner_mod
from mussel_dispatcher.config import RemoteTargetConfig
from mussel_dispatcher.hook_hardening import HookFailureCategory, classify_hook_failure
from mussel_dispatcher.hook_launcher import LAUNCHER, launch_command
from mussel_dispatcher.remote import RemoteAgent
from mussel_dispatcher.runner import NextflowRunner
from mussel_dispatcher.state import StateStore
from tests.test_batch_lifecycle import make_config

# A stand-in `nextflow`: each run counts itself, sleeps, records the slide status the
# database shows while it is still running, and exits with the next code in FAKE_NF_EXITS.
FAKE_NF = """#!/bin/bash
[ "$1" = log ] && exit 1
n=$(( $(cat "$FAKE_NF_DIR/count" 2>/dev/null || echo 0) + 1 ))
echo $n > "$FAKE_NF_DIR/count"
sleep 1.5
"$FAKE_NF_PY" -c 'import sqlite3, sys; print(sqlite3.connect(sys.argv[1]).execute("SELECT status FROM slides").fetchone()[0])' "$FAKE_NF_DB" >> "$FAKE_NF_DIR/seen"
codes=($FAKE_NF_EXITS)
echo done >> "$FAKE_NF_DIR/finished"
exit ${codes[$((n - 1))]}
"""


class FlakyPidState(StateStore):
    """set_batch_nf_pid raises for the listed call numbers (1-based), like a Postgres blip."""

    def __init__(self, path, fail_calls=(), fail_clear=False):
        super().__init__(path)
        self.fail_calls, self.fail_clear, self.pid_calls = set(fail_calls), fail_clear, []

    def set_batch_nf_pid(self, batch_id, pid):
        self.pid_calls.append(pid)
        if len(self.pid_calls) in self.fail_calls or (self.fail_clear and pid is None):
            raise RuntimeError("server closed the connection unexpectedly")
        return super().set_batch_nf_pid(batch_id, pid)


def _runner(tmp_path, monkeypatch, exits, **state_kw):
    for d in ("batches", "state", "logs", "work", "nf"):
        (tmp_path / d).mkdir()
    fake = tmp_path / "nf" / "nextflow"
    fake.write_text(FAKE_NF)
    fake.chmod(0o755)
    db = tmp_path / "state" / "test.db"
    monkeypatch.setenv("MUSSEL_NEXTFLOW_BIN", str(fake))
    monkeypatch.setenv("FAKE_NF_DIR", str(tmp_path / "nf"))
    monkeypatch.setenv("FAKE_NF_PY", sys.executable)
    monkeypatch.setenv("FAKE_NF_DB", str(db))
    monkeypatch.setenv("FAKE_NF_EXITS", exits)
    monkeypatch.setattr(runner_mod, "_NF_PID_WRITE_RETRY_SECONDS", 0.0)
    cfg = make_config(
        tmp_path,
        repo_dir=str(tmp_path),
        dispatch_dir=str(tmp_path / "batches"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        work_base_dir=str(tmp_path / "work"),
    )
    state = FlakyPidState(str(db), **state_kw)
    slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
    state.add_slide("/slides/a.svs", "a")
    return NextflowRunner(cfg, "batch-001", slides, state), state


def _nf_files(tmp_path, name):
    path = tmp_path / "nf" / name
    return path.read_text().split() if path.exists() else []


def _slide(state):
    return state._conn().execute(
        "SELECT status, fail_count FROM slides WHERE slide_path='/slides/a.svs'"
    ).fetchone()


# 4. A failed nf_pid write after Popen must not abandon a running Nextflow.
def test_pid_write_failure_after_launch_still_waits_for_nextflow(tmp_path, monkeypatch):
    runner, state = _runner(tmp_path, monkeypatch, "0", fail_calls={1})

    assert runner.run() == 0

    assert _nf_files(tmp_path, "finished") == ["done"]     # run() returned after Nextflow did
    assert _nf_files(tmp_path, "count") == ["1"]           # and launched it once
    assert "PENDING" not in _nf_files(tmp_path, "seen")    # slides not requeued while it ran
    assert _slide(state)["status"] == "SUCCEEDED"
    assert state.pid_calls[:2] == [state.pid_calls[0], state.pid_calls[0]]  # retried the record
    assert state.get_batch_nf_pid("batch-001") is None
    assert runner.work_may_be_running is False


def test_pid_clear_failure_does_not_raise_out_of_run(tmp_path, monkeypatch, caplog):
    runner, state = _runner(tmp_path, monkeypatch, "0", fail_clear=True)

    assert runner.run() == 0

    assert _slide(state)["status"] == "SUCCEEDED"
    assert state.pid_calls[1:] == [None] * runner_mod._NF_PID_WRITE_ATTEMPTS
    assert "could not write nf_pid=None" in caplog.text


def test_pid_write_failure_after_auto_resume_launch_still_waits(tmp_path, monkeypatch):
    # First run fails after "60 s"; the auto-resume's PID record (3rd write) blips.
    runner, state = _runner(tmp_path, monkeypatch, "1 0", fail_calls={3})
    calls = []

    def fake_time():
        calls.append(None)
        return time.time() - (1000 if len(calls) == 1 else 0)  # run_started_at

    class FakeClock(types.ModuleType):                    # only time() is faked
        def __getattr__(self, name):
            return getattr(time, name)

    clock = FakeClock("time")
    clock.time = fake_time
    monkeypatch.setattr(runner_mod, "time", clock)

    assert runner.run() == 0

    assert _nf_files(tmp_path, "count") == ["2"]
    assert _nf_files(tmp_path, "finished") == ["done", "done"]
    assert "PENDING" not in _nf_files(tmp_path, "seen")
    row = _slide(state)
    assert (row["status"], row["fail_count"]) == ("SUCCEEDED", 0)
    assert state.get_batch_nf_pid("batch-001") is None


# 5. Exit 2 from a launched WDS hook is coverage only for wds.main's validation failure.
WDS_HOOK = "python -m mussel_dispatcher.wds --config cfg.yaml append --model titan"


def _tail(proc):
    return ((proc.stderr or "") + (proc.stdout or ""))[-1000:]


def test_missing_launcher_file_is_release_not_coverage(tmp_path):
    missing = tmp_path / "removed-release" / "hook_launcher.py"
    proc = subprocess.run(
        [sys.executable, str(missing), "--release", str(tmp_path), "mussel_dispatcher.wds"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2 and "can't open file" in proc.stderr
    outcome = classify_hook_failure(proc.returncode, _tail(proc), hook_command=WDS_HOOK)
    assert outcome.category == HookFailureCategory.RELEASE_OR_CONFIG
    # Other "can't open file" text (e.g. a library warning) is not the interpreter's message.
    assert classify_hook_failure(2, "warning: can't open file cache, continuing",
                                 hook_command=WDS_HOOK).category == HookFailureCategory.COVERAGE_MISSING


def test_launcher_usage_error_is_release_not_coverage():
    proc = subprocess.run([sys.executable, LAUNCHER, "mussel_dispatcher.wds"],
                          capture_output=True, text=True)
    assert proc.returncode == 78 and "usage: hook_launcher.py" in proc.stderr
    outcome = classify_hook_failure(proc.returncode, _tail(proc), hook_command=WDS_HOOK)
    assert outcome.category == HookFailureCategory.RELEASE_OR_CONFIG
    # A launcher from before the exit-78 change exited 2 with the bare usage line.
    legacy = "usage: hook_launcher.py --release DIR MODULE [ARGS...]\n"
    assert classify_hook_failure(2, legacy, hook_command=WDS_HOOK).category == (
        HookFailureCategory.RELEASE_OR_CONFIG
    )


def test_wds_validation_failure_stays_coverage_missing(tmp_path, caplog):
    import logging

    from mussel_dispatcher.batch_lifecycle import PostBatchHooks
    from tests.test_review_followups import _real_wds_batch, _run_wds

    cfg, state, csv_path, _ = _real_wds_batch(tmp_path, {"model-a": "a", "model-b": "ab"})
    seen = {}

    def executor(cmd, **kwargs):
        if "mussel_dispatcher.wds" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            model, rc = _run_wds(cmd)
        seen[model] = (rc, caplog.text)
        return subprocess.CompletedProcess(cmd, rc, "", caplog.text)

    PostBatchHooks(cfg, "batch-1", state).run(str(csv_path), phase="wds", executor=executor)

    rc, text = seen["model-a"]                  # model-a lacks slide b: wds.main's real exit 2
    assert rc == 2 and "requested slide(s) have no index entry" in text
    categories = {r["command"].split("--model-type=")[1].split()[0]: r["failure_category"]
                  for r in state.get_batch_hooks(["batch-1"])["batch-1"] if "--model-type=" in r["command"]}
    assert categories["model-a"] == HookFailureCategory.COVERAGE_MISSING
    # And, conservatively, an exit 2 from a WDS hook with no recognizable marker.
    assert classify_hook_failure(2, "", hook_command=WDS_HOOK).category == (
        HookFailureCategory.COVERAGE_MISSING
    )


def test_remote_preflight_refuses_a_missing_launcher(tmp_path, monkeypatch):
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.target = RemoteTargetConfig(name="hpc")
    agent.state = MagicMock()
    agent.state.get_batch_hooks.return_value = {"batch-1": []}
    order = {"batch_id": "batch-1", "remote_run_dir": str(tmp_path / "run"),
             "remote_batch_csv": str(tmp_path / "run" / "batch.csv")}
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "batch.csv").write_text("slide_id\n")
    (tmp_path / "dispatcher" / "mussel_dispatcher").mkdir(parents=True)
    scripts = []

    def fake_run(args, **kwargs):
        if args[0] == "sbatch":
            scripts.append(Path(args[-1]).read_text())
            return subprocess.CompletedProcess(args, 0, "12345\n", "")
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "sacct":
            return subprocess.CompletedProcess(args, 0, "COMPLETED|0:0|node032\n", "")
        raise AssertionError(args)

    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", fake_run)
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.wds", "append"], str(tmp_path))
    agent._run_hook_slurm(order, cmd, cwd=str(tmp_path), env={}, hook_index=0)
    monkeypatch.undo()                                   # subprocess.run is real again
    script = scripts[0]
    assert "hook-launcher-missing" in script

    # Run the generated preflight (not the hook) against a launcher path that is gone.
    preflight = script.split("exec ", 1)[0].replace(LAUNCHER, str(tmp_path / "gone.py"))
    proc = subprocess.run(["bash", "-c", preflight], capture_output=True, text=True)
    assert proc.returncode == 78
    assert classify_hook_failure(proc.returncode, _tail(proc), hook_command=WDS_HOOK).category == (
        HookFailureCategory.RELEASE_OR_CONFIG
    )
    preflight_ok = script.split("exec ", 1)[0]
    assert subprocess.run(["bash", "-c", preflight_ok], capture_output=True).returncode == 0
