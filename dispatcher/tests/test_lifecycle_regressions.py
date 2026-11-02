"""Regression tests for the local batch lifecycle (finalization retry, restart
recovery, shutdown during auto-resume, watcher resilience, PID identity and
execution capacity)."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from mussel_dispatcher import scheduler as sched
from mussel_dispatcher.batch_lifecycle import (
    BatchFinalizer,
    FinalizationContext,
    write_batch_csv,
)
from mussel_dispatcher.config import Config, WatcherConfig
from mussel_dispatcher.runner import NextflowRunner
from mussel_dispatcher.state import StateStore
from mussel_dispatcher.watchers import TcgaWatcher


def make_config(tmp_path: Path, **overrides) -> Config:
    defaults = dict(
        repo_dir=str(tmp_path / "repo"),
        nextflow_profiles="standard",
        outdir=str(tmp_path / "results"),
        work_base_dir=str(tmp_path / "work"),
        dispatch_dir=str(tmp_path / "dispatch"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )
    defaults.update(overrides)
    return Config(**defaults)


def add_dispatched_batch(state: StateStore, tmp_path: Path, batch_id: str = "batch-1",
                         *, work_dir: Path | None = None) -> Path:
    csv_path = tmp_path / f"{batch_id}.csv"
    write_batch_csv(csv_path, [{"slide_id": "a", "slide_path": "s3://slides/a.svs"}])
    state.add_slide("s3://slides/a.svs", "a")
    state.add_batch(
        batch_id,
        str(csv_path),
        str(work_dir or tmp_path / "work" / batch_id),
        1,
        str(tmp_path / f"batch_{batch_id}.log"),
    )
    assert state.mark_dispatched(["s3://slides/a.svs"], batch_id) == 1
    return csv_path


def slide_row(state: StateStore) -> dict:
    return dict(state._conn().execute(
        "SELECT status, fail_count, batch_id FROM slides WHERE slide_id='a'"
    ).fetchone())


# ---------------------------------------------------------------------------
# 1. A local FINALIZING batch must be retryable after its first attempt.
# ---------------------------------------------------------------------------

def _hook(tmp_path: Path, phase: str, flag: Path | None = None) -> dict:
    """A real hook command: fails as a transient error until ``flag`` exists."""
    if flag is None:
        code = "pass"
    else:
        code = (
            "import os, sys; ok = os.path.exists(%r); "
            "ok or sys.stderr.write('connection refused'); sys.exit(0 if ok else 1)" % str(flag)
        )
    return {"command": f'python -c "{code}"', "_phase": phase}


def _wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _make_retry_due(state: StateStore, batch_id: str) -> None:
    """Fast-forward both the batch and the hook backoff."""
    conn = state._conn()
    conn.execute("UPDATE batches SET finalization_retry_at='2000-01-01T00:00:00+00:00' "
                 "WHERE batch_id=?", (batch_id,))
    conn.execute("UPDATE batch_hooks SET next_retry_at=NULL WHERE batch_id=?", (batch_id,))
    conn.commit()


def _retry_delay_seconds(state: StateStore, batch_id: str) -> float:
    retry_at = state.get_batch(batch_id)["finalization_retry_at"]
    return datetime.fromisoformat(retry_at).timestamp() - time.time()


def test_retry_loop_replays_local_finalization_with_exponential_backoff(tmp_path):
    flag = tmp_path / "hook-ok"
    cfg = make_config(tmp_path, post_batch_hooks=[_hook(tmp_path, "wds", flag)])
    for name in ("results", "logs", "repo"):
        (tmp_path / name).mkdir()
    state = StateStore(str(tmp_path / "state.db"))
    add_dispatched_batch(state, tmp_path)
    state.set_batch_status("batch-1", "FINALIZING")
    rm = sched.RunManager(cfg, state)
    scheduler = sched.BatchScheduler(cfg, state, rm, threading.Event())

    def attempt_recorded(n):
        return lambda: (
            f"attempt {n}," in str(state.get_batch("batch-1")["finalization_error"] or "")
            and "batch-1" not in rm._finalization_pool._outcomes
        )

    try:
        scheduler._retry_finalizing_batches()
        assert _wait_for(attempt_recorded(1)), state.get_batch("batch-1")
        error = state.get_batch("batch-1")["finalization_error"]
        assert "TRANSIENT_SERVICE" in error and "after 1 attempt(s)" in error
        assert 40 <= _retry_delay_seconds(state, "batch-1") <= 61

        scheduler._retry_finalizing_batches()  # backoff not elapsed: nothing runs
        time.sleep(0.3)
        assert "attempt 1," in state.get_batch("batch-1")["finalization_error"]

        _make_retry_due(state, "batch-1")
        scheduler._retry_finalizing_batches()
        assert _wait_for(attempt_recorded(2)), state.get_batch("batch-1")
        assert 100 <= _retry_delay_seconds(state, "batch-1") <= 121
        assert slide_row(state)["fail_count"] == 0

        flag.touch()
        _make_retry_due(state, "batch-1")
        scheduler._retry_finalizing_batches()
        assert _wait_for(lambda: state.get_batch("batch-1")["status"] == "SUCCEEDED")
        assert _wait_for(lambda: "batch-1" not in rm._finalization_attempts)
        assert slide_row(state)["status"] == "SUCCEEDED"
    finally:
        rm.shutdown(wait=True)


def test_finalization_backoff_is_capped():
    rm = sched.RunManager.__new__(sched.RunManager)
    delays = [rm._finalization_retry_delay(n) for n in range(1, 10)]
    assert delays[:6] == [60, 120, 240, 480, 960, 1800]
    assert set(delays[5:]) == {1800}


# ---------------------------------------------------------------------------
# 2. Restart recovery replays FINALIZING batches after the work dir is gone.
# ---------------------------------------------------------------------------

def test_recover_in_flight_replays_finalizing_batch_without_work_dir(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    work_dir = tmp_path / "work" / "batch-1"  # never created: cleanup_work_dir removed it
    csv_path = add_dispatched_batch(state, tmp_path, work_dir=work_dir)
    state.set_batch_status("batch-1", "FINALIZING")

    resume_specs, finalize_specs = sched.recover_in_flight(
        state, deque(), retry_failed=False, cfg=make_config(tmp_path)
    )

    assert resume_specs == []
    assert [spec[:2] for spec in finalize_specs] == [("batch-1", str(csv_path))]
    assert state.get_batch("batch-1")["status"] == "FINALIZING"
    assert slide_row(state) == {"status": "DISPATCHED", "fail_count": 0, "batch_id": "batch-1"}


def test_recovered_finalizing_batch_finalizes_without_work_dir(tmp_path):
    cfg = make_config(tmp_path, post_batch_hooks=[_hook(tmp_path, "db_sync")])
    for name in ("results", "logs", "repo"):
        (tmp_path / name).mkdir()
    state = StateStore(str(tmp_path / "state.db"))
    work_dir = tmp_path / "work" / "batch-1"  # deleted when Nextflow exited 0
    add_dispatched_batch(state, tmp_path, work_dir=work_dir)
    state.set_batch_status("batch-1", "FINALIZING")

    _, finalize_specs = sched.recover_in_flight(state, deque(), retry_failed=False, cfg=cfg)
    rm = sched.RunManager(cfg, state)
    try:
        for batch_id, csv_path, log_path, wdir, started, nf_log in finalize_specs:
            assert rm.submit_finalize_recovered_success(
                batch_id, csv_path, log_path, wdir, run_started_at=started, nf_log_path=nf_log
            )
        assert _wait_for(lambda: state.get_batch("batch-1")["status"] == "SUCCEEDED"), \
            state.get_batch("batch-1")
    finally:
        rm.shutdown(wait=True)
    assert not work_dir.exists()
    assert slide_row(state)["status"] == "SUCCEEDED"
    hooks = state.get_batch_hooks(["batch-1"])["batch-1"]
    assert [h["status"] for h in hooks] == ["SUCCEEDED"]


def test_recover_in_flight_fails_finalizing_batch_whose_csv_is_gone(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    csv_path = add_dispatched_batch(state, tmp_path)
    csv_path.unlink()
    state.set_batch_status("batch-1", "FINALIZING")

    resume_specs, finalize_specs = sched.recover_in_flight(
        state, deque(), retry_failed=False, cfg=make_config(tmp_path)
    )

    assert resume_specs == [] and finalize_specs == []
    assert slide_row(state)["status"] == "PENDING"


# ---------------------------------------------------------------------------
# 3. WDS hook failures: transient ones keep the slides untouched for a retry;
#    coverage-only and terminal ones fall back to coverage reconciliation.
# ---------------------------------------------------------------------------

def _wds_finalizer(tmp_path: Path, state: StateStore, **overrides):
    outdir = tmp_path / "results"
    outdir.mkdir(exist_ok=True)
    cfg = make_config(
        tmp_path,
        outdir=str(outdir),
        hook_preflight=False,
        watchers=[WatcherConfig(type="s3", wds_destinations={"model-a": "s3://wds"})],
        post_batch_hooks=[
            {"command": "python -m mussel_dispatcher.wds append", "_phase": "wds"},
            {"command": "db-hook", "_phase": "db_sync"},
        ],
        **overrides,
    )
    csv_path = add_dispatched_batch(state, tmp_path)
    return BatchFinalizer(cfg, "batch-1", state), csv_path, cfg


def _wds_executor(returncode: int, stderr: str):
    def executor(cmd, **kwargs):
        if "mussel_dispatcher.wds" in cmd:
            return subprocess.CompletedProcess(cmd, returncode, "", stderr)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return executor


def _finalize(finalizer, csv_path, cfg, executor, verify_wds=None):
    return finalizer.finalize_success(
        FinalizationContext(batch_csv=str(csv_path), outdir=cfg.outdir, repo_dir=cfg.repo_dir),
        hook_executor=executor,
        collect_manifest=lambda started: None,
        verify_wds=verify_wds,
    )


def _clear_hook_backoff(state: StateStore) -> None:
    state._conn().execute("UPDATE batch_hooks SET next_retry_at=NULL")
    state._conn().commit()


@pytest.mark.parametrize("returncode,stderr,category", [
    (1, "botocore: connection refused by ECS endpoint", "TRANSIENT_SERVICE"),
    (78, "hook preflight failed", "RELEASE_OR_CONFIG"),
])
def test_transient_wds_failure_keeps_slides_and_skips_reconciliation(
    tmp_path, returncode, stderr, category
):
    state = StateStore(str(tmp_path / "state.db"))
    finalizer, csv_path, cfg = _wds_finalizer(tmp_path, state)
    verify_calls = []

    result = _finalize(
        finalizer, csv_path, cfg, _wds_executor(returncode, stderr),
        verify_wds=lambda batch_csv: verify_calls.append(batch_csv) or False,
    )

    assert result.wds_hooks_succeeded is False
    assert verify_calls == [], "a retryable failure must not trigger coverage reconciliation"
    assert state.get_batch("batch-1")["status"] == "FINALIZING"
    assert state.get_pending_slides() == []
    assert slide_row(state)["fail_count"] == 0
    assert slide_row(state)["status"] != "PENDING"
    # The status page shows why: hook, category and attempt count.
    assert category in result.detail and "after 1 attempt(s)" in result.detail


def test_coverage_only_wds_failure_still_reconciles_coverage(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    finalizer, csv_path, cfg = _wds_finalizer(tmp_path, state)
    verify_calls = []

    _finalize(
        finalizer, csv_path, cfg, _wds_executor(2, "missing outputs"),
        verify_wds=lambda batch_csv: verify_calls.append(batch_csv) or True,
    )

    assert verify_calls, "exit-2 coverage failures must still be reconciled"


@pytest.mark.parametrize("stderr,overrides", [
    # Unmatched text -> HOOK_BUG, which is never re-run automatically.
    ("Traceback: KeyError 'shard_file'", {}),
    # NODE_INFRA past hook_max_infra_retries.
    ("OSError: no space left on device", {"hook_max_infra_retries": 1}),
])
def test_terminal_wds_failure_reconciles_coverage_and_completes_batch(tmp_path, stderr, overrides):
    state = StateStore(str(tmp_path / "state.db"))
    finalizer, csv_path, cfg = _wds_finalizer(tmp_path, state, max_slide_retries=3, **overrides)
    # The WDS manifest does not list slide "a": it never reached WDS.

    result = _finalize(finalizer, csv_path, cfg, _wds_executor(1, stderr))

    assert result.wds_hooks_succeeded and result.db_sync_hooks_succeeded
    assert state.get_batch("batch-1")["status"] == "SUCCEEDED"
    row = slide_row(state)
    assert row["status"] == "PENDING" and row["fail_count"] == 1, "missing slide is requeued"
    hooks = state.get_batch_hooks(["batch-1"])["batch-1"]
    assert hooks[0]["status"] != "FAILED"  # waived after reconciliation


def test_terminal_wds_failure_requeue_stays_bounded_by_max_slide_retries(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    finalizer, csv_path, cfg = _wds_finalizer(tmp_path, state, max_slide_retries=1)
    state._conn().execute("UPDATE slides SET fail_count=1")
    state._conn().commit()

    _finalize(finalizer, csv_path, cfg, _wds_executor(1, "Traceback: KeyError"))

    assert slide_row(state)["status"] == "FAILED"
    assert state.get_pending_slides() == []


def test_transient_wds_failure_falls_back_after_attempt_cap(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    finalizer, csv_path, cfg = _wds_finalizer(
        tmp_path, state, max_slide_retries=3, wds_hook_max_attempts=3
    )
    executor = _wds_executor(1, "connection refused")

    for attempt in (1, 2):
        result = _finalize(finalizer, csv_path, cfg, executor)
        assert state.get_batch("batch-1")["status"] == "FINALIZING"
        assert slide_row(state)["fail_count"] == 0, f"slides touched on attempt {attempt}"
        assert f"after {attempt} attempt(s)" in result.detail
        _clear_hook_backoff(state)

    _finalize(finalizer, csv_path, cfg, executor)

    assert state.get_batch("batch-1")["status"] == "SUCCEEDED"
    assert slide_row(state)["status"] == "PENDING" and slide_row(state)["fail_count"] == 1


# ---------------------------------------------------------------------------
# 4. Shutdown during the auto-resume Nextflow run preserves recovery state.
# ---------------------------------------------------------------------------

def _make_runner(tmp_path: Path, shutdown_event: threading.Event):
    for name in ("batches", "state", "logs", "work"):
        (tmp_path / name).mkdir(exist_ok=True)
    cfg = make_config(
        tmp_path,
        repo_dir=str(tmp_path),
        dispatch_dir=str(tmp_path / "batches"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        work_base_dir=str(tmp_path / "work"),
    )
    state = StateStore(str(tmp_path / "state" / "test.db"))
    state.add_slide("/slides/a.svs", "a")
    runner = NextflowRunner(
        cfg, "batch-001", [{"slide_path": "/slides/a.svs", "slide_id": "a"}], state,
        shutdown_event=shutdown_event,
    )
    return runner, state


def _stop_for_restart_recovery(state, shutdown_event):
    """What RunManager.stop_for_restart_recovery() does before SIGTERMing NF."""
    state.set_batch_status("batch-001", "RECOVERING")
    shutdown_event.set()


def _run_with_procs(runner, waits):
    procs = []

    def fake_popen(*args, **kwargs):
        wait = waits[len(procs)]
        log_line = getattr(wait, "log_line", "")
        if log_line:  # NF writes to the log while running, before wait()
            kwargs["stdout"].write(log_line)
            kwargs["stdout"].flush()
        proc = mock.Mock()
        proc.pid = 1000 + len(procs)
        proc.wait.side_effect = wait
        procs.append(proc)
        return proc

    clock = iter([0.0] + [120.0 + i for i in range(100)])
    with mock.patch("subprocess.Popen", side_effect=fake_popen), \
         mock.patch("mussel_dispatcher.runner._query_session_id_via_nf_cli", return_value="sess-1"), \
         mock.patch("mussel_dispatcher.runner.time.time", side_effect=lambda: next(clock)), \
         mock.patch.object(runner, "_cleanup") as cleanup, \
         mock.patch.object(runner, "_cleanup_execution_artifacts") as cleanup_exec:
        exit_code = runner.run()
    return exit_code, procs, cleanup, cleanup_exec


def test_shutdown_during_auto_resume_preserves_recovering_batch(tmp_path):
    shutdown_event = threading.Event()
    runner, state = _make_runner(tmp_path, shutdown_event)

    def resume_terminated():
        _stop_for_restart_recovery(state, shutdown_event)
        return 143

    exit_code, procs, cleanup, cleanup_exec = _run_with_procs(
        runner, [lambda: 1, resume_terminated]
    )

    assert exit_code == 143
    assert len(procs) == 2
    assert state.get_batch("batch-001")["status"] == "RECOVERING"
    assert slide_row(state) == {"status": "DISPATCHED", "fail_count": 0, "batch_id": "batch-001"}
    cleanup.assert_not_called()
    cleanup_exec.assert_not_called()


def test_shutdown_during_session_lock_backoff_does_not_relaunch(tmp_path):
    class ShutdownDuringWait(threading.Event):
        def wait(self, timeout=None):
            _stop_for_restart_recovery(state, self)
            return True

    shutdown_event = ShutdownDuringWait()
    runner, state = _make_runner(tmp_path, shutdown_event)

    def resume_lock_failure():
        return 1

    resume_lock_failure.log_line = "ERROR ~ Unable to acquire lock on session with ID sess-1\n"

    with mock.patch("mussel_dispatcher.runner.time.sleep") as sleep:
        exit_code, procs, cleanup, _ = _run_with_procs(
            runner, [lambda: 1, resume_lock_failure, lambda: 0]
        )

    assert len(procs) == 2, "no Nextflow run may be launched during shutdown"
    sleep.assert_not_called()
    assert exit_code == 1
    assert state.get_batch("batch-001")["status"] == "RECOVERING"
    assert slide_row(state)["fail_count"] == 0
    cleanup.assert_not_called()


# ---------------------------------------------------------------------------
# 5. The TCGA watcher survives failing scripts and failing polls.
# ---------------------------------------------------------------------------

def _tcga_watcher(tmp_path: Path, stop_event: threading.Event) -> TcgaWatcher:
    cfg = WatcherConfig(
        type="tcga",
        poll_interval_seconds=0,
        inventory_csv=str(tmp_path / "inv.csv"),
        status_csv=str(tmp_path / "status.csv"),
    )
    state = StateStore(str(tmp_path / "state.db"))
    return TcgaWatcher(cfg, deque(), state, stop_event, str(tmp_path), str(tmp_path / "out"))


def test_tcga_run_script_logs_failure_without_name_error(tmp_path):
    watcher = _tcga_watcher(tmp_path, threading.Event())
    failed = subprocess.CompletedProcess([], 1, "", "boom")
    with mock.patch("mussel_dispatcher.watchers.subprocess.run", return_value=failed):
        assert watcher._run_script("sync_inventory", []) == 1


def test_tcga_watcher_keeps_polling_after_poll_exception(tmp_path):
    stop_event = threading.Event()
    watcher = _tcga_watcher(tmp_path, stop_event)
    calls = []

    def poll():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        stop_event.set()

    with mock.patch.object(watcher, "_poll", side_effect=poll):
        watcher.run()

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# 6. A recorded nf_pid reused by a non-Nextflow process is never killed.
# ---------------------------------------------------------------------------

@pytest.fixture
def child_process():
    procs = []

    def spawn(*argv):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", *argv]
        )
        procs.append(proc)
        # Right after exec the kernel can briefly report an empty cmdline.
        cmdline = Path(f"/proc/{proc.pid}/cmdline")
        assert _wait_for(lambda: cmdline.exists() and cmdline.read_bytes(), timeout=5)
        return proc

    yield spawn
    for proc in procs:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs procfs")
def test_nf_pid_alive_checks_process_identity(child_process):
    assert sched._nf_pid_alive(child_process().pid) is False
    assert sched._nf_pid_alive(child_process("nextflow.cli.Launcher").pid) is True


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs procfs")
def test_nf_pid_alive_requires_this_batchs_run_name_or_log(child_process):
    nf_args = ("nextflow", "-log", "/logs/batch_b1.nf.log", "run", "main.nf", "-name", "rabc123")
    pid = child_process(*nf_args).pid
    assert sched._nf_pid_alive(pid, run_name="rabc123") is True
    assert sched._nf_pid_alive(pid, run_name="rother99") is False, "another batch's Nextflow"
    assert sched._nf_pid_alive(pid, nf_log_path="/x/logs/batch_b1.nf.log") is True
    assert sched._nf_pid_alive(pid, nf_log_path="/x/logs/batch_b2.nf.log") is False
    assert sched._nf_pid_alive(pid) is True  # nothing recorded: bare nextflow check


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs procfs")
def test_kill_orphaned_nf_spares_other_batchs_nextflow(child_process, monkeypatch):
    monkeypatch.setattr(sched, "_scancel_slurm_jobs", lambda *a, **k: 0)
    other = child_process("nextflow", "run", "main.nf", "-name", "rother99")
    sched._kill_orphaned_nf("batch-1", other.pid, sigterm_wait=0.5, run_name="rabc123")
    assert other.poll() is None


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs procfs")
def test_kill_orphaned_nf_spares_reused_pid(child_process, monkeypatch):
    monkeypatch.setattr(sched, "_scancel_slurm_jobs", lambda *a, **k: 0)
    unrelated = child_process()
    sched._kill_orphaned_nf("batch-1", unrelated.pid, sigterm_wait=0.5)
    assert unrelated.poll() is None, "reused PID must not be signalled"


# ---------------------------------------------------------------------------
# 7. Executor size and startup resume cap follow the summed pool limits.
# ---------------------------------------------------------------------------

def _pooled_config(tmp_path: Path) -> Config:
    return make_config(
        tmp_path,
        max_concurrent_runs=2,
        execution_pools=[
            {"name": "local", "backend": "local", "max_concurrent_runs": 3},
            {"name": "hpc", "backend": "remote_agent", "target": "hpc",
             "max_concurrent_runs": 2},
        ],
    )


def test_run_manager_executor_runs_every_slot_concurrently(tmp_path):
    cfg = _pooled_config(tmp_path)
    state = StateStore(str(tmp_path / "state.db"))
    started = []
    release = threading.Event()

    class FakeRunner:
        def __init__(self, cfg, batch_id, *args, **kwargs):
            self.batch_id = batch_id

        def run(self):
            started.append(self.batch_id)
            release.wait(timeout=5)
            return 0

    rm = sched.RunManager(cfg, state)
    try:
        with mock.patch.object(sched, "NextflowRunner", FakeRunner):
            for i in range(5):
                assert rm.submit(f"b{i}", [], known_active_batches=0)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and len(started) < 5:
                time.sleep(0.05)
        assert len(started) == 5, f"only {len(started)} of 5 slot holders ran"
    finally:
        release.set()
        rm.shutdown(wait=True)


def test_startup_resume_cap_matches_execution_slots(tmp_path):
    specs = [(f"b{i}", "c.csv", "w") for i in range(7)]
    capped, excess = sched._split_startup_resumes(_pooled_config(tmp_path), specs)
    assert len(capped) == 5 and len(excess) == 2
