"""Regressions for a dispatcher restart: preserved batches, and staleness by progress.

The service stop SIGTERMed every Nextflow alongside the dispatcher. Workers whose run
ended before stop_for_restart_recovery() reached their batch booked it FAILED, charged
fail_count and deleted its work dir; restart recovery then reset the survivors as
"stale" because they were created more than 24 h earlier, although they had only been
queued behind GPU contention.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mussel_dispatcher import runner as runner_mod
from mussel_dispatcher import scheduler as sched
from mussel_dispatcher.runner import NextflowRunner
from mussel_dispatcher.scheduler import RunManager, recover_in_flight
from mussel_dispatcher.state import StateStore
from tests.test_batch_lifecycle import make_config

# A stand-in `nextflow`: creates its work dir, then runs until SIGTERM and exits 1,
# as Nextflow does when aborted. A -resume run records itself and succeeds.
FAKE_NF = """#!/bin/bash
[ "$1" = log ] && exit 1
work=""; prev=""
for a in "$@"; do [ "$prev" = -work-dir ] && work="$a"; prev="$a"; done
mkdir -p "$work/ab/cdef"
case " $* " in *" -resume "*) echo "$work" >> "$FAKE_NF_DIR/resumed"; exit 0;; esac
trap 'exit 1' TERM
echo "$work" >> "$FAKE_NF_DIR/started"
while :; do sleep 0.05; done
"""


class GatedState(StateStore):
    """Holds shutdown's preservation write for chosen batches until a gate opens."""

    def __init__(self, path):
        super().__init__(path)
        self.gates: dict[str, threading.Event] = {}

    def _gate(self, batch_id, status):
        gate = self.gates.get(batch_id)
        if gate is not None and status in ("RECOVERING", "FINALIZING"):
            assert gate.wait(15), f"{batch_id}: gate never opened"

    def preserve_batch_for_recovery(self, batch_id, status):
        self._gate(batch_id, status)
        return super().preserve_batch_for_recovery(batch_id, status)

    def set_batch_status(self, batch_id, status):
        self._gate(batch_id, status)
        return super().set_batch_status(batch_id, status)


def _env(tmp_path, monkeypatch, **cfg_kw):
    for d in ("batches", "state", "logs", "work", "nf", "results"):
        (tmp_path / d).mkdir(exist_ok=True)
    fake = tmp_path / "nf" / "nextflow"
    fake.write_text(FAKE_NF)
    fake.chmod(0o755)
    monkeypatch.setenv("MUSSEL_NEXTFLOW_BIN", str(fake))
    monkeypatch.setenv("FAKE_NF_DIR", str(tmp_path / "nf"))
    monkeypatch.setattr(runner_mod, "_query_session_id_via_nf_cli", lambda *a: "sess-1")
    cfg_kw.setdefault("cleanup_work_dir", True)
    return make_config(
        tmp_path,
        repo_dir=str(tmp_path),
        dispatch_dir=str(tmp_path / "batches"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        work_base_dir=str(tmp_path / "work"),
        outdir=str(tmp_path / "results"),
        max_concurrent_runs=4,
        **cfg_kw,
    )


def _batch_id(suffix, hours_ago=0.0):
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return ts.strftime("%Y%m%dT%H%M%S") + "_" + suffix


def _submit(manager, state, batch_id, slide):
    state.add_slide(f"/slides/{slide}.svs", slide)
    assert manager.submit(
        batch_id, [{"slide_path": f"/slides/{slide}.svs", "slide_id": slide}],
        known_active_batches=0,
    )


def _wait_running(state, batch_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        pid = state.get_batch_nf_pid(batch_id)
        batch = state.get_batch(batch_id) or {}
        if pid and batch.get("work_dir") and os.path.isdir(batch["work_dir"]):
            return pid
        time.sleep(0.05)
    raise AssertionError(f"{batch_id}: Nextflow never started")


def _slide(state, slide):
    row = state._conn().execute(
        "SELECT status, fail_count FROM slides WHERE slide_id=?", (slide,)
    ).fetchone()
    return row["status"], row["fail_count"]


def _assert_preserved(state, batch_id, slide):
    batch = state.get_batch(batch_id)
    assert batch["status"] == "RECOVERING", batch_id
    assert batch["nextflow_exit"] is None
    assert _slide(state, slide) == ("DISPATCHED", 0)
    assert os.path.isdir(batch["work_dir"]), f"{batch_id}: work dir deleted"


def test_service_stop_preserves_batches_whatever_the_exit_order(tmp_path, monkeypatch):
    """A's run ends after shutdown preserved it; B's run ends first (the stop SIGTERMed
    it directly) and its worker finishes before shutdown reaches B."""
    cfg = _env(tmp_path, monkeypatch)
    state = GatedState(str(tmp_path / "state" / "test.db"))
    manager = RunManager(cfg, state)
    a, b = _batch_id("aaaa0001"), _batch_id("bbbb0002")
    try:
        _submit(manager, state, a, "sa")
        _submit(manager, state, b, "sb")
        _wait_running(state, a)
        pid_b = _wait_running(state, b)
        b_done = threading.Event()
        state.gates[b] = b_done
        manager._futures[b].add_done_callback(lambda _f: b_done.set())

        manager.begin_shutdown()             # the signal handler
        os.kill(pid_b, signal.SIGTERM)       # systemd signals the whole control group
        preserved, failed = manager.stop_for_restart_recovery()
    finally:
        manager.shutdown(wait=True)

    assert (preserved, failed) == (2, 0)
    _assert_preserved(state, a, "sa")
    _assert_preserved(state, b, "sb")

    # Restart recovery resumes both from their work dirs.
    resume_specs, _ = recover_in_flight(state, deque(), cfg=cfg)
    assert sorted(spec[0] for spec in resume_specs) == sorted([a, b])
    restarted = RunManager(cfg, StateStore(str(tmp_path / "state" / "test.db")))
    try:
        for batch_id, csv_path, work_dir in resume_specs:
            assert restarted.submit_resume(batch_id, csv_path, work_dir)
    finally:
        restarted.shutdown(wait=True)
    resumed = (tmp_path / "nf" / "resumed").read_text().split()
    assert sorted(resumed) == sorted(state.get_batch(x)["work_dir"] for x in (a, b))
    assert _slide(state, "sa")[1] == 0 and _slide(state, "sb")[1] == 0


def test_run_ending_just_before_the_shutdown_signal_is_not_booked(tmp_path, monkeypatch):
    """The stop can end Nextflow before the dispatcher's signal handler has run."""
    monkeypatch.setattr(runner_mod, "_SHUTDOWN_EXIT_GRACE_SECONDS", 5.0)
    cfg = _env(tmp_path, monkeypatch)
    state = StateStore(str(tmp_path / "state" / "test.db"))
    manager = RunManager(cfg, state)
    a = _batch_id("cccc0003")
    try:
        _submit(manager, state, a, "sa")
        pid = _wait_running(state, a)
        os.kill(pid, signal.SIGTERM)
        while state.get_batch_nf_pid(a):     # the worker is past proc.wait()
            time.sleep(0.02)
        time.sleep(0.3)
        manager.begin_shutdown()
        manager.stop_for_restart_recovery()
    finally:
        manager.shutdown(wait=True)

    _assert_preserved(state, a, "sa")


def test_failure_booking_yields_to_a_concurrent_preservation(tmp_path, monkeypatch):
    """Shutdown preserves the batch between the worker's shutdown check and its
    failure write: the write must not land."""
    cfg = _env(tmp_path, monkeypatch)

    class PreserveFirst(StateStore):
        def complete_batch(self, batch_id, exit_code, **kw):
            if exit_code != 0:
                self.preserve_batch_for_recovery(batch_id, "RECOVERING")
            return super().complete_batch(batch_id, exit_code, **kw)

    state = PreserveFirst(str(tmp_path / "state" / "test.db"))
    state.add_slide("/slides/sa.svs", "sa")
    a = _batch_id("dddd0004")
    runner = NextflowRunner(
        cfg, a, [{"slide_path": "/slides/sa.svs", "slide_id": "sa"}], state,
        shutdown_event=threading.Event(),
    )
    thread = threading.Thread(target=runner.run)
    thread.start()
    os.kill(_wait_running(state, a), signal.SIGTERM)
    thread.join(30)
    assert not thread.is_alive()

    _assert_preserved(state, a, "sa")


def test_shutdown_does_not_revive_a_batch_its_worker_already_failed(tmp_path, monkeypatch):
    cfg = _env(tmp_path, monkeypatch)
    state = StateStore(str(tmp_path / "state" / "test.db"))
    a = _batch_id("eeee0005")
    work = tmp_path / "work" / f"batch_{a}" / "work"
    work.mkdir(parents=True)
    csv = tmp_path / "batches" / f"{a}.csv"
    csv.write_text("slide_path\n/slides/sa.svs\n")
    state.add_slide("/slides/sa.svs", "sa")
    state.add_batch(a, str(csv), str(work), 1, str(tmp_path / "logs" / f"batch_{a}.log"))
    state.mark_dispatched(["/slides/sa.svs"], a)
    # A genuine failure, booked before shutdown; the worker is still cleaning up.
    state.complete_batch(a, 1)
    state.mark_slides_complete(a, False)
    manager = RunManager(cfg, state)
    manager._futures[a] = Future()
    try:
        manager.stop_for_restart_recovery()
    finally:
        manager._futures.clear()
        manager.shutdown(wait=True)

    assert state.get_batch(a)["status"] == "FAILED"
    assert _slide(state, "sa") == ("FAILED", 1)


def test_conditional_terminal_writes_skip_preserved_batches(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    state.add_slide("/slides/sa.svs", "sa")
    state.add_batch("b1", "/x.csv", "/w", 1, "/x.log")
    state.mark_dispatched(["/slides/sa.svs"], "b1")
    state.set_batch_status("b1", "RECOVERING")

    assert state.complete_batch("b1", 1, unless_preserved=True) is False
    state.mark_slides_complete("b1", False, unless_preserved=True)

    assert state.get_batch("b1")["status"] == "RECOVERING"
    assert _slide(state, "sa") == ("DISPATCHED", 0)
    assert state.complete_batch("b1", 1) is True


# ---------------------------------------------------------------------------
# Staleness is the time since the batch's last progress, not its creation age.
# ---------------------------------------------------------------------------

def _interrupted_batch(tmp_path, state, batch_id, *, log_age_hours, nf_pid=None):
    work = tmp_path / "work" / f"batch_{batch_id}" / "work"
    work.mkdir(parents=True)
    csv = tmp_path / "batches" / f"{batch_id}.csv"
    csv.parent.mkdir(exist_ok=True)
    csv.write_text("slide_path\n/slides/sa.svs\n")
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"batch_{batch_id}.log"
    state.add_slide("/slides/sa.svs", "sa")
    state.add_batch(batch_id, str(csv), str(work), 1, str(log_path))
    state.mark_dispatched(["/slides/sa.svs"], batch_id)
    if nf_pid:
        state.set_batch_nf_pid(batch_id, nf_pid)
    mtime = time.time() - log_age_hours * 3600
    for path in (log_path, logs / f"batch_{batch_id}.nf.log", logs / f"batch_{batch_id}.trace.tsv"):
        path.write_text("x\n")
        os.utime(path, (mtime, mtime))
    return work


def _dead_pid():
    pid = 4_000_000
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except OSError:
            pass
        pid += 1


@pytest.mark.parametrize("nf_pid", [None, "dead"])
def test_old_batch_with_recent_progress_is_resumed(tmp_path, monkeypatch, nf_pid):
    """Created 29 h ago, its Nextflow was alive (waiting for GPUs) until the restart."""
    monkeypatch.setattr(sched, "_scancel_slurm_jobs", lambda *a, **k: None)
    state = StateStore(str(tmp_path / "state.db"))
    batch_id = _batch_id("af8172d3", hours_ago=29.3)
    _interrupted_batch(tmp_path, state, batch_id, log_age_hours=0.01,
                       nf_pid=_dead_pid() if nf_pid else None)

    resume_specs, _ = recover_in_flight(state, deque(), retry_failed=False)

    assert [spec[0] for spec in resume_specs] == [batch_id]
    assert state.get_batch(batch_id)["status"] == "RECOVERING"
    assert _slide(state, "sa") == ("DISPATCHED", 0)


@pytest.mark.parametrize("nf_pid", [None, "dead"])
@pytest.mark.parametrize("created_hours,log_age_hours,limits", [
    (22.0, 0.01, {"stale_batch_max_age_hours": 20}),    # past the hard age bound
    (10.0, 6.0, {"stale_batch_idle_hours": 4}),         # no progress for too long
])
def test_abandoned_batch_is_reset_without_charging_slides(
    tmp_path, monkeypatch, nf_pid, created_hours, log_age_hours, limits
):
    monkeypatch.setattr(sched, "_scancel_slurm_jobs", lambda *a, **k: None)
    state = StateStore(str(tmp_path / "state.db"))
    cfg = make_config(tmp_path, **limits)
    batch_id = _batch_id("8337c01b", hours_ago=created_hours)
    _interrupted_batch(tmp_path, state, batch_id, log_age_hours=log_age_hours,
                       nf_pid=_dead_pid() if nf_pid else None)

    resume_specs, _ = recover_in_flight(state, deque(), retry_failed=False, cfg=cfg)

    assert resume_specs == []
    assert state.get_batch(batch_id)["status"] == "FAILED"
    assert _slide(state, "sa") == ("PENDING", 0)


def test_submit_resume_judges_local_by_progress_and_remote_by_age(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    state = StateStore(str(tmp_path / "state.db"))
    local = _batch_id("7f5aab01", hours_ago=29.3)
    work = _interrupted_batch(tmp_path, state, local, log_age_hours=0.01)
    remote = _batch_id("7c77bc4e", hours_ago=29.3)
    state.add_batch(remote, str(tmp_path / "r.csv"), str(tmp_path / "rw"), 1,
                    str(tmp_path / "logs" / f"batch_{remote}.log"), backend="remote_agent")
    (tmp_path / "logs" / f"batch_{remote}.log").write_text("fresh\n")

    class IdleRunner:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return 0

    monkeypatch.setattr(sched, "NextflowRunner", IdleRunner)
    manager = RunManager(cfg, state)
    try:
        assert manager.submit_resume(local, str(tmp_path / "batches" / f"{local}.csv"), str(work))
        assert not manager.submit_resume(remote, str(tmp_path / "r.csv"), str(tmp_path / "rw"))
    finally:
        manager.shutdown(wait=True)
    assert state.get_batch(remote)["status"] == "FAILED"
    assert _slide(state, "sa") == ("DISPATCHED", 0)
