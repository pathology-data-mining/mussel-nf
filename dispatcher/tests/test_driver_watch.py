"""Lost Slurm jobs and stalled remote drivers.

The driver side runs against tests/fake_slurm (real sbatch/squeue/sacct
processes that can lose a job and hand its id to another user); the agent side
runs RemoteAgent._refresh_order on a real run dir.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mussel_dispatcher import driver_watch, remote
from mussel_dispatcher.driver_watch import LostJobWatcher, assess_stall

from tests.test_remote_recovery import FakeSlurm, _make_agent, _ok, _order

REPO = Path(__file__).resolve().parents[2]
FAKE_SLURM = REPO / "tests" / "fake_slurm"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _stamp(when: datetime) -> str:
    return when.astimezone().strftime("%b-%d %H:%M:%S.") + f"{when.microsecond // 1000:03d}"


class Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """The fake Slurm on PATH; ``conf`` sets its misbehaviour for later submissions."""
    monkeypatch.setenv("PATH", f"{FAKE_SLURM}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(tmp_path / "slurmctld"))
    conf_path = tmp_path / "fake_slurm.conf"
    monkeypatch.setenv("FAKE_SLURM_CONF", str(conf_path))

    class Slurm:
        state_dir = tmp_path / "slurmctld"

        def conf(self, text: str) -> None:
            conf_path.write_text(text + "\n")

        def scancels(self) -> list[str]:
            path = self.state_dir / "scancel.log"
            return path.read_text().splitlines() if path.exists() else []

        def job_state(self, job_id: str) -> str:
            return json.loads((self.state_dir / f"job_{job_id}.json").read_text())["state"]

        def sacct(self, job_id: str) -> str:
            return subprocess.run(
                ["sacct", "-a", "-D", "-X", "-n", "-P", "-j", job_id,
                 "--format=JobID,JobName,User,State,Start"],
                check=True, text=True, capture_output=True,
            ).stdout

    return Slurm()


class FakeNextflow:
    """Writes task dirs and the nextflow.log lines Nextflow writes on a Slurm submit."""

    def __init__(self, run_dir: Path, clock: Clock):
        self.run_dir = run_dir
        self.log = run_dir / "nextflow.log"
        self.clock = clock
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log.touch()
        self.n = 0

    def submit(self, script: str = "sleep 60") -> tuple[str, Path]:
        self.n += 1
        name = f"MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:TESSELLATE ({self.n})"
        wd = self.run_dir / "work" / f"{self.n:02d}" / f"{self.n:030x}"
        wd.mkdir(parents=True)
        (wd / ".command.run").write_text(
            "#!/bin/bash\n"
            f"#SBATCH -J nf-MUSSEL_EXTRACT_FEATURES_TESSELLATE_PATCHES_TESSELLATE_({self.n})\n"
            f"#SBATCH -o {wd}/.command.log\n"
            f"{script}\n"
        )
        out = subprocess.run(["sbatch", ".command.run"], cwd=wd, check=True, text=True,
                             capture_output=True).stdout
        job_id = out.split()[-1]
        with open(self.log, "a") as fh:
            fh.write(f"{_stamp(self.clock())} [Task submitter] DEBUG nextflow.executor.GridTaskHandler - "
                     f"[SLURM] submitted process {name} > jobId: {job_id}; workDir: {wd}\n")
            fh.write(f"{_stamp(self.clock())} [Task submitter] INFO  nextflow.Session - "
                     f"[{self.n:02d}/abcdef] Submitted process > {name}\n")
        return job_id, wd


def _watcher(nf: FakeNextflow, clock: Clock, grace: float = 1800) -> LostJobWatcher:
    return LostJobWatcher(nf.run_dir, nf.log, grace_seconds=grace, clock=clock, log=lambda _m: None)


def _cleanup_jobs(tmp_path):
    subprocess.run(["pkill", "-f", str(tmp_path)], check=False)


# --- driver side: lost jobs ----------------------------------------------------


@pytest.mark.parametrize("mode, reason", [
    ("never_started", "never_started"),        # 9000001: ours, Start=None
    ("reused", "id_reused_by_other_job"),      # 9000002: only another user's job
    ("no_record", "no_accounting_record"),
    ("boot_fail", "ended_before_task_began"),  # got a node, ended before .command.begin
])
def test_lost_job_is_failed_after_the_grace_only(tmp_path, slurm, mode, reason):
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    slurm.conf(f"LOSE=TESSELLATE_\\(1\\)\nLOSE_MODE={mode}")
    lost_id, lost_wd = nf.submit()
    if mode == "reused":
        assert "otheruser" in slurm.sacct(lost_id)
    watcher = _watcher(nf, clock)

    assert watcher.poll() == []
    clock.advance(1700)
    assert watcher.poll() == []
    assert not (lost_wd / ".exitcode").exists()
    clock.advance(200)
    assert watcher.poll() == []   # lost on one look: confirmed on the next poll
    assert not (lost_wd / ".exitcode").exists()
    marked = watcher.poll()

    assert [(m["job_id"], m["reason"]) for m in marked] == [(lost_id, reason)]
    assert (lost_wd / ".command.begin").exists()
    assert (lost_wd / ".exitcode").read_text() == str(driver_watch.LOST_JOB_EXIT_CODE)
    assert json.loads((lost_wd / driver_watch.MARKER_FILE).read_text())["reason"] == reason
    assert "gone without having run" in (lost_wd / ".command.err").read_text()
    status = watcher.status()
    assert [j["job_id"] for j in status["lost_jobs"]] == [lost_id]
    # Our job was scancelled first (by id, user and name), never another user's.
    name = f"nf-MUSSEL_EXTRACT_FEATURES_TESSELLATE_PATCHES_TESSELLATE_(1)"
    expected = [] if mode == "reused" else [f"--user {watcher.user} --name {name} {lost_id}"]
    assert slurm.scancels() == expected
    # Marked once: the task now has an exit code.
    clock.advance(3600)
    assert watcher.poll() == []
    assert not list(lost_wd.glob(".*.tmp"))


def test_running_and_started_tasks_are_never_touched(tmp_path, slurm):
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    try:
        slurm.conf("")
        running_id, running_wd = nf.submit("sleep 120")      # queued/running under our name
        slurm.conf("LOSE=TESSELLATE_\\(2\\)")
        _, started_wd = nf.submit()                          # lost, but it had started
        (started_wd / ".command.begin").touch()
        watcher = _watcher(nf, clock)

        for _ in range(3):
            clock.advance(3600)
            assert watcher.poll() == []
        assert not (running_wd / ".exitcode").exists()
        assert not (running_wd / ".command.begin").exists()
        assert not (started_wd / ".exitcode").exists()
    finally:
        _cleanup_jobs(tmp_path)


def test_unreachable_controller_neither_marks_nor_ages(tmp_path, slurm, monkeypatch):
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    slurm.conf("LOSE=TESSELLATE_\\(1\\)")
    _, lost_wd = nf.submit()
    watcher = _watcher(nf, clock)
    monkeypatch.setenv("FAKE_SLURM_DOWN", "1")

    for _ in range(4):
        clock.advance(3600)
        assert watcher.poll() == []
    assert "squeue" in watcher.status()["last_error"]
    assert not (lost_wd / ".exitcode").exists()

    # Back up: the job has been missing for 0 s as far as anyone knows.
    monkeypatch.delenv("FAKE_SLURM_DOWN")
    assert watcher.poll() == []
    clock.advance(1800)
    assert watcher.poll() == []
    assert len(watcher.poll()) == 1


def test_sacct_down_after_the_grace_does_not_mark(tmp_path, slurm, monkeypatch):
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    slurm.conf("LOSE=TESSELLATE_\\(1\\)")
    _, lost_wd = nf.submit()
    watcher = _watcher(nf, clock)
    watcher.poll()
    clock.advance(3600)
    real = watcher.runner

    def sacct_down(cmd):
        if cmd[0] == "sacct":
            raise RuntimeError("sacct: slurmdbd unreachable")
        return real(cmd)

    watcher.runner = sacct_down
    assert watcher.poll() == []
    assert not (lost_wd / ".exitcode").exists()
    watcher.runner = real
    assert watcher.poll() == []
    assert len(watcher.poll()) == 1


def test_a_job_slurm_reports_late_is_not_failed(tmp_path, slurm):
    """squeue and slurmdbd both lag a fresh job: one lost-looking poll is not enough."""
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    try:
        slurm.conf("")
        job_id, wd = nf.submit("sleep 120")
        watcher = _watcher(nf, clock)
        real = watcher.runner
        lagging = {"on": True}

        def runner(cmd):
            if lagging["on"] and cmd[0] in ("squeue", "sacct"):
                return ""   # not queued, no accounting record (yet)
            return real(cmd)

        watcher.runner = runner
        watcher.poll()
        clock.advance(1900)
        assert watcher.poll() == []          # looks lost once
        lagging["on"] = False
        assert watcher.poll() == []          # Slurm caught up: queued under our name
        lagging["on"] = True
        assert watcher.poll() == []          # the count started over
        assert not (wd / ".exitcode").exists()
        assert slurm.scancels() == []
        assert slurm.job_state(job_id) == "RUNNING"
    finally:
        _cleanup_jobs(tmp_path)


def test_a_misjudged_job_is_cancelled_before_its_task_is_failed(tmp_path, slurm):
    """If Slurm never reports a job that is in fact queued, failing its task would run
    the retry beside it: the watch scancels it first."""
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    try:
        slurm.conf("")
        job_id, wd = nf.submit("sleep 120")
        watcher = _watcher(nf, clock)
        real = watcher.runner
        watcher.runner = lambda cmd: "" if cmd[0] in ("squeue", "sacct") else real(cmd)
        watcher.poll()
        clock.advance(1900)
        watcher.poll()
        marked = watcher.poll()

        assert [m["reason"] for m in marked] == ["no_accounting_record"]
        assert slurm.job_state(job_id) == "CANCELLED"
        assert (wd / ".exitcode").read_text() == "75"
    finally:
        _cleanup_jobs(tmp_path)


def test_scancel_failure_does_not_block_marking(tmp_path, slurm):
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    slurm.conf("LOSE=TESSELLATE_\\(1\\)")
    _, wd = nf.submit()
    watcher = _watcher(nf, clock)
    real = watcher.runner

    def runner(cmd):
        if cmd[0] == "scancel":
            raise RuntimeError("scancel exited 1: slurm_kill_job error")
        return real(cmd)

    watcher.runner = runner
    watcher.poll()
    clock.advance(1900)
    watcher.poll()
    assert len(watcher.poll()) == 1
    assert (wd / ".exitcode").read_text() == "75"


def test_exit_code_appears_whole_or_not_at_all(tmp_path, slurm, monkeypatch):
    """Nextflow reads .exitcode as soon as it exists: a failed write must not leave an
    empty one (nor a stray temp file)."""
    clock = Clock(T0)
    nf = FakeNextflow(tmp_path / "run", clock)
    slurm.conf("LOSE=TESSELLATE_\\(1\\)")
    _, wd = nf.submit()
    watcher = _watcher(nf, clock)
    watcher.poll()
    clock.advance(1900)
    watcher.poll()

    def disk_full(_fd):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m:   # undo only this patch, not the fake Slurm's env
        m.setattr(driver_watch.os, "fsync", disk_full)
        assert watcher.poll() == []
    assert not (wd / ".exitcode").exists()
    assert not list(wd.glob(".*.tmp"))
    assert "No space left" in watcher.status()["last_error"]
    assert len(watcher.poll()) == 1
    assert (wd / ".exitcode").read_text() == "75"


def test_the_watch_script_runs_beside_the_driver_and_exits_with_it(tmp_path, slurm):
    clock = Clock(datetime.now(timezone.utc) - timedelta(hours=2))
    nf = FakeNextflow(tmp_path / "run", clock)
    slurm.conf("LOSE=TESSELLATE_\\(1\\)\nLOSE_MODE=reused")
    _, lost_wd = nf.submit()
    parent = subprocess.Popen(["sleep", "4"])
    threading.Thread(target=parent.wait, daemon=True).start()   # reap it: a zombie still "exists"
    try:
        watch = subprocess.run(
            [sys.executable, str(Path(driver_watch.__file__)), "--run-dir", str(nf.run_dir),
             "--nf-log", str(nf.log), "--grace-seconds", "0", "--interval-seconds", "1",
             "--parent-pid", str(parent.pid)],
            text=True, capture_output=True, timeout=60,
        )
    finally:
        parent.wait()
    assert watch.returncode == 0, watch.stderr
    assert "LOST SLURM JOB" in watch.stdout
    assert (lost_wd / ".exitcode").read_text() == "75"
    status = driver_watch.read_status(nf.run_dir)
    assert status["lost_jobs"][0]["reason"] == "id_reused_by_other_job"


def test_rendered_driver_starts_the_watch_only_when_enabled(tmp_path):
    from mussel_dispatcher.config import RemoteTargetConfig
    order = {
        "batch_id": "b1", "remote_run_dir": str(tmp_path / "run"), "remote_repo_dir": "/remote/repo",
        "nextflow_profiles": "cluster", "nextflow_run_name": "rb1",
    }
    on = remote.render_driver_script(order, RemoteTargetConfig(name="hpc", lost_job_grace_seconds=900))
    off = remote.render_driver_script(order, RemoteTargetConfig(name="hpc", lost_job_detection=False))

    assert "/remote/repo/dispatcher/mussel_dispatcher/driver_watch.py --run-dir" in on
    assert "--grace-seconds 900" in on and "--parent-pid $$" in on
    assert on.index("watch_pid=$!") < on.index("completed batch b1")
    assert "driver_watch.py" not in off
    for script in (on, off):
        syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert syntax.returncode == 0, syntax.stderr


# --- live: Nextflow + the watch + the fake Slurm ------------------------------


@pytest.mark.live_nextflow
def test_lost_tessellate_job_no_longer_hangs_the_pipeline(tmp_path):
    """Stub pipeline on the production cluster+slurm profiles; Slurm loses the second
    TESSELLATE job once (id reused by another user). The watch fails it with exit 75,
    Nextflow retries it, and the run finishes both slides."""
    nf, log = _run_stub_pipeline(
        tmp_path, "LOSE=TESSELLATE_PATCHES_TESSELLATE_\\(2\\)\nLOSE_MODE=reused\nLOSE_TIMES=1",
        slides=("948176", "948177"))
    watch = subprocess.Popen(
        [sys.executable, str(Path(driver_watch.__file__)), "--run-dir", str(tmp_path),
         "--nf-log", str(log), "--grace-seconds", "5", "--interval-seconds", "2",
         "--parent-pid", str(nf.pid)],
        env=dict(os.environ, PATH=f"{FAKE_SLURM}:{os.environ['PATH']}",
                 FAKE_SLURM_DIR=str(tmp_path / "slurmctld")),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out, _ = nf.communicate(timeout=600)
    watch_out, _ = watch.communicate(timeout=60)

    assert not (tmp_path / "fake_slurm_watchdog_fired").exists(), out
    assert nf.returncode == 0, out
    assert "LOST SLURM JOB" in watch_out, watch_out
    for slide in ("948176", "948177"):
        assert (tmp_path / "out" / "tiles_512px" / "9481" / f"{slide}.patch.h5").exists(), out
        assert (tmp_path / "out" / "features" / "titan_slide" / f"{slide}.features.pt").exists(), out


# --- agent side: publish lost jobs, detect and (optionally) cancel stalls ------


INCIDENT_IDLE = """\
{t} [Task monitor] DEBUG n.processor.TaskPollingMonitor - !! executor local > No more task to compute -- The following nodes are still active:
[process] MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:DERIVE_TILE_GRID
  status=ACTIVE
  port 0: (queue) OPEN  ; channel: -
  port 1: (cntrl) -     ; channel: $

{t} [Task monitor] DEBUG n.processor.TaskPollingMonitor - !! executor slurm > No more task to compute -- The following nodes are still active:
[process] MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:DERIVE_TILE_GRID
  status=ACTIVE
"""
COMPLETED = ("{t} [Task monitor] DEBUG n.processor.TaskPollingMonitor - Task completed > TaskHandler[jobId: 9000002; "
             "id: 3; name: MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:TESSELLATE (2); status: COMPLETED; exit: 1]\n")
BUSY = ("{t} [Task monitor] DEBUG n.processor.TaskPollingMonitor - !! executor slurm > tasks to be completed: 1 -- "
        "submitted tasks are shown below\n~> TaskHandler[jobId: 9000002; id: 3; status: SUBMITTED]\n")


def _nf_log(*parts: tuple[str, float]) -> str:
    now = datetime.now(timezone.utc)
    return "".join(text.format(t=_stamp(now - timedelta(seconds=ago))) for text, ago in parts)


def test_assess_stall_on_the_incident_log():
    now = datetime.now(timezone.utc)
    kw = dict(idle_seconds=3 * 3600, no_progress_seconds=12 * 3600)
    # Nov 10 14:03 -> the ignored task completed; every executor idle since.
    stalled = assess_stall(_nf_log((COMPLETED, 4 * 3600), (INCIDENT_IDLE, 60)), now, **kw)
    assert stalled["reason"] == "idle_with_active_nodes"
    assert stalled["idle_executors"] == ["local", "slurm"]
    assert assess_stall(_nf_log((COMPLETED, 3600), (INCIDENT_IDLE, 60)), now, **kw) is None
    assert stalled["cancellable"] is True
    # Nov 9 16:05 -> Nov 10 14:03: a SUBMITTED task Slurm lost keeps slurm "busy"; the
    # lost-job watch fails that task, so a busy executor is never a stall.
    busy = _nf_log((COMPLETED, 13 * 3600), (INCIDENT_IDLE.split("\n\n")[0] + "\n", 60), (BUSY, 30))
    assert assess_stall(busy, now, idle_seconds=3 * 3600, no_progress_seconds=0) is None
    assert assess_stall(busy, now, **kw) is None


def test_a_long_gpu_queue_wait_is_not_a_stall():
    """Local batches waited 29 h for A100s: the slurm executor dumps its pending tasks
    every 5 minutes and nothing completes. That is not a stall."""
    now = datetime.now(timezone.utc)
    kw = dict(idle_seconds=3 * 3600, no_progress_seconds=12 * 3600)
    dumps = [(BUSY, ago) for ago in range(29 * 3600, 0, -300)]
    assert assess_stall(_nf_log((COMPLETED, 29 * 3600 + 60), *dumps), now, **kw) is None
    # Nothing in flight and nothing happening (no executor dump at all): reported, but
    # never cancellable.
    quiet = assess_stall(_nf_log((COMPLETED, 13 * 3600)), now, **kw)
    assert quiet["reason"] == "no_task_progress" and quiet["cancellable"] is False


def _running_driver_agent(tmp_path, monkeypatch, status=None, **target):
    agent = _make_agent(tmp_path, **target)
    order = _order(tmp_path, status="RUNNING", slides=[{"slide_id": "S1", "slide_path": "/s/S1.svs"}],
                   remote_nf_log=str(tmp_path / "run" / "nextflow.log"))
    (tmp_path / "run").mkdir(exist_ok=True)
    agent.state.get_remote_batches_for_target.return_value = [order]
    agent.state.get_slide_statuses.return_value = {
        "S1": {"status": "DISPATCHED", "batch_id": order["batch_id"]}}
    slurm_state = {"state": "RUNNING", "source": "squeue"}
    agent._driver_slurm_state = lambda _job: dict(slurm_state)
    client = MagicMock()
    current = {"status": "RUNNING", "driver_job_id": "4242", **(status or {})}
    client.read_status.side_effect = lambda _b: dict(current)

    def write(_b, payload):
        current.update(payload)
        return True

    client.write_status.side_effect = write
    monkeypatch.setattr(remote, "RemoteDispatchClient", lambda *a, **k: client)
    return agent, client, current, slurm_state


def test_lost_jobs_from_the_watch_are_published_and_alerted(tmp_path, monkeypatch, caplog):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch)
    lost = {"job_id": "9000002", "task": "TESSELLATE (2)", "work_dir": "/w/58/89e4d9",
            "reason": "id_reused_by_other_job", "exit_code": 75}
    driver_watch.write_status(tmp_path / "run" / driver_watch.STATUS_FILE, {
        "heartbeat_at": datetime.now(timezone.utc).isoformat(), "lost_jobs": [lost]})
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({}))

    agent._refresh_claimed_orders()
    agent._refresh_claimed_orders()

    assert current["status"] == "RUNNING"
    assert current["lost_jobs"] == [lost]
    assert "1 lost Slurm job" in current["warning"]
    alerts = [r for r in caplog.records if "ALERT" in r.getMessage() and "9000002" in r.getMessage()]
    assert len(alerts) == 1   # once, not every poll


def test_stale_watch_heartbeat_is_a_warning(tmp_path, monkeypatch):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch)
    driver_watch.write_status(tmp_path / "run" / driver_watch.STATUS_FILE, {
        "heartbeat_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), "lost_jobs": []})
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({}))

    agent._refresh_claimed_orders()

    assert "driver watch is not running" in current["warning"]


def test_stalled_driver_is_reported_but_not_cancelled_by_default(tmp_path, monkeypatch, caplog):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch)
    (tmp_path / "run" / "nextflow.log").write_text(_nf_log((COMPLETED, 4 * 3600), (INCIDENT_IDLE, 60)))
    slurm = FakeSlurm({})   # scancel would raise "unexpected command"
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._refresh_claimed_orders()

    assert current["status"] == "RUNNING"
    assert current["driver_stall"]["failure_class"] == "DRIVER_STALLED"
    assert current["driver_stall"]["reason"] == "idle_with_active_nodes"
    assert current["warning"].startswith("DRIVER_STALLED")
    assert any("ALERT" in r.getMessage() and "stalled" in r.getMessage() for r in caplog.records)
    assert slurm.commands("scancel") == []

    # Progress resumes: the finding is cleared.
    (tmp_path / "run" / "nextflow.log").write_text(_nf_log((COMPLETED, 60)))
    agent._refresh_claimed_orders()
    assert current["driver_stall"] is None and current["warning"] is None


@pytest.mark.parametrize("ended", ["CANCELLED", "FAILED"])   # FAILED: Nextflow exited 1 on the signal
def test_stalled_driver_cancel_is_an_infrastructure_failure(tmp_path, monkeypatch, ended):
    agent, client, current, slurm_state = _running_driver_agent(
        tmp_path, monkeypatch, driver_stall_cancel=True)
    (tmp_path / "run" / "nextflow.log").write_text(_nf_log((COMPLETED, 4 * 3600), (INCIDENT_IDLE, 60)))
    slurm = FakeSlurm({"scancel": _ok()})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._refresh_claimed_orders()
    assert slurm.commands("scancel") == [["scancel", "4242"]]
    assert current["stall_cancelled_at"]

    slurm_state.update(state=ended, exit_code=0, source="sacct")
    agent._refresh_claimed_orders()

    assert current["status"] == "FAILED"
    assert current["failure_class"] == "DRIVER_STALLED"
    assert "idle_with_active_nodes" in current["error"]
    assert current["nextflow_exit"] == -1
    agent.state.mark_slides_complete.assert_called_once_with(
        "20261028T101112_ab12cd34", False, charge_fail_count=False)
    assert len(slurm.commands("scancel")) == 1


def test_watch_findings_reach_the_stored_remote_status(tmp_path):
    from mussel_dispatcher.state import StateStore
    state = StateStore(str(tmp_path / "state.db"))
    state.add_batch("b1", str(tmp_path / "batch.csv"), str(tmp_path / "work"), 0,
                    str(tmp_path / "batch.log"), backend="remote_agent", target="hpc")
    state.write_remote_status("b1", {"status": "RUNNING", "driver_job_id": "4242"})

    stall = {"failure_class": "DRIVER_STALLED", "reason": "idle_with_active_nodes"}
    assert state.write_remote_status("b1", {"status": "RUNNING", "driver_stall": stall,
                                            "lost_jobs": [{"job_id": "1"}], "warning": "w"})
    stored = state.read_remote_status("b1")
    assert stored["driver_stall"] == stall and stored["lost_jobs"] == [{"job_id": "1"}]
    assert state.write_remote_status("b1", {"status": "RUNNING", "stall_cancelled_at": "now"})
    assert state.read_remote_status("b1")["stall_cancelled_at"] == "now"


def test_no_progress_alone_is_never_cancelled(tmp_path, monkeypatch):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch, driver_stall_cancel=True)
    (tmp_path / "run" / "nextflow.log").write_text(_nf_log((COMPLETED, 13 * 3600)))
    slurm = FakeSlurm({})   # scancel would raise "unexpected command"
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._refresh_claimed_orders()

    assert current["driver_stall"]["reason"] == "no_task_progress"
    assert slurm.commands("scancel") == []
    assert not current.get("stall_cancelled_at")


def test_failed_stall_scancel_still_publishes_the_status(tmp_path, monkeypatch):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch, driver_stall_cancel=True)
    (tmp_path / "run" / "nextflow.log").write_text(_nf_log((COMPLETED, 4 * 3600), (INCIDENT_IDLE, 60)))
    slurm = FakeSlurm({"scancel": subprocess.TimeoutExpired(["scancel", "4242"], 30)})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._refresh_claimed_orders()

    assert agent._order_failure_state() == {}   # the refresh did not blow up
    assert current["driver_stall"]["reason"] == "idle_with_active_nodes"
    assert current["heartbeat_at"]
    assert not current.get("stall_cancelled_at")   # retried next poll
    slurm.handlers["scancel"] = _ok()
    agent._refresh_claimed_orders()
    assert len(slurm.commands("scancel")) == 2
    assert current["stall_cancelled_at"]


def test_a_stall_is_not_rewritten_every_poll(tmp_path, monkeypatch):
    """quiet_seconds grows every poll; the stored status must not change with it."""
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch)
    (tmp_path / "run" / "nextflow.log").write_text(_nf_log((COMPLETED, 4 * 3600), (INCIDENT_IDLE, 60)))
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({}))
    payloads = []
    client.write_status.side_effect = lambda _b, payload: (payloads.append(dict(payload)),
                                                           current.update(payload))[-1] is None
    agent._refresh_claimed_orders()

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(minutes=10)

    monkeypatch.setattr(remote, "datetime", Later)
    agent._refresh_claimed_orders()

    assert len(payloads) == 2
    assert payloads[1]["driver_stall"] == payloads[0]["driver_stall"]
    assert payloads[1]["warning"] == payloads[0]["warning"]


def test_a_watch_that_never_started_is_a_warning(tmp_path, monkeypatch):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({}))

    agent._refresh_claimed_orders()
    assert current.get("warning") is None   # just started: give it a few intervals
    batch_id = next(iter(agent._driver_watch_missing_since))
    agent._driver_watch_missing_since[batch_id] -= 11 * 60
    agent._refresh_claimed_orders()
    assert "driver watch never started" in current["warning"]

    # It turns up: the warning clears.
    driver_watch.write_status(tmp_path / "run" / driver_watch.STATUS_FILE, {
        "heartbeat_at": datetime.now(timezone.utc).isoformat(), "lost_jobs": []})
    agent._refresh_claimed_orders()
    assert current["warning"] is None


def test_no_watch_warning_when_detection_is_off(tmp_path, monkeypatch):
    agent, client, current, _ = _running_driver_agent(tmp_path, monkeypatch, lost_job_detection=False)
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({}))
    agent._refresh_claimed_orders()
    agent._driver_watch_missing_since = {k: v - 3600 for k, v in agent.__dict__.get(
        "_driver_watch_missing_since", {}).items()}
    agent._refresh_claimed_orders()
    assert current.get("warning") is None


def _render_watch(tmp_path, **target):
    from mussel_dispatcher.config import RemoteTargetConfig
    order = {"batch_id": "b1", "remote_run_dir": str(tmp_path / "run"), "remote_repo_dir": str(REPO),
             "nextflow_profiles": "cluster", "nextflow_run_name": "rb1"}
    return remote._render_driver_watch(order, RemoteTargetConfig(name="hpc", **target),
                                       str(tmp_path / "run" / "nextflow.log"))


def test_the_driver_runs_the_watch_with_the_configured_python(tmp_path):
    """A system python3 3.6 cannot import driver_watch.py."""
    script = REPO / "dispatcher" / "mussel_dispatcher" / "driver_watch.py"
    configured = _render_watch(tmp_path, driver_watch_python="/opt/py311/bin/python3.11")
    default = _render_watch(tmp_path)

    assert f"  /opt/py311/bin/python3.11 {script} --run-dir" in configured
    assert f"[[ -x /opt/py311/bin/python3.11 && -f {script} ]]" in configured
    assert f"  {sys.executable} {script} --run-dir" in default
    for block in (configured, default):   # never whatever python3 is first on PATH
        assert "command -v python3" not in block
        assert not re.search(r"^\s*python3? ", block, re.M)

    # The rendered block really starts the watch with that interpreter.
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "nextflow.log").touch()
    proc = subprocess.run(["bash", "-c", default + "sleep 3\n"], text=True, capture_output=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    status = driver_watch.read_status(tmp_path / "run")
    assert status and status["version"] == 1, (tmp_path / "run" / "driver_watch.log").read_text()


# --- a driver Nextflow aborted because Slurm would not take its jobs -----------

SUBMIT_ABORT = (
    "{t} [Task submitter] ERROR nextflow.processor.TaskProcessor - Error executing process > "
    "'MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:TESSELLATE (1)'\n\nCaused by:\n"
    "  Failed to submit process to grid scheduler for execution\n\nCommand output:\n"
    "  sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)\n"
    "{t} [Task submitter] DEBUG nextflow.Session - Session aborted -- Cause: Error submitting process "
    "'MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:TESSELLATE (1)' for execution\n"
)
LOST_ABORT = (
    "{t} [Task monitor] DEBUG nextflow.Session - Session aborted -- Cause: Process "
    "`MUSSEL:EXTRACT_FEATURES:TESSELLATE_PATCHES:TESSELLATE (2)` terminated with an error exit status (75)\n"
)
RUN_ABORT = (
    "{t} [Task monitor] DEBUG nextflow.Session - Session aborted -- Cause: Process "
    "`MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH_SLIDE (1)` terminated with an error exit status (1)\n"
)


def test_nextflow_abort_cause():
    assert driver_watch.nextflow_abort_cause(_nf_log((SUBMIT_ABORT, 5)))["reason"] == "submission_failed"
    assert driver_watch.nextflow_abort_cause(_nf_log((LOST_ABORT, 5)))["reason"] == "lost_job"
    assert driver_watch.nextflow_abort_cause(_nf_log((RUN_ABORT, 5))) is None
    # Retried submission failures that did not abort the run are not the cause.
    retried = SUBMIT_ABORT.split("{t} [Task submitter] DEBUG nextflow.Session")[0]
    assert driver_watch.nextflow_abort_cause(_nf_log((retried, 60), (RUN_ABORT, 5))) is None
    assert driver_watch.nextflow_abort_cause(_nf_log((COMPLETED, 5))) is None


def _ended_driver_agent(tmp_path, monkeypatch, nf_log_text: str):
    """A RemoteAgent on a real StateStore whose driver Slurm reports FAILED (exit 1)."""
    from mussel_dispatcher.state import StateStore
    agent = _make_agent(tmp_path)
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/s/S1.svs", "S1")
    order = _order(tmp_path, status="RUNNING", slides=[{"slide_id": "S1", "slide_path": "/s/S1.svs"}],
                   remote_nf_log=str(tmp_path / "run" / "nextflow.log"))
    store.add_batch(order["batch_id"], str(tmp_path / "batch.csv"), str(tmp_path / "work"), 1,
                    str(tmp_path / "batch.log"), backend="remote_agent", target="hpc")
    assert store.mark_dispatched(["/s/S1.svs"], order["batch_id"]) == 1
    store.get_remote_batches_for_target = lambda _t: [order]
    agent.state = store
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "run" / "nextflow.log").write_text(nf_log_text)
    agent._driver_slurm_state = lambda _job: {"state": "FAILED", "exit_code": 1, "source": "sacct"}
    client = MagicMock()
    current = {"status": "RUNNING", "driver_job_id": "4242"}
    client.read_status.side_effect = lambda _b: dict(current)
    client.write_status.side_effect = lambda _b, payload: current.update(payload) or True
    monkeypatch.setattr(remote, "RemoteDispatchClient", lambda *a, **k: client)
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({}))

    def slide():
        return store._conn().execute(
            store._sql("SELECT status, fail_count FROM slides WHERE slide_path=?"), ("/s/S1.svs",)
        ).fetchone()

    return agent, current, slide


@pytest.mark.parametrize("abort", [SUBMIT_ABORT, LOST_ABORT])
def test_a_driver_aborted_by_slurm_charges_no_slide(tmp_path, monkeypatch, abort):
    """Retries exhausted while slurmctld was down: Nextflow aborts, the driver ends FAILED.
    That is Slurm's failure, not the slides' (before, the same outage hung to TIMEOUT)."""
    agent, current, slide = _ended_driver_agent(tmp_path, monkeypatch, _nf_log((abort, 30)))

    agent._refresh_claimed_orders()

    assert current["status"] == "FAILED"
    assert current["failure_class"] == "SLURM_INFRASTRUCTURE"
    assert "Nextflow aborted" in current["error"]
    assert slide()["fail_count"] == 0
    assert slide()["status"] != "DISPATCHED"


def test_a_driver_that_failed_on_its_own_still_charges(tmp_path, monkeypatch):
    agent, current, slide = _ended_driver_agent(tmp_path, monkeypatch, _nf_log((RUN_ABORT, 30)))

    agent._refresh_claimed_orders()

    assert current["failure_class"] == "SLURM_EXECUTION"
    assert slide()["fail_count"] == 1


def _run_stub_pipeline(tmp_path, conf: str, slides=("948176",), timeout=600):
    """The stub pipeline on the production cluster+slurm profiles against the fake Slurm."""
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is not installed")
    env = dict(os.environ, PATH=f"{FAKE_SLURM}:{os.environ['PATH']}",
               FAKE_SLURM_DIR=str(tmp_path / "slurmctld"), NXF_ANSI_LOG="false")
    env.pop("FAKE_SLURM_CONF", None)
    (tmp_path / "fake_slurm.conf").write_text(conf + "\nWATCHDOG=300\n")
    slide = REPO / "tests" / "testdata" / "948176.svs"
    csv = tmp_path / "samples.csv"
    csv.write_text("slide_id,slide_path\n" + "".join(f"{s},{slide}\n" for s in slides))
    log = tmp_path / "nextflow.log"
    nf = subprocess.Popen(
        ["nextflow", "-log", str(log), "run", str(REPO / "main.nf"), "-stub",
         "-profile", "test_stub_grid_from_primary,cluster,slurm",
         "-c", str(FAKE_SLURM / "fake_slurm.config"), "-w", str(tmp_path / "work"),
         "--samples_csv", str(csv), "--outdir", str(tmp_path / "out")],
        cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return nf, log


@pytest.mark.live_nextflow
def test_a_run_aborted_by_failing_submissions_charges_no_slide(tmp_path, monkeypatch):
    """End to end: every sbatch fails (slurmctld down), Nextflow aborts after its retries,
    and the agent reading that driver's real nextflow.log charges no slide."""
    run = tmp_path / "nfrun"
    run.mkdir()
    nf, log = _run_stub_pipeline(run, "FAIL_SUBMIT=TESSELLATE_PATCHES_TESSELLATE")
    out, _ = nf.communicate(timeout=600)
    assert nf.returncode != 0, out
    assert not (run / "fake_slurm_watchdog_fired").exists(), out

    agent, current, slide = _ended_driver_agent(tmp_path, monkeypatch, log.read_text())
    agent._refresh_claimed_orders()

    assert current["failure_class"] == "SLURM_INFRASTRUCTURE", current.get("error")
    assert "submission_failed" in current["error"]
    assert slide()["fail_count"] == 0


@pytest.mark.live_nextflow
def test_a_run_aborted_by_a_job_lost_on_every_attempt_charges_no_slide(tmp_path, monkeypatch):
    """End to end: the watch fails a lost job (exit 75) on every attempt; past maxRetries
    Nextflow aborts, which is again Slurm's failure."""
    run = tmp_path / "nfrun"
    run.mkdir()
    nf, log = _run_stub_pipeline(run, "EXEC_FAIL=TESSELLATE_PATCHES_TESSELLATE\nEXEC_EXIT=75")
    out, _ = nf.communicate(timeout=600)
    assert nf.returncode != 0, out

    agent, current, slide = _ended_driver_agent(tmp_path, monkeypatch, log.read_text())
    agent._refresh_claimed_orders()

    assert current["failure_class"] == "SLURM_INFRASTRUCTURE", current.get("error")
    assert "lost_job" in current["error"]
    assert slide()["fail_count"] == 0
