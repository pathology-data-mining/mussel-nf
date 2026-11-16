"""Regression tests for remote-agent crash recovery and release pinning."""
from __future__ import annotations

import hashlib
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mussel_dispatcher import remote
from mussel_dispatcher.config import Config, RemoteTargetConfig
from mussel_dispatcher.remote import RemoteAgent, _runtime_sha256_manifest

DISPATCHER_DIR = Path(__file__).resolve().parents[1]
_REAL_RUN = subprocess.run
REPLAY = {"replay_capacity": lambda: True}
PURGED = "slurm_load_jobs error: Invalid job id specified"


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=pytest",
         "-c", "user.email=pytest@example.invalid", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_agent(tmp_path: Path, **target_kwargs) -> RemoteAgent:
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.cfg = Config(
        nextflow_profiles="standard",
        repo_dir=str(tmp_path / "local-repo"),
        outdir=str(tmp_path / "results"),
        state_dir=str(tmp_path / "state"),
        state_backend="postgres",
    )
    target_kwargs.setdefault("remote_repo_dir", str(tmp_path / "repo"))
    agent.target = RemoteTargetConfig(name="hpc", **target_kwargs)
    agent.state = MagicMock()
    agent.dry_run = False
    agent.once = False
    agent._driver_slurm_queries = 0
    agent._status_writes_suppressed = 0
    agent._capacity_guard = MagicMock()
    agent._capacity_guard.check.return_value = True
    return agent


def _order(tmp_path: Path, **extra) -> dict:
    order = {
        "schema_version": 1,
        "batch_id": "20261028T101112_ab12cd34",
        "created_at": _iso(600),
        "remote_run_dir": str(tmp_path / "run"),
        "nextflow_params_file": "",
        "nextflow_params_content": None,
        "nextflow_params_sha256": "",
        "runtime_sha256": {},
        "slides": [],
        "nextflow_run_name": "rab12cd34",
    }
    order.update(extra)
    return order


class FakeSlurm:
    """Scriptable stand-in for subprocess.run; records every command."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        if args[0] == "git":
            return _REAL_RUN(args, **kwargs)
        self.calls.append(list(args))
        handler = self.handlers.get(args[0])
        if handler is None:
            raise AssertionError(f"unexpected command: {args}")
        result = handler(args) if callable(handler) else handler
        if isinstance(result, BaseException):
            raise result
        return result

    def commands(self, name: str) -> list[list[str]]:
        return [call for call in self.calls if call[0] == name]


def _ok(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return lambda args: subprocess.CompletedProcess(args, returncode, stdout, stderr)


# --- Finding 1: orphaned CLAIMED orders -------------------------------------


def test_claim_orphaned_by_previous_agent_is_replayed_despite_fresh_heartbeat(tmp_path):
    agent = _make_agent(tmp_path)
    agent._started_at = time.time() - 30
    fresh_agent = {"heartbeat_at": _iso(5)}  # written by this very process
    orphan = {"batch_id": "b1", "status": "CLAIMED", "claimed_at": _iso(600)}

    assert agent._should_replay_claimed_order(orphan, {"status": "CLAIMED"}, fresh_agent)


def test_claim_made_by_this_agent_is_not_replayed_while_heartbeat_fresh(tmp_path):
    agent = _make_agent(tmp_path)
    agent._started_at = time.time() - 1000
    fresh_agent = {"heartbeat_at": _iso(5)}
    mine = {"batch_id": "b1", "status": "CLAIMED", "claimed_at": _iso(600)}

    assert not agent._should_replay_claimed_order(mine, {"status": "CLAIMED"}, fresh_agent)


def test_replay_adopts_existing_driver_job_instead_of_resubmitting(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    slurm = FakeSlurm({"squeue": _ok("4242\n")})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._process_order(_order(tmp_path), **REPLAY)

    assert slurm.commands("sbatch") == []
    squeue = slurm.commands("squeue")[0]
    assert squeue[squeue.index("-n") + 1] == "mussel-driver-20261028T101112_ab12cd34"
    payload = agent.state.write_remote_status.call_args.args[1]
    assert payload["status"] == "RUNNING"
    assert payload["driver_job_id"] == "4242"


def test_replay_adopts_finished_driver_found_in_sacct(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    slurm = FakeSlurm({"squeue": _ok(""), "sacct": _ok("4243|COMPLETED|0:0\n")})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._process_order(_order(tmp_path), **REPLAY)

    assert slurm.commands("sbatch") == []
    payload = agent.state.write_remote_status.call_args.args[1]
    assert (payload["status"], payload["driver_job_id"]) == ("RUNNING", "4243")


def test_unanswerable_driver_lookup_blocks_instead_of_submitting(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    slurm = FakeSlurm({"squeue": _ok("", 1, "slurm_load_jobs error: Socket timed out")})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._process_order(_order(tmp_path), **REPLAY)

    assert slurm.commands("sbatch") == []
    payload = agent.state.write_remote_status.call_args.args[1]
    assert payload["status"] == "BLOCKED"
    assert "Socket timed out" in payload["error"]


# --- Finding 3: BLOCKED retry after an sbatch that actually succeeded -------


def test_blocked_retry_after_sbatch_timeout_adopts_the_submitted_driver(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    submitted: list[str] = []

    def squeue(args):
        return subprocess.CompletedProcess(args, 0, "".join(f"{j}\n" for j in submitted), "")

    def sbatch(args):
        submitted.append("777")  # Slurm accepted it, but the reply was lost.
        return subprocess.TimeoutExpired(args, 60)

    slurm = FakeSlurm({"squeue": squeue, "sacct": _ok(""), "sbatch": sbatch})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._process_order(_order(tmp_path))
    assert agent.state.write_remote_status.call_args.args[1]["status"] == "BLOCKED"

    agent._process_order(_order(tmp_path), **REPLAY)  # the BLOCKED retry

    assert len(slurm.commands("sbatch")) == 1
    payload = agent.state.write_remote_status.call_args.args[1]
    assert (payload["status"], payload["driver_job_id"]) == ("RUNNING", "777")


def test_fresh_claim_submits_without_slurm_lookup(tmp_path, monkeypatch):
    # A freshly claimed QUEUED order cannot have a driver yet, so an
    # slurmdbd/slurmctld outage must not block new work.
    agent = _make_agent(tmp_path)
    slurm = FakeSlurm({"sbatch": _ok("100\n")})
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._process_order(_order(tmp_path))

    assert [call[0] for call in slurm.calls] == ["sbatch"]
    payload = agent.state.write_remote_status.call_args.args[1]
    assert (payload["status"], payload["driver_job_id"]) == ("RUNNING", "100")


@pytest.mark.parametrize(
    "status,order_extra",
    [
        ({"status": "BLOCKED", "retry_at": "2000-01-01T00:00:00+00:00"}, {}),
        ({"status": "CLAIMED"}, {"status": "CLAIMED", "claimed_at": _iso(600)}),
    ],
)
def test_replays_adopt_running_driver_even_when_storage_is_low(
    tmp_path, monkeypatch, status, order_extra
):
    agent = _make_agent(tmp_path)
    agent._started_at = time.time()
    agent._capacity_guard.check.return_value = False
    agent.state.get_remote_batches_for_target.return_value = [_order(tmp_path, **order_extra)]
    agent.state.read_remote_agent.return_value = {"heartbeat_at": _iso(5)}
    client = MagicMock()
    client.read_status.return_value = status
    monkeypatch.setattr(remote, "RemoteDispatchClient", lambda *a, **k: client)
    live: list[str] = []
    slurm = FakeSlurm({
        "squeue": lambda args: subprocess.CompletedProcess(args, 0, "".join(live), ""),
        "sacct": _ok(""),
        "sbatch": _ok("999\n"),
    })
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    agent._refresh_claimed_orders()  # no driver, low storage: nothing submitted
    assert slurm.commands("sbatch") == []
    client.write_status.assert_not_called()

    live.append("4242\n")  # the driver is in fact running
    agent._refresh_claimed_orders()
    assert slurm.commands("sbatch") == []
    payload = client.write_status.call_args.args[1]
    assert (payload["status"], payload["driver_job_id"]) == ("RUNNING", "4242")

    live.clear()
    agent._capacity_guard.check.return_value = True
    agent._refresh_claimed_orders()
    assert len(slurm.commands("sbatch")) == 1


def test_multiple_live_drivers_are_reported(tmp_path, monkeypatch, caplog):
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(remote.subprocess, "run", FakeSlurm({"squeue": _ok("4242\n4250\n")}))

    with caplog.at_level("WARNING", logger="mussel-dispatcher"):
        agent._process_order(_order(tmp_path), **REPLAY)

    assert agent.state.write_remote_status.call_args.args[1]["driver_job_id"] == "4250"
    assert "leaving duplicates 4242 running" in caplog.text


# --- Finding 2: drivers and hooks run from the immutable staged release -----


def _staged_release(tmp_path: Path) -> tuple[Path, str]:
    stage_root = tmp_path / "releases"
    source = tmp_path / "source"
    (source / "dispatcher" / "mussel_dispatcher").mkdir(parents=True)
    (source / "dispatcher" / "mussel_dispatcher" / "__init__.py").write_text("")
    (source / "dispatcher" / "remote_nextflow.config").write_text("// remote\n")
    (source / "main.nf").write_text("workflow {}\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "release")
    revision = _git(source, "rev-parse", "HEAD")
    stage = stage_root / "default" / revision
    stage.parent.mkdir(parents=True)
    _git(source, "worktree", "add", "--detach", str(stage), revision)
    return stage_root, revision


def test_driver_runs_from_staged_release_not_the_active_link(tmp_path):
    stage_root, revision = _staged_release(tmp_path)
    active = tmp_path / "mussel-nf.active"
    agent = _make_agent(
        tmp_path,
        remote_repo_dir=str(active),
        release_stage_root=str(stage_root),
        nextflow_config=str(active / "dispatcher" / "remote_nextflow.config"),
    )
    agent.dry_run = True
    order = _order(tmp_path, source_revision=revision)

    agent._process_order(order)

    stage = stage_root / "default" / revision
    script = (tmp_path / "run" / "driver.sbatch").read_text()
    assert f"git -C {stage} rev-parse HEAD" in script
    assert f"-c {stage}/dispatcher/remote_nextflow.config" in script
    assert str(active) not in script


def test_driver_without_stage_root_uses_active_link(tmp_path):
    active = tmp_path / "mussel-nf.active"
    agent = _make_agent(tmp_path, remote_repo_dir=str(active))
    order = agent._normalize_order(_order(tmp_path, source_revision="a" * 40))

    assert agent._pin_order_to_staged_release(order)["remote_repo_dir"] == str(active)


def test_driver_with_missing_stage_blocks_then_retires_order(tmp_path, monkeypatch):
    active = tmp_path / "mussel-nf.active"
    agent = _make_agent(
        tmp_path, remote_repo_dir=str(active), release_stage_root=str(tmp_path / "none")
    )
    slurm = FakeSlurm({"squeue": _ok(""), "sacct": _ok("")})
    monkeypatch.setattr(remote.subprocess, "run", slurm)
    order = _order(tmp_path, source_revision="a" * 40)

    for attempt in range(1, remote._STAGE_MISSING_MAX_ATTEMPTS + 1):
        agent._process_order(order, **REPLAY)
        payload = agent.state.write_remote_status.call_args.args[1]
        assert payload["status"] == "BLOCKED"
        assert "is missing" in payload["error"]
        if attempt < remote._STAGE_MISSING_MAX_ATTEMPTS:
            agent.state.requeue_blocked_remote_batch.assert_not_called()

    assert slurm.commands("sbatch") == []
    assert payload["failure_class"] == "STALE_ORDER"
    agent.state.requeue_blocked_remote_batch.assert_called_once()


def test_driver_stage_check_timeout_is_transient(tmp_path, monkeypatch):
    stage_root, revision = _staged_release(tmp_path)
    agent = _make_agent(tmp_path, release_stage_root=str(stage_root))
    slurm = FakeSlurm({"squeue": _ok(""), "sacct": _ok("")})

    def run(args, **kwargs):
        if args[0] == "git" and args[-2:] == ["rev-parse", "HEAD"]:
            raise subprocess.TimeoutExpired(args, 15)  # slow/hung filesystem
        return slurm(args, **kwargs)

    monkeypatch.setattr(remote.subprocess, "run", run)
    order = _order(tmp_path, source_revision=revision)
    for _ in range(remote._STAGE_MISSING_MAX_ATTEMPTS + 2):
        agent._process_order(order, **REPLAY)

    payload = agent.state.write_remote_status.call_args.args[1]
    assert payload["status"] == "BLOCKED"
    assert payload["failure_class"] == "REMOTE_PREPARATION"
    assert "could not verify staged release" in payload["error"]
    agent.state.requeue_blocked_remote_batch.assert_not_called()


def test_hook_repo_refuses_active_link_when_staged_release_is_missing(tmp_path):
    active = tmp_path / "mussel-nf.active"
    agent = _make_agent(
        tmp_path, remote_repo_dir=str(active), release_stage_root=str(tmp_path / "none")
    )
    order = {"remote_repo_dir": str(active), "source_revision": "a" * 40}

    with pytest.raises(remote.StagedReleaseError, match="is missing") as raised:
        agent._hook_repo_dir(order)
    assert raised.value.transient is False


def test_missing_hook_stage_backs_off_then_blocks_hooks(tmp_path):
    agent = _make_agent(tmp_path)
    missing = remote.StagedReleaseError("staged release x is missing", transient=False)

    for _ in range(remote._STAGE_MISSING_MAX_ATTEMPTS - 1):
        agent._defer_finalization_for_stage("b1", missing)
    delays = [
        call.kwargs["delay_seconds"]
        for call in agent.state.mark_remote_finalization_retry.call_args_list
    ]
    assert delays == [60, 120]
    agent.state.mark_remote_finalization_blocked.assert_not_called()

    agent._defer_finalization_for_stage("b1", missing)

    blocked = agent.state.mark_remote_finalization_blocked.call_args
    assert blocked.args == ("b1",)
    assert "staged release x is missing" in blocked.kwargs["error"]
    assert blocked.kwargs["payload"]["blocked_reason"] == "staged release missing"


def test_transient_hook_stage_error_never_blocks(tmp_path):
    agent = _make_agent(tmp_path)
    transient = remote.StagedReleaseError("git timed out", transient=True)

    for _ in range(10):
        agent._defer_finalization_for_stage("b1", transient)

    agent.state.mark_remote_finalization_blocked.assert_not_called()
    delays = [
        call.kwargs["delay_seconds"]
        for call in agent.state.mark_remote_finalization_retry.call_args_list
    ]
    assert delays[:3] == [60, 120, 240] and max(delays) == remote._STAGE_RETRY_MAX_SECONDS


def test_finalization_routes_stage_errors_to_backoff(tmp_path):
    agent = _make_agent(tmp_path)
    agent._finalization_owner = "agent-a"
    agent._finalization_lock = threading.RLock()
    agent._finalization_threads = {}
    agent._finalization_lease_names = {}
    agent._finalize_success = MagicMock(
        side_effect=remote.StagedReleaseError("stage gone", transient=False)
    )
    agent.state.release_finalization_lease.return_value = True

    agent._finalize_and_publish({"batch_id": "b1"}, "123", "lease")

    agent.state.mark_remote_finalization_retry.assert_called_once_with(
        "b1", error="stage gone", delay_seconds=60
    )


def test_hook_repo_uses_staged_release(tmp_path):
    stage_root, revision = _staged_release(tmp_path)
    agent = _make_agent(tmp_path, release_stage_root=str(stage_root))
    order = {"remote_repo_dir": str(tmp_path / "active"), "source_revision": revision}

    assert agent._hook_repo_dir(order) == str(stage_root / "default" / revision)


# --- Finding 4: watchdog exits non-zero ---------------------------------------


def test_watchdog_exits_nonzero_instead_of_sigterm(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, agent_summary_interval_seconds=5)
    agent._agent_heartbeat_stop = threading.Event()
    agent._agent_heartbeat_thread = None
    agent._agent_watchdog_thread = None
    agent._main_loop_progress_lock = threading.Lock()
    agent._main_loop_progress_monotonic = time.monotonic() - 10_000
    agent._main_loop_progress_at = _iso(10_000)
    agent._publish_agent_snapshot = lambda **_kwargs: None
    exits: list[int] = []

    def fake_exit(code):
        exits.append(code)
        agent._agent_heartbeat_stop.set()

    monkeypatch.setattr(remote, "_hard_exit", fake_exit)
    monkeypatch.setattr(
        remote.os, "kill", lambda *_a: pytest.fail("watchdog must not SIGTERM itself")
    )

    agent._start_agent_heartbeat()
    agent._agent_watchdog_thread.join(timeout=15)

    assert exits == [70]


def test_hard_exit_flushes_and_calls_os_exit(monkeypatch):
    codes: list[int] = []
    monkeypatch.setattr(remote.os, "_exit", codes.append)
    remote._hard_exit(70)
    assert codes == [70]


# --- Finding 5: hook polling and submission -----------------------------------


def _hook_agent(tmp_path, hook_row=None):
    agent = _make_agent(tmp_path, finalization_poll_interval_seconds=1)
    agent._finalization_owner = "agent-a"
    agent._finalization_lease_names = {}
    agent.state.owns_finalization_lease.return_value = True
    row = {"hook_index": 0, "status": "RUNNING", "active_attempt_number": 3}
    row.update(hook_row or {})
    agent.state.get_batch_hooks.return_value = {"20261028T101112_ab12cd34": [row]}
    (tmp_path / "run").mkdir(exist_ok=True)
    return agent


def test_hook_poll_ignores_failed_squeue_and_non_terminal_sacct(tmp_path, monkeypatch):
    agent = _hook_agent(tmp_path, {"executor_job_id": "555"})
    monkeypatch.setattr(remote.time, "sleep", lambda _s: None)
    squeue = iter([
        subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Socket timed out"),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ])
    sacct = iter(["RUNNING|0:0|node001\n", "COMPLETED|0:0|node001\n"])
    slurm = FakeSlurm({
        "squeue": lambda args: next(squeue),
        "sacct": lambda args: subprocess.CompletedProcess(args, 0, next(sacct), ""),
    })
    monkeypatch.setattr(remote.subprocess, "run", slurm)

    result = agent._run_hook_slurm(
        _order(tmp_path), ["hook"], cwd=str(tmp_path), env={}, hook_index=0
    )

    assert result.returncode == 0
    assert len(slurm.commands("sacct")) == 2
    final = agent.state.set_batch_hook_executor.call_args.kwargs
    assert final["state"] == "COMPLETED"


class FakeCluster:
    """Minimal Slurm model: squeue only knows live jobs, sacct knows all.

    Like real Slurm, ``squeue -j`` on a finished (purged) job exits 1 with
    "Invalid job id specified".
    """

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.next_id = 9001
        self.calls: list[list[str]] = []
        self.sbatch_outcome = "ok"  # or "timeout" (accepted, reply lost)
        self.new_job_state = "COMPLETED"
        self.new_job_exit = 0
        self.running_polls = 0
        self.sacct_down = False

    def add(self, name, state, exit_code=0, running_polls=0):
        job_id = str(self.next_id)
        self.next_id += 1
        self.jobs[job_id] = {
            "name": name, "state": state, "exit": exit_code, "polls": running_polls,
        }
        return job_id

    def commands(self, name):
        return [call for call in self.calls if call[0] == name]

    def __call__(self, args, **kwargs):
        if args[0] == "git":
            return _REAL_RUN(args, **kwargs)
        self.calls.append(list(args))
        live = {"PENDING", "RUNNING"}
        if args[0] == "sbatch":
            if "--job-name" in args:
                name = args[args.index("--job-name") + 1]
            else:  # driver: name comes from the script's #SBATCH header
                header = Path(args[-1]).read_text().split("--job-name=", 1)[1]
                name = header.split()[0]
            state = "RUNNING" if self.running_polls else self.new_job_state
            job_id = self.add(name, state, self.new_job_exit, self.running_polls)
            if self.sbatch_outcome == "timeout":
                raise subprocess.TimeoutExpired(args, 60)
            return subprocess.CompletedProcess(args, 0, f"{job_id}\n", "")
        if args[0] == "squeue" and "-n" in args:
            name = args[args.index("-n") + 1]
            ids = [j for j, job in self.jobs.items() if job["name"] == name and job["state"] in live]
            return subprocess.CompletedProcess(args, 0, "".join(f"{j}\n" for j in ids), "")
        if args[0] == "squeue":
            job = self.jobs.get(args[args.index("-j") + 1])
            if job and job["state"] in live:
                job["polls"] -= 1
                if job["polls"] <= 0:
                    job["state"] = self.new_job_state
                return subprocess.CompletedProcess(args, 0, "RUNNING\n", "")
            return subprocess.CompletedProcess(args, 1, "", PURGED)
        if args[0] == "sacct":
            if self.sacct_down:
                return subprocess.CompletedProcess(
                    args, 1, "", "sacct: error: Problem talking to the database"
                )
            if "--name" in args:
                name = args[args.index("--name") + 1]
                rows = [
                    f"{j}|{job['state']}|{job['exit']}:0"
                    for j, job in self.jobs.items() if job["name"] == name
                ]
            else:
                job = self.jobs[args[args.index("-j") + 1]]
                rows = [f"{job['state']}|{job['exit']}:0|node001"]
            return subprocess.CompletedProcess(args, 0, "".join(r + "\n" for r in rows), "")
        raise AssertionError(f"unexpected command: {args}")


HOOK_NAME = "mussel-hook-ab12cd34-0"


def _run_hook(agent, tmp_path):
    return agent._run_hook_slurm(
        _order(tmp_path), ["hook"], cwd=str(tmp_path), env={}, hook_index=0
    )


def test_hook_poll_reads_sacct_for_purged_job(tmp_path, monkeypatch):
    # Hook row lost its job id; once Slurm purges the finished job squeue
    # exits 1, and polling must still reach sacct instead of spinning.
    cluster = FakeCluster()
    job_id = cluster.add("other-name", "COMPLETED")
    agent = _hook_agent(tmp_path, {"executor_job_id": job_id})
    monkeypatch.setattr(remote.time, "sleep", lambda _s: pytest.fail("must not wait"))
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    assert _run_hook(agent, tmp_path).returncode == 0
    assert agent.state.set_batch_hook_executor.call_args.kwargs["state"] == "COMPLETED"


def test_hook_executor_write_failure_does_not_abandon_submitted_job(tmp_path, monkeypatch):
    agent = _hook_agent(tmp_path, {"executor_job_id": ""})
    agent.state.set_batch_hook_executor.side_effect = [RuntimeError("db down"), None]
    monkeypatch.setattr(remote.time, "sleep", lambda _s: None)
    cluster = FakeCluster()
    cluster.running_polls = 2
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    result = _run_hook(agent, tmp_path)

    assert result.returncode == 0
    assert len(cluster.commands("sbatch")) == 1
    assert cluster.jobs["9001"]["state"] == "COMPLETED"
    assert agent.state.set_batch_hook_executor.call_args.kwargs == {
        "job_id": "9001", "state": "COMPLETED", "expected_attempt_number": 3,
    }


def test_hook_retry_adopts_live_job_after_sbatch_timeout(tmp_path, monkeypatch):
    agent = _hook_agent(tmp_path, {"executor_job_id": ""})
    monkeypatch.setattr(remote.time, "sleep", lambda _s: None)
    cluster = FakeCluster()
    cluster.sbatch_outcome = "timeout"
    cluster.running_polls = 3
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    first = _run_hook(agent, tmp_path)
    assert first.returncode == 78
    assert "category=TRANSIENT_SERVICE" in first.stderr

    second = _run_hook(agent, tmp_path)

    assert second.returncode == 0
    assert len(cluster.commands("sbatch")) == 1
    adopted = agent.state.set_batch_hook_executor.call_args_list[0].kwargs
    assert (adopted["job_id"], adopted["state"]) == ("9001", "SUBMITTED")


def test_hook_retry_adopts_success_of_job_that_finished_after_sbatch_timeout(
    tmp_path, monkeypatch
):
    agent = _hook_agent(tmp_path, {"executor_job_id": ""})
    monkeypatch.setattr(remote.time, "sleep", lambda _s: None)
    cluster = FakeCluster()
    cluster.sbatch_outcome = "timeout"  # job runs to completion, reply lost
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    assert _run_hook(agent, tmp_path).returncode == 78
    second = _run_hook(agent, tmp_path)

    assert second.returncode == 0
    assert len(cluster.commands("sbatch")) == 1  # the WDS append ran once
    assert agent.state.set_batch_hook_executor.call_args.kwargs["job_id"] == "9001"


def test_hook_retry_resubmits_after_previous_job_failed(tmp_path, monkeypatch):
    agent = _hook_agent(tmp_path, {"executor_job_id": ""})
    monkeypatch.setattr(remote.time, "sleep", lambda _s: None)
    cluster = FakeCluster()
    cluster.add(HOOK_NAME, "FAILED", exit_code=1)
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    result = _run_hook(agent, tmp_path)

    assert result.returncode == 0
    assert len(cluster.commands("sbatch")) == 1


def test_hook_is_not_submitted_when_sacct_cannot_answer(tmp_path, monkeypatch):
    agent = _hook_agent(tmp_path, {"executor_job_id": ""})
    cluster = FakeCluster()
    cluster.sacct_down = True
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    result = _run_hook(agent, tmp_path)

    assert result.returncode == 78
    assert "TRANSIENT_SERVICE" in result.stderr
    assert cluster.commands("sbatch") == []


# --- Finding 6: terminal driver failure releases slides before remote_status --


def _failed_driver_agent(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    order = _order(
        tmp_path,
        status="RUNNING",
        slides=[{"slide_id": "S1", "slide_path": "/s/S1.svs"}],
    )
    agent.state.get_remote_batches_for_target.return_value = [order]
    agent._driver_slurm_state = lambda _job: {"state": "FAILED", "exit_code": 1, "source": "sacct"}
    client = MagicMock()
    client.read_status.return_value = {"status": "RUNNING", "driver_job_id": "4242"}
    monkeypatch.setattr(remote, "RemoteDispatchClient", lambda *a, **k: client)
    return agent, client


def _written_statuses(client) -> list[str]:
    return [call.args[1].get("status") for call in client.write_status.call_args_list]


def test_crash_while_releasing_failed_batch_leaves_it_revisitable(tmp_path, monkeypatch):
    agent, client = _failed_driver_agent(tmp_path, monkeypatch)
    agent.state.get_slide_statuses.return_value = {
        "S1": {"status": "DISPATCHED", "batch_id": "20261028T101112_ab12cd34"}
    }
    agent.state.mark_slides_complete.side_effect = RuntimeError("connection lost")

    agent._refresh_claimed_orders()

    assert "FAILED" not in _written_statuses(client)

    # The batch is still listed (not terminal) and is finished on a later pass.
    agent._order_failures.clear()
    agent.state.mark_slides_complete.side_effect = None
    agent._refresh_claimed_orders()
    assert _written_statuses(client)[-1] == "FAILED"


def test_replayed_terminal_failure_does_not_recharge_released_slides(tmp_path, monkeypatch):
    agent, client = _failed_driver_agent(tmp_path, monkeypatch)
    agent.state.get_slide_statuses.return_value = {
        "S1": {"status": "FAILED", "batch_id": "20261028T101112_ab12cd34"}
    }

    agent._refresh_claimed_orders()

    agent.state.complete_batch.assert_called_once()
    agent.state.mark_slides_complete.assert_not_called()
    assert _written_statuses(client)[-1] == "FAILED"


def test_terminal_failure_releases_slides_then_writes_status(tmp_path, monkeypatch):
    agent, client = _failed_driver_agent(tmp_path, monkeypatch)
    agent.state.get_slide_statuses.return_value = {
        "S1": {"status": "DISPATCHED", "batch_id": "20261028T101112_ab12cd34"}
    }
    events: list[str] = []
    agent.state.mark_slides_complete.side_effect = lambda *a, **k: events.append("slides")
    client.write_status.side_effect = lambda _b, payload: events.append(payload["status"])

    agent._refresh_claimed_orders()

    assert events == ["slides", "FAILED"]


# --- Finding 7: runtime fingerprint covers tracked files only ----------------


def test_runtime_manifest_ignores_untracked_and_ignored_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "modules").mkdir(parents=True)
    (repo / "bin").mkdir()
    (repo / "main.nf").write_text("workflow {}\n")
    (repo / "nextflow.config").write_text("\n")
    (repo / "modules" / "pipeline.nf").write_text("process P {}\n")
    (repo / ".gitignore").write_text("*.log\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    (repo / "modules" / "scratch.nf").write_text("untracked\n")
    (repo / "bin" / "debug.log").write_text("ignored\n")

    manifest = _runtime_sha256_manifest(str(repo))

    assert set(manifest) == {"main.nf", "nextflow.config", "modules/pipeline.nf"}


# --- Review: per-order isolation keeps one bad batch from halting the target --


def test_persistently_failing_order_does_not_block_healthy_orders(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    bad = _order(tmp_path, batch_id="bad", status="RUNNING")
    good = _order(tmp_path, batch_id="good", status="RUNNING")
    agent.state.get_remote_batches_for_target.return_value = [bad, good]
    states = {"4242": "FAILED", "4343": "RUNNING"}
    agent._driver_slurm_state = lambda job: {"state": states[job], "exit_code": 1}
    agent.state.complete_batch.side_effect = RuntimeError("deadlock detected")
    client = MagicMock()
    client.read_status.side_effect = lambda batch: {
        "status": "RUNNING", "driver_job_id": "4242" if batch == "bad" else "4343",
    }
    monkeypatch.setattr(remote, "RemoteDispatchClient", lambda *a, **k: client)

    for _ in range(3):
        assert agent._refresh_claimed_orders() is True

    good_writes = [c for c in client.write_status.call_args_list if c.args[0] == "good"]
    assert good_writes and good_writes[0].args[1]["status"] == "RUNNING"
    # The bad order is backed off rather than retried (and logged) every poll.
    assert agent.state.complete_batch.call_count == 1
    assert agent._order_failures["bad"][0] == 1


def test_run_loop_survives_a_failed_refresh_cycle(tmp_path):
    agent = _make_agent(tmp_path)
    agent.once = True
    for name in (
        "_start_agent_heartbeat", "_stop_agent_heartbeat", "_mark_main_loop_progress",
        "_publish_agent_snapshot", "_reconcile_finalization_workers",
        "_reconcile_finalization_state", "_reconcile_execution_cleanup",
        "_reconcile_hook_executors",
    ):
        setattr(agent, name, MagicMock())
    agent._refresh_claimed_orders = MagicMock(side_effect=RuntimeError("listing failed"))
    agent._claim_and_process_one = MagicMock(return_value=False)

    assert agent.run() == 0
    agent._claim_and_process_one.assert_called_once()


# --- Review: schema-v3 order verified end to end against the pinned release ---


def test_schema_v3_order_verifies_and_runs_from_pinned_release(tmp_path, monkeypatch):
    from mussel_dispatcher.contract import (
        execution_profile_hash, release_manifest_hash, tracked_source_manifest,
    )

    source = tmp_path / "source"
    (source / "dispatcher" / "mussel_dispatcher").mkdir(parents=True)
    (source / "dispatcher" / "mussel_dispatcher" / "__init__.py").write_text("")
    (source / "dispatcher" / "remote_nextflow.config").write_text("// remote v1\n")
    (source / "modules").mkdir()
    (source / "modules" / "pipeline.nf").write_text("process P {}\n")
    (source / "main.nf").write_text("workflow {}\n")
    (source / "nextflow.config").write_text("\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "v1")
    revision = _git(source, "rev-parse", "HEAD")
    runtime = _runtime_sha256_manifest(str(source))
    manifest = release_manifest_hash(tracked_source_manifest(str(source)))

    stage_root = tmp_path / "releases"
    stage = stage_root / "default" / revision
    stage.parent.mkdir(parents=True)
    _git(source, "worktree", "add", "--detach", str(stage), revision)

    # Activation has since moved the active link to a newer release.
    (source / "modules" / "pipeline.nf").write_text("process P_V2 {}\n")
    _git(source, "commit", "-qam", "v2")
    active = tmp_path / "mussel-nf.active"
    active.symlink_to(source, target_is_directory=True)

    agent = _make_agent(
        tmp_path,
        remote_repo_dir=str(active),
        release_stage_root=str(stage_root),
        nextflow_config=str(active / "dispatcher" / "remote_nextflow.config"),
    )
    params = (
        "tiling:\n  seg_model: neural\n"
        "featurize:\n  model_types: [titan_slide]\n  slide_batch_size: 1\n"
        "  slide_max_patches: 4096\n  model_precision_overrides: {}\n"
    )
    params_sha = hashlib.sha256(params.encode()).hexdigest()
    order = _order(
        tmp_path,
        schema_version=3,
        source_revision=revision,
        release_manifest_sha256=manifest,
        runtime_sha256=runtime,
        execution_profile_sha256=execution_profile_hash(agent.cfg, target=agent.target),
        nextflow_params_file=str(tmp_path / "run" / "params.yaml"),
        nextflow_params_content=params,
        nextflow_params_sha256=params_sha,
        effective_params_sha256=params_sha,
    )
    cluster = FakeCluster()
    cluster.new_job_state = "RUNNING"
    monkeypatch.setattr(remote.subprocess, "run", cluster)

    agent._process_order(order)

    payload = agent.state.write_remote_status.call_args.args[1]
    assert payload["status"] == "RUNNING", payload.get("error")
    script = (tmp_path / "run" / "driver.sbatch").read_text()
    assert f"git -C {stage} rev-parse HEAD" in script
    assert f"{stage}/modules/pipeline.nf" in script
    assert str(active) not in script

    # The same order run through the active link fails release verification.
    unpinned = _make_agent(tmp_path, remote_repo_dir=str(active))
    unpinned._process_order(order)
    error = unpinned.state.write_remote_status.call_args.args[1]["error"]
    assert "source release drift" in error
