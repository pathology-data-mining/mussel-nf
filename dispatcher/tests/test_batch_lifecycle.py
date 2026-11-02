from __future__ import annotations

import csv
import subprocess
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mussel_dispatcher.batch_lifecycle import (
    BatchFinalizer,
    FinalizationContext,
    FinalizationWorkerPool,
    NextflowRunSpec,
    PostBatchHooks,
    build_nextflow_command,
    write_batch_csv,
)
from mussel_dispatcher.config import Config, RemoteTargetConfig, WatcherConfig
from mussel_dispatcher.remote import RemoteAgent, RemoteDispatchClient
from mussel_dispatcher.state import StateStore


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


def add_running_batch(state: StateStore, tmp_path: Path, batch_id: str = "batch-1") -> None:
    state.add_slide("s3://slides/a.svs", "a")
    state.add_batch(
        batch_id,
        str(tmp_path / "batch.csv"),
        str(tmp_path / "work"),
        1,
        str(tmp_path / "batch.log"),
    )
    assert state.mark_dispatched(["s3://slides/a.svs"], batch_id) == 1


def test_finalization_worker_pool_is_bounded_and_keyed():
    pool = FinalizationWorkerPool(1, thread_name_prefix="test-finalize")
    started = Event()
    release = Event()

    def blocked_work():
        started.set()
        release.wait(timeout=2)

    try:
        first = pool.submit("batch-a", blocked_work)
        assert first is not None
        assert started.wait(timeout=2)
        assert pool.submit("batch-a", lambda: None) is None
        assert pool.submit("batch-b", lambda: None) is None
        release.set()
        first.result(timeout=2)
        second = pool.submit("batch-b", lambda: None)
        assert second is not None
        second.result(timeout=2)
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_finalization_worker_pool_retains_unexpected_failure_until_drained():
    pool = FinalizationWorkerPool(1, thread_name_prefix="test-finalize-failure")

    def fail_work():
        raise RuntimeError("worker exploded")

    try:
        future = pool.submit("batch-failure", fail_work)
        assert future is not None
        with pytest.raises(RuntimeError, match="worker exploded"):
            future.result(timeout=2)
        outcomes = pool.drain_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["batch_id"] == "batch-failure"
        assert isinstance(outcomes[0]["exception"], RuntimeError)
        assert pool.drain_outcomes() == []
    finally:
        pool.shutdown(wait=True)


def test_batch_csv_is_backend_neutral_and_deduplicated(tmp_path):
    slides = [
        {"slide_id": "a", "slide_path": "gdc://file/a.svs", "needs_download": True},
        {"slide_id": "a", "slide_path": "s3://slides/a.svs", "needs_download": False},
        {"slide_id": "b", "slide_path": "s3://slides/b.svs", "oncotree_code": "LUAD"},
    ]
    local_path = tmp_path / "local.csv"
    remote_path = tmp_path / "remote.csv"
    write_batch_csv(local_path, slides, batch_id="local")
    write_batch_csv(remote_path, slides, batch_id="remote")

    assert local_path.read_bytes() == remote_path.read_bytes()
    rows = list(csv.DictReader(local_path.open()))
    assert [row["slide_id"] for row in rows] == ["a", "b"]
    assert rows[0]["slide_path"] == "s3://slides/a.svs"


def test_nextflow_command_builder_keeps_common_argument_order():
    command = build_nextflow_command(NextflowRunSpec(
        executable=["nextflow"],
        log_path="nf.log",
        repo_dir="/repo",
        profiles="cluster,conda",
        work_dir="/work",
        batch_csv="/run/batch.csv",
        outdir="/results",
        run_name="r1234",
        trace_path="trace.tsv",
        config_path="remote.config",
        params_path="params.yaml",
        tower_enabled=True,
        scratch_dir="/local-scratch/task",
        extra_args=["--foo", "bar"],
    ))
    assert command == [
        "nextflow", "-log", "nf.log", "run", "/repo", "-profile", "cluster,conda",
        "-work-dir", "/work", "--samples_csv", "/run/batch.csv", "--outdir", "/results",
        "-name", "r1234", "-with-trace", "trace.tsv", "--scratch_dir", "/local-scratch/task",
        "-with-tower", "-c", "remote.config", "-params-file", "params.yaml", "--foo", "bar",
    ]


def test_successful_hooks_are_skipped_during_replay(tmp_path):
    cfg = make_config(tmp_path, post_batch_hooks=[{
        "command": "python -m example",
        "args": ["--batch={batch_csv}"],
        "_phase": "wds",
    }])
    state = StateStore(str(tmp_path / "state.db"))
    add_running_batch(state, tmp_path)
    calls = []

    def executor(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    hooks = PostBatchHooks(cfg, "batch-1", state)
    assert hooks.run("/remote/batch.csv", phase="wds", executor=executor)
    assert hooks.run("/remote/batch.csv", phase="wds", executor=executor)
    assert len(calls) == 1
    assert state.get_batch_hooks(["batch-1"])["batch-1"][0]["status"] == "SUCCEEDED"

    cfg.post_batch_hooks[0]["args"] = ["--changed"]
    assert hooks.run("/remote/batch.csv", phase="wds", executor=executor)
    assert len(calls) == 2


def test_shared_finalizer_records_hook_failure_without_requeueing_slides(tmp_path):
    # "wds-hook" is not the mussel WDS appender, so its exit 2 is not a
    # coverage-only failure, and "connection refused" is transient: the hook
    # is retried and the slides stay untouched.
    outdir = tmp_path / "results"
    outdir.mkdir()
    batch_csv = tmp_path / "batch.csv"
    write_batch_csv(batch_csv, [{"slide_id": "a", "slide_path": "s3://slides/a.svs"}])
    cfg = make_config(
        tmp_path,
        outdir=str(outdir),
        watchers=[WatcherConfig(type="s3", wds_destinations={"model-a": "s3://wds"})],
        post_batch_hooks=[
            {"command": "wds-hook", "_phase": "wds"},
            {"command": "db-hook", "_phase": "db_sync"},
        ],
    )
    state = StateStore(str(tmp_path / "state.db"))
    add_running_batch(state, tmp_path)

    def executor(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 2 if cmd[0] == "wds-hook" else 0, "",
            "connection refused" if cmd[0] == "wds-hook" else "",
        )

    result = BatchFinalizer(cfg, "batch-1", state).finalize_success(
        FinalizationContext(
            batch_csv=str(batch_csv),
            outdir=str(outdir),
            repo_dir=cfg.repo_dir,
        ),
        hook_executor=executor,
    )

    assert not result.wds_hooks_succeeded
    assert not result.db_sync_hooks_succeeded
    assert state.get_batch("batch-1")["status"] == "FINALIZING"
    assert state.get_pending_slides() == []
    assert state.get_slides_by_id("a")[0]["status"] == "SUCCEEDED"
    assert state.get_slides_by_id("a")[0]["fail_count"] == 0
    assert [row["status"] for row in state.get_batch_hooks(["batch-1"])["batch-1"]] == [
        "FAILED", "PENDING"
    ]


def test_optional_hook_failure_does_not_block_batch_completion(tmp_path):
    cfg = make_config(
        tmp_path,
        post_batch_hooks=[{
            "command": "optional-hook",
            "required": False,
            "_phase": "wds",
        }],
    )
    state = StateStore(str(tmp_path / "state.db"))
    add_running_batch(state, tmp_path)

    def executor(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 7, "", "optional failure")

    result = BatchFinalizer(cfg, "batch-1", state).finalize_success(
        FinalizationContext(
            batch_csv=str(tmp_path / "batch.csv"),
            outdir=str(tmp_path / "results"),
            repo_dir=cfg.repo_dir,
        ),
        hook_executor=executor,
    )

    assert result.wds_hooks_succeeded
    assert result.db_sync_hooks_succeeded
    assert state.get_batch("batch-1")["status"] == "SUCCEEDED"
    assert state.get_batch_hooks(["batch-1"])["batch-1"][0]["status"] == "FAILED"


def test_stale_remote_writer_cannot_regress_claimed_finalization(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    state.add_batch(
        "batch-1", str(tmp_path / "batch.csv"), str(tmp_path / "work"), 0,
        str(tmp_path / "batch.log"), backend="remote_agent", target="hpc",
    )
    state.write_remote_status("batch-1", {"status": "RUNNING", "driver_job_id": "123"})

    assert state.claim_remote_finalization("batch-1", stale_after_seconds=300, owner="agent-a")
    state.write_remote_status("batch-1", {"status": "RUNNING", "heartbeat_at": "stale"})
    assert state.read_remote_status("batch-1")["status"] == "HOOKS_RUNNING"
    assert not state.transition_remote_status(
        "batch-1", expected_statuses={"RUNNING"}, status="SUCCEEDED"
    )


def test_remote_work_cleanup_cannot_remove_results_or_symlink_targets(tmp_path):
    agent, _state = make_remote_agent(tmp_path)
    agent.cfg.cleanup_work_dir = True
    run_dir = tmp_path / "remote" / "run"
    work_dir = run_dir / "work"
    results_dir = tmp_path / "results"
    work_dir.mkdir(parents=True)
    results_dir.mkdir()
    (results_dir / "keep.txt").write_text("keep")
    agent.target.remote_outdir = str(results_dir)

    agent._cleanup_remote_work_dir({
        "remote_run_dir": str(run_dir),
        "remote_work_dir": str(work_dir),
        "remote_outdir": str(results_dir),
    })
    assert not work_dir.exists()
    assert (results_dir / "keep.txt").exists()

    symlink = run_dir / "work-link"
    symlink.symlink_to(results_dir, target_is_directory=True)
    agent._cleanup_remote_work_dir({
        "remote_run_dir": str(run_dir),
        "remote_work_dir": str(symlink),
        "remote_outdir": str(results_dir),
    })
    assert symlink.is_symlink()
    assert (results_dir / "keep.txt").exists()


def test_remote_finalization_claim_is_exclusive_and_stale_recoverable(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    add_running_batch(state, tmp_path)
    state.update_remote_batch(
        "batch-1", remote_status="RUNNING", remote_heartbeat_at="2026-08-04T00:00:00+00:00"
    )

    assert state.claim_remote_finalization("batch-1", stale_after_seconds=300)
    assert not state.claim_remote_finalization("batch-1", stale_after_seconds=300)
    state.update_remote_batch(
        "batch-1", remote_status="HOOKS_RUNNING", remote_heartbeat_at="2000-01-01T00:00:00+00:00"
    )
    assert state.claim_remote_finalization("batch-1", stale_after_seconds=300)


def test_remote_agent_reclaims_persisted_finalization_after_restart(tmp_path, monkeypatch):
    agent, state = make_remote_agent(tmp_path)
    order = {
        "batch_id": "batch-1",
        "remote_run_dir": str(tmp_path / "remote" / "batch-1"),
        "remote_driver_job_id": "123",
    }
    state.get_remote_batches_for_target.return_value = [order]
    state.read_remote_status.return_value = {
        "status": "HOOKS_RETRY",
        "retry_at": "2000-01-01T00:00:00+00:00",
        "driver_job_id": "123",
    }
    state.claim_finalization_slot.return_value = "dispatcher-finalization:remote"
    state.claim_remote_finalization.return_value = True
    client = MagicMock()
    client.read_status.return_value = {
        "status": "HOOKS_RETRY",
        "retry_at": "2000-01-01T00:00:00+00:00",
        "driver_job_id": "123",
    }
    monkeypatch.setattr(
        "mussel_dispatcher.remote.RemoteDispatchClient",
        lambda *args, **kwargs: client,
    )
    agent._finalization_owner = "test-agent"
    agent._finalization_threads = {}
    agent._finalization_pool = MagicMock()
    agent._finalization_pool.running_count.return_value = 0
    agent._driver_slurm_queries = 0
    agent._status_writes_suppressed = 0
    agent._launch_finalization = MagicMock()

    assert agent._refresh_claimed_orders() is True
    state.claim_finalization_slot.assert_called_once_with(
        "batch-1",
        lane="remote",
        max_slots=1,
        owner="test-agent",
        ttl_seconds=300,
    )
    state.claim_remote_finalization.assert_called_once_with(
        "batch-1",
        stale_after_seconds=300,
        owner="test-agent",
        lease_name="dispatcher-finalization:remote",
    )
    agent._launch_finalization.assert_called_once()
    launch_order, launch_job_id = agent._launch_finalization.call_args.args[:2]
    assert launch_order["batch_id"] == "batch-1"
    assert launch_job_id == "123"
    assert agent._launch_finalization.call_args.kwargs["lease_name"] == (
        "dispatcher-finalization:remote"
    )
    payload = client.write_status.call_args.args[1]
    assert payload["status"] == "HOOKS_RUNNING"
    assert payload["nextflow_exit"] == 0


def test_remote_agent_reconciles_unexpected_finalizer_failure(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    agent._finalization_owner = "test-agent"
    agent._finalization_lease_names = {"batch-1": "dispatcher-finalization:remote:2"}
    agent._finalization_reconcile_health = {"errors": []}
    agent._finalization_pool = MagicMock()
    agent._finalization_pool.drain_outcomes.return_value = [{
        "batch_id": "batch-1",
        "cancelled": False,
        "exception": RuntimeError("unexpected worker failure"),
    }]
    state.release_finalization_lease.return_value = True

    agent._reconcile_finalization_workers()

    state.mark_remote_finalization_retry.assert_called_once_with(
        "batch-1", error="finalization worker raised RuntimeError: unexpected worker failure",
        delay_seconds=1, owner="test-agent", lease_name="dispatcher-finalization:remote:2",
    )
    state.release_finalization_lease.assert_called_once_with(
        "batch-1", owner="test-agent", lease_name="dispatcher-finalization:remote:2"
    )
    assert "batch-1" not in agent._finalization_lease_names


def test_remote_agent_marks_worker_slot_mismatch_degraded(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    agent._finalization_owner = "test-agent"
    agent._finalization_reconcile_health = {"errors": []}
    agent._finalization_reconcile_failures = 0
    agent._last_finalization_reconcile_at = 0
    agent._finalization_pool = MagicMock()
    agent._finalization_pool.running_batch_ids.return_value = {"batch-1", "batch-2"}
    state.reconcile_finalization_slots.return_value = {
        "durable_slots": 1,
        "process_active": 2,
        "repaired": [],
        "terminal_released": [],
        "expired": [],
        "foreign_live": [],
    }

    agent._reconcile_finalization_state()

    assert agent._finalization_reconcile_health["state"] == "DEGRADED"
    state.reconcile_finalization_slots.assert_called_once_with(
        lane="remote",
        max_slots=1,
        owner="test-agent",
        active_batch_ids={"batch-1", "batch-2"},
    )


def test_remote_agent_reports_foreign_live_slots_as_recovering(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    agent._finalization_owner = "test-agent"
    agent._finalization_reconcile_health = {"errors": []}
    agent._finalization_reconcile_failures = 0
    agent._last_finalization_reconcile_at = 0
    agent._finalization_pool = MagicMock()
    agent._finalization_pool.running_batch_ids.return_value = set()
    state.reconcile_finalization_slots.return_value = {
        "health_state": "RECOVERING",
        "durable_slots": 4,
        "process_active": 0,
        "repaired": [],
        "terminal_released": [],
        "expired": [],
        "foreign_live": [{"batch_id": f"batch-{i}"} for i in range(4)],
    }

    agent._reconcile_finalization_state()

    assert agent._finalization_reconcile_health["state"] == "RECOVERING"


def test_safe_release_treats_foreign_or_absent_lease_as_released(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    agent._finalization_owner = "test-agent"
    state.release_finalization_lease.return_value = False
    state.finalization_lease_status.return_value = {
        "owner": "replacement-agent",
        "batch_id": "batch-1",
    }

    assert agent._release_finalization_lease_safely("batch-1", "slot-1")
    assert state.release_finalization_lease.call_count == 1

def make_remote_agent(tmp_path: Path):
    cfg = make_config(tmp_path, state_backend="postgres")
    target = RemoteTargetConfig(name="hpc", remote_scratch_root=str(tmp_path / "remote"))
    state = MagicMock()
    agent = object.__new__(RemoteAgent)
    agent.cfg = cfg
    agent.target = target
    agent.once = False
    agent.dry_run = False
    agent.state = state
    return agent, state


def test_remote_log_tail_is_bounded_stable_and_includes_recent_hooks(tmp_path):
    agent, _state = make_remote_agent(tmp_path)
    run_dir = tmp_path / "remote" / "batch-logs"
    hooks_dir = run_dir / "hooks"
    hooks_dir.mkdir(parents=True)
    (run_dir / "driver.out").write_text("d" * (20 * 1024))
    (run_dir / "driver.err").write_text("driver error\n")
    (run_dir / "nextflow.log").write_text("nextflow line\n")
    for index in range(6):
        path = hooks_dir / f"hook_{index}.out"
        path.write_text(f"hook {index}\n")
        path.touch()

    order = {"batch_id": "batch-logs", "remote_run_dir": str(run_dir)}
    first = agent._capture_remote_log_tail(order)
    assert first is not None
    assert first["files"][0]["name"] == "driver.out"
    assert first["files"][0]["truncated"] is True
    assert len([row for row in first["files"] if row["name"].startswith("hooks/")]) == 4
    assert sum(len(row["text"].encode()) for row in first["files"]) <= 64 * 1024

    unchanged = agent._capture_remote_log_tail(order, first)
    assert unchanged == first
    assert unchanged["captured_at"] == first["captured_at"]

    (run_dir / "driver.err").write_text("different error\n")
    changed = agent._capture_remote_log_tail(order, first)
    assert changed is not None
    assert changed["content_sha256"] != first["content_sha256"]


def test_remote_agent_mirrors_logs_while_hooks_are_running(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    run_dir = tmp_path / "remote" / "batch-hooks"
    run_dir.mkdir(parents=True)
    (run_dir / "driver.out").write_text("remote output\n")
    order = {"batch_id": "batch-hooks", "remote_run_dir": str(run_dir)}
    state.get_remote_batches_for_target.return_value = [order]
    state.read_remote_status.return_value = {
        "status": "HOOKS_RUNNING",
        "heartbeat_at": "2099-01-01T00:00:00+00:00",
    }
    state.write_remote_status.return_value = True
    agent._driver_slurm_state = MagicMock()

    agent._refresh_claimed_orders()

    payload = state.write_remote_status.call_args.args[1]
    assert payload["batch_id"] == "batch-hooks"
    assert payload["log_tail"]["files"][0]["text"] == "remote output\n"
    assert not agent._driver_slurm_state.called


def test_remote_agent_uses_shared_finalizer_before_publishing_success(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    order = {"batch_id": "batch-1", "remote_run_dir": str(tmp_path / "remote" / "batch-1")}
    state.read_remote_agent.return_value = {"heartbeat_at": "2026-08-04T00:00:00+00:00"}
    state.get_remote_batches_for_target.return_value = [order]
    state.read_remote_status.return_value = {"status": "RUNNING", "driver_job_id": "123"}
    state.claim_remote_finalization.return_value = True
    state.get_batch_hooks.return_value = {"batch-1": []}
    agent._driver_slurm_state = MagicMock(return_value={"state": "COMPLETED", "exit_code": 0})
    agent._finalize_success = MagicMock(return_value=SimpleNamespace(
        wds_hooks_succeeded=False, db_sync_hooks_succeeded=False
    ))

    agent._refresh_claimed_orders()

    assert agent._finalize_success.call_count == 1
    assert agent._finalize_success.call_args.args[0]["batch_id"] == "batch-1"
    terminal_payloads = [
        call.args[1] for call in state.write_remote_status.call_args_list
        if call.args[1].get("status") == "SUCCEEDED"
    ]
    assert len(terminal_payloads) == 0
    state.mark_remote_finalization_retry.assert_called_once()
    assert not state.mark_slides_complete.called


def test_remote_driver_failure_completes_batch_and_slides_once(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    order = {"batch_id": "batch-1", "remote_run_dir": str(tmp_path / "remote" / "batch-1")}
    state.read_remote_agent.return_value = {}
    state.get_remote_batches_for_target.return_value = [order]
    state.read_remote_status.return_value = {"status": "RUNNING", "driver_job_id": "123"}
    agent._driver_slurm_state = MagicMock(return_value={"state": "OUT_OF_MEMORY", "exit_code": 137})
    agent._cleanup_remote_work_dir = MagicMock()

    agent._refresh_claimed_orders()

    state.complete_batch.assert_called_once_with("batch-1", 137)
    state.mark_slides_complete.assert_called_once_with(
        "batch-1", False, charge_fail_count=False
    )
    assert agent._cleanup_remote_work_dir.call_count == 1
    assert agent._cleanup_remote_work_dir.call_args.args[0]["batch_id"] == "batch-1"


def test_remote_client_rejects_non_postgres_control_plane(tmp_path):
    cfg = make_config(tmp_path, state_backend="sqlite")
    target = RemoteTargetConfig(name="hpc")
    try:
        RemoteDispatchClient(cfg, target)
    except RuntimeError as exc:
        assert "state_backend=postgres" in str(exc)
    else:
        raise AssertionError("remote client accepted a non-PostgreSQL control plane")


def _make_staged_run(tmp_path: Path, batch_id: str) -> Path:
    run_dir = tmp_path / "remote" / batch_id
    staging = run_dir / "wds-staging" / "optimus" / "BRCA"
    staging.mkdir(parents=True)
    (staging / "000000.tar").write_bytes(b"shard")
    (run_dir / "driver.out").write_text("log\n")
    return run_dir


def test_wds_staging_cleanup_removes_only_staging(tmp_path):
    agent, _state = make_remote_agent(tmp_path)
    agent.cfg.cleanup_work_dir = True
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "keep.pt").write_bytes(b"published")
    run_dir = _make_staged_run(tmp_path, "batch-1")

    assert agent._cleanup_remote_wds_staging({
        "batch_id": "batch-1",
        "remote_run_dir": str(run_dir),
        "remote_outdir": str(results_dir),
    })

    assert not (run_dir / "wds-staging").exists()
    assert (run_dir / "driver.out").exists()
    assert (results_dir / "keep.pt").exists()


def test_wds_staging_cleanup_disabled_without_cleanup_work_dir(tmp_path):
    agent, _state = make_remote_agent(tmp_path)
    agent.cfg.cleanup_work_dir = False
    run_dir = _make_staged_run(tmp_path, "batch-1")

    assert not agent._cleanup_remote_wds_staging(
        {"batch_id": "batch-1", "remote_run_dir": str(run_dir)}
    )
    assert agent._reconcile_wds_staging_cleanup(limit=50, budget_seconds=15.0) == 0
    assert (run_dir / "wds-staging").exists()


def test_wds_staging_reconcile_only_removes_succeeded_batches(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    agent.cfg.cleanup_work_dir = True
    rows = {
        "ok": {"backend": "remote_agent", "target": "hpc", "remote_status": "SUCCEEDED"},
        "blocked": {"backend": "remote_agent", "target": "hpc", "remote_status": "HOOKS_BLOCKED"},
        "retry": {"backend": "remote_agent", "target": "hpc", "remote_status": "HOOKS_RETRY"},
        "failed": {"backend": "remote_agent", "target": "hpc", "remote_status": "FAILED"},
        "other-target": {"backend": "remote_agent", "target": "other-hpc", "remote_status": "SUCCEEDED"},
    }
    run_dirs = {name: _make_staged_run(tmp_path, name) for name in [*rows, "unknown"]}
    state.get_batch.side_effect = rows.get

    assert agent._reconcile_wds_staging_cleanup(limit=50, budget_seconds=15.0) == 1

    assert not (run_dirs["ok"] / "wds-staging").exists()
    for name in ("blocked", "retry", "failed", "other-target", "unknown"):
        assert (run_dirs[name] / "wds-staging").exists(), name


def test_wds_staging_reconcile_respects_limit(tmp_path):
    agent, state = make_remote_agent(tmp_path)
    agent.cfg.cleanup_work_dir = True
    for name in ("a", "b", "c"):
        _make_staged_run(tmp_path, name)
    state.get_batch.return_value = {
        "backend": "remote_agent", "target": "hpc", "remote_status": "SUCCEEDED",
    }

    assert agent._reconcile_wds_staging_cleanup(limit=2, budget_seconds=15.0) == 2
    assert len(list((tmp_path / "remote").glob("*/wds-staging"))) == 1


def _prepare_finalize_and_publish(agent, monkeypatch, *, hooks_ok: bool, transitioned: bool):
    import threading

    agent.cfg.cleanup_work_dir = True
    agent._finalization_lock = threading.Lock()
    agent._finalization_threads = {}
    agent.state.read_remote_status.return_value = {}
    agent.state.get_batch_hooks.return_value = {}
    agent.state.transition_remote_status.return_value = transitioned
    monkeypatch.setattr(
        agent,
        "_finalize_success",
        lambda order: SimpleNamespace(
            wds_hooks_succeeded=hooks_ok, db_sync_hooks_succeeded=hooks_ok
        ),
    )
    monkeypatch.setattr(agent, "_attach_remote_log_tail", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_publish_remote_log_tail", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_release_finalization_lease_safely", lambda *args: None)


def test_finalize_and_publish_removes_wds_staging_after_success(tmp_path, monkeypatch):
    agent, _state = make_remote_agent(tmp_path)
    run_dir = _make_staged_run(tmp_path, "batch-1")
    _prepare_finalize_and_publish(agent, monkeypatch, hooks_ok=True, transitioned=True)

    agent._finalize_and_publish({"batch_id": "batch-1", "remote_run_dir": str(run_dir)}, "123")

    assert not (run_dir / "wds-staging").exists()
    assert (run_dir / "driver.out").exists()


@pytest.mark.parametrize("hooks_ok, transitioned", [(False, True), (True, False)])
def test_finalize_and_publish_keeps_wds_staging_unless_succeeded(
    tmp_path, monkeypatch, hooks_ok, transitioned
):
    agent, _state = make_remote_agent(tmp_path)
    run_dir = _make_staged_run(tmp_path, "batch-1")
    _prepare_finalize_and_publish(
        agent, monkeypatch, hooks_ok=hooks_ok, transitioned=transitioned
    )

    agent._finalize_and_publish({"batch_id": "batch-1", "remote_run_dir": str(run_dir)}, "123")

    assert (run_dir / "wds-staging").exists()
