from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from threading import Event
from unittest.mock import MagicMock, patch

from mussel_dispatcher.config import Config, RemoteTargetConfig
from mussel_dispatcher.remote import RemoteAgent
from mussel_dispatcher.runner import NextflowRunner
from mussel_dispatcher.scheduler import RunManager
from mussel_dispatcher.state import (
    DB_SYNC_LEASE_NAME,
    LOCAL_FINALIZATION_LEASE_NAME,
    REMOTE_FINALIZATION_LEASE_NAME,
    StateStore,
)


def test_global_finalization_lease_is_exclusive_and_reclaimable(tmp_path):
    store_a = StateStore(str(tmp_path / "state.db"))
    store_b = StateStore(str(tmp_path / "state.db"))

    assert store_a.claim_finalization_lease(
        "batch-a", owner="owner-a", ttl_seconds=300
    )
    assert not store_b.claim_finalization_lease(
        "batch-b", owner="owner-b", ttl_seconds=300
    )
    assert store_a.heartbeat_finalization_lease(
        "batch-a", owner="owner-a", ttl_seconds=300
    )
    assert store_a.release_finalization_lease("batch-a", owner="owner-a")
    assert store_b.claim_finalization_lease(
        "batch-b", owner="owner-b", ttl_seconds=300
    )


def test_expired_finalization_lease_can_be_taken_over(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    assert store.claim_finalization_lease("batch-a", owner="owner-a", ttl_seconds=1)

    conn = store._conn()
    expired = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    conn.execute(
        "UPDATE coordination_leases SET expires_at=? WHERE lease_name=?",
        (expired, "dispatcher-finalization"),
    )

    assert store.claim_finalization_lease("batch-b", owner="owner-b", ttl_seconds=300)
    lease = store.finalization_lease_status()
    assert lease["owner"] == "owner-b"
    assert lease["batch_id"] == "batch-b"


def test_unchanged_remote_status_is_not_rewritten(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        "INSERT INTO batches (batch_id, status, backend, target) VALUES (?, ?, ?, ?)",
        ("batch-a", "RUNNING", "remote_agent", "hpc"),
    )
    payload = {
        "status": "RUNNING",
        "batch_id": "batch-a",
        "driver_job_id": "123",
        "slurm": {"state": "RUNNING", "source": "squeue"},
    }
    assert store.write_remote_status("batch-a", payload) is True
    assert store.write_remote_status(
        "batch-a", {**payload, "heartbeat_at": "later", "updated_at": "later"}
    ) is False
    assert store.write_remote_status(
        "batch-a", {**payload, "slurm": {"state": "COMPLETED"}}
    ) is True

    with_tail = {
        **payload,
        "log_tail": {"content_sha256": "abc", "files": [{"name": "driver.out", "text": "one"}]},
    }
    assert store.write_remote_status("batch-a", with_tail) is True
    assert store.write_remote_status("batch-a", with_tail) is False
    with_tail["log_tail"] = {
        "content_sha256": "def",
        "files": [{"name": "driver.out", "text": "two"}],
    }
    assert store.write_remote_status("batch-a", with_tail) is True


def test_agent_snapshot_is_written_as_one_complete_payload(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.write_remote_agent_snapshot(
        "hpc",
        {
            "host": "login01",
            "status": "ALIVE",
            "heartbeat_at": "2026-08-04T00:00:00+00:00",
            "slurm": {"running": 2, "pending": 1},
        },
    )

    agent = store.read_remote_agent("hpc")
    assert agent["host"] == "login01"
    assert agent["slurm"] == {"running": 2, "pending": 1}


def test_remote_target_release_gate_pauses_source_drift(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.publish_deployment_release(
        "default",
        desired_revision="new-release",
        desired_release_manifest_sha256="manifest",
    )
    store.write_deployment_node(
        "default",
        "hpc:login01",
        target="hpc",
        status="READY",
        staged_revision="new-release",
        active_revision="old-release",
    )

    ready, reason = store.remote_target_release_ready("hpc")

    assert not ready
    assert "old-release" in reason

    store.write_deployment_node(
        "default",
        "hpc:login01",
        target="hpc",
        status="ACTIVE",
        staged_revision="new-release",
        active_revision="new-release",
    )
    assert store.remote_target_release_ready("hpc") == (True, "")


def test_remote_target_without_release_record_remains_compatible(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))

    assert store.remote_target_release_ready("hpc") == (True, "")


def test_remote_finalization_claim_uses_remote_lane(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (
            batch_id, status, backend, target, remote_status,
            remote_driver_job_id, remote_payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "batch-a", "RUNNING", "remote_agent", "hpc", "RUNNING", "123",
            '{"batch_id":"batch-a","status":"RUNNING","driver_job_id":"123"}',
        ),
    )
    assert store.claim_finalization_lease("other", owner="other-owner", ttl_seconds=300)
    assert store.claim_remote_finalization(
        "batch-a", stale_after_seconds=300, owner="hpc-owner"
    )
    assert store.read_remote_status("batch-a")["status"] == "HOOKS_RUNNING"
    assert store.finalization_lease_status(REMOTE_FINALIZATION_LEASE_NAME)["owner"] == "hpc-owner"


def test_local_and_remote_finalization_lanes_are_independent(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("ready-remote", "RECOVERING", "remote_agent", "hpc", "HOOKS_RUNNING"),
    )

    assert store.claim_finalization_lease(
        "local-batch", owner="local-owner", ttl_seconds=300,
        lease_name=LOCAL_FINALIZATION_LEASE_NAME,
    )
    assert store.claim_finalization_lease(
        "remote-batch", owner="remote-owner", ttl_seconds=300,
        lease_name=REMOTE_FINALIZATION_LEASE_NAME,
    )
    assert not store.claim_finalization_lease(
        "another-local", owner="other-local", ttl_seconds=300,
        lease_name=LOCAL_FINALIZATION_LEASE_NAME,
    )


def test_finalization_slots_bound_each_backend_and_reclaim_expired(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))

    first = store.claim_finalization_slot(
        "local-a", lane="local", max_slots=2, owner="owner-a", ttl_seconds=300
    )
    second = store.claim_finalization_slot(
        "local-b", lane="local", max_slots=2, owner="owner-b", ttl_seconds=300
    )
    assert first.endswith(":local")
    assert second.endswith(":local:1")
    assert store.claim_finalization_slot(
        "local-c", lane="local", max_slots=2, owner="owner-c", ttl_seconds=300
    ) is None
    assert store.claim_finalization_slot(
        "remote-a", lane="remote", max_slots=2, owner="remote-a", ttl_seconds=300
    ) is not None

    conn = store._conn()
    expired = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    conn.execute(
        "UPDATE coordination_leases SET expires_at=? WHERE batch_id='local-a'",
        (expired,),
    )
    conn.commit()
    assert store.claim_finalization_slot(
        "local-c", lane="local", max_slots=2, owner="owner-c", ttl_seconds=300
    ) == first


def test_claim_and_reconcile_share_the_lane_lock(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    original = store._lock_finalization_lane
    store._lock_finalization_lane = MagicMock(side_effect=original)

    store.claim_finalization_slot(
        "batch-a", lane="remote", max_slots=1, owner="agent-a", ttl_seconds=300
    )
    store.reconcile_finalization_slots(
        lane="remote", max_slots=1, owner="agent-a",
        active_batch_ids={"batch-a"},
    )

    assert [call.args[1] for call in store._lock_finalization_lane.call_args_list] == [
        "remote", "remote"
    ]


def test_reconcile_remote_slot_returns_current_owner_orphan_to_retry(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("orphan-batch", "RUNNING", "remote_agent", "hpc", "HOOKS_RUNNING"),
    )
    lease = store.claim_finalization_slot(
        "orphan-batch", lane="remote", max_slots=1,
        owner="current-agent", ttl_seconds=300,
    )
    assert lease == REMOTE_FINALIZATION_LEASE_NAME

    result = store.reconcile_finalization_slots(
        lane="remote",
        max_slots=1,
        owner="current-agent",
        active_batch_ids=set(),
    )

    assert [row["batch_id"] for row in result["repaired"]] == ["orphan-batch"]
    assert store.read_remote_status("orphan-batch")["status"] == "HOOKS_RETRY"
    assert store.finalization_lease_status(REMOTE_FINALIZATION_LEASE_NAME) is None


def test_reconcile_remote_slot_releases_terminal_batch_lease(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("terminal-batch", "SUCCEEDED", "remote_agent", "hpc", "SUCCEEDED"),
    )
    assert store.claim_finalization_slot(
        "terminal-batch", lane="remote", max_slots=1,
        owner="old-agent", ttl_seconds=300,
    ) == REMOTE_FINALIZATION_LEASE_NAME

    result = store.reconcile_finalization_slots(
        lane="remote", max_slots=1, owner="current-agent", active_batch_ids=set()
    )

    assert result["terminal_released"][0]["batch_id"] == "terminal-batch"
    assert store.finalization_lease_status(REMOTE_FINALIZATION_LEASE_NAME) is None


def test_reconcile_expired_foreign_slot_requeues_and_releases(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("expired-batch", "RUNNING", "remote_agent", "hpc", "HOOKS_RUNNING"),
    )
    lease = store.claim_finalization_slot(
        "expired-batch", lane="remote", max_slots=1,
        owner="old-agent", ttl_seconds=300,
    )
    store._conn().execute(
        "UPDATE coordination_leases SET expires_at=? WHERE lease_name=?",
        ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), lease),
    )

    result = store.reconcile_finalization_slots(
        lane="remote", max_slots=1, owner="new-agent", active_batch_ids=set()
    )

    assert result["health_state"] == "RECOVERING"
    assert [row["batch_id"] for row in result["expired"]] == ["expired-batch"]
    assert result["durable_slots"] == 0
    assert store.read_remote_status("expired-batch")["status"] == "HOOKS_RETRY"
    assert store.finalization_lease_status(lease) is None


def test_reconcile_live_foreign_slots_reports_recovering(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("foreign-batch", "RUNNING", "remote_agent", "hpc", "HOOKS_RUNNING"),
    )
    lease = store.claim_finalization_slot(
        "foreign-batch", lane="remote", max_slots=1,
        owner="old-agent", ttl_seconds=300,
    )

    result = store.reconcile_finalization_slots(
        lane="remote", max_slots=1, owner="new-agent", active_batch_ids=set()
    )

    assert result["health_state"] == "RECOVERING"
    assert [row["batch_id"] for row in result["foreign_live"]] == ["foreign-batch"]
    assert result["durable_slots"] == 1
    assert store.finalization_lease_status(lease)["owner"] == "old-agent"


def test_exact_finalization_ownership_requires_live_matching_lease(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    lease = store.claim_finalization_slot(
        "batch-a", lane="remote", max_slots=1,
        owner="agent-a", ttl_seconds=300,
    )
    assert store.owns_finalization_lease(
        "batch-a", owner="agent-a", lease_name=lease
    )
    assert not store.owns_finalization_lease(
        "batch-a", owner="agent-b", lease_name=lease
    )
    store._conn().execute(
        "UPDATE coordination_leases SET expires_at=? WHERE lease_name=?",
        ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), lease),
    )
    assert not store.owns_finalization_lease(
        "batch-a", owner="agent-a", lease_name=lease
    )


def test_remote_finalization_heartbeat_requires_exact_lease_owner(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("heartbeat-batch", "RUNNING", "remote_agent", "hpc", "RUNNING"),
    )
    assert store.claim_remote_finalization(
        "heartbeat-batch", stale_after_seconds=300, owner="agent-a"
    )
    assert not store.heartbeat_remote_finalization(
        "heartbeat-batch", owner="agent-b", lease_ttl_seconds=300
    )
    assert store.heartbeat_remote_finalization(
        "heartbeat-batch", owner="agent-a", lease_ttl_seconds=300
    )


def test_finalization_lane_limits_use_remote_target_override(tmp_path):
    cfg = Config(
        nextflow_profiles="standard",
        outdir=str(tmp_path / "results"),
        state_dir=str(tmp_path / "state"),
        max_concurrent_finalizations=2,
        dispatch_targets=[
            RemoteTargetConfig(name="hpc", max_concurrent_finalizations=4),
        ],
        default_dispatch_target="hpc",
    )
    assert cfg.effective_max_concurrent_finalizations(lane="local") == 2
    assert cfg.effective_max_concurrent_finalizations(
        lane="remote", target="hpc"
    ) == 4

    store = StateStore(cfg)
    status = store.finalization_coordination_status()
    assert status["max_slots_by_lane"] == {"local": 2, "remote": 4}
    assert RemoteAgent._max_concurrent_finalizations(cfg, cfg.dispatch_targets[0]) == 4


def test_remote_finalization_queue_is_prioritized_over_running_orders(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    rows = [
        ("driver-old", "RUNNING", "2026-08-11T00:00:00+00:00"),
        ("retry-old", "HOOKS_RETRY", "2026-08-11T02:00:00+00:00"),
        ("queued-new", "FINALIZATION_QUEUED", "2026-08-11T03:00:00+00:00"),
        ("hooks-active", "HOOKS_RUNNING", "2026-08-11T01:00:00+00:00"),
    ]
    store._conn().executemany(
        """
        INSERT INTO batches (
            batch_id, status, backend, target, remote_status, dispatched_at
        ) VALUES (?, 'RUNNING', 'remote_agent', 'hpc', ?, ?)
        """,
        [(batch_id, status, dispatched_at) for batch_id, status, dispatched_at in rows],
    )

    assert [
        row["batch_id"] for row in store.get_remote_batches_for_target("hpc")
    ] == ["queued-new", "retry-old", "hooks-active", "driver-old"]


def test_db_sync_lease_remains_shared_across_backends(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))

    assert store.claim_finalization_lease(
        "remote-batch", owner="remote-db", ttl_seconds=300,
        lease_name=DB_SYNC_LEASE_NAME,
    )
    assert not store.claim_finalization_lease(
        "local-batch", owner="local-db", ttl_seconds=300,
        lease_name=DB_SYNC_LEASE_NAME,
    )


def test_finalization_only_batches_do_not_consume_dispatch_capacity(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    rows = [
        ("local-finalizing", "FINALIZING", "local", "", ""),
        ("hpc-queued", "RUNNING", "remote_agent", "hpc", "FINALIZATION_QUEUED"),
        ("hpc-hooks", "RUNNING", "remote_agent", "hpc", "HOOKS_RUNNING"),
        ("hpc-retry", "RUNNING", "remote_agent", "hpc", "HOOKS_RETRY"),
        # A HOOKS_BLOCKED (quarantined) batch must not hold a run slot.
        ("hpc-blocked", "RECOVERING", "remote_agent", "hpc", "HOOKS_BLOCKED"),
        ("hpc-running", "RUNNING", "remote_agent", "hpc", "RUNNING"),
    ]
    store._conn().executemany(
        """
        INSERT INTO batches (batch_id, status, backend, target, remote_status)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    assert store.count_scheduling_active_batches() == 1
    assert [row["batch_id"] for row in store.get_scheduling_active_batches()] == [
        "hpc-running"
    ]


def test_recovered_finalization_does_not_consume_nextflow_slot(tmp_path):
    cfg = Config(nextflow_profiles="standard", outdir=str(tmp_path), max_concurrent_runs=1)
    state = MagicMock()
    manager = RunManager(cfg, state)
    finalizer_started = Event()
    release_finalizer = Event()

    def fake_finalizer(*_args, **_kwargs):
        finalizer_started.set()
        release_finalizer.wait(timeout=2)

    runner = MagicMock()
    runner.run.return_value = 0
    try:
        with patch.object(manager, "_run_finalize_recovered_success", side_effect=fake_finalizer), \
             patch("mussel_dispatcher.scheduler.NextflowRunner", return_value=runner):
            assert manager.submit_finalize_recovered_success(
                "finalizing", "/tmp/batch.csv", "/tmp/batch.log", "/tmp/work"
            )
            assert finalizer_started.wait(timeout=2)
            assert manager.has_capacity()
            assert manager.submit("new-batch", [], known_active_batches=0)
            release_finalizer.set()
    finally:
        release_finalizer.set()
        manager.shutdown(wait=True)


def test_live_success_finalization_does_not_hold_nextflow_slot(tmp_path):
    """A completed driver must free capacity while hooks/WDS finalization runs."""
    cfg = Config(nextflow_profiles="standard", outdir=str(tmp_path), max_concurrent_runs=1)
    state = MagicMock()
    manager = RunManager(cfg, state)
    finalizer_started = Event()
    release_finalizer = Event()
    normal_runner = MagicMock()
    new_runner = MagicMock()
    finalizer_runner = MagicMock()
    new_runner.run.return_value = 1

    def fake_finalizer(*_args, **_kwargs):
        finalizer_started.set()
        release_finalizer.wait(timeout=2)

    finalizer_runner.finalize_recovered_success.side_effect = fake_finalizer

    live_runner_count = 0

    def make_runner(*_args, **kwargs):
        nonlocal live_runner_count
        if kwargs.get("finalization_submitter") is not None:
            live_runner_count += 1
            if live_runner_count > 1:
                return new_runner
            submitter = kwargs["finalization_submitter"]
            normal_runner.run.side_effect = lambda: (
                submitter(
                    "live",
                    "/tmp/live.csv",
                    "/tmp/live.log",
                    "/tmp/live-work",
                    1.0,
                ),
                0,
            )[1]
            return normal_runner
        return finalizer_runner

    try:
        with patch("mussel_dispatcher.scheduler.NextflowRunner", side_effect=make_runner):
            assert manager.submit("live", [], known_active_batches=0)
            assert finalizer_started.wait(timeout=2)

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not manager.has_capacity():
                time.sleep(0.01)
            assert manager.has_capacity(), "driver slot should be free while finalizer is blocked"
            assert manager.submit("new-local", [], known_active_batches=0)
            state.set_batch_status.assert_called_once_with("live", "FINALIZING")
        release_finalizer.set()
    finally:
        release_finalizer.set()
        manager.shutdown(wait=True)


def test_remote_waiter_releases_slot_when_agent_starts_finalization(tmp_path):
    cfg = Config(
        nextflow_profiles="standard",
        outdir=str(tmp_path),
        execution_backend="remote_agent",
        default_dispatch_target="hpc",
        dispatch_targets=[RemoteTargetConfig(name="hpc")],
        cleanup_work_dir=False,
    )
    client = MagicMock()
    client.read_status.return_value = {"status": "HOOKS_RUNNING"}
    runner = NextflowRunner(
        cfg,
        "remote-batch",
        [],
        MagicMock(),
        resume=True,
        existing_csv_path=str(tmp_path / "batch.csv"),
        existing_work_dir=str(tmp_path / "work"),
    )

    with patch("mussel_dispatcher.runner.RemoteDispatchClient", return_value=client):
        assert runner._run_remote_agent() == 0

    client.read_status.assert_called_once_with("remote-batch")


def test_completed_remote_driver_is_queued_while_another_finalizer_runs(tmp_path, monkeypatch):
    cfg = Config(
        nextflow_profiles="standard",
        outdir=str(tmp_path),
        state_backend="postgres",
    )
    target = RemoteTargetConfig(name="hpc", finalization_lease_ttl_seconds=300)
    state = MagicMock()
    state.get_remote_batches_for_target.return_value = [
        {
            "batch_id": "completed-remote",
            "remote_run_dir": str(tmp_path / "run"),
            "remote_work_dir": str(tmp_path / "work"),
            "remote_outdir": str(tmp_path / "results"),
            "remote_log_path": str(tmp_path / "driver.out"),
        }
    ]
    state.claim_remote_finalization.return_value = False
    client = MagicMock()
    client.read_status.return_value = {
        "status": "RUNNING",
        "driver_job_id": "5999999",
    }
    monkeypatch.setattr(
        "mussel_dispatcher.remote.RemoteDispatchClient",
        lambda *args, **kwargs: client,
    )

    agent = RemoteAgent.__new__(RemoteAgent)
    agent.cfg = cfg
    agent.target = target
    agent.state = state
    agent._finalization_owner = "test-agent"
    agent._finalization_threads = {"other-batch": MagicMock(is_alive=lambda: True)}
    agent._driver_slurm_queries = 0
    agent._status_writes_suppressed = 0
    agent._driver_slurm_state = lambda _job_id: {
        "state": "COMPLETED",
        "exit_code": 0,
        "source": "sacct",
    }

    assert agent._refresh_claimed_orders() is True

    payload = client.write_status.call_args.args[1]
    assert payload["status"] == "FINALIZATION_QUEUED"
    assert payload["nextflow_exit"] == 0
    state.claim_remote_finalization.assert_not_called()


def test_blocked_dispatch_is_logged_once_per_change(caplog):
    from unittest.mock import MagicMock
    from mussel_dispatcher.scheduler import RunManager

    def pool(name, active, limit):
        blockers = [] if active < limit else [
            {"code": "CONCURRENCY_LIMIT", "message": f"{active} active batch(es) at limit {limit}"}]
        return {"name": name, "available": not blockers, "blockers": blockers,
                "active_batches": active, "max_batches": limit, "_order": 0}

    rm = RunManager.__new__(RunManager)
    rm.state = MagicMock()
    rm.cfg = MagicMock()
    states = iter([[pool("hpc", 2, 2)], [pool("hpc", 2, 2)], [pool("hpc", 1, 2)]])
    rm._evaluate_execution_pools = lambda: next(states)
    caplog.set_level("INFO", logger="mussel-dispatcher")

    assert rm.choose_execution_pool(queued=320) is None
    assert rm.choose_execution_pool(queued=640) is None  # only the count changed: no second line
    rm.choose_execution_pool(queued=640)                 # a slot frees up
    lines = [r.getMessage() for r in caplog.records if "Fresh dispatch" in r.getMessage()]
    assert lines == ["Fresh dispatch blocked (320 slide(s) queued): hpc: CONCURRENCY_LIMIT "
                     "(2 active batch(es) at limit 2)",
                     "Fresh dispatch unblocked: hpc 1/2"]


def test_global_dispatch_limit_is_logged_once_per_change(caplog):
    from unittest.mock import MagicMock
    from mussel_dispatcher.scheduler import RunManager

    rm = RunManager.__new__(RunManager)
    rm.cfg = MagicMock()
    rm.cfg.effective_max_concurrent_runs.return_value = 5
    rm.has_capacity = lambda: True
    caplog.set_level("INFO", logger="mussel-dispatcher")

    assert rm.has_fresh_dispatch_capacity(5) is False
    assert rm.has_fresh_dispatch_capacity(6) is False   # still blocked: no second line
    assert rm.has_fresh_dispatch_capacity(4) is True
    assert rm.has_fresh_dispatch_capacity(3) is True    # still open: no line
    lines = [r.getMessage() for r in caplog.records if "Fresh dispatch" in r.getMessage()]
    assert lines == ["Fresh dispatch blocked: 5 active batch(es) in DB >= effective max 5",
                     "Fresh dispatch no longer at the global limit: 4 active batch(es) < 5"]
