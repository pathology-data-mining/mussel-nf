"""Regression tests for dispatcher state-store and slide-claiming fixes."""
from __future__ import annotations

import csv
import dataclasses
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest

import mussel_dispatcher.db as db_mod
import mussel_dispatcher.state as state_mod
from mussel_dispatcher import release
from mussel_dispatcher.config import Config, RemoteTargetConfig
from mussel_dispatcher.remote import RemoteAgent
from mussel_dispatcher.runner import NextflowRunner
from mussel_dispatcher.state import REMOTE_FINALIZATION_LEASE_NAME, StateStore


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


# ---------------------------------------------------------------------------
# 1. Remote finalization retry/blocked transitions are fenced by ownership
# ---------------------------------------------------------------------------

def _hand_finalization_to(store: StateStore, batch_id: str, owner: str) -> None:
    """Simulate another agent reclaiming an expired finalization lease."""
    conn = store._conn()
    conn.execute(
        "UPDATE coordination_leases SET owner=? WHERE batch_id=?", (owner, batch_id)
    )
    conn.execute(
        "UPDATE batches SET finalization_owner=? WHERE batch_id=?", (owner, batch_id)
    )


def _remote_batch_in_finalization(tmp_path) -> StateStore:
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().execute(
        "INSERT INTO batches (batch_id, status, backend, target, remote_status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fin-batch", "RUNNING", "remote_agent", "hpc", "RUNNING"),
    )
    assert store.claim_remote_finalization(
        "fin-batch", stale_after_seconds=300, owner="agent-a"
    )
    return store


def _remote_status(store: StateStore, batch_id: str) -> str:
    row = store._conn().execute(
        "SELECT remote_status FROM batches WHERE batch_id=?", (batch_id,)
    ).fetchone()
    return row["remote_status"]


def test_stale_owner_cannot_mark_finalization_retry(tmp_path):
    store = _remote_batch_in_finalization(tmp_path)
    _hand_finalization_to(store, "fin-batch", "agent-b")

    assert not store.mark_remote_finalization_retry(
        "fin-batch", error="old worker", owner="agent-a",
        lease_name=REMOTE_FINALIZATION_LEASE_NAME,
    )
    assert _remote_status(store, "fin-batch") == "HOOKS_RUNNING"

    assert store.mark_remote_finalization_retry(
        "fin-batch", error="current worker", owner="agent-b",
        lease_name=REMOTE_FINALIZATION_LEASE_NAME,
    )
    assert _remote_status(store, "fin-batch") == "HOOKS_RETRY"


def test_stale_owner_cannot_mark_finalization_blocked(tmp_path):
    store = _remote_batch_in_finalization(tmp_path)
    _hand_finalization_to(store, "fin-batch", "agent-b")

    assert not store.mark_remote_finalization_blocked(
        "fin-batch", error="old worker", owner="agent-a",
        lease_name=REMOTE_FINALIZATION_LEASE_NAME,
    )
    assert _remote_status(store, "fin-batch") == "HOOKS_RUNNING"

    assert store.mark_remote_finalization_blocked(
        "fin-batch", error="needs operator", owner="agent-b",
        lease_name=REMOTE_FINALIZATION_LEASE_NAME,
    )
    assert _remote_status(store, "fin-batch") == "HOOKS_BLOCKED"


def test_finalization_retry_requires_live_lease_row(tmp_path):
    store = _remote_batch_in_finalization(tmp_path)
    store._conn().execute("DELETE FROM coordination_leases WHERE batch_id='fin-batch'")

    assert not store.mark_remote_finalization_retry(
        "fin-batch", owner="agent-a", lease_name=REMOTE_FINALIZATION_LEASE_NAME
    )
    assert _remote_status(store, "fin-batch") == "HOOKS_RUNNING"


def test_finalization_fence_locks_lease_before_batch_row(tmp_path):
    store = _remote_batch_in_finalization(tmp_path)
    seen: list[str] = []
    real_run_write = store._run_write

    class _Recorder:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            seen.append(sql)
            return self._conn.execute(sql, params)

    store._run_write = lambda op, fn: real_run_write(op, lambda c: fn(_Recorder(c)))
    assert store.mark_remote_finalization_retry(
        "fin-batch", owner="agent-a", lease_name=REMOTE_FINALIZATION_LEASE_NAME
    )
    lease_at = next(i for i, q in enumerate(seen) if "coordination_leases" in q)
    batch_at = next(i for i, q in enumerate(seen) if "FROM batches" in q)
    assert lease_at < batch_at


def _pool_full_agent(tmp_path):
    agent = object.__new__(RemoteAgent)
    agent.cfg = make_config(tmp_path, state_backend="postgres")
    agent.target = RemoteTargetConfig(name="hpc", remote_scratch_root=str(tmp_path / "r"))
    agent.state = MagicMock()
    agent.once = False
    agent._finalization_owner = "agent-a"
    agent._finalization_lock = threading.RLock()
    agent._finalization_threads = {}
    agent._finalization_lease_names = {}
    agent._finalization_pool = MagicMock()
    agent._finalization_pool.running_count.return_value = 0
    agent._finalization_pool.submit.return_value = None
    agent._finalization_limit = MagicMock(return_value=4)
    agent._release_finalization_lease_safely = MagicMock(return_value=True)
    return agent


def test_pool_full_retry_passes_owner(tmp_path):
    agent = _pool_full_agent(tmp_path)

    agent._launch_finalization({"batch_id": "fin-batch"}, "123", lease_name="lease-x")

    kwargs = agent.state.mark_remote_finalization_retry.call_args.kwargs
    assert (kwargs["owner"], kwargs["lease_name"]) == ("agent-a", "lease-x")


def test_remote_agent_passes_owner_to_finalization_retry(tmp_path):
    agent = object.__new__(RemoteAgent)
    agent.cfg = make_config(tmp_path, state_backend="postgres")
    agent.target = RemoteTargetConfig(name="hpc", remote_scratch_root=str(tmp_path / "r"))
    agent.state = MagicMock()
    agent.state.read_remote_status.return_value = {}
    agent._finalization_owner = "agent-a"
    agent._finalization_lock = threading.RLock()
    agent._finalization_threads = {}
    agent._finalization_lease_names = {"fin-batch": "lease-x"}
    agent._finalize_success = MagicMock(side_effect=RuntimeError("boom"))
    agent._publish_remote_log_tail = MagicMock()
    agent._release_finalization_lease_safely = MagicMock(return_value=True)

    with patch("mussel_dispatcher.remote.RemoteDispatchClient"):
        agent._finalize_and_publish({"batch_id": "fin-batch"}, "123")

    kwargs = agent.state.mark_remote_finalization_retry.call_args.kwargs
    assert kwargs["owner"] == "agent-a"
    assert kwargs["lease_name"] == "lease-x"


# ---------------------------------------------------------------------------
# 2. Schema migration / startup guard
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, row=None, rows=None, rowcount=0):
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _FakePgConn:
    """Minimal psycopg stand-in that records SQL and can inject failures."""

    def __init__(self, *, columns=None, fail_on=None, fail_exc=None, existing_indexes=()):
        self.existing_indexes = set(existing_indexes)
        self.queries: list[str] = []
        self.columns = columns  # table -> set(columns) for information_schema
        self.fail_on = fail_on
        self.fail_exc = fail_exc
        self.commits = 0
        self.closed = False

    def execute(self, sql, params=()):
        self.queries.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise self.fail_exc
        if "information_schema.columns" in sql:
            if self.columns is None:
                return _FakeCursor(None)
            table = params[0]
            if "column_name=%s" in sql:
                present = params[1] in self.columns.get(table, set())
                return _FakeCursor((1,) if present else None)
            return _FakeCursor(
                rows=[{"column_name": c} for c in sorted(self.columns.get(table, set()))]
            )
        if "FROM pg_indexes" in sql:
            return _FakeCursor((1,) if params[0] in self.existing_indexes else None)
        if "to_regclass" in sql:
            return _FakeCursor({"to_regclass": params[0]})
        return _FakeCursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _patch_postgres(monkeypatch, conn):
    monkeypatch.setattr(state_mod, "connect_database", lambda *a, **k: conn)
    monkeypatch.setattr(
        state_mod,
        "resolve_database_config",
        lambda cfg: db_mod.ResolvedDatabaseConfig(
            backend="postgres", host="db.example", database="dispatcher", user="app",
        ),
    )


def test_skip_init_guard_checks_every_migrated_column(tmp_path, monkeypatch):
    columns = {t: set(c) for t, c in state_mod._required_schema_columns().items()}
    columns["slides"].discard("priority")
    columns["batches"].discard("driver_completed_at")
    conn = _FakePgConn(columns=columns)
    _patch_postgres(monkeypatch, conn)
    monkeypatch.setenv("MUSSEL_POSTGRES_SKIP_SCHEMA_INIT", "1")

    with pytest.raises(RuntimeError) as excinfo:
        StateStore(make_config(tmp_path, state_backend="postgres"))
    assert "slides.priority" in str(excinfo.value)
    assert "batches.driver_completed_at" in str(excinfo.value)


def test_skip_init_guard_accepts_complete_schema(tmp_path, monkeypatch):
    columns = {t: set(c) for t, c in state_mod._required_schema_columns().items()}
    conn = _FakePgConn(columns=columns)
    _patch_postgres(monkeypatch, conn)
    monkeypatch.setenv("MUSSEL_POSTGRES_SKIP_SCHEMA_INIT", "1")

    StateStore(make_config(tmp_path, state_backend="postgres"))


def test_required_columns_cover_alter_only_columns():
    required = state_mod._required_schema_columns()
    for table, column in (
        ("slides", "priority"),
        ("batches", "nf_pid"),
        ("batches", "driver_completed_at"),
        ("batches", "execution_cleanup_at"),
        ("batches", "execution_cleanup_error"),
        ("batches", "remote_status"),
        ("batches", "remote_payload"),
        ("remote_agents", "payload"),
    ):
        assert column in required[table], f"{table}.{column}"


def test_sqlite_schema_satisfies_required_columns(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    assert store._missing_schema_columns(store._conn()) == []
    store._conn().execute("ALTER TABLE batches DROP COLUMN execution_cleanup_error")
    assert store._missing_schema_columns(store._conn()) == [
        "batches.execution_cleanup_error"
    ]


def test_migrate_fails_when_ddl_lacks_privilege(tmp_path, monkeypatch):
    conn = _FakePgConn(
        fail_on="CREATE INDEX IF NOT EXISTS idx_slides_status ",
        fail_exc=RuntimeError("must be owner of relation slides"),
    )
    _patch_postgres(monkeypatch, conn)

    with pytest.raises(RuntimeError, match="must be owner"):
        release.migrate(make_config(tmp_path, state_backend="postgres"))
    assert conn.commits == 0


def test_service_startup_still_tolerates_ddl_privilege_error(tmp_path, monkeypatch):
    conn = _FakePgConn(
        fail_on="CREATE INDEX IF NOT EXISTS idx_slides_status ",
        fail_exc=RuntimeError("permission denied for table slides"),
    )
    _patch_postgres(monkeypatch, conn)

    StateStore(make_config(tmp_path, state_backend="postgres"))
    assert conn.commits == 1


def test_existing_postgres_index_is_not_recreated(tmp_path, monkeypatch):
    conn = _FakePgConn(existing_indexes={"idx_slides_status"})
    _patch_postgres(monkeypatch, conn)

    StateStore(make_config(tmp_path, state_backend="postgres"))
    assert not any(
        "CREATE INDEX IF NOT EXISTS idx_slides_status " in sql for sql in conn.queries
    )
    assert any("idx_slides_batch_id" in sql and "CREATE INDEX" in sql for sql in conn.queries)


def test_has_column_is_scoped_to_current_schema(tmp_path, monkeypatch):
    conn = _FakePgConn()
    _patch_postgres(monkeypatch, conn)

    StateStore(make_config(tmp_path, state_backend="postgres"))
    probes = [
        sql for sql in conn.queries
        if "information_schema.columns" in sql and "column_name=%s" in sql
    ]
    assert probes
    assert all("current_schema()" in sql for sql in probes)


def test_migrate_fails_when_ddl_times_out(tmp_path, monkeypatch):
    conn = _FakePgConn(
        fail_on="ADD COLUMN driver_completed_at",
        fail_exc=RuntimeError("canceling statement due to lock timeout"),
    )
    _patch_postgres(monkeypatch, conn)
    monkeypatch.setenv("MUSSEL_POSTGRES_SKIP_SCHEMA_INIT", "1")

    with pytest.raises(RuntimeError, match="lock timeout"):
        release.migrate(make_config(tmp_path, state_backend="postgres"))
    assert conn.commits == 0


def test_service_startup_still_tolerates_ddl_timeout(tmp_path, monkeypatch):
    conn = _FakePgConn(
        fail_on="ADD COLUMN driver_completed_at",
        fail_exc=RuntimeError("canceling statement due to lock timeout"),
    )
    _patch_postgres(monkeypatch, conn)

    StateStore(make_config(tmp_path, state_backend="postgres"))
    assert "ROLLBACK TO SAVEPOINT schema_ddl" in conn.queries
    assert conn.commits == 1


def test_rollup_backfill_failure_does_not_abort_schema_transaction(tmp_path, monkeypatch):
    conn = _FakePgConn(
        fail_on="INSERT INTO slide_rollups",
        fail_exc=RuntimeError("canceling statement due to statement timeout"),
    )
    _patch_postgres(monkeypatch, conn)

    store = StateStore(make_config(tmp_path, state_backend="postgres"))

    failing = next(
        i for i, sql in enumerate(conn.queries) if "INSERT INTO slide_rollups" in sql
    )
    assert "SAVEPOINT slide_rollups_backfill" in conn.queries[:failing]
    assert "ROLLBACK TO SAVEPOINT slide_rollups_backfill" in conn.queries[failing:]
    assert conn.commits == 1
    assert store._slide_rollups_ready is False


# ---------------------------------------------------------------------------
# 3. Slide claiming restricts the batch to what it owns
# ---------------------------------------------------------------------------

def test_claim_slides_returns_claimed_paths(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.add_slide("/slides/b.svs", "b")
    assert store.claim_slides(["/slides/b.svs"], "other") == ["/slides/b.svs"]

    assert store.claim_slides(["/slides/a.svs", "/slides/b.svs"], "mine") == [
        "/slides/a.svs"
    ]
    assert store.mark_dispatched(["/slides/a.svs"], "again") == 0


def _runner(tmp_path, store, slides, batch_id="batch-x"):
    cfg = make_config(tmp_path)
    runner = NextflowRunner(cfg, batch_id, slides, store)
    csv_path = runner._write_csv()
    return runner, csv_path


def _csv_paths(csv_path) -> list[str]:
    with open(csv_path, newline="") as fh:
        return [row["slide_path"] for row in csv.DictReader(fh)]


def _slide(store, path):
    return store._conn().execute(
        "SELECT status, batch_id, error_msg FROM slides WHERE slide_path=?", (path,)
    ).fetchone()


def test_runner_runs_only_slides_it_claimed(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.add_slide("/slides/b.svs", "b")
    store.claim_slides(["/slides/b.svs"], "racing-batch")
    slides = [
        {"slide_path": "/slides/a.svs", "slide_id": "a"},
        {"slide_path": "/slides/b.svs", "slide_id": "b"},
    ]
    runner, csv_path = _runner(tmp_path, store, slides)

    assert runner._register_new_batch(
        csv_path, str(tmp_path / "w"), str(tmp_path / "l.log"), nf_run_name="rx"
    )

    assert [s["slide_path"] for s in runner.slides] == ["/slides/a.svs"]
    assert _csv_paths(csv_path) == ["/slides/a.svs"]
    batch = store._conn().execute(
        "SELECT slide_count FROM batches WHERE batch_id='batch-x'"
    ).fetchone()
    assert batch["slide_count"] == 1
    assert _slide(store, "/slides/b.svs")["batch_id"] == "racing-batch"


def test_runner_abandons_batch_when_nothing_claimed(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.claim_slides(["/slides/a.svs"], "racing-batch")
    runner, csv_path = _runner(
        tmp_path, store, [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
    )

    assert not runner._register_new_batch(
        csv_path, str(tmp_path / "w"), str(tmp_path / "l.log"), nf_run_name="rx"
    )
    assert store._conn().execute(
        "SELECT COUNT(*) AS n FROM batches WHERE batch_id='batch-x'"
    ).fetchone()["n"] == 0
    assert not Path(csv_path).exists()


def test_slide_dropped_from_csv_is_not_marked_succeeded(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a/X.svs", "X")
    store.add_slide("/slides/b/X.svs", "X")
    slides = [
        {"slide_path": "/slides/a/X.svs", "slide_id": "X"},
        {"slide_path": "/slides/b/X.svs", "slide_id": "X"},
    ]
    runner, csv_path = _runner(tmp_path, store, slides)

    assert runner._register_new_batch(
        csv_path, str(tmp_path / "w"), str(tmp_path / "l.log"), nf_run_name="rx"
    )
    assert _csv_paths(csv_path) == ["/slides/a/X.svs"]
    store.mark_slides_complete("batch-x", succeeded=True)

    assert _slide(store, "/slides/a/X.svs")["status"] == "SUCCEEDED"
    dropped = _slide(store, "/slides/b/X.svs")
    assert dropped["status"] == "SKIPPED"
    assert "duplicate slide_id X" in dropped["error_msg"]
    assert "/slides/a/X.svs" in dropped["error_msg"]
    fail_count = store._conn().execute(
        "SELECT fail_count FROM slides WHERE slide_path='/slides/b/X.svs'"
    ).fetchone()["fail_count"]
    assert fail_count == 0
    rollup = store._conn().execute(
        "SELECT has_succeeded, has_failed, has_pending FROM slide_rollups WHERE slide_id='X'"
    ).fetchone()
    assert tuple(rollup) == (1, 0, 0)


def test_duplicate_dropped_from_csv_is_never_rebatched_by_retry_sweep(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a/X.svs", "X")
    store.add_slide("/slides/b/X.svs", "X")
    runner, csv_path = _runner(tmp_path, store, [
        {"slide_path": "/slides/a/X.svs", "slide_id": "X"},
        {"slide_path": "/slides/b/X.svs", "slide_id": "X"},
    ])
    assert runner._register_new_batch(
        csv_path, str(tmp_path / "w"), str(tmp_path / "l.log"), nf_run_name="rx"
    )
    store.mark_slides_complete("batch-x", succeeded=True)

    store.reset_failed_to_pending(max_retries=5)
    store.reset_failed_to_pending(max_retries=0)
    store.add_slide("/slides/b/X.svs", "X")  # watcher rediscovers the file

    assert store.get_pending_slides() == []
    assert _slide(store, "/slides/b/X.svs")["status"] == "SKIPPED"


def test_invalid_row_dropped_from_csv_is_failed_and_charged(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.add_slide("gdc-broken", "b")
    runner, csv_path = _runner(tmp_path, store, [
        {"slide_path": "/slides/a.svs", "slide_id": "a"},
        {"slide_path": "gdc-broken", "slide_id": "b", "needs_download": 1},
    ])

    assert runner._register_new_batch(
        csv_path, str(tmp_path / "w"), str(tmp_path / "l.log"), nf_run_name="rx"
    )
    row = store._conn().execute(
        "SELECT status, fail_count, error_msg FROM slides WHERE slide_path='gdc-broken'"
    ).fetchone()
    assert (row["status"], row["fail_count"]) == ("FAILED", 1)
    assert "no file_id" in row["error_msg"]


# ---------------------------------------------------------------------------
# 4. reset_failed_to_pending(max_retries>0) refreshes rollups
# ---------------------------------------------------------------------------

def test_reset_failed_with_retry_limit_refreshes_rollups(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.mark_dispatched(["/slides/a.svs"], "batch-1")
    store.mark_slides_complete("batch-1", succeeded=False)

    assert store.reset_failed_to_pending(max_retries=3) == 1

    rollup = store._conn().execute(
        "SELECT has_failed, has_pending FROM slide_rollups WHERE slide_id='a'"
    ).fetchone()
    assert (rollup["has_failed"], rollup["has_pending"]) == (0, 1)


# ---------------------------------------------------------------------------
# 5. Scheduling-active count and listing agree
# ---------------------------------------------------------------------------

def test_scheduling_active_count_matches_listing_for_blank_remote_status(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store._conn().executemany(
        "INSERT INTO batches (batch_id, status, backend, target, remote_status) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("remote-null", "RUNNING", "remote_agent", "hpc", None),
            ("remote-blank", "RUNNING", "remote_agent", "hpc", ""),
            ("remote-done", "RUNNING", "remote_agent", "hpc", "SUCCEEDED"),
            ("local", "RUNNING", "local", None, None),
        ],
    )

    listed = {row["batch_id"] for row in store.get_scheduling_active_batches()}
    assert store.count_scheduling_active_batches() == len(listed)
    assert listed == {"remote-null", "remote-blank", "local"}


# ---------------------------------------------------------------------------
# 6. _run_write retries a write whose session died before COMMIT
# ---------------------------------------------------------------------------

class _DyingConn:
    def __init__(self, *, fail_in_fn=False, fail_on_commit=False):
        self.closed = False
        self.fail_in_fn = fail_in_fn
        self.fail_on_commit = fail_on_commit

    def execute(self, sql, params=()):
        if sql == "BEGIN":
            return _FakeCursor()
        if self.fail_in_fn:
            self.closed = True
            raise psycopg.OperationalError(
                "consuming input failed: server closed the connection unexpectedly"
            )
        return _FakeCursor(rowcount=1)

    def commit(self):
        if self.fail_on_commit:
            self.closed = True
            raise psycopg.OperationalError("server closed the connection unexpectedly")

    def rollback(self):
        if self.closed:
            raise psycopg.OperationalError("the connection is closed")

    def close(self):
        self.closed = True


def _postgres_store_with_conns(tmp_path, monkeypatch, conns):
    store = StateStore(str(tmp_path / "state.db"))
    store.db_cfg = dataclasses.replace(store.db_cfg, backend="postgres")
    queue = list(conns)
    monkeypatch.setattr(store, "_conn", lambda: queue[0])
    monkeypatch.setattr(store, "close_thread_connection", lambda: queue.pop(0))
    monkeypatch.setattr(state_mod.time, "sleep", lambda _s: None)
    return store


def test_run_write_retries_when_connection_lost_before_commit(tmp_path, monkeypatch):
    store = _postgres_store_with_conns(
        tmp_path, monkeypatch, [_DyingConn(fail_in_fn=True), _DyingConn()]
    )
    calls = []

    def _write(conn):
        calls.append(conn)
        conn.execute("UPDATE slides SET status='PENDING'")
        return "ok"

    assert store._run_write("test", _write) == "ok"
    assert len(calls) == 2


def test_run_write_does_not_replay_when_connection_lost_during_commit(tmp_path, monkeypatch):
    store = _postgres_store_with_conns(
        tmp_path, monkeypatch, [_DyingConn(fail_on_commit=True), _DyingConn()]
    )
    calls = []

    with pytest.raises(psycopg.OperationalError):
        store._run_write("test", lambda conn: calls.append(conn))
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 7. mark_slides_complete is idempotent
# ---------------------------------------------------------------------------

def test_replayed_failure_charges_fail_count_once(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.mark_dispatched(["/slides/a.svs"], "batch-1")

    store.mark_slides_complete("batch-1", succeeded=False)
    store.mark_slides_complete("batch-1", succeeded=False)

    row = store._conn().execute(
        "SELECT status, fail_count FROM slides WHERE slide_path='/slides/a.svs'"
    ).fetchone()
    assert (row["status"], row["fail_count"]) == ("FAILED", 1)


def test_replayed_completion_does_not_touch_slides_requeued_elsewhere(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_slide("/slides/a.svs", "a")
    store.mark_dispatched(["/slides/a.svs"], "batch-1")
    store.mark_slides_complete("batch-1", succeeded=False)
    store.reset_failed_to_pending(max_retries=3)
    store.mark_dispatched(["/slides/a.svs"], "batch-2")

    store.mark_slides_complete("batch-1", succeeded=False)

    row = store._conn().execute(
        "SELECT status, batch_id, fail_count FROM slides WHERE slide_path='/slides/a.svs'"
    ).fetchone()
    assert (row["status"], row["batch_id"], row["fail_count"]) == ("DISPATCHED", "batch-2", 1)


# ---------------------------------------------------------------------------
# 8. SKIPPED duplicates never stand in for their slide_id downstream
# ---------------------------------------------------------------------------

def test_dashboard_db_total_excludes_skipped_rows(tmp_path, monkeypatch):
    import json
    import tests.test_dispatcher as td

    routes = td.TestTowerShimServerRoutes()
    Handler = routes._make_server(tmp_path)
    (tmp_path / "state" / "dispatcher_status.json").write_text(json.dumps({
        "counts": {"SUCCEEDED": 7, "PENDING": 3, "SKIPPED": 2},
        "wds": {"models": {}, "total": 0, "inventory_total": 0, "error": None},
        "recent_batches": [],
    }))

    h, responses = routes._fake_request(Handler, "GET", "/api/wds")
    assert h._response_code == 200
    assert json.loads(responses[0])["db_total"] == 10
