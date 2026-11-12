"""StateStore under Postgres-like failures: commit-unknown and dropped sessions.

FaultyConnection sits where ``connect_database`` hands StateStore its session and
fails the way psycopg does when the server goes away: a COMMIT that the server
applied but whose reply is lost, or a session that dies before a statement. The
default backend is SQLite behind the Postgres dialect (StateStore takes its
Postgres code paths); MUSSEL_TEST_POSTGRES=1 also runs every test against a
throwaway real PostgreSQL cluster (initdb/pg_ctl on PATH).
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from mussel_dispatcher import runner as runner_mod
from mussel_dispatcher import scheduler as sched
from mussel_dispatcher import state as state_mod
from mussel_dispatcher.config import Config, PostgresConfig, RemoteTargetConfig
from mussel_dispatcher.remote import RemoteDispatchClient
from mussel_dispatcher.runner import NextflowRunner
from mussel_dispatcher.state import StateStore

_REAL_CONNECT = state_mod.connect_database
_LOST = "consuming input failed: server closed the connection unexpectedly"
_FOR_UPDATE = re.compile(r"\s+FOR UPDATE(\s+SKIP LOCKED)?", re.IGNORECASE)


class Faults:
    """Armed failures; each fires once on the first statement/commit it matches."""

    def __init__(self):
        self._armed: list[tuple[str, object]] = []
        self.fired: list[str] = []
        self.connects = 0

    def commit_unknown(self, pattern: str) -> None:
        self._armed.append(("commit", pattern))

    def drop_before(self, pattern: str) -> None:
        self._armed.append(("drop", pattern))

    def take(self, kind: str, statements: list[str]) -> bool:
        for i, (k, pattern) in enumerate(self._armed):
            if k == kind and any(pattern in s for s in statements):
                del self._armed[i]
                self.fired.append(f"{kind}:{pattern}")
                return True
        return False


class FaultyConnection:
    """A DB session that dies like a psycopg session whose server went away."""

    def __init__(self, raw, faults: Faults, *, sqlite_dialect: bool):
        self._raw, self._faults, self._sqlite = raw, faults, sqlite_dialect
        self._tx: list[str] = []
        self.closed = False
        self.broken = False

    def _die(self):
        self.closed = self.broken = True
        try:
            self._raw.close()
        except Exception:
            pass

    def execute(self, sql, params=()):
        if self.closed:
            raise psycopg.OperationalError("the connection is closed")
        text = f"{sql} {params!r}"
        if sql.strip().upper() == "BEGIN":
            self._tx = []
        elif self._faults.take("drop", [text]):
            self._die()
            raise psycopg.OperationalError(_LOST)
        self._tx.append(text)
        if self._sqlite:
            sql = _FOR_UPDATE.sub("", sql.replace("%s", "?"))
        return self._raw.execute(sql, params)

    def commit(self):
        if self.closed:
            raise psycopg.OperationalError("the connection is closed")
        unknown = self._faults.take("commit", self._tx)
        self._raw.commit()          # the server applied it ...
        if unknown:
            self._die()             # ... and the reply never arrived
            raise psycopg.OperationalError(_LOST)

    def rollback(self):
        if self.closed:
            raise psycopg.OperationalError("the connection is closed")
        self._raw.rollback()

    def close(self):
        self.closed = True
        self._raw.close()


# ── backends ──────────────────────────────────────────────────────────────────

_PG = {}


def _real_postgres():
    """Start one throwaway cluster per session (unix socket only, trust auth)."""
    if "admin" in _PG:
        return _PG
    data = Path(tempfile.mkdtemp(prefix="mussel-pg-data-"))
    sock = tempfile.mkdtemp(prefix="pg", dir="/tmp")  # socket paths are capped at 107 bytes
    subprocess.run(["initdb", "-D", str(data), "-A", "trust", "-U", "tester"],
                   check=True, capture_output=True)
    port = 20000 + os.getpid() % 20000
    subprocess.run(["pg_ctl", "-D", str(data), "-l", str(data / "pg.log"), "-w", "start",
                    "-o", f"-k {sock} -p {port} -c listen_addresses=''"],
                   check=True, capture_output=True)
    admin = psycopg.connect(host=sock, port=port, user="tester", dbname="postgres", autocommit=True)
    _PG.update(admin=admin, data=data, sock=sock, port=port, n=itertools.count())
    return _PG


@pytest.fixture(scope="session", autouse=True)
def _stop_real_postgres():
    yield
    if "admin" in _PG:
        _PG["admin"].close()
        subprocess.run(["pg_ctl", "-D", str(_PG["data"]), "-m", "immediate", "stop"],
                       capture_output=True)
        shutil.rmtree(_PG["data"], ignore_errors=True)
        shutil.rmtree(_PG["sock"], ignore_errors=True)


_BACKENDS = ["pg-on-sqlite"]
if os.environ.get("MUSSEL_TEST_POSTGRES") == "1" and shutil.which("initdb") and shutil.which("pg_ctl"):
    _BACKENDS.append("postgres")


@pytest.fixture(params=_BACKENDS)
def db(request, tmp_path, monkeypatch):
    """(make_store, faults): StateStores whose sessions carry the armed faults."""
    faults = Faults()
    monkeypatch.setattr(state_mod, "_LOCK_RETRY_BASE_SECONDS", 0.0)
    if request.param == "postgres":
        pg = _real_postgres()
        name = f"t{next(pg['n'])}"
        pg["admin"].execute(f"CREATE DATABASE {name}")
        pg_cfg = PostgresConfig(host=pg["sock"], port=pg["port"], database=name, user="tester")

        def connect(db_cfg, **kwargs):
            faults.connects += 1
            return FaultyConnection(_REAL_CONNECT(db_cfg, **kwargs), faults, sqlite_dialect=False)

        monkeypatch.setattr(state_mod, "connect_database", connect)

        def make_store(cfg: Config | None = None) -> StateStore:
            cfg = cfg or _config(tmp_path)
            return StateStore(dataclasses.replace(cfg, state_backend="postgres", postgres=pg_cfg))
    else:
        path = str(tmp_path / "state" / "pg.db")

        def connect(db_cfg, **kwargs):
            faults.connects += 1
            raw = sqlite3.connect(path, check_same_thread=False, timeout=60, isolation_level=None)
            raw.row_factory = sqlite3.Row
            raw.create_function("pg_advisory_xact_lock", 1, lambda key: None)  # SQLite serializes writers
            return FaultyConnection(raw, faults, sqlite_dialect=True)

        def make_store(cfg: Config | None = None) -> StateStore:
            # The SQLite schema is created unwrapped; sessions after that take the
            # Postgres dialect and the faults.
            monkeypatch.setattr(state_mod, "connect_database", _REAL_CONNECT)
            store = StateStore(path)
            store.close_thread_connection()
            monkeypatch.setattr(state_mod, "connect_database", connect)
            store.db_cfg = dataclasses.replace(store.db_cfg, backend="postgres")
            return store

    return make_store, faults


def _config(tmp_path: Path, **overrides) -> Config:
    for d in ("repo", "results", "work", "dispatch", "state", "logs"):
        (tmp_path / d).mkdir(exist_ok=True)
    values = dict(
        repo_dir=str(tmp_path / "repo"), nextflow_profiles="standard",
        outdir=str(tmp_path / "results"), work_base_dir=str(tmp_path / "work"),
        dispatch_dir=str(tmp_path / "dispatch"), state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )
    values.update(overrides)
    return Config(**values)


def _slide(store, path="/slides/a.svs"):
    return store._conn().execute(
        store._sql("SELECT status, batch_id, fail_count FROM slides WHERE slide_path=?"), (path,)
    ).fetchone()


def _batch(store, batch_id):
    return store._conn().execute(
        store._sql("SELECT status, remote_status, nf_pid FROM batches WHERE batch_id=?"), (batch_id,)
    ).fetchone()


# ── _run_write under the two failure shapes ──────────────────────────────────

def test_commit_unknown_is_raised_not_replayed(db):
    # mark_slides_complete charges fail_count per call: a replay after an applied
    # COMMIT would charge the slide twice.
    make_store, faults = db
    store = make_store()
    store.add_slide("/slides/a.svs", "a")
    store.add_batch("b1", "/b1.csv", "/w1", 1, "/b1.log")
    assert store.mark_dispatched(["/slides/a.svs"], "b1") == 1
    faults.commit_unknown("fail_count")

    with pytest.raises(psycopg.OperationalError, match="server closed the connection"):
        store.mark_slides_complete("b1", False, charge_fail_count=True)

    assert faults.fired == ["commit:fail_count"]
    row = _slide(store)                  # a fresh session sees the applied write, once
    assert (row["status"], row["fail_count"]) == ("FAILED", 1)


def test_session_lost_before_commit_reconnects_and_retries(db):
    make_store, faults = db
    store = make_store()
    store.add_batch("b1", "/b1.csv", "/w1", 1, "/b1.log", backend="remote_agent", target="hpc")
    assert store.write_remote_status("b1", {"status": "QUEUED"})
    connects = faults.connects
    faults.drop_before("SELECT remote_status FROM batches")

    assert store.withdraw_remote_order("b1") is True

    assert faults.fired == ["drop:SELECT remote_status FROM batches"]
    assert faults.connects == connects + 1
    assert _batch(store, "b1")["remote_status"] == "CANCELLED"


# ── remote submit: the order committed, the reply was lost ──────────────────

def _remote_config(tmp_path):
    target = RemoteTargetConfig(name="hpc", remote_scratch_root="/remote/scratch",
                                remote_repo_dir="/remote/repo", poll_interval_seconds=1)
    return _config(tmp_path, state_backend="postgres", execution_backend="remote_agent",
                   dispatch_targets=[target], default_dispatch_target="hpc")


def test_remote_submit_commit_unknown_withdraws_the_committed_order(db, tmp_path, monkeypatch):
    # The QUEUED order is in the database but the runner saw an error: the batch
    # must not stay claimable by the agent once its slides are requeued.
    make_store, faults = db
    cfg = _remote_config(tmp_path)
    for name in ("main.nf", "nextflow.config"):      # the order fingerprints the runtime
        (tmp_path / "repo" / name).write_text("// stub\n")
    store = make_store(cfg)
    monkeypatch.setattr(runner_mod, "build_contract", lambda cfg, target=None: {
        "effective_params_yaml": "", "effective_params_sha256": "p", "source_revision": "r",
        "release_manifest_sha256": "m", "semantic_contract_sha256": "s",
        "execution_profile": {"name": "hpc"}, "execution_profile_sha256": "e",
    })
    monkeypatch.setattr(runner_mod, "RemoteDispatchClient",
                        lambda cfg, target: RemoteDispatchClient(cfg, target, state=store))
    batch_id = "20991231T000000_ab12cd34"
    store.add_slide("/slides/a.svs", "a")
    slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
    faults.commit_unknown('"status": "QUEUED"')
    runner = NextflowRunner(cfg, batch_id, slides, store)

    future: Future = Future()
    with pytest.raises(psycopg.OperationalError) as raised:
        runner.run()
    future.set_exception(raised.value)  # what the worker thread's future carries
    assert faults.fired == ['commit:"status": "QUEUED"']

    manager = sched.RunManager(cfg, store)
    try:
        assert manager._slots.acquire(blocking=False)
        manager._on_done(batch_id, future, runner)
    finally:
        manager.shutdown(wait=False)

    assert store.claim_remote_batch("hpc") is None        # the agent cannot start it
    assert _batch(store, batch_id)["remote_status"] == "CANCELLED"
    assert _batch(store, batch_id)["status"] == "FAILED"
    row = _slide(store)
    assert (row["status"], row["batch_id"]) == ("PENDING", None)


# ── nf_pid: the record commits, the reply is lost, Nextflow keeps running ────

FAKE_NF = """#!/bin/bash
[ "$1" = log ] && exit 1
n=$(( $(cat "$FAKE_NF_DIR/count" 2>/dev/null || echo 0) + 1 ))
echo $n > "$FAKE_NF_DIR/count"
sleep 1.5
echo $$ >> "$FAKE_NF_DIR/pids"
echo done >> "$FAKE_NF_DIR/finished"
exit 0
"""


def test_nf_pid_commit_unknown_keeps_waiting_on_nextflow(db, tmp_path, monkeypatch):
    make_store, faults = db
    cfg = _config(tmp_path, repo_dir=str(tmp_path))
    store = make_store(cfg)
    nf_dir = tmp_path / "nf"
    nf_dir.mkdir()
    fake = nf_dir / "nextflow"
    fake.write_text(FAKE_NF)
    fake.chmod(0o755)
    monkeypatch.setenv("MUSSEL_NEXTFLOW_BIN", str(fake))
    monkeypatch.setenv("FAKE_NF_DIR", str(nf_dir))
    monkeypatch.setattr(runner_mod, "_NF_PID_WRITE_RETRY_SECONDS", 0.0)
    store.add_slide("/slides/a.svs", "a")
    slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
    faults.commit_unknown("SET nf_pid=")      # the first record (the PID) lands, its reply is lost
    seen = {}

    def watch():                               # what restart recovery would read mid-run
        reader = store
        while not (nf_dir / "finished").exists():
            row = _batch(reader, "batch-001") if (nf_dir / "count").exists() else None
            if row and row["nf_pid"]:
                seen["nf_pid"] = row["nf_pid"]
            threading.Event().wait(0.1)
        reader.close_thread_connection()

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        assert NextflowRunner(cfg, "batch-001", slides, store).run() == 0
    finally:
        watcher.join(10)

    assert faults.fired == ["commit:SET nf_pid="]
    assert (nf_dir / "count").read_text().split() == ["1"]       # launched once
    assert (nf_dir / "finished").read_text().split() == ["done"]  # and waited for
    assert seen.get("nf_pid")                                    # recorded while it ran
    assert _batch(store, "batch-001")["nf_pid"] is None
    assert _slide(store)["status"] == "SUCCEEDED"


def test_remote_waiter_reconnects_after_its_session_drops(db, tmp_path, monkeypatch):
    # A read is not retried inside StateStore: the waiter's next poll must get a
    # fresh session from _conn(), not the dead one.
    make_store, faults = db
    cfg = _remote_config(tmp_path)
    store = make_store(cfg)
    monkeypatch.setattr(runner_mod, "RemoteDispatchClient",
                        lambda cfg, target: RemoteDispatchClient(cfg, target, state=store))
    batch_id = "20991231T000000_ab12cd34"
    store.add_batch(batch_id, "/b.csv", "/w", 1, "/b.log", backend="remote_agent", target="hpc")
    assert store.write_remote_status(batch_id, {"status": "SUCCEEDED", "nextflow_exit": 0})
    connects = faults.connects
    faults.drop_before("remote_payload, finalization_retry_at")
    stop = threading.Event()
    runner = NextflowRunner(cfg, batch_id, [], store, resume=True, shutdown_event=stop,
                            existing_csv_path=str(tmp_path / "b.csv"),
                            existing_work_dir=str(tmp_path / "work" / "b"))
    result = {}
    waiter = threading.Thread(target=lambda: result.setdefault("rc", runner.run()))
    waiter.start()
    waiter.join(10)
    finished = not waiter.is_alive()
    stop.set()                       # a waiter stuck on the dead session exits here
    waiter.join(5)

    assert faults.fired == ["drop:remote_payload, finalization_retry_at"]
    assert finished, "waiter kept polling a dead session"
    assert faults.connects > connects
    assert result == {"rc": 0}
