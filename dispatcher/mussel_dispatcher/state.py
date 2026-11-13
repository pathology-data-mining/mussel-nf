"""Database-backed state store for mussel-dispatcher."""
from __future__ import annotations

import logging
import copy
import os
import json
import hashlib
import re as _re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .db import connect_database, resolve_database_config, sql_for_backend, placeholders_for_backend
from .db import psycopg as _psycopg
from .tower_state import newest_tower_state_for_batch, normalize_tower_state
from .wds_status import build_wds_snapshot

try:
    from nextflow_turret import RunStore as _TowerRunStore
except ImportError:  # pragma: no cover - optional in sqlite-only test envs
    _TowerRunStore = None

log = logging.getLogger("mussel-dispatcher")

_ACTIVE_BATCH_STATUSES = ("RUNNING", "RECOVERING", "FINALIZING")
_LOCK_RETRY_ATTEMPTS = 3
_LOCK_RETRY_BASE_SECONDS = 1.0
_POSTGRES_STATUS_READ_TIMEOUT_MS = 5000
_POSTGRES_SCHEMA_LOCK_TIMEOUT_MS = 1000
_POSTGRES_SCHEMA_STATEMENT_TIMEOUT_MS = 5000
# A thread's session idle longer than this is reconnected before use (StateStore._idle_too_long).
_POSTGRES_MAX_IDLE_SECONDS = 600
_SNAPSHOT_ENRICHMENT_TTL_SECONDS = 5 * 60
_SNAPSHOT_ENRICHMENT_RETRY_SECONDS = 60
_RUN_NAME_LOG_TAIL_BYTES = 128 * 1024
# Retained for the observation query below; burst completions are not used by
# estimate_throughput_and_eta for the displayed sustained rate.
_BURST_WINDOW_HOURS = 1.0
# Local and remote finalizers share the same Postgres state database, but WDS
# writes already serialize at the destination/model level in wds.py.  Keep a
# lane per execution backend so a long-running remote hook cannot starve local
# finalization (or vice versa).
_FINALIZATION_LEASE_NAME = "dispatcher-finalization"
LOCAL_FINALIZATION_LEASE_NAME = f"{_FINALIZATION_LEASE_NAME}:local"
REMOTE_FINALIZATION_LEASE_NAME = f"{_FINALIZATION_LEASE_NAME}:remote"
# Databricks sync purges/uploads a shared volume folder before merging.  Keep
# this on the legacy name so older dispatcher/remote-agent processes also
# participate in the same cross-backend exclusion.
DB_SYNC_LEASE_NAME = _FINALIZATION_LEASE_NAME
_TOWER_LIVE_REMOTE_STATUSES = frozenset({"RUNNING", "CANCEL_REQUESTED"})
# Batches that consume fresh-dispatch capacity.  get_ and count_ variants must
# share this predicate so the global slot count and the per-pool breakdown
# agree.  A remote row whose remote_status is still NULL/'' (registered, order
# not yet queued, or stuck there) occupies a slot and is therefore listed too.
# HOOKS_BLOCKED is a quarantine that lasts until an operator retries or waives the
# batch's hooks: it runs nothing, and counting it would hold a run slot indefinitely.
_SCHEDULING_ACTIVE_BATCH_PREDICATE = """
    status IN ('RUNNING','RECOVERING')
    AND (
        backend!='remote_agent'
        OR COALESCE(remote_status, '') NOT IN (
            'SUCCEEDED', 'FAILED', 'CANCELLED', 'STALE',
            'FINALIZATION_QUEUED', 'HOOKS_RUNNING', 'HOOKS_RETRY', 'HOOKS_BLOCKED'
        )
    )
"""


# CREATE TABLE definitions for the Postgres backend.  The SQLite schema in
# StateStore._init_schema mirrors these; columns added after a table first
# shipped live in _COLUMN_MIGRATIONS below.  Together the two are the source of
# truth for the startup schema guard (_required_schema_columns).
_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS slides (
    slide_path    TEXT PRIMARY KEY,
    slide_id      TEXT NOT NULL,
    oncotree_code TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'PENDING',
    batch_id      TEXT,
    download_path TEXT,
    fail_count    INTEGER NOT NULL DEFAULT 0,
    first_seen_at  TEXT,
    dispatched_at  TEXT,
    completed_at   TEXT,
    error_msg      TEXT,
    file_id        TEXT,
    file_name      TEXT,
    needs_download INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id      TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'RUNNING',
    slide_count   INTEGER,
    dispatched_at TEXT,
    completed_at  TEXT,
    nextflow_exit INTEGER,
    csv_path      TEXT,
    work_dir      TEXT,
    log_path      TEXT,
    manifest_path TEXT,
    session_id    TEXT,
    run_name      TEXT,
    source_revision TEXT,
    release_manifest_sha256 TEXT,
    semantic_contract_sha256 TEXT,
    execution_profile TEXT,
    execution_profile_sha256 TEXT,
    params_path TEXT,
    effective_params_sha256 TEXT,
    finalization_stage TEXT NOT NULL DEFAULT 'PENDING',
    finalization_owner TEXT,
    finalization_heartbeat_at TEXT,
    finalization_retry_at TEXT,
    finalization_error TEXT,
    driver_completed_at TEXT,
    execution_cleanup_at TEXT,
    execution_cleanup_error TEXT
);
CREATE TABLE IF NOT EXISTS batch_hooks (
    batch_id      TEXT NOT NULL,
    hook_index    INTEGER NOT NULL,
    phase         TEXT NOT NULL DEFAULT '',
    hook_name     TEXT NOT NULL DEFAULT '',
    command       TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'PENDING',
    started_at    TEXT,
    completed_at  TEXT,
    exit_code     INTEGER,
    error_msg     TEXT,
    required      INTEGER NOT NULL DEFAULT 1,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    executor_job_id TEXT,
    executor_state TEXT,
    executor_submitted_at TEXT,
    executor_last_polled_at TEXT,
    executor_missing_since TEXT,
    failure_category TEXT,
    failure_detail TEXT,
    runtime_revision TEXT,
    execution_node TEXT,
    idempotency_key TEXT,
    active_attempt_number INTEGER,
    PRIMARY KEY (batch_id, hook_index)
);
CREATE TABLE IF NOT EXISTS hook_circuit_breakers (
    target TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT '',
    failure_times TEXT NOT NULL DEFAULT '[]',
    paused INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'CLOSED',
    canary_batch_id TEXT,
    canary_started_at TEXT,
    reason TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hook_failure_events (
    target TEXT NOT NULL,
    hook_name TEXT NOT NULL,
    category TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '',
    failed_at TEXT NOT NULL,
    PRIMARY KEY (target, hook_name, category, batch_id)
);
CREATE TABLE IF NOT EXISTS hook_attempts (
    batch_id TEXT NOT NULL,
    hook_index INTEGER NOT NULL,
    attempt_number INTEGER NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    exit_code INTEGER,
    failure_category TEXT,
    failure_detail TEXT,
    runtime_revision TEXT,
    execution_node TEXT,
    idempotency_key TEXT,
    PRIMARY KEY (batch_id, hook_index, attempt_number)
);
CREATE TABLE IF NOT EXISTS remote_agents (
    target        TEXT PRIMARY KEY,
    heartbeat_at  TEXT,
    host          TEXT,
    status        TEXT,
    payload       TEXT
);
CREATE TABLE IF NOT EXISTS coordination_leases (
    lease_name    TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    batch_id      TEXT,
    acquired_at   TEXT NOT NULL,
    heartbeat_at  TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deployment_releases (
    deployment_id TEXT PRIMARY KEY,
    desired_revision TEXT NOT NULL,
    desired_release_manifest_sha256 TEXT NOT NULL,
    desired_semantic_contract_sha256 TEXT,
    phase TEXT NOT NULL DEFAULT 'STAGED',
    requested_at TEXT NOT NULL,
    requested_by TEXT,
    error TEXT,
    controller_status TEXT,
    controller_message TEXT,
    controller_revision TEXT,
    controller_heartbeat_at TEXT
);
CREATE TABLE IF NOT EXISTS deployment_nodes (
    deployment_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    target TEXT NOT NULL,
    host TEXT,
    status TEXT NOT NULL DEFAULT 'UNKNOWN',
    staged_revision TEXT,
    active_revision TEXT,
    semantic_contract_sha256 TEXT,
    execution_profile_sha256 TEXT,
    heartbeat_at TEXT,
    updated_at TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (deployment_id, node_id)
);
CREATE TABLE IF NOT EXISTS _schema_migrations (name TEXT PRIMARY KEY)
"""

_SLIDE_ROLLUPS_DDL = """
CREATE TABLE IF NOT EXISTS slide_rollups (
    slide_id TEXT PRIMARY KEY,
    has_succeeded INTEGER NOT NULL DEFAULT 0,
    has_retry_pending INTEGER NOT NULL DEFAULT 0,
    has_failed INTEGER NOT NULL DEFAULT 0,
    has_pending INTEGER NOT NULL DEFAULT 0,
    has_dispatched INTEGER NOT NULL DEFAULT 0
)
""".strip()

# Additive column migrations applied to databases created by older releases.
_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "slides": {
        "oncotree_code": "ALTER TABLE slides ADD COLUMN oncotree_code TEXT NOT NULL DEFAULT ''",
        "priority": "ALTER TABLE slides ADD COLUMN priority INTEGER NOT NULL DEFAULT 0",
    },
    "batches": {
        "nf_pid": "ALTER TABLE batches ADD COLUMN nf_pid INTEGER",
        "run_name": "ALTER TABLE batches ADD COLUMN run_name TEXT",
        "backend": "ALTER TABLE batches ADD COLUMN backend TEXT NOT NULL DEFAULT 'local'",
        "target": "ALTER TABLE batches ADD COLUMN target TEXT",
        "remote_run_id": "ALTER TABLE batches ADD COLUMN remote_run_id TEXT",
        "remote_status": "ALTER TABLE batches ADD COLUMN remote_status TEXT",
        "remote_heartbeat_at": "ALTER TABLE batches ADD COLUMN remote_heartbeat_at TEXT",
        "remote_driver_job_id": "ALTER TABLE batches ADD COLUMN remote_driver_job_id TEXT",
        "remote_work_dir": "ALTER TABLE batches ADD COLUMN remote_work_dir TEXT",
        "remote_outdir": "ALTER TABLE batches ADD COLUMN remote_outdir TEXT",
        "remote_log_path": "ALTER TABLE batches ADD COLUMN remote_log_path TEXT",
        "remote_payload": "ALTER TABLE batches ADD COLUMN remote_payload TEXT",
        "source_revision": "ALTER TABLE batches ADD COLUMN source_revision TEXT",
        "release_manifest_sha256": "ALTER TABLE batches ADD COLUMN release_manifest_sha256 TEXT",
        "semantic_contract_sha256": "ALTER TABLE batches ADD COLUMN semantic_contract_sha256 TEXT",
        "execution_profile": "ALTER TABLE batches ADD COLUMN execution_profile TEXT",
        "execution_profile_sha256": "ALTER TABLE batches ADD COLUMN execution_profile_sha256 TEXT",
        "params_path": "ALTER TABLE batches ADD COLUMN params_path TEXT",
        "effective_params_sha256": "ALTER TABLE batches ADD COLUMN effective_params_sha256 TEXT",
        "finalization_stage": "ALTER TABLE batches ADD COLUMN finalization_stage TEXT NOT NULL DEFAULT 'PENDING'",
        "finalization_owner": "ALTER TABLE batches ADD COLUMN finalization_owner TEXT",
        "finalization_heartbeat_at": "ALTER TABLE batches ADD COLUMN finalization_heartbeat_at TEXT",
        "finalization_retry_at": "ALTER TABLE batches ADD COLUMN finalization_retry_at TEXT",
        "finalization_error": "ALTER TABLE batches ADD COLUMN finalization_error TEXT",
        "driver_completed_at": "ALTER TABLE batches ADD COLUMN driver_completed_at TEXT",
        "execution_cleanup_at": "ALTER TABLE batches ADD COLUMN execution_cleanup_at TEXT",
        "execution_cleanup_error": "ALTER TABLE batches ADD COLUMN execution_cleanup_error TEXT",
    },
    "batch_hooks": {
        "required": "ALTER TABLE batch_hooks ADD COLUMN required INTEGER NOT NULL DEFAULT 1",
        "attempt_count": "ALTER TABLE batch_hooks ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0",
        "next_retry_at": "ALTER TABLE batch_hooks ADD COLUMN next_retry_at TEXT",
        "executor_job_id": "ALTER TABLE batch_hooks ADD COLUMN executor_job_id TEXT",
        "executor_state": "ALTER TABLE batch_hooks ADD COLUMN executor_state TEXT",
        "executor_submitted_at": "ALTER TABLE batch_hooks ADD COLUMN executor_submitted_at TEXT",
        "executor_last_polled_at": "ALTER TABLE batch_hooks ADD COLUMN executor_last_polled_at TEXT",
        "executor_missing_since": "ALTER TABLE batch_hooks ADD COLUMN executor_missing_since TEXT",
        "failure_category": "ALTER TABLE batch_hooks ADD COLUMN failure_category TEXT",
        "failure_detail": "ALTER TABLE batch_hooks ADD COLUMN failure_detail TEXT",
        "runtime_revision": "ALTER TABLE batch_hooks ADD COLUMN runtime_revision TEXT",
        "execution_node": "ALTER TABLE batch_hooks ADD COLUMN execution_node TEXT",
        "idempotency_key": "ALTER TABLE batch_hooks ADD COLUMN idempotency_key TEXT",
        "active_attempt_number": "ALTER TABLE batch_hooks ADD COLUMN active_attempt_number INTEGER",
    },
    "hook_circuit_breakers": {
        "state": "ALTER TABLE hook_circuit_breakers ADD COLUMN state TEXT",
        "canary_batch_id": "ALTER TABLE hook_circuit_breakers ADD COLUMN canary_batch_id TEXT",
        "canary_started_at": "ALTER TABLE hook_circuit_breakers ADD COLUMN canary_started_at TEXT",
    },
    "deployment_releases": {
        "controller_status": "ALTER TABLE deployment_releases ADD COLUMN controller_status TEXT",
        "controller_message": "ALTER TABLE deployment_releases ADD COLUMN controller_message TEXT",
        "controller_revision": "ALTER TABLE deployment_releases ADD COLUMN controller_revision TEXT",
        "controller_heartbeat_at": "ALTER TABLE deployment_releases ADD COLUMN controller_heartbeat_at TEXT",
    },
}

_CREATE_TABLE_RE = _re.compile(
    r"\s*CREATE TABLE IF NOT EXISTS (\w+) \((.*)\)\s*$", _re.S
)
_CREATE_INDEX_RE = _re.compile(r"\s*CREATE INDEX IF NOT EXISTS (\w+)\s", _re.I)
_TABLE_CONSTRAINT_PREFIXES = ("PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT")


def _schema_table_columns(ddl: str) -> dict[str, set[str]]:
    """Parse ``CREATE TABLE IF NOT EXISTS`` statements into table -> columns."""
    tables: dict[str, set[str]] = {}
    for statement in ddl.split(";"):
        match = _CREATE_TABLE_RE.match(statement)
        if not match:
            continue
        columns = tables.setdefault(match.group(1), set())
        for line in match.group(2).splitlines():
            token = line.strip().split(" ", 1)[0].strip(",")
            if token and token.upper() not in _TABLE_CONSTRAINT_PREFIXES:
                columns.add(token)
    return tables


def _required_schema_columns() -> dict[str, set[str]]:
    """Every table and column the StateStore queries rely on.

    ``slide_rollups`` is deliberately absent: it is an optional read
    accelerator and callers fall back to live counts when it is missing.
    """
    required = _schema_table_columns(_PG_SCHEMA)
    for table, columns in _COLUMN_MIGRATIONS.items():
        required.setdefault(table, set()).update(columns)
    return required


# Recompute every slide_id's rollup from slides (the table must be empty first). Used by
# the one-time migration and by scripts/import_sqlite_to_postgres.py after an import.
SLIDE_ROLLUPS_REBUILD_SQL = """
    INSERT INTO slide_rollups (
        slide_id, has_succeeded, has_retry_pending, has_failed, has_pending, has_dispatched
    )
    SELECT
        slide_id,
        MAX(CASE WHEN status='SUCCEEDED' THEN 1 ELSE 0 END) AS has_succeeded,
        MAX(CASE WHEN status='PENDING' AND fail_count > 0 THEN 1 ELSE 0 END) AS has_retry_pending,
        MAX(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) AS has_failed,
        MAX(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS has_pending,
        MAX(CASE WHEN status='DISPATCHED' THEN 1 ELSE 0 END) AS has_dispatched
    FROM slides
    GROUP BY slide_id
"""


def finalization_slot_name(lane: str, slot: int) -> str:
    """Return a durable lease name for one local or remote finalizer slot.

    A remote lane is ``remote:<target>``: each target's limit
    (max_concurrent_finalizations is per target) gets its own slots. A bare
    ``remote`` keeps the shared pre-target names.
    """
    kind, _, target = str(lane).strip().partition(":")
    base = {
        "local": LOCAL_FINALIZATION_LEASE_NAME,
        "remote": REMOTE_FINALIZATION_LEASE_NAME,
    }.get(kind.lower())
    if base is None or (target and kind.lower() != "remote"):
        raise ValueError(f"unknown finalization lane: {lane!r}")
    if target:
        base = f"{base}:{target}"
    return base if int(slot) == 0 else f"{base}:{int(slot)}"


def _tower_is_live_for_batch(batch: dict) -> bool:
    """Whether Tower represents active Nextflow execution for this batch."""
    if batch.get("status") != "RUNNING":
        return False
    if batch.get("backend") == "remote_agent":
        return batch.get("remote_status") in _TOWER_LIVE_REMOTE_STATUSES
    return True

def estimate_throughput_and_eta(
    *,
    completed_in_window: int,
    oldest_completed_at: str | None,
    remaining: int,
    window_hours: float,
    recent_completed: int = 0,
    recent_oldest_completed_at: str | None = None,
) -> dict:
    """Estimate throughput and ETA from sustained rolling-window progress.

    Recent batch-completion bursts are deliberately not used: they reflect
    dispatch/finalization timing and can make a large backlog appear to have
    an unrealistically high slides/hour rate.
    """

    def _parse_iso(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            parsed = datetime.fromisoformat(str(ts).replace(" ", "T"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _rate(count: int, oldest: str | None, max_hours: float, min_elapsed_seconds: float = 60.0) -> tuple[float | None, float | None]:
        if count < 2:
            return None, None
        t0 = _parse_iso(oldest)
        if t0 is None:
            return None, None
        now = datetime.now(timezone.utc)
        elapsed_secs = min((now - t0).total_seconds(), max_hours * 3600)
        if elapsed_secs <= min_elapsed_seconds:
            return None, None
        elapsed_hours = elapsed_secs / 3600
        return count / elapsed_hours, elapsed_hours

    throughput = None
    window_elapsed_hours = None
    eta_seconds = None

    base_rate, base_elapsed = _rate(
        completed_in_window, oldest_completed_at, window_hours
    )
    if base_rate is not None:
        throughput = base_rate
        window_elapsed_hours = base_elapsed

    if throughput and throughput > 0:
        eta_seconds = (remaining / throughput) * 3600

    return {
        "window_hours": window_elapsed_hours,
        "throughput_per_hour": round(throughput, 1) if throughput else None,
        "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
    }


class StateStore:
    _POSTGRES_PRIORITY_UPDATE_CHUNK_SIZE = 10_000

    _WRITE_LOCKS: dict[str, threading.RLock] = {}
    _WRITE_LOCKS_GUARD = threading.Lock()

    def __init__(
        self,
        db_or_cfg: str | Config,
        manifest_id_set: frozenset | None = None,
        *,
        strict_schema: bool = False,
    ):
        # strict_schema: used by `release migrate`.  Schema DDL that times out
        # (or the rollup backfill) raises instead of being skipped with a
        # warning, and the resulting schema is verified before commit.
        self._strict_schema = strict_schema
        self.cfg = db_or_cfg if isinstance(db_or_cfg, Config) else None
        self.db_cfg = resolve_database_config(db_or_cfg) if isinstance(db_or_cfg, Config) else None
        self.db_path = (
            self.db_cfg.sqlite_path
            if self.db_cfg and self.db_cfg.backend == "sqlite"
            else str(db_or_cfg if not isinstance(db_or_cfg, Config) else self.db_cfg.location())
        )
        if self.db_cfg is None:
            class _CompatCfg:
                state_backend = "sqlite"
                def state_db_path(self_nonlocal):
                    return str(db_or_cfg)
            self.db_cfg = resolve_database_config(_CompatCfg())  # type: ignore[arg-type]
        self._manifest_id_set = manifest_id_set  # slide_ids with priority=0; others get priority=1
        if self.db_cfg.backend == "sqlite":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._slide_rollups_ready = False
        self._snapshot_enrichment_lock = threading.Lock()
        self._wds_snapshot_cache: dict[str, object] | None = None
        self._tower_snapshot_cache: dict[str, object] | None = None
        self._runtime_status_lock = threading.Lock()
        self._runtime_status: dict[str, object] = {}
        runtime_state_dir = getattr(self.cfg, "state_dir", "") or os.path.dirname(self.db_path)
        self._runtime_status_path = os.path.join(runtime_state_dir, "runtime_status.json")
        with self._WRITE_LOCKS_GUARD:
            self._write_lock = self._WRITE_LOCKS.setdefault(
                os.path.realpath(self.db_path), threading.RLock()
            )
        self._init_schema()

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        manifest_id_set: frozenset | None = None,
        *,
        strict_schema: bool = False,
    ) -> "StateStore":
        return cls(cfg, manifest_id_set=manifest_id_set, strict_schema=strict_schema)

    def _sql(self, sql: str) -> str:
        return sql_for_backend(self.db_cfg, sql)

    def _placeholders(self, n: int) -> str:
        return placeholders_for_backend(self.db_cfg, n)

    def _table_columns(self, conn, table: str) -> set[str]:
        if self.db_cfg.backend == "postgres":
            rows = conn.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = %s
                """,
                (table,),
            ).fetchall()
            return {str(self._row_get(row, "column_name", 0)) for row in rows}
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(self._row_get(row, "name", 1)) for row in rows}

    @staticmethod
    def _row_get(row, key: str, index: int = 0):
        if row is None:
            return None
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return row[index]

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            # Worker threads are long-lived, while PostgreSQL sessions can be
            # closed underneath them after a network/database restart.  A
            # closed psycopg connection must not be handed back to callers;
            # otherwise every subsequent operation in that worker fails before
            # post-batch hooks can even be started.
            now = time.monotonic()
            try:
                if not bool(getattr(conn, "closed", False)) and not self._idle_too_long(conn, now):
                    self._local.conn_used_at = now
                    return conn
            except Exception:
                pass
            self.close_thread_connection()
        self._local.conn = connect_database(self.db_cfg, check_same_thread=False)
        self._local.conn_used_at = time.monotonic()
        return self._local.conn

    def _idle_too_long(self, conn, now: float) -> bool:
        """A PostgreSQL session idle past MUSSEL_POSTGRES_MAX_IDLE_SECONDS is replaced.

        Finalization workers sit idle between batches for an hour or more, long
        enough for the network path to drop the flow without a reset. Reusing
        that socket hung the worker's first write (set_batch_status) inside the
        process-wide write lock until TCP gave up (~15 min), so the main loop and
        the agent heartbeat queued behind it and the stall watchdog killed the
        agent. A fresh session costs one connect.
        """
        if self.db_cfg.backend != "postgres":
            return False
        max_idle = float(os.environ.get(
            "MUSSEL_POSTGRES_MAX_IDLE_SECONDS", str(_POSTGRES_MAX_IDLE_SECONDS)))
        used_at = getattr(self._local, "conn_used_at", None)
        if max_idle <= 0 or used_at is None or now - used_at <= max_idle:
            return False
        # Never drop a session mid-transaction (a nested _conn() in a long write).
        try:
            in_transaction = int(conn.info.transaction_status) != 0   # 0 = IDLE
        except Exception:
            in_transaction = False
        if in_transaction:
            return False
        log.debug("Replacing PostgreSQL session idle for %.0fs", now - used_at)
        return True

    def close_thread_connection(self) -> None:
        """Release the database session owned by the current worker thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass
        try:
            del self._local.conn
        except AttributeError:
            pass

    def _executemany(self, conn, sql: str, rows: list[tuple] | list[list]) -> int:
        """Backend-safe bulk execution.

        sqlite3 connections expose ``executemany()`` directly, while the
        psycopg connection wrapper used here does not. Use a simple loop so the
        same write path works on both backends.
        """
        rowcount = 0
        for params in rows:
            cur = conn.execute(self._sql(sql), params)
            rowcount += max(0, getattr(cur, "rowcount", 0) or 0)
        return rowcount

    def _rollback_read_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        try:
            conn.rollback()
        except Exception:
            pass

    def _refresh_slide_rollups(self, conn, slide_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        slide_ids = sorted({str(slide_id) for slide_id in slide_ids if slide_id})
        if not slide_ids:
            return
        placeholders = self._placeholders(len(slide_ids))
        rows = conn.execute(
            self._sql(
                f"""
                SELECT
                    slide_id,
                    MAX(CASE WHEN status='SUCCEEDED' THEN 1 ELSE 0 END) AS has_succeeded,
                    MAX(CASE WHEN status='PENDING' AND fail_count > 0 THEN 1 ELSE 0 END) AS has_retry_pending,
                    MAX(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) AS has_failed,
                    MAX(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS has_pending,
                    MAX(CASE WHEN status='DISPATCHED' THEN 1 ELSE 0 END) AS has_dispatched
                FROM slides
                WHERE slide_id IN ({placeholders})
                GROUP BY slide_id
                ORDER BY slide_id
                """
            ),
            slide_ids,
        ).fetchall()
        present_ids = {self._row_get(row, "slide_id", 0) for row in rows}
        missing_ids = [slide_id for slide_id in slide_ids if slide_id not in present_ids]
        if missing_ids:
            delete_placeholders = self._placeholders(len(missing_ids))
            conn.execute(
                self._sql(f"DELETE FROM slide_rollups WHERE slide_id IN ({delete_placeholders})"),
                missing_ids,
            )
        if not rows:
            return
        if self.db_cfg.backend == "postgres":
            upsert_sql = """
                INSERT INTO slide_rollups (
                    slide_id, has_succeeded, has_retry_pending, has_failed, has_pending, has_dispatched
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(slide_id) DO UPDATE SET
                    has_succeeded=EXCLUDED.has_succeeded,
                    has_retry_pending=EXCLUDED.has_retry_pending,
                    has_failed=EXCLUDED.has_failed,
                    has_pending=EXCLUDED.has_pending,
                    has_dispatched=EXCLUDED.has_dispatched
            """
        else:
            upsert_sql = """
                INSERT INTO slide_rollups (
                    slide_id, has_succeeded, has_retry_pending, has_failed, has_pending, has_dispatched
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(slide_id) DO UPDATE SET
                    has_succeeded=excluded.has_succeeded,
                    has_retry_pending=excluded.has_retry_pending,
                    has_failed=excluded.has_failed,
                    has_pending=excluded.has_pending,
                    has_dispatched=excluded.has_dispatched
            """
        self._executemany(
            conn,
            upsert_sql,
            [
                (
                    self._row_get(row, "slide_id", 0),
                    int(self._row_get(row, "has_succeeded", 1) or 0),
                    int(self._row_get(row, "has_retry_pending", 2) or 0),
                    int(self._row_get(row, "has_failed", 3) or 0),
                    int(self._row_get(row, "has_pending", 4) or 0),
                    int(self._row_get(row, "has_dispatched", 5) or 0),
                )
                for row in rows
            ],
        )

    def _refresh_slide_rollups_for_paths(self, conn, slide_paths: list[str] | tuple[str, ...] | set[str]) -> None:
        slide_paths = sorted({str(slide_path) for slide_path in slide_paths if slide_path})
        if not slide_paths:
            return
        placeholders = self._placeholders(len(slide_paths))
        rows = conn.execute(
            self._sql(f"SELECT DISTINCT slide_id FROM slides WHERE slide_path IN ({placeholders})"),
            slide_paths,
        ).fetchall()
        self._refresh_slide_rollups(
            conn,
            [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)],
        )

    def _refresh_slide_rollups_for_batch(self, conn, batch_id: str) -> None:
        rows = conn.execute(
            self._sql("SELECT DISTINCT slide_id FROM slides WHERE batch_id=?"),
            (batch_id,),
        ).fetchall()
        self._refresh_slide_rollups(
            conn,
            [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)],
        )

    def _set_postgres_status_read_timeout(self, conn) -> bool:
        if self.db_cfg.backend != "postgres":
            return False
        timeout_ms = int(
            os.environ.get(
                "MUSSEL_POSTGRES_STATUS_READ_TIMEOUT_MS",
                str(_POSTGRES_STATUS_READ_TIMEOUT_MS),
            )
        )
        conn.execute(f"SET statement_timeout TO '{timeout_ms}ms'")
        return True

    def _reset_postgres_status_read_timeout(self, conn) -> None:
        if self.db_cfg.backend != "postgres":
            return
        try:
            conn.execute("SET statement_timeout TO DEFAULT")
        except Exception:
            pass

    @staticmethod
    def _load_snapshot_file(path: str) -> dict | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _missing_schema_columns(self, conn) -> list[str]:
        """Return ``table`` / ``table.column`` entries absent from the database."""
        missing: list[str] = []
        for table, columns in sorted(_required_schema_columns().items()):
            present = self._table_columns(conn, table)
            if not present:
                missing.append(table)
                continue
            missing.extend(f"{table}.{column}" for column in sorted(columns - present))
        return missing

    def _init_schema(self):
        if (
            self.db_cfg.backend == "postgres"
            and os.environ.get("MUSSEL_POSTGRES_SKIP_SCHEMA_INIT", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            conn = self._conn()
            missing = self._missing_schema_columns(conn)
            if missing:
                raise RuntimeError(
                    "Postgres schema init was skipped but required tables/columns are missing: "
                    + ", ".join(missing)
                    + "; run `release migrate` before starting services"
                )
            self._slide_rollups_ready = bool(
                self._row_get(
                    conn.execute("SELECT to_regclass(%s)", ("public.slide_rollups",)).fetchone(),
                    "to_regclass",
                    0,
                )
            )
            log.info(
                "Skipping Postgres schema init on startup; required tables already exist"
            )
            return

        def _has_column(conn, table: str, column: str) -> bool:
            if self.db_cfg.backend == "postgres":
                row = conn.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name=%s AND column_name=%s",
                    (table, column),
                ).fetchone()
                return bool(row)
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return any(self._row_get(row, "name", 1) == column for row in rows)

        def _init(conn) -> None:
            if self.db_cfg.backend == "postgres":
                conn.execute(
                    f"SET LOCAL lock_timeout TO '{_POSTGRES_SCHEMA_LOCK_TIMEOUT_MS}ms'"
                )
                # Startup DDL is bounded; an explicit migration (index builds,
                # the rollup backfill over every slide) must be allowed to finish.
                statement_timeout_ms = 0 if self._strict_schema else _POSTGRES_SCHEMA_STATEMENT_TIMEOUT_MS
                conn.execute(f"SET LOCAL statement_timeout TO '{statement_timeout_ms}ms'")

            def _exec_schema_ddl(sql: str) -> None:
                if self.db_cfg.backend == "postgres":
                    index = _CREATE_INDEX_RE.match(sql)
                    if index and conn.execute(
                        "SELECT 1 FROM pg_indexes "
                        "WHERE schemaname = current_schema() AND indexname = %s",
                        (index.group(1),),
                    ).fetchone():
                        # CREATE INDEX IF NOT EXISTS still takes a SHARE lock
                        # on the table; skip it when the index already exists.
                        return
                    conn.execute("SAVEPOINT schema_ddl")
                try:
                    conn.execute(sql)
                except Exception as exc:
                    msg = str(exc).lower()
                    timed_out = (
                        "statement timeout" in msg
                        or "lock timeout" in msg
                        or "deadlock detected" in msg
                    )
                    if self.db_cfg.backend == "postgres" and (
                        "must be owner of relation" in msg
                        or "permission denied" in msg
                        or timed_out
                    ):
                        if self._strict_schema:
                            # An explicit migration must not report success
                            # while leaving columns, indexes or backfills
                            # unapplied (timeouts or missing privileges).
                            raise
                        conn.execute("ROLLBACK TO SAVEPOINT schema_ddl")
                        conn.execute("RELEASE SAVEPOINT schema_ddl")
                        log.warning("Postgres schema DDL skipped: %s", sql.strip())
                        return
                    raise
                else:
                    if self.db_cfg.backend == "postgres":
                        conn.execute("RELEASE SAVEPOINT schema_ddl")

            sqlite_schema = """
                CREATE TABLE IF NOT EXISTS slides (
                    slide_path    TEXT PRIMARY KEY,
                    slide_id      TEXT NOT NULL,
                    oncotree_code TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'PENDING',
                    batch_id      TEXT,
                    download_path TEXT,
                    fail_count    INTEGER NOT NULL DEFAULT 0,
                    first_seen_at  TEXT,
                    dispatched_at  TEXT,
                    completed_at   TEXT,
                    error_msg      TEXT,
                    file_id        TEXT,
                    file_name      TEXT,
                    needs_download INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id      TEXT PRIMARY KEY,
                    status        TEXT NOT NULL DEFAULT 'RUNNING',
                    slide_count   INTEGER,
                    dispatched_at TEXT,
                    completed_at  TEXT,
                    nextflow_exit INTEGER,
                    csv_path      TEXT,
                    work_dir      TEXT,
                    log_path      TEXT,
                    manifest_path TEXT,
                    session_id    TEXT,
                    backend       TEXT NOT NULL DEFAULT 'local',
                    target        TEXT,
                    remote_run_id TEXT,
                    remote_status TEXT,
                    remote_heartbeat_at TEXT,
                    remote_driver_job_id TEXT,
                    remote_work_dir TEXT,
                    remote_outdir TEXT,
                    remote_log_path TEXT,
                    remote_payload TEXT,
                    run_name      TEXT,
                    source_revision TEXT,
                    release_manifest_sha256 TEXT,
                    semantic_contract_sha256 TEXT,
                    execution_profile TEXT,
                    execution_profile_sha256 TEXT,
                    params_path TEXT,
                    effective_params_sha256 TEXT,
                    finalization_stage TEXT NOT NULL DEFAULT 'PENDING',
                    finalization_owner TEXT,
                    finalization_heartbeat_at TEXT,
                    finalization_retry_at TEXT,
                    finalization_error TEXT,
                    driver_completed_at TEXT,
                    execution_cleanup_at TEXT,
                    execution_cleanup_error TEXT
                );
                CREATE TABLE IF NOT EXISTS batch_hooks (
                    batch_id      TEXT NOT NULL,
                    hook_index    INTEGER NOT NULL,
                    phase         TEXT NOT NULL DEFAULT '',
                    hook_name     TEXT NOT NULL DEFAULT '',
                    command       TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'PENDING',
                    started_at    TEXT,
                    completed_at  TEXT,
                    exit_code     INTEGER,
                    error_msg     TEXT,
                    required      INTEGER NOT NULL DEFAULT 1,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT,
                    executor_job_id TEXT,
                    executor_state TEXT,
                    executor_submitted_at TEXT,
                    executor_last_polled_at TEXT,
                    executor_missing_since TEXT,
                    failure_category TEXT,
                    failure_detail TEXT,
                    runtime_revision TEXT,
                    execution_node TEXT,
                    idempotency_key TEXT,
                    active_attempt_number INTEGER,
                    PRIMARY KEY (batch_id, hook_index)
                );
                CREATE TABLE IF NOT EXISTS hook_circuit_breakers (
                    target TEXT PRIMARY KEY,
                    category TEXT NOT NULL DEFAULT '',
                    failure_times TEXT NOT NULL DEFAULT '[]',
                    paused INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'CLOSED',
                    canary_batch_id TEXT,
                    canary_started_at TEXT,
                    reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hook_failure_events (
                    target TEXT NOT NULL,
                    hook_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL DEFAULT 0,
                    detail TEXT NOT NULL DEFAULT '',
                    failed_at TEXT NOT NULL,
                    PRIMARY KEY (target, hook_name, category, batch_id)
                );
                CREATE TABLE IF NOT EXISTS hook_attempts (
                    batch_id TEXT NOT NULL,
                    hook_index INTEGER NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    exit_code INTEGER,
                    failure_category TEXT,
                    failure_detail TEXT,
                    runtime_revision TEXT,
                    execution_node TEXT,
                    idempotency_key TEXT,
                    PRIMARY KEY (batch_id, hook_index, attempt_number)
                );
                CREATE TABLE IF NOT EXISTS remote_agents (
                    target        TEXT PRIMARY KEY,
                    heartbeat_at  TEXT,
                    host          TEXT,
                    status        TEXT,
                    payload       TEXT
                );
                CREATE TABLE IF NOT EXISTS coordination_leases (
                    lease_name    TEXT PRIMARY KEY,
                    owner         TEXT NOT NULL,
                    batch_id      TEXT,
                    acquired_at   TEXT NOT NULL,
                    heartbeat_at  TEXT NOT NULL,
                    expires_at    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deployment_releases (
                    deployment_id TEXT PRIMARY KEY,
                    desired_revision TEXT NOT NULL,
                    desired_release_manifest_sha256 TEXT NOT NULL,
                    desired_semantic_contract_sha256 TEXT,
                    phase TEXT NOT NULL DEFAULT 'STAGED',
                    requested_at TEXT NOT NULL,
                    requested_by TEXT,
                    error TEXT,
                    controller_status TEXT,
                    controller_message TEXT,
                    controller_revision TEXT,
                    controller_heartbeat_at TEXT
                );
                CREATE TABLE IF NOT EXISTS deployment_nodes (
                    deployment_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    host TEXT,
                    status TEXT NOT NULL DEFAULT 'UNKNOWN',
                    staged_revision TEXT,
                    active_revision TEXT,
                    semantic_contract_sha256 TEXT,
                    execution_profile_sha256 TEXT,
                    heartbeat_at TEXT,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    PRIMARY KEY (deployment_id, node_id)
                );
                CREATE TABLE IF NOT EXISTS _schema_migrations (name TEXT PRIMARY KEY);
            """
            if self.db_cfg.backend == "sqlite":
                # Not executescript(): it COMMITs the BEGIN IMMEDIATE first, so the
                # migration below would run in autocommit (and concurrently).
                # Split on ";": keep the schema free of triggers and of ";" in comments or strings.
                for stmt in [s.strip() for s in sqlite_schema.split(";") if s.strip()]:
                    conn.execute(stmt)
            else:
                for stmt in [s.strip() for s in _PG_SCHEMA.split(";") if s.strip()]:
                    _exec_schema_ddl(stmt)
            _exec_schema_ddl(_SLIDE_ROLLUPS_DDL)

            def _add_missing_columns(table: str) -> None:
                for col, ddl in _COLUMN_MIGRATIONS[table].items():
                    if not _has_column(conn, table, col):
                        _exec_schema_ddl(ddl)

            # Migrate existing DBs that predate oncotree_code / priority.
            _add_missing_columns("slides")
            # Migrate existing DBs where batches column was named nf_session_id instead of session_id.
            has_session_id = _has_column(conn, "batches", "session_id")
            has_nf_session_id = _has_column(conn, "batches", "nf_session_id")
            if has_nf_session_id and not has_session_id:
                _exec_schema_ddl("ALTER TABLE batches RENAME COLUMN nf_session_id TO session_id")
            _add_missing_columns("batches")
            _add_missing_columns("batch_hooks")

            # Older batches have no explicit driver-terminal marker.  Their
            # completed timestamps are safe recovery anchors for cleanup;
            # remote finalization rows use the durable remote heartbeat.
            if _has_column(conn, "batches", "completed_at"):
                _exec_schema_ddl(
                    "UPDATE batches SET driver_completed_at=COALESCE(driver_completed_at, completed_at) "
                    "WHERE status IN ('SUCCEEDED','FAILED','CANCELLED') AND completed_at IS NOT NULL"
                )
            if all(
                _has_column(conn, "batches", column)
                for column in ("backend", "remote_status", "remote_heartbeat_at")
            ):
                _exec_schema_ddl(
                    "UPDATE batches SET driver_completed_at=COALESCE(driver_completed_at, remote_heartbeat_at) "
                    "WHERE backend='remote_agent' AND remote_status IN "
                    "('FINALIZATION_QUEUED','HOOKS_RUNNING','HOOKS_RETRY','SUCCEEDED','FAILED','CANCELLED') "
                    "AND remote_heartbeat_at IS NOT NULL"
                )

            _exec_schema_ddl(
                "CREATE TABLE IF NOT EXISTS hook_circuit_breakers ("
                "target TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT '', "
                "failure_times TEXT NOT NULL DEFAULT '[]', paused INTEGER NOT NULL DEFAULT 0, "
                "state TEXT NOT NULL DEFAULT 'CLOSED', canary_batch_id TEXT, "
                "canary_started_at TEXT, "
                "reason TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)"
            )
            _add_missing_columns("hook_circuit_breakers")
            _exec_schema_ddl(
                "UPDATE hook_circuit_breakers SET state=CASE WHEN paused=1 "
                "THEN 'OPEN' ELSE 'CLOSED' END WHERE state IS NULL OR state=''"
            )
            _exec_schema_ddl(
                "CREATE TABLE IF NOT EXISTS hook_failure_events ("
                "target TEXT NOT NULL, hook_name TEXT NOT NULL, category TEXT NOT NULL, "
                "batch_id TEXT NOT NULL, attempt_number INTEGER NOT NULL DEFAULT 0, "
                "detail TEXT NOT NULL DEFAULT '', failed_at TEXT NOT NULL, "
                "PRIMARY KEY (target, hook_name, category, batch_id))"
            )

            _add_missing_columns("deployment_releases")

            # Ensure performance indexes exist (idempotent).
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_slides_status ON slides(status)")
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_slides_batch_id ON slides(batch_id)")
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_slides_slide_id ON slides(slide_id)")
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status)")
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_batches_remote_queue "
                "ON batches(target, remote_status)"
            )
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_batches_remote_finalization "
                "ON batches(target, remote_status, finalization_retry_at)"
            )
            if all(
                _has_column(conn, "batches", column)
                for column in ("target", "remote_status", "dispatched_at", "batch_id")
            ):
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_batches_remote_queue_order "
                    "ON batches(target, remote_status, dispatched_at, batch_id)"
                )
            if all(
                _has_column(conn, "batches", column)
                for column in (
                    "target", "remote_status", "finalization_retry_at",
                    "dispatched_at", "batch_id",
                )
            ):
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_batches_remote_finalization_order "
                    "ON batches(target, remote_status, finalization_retry_at, dispatched_at, batch_id)"
                )
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_batch_hooks_batch_id ON batch_hooks(batch_id)")
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_batch_hooks_status ON batch_hooks(status)")
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_batch_hooks_executor_poll "
                "ON batch_hooks(status, executor_state, executor_job_id, batch_id)"
            )
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_hook_failure_events_window "
                "ON hook_failure_events(target, hook_name, category, failed_at)"
            )
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_batch_hooks_retry "
                "ON batch_hooks(batch_id, status, next_retry_at)"
            )
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_remote_agents_heartbeat_at ON remote_agents(heartbeat_at)")
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_deployment_nodes_status "
                "ON deployment_nodes(deployment_id, status)"
            )
            _exec_schema_ddl(
                "CREATE INDEX IF NOT EXISTS idx_coordination_leases_expires_at "
                "ON coordination_leases(expires_at)"
            )
            _exec_schema_ddl("CREATE INDEX IF NOT EXISTS idx_slide_rollups_pending ON slide_rollups(has_pending)")
            if self.db_cfg.backend == "postgres":
                # Keep Postgres indexes on the raw ISO-8601 text columns instead
                # of cast expressions so startup schema checks stay restart-safe.
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_slides_status_slide_id "
                    "ON slides(status, slide_id)"
                )
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_slides_pending_retry "
                    "ON slides(status, fail_count, slide_id) "
                    "WHERE status = 'PENDING' AND fail_count > 0"
                )
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_slides_succeeded_completed_at "
                    "ON slides(status, completed_at) "
                    "WHERE status = 'SUCCEEDED' AND completed_at IS NOT NULL AND completed_at <> ''"
                )
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_slides_pending_priority_queue "
                    "ON slides(status, priority DESC, first_seen_at, slide_path) "
                    "WHERE status = 'PENDING'"
                )
                _exec_schema_ddl(
                    "CREATE INDEX IF NOT EXISTS idx_batches_running_dispatched_at "
                    "ON batches(status, dispatched_at) "
                    "WHERE status IN ('RUNNING', 'RECOVERING', 'FINALIZING') "
                    "AND dispatched_at IS NOT NULL AND dispatched_at <> ''"
                )
            # Backfilling this table rewrites every rollup and its indexes.  It must
            # be a migration, not a startup task: on GPFS a restart loop otherwise
            # turns a small control-plane outage into sustained database I/O.
            rollup_migration = "slide_rollups_v1"
            migrated = conn.execute(
                self._sql("SELECT 1 FROM _schema_migrations WHERE name=?"),
                (rollup_migration,),
            ).fetchone()
            # On Postgres a failed statement aborts the whole transaction; the
            # savepoint confines a backfill failure so the DDL above still
            # commits instead of being silently rolled back with it.
            use_savepoint = self.db_cfg.backend == "postgres" and not migrated
            if use_savepoint:
                conn.execute("SAVEPOINT slide_rollups_backfill")
            try:
                if not migrated:
                    rollup_count = self._row_get(
                        conn.execute("SELECT COUNT(*) AS cnt FROM slide_rollups").fetchone(),
                        "cnt",
                        0,
                    ) or 0
                    if not rollup_count:
                        conn.execute(SLIDE_ROLLUPS_REBUILD_SQL)
                    # Existing populated rollups were produced by the previous
                    # startup backfill.  Preserve them rather than rewriting the
                    # entire table during this upgrade.
                    conn.execute(
                        self._sql("INSERT INTO _schema_migrations (name) VALUES (?)"),
                        (rollup_migration,),
                    )
                if use_savepoint:
                    conn.execute("RELEASE SAVEPOINT slide_rollups_backfill")
                self._slide_rollups_ready = True
            except Exception as exc:
                if use_savepoint:
                    conn.execute("ROLLBACK TO SAVEPOINT slide_rollups_backfill")
                    conn.execute("RELEASE SAVEPOINT slide_rollups_backfill")
                self._slide_rollups_ready = False
                if self._strict_schema:
                    raise
                log.warning("Slide rollup migration skipped; falling back to live distinct counts: %s", exc)

            if self._strict_schema:
                missing = self._missing_schema_columns(conn)
                if missing:
                    raise RuntimeError(
                        "Schema migration finished but required tables/columns are still missing: "
                        + ", ".join(missing)
                    )

        self._run_write("init_schema", _init)

    @staticmethod
    def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
        msg = str(exc).lower()
        return "database is locked" in msg or "database table is locked" in msg

    @staticmethod
    def _is_postgres_retryable_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "deadlock detected" in msg
            or "could not serialize access" in msg
            or "serialization failure" in msg
        )

    @staticmethod
    def _is_postgres_connection_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "another command is already in progress",
                "connection is closed",
                "connection already closed",
                "sending query and params failed",
                "the cursor is closed",
            )
        )

    @staticmethod
    def _is_postgres_connection_lost(exc: Exception, conn) -> bool:
        """The server session died mid-transaction (after statements were sent).

        The server rolls back an open transaction when its session ends, so a
        loss detected before COMMIT was sent leaves nothing applied.
        """
        msg = str(exc).lower()
        if any(
            marker in msg
            for marker in (
                "server closed the connection unexpectedly",
                "consuming input failed",
                "terminating connection due to administrator command",
                "ssl connection has been closed unexpectedly",
                "connection to server was lost",
            )
        ):
            return True
        if _psycopg is None or not isinstance(exc, _psycopg.OperationalError):
            return False
        if isinstance(exc, getattr(_psycopg.errors, "AdminShutdown", ())):
            return True
        try:
            return bool(getattr(conn, "closed", False) or getattr(conn, "broken", False))
        except Exception:
            return False

    def _run_write(self, op_name: str, fn):
        for attempt in range(1, _LOCK_RETRY_ATTEMPTS + 1):
            # Resolve the connection for each attempt so a connection closed
            # by a previous database error is replaced before BEGIN.
            conn = self._conn()
            committing = False
            with self._write_lock:
                try:
                    if self.db_cfg.backend == "sqlite":
                        conn.execute("BEGIN IMMEDIATE")
                    else:
                        conn.execute("BEGIN")
                    result = fn(conn)
                    committing = True
                    conn.commit()
                    return result
                except sqlite3.OperationalError as exc:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                    if self._is_locked_error(exc) and attempt < _LOCK_RETRY_ATTEMPTS:
                        sleep_s = _LOCK_RETRY_BASE_SECONDS * attempt
                        log.warning(
                            "SQLite write lock during %s on %s (attempt %d/%d); retrying in %.1fs",
                            op_name, self.db_path, attempt, _LOCK_RETRY_ATTEMPTS, sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    raise
                except Exception as exc:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    connection_error = (
                        self.db_cfg.backend == "postgres"
                        and self._is_postgres_connection_error(exc)
                    )
                    connection_lost = (
                        self.db_cfg.backend == "postgres"
                        and not connection_error
                        and self._is_postgres_connection_lost(exc, conn)
                    )
                    if connection_error or connection_lost:
                        self.close_thread_connection()
                    if connection_lost and committing:
                        # COMMIT may have reached the server before the
                        # session died; the outcome is unknown, so replaying
                        # a non-idempotent write here is unsafe.
                        raise
                    connection_error = connection_error or connection_lost
                    if (
                        self.db_cfg.backend == "postgres"
                        and (
                            connection_error
                            or self._is_postgres_retryable_error(exc)
                        )
                        and attempt < _LOCK_RETRY_ATTEMPTS
                    ):
                        sleep_s = _LOCK_RETRY_BASE_SECONDS * attempt
                        log.warning(
                            "Postgres write retry during %s on %s (attempt %d/%d): %s; retrying in %.1fs",
                            op_name,
                            self.db_path,
                            attempt,
                            _LOCK_RETRY_ATTEMPTS,
                            exc,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    raise

    # -----------------------------------------------------------------------
    # Slide operations
    # -----------------------------------------------------------------------

    def resolve_priority(self, slide_id: str, priority_override: int | None = None) -> int:
        base_priority = (
            1 if (self._manifest_id_set is not None and slide_id not in self._manifest_id_set)
            else 0
        )
        if priority_override is None:
            return base_priority
        try:
            return max(base_priority, int(priority_override))
        except (TypeError, ValueError):
            return base_priority

    def add_slide(self, slide_path: str, slide_id: str, *,
                  oncotree_code: str = "",
                  file_id: str = "", file_name: str = "",
                  needs_download: bool = False,
                  priority_override: int | None = None) -> bool:
        """Insert a new slide. Returns True if inserted, False if already known."""
        priority = self.resolve_priority(slide_id, priority_override)
        def _insert(conn) -> int:
            cur = conn.execute(
                self._sql("""INSERT INTO slides (slide_path, slide_id, oncotree_code, status, first_seen_at,
                       file_id, file_name, needs_download, priority)
                       VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)
                       ON CONFLICT(slide_path) DO NOTHING"""),
                    (slide_path, slide_id, oncotree_code,
                     datetime.now(timezone.utc).isoformat(),
                     file_id, file_name, int(needs_download), priority),
                )
            if cur.rowcount:
                self._refresh_slide_rollups(conn, [slide_id])
            return cur.rowcount
        if self.db_cfg.backend == "sqlite":
            try:
                def _insert_sqlite(conn) -> int:
                    cur = conn.execute(
                        """INSERT INTO slides (slide_path, slide_id, oncotree_code, status, first_seen_at,
                           file_id, file_name, needs_download, priority)
                           VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)""",
                        (slide_path, slide_id, oncotree_code,
                         datetime.now(timezone.utc).isoformat(),
                         file_id, file_name, int(needs_download), priority),
                    )
                    if cur.rowcount:
                        self._refresh_slide_rollups(conn, [slide_id])
                    return cur.rowcount
                return bool(self._run_write("add_slide", _insert_sqlite))
            except sqlite3.IntegrityError:
                return False
        return bool(self._run_write("add_slide", _insert))

    def is_known(self, slide_path: str) -> bool:
        return bool(self._conn().execute(
            self._sql("SELECT 1 FROM slides WHERE slide_path=?"), (slide_path,)
        ).fetchone())

    def is_known_by_id(self, slide_id: str) -> bool:
        return bool(self._conn().execute(
            self._sql("SELECT 1 FROM slides WHERE slide_id=?"), (slide_id,)
        ).fetchone())

    def get_slides_by_id(self, slide_id: str) -> list:
        rows = self._conn().execute(
            self._sql("SELECT * FROM slides WHERE slide_id=?"), (slide_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # When a slide_id has several paths (rows), the one that speaks for it: work in
    # flight first, then done, then failed. SKIPPED marks a twin path processed
    # through another row, so it represents the slide only if nothing else does.
    _SLIDE_STATUS_PRECEDENCE = {"DISPATCHED": 0, "PENDING": 1, "SUCCEEDED": 2, "FAILED": 3, "SKIPPED": 5}

    def get_slide_statuses(self, slide_ids: list[str]) -> dict[str, dict[str, str | None]]:
        """Return status and current batch ownership for requested slides.

        One entry per slide_id. With several rows (paths) for a slide_id, the row
        with the highest-precedence status is reported (``_SLIDE_STATUS_PRECEDENCE``;
        ties go to the row with a batch, then the lowest batch_id), never an
        arbitrary one: a SKIPPED twin must not hide a FAILED or DISPATCHED row.
        """
        if not slide_ids:
            return {}
        ordered_ids = sorted(set(slide_ids))
        placeholders = self._placeholders(len(ordered_ids))
        rows = self._conn().execute(
            self._sql(
                f"SELECT slide_id, status, batch_id FROM slides "
                f"WHERE slide_id IN ({placeholders})"
            ),
            ordered_ids,
        ).fetchall()

        def rank(entry: dict[str, str | None]) -> tuple[int, int, str]:
            status, batch = str(entry["status"] or ""), entry["batch_id"]
            return (self._SLIDE_STATUS_PRECEDENCE.get(status, 4), 0 if batch else 1, str(batch or ""))

        out: dict[str, dict[str, str | None]] = {}
        for row in rows:
            slide_id = str(self._row_get(row, "slide_id", 0))
            entry = {"status": self._row_get(row, "status", 1), "batch_id": self._row_get(row, "batch_id", 2)}
            if slide_id not in out or rank(entry) < rank(out[slide_id]):
                out[slide_id] = entry
        return out

    def remove_slide(self, slide_path: str):
        def _remove(conn) -> None:
            row = conn.execute(
                self._sql("SELECT slide_id FROM slides WHERE slide_path=?"),
                (slide_path,),
            ).fetchone()
            conn.execute(self._sql("DELETE FROM slides WHERE slide_path=?"), (slide_path,))
            if row:
                self._refresh_slide_rollups(conn, [self._row_get(row, "slide_id", 0)])
        self._run_write("remove_slide", _remove)

    def get_pending_slides(self, *, limit: int | None = None) -> list:
        sql = (
            """SELECT slide_path, slide_id, oncotree_code, download_path, file_id, file_name, needs_download
               FROM slides WHERE status='PENDING' AND slide_path != ''
               ORDER BY priority DESC, first_seen_at ASC, slide_path ASC"""
        )
        params: tuple[object, ...] = ()
        if limit is not None and limit > 0:
            sql += f" LIMIT {self._placeholders(1)}"
            params = (int(limit),)
        rows = self._conn().execute(self._sql(sql), params).fetchall()
        slides = [dict(r) for r in rows]
        # Backfill file_id/file_name from gdc:// URIs for rows from older DB schema
        for s in slides:
            if s.get("needs_download") and not s.get("file_id"):
                sp = s.get("slide_path", "")
                if sp.startswith("gdc://"):
                    rest = sp[len("gdc://"):]
                    slash = rest.find("/")
                    if slash > 0:
                        s["file_id"] = rest[:slash]
                        s["file_name"] = rest[slash + 1:]
        return slides

    def count_pending_slides(self) -> int:
        row = self._conn().execute(
            """SELECT COUNT(*) AS cnt
               FROM slides
               WHERE status='PENDING' AND slide_path != ''"""
        ).fetchone()
        return int(self._row_get(row, "cnt", 0) or 0)

    def get_pending_slide_rows(self, *, limit: int | None = None) -> list[dict]:
        sql = (
            """SELECT slide_path, slide_id, priority
               FROM slides
               WHERE status='PENDING' AND slide_path != ''
               ORDER BY priority DESC, first_seen_at ASC, slide_path ASC"""
        )
        params: tuple[object, ...] = ()
        if limit is not None and limit > 0:
            sql += f" LIMIT {self._placeholders(1)}"
            params = (int(limit),)
        rows = self._conn().execute(self._sql(sql), params).fetchall()
        return [dict(r) for r in rows]

    def update_slide_priorities(self, rows: list[tuple[str, int]]) -> int:
        if not rows:
            return 0
        def _update(conn) -> int:
            if self.db_cfg.backend == "postgres":
                updated = 0
                for start in range(0, len(rows), self._POSTGRES_PRIORITY_UPDATE_CHUNK_SIZE):
                    chunk = rows[start : start + self._POSTGRES_PRIORITY_UPDATE_CHUNK_SIZE]
                    placeholders = ",".join(["(%s, %s)"] * len(chunk))
                    params: list[object] = []
                    for slide_path, priority in chunk:
                        params.extend((slide_path, priority))
                    cur = conn.execute(
                        f"""
                        UPDATE slides AS s
                           SET priority = v.priority
                          FROM (VALUES {placeholders}) AS v(slide_path, priority)
                         WHERE s.slide_path = v.slide_path
                           AND s.status = 'PENDING'
                        """,
                        params,
                    )
                    updated += max(0, getattr(cur, "rowcount", 0) or 0)
                return updated
            return self._executemany(
                conn,
                "UPDATE slides SET priority=? WHERE slide_path=? AND status='PENDING'",
                [(priority, slide_path) for slide_path, priority in rows],
            )
        return self._run_write("update_slide_priorities", _update)

    def get_all_slides(self) -> list[dict]:
        """Return all slide rows with a standard set of columns.

        Columns: slide_path, slide_id, oncotree_code, status, error_msg,
                 first_seen_at, completed_at
        """
        rows = self._conn().execute(
            """SELECT slide_path, slide_id, oncotree_code, status, error_msg,
                      first_seen_at, completed_at
               FROM slides"""
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_dispatched(self, slide_paths: list, batch_id: str) -> int:
        """Atomically claim PENDING slides for a batch; return how many were claimed.

        See ``claim_slides`` for the claimed paths themselves.
        """
        return len(self.claim_slides(slide_paths, batch_id))

    def claim_slides(self, slide_paths: list, batch_id: str) -> list[str]:
        """Atomically claim PENDING slides for a batch.

        Only slides still in PENDING status are updated.  Returns the
        slide_paths actually claimed; anything missing was claimed first by a
        racing dispatcher (or is no longer PENDING) and must not be run by
        this batch.
        """
        now = datetime.now(timezone.utc).isoformat()
        slide_paths = list(dict.fromkeys(slide_paths))

        def _mark(conn) -> list[str]:
            if not slide_paths:
                return []
            if self.db_cfg.backend == "postgres":
                placeholders = self._placeholders(len(slide_paths))
                rows = conn.execute(
                    self._sql(
                        f"""
                        UPDATE slides
                           SET status='DISPATCHED', batch_id=?, dispatched_at=?
                         WHERE slide_path IN ({placeholders})
                           AND status='PENDING'
                     RETURNING slide_id, slide_path
                        """
                    ),
                    (batch_id, now, *slide_paths),
                ).fetchall()
                claimed = [str(self._row_get(row, "slide_path", 1)) for row in rows]
                slide_ids = [
                    self._row_get(row, "slide_id", 0)
                    for row in rows
                    if self._row_get(row, "slide_id", 0)
                ]
            else:
                claimed = []
                for p in slide_paths:
                    cur = conn.execute(
                        self._sql("UPDATE slides SET status='DISPATCHED', batch_id=?, dispatched_at=?"
                        " WHERE slide_path=? AND status='PENDING'"),
                        (batch_id, now, p),
                    )
                    if cur.rowcount == 1:
                        claimed.append(p)
                slide_ids = []
                if claimed:
                    rows = conn.execute(
                        self._sql(
                            "SELECT DISTINCT slide_id FROM slides WHERE slide_path IN "
                            f"({self._placeholders(len(claimed))})"
                        ),
                        claimed,
                    ).fetchall()
                    slide_ids = [self._row_get(row, "slide_id", 0) for row in rows]
            self._refresh_slide_rollups(conn, [sid for sid in slide_ids if sid])
            return claimed

        return list(self._run_write("mark_dispatched", _mark) or [])

    def reject_pending_slides(
        self, reasons: dict[str, str], *, terminal_status: str = "FAILED"
    ) -> int:
        """Retire PENDING slides that cannot be part of a batch (e.g. dropped from its CSV).

        ``reasons`` maps slide_path -> error message.  Only PENDING rows are
        touched, so a slide another dispatcher already claimed is left alone.

        ``terminal_status``:
          * ``FAILED`` (default) charges fail_count, so the row follows the
            normal retry/blacklist policy.  For rows that cannot be processed.
          * ``SKIPPED`` is terminal and never charged: the retry sweep only
            resets FAILED rows, slide rollups ignore it, and watchers'
            ``add_slide`` never revives an existing row.  Used for a second
            path of a slide_id that is processed through its twin, so the
            slide_id is computed (and appended to WDS/h5) exactly once.
        """
        if terminal_status not in {"FAILED", "SKIPPED"}:
            raise ValueError(f"unsupported terminal_status {terminal_status!r}")
        if not reasons:
            return 0
        charge = 1 if terminal_status == "FAILED" else 0
        now = datetime.now(timezone.utc).isoformat()

        def _reject(conn) -> int:
            rejected = 0
            slide_ids: list[str] = []
            for slide_path, reason in reasons.items():
                row = conn.execute(
                    self._sql("SELECT slide_id FROM slides WHERE slide_path=? AND status='PENDING'"),
                    (slide_path,),
                ).fetchone()
                if row is None:
                    continue
                cur = conn.execute(
                    self._sql(
                        "UPDATE slides SET status=?, completed_at=?, error_msg=?, "
                        "fail_count=fail_count+? WHERE slide_path=? AND status='PENDING'"
                    ),
                    (terminal_status, now, str(reason)[:2000], charge, slide_path),
                )
                if cur.rowcount == 1:
                    rejected += 1
                    slide_ids.append(self._row_get(row, "slide_id", 0))
            self._refresh_slide_rollups(conn, slide_ids)
            return rejected

        return int(self._run_write("reject_pending_slides", _reject) or 0)

    def mark_slides_complete(self, batch_id: str, succeeded: bool,
                             per_slide_status: dict | None = None,
                             charge_fail_count: bool = True,
                             unless_preserved: bool = False):
        """Mark all slides in a batch as SUCCEEDED or FAILED.

        When charge_fail_count=False (fast infra failure), slides are reset to
        PENDING instead of FAILED so they don't burn a retry slot.
        ``unless_preserved`` leaves the slides alone while the batch is
        RECOVERING/FINALIZING (shutdown preserved it for restart recovery).
        """
        now = datetime.now(timezone.utc).isoformat()
        def _mark(conn) -> None:
            slide_ids: list[str] = []
            if unless_preserved:
                lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
                row = conn.execute(
                    self._sql("SELECT status FROM batches WHERE batch_id=?" + lock),
                    (batch_id,),
                ).fetchone()
                if self._row_get(row, "status", 0) in ("RECOVERING", "FINALIZING"):
                    return
            if per_slide_status:
                slide_paths = list(per_slide_status.keys())
                if slide_paths:
                    placeholders = self._placeholders(len(slide_paths))
                    rows = conn.execute(
                        self._sql(f"SELECT DISTINCT slide_id FROM slides WHERE slide_path IN ({placeholders})"),
                        slide_paths,
                    ).fetchall()
                    slide_ids = [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)]
                for slide_path, ok in per_slide_status.items():
                    status = "SUCCEEDED" if ok else "FAILED"
                    conn.execute(
                        self._sql(
                            "UPDATE slides SET status=?, completed_at=? "
                            "WHERE slide_path=? AND batch_id=? AND status='DISPATCHED'"
                        ),
                        (status, now, slide_path, batch_id),
                    )
            elif not succeeded and not charge_fail_count:
                rows = conn.execute(
                    self._sql("SELECT DISTINCT slide_id FROM slides WHERE batch_id=? AND status='DISPATCHED'"),
                    (batch_id,),
                ).fetchall()
                slide_ids = [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)]
                conn.execute(
                    self._sql("""UPDATE slides
                       SET status='PENDING', batch_id=NULL, completed_at=NULL, error_msg=NULL
                       WHERE batch_id=? AND status='DISPATCHED'"""),
                    (batch_id,),
                )
            else:
                rows = conn.execute(
                    self._sql("SELECT DISTINCT slide_id FROM slides WHERE batch_id=? AND status='DISPATCHED'"),
                    (batch_id,),
                ).fetchall()
                slide_ids = [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)]
                status = "SUCCEEDED" if succeeded else "FAILED"
                # Charge fail_count in the same statement that leaves
                # DISPATCHED, so a replayed terminal update (no rows still
                # DISPATCHED) neither re-marks nor re-charges anything.
                conn.execute(
                    self._sql("""UPDATE slides SET status=?, completed_at=?, fail_count=fail_count+?
                       WHERE batch_id=? AND status='DISPATCHED'"""),
                    (status, now, 0 if succeeded else 1, batch_id),
                )
            self._refresh_slide_rollups(conn, slide_ids)
        self._run_write("mark_slides_complete", _mark)

    def reset_dispatched_to_pending(self, batch_id: str):
        def _reset(conn) -> None:
            rows = conn.execute(
                self._sql("SELECT DISTINCT slide_id FROM slides WHERE batch_id=? AND status='DISPATCHED'"),
                (batch_id,),
            ).fetchall()
            conn.execute(
                self._sql("""UPDATE slides
                   SET status='PENDING', batch_id=NULL, completed_at=NULL, error_msg=NULL
                   WHERE batch_id=? AND status='DISPATCHED'"""),
                (batch_id,),
            )
            self._refresh_slide_rollups(
                conn,
                [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)],
            )
        self._run_write("reset_dispatched_to_pending", _reset)

    def reset_failed_to_pending(self, max_retries: int = 0) -> int:
        """Reset FAILED slides to PENDING. Slides at/above max_retries are left FAILED."""
        skipped_box: dict[str, int] = {"count": 0}

        def _reset(conn) -> int:
            rows = conn.execute(
                self._sql(
                    "SELECT DISTINCT slide_id FROM slides WHERE status='FAILED'"
                    + (" AND fail_count < ?" if max_retries > 0 else "")
                ),
                ((max_retries,) if max_retries > 0 else ()),
            ).fetchall()
            slide_ids = [self._row_get(row, "slide_id", 0) for row in rows if self._row_get(row, "slide_id", 0)]
            if max_retries > 0:
                cur = conn.execute(
                    self._sql("UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL, "
                    "completed_at=NULL, error_msg=NULL "
                    "WHERE status='FAILED' AND fail_count < ?"),
                    (max_retries,),
                )
                skipped_box["count"] = conn.execute(
                    self._sql("SELECT COUNT(*) FROM slides WHERE status='FAILED' AND fail_count >= ?"),
                    (max_retries,),
                ).fetchone()
                skipped_box["count"] = int(self._row_get(skipped_box["count"], "count", 0) or 0)
                self._refresh_slide_rollups(conn, slide_ids)
                return cur.rowcount
            cur = conn.execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL, "
                "completed_at=NULL, error_msg=NULL WHERE status='FAILED'"
            )
            self._refresh_slide_rollups(conn, slide_ids)
            return cur.rowcount

        rowcount = self._run_write("reset_failed_to_pending", _reset)
        if skipped_box["count"]:
            log.warning("Permanently skipping %d slide(s) with fail_count >= %d.",
                        skipped_box["count"], max_retries)
        return rowcount

    def reset_succeeded_to_pending(self, slide_paths: list[str]) -> int:
        """Reset specific SUCCEEDED slides back to PENDING for re-dispatch.

        Used when an external status check (e.g. prepare_samples) reports that
        a slide still needs processing even though the dispatcher DB recorded it
        as SUCCEEDED — for example when output files were deleted after the run.
        Only transitions SUCCEEDED → PENDING; DISPATCHED/FAILED slides are left
        unchanged.

        Returns the number of slides actually reset.
        """
        if not slide_paths:
            return 0
        placeholders = self._placeholders(len(slide_paths))
        def _reset(conn) -> int:
            rows = conn.execute(
                self._sql(f"SELECT DISTINCT slide_id FROM slides WHERE slide_path IN ({placeholders})"),
                slide_paths,
            ).fetchall()
            conn.execute(
                self._sql(f"""UPDATE slides
                    SET status='PENDING', batch_id=NULL, fail_count=0,
                        dispatched_at=NULL, completed_at=NULL, error_msg=NULL """
                f"WHERE slide_path IN ({placeholders}) AND status='SUCCEEDED'"),
                slide_paths,
            )
            row = conn.execute(
                self._sql(f"SELECT COUNT(*) AS cnt FROM slides WHERE slide_path IN ({placeholders}) AND status='PENDING'"),
                slide_paths,
            ).fetchone()
            self._refresh_slide_rollups(
                conn,
                [self._row_get(item, "slide_id", 0) for item in rows if self._row_get(item, "slide_id", 0)],
            )
            return int(self._row_get(row, "cnt", 0) or 0)
        return self._run_write("reset_succeeded_to_pending", _reset)

    def reconcile_incomplete_wds_slides(
        self,
        slide_ids: list[str],
        *,
        source_batch_id: str = "",
        max_retries: int = 0,
        failure_reason: str = "Permanently failed: missing from WDS after max retries",
    ) -> tuple[list[str], list[str]]:
        """Reset WDS-incomplete slides to PENDING or permanently FAIL them.

        Returns ``(retryable, permanently_failed)`` slide ID lists.
        """
        if not slide_ids:
            return [], []
        ordered_ids = sorted(set(slide_ids))
        placeholders = self._placeholders(len(ordered_ids))

        def _reconcile(conn) -> tuple[list[str], list[str]]:
            ownership = " AND batch_id=?" if source_batch_id else ""
            query_params = [*ordered_ids]
            if source_batch_id:
                query_params.append(source_batch_id)
            rows = conn.execute(
                self._sql(f"SELECT slide_id, fail_count FROM slides "
                f"WHERE slide_id IN ({placeholders}) AND status='SUCCEEDED'{ownership}",
                ),
                query_params,
            ).fetchall()
            fail_counts = {self._row_get(row, "slide_id"): int(self._row_get(row, "fail_count") or 0) for row in rows}
            retryable: list[str] = []
            permanently_failed: list[str] = []
            for slide_id in ordered_ids:
                current = fail_counts.get(slide_id)
                if current is None:
                    continue
                if max_retries > 0 and current + 1 >= max_retries:
                    permanently_failed.append(slide_id)
                else:
                    retryable.append(slide_id)

            if permanently_failed:
                perm_placeholders = self._placeholders(len(permanently_failed))
                conn.execute(
                    self._sql(f"UPDATE slides SET status='FAILED', fail_count=fail_count+1, "
                    f"batch_id=NULL, dispatched_at=NULL, error_msg=? "
                    f"WHERE slide_id IN ({perm_placeholders}) AND status='SUCCEEDED'"
                    f"{(' AND batch_id=?' if source_batch_id else '')}"),
                    [failure_reason, *permanently_failed, *([source_batch_id] if source_batch_id else [])],
                )
            if retryable:
                retry_placeholders = self._placeholders(len(retryable))
                conn.execute(
                    self._sql(f"UPDATE slides SET status='PENDING', fail_count=fail_count+1, "
                    f"batch_id=NULL, dispatched_at=NULL, completed_at=NULL "
                    f"WHERE slide_id IN ({retry_placeholders}) AND status='SUCCEEDED'"
                    f"{(' AND batch_id=?' if source_batch_id else '')}"),
                    [*retryable, *([source_batch_id] if source_batch_id else [])],
                )
            self._refresh_slide_rollups(conn, ordered_ids)
            return retryable, permanently_failed

        return self._run_write("reconcile_incomplete_wds_slides", _reconcile)

    def blacklist_slide(self, slide_id: str, reason: str, max_retries: int = 999):
        """Permanently exclude a slide from future dispatch.

        Sets status=FAILED and fail_count=max_retries so reset_failed_to_pending()
        never re-queues it. Inserts a tombstone row if the slide is not yet known.
        """
        def _blacklist(conn) -> None:
            existing = conn.execute(
                self._sql("SELECT slide_path FROM slides WHERE slide_id=?"), (slide_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    self._sql("""UPDATE slides SET status='FAILED', fail_count=?, error_msg=?,
                       completed_at=?, batch_id=NULL WHERE slide_id=?"""),
                    (max_retries, reason, datetime.now(timezone.utc).isoformat(), slide_id),
                )
            else:
                conn.execute(
                    self._sql("""INSERT INTO slides (slide_path, slide_id, status, fail_count, error_msg, first_seen_at)
                       VALUES ('', ?, 'FAILED', ?, ?, ?)"""),
                    (slide_id, max_retries, reason, datetime.now(timezone.utc).isoformat()),
                )
            self._refresh_slide_rollups(conn, [slide_id])
        self._run_write("blacklist_slide", _blacklist)
        log.warning("Blacklisted slide %s: %s", slide_id, reason)

    def set_download_path(self, slide_path: str, download_path: str):
        self._run_write(
            "set_download_path",
            lambda conn: conn.execute(
                self._sql("UPDATE slides SET download_path=? WHERE slide_path=?"), (download_path, slide_path)
            ),
        )

    def get_batch_download_paths(self, batch_id: str) -> list:
        rows = self._conn().execute(
            self._sql("SELECT download_path FROM slides WHERE batch_id=? AND download_path IS NOT NULL"),
            (batch_id,),
        ).fetchall()
        return [self._row_get(r, "download_path") for r in rows]

    def get_old_batch_logs(self, older_than_days: int) -> list:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        rows = self._conn().execute(
            self._sql("""SELECT batch_id, log_path FROM batches
               WHERE status='SUCCEEDED' AND log_path IS NOT NULL AND completed_at < ?"""),
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Batch operations
    # -----------------------------------------------------------------------

    def add_batch(self, batch_id: str, csv_path: str, work_dir: str,
                  slide_count: int, log_path: str, run_name: str | None = None,
                  *, backend: str = "local",
                  target: str | None = None,
                  source_revision: str | None = None,
                  release_manifest_sha256: str | None = None,
                  semantic_contract_sha256: str | None = None,
                  execution_profile: str | None = None,
                  execution_profile_sha256: str | None = None,
                  params_path: str | None = None,
                  effective_params_sha256: str | None = None):
        def _add(conn) -> None:
            sql = self._sql(
                """INSERT INTO batches
                   (batch_id, status, slide_count, dispatched_at, csv_path, work_dir, log_path, run_name, backend, target,
                    source_revision, release_manifest_sha256, semantic_contract_sha256, execution_profile,
                    execution_profile_sha256, params_path, effective_params_sha256)
                   VALUES (?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(batch_id) DO UPDATE SET
                     status=excluded.status,
                     slide_count=excluded.slide_count,
                     dispatched_at=excluded.dispatched_at,
                     csv_path=excluded.csv_path,
                     work_dir=excluded.work_dir,
                     log_path=excluded.log_path,
                     run_name=excluded.run_name,
                     backend=excluded.backend,
                     target=excluded.target,
                     source_revision=excluded.source_revision,
                     release_manifest_sha256=excluded.release_manifest_sha256,
                     semantic_contract_sha256=excluded.semantic_contract_sha256,
                     execution_profile=excluded.execution_profile,
                     execution_profile_sha256=excluded.execution_profile_sha256,
                     params_path=excluded.params_path,
                     effective_params_sha256=excluded.effective_params_sha256"""
                if self.db_cfg.backend == "postgres"
                else
                """INSERT OR REPLACE INTO batches
                   (batch_id, status, slide_count, dispatched_at, csv_path, work_dir, log_path, run_name, backend, target,
                    source_revision, release_manifest_sha256, semantic_contract_sha256, execution_profile,
                    execution_profile_sha256, params_path, effective_params_sha256)
                   VALUES (?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            )
            conn.execute(
                sql,
                (batch_id, slide_count, datetime.now(timezone.utc).isoformat(),
                 csv_path, work_dir, log_path, run_name, backend, target,
                 source_revision, release_manifest_sha256, semantic_contract_sha256,
                 execution_profile, execution_profile_sha256, params_path,
                 effective_params_sha256),
            )
        self._run_write("add_batch", _add)

    def restart_batch(self, batch_id: str):
        """Mark a recovered batch as RUNNING again (for -resume runs)."""
        self._run_write(
            "restart_batch",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET status='RUNNING', completed_at=NULL, nextflow_exit=NULL WHERE batch_id=?"),
                (batch_id,),
            ),
        )

    def set_batch_status(self, batch_id: str, status: str):
        """Set a batch status without marking it terminal."""
        self._run_write(
            "set_batch_status",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET status=?, completed_at=NULL WHERE batch_id=?"),
                (status, batch_id),
            ),
        )

    def set_batch_finalization_stage(self, batch_id: str, stage: str) -> None:
        self._run_write(
            "set_batch_finalization_stage",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET finalization_stage=? WHERE batch_id=?"),
                (stage, batch_id),
            ),
        )

    def defer_local_finalization_retry(
        self, batch_id: str, *, error: str = "", delay_seconds: int = 60
    ) -> None:
        """Back off local finalization without losing its durable state."""
        now = datetime.now(timezone.utc)
        retry_at = (now + timedelta(seconds=max(1, delay_seconds))).isoformat()
        self._run_write(
            "defer_local_finalization_retry",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batches SET finalization_retry_at=?, finalization_error=? "
                    "WHERE batch_id=? AND status='FINALIZING'"
                ),
                (retry_at, error[:2000], batch_id),
            ),
        )

    def cancel_batch(self, batch_id: str):
        """Remove a batch record that was never actually started (race-condition abort)."""
        def _cancel(conn) -> None:
            conn.execute(self._sql("DELETE FROM batch_hooks WHERE batch_id=?"), (batch_id,))
            conn.execute(self._sql("DELETE FROM batches WHERE batch_id=?"), (batch_id,))

        self._run_write("cancel_batch", _cancel)

    def set_batch_slide_count(self, batch_id: str, slide_count: int) -> None:
        self._run_write(
            "set_batch_slide_count",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET slide_count=? WHERE batch_id=?"),
                (int(slide_count), batch_id),
            ),
        )

    def complete_batch(self, batch_id: str, exit_code: int, *, unless_preserved: bool = False) -> bool:
        """Mark a batch terminal; returns False when no row was updated.

        ``unless_preserved`` leaves a batch that shutdown handed to restart
        recovery (RECOVERING/FINALIZING) untouched, in the same statement.
        """
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        completed_at = datetime.now(timezone.utc).isoformat()
        guard = " AND status NOT IN ('RECOVERING', 'FINALIZING')" if unless_preserved else ""
        def _complete(conn) -> bool:
            cur = conn.execute(
                self._sql("""UPDATE batches SET status=?, completed_at=?, nextflow_exit=?, nf_pid=NULL,
                                      driver_completed_at=COALESCE(driver_completed_at, ?),
                                      finalization_stage=CASE WHEN ?=0 THEN 'SUCCEEDED' ELSE finalization_stage END,
                                      finalization_retry_at=NULL, finalization_error=NULL
                   WHERE batch_id=?""" + guard),
                (status, completed_at, exit_code, completed_at, exit_code, batch_id),
            )
            return (getattr(cur, "rowcount", 0) or 0) > 0
        return bool(self._run_write("complete_batch", _complete))

    def preserve_batch_for_recovery(self, batch_id: str, status: str) -> bool:
        """Hand a live batch to restart recovery unless it already ended.

        Returns False when the batch is already terminal: its worker booked
        the outcome first, and flipping it back would resume a batch whose
        slides were already released or charged.
        """
        def _preserve(conn) -> bool:
            cur = conn.execute(
                self._sql(
                    "UPDATE batches SET status=?, completed_at=NULL "
                    "WHERE batch_id=? AND status NOT IN ('SUCCEEDED', 'FAILED')"
                ),
                (status, batch_id),
            )
            return (getattr(cur, "rowcount", 0) or 0) > 0
        return bool(self._run_write("preserve_batch_for_recovery", _preserve))

    def mark_driver_terminal(self, batch_id: str, exit_code: int) -> None:
        """Record driver completion before finalization or hook execution."""
        completed_at = datetime.now(timezone.utc).isoformat()
        self._run_write(
            "mark_driver_terminal",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batches SET driver_completed_at=COALESCE(driver_completed_at, ?), "
                    "nextflow_exit=?, nf_pid=NULL WHERE batch_id=?"
                ),
                (completed_at, exit_code, batch_id),
            ),
        )

    def record_execution_cleanup(
        self,
        batch_id: str,
        *,
        error: str = "",
        retryable: bool = True,
    ) -> None:
        """Persist the result of the shared execution-artifact cleanup."""
        now = datetime.now(timezone.utc).isoformat()
        self._run_write(
            "record_execution_cleanup",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batches SET execution_cleanup_at=?, execution_cleanup_error=? "
                    "WHERE batch_id=?"
                ),
                (now if not error or not retryable else None, error[:2000], batch_id),
            ),
        )

    def get_execution_cleanup_candidates(self) -> list[dict]:
        rows = self._conn().execute(
            self._sql(
                "SELECT * FROM batches WHERE driver_completed_at IS NOT NULL "
                "AND execution_cleanup_at IS NULL "
                "AND (status NOT IN ('RUNNING','RECOVERING','CANCEL_REQUESTED') "
                "OR (backend='remote_agent' AND remote_status IN "
                "('FINALIZATION_QUEUED','HOOKS_RUNNING','HOOKS_RETRY','SUCCEEDED','FAILED','CANCELLED')))"
            )
        ).fetchall()
        return [dict(row) for row in rows]

    def update_remote_batch(self, batch_id: str, **fields):
        allowed = {
            "remote_run_id", "remote_status", "remote_heartbeat_at",
            "remote_driver_job_id", "remote_work_dir", "remote_outdir",
            "remote_log_path",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        payload = fields.get("remote_payload")
        if payload is not None:
            updates["remote_payload"] = json.dumps(payload, sort_keys=True)
        if not updates:
            return
        assignments = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [batch_id]
        self._run_write(
            "update_remote_batch",
            lambda conn: conn.execute(
                self._sql(f"UPDATE batches SET {assignments} WHERE batch_id=?"), values
            ),
        )

    def transition_remote_status(
        self,
        batch_id: str,
        *,
        expected_statuses: set[str] | tuple[str, ...],
        status: str,
        payload: dict[str, object] | None = None,
        heartbeat_at: str | None = None,
        finalization_retry_at: str | None = None,
    ) -> bool:
        """Atomically transition a remote batch under the database write lock.

        The payload is merged while the row is locked.  Callers must provide the
        states from which their transition is valid; this prevents stale pollers
        from moving a batch backwards (for example HOOKS_RUNNING -> RUNNING).
        """
        expected = {str(value).upper() for value in expected_statuses}
        now = heartbeat_at or datetime.now(timezone.utc).isoformat()

        def _transition(conn) -> bool:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql(
                    "SELECT batch_id, remote_status, remote_heartbeat_at, "
                    "remote_driver_job_id, remote_work_dir, remote_outdir, "
                    "remote_log_path, remote_payload FROM batches WHERE batch_id=?" + lock
                ),
                (batch_id,),
            ).fetchone()
            current = self._remote_status_from_row(row)
            current_status = str((current or {}).get("status") or "").upper()
            if current_status not in expected:
                return False
            merged = dict(current or {"batch_id": batch_id})
            merged.update(payload or {})
            merged["status"] = status
            merged.setdefault("batch_id", batch_id)
            merged["heartbeat_at"] = now
            updates = {
                "remote_status": status,
                "remote_heartbeat_at": now,
                "remote_driver_job_id": merged.get("driver_job_id"),
                "remote_work_dir": merged.get("remote_work_dir"),
                "remote_outdir": merged.get("remote_outdir"),
                "remote_log_path": merged.get("remote_log_path"),
                "remote_payload": json.dumps(merged, sort_keys=True),
                "finalization_retry_at": finalization_retry_at,
            }
            assignments = ", ".join(f"{key}=?" for key in updates)
            conn.execute(
                self._sql(f"UPDATE batches SET {assignments} WHERE batch_id=?"),
                [*updates.values(), batch_id],
            )
            return True

        return bool(self._run_write("transition_remote_status", _transition))

    @staticmethod
    def _decode_remote_payload(raw_payload) -> dict:
        if isinstance(raw_payload, dict):
            return dict(raw_payload)
        if not raw_payload:
            return {}
        try:
            payload = json.loads(raw_payload)
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _remote_status_from_row(self, row) -> dict | None:
        if not row:
            return None
        payload = self._decode_remote_payload(self._row_get(row, "remote_payload", 0))
        payload["batch_id"] = self._row_get(row, "batch_id", 1)
        for key, col in (
            ("status", "remote_status"),
            ("heartbeat_at", "remote_heartbeat_at"),
            ("driver_job_id", "remote_driver_job_id"),
            ("remote_work_dir", "remote_work_dir"),
            ("remote_outdir", "remote_outdir"),
            ("remote_log_path", "remote_log_path"),
        ):
            value = self._row_get(row, col)
            if value:
                payload[key] = value
        # Some legacy callers select a narrower batch column set. Avoid the
        # generic positional fallback here, which would otherwise interpret
        # batch_id as finalization_retry_at when the column is absent.
        try:
            retry_at = row["finalization_retry_at"]
        except (KeyError, IndexError, TypeError):
            retry_at = None
        if retry_at:
            payload["finalization_retry_at"] = retry_at
        return payload

    def read_remote_status(self, batch_id: str) -> dict | None:
        row = self._conn().execute(
            self._sql(
                """
                SELECT batch_id, remote_status, remote_heartbeat_at,
                       remote_driver_job_id, remote_work_dir, remote_outdir,
                       remote_log_path, remote_payload, finalization_retry_at
                  FROM batches
                 WHERE batch_id=?
                """
            ),
            (batch_id,),
        ).fetchone()
        return self._remote_status_from_row(row)

    def write_remote_status(self, batch_id: str, payload: dict[str, object]) -> bool:
        incoming = dict(payload or {})
        requested_status = str(incoming.get("status") or "").upper()
        terminal = {"SUCCEEDED", "FAILED", "CANCELLED"}
        active = {
            "QUEUED", "CLAIMED", "STARTING", "SUBMITTED", "RUNNING",
            "CANCEL_REQUESTED", "FINALIZATION_QUEUED", "HOOKS_RETRY",
            "HOOKS_RUNNING", "HOOKS_BLOCKED",
        }

        def _write(conn) -> None:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql(
                    "SELECT batch_id, remote_status, remote_heartbeat_at, "
                    "remote_driver_job_id, remote_work_dir, remote_outdir, "
                    "remote_log_path, remote_payload, finalization_retry_at "
                    "FROM batches WHERE batch_id=?" + lock
                ),
                (batch_id,),
            ).fetchone()
            existing = self._remote_status_from_row(row) or {"batch_id": batch_id}
            current_status = str(existing.get("status") or "").upper()
            # Compatibility callers may still use this method, but they may not
            # regress a state owned by a newer transition writer.
            if current_status in terminal and requested_status not in {"", current_status}:
                incoming.pop("status", None)
            elif current_status in {"HOOKS_RUNNING", "HOOKS_RETRY", "HOOKS_BLOCKED"} and requested_status in active - {"HOOKS_RUNNING", "HOOKS_RETRY", "HOOKS_BLOCKED"}:
                incoming.pop("status", None)
            merged = dict(existing)
            merged.update(incoming)
            merged.setdefault("batch_id", batch_id)
            heartbeat = merged.get("heartbeat_at") or merged.get("updated_at")
            if heartbeat:
                merged["heartbeat_at"] = heartbeat
            meaningful_keys = (
                "status", "driver_job_id", "remote_run_dir", "remote_work_dir",
                "remote_outdir", "remote_log_path", "slurm", "nextflow_exit",
                "failure_class", "error", "cancel_requested_at", "log_tail",
                "lost_jobs", "driver_stall", "warning", "stall_cancelled_at",
            )
            if all(existing.get(key) == merged.get(key) for key in meaningful_keys):
                return False
            updates = {
                "remote_status": merged.get("status"),
                "remote_heartbeat_at": heartbeat,
                "remote_driver_job_id": merged.get("driver_job_id"),
                "remote_work_dir": merged.get("remote_work_dir"),
                "remote_outdir": merged.get("remote_outdir"),
                "remote_log_path": merged.get("remote_log_path"),
                "remote_payload": json.dumps(merged, sort_keys=True),
            }
            assignments = ", ".join(f"{key}=?" for key in updates)
            conn.execute(
                self._sql(f"UPDATE batches SET {assignments} WHERE batch_id=?"),
                [*updates.values(), batch_id],
            )
            return True

        return bool(self._run_write("write_remote_status", _write))

    def read_remote_agent(self, target: str) -> dict | None:
        row = self._conn().execute(
            self._sql(
                """
                SELECT target, heartbeat_at, host, status, payload
                  FROM remote_agents
                 WHERE target=?
                """
            ),
            (target,),
        ).fetchone()
        if not row:
            return None
        payload = self._decode_remote_payload(self._row_get(row, "payload", 0))
        payload["target"] = self._row_get(row, "target", 1)
        for key, col in (
            ("heartbeat_at", "heartbeat_at"),
            ("host", "host"),
            ("status", "status"),
        ):
            value = self._row_get(row, col)
            if value:
                payload[key] = value
        return payload

    def write_remote_agent(self, target: str, payload: dict[str, object]) -> None:
        merged = dict(self.read_remote_agent(target) or {"target": target})
        merged.update(payload or {})
        heartbeat_at = merged.get("heartbeat_at")
        host = merged.get("host")
        status = merged.get("status")
        raw_payload = json.dumps(merged, sort_keys=True)

        def _write(conn):
            if self.db_cfg.backend == "postgres":
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO remote_agents (target, heartbeat_at, host, status, payload)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (target) DO UPDATE
                            SET heartbeat_at=EXCLUDED.heartbeat_at,
                                host=EXCLUDED.host,
                                status=EXCLUDED.status,
                                payload=EXCLUDED.payload
                        """
                    ),
                    (target, heartbeat_at, host, status, raw_payload),
                )
            else:
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO remote_agents (target, heartbeat_at, host, status, payload)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(target) DO UPDATE SET
                            heartbeat_at=excluded.heartbeat_at,
                            host=excluded.host,
                            status=excluded.status,
                            payload=excluded.payload
                        """
                    ),
                    (target, heartbeat_at, host, status, raw_payload),
                )

        self._run_write("write_remote_agent", _write)

    def write_remote_agent_snapshot(self, target: str, payload: dict[str, object]) -> None:
        """Replace the complete agent snapshot in one write without a read-before-write."""
        merged = dict(payload or {})
        merged["target"] = target
        raw_payload = json.dumps(merged, sort_keys=True)

        def _write(conn):
            if self.db_cfg.backend == "postgres":
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO remote_agents (target, heartbeat_at, host, status, payload)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (target) DO UPDATE
                            SET heartbeat_at=EXCLUDED.heartbeat_at,
                                host=EXCLUDED.host,
                                status=EXCLUDED.status,
                                payload=EXCLUDED.payload
                        """
                    ),
                    (
                        target, merged.get("heartbeat_at"), merged.get("host"),
                        merged.get("status"), raw_payload,
                    ),
                )
            else:
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO remote_agents (target, heartbeat_at, host, status, payload)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(target) DO UPDATE SET
                            heartbeat_at=excluded.heartbeat_at,
                            host=excluded.host,
                            status=excluded.status,
                            payload=excluded.payload
                        """
                    ),
                    (
                        target, merged.get("heartbeat_at"), merged.get("host"),
                        merged.get("status"), raw_payload,
                    ),
                )

        self._run_write("write_remote_agent_snapshot", _write)

    # -----------------------------------------------------------------------
    # Release coordination
    # -----------------------------------------------------------------------

    def publish_deployment_release(
        self,
        deployment_id: str,
        *,
        desired_revision: str,
        desired_release_manifest_sha256: str,
        desired_semantic_contract_sha256: str = "",
        requested_by: str = "",
    ) -> None:
        """Publish the desired immutable release for all deployment agents."""
        now = datetime.now(timezone.utc).isoformat()

        def _write(conn) -> None:
            values = (
                deployment_id, desired_revision, desired_release_manifest_sha256,
                desired_semantic_contract_sha256 or None, "STAGED", now,
                requested_by or None, None, "SYNCING",
                f"Published {desired_revision[:12]}; waiting for release agents",
                desired_revision, now,
            )
            conflict = (
                "ON CONFLICT (deployment_id) DO UPDATE SET"
                if self.db_cfg.backend == "postgres"
                else "ON CONFLICT(deployment_id) DO UPDATE SET"
            )
            conn.execute(
                self._sql(
                    "INSERT INTO deployment_releases "
                    "(deployment_id, desired_revision, desired_release_manifest_sha256, "
                    "desired_semantic_contract_sha256, phase, requested_at, requested_by, error, "
                    "controller_status, controller_message, controller_revision, controller_heartbeat_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    f"{conflict} desired_revision=excluded.desired_revision, "
                    "desired_release_manifest_sha256=excluded.desired_release_manifest_sha256, "
                    "desired_semantic_contract_sha256=excluded.desired_semantic_contract_sha256, "
                    "phase=excluded.phase, requested_at=excluded.requested_at, "
                    "requested_by=excluded.requested_by, error=NULL, "
                    "controller_status=excluded.controller_status, "
                    "controller_message=excluded.controller_message, "
                    "controller_revision=excluded.controller_revision, "
                    "controller_heartbeat_at=excluded.controller_heartbeat_at"
                ),
                values,
            )

        self._run_write("publish_deployment_release", _write)

    def get_deployment_release(self, deployment_id: str) -> dict | None:
        row = self._conn().execute(
            self._sql(
                "SELECT deployment_id, desired_revision, desired_release_manifest_sha256, "
                "desired_semantic_contract_sha256, phase, requested_at, requested_by, error, "
                "controller_status, controller_message, controller_revision, controller_heartbeat_at "
                "FROM deployment_releases WHERE deployment_id=?"
            ),
            (deployment_id,),
        ).fetchone()
        return dict(row) if row else None

    def set_deployment_phase(
        self,
        deployment_id: str,
        *,
        phase: str,
        expected_phases: set[str] | tuple[str, ...] | None = None,
        expected_revision: str | None = None,
        error: str | None = None,
    ) -> bool:
        """CAS the release phase so agents cannot promote stale intent."""
        expected = tuple(str(item) for item in (expected_phases or ()))

        def _write(conn) -> bool:
            predicates = ["deployment_id=?"]
            where_params: list[object] = [deployment_id]
            if expected:
                predicates.append(
                    "phase IN (" + ",".join("?" for _ in expected) + ")"
                )
                where_params.extend(expected)
            if expected_revision:
                predicates.append("desired_revision=?")
                where_params.append(expected_revision)
            result = conn.execute(
                self._sql(
                    "UPDATE deployment_releases SET phase=?, error=? WHERE "
                    + " AND ".join(predicates)
                ),
                [phase, error, *where_params],
            )
            return result.rowcount == 1

        return bool(self._run_write("set_deployment_phase", _write))

    def report_release_controller(
        self,
        deployment_id: str,
        *,
        status: str,
        message: str = "",
        revision: str = "",
        error: str | None = None,
    ) -> bool:
        """Persist the local release controller heartbeat for the dashboard."""
        now = datetime.now(timezone.utc).isoformat()

        def _write(conn) -> bool:
            result = conn.execute(
                self._sql(
                    "UPDATE deployment_releases SET controller_status=?, "
                    "controller_message=?, controller_revision=?, "
                    "controller_heartbeat_at=?, error=? WHERE deployment_id=?"
                ),
                (status, message or None, revision or None, now, error, deployment_id),
            )
            return result.rowcount == 1

        return bool(self._run_write("report_release_controller", _write))

    def write_deployment_node(
        self,
        deployment_id: str,
        node_id: str,
        *,
        target: str,
        host: str = "",
        status: str = "UNKNOWN",
        staged_revision: str = "",
        active_revision: str = "",
        semantic_contract_sha256: str = "",
        execution_profile_sha256: str = "",
        heartbeat_at: str | None = None,
        error: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        heartbeat = heartbeat_at or now
        values = (
            deployment_id, node_id, target, host or None, status,
            staged_revision or None, active_revision or None,
            semantic_contract_sha256 or None, execution_profile_sha256 or None,
            heartbeat, now, error or None,
        )
        columns = (
            "deployment_id, node_id, target, host, status, staged_revision, "
            "active_revision, semantic_contract_sha256, execution_profile_sha256, "
            "heartbeat_at, updated_at, error"
        )
        conflict = (
            "ON CONFLICT (deployment_id, node_id) DO UPDATE SET"
            if self.db_cfg.backend == "postgres"
            else "ON CONFLICT(deployment_id, node_id) DO UPDATE SET"
        )

        def _write(conn) -> None:
            conn.execute(
                self._sql(
                    f"INSERT INTO deployment_nodes ({columns}) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    f"{conflict} target=excluded.target, host=excluded.host, "
                    "status=excluded.status, staged_revision=excluded.staged_revision, "
                    "active_revision=excluded.active_revision, "
                    "semantic_contract_sha256=excluded.semantic_contract_sha256, "
                    "execution_profile_sha256=excluded.execution_profile_sha256, "
                    "heartbeat_at=excluded.heartbeat_at, updated_at=excluded.updated_at, "
                    "error=excluded.error"
                ),
                values,
            )

        self._run_write("write_deployment_node", _write)

    def list_deployment_nodes(self, deployment_id: str) -> list[dict]:
        rows = self._conn().execute(
            self._sql(
                "SELECT deployment_id, node_id, target, host, status, staged_revision, "
                "active_revision, semantic_contract_sha256, execution_profile_sha256, "
                "heartbeat_at, updated_at, error FROM deployment_nodes "
                "WHERE deployment_id=? ORDER BY target, node_id"
            ),
            (deployment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def request_remote_cancel(self, batch_id: str) -> None:
        requested_at = datetime.now(timezone.utc).isoformat()

        def _cancel(conn) -> None:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql(
                    "SELECT batch_id, remote_status, remote_heartbeat_at, "
                    "remote_driver_job_id, remote_work_dir, remote_outdir, "
                    "remote_log_path, remote_payload FROM batches WHERE batch_id=?" + lock
                ),
                (batch_id,),
            ).fetchone()
            payload = dict(self._remote_status_from_row(row) or {"batch_id": batch_id})
            current = str(payload.get("status") or "").upper()
            payload["cancel_requested_at"] = requested_at
            if current not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                payload["status"] = "CANCEL_REQUESTED"
            payload["heartbeat_at"] = requested_at
            updates = {
                "remote_status": payload.get("status"),
                "remote_heartbeat_at": requested_at,
                "remote_driver_job_id": payload.get("driver_job_id"),
                "remote_work_dir": payload.get("remote_work_dir"),
                "remote_outdir": payload.get("remote_outdir"),
                "remote_log_path": payload.get("remote_log_path"),
                "remote_payload": json.dumps(payload, sort_keys=True),
            }
            assignments = ", ".join(f"{key}=?" for key in updates)
            conn.execute(
                self._sql(f"UPDATE batches SET {assignments} WHERE batch_id=?"),
                [*updates.values(), batch_id],
            )

        self._run_write("request_remote_cancel", _cancel)

    def requeue_blocked_remote_batch(self, batch_id: str, *, reason: str) -> int:
        """Retire a stale remote order and return its slides to the queue.

        The old order remains auditable as a failed batch, but its slides are
        made PENDING without charging a scientific retry.  The dispatcher can
        then create a fresh order carrying the current source/profile contract.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _requeue(conn) -> int:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql(
                    "SELECT remote_status, remote_payload FROM batches "
                    "WHERE batch_id=?" + lock
                ),
                (batch_id,),
            ).fetchone()
            current = str(self._row_get(row, "remote_status", 0) or "").upper() if row else ""
            if current not in {"BLOCKED", "CLAIMED", "STARTING"}:
                return 0
            payload = self._decode_remote_payload(
                self._row_get(row, "remote_payload", 1)
            )
            payload["batch_id"] = batch_id
            payload["status"] = current
            payload.update({
                "status": "FAILED",
                "failure_class": "STALE_ORDER_REQUEUED",
                "error": reason[:2000],
                "requeued": True,
                "nextflow_exit": -1,
                "updated_at": now,
                "heartbeat_at": now,
            })
            slides = conn.execute(
                self._sql(
                    "SELECT DISTINCT slide_id FROM slides "
                    "WHERE batch_id=? AND status='DISPATCHED'"
                ),
                (batch_id,),
            ).fetchall()
            count = len(slides)
            conn.execute(
                self._sql(
                    "UPDATE slides SET status='PENDING', batch_id=NULL, "
                    "dispatched_at=NULL, completed_at=NULL, error_msg=NULL "
                    "WHERE batch_id=? AND status='DISPATCHED'"
                ),
                (batch_id,),
            )
            conn.execute(
                self._sql(
                    "UPDATE batches SET status='FAILED', completed_at=?, nextflow_exit=-1, "
                    "remote_status='FAILED', remote_heartbeat_at=?, remote_payload=? "
                    "WHERE batch_id=?"
                ),
                (now, now, json.dumps(payload, sort_keys=True), batch_id),
            )
            self._refresh_slide_rollups(
                conn,
                [self._row_get(item, "slide_id", 0) for item in slides],
            )
            return count

        return int(self._run_write("requeue_blocked_remote_batch", _requeue) or 0)

    def claim_remote_batch(self, target: str, *, claimer_host: str = "") -> dict | None:
        now = datetime.now(timezone.utc).isoformat()

        def _claim(conn):
            if self.db_cfg.backend == "postgres":
                row = conn.execute(
                    self._sql(
                        """
                        SELECT batch_id, remote_payload
                          FROM batches
                         WHERE backend='remote_agent'
                           AND target=?
                           AND COALESCE(remote_status, '')='QUEUED'
                           AND status IN ('RUNNING', 'RECOVERING')
                         ORDER BY dispatched_at ASC NULLS FIRST, batch_id ASC
                         FOR UPDATE SKIP LOCKED
                         LIMIT 1
                        """
                    ),
                    (target,),
                ).fetchone()
            else:
                row = conn.execute(
                    self._sql(
                        """
                        SELECT batch_id, remote_payload
                          FROM batches
                         WHERE backend='remote_agent'
                           AND target=?
                           AND COALESCE(remote_status, '')='QUEUED'
                           AND status IN ('RUNNING', 'RECOVERING')
                         ORDER BY dispatched_at ASC, batch_id ASC
                         LIMIT 1
                        """
                    ),
                    (target,),
                ).fetchone()
            if not row:
                return None
            batch_id = self._row_get(row, "batch_id", 0)
            payload = self._decode_remote_payload(self._row_get(row, "remote_payload", 1))
            payload["batch_id"] = batch_id
            payload["claimed_at"] = now
            if claimer_host:
                payload["claimer_host"] = claimer_host
            conn.execute(
                self._sql(
                    """
                    UPDATE batches
                       SET remote_status=?, remote_heartbeat_at=?, remote_payload=?
                     WHERE batch_id=?
                    """
                ),
                ("CLAIMED", now, json.dumps(payload, sort_keys=True), batch_id),
            )
            return payload

        return self._run_write("claim_remote_batch", _claim)

    def withdraw_remote_order(self, batch_id: str) -> bool:
        """Cancel a remote order the agent has not claimed; False if it may be running.

        True when no order is live: none was written, it was already cancelled,
        or it was still QUEUED (now CANCELLED, atomically with the agent's claim).
        """
        def _withdraw(conn) -> bool:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql("SELECT remote_status FROM batches WHERE batch_id=?" + lock),
                (batch_id,),
            ).fetchone()
            status = str(self._row_get(row, "remote_status", 0) or "").upper() if row else ""
            if status == "QUEUED":
                conn.execute(
                    self._sql("UPDATE batches SET remote_status='CANCELLED', remote_heartbeat_at=? "
                              "WHERE batch_id=?"),
                    (datetime.now(timezone.utc).isoformat(), batch_id),
                )
                return True
            return status in {"", "CANCELLED"}

        return bool(self._run_write("withdraw_remote_order", _withdraw))

    def claim_remote_finalization(
        self,
        batch_id: str,
        *,
        stale_after_seconds: int,
        owner: str = "",
        lease_name: str = REMOTE_FINALIZATION_LEASE_NAME,
    ) -> bool:
        """Atomically acquire success finalization for a completed remote driver.

        A stale ``HOOKS_RUNNING`` lease may be reclaimed after an agent crash.
        Successfully completed hooks are independently idempotent and are
        skipped by the shared hook runner during that replay.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        stale_before = (now - timedelta(seconds=max(1, stale_after_seconds))).isoformat()

        lease_owner = owner or f"remote-finalizer:{batch_id}"
        if not self.claim_finalization_lease(
            batch_id,
            owner=lease_owner,
            ttl_seconds=max(1, stale_after_seconds),
            lease_name=lease_name,
        ):
            return False

        def _claim(conn) -> bool:
            cur = conn.execute(
                self._sql(
                    """
                    UPDATE batches
                       SET remote_status='HOOKS_RUNNING', remote_heartbeat_at=?,
                           finalization_stage=COALESCE(NULLIF(finalization_stage, 'PENDING'), 'PREPARE'),
                           finalization_owner=?, finalization_heartbeat_at=?,
                           finalization_retry_at=NULL, finalization_error=NULL
                     WHERE batch_id=?
                       AND (
                            COALESCE(remote_status, '') IN
                                ('CLAIMED','STARTING','SUBMITTED','RUNNING','CANCEL_REQUESTED')
                            OR (
                                remote_status='HOOKS_RUNNING'
                                AND (remote_heartbeat_at IS NULL OR remote_heartbeat_at < ?)
                            )
                            OR (
                                remote_status IN ('HOOKS_RETRY', 'FINALIZATION_QUEUED')
                                AND (finalization_retry_at IS NULL OR finalization_retry_at <= ?)
                            )
                       )
                    """
                ),
                (now_iso, owner or "", now_iso, batch_id, stale_before, now_iso),
            )
            return cur.rowcount == 1

        try:
            claimed = bool(self._run_write("claim_remote_finalization", _claim))
        except Exception:
            self.release_finalization_lease(
                batch_id, owner=lease_owner, lease_name=lease_name
            )
            raise
        if not claimed:
            self.release_finalization_lease(
                batch_id, owner=lease_owner, lease_name=lease_name
            )
        return claimed

    def claim_finalization_lease(
        self,
        batch_id: str,
        *,
        owner: str,
        ttl_seconds: int = 300,
        lease_name: str = _FINALIZATION_LEASE_NAME,
    ) -> bool:
        """Atomically acquire a backend-scoped finalization lease."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires_iso = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat()

        def _claim(conn) -> bool:
            if self.db_cfg.backend == "postgres":
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO coordination_leases
                            (lease_name, owner, batch_id, acquired_at, heartbeat_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT (lease_name) DO NOTHING
                        """
                    ),
                    (lease_name, owner, batch_id, now_iso, now_iso, expires_iso),
                )
            else:
                conn.execute(
                    self._sql(
                        """
                        INSERT OR IGNORE INTO coordination_leases
                            (lease_name, owner, batch_id, acquired_at, heartbeat_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """
                    ),
                    (lease_name, owner, batch_id, now_iso, now_iso, expires_iso),
                )
            cur = conn.execute(
                self._sql(
                    """
                    UPDATE coordination_leases
                       SET owner=?, batch_id=?, acquired_at=CASE
                               WHEN owner=? THEN acquired_at ELSE ? END,
                           heartbeat_at=?, expires_at=?
                     WHERE lease_name=?
                       AND ((owner=? AND batch_id=?) OR expires_at <= ?)
                    """
                ),
                (
                    owner, batch_id, owner, now_iso, now_iso, expires_iso,
                    lease_name, owner, batch_id, now_iso,
                ),
            )
            return cur.rowcount == 1

        return bool(self._run_write("claim_finalization_lease", _claim))

    def claim_finalization_slot(
        self,
        batch_id: str,
        *,
        lane: str,
        max_slots: int,
        owner: str,
        ttl_seconds: int = 300,
    ) -> str | None:
        """Atomically claim one bounded finalization slot for a batch.

        Slots are durable PostgreSQL/SQLite leases. A transaction-scoped
        PostgreSQL advisory lock serializes slot selection across dispatcher
        processes; SQLite's write transaction provides the equivalent guard.
        """
        count = max(1, int(max_slots or 1))
        lease_names = [finalization_slot_name(lane, index) for index in range(count)]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires_iso = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat()

        def _claim(conn) -> str | None:
            self._lock_finalization_lane(conn, lane)

            placeholders = self._placeholders(len(lease_names))
            rows = conn.execute(
                self._sql(
                    f"SELECT lease_name, owner, batch_id, expires_at "
                    f"FROM coordination_leases WHERE lease_name IN ({placeholders})"
                ),
                lease_names,
            ).fetchall()
            current = {
                str(self._row_get(row, "lease_name", 0)): row for row in rows
            }
            for lease_name in lease_names:
                row = current.get(lease_name)
                if not row:
                    continue
                row_batch = self._row_get(row, "batch_id", 2)
                row_owner = self._row_get(row, "owner", 1)
                expires = str(self._row_get(row, "expires_at", 3) or "")
                try:
                    live = datetime.fromisoformat(expires).timestamp() > time.time()
                except (TypeError, ValueError, OSError):
                    live = False
                if live and row_batch == batch_id:
                    return lease_name if row_owner == owner else None

            for lease_name in lease_names:
                row = current.get(lease_name)
                if row:
                    expires = str(self._row_get(row, "expires_at", 3) or "")
                    try:
                        live = datetime.fromisoformat(expires).timestamp() > time.time()
                    except (TypeError, ValueError, OSError):
                        live = False
                    if live:
                        continue
                    cur = conn.execute(
                        self._sql(
                            "UPDATE coordination_leases SET owner=?, batch_id=?, "
                            "acquired_at=?, heartbeat_at=?, expires_at=? "
                            "WHERE lease_name=? AND expires_at<=?"
                        ),
                        (owner, batch_id, now_iso, now_iso, expires_iso, lease_name, now_iso),
                    )
                    if cur.rowcount == 1:
                        return lease_name
                    continue
                conn.execute(
                    self._sql(
                        "INSERT INTO coordination_leases "
                        "(lease_name, owner, batch_id, acquired_at, heartbeat_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)"
                    ),
                    (lease_name, owner, batch_id, now_iso, now_iso, expires_iso),
                )
                return lease_name
            return None

        return self._run_write("claim_finalization_slot", _claim)

    @staticmethod
    def _finalization_lane_lock_key(lane: str) -> int:
        key_bytes = hashlib.sha256(
            f"finalization-slots:{lane}".encode("utf-8")
        ).digest()[:8]
        return int.from_bytes(key_bytes, byteorder="big", signed=True)

    def _lock_finalization_lane(self, conn, lane: str) -> None:
        """Serialize every mutation of a bounded finalization lane."""
        if self.db_cfg.backend == "postgres":
            conn.execute(
                self._sql("SELECT pg_advisory_xact_lock(?)"),
                (self._finalization_lane_lock_key(lane),),
            )

    def heartbeat_finalization_lease(
        self,
        batch_id: str,
        *,
        owner: str,
        ttl_seconds: int = 300,
        lease_name: str = _FINALIZATION_LEASE_NAME,
    ) -> bool:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires_iso = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat()

        def _heartbeat(conn) -> bool:
            cur = conn.execute(
                self._sql(
                    """
                    UPDATE coordination_leases
                       SET heartbeat_at=?, expires_at=?
                     WHERE lease_name=? AND owner=? AND batch_id=?
                    """
                ),
                (now_iso, expires_iso, lease_name, owner, batch_id),
            )
            return cur.rowcount == 1

        return bool(self._run_write("heartbeat_finalization_lease", _heartbeat))

    def release_finalization_lease(
        self,
        batch_id: str,
        *,
        owner: str,
        lease_name: str = _FINALIZATION_LEASE_NAME,
    ) -> bool:
        def _release(conn) -> bool:
            cur = conn.execute(
                self._sql(
                    "DELETE FROM coordination_leases "
                    "WHERE lease_name=? AND owner=? AND batch_id=?"
                ),
                (lease_name, owner, batch_id),
            )
            return cur.rowcount == 1

        return bool(self._run_write("release_finalization_lease", _release))

    def finalization_lease_status(
        self, lease_name: str = _FINALIZATION_LEASE_NAME
    ) -> dict | None:
        row = self._conn().execute(
            self._sql(
                "SELECT lease_name, owner, batch_id, acquired_at, heartbeat_at, expires_at "
                "FROM coordination_leases WHERE lease_name=?"
            ),
            (lease_name,),
        ).fetchone()
        return dict(row) if row else None

    def owns_finalization_lease(
        self,
        batch_id: str,
        *,
        owner: str,
        lease_name: str = _FINALIZATION_LEASE_NAME,
    ) -> bool:
        """Return whether this exact worker owns a currently live lease."""
        row = self._conn().execute(
            self._sql(
                "SELECT expires_at FROM coordination_leases "
                "WHERE lease_name=? AND owner=? AND batch_id=?"
            ),
            (lease_name, owner, batch_id),
        ).fetchone()
        if not row:
            return False
        try:
            return (
                datetime.fromisoformat(
                    str(self._row_get(row, "expires_at", 0) or "")
                ).timestamp()
                > time.time()
            )
        except (TypeError, ValueError, OSError):
            return False

    def finalization_coordination_status(self) -> dict:
        row = self._conn().execute(
            self._sql(
                """
                SELECT COUNT(*) AS queued
                  FROM batches
                 WHERE status='FINALIZING'
                    OR remote_status='FINALIZATION_QUEUED'
                """
            )
        ).fetchone()
        def _lane_limit(lane: str, target: str = "") -> int:
            resolver = getattr(self.cfg, "effective_max_concurrent_finalizations", None)
            if resolver is not None:
                return max(1, int(resolver(lane=lane, target=target)))
            return max(1, int(getattr(self.cfg, "max_concurrent_finalizations", 1) or 1))

        max_slots = _lane_limit("local")
        # Every remote target has its own slots: report them all under "remote".
        remote_targets = [
            str(t.name) for t in (getattr(self.cfg, "dispatch_targets", None) or [])
            if str(getattr(t, "backend", "remote_agent") or "remote_agent") == "remote_agent" and t.name
        ] or [getattr(self.cfg, "default_dispatch_target", "") or ""]
        remote_slots = [
            (f"remote:{target}" if target else "remote", index)
            for target in remote_targets
            for index in range(_lane_limit("remote", target))
        ]
        lane_limits = {"local": max_slots, "remote": len(remote_slots)}
        lanes = {
            "local": [
                lease for index in range(max_slots)
                if (lease := self.finalization_lease_status(finalization_slot_name("local", index)))
            ],
            "remote": [
                lease for lane, index in remote_slots
                if (lease := self.finalization_lease_status(finalization_slot_name(lane, index)))
            ],
        }
        active_slots = [lease for slots in lanes.values() for lease in slots]
        return {
            # Keep ``lease`` for the existing dashboard contract, but point it
            # at an actual running finalizer when one exists. The old base
            # lease remains the shared DB-sync lease.
            "lease": active_slots[0] if active_slots else self.finalization_lease_status(),
            "lanes": lanes,
            "active_slots": len(active_slots),
            "max_slots_per_lane": max_slots,
            "max_slots_by_lane": lane_limits,
            "queued_batches": int(self._row_get(row, "queued", 0) or 0),
        }

    def reconcile_finalization_slots(
        self,
        *,
        lane: str,
        max_slots: int,
        owner: str,
        active_batch_ids: set[str] | list[str] | tuple[str, ...] = (),
    ) -> dict:
        """Reconcile durable finalization slots with one agent's live workers.

        Process-local futures are only an observation of ownership. PostgreSQL
        leases remain authoritative, but a lease owned by this exact process
        with no live future is unambiguously orphaned and can be returned to
        the durable retry queue. Terminal batches and missing batch rows never
        need to retain a lease. Foreign, unexpired leases are reported and
        left alone so a healthy agent is not interrupted.
        """
        names = [
            finalization_slot_name(lane, index)
            for index in range(max(1, int(max_slots or 1)))
        ]
        active = {str(batch_id) for batch_id in active_batch_ids if batch_id}
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        def _reconcile(conn) -> dict:
            self._lock_finalization_lane(conn, lane)
            placeholders = self._placeholders(len(names))
            rows = conn.execute(
                self._sql(
                    f"SELECT lease_name, owner, batch_id, expires_at "
                    f"FROM coordination_leases WHERE lease_name IN ({placeholders})"
                ),
                names,
            ).fetchall()
            result = {
                "lane": lane,
                "owner": owner,
                "durable_slots": len(rows),
                "process_active": len(active),
                "repaired": [],
                "terminal_released": [],
                "expired": [],
                "foreign_live": [],
                "workers_without_lease": [],
                "unresolved_orphans": [],
            }
            terminal_statuses = {"SUCCEEDED", "FAILED", "CANCELLED"}
            active_statuses = {
                "HOOKS_RUNNING", "HOOKS_RETRY", "FINALIZATION_QUEUED",
            }
            for row in rows:
                lease_name = str(self._row_get(row, "lease_name", 0) or "")
                lease_owner = str(self._row_get(row, "owner", 1) or "")
                batch_id = str(self._row_get(row, "batch_id", 2) or "")
                expires_at = str(self._row_get(row, "expires_at", 3) or "")
                try:
                    expired = datetime.fromisoformat(expires_at).timestamp() <= time.time()
                except (TypeError, ValueError, OSError):
                    expired = True
                batch = conn.execute(
                    self._sql("SELECT remote_status FROM batches WHERE batch_id=?"),
                    (batch_id,),
                ).fetchone()
                status = str(self._row_get(batch, "remote_status", 0) or "").upper() if batch else ""
                if not batch or status in terminal_statuses:
                    conn.execute(
                        self._sql(
                            "DELETE FROM coordination_leases "
                            "WHERE lease_name=? AND owner=? AND batch_id=?"
                        ),
                        (lease_name, lease_owner, batch_id),
                    )
                    result["terminal_released"].append({
                        "lease_name": lease_name,
                        "batch_id": batch_id,
                        "status": status or "MISSING",
                    })
                    continue
                if expired:
                    if status in active_statuses:
                        conn.execute(
                            self._sql(
                                "UPDATE batches SET remote_status='HOOKS_RETRY', "
                                "remote_heartbeat_at=?, finalization_retry_at=?, "
                                "finalization_error=? WHERE batch_id=?"
                            ),
                            (
                                now_iso,
                                now_iso,
                                "finalization lease expired; recovered automatically",
                                batch_id,
                            ),
                        )
                    deleted = conn.execute(
                        self._sql(
                            "DELETE FROM coordination_leases "
                            "WHERE lease_name=? AND owner=? AND batch_id=?"
                        ),
                        (lease_name, lease_owner, batch_id),
                    )
                    item = {
                        "lease_name": lease_name,
                        "batch_id": batch_id,
                        "status": status,
                        "owner": lease_owner,
                    }
                    result["expired"].append(item)
                    if deleted.rowcount:
                        result["repaired"].append(item)
                    else:
                        result["unresolved_orphans"].append(item)
                    continue
                if lease_owner == owner and batch_id not in active:
                    if status in active_statuses:
                        conn.execute(
                            self._sql(
                                "UPDATE batches SET remote_status='HOOKS_RETRY', "
                                "remote_heartbeat_at=?, finalization_retry_at=?, "
                                "finalization_error=? WHERE batch_id=?"
                            ),
                            (
                                now_iso,
                                now_iso,
                                "finalization lease had no live worker; recovered automatically",
                                batch_id,
                            ),
                        )
                    deleted = conn.execute(
                        self._sql(
                            "DELETE FROM coordination_leases "
                            "WHERE lease_name=? AND owner=? AND batch_id=?"
                        ),
                        (lease_name, lease_owner, batch_id),
                    )
                    item = {
                        "lease_name": lease_name,
                        "batch_id": batch_id,
                        "owner": lease_owner,
                        "status": status,
                    }
                    if deleted.rowcount:
                        result["repaired"].append(item)
                    else:
                        result["unresolved_orphans"].append(item)
                elif lease_owner != owner:
                    result["foreign_live"].append({
                        "lease_name": lease_name,
                        "batch_id": batch_id,
                        "owner": lease_owner,
                    })
            remaining = conn.execute(
                self._sql(
                    f"SELECT owner, batch_id, expires_at FROM coordination_leases "
                    f"WHERE lease_name IN ({placeholders})"
                ),
                names,
            ).fetchall()
            owned_live = set()
            for row in remaining:
                lease_owner = str(self._row_get(row, "owner", 0) or "")
                batch_id = str(self._row_get(row, "batch_id", 1) or "")
                expires_at = str(self._row_get(row, "expires_at", 2) or "")
                try:
                    live = datetime.fromisoformat(expires_at).timestamp() > time.time()
                except (TypeError, ValueError, OSError):
                    live = False
                if live and lease_owner == owner:
                    owned_live.add(batch_id)
            result["durable_slots"] = len(remaining)
            result["owned_live"] = sorted(owned_live)
            result["workers_without_lease"] = sorted(active - owned_live)
            if result["workers_without_lease"] or result["unresolved_orphans"]:
                result["health_state"] = "DEGRADED"
            elif result["foreign_live"] or result["expired"]:
                result["health_state"] = "RECOVERING"
            else:
                result["health_state"] = "HEALTHY"
            return result

        return dict(self._run_write("reconcile_finalization_slots", _reconcile) or {})

    def remote_target_snapshot(self, release_sync: dict | None = None) -> list[dict]:
        """Build remote health from one state-store connection for the dashboard."""
        if self.cfg is None:
            return []
        targets = getattr(self.cfg, "dispatch_targets", []) or []
        if not targets:
            return []
        agents = {
            str(self._row_get(row, "target", 0)): self._decode_remote_payload(
                self._row_get(row, "payload", 1)
            )
            for row in self._conn().execute(
                self._sql("SELECT target, payload FROM remote_agents")
            ).fetchall()
        }
        queue_rows = self._conn().execute(
            self._sql(
                """
                SELECT target, remote_status, COUNT(*) AS n
                  FROM batches
                 WHERE backend='remote_agent'
                   AND COALESCE(remote_status, '') NOT IN ('', 'SUCCEEDED', 'FAILED', 'CANCELLED')
                 GROUP BY target, remote_status
                """
            )
        ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        for row in queue_rows:
            target_name = str(self._row_get(row, "target", 0) or "")
            counts.setdefault(target_name, {})[str(self._row_get(row, "remote_status", 1) or "")] = int(
                self._row_get(row, "n", 2) or 0
            )
        now = time.time()
        release_targets = {
            str(item.get("name") or ""): item
            for item in ((release_sync or self.release_sync_snapshot()).get("targets") or [])
        }
        result = []
        for target in targets:
            agent = agents.get(target.name) or {}
            heartbeat = agent.get("heartbeat_at")
            age_s = None
            state = "unknown"
            if heartbeat:
                try:
                    age_s = max(0, int(now - datetime.fromisoformat(str(heartbeat)).timestamp()))
                    state = "alive" if age_s <= target.agent_heartbeat_timeout_seconds else "stale"
                except (TypeError, ValueError, OSError):
                    state = "unknown"
            target_counts = counts.get(target.name, {})
            result.append({
                "name": target.name,
                "backend": target.backend,
                "login_host": target.login_host,
                "state": state,
                "agent_heartbeat_at": heartbeat,
                "agent_heartbeat_age_s": age_s,
                "queue_depth": target_counts.get("QUEUED", 0),
                "active_batches": sum(target_counts.values()),
                "slurm_running": int((agent.get("slurm") or {}).get("running", 0) or 0),
                "slurm_pending": int((agent.get("slurm") or {}).get("pending", 0) or 0),
                "slurm_completing": int((agent.get("slurm") or {}).get("completing", 0) or 0),
                "slurm_total": int((agent.get("slurm") or {}).get("total", 0) or 0),
                "slurm_nodes": list((agent.get("slurm") or {}).get("nodes") or []),
                "slurm_error": (agent.get("slurm") or {}).get("error"),
                "coordination": dict(agent.get("coordination") or {}),
                "hook_executor_health": dict(agent.get("hook_executor_health") or {}),
                "finalization": dict(agent.get("finalization") or {}),
                "release": dict(release_targets.get(target.name) or {}),
                "status_source": "dispatcher_snapshot",
            })
        return result

    def _lock_remote_finalization_lease(
        self, conn, batch_id: str, *, owner: str, lease_name: str
    ) -> bool:
        """Lock ``owner``'s finalization lease row; False if it no longer holds it.

        Callers take this lock before the batches row, the same order as
        ``heartbeat_remote_finalization``, so the two cannot deadlock.
        """
        lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
        lease = conn.execute(
            self._sql(
                "SELECT 1 FROM coordination_leases "
                "WHERE lease_name=? AND owner=? AND batch_id=?" + lock
            ),
            (lease_name, owner, batch_id),
        ).fetchone()
        return lease is not None

    def mark_remote_finalization_retry(
        self,
        batch_id: str,
        *,
        error: str = "",
        delay_seconds: int = 60,
        owner: str = "",
        lease_name: str = REMOTE_FINALIZATION_LEASE_NAME,
    ) -> bool:
        """Return a claimed finalization to a durable retryable state.

        With ``owner`` the transition applies only while ``owner`` still holds
        the batch's finalization lease and ``finalization_owner`` (as
        ``heartbeat_remote_finalization`` requires), so a worker that lost
        its lease cannot demote a batch another agent is now finalizing.
        Returns whether the transition was applied.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        retry_at = (now + timedelta(seconds=max(1, delay_seconds))).isoformat()

        def _retry(conn) -> bool:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            owns_lease = bool(owner) and self._lock_remote_finalization_lease(
                conn, batch_id, owner=owner, lease_name=lease_name
            )
            row = conn.execute(
                self._sql(
                    "SELECT remote_status, finalization_owner FROM batches WHERE batch_id=?"
                    + lock
                ),
                (batch_id,),
            ).fetchone()
            current = str(self._row_get(row, "remote_status", 0) or "").upper() if row else ""
            if current not in {"HOOKS_RUNNING", "HOOKS_RETRY"}:
                return False
            if owner and (
                not owns_lease
                or str(self._row_get(row, "finalization_owner", 1) or "") != owner
            ):
                log.warning(
                    "Batch %s: not marking HOOKS_RETRY; %s no longer owns finalization",
                    batch_id, owner,
                )
                return False
            conn.execute(
                self._sql(
                    "UPDATE batches SET remote_status='HOOKS_RETRY', "
                    "remote_heartbeat_at=?, finalization_retry_at=?, "
                    "finalization_heartbeat_at=?, finalization_error=? WHERE batch_id=?"
                ),
                (now_iso, retry_at, now_iso, error[:2000], batch_id),
            )
            return True

        return bool(self._run_write("mark_remote_finalization_retry", _retry))

    def mark_remote_finalization_blocked(
        self,
        batch_id: str,
        *,
        error: str = "",
        payload: dict | None = None,
        owner: str = "",
        lease_name: str = REMOTE_FINALIZATION_LEASE_NAME,
    ) -> bool:
        """Quarantine a batch whose required hooks need operator intervention.

        ``owner`` fences the transition exactly as in
        ``mark_remote_finalization_retry``.  Returns whether it was applied.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _block(conn) -> bool:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            owns_lease = bool(owner) and self._lock_remote_finalization_lease(
                conn, batch_id, owner=owner, lease_name=lease_name
            )
            row = conn.execute(
                self._sql(
                    "SELECT remote_status, remote_payload, finalization_owner FROM batches "
                    "WHERE batch_id=?" + lock
                ),
                (batch_id,),
            ).fetchone()
            current = str(self._row_get(row, "remote_status", 0) or "").upper() if row else ""
            if current not in {"HOOKS_RUNNING", "HOOKS_RETRY", "FINALIZATION_QUEUED"}:
                return False
            if owner and (
                not owns_lease
                or str(self._row_get(row, "finalization_owner", 2) or "") != owner
            ):
                log.warning(
                    "Batch %s: not marking HOOKS_BLOCKED; %s no longer owns finalization",
                    batch_id, owner,
                )
                return False
            merged = self._decode_remote_payload(
                self._row_get(row, "remote_payload", 1)
            )
            merged.update(dict(payload or {}))
            merged.update({
                "batch_id": batch_id,
                "status": "HOOKS_BLOCKED",
                "heartbeat_at": now,
                "updated_at": now,
            })
            cur = conn.execute(
                self._sql(
                    "UPDATE batches SET remote_status='HOOKS_BLOCKED', "
                    "remote_heartbeat_at=?, remote_payload=?, "
                    "finalization_retry_at=NULL, finalization_heartbeat_at=?, "
                    "finalization_error=?, finalization_stage='BLOCKED' "
                    "WHERE batch_id=?"
                ),
                (now, json.dumps(merged, sort_keys=True), now, error[:2000], batch_id),
            )
            return cur.rowcount == 1

        return bool(self._run_write("mark_remote_finalization_blocked", _block))


    def heartbeat_remote_finalization(
        self,
        batch_id: str,
        *,
        owner: str = "",
        lease_ttl_seconds: int = 300,
        lease_name: str = REMOTE_FINALIZATION_LEASE_NAME,
    ) -> bool:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_expires = (
            now_dt + timedelta(seconds=max(1, lease_ttl_seconds))
        ).isoformat()

        class _LeaseOwnershipLost(Exception):
            pass

        def _heartbeat(conn) -> bool:
            if owner:
                lease_cur = conn.execute(
                    self._sql(
                        "UPDATE coordination_leases "
                        "SET heartbeat_at=?, expires_at=? "
                        "WHERE lease_name=? AND owner=? AND batch_id=?"
                    ),
                    (now, lease_expires, lease_name, owner, batch_id),
                )
                if lease_cur.rowcount != 1:
                    raise _LeaseOwnershipLost
            owner_predicate = "finalization_owner=?" if owner else "1=1"
            cur = conn.execute(
                self._sql(
                    "UPDATE batches SET remote_heartbeat_at=?, finalization_heartbeat_at=? "
                    "WHERE batch_id=? AND remote_status IN ('HOOKS_RUNNING','HOOKS_RETRY') "
                    f"AND {owner_predicate}"
                ),
                (now, now, batch_id, owner) if owner else (now, now, batch_id),
            )
            if cur.rowcount != 1 and owner:
                raise _LeaseOwnershipLost
            return cur.rowcount == 1

        try:
            return bool(self._run_write("heartbeat_remote_finalization", _heartbeat))
        except _LeaseOwnershipLost:
            return False

    def get_remote_batches_for_target(self, target: str) -> list[dict]:
        rows = self._conn().execute(
            self._sql(
                """
                SELECT batch_id, remote_status, remote_heartbeat_at,
                       remote_driver_job_id, remote_work_dir, remote_outdir,
                       remote_log_path, remote_payload, finalization_retry_at
                  FROM batches
                 WHERE backend='remote_agent'
                   AND target=?
                   AND COALESCE(remote_status, '') NOT IN ('', 'SUCCEEDED', 'FAILED', 'CANCELLED')
                 ORDER BY CASE remote_status
                              WHEN 'FINALIZATION_QUEUED' THEN 0
                              WHEN 'HOOKS_RETRY' THEN 1
                              WHEN 'HOOKS_RUNNING' THEN 2
                              WHEN 'RUNNING' THEN 3
                              WHEN 'QUEUED' THEN 4
                              ELSE 5
                          END,
                          CASE
                              WHEN remote_status IN ('FINALIZATION_QUEUED', 'HOOKS_RETRY')
                              THEN COALESCE(finalization_retry_at, dispatched_at)
                              ELSE dispatched_at
                          END ASC,
                          dispatched_at ASC,
                          batch_id ASC
                """
            ),
            (target,),
        ).fetchall()
        return [payload for row in rows if (payload := self._remote_status_from_row(row))]

    def remote_finalization_queue_health(self, target: str) -> dict:
        """Return age and count for remote batches waiting on finalization."""
        row = self._conn().execute(
            self._sql(
                """
                SELECT COUNT(*) AS queued,
                       MIN(COALESCE(finalization_retry_at, dispatched_at)) AS oldest
                  FROM batches
                 WHERE backend='remote_agent'
                   AND target=?
                   AND remote_status IN ('FINALIZATION_QUEUED', 'HOOKS_RETRY')
                """
            ),
            (target,),
        ).fetchone()
        queued = int(self._row_get(row, "queued", 0) or 0)
        oldest = self._row_get(row, "oldest", 1)
        age_s = None
        if oldest:
            try:
                age_s = max(
                    0,
                    int(
                        (
                            datetime.now(timezone.utc)
                            - datetime.fromisoformat(str(oldest).replace("Z", "+00:00"))
                        ).total_seconds()
                    ),
                )
            except (TypeError, ValueError, OSError):
                age_s = None
        return {
            "queued": queued,
            "oldest_queued_at": oldest,
            "oldest_queued_age_s": age_s,
        }

    def get_remote_execution_cleanup_batches(self, target: str) -> list[dict]:
        rows = self._conn().execute(
            self._sql(
                """
                SELECT batch_id, remote_status, remote_heartbeat_at,
                       remote_driver_job_id, remote_work_dir, remote_outdir,
                       remote_log_path, remote_payload, finalization_retry_at
                  FROM batches
                 WHERE backend='remote_agent' AND target=?
                   AND driver_completed_at IS NOT NULL
                   AND execution_cleanup_at IS NULL
                   AND remote_status IN ('SUCCEEDED','FAILED','CANCELLED')
                 ORDER BY dispatched_at ASC, batch_id ASC
                """
            ),
            (target,),
        ).fetchall()
        return [payload for row in rows if (payload := self._remote_status_from_row(row))]

    def get_batch(self, batch_id: str) -> dict | None:
        row = self._conn().execute(
            self._sql("SELECT * FROM batches WHERE batch_id=?"), (batch_id,)
        ).fetchone()
        return dict(row) if row else None

    def record_batch_manifest(self, batch_id: str, manifest_path: str):
        self._run_write(
            "record_batch_manifest",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET manifest_path=? WHERE batch_id=?"), (manifest_path, batch_id)
            ),
        )

    def get_all_manifest_paths(self) -> list:
        rows = self._conn().execute(
            "SELECT manifest_path FROM batches WHERE manifest_path IS NOT NULL"
        ).fetchall()
        return [r["manifest_path"] for r in rows]

    def get_active_batches(self) -> list:
        rows = self._conn().execute(
            """
            SELECT *
              FROM batches
             WHERE status IN ('RUNNING','RECOVERING','FINALIZING')
                OR (
                    backend='remote_agent'
                    AND COALESCE(remote_status, '') NOT IN ('', 'SUCCEEDED', 'FAILED', 'CANCELLED')
                )
             ORDER BY COALESCE(remote_heartbeat_at, dispatched_at) DESC, dispatched_at DESC, batch_id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def remote_target_release_ready(
        self, target: str, *, deployment_id: str | None = None
    ) -> tuple[bool, str]:
        """Return whether a remote target may receive fresh work.

        A published release is an opt-in gate.  Deployments that do not use
        the release-coordination tables remain backwards compatible, but once
        a desired release exists every registered node for the target must
        report that exact revision as active.  This prevents a dispatcher
        running a newer contract from repeatedly creating orders that an old
        remote checkout will reject as source drift.
        """
        release_id = (
            deployment_id
            or getattr(self.cfg, "release_deployment_id", "")
            or os.environ.get("MUSSEL_RELEASE_DEPLOYMENT_ID", "default")
        ).strip()
        release = self.get_deployment_release(release_id)
        managed = bool(getattr(self.cfg, "auto_sync_release", False))
        if not release:
            if managed:
                return False, f"deployment {release_id} is waiting for its first managed release"
            return True, ""

        desired = str(release.get("desired_revision") or "").strip()
        if not desired:
            return False, f"deployment {release_id} has no desired revision"

        if managed:
            controller_status = str(release.get("controller_status") or "").upper()
            if controller_status in {"BLOCKED_LOCAL", "ERROR"}:
                return False, str(
                    release.get("controller_message")
                    or release.get("error")
                    or f"release controller is {controller_status.lower()}"
                )
            phase = str(release.get("phase") or "").upper()
            if phase != "ACTIVE":
                return False, f"release {desired} is {phase.lower() or 'syncing'}"

        nodes = [
            node for node in self.list_deployment_nodes(release_id)
            if str(node.get("target") or "").strip() == str(target).strip()
        ]
        if not nodes:
            return False, f"deployment {release_id} has no registered node for target {target}"

        not_active = [
            f"{node.get('node_id')}: active={node.get('active_revision') or '<none>'}"
            for node in nodes
            if str(node.get("active_revision") or "").strip() != desired
        ]
        if not_active:
            return False, (
                f"target {target} is not on desired release {desired}: "
                + ", ".join(not_active)
            )
        if managed:
            target_cfg = self.cfg.get_dispatch_target(target) if self.cfg else None
            timeout = max(
                1,
                int(getattr(target_cfg, "agent_heartbeat_timeout_seconds", 300) or 300),
            )
            now = datetime.now(timezone.utc)
            stale_nodes = []
            for node in nodes:
                heartbeat = node.get("heartbeat_at")
                try:
                    age = (now - datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))).total_seconds()
                except (TypeError, ValueError, OSError):
                    age = timeout + 1
                if age > timeout:
                    stale_nodes.append(str(node.get("node_id") or target))
            if stale_nodes:
                return False, "release agent heartbeat stale: " + ", ".join(stale_nodes)
            errors = [
                str(node.get("error")) for node in nodes
                if str(node.get("status") or "").upper() == "ERROR" and node.get("error")
            ]
            if errors:
                return False, "; ".join(errors)
            agent = self.read_remote_agent(target) or {}
            agent_heartbeat = agent.get("heartbeat_at")
            try:
                agent_age = (
                    now - datetime.fromisoformat(str(agent_heartbeat).replace("Z", "+00:00"))
                ).total_seconds()
            except (TypeError, ValueError, OSError):
                agent_age = timeout + 1
            if agent_age > timeout:
                return False, f"target {target} worker heartbeat is stale"
            loaded_revision = str(agent.get("agent_revision") or "").strip()
            if loaded_revision != desired:
                return False, (
                    f"target {target} active link is {desired[:12]} but running agent loaded "
                    f"{loaded_revision[:12] if loaded_revision else '<unknown>'}"
                )
        return True, ""

    def release_sync_snapshot(self, deployment_id: str | None = None) -> dict:
        """Return dashboard-ready release coordination and dispatch-gate state."""
        release_id = (
            deployment_id
            or getattr(self.cfg, "release_deployment_id", "")
            or os.environ.get("MUSSEL_RELEASE_DEPLOYMENT_ID", "default")
        ).strip()
        release = self.get_deployment_release(release_id)
        managed = bool(getattr(self.cfg, "auto_sync_release", False))
        if not release:
            return {
                "deployment_id": release_id,
                "state": "SYNCING" if managed else "UNMANAGED",
                "managed": managed,
                "targets": [],
                "blocked_targets": [],
                "message": (
                    "Waiting for the local release controller to publish its first release"
                    if managed else ""
                ),
            }

        desired = str(release.get("desired_revision") or "")
        nodes = self.list_deployment_nodes(release_id)
        target_names = [
            target.name for target in (getattr(self.cfg, "dispatch_targets", []) or [])
        ]
        targets = []
        blocked_targets = []
        for target_name in target_names:
            target_nodes = [
                node for node in nodes if str(node.get("target") or "") == target_name
            ]
            agent = self.read_remote_agent(target_name) or {}
            ready, reason = self.remote_target_release_ready(
                target_name, deployment_id=release_id
            )
            if not ready:
                blocked_targets.append(target_name)
            targets.append({
                "name": target_name,
                "dispatch_ready": ready,
                "dispatch_block_reason": reason,
                "node_status": ",".join(
                    sorted({str(node.get("status") or "UNKNOWN") for node in target_nodes})
                ) or "MISSING",
                "staged_revision": next((
                    str(node.get("staged_revision") or "") for node in target_nodes
                    if node.get("staged_revision")
                ), ""),
                "active_revision": next((
                    str(node.get("active_revision") or "") for node in target_nodes
                    if node.get("active_revision")
                ), ""),
                "agent_revision": str(agent.get("agent_revision") or ""),
                "heartbeat_at": next((
                    node.get("heartbeat_at") for node in target_nodes if node.get("heartbeat_at")
                ), None),
                "error": next((
                    str(node.get("error")) for node in target_nodes if node.get("error")
                ), ""),
            })

        controller_status = str(release.get("controller_status") or "").upper()
        if controller_status in {"BLOCKED_LOCAL", "ERROR"}:
            state = controller_status
        elif managed and not blocked_targets and str(release.get("phase") or "").upper() == "ACTIVE":
            state = "ACTIVE"
        elif managed:
            state = "SYNCING"
        else:
            state = "ACTIVE" if not blocked_targets else "SYNCING"
        return {
            "deployment_id": release_id,
            "managed": managed,
            "state": state,
            "phase": str(release.get("phase") or ""),
            "desired_revision": desired,
            "local_revision": str(release.get("controller_revision") or ""),
            "message": str(release.get("controller_message") or release.get("error") or ""),
            "updated_at": release.get("controller_heartbeat_at") or release.get("requested_at"),
            "requested_at": release.get("requested_at"),
            "requested_by": release.get("requested_by"),
            "blocked_targets": blocked_targets,
            "targets": targets,
        }

    def get_scheduling_active_batches(self) -> list:
        """Batches that should consume fresh-dispatch capacity.

        This is intentionally narrower than ``get_active_batches()``. The wider
        query keeps recovered remote-agent rows visible to the dashboard even if
        the local batch row drifted into a terminal state during a dispatcher
        restart. Those recovered rows should not indefinitely block fresh local
        dispatch capacity.
        """
        rows = self._conn().execute(
            "SELECT * FROM batches WHERE "
            + _SCHEDULING_ACTIVE_BATCH_PREDICATE
            + " ORDER BY dispatched_at DESC, batch_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_running_batches(self) -> list:
        rows = self._conn().execute(
            "SELECT * FROM batches WHERE status='RUNNING'"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_active_batches(self) -> int:
        row = self._conn().execute(
            """
            SELECT COUNT(*) AS cnt
              FROM batches
             WHERE status IN ('RUNNING','RECOVERING','FINALIZING')
                OR (
                    backend='remote_agent'
                    AND COALESCE(remote_status, '') NOT IN ('', 'SUCCEEDED', 'FAILED', 'CANCELLED')
                )
            """
        ).fetchone()
        return int(self._row_get(row, "cnt", 0) or 0)

    def count_scheduling_active_batches(self) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) AS cnt FROM batches WHERE "
            + _SCHEDULING_ACTIVE_BATCH_PREDICATE
        ).fetchone()
        return int(self._row_get(row, "cnt", 0) or 0)

    def count_running_batches(self) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) AS cnt FROM batches WHERE status='RUNNING'"
        ).fetchone()
        return int(self._row_get(row, "cnt", 0) or 0)

    def active_batch_breakdown(self) -> dict:
        rows = self.get_scheduling_active_batches()
        local = 0
        remote = 0
        for row in rows:
            if str(row.get("backend") or "") == "remote_agent":
                remote += 1
            else:
                local += 1
        total = local + remote
        return {
            "running_batches": total,
            "running_batches_total": total,
            "running_batches_local": local,
            "running_batches_remote": remote,
        }

    def set_batch_session_id(self, batch_id: str, session_id: str):
        self._run_write(
            "set_batch_session_id",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET session_id=? WHERE batch_id=?"), (session_id, batch_id)
            ),
        )

    def set_batch_run_name(self, batch_id: str, run_name: str):
        """Persist the current NF run name for Tower matching without log reads."""
        self._run_write(
            "set_batch_run_name",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET run_name=? WHERE batch_id=?"),
                (run_name, batch_id),
            ),
        )
        # A resume gets a new NF run name; force the next snapshot to rematch
        # its Tower state rather than waiting for the enrichment TTL.
        with self._snapshot_enrichment_lock:
            self._tower_snapshot_cache = None

    def get_batch_session_id(self, batch_id: str) -> str | None:
        row = self._conn().execute(
            self._sql("SELECT session_id FROM batches WHERE batch_id=?"), (batch_id,)
        ).fetchone()
        return self._row_get(row, "session_id") if row else None

    def set_batch_nf_pid(self, batch_id: str, pid: int):
        self._run_write(
            "set_batch_nf_pid",
            lambda conn: conn.execute(
                self._sql("UPDATE batches SET nf_pid=? WHERE batch_id=?"), (pid, batch_id)
            ),
        )

    def get_batch_nf_pid(self, batch_id: str) -> int | None:
        row = self._conn().execute(
            self._sql("SELECT nf_pid FROM batches WHERE batch_id=?"), (batch_id,)
        ).fetchone()
        return self._row_get(row, "nf_pid") if row else None

    def get_finished_batches_with_work_dirs(self) -> list:
        rows = self._conn().execute(
            "SELECT batch_id, work_dir, status FROM batches WHERE status IN ('SUCCEEDED','FAILED') AND work_dir IS NOT NULL"
        ).fetchall()
        return [dict(r) for r in rows]

    def sync_batch_hooks(self, batch_id: str, hooks: list[dict]) -> None:
        if not hooks:
            return

        def _sync(conn) -> None:
            self._executemany(
                conn,
                """
                INSERT INTO batch_hooks (
                    batch_id, hook_index, phase, hook_name, command, required, status,
                    runtime_revision, idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                ON CONFLICT(batch_id, hook_index) DO UPDATE SET
                    phase=excluded.phase,
                    hook_name=excluded.hook_name,
                    command=excluded.command,
                    required=excluded.required,
                    runtime_revision=excluded.runtime_revision,
                    idempotency_key=excluded.idempotency_key,
                    status=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN 'PENDING' ELSE batch_hooks.status END,
                    started_at=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.started_at END,
                    completed_at=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.completed_at END,
                    exit_code=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.exit_code END,
                    error_msg=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.error_msg END,
                    failure_category=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.failure_category END,
                    failure_detail=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.failure_detail END,
                    execution_node=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.execution_node END,
                    attempt_count=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN 0 ELSE batch_hooks.attempt_count END,
                    next_retry_at=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.next_retry_at END,
                    executor_job_id=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.executor_job_id END,
                    executor_state=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.executor_state END,
                    executor_submitted_at=CASE
                        WHEN batch_hooks.phase <> excluded.phase OR batch_hooks.hook_name <> excluded.hook_name OR batch_hooks.command <> excluded.command OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.executor_submitted_at END,
                    executor_last_polled_at=CASE
                        WHEN batch_hooks.phase <> excluded.phase OR batch_hooks.hook_name <> excluded.hook_name OR batch_hooks.command <> excluded.command OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.executor_last_polled_at END,
                    executor_missing_since=CASE
                        WHEN batch_hooks.phase <> excluded.phase OR batch_hooks.hook_name <> excluded.hook_name OR batch_hooks.command <> excluded.command OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.executor_missing_since END,
                    active_attempt_number=CASE
                        WHEN batch_hooks.phase <> excluded.phase
                          OR batch_hooks.hook_name <> excluded.hook_name
                          OR batch_hooks.command <> excluded.command
                          OR batch_hooks.required <> excluded.required
                        THEN NULL ELSE batch_hooks.active_attempt_number END
                """,
                [
                    (
                        batch_id,
                        int(h["hook_index"]),
                        h.get("phase", ""),
                        h.get("hook_name", ""),
                        h.get("command", ""),
                        1 if h.get("required", True) else 0,
                        h.get("runtime_revision", ""),
                        h.get("idempotency_key")
                        or hashlib.sha256(
                            f"{batch_id}:{int(h['hook_index'])}:{h.get('command', '')}".encode()
                        ).hexdigest(),
                    )
                    for h in hooks
                ],
            )

        self._run_write("sync_batch_hooks", _sync)

    def start_batch_hook(self, batch_id: str, hook_index: int) -> int:
        started_at = datetime.now(timezone.utc).isoformat()
        def _start(conn) -> None:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            existing = conn.execute(
                self._sql(
                    "SELECT status, active_attempt_number, idempotency_key, runtime_revision, "
                    "executor_job_id, attempt_count FROM batch_hooks "
                    "WHERE batch_id=? AND hook_index=?" + lock
                ),
                (batch_id, hook_index),
            ).fetchone()
            old_status = str(self._row_get(existing, "status", 0) or "") if existing else ""
            active = self._row_get(existing, "active_attempt_number", 1) if existing else None
            if old_status == "RUNNING" and active is not None:
                attempt = int(active)
                attempt_count_sql = "attempt_count=COALESCE(attempt_count, 0)"
            else:
                max_row = conn.execute(
                    self._sql(
                        "SELECT MAX(attempt_number) AS max_attempt FROM hook_attempts "
                        "WHERE batch_id=? AND hook_index=?"
                    ),
                    (batch_id, hook_index),
                ).fetchone()
                attempt = int(self._row_get(max_row, "max_attempt", 0) or 0) + 1
                attempt_count_sql = "attempt_count=COALESCE(attempt_count, 0)+1"
            conn.execute(
                self._sql("""
                UPDATE batch_hooks
                   SET status='RUNNING', started_at=?, completed_at=NULL, exit_code=NULL,
                       error_msg=NULL, {attempt_count_sql}, active_attempt_number=?,
                       next_retry_at=NULL,
                       executor_job_id=CASE WHEN status='RUNNING' THEN executor_job_id ELSE NULL END,
                       executor_state='RUNNING',
                       executor_submitted_at=CASE WHEN status='RUNNING' THEN executor_submitted_at ELSE NULL END,
                       executor_last_polled_at=CASE WHEN status='RUNNING' THEN executor_last_polled_at ELSE NULL END,
                       executor_missing_since=CASE WHEN status='RUNNING' THEN executor_missing_since ELSE NULL END
                 WHERE batch_id=? AND hook_index=?
                """.format(attempt_count_sql=attempt_count_sql)),
                (started_at, attempt, batch_id, hook_index),
            )
            if old_status != "RUNNING" or active is None:
                conn.execute(
                    self._sql(
                        "INSERT INTO hook_attempts "
                        "(batch_id, hook_index, attempt_number, started_at, status, "
                        "runtime_revision, idempotency_key) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)"
                    ),
                    (
                        batch_id, hook_index, attempt, started_at,
                        self._row_get(existing, "runtime_revision", 3) or "",
                        self._row_get(existing, "idempotency_key", 2) or "",
                    ),
                )
            return attempt

        return int(self._run_write("start_batch_hook", _start) or 1)

    def complete_batch_hook(
        self,
        batch_id: str,
        hook_index: int,
        *,
        exit_code: int | None,
        error_msg: str = "",
        failure_category: str = "",
        failure_detail: str = "",
        runtime_revision: str = "",
        execution_node: str = "",
        expected_executor_job_id: str | None = None,
        expected_attempt_number: int | None = None,
        executor_polled_at: str | None = None,
    ) -> bool:
        completed_at = datetime.now(timezone.utc).isoformat()
        executor_polled_at = executor_polled_at or completed_at
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        def _complete(conn) -> bool:
            if expected_executor_job_id is None:
                lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
                current = conn.execute(
                    self._sql("SELECT status, executor_job_id FROM batch_hooks WHERE batch_id=? AND hook_index=?" + lock),
                    (batch_id, hook_index),
                ).fetchone()
                if current and str(self._row_get(current, "status", 0) or "") != "RUNNING" and self._row_get(current, "executor_job_id", 1):
                    return False
            retry_at = None
            if status == "FAILED":
                row = conn.execute(
                    self._sql(
                        "SELECT attempt_count FROM batch_hooks "
                        "WHERE batch_id=? AND hook_index=?"
                    ),
                    (batch_id, hook_index),
                ).fetchone()
                attempt = int(self._row_get(row, "attempt_count", 0) or 1)
                base_delay = min(900, 60 * (2 ** max(0, attempt - 1)))
                # Deterministic jitter spreads a shared retry cohort without
                # making tests or incident reconstruction depend on RNG state.
                jitter_seed = int(
                    hashlib.sha256(f"{batch_id}:{hook_index}:{attempt}".encode()).hexdigest()[:8],
                    16,
                )
                jitter = 0.8 + (jitter_seed % 4001) / 10000.0
                delay = int(base_delay * jitter)
                retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            cursor = conn.execute(
                self._sql("""
                UPDATE batch_hooks
                   SET status=?, completed_at=?, exit_code=?, error_msg=?,
                       next_retry_at=?, executor_state=?, executor_last_polled_at=COALESCE(?, executor_last_polled_at), executor_missing_since=NULL, failure_category=?,
                       failure_detail=?, runtime_revision=COALESCE(NULLIF(?, ''), runtime_revision),
                       execution_node=COALESCE(NULLIF(?, ''), execution_node)
                 WHERE batch_id=? AND hook_index=?
                   AND (CAST(? AS TEXT) IS NULL OR (status='RUNNING' AND executor_job_id=? AND active_attempt_number=?))
                """),
                (
                    status, completed_at, exit_code, error_msg, retry_at, status, executor_polled_at,
                    failure_category or None, failure_detail or error_msg,
                    runtime_revision, execution_node, batch_id, hook_index,
                    expected_executor_job_id, expected_executor_job_id,
                    expected_attempt_number if expected_attempt_number is not None else -1,
                ),
            )
            if expected_executor_job_id is not None and not cursor.rowcount:
                return False
            row = conn.execute(
                self._sql(
                    "SELECT attempt_count, active_attempt_number, idempotency_key FROM batch_hooks "
                    "WHERE batch_id=? AND hook_index=?"
                ),
                (batch_id, hook_index),
            ).fetchone()
            attempt = int(
                self._row_get(row, "active_attempt_number", 1)
                or self._row_get(row, "attempt_count", 0)
                or 1
            )
            conn.execute(
                self._sql(
                    "UPDATE hook_attempts SET completed_at=?, status=?, exit_code=?, "
                    "failure_category=?, failure_detail=?, runtime_revision=COALESCE(NULLIF(?, ''), runtime_revision), "
                    "execution_node=COALESCE(NULLIF(?, ''), execution_node), "
                    "idempotency_key=COALESCE(NULLIF(?, ''), idempotency_key) "
                    "WHERE batch_id=? AND hook_index=? AND attempt_number=?"
                ),
                (
                    completed_at, status, exit_code, failure_category or None,
                    failure_detail or error_msg, runtime_revision, execution_node,
                    self._row_get(row, "idempotency_key", 2) or "",
                    batch_id, hook_index, attempt,
                ),
            )
            conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET active_attempt_number=NULL "
                    "WHERE batch_id=? AND hook_index=?"
                ),
                (batch_id, hook_index),
            )
            return True

        return bool(self._run_write("complete_batch_hook", _complete))

    def skip_batch_hook(self, batch_id: str, hook_index: int, *, reason: str) -> None:
        completed_at = datetime.now(timezone.utc).isoformat()
        self._run_write(
            "skip_batch_hook",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET status='SKIPPED', completed_at=?, exit_code=0, "
                    "error_msg=?, next_retry_at=NULL, executor_state='SKIPPED' "
                    "WHERE batch_id=? AND hook_index=?"
                ),
                (completed_at, reason[:1000], batch_id, hook_index),
            ),
        )

    def skip_unstarted_batch_hooks(self, batch_id: str, *, reason: str) -> int:
        """Mark hooks that never started after a driver-terminal failure."""
        completed_at = datetime.now(timezone.utc).isoformat()

        def _skip(conn) -> int:
            cursor = conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET status='SKIPPED', completed_at=?, exit_code=0, "
                    "error_msg=?, next_retry_at=NULL, executor_state='SKIPPED' "
                    "WHERE batch_id=? AND status='PENDING' AND started_at IS NULL "
                    "AND executor_job_id IS NULL"
                ),
                (completed_at, reason[:1000], batch_id),
            )
            return int(cursor.rowcount or 0)

        return int(self._run_write("skip_unstarted_batch_hooks", _skip) or 0)

    def retry_batch_hook(self, batch_id: str, hook_index: int) -> None:
        self._run_write(
            "retry_batch_hook",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET status='PENDING', started_at=NULL, completed_at=NULL, "
                    "exit_code=NULL, error_msg=NULL, next_retry_at=NULL, executor_job_id=NULL, "
                    "executor_state='PENDING', executor_submitted_at=NULL, executor_last_polled_at=NULL, executor_missing_since=NULL, failure_category=NULL, failure_detail=NULL, "
                    "execution_node=NULL WHERE batch_id=? AND hook_index=?"
                ),
                (batch_id, hook_index),
            ),
        )

    def waive_batch_hook(self, batch_id: str, hook_index: int, *, reason: str) -> None:
        completed_at = datetime.now(timezone.utc).isoformat()
        self._run_write(
            "waive_batch_hook",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET status='WAIVED', completed_at=?, exit_code=0, "
                    "error_msg=?, next_retry_at=NULL, executor_state='WAIVED' "
                    "WHERE batch_id=? AND hook_index=?"
                ),
                (completed_at, reason[:1000], batch_id, hook_index),
            ),
        )

    def reopen_remote_finalization(self, batch_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()

        def _reopen(conn) -> bool:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql(
                    "SELECT remote_status, remote_payload FROM batches WHERE batch_id=?" + lock
                ),
                (batch_id,),
            ).fetchone()
            if not row:
                return False
            payload = self._decode_remote_payload(self._row_get(row, "remote_payload", 1))
            payload.update({"batch_id": batch_id, "status": "HOOKS_RETRY", "heartbeat_at": now})
            conn.execute(
                self._sql(
                    "UPDATE batches SET remote_status='HOOKS_RETRY', remote_heartbeat_at=?, "
                    "remote_payload=?, finalization_retry_at=?, finalization_error=NULL "
                    "WHERE batch_id=?"
                ),
                (now, json.dumps(payload, sort_keys=True), now, batch_id),
            )
            return True

        return bool(self._run_write("reopen_remote_finalization", _reopen))

    def set_batch_hook_executor(
        self, batch_id: str, hook_index: int, *, job_id: str, state: str = "SUBMITTED",
        submitted_at: str | None = None, polled_at: str | None = None,
        missing_since: str | None = None,
        expected_attempt_number: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        submitted_at = submitted_at or (now if state == "SUBMITTED" else None)
        polled_at = polled_at or now
        self._run_write(
            "set_batch_hook_executor",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET executor_job_id=?, executor_state=?, "
                    "executor_submitted_at=COALESCE(?, executor_submitted_at), "
                    "executor_last_polled_at=?, executor_missing_since=? "
                    "WHERE batch_id=? AND hook_index=? "
                    "AND (CAST(? AS INTEGER) IS NULL OR (status=? AND (executor_job_id IS NULL OR executor_job_id=?) AND active_attempt_number=?))"
                ),
                (job_id, state, submitted_at, polled_at, missing_since, batch_id, hook_index,
                 expected_attempt_number, "RUNNING", job_id, expected_attempt_number),
            ),
        )

    def get_active_hook_executors(self, target: str | None = None) -> list[dict]:
        query = (
            "SELECT bh.batch_id, bh.hook_index, bh.hook_name, bh.attempt_count, "
            "bh.active_attempt_number, bh.executor_job_id, bh.executor_state, "
            "bh.executor_submitted_at, bh.executor_last_polled_at, bh.executor_missing_since, "
            "b.target, b.backend, b.remote_status "
            "FROM batch_hooks bh JOIN batches b ON b.batch_id=bh.batch_id "
            "WHERE bh.status=? AND bh.executor_job_id IS NOT NULL "
            "AND bh.executor_state IN (?, ?, ?, ?) AND b.backend=?")
        params = ["RUNNING", "SUBMITTED", "RUNNING", "PENDING", "UNKNOWN", "remote_agent"]
        if target:
            query += " AND b.target=?"
            params.append(target)
        query += " ORDER BY bh.executor_submitted_at, bh.batch_id, bh.hook_index"
        rows = self._conn().execute(self._sql(query), params).fetchall()
        return [dict(row) for row in rows]

    def mark_batch_hook_executor_polled(
        self, batch_id: str, hook_index: int, *, job_id: str,
        attempt_number: int, state: str, polled_at: str,
        missing_since: str | None = None,
    ) -> bool:
        def _mark(conn) -> bool:
            cursor = conn.execute(
                self._sql(
                    "UPDATE batch_hooks SET executor_state=?, executor_last_polled_at=?, "
                    "executor_missing_since=? WHERE batch_id=? AND hook_index=? "
                    "AND status=? AND executor_job_id=? AND active_attempt_number=?"
                ),
                (state, polled_at, missing_since, batch_id, hook_index, "RUNNING", job_id, attempt_number),
            )
            return bool(cursor.rowcount)

        return bool(self._run_write("mark_batch_hook_executor_polled", _mark))

    def get_batch_hooks(self, batch_ids: list[str]) -> dict[str, list[dict]]:
        if not batch_ids:
            return {}
        placeholders = self._placeholders(len(batch_ids))
        rows = self._conn().execute(
            self._sql(f"""
            SELECT batch_id, hook_index, phase, hook_name, command, status,
                   started_at, completed_at, exit_code, error_msg, required,
                   attempt_count, next_retry_at, executor_job_id, executor_state,
                   executor_submitted_at, executor_last_polled_at, executor_missing_since,
                   failure_category, failure_detail, runtime_revision,
                   execution_node, idempotency_key, active_attempt_number
              FROM batch_hooks
             WHERE batch_id IN ({placeholders})
             ORDER BY batch_id DESC, hook_index ASC
            """),
            batch_ids,
        ).fetchall()
        return self._group_batch_hook_rows(rows)

    def _group_batch_hook_rows(self, rows) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            item = dict(row)
            started = item.get("started_at")
            completed = item.get("completed_at")
            duration_s = None
            if started and completed:
                try:
                    t0 = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
                    duration_s = max(0, int((t1 - t0).total_seconds()))
                except Exception:
                    duration_s = None
            item["duration_s"] = duration_s
            grouped.setdefault(item["batch_id"], []).append(item)
        return grouped

    def hook_circuit_status(self, target: str) -> dict | None:
        """Return the durable circuit-breaker state for one execution lane."""
        row = self._conn().execute(
            self._sql(
                "SELECT target, category, failure_times, paused, state, canary_batch_id, "
                "canary_started_at, reason, updated_at "
                "FROM hook_circuit_breakers WHERE target=?"
            ),
            (target,),
        ).fetchone()
        return dict(row) if row else None

    def record_hook_failure(
        self,
        target: str,
        category: str,
        detail: str,
        *,
        batch_id: str,
        hook_name: str,
        attempt_number: int = 0,
        threshold: int = 3,
        window_seconds: int = 300,
    ) -> bool:
        """Record a classified failure and trip the lane circuit if needed.

        A batch contributes at most one event to a hook/category cohort. This
        prevents retries and multiple hooks in one batch from creating a false
        circuit trip while still stopping a repeated failure across batches.
        """
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(seconds=max(1, int(window_seconds)))).isoformat()

        def _record(conn) -> bool:
            event_hook = hook_name
            event_batch = batch_id
            conn.execute(
                self._sql(
                    "INSERT INTO hook_failure_events "
                    "(target, hook_name, category, batch_id, attempt_number, detail, failed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(target, hook_name, category, batch_id) DO UPDATE SET "
                    "attempt_number=excluded.attempt_number, detail=excluded.detail, "
                    "failed_at=excluded.failed_at"
                ),
                (
                    target, event_hook, category, event_batch, int(attempt_number or 0),
                    detail[:1000], now.isoformat(),
                ),
            )
            conn.execute(
                self._sql("DELETE FROM hook_failure_events WHERE target=? AND failed_at < ?"),
                (target, cutoff),
            )
            row = conn.execute(
                self._sql(
                    "SELECT category, failure_times, paused, state, canary_batch_id "
                    "FROM hook_circuit_breakers WHERE target=?"
                    + (" FOR UPDATE" if self.db_cfg.backend == "postgres" else "")
                ),
                (target,),
            ).fetchone()
            state = str(self._row_get(row, "state", 3) or "") if row else "CLOSED"
            if state not in {"CLOSED", "OPEN", "HALF_OPEN"}:
                state = "OPEN" if int(self._row_get(row, "paused", 2) or 0) else "CLOSED"
            count_row = conn.execute(
                self._sql(
                    "SELECT COUNT(*) AS failure_count FROM hook_failure_events "
                    "WHERE target=? AND hook_name=? AND category=? AND failed_at >= ?"
                ),
                (target, event_hook, category, cutoff),
            ).fetchone()
            failure_count = int(self._row_get(count_row, "failure_count", 0) or 0)
            if state == "HALF_OPEN" or failure_count >= max(1, int(threshold)):
                state = "OPEN"
            paused = 1 if state in {"OPEN", "HALF_OPEN"} else 0
            reason = (
                f"{category}/{event_hook}: {detail[:850]} "
                f"({failure_count} affected batches in {window_seconds}s)"
                if state == "OPEN" else ""
            )
            failure_times = json.dumps([now.timestamp()])
            conn.execute(
                self._sql(
                    "INSERT INTO hook_circuit_breakers "
                    "(target, category, failure_times, paused, state, canary_batch_id, "
                    "canary_started_at, reason, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?) "
                    "ON CONFLICT(target) DO UPDATE SET category=excluded.category, "
                    "failure_times=excluded.failure_times, paused=excluded.paused, "
                    "state=excluded.state, canary_batch_id=NULL, canary_started_at=NULL, "
                    "reason=excluded.reason, updated_at=excluded.updated_at"
                ),
                (target, category, failure_times, paused, state, reason, now.isoformat()),
            )
            return state == "OPEN"

        return bool(self._run_write("record_hook_failure", _record))

    def begin_hook_canary(self, target: str, *, reason: str = "operator resume") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._run_write(
            "begin_hook_canary",
            lambda conn: conn.execute(
                self._sql(
                    "INSERT INTO hook_circuit_breakers "
                    "(target, category, failure_times, paused, state, canary_batch_id, "
                    "canary_started_at, reason, updated_at) "
                    "VALUES (?, 'OPERATOR', '[]', 1, 'HALF_OPEN', NULL, NULL, ?, ?) "
                    "ON CONFLICT(target) DO UPDATE SET paused=1, state='HALF_OPEN', "
                    "canary_batch_id=NULL, canary_started_at=NULL, reason=?, updated_at=?"
                ),
                (target, reason[:1000], now, reason[:1000], now),
            ),
        )

    def claim_hook_canary(self, target: str, batch_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()

        def _claim(conn) -> bool:
            lock = " FOR UPDATE" if self.db_cfg.backend == "postgres" else ""
            row = conn.execute(
                self._sql(
                    "SELECT state, canary_batch_id FROM hook_circuit_breakers WHERE target=?" + lock
                ),
                (target,),
            ).fetchone()
            if not row:
                return True
            state = str(self._row_get(row, "state", 0) or "CLOSED")
            canary_batch = self._row_get(row, "canary_batch_id", 1)
            if state == "CLOSED":
                return True
            if state == "OPEN":
                return False
            if canary_batch and str(canary_batch) != batch_id:
                return False
            if not canary_batch:
                conn.execute(
                    self._sql(
                        "UPDATE hook_circuit_breakers SET canary_batch_id=?, "
                        "canary_started_at=?, updated_at=? WHERE target=?"
                    ),
                    (batch_id, now, now, target),
                )
            return True

        return bool(self._run_write("claim_hook_canary", _claim))

    def finish_hook_canary(self, target: str, batch_id: str, *, success: bool, reason: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        state = "CLOSED" if success else "OPEN"
        self._run_write(
            "finish_hook_canary",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE hook_circuit_breakers SET state=?, paused=?, canary_batch_id=NULL, "
                    "canary_started_at=NULL, reason=?, updated_at=? "
                    "WHERE target=? AND state='HALF_OPEN' AND canary_batch_id=?"
                ),
                (state, 0 if success else 1, "" if success else reason[:1000], now, target, batch_id),
            ),
        )

    def clear_hook_circuit(self, target: str) -> None:
        self._run_write(
            "clear_hook_circuit",
            lambda conn: conn.execute(
                self._sql(
                    "UPDATE hook_circuit_breakers SET paused=0, state='CLOSED', "
                    "failure_times='[]', canary_batch_id=NULL, canary_started_at=NULL, "
                    "reason='', updated_at=? WHERE target=?"
                ),
                (datetime.now(timezone.utc).isoformat(), target),
            ),
        )

    def pause_hook_circuit(self, target: str, *, reason: str, category: str = "OPERATOR") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._run_write(
            "pause_hook_circuit",
            lambda conn: conn.execute(
                self._sql(
                    "INSERT INTO hook_circuit_breakers "
                    "(target, category, failure_times, paused, state, canary_batch_id, "
                    "canary_started_at, reason, updated_at) "
                    "VALUES (?, ?, '[]', 1, 'OPEN', NULL, NULL, ?, ?) "
                    "ON CONFLICT(target) DO UPDATE SET category=excluded.category, "
                    "paused=1, state='OPEN', canary_batch_id=NULL, canary_started_at=NULL, "
                    "reason=excluded.reason, updated_at=excluded.updated_at"
                ),
                (target, category, reason[:1000], now),
            ),
        )

    def _active_run_name_map_for_batches(self, batches: list[dict]) -> dict[str, str]:
        mapping = {
            str(batch["run_name"]): str(batch["batch_id"])
            for batch in batches
            if batch.get("run_name")
        }
        if not self.cfg:
            return mapping
        for batch in batches:
            if batch.get("run_name"):
                continue
            batch_id = str(batch["batch_id"])
            nf_log = os.path.join(self.cfg.log_dir, f"batch_{batch_id}.nf.log")
            if not os.path.exists(nf_log):
                continue
            try:
                with open(nf_log, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    fh.seek(max(0, fh.tell() - _RUN_NAME_LOG_TAIL_BYTES))
                    text = fh.read().decode(errors="replace")
            except OSError:
                continue
            match = _re.search(r"Run name:\s+(\S+)", text)
            if not match:
                match = _re.search(r"-name\s+(\S+)", text)
            if match:
                mapping[match.group(1)] = batch_id
        return mapping

    def _load_tower_store_states(self) -> list[dict]:
        if not self.cfg or _TowerRunStore is None:
            return []
        tower_db = os.path.join(self.cfg.state_dir, "turret_runs.db")
        if not os.path.exists(tower_db):
            return []
        store = _TowerRunStore(tower_db)
        rows = store.load_all()
        now = time.time()
        return [normalize_tower_state(row, now=now) for row in rows]

    @staticmethod
    def _enrichment_metadata(*, updated_at: float, stale: bool, error: str | None) -> dict:
        return {
            "updated_at": datetime.fromtimestamp(updated_at, timezone.utc).isoformat() if updated_at else None,
            "stale": stale,
            "error": error,
        }

    def _cached_wds_snapshot(self) -> tuple[dict, dict]:
        now = time.time()
        with self._snapshot_enrichment_lock:
            cached = self._wds_snapshot_cache
            if cached and now - float(cached["updated_at"]) < _SNAPSHOT_ENRICHMENT_TTL_SECONDS:
                return dict(cached["value"]), dict(cached["metadata"])
            if cached and now < float(cached.get("next_retry_at", 0.0)):
                value = dict(cached["value"])
                metadata = self._enrichment_metadata(
                    updated_at=float(cached["updated_at"]), stale=True,
                    error=str(cached.get("last_error") or "enrichment refresh deferred"),
                )
                value["stale"] = True
                return value, metadata
            try:
                value = build_wds_snapshot(self.cfg) if self.cfg is not None else {
                    "models": {}, "total": 0, "inventory_total": 0,
                    "error": None, "source": "dispatcher_snapshot",
                }
                wds_errors = [
                    str(model.get("error"))
                    for model in (value.get("models") or {}).values()
                    if isinstance(model, dict) and model.get("error")
                ]
                if value.get("error") or wds_errors:
                    raise RuntimeError(str(value.get("error") or "; ".join(wds_errors)))
                metadata = self._enrichment_metadata(updated_at=now, stale=False, error=None)
            except Exception as exc:
                retry_delay = min(
                    _SNAPSHOT_ENRICHMENT_TTL_SECONDS,
                    max(
                        _SNAPSHOT_ENRICHMENT_RETRY_SECONDS,
                        float((cached or {}).get("failure_backoff_s", _SNAPSHOT_ENRICHMENT_RETRY_SECONDS)) * 2,
                    ),
                )
                if cached:
                    value = dict(cached["value"])
                    cached["last_error"] = str(exc)
                    cached["failure_backoff_s"] = retry_delay
                    cached["next_retry_at"] = now + retry_delay
                    metadata = self._enrichment_metadata(
                        updated_at=float(cached["updated_at"]), stale=True, error=str(exc)
                    )
                else:
                    value = {"models": {}, "total": 0, "inventory_total": 0,
                             "error": str(exc), "source": "dispatcher_snapshot"}
                    metadata = self._enrichment_metadata(updated_at=0.0, stale=True, error=str(exc))
                    self._wds_snapshot_cache = {
                        "updated_at": 0.0, "value": dict(value), "metadata": dict(metadata),
                        "last_error": str(exc), "failure_backoff_s": retry_delay,
                        "next_retry_at": now + retry_delay,
                    }
            else:
                self._wds_snapshot_cache = {
                    "updated_at": now, "value": dict(value), "metadata": dict(metadata),
                    "last_error": None, "failure_backoff_s": _SNAPSHOT_ENRICHMENT_RETRY_SECONDS,
                    "next_retry_at": 0.0,
                }
            value["stale"] = bool(metadata["stale"])
            return value, metadata

    def _cached_tower_states(self, batches: list[dict]) -> tuple[list[dict], dict[str, str], dict]:
        now = time.time()
        with self._snapshot_enrichment_lock:
            cached = self._tower_snapshot_cache
            if cached and now - float(cached["updated_at"]) < _SNAPSHOT_ENRICHMENT_TTL_SECONDS:
                return list(cached["states"]), dict(cached["active_runs"]), dict(cached["metadata"])
            if cached and now < float(cached.get("next_retry_at", 0.0)):
                return (
                    list(cached["states"]), dict(cached["active_runs"]),
                    self._enrichment_metadata(
                        updated_at=float(cached["updated_at"]), stale=True,
                        error=str(cached.get("last_error") or "enrichment refresh deferred"),
                    ),
                )
            try:
                states = self._load_tower_store_states()
                active_runs = self._active_run_name_map_for_batches(batches)
                metadata = self._enrichment_metadata(updated_at=now, stale=False, error=None)
            except Exception as exc:
                retry_delay = min(
                    _SNAPSHOT_ENRICHMENT_TTL_SECONDS,
                    max(
                        _SNAPSHOT_ENRICHMENT_RETRY_SECONDS,
                        float((cached or {}).get("failure_backoff_s", _SNAPSHOT_ENRICHMENT_RETRY_SECONDS)) * 2,
                    ),
                )
                if cached:
                    cached["last_error"] = str(exc)
                    cached["failure_backoff_s"] = retry_delay
                    cached["next_retry_at"] = now + retry_delay
                    return (
                        list(cached["states"]), dict(cached["active_runs"]),
                        self._enrichment_metadata(
                            updated_at=float(cached["updated_at"]), stale=True, error=str(exc)
                        ),
                    )
                self._tower_snapshot_cache = {
                    "updated_at": 0.0, "states": [], "active_runs": {},
                    "metadata": {}, "last_error": str(exc),
                    "failure_backoff_s": retry_delay, "next_retry_at": now + retry_delay,
                }
                return [], {}, self._enrichment_metadata(updated_at=0.0, stale=True, error=str(exc))
            self._tower_snapshot_cache = {
                "updated_at": now, "states": list(states), "active_runs": dict(active_runs),
                "metadata": dict(metadata), "last_error": None,
                "failure_backoff_s": _SNAPSHOT_ENRICHMENT_RETRY_SECONDS, "next_retry_at": 0.0,
            }
            return states, active_runs, metadata

    def _tower_state_for_batch(
        self,
        batch_id: str,
        *,
        tower_states: list[dict],
        active_runs: dict[str, str],
    ) -> dict | None:
        return newest_tower_state_for_batch(
            batch_id, tower_states, active_runs=active_runs
        )

    def _load_recent_batches_snapshot_from_conn(self, conn, *, limit: int = 30) -> list[dict]:
        # StateStore initialization performs the schema migration once. Do not
        # query information_schema on every 60-second dashboard snapshot.
        cols = {
            "backend", "target", "remote_status", "remote_heartbeat_at",
            "remote_driver_job_id", "remote_work_dir", "remote_outdir",
            "remote_log_path", "remote_payload",
            "source_revision", "release_manifest_sha256",
            "semantic_contract_sha256", "execution_profile",
            "execution_profile_sha256", "params_path", "effective_params_sha256",
        }
        remote_select = []
        for col in (
            "backend",
            "target",
            "remote_status",
            "remote_heartbeat_at",
            "remote_driver_job_id",
            "remote_work_dir",
            "remote_outdir",
            "remote_log_path",
            "remote_payload",
            "source_revision",
            "release_manifest_sha256",
            "semantic_contract_sha256",
            "execution_profile",
            "execution_profile_sha256",
            "params_path",
            "effective_params_sha256",
        ):
            remote_select.append(col if col in cols else f"NULL AS {col}")
        select_sql = (
            """
            SELECT batch_id, status, slide_count, dispatched_at, run_name,
                   completed_at, nextflow_exit, log_path,
                   """
            + ", ".join(remote_select)
            + """
            FROM batches
            """
        )
        recent_rows = conn.execute(
            self._sql(select_sql + "ORDER BY dispatched_at DESC LIMIT ?"),
            (int(limit),),
        ).fetchall()
        active_rows = conn.execute(
            self._sql(
                select_sql
                + """
                WHERE status IN ('RUNNING', 'RECOVERING', 'FINALIZING')
                   OR (
                        backend='remote_agent'
                        AND COALESCE(remote_status, '') NOT IN (
                            '', 'SUCCEEDED', 'FAILED', 'CANCELLED', 'STALE',
                            'FINALIZATION_QUEUED', 'HOOKS_RUNNING', 'HOOKS_RETRY'
                        )
                   )
                ORDER BY COALESCE(remote_heartbeat_at, dispatched_at) DESC,
                         dispatched_at DESC, batch_id DESC
                """
            )
        ).fetchall()
        items = [dict(row) for row in recent_rows]
        seen_ids = {str(item.get("batch_id") or "") for item in items}
        items.extend(
            dict(row) for row in active_rows
            if str(row["batch_id"] or "") not in seen_ids
        )
        if not items:
            return []
        batch_ids = [item["batch_id"] for item in items]
        placeholders = self._placeholders(len(batch_ids))
        hook_rows = self._group_batch_hook_rows(
            conn.execute(
                self._sql(f"""
                SELECT batch_id, hook_index, phase, hook_name, command, status,
                       started_at, completed_at, exit_code, error_msg
                  FROM batch_hooks
                 WHERE batch_id IN ({placeholders})
                 ORDER BY batch_id DESC, hook_index ASC
                """),
                batch_ids,
            ).fetchall()
        )
        for item in items:
            item["hooks"] = hook_rows.get(item["batch_id"], [])
        return items

    def _attach_tower_snapshot(self, items: list[dict]) -> tuple[list[dict], dict]:
        tower_items = [item for item in items if _tower_is_live_for_batch(item)]
        if not tower_items:
            for item in items:
                item["tower"] = None
                item["tower_expected"] = False
            return items, self._enrichment_metadata(updated_at=time.time(), stale=False, error=None)

        tower_states, active_runs, tower_metadata = self._cached_tower_states(tower_items)
        for item in items:
            tower_expected = _tower_is_live_for_batch(item)
            item["tower_expected"] = tower_expected
            if tower_expected:
                item["tower"] = self._tower_state_for_batch(
                    item["batch_id"],
                    tower_states=tower_states,
                    active_runs=active_runs,
                )
            else:
                item["tower"] = None
        return items, tower_metadata

    def get_throughput_stats(self, window_hours: float = 6.0) -> dict:
        """Return slide throughput stats based on recently completed slides.

        Looks at slides that completed within the last *window_hours* and
        computes wall-clock throughput (slides per hour) as:
            count / elapsed

        where elapsed = min(window_hours, hours since oldest completion in window).
        This avoids inflated rates when many slides complete in a short burst
        (which would make MAX-MIN very small).

        Returns a dict with keys:
            completed_in_window  – slide count used for the calculation
            window_hours         – actual elapsed hours in the observation window
            throughput_per_hour  – slides/hour (float), or None if insufficient data
            eta_seconds          – seconds until all remaining slides finish, or None
            remaining            – PENDING + DISPATCHED slide count
        """
        if self.db_cfg.backend == "postgres":
            now = datetime.now(timezone.utc)
            window_cutoff = (now - timedelta(hours=float(window_hours))).isoformat()
            burst_cutoff = (now - timedelta(hours=_BURST_WINDOW_HOURS)).isoformat()
            row = self._conn().execute(
                """
                SELECT COUNT(*) AS cnt, MIN(completed_at) AS oldest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND completed_at <> ''
                  AND completed_at >= %s
                """,
                (window_cutoff,),
            ).fetchone()
            recent_row = self._conn().execute(
                """
                SELECT COUNT(*) AS cnt, MIN(completed_at) AS oldest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND completed_at <> ''
                  AND completed_at >= %s
                """,
                (burst_cutoff,),
            ).fetchone()
        else:
            row = self._conn().execute(
                """
                SELECT
                    COUNT(*) AS cnt,
                    MIN(completed_at) AS oldest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND julianday(completed_at) >= julianday('now') - (? / 24.0)
                """,
                (float(window_hours),),
            ).fetchone()
            recent_row = self._conn().execute(
                """
                SELECT
                    COUNT(*) AS cnt,
                    MIN(completed_at) AS oldest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND julianday(completed_at) >= julianday('now') - (? / 24.0)
                """,
                (_BURST_WINDOW_HOURS,),
            ).fetchone()

        cnt = self._row_get(row, "cnt", 0) or 0

        remaining_row = self._conn().execute(
            "SELECT COUNT(*) AS cnt FROM slides WHERE status IN ('PENDING', 'DISPATCHED')"
        ).fetchone()
        remaining = int(self._row_get(remaining_row, "cnt", 0) or 0)

        stats = estimate_throughput_and_eta(
            completed_in_window=cnt,
            oldest_completed_at=self._row_get(row, "oldest"),
            remaining=remaining,
            window_hours=window_hours,
            recent_completed=self._row_get(recent_row, "cnt", 0) or 0,
            recent_oldest_completed_at=self._row_get(recent_row, "oldest"),
        )

        return {
            "completed_in_window": cnt,
            "window_hours": stats["window_hours"],
            "throughput_per_hour": stats["throughput_per_hour"],
            "eta_seconds": stats["eta_seconds"],
            "remaining": remaining,
        }

    def update_runtime_status(self, **fields) -> None:
        """Publish scheduler-owned health data for the dashboard snapshot."""
        with self._runtime_status_lock:
            self._runtime_status.update(fields)
            snapshot = copy.deepcopy(self._runtime_status)
            try:
                Path(self._runtime_status_path).parent.mkdir(parents=True, exist_ok=True)
                temp_path = f"{self._runtime_status_path}.{os.getpid()}.{threading.get_ident()}.tmp"
                with open(temp_path, "w", encoding="utf-8") as handle:
                    json.dump(snapshot, handle, sort_keys=True)
                os.replace(temp_path, self._runtime_status_path)
            except (OSError, TypeError) as exc:
                log.warning("Could not persist runtime status: %s", exc)

    def runtime_status_snapshot(self) -> dict:
        with self._runtime_status_lock:
            try:
                with open(self._runtime_status_path, encoding="utf-8") as handle:
                    persisted = json.load(handle)
                if isinstance(persisted, dict):
                    self._runtime_status.update(persisted)
            except (FileNotFoundError, OSError, TypeError, ValueError):
                pass
            return copy.deepcopy(self._runtime_status)

    def build_status_snapshot(self) -> dict:
        """Return a dashboard-oriented status payload from the current state store."""
        conn = self._conn()
        timeout_set = False
        try:
            timeout_set = self._set_postgres_status_read_timeout(conn)
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM slides GROUP BY status"
            ).fetchall()
            counts = {self._row_get(row, "status", 0): int(self._row_get(row, "n", 1) or 0) for row in rows}
            if self._slide_rollups_ready:
                distinct_counts = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_unique,
                        SUM(has_succeeded) AS succeeded_unique,
                        SUM(has_retry_pending) AS retry_pending,
                        SUM(has_failed) AS terminal_failed,
                        SUM(has_pending) AS pending_total,
                        SUM(has_dispatched) AS dispatched_total
                    FROM slide_rollups
                    """
                ).fetchone()
                total_unique = int(self._row_get(distinct_counts, "total_unique", 0) or 0)
                succeeded_unique = int(self._row_get(distinct_counts, "succeeded_unique", 1) or 0)
                retry_pending = int(self._row_get(distinct_counts, "retry_pending", 2) or 0)
                terminal_failed = int(self._row_get(distinct_counts, "terminal_failed", 3) or 0)
                pending_total = int(self._row_get(distinct_counts, "pending_total", 4) or 0)
                dispatched_total = int(self._row_get(distinct_counts, "dispatched_total", 5) or 0)
            else:
                distinct_counts = conn.execute(
                    self._sql(
                        """
                        SELECT
                            COUNT(DISTINCT slide_id) AS total_unique,
                            COUNT(DISTINCT CASE WHEN status='SUCCEEDED' THEN slide_id END) AS succeeded_unique,
                            COUNT(DISTINCT CASE WHEN status='PENDING' AND fail_count > 0 THEN slide_id END) AS retry_pending,
                            COUNT(DISTINCT CASE WHEN status='FAILED' THEN slide_id END) AS terminal_failed,
                            COUNT(DISTINCT CASE WHEN status='PENDING' THEN slide_id END) AS pending_total,
                            COUNT(DISTINCT CASE WHEN status='DISPATCHED' THEN slide_id END) AS dispatched_total
                        FROM slides
                        """
                    ),
                ).fetchone()
                total_unique = int(self._row_get(distinct_counts, "total_unique", 0) or 0)
                succeeded_unique = int(self._row_get(distinct_counts, "succeeded_unique", 1) or 0)
                retry_pending = int(self._row_get(distinct_counts, "retry_pending", 2) or 0)
                terminal_failed = int(self._row_get(distinct_counts, "terminal_failed", 3) or 0)
                pending_total = int(self._row_get(distinct_counts, "pending_total", 4) or 0)
                dispatched_total = int(self._row_get(distinct_counts, "dispatched_total", 5) or 0)
            blacklisted_row = conn.execute(
                self._sql(
                    "SELECT COUNT(*) AS cnt FROM slides "
                    "WHERE status='FAILED' AND fail_count >= ?"
                ),
                (getattr(self.cfg, "max_slide_retries", 0) or 0,),
            ).fetchone()
            blacklisted = int(self._row_get(blacklisted_row, "cnt", 0) or 0)

            throughput = self.get_throughput_stats(window_hours=6.0)
            recent_batches = self._load_recent_batches_snapshot_from_conn(conn)
            try:
                running_counts = self.active_batch_breakdown()
            except Exception as exc:
                # Counts are advisory dashboard data. Preserve the rest of a
                # usable snapshot if this optional enrichment times out.
                log.warning("Active batch breakdown unavailable: %s", exc)
                running_counts = {
                    "running_batches": 0,
                    "running_batches_total": 0,
                    "running_batches_local": 0,
                    "running_batches_remote": 0,
                }
            try:
                finalization_coordination = self.finalization_coordination_status()
            except Exception as exc:
                log.warning("Finalization coordination status unavailable: %s", exc)
                finalization_coordination = {"lease": None, "queued_batches": 0}
            try:
                release_sync = self.release_sync_snapshot()
            except Exception as exc:
                log.warning("Release sync status unavailable: %s", exc)
                release_sync = {
                    "deployment_id": getattr(self.cfg, "release_deployment_id", "default"),
                    "managed": bool(getattr(self.cfg, "auto_sync_release", False)),
                    "state": "ERROR",
                    "message": str(exc),
                    "targets": [],
                    "blocked_targets": [],
                }
            runtime_status = self.runtime_status_snapshot()
            core_snapshot = {
                "counts": counts,
                "chart_counts": {
                    "SUCCEEDED": succeeded_unique,
                    "RETRY_PENDING": retry_pending,
                    "PENDING": max(0, pending_total - retry_pending),
                    "DISPATCHED": dispatched_total,
                    "FAILED": terminal_failed,
                },
                "total": total_unique,
                "running_batches": running_counts["running_batches"],
                "running_batches_total": running_counts["running_batches_total"],
                "running_batches_local": running_counts["running_batches_local"],
                "running_batches_remote": running_counts["running_batches_remote"],
                "blacklisted": blacklisted,
                "retry_pending": retry_pending,
                "throughput_per_hour": throughput["throughput_per_hour"],
                "eta_seconds": throughput["eta_seconds"],
                "remaining": throughput["remaining"],
                "completed_in_window": throughput["completed_in_window"],
                "recent_batches": recent_batches,
                "finalization_coordination": finalization_coordination,
                "release_sync": release_sync,
                "remote_targets": self.remote_target_snapshot(release_sync),
                "execution_pools": runtime_status.get("execution_pools", []),
                "dispatch_summary": runtime_status.get("dispatch_summary", {"available_pool_count": 0, "queued_work_blocked": False, "blockers": []}),
            }
        finally:
            if timeout_set:
                self._reset_postgres_status_read_timeout(conn)

        recent_batches, tower_metadata = self._attach_tower_snapshot(core_snapshot["recent_batches"])
        wds_snapshot, wds_metadata = self._cached_wds_snapshot()
        inventory_total = int(wds_snapshot.get("inventory_total", 0) or 0)
        wds_models = wds_snapshot.get("models") or {}
        if wds_models:
            wds_done = min(int((item or {}).get("slides", 0) or 0) for item in wds_models.values())
        else:
            wds_done = int(core_snapshot["chart_counts"]["SUCCEEDED"])
        denom = inventory_total or int(core_snapshot["total"])
        if denom:
            wds_done = min(wds_done, denom)
        core_snapshot.update({
            "wds_done": wds_done,
            "pct_done": round((wds_done / denom) * 100, 1) if denom else 0,
            "pct_succeeded": round((int(core_snapshot["chart_counts"]["SUCCEEDED"]) / denom) * 100, 1) if denom else 0,
            "recent_batches": recent_batches,
            "wds": wds_snapshot,
            "enrichment": {"wds": wds_metadata, "tower": tower_metadata},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        return core_snapshot

    def write_status_snapshot(self, path: str) -> dict:
        try:
            payload = self.build_status_snapshot()
        except Exception as exc:
            if self.db_cfg.backend == "postgres":
                self._rollback_read_conn()
            payload = self._load_snapshot_file(path)
            if payload is None:
                raise
            payload = dict(payload)
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            payload["snapshot_error"] = str(exc)
            log.warning("Status snapshot degraded to last known payload: %s", exc)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, path)
        return payload
