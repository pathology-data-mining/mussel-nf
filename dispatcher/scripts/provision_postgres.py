#!/usr/bin/env python3
"""Provision a Postgres database and schema for mussel-dispatcher.

Creates:
1. Application role
2. Application database
3. Dispatcher tables and indexes

Usage:
  python dispatcher/scripts/provision_postgres.py \
    --admin-host db.example.org \
    --admin-user postgres \
    --admin-password-env POSTGRES_ADMIN_PASSWORD \
    --db-host db.example.org \
    --db-name mussel_dispatcher_cohort \
    --app-user mussel_dispatcher_cohort \
    --app-password-env POSTGRES_APP_PASSWORD
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg
from psycopg import sql


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS slides (
        slide_path TEXT PRIMARY KEY,
        slide_id TEXT NOT NULL,
        oncotree_code TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'PENDING',
        batch_id TEXT,
        download_path TEXT,
        fail_count INTEGER NOT NULL DEFAULT 0,
        first_seen_at TEXT,
        dispatched_at TEXT,
        completed_at TEXT,
        error_msg TEXT,
        file_id TEXT,
        file_name TEXT,
        needs_download INTEGER NOT NULL DEFAULT 0,
        priority INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS batches (
        batch_id TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'RUNNING',
        slide_count INTEGER,
        dispatched_at TEXT,
        completed_at TEXT,
        nextflow_exit INTEGER,
        csv_path TEXT,
        work_dir TEXT,
        log_path TEXT,
        manifest_path TEXT,
        session_id TEXT,
        nf_pid INTEGER,
        run_name TEXT,
        backend TEXT NOT NULL DEFAULT 'local',
        target TEXT,
        remote_run_id TEXT,
        remote_status TEXT,
        remote_heartbeat_at TEXT,
        remote_driver_job_id TEXT,
        remote_work_dir TEXT,
        remote_outdir TEXT,
        remote_log_path TEXT,
        remote_payload TEXT,
        finalization_stage TEXT NOT NULL DEFAULT 'PENDING',
        finalization_owner TEXT,
        finalization_heartbeat_at TEXT,
        finalization_retry_at TEXT,
        finalization_error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS batch_hooks (
        batch_id TEXT NOT NULL,
        hook_index INTEGER NOT NULL,
        phase TEXT NOT NULL DEFAULT '',
        hook_name TEXT NOT NULL DEFAULT '',
        command TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'PENDING',
        started_at TEXT,
        completed_at TEXT,
        exit_code INTEGER,
        error_msg TEXT,
        required INTEGER NOT NULL DEFAULT 1,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        next_retry_at TEXT,
        executor_job_id TEXT,
        executor_state TEXT,
        failure_category TEXT,
        failure_detail TEXT,
        runtime_revision TEXT,
        execution_node TEXT,
        idempotency_key TEXT,
        active_attempt_number INTEGER,
        PRIMARY KEY (batch_id, hook_index)
    )
    """,
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hook_failure_events (
        target TEXT NOT NULL,
        hook_name TEXT NOT NULL,
        category TEXT NOT NULL,
        batch_id TEXT NOT NULL,
        attempt_number INTEGER NOT NULL DEFAULT 0,
        detail TEXT NOT NULL DEFAULT '',
        failed_at TEXT NOT NULL,
        PRIMARY KEY (target, hook_name, category, batch_id)
    )
    """,
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS remote_agents (
        target TEXT PRIMARY KEY,
        heartbeat_at TEXT,
        host TEXT,
        status TEXT,
        payload TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coordination_leases (
        lease_name TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        batch_id TEXT,
        acquired_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )
    """,
    "CREATE TABLE IF NOT EXISTS _schema_migrations (name TEXT PRIMARY KEY)",
    "CREATE INDEX IF NOT EXISTS idx_slides_status ON slides(status)",
    "CREATE INDEX IF NOT EXISTS idx_slides_batch_id ON slides(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_slides_slide_id ON slides(slide_id)",
    "CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status)",
    "CREATE INDEX IF NOT EXISTS idx_batches_remote_queue ON batches(target, remote_status)",
    "CREATE INDEX IF NOT EXISTS idx_batches_remote_finalization ON batches(target, remote_status, finalization_retry_at)",
    "CREATE INDEX IF NOT EXISTS idx_batches_remote_queue_order ON batches(target, remote_status, dispatched_at, batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_batches_remote_finalization_order ON batches(target, remote_status, finalization_retry_at, dispatched_at, batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_batch_hooks_batch_id ON batch_hooks(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_batch_hooks_status ON batch_hooks(status)",
    "CREATE INDEX IF NOT EXISTS idx_batch_hooks_executor_poll ON batch_hooks(status, executor_state, executor_job_id, batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_hook_failure_events_window ON hook_failure_events(target, hook_name, category, failed_at)",
    "CREATE INDEX IF NOT EXISTS idx_coordination_leases_expires_at ON coordination_leases(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_remote_agents_heartbeat_at ON remote_agents(heartbeat_at)",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS run_name TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS backend TEXT NOT NULL DEFAULT 'local'",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS target TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_run_id TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_status TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_heartbeat_at TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_driver_job_id TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_work_dir TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_outdir TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_log_path TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS remote_payload TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS finalization_stage TEXT NOT NULL DEFAULT 'PENDING'",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS finalization_owner TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS finalization_heartbeat_at TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS finalization_retry_at TEXT",
    "ALTER TABLE batches ADD COLUMN IF NOT EXISTS finalization_error TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS required INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS next_retry_at TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS executor_job_id TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS executor_state TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS failure_category TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS failure_detail TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS runtime_revision TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS execution_node TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
    "ALTER TABLE batch_hooks ADD COLUMN IF NOT EXISTS active_attempt_number INTEGER",
    "ALTER TABLE hook_circuit_breakers ADD COLUMN IF NOT EXISTS state TEXT",
    "ALTER TABLE hook_circuit_breakers ADD COLUMN IF NOT EXISTS canary_batch_id TEXT",
    "ALTER TABLE hook_circuit_breakers ADD COLUMN IF NOT EXISTS canary_started_at TEXT",
    "UPDATE hook_circuit_breakers SET state=CASE WHEN paused=1 THEN 'OPEN' ELSE 'CLOSED' END WHERE state IS NULL",
]


def _password_from_env(name: str, *, required: bool = True) -> str:
    if not name:
        if required:
            raise RuntimeError("Password environment variable name is required")
        return ""
    value = os.environ.get(name, "")
    if not value and required:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def _connect(*, host: str, port: int, dbname: str, user: str, password: str, sslmode: str, autocommit: bool):
    kwargs = {
        "dbname": dbname,
        "user": user,
        "sslmode": sslmode,
    }
    if host:
        kwargs["host"] = host
        kwargs["port"] = port
    if password:
        kwargs["password"] = password
    conn = psycopg.connect(**kwargs)
    conn.autocommit = autocommit
    return conn


def provision(args: argparse.Namespace) -> None:
    admin_password = _password_from_env(args.admin_password_env, required=False)
    app_password = _password_from_env(args.app_password_env)

    if args.dry_run:
        print(f"[dry-run] ensure role {args.app_user}")
        print(f"[dry-run] ensure database {args.db_name}")
        print(f"[dry-run] apply dispatcher schema in {args.db_name}")
        return

    with _connect(
        host=args.admin_host,
        port=args.admin_port,
        dbname=args.admin_db,
        user=args.admin_user,
        password=admin_password,
        sslmode=args.sslmode,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (args.app_user,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                        sql.Identifier(args.app_user),
                        sql.Literal(app_password),
                    )
                )
            else:
                cur.execute(
                    sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                        sql.Identifier(args.app_user),
                        sql.Literal(app_password),
                    )
                )

            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (args.db_name,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} OWNER {}").format(
                        sql.Identifier(args.db_name),
                        sql.Identifier(args.app_user),
                    )
                )

    with _connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.admin_user,
        password=admin_password,
        sslmode=args.sslmode,
        autocommit=False,
    ) as conn:
        with conn.cursor() as cur:
            if args.recreate_schema:
                cur.execute("DROP TABLE IF EXISTS batch_hooks")
                cur.execute("DROP TABLE IF EXISTS batches")
                cur.execute("DROP TABLE IF EXISTS slides")
                cur.execute("DROP TABLE IF EXISTS remote_agents")
                cur.execute("DROP TABLE IF EXISTS coordination_leases")
                cur.execute("DROP TABLE IF EXISTS _schema_migrations")
            for stmt in SCHEMA_STATEMENTS:
                cur.execute(stmt)
            cur.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(args.db_name),
                    sql.Identifier(args.app_user),
                )
            )
            cur.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {}").format(
                    sql.Identifier(args.app_user),
                )
            )
            conn.commit()

    with _connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.app_user,
        password=app_password,
        sslmode=args.sslmode,
        autocommit=False,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM slides")
            cur.fetchone()
            conn.commit()

    print(f"Provisioned database {args.db_name} for role {args.app_user} on {args.db_host}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-host", default="")
    parser.add_argument("--admin-port", type=int, default=5432)
    parser.add_argument("--admin-db", default="postgres")
    parser.add_argument("--admin-user", required=True)
    parser.add_argument("--admin-password-env", default="")
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--app-user", required=True)
    parser.add_argument("--app-password-env", required=True)
    parser.add_argument("--sslmode", default="prefer")
    parser.add_argument("--recreate-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    provision(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
