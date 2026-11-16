#!/usr/bin/env python3
"""Import dispatcher state from SQLite into a Postgres backend.

Usage:
  python dispatcher/scripts/import_sqlite_to_postgres.py \
    --sqlite-db /path/to/dispatcher.db \
    --config dispatcher/dispatcher.yaml
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import psycopg

from mussel_dispatcher.config import Config
from mussel_dispatcher.db import resolve_database_config
from mussel_dispatcher.state import SLIDE_ROLLUPS_REBUILD_SQL

# (table, conflict columns), in dependency order.
TABLES = (
    ("slides", ["slide_path"]),
    ("batches", ["batch_id"]),
    ("batch_hooks", ["batch_id", "hook_index"]),
    ("hook_attempts", ["batch_id", "hook_index", "attempt_number"]),
)


def _pg_connect(cfg):
    db_cfg = resolve_database_config(cfg)
    return psycopg.connect(
        host=db_cfg.host,
        port=db_cfg.port,
        dbname=db_cfg.database,
        user=db_cfg.user,
        password=db_cfg.password,
        sslmode=db_cfg.sslmode,
    )


def _open_sqlite(sqlite_db: str) -> sqlite3.Connection:
    """Read-only: a mistyped path must fail, not create an empty database to import."""
    if not os.path.isfile(sqlite_db):
        raise FileNotFoundError(f"SQLite state database not found: {sqlite_db}")
    return sqlite3.connect(f"file:{os.path.abspath(sqlite_db)}?mode=ro", uri=True)


def _read_rows(sqlite_db: str, table: str) -> tuple[list[str], list[tuple]]:
    conn = _open_sqlite(sqlite_db)
    try:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if not cols:
            return [], []
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        return cols, rows
    finally:
        conn.close()


def _upsert(cur, table: str, columns: list[str], rows: list[tuple], conflict_cols: list[str]) -> None:
    if not rows:
        return
    col_csv = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_cols
    )
    sql = (
        f"INSERT INTO {table} ({col_csv}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET {updates}"
    )
    cur.executemany(sql, rows)


def import_state(sqlite_db: str, cfg: Config) -> dict[str, int]:
    """Upsert every state table, then rebuild slide_rollups; returns the row counts."""
    _open_sqlite(sqlite_db).close()          # fail before touching Postgres
    counts = {}
    with _pg_connect(cfg) as conn:
        with conn.cursor() as cur:
            for table, conflict in TABLES:
                cols, rows = _read_rows(sqlite_db, table)
                if not cols:
                    continue                 # an older SQLite schema without this table
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = %s", (table,))
                pg_cols = {r[0] for r in cur.fetchall()}
                keep = [i for i, c in enumerate(cols) if c in pg_cols]
                _upsert(cur, table, [cols[i] for i in keep],
                        [tuple(row[i] for i in keep) for row in rows], conflict)
            # Provisioning ran the rollup migration on an empty database, so the
            # imported slides have none: rebuild them all.
            cur.execute("DELETE FROM slide_rollups")
            cur.execute(SLIDE_ROLLUPS_REBUILD_SQL)
            conn.commit()
            for table, _ in TABLES + (("slide_rollups", []),):
                counts[table] = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print("Imported " + " ".join(f"{t}={n}" for t, n in counts.items()))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-db", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if (cfg.state_backend or "sqlite").lower() != "postgres":
        raise RuntimeError("Target config must set state_backend: postgres")
    import_state(args.sqlite_db, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
