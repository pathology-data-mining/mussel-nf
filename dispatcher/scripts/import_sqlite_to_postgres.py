#!/usr/bin/env python3
"""Import dispatcher state from SQLite into a Postgres backend.

Usage:
  python dispatcher/scripts/import_sqlite_to_postgres.py \
    --sqlite-db /path/to/dispatcher.db \
    --config dispatcher/cohort_dispatcher.yaml
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import psycopg

from mussel_dispatcher.config import Config
from mussel_dispatcher.db import resolve_database_config


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


def _read_rows(sqlite_db: str, table: str) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(sqlite_db)
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


def import_state(sqlite_db: str, cfg: Config) -> None:
    with _pg_connect(cfg) as conn:
        with conn.cursor() as cur:
            slide_cols, slide_rows = _read_rows(sqlite_db, "slides")
            _upsert(cur, "slides", slide_cols, slide_rows, ["slide_path"])

            batch_cols, batch_rows = _read_rows(sqlite_db, "batches")
            _upsert(cur, "batches", batch_cols, batch_rows, ["batch_id"])

            hook_cols, hook_rows = _read_rows(sqlite_db, "batch_hooks")
            if hook_cols:
                _upsert(cur, "batch_hooks", hook_cols, hook_rows, ["batch_id", "hook_index"])

            conn.commit()

            slide_count = cur.execute("SELECT COUNT(*) FROM slides").fetchone()[0]
            batch_count = cur.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
            hook_count = cur.execute("SELECT COUNT(*) FROM batch_hooks").fetchone()[0]
    print(f"Imported slides={slide_count} batches={batch_count} hooks={hook_count}")


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
