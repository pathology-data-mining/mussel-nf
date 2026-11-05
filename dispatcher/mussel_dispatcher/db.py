"""Database helpers for dispatcher state backends."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional dependency in sqlite-only test envs
    psycopg = None
    dict_row = None


@dataclass(frozen=True)
class ResolvedDatabaseConfig:
    backend: str
    sqlite_path: str = ""
    host: str = ""
    port: int = 5432
    database: str = ""
    user: str = ""
    password_env: str = ""
    sslmode: str = "prefer"

    @property
    def password(self) -> str:
        return os.environ.get(self.password_env, "") if self.password_env else ""

    def location(self) -> str:
        if self.backend == "postgres":
            return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"
        return self.sqlite_path


def resolve_database_config(cfg: "Config") -> ResolvedDatabaseConfig:
    backend = (cfg.state_backend or "sqlite").strip().lower()
    if backend == "postgres":
        return ResolvedDatabaseConfig(
            backend="postgres",
            host=cfg.postgres_host,
            port=int(cfg.postgres_port or 5432),
            database=cfg.postgres_database,
            user=cfg.postgres_user,
            password_env=cfg.postgres_password_env,
            sslmode=cfg.postgres_sslmode or "prefer",
        )
    return ResolvedDatabaseConfig(backend="sqlite", sqlite_path=cfg.state_db_path())


def connect_database(
    db_cfg: ResolvedDatabaseConfig,
    *,
    readonly: bool = False,
    check_same_thread: bool = False,
) -> Any:
    if db_cfg.backend == "postgres":
        if psycopg is None:  # pragma: no cover - exercised only without optional dep
            raise RuntimeError("psycopg is required for PostgreSQL state backend")
        connect_kwargs = dict(
            host=db_cfg.host,
            port=db_cfg.port,
            dbname=db_cfg.database,
            user=db_cfg.user,
            password=db_cfg.password,
            sslmode=db_cfg.sslmode,
            gssencmode=os.environ.get("MUSSEL_POSTGRES_GSSENCMODE", "disable"),
            application_name=os.environ.get(
                "MUSSEL_POSTGRES_APPLICATION_NAME",
                "mussel-dispatcher-ro" if readonly else "mussel-dispatcher",
            ),
            row_factory=dict_row,
        )
        connect_timeout = os.environ.get("MUSSEL_POSTGRES_CONNECT_TIMEOUT", "").strip()
        if connect_timeout:
            connect_kwargs["connect_timeout"] = int(connect_timeout)
        else:
            connect_kwargs["connect_timeout"] = 5

        if readonly:
            options = os.environ.get("MUSSEL_POSTGRES_READONLY_OPTIONS", "").strip()
            default_statement_timeout_ms = 5000
        else:
            options = os.environ.get("MUSSEL_POSTGRES_OPTIONS", "").strip()
            default_statement_timeout_ms = 15000
        if not options:
            statement_timeout_ms = int(
                os.environ.get(
                    "MUSSEL_POSTGRES_STATEMENT_TIMEOUT_MS",
                    str(default_statement_timeout_ms),
                )
            )
            lock_timeout_ms = int(os.environ.get("MUSSEL_POSTGRES_LOCK_TIMEOUT_MS", "3000"))
            idle_tx_timeout_ms = int(
                os.environ.get("MUSSEL_POSTGRES_IDLE_TX_TIMEOUT_MS", "10000")
            )
            options = (
                f"-c statement_timeout={statement_timeout_ms} "
                f"-c lock_timeout={lock_timeout_ms} "
                f"-c idle_in_transaction_session_timeout={idle_tx_timeout_ms}"
            )
        connect_kwargs["options"] = options

        conn = psycopg.connect(**connect_kwargs)
        # State reads must not retain an implicit transaction between polling
        # cycles. Writes explicitly use BEGIN/COMMIT in StateStore._run_write.
        conn.autocommit = True
        return conn

    conn = sqlite3.connect(
        db_cfg.sqlite_path,
        check_same_thread=check_same_thread,
        timeout=60,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
    return conn


def sql_for_backend(db_cfg: ResolvedDatabaseConfig, sql: str) -> str:
    if db_cfg.backend == "postgres":
        return sql.replace("?", "%s")
    return sql


def placeholders_for_backend(db_cfg: ResolvedDatabaseConfig, n: int) -> str:
    token = "%s" if db_cfg.backend == "postgres" else "?"
    return ",".join(token for _ in range(n))
