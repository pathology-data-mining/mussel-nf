#!/usr/bin/env python3
"""Provision a Postgres database and schema for mussel-dispatcher.

Creates:
1. Application role
2. Application database (owned by the application role)
3. Dispatcher tables and indexes, created by the application role through StateStore,
   so it owns them and its schema migrations can ALTER them

Usage:
  python dispatcher/scripts/provision_postgres.py \
    --admin-host db.example.org \
    --admin-user postgres \
    --admin-password-env POSTGRES_ADMIN_PASSWORD \
    --db-host db.example.org \
    --db-name mussel_dispatcher \
    --app-user mussel_dispatcher \
    --app-password-env POSTGRES_APP_PASSWORD
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg
from psycopg import sql


def schema_tables() -> list[str]:
    """Every table the dispatcher's Postgres schema defines (the state module is the
    source of truth; this script used to keep its own, stale copy)."""
    from mussel_dispatcher.state import _PG_SCHEMA, _SLIDE_ROLLUPS_DDL, _schema_table_columns
    return sorted(_schema_table_columns(_PG_SCHEMA + ";" + _SLIDE_ROLLUPS_DDL))


def create_schema_as_app_user(*, host: str, port: int, db_name: str, app_user: str,
                              app_password: str, sslmode: str) -> None:
    """Create and migrate the schema with StateStore, connected as the app user, so the
    app owns every table: its later ALTER TABLE migrations must be run by the owner."""
    from mussel_dispatcher.config import Config, PostgresConfig
    from mussel_dispatcher.state import StateStore
    env_name = "MUSSEL_PROVISION_APP_PASSWORD"
    os.environ[env_name] = app_password
    try:
        StateStore(Config(
            nextflow_profiles="", outdir="", state_backend="postgres",
            postgres=PostgresConfig(host=host, port=port, database=db_name, user=app_user,
                                    password_env=env_name, sslmode=sslmode),
        ), strict_schema=True)
    finally:
        os.environ.pop(env_name, None)


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
            tables = schema_tables()
            if args.recreate_schema:
                # Every table of the schema, so no stale rollups, breakers or
                # deployment state survive a recreate.
                for table in tables:
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
            cur.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(args.db_name),
                    sql.Identifier(args.app_user),
                )
            )
            cur.execute(sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(sql.Identifier(args.app_user)))
            # Tables an older provisioning created as the admin: hand them to the app,
            # whose StateStore migrations ALTER them.
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            for (table,) in cur.fetchall():
                if table in tables:
                    cur.execute(sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                        sql.Identifier(table), sql.Identifier(args.app_user)))
            conn.commit()

    create_schema_as_app_user(
        host=args.db_host, port=args.db_port, db_name=args.db_name, app_user=args.app_user,
        app_password=app_password, sslmode=args.sslmode,
    )

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
