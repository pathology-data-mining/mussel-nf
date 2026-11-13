"""The WDS append lock against a real, throwaway PostgreSQL server.

Starts a private server (initdb into a temp dir, unix socket only, trust auth)
and stops it afterwards. Skipped when initdb/pg_ctl or psycopg are unavailable;
set MUSSEL_SKIP_PG_TESTS=1 to skip it anyway. Never touches a configured server.
"""
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from mussel_dispatcher import wds as wds_mod
from mussel_dispatcher.db import ResolvedDatabaseConfig, connect_database

psycopg = pytest.importorskip("psycopg")

INITDB, PG_CTL = shutil.which("initdb"), shutil.which("pg_ctl")
pytestmark = pytest.mark.skipif(
    not (INITDB and PG_CTL) or os.environ.get("MUSSEL_SKIP_PG_TESTS") == "1",
    reason="no local PostgreSQL binaries (initdb/pg_ctl)",
)

# Keys covering both signs and both halves >= 2^31 (classid and objid are oids).
KEYS = [42, -1234567890123, 2**63 - 1, -(2**63), 0x0000000180000000, -1]


@pytest.fixture(scope="module")
def pg_cfg(tmp_path_factory):
    data = tmp_path_factory.mktemp("pgdata") / "data"
    # The unix socket path must stay under ~107 bytes, so it gets a short dir.
    sock = Path(tempfile.mkdtemp(prefix="pgsock-", dir="/tmp"))
    port = random.randint(20000, 60000)
    subprocess.run([INITDB, "-D", str(data), "-A", "trust", "-U", "postgres", "-N"],
                   check=True, capture_output=True)
    subprocess.run([PG_CTL, "-D", str(data), "-w", "-l", str(data.parent / "server.log"),
                    "-o", f"-k {sock} -p {port} -c listen_addresses=''", "start"],
                   check=True, capture_output=True)
    try:
        yield ResolvedDatabaseConfig(backend="postgres", host=str(sock), port=port,
                                     database="postgres", user="postgres", sslmode="disable")
    finally:
        subprocess.run([PG_CTL, "-D", str(data), "-m", "immediate", "-w", "stop"],
                       capture_output=True)
        shutil.rmtree(sock, ignore_errors=True)


def _held(conn, key):
    return conn.execute(wds_mod._ADVISORY_LOCK_HELD_SQL, (key,)).fetchone()["held"]


@pytest.mark.parametrize("key", KEYS)
def test_held_query_matches_pg_advisory_lock(pg_cfg, key):
    owner, other = connect_database(pg_cfg), connect_database(pg_cfg)
    try:
        assert _held(owner, key) is False
        owner.execute("SELECT pg_advisory_lock(%s)", (key,))
        assert _held(owner, key) is True
        assert _held(other, key) is False  # another session does not hold it
        assert _held(owner, key ^ 1) is False  # a different key is not held
        owner.execute("SELECT pg_advisory_unlock(%s)", (key,))
        assert _held(owner, key) is False
    finally:
        owner.close()
        other.close()


def test_verify_lock_detects_a_terminated_lock_session(pg_cfg):
    admin = connect_database(pg_cfg)
    try:
        with wds_mod._wds_postgres_lock(pg_cfg, "s3://bucket/wds", "optimus") as verify_lock:
            verify_lock()
            key = wds_mod._postgres_lock_key("s3://bucket/wds::optimus")
            pids = [r["pid"] for r in admin.execute(
                "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted"
                " AND ((classid::bigint << 32) | objid::bigint) = %s", (key,)).fetchall()]
            assert len(pids) == 1
            admin.execute("SELECT pg_terminate_backend(%s)", (pids[0],))
            with pytest.raises(wds_mod.WdsLockLost, match="WDS append lock lost"):
                verify_lock()
        # Leaving the context after the loss does not raise, and the lock is free.
        admin.execute("SELECT pg_advisory_lock(%s)", (key,))
        assert _held(admin, key) is True
    finally:
        admin.close()


def test_lock_session_settings_apply(pg_cfg):
    """The keepalive SETs succeed; idle_session_timeout is skipped below PG 14."""
    with wds_mod._wds_postgres_lock(pg_cfg, "s3://bucket/wds", "settings") as verify_lock:
        verify_lock()
