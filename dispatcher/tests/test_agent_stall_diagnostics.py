"""Remote agent main-loop stalls: evidence on a watchdog trip, and bounded DB sockets.

A remote agent repeatedly exited with "main loop stalled" and no record
of what blocked. Each stall began as a driver completed and its finalization
started; the finalization worker's first write ran on a session left idle since
its previous batch, which the network had dropped without a reset.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from mussel_dispatcher import remote as remote_mod
from mussel_dispatcher.remote import RemoteAgent

DISPATCHER_DIR = Path(__file__).resolve().parents[1]


def _agent(progress_age_s: float) -> RemoteAgent:
    agent = object.__new__(RemoteAgent)
    stamp = time.monotonic() - progress_age_s
    agent._main_loop_progress = lambda: ("2026-01-01T12:00:00+00:00", stamp)
    return agent


def test_watchdog_trip_logs_the_phase_and_every_thread_stack(monkeypatch, caplog):
    agent = _agent(progress_age_s=400)
    agent._set_main_loop_phase("refresh_order:launch_finalization", "20260101T120000_abcd1234")
    exits = []
    monkeypatch.setattr(remote_mod, "_hard_exit", exits.append)
    release = threading.Event()

    def blocked_on_postgres_write():
        release.wait(10)

    worker = threading.Thread(target=blocked_on_postgres_write,
                              name="remote-post-batch-finalize_0", daemon=True)
    worker.start()
    try:
        with caplog.at_level(logging.ERROR, logger="mussel-dispatcher"):
            assert agent._check_main_loop_stall(300) is True
    finally:
        release.set()
        worker.join(5)

    assert exits == [70]
    assert re.search(
        r"stalled for 4\d\ds \(last progress 2026-01-01T12:00:00\+00:00; "
        r"phase refresh_order:launch_finalization batch 20260101T120000_abcd1234 for \d+s\)",
        caplog.text,
    ), caplog.text
    assert "--- thread remote-post-batch-finalize_0" in caplog.text
    assert "blocked_on_postgres_write" in caplog.text
    assert "_check_main_loop_stall" in caplog.text        # the watchdog's own thread too


def test_watchdog_does_nothing_while_the_loop_progresses(monkeypatch, caplog):
    agent = _agent(progress_age_s=10)
    exits = []
    monkeypatch.setattr(remote_mod, "_hard_exit", exits.append)
    with caplog.at_level(logging.ERROR, logger="mussel-dispatcher"):
        assert agent._check_main_loop_stall(300) is False
    assert exits == [] and caplog.text == ""


def test_refresh_records_the_order_being_refreshed(monkeypatch):
    agent = object.__new__(RemoteAgent)
    agent.cfg = MagicMock(state_backend="postgres")
    agent.target = MagicMock()
    agent.state = MagicMock()
    agent.state.get_remote_batches_for_target.return_value = [
        {"batch_id": "batch-1"}, {"batch_id": "batch-2"}]
    seen = []
    agent._refresh_order = lambda order, client, cycle: seen.append(agent._main_loop_phase[:2])

    agent._refresh_claimed_orders()

    assert seen == [("refresh_order", "batch-1"), ("refresh_order", "batch-2")]


def test_sigusr1_dumps_stacks_and_log_lines_carry_timestamps():
    code = (
        "import logging, os, signal\n"
        "from mussel_dispatcher.remote import _install_process_diagnostics\n"
        "_install_process_diagnostics()\n"
        "os.kill(os.getpid(), signal.SIGUSR1)\n"
        "logging.getLogger('mussel-dispatcher').warning('still running')\n"
        "logging.getLogger('botocore.credentials').info('Found credentials in shared credentials file')\n"
        "logging.getLogger('mussel-dispatcher').info('agent info line')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=DISPATCHER_DIR,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr        # SIGUSR1 would otherwise kill it
    assert "most recent call first" in result.stderr    # faulthandler's dump
    assert re.search(
        r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4} WARNING +MainThread mussel-dispatcher: "
        r"still running$",
        result.stderr, re.MULTILINE,
    ), result.stderr
    assert "agent info line" in result.stderr                    # the agent's INFO is kept
    assert "Found credentials" not in result.stderr              # the S3 stack's INFO is not


def _connect_kwargs(monkeypatch, libpq_version: int, **env) -> dict:
    import mussel_dispatcher.db as db_mod

    calls = {}

    class DummyConn:
        closed = False
        autocommit = False

    def fake_connect(**kwargs):
        calls.update(kwargs)
        return DummyConn()

    fake = MagicMock(connect=fake_connect)
    fake.pq.version.return_value = libpq_version
    monkeypatch.setattr(db_mod, "psycopg", fake)
    monkeypatch.setattr(db_mod, "dict_row", object())
    for name in ("MUSSEL_POSTGRES_KEEPALIVES_IDLE", "MUSSEL_POSTGRES_TCP_USER_TIMEOUT_MS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    db_mod.connect_database(db_mod.ResolvedDatabaseConfig(
        backend="postgres", host="db.example", port=5433, database="dispatcher", user="app"))
    return calls


def test_postgres_sessions_bound_a_dead_socket(monkeypatch):
    calls = _connect_kwargs(monkeypatch, 180000)
    assert calls["keepalives"] == 1
    assert calls["keepalives_idle"] == 60       # an idle flow is never silent for long
    assert calls["keepalives_interval"] == 10 and calls["keepalives_count"] == 3
    assert calls["tcp_user_timeout"] == 30000   # an unacknowledged send fails in 30 s


def test_tcp_user_timeout_is_skipped_on_old_libpq_and_can_be_disabled(monkeypatch):
    assert "tcp_user_timeout" not in _connect_kwargs(monkeypatch, 110000)
    assert "tcp_user_timeout" not in _connect_kwargs(
        monkeypatch, 180000, MUSSEL_POSTGRES_TCP_USER_TIMEOUT_MS="0")
