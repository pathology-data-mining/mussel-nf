from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from threading import Event
from pathlib import Path
from unittest.mock import MagicMock

from mussel_dispatcher.config import Config
from mussel_dispatcher.hook_hardening import (
    HookFailureCategory,
    classify_hook_failure,
    preflight_hook,
)
from mussel_dispatcher.state import StateStore
from mussel_dispatcher.config import RemoteTargetConfig
from mussel_dispatcher.batch_lifecycle import PostBatchHooks, _resolve_hook_runtime
from mussel_dispatcher.remote import RemoteAgent


def test_classify_hook_failure_separates_coverage_from_environment():
    assert classify_hook_failure(
        2,
        "missing slide output",
        hook_command="python -m mussel_dispatcher.wds",
    ).category == HookFailureCategory.COVERAGE_MISSING
    assert classify_hook_failure(
        1,
        "ModuleNotFoundError: No module named mussel_dispatcher",
    ).category == HookFailureCategory.RELEASE_OR_CONFIG
    assert classify_hook_failure(1, "Slurm hook state=OUT_OF_MEMORY").category == HookFailureCategory.NODE_INFRA
    assert classify_hook_failure(1, "database connection timed out").category == HookFailureCategory.TRANSIENT_SERVICE
    assert classify_hook_failure(
        78,
        "[mussel-hook-preflight] category=NODE_INFRA detail=batch-csv-unreadable",
    ).category == HookFailureCategory.NODE_INFRA



def test_classify_hook_failure_ignores_paths_in_tracebacks():
    key_error = (
        "Traceback (most recent call last):\n"
        '  File "/opt/mussel-nf/dispatcher/mussel_dispatcher/wds.py", '
        "line 1510, in append_wds\n"
        "KeyError: 'shard_file'"
    )
    assert classify_hook_failure(1, key_error).category == HookFailureCategory.HOOK_BUG
    read_timeout = (
        "Traceback (most recent call last):\n"
        '  File "/opt/envs/mussel/lib/python3.11/site-packages/botocore/'
        'httpsession.py", line 501, in send\n'
        "botocore.exceptions.ReadTimeoutError: Read timeout on endpoint URL: "
        '"https://s3.example.org/bucket/wds/optimus/wds_index.json"'
    )
    assert classify_hook_failure(1, read_timeout).category == HookFailureCategory.TRANSIENT_SERVICE

def test_preflight_imports_dispatcher_from_repo(tmp_path):
    repo = tmp_path / "repo"
    package = repo / "dispatcher" / "mussel_dispatcher"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n")
    (package / "wds.py").write_text("\n")
    batch_csv = tmp_path / "batch.csv"
    batch_csv.write_text("slide_id,slide_path\na,/slides/a.svs\n")

    assert preflight_hook(
        command=[sys.executable, "-m", "mussel_dispatcher.wds"],
        cwd=str(repo),
        env={},
        paths=(str(batch_csv), str(repo)),
        active_revision="",
    ) is None


def test_preflight_import_is_cached_per_release_runtime(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    package = repo / "dispatcher" / "mussel_dispatcher"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n")
    (package / "wds.py").write_text("\n")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr("mussel_dispatcher.hook_hardening.subprocess.run", fake_run)
    command = [sys.executable, "-m", "mussel_dispatcher.wds"]
    assert preflight_hook(command=command, cwd=str(repo), env={}) is None
    assert preflight_hook(command=command, cwd=str(repo), env={}) is None
    assert len(calls) == 1


def test_hook_circuit_breaker_persists_and_clears(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    for batch_id in ("batch-1", "batch-2"):
        assert not state.record_hook_failure(
            "hpc", "RELEASE_OR_CONFIG", "bad import",
            batch_id=batch_id, hook_name="wds:uni2h", threshold=3,
        )
    assert state.record_hook_failure(
        "hpc", "RELEASE_OR_CONFIG", "bad import",
        batch_id="batch-3", hook_name="wds:uni2h", threshold=3,
    )
    status = state.hook_circuit_status("hpc")
    assert status["paused"] == 1
    assert status["category"] == "RELEASE_OR_CONFIG"
    state.clear_hook_circuit("hpc")
    assert state.hook_circuit_status("hpc")["paused"] == 0


def test_hook_circuit_deduplicates_retries_and_canary_is_single_batch(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    for _ in range(3):
        assert not state.record_hook_failure(
            "hpc", "RELEASE_OR_CONFIG", "bad import",
            batch_id="same-batch", hook_name="wds:uni2h", threshold=3,
        )
    assert state.hook_circuit_status("hpc") is not None
    assert state.hook_circuit_status("hpc")["paused"] == 0
    for batch_id in ("batch-1", "batch-2", "batch-3"):
        state.record_hook_failure(
            "hpc", "RELEASE_OR_CONFIG", "bad import",
            batch_id=batch_id, hook_name="wds:uni2h", threshold=3,
        )
    state.begin_hook_canary("hpc")
    assert state.hook_circuit_status("hpc")["state"] == "HALF_OPEN"
    assert state.claim_hook_canary("hpc", "canary-1")
    assert not state.claim_hook_canary("hpc", "canary-2")
    state.finish_hook_canary("hpc", "canary-1", success=True)
    assert state.hook_circuit_status("hpc")["state"] == "CLOSED"


def test_hook_attempt_history_survives_command_change(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    state.sync_batch_hooks("batch-1", [{
        "hook_index": 0, "phase": "wds", "hook_name": "wds:test",
        "command": "false", "required": 1,
    }])
    state.start_batch_hook("batch-1", 0)
    state.complete_batch_hook("batch-1", 0, exit_code=1, error_msg="first")
    state.sync_batch_hooks("batch-1", [{
        "hook_index": 0, "phase": "wds", "hook_name": "wds:test",
        "command": "true", "required": 1,
    }])
    state.start_batch_hook("batch-1", 0)
    state.complete_batch_hook("batch-1", 0, exit_code=0)
    attempts = state._conn().execute(
        "SELECT attempt_number, status FROM hook_attempts "
        "WHERE batch_id='batch-1' AND hook_index=0 ORDER BY attempt_number"
    ).fetchall()
    assert [tuple(row) for row in attempts] == [(1, "FAILED"), (2, "SUCCEEDED")]


def test_each_hook_gets_its_own_idempotency_key(tmp_path):
    cfg = Config(
        repo_dir=str(tmp_path), nextflow_profiles="standard",
        outdir=str(tmp_path / "results"), state_dir=str(tmp_path / "state"),
        post_batch_hooks=[
            {"command": "true", "_phase": "wds"},
            {"command": "true", "_phase": "wds"},
        ], hook_preflight=False,
    )
    state = StateStore(str(tmp_path / "state.db"))
    seen = []

    def executor(cmd, *, cwd, env, hook_index):
        del cmd, cwd, hook_index
        seen.append(env["MUSSEL_HOOK_IDEMPOTENCY_KEY"])
        return subprocess.CompletedProcess([], 0, "", "")

    hooks = PostBatchHooks(cfg, "batch-1", state, target="local", runtime_revision="rev")
    assert hooks._run_unlocked(str(tmp_path / "batch.csv"), executor=executor)
    assert len(seen) == 2
    assert seen[0] != seen[1]


def test_canary_spans_wds_and_db_sync_phases(tmp_path):
    cfg = Config(
        repo_dir=str(tmp_path), nextflow_profiles="standard",
        outdir=str(tmp_path / "results"), state_dir=str(tmp_path / "state"),
        post_batch_hooks=[
            {"command": "true", "_phase": "wds"},
            {"command": "true", "_phase": "db_sync"},
        ], hook_preflight=False,
    )
    state = StateStore(str(tmp_path / "state.db"))
    state.begin_hook_canary("hpc")
    hooks = PostBatchHooks(cfg, "batch-1", state, target="hpc", runtime_revision="rev")
    executor = lambda cmd, *, cwd, env, hook_index: subprocess.CompletedProcess([], 0, "", "")
    assert hooks.run(str(tmp_path / "batch.csv"), phase="wds", executor=executor)
    assert state.hook_circuit_status("hpc")["state"] == "HALF_OPEN"
    assert hooks.run(str(tmp_path / "batch.csv"), phase="db_sync", executor=executor)
    assert state.hook_circuit_status("hpc")["state"] == "CLOSED"


def test_old_paused_breaker_schema_migrates_to_open_state(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE hook_circuit_breakers ("
        "target TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT '', "
        "failure_times TEXT NOT NULL DEFAULT '[]', paused INTEGER NOT NULL DEFAULT 0, "
        "reason TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO hook_circuit_breakers(target, category, paused, reason, updated_at) "
        "VALUES ('hpc', 'RELEASE_OR_CONFIG', 1, 'legacy pause', 'now')"
    )
    conn.commit()
    conn.close()
    state = StateStore(str(db_path))
    status = state.hook_circuit_status("hpc")
    assert status["state"] == "OPEN"


def test_hook_failure_diagnostics_are_persisted(tmp_path):
    cfg = Config(
        nextflow_profiles="standard",
        repo_dir=str(tmp_path),
        outdir=str(tmp_path / "results"),
        state_dir=str(tmp_path / "state"),
        post_batch_hooks=[{"command": "false", "_phase": "wds"}],
        hook_preflight=False,
    )
    state = StateStore(str(tmp_path / "state.db"))
    state.sync_batch_hooks("batch-1", [{
        "hook_index": 0,
        "phase": "wds",
        "hook_name": "wds:test",
        "command": "false",
        "required": 1,
    }])
    state.start_batch_hook("batch-1", 0)
    state.complete_batch_hook(
        "batch-1",
        0,
        exit_code=78,
        error_msg="bad release",
        failure_category=HookFailureCategory.RELEASE_OR_CONFIG,
        failure_detail="bad release",
        runtime_revision="release-a",
        execution_node="node031",
    )
    row = state.get_batch_hooks(["batch-1"])["batch-1"][0]
    assert row["failure_category"] == HookFailureCategory.RELEASE_OR_CONFIG
    assert row["runtime_revision"] == "release-a"
    assert row["execution_node"] == "node031"
    assert row["idempotency_key"]
    attempt = state._conn().execute(
        "SELECT status, failure_category, execution_node FROM hook_attempts "
        "WHERE batch_id='batch-1' AND hook_index=0 AND attempt_number=1"
    ).fetchone()
    assert tuple(attempt) == ("FAILED", HookFailureCategory.RELEASE_OR_CONFIG, "node031")


def test_remote_hook_excludes_node_from_previous_infra_failure(tmp_path, monkeypatch):
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.target = RemoteTargetConfig(name="hpc", hook_exclude="node000")
    agent.state = MagicMock()
    agent.state.get_batch_hooks.return_value = {
        "batch-1": [{
            "hook_index": 0,
            "executor_job_id": "",
            "failure_category": HookFailureCategory.NODE_INFRA,
            "execution_node": "node031",
        }]
    }
    order = {
        "batch_id": "batch-1",
        "remote_run_dir": str(tmp_path / "run"),
        "remote_batch_csv": str(tmp_path / "run" / "batch.csv"),
    }
    (tmp_path / "run").mkdir()
    calls = []
    submitted_script = {}

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == "sbatch":
            submitted_script["text"] = Path(args[-1]).read_text()
            return subprocess.CompletedProcess(args, 0, "12345\n", "")
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "sacct":
            return subprocess.CompletedProcess(args, 0, "COMPLETED|0:0|node032\n", "")
        raise AssertionError(args)

    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", fake_run)
    result = agent._run_hook_slurm(
        order,
        ["python", "-m", "mussel_dispatcher.wds"],
        cwd=str(tmp_path),
        env={},
        hook_index=0,
    )
    assert result.returncode == 0
    sbatch = next(call for call in calls if call[0] == "sbatch")
    assert sbatch[sbatch.index("--exclude") + 1] == "node000,node031"
    assert "preflight node" in submitted_script["text"]


def test_remote_hook_refuses_new_submission_without_exact_lease(tmp_path, monkeypatch):
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.target = RemoteTargetConfig(name="hpc")
    agent._finalization_owner = "agent-a"
    agent._finalization_lease_names = {
        "batch-1": "dispatcher-finalization:remote"
    }
    agent.state = MagicMock()
    agent.state.get_batch_hooks.return_value = {"batch-1": []}
    agent.state.owns_finalization_lease.return_value = False
    order = {"batch_id": "batch-1", "remote_run_dir": str(tmp_path / "run")}
    (tmp_path / "run").mkdir()
    run = MagicMock()
    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", run)

    result = agent._run_hook_slurm(
        order, ["hook"], cwd=str(tmp_path), env={}, hook_index=0
    )

    assert result.returncode == 78
    assert "finalization-lease-lost" in result.stderr
    run.assert_not_called()


def test_remote_hook_can_poll_persisted_job_after_ownership_loss(tmp_path, monkeypatch):
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.target = RemoteTargetConfig(name="hpc", finalization_poll_interval_seconds=1)
    agent.state = MagicMock()
    agent.state.get_batch_hooks.return_value = {
        "batch-1": [{
            "hook_index": 0,
            "status": "RUNNING",
            "executor_job_id": "12345",
            "active_attempt_number": 7,
        }]
    }
    order = {"batch_id": "batch-1", "remote_run_dir": str(tmp_path / "run")}
    (tmp_path / "run" / "hooks").mkdir(parents=True)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "sacct":
            return subprocess.CompletedProcess(args, 0, "COMPLETED|0:0|node001\n", "")
        raise AssertionError(args)

    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", fake_run)
    lost = Event()
    lost.set()

    result = agent._run_hook_slurm(
        order, ["hook"], cwd=str(tmp_path), env={}, hook_index=0,
        ownership_lost=lost,
    )

    assert result.returncode == 0
    assert all(call[0] != "sbatch" for call in calls)
    agent.state.owns_finalization_lease.assert_not_called()


def test_heartbeat_outage_fences_next_hook_before_lease_expiry(tmp_path, monkeypatch):
    cfg = Config(
        nextflow_profiles="standard",
        repo_dir=str(tmp_path),
        outdir=str(tmp_path / "results"),
        state_dir=str(tmp_path / "state"),
    )
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.cfg = cfg
    agent.target = RemoteTargetConfig(
        name="hpc",
        hook_use_slurm=False,
        finalization_lease_ttl_seconds=1,
        finalization_poll_interval_seconds=1,
    )
    agent.state = MagicMock()
    agent.state.heartbeat_remote_finalization.side_effect = RuntimeError("db unavailable")
    agent.state.owns_finalization_lease.return_value = True
    agent._finalization_owner = "agent-a"
    agent._finalization_lease_names = {"batch-1": "slot-1"}
    agent._finalization_heartbeat_health = {}
    agent._finalization_reconcile_health = {}
    agent._agent_revision = "revision"

    class FakeFinalizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def finalize_success(self, _context, *, hook_executor):
            time.sleep(0.25)
            return hook_executor(
                ["hook"], cwd=str(tmp_path), env={}, hook_index=0
            )

    monkeypatch.setattr("mussel_dispatcher.remote.BatchFinalizer", FakeFinalizer)
    run = MagicMock()
    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", run)
    order = {
        "batch_id": "batch-1",
        "remote_run_dir": str(tmp_path / "run"),
        "remote_batch_csv": str(tmp_path / "run" / "batch.csv"),
    }

    result = agent._finalize_success(order, lease_name="slot-1")

    assert result.returncode == 78
    assert "finalization-lease-lost" in result.stderr
    run.assert_not_called()
    assert agent._finalization_reconcile_health["last_successful_heartbeat_at"]


def test_hook_config_release_selects_matching_runtime(tmp_path, monkeypatch):
    old_repo = tmp_path / "old-release"
    new_repo = tmp_path / "releases" / "new-release"
    config_path = new_repo / "dispatcher" / "cohort_dispatcher.yaml"
    (new_repo / "dispatcher" / "mussel_dispatcher").mkdir(parents=True)
    config_path.write_text("")

    def fake_run(args, **kwargs):
        assert args == ["git", "-C", str(new_repo), "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(args, 0, "new-revision\n", "")

    monkeypatch.setattr("mussel_dispatcher.batch_lifecycle.subprocess.run", fake_run)
    repo_dir, revision = _resolve_hook_runtime(
        ["python", "-m", "mussel_dispatcher.tcga.sync_databricks",
         f"--config={config_path}"],
        str(old_repo),
        "old-revision",
    )
    assert Path(repo_dir) == new_repo
    assert revision == "new-revision"
