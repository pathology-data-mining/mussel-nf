"""Regressions for dispatcher re-review findings."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mussel_dispatcher.batch_lifecycle import PostBatchHooks
from mussel_dispatcher.config import Config, RemoteTargetConfig
from mussel_dispatcher.hook_hardening import HookFailureCategory
from mussel_dispatcher.hook_launcher import LAUNCHER
from mussel_dispatcher.remote import RemoteAgent
from mussel_dispatcher.state import StateStore


def _slurm_hook_run(tmp_path, monkeypatch, *, sacct, out, err, cmd=None):
    """Run one required WDS hook through PostBatchHooks and the real Slurm executor.

    The fake sbatch writes the job's .out/.err files the way Slurm would.
    Returns (hook row, the hooks object, the sbatch calls).
    """
    cfg = Config(
        repo_dir=str(tmp_path), nextflow_profiles="standard",
        outdir=str(tmp_path / "results"), state_dir=str(tmp_path / "state"),
        post_batch_hooks=[{"command": "python -m mussel_dispatcher.wds", "_phase": "wds"}],
        hook_preflight=False,
    )
    state = StateStore(str(tmp_path / "state.db"))
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.target = RemoteTargetConfig(name="hpc", finalization_poll_interval_seconds=1)
    agent.state = state
    monkeypatch.setattr(state, "owns_finalization_lease", lambda *a, **k: True)
    order = {
        "batch_id": "batch-1",
        "remote_run_dir": str(tmp_path / "run"),
        "remote_batch_csv": str(tmp_path / "run" / "batch.csv"),
    }
    sbatches = []

    def fake_run(args, **kwargs):
        if args[0] == "sbatch":
            sbatches.append(list(args))
            Path(args[args.index("--output") + 1]).write_text(out)
            Path(args[args.index("--error") + 1]).write_text(err)
            return subprocess.CompletedProcess(args, 0, "4242\n", "")
        if args[0] == "squeue":
            # The orphan lookup by name finds nothing; the job itself has left the queue.
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "sacct":
            if "--name" in args or any(str(a).startswith("--name") for a in args):
                return subprocess.CompletedProcess(args, 0, "", "")
            return subprocess.CompletedProcess(args, 0, sacct, "")
        raise AssertionError(args)

    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", fake_run)
    hook_cmd = cmd or ["python", "-m", "mussel_dispatcher.wds"]

    def executor(_cmd, *, cwd, env, hook_index):
        return agent._run_hook_slurm(order, hook_cmd, cwd=str(tmp_path), env=env, hook_index=hook_index)

    hooks = PostBatchHooks(cfg, "batch-1", state, target="hpc", runtime_revision="rev")
    assert not hooks.run(str(tmp_path / "batch.csv"), phase="wds", executor=executor)
    row = state.get_batch_hooks(["batch-1"])["batch-1"][0]
    return row, hooks, sbatches, state


@pytest.mark.parametrize("sacct_state", ["TIMEOUT", "PREEMPTED", "NODE_FAIL"])
def test_slurm_hook_killed_by_slurm_is_node_infra_despite_a_nonempty_err(tmp_path, monkeypatch, sacct_state):
    # Finding 2: Slurm's own "CANCELLED ... DUE TO TIME LIMIT" fills the .err, which
    # used to hide the state marker and make the hook a HOOK_BUG (no automatic retry).
    row, hooks, _, state = _slurm_hook_run(
        tmp_path, monkeypatch,
        sacct=f"{sacct_state}|0:15|node031\n",
        out="[mussel-hook] preflight node=node031\nappending shards\n",
        err=f"slurmstepd: error: *** JOB 4242 ON node031 CANCELLED AT 2026-01-01T12:00:00 "
            f"DUE TO {sacct_state.replace('_', ' ')} ***\n",
    )
    assert row["failure_category"] == HookFailureCategory.NODE_INFRA
    assert row["execution_node"] == "node031"
    assert hooks._terminal_failure_reason(row) == ""
    assert state.hook_circuit_status("hpc")["category"] == HookFailureCategory.NODE_INFRA


@pytest.mark.parametrize("detail", ["batch-csv-unreadable", "dispatcher-package-missing"])
def test_slurm_hook_node_preflight_failure_is_node_infra_and_excludes_the_node(tmp_path, monkeypatch, detail):
    # Finding 3: the preflight marker is on stdout, and stderr is never empty (Slurm's
    # node line at least), so the marker was lost and the node never excluded.
    row, _, _, _ = _slurm_hook_run(
        tmp_path, monkeypatch,
        sacct="FAILED|78:0|node031\n",
        out=f"[mussel-hook] preflight node=node031\n"
            f"[mussel-hook-preflight] category=NODE_INFRA detail={detail}\n",
        err="",
    )
    assert row["failure_category"] == HookFailureCategory.NODE_INFRA
    assert row["failure_detail"] == detail
    assert row["execution_node"] == "node031"
    # The retry goes elsewhere.
    agent = RemoteAgent.__new__(RemoteAgent)
    agent.target = RemoteTargetConfig(name="hpc", finalization_poll_interval_seconds=1)
    agent.state = MagicMock()
    agent.state.get_batch_hooks.return_value = {"batch-1": [dict(row, executor_job_id="")]}
    sbatches = []

    def fake_run(args, **kwargs):
        if args[0] == "sbatch":
            sbatches.append(list(args))
            return subprocess.CompletedProcess(args, 0, "4243\n", "")
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "COMPLETED|0:0|node032\n", "")

    monkeypatch.setattr("mussel_dispatcher.remote.subprocess.run", fake_run)
    order = {"batch_id": "batch-1", "remote_run_dir": str(tmp_path / "run")}
    agent._run_hook_slurm(order, ["python", "-m", "mussel_dispatcher.wds"],
                          cwd=str(tmp_path), env={}, hook_index=0)
    assert sbatches[0][sbatches[0].index("--exclude") + 1] == "node031"


def test_slurm_hook_import_preflight_failure_stays_release_or_config(tmp_path, monkeypatch):
    row, _, _, _ = _slurm_hook_run(
        tmp_path, monkeypatch,
        sacct="FAILED|78:0|node031\n",
        out="[mussel-hook] preflight node=node031\n"
            "[mussel-hook-preflight] category=RELEASE_OR_CONFIG detail=dispatcher-import-failed\n",
        err="Traceback (most recent call last):\nModuleNotFoundError: No module named 'webdataset'\n",
    )
    assert row["failure_category"] == HookFailureCategory.RELEASE_OR_CONFIG
    assert row["failure_detail"] == "dispatcher-import-failed"


def test_slurm_hook_missing_launcher_stays_release_or_config(tmp_path, monkeypatch):
    row, _, _, _ = _slurm_hook_run(
        tmp_path, monkeypatch,
        sacct="FAILED|78:0|node031\n",
        out="[mussel-hook] preflight node=node031\n"
            "[mussel-hook-preflight] category=RELEASE_OR_CONFIG detail=hook-launcher-missing\n",
        err="",
        cmd=["python", LAUNCHER, "--release", "/remote/repo/dispatcher", "mussel_dispatcher.wds"],
    )
    assert row["failure_category"] == HookFailureCategory.RELEASE_OR_CONFIG
    assert row["failure_detail"] == "hook-launcher-missing"


def _dashboard_status(tmp_path, *, wds_staging_dir=""):
    """Drive /api/status on a SQLite dashboard with no dispatcher snapshot."""
    import json
    import time

    from mussel_dispatcher.config import WatcherConfig
    from mussel_dispatcher.dashboard.server import _build_handler
    from tests.test_dispatcher import TestTowerShimServerRoutes

    watcher = WatcherConfig(type="tcga", wds_destinations={"optimus": "s3://bucket/wds/optimus/"},
                            wds_staging_dir=wds_staging_dir)
    cfg = Config(
        repo_dir=str(tmp_path), nextflow_profiles="standard",
        outdir=str(tmp_path / "results"), state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"), watchers=[watcher],
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    state = StateStore(cfg.state_db_path())
    for slide in ("A", "B", "C"):
        state.add_slide(f"/slides/{slide}.svs", slide)
    state._conn().execute("UPDATE slides SET status='SUCCEEDED'")
    state._conn().commit()
    helper = TestTowerShimServerRoutes()
    handler = _build_handler(cfg)
    deadline = time.time() + 20
    while True:
        h, responses = helper._fake_request(handler, "GET", "/api/status")
        assert h._response_code == 200
        body = json.loads(responses[0])
        if body["status_cache"].get("age_seconds") is not None or time.time() > deadline:
            return body
        time.sleep(0.1)


def test_dashboard_status_without_snapshot_refreshes_on_sqlite(tmp_path):
    # Finding 1: _compute_status_payload used release_sync without defining it.
    body = _dashboard_status(tmp_path)
    assert body["status_cache"]["error"] is None
    assert body["status_cache"]["age_seconds"] is not None
    assert body["release_sync"]["state"]
    assert body["pct_succeeded"] == 100


def test_dashboard_reads_wds_index_from_the_watcher_staging_dir(tmp_path):
    # Finding 13: the dashboard ignored watcher.wds_staging_dir.
    import json

    staging = tmp_path / "custom-staging"
    (staging / "optimus").mkdir(parents=True)
    (staging / "optimus" / "wds_index.json").write_text(json.dumps({
        "A": {"shard_file": "shard-000000.tar"}, "B": {"shard_file": "shard-000000.tar"},
    }))
    body = _dashboard_status(tmp_path, wds_staging_dir=str(staging))
    assert body["status_cache"]["error"] is None
    assert body["wds_done"] == 2


def _release_agent_stage_root(tmp_path, target):
    from mussel_dispatcher.release_agent import ReleaseAgent

    cfg = Config(repo_dir=str(tmp_path / "local-repo"), nextflow_profiles="standard",
                 outdir=str(tmp_path / "results"), state_dir=str(tmp_path / "ra-state"))
    (tmp_path / "ra-state").mkdir(exist_ok=True)
    return ReleaseAgent(cfg, target, deployment_id="default").stage_root


def test_remote_agent_runs_from_the_release_agents_env_stage_root(tmp_path, monkeypatch):
    # Finding 4: MUSSEL_RELEASE_STAGE_ROOT steered only the release agent.
    from tests.test_remote_recovery import _make_agent, _order, _staged_release

    stage_root, revision = _staged_release(tmp_path)
    monkeypatch.setenv("MUSSEL_RELEASE_STAGE_ROOT", str(stage_root))
    agent = _make_agent(tmp_path, remote_repo_dir=str(tmp_path / "mussel-nf.active"))
    order = agent._normalize_order(_order(tmp_path, source_revision=revision))
    pinned = agent._pin_order_to_staged_release(order)["remote_repo_dir"]
    assert pinned == str(stage_root / "default" / revision)
    assert _release_agent_stage_root(tmp_path, agent.target) == stage_root


def test_remote_agent_runs_from_the_release_agents_default_stage_root(tmp_path):
    from tests.test_remote_recovery import _make_agent, _order, _staged_release

    stage_root, revision = _staged_release(tmp_path)
    active = tmp_path / "mussel-nf.active"
    default_root = Path(f"{active}.releases")
    default_root.symlink_to(stage_root)
    agent = _make_agent(tmp_path, remote_repo_dir=str(active))
    assert _release_agent_stage_root(tmp_path, agent.target) == default_root
    order = agent._normalize_order(_order(tmp_path, source_revision=revision))
    assert agent._pin_order_to_staged_release(order)["remote_repo_dir"] == str(
        default_root / "default" / revision
    )


def test_driver_command_log_line_does_not_expand_shell_metacharacters(tmp_path):
    # Finding 5: the shlex-quoted command sat inside a double-quoted echo.
    import shlex

    from mussel_dispatcher.remote import render_driver_script

    target = RemoteTargetConfig(name="hpc", remote_repo_dir="/remote/repo", lost_job_detection=False)
    hostile = ['--label=$(touch PWNED)', '--note=`touch PWNED`', '--quote="x', "--var=$HOME"]
    order = {
        "batch_id": "batch-001",
        "remote_run_dir": str(tmp_path / "run"),
        "remote_work_dir": str(tmp_path / "run" / "work"),
        "remote_outdir": str(tmp_path / "results"),
        "remote_repo_dir": "/remote/repo",
        "nextflow_profiles": "standard",
        "nextflow_run_name": "r$(touch PWNED)",
        "nextflow_cli_args": hostile,
    }
    script = render_driver_script(order, target)
    line = next(l for l in script.splitlines() if "[mussel-driver] command:" in l)
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + line], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "PWNED").exists()
    logged = result.stdout.rstrip("\n").split("[mussel-driver] command: ", 1)[1]
    parts = shlex.split(logged)
    assert parts[-len(hostile):] == hostile
    assert "r$(touch PWNED)" in parts


def test_capacity_guard_without_resume_thresholds_stays_paused_below_min(tmp_path, monkeypatch):
    # Finding 6: resume_free_* default 0, so a paused guard resumed onto a near-full FS.
    from mussel_dispatcher.config import StoragePolicyConfig
    from mussel_dispatcher.storage import CapacityGuard, CapacitySnapshot

    free = {"gib": 5.0}
    monkeypatch.setattr(CapacityGuard, "_snapshot", staticmethod(
        lambda path: CapacitySnapshot(path, 100 * 2**30, int(free["gib"] * 2**30), free["gib"])
    ))
    guard = CapacityGuard(StoragePolicyConfig(min_free_gib=10.0), name="local")
    assert not guard.check([str(tmp_path)])
    for _ in range(3):
        assert not guard.check([str(tmp_path)])
    free["gib"] = 20.0
    assert not guard.check([str(tmp_path)])  # hysteresis: two healthy polls
    assert guard.check([str(tmp_path)])


@pytest.mark.parametrize("backend,want", [("local", ("local", "")), ("remote_agent", ("remote_agent", "hpc"))])
def test_submit_resume_runs_a_batch_on_its_own_backend(tmp_path, monkeypatch, backend, want):
    # Finding 7: under execution_backend=remote_agent a resumed local batch ran
    # _run_remote_agent and polled for an agent forever.
    from mussel_dispatcher import scheduler as sched
    from mussel_dispatcher.config import ExecutionPoolConfig
    from mussel_dispatcher.scheduler import RunManager
    from tests.test_batch_lifecycle import make_config
    from tests.test_restart_preservation import _batch_id, _interrupted_batch

    cfg = make_config(
        tmp_path, execution_backend="remote_agent", default_dispatch_target="hpc",
        execution_pools=[ExecutionPoolConfig(name="local", backend="local", max_concurrent_runs=2),
                         ExecutionPoolConfig(name="hpc", backend="remote_agent", target="hpc",
                                             max_concurrent_runs=2)],
    )
    state = StateStore(str(tmp_path / "state.db"))
    batch_id = _batch_id("5eed0001", hours_ago=1.0)
    work = _interrupted_batch(tmp_path, state, batch_id, log_age_hours=0.01)
    if backend == "remote_agent":
        state._conn().execute("UPDATE batches SET backend='remote_agent', target='hpc' WHERE batch_id=?",
                              (batch_id,))
        state._conn().commit()
    seen = []

    class RecordingRunner(sched.NextflowRunner):
        def run(self):
            seen.append((self.cfg.execution_backend, self.cfg.default_dispatch_target))
            return 0

    monkeypatch.setattr(sched, "NextflowRunner", RecordingRunner)
    manager = RunManager(cfg, state)
    try:
        assert manager.submit_resume(batch_id, str(tmp_path / "batches" / f"{batch_id}.csv"), str(work))
    finally:
        manager.shutdown(wait=True)
    assert seen == [want]


def test_local_finalization_uses_the_local_lane_limit(tmp_path):
    # Finding 8: the slot claim used max_concurrent_finalizations (1) while the
    # pool was sized from finalization_lane_limits["local"] (4).
    from mussel_dispatcher.runner import NextflowRunner
    from tests.test_batch_lifecycle import make_config

    cfg = make_config(tmp_path, max_concurrent_finalizations=1,
                      finalization_lane_limits={"local": 4})
    state = StateStore(str(tmp_path / "state.db"))
    for batch_id in ("busy", "batch-2"):
        state.add_batch(batch_id, str(tmp_path / f"{batch_id}.csv"), str(tmp_path / "work"), 1,
                        str(tmp_path / f"{batch_id}.log"))
    assert state.claim_finalization_slot("busy", lane="local", max_slots=4, owner="other")
    runner = NextflowRunner(cfg, "batch-2", [], state)
    runner._lifecycle = MagicMock()
    runner._lifecycle.finalize_success.return_value = "finalized"
    assert runner._finalize_success(str(tmp_path / "batch-2.csv"), "", "", 0.0) == "finalized"


@pytest.fixture(scope="module")
def real_pg(tmp_path_factory):
    import os
    import random
    import shutil
    import tempfile

    psycopg = pytest.importorskip("psycopg")
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not (initdb and pg_ctl) or os.environ.get("MUSSEL_SKIP_PG_TESTS") == "1":
        pytest.skip("no local PostgreSQL binaries (initdb/pg_ctl)")
    data = tmp_path_factory.mktemp("pgdata") / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgsock-", dir="/tmp"))
    port = random.randint(20000, 60000)
    subprocess.run([initdb, "-D", str(data), "-A", "trust", "-U", "postgres", "-N"],
                   check=True, capture_output=True)
    subprocess.run([pg_ctl, "-D", str(data), "-w", "-l", str(data.parent / "server.log"),
                    "-o", f"-k {sock} -p {port} -c listen_addresses=''", "start"],
                   check=True, capture_output=True)
    admin = psycopg.connect(host=str(sock), port=port, user="postgres", dbname="postgres",
                            autocommit=True)
    names = iter(range(1000))
    try:
        def new_database():
            name = f"rr{next(names)}"
            admin.execute(f"CREATE DATABASE {name}")
            from mussel_dispatcher.config import PostgresConfig

            return {"postgres": PostgresConfig(host=str(sock), port=port, database=name,
                                               user="postgres", sslmode="disable")}
        yield new_database
    finally:
        admin.close()
        subprocess.run([pg_ctl, "-D", str(data), "-m", "immediate", "-w", "stop"], capture_output=True)
        shutil.rmtree(sock, ignore_errors=True)


def test_strict_postgres_migration_is_not_cut_off_by_the_startup_statement_timeout(
    tmp_path, monkeypatch, real_pg
):
    # Finding 9: `release migrate` ran the rollup backfill under the 5 s startup cap.
    from mussel_dispatcher import state as state_mod
    from tests.test_batch_lifecycle import make_config

    monkeypatch.setattr(state_mod, "_POSTGRES_SCHEMA_STATEMENT_TIMEOUT_MS", 1500)
    # A backfill slower than the cap, as over a large slides table.
    monkeypatch.setattr(state_mod, "SLIDE_ROLLUPS_REBUILD_SQL", state_mod.SLIDE_ROLLUPS_REBUILD_SQL.replace(
        "FROM slides", "FROM slides, (SELECT pg_sleep(2.5)) AS slow"))
    startup = StateStore.from_config(make_config(tmp_path, state_backend="postgres", **real_pg()))
    assert startup._slide_rollups_ready is False  # startup stays bounded and skips it

    cfg = make_config(tmp_path, state_backend="postgres", **real_pg())
    migrated = StateStore.from_config(cfg, strict_schema=True)
    assert migrated._slide_rollups_ready is True
    assert migrated._conn().execute(
        "SELECT 1 FROM _schema_migrations WHERE name='slide_rollups_v1'").fetchone()


def test_strict_postgres_schema_init_lifts_only_the_statement_timeout(tmp_path, monkeypatch):
    # The same check without a local PostgreSQL (CI).
    from mussel_dispatcher import state as state_mod
    from tests.test_batch_lifecycle import make_config
    from tests.test_state_hardening import _FakePgConn, _patch_postgres

    for strict, want in ((False, "SET LOCAL statement_timeout TO '5000ms'"),
                         (True, "SET LOCAL statement_timeout TO '0ms'")):
        conn = _FakePgConn(columns={t: set(c) for t, c in state_mod._required_schema_columns().items()})
        _patch_postgres(monkeypatch, conn)
        StateStore.from_config(make_config(tmp_path, state_backend="postgres"), strict_schema=strict)
        assert want in conn.queries
        assert "SET LOCAL lock_timeout TO '1000ms'" in conn.queries


def test_pool_limit_counts_submissions_whose_batch_row_is_not_written_yet(tmp_path, monkeypatch):
    # Finding 10: the worker writes the batch row after build_contract, which can
    # outlast a scheduler tick, so the pool limit read from the DB was overshot.
    import threading
    import time

    from mussel_dispatcher import scheduler as sched
    from mussel_dispatcher.config import ExecutionPoolConfig
    from mussel_dispatcher.scheduler import RunManager
    from tests.test_batch_lifecycle import make_config

    cfg = make_config(tmp_path, execution_pools=[
        ExecutionPoolConfig(name="local", backend="local", max_concurrent_runs=1)])
    state = StateStore(str(tmp_path / "state.db"))
    release = threading.Event()

    class SlowContractRunner:
        def __init__(self, *a, **k):
            pass

        def run(self):
            release.wait(10)  # still hashing the repo; no batch row yet
            return 0

    monkeypatch.setattr(sched, "NextflowRunner", SlowContractRunner)
    manager = RunManager(cfg, state)
    try:
        batch_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_aa000001"
        assert manager.submit(batch_id, [{"slide_id": "a"}], execution_pool=cfg.get_execution_pool("local"))
        assert manager.active_batch_counts_by_pool() == {"local": 1}
        assert manager.choose_execution_pool() is None
    finally:
        release.set()
        manager.shutdown(wait=True)
    assert manager.active_batch_counts_by_pool() == {}


def test_remote_resumes_over_the_startup_cap_are_resumed_when_a_slot_frees(tmp_path, monkeypatch):
    # Finding 11: they were logged "until a resume slot is available" but never
    # resubmitted, and stayed RECOVERING with no waiter.
    import threading

    from mussel_dispatcher import scheduler as sched
    from mussel_dispatcher.config import ExecutionPoolConfig
    from mussel_dispatcher.scheduler import RunManager, _submit_startup_resumes
    from tests.test_batch_lifecycle import make_config
    from tests.test_restart_preservation import _batch_id

    cfg = make_config(tmp_path, execution_backend="remote_agent", default_dispatch_target="hpc",
                      execution_pools=[ExecutionPoolConfig(name="hpc", backend="remote_agent",
                                                           target="hpc", max_concurrent_runs=1)])
    state = StateStore(str(tmp_path / "state.db"))
    specs = []
    for n in range(2):
        batch_id = _batch_id(f"5eed000{n}", hours_ago=1.0)
        state.add_batch(batch_id, str(tmp_path / f"{n}.csv"), str(tmp_path / f"w{n}"), 1,
                        str(tmp_path / f"{n}.log"), backend="remote_agent", target="hpc")
        state.set_batch_status(batch_id, "RECOVERING")
        specs.append((batch_id, str(tmp_path / f"{n}.csv"), str(tmp_path / f"w{n}")))
    release = {spec[0]: threading.Event() for spec in specs}
    started = []

    class Waiter(sched.NextflowRunner):
        def run(self):
            started.append(self.batch_id)
            release[self.batch_id].wait(10)
            return 0

    monkeypatch.setattr(sched, "NextflowRunner", Waiter)
    manager = RunManager(cfg, state)
    try:
        assert _submit_startup_resumes(cfg, state, manager, specs) == 1
        assert manager.submit_deferred_resumes() == 0  # the only slot is taken
        assert state.get_batch(specs[1][0])["status"] == "RECOVERING"
        release[specs[0][0]].set()
        manager._futures.get(specs[0][0]) and manager._futures[specs[0][0]].result(10)
        for _ in range(100):
            if manager.submit_deferred_resumes():
                break
            threading.Event().wait(0.05)
        for _ in range(100):
            if len(started) == 2:
                break
            threading.Event().wait(0.05)
        assert started == [specs[0][0], specs[1][0]]
    finally:
        for event in release.values():
            event.set()
        manager.shutdown(wait=True)


def test_s3_append_does_not_undo_its_own_coords_repair(tmp_path, monkeypatch):
    # Finding 12: a new slide opened the writer on the project's staged shard, then an
    # indexed slide's coords were repaired on S3, and flush uploaded the stale staged
    # copy over the repair.
    import pandas as pd

    from mussel_dispatcher import wds as wds_mod
    from tests.test_wds_integrity import _FakeS3, _members, _write_slide

    fake = _FakeS3(tmp_path / "remote")
    monkeypatch.setattr(wds_mod, "_s3_upload", fake.upload)
    monkeypatch.setattr(wds_mod, "_s3_download", fake.download)
    monkeypatch.setattr(wds_mod, "_s3_client", fake.client)
    for d in ("pt", "tiles"):
        (tmp_path / d).mkdir()

    def run(slides):
        return wds_mod.append_wds(
            pt_dir=tmp_path / "pt", h5_dir=tmp_path / "tiles",
            inventory_df=pd.DataFrame({"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ", "PROJ"]}),
            wds_dest="s3://bucket/wds", model_type="optimus", staging_dir=tmp_path / "staging",
            max_shard_bytes=10 * 1024 ** 3, slide_id_filter=slides, lock_dir=tmp_path / "locks")

    _write_slide(tmp_path / "pt", tmp_path / "tiles", "B", rows=2, value=2)
    (tmp_path / "tiles" / "B.patch.h5").unlink()            # B is indexed without coords
    run({"B"})
    assert "B.coords.npy" not in fake.shards()["PROJ/000000.tar"]

    _write_slide(tmp_path / "pt", tmp_path / "tiles", "A", rows=2, value=1)   # new slide, sorts first
    _write_slide(tmp_path / "pt", tmp_path / "tiles", "B", rows=2, value=2)   # B's coords are back
    index = run(None)  # unfiltered: indexed B takes the in-place repair path
    assert index["_stats"]["n_indexed_repaired"] == 1

    shard = fake.shards()[index["B"]["shard_file"]]
    assert "B.coords.npy" in shard
    assert "A.features.npy" in fake.shards()[index["A"]["shard_file"]]
    staged = tmp_path / "staging" / "optimus" / index["B"]["shard_file"]
    assert not staged.exists() or "B.coords.npy" in _members(staged)


def test_sqlite_schema_init_is_one_transaction(tmp_path, monkeypatch):
    # Finding 15: executescript() committed the BEGIN IMMEDIATE, so the schema and
    # the rollup migration ran in autocommit and concurrent starts could both rebuild.
    import sqlite3

    from mussel_dispatcher import state as state_mod

    monkeypatch.setattr(state_mod, "SLIDE_ROLLUPS_REBUILD_SQL", "SELECT no_such_function()")
    db = tmp_path / "state.db"
    with pytest.raises(sqlite3.OperationalError, match="no_such_function"):
        StateStore(str(db), strict_schema=True)
    tables = sqlite3.connect(db).execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert tables == []  # the failed migration rolled back everything with it


def test_a_deferred_remote_resume_older_than_a_day_is_resumed_not_requeued(tmp_path, monkeypatch):
    # The agent still owns a deferred remote batch: the 24 h creation-age rule must not
    # reset its slides to PENDING (they would be dispatched again while it runs).
    from mussel_dispatcher import scheduler as sched
    from mussel_dispatcher.config import ExecutionPoolConfig
    from mussel_dispatcher.scheduler import RunManager
    from tests.test_batch_lifecycle import make_config
    from tests.test_restart_preservation import _batch_id

    cfg = make_config(tmp_path, execution_backend="remote_agent", default_dispatch_target="hpc",
                      execution_pools=[ExecutionPoolConfig(name="hpc", backend="remote_agent",
                                                           target="hpc", max_concurrent_runs=1)])
    state = StateStore(str(tmp_path / "state.db"))
    batch_id = _batch_id("5eedd0ld", hours_ago=30.0)
    state.add_slide("/s/a.svs", "a")
    state.add_batch(batch_id, str(tmp_path / "b.csv"), str(tmp_path / "w"), 1,
                    str(tmp_path / "b.log"), backend="remote_agent", target="hpc")
    state.mark_dispatched(["/s/a.svs"], batch_id)
    state.set_batch_status(batch_id, "RECOVERING")
    started = []

    class Waiter(sched.NextflowRunner):
        def run(self):
            started.append(self.batch_id)
            return 0

    monkeypatch.setattr(sched, "NextflowRunner", Waiter)
    manager = RunManager(cfg, state)
    try:
        manager.defer_resume(batch_id, str(tmp_path / "b.csv"), str(tmp_path / "w"))
        assert manager.submit_deferred_resumes() == 1
        manager._futures.get(batch_id) and manager._futures[batch_id].result(10)
        assert started == [batch_id]
        slide = state._conn().execute("SELECT status FROM slides").fetchone()
        assert slide["status"] == "DISPATCHED"
    finally:
        manager.shutdown(wait=True)


def test_a_truncated_staged_shard_does_not_fail_an_s3_coords_repair(tmp_path, monkeypatch):
    # S3 is repaired first; a staged copy a killed hook left truncated is quarantined.
    from mussel_dispatcher import wds as wds_mod

    staging = tmp_path / "staging"
    staged = staging / "optimus" / "P" / "000003.tar"
    staged.parent.mkdir(parents=True)
    import io
    import tarfile as _tarfile
    buf = io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w") as tar:   # a member cut off mid-write
        data = b"x" * 4096
        info = _tarfile.TarInfo("S0.features.npy"); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    staged.write_bytes(buf.getvalue()[:1500])
    h5 = tmp_path / "S1.patch.h5"
    h5.write_bytes(b"")

    def fake_download(uri, local, *a):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(b"")
        return True

    monkeypatch.setattr(wds_mod, "_s3_tar_index", lambda uri: {"S1.features.npy": [(0, 1)]})
    monkeypatch.setattr(wds_mod, "_s3_shard_needs_repair", lambda *a, **k: True)
    monkeypatch.setattr(wds_mod, "_s3_download", fake_download)
    monkeypatch.setattr(wds_mod, "_repair_coords_in_tar",
                        lambda stats, *a, **k: {**stats, "repaired": 1})

    stats = wds_mod._repair_indexed_slide_coords_if_possible(
        slide_id="S1", index_entry={"shard_file": "P/000003.tar"}, wds_dest="s3://bucket/wds",
        model_type="optimus", staging_dir=staging, h5_lookup={"S1": h5}, dry_run=False,
        s3_max_concurrency=1)

    assert stats["repaired"] == 1
    assert not staged.exists()
    assert list(staged.parent.glob("000003.tar.corrupt.*"))
