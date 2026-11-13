"""Regressions for dispatcher review follow-ups."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mussel_dispatcher import wds as wds_mod
from mussel_dispatcher.batch_lifecycle import PostBatchHooks
from mussel_dispatcher.config import _read_nf_model_types
from mussel_dispatcher.remote import RemoteAgent
from mussel_dispatcher.state import StateStore, finalization_slot_name
from tests.test_batch_lifecycle import add_running_batch, make_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


# 6. --once runs finalization in the main thread: no stall watchdog to kill it mid-append.
@pytest.mark.parametrize("once,watchdog", [(True, False), (False, True)])
def test_watchdog_runs_only_for_the_long_running_agent(once, watchdog):
    agent = object.__new__(RemoteAgent)
    agent.once = once
    agent.target = MagicMock(agent_summary_interval_seconds=60, agent_heartbeat_timeout_seconds=300)
    agent._agent_heartbeat_thread = None
    agent._agent_watchdog_thread = None
    import threading
    agent._agent_heartbeat_stop = threading.Event()
    agent._publish_agent_snapshot = lambda force=False: None
    agent._main_loop_progress = lambda: ("now", 0.0)

    agent._start_agent_heartbeat()
    try:
        assert agent._agent_watchdog_thread.is_alive() is watchdog
    finally:
        agent._agent_heartbeat_stop.set()


# 9. Remote finalization slots are per target (max_concurrent_finalizations is per target).
def test_each_remote_target_has_its_own_finalization_slots(tmp_path):
    assert finalization_slot_name("remote:hpc", 1) != finalization_slot_name("remote:hpc2", 1)
    assert finalization_slot_name("remote", 0) == finalization_slot_name("remote", 0)  # legacy name kept
    with pytest.raises(ValueError):
        finalization_slot_name("local:hpc", 0)
    state = StateStore(str(tmp_path / "s.db"))
    a = state.claim_finalization_slot("batch-a", lane="remote:hpc", max_slots=1, owner="a", ttl_seconds=300)
    b = state.claim_finalization_slot("batch-b", lane="remote:hpc2", max_slots=1, owner="b", ttl_seconds=300)
    c = state.claim_finalization_slot("batch-c", lane="remote:hpc", max_slots=1, owner="c", ttl_seconds=300)
    assert a and b and a != b      # one target's slot does not starve the other
    assert c is None               # each target still respects its own limit


# 10. Provisioning drops and creates every table of the real schema, as the app user.
def test_provisioning_covers_every_schema_table(tmp_path):
    provision = _script("provision_postgres")
    tables = set(provision.schema_tables())
    state = StateStore(str(tmp_path / "s.db"))          # SQLite mirror of the same schema
    sqlite_tables = {r[0] for r in state._conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    assert {"slide_rollups", "hook_attempts", "hook_circuit_breakers", "hook_failure_events",
            "deployment_releases", "deployment_nodes"} <= tables
    assert sqlite_tables <= tables | {"_schema_migrations"}


def test_provisioning_builds_the_schema_through_statestore_as_the_app_user(monkeypatch):
    provision = _script("provision_postgres")
    seen = {}

    class FakeStore:
        def __init__(self, cfg, strict_schema=False):
            import os
            seen.update(user=cfg.postgres.user, password=os.environ.get(cfg.postgres.password_env),
                        strict=strict_schema)

    monkeypatch.setattr("mussel_dispatcher.state.StateStore", FakeStore)
    provision.create_schema_as_app_user(host="db.example", port=5432, db_name="d", app_user="app",
                                        app_password="secret", sslmode="disable")
    assert seen == {"user": "app", "password": "secret", "strict": True}


# 11. Import refuses a missing SQLite path instead of creating and importing an empty DB.
def test_import_refuses_a_missing_sqlite_db(tmp_path, monkeypatch):
    importer = _script("import_sqlite_to_postgres")
    connects = []
    monkeypatch.setattr(importer, "_pg_connect", lambda cfg: connects.append(cfg))
    missing = tmp_path / "dispatcher.db.typo"
    with pytest.raises(FileNotFoundError):
        importer.import_state(str(missing), MagicMock())
    assert not missing.exists() and connects == []
    assert ("hook_attempts", ["batch_id", "hook_index", "attempt_number"]) in importer.TABLES


# 12. A params file that does not set model_types falls back to nextflow.config.
def test_model_types_fall_back_when_the_params_file_does_not_set_them(tmp_path):
    (tmp_path / "nextflow.config").write_text("params { featurize { model_types = ['uni', 'conch'] } }\n")
    params = tmp_path / "params.yaml"
    params.write_text("featurize:\n  batch_size: 8\n")
    assert _read_nf_model_types(str(tmp_path), str(params)) == ["uni", "conch"]
    params.write_text("featurize:\n  model_types: [titan]\n")
    assert _read_nf_model_types(str(tmp_path), str(params)) == ["titan"]


# 14. Every model's WDS hook may delete the shared coords, but only for slides no
#     model still holds a .pt for: whichever append finishes a slide last cleans it up.
def test_every_wds_hook_deletes_coords_once_no_model_still_needs_them(tmp_path):
    from mussel_dispatcher.config import WatcherConfig
    cfg = make_config(tmp_path, cleanup_results=True, watchers=[WatcherConfig(
        type="databricks", wds_destinations={"model-a": "s3://wds", "model-b": "s3://wds"})])
    wds = [h for h in cfg._build_auto_hooks() if "mussel_dispatcher.wds" in h.get("command", "")]
    assert len(wds) == 2
    for hook, other in zip(wds, ("model-b", "model-a")):
        assert "--delete-coords-local" in hook["args"]
        assert "--coords-shared-pt-dirs={outdir}/features/" + other in hook["args"]
        assert not any(k.startswith("_args") for k in hook)


def test_coords_survive_while_another_model_still_has_the_slide(tmp_path):
    for d in ("a", "b", "tiles"):
        (tmp_path / d).mkdir()
    for sid in ("s1", "s2"):
        (tmp_path / "tiles" / f"{sid}.patch.h5").write_bytes(b"")
    (tmp_path / "b" / "s1.features.pt").write_bytes(b"")        # model-b has not appended s1

    lookup = wds_mod._build_h5_lookup(tmp_path / "tiles")
    n = wds_mod._delete_consumed_coords(lookup, {"s1", "s2"}, [tmp_path / "a", tmp_path / "b"])

    assert n == 1
    assert (tmp_path / "tiles" / "s1.patch.h5").exists()
    assert not (tmp_path / "tiles" / "s2.patch.h5").exists()


def _real_wds_batch(tmp_path, outputs):
    """Generated hooks for two models whose executor runs the real WDS append."""
    import csv as _csv

    from mussel_dispatcher.config import WatcherConfig
    from tests.test_wds_integrity import _write_slide

    outdir = tmp_path / "results"
    (outdir / "tiles").mkdir(parents=True)
    wds_root = tmp_path / "wds"
    cfg = make_config(tmp_path, outdir=str(outdir), hook_preflight=False, cleanup_results=True,
                      max_slide_retries=3, watchers=[WatcherConfig(
                          type="databricks",
                          wds_destinations={"model-a": str(wds_root), "model-b": str(wds_root)})])
    cfg.post_batch_hooks = cfg._build_auto_hooks()                     # what production runs
    csv_path = tmp_path / "batch-1.csv"
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["slide_id", "slide_path", "oncotree_code"])
        w.writerows([[s, f"s3://slides/{s}.svs", "P"] for s in ("a", "b")])
    for model, slides in outputs.items():
        (outdir / "features" / model).mkdir(parents=True)
        for sid in slides:
            _write_slide(outdir / "features" / model, outdir / "tiles", sid, rows=2, value=1)
    state = StateStore(str(tmp_path / "state.db"))
    for s in ("a", "b"):
        state.add_slide(f"s3://slides/{s}.svs", s)
    state.add_batch("batch-1", str(csv_path), str(tmp_path / "work"), 2, str(tmp_path / "b.log"))
    state.mark_dispatched(["s3://slides/a.svs", "s3://slides/b.svs"], "batch-1")
    return cfg, state, csv_path, outdir


def _run_wds(cmd):
    args = [str(a) for a in cmd[cmd.index("mussel_dispatcher.wds") + 1:]]
    model = next(a.split("=", 1)[1] for a in args if a.startswith("--model-type="))
    return model, wds_mod.main(args)


def test_coords_are_deleted_when_the_failed_earlier_hook_is_retried(tmp_path):
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {"model-a": "ab", "model-b": "ab"})
    ran, fail_a = [], [True]

    def executor(cmd, **kwargs):
        if "mussel_dispatcher.wds" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if fail_a[0] and "--model-type=model-a" in cmd:
            ran.append("model-a failed")
            return subprocess.CompletedProcess(cmd, 1, "", "connection refused")
        model, rc = _run_wds(cmd)
        ran.append(model)
        return subprocess.CompletedProcess(cmd, rc, "", "")

    PostBatchHooks(cfg, "batch-1", state).run(str(csv_path), phase="wds", executor=executor)
    assert ran == ["model-a failed", "model-b"]
    assert sorted(p.name for p in (outdir / "tiles").iterdir()) == ["a.patch.h5", "b.patch.h5"]

    state._conn().execute("UPDATE batch_hooks SET next_retry_at=NULL")
    state._conn().commit()
    fail_a[0] = False
    PostBatchHooks(cfg, "batch-1", state).run(str(csv_path), phase="wds", executor=executor)

    assert ran == ["model-a failed", "model-b", "model-a"]              # B is not replayed
    assert list((outdir / "tiles").iterdir()) == []                     # A's retry cleaned up


# 15a. A corrupt published shard of a LOCAL destination is not renamed away.
def test_local_destination_refuses_a_corrupt_published_shard(tmp_path):
    shard = tmp_path / "wds" / "optimus" / "PROJ" / "000000.tar"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"not a tar" * 100)
    with pytest.raises(RuntimeError, match="unreadable"):
        wds_mod._ShardWriter(str(tmp_path / "wds"), "optimus", "PROJ", None, max_shard_bytes=10**9)
    assert shard.exists()


# 15b. The index merge keeps another writer's newer entry for a slide this run did not touch.
def test_a_stale_remote_index_never_overrides_this_runs_entries(tmp_path, monkeypatch):
    """Real appends: run 2's save-time merge reads a stale index (A still in an old
    shard it was stripped from); A must keep the entry run 2 loaded under the lock."""
    import pandas as pd

    from tests.test_wds_integrity import _FakeS3, _write_slide

    fake = _FakeS3(tmp_path / "remote")
    monkeypatch.setattr(wds_mod, "_s3_upload", fake.upload)
    monkeypatch.setattr(wds_mod, "_s3_client", fake.client)
    index_downloads = []                                      # during run 2

    def download(uri, local, *a):
        ok = fake.download(uri, local, *a)
        if uri.endswith("/wds_index.json") and ok and run_2[0]:
            index_downloads.append(uri)
            if len(index_downloads) > 1:                      # run 2's save-time merge read
                stale = json.loads(Path(local).read_text())
                stale["A"] = dict(stale["A"], shard_file="PROJ/999999.tar")
                Path(local).write_text(json.dumps(stale))
        return ok

    monkeypatch.setattr(wds_mod, "_s3_download", download)
    for d in ("pt", "tiles"):
        (tmp_path / d).mkdir()
    for sid in "AB":
        _write_slide(tmp_path / "pt", tmp_path / "tiles", sid, rows=2, value=1)

    def run(ids):
        return wds_mod.append_wds(
            pt_dir=tmp_path / "pt", h5_dir=tmp_path / "tiles",
            inventory_df=pd.DataFrame({"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ"] * 2}),
            wds_dest="s3://bucket/wds", model_type="optimus", staging_dir=tmp_path / "staging",
            max_shard_bytes=10 * 1024 ** 3, slide_id_filter=set(ids), lock_dir=tmp_path / "locks")

    run_2 = [False]
    a_entry = dict(run("A")["A"])
    run_2[0] = True
    run("B")

    published = json.loads(fake.path("s3://bucket/wds/optimus/wds_index.json").read_text())
    assert len(index_downloads) >= 2                          # the stale read did happen
    assert published["A"] == a_entry and "B" in published


# 9 (status). Every remote target's finalization slots show in the coordination status.
def test_finalization_status_reports_every_remote_target(tmp_path):
    from mussel_dispatcher.config import RemoteTargetConfig
    cfg = make_config(tmp_path, max_concurrent_finalizations=1,
                      dispatch_targets=[RemoteTargetConfig(name="hpc"), RemoteTargetConfig(name="hpc2")],
                      default_dispatch_target="hpc")
    state = StateStore(str(tmp_path / "s.db"))
    state.cfg = cfg
    state.claim_finalization_slot("b-hpc2", lane="remote:hpc2", max_slots=1, owner="x", ttl_seconds=300)
    status = state.finalization_coordination_status()
    assert [lease["batch_id"] for lease in status["lanes"]["remote"]] == ["b-hpc2"]


# 14 (end to end). Real generated hooks -> real WDS appends -> BatchFinalizer: a coverage
# failure in an earlier model requeues only the slide it lacks; the last model still appends.
def test_coverage_failure_in_an_earlier_model_requeues_only_the_missing_slide(tmp_path):
    from mussel_dispatcher.batch_lifecycle import BatchFinalizer, FinalizationContext

    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {"model-a": "a", "model-b": "ab"})
    ran = []

    def executor(cmd, **kwargs):
        if "mussel_dispatcher.wds" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        model, rc = _run_wds(cmd)
        ran.append((model, rc))
        return subprocess.CompletedProcess(cmd, rc, "", "")

    BatchFinalizer(cfg, "batch-1", state).finalize_success(
        FinalizationContext(batch_csv=str(csv_path), outdir=str(outdir), repo_dir=cfg.repo_dir),
        hook_executor=executor, collect_manifest=lambda started: None)

    assert ran == [("model-a", 2), ("model-b", 0)]       # A lacks b (exit 2); B still appends
    slides = {r["slide_id"]: dict(r) for r in state._conn().execute(
        "SELECT slide_id, status, fail_count FROM slides")}
    assert slides["b"]["status"] == "PENDING" and slides["b"]["fail_count"] == 1   # only b requeued
    assert slides["a"]["status"] != "PENDING" and slides["a"]["fail_count"] == 0
    # Neither model has a .pt left for either slide: no append still needs the coords.
    assert list((outdir / "tiles").iterdir()) == []


# 16. A remote submit that fails after its order committed (e.g. the COMMIT's reply was
#     lost) must not leave the order claimable once its slides are requeued.
def _remote_batch_with_order(tmp_path, remote_status):
    state = StateStore(str(tmp_path / "s.db"))
    state.add_slide("/s/a.svs", "a")
    state.add_batch("b1", "/c.csv", "/w", 1, "/l", backend="remote_agent", target="hpc")
    state.mark_dispatched(["/s/a.svs"], "b1")
    if remote_status:
        state._conn().execute("UPDATE batches SET remote_status=?, remote_payload='{}' WHERE batch_id='b1'",
                              (remote_status,))
        state._conn().commit()
    return state


def _fail_submit(state, monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from mussel_dispatcher.scheduler import RunManager
    from tests.test_dispatcher import make_config as make_dispatcher_config

    rm = RunManager(make_dispatcher_config(max_concurrent_runs=2), state)
    reattached = []
    monkeypatch.setattr(rm, "_reattach_remote_waiter", lambda batch_id, batch: reattached.append(batch_id))
    future = Future()
    future.set_exception(RuntimeError("connection lost during COMMIT"))
    assert rm._slots.acquire(blocking=False)
    rm._futures["b1"] = future
    # The runner died inside client.submit(): work_may_be_running was never set.
    rm._on_done("b1", future, SimpleNamespace(work_may_be_running=False, driver_succeeded=False))
    rm.shutdown(wait=False)
    slide = state._conn().execute("SELECT status FROM slides").fetchone()["status"]
    return slide, reattached


@pytest.mark.parametrize("remote_status", ["", "QUEUED"])
def test_an_unclaimed_order_is_withdrawn_before_its_slides_are_requeued(tmp_path, monkeypatch, remote_status):
    remote_status_before = remote_status
    state = _remote_batch_with_order(tmp_path, remote_status)
    slide, reattached = _fail_submit(state, monkeypatch)
    assert slide == "PENDING" and reattached == []
    remote_status = state._conn().execute("SELECT remote_status FROM batches").fetchone()[0]
    # CANCELLED, not just a FAILED batch: an agent on older code claims by remote_status alone.
    assert (remote_status or "") == ("CANCELLED" if remote_status_before else "")
    assert state.claim_remote_batch("hpc") is None             # the agent can't run it too


def test_an_order_the_agent_already_claimed_is_waited_on_not_requeued(tmp_path, monkeypatch):
    state = _remote_batch_with_order(tmp_path, "CLAIMED")
    slide, reattached = _fail_submit(state, monkeypatch)
    assert slide == "DISPATCHED" and reattached == ["b1"]


@pytest.mark.parametrize("batch_status,claimable", [
    ("RUNNING", True), ("RECOVERING", True), ("FAILED", False), ("SUCCEEDED", False)])
def test_only_an_active_batch_order_is_claimable(tmp_path, batch_status, claimable):
    state = _remote_batch_with_order(tmp_path, "QUEUED")
    state._conn().execute("UPDATE batches SET status=? WHERE batch_id='b1'", (batch_status,))
    state._conn().commit()
    assert (state.claim_remote_batch("hpc") is not None) is claimable


