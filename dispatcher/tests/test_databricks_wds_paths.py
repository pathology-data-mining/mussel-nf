"""The cohort Databricks export takes wds_path from the WDS index, and the sync
refuses a MERGE that would blank existing paths.

A partial wds_manifest.csv (e.g. an out-of-band run's few rows) must never be
the export's wds_path source: the MERGE would blank every other slide's path.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from mussel_dispatcher import databricks_sync as dbs
from mussel_dispatcher import wds as wds_mod
from mussel_dispatcher.cohort import sync_databricks as sync
from mussel_dispatcher.state import StateStore

from .test_wds_integrity import _FakeS3

DEST = "s3://bucket/wds"


@pytest.fixture(autouse=True)
def _fresh_cache():
    sync.clear_index_cache()
    yield
    sync.clear_index_cache()


@pytest.fixture
def s3(tmp_path, monkeypatch):
    fake = _FakeS3(tmp_path / "remote")
    monkeypatch.setattr(wds_mod, "_s3_upload", fake.upload)
    monkeypatch.setattr(wds_mod, "_s3_download", fake.download)
    monkeypatch.setattr(wds_mod, "_s3_client", fake.client)
    return fake


def _put_index(tmp_path, model: str, shards: dict[str, str]) -> None:
    p = tmp_path / "remote" / "wds" / model / "wds_index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    index = {sid: {"project_id": shard.split("/")[0], "shard_file": shard}
             for sid, shard in shards.items()}
    index["_stats"] = {"n_slides": len(shards)}
    p.write_text(json.dumps(index))


def _db(tmp_path, succeeded: list[str], pending: list[str] = ()) -> str:
    db_path = tmp_path / "dispatcher.db"
    store = StateStore(str(db_path))
    for sid in [*succeeded, *pending]:
        store.add_slide(f"/slides/{sid}.svs", sid, oncotree_code="COAD")
    if succeeded:
        store.mark_dispatched([f"/slides/{sid}.svs" for sid in succeeded], "b1")
        store.mark_slides_complete("b1", succeeded=True)
    return str(db_path)


def _paths(df: pd.DataFrame, model: str = "optimus") -> dict[str, str]:
    sub = df[df["model"] == model]
    return dict(zip(sub["slide_id"], sub["wds_path"]))


# ---------------------------------------------------------------------------
# build_export: wds_path from the index
# ---------------------------------------------------------------------------

def test_partial_manifest_does_not_blank_other_slides(tmp_path, s3):
    """(a) A 1-row manifest (an out-of-band run's) leaves every slide its index path."""
    sids = [f"s{i}" for i in range(5)]
    _put_index(tmp_path, "optimus", {sid: f"COAD/00000{i}.tar" for i, sid in enumerate(sids)})
    manifest = tmp_path / "wds_manifest.csv"
    manifest.write_text("slide_id,model,wds_path\ns0,optimus,s3://bucket/wds/optimus/COAD/000000.tar\n")

    df = sync.build_export(_db(tmp_path, sids), str(manifest), ["optimus"],
                           wds_destinations={"optimus": DEST})

    assert _paths(df) == {sid: f"{DEST}/optimus/COAD/00000{i}.tar" for i, sid in enumerate(sids)}


def test_stale_manifest_path_is_replaced_by_index_path(tmp_path, s3):
    """(b) A shard rewritten by a repair/dedupe: the index path wins over the manifest's."""
    _put_index(tmp_path, "optimus", {"a": "COAD/000009.tar", "b": "COAD/000003.tar"})
    manifest = tmp_path / "wds_manifest.csv"
    manifest.write_text(
        "slide_id,model,wds_path\n"
        "a,optimus,s3://bucket/wds/optimus/COAD/000001.tar\n"   # stale: a moved to 000009
        "b,optimus,s3://bucket/wds/optimus/COAD/000003.tar\n"
    )

    df = sync.build_export(_db(tmp_path, ["a", "b"]), str(manifest), ["optimus"],
                           wds_destinations={"optimus": DEST})

    assert _paths(df) == {"a": f"{DEST}/optimus/COAD/000009.tar",
                          "b": f"{DEST}/optimus/COAD/000003.tar"}


def test_succeeded_slide_missing_from_index_is_a_logged_discrepancy(tmp_path, s3, caplog):
    """(d) SUCCEEDED in the DB but absent from the index: empty path, counted, no crash."""
    _put_index(tmp_path, "optimus", {"a": "COAD/000000.tar"})
    caplog.set_level(logging.WARNING, logger=sync.__name__)

    df = sync.build_export(_db(tmp_path, ["a", "ghost"], pending=["p"]), None, ["optimus"],
                           wds_destinations={"optimus": DEST})

    assert _paths(df) == {"a": f"{DEST}/optimus/COAD/000000.tar", "ghost": "", "p": ""}
    msgs = [r.getMessage() for r in caplog.records if "discrepancy" in r.getMessage()]
    assert len(msgs) == 1 and "1 SUCCEEDED slides have no optimus entry" in msgs[0]
    assert "ghost" in msgs[0]


def test_each_model_reads_its_own_index(tmp_path, s3):
    _put_index(tmp_path, "optimus", {"a": "COAD/000000.tar"})
    _put_index(tmp_path, "titan_slide", {"a": "COAD/000042.tar"})

    df = sync.build_export(_db(tmp_path, ["a"]), None, ["optimus", "titan_slide"],
                           wds_destinations={"optimus": DEST, "titan_slide": DEST})

    assert _paths(df, "optimus") == {"a": f"{DEST}/optimus/COAD/000000.tar"}
    assert _paths(df, "titan_slide") == {"a": f"{DEST}/titan_slide/COAD/000042.tar"}


def test_missing_index_fails_loudly(tmp_path, s3):
    """No index for a model with SUCCEEDED slides: fail, never export blank paths."""
    _put_index(tmp_path, "optimus", {"a": "COAD/000000.tar"})

    with pytest.raises(sync.WdsIndexUnavailable, match="titan_slide"):
        sync.build_export(_db(tmp_path, ["a"]), None, ["optimus", "titan_slide"],
                          wds_destinations={"optimus": DEST, "titan_slide": DEST})


def test_no_wds_destination_fails_loudly_even_with_a_manifest(tmp_path, s3):
    manifest = tmp_path / "wds_manifest.csv"
    manifest.write_text("slide_id,model,wds_path\na,optimus,s3://bucket/wds/optimus/COAD/000000.tar\n")

    with pytest.raises(sync.WdsIndexUnavailable, match="no WDS destination"):
        sync.build_export(_db(tmp_path, ["a"]), str(manifest), ["optimus"])


def test_main_aborts_without_uploading_when_an_index_is_missing(tmp_path, s3, monkeypatch):
    uploads = []
    monkeypatch.setattr(sync, "upload_and_trigger", lambda *a, **k: uploads.append(a))

    with pytest.raises(SystemExit, match="aborted"):
        sync.main(["--db", _db(tmp_path, ["a"]), f"--wds-dest={DEST}", "--model-types=optimus",
                   "--volume-folder=/Volumes/c/s/v", "--table=c.s.t", "--warehouse-id=w"])
    assert uploads == []


def test_index_is_downloaded_to_a_private_dir_not_the_hooks_staging_copy(tmp_path, s3, monkeypatch):
    """The WDS hook writes <staging>/<model>/wds_index.json then uploads it; the sync
    must not overwrite that file in between."""
    _put_index(tmp_path, "optimus", {"a": "COAD/000000.tar"})
    staging = tmp_path / "wds-staging"
    hook_copy = staging / "optimus" / "wds_index.json"
    hook_copy.parent.mkdir(parents=True)
    hook_copy.write_text('{"being": "saved"}')
    seen = []
    real = s3.download
    monkeypatch.setattr(wds_mod, "_s3_download",
                        lambda uri, local, *a: (seen.append(Path(local)), real(uri, local, *a))[1])

    sync.build_export(_db(tmp_path, ["a"]), None, ["optimus"],
                      wds_destinations={"optimus": DEST}, staging_dir=staging)

    assert hook_copy.read_text() == '{"being": "saved"}'
    assert seen and all(p != hook_copy for p in seen)
    assert not list(staging.glob(".databricks-sync-*"))  # temp dir cleaned up


def test_index_is_loaded_once_per_process(tmp_path, s3, monkeypatch):
    _put_index(tmp_path, "optimus", {"a": "COAD/000000.tar"})
    calls = []
    real = s3.download
    monkeypatch.setattr(wds_mod, "_s3_download",
                        lambda uri, local, *a: (calls.append(uri), real(uri, local, *a))[1])
    db = _db(tmp_path, ["a"])

    for _ in range(3):
        sync.build_export(db, None, ["optimus"], wds_destinations={"optimus": DEST})
    assert len(calls) == 1


def test_wds_dest_arg_parsing():
    assert sync._parse_wds_dest_args([DEST], ["optimus", "titan_slide"]) == {
        "optimus": DEST, "titan_slide": DEST}
    assert sync._parse_wds_dest_args(["optimus=s3://b/x"], ["optimus"]) == {"optimus": "s3://b/x"}


# ---------------------------------------------------------------------------
# upload_and_trigger: the MERGE guard
# ---------------------------------------------------------------------------

def _export(n: int, n_with_path: int, model: str = "optimus") -> pd.DataFrame:
    return pd.DataFrame({
        "slide_id": [f"s{i}" for i in range(n)],
        "model": [model] * n,
        "status": ["SUCCEEDED"] * n,
        "wds_path": [f"{DEST}/{model}/COAD/{i:06d}.tar" if i < n_with_path else "" for i in range(n)],
        "completed_at": ["2026-01-01"] * n,
    })


def _args(tmp_path, **kw) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    dbs.add_upload_args(parser)
    args = parser.parse_args([
        "--databricks-host=https://dbx.example", "--token=t",
        "--volume-folder=/Volumes/c/s/v", "--table=c.s.t", "--warehouse-id=w",
        f"--status-file={tmp_path / 'sync_status.json'}",
    ])
    for k, v in kw.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def dbx(monkeypatch):
    """Fake Databricks: records uploads/merges/deletes; the guard query returns ``counts``."""
    calls = {"upload": [], "merge": [], "delete": [], "sql": []}
    state = {"counts": [[100, 0]], "sql_error": None}

    def sql_rows(statement, *a, **k):
        calls["sql"].append(statement)
        if state["sql_error"]:
            raise RuntimeError(state["sql_error"])
        return state["counts"]

    monkeypatch.setattr(dbs, "ensure_volume_exists", lambda *a, **k: None)
    monkeypatch.setattr(dbs, "purge_volume_folder", lambda *a, **k: None)
    monkeypatch.setattr(dbs, "upload_parquet", lambda local, vp, *a: calls["upload"].append(vp))
    monkeypatch.setattr(dbs, "merge_via_warehouse",
                        lambda *a, **k: (calls["merge"].append(a), (True, "ok"))[1])
    monkeypatch.setattr(dbs, "_delete_volume_file", lambda vp, *a: calls["delete"].append(vp))
    monkeypatch.setattr(dbs, "_sql_rows", sql_rows)
    calls["state"] = state
    return calls


def test_guard_refuses_an_export_that_would_blank_most_paths(tmp_path, dbx):
    """(c) A partial-manifest export: SUCCEEDED rows with (almost) no paths."""
    with pytest.raises(SystemExit, match="refused"):
        dbs.upload_and_trigger(_export(100, 1), _args(tmp_path), filename_prefix="cohort_inventory_",
                               wds_path_guard=True)
    assert dbx["upload"] == [] and dbx["merge"] == []
    status = json.loads((tmp_path / "sync_status.json").read_text())
    assert status["status"] == "FAILED" and "allow-wds-path-loss" in status["message"]


def test_guard_checks_each_model_separately(tmp_path, dbx):
    df = pd.concat([_export(100, 100, "optimus"), _export(100, 100, "hoptimus1"),
                    _export(100, 85, "titan_slide")], ignore_index=True)
    with pytest.raises(SystemExit, match="titan_slide"):
        dbs.upload_and_trigger(df, _args(tmp_path), filename_prefix="cohort_inventory_",
                               wds_path_guard=True)
    assert dbx["merge"] == []


def test_guard_refuses_a_merge_that_would_blank_existing_table_paths(tmp_path, dbx):
    """(c) Export looks complete, but the table holds paths this MERGE would blank."""
    dbx["state"]["counts"] = [["1000", "500"]]
    with pytest.raises(SystemExit, match="blank 500 of 1000"):
        dbs.upload_and_trigger(_export(100, 100), _args(tmp_path), filename_prefix="cohort_inventory_",
                               wds_path_guard=True)
    assert dbx["merge"] == []
    assert len(dbx["upload"]) == 1 and dbx["delete"] == dbx["upload"]  # refused parquet removed
    assert "parquet.`/Volumes/c/s/v/`" in dbx["sql"][0] and "c.s.t" in dbx["sql"][0]


def test_guard_fails_closed_when_the_table_check_errors(tmp_path, dbx):
    dbx["state"]["sql_error"] = "warehouse unavailable"
    with pytest.raises(SystemExit, match="could not check"):
        dbs.upload_and_trigger(_export(100, 100), _args(tmp_path), filename_prefix="cohort_inventory_",
                               wds_path_guard=True)
    assert dbx["merge"] == []


def test_guard_allows_a_healthy_merge_and_a_new_table(tmp_path, dbx):
    dbx["state"]["counts"] = [["1000", "3"]]  # 0.3% < 1%: e.g. a few re-queued slides
    dbs.upload_and_trigger(_export(100, 99), _args(tmp_path), filename_prefix="cohort_inventory_",
                           wds_path_guard=True)
    assert len(dbx["merge"]) == 1

    dbx["state"]["sql_error"] = "[TABLE_OR_VIEW_NOT_FOUND] c.s.t"
    dbs.upload_and_trigger(_export(100, 99), _args(tmp_path), filename_prefix="cohort_inventory_",
                           wds_path_guard=True)
    assert len(dbx["merge"]) == 2


def test_guard_override_flag(tmp_path, dbx):
    dbx["state"]["counts"] = [["1000", "900"]]
    dbs.upload_and_trigger(_export(100, 0), _args(tmp_path, allow_wds_path_loss=True),
                           filename_prefix="cohort_inventory_", wds_path_guard=True)
    assert len(dbx["merge"]) == 1 and dbx["sql"] == []


def test_guard_is_off_unless_requested(tmp_path, dbx):
    """TCGA and other callers keep the old behaviour."""
    dbs.upload_and_trigger(_export(100, 0), _args(tmp_path), filename_prefix="tcga_inventory_")
    assert len(dbx["merge"]) == 1 and dbx["sql"] == []


def test_cohort_main_enables_the_guard(tmp_path, s3, monkeypatch):
    _put_index(tmp_path, "optimus", {"a": "COAD/000000.tar"})
    seen = {}
    monkeypatch.setattr(sync, "upload_and_trigger",
                        lambda df, args, **k: seen.update(k, df=df))
    sync.main(["--db", _db(tmp_path, ["a"]), f"--wds-dest={DEST}", "--model-types=optimus",
               "--volume-folder=/Volumes/c/s/v"])
    assert seen["wds_path_guard"] is True
    assert _paths(seen["df"]) == {"a": f"{DEST}/optimus/COAD/000000.tar"}
