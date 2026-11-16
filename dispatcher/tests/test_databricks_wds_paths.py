"""The Databricks sync refuses an export or MERGE that would blank existing wds_paths.

A partial or stale export (e.g. one whose wds_path values came from a small
out-of-band manifest) must never MERGE away the table's existing paths.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from mussel_dispatcher import databricks_sync as dbs

DEST = "s3://bucket/wds"


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
    """SUCCEEDED rows with (almost) no paths are refused before upload."""
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
    """Callers that do not request the guard keep the old behaviour."""
    dbs.upload_and_trigger(_export(100, 0), _args(tmp_path), filename_prefix="tcga_inventory_")
    assert len(dbx["merge"]) == 1 and dbx["sql"] == []
