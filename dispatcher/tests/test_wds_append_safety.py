"""Regression tests for WDS append failure paths that could lose or duplicate slides.

- Local sources are deleted, and manifest rows written, only after the index save.
- A local destination keeps a re-processed slide's indexed copy until the index save.
- A staged shard whose S3 copy was stripped and deleted is not re-uploaded.
- The Postgres append lock uses keepalives and is verified before mutating.
- A 403 on an object that exists is an error, not "missing".
- The final index save does not resurrect slides pruned as failed.

(A failed S3 shard listing failing the writer, rather than guessing shard 0, is
tested in test_dispatcher.TestShardWriterInit.)
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from mussel_dispatcher import wds as wds_mod
from tests.test_wds_integrity import _FakeS3, _append, _members, _write_slide, dirs  # noqa: F401


@pytest.fixture
def s3(tmp_path, monkeypatch):
    fake = _FakeS3(tmp_path / "remote")
    monkeypatch.setattr(wds_mod, "_s3_upload", fake.upload)
    monkeypatch.setattr(wds_mod, "_s3_download", fake.download)
    monkeypatch.setattr(wds_mod, "_s3_client", fake.client)
    return fake


def _run_s3(tmp_path, staging, slide_id_filter, **kwargs):
    return wds_mod.append_wds(
        pt_dir=tmp_path / "pt", h5_dir=tmp_path / "tiles",
        inventory_df=pd.DataFrame({"file_name": ["A.svs", "B.svs", "C.svs"],
                                   "project_id": ["PROJ", "PROJ", "PROJ"]}),
        wds_dest="s3://bucket/wds", model_type="optimus", staging_dir=tmp_path / staging,
        max_shard_bytes=10 * 1024 ** 3, slide_id_filter=slide_id_filter, **kwargs)


def _die_on_first_index_save(monkeypatch):
    real = wds_mod._save_index
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("index save failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(wds_mod, "_save_index", flaky)


def test_sources_and_manifest_wait_for_the_index_save(dirs, s3, monkeypatch):
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    pt, coords = dirs / "pt" / "A.features.pt", dirs / "tiles" / "A.patch.h5"
    manifest = dirs / "manifest.csv"
    _die_on_first_index_save(monkeypatch)

    with pytest.raises(RuntimeError):
        _run_s3(dirs, "staging", {"A"}, lock_dir=dirs / "locks", delete_local=True,
                delete_coords_local=True, manifest_csv=manifest)

    assert pt.exists() and coords.exists()
    assert not manifest.exists()

    # The retry strips the dead run's copy, appends A again from the kept sources,
    # and only then deletes them and writes the manifest.
    index = _run_s3(dirs, "staging", {"A"}, lock_dir=dirs / "locks", delete_local=True,
                    delete_coords_local=True, manifest_csv=manifest)

    holders = [sh for sh, m in s3.shards().items() if "A.features.npy" in m]
    assert holders == [index["A"]["shard_file"]]
    assert not pt.exists() and not coords.exists()
    assert index["A"]["shard_file"] in manifest.read_text()


def test_local_reprocessed_slide_keeps_indexed_copy_until_index_save(dirs, monkeypatch):
    wds_root = dirs / "wds"
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    _write_slide(dirs / "pt", dirs / "tiles", "B", rows=2, value=2)
    first = _append(dirs, wds_root, slide_id_filter={"A", "B"}, lock_dir=dirs / "locks")
    old_shard = wds_root / "optimus" / first["A"]["shard_file"]

    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=3)  # re-processed
    _die_on_first_index_save(monkeypatch)
    with pytest.raises(RuntimeError):
        _append(dirs, wds_root, slide_id_filter={"A"}, lock_dir=dirs / "locks")

    # The index still points at the old copy, which must still be there.
    saved = json.loads((wds_root / "optimus" / "wds_index.json").read_text())
    assert saved["A"]["shard_file"] == first["A"]["shard_file"]
    assert "A.features.npy" in _members(old_shard)

    index = _append(dirs, wds_root, slide_id_filter={"A"}, lock_dir=dirs / "locks")

    holders = {str(p.relative_to(wds_root / "optimus")): _members(p).count("A.features.npy")
               for p in (wds_root / "optimus").rglob("*.tar") if "A.features.npy" in _members(p)}
    assert holders == {index["A"]["shard_file"]: 1}
    assert index["A"]["shard_file"] != first["A"]["shard_file"]
    assert "B.features.npy" in _members(old_shard)


def test_emptied_shard_is_not_resurrected_from_staging(dirs, s3, monkeypatch):
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    _die_on_first_index_save(monkeypatch)
    with pytest.raises(RuntimeError):
        _run_s3(dirs, "staging", {"A"}, lock_dir=dirs / "locks")
    (dirs / "pt" / "A.features.pt").unlink()  # batch A abandoned; the next batch is B

    _write_slide(dirs / "pt", dirs / "tiles", "B", rows=2, value=2)
    index = _run_s3(dirs, "staging", {"B"}, lock_dir=dirs / "locks")

    assert "A" not in index
    assert all("A.features.npy" not in members for members in s3.shards().values())
    assert [sh for sh, m in s3.shards().items() if "B.features.npy" in m] == [index["B"]["shard_file"]]


class _FakeLockConn:
    def __init__(self, held):
        self.held = held
        self.sql = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        held = self.held() if "pg_locks" in sql and callable(self.held) else self.held
        return SimpleNamespace(fetchone=lambda: {"held": held})

    def close(self):
        pass


def _postgres_run(dirs, monkeypatch, conn):
    monkeypatch.setattr(wds_mod, "connect_database", lambda cfg, readonly=False: conn)
    return _run_s3(dirs, "staging", {"A"}, lock_backend="postgres",
                   postgres_db_cfg=SimpleNamespace(backend="postgres"))


def test_postgres_lock_uses_keepalives_and_is_verified(dirs, s3, monkeypatch):
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    conn = _FakeLockConn(held=True)

    index = _postgres_run(dirs, monkeypatch, conn)

    assert "A" in index
    assert any("tcp_keepalives_idle" in sql for sql in conn.sql)
    checks = [sql for sql in conn.sql if "pg_locks" in sql]
    assert len(checks) >= 3  # reconcile, shard upload, index save


def test_lost_postgres_lock_aborts_before_upload(dirs, s3, monkeypatch):
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    checks = {"n": 0}

    def held():
        checks["n"] += 1
        return checks["n"] == 1  # held at reconcile, lost before the shard upload

    conn = _FakeLockConn(held=held)

    with pytest.raises(wds_mod.WdsLockLost):
        _postgres_run(dirs, monkeypatch, conn)

    assert s3.shards() == {}
    assert not (dirs / "remote" / "wds" / "optimus" / "wds_index.json").exists()


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")


def test_access_denied_on_existing_object_raises(tmp_path, monkeypatch):
    client = MagicMock()
    client.download_file.side_effect = _client_error("403")
    client.list_objects_v2.return_value = {"Contents": [{"Key": "wds/optimus/wds_index.json"}]}
    monkeypatch.setattr(wds_mod, "_s3_client", lambda: client)

    with pytest.raises(ClientError):
        wds_mod._s3_download("s3://bucket/wds/optimus/wds_index.json", tmp_path / "i.json")
    with pytest.raises(ClientError):
        wds_mod._load_index("s3://bucket/wds", "optimus", tmp_path)


def test_access_denied_on_unlisted_object_is_missing(tmp_path, monkeypatch):
    # ECS answers 403 for a key that does not exist; the listing confirms it.
    client = MagicMock()
    client.download_file.side_effect = _client_error("AccessDenied")
    client.list_objects_v2.return_value = {"Contents": [{"Key": "wds/optimus/wds_index.json.bak"}]}
    monkeypatch.setattr(wds_mod, "_s3_client", lambda: client)

    assert wds_mod._s3_download("s3://bucket/wds/optimus/wds_index.json", tmp_path / "i.json") is False


def test_final_save_does_not_resurrect_pruned_failed_slide(dirs, s3, monkeypatch):
    index_uri_path = s3.root / "wds" / "optimus" / "wds_index.json"
    index_uri_path.parent.mkdir(parents=True)
    stale = {sid: {"project_id": "PROJ", "shard_file": "PROJ/000000.tar"} for sid in ("A", "B")}
    index_uri_path.write_text(json.dumps(stale))

    # ECS keeps serving the pre-prune index to every read.
    real_download = s3.download

    def stale_download(uri, local, *a):
        if uri.endswith("wds_index.json"):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(stale))
            return True
        return real_download(uri, local, *a)

    monkeypatch.setattr(wds_mod, "_s3_download", stale_download)
    _write_slide(dirs / "pt", dirs / "tiles", "C", rows=2, value=3)

    index = _run_s3(dirs, "staging", {"A", "C"}, failed_slides={"A"})

    assert "A" not in index and "C" in index
    assert "A" not in json.loads(index_uri_path.read_text())


def test_lost_lock_is_classified_transient(dirs, s3, monkeypatch):
    from mussel_dispatcher.hook_hardening import HookFailureCategory, classify_hook_failure

    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    with pytest.raises(wds_mod.WdsLockLost) as lost:
        _postgres_run(dirs, monkeypatch, _FakeLockConn(held=False))

    class _Broken(_FakeLockConn):
        def execute(self, sql, params=None):
            if "pg_locks" in sql:
                raise OSError("server closed the connection unexpectedly")
            return super().execute(sql, params)

    with pytest.raises(wds_mod.WdsLockLost) as unverifiable:
        _postgres_run(dirs, monkeypatch, _Broken(held=True))

    for exc in (lost.value, unverifiable.value):
        # What wds.main logs when append_wds raises.
        output = f"ERROR Failed to append WDS for model optimus: {exc}"
        assert classify_hook_failure(1, output).category == HookFailureCategory.TRANSIENT_SERVICE


class _UndefinedSetting(Exception):
    sqlstate = "42704"


def test_idle_session_timeout_is_skipped_on_old_servers(monkeypatch):
    class _OldServer(_FakeLockConn):
        def execute(self, sql, params=None):
            if "idle_session_timeout" in sql:
                raise _UndefinedSetting('unrecognized configuration parameter "idle_session_timeout"')
            return super().execute(sql, params)

    conn = _OldServer(held=True)
    monkeypatch.setattr(wds_mod, "connect_database", lambda cfg, readonly=False: conn)
    with wds_mod._wds_postgres_lock(SimpleNamespace(backend="postgres"), "s3://b/wds", "optimus") as verify:
        verify()
    assert any("pg_advisory_lock" in sql for sql in conn.sql)


def test_idle_session_timeout_other_errors_propagate(monkeypatch):
    class _Failing(_FakeLockConn):
        def execute(self, sql, params=None):
            if "idle_session_timeout" in sql:
                raise RuntimeError("connection lost")
            return super().execute(sql, params)

    monkeypatch.setattr(wds_mod, "connect_database", lambda cfg, readonly=False: _Failing(held=True))
    with pytest.raises(RuntimeError, match="connection lost"):
        with wds_mod._wds_postgres_lock(SimpleNamespace(backend="postgres"), "s3://b/wds", "optimus"):
            pass


def test_idle_session_timeout_is_disabled(monkeypatch):
    conn = _FakeLockConn(held=True)
    monkeypatch.setattr(wds_mod, "connect_database", lambda cfg, readonly=False: conn)
    with wds_mod._wds_postgres_lock(SimpleNamespace(backend="postgres"), "s3://b/wds", "optimus"):
        pass
    assert "SET idle_session_timeout TO 0" in conn.sql


def test_manifest_lists_only_this_runs_slides_after_a_stale_read(dirs, s3, monkeypatch):
    # Another batch indexed X; this run's first index read is stale and misses it.
    remote_index = s3.root / "wds" / "optimus" / "wds_index.json"
    remote_index.parent.mkdir(parents=True)
    remote_index.write_text(json.dumps({"X": {"project_id": "PROJ", "shard_file": "PROJ/000007.tar"}}))
    real_load = wds_mod._load_index
    calls = {"n": 0}

    def stale_first(*args, **kwargs):
        calls["n"] += 1
        return {} if calls["n"] == 1 else real_load(*args, **kwargs)

    monkeypatch.setattr(wds_mod, "_load_index", stale_first)
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    manifest = dirs / "manifest.csv"

    index = _run_s3(dirs, "staging", {"A"}, manifest_csv=manifest)

    assert "X" in json.loads(remote_index.read_text())  # the save still keeps X
    rows = pd.read_csv(manifest)
    assert list(rows["slide_id"]) == ["A"]
    assert rows["wds_path"].iloc[0].endswith(index["A"]["shard_file"])


def _lost():
    raise wds_mod.WdsLockLost("WDS append lock lost: test")


def test_reconcile_strips_only_under_a_verified_lock(dirs, s3, monkeypatch):
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    _die_on_first_index_save(monkeypatch)
    with pytest.raises(RuntimeError):
        _run_s3(dirs, "staging", {"A"}, lock_dir=dirs / "locks")
    before, journals = s3.shards(), s3.journals()
    assert journals and any("A.features.npy" in m for m in before.values())

    with pytest.raises(wds_mod.WdsLockLost):
        wds_mod._reconcile_journals("s3://bucket/wds", "optimus", {}, dirs / "staging",
                                    verify_lock=_lost)

    assert s3.shards() == before and s3.journals() == journals


def _staged_shard_needing_repair(staging, slide_id="A"):
    """A staged shard holding features (2 rows) but no coords, as an indexed slide."""
    shard = staging / "optimus" / "PROJ" / "000000.tar"
    shard.parent.mkdir(parents=True)
    import io
    import tarfile
    import numpy as np
    with tarfile.open(shard, "w") as tar:
        buf = io.BytesIO(); np.save(buf, np.zeros((2, 4), np.float32)); data = buf.getvalue()
        info = tarfile.TarInfo(f"{slide_id}.features.npy"); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return shard


def test_coords_repairs_upload_only_under_a_verified_lock(dirs, monkeypatch):
    uploads = []
    monkeypatch.setattr(wds_mod, "_s3_upload", lambda *a, **k: uploads.append(a))
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)  # .patch.h5 with 2 coords
    h5 = {"A": dirs / "tiles" / "A.patch.h5"}

    _staged_shard_needing_repair(dirs / "staging1")
    with pytest.raises(wds_mod.WdsLockLost):
        wds_mod._repair_indexed_slide_coords_if_possible(
            slide_id="A", index_entry={"shard_file": "PROJ/000000.tar"},
            wds_dest="s3://bucket/wds", model_type="optimus", staging_dir=dirs / "staging1",
            h5_lookup=h5, dry_run=False, s3_max_concurrency=1, verify_lock=_lost)
    _staged_shard_needing_repair(dirs / "staging2")
    with pytest.raises(wds_mod.WdsLockLost):
        wds_mod.repair_wds_shards("s3://bucket/wds", "optimus", dirs / "staging2", dirs / "tiles",
                                  verify_lock=_lost)
    assert uploads == []

    # Local destinations rewrite in place: verified before the rewrite.
    local_root = dirs / "local"
    shard = _staged_shard_needing_repair(local_root)
    before = shard.read_bytes()
    with pytest.raises(wds_mod.WdsLockLost):
        wds_mod.repair_wds_shards(str(local_root), "optimus", None, dirs / "tiles",
                                  verify_lock=_lost)
    assert shard.read_bytes() == before


def test_repair_existing_shards_runs_under_the_append_lock(dirs, monkeypatch):
    held = []
    real_lock = wds_mod._wds_mutation_lock

    def spy(**kwargs):
        held.append(kwargs)
        return real_lock(**kwargs)

    def repair(*a, verify_lock=None, **k):
        assert held, "repair ran without the append lock"
        verify_lock()
        return {}

    monkeypatch.setattr(wds_mod, "_wds_mutation_lock", spy)
    monkeypatch.setattr(wds_mod, "repair_wds_shards", repair)
    monkeypatch.setattr(wds_mod, "append_wds", lambda **k: {})
    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    pd.DataFrame({"file_name": ["A.svs"], "project_id": ["PROJ"]}).to_csv(dirs / "inv.csv", index=False)

    rc = wds_mod.main(["--pt-dir", str(dirs / "pt"), "--h5-dir", str(dirs / "tiles"),
                       "--inventory", str(dirs / "inv.csv"), "--wds-dest", str(dirs / "wds"),
                       "--model-type", "optimus", "--lock-dir", str(dirs / "locks"),
                       "--repair-existing-shards"])

    assert rc == 0
    assert held and held[0]["lock_dir"] == dirs / "locks"


class _EcsClient:
    """boto3 S3 client over a directory that, like ECS, answers 403 for a missing key."""

    def __init__(self, root, unreadable=()):
        self.root = root
        self.unreadable = set(unreadable)

    @staticmethod
    def _error(code, op):
        return ClientError({"Error": {"Code": code, "Message": code}}, op)

    def download_file(self, bucket, key, filename, Config=None):
        path = self.root / key
        if key in self.unreadable or not path.exists():
            raise self._error("403", "HeadObject")
        with open(filename, "wb") as fh:
            fh.write(path.read_bytes())

    def upload_file(self, filename, bucket, key, Config=None):
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "rb") as fh:
            path.write_bytes(fh.read())

    def head_object(self, Bucket, Key):
        path = self.root / Key
        if not path.exists():
            raise self._error("403", "HeadObject")
        return {"ContentLength": path.stat().st_size}

    def _keys(self, prefix):
        return sorted(k for k in (str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())
                      if k.startswith(prefix))

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):
        return {"Contents": [{"Key": k} for k in self._keys(Prefix)[:MaxKeys]]}

    def get_paginator(self, name):
        return SimpleNamespace(paginate=lambda Bucket, Prefix: [self.list_objects_v2(Bucket, Prefix)])

    def delete_object(self, Bucket, Key):
        (self.root / Key).unlink()


def test_append_through_ecs_403_for_missing_keys(dirs, monkeypatch):
    root = dirs / "remote"
    root.mkdir()
    client = _EcsClient(root)
    monkeypatch.setattr(wds_mod, "_s3_client", lambda: client)
    remote_index = root / "wds" / "optimus" / "wds_index.json"

    _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
    first = _run_s3(dirs, "staging", {"A"}, lock_dir=dirs / "locks")  # no index, no journals yet
    _write_slide(dirs / "pt", dirs / "tiles", "B", rows=2, value=2)
    second = _run_s3(dirs, "staging", {"B"}, lock_dir=dirs / "locks")

    saved = json.loads(remote_index.read_text())
    assert {"A", "B"} <= set(saved)
    assert saved["A"]["shard_file"] == first["A"]["shard_file"]
    assert saved["B"]["shard_file"] == second["B"]["shard_file"]
    assert not list((root / "wds" / "optimus").glob("_journal/*.json"))

    # An index that exists but cannot be read fails the append instead of being replaced.
    client.unreadable.add("wds/optimus/wds_index.json")
    _write_slide(dirs / "pt", dirs / "tiles", "C", rows=2, value=3)
    before = remote_index.read_text()
    with pytest.raises(ClientError):
        _run_s3(dirs, "staging", {"C"}, lock_dir=dirs / "locks")
    assert remote_index.read_text() == before
