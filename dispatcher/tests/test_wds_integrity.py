"""Regression tests for the WDS integrity fixes.

1. A re-processed slide that is already indexed replaces its stale WDS copy.
2. Index saves never resurrect entries this writer removed.
3. Shard numbering never reuses a name another writer already uploaded.
4. --force-new-shard reaches the shard writer.
5. Replacing a slide without its .patch.h5 keeps the indexed MPP.
6. An append that dies before saving the index leaves no copies behind, and the
   cleanup never strips a copy the index may still reference.
"""
import io
import json
import tarfile
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from mussel_dispatcher import wds as wds_mod


def _members(path):
    with tarfile.open(path, "r") as tar:
        return [m.name for m in tar.getmembers()]


def _write_slide(pt_dir, h5_dir, sid, rows, value):
    torch.save(torch.full((rows, 4), float(value)), pt_dir / f"{sid}.features.pt")
    with h5py.File(h5_dir / f"{sid}.patch.h5", "w") as f:
        f.create_dataset("coords", data=np.arange(rows * 2, dtype=np.int64).reshape(rows, 2))


def _append(tmp_path, wds_root, **kwargs):
    return wds_mod.append_wds(
        pt_dir=tmp_path / "pt",
        h5_dir=tmp_path / "tiles",
        inventory_df=pd.DataFrame({"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ", "PROJ"]}),
        wds_dest=str(wds_root),
        model_type="optimus",
        staging_dir=None,
        max_shard_bytes=10 * 1024 ** 3,
        **kwargs,
    )


@pytest.fixture
def dirs(tmp_path):
    (tmp_path / "pt").mkdir()
    (tmp_path / "tiles").mkdir()
    return tmp_path


class TestReplaceReprocessed:
    def _seed_stale(self, tmp_path, native_mpp=None, mpp_is_fallback=None):
        """Index A in an old shard holding a 2-row tiling (the stale copy)."""
        wds_root = tmp_path / "wds"
        old = wds_root / "optimus" / "PROJ" / "000000.tar"
        old.parent.mkdir(parents=True)
        with tarfile.open(old, "w") as tar:
            for sid in ("A", "B"):
                for kind, arr in (("features", np.zeros((2, 4), np.float32)),
                                  ("coords", np.zeros((2, 2), np.int64))):
                    buf = io.BytesIO(); np.save(buf, arr); data = buf.getvalue()
                    info = tarfile.TarInfo(f"{sid}.{kind}.npy"); info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        (wds_root / "optimus" / "wds_index.json").write_text(json.dumps({
            sid: {"project_id": "PROJ", "shard_file": "PROJ/000000.tar",
                  "native_mpp": native_mpp, "mpp_is_fallback": mpp_is_fallback}
            for sid in ("A", "B")
        }))
        return wds_root, old

    def test_reprocessed_slide_replaces_stale_copy(self, dirs):
        wds_root, old = self._seed_stale(dirs)
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=1)  # new 3-row tiling

        result = _append(dirs, wds_root, slide_id_filter={"A"})

        assert result["_stats"]["n_replaced"] == 1
        entry = result["A"]
        assert entry["content_sha1"]
        new_path = wds_root / "optimus" / entry["shard_file"]
        with tarfile.open(new_path, "r") as tar:
            feats = np.load(io.BytesIO(tar.extractfile("A.features.npy").read()))
        assert feats.shape == (3, 4)
        # exactly one copy of A remains across all shards; B is untouched
        all_members = [n for p in (wds_root / "optimus" / "PROJ").glob("*.tar") for n in _members(p)]
        assert all_members.count("A.features.npy") == 1
        assert all_members.count("A.coords.npy") == 1
        assert "B.features.npy" in _members(old)

    def test_identical_content_is_not_rewritten(self, dirs):
        wds_root, _old = self._seed_stale(dirs)
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=1)
        first = _append(dirs, wds_root, slide_id_filter={"A"})
        shard = wds_root / "optimus" / first["A"]["shard_file"]
        before = shard.stat().st_mtime_ns

        second = _append(dirs, wds_root, slide_id_filter={"A"})  # e.g. a hook retry

        assert second["_stats"]["n_appended"] == 0
        assert second["_stats"]["n_replaced"] == 0
        assert shard.stat().st_mtime_ns == before

    def test_slides_outside_the_batch_are_never_replaced(self, dirs):
        wds_root, old = self._seed_stale(dirs)
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=1)

        result = _append(dirs, wds_root, slide_id_filter={"B"})

        assert result["A"]["shard_file"] == "PROJ/000000.tar"
        assert "A.features.npy" in _members(old)

    def test_legacy_behaviour_flag_keeps_indexed_copy(self, dirs):
        wds_root, old = self._seed_stale(dirs)
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=1)

        result = _append(dirs, wds_root, slide_id_filter={"A"}, replace_reprocessed=False)

        assert result["_stats"]["n_replaced"] == 0
        assert result["A"]["shard_file"] == "PROJ/000000.tar"


    def test_replacing_without_h5_keeps_indexed_mpp(self, dirs):
        # A slide encoder's features carry no .patch.h5, so the replacement has no MPP of
        # its own; the slide's MPP must survive rather than be erased.
        wds_root, _old = self._seed_stale(dirs, native_mpp=0.2527, mpp_is_fallback=False)
        torch.save(torch.full((1, 4), 1.0), dirs / "pt" / "A.features.pt")

        result = _append(dirs, wds_root, slide_id_filter={"A"})

        assert result["_stats"]["n_replaced"] == 1
        assert result["A"]["native_mpp"] == 0.2527
        assert result["A"]["mpp_is_fallback"] is False

    def test_replacing_with_h5_mpp_uses_the_new_value(self, dirs):
        wds_root, _old = self._seed_stale(dirs, native_mpp=0.2527, mpp_is_fallback=False)
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=1)
        with h5py.File(dirs / "tiles" / "A.patch.h5", "a") as f:
            f["coords"].attrs["native_mpp"] = 0.5
            f["coords"].attrs["mpp_is_fallback"] = True

        result = _append(dirs, wds_root, slide_id_filter={"A"})

        assert result["A"]["native_mpp"] == 0.5
        assert result["A"]["mpp_is_fallback"] is True


class TestSaveIndexDoesNotResurrect:
    def test_removed_entries_are_not_merged_back(self, tmp_path, monkeypatch):
        (tmp_path / "optimus").mkdir()  # created by _load_index in normal use
        remote = {"A": {"shard_file": "P/000000.tar"}, "B": {"shard_file": "P/000000.tar"}}

        def fake_download(uri, local_path, *_args):
            local_path.write_text(json.dumps(remote))
            return True

        uploaded = {}
        monkeypatch.setattr(wds_mod, "_s3_download", fake_download)
        monkeypatch.setattr(wds_mod, "_s3_upload",
                            lambda path, uri, *_a: uploaded.update(json.loads(path.read_text())))

        index = {"B": remote["B"]}  # A was pruned by this writer
        wds_mod._save_index(index, "s3://bucket/wds", "optimus", tmp_path, removed={"A"})

        assert "A" not in uploaded
        assert "B" in uploaded

    def test_other_writers_entries_are_still_merged(self, tmp_path, monkeypatch):
        (tmp_path / "optimus").mkdir()
        remote = {"C": {"shard_file": "P/000001.tar"}}
        monkeypatch.setattr(wds_mod, "_s3_download",
                            lambda uri, p, *_a: (p.write_text(json.dumps(remote)), True)[1])
        uploaded = {}
        monkeypatch.setattr(wds_mod, "_s3_upload",
                            lambda path, uri, *_a: uploaded.update(json.loads(path.read_text())))

        wds_mod._save_index({"B": {}}, "s3://bucket/wds", "optimus", tmp_path, removed={"A"})

        assert set(uploaded) == {"B", "C"}


class TestShardNumbering:
    def _client(self, keys, head_sizes=None):
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": k} for k in keys]}
        ]

        def head_object(Bucket, Key):
            if head_sizes is None or Key not in head_sizes:
                err = Exception("not found"); err.response = {"Error": {"Code": "404"}}
                raise err
            return {"ContentLength": head_sizes[Key]}

        client.head_object.side_effect = head_object
        return client

    def _staged(self, tmp_path, idx):
        work = tmp_path / "staging" / "optimus" / "PROJ"
        work.mkdir(parents=True)
        shard = work / f"{idx:06d}.tar"
        with tarfile.open(shard, "w") as tar:
            data = b"x" * 10
            info = tarfile.TarInfo("A.features.npy"); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return shard

    def test_staged_shard_not_reused_when_s3_copy_differs(self, tmp_path, monkeypatch):
        shard = self._staged(tmp_path, 3)
        prefix = "wds/optimus/PROJ/"
        keys = [f"{prefix}{i:06d}.tar" for i in range(6)]
        monkeypatch.setattr(wds_mod, "_s3_client",
                            lambda: self._client(keys, {f"{prefix}000003.tar": shard.stat().st_size + 512}))

        writer = wds_mod._ShardWriter("s3://bucket/wds", "optimus", "PROJ", tmp_path / "staging",
                                      max_shard_bytes=10 * 1024 ** 3)

        assert writer._current_path is None
        assert writer._current_index == 6

    def test_staged_shard_reused_when_s3_copy_matches(self, tmp_path, monkeypatch):
        shard = self._staged(tmp_path, 3)
        prefix = "wds/optimus/PROJ/"
        monkeypatch.setattr(wds_mod, "_s3_client",
                            lambda: self._client([f"{prefix}000003.tar"],
                                                 {f"{prefix}000003.tar": shard.stat().st_size}))

        writer = wds_mod._ShardWriter("s3://bucket/wds", "optimus", "PROJ", tmp_path / "staging",
                                      max_shard_bytes=10 * 1024 ** 3)

        assert writer._current_index == 3
        assert writer._current_path == shard

    def test_rotation_skips_names_owned_by_other_writers(self, tmp_path, monkeypatch):
        prefix = "wds/optimus/PROJ/"
        keys = [f"{prefix}{i:06d}.tar" for i in range(9)]  # another writer reached 000008
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: self._client(keys))
        monkeypatch.setattr(wds_mod, "_s3_upload", lambda *a, **k: None)
        writer = wds_mod._ShardWriter("s3://bucket/wds", "optimus", "PROJ", tmp_path / "staging",
                                      max_shard_bytes=10 * 1024 ** 3)
        writer._current_index = 4
        writer._current_path = tmp_path / "staging" / "optimus" / "PROJ" / "000004.tar"

        writer._seal_and_rotate()

        assert writer._current_index == 9

    def test_force_new_shard_reaches_writer(self, dirs, monkeypatch):
        seen = {}
        real = wds_mod._ShardWriter

        def spy(*args, **kwargs):
            seen["force_new_shard"] = kwargs.get("force_new_shard")
            return real(*args, **kwargs)

        monkeypatch.setattr(wds_mod, "_ShardWriter", spy)
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
        _append(dirs, dirs / "wds", slide_id_filter={"A"}, force_new_shard=True)

        assert seen["force_new_shard"] is True


class _FakeS3:
    """s3://bucket/<key> backed by <root>/<key>, for the helpers append_wds uses."""

    def __init__(self, root):
        self.root = root

    def path(self, uri):
        return self.root / uri[len("s3://"):].split("/", 1)[1]

    def upload(self, local, uri, *_a):
        dst = self.path(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())

    def download(self, uri, local, *_a):
        src = self.path(uri)
        if not src.exists():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(src.read_bytes())
        return True

    def client(self):
        fake = MagicMock()

        def paginate(Bucket, Prefix):
            keys = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())
            return [{"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}]

        def head_object(Bucket, Key):
            p = self.root / Key
            if not p.exists():
                err = Exception("not found"); err.response = {"Error": {"Code": "404"}}
                raise err
            return {"ContentLength": p.stat().st_size}

        def get_object(Bucket, Key, Range=None):
            data = (self.root / Key).read_bytes()
            if Range:
                lo, hi = (int(x) for x in Range.removeprefix("bytes=").split("-"))
                data = data[lo:hi + 1]
            return {"Body": io.BytesIO(data)}

        fake.get_paginator.return_value.paginate.side_effect = paginate
        fake.head_object.side_effect = head_object
        fake.get_object.side_effect = get_object
        fake.delete_object.side_effect = lambda Bucket, Key: (self.root / Key).unlink()
        return fake

    def shards(self):
        return {str(p.relative_to(self.root / "wds" / "optimus")): _members(p)
                for p in sorted((self.root / "wds" / "optimus").rglob("*.tar"))}

    def journals(self):
        return sorted((self.root / "wds" / "optimus" / "_journal").glob("*.json"))


def s3_index(tmp_path):
    return json.loads((tmp_path / "remote" / "wds" / "optimus" / "wds_index.json").read_text())


class TestInterruptedAppend:
    """An append that dies between uploading shards and saving the index leaves no copies behind."""

    @pytest.fixture
    def s3(self, tmp_path, monkeypatch):
        fake = _FakeS3(tmp_path / "remote")
        monkeypatch.setattr(wds_mod, "_s3_upload", fake.upload)
        monkeypatch.setattr(wds_mod, "_s3_download", fake.download)
        monkeypatch.setattr(wds_mod, "_s3_client", fake.client)
        return fake

    def _run(self, tmp_path, staging, **kwargs):
        return wds_mod.append_wds(
            pt_dir=tmp_path / "pt", h5_dir=tmp_path / "tiles",
            inventory_df=pd.DataFrame({"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ", "PROJ"]}),
            wds_dest="s3://bucket/wds", model_type="optimus", staging_dir=tmp_path / staging,
            max_shard_bytes=10 * 1024 ** 3, slide_id_filter={"A", "B"}, **kwargs)

    def _die_on_index_save(self, monkeypatch):
        real = wds_mod._save_index
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("hook killed before the index was saved")
            return real(*args, **kwargs)

        monkeypatch.setattr(wds_mod, "_save_index", flaky)

    def test_retry_into_a_new_shard_strips_the_first_copies(self, dirs, s3, monkeypatch):
        for sid, v in (("A", 1), ("B", 2)):
            _write_slide(dirs / "pt", dirs / "tiles", sid, rows=2, value=v)
        self._die_on_index_save(monkeypatch)
        with pytest.raises(RuntimeError):
            self._run(dirs, "staging1", lock_dir=dirs / "locks")
        assert s3.shards() and len(s3.journals()) == 1  # uploaded, never indexed

        # The retry starts from a fresh staging dir with --force-new-shard, as recovery runs do.
        index = self._run(dirs, "staging2", lock_dir=dirs / "locks", force_new_shard=True)

        shards = s3.shards()
        for sid in ("A", "B"):
            holders = [sh for sh, members in shards.items() if f"{sid}.features.npy" in members]
            assert holders == [index[sid]["shard_file"]]
            assert shards[holders[0]].count(f"{sid}.features.npy") == 1
        assert s3.journals() == []

    def test_crash_after_index_save_keeps_every_slide(self, dirs, s3, monkeypatch):
        for sid, v in (("A", 1), ("B", 2)):
            _write_slide(dirs / "pt", dirs / "tiles", sid, rows=2, value=v)
        real_close = wds_mod._AppendJournal.close
        monkeypatch.setattr(wds_mod._AppendJournal, "close",
                            lambda self: (_ for _ in ()).throw(RuntimeError("died after the index save")))
        with pytest.raises(RuntimeError):
            self._run(dirs, "staging1", lock_dir=dirs / "locks")
        monkeypatch.setattr(wds_mod._AppendJournal, "close", real_close)
        before = s3.shards()

        index = self._run(dirs, "staging2", lock_dir=dirs / "locks")  # identical content: nothing to append

        assert s3.shards() == before
        assert {index["A"]["shard_file"], index["B"]["shard_file"]} <= set(before)
        assert s3.journals() == []

    def test_journals_are_left_alone_without_a_lock(self, dirs, s3, monkeypatch):
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
        self._die_on_index_save(monkeypatch)
        with pytest.raises(RuntimeError):
            self._run(dirs, "staging1", lock_dir=dirs / "locks")
        before = s3.shards()
        leftover = s3.journals()
        assert len(leftover) == 1

        self._run(dirs, "staging2", force_new_shard=True)

        # Unlocked: the leftover journal may belong to a live run, so nothing is stripped,
        # and the unlocked run writes no journal of its own.
        assert s3.journals() == leftover
        assert set(before) <= set(s3.shards())

    def test_unlocked_append_writes_no_journal(self, dirs, s3, monkeypatch):
        # A locked run treats every journal as left by a dead run, so an unlocked run's
        # journal would get its slides stripped while it is still appending.
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
        self._die_on_index_save(monkeypatch)
        with pytest.raises(RuntimeError):
            self._run(dirs, "staging1")
        assert s3.shards() and s3.journals() == []

    def _serve_stale_index(self, monkeypatch, stale_reads):
        """Make the first ``stale_reads`` index reads return the empty pre-append index."""
        real = wds_mod._load_index
        calls = {"n": 0}

        def load(*args, **kwargs):
            calls["n"] += 1
            return {} if calls["n"] <= stale_reads else real(*args, **kwargs)

        monkeypatch.setattr(wds_mod, "_load_index", load)

    def _die_after_index_save(self, dirs, monkeypatch, *, before_marking):
        for sid, v in (("A", 1), ("B", 2)):
            _write_slide(dirs / "pt", dirs / "tiles", sid, rows=2, value=v)
        target = "mark_index_saved" if before_marking else "close"
        real = getattr(wds_mod._AppendJournal, target)
        monkeypatch.setattr(wds_mod._AppendJournal, target,
                            lambda self: (_ for _ in ()).throw(RuntimeError("died after the index save")))
        with pytest.raises(RuntimeError):
            self._run(dirs, "staging1", lock_dir=dirs / "locks")
        monkeypatch.setattr(wds_mod._AppendJournal, target, real)
        # The next run is another batch: it does not re-append A or B, so a wrong
        # strip cannot be masked by writing them again.
        for p in (dirs / "pt").glob("*.features.pt"):
            p.unlink()
        return s3_index(dirs)

    def test_stale_index_read_never_strips_a_saved_runs_slides(self, dirs, s3, monkeypatch):
        # The dead run saved the index and marked its journal; ECS then serves the next
        # run an index from before that save, on every read.
        saved = self._die_after_index_save(dirs, monkeypatch, before_marking=False)
        before = s3.shards()
        self._serve_stale_index(monkeypatch, stale_reads=2)

        self._run(dirs, "staging2", lock_dir=dirs / "locks", force_new_shard=True)

        assert s3.shards() == before
        assert {saved["A"]["shard_file"], saved["B"]["shard_file"]} <= set(before)
        assert s3.journals() == []

    def test_unmarked_journal_rereads_the_index_before_stripping(self, dirs, s3, monkeypatch):
        # Died between the index save and marking the journal; only the run's first
        # index read is stale, the re-read before stripping is current.
        saved = self._die_after_index_save(dirs, monkeypatch, before_marking=True)
        self._serve_stale_index(monkeypatch, stale_reads=1)

        before = s3.shards()

        self._run(dirs, "staging2", lock_dir=dirs / "locks", force_new_shard=True)

        assert s3.shards() == before
        assert {saved["A"]["shard_file"], saved["B"]["shard_file"]} <= set(before)

    def test_successful_append_leaves_no_journal(self, dirs, s3):
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
        self._run(dirs, "staging1", lock_dir=dirs / "locks")
        assert s3.journals() == []
        assert list(s3.shards().values()) == [["A.features.npy", "A.coords.npy"]]

    def test_superseded_copy_left_by_a_crash_is_stripped_next_run(self, dirs, s3, monkeypatch):
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=2, value=1)
        first = self._run(dirs, "staging1", lock_dir=dirs / "locks", force_new_shard=True)
        old_shard = first["A"]["shard_file"]

        # Re-processed: new content replaces the indexed copy, but the run dies after
        # saving the index and before stripping the old copy.
        _write_slide(dirs / "pt", dirs / "tiles", "A", rows=3, value=2)
        real_strip = wds_mod._strip_slides_from_shard
        monkeypatch.setattr(wds_mod, "_strip_slides_from_shard",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("died mid-strip")))
        with pytest.raises(RuntimeError):
            self._run(dirs, "staging2", lock_dir=dirs / "locks", force_new_shard=True)
        monkeypatch.setattr(wds_mod, "_strip_slides_from_shard", real_strip)
        assert "A.features.npy" in s3.shards()[old_shard]  # stale copy still there

        index = self._run(dirs, "staging3", lock_dir=dirs / "locks")  # nothing new to append

        holders = [sh for sh, m in s3.shards().items() if "A.features.npy" in m]
        assert holders == [index["A"]["shard_file"]] and holders != [old_shard]
        assert s3.journals() == []


class TestLocalDestinationJournal:
    """Local destinations are written in place; the journal covers them too."""

    def test_interrupted_local_append_leaves_one_copy(self, dirs, monkeypatch):
        for sid, v in (("A", 1), ("B", 2)):
            _write_slide(dirs / "pt", dirs / "tiles", sid, rows=2, value=v)
        real = wds_mod._save_index
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("killed before the index was saved")
            return real(*args, **kwargs)

        monkeypatch.setattr(wds_mod, "_save_index", flaky)
        wds_root = dirs / "wds"
        with pytest.raises(RuntimeError):
            _append(dirs, wds_root, slide_id_filter={"A", "B"}, lock_dir=dirs / "locks")
        journal_dir = wds_root / "optimus" / "_journal"
        assert len(list(journal_dir.glob("*.json"))) == 1

        index = _append(dirs, wds_root, slide_id_filter={"A", "B"}, lock_dir=dirs / "locks")

        members = [m for p in sorted((wds_root / "optimus").rglob("*.tar")) for m in _members(p)]
        assert sorted(members) == ["A.coords.npy", "A.features.npy", "B.coords.npy", "B.features.npy"]
        assert {index["A"]["shard_file"], index["B"]["shard_file"]} == {"PROJ/000000.tar"}
        assert list(journal_dir.glob("*.json")) == []
