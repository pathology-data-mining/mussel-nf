"""Randomized WDS appends against an S3 fake whose index reads may lag.

ECS can serve an object older than its last upload. The append lock serializes
writers, but not what they read: each index GET may return an earlier version.
Real append_wds runs (new slides, re-processed slides, hook retries, prunes,
runs that die before or after the index save) alternate between two writers;
after every run the published index must point every slide at a shard that holds
its current features and coords, and must not have lost a slide.
"""
import io
import json
import random
import tarfile

import numpy as np
import pandas as pd
import pytest

from mussel_dispatcher import wds as wds_mod
from tests.test_wds_integrity import _FakeS3, _write_slide

INDEX_KEY = "wds/optimus/wds_index.json"
LOAD, MERGE = "load", "merge"  # _load_index (also the journal re-read) / _save_index's merge read


class _LaggingS3(_FakeS3):
    """_FakeS3 whose index GETs may return one of the last ``lag`` earlier versions.

    Each read point is stale with probability ``p_stale`` while ``budget`` allows;
    a version from before the first upload reads as a missing object.
    """

    def __init__(self, root, rng, *, lag=3, p_stale=0.5, points=(LOAD, MERGE)):
        super().__init__(root)
        self.rng, self.lag, self.p_stale, self.points = rng, lag, p_stale, set(points)
        self.versions: list[bytes] = []
        self.budget = 0
        self.stale_served = 0

    def upload(self, local, uri, *a):
        super().upload(local, uri, *a)
        if self.path(uri) == self.root / INDEX_KEY:
            self.versions.append(local.read_bytes())

    def download(self, uri, local, *a):
        if self.path(uri) != self.root / INDEX_KEY or not self.versions:
            return super().download(uri, local, *a)
        point = MERGE if local.name.startswith(".wds_index.remote.") else LOAD
        if point in self.points and self.budget > 0 and self.rng.random() < self.p_stale:
            self.budget -= 1
            back = self.rng.randint(1, self.lag)
            if len(self.versions) - back <= 0:
                self.stale_served += 1
                return False  # read from before the index existed
            if len(self.versions) > 1:
                self.stale_served += 1
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(self.versions[max(0, len(self.versions) - 1 - back)])
            return True
        return super().download(uri, local, *a)


def _npy(tar, name):
    return np.load(io.BytesIO(tar.extractfile(name).read()))


def _check(s3, expected, optional, step):
    """Every indexed slide resolves to its current copy; no expected slide is missing."""
    index_path = s3.root / INDEX_KEY
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index = {k: v for k, v in index.items() if not k.startswith("_")}
    missing = set(expected) - set(index)
    assert not missing, f"step {step}: slides lost from the index: {sorted(missing)}"
    for sid, entry in index.items():
        assert sid in expected or sid in optional, f"step {step}: unexpected slide {sid}"
        shard = s3.root / "wds" / "optimus" / entry["shard_file"]
        assert shard.exists(), f"step {step}: {sid} -> missing shard {entry['shard_file']}"
        with tarfile.open(shard, "r") as tar:
            names = tar.getnames()
            assert f"{sid}.features.npy" in names and f"{sid}.coords.npy" in names, (
                f"step {step}: {sid} -> {entry['shard_file']}, which does not hold it ({names})")
            feats, coords = _npy(tar, f"{sid}.features.npy"), _npy(tar, f"{sid}.coords.npy")
        rows, value = expected.get(sid) or optional[sid]
        assert feats.shape == (rows, 4) and float(feats[0, 0]) == value, (
            f"step {step}: {sid} -> {entry['shard_file']} holds a superseded copy "
            f"({feats.shape}, {float(feats[0, 0])}; want ({rows}, 4), {value})")
        assert coords.shape == (rows, 2), f"step {step}: {sid} coords {coords.shape}"


def _run_sequence(tmp_path, monkeypatch, seed, steps, points):
    rng = random.Random(seed)
    s3 = _LaggingS3(tmp_path / "remote", random.Random(seed + 1), points=points)
    monkeypatch.setattr(wds_mod, "_s3_upload", s3.upload)
    monkeypatch.setattr(wds_mod, "_s3_download", s3.download)
    monkeypatch.setattr(wds_mod, "_s3_client", s3.client)

    expected: dict[str, tuple[int, float]] = {}  # slide -> (rows, value) the index must serve
    optional: dict[str, tuple[int, float]] = {}  # pruned, or a dead run's content: may be served
    project = {}
    next_id, next_value = 0, 1.0
    real_save, real_mark = wds_mod._save_index, wds_mod._AppendJournal.mark_index_saved

    for step in range(steps):
        writer = rng.randrange(2)
        pt, tiles = tmp_path / f"w{writer}" / "pt", tmp_path / f"w{writer}" / "tiles"
        for d in (pt, tiles):
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()
        action = rng.choices(["new", "reprocess", "retry", "prune"], [4, 3, 1, 1])[0]
        indexed = sorted(expected)
        if action != "new" and not indexed:
            action = "new"
        batch, failed, content = set(), set(), {}
        if action == "new":
            for _ in range(rng.randint(1, 3)):
                sid = f"S{next_id:04d}"
                next_id += 1
                project[sid] = rng.choice(["P1", "P2"])
                batch.add(sid)
        elif action == "prune":
            batch = failed = {rng.choice(indexed)}
        else:
            batch = set(rng.sample(indexed, min(len(indexed), rng.randint(1, 2))))
        for sid in batch - failed:
            if action == "retry":
                content[sid] = expected[sid]  # identical output: a hook retry
            else:
                content[sid] = (rng.randint(1, 4), next_value)
                next_value += 1
            _write_slide(pt, tiles, sid, *content[sid])

        crash = rng.random() < 0.15 and action in ("new", "reprocess")
        crash_at = rng.choice(["before_save", "after_save"]) if crash else None
        if crash_at == "before_save":
            monkeypatch.setattr(wds_mod, "_save_index",
                                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("died before save")))
        elif crash_at == "after_save":
            monkeypatch.setattr(wds_mod._AppendJournal, "mark_index_saved",
                                lambda self: (_ for _ in ()).throw(RuntimeError("died after save")))

        s3.budget = 2
        kwargs = dict(pt_dir=pt, h5_dir=tiles, inventory_df=None, wds_dest="s3://bucket/wds",
                      model_type="optimus", staging_dir=tmp_path / f"w{writer}" / f"staging{step}",
                      max_shard_bytes=rng.choice([2048, 8192, 10 ** 9]), slide_id_filter=batch,
                      lock_dir=tmp_path / "locks", slide_to_project=project,
                      failed_slides=failed or None, force_new_shard=rng.random() < 0.3)
        try:
            wds_mod.append_wds(**kwargs)
        except RuntimeError:
            assert crash
        finally:
            monkeypatch.setattr(wds_mod, "_save_index", real_save)
            monkeypatch.setattr(wds_mod._AppendJournal, "mark_index_saved", real_mark)

        if crash_at == "before_save":
            for sid in content:  # never indexed: the next locked run strips these copies
                if sid not in expected:
                    optional.pop(sid, None)
        elif crash_at == "after_save":
            # Indexed, though the old copies may remain until the next run.
            for sid, c in content.items():
                expected[sid] = c
                optional.pop(sid, None)
        elif action == "prune":
            for sid in failed:  # a stale load may not see it to prune: either is fine
                optional[sid] = expected.pop(sid)
        else:
            for sid, c in content.items():
                expected[sid] = c
                optional.pop(sid, None)
        _check(s3, expected, optional, step)
    return s3, expected


@pytest.mark.parametrize("seed", [1, 7, 23, 101])
def test_random_appends_with_stale_save_time_reads_keep_the_index_valid(tmp_path, monkeypatch, seed):
    # The merge read in _save_index (this run's entries win over a stale read): a stale copy there
    # must not point slides back at shards they were stripped from.
    s3, expected = _run_sequence(tmp_path, monkeypatch, seed, steps=60, points=(MERGE,))
    assert s3.stale_served > 5 and len(expected) > 10


@pytest.mark.xfail(strict=True, reason=(
    "Known gap: the run's own entries always win, so an index LOADED stale keeps a slide's "
    "pointer to a shard another writer since stripped it from, and a stale load plus a "
    "stale merge read drops slides outright. Needs per-entry ordering (e.g. a generation "
    "counter) rather than local-wins or remote-wins."))
@pytest.mark.parametrize("seed", [1, 7, 23])
def test_random_appends_with_stale_loads_keep_the_index_valid(tmp_path, monkeypatch, seed):
    _run_sequence(tmp_path, monkeypatch, seed, steps=60, points=(LOAD, MERGE))
