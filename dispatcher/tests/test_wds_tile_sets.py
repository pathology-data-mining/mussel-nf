"""WDS coords come from the tile set the model's features were extracted on.

mussel-nf tiles each model at its recommended size: the first model's tiles
are published under tiles/, other sizes under tiles_<size>px/. append_wds is
always given --h5-dir=<outdir>/tiles. A patch model's own features.h5 (written
next to its .pt) carries the exact coords of its rows and is used first; the
tile sets are only a fallback, matched by row count. A slide encoder's vector
is written without coords.
"""
import io
import tarfile

import h5py
import numpy as np
import pandas as pd
import torch

from mussel_dispatcher import wds as wds_mod


def _write_tiles(tile_dir, sid, rows, offset):
    tile_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(tile_dir / f"{sid}.patch.h5", "w") as f:
        coords = np.arange(rows * 2, dtype=np.int64).reshape(rows, 2) + offset
        f.create_dataset("coords", data=coords)


def _shard_arrays(wds_root, model):
    out = {}
    for tar_path in (wds_root / model).rglob("*.tar"):
        with tarfile.open(tar_path, "r") as tar:
            for m in tar.getmembers():
                out[m.name] = np.load(io.BytesIO(tar.extractfile(m).read()))
    return out


def _append(tmp_path, model, pt_rows, own_coords_offset=None):
    pt_dir = tmp_path / "features" / model
    pt_dir.mkdir(parents=True)
    torch.save(torch.zeros(pt_rows, 4) if pt_rows else torch.zeros(4), pt_dir / "A.features.pt")
    if own_coords_offset is not None:  # the model's own features.h5, as FEATURIZE_BATCH writes it
        with h5py.File(pt_dir / "A.features.h5", "w") as f:
            f.create_dataset("features", data=np.zeros((pt_rows, 4), np.float32))
            ds = f.create_dataset("coords", data=np.arange(pt_rows * 2, dtype=np.int64).reshape(pt_rows, 2) + own_coords_offset)
            ds.attrs["native_mpp"] = 0.25
    wds_root = tmp_path / "wds"
    wds_mod.append_wds(
        pt_dir=pt_dir,
        h5_dir=tmp_path / "tiles",
        inventory_df=pd.DataFrame({"file_name": ["A.svs"], "project_id": ["PROJ"]}),
        wds_dest=str(wds_root),
        model_type=model,
        staging_dir=None,
        max_shard_bytes=10 * 1024 ** 3,
    )
    return _shard_arrays(wds_root, model)


def test_primary_tile_set_used_when_rows_match(tmp_path):
    _write_tiles(tmp_path / "tiles", "A", rows=6, offset=0)
    _write_tiles(tmp_path / "tiles_512px", "A", rows=2, offset=1000)
    arrays = _append(tmp_path, "hoptimus1", pt_rows=6)
    assert arrays["A.coords.npy"].shape == (6, 2)
    assert arrays["A.coords.npy"].min() == 0


def test_other_tile_set_used_for_model_at_another_size(tmp_path):
    _write_tiles(tmp_path / "tiles", "A", rows=6, offset=0)
    _write_tiles(tmp_path / "tiles_512px", "A", rows=2, offset=1000)
    arrays = _append(tmp_path, "conch1_5", pt_rows=2)
    assert arrays["A.coords.npy"].shape == (2, 2)
    assert arrays["A.coords.npy"].min() == 1000


def test_no_matching_tile_set_drops_coords(tmp_path):
    _write_tiles(tmp_path / "tiles", "A", rows=6, offset=0)
    arrays = _append(tmp_path, "conch1_5", pt_rows=2)
    assert "A.features.npy" in arrays
    assert "A.coords.npy" not in arrays


def test_own_features_h5_coords_win_when_tile_sets_tie(tmp_path):
    """A tiny slide whose 224 px and 512 px tile sets have the same row count: row
    matching would pick tiles/ for a 512 px model; its own features.h5 is exact."""
    _write_tiles(tmp_path / "tiles", "A", rows=2, offset=0)
    _write_tiles(tmp_path / "tiles_512px", "A", rows=2, offset=1000)
    arrays = _append(tmp_path, "conch1_5", pt_rows=2, own_coords_offset=1000)
    assert arrays["A.coords.npy"].min() == 1000


def test_own_coords_ignored_when_rows_differ(tmp_path):
    _write_tiles(tmp_path / "tiles", "A", rows=6, offset=0)
    pt_dir = tmp_path / "features" / "hoptimus1"
    pt_dir.mkdir(parents=True)
    with h5py.File(pt_dir / "A.features.h5", "w") as f:  # stale / partial file
        f.create_dataset("coords", data=np.zeros((3, 2), np.int64))
    torch.save(torch.zeros(6, 4), pt_dir / "A.features.pt")
    wds_mod.append_wds(pt_dir=pt_dir, h5_dir=tmp_path / "tiles",
                       inventory_df=pd.DataFrame({"file_name": ["A.svs"], "project_id": ["PROJ"]}),
                       wds_dest=str(tmp_path / "wds"), model_type="hoptimus1", staging_dir=None,
                       max_shard_bytes=10 * 1024 ** 3)
    assert _shard_arrays(tmp_path / "wds", "hoptimus1")["A.coords.npy"].shape == (6, 2)


def test_tile_set_at_another_resolution(tmp_path):
    """A set at another resolution (tiles_<size>px_<mpp>mpp/, e.g. a patch model at 1.0 µm/px) is found too."""
    _write_tiles(tmp_path / "tiles", "A", rows=6, offset=0)
    _write_tiles(tmp_path / "tiles_256px_1mpp", "A", rows=3, offset=2000)
    _write_tiles(tmp_path / "tiles_0_old", "A", rows=3, offset=9000)  # not a tile set; sorts first
    arrays = _append(tmp_path, "ctranspath", pt_rows=3)
    assert arrays["A.coords.npy"].shape == (3, 2)
    assert arrays["A.coords.npy"].min() == 2000


def _write_tiles_with_mpp(tile_dir, sid, rows, offset, native_mpp):
    _write_tiles(tile_dir, sid, rows, offset)
    with h5py.File(tile_dir / f"{sid}.patch.h5", "a") as f:
        f["coords"].attrs["native_mpp"] = native_mpp
        f["coords"].attrs["mpp_is_fallback"] = False


def _index(tmp_path, model):
    import json
    return json.loads((tmp_path / "wds" / model / "wds_index.json").read_text())


def test_slide_encoder_vector_has_no_coords_but_keeps_mpp(tmp_path):
    # A TITAN .pt is one 1-D vector; no tile set's coords belong with it.
    _write_tiles_with_mpp(tmp_path / "tiles", "A", rows=6, offset=0, native_mpp=0.2527)
    _write_tiles(tmp_path / "tiles_512px", "A", rows=2, offset=1000)
    arrays = _append(tmp_path, "titan_slide", pt_rows=0)
    assert sorted(arrays) == ["A.features.npy"]
    assert arrays["A.features.npy"].shape == (1, 4)
    assert _index(tmp_path, "titan_slide")["A"]["native_mpp"] == 0.2527


def test_slide_encoder_on_a_one_tile_slide_gets_no_coords(tmp_path):
    # One tile in each set: the 1-row vector must not pick up a tile's coords.
    _write_tiles(tmp_path / "tiles", "A", rows=1, offset=0)
    _write_tiles(tmp_path / "tiles_512px", "A", rows=1, offset=1000)
    arrays = _append(tmp_path, "titan_slide", pt_rows=0)
    assert sorted(arrays) == ["A.features.npy"]


def test_indexed_slide_encoder_is_not_given_coords_on_reappend(tmp_path):
    # An already-indexed slide vector is never "repaired" with coords from its tile file,
    # even when the tile count (1) matches its single row.
    _write_tiles(tmp_path / "tiles", "A", rows=1, offset=0)
    _append(tmp_path, "titan_slide", pt_rows=0)
    pt_dir = tmp_path / "features" / "titan_slide"
    wds_mod.append_wds(
        pt_dir=pt_dir, h5_dir=tmp_path / "tiles",
        inventory_df=pd.DataFrame({"file_name": ["A.svs"], "project_id": ["PROJ"]}),
        wds_dest=str(tmp_path / "wds"), model_type="titan_slide", staging_dir=None,
        max_shard_bytes=10 * 1024 ** 3,
    )
    assert sorted(_shard_arrays(tmp_path / "wds", "titan_slide")) == ["A.features.npy"]
