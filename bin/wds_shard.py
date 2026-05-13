#!/usr/bin/env python3
"""Create paladin-compatible WebDataset tar shards from feature files.

Called directly by Nextflow processes. For incremental, routed, S3-aware
sharding (dispatcher post-batch hooks), see mussel_dispatcher/wds.py.

Each shard is a standard tar archive where every sample consists of:
  - ``{slide_id}.features.npy``  — float32 [N_tiles, D] feature array
  - ``{slide_id}.coords.npy``    — int64  [N_tiles, 2] tile coords (optional)

The format matches the paladin training pipeline and is directly readable by
the ``webdataset`` Python library:

    import io, numpy as np, torch, webdataset as wds
    ds = wds.WebDataset("results/wds/optimus/all/000000.tar")
    for sample in ds:
        slide_id = sample["__key__"]
        features = torch.from_numpy(np.load(io.BytesIO(sample["features.npy"])))

Usage
-----
    wds_shard.py \\
        --pt_files slide1.features.pt slide2.features.pt ... \\
        --slide_ids slide1,slide2,... \\
        --output_dir ./wds_out \\
        [--h5_files slide1.patch.h5 ...] \\
        [--max_shard_size 1000] \\
        [--prefix ""]

Output
------
    {output_dir}/000000.tar
    {output_dir}/000001.tar
    ...
"""

import argparse
import math
import tarfile
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch


def _shard_path(output_dir: Path, prefix: str, index: int, total: int) -> Path:
    width = max(6, len(str(total - 1)))
    return output_dir / f"{prefix}{str(index).zfill(width)}.tar"


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _add_npy(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, BytesIO(data))


def create_shards(
    pt_files: list[Path],
    slide_ids: list[str],
    output_dir: Path,
    h5_files: list[Path] | None = None,
    max_shard_size: int = 1000,
    prefix: str = "",
) -> list[Path]:
    """Write paladin-compatible WDS tar shards and return the list of shard paths."""
    assert len(pt_files) == len(slide_ids), (
        f"pt_files ({len(pt_files)}) and slide_ids ({len(slide_ids)}) must match"
    )
    if h5_files is not None:
        assert len(h5_files) == len(slide_ids), (
            f"h5_files ({len(h5_files)}) and slide_ids ({len(slide_ids)}) must match"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(slide_ids)
    num_shards = max(1, math.ceil(n / max_shard_size))
    shard_paths: list[Path] = []

    for shard_idx in range(num_shards):
        start = shard_idx * max_shard_size
        end = min(start + max_shard_size, n)
        shard_path = _shard_path(output_dir, prefix, shard_idx, num_shards)

        with tarfile.open(shard_path, "w") as tar:
            for i in range(start, end):
                slide_id = slide_ids[i]

                # features.npy — convert .pt tensor to float32 numpy.
                # Use .to(float32) before .numpy() so bfloat16/float16 tensors
                # are handled safely (numpy has no native bfloat16 type).
                tensor = torch.load(pt_files[i], weights_only=True)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                _add_npy(tar, f"{slide_id}.features.npy", _npy_bytes(tensor.to(torch.float32).numpy()))

                # coords.npy — extract tile coordinates from .h5 (optional)
                if h5_files is not None:
                    try:
                        with h5py.File(h5_files[i], "r") as hf:
                            coords = hf["coords"][:].astype(np.int64)
                        if coords.ndim == 1:
                            coords = coords.reshape(1, -1)
                        _add_npy(tar, f"{slide_id}.coords.npy", _npy_bytes(coords))
                    except Exception as exc:
                        print(f"WARNING: could not read coords from {h5_files[i]}: {exc}")

        shard_paths.append(shard_path)
        print(f"Wrote {shard_path} ({end - start} samples)")

    return shard_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt_files",
        nargs="+",
        required=True,
        help="Paths to .pt feature files (one per slide, in same order as --slide_ids)",
    )
    parser.add_argument(
        "--slide_ids",
        required=True,
        help="Comma-separated slide IDs matching the order of --pt_files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where shard tar files will be written",
    )
    parser.add_argument(
        "--h5_files",
        nargs="+",
        default=None,
        help="Optional .h5 patch-feature files; tile coords are extracted and stored as .coords.npy",
    )
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=1000,
        help="Maximum number of slides per shard (default: 1000)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix for shards (default: '' produces 000000.tar)",
    )
    args = parser.parse_args()

    slide_ids = [s.strip() for s in args.slide_ids.split(",") if s.strip()]
    pt_files = [Path(p) for p in args.pt_files]
    h5_files = [Path(p) for p in args.h5_files] if args.h5_files else None
    output_dir = Path(args.output_dir)

    # Sort by slide_id for deterministic ordering within shards
    order = sorted(range(len(slide_ids)), key=lambda i: slide_ids[i])
    slide_ids = [slide_ids[i] for i in order]
    pt_files = [pt_files[i] for i in order]
    if h5_files is not None:
        h5_files = [h5_files[i] for i in order]

    shards = create_shards(
        pt_files=pt_files,
        slide_ids=slide_ids,
        output_dir=output_dir,
        h5_files=h5_files,
        max_shard_size=args.max_shard_size,
        prefix=args.prefix,
    )
    print(f"Created {len(shards)} shard(s) in {output_dir}")


if __name__ == "__main__":
    main()
