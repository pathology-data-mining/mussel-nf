#!/usr/bin/env python3
"""Create WebDataset-format .tar shards from feature files.

Each shard is a standard tar archive where every sample consists of one or more
files named ``{slide_id}.{ext}``.  WebDataset identifies samples by the stem
before the first dot, so the resulting shards are drop-in compatible with the
``webdataset`` Python library.

Usage
-----
    wds_shard.py \\
        --pt_files slide1.features.pt slide2.features.pt ... \\
        --slide_ids slide1,slide2,... \\
        --output_dir ./wds_out \\
        [--h5_files slide1.features.h5 ...] \\
        [--max_shard_size 1000] \\
        [--prefix shard-]

Output
------
    {output_dir}/shard-000000.tar
    {output_dir}/shard-000001.tar
    ...
"""

import argparse
import math
import os
import tarfile
from pathlib import Path


def _shard_path(output_dir: Path, prefix: str, index: int, total: int) -> Path:
    width = max(6, len(str(total - 1)))
    return output_dir / f"{prefix}{str(index).zfill(width)}.tar"


def create_shards(
    pt_files: list[Path],
    slide_ids: list[str],
    output_dir: Path,
    h5_files: list[Path] | None = None,
    max_shard_size: int = 1000,
    prefix: str = "shard-",
) -> list[Path]:
    """Write one or more WDS tar shards and return the list of shard paths."""
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
                # .pt file — always present
                _add_file(tar, pt_files[i], f"{slide_id}.pt")
                # .h5 file — optional
                if h5_files is not None:
                    _add_file(tar, h5_files[i], f"{slide_id}.features.h5")

        shard_paths.append(shard_path)
        print(f"Wrote {shard_path} ({end - start} samples)")

    return shard_paths


def _add_file(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(src), arcname=arcname)
    # Strip mtime/uid/gid noise for reproducibility
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with open(src, "rb") as fh:
        tar.addfile(info, fh)


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
        help="Optional .h5 patch-feature files to bundle alongside .pt files",
    )
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=1000,
        help="Maximum number of slides per shard (default: 1000)",
    )
    parser.add_argument(
        "--prefix",
        default="shard-",
        help="Filename prefix for shards (default: 'shard-')",
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
