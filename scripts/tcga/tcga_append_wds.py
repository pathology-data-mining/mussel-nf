#!/usr/bin/env python3
"""Append new slides to per-cancer-type WDS shard directories.

Reads .features.pt and .patch.h5 files produced by the mussel-nf pipeline
and appends them to WebDataset tar shards organised by cancer type:

    wds/<model>/<project_id>/000000.tar
    wds/<model>/<project_id>/000001.tar
    ...
    wds/<model>/wds_index.json   ← slide_id → {project_id, shard_file}

Slides are routed to the correct project directory using tcga_inventory.csv.
Shards are capped by byte size (--max-shard-bytes) rather than slide count,
which keeps shard files consistently sized across different model feature
dimensions (e.g. ctranspath 768-d vs uni2h 1024-d).

For local destinations, shards are written directly.
For S3 destinations, a --staging-dir is required; in-progress shards are
accumulated locally and uploaded to S3 when sealed (size limit reached).
Partially-filled staging shards are uploaded at the end of each run so that
incremental progress is immediately visible on S3.

Usage
-----
    # Local destination
    python tcga_append_wds.py \\
        --pt-dir /data/tcga-results/features/ctranspath/pt \\
        --h5-dir /data/tcga-results/features/ctranspath/h5 \\
        --inventory tcga_inventory.csv \\
        --wds-dest /data/tcga-wds \\
        --model-type ctranspath

    # S3 destination
    python tcga_append_wds.py \\
        --pt-dir /data/tcga-results/features/ctranspath/pt \\
        --h5-dir /data/tcga-results/features/ctranspath/h5 \\
        --inventory tcga_inventory.csv \\
        --wds-dest s3://pathology/tcga-features/wds \\
        --staging-dir /data/tcga-wds-staging \\
        --model-type ctranspath
"""

import argparse
import csv
import io
import json
import logging
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numpy / tensor helpers (mirrors bin/wds_shard.py)
# ---------------------------------------------------------------------------

def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _add_npy_to_tar(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def _load_features(pt_path: Path) -> np.ndarray:
    tensor = torch.load(pt_path, weights_only=True)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.numpy().astype(np.float32)


def _load_coords(h5_path: Path) -> np.ndarray | None:
    try:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:].astype(np.int64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        return coords
    except Exception as exc:
        log.warning("Could not read coords from %s: %s", h5_path, exc)
        return None


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _s3_parts(s3_uri: str) -> tuple[str, str]:
    bucket, _, key = s3_uri[5:].partition("/")
    return bucket, key


def _s3_upload(local_path: Path, s3_uri: str) -> None:
    import boto3
    bucket, key = _s3_parts(s3_uri)
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    log.debug("Uploaded %s → %s", local_path.name, s3_uri)


def _s3_download(s3_uri: str, local_path: Path) -> bool:
    import boto3
    from botocore.exceptions import ClientError
    bucket, key = _s3_parts(s3_uri)
    try:
        boto3.client("s3").download_file(bucket, key, str(local_path))
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _load_index(wds_dest: str, model: str, staging_dir: Path | None) -> dict:
    if _is_s3(wds_dest):
        assert staging_dir, "--staging-dir required for S3 destinations"
        tmp = staging_dir / model / "wds_index.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        index_uri = f"{wds_dest.rstrip('/')}/{model}/wds_index.json"
        return json.loads(tmp.read_text()) if (_s3_download(index_uri, tmp) and tmp.exists()) else {}
    else:
        p = Path(wds_dest) / model / "wds_index.json"
        return json.loads(p.read_text()) if p.exists() else {}


def _save_index(index: dict, wds_dest: str, model: str, staging_dir: Path | None) -> None:
    data = json.dumps(index, indent=2, sort_keys=True)
    if _is_s3(wds_dest):
        tmp = staging_dir / model / "wds_index.json"
        tmp.write_text(data)
        _s3_upload(tmp, f"{wds_dest.rstrip('/')}/{model}/wds_index.json")
    else:
        p = Path(wds_dest) / model / "wds_index.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        tmp_p.write_text(data)
        tmp_p.replace(p)


# ---------------------------------------------------------------------------
# Per-project shard writer
# ---------------------------------------------------------------------------

class _ShardWriter:
    """Manages the current (unsealed) shard for one (model, project_id) pair."""

    def __init__(
        self,
        wds_dest: str,
        model: str,
        project_id: str,
        staging_dir: Path | None,
        max_shard_bytes: int,
        dry_run: bool = False,
    ):
        self._wds_dest = wds_dest
        self._model = model
        self._project_id = project_id
        self._staging_dir = staging_dir
        self._max_shard_bytes = max_shard_bytes
        self._dry_run = dry_run
        self._use_s3 = _is_s3(wds_dest)

        if self._use_s3:
            assert staging_dir, "--staging-dir required for S3 destinations"
            self._work_dir = staging_dir / model / project_id
        else:
            self._work_dir = Path(wds_dest) / model / project_id
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._current_path: Path | None = None
        self._current_index: int = 0
        self._current_bytes: int = 0
        self._init_current_shard()

    def _init_current_shard(self) -> None:
        shards = sorted(self._work_dir.glob("*.tar"))
        if not shards:
            return
        last = shards[-1]
        size = last.stat().st_size
        try:
            idx = int(last.stem)
        except ValueError:
            idx = len(shards) - 1
        if size < self._max_shard_bytes:
            self._current_path = last
            self._current_index = idx
            self._current_bytes = size
        else:
            self._current_index = idx + 1

    def _active_path(self) -> Path:
        return self._work_dir / f"{self._current_index:06d}.tar"

    def append(
        self, slide_id: str, features: np.ndarray, coords: np.ndarray | None
    ) -> str:
        """Write one slide into the current shard. Returns the relative shard filename."""
        feat_bytes = _npy_bytes(features)
        coord_bytes = _npy_bytes(coords) if coords is not None else None
        entry_size = len(feat_bytes) + (len(coord_bytes) if coord_bytes else 0)

        if self._current_path is not None and self._current_bytes + entry_size > self._max_shard_bytes:
            self._seal_and_rotate()

        shard_path = self._active_path()

        if self._dry_run:
            log.info("[dry-run] %s → %s/%s/%s",
                     slide_id, self._model, self._project_id, shard_path.name)
            return shard_path.name

        mode = "a" if shard_path.exists() else "w"
        with tarfile.open(shard_path, mode) as tar:
            _add_npy_to_tar(tar, f"{slide_id}.features.npy", feat_bytes)
            if coord_bytes:
                _add_npy_to_tar(tar, f"{slide_id}.coords.npy", coord_bytes)

        self._current_path = shard_path
        self._current_bytes = shard_path.stat().st_size
        return shard_path.name

    def _seal_and_rotate(self) -> None:
        """Upload the full shard to S3 (if needed) and start a new index slot."""
        if self._current_path and self._use_s3 and not self._dry_run:
            s3_uri = (f"{self._wds_dest.rstrip('/')}/{self._model}/"
                      f"{self._project_id}/{self._current_path.name}")
            _s3_upload(self._current_path, s3_uri)
            self._current_path.unlink()
        self._current_index += 1
        self._current_path = None
        self._current_bytes = 0

    def flush(self) -> None:
        """Upload the current (possibly unsealed) staging shard to S3 at end of run."""
        if self._current_path and self._use_s3 and not self._dry_run:
            s3_uri = (f"{self._wds_dest.rstrip('/')}/{self._model}/"
                      f"{self._project_id}/{self._current_path.name}")
            _s3_upload(self._current_path, s3_uri)


# ---------------------------------------------------------------------------
# Main append logic
# ---------------------------------------------------------------------------

def _slide_id_from_pt(pt_path: Path) -> str:
    return pt_path.name.replace(".features.pt", "")


def append_wds(
    pt_dir: Path,
    h5_dir: Path | None,
    inventory_df,
    wds_dest: str,
    model_type: str,
    staging_dir: Path | None,
    max_shard_bytes: int,
    dry_run: bool = False,
    slide_id_filter: set[str] | None = None,
    delete_local: bool = False,
    manifest_csv: Path | None = None,
) -> dict:
    """Append all .features.pt files in pt_dir to WDS shards.

    If slide_id_filter is provided, only those slide_ids are appended.
    If delete_local is True, the source .pt and .patch.h5 files are deleted
    after all writers have been flushed (i.e. after S3 upload completes).
    If manifest_csv is set, appends rows (slide_id, model, wds_path) to that
    CSV so callers can look up the full S3 shard path for each slide.
    Returns the updated index dict.
    """
    # Build slide_id → project_id lookup from inventory
    inv = inventory_df.copy()
    inv["slide_id"] = inv["file_name"].apply(lambda fn: fn.split(".")[0])
    slide_to_project: dict[str, str] = dict(zip(inv["slide_id"], inv["project_id"]))

    index = _load_index(wds_dest, model_type, staging_dir)
    already_indexed = set(index.keys())
    log.info("Loaded index: %d slides already in WDS", len(already_indexed))

    pt_files = sorted(pt_dir.rglob("*.features.pt"))
    log.info("Found %d .pt files in %s", len(pt_files), pt_dir)

    writers: dict[str, _ShardWriter] = {}
    n_appended = n_skipped = n_missing_project = 0
    appended_locals: list[tuple[Path, Path | None]] = []  # (pt_path, h5_path) for delete_local

    for pt_path in pt_files:
        slide_id = _slide_id_from_pt(pt_path)

        if slide_id_filter is not None and slide_id not in slide_id_filter:
            continue

        if slide_id in already_indexed:
            n_skipped += 1
            continue

        project_id = slide_to_project.get(slide_id)
        if not project_id:
            log.warning("No project_id for %s — skipping", slide_id)
            n_missing_project += 1
            continue

        try:
            features = _load_features(pt_path)
        except Exception as exc:
            log.error("Failed to load %s: %s", pt_path, exc)
            continue

        h5_path_for_slide: Path | None = None
        coords: np.ndarray | None = None
        if h5_dir is not None:
            h5_candidates = list(h5_dir.rglob(f"{slide_id}.patch.h5"))
            if h5_candidates:
                h5_path_for_slide = h5_candidates[0]
                coords = _load_coords(h5_path_for_slide)

        if project_id not in writers:
            writers[project_id] = _ShardWriter(
                wds_dest=wds_dest,
                model=model_type,
                project_id=project_id,
                staging_dir=staging_dir,
                max_shard_bytes=max_shard_bytes,
                dry_run=dry_run,
            )

        shard_name = writers[project_id].append(slide_id, features, coords)
        index[slide_id] = {
            "project_id": project_id,
            "shard_file": f"{project_id}/{shard_name}",
        }
        n_appended += 1
        if delete_local:
            appended_locals.append((pt_path, h5_path_for_slide))
        if n_appended % 100 == 0:
            log.info("  %d appended, %d skipped", n_appended, n_skipped)

    # Flush unsealed staging shards to S3 — must complete before deleting local files
    for writer in writers.values():
        writer.flush()

    # Delete local source files now that data is durably in WDS / S3
    if delete_local and not dry_run and appended_locals:
        n_deleted = 0
        for pt_path, h5_path in appended_locals:
            pt_path.unlink(missing_ok=True)
            if h5_path:
                h5_path.unlink(missing_ok=True)
            n_deleted += 1
        log.info("Deleted %d local source file pair(s) (pt + patch.h5)", n_deleted)

    # Write / append WDS manifest CSV: slide_id, model, wds_path (full S3 or local path)
    if manifest_csv is not None and not dry_run and n_appended > 0:
        manifest_csv = Path(manifest_csv)
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not manifest_csv.exists()
        with manifest_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["slide_id", "model", "wds_path"])
            if write_header:
                writer.writeheader()
            for slide_id, info in index.items():
                if slide_id not in already_indexed:
                    shard_file = info["shard_file"]
                    wds_path = f"{wds_dest.rstrip('/')}/{model_type}/{shard_file}"
                    writer.writerow({"slide_id": slide_id, "model": model_type,
                                     "wds_path": wds_path})
        log.info("Wrote %d WDS path entries to %s", n_appended, manifest_csv)

    if n_missing_project:
        log.warning("%d slides skipped (no project_id in inventory)", n_missing_project)
    log.info("Done: %d appended, %d already indexed", n_appended, n_skipped)

    if n_appended > 0 and not dry_run:
        _save_index(index, wds_dest, model_type, staging_dir)
        log.info("Updated wds_index.json (%d total entries)", len(index))

    return index


def _discover_models(results_dir: Path) -> list[str]:
    """Auto-discover model types from results/features/*/pt/ directories."""
    features_dir = results_dir / "features"
    if not features_dir.exists():
        return []
    return sorted(
        d.name for d in features_dir.iterdir()
        if d.is_dir() and (d / "pt").exists()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pt-dir", default=None,
                        help="Directory containing .features.pt files. "
                             "If omitted, --results-dir must be set and model auto-discovered.")
    parser.add_argument("--results-dir", default=None,
                        help="Nextflow results dir. Used to auto-discover models when "
                             "--pt-dir / --model-type are not specified.")
    parser.add_argument("--h5-dir", default=None,
                        help="Directory containing .patch.h5 files (coords; optional)")
    parser.add_argument("--inventory", default="tcga_inventory.csv",
                        help="tcga_inventory.csv for project_id lookup")
    parser.add_argument("--wds-dest", required=True,
                        help="WDS destination: local path or s3://bucket/prefix")
    parser.add_argument("--staging-dir", default=None,
                        help="Local staging dir for unsealed shards (required for S3)")
    parser.add_argument("--model-type", default=None,
                        help="Model type label, e.g. ctranspath. "
                             "If omitted with --results-dir, all discovered models are processed.")
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024 ** 3,
                        help="Max shard size in bytes (default: 2 GB)")
    parser.add_argument("--slide-ids-csv", default=None,
                        help="CSV with a 'slide_id' column; only append these slides. "
                             "Use to restrict each orchestrator chunk to its own outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without writing anything")
    parser.add_argument("--manifest-csv", default=None,
                        help="Path to a CSV that accumulates slide_id, model, wds_path rows "
                             "for every slide successfully added to WDS. "
                             "Appended to on each run so it builds up over time. "
                             "Created with a header row if it does not yet exist.")
    parser.add_argument("--delete-local", action="store_true",
                        help="Delete local .pt and .patch.h5 source files after they are "
                             "successfully flushed to WDS (including S3 upload). "
                             "Has no effect with --dry-run.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    import pandas as pd
    inventory_df = pd.read_csv(args.inventory, dtype=str).fillna("")

    # Optional: restrict to a specific set of slide_ids (e.g. the current orchestrator chunk)
    slide_id_filter: set[str] | None = None
    if args.slide_ids_csv:
        ids_df = pd.read_csv(args.slide_ids_csv, dtype=str).fillna("")
        if "slide_id" in ids_df.columns:
            slide_id_filter = set(ids_df["slide_id"].str.strip())
            log.info("Restricting append to %d slide_ids from %s",
                     len(slide_id_filter), args.slide_ids_csv)
        else:
            log.warning("--slide-ids-csv has no 'slide_id' column — ignoring filter")

    staging_dir = Path(args.staging_dir) if args.staging_dir else None

    # Resolve models to process
    if args.model_type:
        models_to_run = [args.model_type]
        results_dir = Path(args.results_dir) if args.results_dir else None
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        models_to_run = _discover_models(results_dir)
        if not models_to_run:
            log.error("No model output directories found under %s/features/*/pt/", args.results_dir)
            return 1
        log.info("Auto-discovered models: %s", ", ".join(models_to_run))
    else:
        log.error("Either --model-type or --results-dir must be specified")
        return 1

    rc = 0
    for model in models_to_run:
        if args.pt_dir:
            pt_dir = Path(args.pt_dir)
            h5_dir = Path(args.h5_dir) if args.h5_dir else None
        else:
            pt_dir = results_dir / "features" / model / "pt"
            h5_dir = results_dir / "features" / model / "tile_h5"

        if not pt_dir.exists():
            log.warning("pt dir does not exist, skipping: %s", pt_dir)
            continue

        try:
            append_wds(
                pt_dir=pt_dir,
                h5_dir=h5_dir if h5_dir and h5_dir.exists() else None,
                inventory_df=inventory_df,
                wds_dest=args.wds_dest,
                model_type=model,
                staging_dir=staging_dir,
                max_shard_bytes=args.max_shard_bytes,
                dry_run=args.dry_run,
                slide_id_filter=slide_id_filter,
                delete_local=args.delete_local,
                manifest_csv=Path(args.manifest_csv) if args.manifest_csv else None,
            )
        except Exception as exc:
            log.error("Failed to append WDS for model %s: %s", model, exc)
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
