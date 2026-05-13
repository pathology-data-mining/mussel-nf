#!/usr/bin/env python3
"""Append new slides to per-group WDS shard directories.

Reads .features.pt and .patch.h5 files produced by the mussel-nf pipeline
and appends them to WebDataset tar shards organised by a routing key (e.g.
cancer type, oncotree code, or any project identifier):

    wds/<model>/<group>/000000.tar
    wds/<model>/<group>/000001.tar
    ...
    wds/<model>/wds_index.json   ← slide_id → {project_id, shard_file}

Slides are routed to the correct group directory using either:
  - an inventory CSV with slide_id → project_id mapping (--inventory), or
  - a column in the batch slide-ids CSV (--project-id-column), e.g. oncotree_code.

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
    # Route by inventory CSV (e.g. TCGA project_id)
    python -m mussel_dispatcher.wds \\
        --pt-dir /data/results/features/ctranspath/pt \\
        --h5-dir /data/results/features/ctranspath/h5 \\
        --inventory tcga_inventory.csv \\
        --wds-dest /data/wds \\
        --model-type ctranspath

    # Route by column in batch CSV (e.g. oncotree_code from IMPACT dispatcher)
    python -m mussel_dispatcher.wds \\
        --pt-dir /data/results/features/ctranspath/pt \\
        --slide-ids-csv batch.csv \\
        --project-id-column oncotree_code \\
        --wds-dest s3://bucket/wds \\
        --staging-dir /data/wds-staging \\
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


def _load_slide_meta(h5_path: Path) -> tuple[np.ndarray | None, float | None, bool | None]:
    """Load coords array and MPP metadata from a tile-coords HDF5 file.

    Returns ``(coords, native_mpp, mpp_is_fallback)`` where any may be ``None``
    on error.  ``mpp_is_fallback`` is ``True`` when Mussel used the 0.5 µm/px
    default because no MPP metadata was found in the slide.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:].astype(np.int64)
            attrs = dict(f["coords"].attrs)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        native_mpp = attrs.get("native_mpp")
        if native_mpp is not None:
            native_mpp = float(native_mpp)
        mpp_is_fallback = attrs.get("mpp_is_fallback")
        if mpp_is_fallback is not None:
            mpp_is_fallback = bool(mpp_is_fallback)
        return coords, native_mpp, mpp_is_fallback
    except Exception as exc:
        log.warning("Could not read slide meta from %s: %s", h5_path, exc)
        return None, None, None


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _s3_parts(s3_uri: str) -> tuple[str, str]:
    bucket, _, key = s3_uri[5:].partition("/")
    return bucket, key


def _s3_transfer_config(max_concurrency: int = 4):
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(max_concurrency=max_concurrency)


def _make_s3_client(endpoint_url: str | None = None,
                    access_key: str | None = None,
                    secret_key: str | None = None):
    """Create a boto3 S3 client, optionally targeting a custom endpoint (e.g. ECS).

    Credentials fall back to environment variables (ECS_ACCESS_KEY / ECS_SECRET_KEY
    or standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) if not explicitly provided.
    Endpoint falls back to the S3_ENDPOINT_URL environment variable if not provided.
    """
    import boto3, os
    endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL") or os.environ.get("ECS_ENDPOINT_URL")
    access_key = access_key or os.environ.get("ECS_ACCESS_KEY")
    secret_key = secret_key or os.environ.get("ECS_SECRET_KEY")
    kwargs: dict = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    from botocore.config import Config
    kwargs["config"] = Config(
        connect_timeout=10,
        read_timeout=60,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    return boto3.client("s3", **kwargs)


# Module-level client cache — populated by main() after arg parsing.
_s3_client_instance = None


def _s3_client():
    """Return the module-level S3 client (configured by main or _make_s3_client defaults)."""
    global _s3_client_instance
    if _s3_client_instance is None:
        _s3_client_instance = _make_s3_client()
    return _s3_client_instance


def _s3_upload(local_path: Path, s3_uri: str, max_concurrency: int = 4) -> None:
    bucket, key = _s3_parts(s3_uri)
    _s3_client().upload_file(
        str(local_path), bucket, key, Config=_s3_transfer_config(max_concurrency)
    )
    log.debug("Uploaded %s → %s", local_path.name, s3_uri)


def _s3_download(s3_uri: str, local_path: Path, max_concurrency: int = 4) -> bool:
    from botocore.exceptions import ClientError
    bucket, key = _s3_parts(s3_uri)
    try:
        _s3_client().download_file(
            bucket, key, str(local_path), Config=_s3_transfer_config(max_concurrency)
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _load_index(wds_dest: str, model: str, staging_dir: Path | None, s3_max_concurrency: int = 4) -> dict:
    if _is_s3(wds_dest):
        assert staging_dir, "--staging-dir required for S3 destinations"
        tmp = staging_dir / model / "wds_index.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        index_uri = f"{wds_dest.rstrip('/')}/{model}/wds_index.json"
        return json.loads(tmp.read_text()) if (_s3_download(index_uri, tmp, s3_max_concurrency) and tmp.exists()) else {}
    else:
        p = Path(wds_dest) / model / "wds_index.json"
        return json.loads(p.read_text()) if p.exists() else {}


def _save_index(index: dict, wds_dest: str, model: str, staging_dir: Path | None, s3_max_concurrency: int = 4) -> None:
    data = json.dumps(index, indent=2, sort_keys=True)
    if _is_s3(wds_dest):
        tmp = staging_dir / model / "wds_index.json"
        tmp.write_text(data)
        _s3_upload(tmp, f"{wds_dest.rstrip('/')}/{model}/wds_index.json", s3_max_concurrency)
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
        s3_max_concurrency: int = 4,
    ):
        self._wds_dest = wds_dest
        self._model = model
        self._project_id = project_id
        self._staging_dir = staging_dir
        self._max_shard_bytes = max_shard_bytes
        self._dry_run = dry_run
        self._use_s3 = _is_s3(wds_dest)
        self._s3_max_concurrency = s3_max_concurrency

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
            _s3_upload(self._current_path, s3_uri, self._s3_max_concurrency)
            self._current_path.unlink()
        self._current_index += 1
        self._current_path = None
        self._current_bytes = 0

    def flush(self) -> None:
        """Upload the current (possibly unsealed) staging shard to S3 at end of run."""
        if self._current_path and self._use_s3 and not self._dry_run:
            s3_uri = (f"{self._wds_dest.rstrip('/')}/{self._model}/"
                      f"{self._project_id}/{self._current_path.name}")
            _s3_upload(self._current_path, s3_uri, self._s3_max_concurrency)


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
    s3_max_concurrency: int = 4,
    slide_to_project: "dict[str, str] | None" = None,
    failed_slides: "set[str] | None" = None,
    also_delete_pt_dirs: "list[Path] | None" = None,
) -> dict:
    """Append all .features.pt files in pt_dir to WDS shards.

    If slide_id_filter is provided, only those slide_ids are appended.
    If delete_local is True, the source .pt and .patch.h5 files are deleted
    after all writers have been flushed (i.e. after S3 upload completes).
    If manifest_csv is set, appends rows (slide_id, model, wds_path) to that
    CSV so callers can look up the full S3 shard path for each slide.
    s3_max_concurrency limits boto3 multipart threads per upload/download.
    slide_to_project: pre-built slide_id → routing key (e.g. oncotree_code) dict.
      When provided, inventory_df is not used for routing. Useful when the
      routing key is already present in the batch CSV (e.g. Databricks watcher).
    also_delete_pt_dirs: when delete_local is True, also delete any files
      matching {slide_id}.* from these additional directories (e.g. patch encoder
      .pt files that are no longer needed after the slide encoder is in WDS).
    Returns the updated index dict.
    """
    # Build slide_id → project_id lookup: prefer explicit dict, fall back to inventory_df
    if slide_to_project is None:
        inv = inventory_df.copy()
        inv["slide_id"] = inv["file_name"].apply(lambda fn: fn.split(".")[0])
        slide_to_project = dict(zip(inv["slide_id"], inv["project_id"]))

    index = _load_index(wds_dest, model_type, staging_dir, s3_max_concurrency)
    already_indexed = set(index.keys())
    log.info("Loaded index: %d slides already in WDS", len(already_indexed))

    # Remove any stale index entries for slides that are now known failures.
    if failed_slides:
        stale = already_indexed & failed_slides
        if stale:
            log.warning("Removing %d failed slides from wds_index: %s",
                        len(stale), ", ".join(sorted(stale)[:10]))
            for sid in stale:
                del index[sid]
            already_indexed -= stale
            if not dry_run:
                _save_index(index, wds_dest, model_type, staging_dir, s3_max_concurrency)
                log.info("Pruned wds_index.json (%d entries remaining)", len(index))

    pt_files = sorted(pt_dir.rglob("*.features.pt"))
    log.info("Found %d .pt files in %s", len(pt_files), pt_dir)

    # Pre-build a slide_id → h5_path lookup to avoid O(N²) rglob calls in the loop.
    h5_lookup: dict[str, Path] = {}
    if h5_dir is not None:
        for h5_path in h5_dir.rglob("*.patch.h5"):
            sid = h5_path.name.split(".")[0]
            h5_lookup[sid] = h5_path
        log.info("Found %d .patch.h5 files in %s", len(h5_lookup), h5_dir)

    writers: dict[str, _ShardWriter] = {}
    n_appended = n_skipped = n_missing_project = 0
    appended_locals: list[tuple[Path, Path | None]] = []  # (pt_path, h5_path) for delete_local

    for pt_path in pt_files:
        slide_id = _slide_id_from_pt(pt_path)

        if slide_id_filter is not None and slide_id not in slide_id_filter:
            continue

        if failed_slides and slide_id in failed_slides:
            log.debug("Skipping failed slide: %s", slide_id)
            continue

        if slide_id in already_indexed:
            n_skipped += 1
            # Even though already in WDS, clean up the local file if requested.
            if delete_local:
                appended_locals.append((pt_path, h5_lookup.get(slide_id)))
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

        h5_path_for_slide: Path | None = h5_lookup.get(slide_id)
        coords: np.ndarray | None = None
        native_mpp: float | None = None
        mpp_is_fallback: bool | None = None
        if h5_path_for_slide is not None:
            coords, native_mpp, mpp_is_fallback = _load_slide_meta(h5_path_for_slide)

        if project_id not in writers:
            writers[project_id] = _ShardWriter(
                wds_dest=wds_dest,
                model=model_type,
                project_id=project_id,
                staging_dir=staging_dir,
                max_shard_bytes=max_shard_bytes,
                dry_run=dry_run,
                s3_max_concurrency=s3_max_concurrency,
            )

        shard_name = writers[project_id].append(slide_id, features, coords)
        index[slide_id] = {
            "project_id": project_id,
            "shard_file": f"{project_id}/{shard_name}",
            "native_mpp": native_mpp,
            "mpp_is_fallback": mpp_is_fallback,
        }
        n_appended += 1
        if delete_local:
            appended_locals.append((pt_path, h5_path_for_slide))
        if n_appended % 100 == 0:
            log.info("  %d appended, %d skipped", n_appended, n_skipped)

    # Flush unsealed staging shards to S3 — must complete before deleting local files
    for writer in writers.values():
        writer.flush()

    # Delete local source files now that data is durably in WDS / S3.
    # Includes both newly-appended slides and already-indexed slides whose
    # local files were still present (cleanup of files from prior runs).
    # Also deletes the companion .features.h5 (H5 duplicate of the .pt data)
    # that the pipeline publishes alongside every .features.pt file.
    if delete_local and not dry_run and appended_locals:
        n_deleted = 0
        for pt_path, h5_path in appended_locals:
            pt_path.unlink(missing_ok=True)
            # Delete the .features.h5 companion published next to the .pt file
            # pt_path ends in .features.pt; .with_suffix(".h5") gives .features.h5
            companion_h5 = pt_path.with_suffix(".h5")
            companion_h5.unlink(missing_ok=True)
            if h5_path:
                h5_path.unlink(missing_ok=True)
            n_deleted += 1
        log.info("Deleted %d local source file pair(s) (pt + features.h5 + patch.h5)", n_deleted)

    # Delete companion files from additional directories (e.g. patch encoder .pt files
    # whose features have been consumed by a slide encoder now safely in WDS).
    if delete_local and not dry_run and appended_locals and also_delete_pt_dirs:
        processed_ids = {_slide_id_from_pt(pt) for pt, _ in appended_locals}
        n_extra_deleted = 0
        for extra_dir in also_delete_pt_dirs:
            extra_dir = Path(extra_dir)
            if not extra_dir.exists():
                continue
            for slide_id in processed_ids:
                for f in extra_dir.glob(f"{slide_id}.*"):
                    f.unlink(missing_ok=True)
                    n_extra_deleted += 1
        if n_extra_deleted:
            log.info("Deleted %d extra file(s) from also-delete-pt-dirs", n_extra_deleted)

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
        _save_index(index, wds_dest, model_type, staging_dir, s3_max_concurrency)
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
    parser.add_argument("--inventory", default=None,
                        help="tcga_inventory.csv for project_id lookup. "
                             "Required unless --project-id-column is set.")
    parser.add_argument("--project-id-column", default=None,
                        help="Column name in --slide-ids-csv to use as the routing key "
                             "(e.g. 'oncotree_code'). When set, --inventory is not needed; "
                             "slides are routed into per-value subdirectories using this column.")
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
    parser.add_argument("--also-delete-pt-dirs", default=None,
                        help="Comma-separated list of additional directories from which to "
                             "delete all files matching {slide_id}.* when --delete-local is "
                             "active. Use to clean up patch encoder .pt files (e.g. conch1_5) "
                             "after their slide encoder (e.g. titan_slide) is written to WDS.")
    parser.add_argument("--s3-max-concurrency", type=int, default=4,
                        help="Maximum number of parallel boto3 transfer threads per S3 "
                             "upload/download (default: 4). Reduce to limit ECS endpoint load "
                             "when multiple batches are running concurrently.")
    parser.add_argument("--s3-endpoint", default=None,
                        help="Custom S3 endpoint URL (e.g. http://pmindecs.mskcc.org:9020 for "
                             "ECS). Falls back to S3_ENDPOINT_URL / ECS_ENDPOINT_URL env vars.")
    parser.add_argument("--s3-access-key", default=None,
                        help="S3 access key ID. Falls back to ECS_ACCESS_KEY env var.")
    parser.add_argument("--s3-secret-key", default=None,
                        help="S3 secret access key. Falls back to ECS_SECRET_KEY env var.")
    parser.add_argument("--status-csv", default=None,
                        help="Path to tcga_status.csv (slide_id, model, status). "
                             "Slides with status='failed' are skipped and any stale "
                             "index entries for them are removed.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Configure the module-level S3 client once (used by _s3_upload / _s3_download).
    global _s3_client_instance
    _s3_client_instance = _make_s3_client(
        endpoint_url=args.s3_endpoint,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
    )

    import pandas as pd

    # Validate: need either --inventory or --project-id-column (with --slide-ids-csv)
    if not args.project_id_column and not args.inventory:
        log.error("Either --inventory or --project-id-column (with --slide-ids-csv) must be set")
        return 1
    if args.project_id_column and not args.slide_ids_csv:
        log.error("--project-id-column requires --slide-ids-csv")
        return 1

    # Optional: restrict to a specific set of slide_ids AND/OR build routing lookup
    slide_id_filter: set[str] | None = None
    slide_to_project: dict[str, str] | None = None
    ids_df = None
    if args.slide_ids_csv:
        ids_df = pd.read_csv(args.slide_ids_csv, dtype=str).fillna("")
        if "slide_id" in ids_df.columns:
            slide_id_filter = set(ids_df["slide_id"].str.strip())
            log.info("Restricting append to %d slide_ids from %s",
                     len(slide_id_filter), args.slide_ids_csv)
        else:
            log.warning("--slide-ids-csv has no 'slide_id' column — ignoring filter")

    inventory_df = None
    if args.project_id_column:
        # Route slides by a column in the batch CSV (e.g. oncotree_code)
        if ids_df is None or "slide_id" not in ids_df.columns:
            log.error("--slide-ids-csv must have a 'slide_id' column when using --project-id-column")
            return 1
        col = args.project_id_column
        if col not in ids_df.columns:
            log.error("Column '%s' not found in %s (available: %s)",
                      col, args.slide_ids_csv, ", ".join(ids_df.columns))
            return 1
        slide_to_project = dict(zip(ids_df["slide_id"].str.strip(), ids_df[col].str.strip()))
        log.info("Routing %d slides by column '%s' from %s",
                 len(slide_to_project), col, args.slide_ids_csv)
    else:
        inventory_df = pd.read_csv(args.inventory, dtype=str).fillna("")

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
    # Build failed_slides set from status CSV (per-model, using requested model(s)).
    failed_slides: set[str] | None = None
    if args.status_csv and Path(args.status_csv).exists():
        import pandas as _pd_status
        try:
            sdf = _pd_status.read_csv(args.status_csv, dtype=str).fillna("")
            model_col = "model" if "model" in sdf.columns else None
            status_col = "status" if "status" in sdf.columns else None
            if "slide_id" in sdf.columns and status_col:
                mask = sdf[status_col].str.lower() == "failed"
                if model_col:
                    mask = mask & (sdf[model_col].isin(models_to_run))
                failed_slides = set(sdf.loc[mask, "slide_id"].str.strip())
                log.info("Status CSV: %d failed slide(s) will be excluded from WDS",
                         len(failed_slides))
        except Exception as exc:
            log.warning("Could not read --status-csv %s: %s", args.status_csv, exc)

    for model in models_to_run:
        if args.pt_dir:
            pt_dir = Path(args.pt_dir)
            h5_dir = Path(args.h5_dir) if args.h5_dir else None
        else:
            pt_dir = results_dir / "features" / model
            h5_dir = results_dir / "tiles"

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
                s3_max_concurrency=args.s3_max_concurrency,
                slide_to_project=slide_to_project,
                failed_slides=failed_slides,
                also_delete_pt_dirs=[Path(d) for d in args.also_delete_pt_dirs.split(",")]
                    if args.also_delete_pt_dirs else None,
            )
        except Exception as exc:
            log.error("Failed to append WDS for model %s: %s", model, exc)
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
