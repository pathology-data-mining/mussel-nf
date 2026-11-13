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

    # Route by column in batch CSV (e.g. oncotree_code from Databricks watcher)
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
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import tarfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch

from .db import ResolvedDatabaseConfig, connect_database

log = logging.getLogger(__name__)


def _lock_filename_for_key(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
    return (safe or "wds") + ".lock"


@contextmanager
def _wds_append_lock(lock_dir: Path | None, wds_dest: str, model_type: str):
    """Serialize WDS index/shard mutation across dispatcher configs."""
    if lock_dir is None:
        yield
        return
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / _lock_filename_for_key(f"{wds_dest.rstrip('/')}::{model_type}")
    with lock_path.open("a+") as fh:
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        wait_started = time.monotonic()
        log.info("Waiting for WDS append lock: %s", lock_path)
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        acquired_at = time.monotonic()
        log.info(
            "Acquired WDS append lock: %s wait_seconds=%.3f",
            lock_path,
            acquired_at - wait_started,
        )
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            log.info(
                "Released WDS append lock: %s hold_seconds=%.3f",
                lock_path,
                time.monotonic() - acquired_at,
            )


class WdsLockLost(RuntimeError):
    """The Postgres WDS append lock is no longer held; another writer may append.

    Its message starts with "WDS append lock lost", which hook_hardening
    classifies as a transient service failure, so the hook is retried.
    """


# pg_advisory_lock(bigint) shows in pg_locks as classid = high 32 bits,
# objid = low 32 bits, objsubid = 1.
_ADVISORY_LOCK_HELD_SQL = (
    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND granted"
    " AND pid = pg_backend_pid() AND objsubid = 1"
    " AND ((classid::bigint << 32) | objid::bigint) = %s) AS held"
)


def _postgres_lock_key(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


@contextmanager
def _wds_postgres_lock(db_cfg: ResolvedDatabaseConfig, wds_dest: str, model_type: str):
    if db_cfg.backend != "postgres":
        raise ValueError("Postgres WDS lock requires postgres backend")
    lock_name = f"{wds_dest.rstrip('/')}::{model_type}"
    lock_key = _postgres_lock_key(lock_name)
    conn = connect_database(db_cfg, readonly=False)
    try:
        conn.execute("SET statement_timeout TO 0")
        # The dispatcher-wide lock timeout protects ordinary state writes, but
        # it must not abort a WDS advisory-lock wait. Local and remote writers
        # intentionally serialize on the same destination/model key, and a
        # normal append can take several minutes. The connection is dedicated
        # to this lock and PostgreSQL releases the advisory lock if the
        # process exits, so an unbounded wait is safe here.
        conn.execute("SET lock_timeout TO 0")
        # The connection then sits idle for the whole append. Keepalives stop
        # a firewall/NAT idle timeout from dropping it (which would release
        # the lock); verify_lock() below detects a drop that happens anyway.
        conn.execute("SET tcp_keepalives_idle TO 60")
        conn.execute("SET tcp_keepalives_interval TO 10")
        conn.execute("SET tcp_keepalives_count TO 6")
        try:
            # PostgreSQL >= 14 can end idle sessions server-side.
            conn.execute("SET idle_session_timeout TO 0")
        except Exception as exc:
            # undefined_object: the server predates the setting, so it has no
            # idle session timeout to disable.
            if getattr(exc, "sqlstate", None) != "42704":
                raise
        wait_started = time.monotonic()
        conn.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
        acquired_at = time.monotonic()
        log.info(
            "Acquired Postgres WDS append lock: %s key=%d wait_seconds=%.3f",
            lock_name,
            lock_key,
            acquired_at - wait_started,
        )

        def verify_lock() -> None:
            """Raise WdsLockLost unless this session still holds the lock."""
            try:
                row = conn.execute(_ADVISORY_LOCK_HELD_SQL, (lock_key,)).fetchone()
            except Exception as exc:
                raise WdsLockLost(
                    f"WDS append lock lost: Postgres lock {lock_name} is unverifiable: {exc}"
                ) from exc
            if not row or not row["held"]:
                raise WdsLockLost(
                    f"WDS append lock lost: Postgres lock {lock_name} is no longer held"
                )

        yield verify_lock
    finally:
        try:
            conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
            log.info(
                "Released Postgres WDS append lock: %s key=%d hold_seconds=%.3f",
                lock_name,
                lock_key,
                time.monotonic() - acquired_at if "acquired_at" in locals() else 0.0,
            )
        except Exception as exc:
            # A dead connection already released the lock server-side; do not
            # mask the error that ended the append.
            log.warning("Could not release Postgres WDS append lock %s: %s", lock_name, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass


@contextmanager
def _wds_mutation_lock(
    *,
    lock_backend: str,
    lock_dir: Path | None,
    postgres_db_cfg: ResolvedDatabaseConfig | None,
    wds_dest: str,
    model_type: str,
):
    """Hold the WDS append lock for a mutation outside append_wds.

    Yields the Postgres lock's verify function, or a no-op for file locks.
    """
    if lock_backend == "postgres":
        if postgres_db_cfg is None:
            raise ValueError("Postgres WDS lock backend requires PostgreSQL config")
        with _wds_postgres_lock(postgres_db_cfg, wds_dest, model_type) as verify_lock:
            yield verify_lock
    else:
        with _wds_append_lock(lock_dir, wds_dest, model_type):
            yield lambda: None


def _validate_shared_lock_args(*, wds_dest: str, lock_dir: Path | None) -> None:
    if _is_s3(wds_dest) and lock_dir is None:
        raise ValueError(
            "S3/ECS WDS mutation requires --lock-dir on shared storage to avoid shard/index races"
        )


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


def _slide_id_from_member(name: str) -> str | None:
    if name.endswith(".features.npy"):
        return name[:-len(".features.npy")]
    if name.endswith(".coords.npy"):
        return name[:-len(".coords.npy")]
    return None


def _rewrite_tar_excluding_slides(tar_path: Path, slide_ids: set[str]) -> bool:
    """Rewrite a tar in place without entries for slide_ids.

    Returns True if any entry was removed. Used before retry appends so a failed
    index save followed by resume does not leave duplicate tar member names.
    """
    if not tar_path.exists() or not slide_ids:
        return False

    tmp_path = tar_path.with_suffix(tar_path.suffix + ".tmp")
    removed = 0
    try:
        with tarfile.open(tar_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
            for member in src.getmembers():
                sid = _slide_id_from_member(member.name)
                if sid in slide_ids:
                    removed += 1
                    continue
                fh = src.extractfile(member) if member.isfile() else None
                dst.addfile(member, fh)
                if fh is not None:
                    fh.close()
        if removed:
            tmp_path.replace(tar_path)
            log.warning("Compacted %s: removed %d stale duplicate member(s)", tar_path, removed)
            return True
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_npy_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    fh = tar.extractfile(member)
    if fh is None:
        raise RuntimeError(f"Could not read tar member {member.name}")
    try:
        return np.load(io.BytesIO(fh.read()))
    finally:
        fh.close()


def _repair_wds_tar(
    tar_path: Path,
    *,
    h5_lookup: dict[str, Path] | None = None,
    require_coords: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Compact duplicate WDS tar entries and restore coords when possible.

    For each slide, keeps the last features entry and one coords entry matching
    feature row count. If no matching coords entry exists, a local .patch.h5 can
    provide coords via h5_lookup. This repairs retry-created duplicate tar
    members without rerunning feature extraction.

    TODO: When compacting titan_slide shards, drop coords members
    entirely. Titan slide embeddings are the payload; coords are currently being
    packaged as shared patch-level baggage and should be removed during the
    planned compaction cleanup.
    """
    stats = {
        "slides": 0,
        "duplicate_slides": 0,
        "missing_coords": 0,
        "repaired_coords_from_h5": 0,
        "mismatched_coords": 0,
        "rewritten": 0,
    }
    if not tar_path.exists():
        return stats

    grouped: dict[str, dict[str, list[tarfile.TarInfo]]] = {}
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            sid = _slide_id_from_member(member.name)
            if sid is None:
                continue
            kind = "features" if member.name.endswith(".features.npy") else "coords"
            grouped.setdefault(sid, {"features": [], "coords": []})[kind].append(member)

        repaired: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        for sid, parts in grouped.items():
            stats["slides"] += 1
            if len(parts["features"]) > 1 or len(parts["coords"]) > 1:
                stats["duplicate_slides"] += 1
            if not parts["features"]:
                continue

            features = _read_npy_member(tar, parts["features"][-1])
            feature_rows = features.shape[0] if features.ndim > 1 else 1
            coords = None
            for coord_member in reversed(parts["coords"]):
                candidate = _read_npy_member(tar, coord_member)
                coord_rows = candidate.shape[0] if candidate.ndim > 1 else 1
                if coord_rows == feature_rows:
                    coords = candidate
                    break
                stats["mismatched_coords"] += 1

            if coords is None and h5_lookup and sid in h5_lookup:
                candidate = _load_coords(h5_lookup[sid])
                if candidate is not None:
                    coord_rows = candidate.shape[0] if candidate.ndim > 1 else 1
                    if coord_rows == feature_rows:
                        coords = candidate
                        stats["repaired_coords_from_h5"] += 1
                    else:
                        stats["mismatched_coords"] += 1

            if coords is None and require_coords:
                stats["missing_coords"] += 1
            repaired.append((sid, features, coords))

    expected_members = sum(1 + (1 if coords is not None else 0) for _, _, coords in repaired)
    existing_data_members = sum(
        len(parts["features"]) + len(parts["coords"]) for parts in grouped.values()
    )
    needs_rewrite = (
        existing_data_members != expected_members
        or stats["duplicate_slides"] > 0
        or stats["repaired_coords_from_h5"] > 0
    )
    if not needs_rewrite or dry_run:
        return stats

    tmp_path = tar_path.with_suffix(tar_path.suffix + ".repair.tmp")
    try:
        with tarfile.open(tmp_path, "w") as out:
            for sid, features, coords in repaired:
                _add_npy_to_tar(out, f"{sid}.features.npy", _npy_bytes(features))
                if coords is not None:
                    _add_npy_to_tar(out, f"{sid}.coords.npy", _npy_bytes(coords))
        tmp_path.replace(tar_path)
        stats["rewritten"] = 1
        return stats
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_features(pt_path: Path) -> np.ndarray:
    tensor = torch.load(pt_path, weights_only=True)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dtype == torch.bfloat16:
        # numpy has no native bfloat16; store raw bits as uint16
        # Paladin's WDS loader detects uint16 and reinterprets as bfloat16
        return tensor.view(torch.int16).numpy().view(np.uint16)
    return tensor.numpy()


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
                    secret_key: str | None = None,
                    aws_profile: str | None = None):
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
    session_kwargs: dict = {}
    use_explicit_keys = bool(access_key and secret_key)
    if aws_profile and not use_explicit_keys:
        session_kwargs["profile_name"] = aws_profile
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if access_key and use_explicit_keys:
        kwargs["aws_access_key_id"] = access_key
    if secret_key and use_explicit_keys:
        kwargs["aws_secret_access_key"] = secret_key
    from botocore.config import Config
    kwargs["config"] = Config(
        connect_timeout=10,
        read_timeout=60,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    session = boto3.Session(**session_kwargs)
    return session.client("s3", **kwargs)


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
    local_size = local_path.stat().st_size
    _s3_client().upload_file(
        str(local_path), bucket, key, Config=_s3_transfer_config(max_concurrency)
    )
    # Verify the object exists and has the correct size after upload.
    # Some S3-compatible backends (e.g. ECS under storage pressure) can return a
    # successful upload response while buffering the write — if storage is full the
    # object is silently evicted from cache and never committed to disk.  Checking
    # ContentLength here detects both "key not found" and "partial write" cases and
    # ensures the manifest is never written with a path pointing to a missing shard.
    head = _s3_client().head_object(Bucket=bucket, Key=key)
    remote_size = head["ContentLength"]
    if remote_size != local_size:
        raise RuntimeError(
            f"Upload size mismatch for {s3_uri}: "
            f"uploaded {local_size} bytes but S3 reports {remote_size} bytes"
        )
    log.debug("Uploaded %s → %s (%d bytes)", local_path.name, s3_uri, local_size)


def _s3_download(s3_uri: str, local_path: Path, max_concurrency: int = 4) -> bool:
    from botocore.exceptions import ClientError
    bucket, key = _s3_parts(s3_uri)
    try:
        _s3_client().download_file(
            bucket, key, str(local_path), Config=_s3_transfer_config(max_concurrency)
        )
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return False
        # Some S3-compatible endpoints (e.g. ECS) return 403/AccessDenied for
        # non-existent objects when the bucket exists but the key does not.
        # A 403 can equally mean an existing object we may not read, and
        # treating that as missing makes _load_index start from {} and the
        # next save overwrite the index, so only a listing that confirms the
        # key is absent makes it "not found".
        if code in ("403", "AccessDenied") and not _s3_key_listed(bucket, key):
            log.warning("Got %s on %s and the key is not listed — treating as not found",
                        code, s3_uri)
            return False
        raise


def _s3_key_listed(bucket: str, key: str) -> bool:
    resp = _s3_client().list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(obj.get("Key") == key for obj in resp.get("Contents", []))


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


def _save_index(
    index: dict,
    wds_dest: str,
    model: str,
    staging_dir: Path | None,
    s3_max_concurrency: int = 4,
    removed: "set[str] | None" = None,
) -> None:
    """Write the index; for S3, first add remote entries this run's copy lacks.

    This run's entries always win: the remote read may be stale (older than the
    index this run loaded under the lock), and taking its entry for a slide another
    writer re-appended would point the slide back at a shard it was stripped from.
    """
    removed = removed or set()
    if _is_s3(wds_dest):
        tmp = staging_dir / model / "wds_index.json"
        # ECS can return a stale object after a previous writer has uploaded a
        # newer index.  The Postgres lock serializes mutations, but it cannot
        # make an eventually-consistent object read current.  Merge the latest
        # readable remote snapshot before saving so an append can never discard
        # entries that another completed hook already published.  Entries this
        # writer deliberately removed are excluded from the merge; otherwise
        # every prune is silently undone by the next save.
        remote_tmp = tmp.with_name(f".wds_index.remote.{os.getpid()}.json")
        try:
            if _s3_download(
                f"{wds_dest.rstrip('/')}/{model}/wds_index.json",
                remote_tmp,
                s3_max_concurrency,
            ) and remote_tmp.exists():
                remote_index = json.loads(remote_tmp.read_text())
                remote_entries = {
                    str(key): value
                    for key, value in remote_index.items()
                    if not str(key).startswith("_")
                }
                before = len(index)
                for slide_id, entry in remote_entries.items():
                    if slide_id not in removed:
                        index.setdefault(slide_id, entry)
                merged = len(index) - before
                if merged:
                    log.warning(
                        "Merged %d remote WDS index entries before save for %s (%d total)",
                        merged,
                        model,
                        len(index),
                    )
        except (OSError, ValueError, TypeError) as exc:
            # Do not turn a protective read into a new hook failure.  The
            # normal upload still has its existing size verification, and the
            # Postgres mutation lock remains held for the whole operation.
            log.warning("Could not merge current remote WDS index for %s: %s", model, exc)
        finally:
            remote_tmp.unlink(missing_ok=True)
        data = json.dumps(index, indent=2, sort_keys=True)
        tmp.write_text(data)
        _s3_upload(tmp, f"{wds_dest.rstrip('/')}/{model}/wds_index.json", s3_max_concurrency)
    else:
        data = json.dumps(index, indent=2, sort_keys=True)
        p = Path(wds_dest) / model / "wds_index.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        tmp_p.write_text(data)
        tmp_p.replace(p)


def _build_h5_lookup(h5_dir: Path | None) -> dict[str, Path]:
    h5_lookup: dict[str, Path] = {}
    if h5_dir is not None:
        for h5_path in h5_dir.rglob("*.patch.h5"):
            sid = h5_path.name.split(".")[0]
            h5_lookup[sid] = h5_path
        log.info("Found %d .patch.h5 files in %s", len(h5_lookup), h5_dir)
    return h5_lookup


def _delete_consumed_coords(h5_lookup: dict[str, Path], slide_ids: set[str],
                            pt_dirs: list[Path]) -> int:
    """Delete the .patch.h5 of each slide no pt_dir still holds a .features.pt for.

    A .pt left in any model's dir is an append still to come (a failed hook's
    retry), and that append reads these coords.
    """
    pending = {_slide_id_from_pt(pt) for d in pt_dirs if Path(d).exists()
               for pt in Path(d).rglob("*.features.pt")}
    n = 0
    for slide_id in sorted(slide_ids - pending):
        h5_path = h5_lookup.get(slide_id)
        if h5_path is not None and h5_path.exists():
            h5_path.unlink(missing_ok=True)
            n += 1
    return n


def _own_coords(pt_path: Path, features: np.ndarray):
    """(coords, native_mpp, mpp_is_fallback) from the model's own features.h5, if it has
    coords for every feature row; else None. Patch models write one next to the .pt."""
    if features.ndim < 2:
        return None
    own_h5 = pt_path.with_name(pt_path.name[: -len(".pt")] + ".h5") if pt_path.name.endswith(".pt") else None
    if own_h5 is None or not own_h5.exists():
        return None
    try:
        with h5py.File(own_h5, "r") as f:
            if "coords" not in f:
                return None
    except Exception:
        return None
    coords, native_mpp, mpp_is_fallback = _load_slide_meta(own_h5)
    if coords is None or len(coords) != len(features):
        return None
    return coords, native_mpp, mpp_is_fallback


_TILE_SET_DIR = re.compile(r"tiles_\d+px(_[0-9.]+mpp)?")


def _alt_paths(lookups: list[dict[str, Path]], slide_id: str) -> list[Path]:
    return [lk[slide_id] for lk in lookups if slide_id in lk]


def _build_alt_h5_lookups(h5_dir: Path | None) -> list[dict[str, Path]]:
    """Lookups for the other tile sets next to h5_dir (tiles_<size>px[_<mpp>mpp]/).

    mussel-nf tiles each model at its recommended size and resolution; the
    first model's tile set is published under tiles/, any other size under
    tiles_<size>px/, and a set at another resolution under
    tiles_<size>px_<mpp>mpp/. A patch model's coords come from its own
    features.h5, else from whichever set matches its feature row count
    (_load_slide_payload).
    """
    if h5_dir is None or h5_dir.name != "tiles" or not h5_dir.parent.exists():
        return []
    return [_build_h5_lookup(d) for d in sorted(h5_dir.parent.glob("tiles_*"))
            if d.is_dir() and _TILE_SET_DIR.fullmatch(d.name)]


def repair_wds_shards(
    wds_dest: str,
    model: str,
    staging_dir: Path | None,
    h5_dir: Path | None = None,
    *,
    require_coords: bool = False,
    dry_run: bool = False,
    s3_max_concurrency: int = 4,
    verify_lock: "Callable[[], None] | None" = None,
) -> dict[str, int]:
    """Repair local/staged WDS shards for one model and upload replacements if S3."""
    verify_lock = verify_lock or (lambda: None)
    use_s3 = _is_s3(wds_dest)
    if use_s3:
        assert staging_dir, "--staging-dir required for S3 destinations"
        root = staging_dir / model
    else:
        root = Path(wds_dest) / model
    h5_lookup = _build_h5_lookup(h5_dir)
    totals = {
        "shards": 0,
        "rewritten": 0,
        "slides": 0,
        "duplicate_slides": 0,
        "missing_coords": 0,
        "repaired_coords_from_h5": 0,
        "mismatched_coords": 0,
    }
    if use_s3:
        # Staging no longer mirrors S3 (sealed shards are deleted once indexed):
        # repair the S3 objects, one temporary download at a time.
        shard_files = _list_s3_shards(wds_dest, model)
    elif root.exists():
        shard_files = [str(p.relative_to(root)) for p in sorted(root.rglob("*.tar"))]
    else:
        return totals
    for shard_file in shard_files:
        tar_path = root / ("_repair" if use_s3 else "") / shard_file
        if use_s3:
            uri = f"{wds_dest.rstrip('/')}/{model}/{shard_file}"
            members = _s3_tar_index(uri)
            # Headers first: download only shards _repair_wds_tar would change (a
            # model holds ~74k shards of ~2 GiB), and never for a dry run.
            if members is None:
                continue
            if not _s3_shard_needs_repair(uri, members, None, require_coords=require_coords,
                                          with_tiles=h5_lookup):
                totals["shards"] += 1
                continue
            if dry_run:
                totals["shards"] += 1
                totals["would_repair"] = totals.get("would_repair", 0) + 1
                continue
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            if not _s3_download(uri, tar_path, s3_max_concurrency):
                continue
        if not use_s3 and not dry_run:
            verify_lock()  # rewritten in place
        try:
            stats = _repair_wds_tar(
                tar_path,
                h5_lookup=h5_lookup,
                require_coords=require_coords,
                dry_run=dry_run,
            )
            if stats["rewritten"] and use_s3 and not dry_run:
                verify_lock()
                _s3_upload(tar_path, f"{wds_dest.rstrip('/')}/{model}/{shard_file}", s3_max_concurrency)
        finally:
            if use_s3:
                tar_path.unlink(missing_ok=True)
        totals["shards"] += 1
        for key in (
            "rewritten",
            "slides",
            "duplicate_slides",
            "missing_coords",
            "repaired_coords_from_h5",
            "mismatched_coords",
        ):
            totals[key] += stats[key]
    return totals


def _list_s3_shards(wds_dest: str, model: str) -> list[str]:
    """``<project>/<NNNNNN>.tar`` for every shard of one model on S3."""
    bucket, prefix = _s3_parts(f"{wds_dest.rstrip('/')}/{model}/")
    names = []
    for page in _s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if rel.endswith(".tar") and "/" in rel and not rel.startswith("_"):
                names.append(rel)
    return sorted(names)


# (uri, etag) -> {member name: [(data offset, size), ...]}. Keyed by ETag, so a shard
# rewritten in between is scanned again; bounded, since one append touches few shards.
_S3_TAR_INDEX_CACHE: dict[tuple[str, str, int], dict[str, list[tuple[int, int]]]] = {}
_S3_TAR_INDEX_CACHE_MAX = 256


def _s3_tar_index(uri: str) -> dict[str, list[tuple[int, int]]] | None:
    """Members of an uncompressed tar on S3 -> [(data offset, size)], from its headers alone.

    None if the object does not exist. One 512-byte range read per member instead of
    downloading a ~2 GiB shard; a name listed twice is a duplicate entry.
    """
    bucket, key = _s3_parts(uri)
    client = _s3_client()
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    size = head["ContentLength"]
    etag = str(head.get("ETag") or "").strip('"')
    cache_key = (uri, etag, size) if etag else None    # no ETag: nothing to key a cache on
    if cache_key in _S3_TAR_INDEX_CACHE:
        return _S3_TAR_INDEX_CACHE[cache_key]
    members, pos = {}, 0
    while pos + 512 <= size:
        block = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={pos}-{pos + 511}")["Body"].read()
        if block == b"\0" * 512:
            break
        info = tarfile.TarInfo.frombuf(block, tarfile.ENCODING, "surrogateescape")
        members.setdefault(info.name, []).append((pos + 512, info.size))
        pos += 512 + math.ceil(info.size / 512) * 512
    if cache_key is not None:
        if len(_S3_TAR_INDEX_CACHE) >= _S3_TAR_INDEX_CACHE_MAX:
            _S3_TAR_INDEX_CACHE.clear()
        _S3_TAR_INDEX_CACHE[cache_key] = members
    return members


def _s3_npy_rows(uri: str, offset: int, size: int) -> int:
    """Row count of an .npy member, as _repair_wds_tar counts it, from its header only."""
    bucket, key = _s3_parts(uri)
    head = _s3_client().get_object(
        Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset + min(size, 4096) - 1}")["Body"].read()
    buf = io.BytesIO(head)
    major, _minor = np.lib.format.read_magic(buf)
    read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
    shape, _fortran, _dtype = read_header(buf)
    return shape[0] if len(shape) > 1 else 1


def _s3_shard_needs_repair(uri: str, members: dict[str, list[tuple[int, int]]],
                           slide_ids: set[str] | None, *, require_coords: bool,
                           with_tiles: "set[str] | dict" = frozenset()) -> bool:
    """Whether _repair_wds_tar would change anything for these slides (all if None).

    Mirrors its rules from headers: a duplicate entry, coords whose row count differs
    from the features', or no coords where they are required or a tile file
    (``with_tiles``, slide ids with a .patch.h5) can supply them.
    """
    grouped: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for name, entries in members.items():
        sid = _slide_id_from_member(name)
        if sid is None or (slide_ids is not None and sid not in slide_ids):
            continue
        kind = "features" if name.endswith(".features.npy") else "coords"
        grouped.setdefault(sid, {"features": [], "coords": []})[kind].extend(entries)
    for sid, parts in grouped.items():
        if len(parts["features"]) > 1 or len(parts["coords"]) > 1:
            return True
        if not parts["features"]:
            continue
        if not parts["coords"]:
            if require_coords or sid in with_tiles:
                return True
            continue
        if _s3_npy_rows(uri, *parts["coords"][0]) != _s3_npy_rows(uri, *parts["features"][0]):
            return True
    return False


def _local_shard_path_for_index_entry(
    wds_dest: str,
    model_type: str,
    staging_dir: Path | None,
    shard_file: str,
) -> Path:
    if _is_s3(wds_dest):
        assert staging_dir, "--staging-dir required for S3 destinations"
        return staging_dir / model_type / shard_file
    return Path(wds_dest) / model_type / shard_file


def _repair_indexed_slide_coords_if_possible(
    *,
    slide_id: str,
    index_entry: dict,
    wds_dest: str,
    model_type: str,
    staging_dir: Path | None,
    h5_lookup: dict[str, Path],
    dry_run: bool,
    s3_max_concurrency: int,
    verify_lock: "Callable[[], None] | None" = None,
) -> dict[str, int]:
    """Repair coords for one already-indexed slide if its shard is reachable.

    This is the incremental recovery path for batches that re-emit a slide
    whose WDS entry exists but lacks coords. The shard is rewritten in place
    from local .patch.h5 when available, then uploaded back to S3 if needed.
    """
    stats = {
        "checked": 0,
        "downloaded": 0,
        "repaired": 0,
        "missing_h5": 0,
        "missing_shard": 0,
        "mismatched_coords": 0,
    }
    h5_path = h5_lookup.get(slide_id)
    if h5_path is None:
        stats["missing_h5"] = 1
        return stats

    shard_file = str(index_entry.get("shard_file") or "").strip()
    if not shard_file:
        stats["missing_shard"] = 1
        return stats

    temp_copy = False
    if _is_s3(wds_dest):
        # S3 is the copy the index points at; staging may hold a stale one. Read
        # the tar headers first and download only if this slide lacks coords.
        shard_uri = f"{wds_dest.rstrip('/')}/{model_type}/{shard_file}"
        members = _s3_tar_index(shard_uri)
        if members is None:
            stats["missing_shard"] = 1
            return stats
        if not _s3_shard_needs_repair(shard_uri, members, {slide_id}, require_coords=True):
            stats["checked"] = 1
            return stats
        tar_path = staging_dir / model_type / "_repair" / shard_file
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        if not _s3_download(shard_uri, tar_path, s3_max_concurrency):
            stats["missing_shard"] = 1
            return stats
        stats["downloaded"] = 1
        temp_copy = True
    else:
        tar_path = _local_shard_path_for_index_entry(wds_dest, model_type, staging_dir, shard_file)
        if not tar_path.exists():
            stats["missing_shard"] = 1
            return stats
    try:
        stats = _repair_coords_in_tar(
            stats, tar_path, slide_id=slide_id, h5_path=h5_path, wds_dest=wds_dest,
            model_type=model_type, shard_file=shard_file, dry_run=dry_run,
            s3_max_concurrency=s3_max_concurrency, verify_lock=verify_lock)
    finally:
        if temp_copy:
            tar_path.unlink(missing_ok=True)
    staged = staging_dir / model_type / shard_file if temp_copy else None
    if staged is not None and stats["repaired"] and not dry_run and staged.exists():
        # A writer may hold this shard open and upload its staged copy at flush;
        # repair that copy too, or the flush undoes the repair. Best effort: S3 is
        # already repaired, and a truncated staged copy (a killed hook's) must not
        # fail the hook; quarantine it so no writer reopens it.
        try:
            _repair_wds_tar(staged, h5_lookup={slide_id: h5_path}, require_coords=True, dry_run=False)
        except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
            quarantine = staged.with_name(f"{staged.name}.corrupt.{int(time.time())}")
            try:
                staged.replace(quarantine)
            except OSError:
                quarantine = staged
            log.warning("Could not repair staged WDS shard %s (%s); quarantined as %s",
                        staged, exc, quarantine)
    return stats


def _repair_coords_in_tar(
    stats: dict[str, int],
    tar_path: Path,
    *,
    slide_id: str,
    h5_path: Path,
    wds_dest: str,
    model_type: str,
    shard_file: str,
    dry_run: bool,
    s3_max_concurrency: int,
    verify_lock: "Callable[[], None] | None",
) -> dict[str, int]:
    """Rewrite one slide's coords in ``tar_path`` from its .patch.h5; upload it back if S3."""
    stats["checked"] = 1
    verify_lock = verify_lock or (lambda: None)
    if not _is_s3(wds_dest) and not dry_run:
        verify_lock()  # rewritten in place
    repair = _repair_wds_tar(
        tar_path,
        h5_lookup={slide_id: h5_path},
        require_coords=True,
        dry_run=dry_run,
    )
    if repair["mismatched_coords"] > 0:
        stats["mismatched_coords"] = 1
    if repair["repaired_coords_from_h5"] > 0:
        stats["repaired"] = repair["repaired_coords_from_h5"]
        if repair["rewritten"] and _is_s3(wds_dest) and not dry_run:
            verify_lock()
            shard_uri = f"{wds_dest.rstrip('/')}/{model_type}/{shard_file}"
            _s3_upload(tar_path, shard_uri, s3_max_concurrency)
    return stats


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
        force_new_shard: bool = False,
        journal: "_AppendJournal | None" = None,
        verify_lock: "Callable[[], None] | None" = None,
    ):
        self._wds_dest = wds_dest
        self._model = model
        self._project_id = project_id
        self._staging_dir = staging_dir
        self._max_shard_bytes = max_shard_bytes
        self._dry_run = dry_run
        self._use_s3 = _is_s3(wds_dest)
        self._s3_max_concurrency = s3_max_concurrency
        self._force_new_shard = force_new_shard
        self._journal = journal
        self._verify_lock = verify_lock or (lambda: None)

        if self._use_s3:
            assert staging_dir, "--staging-dir required for S3 destinations"
            self._work_dir = staging_dir / model / project_id
        else:
            self._work_dir = Path(wds_dest) / model / project_id
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._current_path: Path | None = None
        self._current_index: int = 0
        self._current_bytes: int = 0
        self._local_slide_locations: dict[str, set[Path]] | None = None
        self._sealed_staging: list[Path] = []  # uploaded and size-verified; see discard_sealed_staging
        self._init_current_shard()

    def _init_current_shard(self) -> None:
        if self._force_new_shard and self._use_s3:
            # Recovery runs should not download the previous (often ~2 GiB)
            # shard for every project merely to append a handful of missing
            # slides.  Allocate a collision-free shard and leave existing
            # ECS objects untouched; the index will point at the new shard.
            self._current_index = self._next_s3_shard_index()
            return
        shards = sorted(self._work_dir.glob("*.tar"))
        if shards:
            last = shards[-1]
            size = last.stat().st_size
            try:
                idx = int(last.stem)
            except ValueError:
                idx = len(shards) - 1
            try:
                with tarfile.open(last, "r") as tar:
                    tar.getmembers()
            except (OSError, tarfile.TarError) as exc:
                if not self._use_s3:
                    # For a local destination this is the published shard, which
                    # the index may point at: renaming it away would drop indexed
                    # slides. Stop and let an operator repair it.
                    raise RuntimeError(
                        f"Published WDS shard {last} is unreadable ({exc}); repair it "
                        "before appending (renaming it would lose indexed slides)"
                    ) from exc
                # A prior hook can be killed while tarfile is appending, leaving
                # a truncated staging shard. Never append to it or overwrite a
                # possibly healthy object with the same name in ECS. Preserve
                # the bad file for diagnosis and advance to a fresh shard slot.
                quarantine = last.with_name(
                    f"{last.name}.corrupt.{int(time.time())}"
                )
                try:
                    last.replace(quarantine)
                except OSError:
                    quarantine = last
                next_index = idx + 1
                if self._use_s3:
                    next_index = max(next_index, self._next_s3_shard_index())
                self._current_index = next_index
                log.warning(
                    "Quarantined corrupt WDS staging shard %s (%s); using %06d",
                    quarantine, exc, self._current_index,
                )
                return
            if size < self._max_shard_bytes and self._remote_matches_local(idx, size, last):
                self._current_path = last
                self._current_index = idx
                self._current_bytes = size
            else:
                # Either full, or another writer (a different staging dir) has
                # since uploaded different content under this shard name.
                # Appending to our stale copy and re-uploading would silently
                # drop that writer's slides while the index still points at
                # the shard, so allocate a fresh slot past everything on S3.
                next_index = idx + 1
                if self._use_s3:
                    next_index = max(next_index, self._next_s3_shard_index())
                self._current_index = next_index
        elif self._use_s3:
            # No local staging — check S3 to avoid overwriting existing shards.
            # Start the shard index at max_existing_s3_index + 1 so we never
            # clobber shards from a previous run that were already uploaded.
            self._current_index = self._next_s3_shard_index()

    def _next_s3_shard_index(self) -> int:
        """Return the next shard index that doesn't collide with existing S3 objects.

        A failed listing propagates: guessing (e.g. 0) would overwrite a shard
        that holds indexed slides.
        """
        client = _s3_client()
        bucket, prefix = _s3_parts(
            f"{self._wds_dest.rstrip('/')}/{self._model}/{self._project_id}/"
        )
        paginator = client.get_paginator("list_objects_v2")
        max_idx = -1
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                stem = obj["Key"].split("/")[-1].replace(".tar", "")
                try:
                    max_idx = max(max_idx, int(stem))
                except ValueError:
                    pass
        return max_idx + 1 if max_idx >= 0 else 0

    def _remote_matches_local(self, idx: int, local_size: int, local_path: Path) -> bool:
        """True if our staged copy of shard ``idx`` is the same size as the S3 object,
        or the S3 object is absent and the staged copy holds no slides."""
        if not self._use_s3 or self._dry_run:
            return True
        bucket, key = _s3_parts(
            f"{self._wds_dest.rstrip('/')}/{self._model}/{self._project_id}/{idx:06d}.tar"
        )
        try:
            remote_size = _s3_client().head_object(Bucket=bucket, Key=key)["ContentLength"]
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                # Every run uploads its staging shards before saving the index,
                # so slides in a staged copy that is not on S3 are unindexed:
                # left by a dead run, or stripped and the emptied shard deleted.
                # Re-uploading them would publish unindexed duplicates.
                if _tar_slide_ids(local_path):
                    log.warning("Staged shard %06d holds slides but is not on S3; not reusing it", idx)
                    return False
                return True
            log.warning("Could not verify staged shard %06d against S3 (%s); not reusing it", idx, exc)
            return False
        if remote_size != local_size:
            log.warning(
                "Staged shard %s/%s/%06d.tar (%d bytes) differs from S3 (%d bytes); "
                "another writer owns it, starting a new shard",
                self._model, self._project_id, idx, local_size, remote_size,
            )
            return False
        return True

    def _active_path(self) -> Path:
        return self._work_dir / f"{self._current_index:06d}.tar"

    def _build_local_slide_locations(self) -> dict[str, set[Path]]:
        locations: dict[str, set[Path]] = {}
        for shard_path in sorted(self._work_dir.glob("*.tar")):
            try:
                with tarfile.open(shard_path) as tar:
                    for member in tar.getmembers():
                        sid = _slide_id_from_member(member.name)
                        if sid is not None:
                            locations.setdefault(sid, set()).add(shard_path)
            except tarfile.TarError as exc:
                log.warning("Could not inspect existing shard %s: %s", shard_path, exc)
        return locations

    def _remove_existing_slide_entries(self, slide_id: str, keep: Path | None = None) -> None:
        """Remove slide_id from this writer's shards, except ``keep`` (its indexed copy)."""
        if self._local_slide_locations is None:
            self._local_slide_locations = self._build_local_slide_locations()
        locations = self._local_slide_locations.pop(slide_id, set())
        for shard_path in sorted(locations - {keep}):
            if shard_path.exists():
                _rewrite_tar_excluding_slides(shard_path, {slide_id})
        if keep in locations:
            self._local_slide_locations[slide_id] = {keep}
        if self._current_path and self._current_path.exists():
            self._current_bytes = self._current_path.stat().st_size

    def append(
        self,
        slide_id: str,
        features: np.ndarray,
        coords: np.ndarray | None,
        indexed_shard: str | None = None,
    ) -> str:
        """Write one slide into the current shard. Returns the relative shard filename.

        ``indexed_shard`` is the index's current shard_file for a re-processed
        slide. A local destination is written in place, so that copy is kept
        until the index points at the new one (append_wds strips it after the
        save); a crash before the save must not leave the slide nowhere.
        """
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

        keep = None
        if indexed_shard and not self._use_s3:
            keep = self._work_dir.parent / indexed_shard
            if keep == shard_path:
                # Never write the new copy next to the indexed one.
                self._seal_and_rotate()
                shard_path = self._active_path()
        self._remove_existing_slide_entries(slide_id, keep=keep)
        if self._journal is not None and not self._use_s3:
            # Local destinations are written in place: journal before writing.
            self._journal.record(f"{self._project_id}/{shard_path.name}", {slide_id})

        mode = "a" if shard_path.exists() else "w"
        with tarfile.open(shard_path, mode) as tar:
            _add_npy_to_tar(tar, f"{slide_id}.features.npy", feat_bytes)
            if coord_bytes:
                _add_npy_to_tar(tar, f"{slide_id}.coords.npy", coord_bytes)

        self._current_path = shard_path
        self._current_bytes = shard_path.stat().st_size
        if self._local_slide_locations is not None:
            self._local_slide_locations.setdefault(slide_id, set()).add(shard_path)
        return shard_path.name

    def _upload_current(self) -> None:
        """Journal the current staging shard's slides, then upload it to S3."""
        name = self._current_path.name
        self._verify_lock()
        if self._journal is not None:
            self._journal.record(f"{self._project_id}/{name}", _tar_slide_ids(self._current_path))
        s3_uri = f"{self._wds_dest.rstrip('/')}/{self._model}/{self._project_id}/{name}"
        _s3_upload(self._current_path, s3_uri, self._s3_max_concurrency)

    def _seal_and_rotate(self) -> None:
        """Upload the full shard to S3 (if needed) and start a new index slot."""
        if self._current_path and self._use_s3 and not self._dry_run:
            self._upload_current()
            # Kept until the index is saved (a failed save leaves the journal to
            # clean up), then deleted by discard_sealed_staging: S3 holds it.
            self._sealed_staging.append(self._current_path)
        self._current_index += 1
        if self._use_s3 and not self._dry_run:
            # Another staging dir may already own the next number on S3.
            self._current_index = max(self._current_index, self._next_s3_shard_index())
        self._current_path = None
        self._current_bytes = 0

    def discard_sealed_staging(self) -> int:
        """Delete the staging copies of shards this writer sealed and uploaded.

        Call only after the index is saved. Sealed shards are never appended to
        again (the writer reuses only its last, open shard), and keeping them made
        the shared staging dir grow without bound (sealed copies add up to many TiB).
        """
        n = 0
        for path in self._sealed_staging:
            if path != self._current_path and path.exists():
                path.unlink()
                n += 1
        self._sealed_staging = []
        self._local_slide_locations = None
        return n

    def flush(self) -> None:
        """Upload the current (possibly unsealed) staging shard to S3 at end of run."""
        if self._current_path and self._use_s3 and not self._dry_run:
            self._upload_current()


@dataclass(frozen=True)
class _SlidePayload:
    features: np.ndarray
    coords: np.ndarray | None
    native_mpp: float | None
    mpp_is_fallback: bool | None
    content_sha1: str


def _is_slide_encoder(model_type: str) -> bool:
    """Slide encoders (mussel-nf slide_to_patch_mapping: titan_slide, gigapath_slide, ...)
    write one vector per slide."""
    return model_type.endswith("_slide")


def _load_slide_payload(
    pt_path: Path, h5_path: Path | None, alt_h5_paths: list[Path] = (), slide_level: bool = False
) -> "_SlidePayload | None":
    """Load what append() writes for one slide, plus a fingerprint of it.

    A slide encoder's vector (``slide_level``) is written without coords: it has no
    tile rows. Its tile file still supplies the slide's MPP.

    Patch-level features (2-D) take their coords from the model's own
    <slide>.features.h5 next to the .pt (written with the features, so the
    rows match by construction, whichever tile set the model used). Without
    one, coords come from h5_path or, if its row count differs, from the
    other tile sets (alt_h5_paths); if none match, the slide is written
    without coords rather than with another tiling's coords.
    """
    try:
        features = _load_features(pt_path)
    except Exception as exc:
        log.error("Failed to load %s: %s", pt_path, exc)
        return None
    coords = native_mpp = mpp_is_fallback = None
    if h5_path is not None:
        coords, native_mpp, mpp_is_fallback = _load_slide_meta(h5_path)
    own = None if slide_level else _own_coords(pt_path, features)
    if slide_level:
        coords = None
    elif own is not None:
        coords = own[0]
        native_mpp = own[1] if own[1] is not None else native_mpp
        mpp_is_fallback = own[2] if own[2] is not None else mpp_is_fallback
    elif features.ndim > 1 and coords is not None and len(coords) != len(features):
        for alt in alt_h5_paths:
            alt_coords, alt_mpp, alt_fallback = _load_slide_meta(alt)
            if alt_coords is not None and len(alt_coords) == len(features):
                coords, native_mpp, mpp_is_fallback = alt_coords, alt_mpp, alt_fallback
                break
        else:
            log.warning(
                "%s: %d feature rows but no tile set with matching coords (%s has %d); "
                "writing features without coords", pt_path, len(features), h5_path, len(coords))
            coords = None
    digest = hashlib.sha1(_npy_bytes(features))
    if coords is not None:
        digest.update(_npy_bytes(coords))
    return _SlidePayload(features, coords, native_mpp, mpp_is_fallback, digest.hexdigest())


def _strip_slides_from_shard(
    wds_dest: str,
    model: str,
    shard_file: str,
    slide_ids: set[str],
    staging_dir: Path | None,
    s3_max_concurrency: int = 4,
) -> None:
    """Remove every member of slide_ids from one existing shard (local or S3)."""
    if not _is_s3(wds_dest):
        _rewrite_tar_excluding_slides(Path(wds_dest) / model / shard_file, slide_ids)
        return
    uri = f"{wds_dest.rstrip('/')}/{model}/{shard_file}"
    work = staging_dir / model / "_strip" / shard_file
    work.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not _s3_download(uri, work, s3_max_concurrency):
            log.warning("Superseded shard %s not found; nothing to strip", uri)
            return
        if not _rewrite_tar_excluding_slides(work, slide_ids):
            return
        with tarfile.open(work, "r") as tar:
            remaining = sum(1 for m in tar.getmembers() if m.isfile())
        if remaining:
            _s3_upload(work, uri, s3_max_concurrency)
        else:
            bucket, key = _s3_parts(uri)
            _s3_client().delete_object(Bucket=bucket, Key=key)
        log.info("Stripped %d superseded slide(s) from %s (%d members left)",
                 len(slide_ids), uri, remaining)
    finally:
        work.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Append journal
# ---------------------------------------------------------------------------

_JOURNAL_DIR = "_journal"


def _tar_slide_ids(path: Path) -> set[str]:
    with tarfile.open(path, "r") as tar:
        return {sid for m in tar.getmembers() if (sid := _slide_id_from_member(m.name)) is not None}


def _put_json(uri: str, obj: dict, staging_dir: Path | None) -> None:
    data = json.dumps(obj)
    if _is_s3(uri):
        tmp = Path(staging_dir) / f".{uri.rsplit('/', 1)[-1]}.upload"
        tmp.write_text(data)
        try:
            _s3_upload(tmp, uri)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(data)
        tmp.replace(path)


def _read_json(uri: str, staging_dir: Path | None) -> dict | None:
    if not _is_s3(uri):
        return json.loads(Path(uri).read_text()) if Path(uri).exists() else None
    tmp = Path(staging_dir) / f".{uri.rsplit('/', 1)[-1]}.download"
    try:
        return json.loads(tmp.read_text()) if _s3_download(uri, tmp) else None
    finally:
        tmp.unlink(missing_ok=True)


def _delete_object(uri: str) -> None:
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        _s3_client().delete_object(Bucket=bucket, Key=key)
    else:
        Path(uri).unlink(missing_ok=True)


def _list_journals(wds_dest: str, model: str) -> list[str]:
    root = f"{wds_dest.rstrip('/')}/{model}/{_JOURNAL_DIR}"
    if not _is_s3(root):
        return sorted(str(p) for p in Path(root).glob("*.json"))
    bucket, prefix = _s3_parts(root + "/")
    keys = []
    for page in _s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".json")]
    return [f"s3://{bucket}/{k}" for k in sorted(keys)]


class _AppendJournal:
    """Shards this append uploads before it saves the index.

    append_wds uploads each shard as it fills and saves wds_index.json only at
    the end. If a run dies in between, the shards it uploaded hold slides the
    index never recorded, and a retry writes them again into a new shard (a
    fresh staging dir or --force-new-shard does not see the first copies), so
    the first copies stay behind for plain WebDataset readers. Each run records
    every shard and the slides in it here (<dest>/<model>/_journal/<run>.json)
    before uploading it, and, before saving the index, the old shards of the
    re-processed slides it replaces (stripped only after the save), then marks
    the record once the index is saved. It deletes the record once both are
    done; the next run holding the append lock strips whatever a dead run left
    (_reconcile_journals).
    """

    def __init__(self, wds_dest: str, model: str, staging_dir: Path | None):
        run = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.uri = f"{wds_dest.rstrip('/')}/{model}/{_JOURNAL_DIR}/{run}.json"
        self._staging_dir = staging_dir
        self.shards: dict[str, set[str]] = {}
        self.superseded: dict[str, set[str]] = {}
        self.index_saved = False

    def _write(self) -> None:
        _put_json(self.uri, {"shards": {k: sorted(v) for k, v in self.shards.items()},
                             "superseded": {k: sorted(v) for k, v in self.superseded.items()},
                             "index_saved": self.index_saved},
                  self._staging_dir)

    def record(self, shard_file: str, slide_ids: set[str]) -> None:
        """Note that ``shard_file`` (``<project>/<NNNNNN>.tar``) is about to hold ``slide_ids``."""
        if not slide_ids or slide_ids <= self.shards.get(shard_file, set()):
            return
        self.shards.setdefault(shard_file, set()).update(slide_ids)
        self._write()

    def record_superseded(self, old_shards: dict[str, set[str]]) -> None:
        """Note the old shards whose copies of re-processed slides are about to be replaced."""
        if not old_shards:
            return
        for shard_file, slide_ids in old_shards.items():
            self.superseded.setdefault(shard_file, set()).update(slide_ids)
        self._write()

    def mark_index_saved(self) -> None:
        """Note that the index now records every slide this run uploaded."""
        if self.shards or self.superseded:
            self.index_saved = True
            self._write()

    def close(self) -> None:
        if self.shards or self.superseded:
            _delete_object(self.uri)


def _reconcile_journals(
    wds_dest: str,
    model: str,
    index: dict,
    staging_dir: Path | None,
    s3_max_concurrency: int = 4,
    verify_lock: "Callable[[], None] | None" = None,
) -> int:
    """Strip slides that interrupted appends left in their shards; return how many.

    Every journal present was left by a run that died before closing it (the
    caller holds the append lock, and only locked runs journal, so no other run
    is mid-append). A journaled slide (uploaded, or superseded in an old shard)
    is kept where the index places it and stripped from the journaled shard
    otherwise. Stripping deletes data, so it never trusts one index read: ECS
    can serve an index older than the last save, which would make a dead run's
    indexed copies look orphaned. The index is read again here, and a slide
    either read places in the journaled shard is kept. A run that died after
    saving the index marked its journal; its uploaded copies are indexed and
    only its superseded copies are stripped.
    """
    verify_lock = verify_lock or (lambda: None)
    journals = _list_journals(wds_dest, model)
    if not journals:
        return 0
    fresh = _load_index(wds_dest, model, staging_dir, s3_max_concurrency)
    n_stripped = 0
    for uri in journals:
        entry = _read_json(uri, staging_dir) or {}
        sections = ("superseded",) if entry.get("index_saved") else ("shards", "superseded")
        by_shard: dict[str, set[str]] = {}
        for section in sections:
            for shard_file, slide_ids in entry.get(section, {}).items():
                by_shard.setdefault(shard_file, set()).update(slide_ids)
        for shard_file, slide_ids in sorted(by_shard.items()):
            stale = {s for s in slide_ids
                     if shard_file not in (index.get(s, {}).get("shard_file"),
                                           fresh.get(s, {}).get("shard_file"))}
            if stale:
                log.warning("Journal %s: stripping %d slide(s) an interrupted append left in %s",
                            uri, len(stale), shard_file)
                verify_lock()
                _strip_slides_from_shard(wds_dest, model, shard_file, stale, staging_dir,
                                         s3_max_concurrency)
                n_stripped += len(stale)
        verify_lock()
        _delete_object(uri)
    return n_stripped


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
    delete_coords_local: bool = False,
    manifest_csv: Path | None = None,
    s3_max_concurrency: int = 4,
    lock_dir: Path | None = None,
    lock_backend: str = "file",
    postgres_db_cfg: ResolvedDatabaseConfig | None = None,
    slide_to_project: "dict[str, str] | None" = None,
    failed_slides: "set[str] | None" = None,
    also_delete_pt_dirs: "list[Path] | None" = None,
    coords_shared_pt_dirs: "list[Path] | None" = None,
    force_new_shard: bool = False,
    skip_indexed_repairs: bool = False,
    replace_reprocessed: bool = True,
    _lock_held: bool = False,
    _verify_lock: "Callable[[], None] | None" = None,
) -> dict:
    """Append all .features.pt files in pt_dir to WDS shards.

    If slide_id_filter is provided, only those slide_ids are appended.
    If delete_local is True, the source .pt and companion .features.h5 files
    are deleted after all writers have been flushed (i.e. after S3 upload
    completes). If delete_coords_local is also True, shared .patch.h5 coordinate
    files are deleted too. Keep this off until every model sharing the same
    coordinates has been written to WDS, or pass coords_shared_pt_dirs.
    coords_shared_pt_dirs: with delete_coords_local, the .pt dirs of the other
      models sharing these coords. A requested slide's coords are then deleted
      once no .features.pt for it is left in pt_dir or any of these dirs, i.e.
      once every model's append has consumed it, whichever hook runs last.
    If manifest_csv is set, appends rows (slide_id, model, wds_path) to that
    CSV so callers can look up the full S3 shard path for each slide.
    s3_max_concurrency limits boto3 multipart threads per upload/download.
    slide_to_project: pre-built slide_id → routing key (e.g. oncotree_code) dict.
      When provided, inventory_df is not used for routing. Useful when the
      routing key is already present in the batch CSV (e.g. Databricks watcher).
    also_delete_pt_dirs: when delete_local is True, also delete any files
      matching {slide_id}.* from these additional directories (e.g. patch encoder
      .pt files that are no longer needed after the slide encoder is in WDS).
    replace_reprocessed: when a slide in slide_id_filter is already indexed but
      its local output differs from the indexed content (content_sha1), append
      the new output and strip the old copy from its shard. Without this a
      requeued slide keeps its stale WDS entry forever.

    Slide encoders (``model_type`` ending in ``_slide``) are written without
    coords; their tile files (--h5-dir) only supply the slide's MPP.
    Returns the updated index dict.
    """
    if lock_backend == "postgres":
        if postgres_db_cfg is None:
            raise ValueError("Postgres WDS lock backend requires PostgreSQL config")
        with _wds_postgres_lock(postgres_db_cfg, wds_dest, model_type) as verify_lock:
            return append_wds(
                pt_dir=pt_dir,
                h5_dir=h5_dir,
                inventory_df=inventory_df,
                wds_dest=wds_dest,
                model_type=model_type,
                staging_dir=staging_dir,
                max_shard_bytes=max_shard_bytes,
                dry_run=dry_run,
                slide_id_filter=slide_id_filter,
                delete_local=delete_local,
                delete_coords_local=delete_coords_local,
                manifest_csv=manifest_csv,
                s3_max_concurrency=s3_max_concurrency,
                lock_dir=None,
                lock_backend="file",
                postgres_db_cfg=None,
                slide_to_project=slide_to_project,
                failed_slides=failed_slides,
                also_delete_pt_dirs=also_delete_pt_dirs,
                coords_shared_pt_dirs=coords_shared_pt_dirs,
                force_new_shard=force_new_shard,
                skip_indexed_repairs=skip_indexed_repairs,
                replace_reprocessed=replace_reprocessed,
                _lock_held=True,
                _verify_lock=verify_lock,
            )
    if lock_dir is not None:
        with _wds_append_lock(lock_dir, wds_dest, model_type):
            return append_wds(
                pt_dir=pt_dir,
                h5_dir=h5_dir,
                inventory_df=inventory_df,
                wds_dest=wds_dest,
                model_type=model_type,
                staging_dir=staging_dir,
                max_shard_bytes=max_shard_bytes,
                dry_run=dry_run,
                slide_id_filter=slide_id_filter,
                delete_local=delete_local,
                delete_coords_local=delete_coords_local,
                manifest_csv=manifest_csv,
                s3_max_concurrency=s3_max_concurrency,
                lock_dir=None,
                lock_backend="file",
                postgres_db_cfg=None,
                slide_to_project=slide_to_project,
                failed_slides=failed_slides,
                also_delete_pt_dirs=also_delete_pt_dirs,
                coords_shared_pt_dirs=coords_shared_pt_dirs,
                force_new_shard=force_new_shard,
                skip_indexed_repairs=skip_indexed_repairs,
                replace_reprocessed=replace_reprocessed,
                _lock_held=True,
            )

    # Build slide_id → project_id lookup: prefer explicit dict, fall back to inventory_df
    if slide_to_project is None:
        inv = inventory_df.copy()
        inv["slide_id"] = inv["file_name"].apply(lambda fn: fn.split(".")[0])
        slide_to_project = dict(zip(inv["slide_id"], inv["project_id"]))

    # Postgres lock: abort before each mutating phase if the lock was lost.
    verify_lock = _verify_lock or (lambda: None)

    index = _load_index(wds_dest, model_type, staging_dir, s3_max_concurrency)
    already_indexed = set(index.keys())
    log.info("Loaded index: %d slides already in WDS", len(already_indexed))

    # Remove stale index entries only for slides in the CURRENT batch that are
    # known permanent failures.  Scoping to slide_id_filter prevents cascade
    # eviction: if slides from a *different* batch are temporarily FAILED (e.g.
    # GPU node outage) while this batch's WDS hook runs, their previously-good
    # index entries must not be deleted — the features are still safely in S3.
    pruned: set[str] = set()
    if failed_slides:
        scope = failed_slides & slide_id_filter if slide_id_filter is not None else failed_slides
        stale = already_indexed & scope
        pruned = stale
        if stale:
            log.warning("Removing %d failed slides from wds_index (current-batch scope): %s",
                        len(stale), ", ".join(sorted(stale)[:10]))
            for sid in stale:
                del index[sid]
            already_indexed -= stale
            if not dry_run:
                verify_lock()
                _save_index(index, wds_dest, model_type, staging_dir, s3_max_concurrency,
                            removed=stale)
                log.info("Pruned wds_index.json (%d entries remaining)", len(index))

    # Clean up after appends that died between uploading shards and saving the
    # index. Only under the append lock: without it a journal may belong to a
    # run that is still appending.
    # Only locked runs journal: a locked run treats every journal it finds as
    # left by a dead run, so an unlocked run's journal could be stripped mid-append.
    journal = None
    if not dry_run:
        if _lock_held:
            verify_lock()
            n_cleaned = _reconcile_journals(wds_dest, model_type, index, staging_dir,
                                            s3_max_concurrency, verify_lock=verify_lock)
            if n_cleaned:
                log.warning("Removed %d slide copies left by interrupted appends", n_cleaned)
            journal = _AppendJournal(wds_dest, model_type, staging_dir)
        else:
            log.info("No WDS append lock held: not journaling, and leaving any "
                     "interrupted-append journals for a locked run")

    pt_files = sorted(pt_dir.rglob("*.features.pt"))
    log.info("Found %d .pt files in %s", len(pt_files), pt_dir)

    # Pre-build a slide_id → h5_path lookup to avoid O(N²) rglob calls in the loop.
    h5_lookup = _build_h5_lookup(h5_dir)
    alt_h5_lookups = _build_alt_h5_lookups(h5_dir)
    slide_level = _is_slide_encoder(model_type)

    writers: dict[str, _ShardWriter] = {}
    replaced_from: dict[str, set[str]] = {}  # old shard_file -> re-processed slides moved out of it
    n_replaced = 0
    n_appended = n_skipped = n_missing_project = 0
    n_without_mpp = 0
    n_indexed_repaired = n_indexed_missing_h5 = n_indexed_missing_shard = 0
    n_indexed_mismatched_coords = 0
    appended_locals: list[tuple[Path, Path | None]] = []  # (pt_path, h5_path) for delete_local
    appended_ids: set[str] = set()

    for pt_path in pt_files:
        slide_id = _slide_id_from_pt(pt_path)

        if slide_id_filter is not None and slide_id not in slide_id_filter:
            continue

        if failed_slides and slide_id in failed_slides:
            log.debug("Skipping failed slide: %s", slide_id)
            continue

        payload: _SlidePayload | None = None
        if slide_id in already_indexed and replace_reprocessed and slide_id_filter is not None:
            # This batch re-processed a slide that is already in WDS (e.g. a
            # requeue after a tessellation fix).  Its new output must replace
            # the indexed copy unless the content is identical, which is the
            # case for a plain hook retry of the batch that appended it.
            payload = _load_slide_payload(pt_path, h5_lookup.get(slide_id), _alt_paths(alt_h5_lookups, slide_id),
                                          slide_level=slide_level)
            if payload is not None and index.get(slide_id, {}).get("content_sha1") == payload.content_sha1:
                payload = None

        if slide_id in already_indexed and payload is None:
            n_skipped += 1
            if skip_indexed_repairs:
                continue
            if slide_level:
                # A slide vector has no coords to repair.
                if delete_local:
                    appended_locals.append((pt_path, h5_lookup.get(slide_id)))
                continue
            repair_stats = _repair_indexed_slide_coords_if_possible(
                slide_id=slide_id,
                index_entry=index.get(slide_id, {}),
                wds_dest=wds_dest,
                model_type=model_type,
                staging_dir=staging_dir,
                h5_lookup=h5_lookup,
                dry_run=dry_run,
                s3_max_concurrency=s3_max_concurrency,
                verify_lock=verify_lock,
            )
            n_indexed_repaired += repair_stats["repaired"]
            n_indexed_missing_h5 += repair_stats["missing_h5"]
            n_indexed_missing_shard += repair_stats["missing_shard"]
            n_indexed_mismatched_coords += repair_stats["mismatched_coords"]
            # Even though already in WDS, clean up the local file if requested.
            if delete_local:
                appended_locals.append((pt_path, h5_lookup.get(slide_id)))
            continue

        project_id = slide_to_project.get(slide_id)
        if not project_id:
            log.warning("No project_id for %s — skipping", slide_id)
            n_missing_project += 1
            continue

        h5_path_for_slide: Path | None = h5_lookup.get(slide_id)
        if payload is None:
            payload = _load_slide_payload(pt_path, h5_path_for_slide, _alt_paths(alt_h5_lookups, slide_id),
                                          slide_level=slide_level)
            if payload is None:
                continue
        elif index.get(slide_id, {}).get("shard_file"):
            replaced_from.setdefault(index[slide_id]["shard_file"], set()).add(slide_id)
            n_replaced += 1

        if project_id not in writers:
            writers[project_id] = _ShardWriter(
                wds_dest=wds_dest,
                model=model_type,
                project_id=project_id,
                staging_dir=staging_dir,
                max_shard_bytes=max_shard_bytes,
                dry_run=dry_run,
                s3_max_concurrency=s3_max_concurrency,
                force_new_shard=force_new_shard,
                journal=journal,
                verify_lock=verify_lock,
            )

        native_mpp, mpp_is_fallback = payload.native_mpp, payload.mpp_is_fallback
        if native_mpp is None:
            # MPP comes from the slide header, not the features, so replacing a
            # slide without its .patch.h5 (e.g. a slide encoder's features only)
            # keeps the indexed copy's MPP instead of erasing it.
            previous = index.get(slide_id, {})
            native_mpp = previous.get("native_mpp")
            mpp_is_fallback = previous.get("mpp_is_fallback")
        if native_mpp is None:
            n_without_mpp += 1

        shard_name = writers[project_id].append(slide_id, payload.features, payload.coords,
                                                indexed_shard=index.get(slide_id, {}).get("shard_file"))
        index[slide_id] = {
            "project_id": project_id,
            "shard_file": f"{project_id}/{shard_name}",
            "native_mpp": native_mpp,
            "mpp_is_fallback": mpp_is_fallback,
            "content_sha1": payload.content_sha1,
        }
        n_appended += 1
        appended_ids.add(slide_id)
        if delete_local:
            appended_locals.append((pt_path, h5_path_for_slide))
        if n_appended % 100 == 0:
            log.info("  %d appended, %d skipped", n_appended, n_skipped)

    # Flush unsealed staging shards to S3 — must complete before deleting local files
    for writer in writers.values():
        writer.flush()

    if n_missing_project:
        log.warning("%d slides skipped (no project_id in inventory)", n_missing_project)
    if n_indexed_repaired:
        log.info("Repaired coords in place for %d already-indexed slide(s)", n_indexed_repaired)
    if n_indexed_missing_h5:
        log.info("%d already-indexed slide(s) had no readable local .patch.h5 for repair",
                 n_indexed_missing_h5)
    if n_indexed_missing_shard:
        log.warning("%d already-indexed slide(s) could not locate their indexed shard for repair",
                    n_indexed_missing_shard)
    if n_indexed_mismatched_coords:
        log.warning("%d already-indexed slide(s) had coord/feature row mismatches during repair",
                    n_indexed_mismatched_coords)
    if n_without_mpp:
        log.warning("%d appended slide(s) have no native_mpp: no .patch.h5 with MPP attrs and "
                    "no earlier index entry to keep it from", n_without_mpp)
    log.info("Done: %d appended, %d already indexed, %d missing project",
             n_appended, n_skipped, n_missing_project)

    if n_appended > 0 and not dry_run:
        if journal is not None:
            # Stripped below, after the save; if the run dies in between, the
            # next locked run strips them (_reconcile_journals).
            journal.record_superseded(replaced_from)
        verify_lock()
        # removed=pruned: a stale remote index must not bring back slides pruned above.
        _save_index(index, wds_dest, model_type, staging_dir, s3_max_concurrency,
                    removed=pruned)
        log.info("Updated wds_index.json (%d total entries)", len(index))
        if journal is not None:
            journal.mark_index_saved()

    # Only after the index points at the new copies: remove the superseded
    # copies, which plain WebDataset readers would otherwise still iterate.
    if replaced_from and not dry_run:
        verify_lock()
        for old_shard, sids in sorted(replaced_from.items()):
            moved = {sid for sid in sids if index.get(sid, {}).get("shard_file") != old_shard}
            if moved:
                _strip_slides_from_shard(wds_dest, model_type, old_shard, moved,
                                         staging_dir, s3_max_concurrency)
        log.info("Replaced %d re-processed slide(s) already in WDS", n_replaced)

    # The index now records every slide this run uploaded, and its superseded
    # copies are gone.
    if journal is not None:
        journal.close()
    if not dry_run and _is_s3(wds_dest):
        n_discarded = sum(w.discard_sealed_staging() for w in writers.values())
        if n_discarded:
            log.info("Deleted %d sealed staging shard(s); S3 holds them", n_discarded)

    # Delete local source files now that data is durably in WDS / S3 and the
    # index records it: if the index save failed, the next run strips this
    # run's copies as orphans and the sources are needed to append them again.
    # Includes both newly-appended slides and already-indexed slides whose
    # local files were still present (cleanup of files from prior runs).
    # Also deletes the companion .features.h5 (H5 duplicate of the .pt data)
    # that the pipeline publishes alongside every .features.pt file. The
    # .patch.h5 coordinate file is shared by all model hooks, so it is deleted
    # only when explicitly requested by the final hook.
    if delete_local and not dry_run and appended_locals:
        n_deleted = 0
        n_coords_deleted = 0
        for pt_path, h5_path in appended_locals:
            pt_path.unlink(missing_ok=True)
            # Delete the .features.h5 companion published next to the .pt file
            # pt_path ends in .features.pt; .with_suffix(".h5") gives .features.h5
            companion_h5 = pt_path.with_suffix(".h5")
            companion_h5.unlink(missing_ok=True)
            if delete_coords_local and coords_shared_pt_dirs is None and h5_path:
                h5_path.unlink(missing_ok=True)
                n_coords_deleted += 1
            n_deleted += 1
        log.info(
            "Deleted %d local feature source file pair(s) (pt + features.h5); "
            "deleted %d shared coordinate file(s)",
            n_deleted,
            n_coords_deleted,
        )

    if delete_local and delete_coords_local and coords_shared_pt_dirs is not None and not dry_run:
        candidates = (set(slide_id_filter) if slide_id_filter is not None
                      else {_slide_id_from_pt(pt) for pt, _ in appended_locals})
        n_coords = _delete_consumed_coords(h5_lookup, candidates, [pt_dir, *coords_shared_pt_dirs])
        log.info("Deleted %d shared coordinate file(s) no model still needs", n_coords)

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
    # Include both newly-appended slides AND batch slides that were already indexed in WDS.
    # The latter ensures the manifest stays consistent even when a batch's slides were all
    # already in the WDS index (n_appended == 0), which happens after a manifest reset/rename.
    # Only this run's slides: _save_index merges other writers' entries into
    # ``index``, and those are not this batch's to record.
    if manifest_csv is not None and not dry_run:
        slides_to_manifest: dict[str, dict] = {}
        for slide_id, info in index.items():
            if slide_id in appended_ids:
                slides_to_manifest[slide_id] = info  # newly appended
            elif (slide_id in already_indexed and slide_id_filter is not None
                  and slide_id in slide_id_filter):
                slides_to_manifest[slide_id] = info  # batch slide already in WDS — confirm in manifest
        if slides_to_manifest:
            manifest_csv = Path(manifest_csv)
            manifest_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not manifest_csv.exists()
            with manifest_csv.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["slide_id", "model", "wds_path"])
                if write_header:
                    writer.writeheader()
                for slide_id, info in slides_to_manifest.items():
                    shard_file = info["shard_file"]
                    wds_path = f"{wds_dest.rstrip('/')}/{model_type}/{shard_file}"
                    writer.writerow({"slide_id": slide_id, "model": model_type,
                                     "wds_path": wds_path})
            n_already_written = len(slides_to_manifest) - n_appended
            log.info("Wrote %d WDS path entries to %s (%d new, %d already-indexed)",
                     len(slides_to_manifest), manifest_csv, n_appended, n_already_written)

    index["_stats"] = {
        "n_appended": n_appended,
        "n_replaced": n_replaced,
        "n_skipped": n_skipped,
        "n_missing_project": n_missing_project,
        "n_indexed_repaired": n_indexed_repaired,
        "n_indexed_missing_h5": n_indexed_missing_h5,
        "n_indexed_missing_shard": n_indexed_missing_shard,
        "n_indexed_mismatched_coords": n_indexed_mismatched_coords,
    }
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
                        help="Delete local .pt and companion .features.h5 source files "
                             "after they are successfully flushed to WDS (including S3 upload). "
                             "Has no effect with --dry-run.")
    parser.add_argument("--delete-coords-local", action="store_true",
                        help="Also delete shared .patch.h5 coordinate files after WDS flush. "
                             "Use only on the final WDS hook for a batch because coords are "
                             "shared by all model outputs. Requires --delete-local.")
    parser.add_argument("--repair-existing-shards", action="store_true",
                        help="Before appending, compact existing local/staged WDS shards for "
                             "this model. Duplicate slide entries are collapsed to one "
                             "features/coords pair; missing coords are restored from --h5-dir "
                             "when available. For S3 destinations, repaired staged shards are "
                             "uploaded back to the same S3 keys.")
    parser.add_argument("--repair-require-coords", action="store_true",
                        help="When repairing shards, count slides still missing coords after "
                             "tar/H5 recovery. Intended for tile embedding models.")
    parser.add_argument("--coords-shared-pt-dirs", default=None,
                        help="Comma-separated .pt dirs of the other models sharing --h5-dir. "
                             "With --delete-coords-local, delete a requested slide's coords "
                             "once none of these dirs (nor --pt-dir) still holds its .pt.")
    parser.add_argument("--also-delete-pt-dirs", default=None,
                        help="Comma-separated list of additional directories from which to "
                             "delete all files matching {slide_id}.* when --delete-local is "
                             "active. Use to clean up patch encoder .pt files (e.g. conch1_5) "
                             "after their slide encoder (e.g. titan_slide) is written to WDS.")
    parser.add_argument("--s3-max-concurrency", type=int, default=4,
                        help="Maximum number of parallel boto3 transfer threads per S3 "
                             "upload/download (default: 4). Reduce to limit ECS endpoint load "
                             "when multiple batches are running concurrently.")
    parser.add_argument("--force-new-shard", action="store_true",
                        help="For S3 recovery, allocate fresh shards instead of downloading "
                             "the prior project shard; existing ECS shards stay untouched.")
    parser.add_argument("--skip-indexed-repairs", action="store_true",
                        help="Do not download/repair shards for slides already present in the index; intended for WDS backlog recovery.")
    parser.add_argument("--no-replace-reprocessed", dest="replace_reprocessed", action="store_false",
                        help="Keep the existing WDS copy of re-processed slides that are already indexed "
                             "(legacy behaviour). By default their new output replaces the indexed copy.")
    parser.add_argument("--require-all-slide-ids", action="store_true",
                        help="Fail if any requested slide_id is still absent from the resulting WDS index.")
    parser.add_argument("--lock-dir", default=None,
                        help="Shared directory for WDS append locks. When set, append operations "
                             "for the same wds-dest/model are serialized across dispatchers.")
    parser.add_argument("--lock-backend", choices=["file", "postgres"], default="file",
                        help="Lock authority for WDS append serialization.")
    parser.add_argument("--lock-postgres-host", default=None)
    parser.add_argument("--lock-postgres-port", type=int, default=5432)
    parser.add_argument("--lock-postgres-database", default=None)
    parser.add_argument("--lock-postgres-user", default=None)
    parser.add_argument("--lock-postgres-password-env", default=None)
    parser.add_argument("--lock-postgres-sslmode", default="prefer")
    parser.add_argument("--s3-endpoint", default=None,
                        help="Custom S3 endpoint URL (e.g. http://your-s3-endpoint:9020 for "
                             "ECS). Falls back to S3_ENDPOINT_URL / ECS_ENDPOINT_URL env vars.")
    parser.add_argument("--aws-profile", default=None,
                        help="AWS shared credentials profile to use for S3 operations.")
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
        aws_profile=args.aws_profile,
    )

    import pandas as pd

    # Validate: need either --inventory or --project-id-column (with --slide-ids-csv)
    if not args.project_id_column and not args.inventory:
        log.error("Either --inventory or --project-id-column (with --slide-ids-csv) must be set")
        return 1
    if args.project_id_column and not args.slide_ids_csv:
        log.error("--project-id-column requires --slide-ids-csv")
        return 1
    if args.delete_coords_local and not args.delete_local:
        log.error("--delete-coords-local requires --delete-local")
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
    lock_dir = Path(args.lock_dir) if args.lock_dir else None
    postgres_lock_cfg = None
    if args.lock_backend == "postgres":
        postgres_lock_cfg = ResolvedDatabaseConfig(
            backend="postgres",
            host=args.lock_postgres_host or "",
            port=int(args.lock_postgres_port or 5432),
            database=args.lock_postgres_database or "",
            user=args.lock_postgres_user or "",
            password_env=args.lock_postgres_password_env or "",
            sslmode=args.lock_postgres_sslmode or "prefer",
        )
    try:
        if args.lock_backend == "postgres":
            if not _is_s3(args.wds_dest):
                pass
            elif not all([
                postgres_lock_cfg.host,
                postgres_lock_cfg.database,
                postgres_lock_cfg.user,
                postgres_lock_cfg.password_env,
            ]):
                raise ValueError(
                    "S3/ECS WDS mutation with --lock-backend=postgres requires complete PostgreSQL lock config"
                )
        else:
            _validate_shared_lock_args(wds_dest=args.wds_dest, lock_dir=lock_dir)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

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
            if args.repair_existing_shards:
                # Repair rewrites and re-uploads shards: hold the append lock.
                with _wds_mutation_lock(
                    lock_backend=args.lock_backend,
                    lock_dir=lock_dir,
                    postgres_db_cfg=postgres_lock_cfg,
                    wds_dest=args.wds_dest,
                    model_type=model,
                ) as verify_lock:
                    repair_stats = repair_wds_shards(
                        wds_dest=args.wds_dest,
                        model=model,
                        staging_dir=staging_dir,
                        h5_dir=h5_dir if h5_dir and h5_dir.exists() else None,
                        require_coords=args.repair_require_coords,
                        dry_run=args.dry_run,
                        s3_max_concurrency=args.s3_max_concurrency,
                        verify_lock=verify_lock,
                    )
                log.info("WDS repair stats for %s: %s", model, repair_stats)
            result_index = append_wds(
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
                delete_coords_local=args.delete_coords_local,
                manifest_csv=Path(args.manifest_csv) if args.manifest_csv else None,
                s3_max_concurrency=args.s3_max_concurrency,
                lock_dir=lock_dir,
                lock_backend=args.lock_backend,
                postgres_db_cfg=postgres_lock_cfg,
                slide_to_project=slide_to_project,
                failed_slides=failed_slides,
                also_delete_pt_dirs=[Path(d) for d in args.also_delete_pt_dirs.split(",")]
                    if args.also_delete_pt_dirs else None,
                coords_shared_pt_dirs=[Path(d) for d in args.coords_shared_pt_dirs.split(",") if d]
                    if args.coords_shared_pt_dirs is not None else None,
                force_new_shard=args.force_new_shard,
                skip_indexed_repairs=args.skip_indexed_repairs,
                replace_reprocessed=args.replace_reprocessed,
            )
            if args.require_all_slide_ids and slide_id_filter:
                indexed_ids = {
                    str(slide_id) for slide_id in result_index
                    if not str(slide_id).startswith("_")
                }
                missing_ids = sorted(set(slide_id_filter) - indexed_ids)
                if missing_ids:
                    log.error(
                        "WDS validation failed for %s: %d requested slide(s) have no index entry: %s",
                        model, len(missing_ids), ", ".join(missing_ids[:20]),
                    )
                    rc = 2
        except Exception as exc:
            log.error("Failed to append WDS for model %s: %s", model, exc)
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
