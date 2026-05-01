#!/usr/bin/env python3
"""tcga_poll_mpp.py — backfill native_mpp for all TCGA slides.

Strategy
--------
1. Load ``wds_index.json`` from each model on ECS S3; collect slide_id →
   native_mpp for entries that already have it (newly uploaded slides).
2. For slides missing native_mpp: read the SVS TIFF header from ECS S3
   (two targeted range requests — header → IFD offset, IFD window → MPP).
3. For slides whose SVS is NOT present on ECS S3 (usually pending/failed
   slides): download the full SVS from GDC, upload it to ECS S3 under
   ``s3://pathology/TCGA/<file_id>/<file_name>``, then extract MPP.
4. Patch each model's ``wds_index.json`` on S3 with the resolved MPP values.
5. Write ``tcga_slide_mpp.csv`` (slide_id, native_mpp, source).

Usage
-----
  python scripts/tcga/tcga_poll_mpp.py \\
      --inventory  dispatcher/tcga_inventory.csv \\
      --wds-dest   s3://reef-tcga-v2-0/wds \\
      --models     hoptimus1,titan_slide \\
      --slide-base s3://pathology/TCGA \\
      --s3-endpoint http://pmindecs.mskcc.org:9020 \\
      --output     dispatcher/tcga_slide_mpp.csv \\
      [--workers 16] [--dry-run] [--no-upload] [--gdc-token /path/token]

The script is resumable: any slide_id already present in ``--output`` with a
non-empty native_mpp is skipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import boto3
import requests

log = logging.getLogger("tcga_poll_mpp")

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_S3_CLIENT: Optional[object] = None


def _make_s3(endpoint: str, access_key: str, secret_key: str):
    from botocore.config import Config
    cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3})
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        config=cfg,
    )


def _s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _s3_range(s3, bucket: str, key: str, start: int, end: int) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def _s3_download(s3, bucket: str, key: str, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        return True
    except Exception as exc:
        log.warning("S3 download failed %s/%s: %s", bucket, key, exc)
        return False


def _s3_upload(s3, local: Path, bucket: str, key: str) -> bool:
    try:
        s3.upload_file(str(local), bucket, key)
        return True
    except Exception as exc:
        log.warning("S3 upload failed %s/%s: %s", bucket, key, exc)
        return False


def _s3_json_get(s3, bucket: str, key: str) -> dict:
    try:
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(data)
    except Exception as exc:
        log.debug("Could not load s3://%s/%s: %s", bucket, key, exc)
        return {}


def _s3_json_put(s3, obj: dict, bucket: str, key: str, dry_run: bool) -> None:
    if dry_run:
        log.info("[dry-run] would write s3://%s/%s (%d entries)", bucket, key, len(obj))
        return
    body = json.dumps(obj, indent=2, sort_keys=True).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    log.info("Wrote s3://%s/%s (%d entries)", bucket, key, len(obj))


def _split_s3(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` → (bucket, key)."""
    uri = uri.lstrip("s3://")
    bucket, _, key = uri.partition("/")
    return bucket, key


# ---------------------------------------------------------------------------
# TIFF header parser — 2 range requests
# ---------------------------------------------------------------------------

def _mpp_from_svs_s3(s3, bucket: str, key: str) -> Optional[float]:
    """Extract native MPP from an Aperio SVS on S3 using two range requests.

    Request 1: bytes 0–7  → byte order + IFD offset.
    Request 2: IFD window → parse tag 270 (ImageDescription) and compute MPP.

    Returns None if parsing fails or MPP cannot be determined.
    """
    try:
        hdr = _s3_range(s3, bucket, key, 0, 7)
        if len(hdr) < 8:
            return None

        endian = "<" if hdr[:2] == b"II" else ">"
        bigtiff = struct.unpack_from(endian + "H", hdr, 2)[0] == 43
        ifd_off = struct.unpack_from(
            endian + ("Q" if bigtiff else "I"), hdr, 8 if bigtiff else 4
        )[0]

        # 8 KB window starting at IFD covers entries + inline description.
        win = _s3_range(s3, bucket, key, ifd_off, ifd_off + 8191)
        return _mpp_from_tiff_window(win, ifd_off, endian, bigtiff)

    except Exception as exc:
        log.debug("TIFF parse error for s3://%s/%s: %s", bucket, key, exc)
        return None


def _mpp_from_local_svs(path: Path) -> Optional[float]:
    """Extract MPP from a local SVS file."""
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(8)
            if len(hdr) < 8:
                return None
            endian = "<" if hdr[:2] == b"II" else ">"
            bigtiff = struct.unpack_from(endian + "H", hdr, 2)[0] == 43
            ifd_off = struct.unpack_from(
                endian + ("Q" if bigtiff else "I"), hdr, 8 if bigtiff else 4
            )[0]
            fh.seek(ifd_off)
            win = fh.read(8192)
        return _mpp_from_tiff_window(win, ifd_off, endian, bigtiff)
    except Exception as exc:
        log.debug("TIFF parse error for %s: %s", path, exc)
        return None


def _mpp_from_tiff_window(win: bytes, ifd_off: int, endian: str, bigtiff: bool) -> Optional[float]:
    """Parse MPP out of a bytes window that starts at the IFD offset."""
    entry_size = 20 if bigtiff else 12
    fmt_entry = endian + ("HHQQ" if bigtiff else "HHII")

    try:
        n = struct.unpack_from(endian + "H", win, 0)[0]
    except struct.error:
        return None

    desc_off = desc_n = 0
    xres_num = xres_den = 0
    res_unit = 2  # INCH by default

    for i in range(min(n, 64)):  # cap to avoid corrupt data
        base = 2 + i * entry_size
        if base + entry_size > len(win):
            break
        try:
            tag, dtype, count, val = struct.unpack_from(fmt_entry, win, base)
        except struct.error:
            break

        if tag == 270:  # ImageDescription (ASCII)
            desc_n = count
            inline_limit = 8 if bigtiff else 4
            desc_off = val if count > inline_limit else 0
        elif tag == 282:  # XResolution (RATIONAL = two LONGs)
            if count == 1 and 0 <= val - ifd_off + 8 <= len(win):
                r = win[val - ifd_off: val - ifd_off + 8]
                if len(r) == 8:
                    xres_num, xres_den = struct.unpack_from(endian + "II", r)
        elif tag == 296:  # ResolutionUnit
            res_unit = val & 0xFFFF

    # Try ImageDescription → Aperio "MPP = X.XXXX"
    if desc_off and 0 <= desc_off - ifd_off < len(win):
        start = desc_off - ifd_off
        desc = win[start: start + desc_n]
        m = re.search(rb"MPP\s*=\s*([0-9.]+)", desc)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Fallback: XResolution + ResolutionUnit
    if xres_num and xres_den:
        scale = {2: 25400.0, 3: 10000.0}.get(res_unit, 0.0)
        if scale:
            return round(scale / (xres_num / xres_den), 4)

    return None


# ---------------------------------------------------------------------------
# GDC download
# ---------------------------------------------------------------------------

_GDC_DATA = "https://api.gdc.cancer.gov/data"
_GDC_TIMEOUT = 60
_CHUNK = 8 * 1024 * 1024  # 8 MB


def _gdc_mpp_range(file_id: str, token: str = "") -> Optional[float]:
    """Get native MPP from a GDC file using two HTTP range requests.

    Reads only the TIFF header bytes — no full download needed.
    """
    url = f"{_GDC_DATA}/{file_id}"
    headers = {"X-Auth-Token": token} if token else {}
    try:
        # Request 1: 8-byte TIFF header → IFD offset
        r = requests.get(url, headers={**headers, "Range": "bytes=0-7"},
                         timeout=_GDC_TIMEOUT, stream=False)
        r.raise_for_status()
        hdr = r.content
        if len(hdr) < 8:
            return None
        endian = "<" if hdr[:2] == b"II" else ">"
        bigtiff = struct.unpack_from(endian + "H", hdr, 2)[0] == 43
        ifd_off = struct.unpack_from(
            endian + ("Q" if bigtiff else "I"), hdr, 8 if bigtiff else 4
        )[0]
        # Request 2: 8 KB window at IFD offset → IFD entries + description
        r2 = requests.get(url,
                          headers={**headers, "Range": f"bytes={ifd_off}-{ifd_off+8191}"},
                          timeout=_GDC_TIMEOUT, stream=False)
        r2.raise_for_status()
        return _mpp_from_tiff_window(r2.content, ifd_off, endian, bigtiff)
    except Exception as exc:
        log.debug("GDC range MPP failed %s: %s", file_id, exc)
        return None


def _gdc_download(file_id: str, dest: Path, token: str = "") -> bool:
    """Download a full GDC file to *dest*. Returns True on success."""
    url = f"{_GDC_DATA}/{file_id}"
    headers = {"X-Auth-Token": token} if token else {}
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, headers=headers, stream=True,
                          timeout=_GDC_TIMEOUT) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(_CHUNK):
                    fh.write(chunk)
        return True
    except Exception as exc:
        log.warning("GDC download failed %s: %s", file_id, exc)
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# WDS index helpers
# ---------------------------------------------------------------------------

def _load_wds_index(s3, wds_dest: str, model: str) -> dict:
    """Download wds_index.json for *model* from *wds_dest*."""
    bucket, prefix = _split_s3(wds_dest)
    key = f"{prefix.rstrip('/')}/{model}/wds_index.json"
    return _s3_json_get(s3, bucket, key)


def _save_wds_index(s3, index: dict, wds_dest: str, model: str, dry_run: bool) -> None:
    bucket, prefix = _split_s3(wds_dest)
    key = f"{prefix.rstrip('/')}/{model}/wds_index.json"
    _s3_json_put(s3, index, bucket, key, dry_run)


# ---------------------------------------------------------------------------
# Per-slide worker
# ---------------------------------------------------------------------------

def _process_slide(
    *,
    slide_id: str,
    file_id: str,
    file_name: str,
    s3,
    slide_bucket: str,
    slide_prefix: str,
    gdc_token: str,
    full_download: bool,
    no_upload: bool,
    dry_run: bool,
    tmp_dir: Path,
) -> tuple[str, Optional[float], str]:
    """Resolve native_mpp for one slide.  Returns (slide_id, mpp, source).

    Strategy:
    1. SVS on ECS S3 → 2 range requests → MPP.
    2. SVS not on ECS S3:
       a. Default: 2 GDC range requests for MPP only (fast, no storage).
       b. ``full_download=True``: download full SVS from GDC, upload to ECS S3,
          extract MPP from local file (useful for pending slides that need to
          be available for future Nextflow runs).
    """
    key = f"{slide_prefix.rstrip('/')}/{file_id}/{file_name}"

    if _s3_exists(s3, slide_bucket, key):
        mpp = _mpp_from_svs_s3(s3, slide_bucket, key)
        if mpp is not None:
            return slide_id, mpp, "ecs_s3"
        log.warning("%s: MPP not found in SVS header on ECS S3", slide_id)
        return slide_id, None, "ecs_s3_no_mpp"

    # SVS not on ECS S3.
    if not full_download:
        # Fast path: two range requests to GDC, no local storage.
        log.debug("%s: not on ECS S3, polling MPP from GDC (%s)", slide_id, file_id)
        mpp = _gdc_mpp_range(file_id, gdc_token)
        source = "gdc_range" if mpp is not None else "gdc_range_no_mpp"
        return slide_id, mpp, source

    # Full download + upload path.
    log.info("%s: not on ECS S3, downloading full SVS from GDC (%s)", slide_id, file_id)
    local = tmp_dir / file_id / file_name
    if not _gdc_download(file_id, local, gdc_token):
        return slide_id, None, "gdc_download_failed"

    mpp = _mpp_from_local_svs(local)

    if not no_upload and not dry_run:
        ok = _s3_upload(s3, local, slide_bucket, key)
        source = "gdc_uploaded" if ok else "gdc_not_uploaded"
    elif dry_run:
        source = "gdc_dry_run"
    else:
        source = "gdc_no_upload"

    try:
        shutil.rmtree(local.parent, ignore_errors=True)
    except Exception:
        pass

    return slide_id, mpp, source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--inventory", required=True,
                    help="tcga_inventory.csv (file_id, file_name, slide_type, …)")
    ap.add_argument("--wds-dest", required=True,
                    help="WDS S3 base, e.g. s3://reef-tcga-v2-0/wds")
    ap.add_argument("--models", default="hoptimus1",
                    help="Comma-separated model names whose wds_index.json to patch")
    ap.add_argument("--slide-base", default="s3://pathology/TCGA",
                    help="ECS S3 base for raw SVS slides")
    ap.add_argument("--s3-endpoint", default="http://pmindecs.mskcc.org:9020")
    ap.add_argument("--s3-access-key", default="",
                    help="ECS access key (or set AWS_ACCESS_KEY_ID)")
    ap.add_argument("--s3-secret-key", default="",
                    help="ECS secret key (or set AWS_SECRET_ACCESS_KEY)")
    ap.add_argument("--secrets-env", default="",
                    help="Shell env file with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
    ap.add_argument("--output", default="dispatcher/tcga_slide_mpp.csv",
                    help="Output CSV: slide_id, native_mpp, source")
    ap.add_argument("--gdc-token", default="",
                    help="Path to GDC user token file (for controlled-access data)")
    ap.add_argument("--full-download", action="store_true",
                    help="For slides not on ECS S3: download full SVS from GDC and "
                         "upload to ECS S3 (makes them available for future NF runs). "
                         "Default: two GDC range requests for MPP only (fast).")
    ap.add_argument("--no-upload", action="store_true",
                    help="With --full-download: skip uploading to ECS S3 after download")
    ap.add_argument("--slide-type", default="DX",
                    help="Slide type prefix to filter (default: DX)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel worker threads for S3 range requests")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse MPP but do not write wds_index.json back to S3")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # Suppress noisy boto3/botocore DEBUG output regardless of verbosity.
    for noisy in ("boto3", "botocore", "s3transfer", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Load secrets env file if provided
    if args.secrets_env and Path(args.secrets_env).exists():
        with open(args.secrets_env) as fh:
            for line in fh:
                line = line.strip().lstrip("export").strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    access_key = args.s3_access_key or os.environ.get("ECS_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = args.s3_secret_key or os.environ.get("ECS_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    s3 = _make_s3(args.s3_endpoint, access_key, secret_key)

    gdc_token = ""
    if args.gdc_token and Path(args.gdc_token).exists():
        gdc_token = Path(args.gdc_token).read_text().strip()

    # Load inventory
    import pandas as pd
    inv = pd.read_csv(args.inventory, dtype=str).fillna("")
    if args.slide_type:
        inv = inv[inv["slide_type"].str.startswith(args.slide_type, na=False)]
    log.info("Inventory: %d slides (type=%s*)", len(inv), args.slide_type)

    # Build file_id / file_name lookup from inventory
    slide_info: dict[str, dict] = {}  # slide_id → {file_id, file_name}
    for _, row in inv.iterrows():
        file_name = row["file_name"]
        slide_id = _slide_id_from_filename(file_name)
        slide_info[slide_id] = {"file_id": row["file_id"], "file_name": file_name}

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Load wds_index.json for all models; collect already-known MPPs
    log.info("Loading wds_index.json for models: %s", ", ".join(models))
    wds_indexes: dict[str, dict] = {}
    known_mpp: dict[str, float] = {}  # slide_id → mpp (from index)
    for model in models:
        idx = _load_wds_index(s3, args.wds_dest, model)
        wds_indexes[model] = idx
        for sid, entry in idx.items():
            mpp = entry.get("native_mpp")
            if mpp is not None:
                known_mpp[sid] = float(mpp)
    log.info("Already have native_mpp for %d slides across all models", len(known_mpp))

    # All slides that appear in any wds_index
    indexed_slides = set()
    for idx in wds_indexes.values():
        indexed_slides.update(idx.keys())

    # Load existing output CSV for resumability
    existing: dict[str, tuple[Optional[float], str]] = {}  # slide_id → (mpp, source)
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path, newline="") as fh:
            for row in csv.DictReader(fh):
                mpp_str = row.get("native_mpp", "")
                try:
                    mpp_val: Optional[float] = float(mpp_str) if mpp_str else None
                except ValueError:
                    mpp_val = None
                existing[row["slide_id"]] = (mpp_val, row.get("source", ""))
        log.info("Resuming: %d slides already in %s", len(existing), out_path)

    # Determine which slides need resolution
    todo: list[dict] = []
    for slide_id, info in slide_info.items():
        # Skip if already resolved with a valid MPP
        if slide_id in existing and existing[slide_id][0] is not None:
            continue
        # Skip if already in wds_index with MPP
        if slide_id in known_mpp:
            continue
        todo.append({"slide_id": slide_id, **info})

    log.info("%d slides need MPP resolution (%d already known, %d in prior CSV)",
             len(todo), len(known_mpp), sum(1 for m, _ in existing.values() if m is not None))

    slide_bucket, slide_prefix = _split_s3(args.slide_base)
    results: dict[str, tuple[Optional[float], str]] = dict(existing)

    # Copy already-known MPPs from wds_index into results
    for sid, mpp in known_mpp.items():
        if sid not in results or results[sid][0] is None:
            results[sid] = (mpp, "wds_index")

    if todo:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tcga_poll_mpp_"))
        log.info("Temp dir for GDC downloads: %s", tmp_dir)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        _process_slide,
                        slide_id=s["slide_id"],
                        file_id=s["file_id"],
                        file_name=s["file_name"],
                        s3=s3,
                        slide_bucket=slide_bucket,
                        slide_prefix=slide_prefix,
                        gdc_token=gdc_token,
                        full_download=args.full_download,
                        no_upload=args.no_upload,
                        dry_run=args.dry_run,
                        tmp_dir=tmp_dir,
                    ): s["slide_id"]
                    for s in todo
                }
                done = 0
                total = len(futures)
                for fut in as_completed(futures):
                    done += 1
                    try:
                        slide_id, mpp, source = fut.result()
                        results[slide_id] = (mpp, source)
                        if done % 100 == 0 or done == total:
                            resolved = sum(1 for m, _ in results.values() if m is not None)
                            log.info("Progress: %d/%d — %d resolved", done, total, resolved)
                    except Exception as exc:
                        sid = futures[fut]
                        log.error("Worker failed for %s: %s", sid, exc)
                        results[sid] = (None, "error")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Patch wds_index.json entries with resolved native_mpp
    mpp_lookup = {sid: mpp for sid, (mpp, _) in results.items() if mpp is not None}
    for model in models:
        idx = wds_indexes[model]
        patched = 0
        for sid, entry in idx.items():
            if entry.get("native_mpp") is None and sid in mpp_lookup:
                entry["native_mpp"] = mpp_lookup[sid]
                patched += 1
        log.info("Model %-20s: patching %d / %d index entries", model, patched, len(idx))
        if patched:
            _save_wds_index(s3, idx, args.wds_dest, model, args.dry_run)

    # Write output CSV
    all_slide_ids = sorted(set(slide_info.keys()) | set(results.keys()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_count = sum(1 for sid in all_slide_ids
                         if results.get(sid, (None,))[0] is not None)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "native_mpp", "source"])
        for sid in all_slide_ids:
            mpp, source = results.get(sid, (None, ""))
            w.writerow([sid, "" if mpp is None else f"{mpp:.6g}", source])

    log.info("Wrote %s (%d slides, %d resolved / %d total)",
             out_path, len(all_slide_ids), resolved_count, len(all_slide_ids))

    null_count = len(all_slide_ids) - resolved_count
    if null_count:
        log.warning("%d slides have null native_mpp (SVS missing MPP metadata or download failed)",
                    null_count)
    return 0


def _slide_id_from_filename(file_name: str) -> str:
    """Extract TCGA-XX-XXXX-XX slide ID from SVS filename.

    E.g. ``TCGA-27-1835-01Z-00-DX1.abc.svs`` → ``TCGA-27-1835-01Z-00-DX1``
    """
    stem = Path(file_name).stem
    # Aperio filenames: <slide_id>.<uuid>[.<ext>]
    parts = stem.split(".")
    return parts[0]


if __name__ == "__main__":
    sys.exit(main())
