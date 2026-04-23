#!/usr/bin/env python3
"""Prepare a nextflow samples CSV from the TCGA inventory with path resolution.

For each pending slide, resolves slide_path using this priority chain:
  1. Local disk  — <local-slides-dir>/<file_id>/<file_name>  (exists on disk)
  2. S3          — <s3-base>/<file_id>/<file_name>           (constructed or verified)
  3. Needs download — flagged as needs_download=true in the sidecar meta CSV

Writes two files:
  <output>           — slide_id,slide_path  (nextflow-compatible)
  <output>.meta.csv  — slide_id,slide_path,needs_download,file_id,project_id
                       (used by the orchestrator to build gdc-client manifests)

Exit codes
----------
    0  -- success, output CSV written
    1  -- error
    2  -- no pending slides (useful for orchestrator loop termination)

Usage
-----
    python tcga_prepare_samples.py \\
        --inventory tcga_inventory.csv \\
        --status tcga_status.csv \\
        --s3-base s3://pathology/TCGA \\
        --local-slides-dir /data/tcga-slides \\
        --output samples_to_run.csv \\
        [--model ctranspath] \\
        [--slide-type DX1] \\
        [--project TCGA-BRCA] \\
        [--limit 500]
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

META_COLUMNS = ["slide_id", "slide_path", "needs_download", "file_id", "file_name", "project_id"]


def _slide_id_from_filename(file_name: str) -> str:
    """Extract the full TCGA slide barcode from a filename or bare barcode.

    TCGA filenames embed a full slide barcode before the first dot:
        TCGA-BR-A44T-01Z-00-DX1.<uuid>.svs  →  TCGA-BR-A44T-01Z-00-DX1

    Using the full barcode (rather than the 4-part sample barcode) is essential
    because a single patient-sample can have multiple slides (DX1, DX2, …) that
    would otherwise collapse to the same identifier.

    Also handles bare barcodes passed directly (no dot → returned unchanged).
    """
    return file_name.split(".")[0]


def _load_nextflow_config() -> dict[str, str]:
    """Return a flat dict of nextflow config key=value pairs.

    Runs `nextflow config -flat` from the repo root (parent of scripts/tcga/).
    Returns empty dict if nextflow is not available or config cannot be parsed.
    """
    repo_root = Path(__file__).parent.parent.parent
    try:
        result = subprocess.run(
            ["nextflow", "config", "-flat"],
            capture_output=True, text=True, check=False,
            cwd=str(repo_root),
        )
        cfg: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip("'\"")
        return cfg
    except FileNotFoundError:
        return {}


def _resolve_s3_credentials(
    access_key: str | None = None,
    secret_key: str | None = None,
    endpoint_url: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve S3 credentials and endpoint URL.

    Resolution order for credentials (first non-empty wins):
      1. Explicit arguments
      2. ECS_ACCESS_KEY / ECS_SECRET_KEY environment variables
      3. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables
      4. nextflow secrets store (ECS_ACCESS_KEY / ECS_SECRET_KEY)

    Resolution order for endpoint (first non-empty wins):
      1. Explicit endpoint_url argument
      2. ECS_ENDPOINT_URL environment variable
      3. aws.client.endpoint from `nextflow config -flat`

    Returns (key, secret, endpoint).
    """
    import os

    key = access_key or os.environ.get("ECS_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = secret_key or os.environ.get("ECS_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = endpoint_url or os.environ.get("ECS_ENDPOINT_URL")

    nf_cfg: dict[str, str] = {}

    if not key or not secret:
        try:
            result_k = subprocess.run(
                ["nextflow", "secrets", "get", "ECS_ACCESS_KEY"],
                capture_output=True, text=True, check=False,
            )
            result_s = subprocess.run(
                ["nextflow", "secrets", "get", "ECS_SECRET_KEY"],
                capture_output=True, text=True, check=False,
            )
            nf_key = result_k.stdout.strip()
            nf_secret = result_s.stdout.strip()
            if nf_key and nf_secret:
                log.debug("Using ECS credentials from nextflow secrets store")
                key = key or nf_key
                secret = secret or nf_secret
        except FileNotFoundError:
            pass  # nextflow not on PATH

    if not endpoint:
        nf_cfg = _load_nextflow_config()
        endpoint = nf_cfg.get("aws.client.endpoint")
        if endpoint:
            log.debug("Using S3 endpoint from nextflow config: %s", endpoint)

    return key, secret, endpoint


def _list_s3_file_ids(
    s3_base: str,
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> set[str]:
    """Return the set of file_id prefixes that exist under s3_base."""
    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed — skipping S3 existence check")
        return set()

    key, secret, endpoint = _resolve_s3_credentials(access_key, secret_key, endpoint_url)

    # Fall back to ECS_ENDPOINT_URL env var if no explicit endpoint given
    if not endpoint:
        log.warning("No S3 endpoint URL configured — set ECS_ENDPOINT_URL, --s3-endpoint, "
                    "or aws.client.endpoint in nextflow.config")

    s3_base = s3_base.rstrip("/")
    without_scheme = s3_base[5:]  # strip "s3://"
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""

    client_kwargs: dict = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if key and secret:
        client_kwargs["aws_access_key_id"] = key
        client_kwargs["aws_secret_access_key"] = secret

    s3 = boto3.client("s3", **client_kwargs)

    existing: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes") or []:
                part = cp["Prefix"].rstrip("/").split("/")[-1]
                existing.add(part)
    except Exception as exc:
        log.warning("S3 listing failed (%s) — will flag slides as needs_download", exc)
        return set()

    log.info("S3 listing found %d existing file_id prefixes under %s", len(existing), s3_base)
    return existing


def prepare_samples(
    inventory_df: pd.DataFrame,
    status_df: pd.DataFrame | None,
    *,
    model: str | None = None,
    slide_type_filter: str | None = None,
    sample_type_filter: str | None = None,
    project_filter: str | None = None,
    skip_done: bool = True,
    local_slides_dir: Path | None = None,
    s3_base: str | None = None,
    check_s3_exists: bool = False,
    s3_endpoint: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    df = inventory_df.copy()
    df["slide_id"] = df["file_name"].apply(_slide_id_from_filename)
    df = df[df["slide_id"] != ""].copy()

    if slide_type_filter and slide_type_filter.lower() != "all":
        # Support comma-separated types and prefix matching.
        # e.g. "DX1" → exact; "DX" → prefix (matches DX1, DX2, …); "DX1,DX2" → either
        types = [t.strip() for t in slide_type_filter.split(",") if t.strip()]
        mask = df["slide_type"].apply(
            lambda st: any(st == t or st.startswith(t) for t in types)
        )
        df = df[mask]
        log.info("slide_type filter '%s' matched %d slides", slide_type_filter, mask.sum())

    if sample_type_filter and sample_type_filter.lower() != "all":
        # Comma-separated substrings matched case-insensitively against the
        # GDC sample_type field (e.g. "Primary Tumor", "Metastatic").
        # Use "tumor" to match "Primary Tumor" + "Metastatic" + "Recurrent Tumor".
        parts = [p.strip().lower() for p in sample_type_filter.split(",") if p.strip()]
        mask = df["sample_type"].apply(
            lambda st: any(p in st.lower() for p in parts)
        )
        df = df[mask]
        log.info("sample_type filter '%s' matched %d slides", sample_type_filter, mask.sum())

    if project_filter:
        projects = {p.strip() for p in project_filter.split(",")}
        df = df[df["project_id"].isin(projects)]

    if skip_done and status_df is not None and not status_df.empty:
        done_query = status_df[status_df["status"] == "done"]
        if model:
            done_query = done_query[done_query["model"] == model]
        done_ids = set(done_query["slide_id"])
        df = df[~df["slide_id"].isin(done_ids)]

    log.info("%d slides pending after filters", len(df))

    df = df.sort_values(["project_id", "slide_id"]).reset_index(drop=True)

    if limit is not None and limit > 0:
        df = df.head(limit)
        log.info("Limited to first %d slides", limit)

    # Pre-fetch S3 listing once (fast batch check) instead of per-file head_object
    s3_existing: set[str] | None = None
    if check_s3_exists and s3_base:
        log.info("Fetching S3 file_id listing from %s …", s3_base)
        s3_existing = _list_s3_file_ids(
            s3_base,
            endpoint_url=s3_endpoint,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
        )

    records = []
    for _, row in df.iterrows():
        slide_id = row["slide_id"]
        file_id = row["file_id"]
        file_name = row["file_name"]
        project_id = row.get("project_id", "")
        slide_path: str | None = None
        needs_download = False

        # 1. Check local disk — verify size matches inventory to detect partial downloads
        if local_slides_dir is not None:
            local_path = local_slides_dir / file_id / file_name
            if local_path.exists():
                expected_size = int(row.get("file_size", 0) or 0)
                actual_size = local_path.stat().st_size
                if expected_size == 0 or actual_size == expected_size:
                    slide_path = str(local_path)
                else:
                    log.debug(
                        "Partial download detected for %s (%d / %d bytes) — will re-download",
                        file_name, actual_size, expected_size,
                    )

        # 2. Check S3
        if slide_path is None and s3_base:
            candidate = f"{s3_base.rstrip('/')}/{file_id}/{file_name}"
            if check_s3_exists:
                if s3_existing is not None and file_id in s3_existing:
                    slide_path = candidate
            else:
                slide_path = candidate  # assume present

        # 3. Flag for download
        if slide_path is None:
            needs_download = True
            if local_slides_dir is not None:
                # Will exist here after gdc-client downloads it
                slide_path = str(local_slides_dir / file_id / file_name)
            else:
                slide_path = ""

        records.append({
            "slide_id": slide_id,
            "slide_path": slide_path,
            "needs_download": needs_download,
            "file_id": file_id,
            "file_name": file_name,
            "project_id": project_id,
        })

    return pd.DataFrame(records, columns=META_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--inventory", default="tcga_inventory.csv")
    parser.add_argument("--status", default=None,
                        help="tcga_status.csv path (optional)")
    parser.add_argument("--model", default=None,
                        help="Model type for done-status filtering (e.g. ctranspath)")
    parser.add_argument("--s3-base", default=None,
                        help="S3 base URI, e.g. s3://pathology/TCGA")
    parser.add_argument("--s3-endpoint", default=None,
                        help="S3 endpoint URL for non-AWS stores, e.g. http://pmindecs.mskcc.org:9020")
    parser.add_argument("--s3-access-key", default=None,
                        help="S3 access key (overrides ECS_ACCESS_KEY env var)")
    parser.add_argument("--s3-secret-key", default=None,
                        help="S3 secret key (overrides ECS_SECRET_KEY env var)")
    parser.add_argument("--local-slides-dir", default=None,
                        help="Local directory where slides may already be downloaded")
    parser.add_argument("--check-s3-exists", action="store_true",
                        help="Verify S3 presence via a single paginated listing (recommended)")
    parser.add_argument("--no-check-s3-exists", dest="check_s3_exists", action="store_false",
                        help="Assume all constructed S3 paths exist (faster, may cause staging failures)")
    parser.add_argument("--slide-type", default="all",
                        help="Slide type filter. Prefix matching and comma-separated supported. "
                             "e.g. 'DX1', 'DX' (all diagnostic), 'DX1,TS1', or 'all' (default: all)")
    parser.add_argument("--sample-type", default="Primary Tumor",
                        help="GDC sample_type filter (case-insensitive substring, comma-separated). "
                             "e.g. 'Primary Tumor', 'tumor' (matches Primary Tumor + Metastatic + "
                             "Recurrent Tumor), 'all' to disable. Default: 'Primary Tumor'")
    parser.add_argument("--project", default=None,
                        help="Comma-separated project filter, e.g. TCGA-BRCA,TCGA-GBM")
    parser.add_argument("--skip-done", action="store_true", default=True,
                        help="Skip slides already marked done in status CSV (default: true)")
    parser.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of slides to output (for chunked runs)")
    parser.add_argument("--output", default="samples_to_run.csv")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    inventory_df = pd.read_csv(args.inventory, dtype=str).fillna("")
    status_df: pd.DataFrame | None = None
    if args.status and Path(args.status).exists():
        status_df = pd.read_csv(args.status, dtype=str).fillna("")

    out_df = prepare_samples(
        inventory_df,
        status_df,
        model=args.model,
        slide_type_filter=args.slide_type,
        sample_type_filter=args.sample_type,
        project_filter=args.project,
        skip_done=args.skip_done,
        local_slides_dir=Path(args.local_slides_dir) if args.local_slides_dir else None,
        s3_base=args.s3_base,
        check_s3_exists=args.check_s3_exists,
        s3_endpoint=args.s3_endpoint,
        s3_access_key=args.s3_access_key,
        s3_secret_key=args.s3_secret_key,
        limit=args.limit,
    )

    if len(out_df) == 0:
        log.info("No pending slides — nothing to do")
        return 2

    n_download = int(out_df["needs_download"].sum())
    n_ready = len(out_df) - n_download
    log.info("%d slides: %d on disk/S3, %d need download", len(out_df), n_ready, n_download)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Main CSV — includes extra columns that nf-schema picks up as meta fields:
    #   file_id, file_name  → used by DOWNLOAD_SLIDE to fetch the slide
    #   needs_download      → routes slide through DOWNLOAD_SLIDE in main.nf
    # Columns unknown to the schema are silently ignored, so this is backward
    # compatible with runs that don't use DOWNLOAD_SLIDE.
    nf_df = out_df[["slide_id", "slide_path", "file_id", "file_name", "needs_download"]].copy()
    nf_df["needs_download"] = nf_df["needs_download"].apply(lambda x: str(x).lower())
    nf_df.to_csv(output_path, index=False)

    # Full metadata sidecar for orchestrator
    meta_path = output_path.with_suffix(".meta.csv")
    out_df.to_csv(meta_path, index=False)

    log.info("Wrote %s (%d slides)", output_path, len(out_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
