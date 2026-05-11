"""Helper functions for the Mussel dispatcher dashboard.

Separated from server.py to keep the main HTTP server module concise.
Covers: SLURM stats (squeue + sacct), S3 shard counting.

Nextflow log/trace parsing and task failure classification are imported
from nextflow_turret where they are maintained as generic NF utilities.
"""
from __future__ import annotations

import os
import re
import subprocess
import time

from nextflow_turret import (
    parse_elapsed_s as _parse_elapsed_s,
    classify_task_failure,
)

# Backward-compat aliases used internally and by some tests
_parse_elapsed_hms = _parse_elapsed_s

# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

_squeue_cache: dict = {}
_SQUEUE_TTL = 15  # seconds

_sacct_cache: dict = {}
_SACCT_TTL = 60   # seconds — sacct is slower




def sacct_stats() -> dict:
    """Return summary of completed mussel SLURM jobs from the last 24 h via sacct."""
    now = time.time()
    if _sacct_cache.get("ts", 0) + _SACCT_TTL > now:
        return _sacct_cache.get("data", {})

    result: dict = {
        "completed": 0, "failed": 0, "cancelled": 0, "timeout": 0,
        "avg_elapsed_s": None, "min_elapsed_s": None, "max_elapsed_s": None,
        "failure_types": {},
        "sacct_error": None,
    }
    try:
        from datetime import datetime, timezone, timedelta
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")
        out = subprocess.check_output(
            ["sacct", f"--user={os.environ.get('USER', '')}", f"--starttime={since}",
             "--format=JobID,JobName,State,ExitCode,Elapsed,WorkDir%120",
             "--noheader", "--parsable2"],
            timeout=20, text=True, stderr=subprocess.DEVNULL,
        )
        elapsed_list = []
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            job_id, job_name, state, exit_code, elapsed, work_dir = parts[:6]
            if "." in job_id or not job_name.startswith("nf-"):
                continue
            state = state.strip()
            if state == "COMPLETED":
                result["completed"] += 1
                s = _parse_elapsed_s(elapsed)
                if s is not None:
                    elapsed_list.append(s)
            elif state.startswith("FAILED"):
                result["failed"] += 1
                cat = classify_task_failure(work_dir, exit_code, state)
                result["failure_types"][cat] = result["failure_types"].get(cat, 0) + 1
            elif state.startswith("CANCEL"):
                result["cancelled"] += 1
                cat = classify_task_failure(work_dir, exit_code, state)
                result["failure_types"][cat] = result["failure_types"].get(cat, 0) + 1
            elif state == "TIMEOUT":
                result["timeout"] += 1
                result["failure_types"]["timeout"] = result["failure_types"].get("timeout", 0) + 1
        if elapsed_list:
            result["avg_elapsed_s"] = int(sum(elapsed_list) / len(elapsed_list))
            result["min_elapsed_s"] = min(elapsed_list)
            result["max_elapsed_s"] = max(elapsed_list)
    except FileNotFoundError:
        result["sacct_error"] = "sacct not found"
    except subprocess.TimeoutExpired:
        result["sacct_error"] = "sacct timed out"
    except Exception as exc:
        result["sacct_error"] = str(exc)[:80]

    _sacct_cache["data"] = result
    _sacct_cache["ts"] = now
    return result


def slurm_stats() -> dict:
    """Return summary of current user's NF/mussel SLURM jobs via squeue + sacct.

    squeue format: job_id | job_name | state | reason | node | workdir | elapsed
    Only rows whose job_name starts with ``nf-`` are counted; unrelated user jobs
    (e.g. other pipelines) are excluded so the counts are mussel-specific.
    """
    now = time.time()
    if _squeue_cache.get("ts", 0) + _SQUEUE_TTL > now:
        return _squeue_cache.get("data", {})

    _BATCH_RE = re.compile(r'batch_(\d{8}T\d{6}_[0-9a-f]+)')
    result: dict = {
        "running": 0, "pending": 0, "total": 0,
        "nodes": {}, "pending_reasons": {},
        "jobs_by_batch": {},
        # Per-process breakdown: {process_short_name: {"running": n, "pending": n}}
        "processes": {},
        # Stalled tasks: running tasks whose elapsed > stall_threshold_s
        "stalled_tasks": [],
        "error": None,
    }
    _STALL_THRESHOLD_S = 7200  # 2 hours — tasks running longer than this are flagged
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader",
             "--format=%i|%j|%T|%R|%N|%Z|%M"],
            timeout=10, text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 7:
                continue
            job_id, job_name, state, reason, node, workdir, elapsed = parts[:7]

            # Only count NF-submitted jobs (job names begin with "nf-")
            if not job_name.startswith("nf-"):
                continue

            # Short process name: strip "nf-" prefix and task index suffix "(N)"
            proc_short = re.sub(r'\s*\(\d+\)$', '', job_name[3:])

            result["total"] += 1
            m = _BATCH_RE.search(workdir)
            batch_id = m.group(1) if m else None
            if batch_id:
                b = result["jobs_by_batch"].setdefault(
                    batch_id, {"running": 0, "pending": 0, "nodes": [], "processes": {}})
                bp = b["processes"].setdefault(
                    proc_short, {"running": 0, "pending": 0, "nodes": [], "elapsed_s": []})
            p = result["processes"].setdefault(
                proc_short, {"running": 0, "pending": 0})

            if state == "RUNNING":
                result["running"] += 1
                p["running"] += 1
                if node and node != "N/A":
                    result["nodes"][node] = result["nodes"].get(node, 0) + 1
                if batch_id:
                    b["running"] += 1
                    if node not in b["nodes"]:
                        b["nodes"].append(node)
                    bp["running"] += 1
                    if node and node != "N/A" and node not in bp["nodes"]:
                        bp["nodes"].append(node)
                elapsed_s = _parse_elapsed_hms(elapsed)
                if elapsed_s:
                    if batch_id:
                        bp["elapsed_s"].append(elapsed_s)
                    if elapsed_s > _STALL_THRESHOLD_S:
                        result["stalled_tasks"].append({
                            "job_id":   job_id,
                            "process":  proc_short,
                            "node":     node,
                            "elapsed_s": elapsed_s,
                            "batch_id": batch_id,
                        })
            elif state == "PENDING":
                result["pending"] += 1
                p["pending"] += 1
                r = reason.strip("()")
                result["pending_reasons"][r] = result["pending_reasons"].get(r, 0) + 1
                if batch_id:
                    b["pending"] += 1
                    bp["pending"] += 1
    except FileNotFoundError:
        result["error"] = "squeue not found"
    except subprocess.TimeoutExpired:
        result["error"] = "squeue timed out"
    except Exception as exc:
        result["error"] = str(exc)[:80]

    result.update(sacct_stats())

    _squeue_cache["data"] = result
    _squeue_cache["ts"] = now
    return result


# ---------------------------------------------------------------------------
# S3 shard stats
# ---------------------------------------------------------------------------

_s3_cache: dict = {}
_S3_CACHE_TTL = 300  # seconds — ECS listing is slow, cache for 5 min


def s3_stats(watcher, wds_prefix: str) -> dict[str, dict]:
    """Return {model: {shards, objects}} from ECS, with TTL cache.

    Each model's S3 listing runs in a separate thread so they proceed in
    parallel rather than sequentially (each listing can take 30-60s on ECS).
    """
    import boto3
    import threading as _threading
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError

    now = time.time()

    client_kwargs: dict = {}
    endpoint = watcher.s3_endpoint or os.environ.get("ECS_ENDPOINT_URL")
    ak = watcher.s3_access_key or os.environ.get("ECS_ACCESS_KEY")
    sk = watcher.s3_secret_key or os.environ.get("ECS_SECRET_KEY")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if ak:
        client_kwargs["aws_access_key_id"] = ak
    if sk:
        client_kwargs["aws_secret_access_key"] = sk
    client_kwargs["config"] = BotoConfig(
        connect_timeout=5, read_timeout=30, retries={"max_attempts": 1}
    )

    results: dict[str, dict] = {}

    def _fetch_model(model: str, dest: str):
        cached = _s3_cache.get(model)
        if cached and (now - cached.get("ts", 0)) < _S3_CACHE_TTL:
            results[model] = {k: v for k, v in cached.items() if k != "ts"}
            return
        if not dest.startswith("s3://"):
            results[model] = {"shards": 0, "objects": 0, "error": "not an s3 path"}
            return
        rest = dest[5:]
        bucket, _, prefix = rest.partition("/")
        try:
            s3 = boto3.client("s3", **client_kwargs)
            paginator = s3.get_paginator("list_objects_v2")
            shards = objects = 0
            model_prefix = f"{prefix}/{model}/"
            for page in paginator.paginate(Bucket=bucket, Prefix=model_prefix):
                for obj in page.get("Contents", []):
                    objects += 1
                    if obj["Key"].endswith(".tar"):
                        shards += 1
            entry = {"shards": shards, "objects": objects, "ts": now}
            _s3_cache[model] = entry
            results[model] = {"shards": shards, "objects": objects}
        except (BotoCoreError, ClientError, Exception) as exc:
            entry = {"shards": 0, "objects": 0, "error": str(exc)[:120], "ts": now}
            _s3_cache[model] = entry
            results[model] = {"shards": 0, "objects": 0, "error": str(exc)[:120]}

    threads = [
        _threading.Thread(target=_fetch_model, args=(m, d), daemon=True)
        for m, d in watcher.wds_destinations.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results
