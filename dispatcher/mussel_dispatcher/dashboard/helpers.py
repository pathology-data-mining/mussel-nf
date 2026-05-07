"""Helper functions for the Mussel dispatcher dashboard.

Separated from dashboard.py to keep the main HTTP server module concise.
Covers: Nextflow log parsing, trace file parsing, SLURM stats (squeue + sacct),
S3 shard counting.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import time

# ---------------------------------------------------------------------------
# Nextflow log parsing
# ---------------------------------------------------------------------------

_NF_PROGRESS_RE = re.compile(r'\[\s*(\d+)%\]\s+(\d+)\s+of\s+(\d+)')
_NF_EXECUTOR_RE = re.compile(r'^executor\s*>\s*\S+\s*\((\d+)\)', re.MULTILINE)
_NF_WARN_RE     = re.compile(r'^WARN[:\s](.+)', re.MULTILINE)
_NF_ERROR_RE    = re.compile(r"^ERROR ~ Error executing process > '([^']+)'", re.MULTILINE)
_NF_KILLED_RE   = re.compile(r'Killing running tasks \((\d+)\)', re.MULTILINE)

_TRACE_DONE_STATUSES = {"COMPLETED", "CACHED"}
_TRACE_FAIL_STATUSES = {"FAILED", "ABORTED"}


def _trace_path_for_log(log_path: str) -> str:
    """Return the expected trace TSV path for a given batch log path."""
    base = log_path[: -len(".log")] if log_path.endswith(".log") else log_path
    return base + ".trace.tsv"


def parse_nf_trace(trace_path: str) -> dict:
    """Parse a Nextflow trace TSV file and return task counts + failure details.

    The trace file (written by ``-with-trace``) is a tab-separated file
    updated in real-time as tasks complete.  It is more reliable than log
    regex for accurate done/failed counts and provides the exact process
    name and exit code of failed tasks.

    Returns a dict with:
        completed  (int)  – tasks with status COMPLETED
        cached     (int)  – tasks with status CACHED (resumed from cache)
        failed     (int)  – tasks with status FAILED or ABORTED
        total      (int)  – all finished tasks seen so far
        failures   (list) – up to 5 dicts {name, exit, hash} for failed tasks
    """
    result: dict = {
        "completed": 0,
        "cached": 0,
        "failed": 0,
        "total": 0,
        "failures": [],
    }
    if not trace_path or not os.path.exists(trace_path):
        return result
    try:
        with open(trace_path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                status = (row.get("status") or "").strip().upper()
                if status in _TRACE_DONE_STATUSES:
                    if status == "CACHED":
                        result["cached"] += 1
                    else:
                        result["completed"] += 1
                    result["total"] += 1
                elif status in _TRACE_FAIL_STATUSES:
                    result["failed"] += 1
                    result["total"] += 1
                    if len(result["failures"]) < 5:
                        result["failures"].append({
                            "name": (row.get("name") or "").strip(),
                            "exit": (row.get("exit") or "").strip(),
                            "hash": (row.get("hash") or "").strip(),
                        })
    except Exception:
        pass
    return result


def parse_nf_log(log_path: str) -> dict:
    """Parse a NF batch stdout log and return a dict with all useful metrics.

    When a companion trace file (``<batch>.trace.tsv``) exists, failure
    counts and details are taken from it instead of log regex — giving the
    exact process name and exit code rather than a best-effort regex match.
    Progress (done/total) still comes from the log's progress line because
    the trace only contains *finished* tasks and cannot report the total
    expected task count while the run is still in progress.
    """
    result = {
        "progress": None,   # {pct, done, total}
        "slurm_jobs": None, # current active SLURM jobs
        "warn_count": 0,
        "last_warn": None,
        "error_count": 0,
        "first_error": None,
        "killed": None,     # N tasks killed (infra kill signal)
        "failures": [],     # [{name, exit, hash}] from trace (up to 5)
    }
    if not log_path:
        return result
    try:
        with open(log_path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")

        # NF progress — take match with largest total (ignores saveParams 1-of-1)
        prog_matches = _NF_PROGRESS_RE.findall(text)
        if prog_matches:
            pct_s, done_s, total_s = max(prog_matches, key=lambda m: int(m[2]))
            result["progress"] = {"pct": int(pct_s), "done": int(done_s), "total": int(total_s)}

        slurm_matches = _NF_EXECUTOR_RE.findall(text)
        if slurm_matches:
            result["slurm_jobs"] = int(slurm_matches[-1])

        warns = _NF_WARN_RE.findall(text)
        result["warn_count"] = len(warns)
        if warns:
            result["last_warn"] = warns[-1].strip()[:120]

        killed = _NF_KILLED_RE.findall(text)
        if killed:
            result["killed"] = int(killed[-1])

        # Prefer trace file for error info — gives exact process name + exit code.
        trace = parse_nf_trace(_trace_path_for_log(log_path))
        if trace["failed"] > 0:
            result["error_count"] = trace["failed"]
            result["failures"] = trace["failures"]
            if trace["failures"]:
                f0 = trace["failures"][0]
                result["first_error"] = f"{f0['name']} (exit {f0['exit']})"
        else:
            # Fall back to log regex when trace is absent or has no failures yet.
            errors = _NF_ERROR_RE.findall(text)
            result["error_count"] = len(errors)
            if errors:
                result["first_error"] = errors[0].strip()[:120]

    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

_squeue_cache: dict = {}
_SQUEUE_TTL = 15  # seconds

_sacct_cache: dict = {}
_SACCT_TTL = 60   # seconds — sacct is slower


def _parse_elapsed_s(elapsed: str) -> int | None:
    """Parse sacct elapsed string (HH:MM:SS or D-HH:MM:SS) to seconds."""
    try:
        if "-" in elapsed:
            days, rest = elapsed.split("-", 1)
            d = int(days)
        else:
            rest, d = elapsed, 0
        parts = rest.split(":")
        if len(parts) == 3:
            return d * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return d * 86400 + int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return None


def classify_task_failure(work_dir: str, exit_code: str, slurm_state: str = "") -> str:
    """Classify a failed NF task by SLURM state, exit code and .command.err content."""
    code = exit_code.split(":")[0] if ":" in exit_code else exit_code
    try:
        code_i = int(code)
    except ValueError:
        code_i = -1

    if slurm_state.startswith("CANCEL"):
        return "sigterm"
    if code_i == 143:
        return "sigterm"

    err_text = ""
    try:
        err_path = os.path.join(work_dir, ".command.err")
        with open(err_path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            err_text = f.read().decode("utf-8", errors="replace").lower()
    except Exception:
        pass

    if "cuda out of memory" in err_text or "cudaoutofmemoryerror" in err_text:
        return "oom_gpu"
    if "out of memory" in err_text or "oom-kill" in err_text or "cannot allocate memory" in err_text:
        return "oom_host"
    if code_i == 137:
        return "oom_host"
    if "no space left" in err_text or "disk quota" in err_text:
        return "disk_full"
    if "s3://" in err_text and ("error" in err_text or "exception" in err_text):
        return "s3_error"
    if "traceback" in err_text or "runtimeerror" in err_text or "valueerror" in err_text:
        return "python_error"
    if code_i == 1:
        return "error_exit1"
    if code_i > 0:
        return f"exit_{code_i}"
    return "unknown"


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


def _parse_elapsed_hms(elapsed: str) -> int | None:
    """Parse squeue elapsed string (D-HH:MM:SS or HH:MM:SS or M:SS) to seconds."""
    try:
        if "-" in elapsed:
            days, rest = elapsed.split("-", 1)
            d = int(days)
        else:
            rest, d = elapsed, 0
        parts = rest.split(":")
        if len(parts) == 3:
            return d * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return d * 86400 + int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return None


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
                    batch_id, {"running": 0, "pending": 0, "nodes": []})
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
                elapsed_s = _parse_elapsed_hms(elapsed)
                if elapsed_s and elapsed_s > _STALL_THRESHOLD_S:
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
