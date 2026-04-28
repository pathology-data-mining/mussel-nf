#!/usr/bin/env python3
"""Mussel-nf Dispatcher Monitoring Dashboard.

Reads configuration from the dispatcher YAML file (same file used to run the
dispatcher) and exposes a browser-based dashboard via stdlib http.server
(no external dependencies required).

Usage:
    python dispatcher/dashboard.py dispatcher/tcga_dispatcher.yaml
    python dispatcher/dashboard.py dispatcher/tcga_dispatcher.yaml --port 8080
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sqlite3
import subprocess
import sys
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import Config / WatcherConfig from the sibling mussel-dispatcher.py
# ---------------------------------------------------------------------------
_DISPATCHER_PY = Path(__file__).parent / "mussel-dispatcher.py"
_spec = importlib.util.spec_from_file_location("mussel_dispatcher", _DISPATCHER_PY)
_disp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_disp)
Config = _disp.Config
WatcherConfig = _disp.WatcherConfig

# ---------------------------------------------------------------------------
# S3 stats cache
# ---------------------------------------------------------------------------
_s3_cache: dict = {}          # model → {shards, objects, ts}
_S3_CACHE_TTL = 300           # seconds — ECS listing is slow, cache for 5 min

# Matches NF progress lines like:
#   [xx/xxxxxx] PROCESS_NAME (tag) [ 16%] 21 of 126
_NF_PROGRESS_RE = re.compile(
    r'\[\s*(\d+)%\]\s+(\d+)\s+of\s+(\d+)'
)


_NF_EXECUTOR_RE = re.compile(r'^executor\s*>\s*\S+\s*\((\d+)\)', re.MULTILINE)
_NF_WARN_RE     = re.compile(r'^WARN[:\s](.+)', re.MULTILINE)
_NF_ERROR_RE    = re.compile(r"^ERROR ~ Error executing process > '([^']+)'", re.MULTILINE)
_NF_KILLED_RE   = re.compile(r'Killing running tasks \((\d+)\)', re.MULTILINE)


def _parse_nf_log(log_path: str) -> dict:
    """Parse a NF batch stdout log and return a dict with all useful metrics."""
    result = {
        "progress": None,   # {pct, done, total}
        "slurm_jobs": None, # current active SLURM jobs
        "warn_count": 0,
        "last_warn": None,
        "error_count": 0,
        "first_error": None,
        "killed": None,     # N tasks killed (infra kill signal)
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

        # Active SLURM jobs — last occurrence
        slurm_matches = _NF_EXECUTOR_RE.findall(text)
        if slurm_matches:
            result["slurm_jobs"] = int(slurm_matches[-1])

        # WARNs
        warns = _NF_WARN_RE.findall(text)
        result["warn_count"] = len(warns)
        if warns:
            result["last_warn"] = warns[-1].strip()[:120]

        # ERRORs
        errors = _NF_ERROR_RE.findall(text)
        result["error_count"] = len(errors)
        if errors:
            result["first_error"] = errors[0].strip()[:120]

        # Infrastructure kills
        killed = _NF_KILLED_RE.findall(text)
        if killed:
            result["killed"] = int(killed[-1])

    except Exception:
        pass
    return result


_squeue_cache: dict = {}
_SQUEUE_TTL = 15  # seconds
_sacct_cache: dict = {}
_SACCT_TTL = 60  # seconds — sacct is slower


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


def _classify_task_failure(work_dir: str, exit_code: str, slurm_state: str = "") -> str:
    """Classify a failed NF task by SLURM state, exit code + .command.err content."""
    code = exit_code.split(":")[0] if ":" in exit_code else exit_code
    try:
        code_i = int(code)
    except ValueError:
        code_i = -1

    # CANCELLED by SLURM/NF = infra stop (dispatcher restart, scancel, preempt)
    if slurm_state.startswith("CANCEL"):
        return "sigterm"
    # SIGTERM (143) = infrastructure kill
    if code_i == 143:
        return "sigterm"

    # Read last 4KB of .command.err for content-based classification
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
        return "oom_host"  # SIGKILL without clear OOM message still likely OOM
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


def _sacct_stats() -> dict:
    """Return summary of completed mussel SLURM jobs from the last 24h via sacct."""
    now = time.time()
    if _sacct_cache.get("ts", 0) + _SACCT_TTL > now:
        return _sacct_cache.get("data", {})

    result: dict = {
        "completed": 0, "failed": 0, "cancelled": 0, "timeout": 0,
        "avg_elapsed_s": None, "min_elapsed_s": None, "max_elapsed_s": None,
        "failure_types": {},   # category -> count
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
            if "." in job_id or "nf-MUSSEL" not in job_name:
                continue
            state = state.strip()
            if state == "COMPLETED":
                result["completed"] += 1
                s = _parse_elapsed_s(elapsed)
                if s is not None:
                    elapsed_list.append(s)
            elif state.startswith("FAILED"):
                result["failed"] += 1
                cat = _classify_task_failure(work_dir, exit_code, state)
                result["failure_types"][cat] = result["failure_types"].get(cat, 0) + 1
            elif state.startswith("CANCEL"):
                result["cancelled"] += 1
                cat = _classify_task_failure(work_dir, exit_code, state)
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


def _slurm_stats() -> dict:
    """Return summary of current user's SLURM jobs via squeue + sacct."""
    now = time.time()
    if _squeue_cache.get("ts", 0) + _SQUEUE_TTL > now:
        return _squeue_cache.get("data", {})

    _BATCH_RE = re.compile(r'batch_(\d{8}T\d{6}_[0-9a-f]+)')
    result: dict = {
        "running": 0, "pending": 0, "total": 0,
        "nodes": {}, "pending_reasons": {},
        "jobs_by_batch": {},  # batch_id -> {running, pending, nodes: []}
        "error": None,
    }
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader", "--format=%T|%R|%N|%Z"],
            timeout=10, text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            state, reason, node, workdir = parts[0], parts[1], parts[2], parts[3]
            result["total"] += 1
            # Extract batch ID from work dir path
            m = _BATCH_RE.search(workdir)
            batch_id = m.group(1) if m else None
            if batch_id:
                b = result["jobs_by_batch"].setdefault(batch_id, {"running": 0, "pending": 0, "nodes": []})
            if state == "RUNNING":
                result["running"] += 1
                if node and node != "N/A":
                    result["nodes"][node] = result["nodes"].get(node, 0) + 1
                if batch_id:
                    b["running"] += 1
                    if node not in b["nodes"]:
                        b["nodes"].append(node)
            elif state == "PENDING":
                result["pending"] += 1
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

    # Merge sacct history (runs on its own TTL, non-blocking relative to squeue)
    result.update(_sacct_stats())

    _squeue_cache["data"] = result
    _squeue_cache["ts"] = now
    return result


def _s3_stats(watcher: WatcherConfig, wds_prefix: str) -> dict[str, dict]:
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
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith(".tar"):
                        shards += 1
                    objects += 1
            entry = {"shards": shards, "objects": objects, "ts": now}
            _s3_cache[model] = entry
            results[model] = {"shards": shards, "objects": objects}
        except (BotoCoreError, ClientError, Exception) as exc:
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


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib — no FastAPI/uvicorn dependency)
# ---------------------------------------------------------------------------

import csv
import json
import re as _re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _build_handler(cfg: Config):
    """Return a BaseHTTPRequestHandler subclass closed over cfg."""

    db_path = os.path.join(cfg.state_dir, "dispatcher.db")
    wds_manifest = os.path.join(cfg.outdir, "wds_manifest.csv")
    nf_log_path = os.path.join(cfg.repo_dir, ".nextflow.log")

    tcga_watcher = None
    for w in cfg.watchers:
        if w.wds_destinations:
            tcga_watcher = w
            break

    # Pre-warm S3 cache in background
    if tcga_watcher and tcga_watcher.wds_destinations:
        threading.Thread(
            target=_s3_stats, args=(tcga_watcher, "wds"), daemon=True, name="s3-prewarm"
        ).start()

    def _db():
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _api_status():
        with _db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM slides GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        total = sum(counts.values())
        succeeded = counts.get("SUCCEEDED", 0)
        pct = round(succeeded / total * 100, 1) if total else 0

        with _db() as conn:
            running_rows = conn.execute(
                "SELECT log_path, slide_count FROM batches WHERE status='RUNNING'"
            ).fetchall()
            n_blacklisted = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE fail_count >= 100"
            ).fetchone()[0]

        n_running = len(running_rows)
        # Improve pct_done: add fractional credit for in-flight NF tasks
        in_flight_done = 0
        in_flight_total = 0
        for rb in running_rows:
            log_info = _parse_nf_log(rb["log_path"]) if rb["log_path"] else {}
            np = log_info.get("progress")
            if np and np["total"] > 0:
                slide_count = rb["slide_count"] or 0
                in_flight_done += slide_count * np["done"] / np["total"]
                in_flight_total += slide_count
        if total:
            effective_done = succeeded + in_flight_done
            pct = round(effective_done / total * 100, 1)

        return {
            "counts": counts,
            "total": total,
            "pct_done": pct,
            "running_batches": n_running,
            "blacklisted": n_blacklisted,
        }

    def _api_batches():
        with _db() as conn:
            rows = conn.execute("""
                SELECT batch_id, status, slide_count, dispatched_at,
                       completed_at, nextflow_exit, log_path
                FROM batches
                ORDER BY dispatched_at DESC
                LIMIT 30
            """).fetchall()
        # Cross-reference with live SLURM jobs (from cache, non-blocking)
        slurm = _slurm_stats()
        jobs_by_batch = slurm.get("jobs_by_batch", {})
        result = []
        for r in rows:
            start = r["dispatched_at"] or ""
            end = r["completed_at"] or ""
            duration = None
            if start and end:
                try:
                    def _parse(s):
                        try:
                            return datetime.fromisoformat(s)
                        except Exception:
                            return None
                    t0, t1 = _parse(start), _parse(end)
                    if t0 and t1:
                        duration = int((t1 - t0).total_seconds())
                except Exception:
                    pass
            log_info = _parse_nf_log(r["log_path"]) if r["log_path"] else {}
            slurm_batch = jobs_by_batch.get(r["batch_id"], {})
            result.append({
                "batch_id": r["batch_id"],
                "status": r["status"],
                "slide_count": r["slide_count"],
                "dispatched_at": start,
                "completed_at": end,
                "duration_s": duration,
                "nextflow_exit": r["nextflow_exit"],
                "has_log": bool(r["log_path"] and os.path.exists(r["log_path"])),
                "nf_progress": log_info.get("progress") if r["status"] == "RUNNING" else None,
                "slurm_jobs": log_info.get("slurm_jobs"),
                "slurm_running": slurm_batch.get("running"),
                "slurm_pending": slurm_batch.get("pending"),
                "slurm_nodes": slurm_batch.get("nodes", []),
                "warn_count": log_info.get("warn_count", 0),
                "last_warn": log_info.get("last_warn"),
                "error_count": log_info.get("error_count", 0),
                "first_error": log_info.get("first_error"),
                "killed": log_info.get("killed"),
            })
        return result

    def _api_logs(batch_id: str):
        with _db() as conn:
            row = conn.execute(
                "SELECT log_path FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if not row or not row["log_path"]:
            return None, "no log path"
        log_path = row["log_path"]
        if not os.path.exists(log_path):
            return None, "log file not found"
        try:
            ansi = _re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfnsuhl]|\r')
            with open(log_path, "r", errors="replace") as fh:
                content = fh.read()
            content = ansi.sub('', content)
            last = content.rfind("executor >")
            block = content[last:].strip() if last >= 0 else content.strip()
            lines = block.splitlines()
        except Exception as exc:
            return None, str(exc)
        nf_lines: list = []
        if os.path.exists(nf_log_path):
            try:
                with open(nf_log_path, "r", errors="replace") as fh:
                    nf_lines = [ln.rstrip() for ln in fh.readlines()[-40:]]
            except Exception:
                pass
        return {"lines": lines, "nf_log": nf_lines}, None

    def _api_wds():
        import glob as _glob

        # SUCCEEDED count from DB (same across all models)
        with _db() as conn:
            db_succeeded = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status='SUCCEEDED'"
            ).fetchone()[0]

        # WDS manifest counts + per-shard distribution
        wds_counts: dict = {}
        shard_slide_counts: dict = {}  # model → {shard_path: slide_count}
        if os.path.exists(wds_manifest):
            try:
                with open(wds_manifest, newline="") as f:
                    for row in csv.DictReader(f):
                        model = row.get("model", "unknown")
                        shard = row.get("wds_path", "")
                        wds_counts[model] = wds_counts.get(model, 0) + 1
                        if model not in shard_slide_counts:
                            shard_slide_counts[model] = {}
                        shard_slide_counts[model][shard] = shard_slide_counts[model].get(shard, 0) + 1
            except Exception as exc:
                return {"models": {}, "total": 0, "db_succeeded": db_succeeded, "error": str(exc)}

        # Local .pt file counts per model (fast glob; shows cleanup status)
        local_pt: dict = {}
        features_dir = os.path.join(cfg.outdir, "features")
        if os.path.isdir(features_dir):
            try:
                for model_dir in os.scandir(features_dir):
                    if model_dir.is_dir():
                        count = sum(1 for _ in _glob.iglob(
                            os.path.join(model_dir.path, "**", "*.features.pt"),
                            recursive=True,
                        ))
                        if count:
                            local_pt[model_dir.name] = count
            except Exception:
                pass

        # S3 shard stats — return from cache only; background thread refreshes
        s3_stats: dict = {}
        if tcga_watcher and tcga_watcher.wds_destinations:
            now = time.time()
            cached = {m: _s3_cache[m] for m in _s3_cache}
            if cached:
                s3_stats = {m: {"shards": v.get("shards", 0), "objects": v.get("objects", 0),
                                "error": v.get("error")} for m, v in cached.items()}
            oldest = min((v.get("ts", 0) for v in _s3_cache.values()), default=0)
            if now - oldest > _S3_CACHE_TTL:
                threading.Thread(
                    target=_s3_stats, args=(tcga_watcher, "wds"), daemon=True, name="s3-refresh"
                ).start()

        models: dict = {}
        # Only include models that have WDS manifest entries or are configured S3 destinations.
        # Local .pt-only entries (e.g. stale ctranspath files) are excluded from the table
        # but their counts are still attached to matching models.
        configured_models = set(tcga_watcher.wds_destinations.keys()) if (
            tcga_watcher and tcga_watcher.wds_destinations
        ) else set()
        all_keys = set(wds_counts) | configured_models
        for m in sorted(all_keys):
            wds_slides = wds_counts.get(m, 0)
            per_shard = list((shard_slide_counts.get(m) or {}).values())
            manifest_shards = len(per_shard)
            shard_stats = {}
            if per_shard:
                shard_stats = {
                    "avg": round(sum(per_shard) / manifest_shards, 1),
                    "min": min(per_shard),
                    "max": max(per_shard),
                    "count": manifest_shards,
                }
            models[m] = {
                "slides": wds_slides,
                "gap": max(0, db_succeeded - wds_slides),
                "shards": s3_stats.get(m, {}).get("shards", 0),
                "manifest_shards": manifest_shards,
                "shard_stats": shard_stats,
                "local_pt": local_pt.get(m, 0),
                "error": s3_stats.get(m, {}).get("error"),
            }
        return {"models": models, "total": sum(wds_counts.values()), "db_succeeded": db_succeeded}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default access log spam
            pass

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    self._send_html(_HTML)
                elif path == "/api/status":
                    self._send_json(_api_status())
                elif path == "/api/batches":
                    self._send_json(_api_batches())
                elif path.startswith("/api/logs/"):
                    batch_id = path[len("/api/logs/"):]
                    data, err = _api_logs(batch_id)
                    if data is None:
                        self._send_json({"lines": [], "error": err}, status=404)
                    else:
                        self._send_json(data)
                elif path == "/api/wds":
                    self._send_json(_api_wds())
                elif path == "/api/slurm":
                    self._send_json(_slurm_stats())
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as exc:
                try:
                    self._send_json({"error": str(exc)}, status=500)
                except Exception:
                    pass

    return Handler


# ---------------------------------------------------------------------------
# Embedded HTML + JS dashboard
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mussel Dispatcher Dashboard</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🦪</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js" async
  onload="onChartJsReady()"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  header { background: #1e293b; border-bottom: 1px solid #334155; padding: 12px 24px;
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 1.2rem; font-weight: 600; color: #f8fafc; }
  #last-updated { font-size: 0.75rem; color: #94a3b8; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
  .full { grid-column: 1 / -1; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;
             letter-spacing: .05em; margin-bottom: 12px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
  .stat { background: #0f172a; border-radius: 6px; padding: 12px 16px; }
  .stat .val { font-size: 1.8rem; font-weight: 700; }
  .stat .lbl { font-size: 0.7rem; color: #94a3b8; margin-top: 2px; text-transform: uppercase; }
  .chart-wrap { display: flex; justify-content: center; align-items: center; max-height: 220px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  th { text-align: left; color: #64748b; font-weight: 600; padding: 6px 8px;
       border-bottom: 1px solid #334155; text-transform: uppercase; font-size: 0.7rem; }
  td { padding: 6px 8px; border-bottom: 1px solid #1e293b; vertical-align: middle; }
  tr:hover td { background: #0f172a; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }
  .badge-running   { background:#0c4a6e; color:#38bdf8; }
  .badge-succeeded { background:#14532d; color:#4ade80; }
  .badge-failed    { background:#450a0a; color:#f87171; }
  .badge-pending   { background:#312e81; color:#a5b4fc; }
  .log-block { background: #0f172a; border-radius: 6px; padding: 10px 12px;
               font-family: monospace; font-size: 0.72rem; line-height: 1.5;
               max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;
               color: #94a3b8; margin-bottom: 8px; }
  .log-header { font-size: 0.75rem; color: #e2e8f0; margin-bottom: 4px; font-weight: 600; }
  .progress-bar-wrap { background: #334155; border-radius: 999px; height: 8px; margin-top: 4px; }
  .progress-bar { background: #22c55e; border-radius: 999px; height: 8px; transition: width .5s; }
  .wds-row { display: flex; justify-content: space-between; padding: 4px 0;
             border-bottom: 1px solid #334155; font-size: 0.8rem; }
  .wds-row:last-child { border-bottom: none; }
  .no-data { color: #475569; font-size: 0.8rem; }
  details summary { cursor: pointer; padding: 4px 0; color: #94a3b8; font-size: 0.8rem; }
  details summary:hover { color: #e2e8f0; }
  details[open] summary { color: #38bdf8; }
  .btn-refresh { background: #334155; border: none; color: #94a3b8; padding: 4px 12px;
                border-radius: 4px; cursor: pointer; font-size: 0.75rem; }
  .btn-refresh:hover { background: #475569; color: #e2e8f0; }
  /* Log modal */
  #log-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.7);
               z-index:1000; align-items:center; justify-content:center; }
  #log-modal.open { display:flex; }
  #log-modal-box { background:#1e293b; border:1px solid #334155; border-radius:10px;
                   width:90vw; max-width:1100px; max-height:85vh; display:flex; flex-direction:column; }
  #log-modal-header { display:flex; justify-content:space-between; align-items:center;
                      padding:12px 16px; border-bottom:1px solid #334155; }
  #log-modal-title { font-size:0.85rem; font-weight:600; color:#f8fafc; font-family:monospace; }
  #log-modal-close { background:none; border:none; color:#94a3b8; font-size:1.4rem;
                     cursor:pointer; line-height:1; padding:0 4px; }
  #log-modal-close:hover { color:#f8fafc; }
  #log-modal-body { flex:1; overflow-y:auto; padding:12px 16px;
                    font-family:monospace; font-size:0.72rem; line-height:1.5;
                    white-space:pre-wrap; word-break:break-all; color:#94a3b8;
                    background:#0f172a; border-radius:0 0 10px 10px; }
</style>
</head>
<body>
<div id="log-modal">
  <div id="log-modal-box">
    <div id="log-modal-header">
      <span id="log-modal-title"></span>
      <button id="log-modal-close" onclick="closeLogModal()">✕</button>
    </div>
    <pre id="log-modal-body">Loading…</pre>
  </div>
</div>
<header>
  <h1>🦪 Mussel Dispatcher Dashboard</h1>
  <div style="display:flex;gap:12px;align-items:center">
    <span id="last-updated">Loading…</span>
    <button class="btn-refresh" onclick="refresh()">↻ Refresh</button>
  </div>
</header>

<div class="grid">
  <!-- Summary -->
  <div class="card full">
    <h2>Overview</h2>
    <div class="summary-grid" id="summary-grid">
      <div class="stat"><div class="val" id="s-total">—</div><div class="lbl">Total Slides</div></div>
      <div class="stat"><div class="val" id="s-done" style="color:#4ade80">—</div><div class="lbl">Succeeded</div></div>
      <div class="stat"><div class="val" id="s-failed" style="color:#f87171">—</div><div class="lbl">Failed</div></div>
      <div class="stat"><div class="val" id="s-pending" style="color:#a5b4fc">—</div><div class="lbl">Pending</div></div>
      <div class="stat"><div class="val" id="s-dispatched" style="color:#38bdf8">—</div><div class="lbl">Dispatched</div></div>
      <div class="stat" title="Slides with pt/h5 features generated (Nextflow succeeded). Does not include WDS upload status."><div class="val" id="s-pct">—</div><div class="lbl">% Features Extracted</div></div>
      <div class="stat" title="Slides confirmed written to WDS shards and uploaded to S3, as a percentage of all SUCCEEDED slides."><div class="val" id="s-wds-pct" style="color:#38bdf8">—</div><div class="lbl">% in WDS</div></div>
      <div class="stat"><div class="val" id="s-running">—</div><div class="lbl">Running Batches</div></div>
      <div class="stat"><div class="val" id="s-blacklisted" style="color:#fb923c">—</div><div class="lbl">Blacklisted</div></div>
    </div>
    <div class="progress-bar-wrap" style="margin-top:12px">
      <div class="progress-bar" id="progress-bar" style="width:0%"></div>
    </div>
  </div>

  <!-- Slide status chart -->
  <div class="card">
    <h2>Slide Status</h2>
    <div class="chart-wrap"><canvas id="statusChart"></canvas></div>
  </div>

  <!-- WDS + S3 -->
  <div class="card">
    <h2>WDS</h2>
    <div id="wds-content"><span class="no-data">Loading…</span></div>
  </div>

  <!-- SLURM -->
  <div class="card">
    <h2>SLURM</h2>
    <div id="slurm-content"><span class="no-data">Loading…</span></div>
  </div>

  <!-- Batch table -->
  <div class="card full">
    <h2>Recent Batches</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Batch ID</th>
            <th>Status</th>
            <th>Tasks</th>
            <th>SLURM</th>
            <th>Started</th>
            <th>Duration</th>
            <th>Alerts</th>
            <th>Log</th>
          </tr>
        </thead>
        <tbody id="batch-tbody"><tr><td colspan="8" class="no-data">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Log tails -->
  <div class="card full" id="logs-card">
    <h2>Live Logs (Running Batches)</h2>
    <div id="logs-content"><span class="no-data">No running batches.</span></div>
  </div>

  <!-- .nextflow.log tail -->
  <div class="card full" id="nflog-card">
    <h2>.nextflow.log <span style="font-weight:400;color:#475569;font-size:0.75rem">(shared, last 40 lines)</span></h2>
    <pre id="nflog-content" style="background:#0f172a;border-radius:6px;padding:10px 12px;font-size:0.7rem;line-height:1.5;color:#64748b;max-height:220px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin:0">Loading…</pre>
  </div>
</div>

<script>
let statusChart = null;
let _chartJsReady = false;

function onChartJsReady() {
  _chartJsReady = true;
  // Draw chart with latest data if already loaded
  loadStatus();
}

function badge(status) {
  const cls = {RUNNING:'running',SUCCEEDED:'succeeded',FAILED:'failed',PENDING:'pending',DISPATCHED:'running'}[status] || 'pending';
  return `<span class="badge badge-${cls}">${status}</span>`;
}

function fmtDuration(s, startIso, ongoing) {
  let sec = s;
  if ((sec === null || sec === undefined) && startIso) {
    // Compute elapsed from start to now for running batches
    try { sec = Math.floor((Date.now() - new Date(startIso).getTime()) / 1000); } catch {}
  }
  if (sec === null || sec === undefined || isNaN(sec) || sec < 0) return '—';
  let str;
  if (sec < 60) str = sec + 's';
  else if (sec < 3600) str = Math.floor(sec/60) + 'm ' + (sec%60) + 's';
  else str = Math.floor(sec/3600) + 'h ' + Math.floor((sec%3600)/60) + 'm';
  return ongoing ? str + ' <span title="ongoing" style="color:#f59e0b">⏳</span>' : str;
}

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch { return iso; }
}

async function apiFetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}

async function loadStatus() {
  try {
    const d = await apiFetch('/api/status');
    const c = d.counts || {};
    document.getElementById('s-total').textContent = d.total ?? '—';
    document.getElementById('s-done').textContent = c.SUCCEEDED ?? 0;
    document.getElementById('s-failed').textContent = c.FAILED ?? 0;
    document.getElementById('s-pending').textContent = c.PENDING ?? 0;
    document.getElementById('s-dispatched').textContent = c.DISPATCHED ?? 0;
    document.getElementById('s-pct').textContent = (d.pct_done ?? 0) + '%';
    document.getElementById('s-running').textContent = d.running_batches ?? 0;
    document.getElementById('s-blacklisted').textContent = d.blacklisted ?? 0;
    document.getElementById('progress-bar').style.width = (d.pct_done ?? 0) + '%';

    const labels = Object.keys(c);
    const values = Object.values(c);
    const colors = labels.map(l => ({
      SUCCEEDED:'#22c55e', FAILED:'#ef4444', PENDING:'#6366f1', DISPATCHED:'#0ea5e9'
    }[l] || '#64748b'));

    if (typeof Chart === 'undefined' || !_chartJsReady) return; // Chart.js not yet loaded
    if (!statusChart) {
      statusChart = new Chart(document.getElementById('statusChart'), {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }] },
        options: { plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
                   cutout: '60%' }
      });
    } else {
      statusChart.data.labels = labels;
      statusChart.data.datasets[0].data = values;
      statusChart.data.datasets[0].backgroundColor = colors;
      statusChart.update();
    }
  } catch(e) {
    document.getElementById('s-total').textContent = 'ERR';
    console.error('loadStatus failed:', e);
  }
}

async function loadBatches() {
  const tbody = document.getElementById('batch-tbody');
  const logsDiv = document.getElementById('logs-content');
  try {
    const batches = await apiFetch('/api/batches');
    if (!batches.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="no-data">No batches yet.</td></tr>';
      logsDiv.innerHTML = '<span class="no-data">No running batches.</span>';
      return;
    }
    tbody.innerHTML = batches.map(b => {
      const np = b.nf_progress;
      const tasksCell = np
        ? `<span title="${np.done} of ${np.total} tasks">${np.done}/${np.total} <span style="color:#6ee7b7">(${np.pct}%)</span></span>`
        : (b.slide_count ?? '—');
      const slurmR = b.slurm_running;
      const slurmP = b.slurm_pending;
      const nodes = (b.slurm_nodes || []).join(', ');
      let slurmCell = '—';
      if (slurmR !== null && slurmR !== undefined) {
        const tip = nodes ? `title="${nodes}"` : '';
        slurmCell = `<span ${tip} style="cursor:${nodes?'help':'default'}">` +
          `<span style="color:#6ee7b7">${slurmR}▶</span>` +
          (slurmP ? ` <span style="color:#fcd34d">${slurmP}⏳</span>` : '') +
          `</span>`;
      }
      // Alerts: killed > errors > warns
      let alertCell = '—';
      const parts = [];
      if (b.killed !== null && b.killed !== undefined)
        parts.push(`<span style="color:#f87171" title="Infra kill: ${b.killed} tasks killed">💀${b.killed}</span>`);
      if (b.error_count)
        parts.push(`<span style="color:#fca5a5" title="${b.first_error || 'errors'}">⛔${b.error_count}</span>`);
      if (b.warn_count)
        parts.push(`<span style="color:#fcd34d" title="${(b.last_warn || '').replace(/"/g,"'")}">⚠️${b.warn_count}</span>`);
      if (b.nextflow_exit !== null && b.nextflow_exit !== undefined && b.nextflow_exit !== 0)
        parts.push(`<span style="color:#f87171" title="NF exit code">exit:${b.nextflow_exit}</span>`);
      if (parts.length) alertCell = parts.join(' ');
      return `
      <tr>
        <td style="font-family:monospace;font-size:0.7rem">${b.batch_id}</td>
        <td>${badge(b.status)}</td>
        <td>${tasksCell}</td>
        <td>${slurmCell}</td>
        <td>${fmtTime(b.dispatched_at)}</td>
        <td>${fmtDuration(b.duration_s, b.dispatched_at, b.status === 'RUNNING')}</td>
        <td>${alertCell}</td>
        <td>${b.has_log ? `<button class="btn-refresh" onclick="showLog('${b.batch_id}')">View</button>` : '—'}</td>
      </tr>`;
    }).join('');

    const running = batches.filter(b => b.status === 'RUNNING');
    if (!running.length) {
      logsDiv.innerHTML = '<span class="no-data">No running batches.</span>';
      return;
    }

    // Add panels for new running batches; keep existing ones (update in-place)
    const existingIds = new Set([...logsDiv.querySelectorAll('[data-batch]')].map(el => el.dataset.batch));
    const runningIds = new Set(running.map(b => b.batch_id));

    // Remove panels for batches no longer running
    logsDiv.querySelectorAll('[data-batch]').forEach(el => {
      if (!runningIds.has(el.dataset.batch)) el.remove();
    });
    if (!logsDiv.querySelector('[data-batch]')) logsDiv.innerHTML = '';

    // Add new panels
    for (const b of running) {
      if (!existingIds.has(b.batch_id)) {
        const card = document.createElement('div');
        card.dataset.batch = b.batch_id;
        card.style.marginBottom = '12px';
        card.innerHTML = `
          <div class="log-header" style="display:flex;justify-content:space-between">
            <span style="font-family:monospace">${b.batch_id} <span style="color:#475569">(${b.slide_count} slides)</span></span>
            <button class="btn-refresh" onclick="showLog('${b.batch_id}')">Full log</button>
          </div>
          <pre class="log-block" id="logtext-${b.batch_id}" style="max-height:160px">Loading…</pre>`;
        logsDiv.appendChild(card);
      }
    }

    // Refresh log content for all running batches
    running.forEach(b => refreshInlineLog(b.batch_id));
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="7" class="no-data">Failed to load batches.</td></tr>';
    console.error('loadBatches failed:', e);
  }
}

async function refreshInlineLog(batch_id) {
  const el = document.getElementById('logtext-' + batch_id);
  if (!el) return;
  try {
    const d = await apiFetch('/api/logs/' + batch_id);
    const text = (d.lines || []).join('\\n') || d.error || '(empty)';
    el.textContent = text;
    el.scrollTop = el.scrollHeight;
    // Update shared .nextflow.log panel (last writer wins, that's fine)
    if (d.nf_log && d.nf_log.length) {
      const nfEl = document.getElementById('nflog-content');
      if (nfEl) {
        const atBottom = nfEl.scrollHeight - nfEl.scrollTop <= nfEl.clientHeight + 20;
        nfEl.textContent = d.nf_log.join('\\n');
        if (atBottom) nfEl.scrollTop = nfEl.scrollHeight;
      }
    }
  } catch { if (el) el.textContent = 'Failed to load log.'; }
}

let _modalBatchId = null;
let _modalInterval = null;

function openLogModal(batch_id) {
  _modalBatchId = batch_id;
  document.getElementById('log-modal-title').textContent = batch_id;
  document.getElementById('log-modal-body').textContent = 'Loading…';
  document.getElementById('log-modal').classList.add('open');
  refreshModalLog();
  clearInterval(_modalInterval);
  _modalInterval = setInterval(refreshModalLog, 5000);
}

function closeLogModal() {
  document.getElementById('log-modal').classList.remove('open');
  clearInterval(_modalInterval);
  _modalInterval = null;
  _modalBatchId = null;
}

async function refreshModalLog() {
  if (!_modalBatchId) return;
  const el = document.getElementById('log-modal-body');
  try {
    const d = await apiFetch('/api/logs/' + _modalBatchId);
    const text = (d.lines || []).join('\\n') + (d.nf_log && d.nf_log.length ? '\\n\\n--- .nextflow.log ---\\n' + d.nf_log.join('\\n') : '');
    const atBottom = el.scrollHeight - el.scrollTop <= el.clientHeight + 20;
    el.textContent = text || d.error || '(empty)';
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch { el.textContent = 'Failed to load log.'; }
}

// Close modal on backdrop click
document.getElementById('log-modal').addEventListener('click', function(e) {
  if (e.target === this) closeLogModal();
});

async function showLog(batch_id) { openLogModal(batch_id); }

async function loadWds() {
  const el = document.getElementById('wds-content');
  try {
    const data = await apiFetch('/api/wds');
    const models = data.models || {};
    const keys = Object.keys(models);
    const dbSucceeded = data.db_succeeded || 0;

    // Update summary grid % in WDS stat
    const totalWds = keys.reduce((a, m) => a + (models[m].slides || 0), 0);
    const wdsPctEl = document.getElementById('s-wds-pct');
    if (wdsPctEl && dbSucceeded > 0) {
      // Use minimum across models with actual slides (weakest link = true completion)
      const activeKeys = keys.filter(m => (models[m].slides || 0) > 0);
      const minWds = activeKeys.length ? Math.min(...activeKeys.map(m => models[m].slides)) : 0;
      const pct = Math.round(minWds / dbSucceeded * 100);
      wdsPctEl.textContent = pct + '%';
    }

    if (!keys.length) { el.innerHTML = '<span class="no-data">No WDS data yet.</span>'; return; }

    const header = `<table style="width:100%;border-collapse:collapse;font-size:0.78rem">
      <thead><tr style="color:#64748b;text-align:left">
        <th style="padding:4px 6px">Model</th>
        <th style="padding:4px 6px;text-align:right">WDS Slides</th>
        <th style="padding:4px 6px;text-align:right" title="SUCCEEDED in DB minus WDS-indexed slides">Gap</th>
        <th style="padding:4px 6px;text-align:right" title="Shards on S3 (from listing) / shards in manifest">S3 / Manifest Shards</th>
        <th style="padding:4px 6px;text-align:right" title="Average · min–max slides per shard (from manifest)">Slides/Shard</th>
        <th style="padding:4px 6px;text-align:right" title="Features .pt files still on local disk (pending cleanup)">Local .pt</th>
      </tr></thead><tbody>`;

    const rows = keys.map(m => {
      const mv = models[m];
      const n = mv.slides || 0;
      const gap = mv.gap || 0;
      const ns = mv.shards || 0;
      const ms = mv.manifest_shards || 0;
      const ss = mv.shard_stats || {};
      const localPt = mv.local_pt || 0;
      const gapColor = gap === 0 ? '#4ade80' : gap < 100 ? '#fcd34d' : '#f87171';
      const shardStr = mv.error
        ? `<span style="color:#f87171" title="${mv.error}">err</span> / ${ms}`
        : `<span style="color:#38bdf8">${ns}</span> / ${ms}`;
      const slidesPerShard = ss.avg != null
        ? `<span style="color:#c4b5fd">${ss.avg}</span> <span style="color:#64748b;font-size:0.7rem">(${ss.min}–${ss.max})</span>`
        : '—';
      const localColor = localPt === 0 ? '#4ade80' : localPt < 200 ? '#fcd34d' : '#f87171';
      return `<tr style="border-top:1px solid #1e293b">
        <td style="padding:4px 6px;font-weight:600">${m}</td>
        <td style="padding:4px 6px;text-align:right;color:#4ade80">${n}</td>
        <td style="padding:4px 6px;text-align:right;color:${gapColor}">${gap > 0 ? '+'+gap : '✓'}</td>
        <td style="padding:4px 6px;text-align:right">${shardStr}</td>
        <td style="padding:4px 6px;text-align:right">${slidesPerShard}</td>
        <td style="padding:4px 6px;text-align:right;color:${localColor}">${localPt}</td>
      </tr>`;
    });

    const footer = dbSucceeded
      ? `<tr style="border-top:1px solid #334155;color:#94a3b8">
          <td style="padding:4px 6px" colspan="2">SUCCEEDED in DB: <b style="color:#fff">${dbSucceeded}</b></td>
          <td colspan="4" style="padding:4px 6px;font-size:0.7rem;color:#64748b">Gap = SUCCEEDED − WDS indexed</td>
        </tr>` : '';

    el.innerHTML = header + rows.join('') + footer + '</tbody></table>';
  } catch(e) {
    el.innerHTML = '<span class="no-data">Failed to load WDS data.</span>';
    console.error('loadWds failed:', e);
  }
}

async function loadS3() { /* merged into loadWds */ }

async function refresh() {
  document.getElementById('last-updated').textContent = 'Refreshing…';
  await Promise.allSettled([loadStatus(), loadBatches(), loadWds(), loadSlurm()]);
  document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

async function loadSlurm() {
  const el = document.getElementById('slurm-content');
  try {
    const d = await apiFetch('/api/slurm');
    if (d.error) { el.innerHTML = `<span class="no-data">squeue error: ${d.error}</span>`; return; }

    // Summary line
    const summary = `<div style="font-size:1.1rem;margin-bottom:0.5rem">
      <span style="color:#6ee7b7;font-weight:600">${d.running}</span> running &nbsp;
      <span style="color:#fcd34d;font-weight:600">${d.pending}</span> pending &nbsp;
      <span style="color:#94a3b8">(${d.total} total)</span>
    </div>`;

    // Node utilization
    const nodes = Object.entries(d.nodes || {}).sort((a,b) => b[1]-a[1]);
    const nodeHtml = nodes.length ? `<div style="margin-bottom:0.4rem"><span style="color:#94a3b8;font-size:0.75rem">NODES</span><br>` +
      nodes.map(([n,c]) => `<span style="font-family:monospace;font-size:0.8rem">${n}</span> <span style="color:#6ee7b7">×${c}</span>`).join(' &nbsp; ') +
      `</div>` : '';

    // Pending reasons
    const reasons = Object.entries(d.pending_reasons || {}).sort((a,b) => b[1]-a[1]);
    const reasonHtml = reasons.length ? `<div style="margin-bottom:0.6rem"><span style="color:#94a3b8;font-size:0.75rem">PENDING REASONS</span><br>` +
      reasons.map(([r,c]) => `<span style="color:#fcd34d">${r}</span>: ${c}`).join(' &nbsp; ') +
      `</div>` : '';

    // sacct history (last 24h)
    let histHtml = '';
    if (d.completed !== undefined) {
      const parts = [];
      if (d.completed) parts.push(`<span style="color:#6ee7b7">✔ ${d.completed} completed</span>`);
      if (d.failed)    parts.push(`<span style="color:#f87171">✖ ${d.failed} failed</span>`);
      if (d.cancelled) parts.push(`<span style="color:#94a3b8">⊘ ${d.cancelled} cancelled</span>`);
      if (d.timeout)   parts.push(`<span style="color:#fcd34d">⏱ ${d.timeout} timeout</span>`);
      let avgStr = '';
      if (d.avg_elapsed_s !== null && d.avg_elapsed_s !== undefined) {
        avgStr = ` &nbsp; <span style="color:#94a3b8">avg ${fmtDuration(d.avg_elapsed_s)} &nbsp; min ${fmtDuration(d.min_elapsed_s)} &nbsp; max ${fmtDuration(d.max_elapsed_s)}</span>`;
      }
      histHtml = `<div style="margin-bottom:0.4rem"><span style="color:#94a3b8;font-size:0.75rem">LAST 24H</span><br>${parts.join(' &nbsp; ')}${avgStr}</div>`;

      // Failure type breakdown
      const ftypes = d.failure_types || {};
      const ftypeLabels = {
        sigterm:      {icon:'⚡', label:'SIGTERM (infra)', color:'#94a3b8'},
        oom_gpu:      {icon:'🔥', label:'GPU OOM',         color:'#f87171'},
        oom_host:     {icon:'💥', label:'Host OOM',        color:'#f87171'},
        disk_full:    {icon:'💾', label:'Disk full',       color:'#fcd34d'},
        s3_error:     {icon:'☁️', label:'S3 error',        color:'#fbbf24'},
        python_error: {icon:'🐍', label:'Python error',    color:'#fb923c'},
        timeout:      {icon:'⏱', label:'Timeout',          color:'#fcd34d'},
        error_exit1:  {icon:'❌', label:'Exit 1',          color:'#f87171'},
        unknown:      {icon:'❓', label:'Unknown',          color:'#94a3b8'},
      };
      const ftypeEntries = Object.entries(ftypes).sort((a,b) => b[1]-a[1]);
      if (ftypeEntries.length) {
        const ftHtml = ftypeEntries.map(([k, n]) => {
          const lbl = ftypeLabels[k] || {icon:'⚠️', label:k, color:'#94a3b8'};
          return `<span style="color:${lbl.color}" title="${lbl.label}">${lbl.icon} ${lbl.label}: ${n}</span>`;
        }).join(' &nbsp; ');
        histHtml += `<div style="font-size:0.8rem">${ftHtml}</div>`;
      }
      if (d.sacct_error) histHtml += `<div style="color:#f87171;font-size:0.75rem">sacct: ${d.sacct_error}</div>`;
    }

    el.innerHTML = summary + nodeHtml + reasonHtml + histHtml;
  } catch(e) {
    el.innerHTML = '<span class="no-data">Failed to load SLURM data.</span>';
  }
}

refresh();
setInterval(refresh, 10000);
setInterval(loadSlurm, 15000);  // matches squeue cache TTL
setInterval(loadWds, 60000);  // S3 stats are expensive — refresh less often via loadWds
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to dispatcher YAML config file")
    parser.add_argument("--port", type=int, default=8050,
                        help="HTTP port to listen on (default: 8050)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    # If ECS credentials aren't in env, try to read them from Nextflow secrets
    for env_key, nf_secret in [("ECS_ACCESS_KEY", "ECS_ACCESS_KEY"),
                                ("ECS_SECRET_KEY", "ECS_SECRET_KEY")]:
        if not os.environ.get(env_key):
            try:
                val = subprocess.check_output(
                    ["nextflow", "secrets", "get", nf_secret],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                ).strip()
                if val:
                    os.environ[env_key] = val
            except Exception:
                pass

    handler_cls = _build_handler(cfg)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Dashboard: http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
