#!/usr/bin/env python3
"""Mussel-nf Dispatcher Monitoring Dashboard.

Reads configuration from the dispatcher YAML file (same file used to run the
dispatcher) and exposes a browser-based dashboard via stdlib http.server
(no external dependencies required).

Usage:
    mussel-dashboard dispatcher/tcga_dispatcher.yaml
    mussel-dashboard dispatcher/tcga_dispatcher.yaml --port 8080
    python -m mussel_dispatcher.dashboard dispatcher/tcga_dispatcher.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re as _re
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mussel_dispatcher.config import Config, WatcherConfig
from mussel_dispatcher.dashboard import helpers as _helpers

parse_nf_log  = _helpers.parse_nf_log
slurm_stats   = _helpers.slurm_stats
s3_stats      = _helpers.s3_stats
_s3_cache     = _helpers._s3_cache
_S3_CACHE_TTL = _helpers._S3_CACHE_TTL

# ---------------------------------------------------------------------------
# HTML template — loaded from static/dashboard.html at startup
# ---------------------------------------------------------------------------
_HTML = (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib — no FastAPI/uvicorn dependency)
# ---------------------------------------------------------------------------

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
            target=s3_stats, args=(tcga_watcher, "wds"), daemon=True, name="s3-prewarm"
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
        in_flight_done = 0
        in_flight_total = 0
        for rb in running_rows:
            log_info = parse_nf_log(rb["log_path"]) if rb["log_path"] else {}
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
        slurm = slurm_stats()
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
            log_info = parse_nf_log(r["log_path"]) if r["log_path"] else {}
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
                "failures": log_info.get("failures", []),
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

        # SUCCEEDED + DISPATCHED count from DB — denominator for WDS % coverage.
        with _db() as conn:
            db_succeeded = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status='SUCCEEDED'"
            ).fetchone()[0]
            db_dispatched = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status='DISPATCHED'"
            ).fetchone()[0]
        db_total_expected = db_succeeded + db_dispatched

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
        cached_s3: dict = {}
        if tcga_watcher and tcga_watcher.wds_destinations:
            now = time.time()
            if _s3_cache:
                cached_s3 = {m: {"shards": v.get("shards", 0), "objects": v.get("objects", 0),
                                  "error": v.get("error")} for m, v in _s3_cache.items()}
            oldest = min((v.get("ts", 0) for v in _s3_cache.values()), default=0)
            if now - oldest > _S3_CACHE_TTL:
                threading.Thread(
                    target=s3_stats, args=(tcga_watcher, "wds"), daemon=True, name="s3-refresh"
                ).start()

        models: dict = {}
        configured_models = set(tcga_watcher.wds_destinations.keys()) if (
            tcga_watcher and tcga_watcher.wds_destinations
        ) else set()
        for m in sorted(set(wds_counts) | configured_models):
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
                "gap": max(0, db_total_expected - wds_slides),
                "shards": cached_s3.get(m, {}).get("shards", 0),
                "manifest_shards": manifest_shards,
                "shard_stats": shard_stats,
                "local_pt": local_pt.get(m, 0),
                "error": cached_s3.get(m, {}).get("error"),
            }
        return {"models": models, "total": sum(wds_counts.values()),
                "db_succeeded": db_succeeded, "db_dispatched": db_dispatched,
                "db_total_expected": db_total_expected}

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
                    self._send_json(slurm_stats())
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
