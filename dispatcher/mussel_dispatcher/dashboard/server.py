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
import nextflow_turret as _tower_shim
from nextflow_turret import TowerRouter as _TowerRouter
from nextflow_turret import TowerHandlerMixin as _TowerHandlerMixin
from nextflow_turret import PersistentWorkflowRegistry as _PersistentRegistry
from nextflow_turret import RunStore as _RunStore
from nextflow_turret import tower_process_to_slurm_name

slurm_stats   = _helpers.slurm_stats
s3_stats      = _helpers.s3_stats
_s3_cache     = _helpers._s3_cache
_S3_CACHE_TTL = _helpers._S3_CACHE_TTL


# ---------------------------------------------------------------------------
# Manifest helpers (module-level so they can be unit-tested)
# ---------------------------------------------------------------------------

def _count_unique_slides_from_manifest(
    manifest_path: str, models: "list[str] | None" = None
) -> "dict[str, int]":
    """Return {model: unique_slide_count} from a WDS manifest CSV.

    Counts unique *slide_ids* per model — not rows — because each slide may
    appear in multiple shards (and across multiple pipeline runs) producing
    duplicate entries.  Over-counting rows would inflate the WDS progress
    percentage past 100 % when shards accumulate across runs.

    ``models`` is an optional allowlist; if None, all models in the file are
    counted.  Returns ``{}`` if the file does not exist or cannot be read.
    """
    unique_slides, _ = _parse_wds_manifest(manifest_path, models=models)
    return {m: len(ids) for m, ids in unique_slides.items()}


def _parse_wds_manifest(
    manifest_path: str,
    models: "list[str] | None" = None,
) -> "tuple[dict[str, set], dict[str, dict[str, int]]]":
    """Parse a WDS manifest CSV and return (unique_slides, shard_slide_counts).

    * ``unique_slides``: ``{model: set(slide_id)}`` — deduplicated.
    * ``shard_slide_counts``: ``{model: {shard_path: row_count}}`` — one entry
      per manifest row, used for per-shard distribution stats.

    Returns empty dicts if the file does not exist or cannot be read.
    ``models`` is an optional allowlist; if None, all models are included.
    """
    unique_slides: dict[str, set] = {}
    shard_slide_counts: dict[str, dict[str, int]] = {}
    if not os.path.exists(manifest_path):
        return unique_slides, shard_slide_counts
    try:
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                m = row.get("model", "unknown")
                slide_id = row.get("slide_id", "")
                shard = row.get("wds_path", "")
                if models is not None and m not in models:
                    continue
                if slide_id:
                    unique_slides.setdefault(m, set()).add(slide_id)
                model_shards = shard_slide_counts.setdefault(m, {})
                model_shards[shard] = model_shards.get(shard, 0) + 1
    except Exception:
        return {}, {}
    return unique_slides, shard_slide_counts

# ---------------------------------------------------------------------------
# HTML template — loaded from static/dashboard.html at startup
# ---------------------------------------------------------------------------
_HTML = (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib — no FastAPI/uvicorn dependency)
# ---------------------------------------------------------------------------

def _seed_registry_from_db(registry, db_path: str) -> None:
    """Seed registry from legacy tasks_done/total/failed columns in dispatcher.db.

    Backfills any batch that has task count data in the old schema columns but
    is not yet present in the persistent RunStore. Runs once at startup.
    """
    import nextflow_turret.state as _nt_state
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        # Check whether the legacy columns still exist
        cols = {row[1] for row in conn.execute("PRAGMA table_info(batches)")}
        if not {"tasks_done", "tasks_total", "tasks_failed"}.issubset(cols):
            return
        rows = conn.execute(
            "SELECT batch_id, tasks_done, tasks_total, tasks_failed "
            "FROM batches WHERE tasks_total IS NOT NULL"
        ).fetchall()
        conn.close()
        for r in rows:
            batch_id = r["batch_id"]
            wf_id = _nt_state.workflow_id_for_batch(batch_id)
            if registry.get_by_id(wf_id) is not None:
                continue  # already in registry
            run_name = f"dispatcher_{batch_id}"
            registry.register(wf_id, batch_id, run_name)
            done   = r["tasks_done"] or 0
            total  = r["tasks_total"] or 0
            failed = r["tasks_failed"] or 0
            progress = {
                "succeeded": done,
                "failed":    failed,
                "cached":    0,
                "running":   0,
                "pending":   max(0, total - done - failed),
            }
            registry.mark_complete(wf_id, progress)
    except Exception as exc:
        import logging
        logging.getLogger("mussel-dispatcher").warning(
            "_seed_registry_from_db: %s", exc
        )


def _build_handler(cfg: Config):
    """Return a BaseHTTPRequestHandler subclass closed over cfg."""

    db_path = os.path.join(cfg.state_dir, "dispatcher.db")
    wds_manifest = os.path.join(cfg.outdir, "wds_manifest.csv")
    nf_log_path = os.path.join(cfg.repo_dir, ".nextflow.log")
    sync_status_path = os.path.join(cfg.state_dir, "sync_status.json")

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

    # Cache inventory total (unique slides the dispatcher should eventually process).
    # Filter by the watcher's slide_type (e.g. "DX" matches DX1, DX2, ...) since
    # the inventory CSV contains all slide types (TS, BS, MS, etc.).
    _inventory_total: int = 0
    if tcga_watcher and getattr(tcga_watcher, "inventory_csv", ""):
        inv_path = tcga_watcher.inventory_csv
        slide_type_prefix = getattr(tcga_watcher, "slide_type", "") or ""
        if os.path.exists(inv_path):
            try:
                with open(inv_path, newline="") as _f:
                    _reader = csv.DictReader(_f)
                    if slide_type_prefix:
                        _inventory_total = sum(
                            1 for r in _reader
                            if r.get("slide_type", "").startswith(slide_type_prefix)
                        )
                    else:
                        _inventory_total = sum(1 for _ in _reader)
            except Exception:
                pass

    # Number of WDS models required per slide.
    _n_models: int = len(tcga_watcher.wds_destinations) if (
        tcga_watcher and tcga_watcher.wds_destinations
    ) else 1

    # WDS manifest done-count cache: (timestamp, {model: count}) updated every 60s.
    _wds_manifest_cache: list = [0.0, {}]  # [last_read_time, {model: count}]

    def _wds_done_counts() -> dict[str, int]:
        """Return {model: unique_slide_count} from wds_manifest, with 60s caching."""
        now = time.time()
        if now - _wds_manifest_cache[0] < 60:
            return _wds_manifest_cache[1]
        counts = _count_unique_slides_from_manifest(wds_manifest)
        _wds_manifest_cache[0] = now
        _wds_manifest_cache[1] = counts
        return counts

    _sync_status_cache: list = [0.0, None]  # [last_read_time, payload|None]

    def _read_sync_status() -> dict | None:
        """Read sync_status.json written by the Databricks sync hook (60s cache)."""
        now = time.time()
        if now - _sync_status_cache[0] < 60:
            return _sync_status_cache[1]
        payload = None
        if os.path.exists(sync_status_path):
            try:
                import json as _json
                payload = _json.loads(Path(sync_status_path).read_text())
            except Exception:
                pass
        _sync_status_cache[0] = now
        _sync_status_cache[1] = payload
        return payload

    # Persistent registry: Tower state survives dashboard restarts.
    _run_store = _RunStore(os.path.join(cfg.state_dir, "turret_runs.db"))
    _registry  = _PersistentRegistry(_run_store)
    # Seed registry from legacy tasks_done/total/failed columns for batches
    # that completed before the PersistentWorkflowRegistry was introduced.
    _seed_registry_from_db(_registry, db_path)

    def _run_name_to_batch_id(run_name: str) -> str:
        """Map NF run name → dispatcher batch_id.

        Supports two formats:
          - Legacy: ``dispatcher_{batch_id}``  (pre-fix, full 35-char name)
          - Modern: ``{hash8}``                (8-char UUID suffix, e.g. "410bb3ce")
        """
        if run_name.startswith("dispatcher_"):
            return run_name[len("dispatcher_"):]
        # Modern short hash: look up batch whose batch_id ends with _{hash}
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT batch_id FROM batches WHERE batch_id LIKE ?",
                    (f"%_{run_name}",)
                ).fetchone()
                if row:
                    return row[0]
        except Exception:
            pass
        return run_name

    # Single TowerRouter instance shared across all requests in this handler.
    _router = _TowerRouter(registry=_registry, run_name_to_batch_id=_run_name_to_batch_id)

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
        db_total = sum(counts.values())
        succeeded = counts.get("SUCCEEDED", 0)

        # Use inventory total as the true denominator (all slides to eventually process).
        # Fall back to db_total if inventory is unavailable.
        true_total = (_inventory_total * _n_models) if _inventory_total else db_total

        # Use WDS manifest counts as the authoritative "done" tally — more reliable
        # than DB SUCCEEDED (which can be reset to PENDING by _verify_wds_coverage).
        wds_counts = _wds_done_counts()
        # "done" = slides confirmed in WDS for EVERY required model (min across models).
        # Cap at _inventory_total: WDS manifests accumulate across all historical runs,
        # so raw counts can exceed the current inventory and produce misleading >100% pct.
        wds_done_raw = min(wds_counts.get(m, 0) for m in tcga_watcher.wds_destinations) if (
            tcga_watcher and tcga_watcher.wds_destinations and wds_counts
        ) else succeeded
        denom = _inventory_total or db_total
        wds_done = min(wds_done_raw, denom)

        # pct_wds: slides confirmed in WDS vs inventory
        pct_wds = round(wds_done / denom * 100, 1) if denom else 0

        # pct_succeeded: DB SUCCEEDED vs inventory — reflects features-extracted progress
        # independently of WDS upload, and is never inflated by historical WDS entries.
        pct_succeeded = round(min(succeeded, denom) / denom * 100, 1) if denom else 0

        with _db() as conn:
            running_rows = conn.execute(
                "SELECT batch_id, log_path, slide_count FROM batches WHERE status='RUNNING'"
            ).fetchall()
            n_blacklisted = conn.execute(
                f"SELECT COUNT(*) FROM slides WHERE fail_count >= {cfg.max_slide_retries}"
            ).fetchone()[0]

        n_running = len(running_rows)

        with _db() as conn:
            tp = conn.execute(
                """
                SELECT COUNT(*) AS cnt,
                       MIN(completed_at) AS oldest,
                       MAX(completed_at) AS newest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND completed_at >= datetime('now', '-6 hours')
                """
            ).fetchone()
            remaining = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status IN ('PENDING','DISPATCHED')"
            ).fetchone()[0]
        # Add undispatched slides to remaining count
        undispatched = max(0, _inventory_total - db_total) if _inventory_total else 0
        total_remaining = remaining + undispatched

        throughput_per_hour = None
        eta_seconds = None
        completed_in_window = tp["cnt"] or 0
        if completed_in_window >= 2 and tp["oldest"] and tp["newest"]:
            try:
                from datetime import datetime as _dt
                t0 = _dt.fromisoformat(tp["oldest"])
                t1 = _dt.fromisoformat(tp["newest"])
                elapsed = (t1 - t0).total_seconds()
                if elapsed > 60:
                    throughput_per_hour = round(completed_in_window / (elapsed / 3600), 1)
                    eta_seconds = round(total_remaining / throughput_per_hour * 3600)
            except Exception:
                pass

        return {
            "counts": counts,
            "total": _inventory_total,
            "wds_done": wds_done,
            "pct_done": pct_wds,
            "pct_succeeded": pct_succeeded,
            "running_batches": n_running,
            "blacklisted": n_blacklisted,
            "throughput_per_hour": throughput_per_hour,
            "eta_seconds": eta_seconds,
            "remaining": total_remaining,
            "completed_in_window": completed_in_window,
            "sync_status": _read_sync_status(),
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
            slurm_batch = jobs_by_batch.get(r["batch_id"], {})
            # Registry is the sole source of progress data (hydrated from DB on restart).
            tower = _registry.get_by_batch(r["batch_id"])
            # Enrich Tower processes with SLURM node/elapsed info.
            # Tower process names (e.g. "MUSSEL:EXTRACT_FEATURES:TESSELLATE_FEATURIZE_BATCH")
            # map to SLURM job name prefixes by replacing ":" with "_".
            if tower and tower.get("processes") and slurm_batch.get("processes"):
                slurm_procs = slurm_batch["processes"]
                for proc in tower["processes"]:
                    slurm_key = tower_process_to_slurm_name(
                        proc.get("process", proc.get("name", ""))
                    )
                    sp = slurm_procs.get(slurm_key) or slurm_procs.get(slurm_key + "_")
                    if sp:
                        proc["slurm_running"] = sp["running"]
                        proc["slurm_nodes"]   = sp["nodes"]
                        elapsed = sp.get("elapsed_s", [])
                        proc["slurm_elapsed_max"] = max(elapsed) if elapsed else None
            result.append({
                "batch_id":      r["batch_id"],
                "status":        r["status"],
                "slide_count":   r["slide_count"],
                "dispatched_at": start,
                "completed_at":  end,
                "duration_s":    duration,
                "nextflow_exit": r["nextflow_exit"],
                "has_log":       bool(r["log_path"] and os.path.exists(r["log_path"])),
                # Tower live data (None when Tower not active for this batch)
                "tower":         tower,
                # SLURM data still useful for node/queue visibility
                "slurm_running": slurm_batch.get("running"),
                "slurm_pending": slurm_batch.get("pending"),
                "slurm_nodes":   slurm_batch.get("nodes", []),
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

        # WDS manifest counts + per-shard distribution
        # unique_slides: deduplicated slide_id per model (correct denominator for % done)
        # shard_slide_counts: row count per shard path (used for shard distribution stats only)
        unique_slides, shard_slide_counts = _parse_wds_manifest(wds_manifest)

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
        for m in sorted(set(unique_slides) | configured_models):
            wds_slides = len(unique_slides.get(m, set()))
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
                "gap": max(0, _inventory_total - wds_slides),
                "shards": cached_s3.get(m, {}).get("shards", 0),
                "manifest_shards": manifest_shards,
                "shard_stats": shard_stats,
                "local_pt": local_pt.get(m, 0),
                "error": cached_s3.get(m, {}).get("error"),
            }
        with _db() as conn:
            db_succeeded = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status='SUCCEEDED'"
            ).fetchone()[0]
        total_rows = sum(len(s) for s in unique_slides.values())
        return {"models": models, "total": total_rows,
                "inventory_total": _inventory_total, "db_succeeded": db_succeeded}

    class Handler(_TowerHandlerMixin, BaseHTTPRequestHandler):
        # Expose registry and router so tests and external callers can inspect Tower state.
        tower_registry = _registry
        tower_router   = _router

        def log_message(self, fmt, *args):  # suppress default access log spam
            pass

        def log_tower(self, method, path, status, note=""):
            if "/trace" in path or "user" in path:
                print(f"TOWER {method} {path} -> {status} {note}", flush=True)

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
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                # Dashboard API routes
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
                    # Delegate to TowerRouter (handles /user-info)
                    result = _router.handle_get(path)
                    if result is not None:
                        status, body = result
                        self.log_tower("GET", path, status)
                        self._send_json(body, status=status)
                    else:
                        self.log_tower("GET", path, 404)
                        self.send_response(404)
                        self.end_headers()
            except Exception as exc:
                try:
                    self._send_json({"error": str(exc)}, status=500)
                except Exception:
                    pass

        def _read_body(self) -> dict:
            """Read JSON request body."""
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                return json.loads(raw)
            except Exception:
                return {}

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
