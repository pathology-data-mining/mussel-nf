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
import logging
import os
import re as _re
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mussel_dispatcher.config import Config, WatcherConfig
from mussel_dispatcher.dashboard import helpers as _helpers
from mussel_dispatcher.scheduler import _find_same_config_dispatchers
from mussel_dispatcher.state import estimate_throughput_and_eta, _tower_is_live_for_batch
from mussel_dispatcher.state import StateStore
from mussel_dispatcher.tower_state import newest_tower_state_for_batch, normalize_tower_state
from mussel_dispatcher.wds_status import parse_wds_index as _parse_wds_index
from mussel_dispatcher.wds_status import parse_wds_manifest as _parse_wds_manifest
import nextflow_turret as _tower_shim
from nextflow_turret import TowerRouter as _TowerRouter
from nextflow_turret import TowerHandlerMixin as _TowerHandlerMixin
from nextflow_turret import PersistentWorkflowRegistry as _PersistentRegistry
from nextflow_turret import RunStore as _RunStore
from nextflow_turret import tower_process_to_slurm_name
from mussel_dispatcher.db import connect_database, resolve_database_config, sql_for_backend

slurm_stats   = _helpers.slurm_stats
s3_stats      = _helpers.s3_stats
_s3_cache     = _helpers._s3_cache
_S3_CACHE_TTL = _helpers._S3_CACHE_TTL
log = logging.getLogger("mussel-dispatcher.dashboard")
_DASHBOARD_FILE_CACHE_TTL = 300.0
_DASHBOARD_LOG_TAIL_BYTES = 256 * 1024


def _table_columns(conn, db_cfg, table: str) -> set[str]:
    """Return column names for a table across supported database backends."""
    if db_cfg.backend == "postgres":
        rows = conn.execute(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema = current_schema()
               AND table_name = %s
            """,
            (table,),
        ).fetchall()
        cols = set()
        for row in rows:
            try:
                value = row["column_name"]
            except Exception:
                try:
                    value = row[0]
                except Exception:
                    value = None
            if value:
                cols.add(str(value))
        return cols
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that rejects excess work instead of queuing it forever."""

    daemon_threads = True

    def __init__(self, server_address, request_handler_class, *, max_requests: int = 16):
        super().__init__(server_address, request_handler_class)
        self._request_slots = threading.BoundedSemaphore(max_requests)

    def process_request(self, request, client_address):
        if not self._request_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return

        def _serve_request() -> None:
            try:
                self.finish_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)
                self._request_slots.release()

        thread = threading.Thread(target=_serve_request, daemon=self.daemon_threads)
        thread.start()


def _read_dispatcher_health(
    pid_path: str,
    log_path: str | None = None,
    *,
    config_path: str | None = None,
) -> dict:
    """Return dispatcher liveness derived from the PID lockfile."""
    def _set_health_from_live_matches(
        message_when_single: str,
        *,
        prefer_pid: int | None = None,
    ) -> dict | None:
        live_pids = sorted(set(health["duplicate_pids"]))
        if prefer_pid is not None and prefer_pid in live_pids:
            health["pid"] = prefer_pid
        elif live_pids:
            health["pid"] = live_pids[0]
        if len(live_pids) > 1:
            health["state"] = "duplicate"
            health["duplicate_pids"] = live_pids
            health["message"] = (
                "Multiple dispatcher processes detected for this config: "
                f"{', '.join(str(p) for p in live_pids)}."
            )
            return health
        if len(live_pids) == 1:
            health["state"] = "alive"
            health["duplicate_pids"] = live_pids
            health["message"] = message_when_single.format(pid=live_pids[0])
            return health
        return None

    health = {
        "state": "unknown",
        "message": "Dispatcher health unknown.",
        "pid": None,
        "pid_path": pid_path,
        "log_path": log_path,
        "log_mtime": None,
        "duplicate_pids": [],
    }
    if config_path:
        try:
            health["duplicate_pids"] = [pid for pid, _ in _find_same_config_dispatchers(config_path)]
        except Exception:
            pass
    if log_path and os.path.exists(log_path):
        try:
            health["log_mtime"] = datetime.fromtimestamp(
                os.path.getmtime(log_path)
            ).astimezone().isoformat()
        except OSError:
            pass
    if not os.path.exists(pid_path):
        fallback = _set_health_from_live_matches(
            "Dispatcher PID file is missing; live dispatcher detected (PID {pid})."
        )
        if fallback is not None:
            return fallback
        health["state"] = "dead"
        health["message"] = "Dispatcher PID file is missing."
        return health
    try:
        raw = Path(pid_path).read_text().strip().split()
        pid = int(raw[0])
        health["pid"] = pid
    except Exception:
        fallback = _set_health_from_live_matches(
            "Dispatcher PID file is unreadable; live dispatcher detected (PID {pid})."
        )
        if fallback is not None:
            return fallback
        health["state"] = "dead"
        health["message"] = "Dispatcher PID file is unreadable."
        return health
    if config_path and health["duplicate_pids"] == [pid]:
        health["state"] = "alive"
        health["message"] = f"Dispatcher is running (PID {pid})."
        return health
    try:
        os.kill(pid, 0)
        if config_path:
            fallback = _set_health_from_live_matches(
                "Dispatcher is running (PID {pid}).",
                prefer_pid=pid,
            )
            if fallback is not None:
                if health["pid"] != pid:
                    health["message"] = (
                        f"Dispatcher PID file points to PID {pid}, but live dispatcher "
                        f"detected at PID {health['pid']}."
                    )
                return health
            health["state"] = "dead"
            health["message"] = (
                f"Dispatcher PID {pid} is running, but it does not match config {config_path}."
            )
            return health
        health["state"] = "alive"
        health["message"] = f"Dispatcher is running (PID {pid})."
        return health
    except OSError:
        fallback = _set_health_from_live_matches(
            "Dispatcher PID file is stale; live dispatcher detected (PID {pid})."
        )
        if fallback is not None:
            return fallback
        health["state"] = "dead"
        health["message"] = f"Dispatcher PID {pid} is not running."
        return health


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


# ---------------------------------------------------------------------------
# HTML template — loaded from static/dashboard.html at startup
# ---------------------------------------------------------------------------
_HTML = (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib — no FastAPI/uvicorn dependency)
# ---------------------------------------------------------------------------

def _seed_registry_from_db(registry, cfg: Config) -> None:
    """Seed registry from legacy tasks_done/total/failed columns in dispatcher.db.

    Backfills any batch that has task count data in the old schema columns but
    is not yet present in the persistent RunStore. Runs once at startup.
    """
    import nextflow_turret.state as _nt_state
    db_cfg = resolve_database_config(cfg)
    if db_cfg.backend != "sqlite":
        return
    try:
        conn = connect_database(db_cfg, readonly=True)
        # Check whether the legacy columns still exist
        cols = _table_columns(conn, db_cfg, "batches")
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

    db_cfg = resolve_database_config(cfg)
    dispatcher_pid_path = os.path.join(cfg.state_dir, "dispatcher.pid")
    dispatcher_log_path = os.path.join(cfg.log_dir, "dispatcher.log")
    if not os.path.exists(dispatcher_log_path):
        dispatcher_log_path = os.path.join(cfg.log_dir, "dispatcher.nohup.out")
    wds_manifest = os.path.join(cfg.outdir, "wds_manifest.csv")
    wds_staging_dir = os.path.join(cfg.state_dir, "wds-staging")
    nf_log_path = os.path.join(cfg.repo_dir, ".nextflow.log")
    sync_status_path = os.path.join(cfg.state_dir, "sync_status.json")
    dispatcher_status_snapshot_path = os.path.join(cfg.state_dir, "dispatcher_status.json")

    tcga_watcher = None
    for w in cfg.watchers:
        if w.wds_destinations:
            tcga_watcher = w
            break

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
    _wds_index_cache: list = [0.0, {}]  # [last_read_time, {model: {"slides", "slide_to_shard"}}]
    _db_slide_ids_cache: list = [0.0, None, None]  # [last_read_time, all_slide_ids, succeeded_slide_ids]
    _local_pt_cache: list = [0.0, {}]  # [last_read_time, {model: pt_count}]

    def _row_val(row, key: str, index: int = 0):
        if row is None:
            return None
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            return row[index]

    def _db_slide_ids_cached() -> tuple[set[str], set[str]]:
        """Return (all_db_slide_ids, succeeded_slide_ids) with a short TTL cache."""
        now = time.time()
        if now - _db_slide_ids_cache[0] < 60 and _db_slide_ids_cache[1] is not None:
            return _db_slide_ids_cache[1], _db_slide_ids_cache[2]
        with _db() as conn:
            all_slide_ids = {
                _row_val(row, "slide_id") for row in conn.execute("SELECT DISTINCT slide_id FROM slides")
                if _row_val(row, "slide_id")
            }
            succeeded_slide_ids = {
                _row_val(row, "slide_id")
                for row in conn.execute(
                    "SELECT DISTINCT slide_id FROM slides WHERE status='SUCCEEDED'"
                )
                if _row_val(row, "slide_id")
            }
        _db_slide_ids_cache[0] = now
        _db_slide_ids_cache[1] = all_slide_ids
        _db_slide_ids_cache[2] = succeeded_slide_ids
        return all_slide_ids, succeeded_slide_ids

    def _wds_index_data_cached() -> dict[str, dict]:
        """Return cached per-model WDS index data from state/wds-staging."""
        now = time.time()
        if now - _wds_index_cache[0] < 60:
            return _wds_index_cache[1]
        data: dict[str, dict] = {}
        models = set(tcga_watcher.wds_destinations.keys()) if (
            tcga_watcher and tcga_watcher.wds_destinations
        ) else set()
        for model in models:
            index_path = os.path.join(wds_staging_dir, model, "wds_index.json")
            slide_ids, slide_to_shard = _parse_wds_index(index_path)
            data[model] = {
                "slides": slide_ids,
                "slide_to_shard": slide_to_shard,
            }
        _wds_index_cache[0] = now
        _wds_index_cache[1] = data
        return data

    def _local_pt_counts_cached() -> dict[str, int]:
        """Return cached recursive .features.pt counts under outdir/features/*."""
        now = time.time()
        if now - _local_pt_cache[0] < _DASHBOARD_FILE_CACHE_TTL:
            return _local_pt_cache[1]
        counts: dict[str, int] = {}
        models = set(tcga_watcher.wds_destinations.keys()) if (
            tcga_watcher and tcga_watcher.wds_destinations
        ) else set()
        features_root = os.path.join(cfg.outdir, "features")
        for model in models:
            model_dir = os.path.join(features_root, model)
            n = 0
            if os.path.isdir(model_dir):
                for _root, _dirs, files in os.walk(model_dir):
                    n += sum(1 for name in files if name.endswith(".features.pt"))
            counts[model] = n
        _local_pt_cache[0] = now
        _local_pt_cache[1] = counts
        return counts

    def _wds_done_counts() -> dict[str, int]:
        """Return current tracked-slide WDS counts per model, with 60s caching."""
        now = time.time()
        if now - _wds_manifest_cache[0] < 60:
            return _wds_manifest_cache[1]
        index_data = _wds_index_data_cached()
        if not index_data:
            _wds_manifest_cache[0] = now
            _wds_manifest_cache[1] = {}
            return {}
        counts = {
            model: len(info.get("slides", set()))
            for model, info in index_data.items()
        }
        _wds_manifest_cache[0] = now
        _wds_manifest_cache[1] = counts
        return counts

    # Full WDS manifest parsing walks a potentially large GPFS file. It is
    # legacy visibility data, so keep it well below the dashboard poll rate.
    _wds_full_cache: list = [0.0, None, None]  # [ts, unique_slides, shard_counts]
    _wds_response_cache: dict[str, object] = {
        "updated_at": 0.0,
        "payload": None,
        "last_error": None,
        "next_retry_at": 0.0,
        "failure_backoff_s": 60.0,
    }
    _wds_cache_lock = threading.Lock()
    _wds_refresh_lock = threading.Lock()
    _WDS_CACHE_TTL = 60.0
    _WDS_STALE_FALLBACK_TTL = 15 * 60.0
    _WDS_FAILURE_BACKOFF_MAX = 10 * 60.0

    def _empty_wds_payload() -> dict:
        return {
            "models": {},
            "total": 0,
            "inventory_total": _inventory_total,
            "db_total": 0,
            "db_succeeded": 0,
        }
    _s3_refresh_lock = threading.Lock()
    _s3_refreshing = False

    def _wds_parsed_cached():
        """Return cached (unique_slides, shard_slide_counts) from wds_manifest (60s TTL)."""
        now = time.time()
        if now - _wds_full_cache[0] < _DASHBOARD_FILE_CACHE_TTL and _wds_full_cache[1] is not None:
            return _wds_full_cache[1], _wds_full_cache[2]
        u, s = _parse_wds_manifest(wds_manifest)
        _wds_full_cache[0] = now
        _wds_full_cache[1] = u
        _wds_full_cache[2] = s
        return u, s

    def _start_s3_refresh() -> None:
        """Start at most one asynchronous ECS listing for all dashboard clients."""
        nonlocal _s3_refreshing
        with _s3_refresh_lock:
            if _s3_refreshing:
                return
            _s3_refreshing = True

        def _refresh() -> None:
            nonlocal _s3_refreshing
            try:
                s3_stats(tcga_watcher, "wds")
            finally:
                with _s3_refresh_lock:
                    _s3_refreshing = False

        threading.Thread(target=_refresh, daemon=True, name="s3-refresh").start()

    _sync_status_cache: list = [0.0, None]  # [last_read_time, payload|None]
    _dispatcher_status_snapshot_cache: list = [0.0, None]  # [last_read_time, payload|None]
    _log_response_cache: dict[str, dict[str, object]] = {}
    _log_cache_lock = threading.Lock()
    _LOG_CACHE_TTL = 15.0
    _LOG_STALE_FALLBACK_TTL = 5 * 60.0
    _LOG_FAILURE_BACKOFF_MAX = 120.0

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

    def _parse_json_or_none(raw: str | None) -> dict | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    _state_store = None
    if db_cfg.backend == "postgres":
        try:
            _state_store = StateStore.from_config(cfg)
        except Exception as exc:
            log.warning("dashboard Postgres state init failed for remote target health: %s", exc)

    _active_batch_cache: list = [0.0, []]
    _active_batch_cache_lock = threading.Lock()
    _ACTIVE_BATCH_CACHE_TTL = 15.0

    def _read_active_batches_for_dashboard() -> list[dict]:
        """Add active rows omitted by the bounded recent-batch snapshot."""
        if _state_store is None:
            return []
        now = time.time()
        with _active_batch_cache_lock:
            if now - _active_batch_cache[0] < _ACTIVE_BATCH_CACHE_TTL:
                return [dict(row) for row in _active_batch_cache[1]]
            try:
                rows = [dict(row) for row in _state_store.get_active_batches()]
                hooks = _state_store.get_batch_hooks(
                    [str(row.get("batch_id")) for row in rows if row.get("batch_id")]
                )
                for row in rows:
                    row["hooks"] = hooks.get(str(row.get("batch_id")), [])
                _active_batch_cache[0] = now
                _active_batch_cache[1] = rows
                return [dict(row) for row in rows]
            except Exception as exc:
                log.warning("dashboard active batch refresh failed: %s", exc)
                return [dict(row) for row in _active_batch_cache[1]]

    def _release_sync_health() -> dict:
        if _state_store is None:
            return {
                "deployment_id": cfg.release_deployment_id,
                "managed": cfg.auto_sync_release,
                "state": "UNMANAGED",
                "targets": [],
                "blocked_targets": [],
            }
        try:
            return _state_store.release_sync_snapshot()
        except Exception as exc:
            log.warning("dashboard release sync refresh failed: %s", exc)
            return {
                "deployment_id": cfg.release_deployment_id,
                "managed": cfg.auto_sync_release,
                "state": "ERROR",
                "message": str(exc),
                "targets": [],
                "blocked_targets": [],
            }

    def _remote_target_health(release_sync: dict | None = None) -> list[dict]:
        targets = []
        now = time.time()
        release_targets = {
            str(item.get("name") or ""): item
            for item in ((release_sync or {}).get("targets") or [])
        }
        for target in cfg.dispatch_targets:
            remote_batches = []
            agent = None
            if _state_store is not None:
                try:
                    agent = _state_store.read_remote_agent(target.name)
                except Exception:
                    agent = None
                try:
                    remote_batches = _state_store.get_remote_batches_for_target(target.name)
                except Exception:
                    remote_batches = []
            heartbeat = (agent or {}).get("heartbeat_at")
            age_s = None
            state = "unknown"
            if heartbeat:
                try:
                    age_s = max(0, int(now - datetime.fromisoformat(heartbeat).timestamp()))
                    state = "alive" if age_s <= target.agent_heartbeat_timeout_seconds else "stale"
                except Exception:
                    state = "unknown"
            queue_depth = sum(
                1 for item in remote_batches if item.get("status") == "QUEUED"
            )
            remote_batch_ids = [str(item.get("batch_id") or "") for item in remote_batches if item.get("batch_id")]
            remote_batch_ids = [batch_id for batch_id in remote_batch_ids if batch_id]
            slurm = _remote_target_slurm_stats(target, remote_batch_ids, agent=agent)
            hook_circuit = {}
            if _state_store is not None:
                try:
                    hook_circuit = _state_store.hook_circuit_status(target.name) or {}
                except Exception:
                    hook_circuit = {}
            targets.append({
                "name": target.name,
                "backend": target.backend,
                "login_host": target.login_host,
                "state": state,
                "agent_heartbeat_at": heartbeat,
                "agent_heartbeat_age_s": age_s,
                "queue_depth": queue_depth,
                "active_batches": len(remote_batch_ids),
                "slurm_running": slurm.get("running"),
                "slurm_pending": slurm.get("pending"),
                "slurm_completing": slurm.get("completing"),
                "slurm_total": slurm.get("total"),
                "slurm_nodes": slurm.get("nodes", []),
                "slurm_error": slurm.get("error"),
                "hook_circuit": hook_circuit,
                "release": dict(release_targets.get(target.name) or {}),
                "status_source": "postgres",
            })
        return targets

    def _remote_target_slurm_stats(target, batch_ids: list[str], *, agent: dict | None = None) -> dict:
        agent_slurm = (agent or {}).get("slurm")
        if isinstance(agent_slurm, dict):
            return {
                "running": int(agent_slurm.get("running", 0) or 0),
                "pending": int(agent_slurm.get("pending", 0) or 0),
                "completing": int(agent_slurm.get("completing", 0) or 0),
                "total": int(agent_slurm.get("total", 0) or 0),
                "nodes": list(agent_slurm.get("nodes") or []),
                "error": agent_slurm.get("error"),
            }
        release_sync = _release_sync_health()
        return {
            "running": 0,
            "pending": 0,
            "completing": 0,
            "total": 0,
            "nodes": [],
            "error": "agent Slurm snapshot unavailable",
        }

    def _read_dispatcher_status_snapshot() -> dict | None:
        now = time.time()
        if now - _dispatcher_status_snapshot_cache[0] < 5:
            return _dispatcher_status_snapshot_cache[1]
        payload = None
        if os.path.exists(dispatcher_status_snapshot_path):
            try:
                payload = json.loads(Path(dispatcher_status_snapshot_path).read_text())
            except Exception:
                payload = None
        _dispatcher_status_snapshot_cache[0] = now
        _dispatcher_status_snapshot_cache[1] = payload
        return payload

    def _snapshot_batch_log_path(batch_id: str) -> str | None:
        snapshot = _read_dispatcher_status_snapshot()
        if not snapshot:
            return None
        for item in snapshot.get("recent_batches") or []:
            if item.get("batch_id") == batch_id and item.get("log_path"):
                return str(item["log_path"])
        return None

    # Persistent registry: Tower state survives dashboard restarts.
    _run_store = _RunStore(os.path.join(cfg.state_dir, "turret_runs.db"))
    _registry  = _PersistentRegistry(_run_store)
    # Seed registry from legacy tasks_done/total/failed columns for batches
    # that completed before the PersistentWorkflowRegistry was introduced.
    _seed_registry_from_db(_registry, cfg)

    _active_run_name_cache: list = [0.0, {}]
    _ACTIVE_RUN_NAME_CACHE_TTL = 15.0

    def _run_name_to_batch_id(run_name: str) -> str:
        """Map NF run name → dispatcher batch_id.

        Supports three formats:
          - Legacy: ``dispatcher_{batch_id}``      (pre-fix, full 35-char name)
          - Modern: ``dispatcher_r{hash8}``        (r-prefixed 8-char UUID suffix)
          - Short:  ``r{hash8}`` or ``{hash8}``   (bare hash, no dispatcher_ prefix)
        """
        active_runs = _active_run_name_map()
        if run_name.startswith("dispatcher_"):
            suffix = run_name[len("dispatcher_"):]
            if mapped := active_runs.get(suffix):
                return mapped
            m = _re.fullmatch(r"r([0-9a-f]{8})", suffix)
            if m and (mapped := _lookup_batch_id_by_suffix(m.group(1))):
                return mapped
            return suffix
        # Bare hash (with or without r-prefix): look up batch whose batch_id ends with _{hash}
        run = run_name.lstrip("r") if len(run_name) == 9 and run_name.startswith("r") else run_name
        if mapped := active_runs.get(run_name):
            return mapped
        if mapped := _lookup_batch_id_by_suffix(run):
            return mapped
        return run_name

    def _tower_state_for_batch(batch_id: str, *, active_runs: dict[str, str] | None = None) -> dict | None:
        if active_runs is None:
            active_runs = _active_run_name_map()
        return newest_tower_state_for_batch(
            batch_id, _registry.get_all(), active_runs=active_runs
        )

    # Single TowerRouter instance shared across all requests in this handler.
    _router = _TowerRouter(registry=_registry, run_name_to_batch_id=_run_name_to_batch_id)

    _db_lock = threading.Lock()
    _db_shared_conn = None

    @contextmanager
    def _db():
        nonlocal _db_shared_conn
        with _db_lock:
            try:
                if _db_shared_conn is None or getattr(_db_shared_conn, "closed", False):
                    _db_shared_conn = connect_database(db_cfg, readonly=True)
                yield _db_shared_conn
            except Exception:
                try:
                    if _db_shared_conn is not None:
                        _db_shared_conn.close()
                except Exception:
                    pass
                _db_shared_conn = None
                raise

    def _close_db_shared_conn() -> None:
        nonlocal _db_shared_conn
        with _db_lock:
            if _db_shared_conn is None:
                return
            try:
                _db_shared_conn.close()
            except Exception:
                pass
            _db_shared_conn = None

    def _lookup_batch_id_by_suffix(run_suffix: str) -> str | None:
        try:
            with _db() as conn:
                row = conn.execute(
                    sql_for_backend(db_cfg, "SELECT batch_id FROM batches WHERE batch_id LIKE ?"),
                    (f"%_{run_suffix}",)
                ).fetchone()
                if row:
                    return row["batch_id"] if isinstance(row, dict) else row[0]
        except Exception:
            pass
        return None

    def _active_run_name_map() -> dict[str, str]:
        now = time.time()
        if now - _active_run_name_cache[0] < _ACTIVE_RUN_NAME_CACHE_TTL:
            return _active_run_name_cache[1]
        try:
            with _db() as conn:
                cols = _table_columns(conn, db_cfg, "batches")
                run_name_expr = "run_name" if "run_name" in cols else "NULL AS run_name"
                backend_col = "backend" if "backend" in cols else "NULL"
                backend_expr = f"{backend_col} AS backend"
                remote_status_col = "remote_status" if "remote_status" in cols else "NULL"
                remote_status_expr = f"{remote_status_col} AS remote_status"
                dispatched_col = "dispatched_at" if "dispatched_at" in cols else "NULL"
                rows = conn.execute(
                    f"""
                    SELECT batch_id, {run_name_expr}, {backend_expr}, {remote_status_expr}
                    FROM batches
                    WHERE status IN ('RUNNING', 'RECOVERING', 'FINALIZING')
                       OR (
                           {backend_col}='remote_agent'
                           AND COALESCE({remote_status_col}, '') NOT IN ('', 'SUCCEEDED', 'FAILED', 'CANCELLED')
                       )
                    ORDER BY {dispatched_col} DESC
                    """
                ).fetchall()
        except Exception:
            return {}

        mapping = {
            str(row["run_name"]): str(row["batch_id"])
            for row in rows
            if row["run_name"]
        }
        for row in rows:
            if row["run_name"]:
                continue
            batch_id = str(row["batch_id"])
            nf_log = os.path.join(cfg.log_dir, f"batch_{batch_id}.nf.log")
            if not os.path.exists(nf_log):
                continue
            try:
                with open(nf_log, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    fh.seek(max(0, fh.tell() - _DASHBOARD_LOG_TAIL_BYTES))
                    text = fh.read().decode(errors="replace")
            except OSError:
                continue
            match = _re.search(r"Run name:\s+(\S+)", text)
            if not match:
                match = _re.search(r"-name\s+(\S+)", text)
            if match:
                mapping[match.group(1)] = batch_id
        _active_run_name_cache[0] = now
        _active_run_name_cache[1] = mapping
        return mapping

    _status_cache: dict[str, object] = {
        "payload": None,
        "updated_at": 0.0,
        "refreshing": False,
        "last_error": None,
    }
    _STATUS_CACHE_TTL = 30.0
    _STATUS_STALE_FALLBACK_TTL = 15 * 60.0
    _status_cache_lock = threading.Lock()

    def _compute_status_payload():
        with _db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM slides GROUP BY status"
            ).fetchall()
            db_total_unique = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt FROM slides"
            ).fetchone(), "cnt", 0) or 0
            succeeded_unique = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt FROM slides WHERE status='SUCCEEDED'"
            ).fetchone(), "cnt", 0) or 0
            retry_pending = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt "
                "FROM slides WHERE status='PENDING' AND fail_count > 0"
            ).fetchone(), "cnt", 0) or 0
            terminal_failed = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt "
                "FROM slides WHERE status='FAILED'"
            ).fetchone(), "cnt", 0) or 0
            pending_total = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt "
                "FROM slides WHERE status='PENDING'"
            ).fetchone(), "cnt", 0) or 0
            dispatched_total = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt "
                "FROM slides WHERE status='DISPATCHED'"
            ).fetchone(), "cnt", 0) or 0
            running_rows = conn.execute(
                """
                SELECT batch_id, status, backend, target, remote_status
                  FROM batches
                 WHERE status IN ('RUNNING','RECOVERING','FINALIZING')
                    OR (
                        backend='remote_agent'
                        AND COALESCE(remote_status, '') IN (
                            'QUEUED', 'CLAIMED', 'STARTING', 'SUBMITTED', 'RUNNING',
                            'CANCEL_REQUESTED', 'FINALIZATION_QUEUED', 'HOOKS_RETRY',
                              'HOOKS_RUNNING', 'HOOKS_BLOCKED', 'STALE'
                        )
                    )
                """
            ).fetchall()
            n_blacklisted = _row_val(conn.execute(
                f"SELECT COUNT(*) AS cnt FROM slides "
                f"WHERE status='FAILED' AND fail_count >= {cfg.max_slide_retries}"
            ).fetchone(), "cnt", 0) or 0

        running_counts = _running_batch_breakdown([dict(row) for row in running_rows])

        with _db() as conn:
            # _compute_status_payload() is reachable only for SQLite: Postgres
            # status always comes from the dispatcher snapshot.
            tp = conn.execute(
                """
                SELECT COUNT(*) AS cnt,
                       MIN(completed_at) AS oldest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND julianday(completed_at) >= julianday('now') - (6.0 / 24.0)
                """
            ).fetchone()
            tp_recent = conn.execute(
                """
                SELECT COUNT(*) AS cnt,
                       MIN(completed_at) AS oldest
                FROM slides
                WHERE status = 'SUCCEEDED'
                  AND completed_at IS NOT NULL
                  AND julianday(completed_at) >= julianday('now') - (0.5 / 24.0)
                """
            ).fetchone()
            remaining = _row_val(conn.execute(
                "SELECT COUNT(DISTINCT slide_id) AS cnt "
                "FROM slides WHERE status IN ('PENDING','DISPATCHED')"
            ).fetchone(), "cnt", 0) or 0
        counts = {r["status"]: r["n"] for r in rows}
        # SKIPPED rows are extra paths of a slide_id counted via its twin.
        db_total = sum(n for status, n in counts.items() if status != "SKIPPED")
        succeeded = counts.get("SUCCEEDED", 0)

        # Use inventory total as the true denominator (all slides to eventually process).
        # Fall back to db_total if inventory is unavailable.
        true_total = (_inventory_total * _n_models) if _inventory_total else db_total_unique

        # Use WDS manifest counts as the authoritative "done" tally — more reliable
        # than DB SUCCEEDED (which can be reset to PENDING by _verify_wds_coverage).
        wds_counts = _wds_done_counts()
        # "done" = slides confirmed in WDS for EVERY required model (min across models).
        # Cap at _inventory_total: WDS manifests accumulate across all historical runs,
        # so raw counts can exceed the current inventory and produce misleading >100% pct.
        wds_done_raw = min(wds_counts.get(m, 0) for m in tcga_watcher.wds_destinations) if (
            tcga_watcher and tcga_watcher.wds_destinations and wds_counts
        ) else succeeded
        denom = _inventory_total or db_total_unique
        wds_done = min(wds_done_raw, denom)

        # pct_wds: slides confirmed in WDS vs inventory
        pct_wds = round(wds_done / denom * 100, 1) if denom else 0

        # pct_succeeded: DB SUCCEEDED vs inventory — reflects features-extracted progress
        # independently of WDS upload, and is never inflated by historical WDS entries.
        pct_succeeded = round(min(succeeded_unique, denom) / denom * 100, 1) if denom else 0
        # Add undispatched slides to remaining count
        undispatched = max(0, _inventory_total - db_total) if _inventory_total else 0
        total_remaining = remaining + undispatched

        completed_in_window = _row_val(tp, "cnt", 0) or 0
        throughput_stats = estimate_throughput_and_eta(
            completed_in_window=completed_in_window,
            oldest_completed_at=_row_val(tp, "oldest"),
            remaining=total_remaining,
            window_hours=6.0,
            recent_completed=_row_val(tp_recent, "cnt", 0) or 0,
            recent_oldest_completed_at=_row_val(tp_recent, "oldest"),
        )

        return {
            "counts": counts,
            "chart_counts": {
                "SUCCEEDED": succeeded_unique,
                "RETRY_PENDING": retry_pending,
                "PENDING": max(0, pending_total - retry_pending),
                "DISPATCHED": dispatched_total,
                "FAILED": terminal_failed,
            },
            "total": true_total,
            "wds_done": wds_done,
            "pct_done": pct_wds,
            "pct_succeeded": pct_succeeded,
            "running_batches": running_counts["running_batches_total"],
            "running_batches_total": running_counts["running_batches_total"],
            "running_batches_local": running_counts["running_batches_local"],
            "running_batches_remote": running_counts["running_batches_remote"],
            "blacklisted": n_blacklisted,
            "retry_pending": retry_pending,
            "throughput_per_hour": throughput_stats["throughput_per_hour"],
            "eta_seconds": throughput_stats["eta_seconds"],
            "remaining": total_remaining,
            "completed_in_window": completed_in_window,
            "dispatcher_health": _read_dispatcher_health(
                dispatcher_pid_path, dispatcher_log_path, config_path=cfg.config_path
            ),
            "sync_status": _read_sync_status(),
            "release_sync": release_sync,
            "remote_targets": _remote_target_health(release_sync),
        }

    def _refresh_status_cache():
        try:
            payload = _compute_status_payload()
            now = time.time()
            with _status_cache_lock:
                _status_cache["payload"] = payload
                _status_cache["updated_at"] = now
                _status_cache["last_error"] = None
        except Exception as exc:
            log.exception("dashboard /api/status refresh failed: %s", exc)
            with _status_cache_lock:
                _status_cache["last_error"] = str(exc)
        finally:
            with _status_cache_lock:
                _status_cache["refreshing"] = False

    def _ensure_status_refresh():
        with _status_cache_lock:
            if _status_cache["refreshing"]:
                return
            _status_cache["refreshing"] = True
        threading.Thread(
            target=_refresh_status_cache,
            name="dashboard-status-refresh",
            daemon=True,
        ).start()

    def _api_status():
        now = time.time()
        snapshot = _read_dispatcher_status_snapshot()
        if snapshot:
            generated_at = snapshot.get("generated_at")
            age_seconds = None
            if generated_at:
                try:
                    age_seconds = max(
                        0,
                        int(
                            (
                                datetime.now().astimezone()
                                - datetime.fromisoformat(str(generated_at).replace("Z", "+00:00")).astimezone()
                            ).total_seconds()
                        ),
                    )
                except Exception:
                    age_seconds = None
            snapshot_stale = bool(age_seconds is not None and age_seconds > 120)
            response = dict(snapshot)
            response["dispatcher_health"] = _read_dispatcher_health(
                dispatcher_pid_path, dispatcher_log_path, config_path=cfg.config_path
            )
            try:
                response["recent_batches"] = _status_recent_batches(snapshot)
            except Exception as exc:
                response["recent_batches"] = list(snapshot.get("recent_batches") or [])
                response["recent_batches_error"] = {
                    "type": type(exc).__name__,
                    "repr": repr(exc),
                    "str": str(exc),
                }
            response["sync_status"] = _read_sync_status()
            # Remote health is part of the dispatcher-owned snapshot. Avoid
            # opening fresh database sessions for every browser status poll.
            response["remote_targets"] = list(snapshot.get("remote_targets") or [])
            response["status_cache"] = {
                "stale": snapshot_stale,
                "age_seconds": age_seconds,
                "error": None,
                "source": "dispatcher_status_snapshot",
            }
            return response

        if db_cfg.backend == "postgres":
            release_sync = _release_sync_health()
            return {
                "counts": {},
                "chart_counts": {},
                "total": 0,
                "wds_done": 0,
                "pct_done": 0,
                "pct_succeeded": 0,
                "running_batches": 0,
                "running_batches_total": 0,
                "running_batches_local": 0,
                "running_batches_remote": 0,
                "blacklisted": 0,
                "retry_pending": 0,
                "throughput_per_hour": None,
                "eta_seconds": None,
                "remaining": 0,
                "completed_in_window": 0,
                "dispatcher_health": _read_dispatcher_health(
                    dispatcher_pid_path, dispatcher_log_path, config_path=cfg.config_path
                ),
                "sync_status": _read_sync_status(),
                "release_sync": release_sync,
                "remote_targets": _remote_target_health(release_sync),
                "status_cache": {
                    "stale": True,
                    "age_seconds": None,
                    "error": "dispatcher status snapshot unavailable",
                    "source": "degraded_no_snapshot",
                },
            }

        with _status_cache_lock:
            payload = _status_cache["payload"]
            updated_at = float(_status_cache["updated_at"])
            last_error = _status_cache["last_error"]

        cache_age = now - updated_at if updated_at else None
        if payload is None:
            _ensure_status_refresh()
            return {
                "counts": {},
                "chart_counts": {},
                "total": 0,
                "wds_done": 0,
                "pct_done": 0,
                "pct_succeeded": 0,
                "running_batches": 0,
                "running_batches_total": 0,
                "running_batches_local": 0,
                "running_batches_remote": 0,
                "blacklisted": 0,
                "retry_pending": 0,
                "throughput_per_hour": None,
                "eta_seconds": None,
                "remaining": 0,
                "completed_in_window": 0,
                "dispatcher_health": _read_dispatcher_health(
                    dispatcher_pid_path, dispatcher_log_path, config_path=cfg.config_path
                ),
                "sync_status": _read_sync_status(),
                "remote_targets": [],
                "status_cache": {
                    "stale": True,
                    "age_seconds": None,
                    "error": last_error,
                },
            }

        if cache_age is not None and cache_age > _STATUS_CACHE_TTL:
            _ensure_status_refresh()

        response = dict(payload)
        response["status_cache"] = {
            "stale": cache_age is not None and cache_age > _STATUS_CACHE_TTL,
            "age_seconds": int(cache_age) if cache_age is not None else None,
            "error": last_error,
        }

        if cache_age is not None and cache_age > _STATUS_STALE_FALLBACK_TTL and last_error:
            log.warning(
                "dashboard /api/status serving stale cache age=%ss after repeated refresh errors: %s",
                int(cache_age), last_error,
            )
        return response

    _REMOTE_ACTIVE_STATUSES = {
        "QUEUED", "CLAIMED", "STARTING", "SUBMITTED", "RUNNING",
        "CANCEL_REQUESTED", "FINALIZATION_QUEUED", "HOOKS_RETRY",
        "HOOKS_RUNNING", "HOOKS_BLOCKED", "BLOCKED", "STALE",
    }

    def _status_recent_batches(snapshot: dict) -> list[dict]:
        """Return recent rows plus active rows outside the bounded window."""
        items = list(snapshot.get("recent_batches") or [])
        seen_ids = {str(row.get("batch_id") or "") for row in items}
        active_rows = [
            row for row in _read_active_batches_for_dashboard()
            if str(row.get("batch_id") or "") not in seen_ids
        ]
        items.extend(active_rows)

        # Active rows outside the dispatcher's bounded snapshot do not carry
        # its Tower enrichment. Fill those rows from the persistent Turret
        # registry so long-running local Slurm batches remain visible.
        missing_tower = [
            row for row in active_rows
            if _tower_is_live_for_batch(row) and not row.get("tower")
        ]
        if missing_tower:
            active_runs = _active_run_name_map()
            for row in missing_tower:
                row["tower_expected"] = True
                row["tower"] = _tower_state_for_batch(
                    row["batch_id"], active_runs=active_runs
                )
        return items

    _ACTIVE_BATCH_STATUSES = {"RUNNING", "RECOVERING", "FINALIZING"}
    _REMOTE_TERMINAL_STATUS_MAP = {
        "SUCCEEDED": "SUCCEEDED",
        "FAILED": "FAILED",
        "CANCELLED": "FAILED",
    }

    def _batch_is_active(row: dict) -> bool:
        status = str(row.get("status") or "")
        if str(row.get("backend") or "") == "remote_agent":
            remote_status = str(row.get("remote_status") or "")
            if remote_status in _REMOTE_TERMINAL_STATUS_MAP:
                return False
            if remote_status in _REMOTE_ACTIVE_STATUSES:
                return True
        if status in _ACTIVE_BATCH_STATUSES:
            return True
        remote_status = str(row.get("remote_status") or "")
        return remote_status in _REMOTE_ACTIVE_STATUSES

    def _running_batch_breakdown(rows: list[dict]) -> dict[str, int]:
        local = 0
        remote = 0
        for row in rows:
            if not _batch_is_active(row):
                continue
            if str(row.get("backend") or "") == "remote_agent":
                remote += 1
            else:
                local += 1
        total = local + remote
        return {
            "running_batches_total": total,
            "running_batches_local": local,
            "running_batches_remote": remote,
        }

    def _display_batch_status(row: dict) -> str:
        status = str(row.get("status") or "")
        if str(row.get("backend") or "") != "remote_agent":
            return status
        remote_status = str(row.get("remote_status") or "")
        if remote_status in _REMOTE_TERMINAL_STATUS_MAP:
            return _REMOTE_TERMINAL_STATUS_MAP[remote_status]
        if remote_status in _REMOTE_ACTIVE_STATUSES:
            return "RUNNING"
        if status in _ACTIVE_BATCH_STATUSES:
            return status
        return status

    def _augment_batch_rows(rows: list[dict], *, allow_live_tower_lookup: bool = True) -> list[dict]:
        slurm = slurm_stats()
        jobs_by_batch = slurm.get("jobs_by_batch", {})
        active_runs: dict[str, str] | None = None
        if allow_live_tower_lookup and any(
            (_tower_is_live_for_batch(row) and not row.get("tower"))
            for row in rows
        ):
            active_runs = _active_run_name_map()
        result = []
        for r in rows:
            start = r.get("dispatched_at") or ""
            end = r.get("completed_at") or ""
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
            tower = r.get("tower")
            tower_expected = _tower_is_live_for_batch(r)
            if tower is None and allow_live_tower_lookup and tower_expected:
                tower = _tower_state_for_batch(r["batch_id"], active_runs=active_runs)
            if not tower_expected:
                tower = None
            elif tower:
                tower = normalize_tower_state(tower)
            if tower and tower.get("processes") and slurm_batch.get("processes"):
                slurm_procs = slurm_batch["processes"]
                for proc in tower["processes"]:
                    slurm_key = tower_process_to_slurm_name(
                        proc.get("process", proc.get("name", ""))
                    )
                    sp = slurm_procs.get(slurm_key) or slurm_procs.get(slurm_key + "_")
                    if sp:
                        proc["slurm_running"] = sp["running"]
                        proc["slurm_nodes"] = sp["nodes"]
                        elapsed = sp.get("elapsed_s", [])
                        proc["slurm_elapsed_max"] = max(elapsed) if elapsed else None
            # An exit code is only meaningful after a batch reaches a terminal
            # state.  Older records may contain the historical -1 running
            # sentinel; never expose that sentinel as a live failure.
            display_nextflow_exit = None if _batch_is_active(r) else r.get("nextflow_exit")
            result.append({
                "batch_id": r["batch_id"],
                "status": _display_batch_status(r),
                "backend": r.get("backend"),
                "target": r.get("target"),
                "slide_count": r.get("slide_count"),
                "dispatched_at": start,
                "completed_at": end,
                "duration_s": duration,
                "nextflow_exit": display_nextflow_exit,
                "has_log": bool(r.get("log_path") and os.path.exists(r["log_path"])),
                "remote_status": r.get("remote_status"),
                "remote_heartbeat_at": r.get("remote_heartbeat_at"),
                "remote_driver_job_id": r.get("remote_driver_job_id"),
                "remote_work_dir": r.get("remote_work_dir"),
                "remote_outdir": r.get("remote_outdir"),
                "remote_log_path": r.get("remote_log_path"),
                "remote_payload": _parse_json_or_none(r.get("remote_payload")),
                "tower": tower,
                "tower_expected": tower_expected,
                "slurm_running": slurm_batch.get("running"),
                "slurm_pending": slurm_batch.get("pending"),
                "slurm_nodes": slurm_batch.get("nodes", []),
                "hooks": r.get("hooks", []),
            })
        return result

    def _api_batches():
        snapshot = _read_dispatcher_status_snapshot()
        if not snapshot or "recent_batches" not in snapshot:
            return []
        try:
            return _augment_batch_rows(
                _status_recent_batches(snapshot),
                allow_live_tower_lookup=db_cfg.backend != "postgres",
            )
        except Exception:
            return []

    def _next_log_retry_delay(current_delay_seconds: float, *, success: bool) -> float:
        if success:
            return _LOG_CACHE_TTL
        return min(_LOG_FAILURE_BACKOFF_MAX, max(_LOG_CACHE_TTL, current_delay_seconds * 2.0))

    def _read_log_payload(batch_id: str) -> dict:
        with _db() as conn:
            cols = _table_columns(conn, db_cfg, "batches")
            backend_col = "backend" if "backend" in cols else "NULL AS backend"
            target_col = "target" if "target" in cols else "NULL AS target"
            remote_log_col = "remote_log_path" if "remote_log_path" in cols else "NULL AS remote_log_path"
            remote_payload_col = "remote_payload" if "remote_payload" in cols else "NULL AS remote_payload"
            row = conn.execute(
                sql_for_backend(
                    db_cfg,
                    f"SELECT log_path, {backend_col}, {target_col}, {remote_log_col}, "
                    f"{remote_payload_col} FROM batches WHERE batch_id=?",
                ),
                (batch_id,),
            ).fetchone()
        if not row:
            snapshot_log_path = _snapshot_batch_log_path(batch_id)
            if not snapshot_log_path:
                raise FileNotFoundError("no log path")
            row = {
                "log_path": snapshot_log_path,
                "backend": None,
                "target": None,
                "remote_log_path": None,
                "remote_payload": None,
            }
        if row["backend"] == "remote_agent":
            remote_lines = []
            remote_log_path = str(row["remote_log_path"] or "").strip()
            if remote_log_path:
                base = Path(remote_log_path)
                for name in ("driver.out", "driver.err", "nextflow.log"):
                    p = base if name == "driver.out" else base.parent / name
                    if p.exists():
                        try:
                            remote_lines.append(f"--- {name} ---")
                            remote_lines.extend([ln.rstrip() for ln in p.read_text(errors="replace").splitlines()[-80:]])
                        except Exception:
                            pass
            try:
                raw_status = row["remote_payload"]
                status = raw_status if isinstance(raw_status, dict) else json.loads(raw_status or "{}")
            except Exception:
                status = {}
            if remote_lines:
                return {
                    "lines": remote_lines,
                    "remote_status": status,
                    "nf_log": [],
                    "stale": False,
                    "source": "live",
                    "log_updated_at": None,
                    "error": None,
                }
            mirrored_lines: list[str] = []
            log_tail = status.get("log_tail") if isinstance(status, dict) else None
            if isinstance(log_tail, dict):
                for log_file in log_tail.get("files") or []:
                    if not isinstance(log_file, dict) or not log_file.get("text"):
                        continue
                    mirrored_lines.append(f"--- {log_file.get('name') or 'remote log'} ---")
                    if log_file.get("truncated"):
                        mirrored_lines.append("[earlier output truncated]")
                    mirrored_lines.extend(str(log_file["text"]).splitlines())
            if mirrored_lines:
                return {
                    "lines": mirrored_lines,
                    "remote_status": status,
                    "nf_log": [],
                    "stale": False,
                    "source": "remote_agent_mirror",
                    "log_updated_at": log_tail.get("captured_at"),
                    "error": None,
                }
            remote_state = str(status.get("status") or "unknown")
            path_hint = remote_log_path or "not reported"
            return {
                "lines": [
                    f"Remote log output is not available yet (status={remote_state}, path={path_hint})."
                ],
                "remote_status": status,
                "nf_log": [],
                "stale": False,
                "source": "remote_agent",
                "log_updated_at": None,
                "error": None,
            }

        log_path = _snapshot_batch_log_path(batch_id) or row["log_path"]
        if not log_path:
            raise FileNotFoundError("no log path")
        if not os.path.exists(log_path):
            raise FileNotFoundError("log file not found")
        try:
            ansi = _re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfnsuhl]|\r')
            with open(log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - _DASHBOARD_LOG_TAIL_BYTES))
                content = fh.read().decode(errors="replace")
            content = ansi.sub('', content)
            last = content.rfind("executor >")
            block = content[last:].strip() if last >= 0 else content.strip()
            lines = block.splitlines()
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        nf_lines: list = []
        if os.path.exists(nf_log_path):
            try:
                with open(nf_log_path, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    fh.seek(max(0, size - _DASHBOARD_LOG_TAIL_BYTES))
                    nf_lines = [ln.rstrip() for ln in fh.read().decode(errors="replace").splitlines()[-40:]]
            except Exception:
                pass
        return {
            "lines": lines,
            "nf_log": nf_lines,
            "stale": False,
            "source": "live",
            "error": None,
        }

    def _api_logs(batch_id: str):
        now = time.time()
        with _log_cache_lock:
            entry = _log_response_cache.get(batch_id)
            if entry:
                cached_at = float(entry.get("updated_at", 0.0) or 0.0)
                cached_payload = entry.get("payload")
                next_retry_at = float(entry.get("next_retry_at", 0.0) or 0.0)
                if cached_payload is not None and now - cached_at < _LOG_CACHE_TTL:
                    return dict(cached_payload), None
                if cached_payload is not None and now < next_retry_at:
                    stale_payload = dict(cached_payload)
                    stale_payload["stale"] = True
                    stale_payload["source"] = "log_cache"
                    stale_payload["error"] = entry.get("last_error")
                    return stale_payload, None
        try:
            payload = _read_log_payload(batch_id)
        except Exception as exc:
            error = str(exc)
            with _log_cache_lock:
                entry = _log_response_cache.get(batch_id) or {}
                previous_backoff = float(entry.get("failure_backoff_s", _LOG_CACHE_TTL) or _LOG_CACHE_TTL)
                next_backoff = _next_log_retry_delay(previous_backoff, success=False)
                cached_at = float(entry.get("updated_at", 0.0) or 0.0)
                cached_payload = entry.get("payload")
                _log_response_cache[batch_id] = {
                    "updated_at": cached_at,
                    "payload": cached_payload,
                    "last_error": error,
                    "next_retry_at": now + next_backoff,
                    "failure_backoff_s": next_backoff,
                }
                if cached_payload is not None and now - cached_at < _LOG_STALE_FALLBACK_TTL:
                    stale_payload = dict(cached_payload)
                    stale_payload["stale"] = True
                    stale_payload["source"] = "log_cache"
                    stale_payload["error"] = error
                    return stale_payload, None
            return None, error
        with _log_cache_lock:
            _log_response_cache[batch_id] = {
                "updated_at": now,
                "payload": payload,
                "last_error": None,
                "next_retry_at": 0.0,
                "failure_backoff_s": _next_log_retry_delay(_LOG_CACHE_TTL, success=True),
            }
        return payload, None

    def _api_wds():
        snapshot = _read_dispatcher_status_snapshot()
        if not snapshot:
            payload = _empty_wds_payload()
            payload["stale"] = True
            payload["source"] = "degraded_no_snapshot"
            payload["error"] = "dispatcher status snapshot unavailable"
            return payload
        payload = dict(snapshot.get("wds") or _empty_wds_payload())
        counts = snapshot.get("counts") or {}
        db_succeeded = int(counts.get("SUCCEEDED", 0) or 0) if isinstance(counts, dict) else 0
        db_total = sum(
            int(v or 0) for status, v in counts.items() if status != "SKIPPED"
        ) if isinstance(counts, dict) else 0
        for model_data in (payload.get("models") or {}).values():
            model_data["gap"] = max(0, db_succeeded - int((model_data or {}).get("slides", 0) or 0))
        payload["db_succeeded"] = db_succeeded
        payload["db_total"] = db_total
        payload["stale"] = bool(payload.get("stale") or snapshot.get("snapshot_error"))
        payload["source"] = payload.get("source") or "dispatcher_status_snapshot"
        payload["snapshot_error"] = snapshot.get("snapshot_error")
        return payload

    class Handler(_TowerHandlerMixin, BaseHTTPRequestHandler):
        # Expose registry and router so tests and external callers can inspect Tower state.
        tower_registry = _registry
        tower_router   = _router
        wds_cache_state = _wds_response_cache
        log_cache_state = _log_response_cache

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
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
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
    max_requests = int(os.environ.get("MUSSEL_DASHBOARD_MAX_REQUESTS", "16"))
    server = BoundedThreadingHTTPServer(
        (args.host, args.port), handler_cls, max_requests=max(1, max_requests)
    )
    print(f"Dashboard: http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
