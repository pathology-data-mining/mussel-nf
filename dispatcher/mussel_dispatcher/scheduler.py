"""BatchScheduler, RunManager, and main() for mussel-dispatcher."""
from __future__ import annotations

import logging
import logging.handlers
import os
import csv
import copy
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import Config
from .state import StateStore
from .watchers import LocalWatcher, S3Watcher, DatabricksWatcher, TcgaWatcher
from .runner import NextflowRunner
from .batch_lifecycle import FinalizationWorkerPool, collect_manifests
from .release import ReleaseController
from .storage import (
    CapacityGuard,
    ExecutionCleanupSpec,
    reconcile_execution_cleanup,
)

log = logging.getLogger("mussel-dispatcher")

_SLURM_JOB_ID_RE = re.compile(r"\bjobId:\s*(\d+)\b")
_SLURM_PROCESSES = (
    "MUSSEL:EXTRACT_FEATURES:TESSELLATE",
    "MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH",
)
_STARTUP_PRIORITY_SYNC_MAX_PENDING = 50000
_DB_SWEEP_MAX_PENDING = 2000
_STARTUP_REQUEUE_MAX_PENDING = 5000


import socket as _socket


def _port_in_use(port: int) -> bool:
    """Return True if *port* is already bound on localhost."""
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _listener_pids(port: int) -> list[int]:
    """Return listening PIDs for a localhost TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _cmdline_for_pid(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_text().replace("\x00", " ").strip()
    except OSError:
        return ""


def _terminate_pid(pid: int, *, label: str, timeout: float = 5.0) -> bool:
    """Terminate a process and wait briefly for the listener to disappear."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            log.info("%s PID %d exited after SIGTERM", label, pid)
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
        log.warning("%s PID %d did not exit after SIGTERM; sent SIGKILL", label, pid)
    except OSError:
        pass
    return not _pid_is_alive(pid)


def _replace_managed_listener(name: str, port: int, marker: str) -> bool:
    """If *port* is occupied by our own managed sidecar, replace it."""
    if not _port_in_use(port):
        return False
    pids = _listener_pids(port)
    if not pids:
        return False
    replaced = False
    for pid in pids:
        cmdline = _cmdline_for_pid(pid)
        if marker not in cmdline:
            log.warning(
                "%s port %d is in use by unrelated PID %d (%s) — reusing existing listener.",
                name, port, pid, cmdline or "<unknown>",
            )
            return False
        log.info(
            "Replacing existing %s listener on port %d: PID %d (%s)",
            name, port, pid, cmdline,
        )
        _terminate_pid(pid, label=name)
        replaced = True
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _port_in_use(port):
            return replaced
        time.sleep(0.2)
    log.warning("%s port %d remained bound after replacement attempt", name, port)
    return replaced


def _apply_manifest_priority_update(state: StateStore, manifest_id_set: frozenset | None):
    if manifest_id_set is None:
        return
    try:
        _conn = state._conn()
        _conn.execute("CREATE TEMP TABLE IF NOT EXISTS _manifest_ids (id TEXT PRIMARY KEY)")
        _conn.execute("DELETE FROM _manifest_ids")
        for mid in manifest_id_set:
            _conn.execute("INSERT OR IGNORE INTO _manifest_ids VALUES (?)", (mid,))
        _updated = _conn.execute(
            "UPDATE slides SET priority=1 WHERE priority=0 "
            "AND slide_id NOT IN (SELECT id FROM _manifest_ids)"
        ).rowcount
        _conn.execute("DROP TABLE IF EXISTS _manifest_ids")
        _conn.commit()
        log.info("Priority update: set priority=1 for %d slide(s) not in manifest", _updated)
    except Exception as _e:
        log.warning("Priority startup update failed: %s", _e)


def _sync_databricks_pending_priorities(cfg: Config, state: StateStore):
    max_pending = int(
        os.environ.get(
            "MUSSEL_STARTUP_PRIORITY_SYNC_MAX_PENDING",
            str(_STARTUP_PRIORITY_SYNC_MAX_PENDING),
        )
    )
    pending_rows = state.get_pending_slide_rows(limit=max_pending + 1)
    if len(pending_rows) > max_pending:
        log.warning(
            "Startup priority sync skipped: pending slide count exceeds limit %d",
            max_pending,
        )
        return

    if not pending_rows:
        return

    pending_by_path = {
        row["slide_path"]: (row["slide_id"], int(row["priority"] or 0))
        for row in pending_rows
    }
    updates_by_path: dict[str, int] = {}
    matched_rows = 0

    for w_cfg in cfg.watchers:
        if w_cfg.type != "databricks" or not w_cfg.warehouse_id:
            continue

        watcher = DatabricksWatcher(w_cfg, deque(), state, threading.Event())
        try:
            query = watcher._build_query()
            log.info(
                "Startup priority sync: querying Databricks watcher for %d pending slide(s)",
                len(pending_by_path),
            )
            for row in watcher._iter_query_rows(query):
                slide_path = row.get("slide_path") or ""
                pending_meta = pending_by_path.get(slide_path)
                if pending_meta is None:
                    continue
                slide_id, existing_priority = pending_meta
                new_priority = state.resolve_priority(
                    slide_id,
                    watcher._parse_priority(row),
                )
                matched_rows += 1
                if new_priority != existing_priority:
                    updates_by_path[slide_path] = new_priority
        except Exception as exc:
            log.warning("Startup priority sync skipped for Databricks watcher: %s", exc)

    if not matched_rows:
        log.info("Startup priority sync: no pending slides matched the Databricks result set")
        return
    if not updates_by_path:
        log.info(
            "Startup priority sync: checked %d matched pending slide(s); no priority changes needed",
            matched_rows,
        )
        return

    updated = state.update_slide_priorities(
        [(slide_path, priority) for slide_path, priority in updates_by_path.items()]
    )
    log.info(
        "Startup priority sync: updated %d pending slide(s) from %d Databricks matches",
        updated,
        matched_rows,
    )


def _start_startup_priority_sync(
    cfg: Config,
    state: StateStore,
    stop_event: threading.Event,
) -> threading.Thread:
    """Run optional Databricks priority reconciliation in the background.

    This must not block dispatcher startup: the sync can scan a very large
    pending set and query a remote warehouse.
    """

    def _run() -> None:
        try:
            if stop_event.wait(5):
                return
            _sync_databricks_pending_priorities(cfg, state)
        except Exception as exc:
            log.warning("Background startup priority sync failed: %s", exc)
        finally:
            close_connection = getattr(state, "close_thread_connection", None)
            if close_connection is not None:
                close_connection()

    thread = threading.Thread(
        target=_run,
        name="startup-priority-sync",
        daemon=True,
    )
    thread.start()
    return thread


def _startup_priority_reconciliation_enabled() -> bool:
    """Return whether expensive priority repair is explicitly authorized.

    New watcher rows receive priority as they are inserted. Reconciling old
    rows scans the entire pending set and must not run during every service
    restart on the production PostgreSQL instance.
    """
    return os.environ.get("MUSSEL_ENABLE_STARTUP_PRIORITY_RECONCILIATION", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _sidecar_log_path(cfg: Config, name: str) -> str:
    return os.path.join(cfg.log_dir, f"{name}.log")


def _dispatcher_log_path(cfg: Config) -> str:
    return os.path.join(cfg.log_dir, "dispatcher.log")


def _dispatcher_status_snapshot_path(cfg: Config) -> str:
    return os.path.join(cfg.state_dir, "dispatcher_status.json")


def _next_snapshot_publish_delay(
    current_delay_seconds: float,
    *,
    success: bool,
    base_interval_seconds: float,
    max_interval_seconds: float,
) -> float:
    if success:
        return float(base_interval_seconds)
    return min(float(max_interval_seconds), max(float(base_interval_seconds), float(current_delay_seconds) * 2.0))


def _start_status_snapshot_publisher(
    cfg: Config,
    state: StateStore,
    stop_event: threading.Event,
    *,
    interval_seconds: float = 60.0,
    max_interval_seconds: float = 300.0,
) -> threading.Thread:
    """Publish a cached dispatcher status snapshot on a fixed interval.

    The dashboard reads this file instead of opening its own DB sessions on
    every poll, which keeps PostgreSQL connection churn bounded.
    """
    snapshot_path = _dispatcher_status_snapshot_path(cfg)

    def _publish() -> None:
        next_delay = float(interval_seconds)
        try:
            while not stop_event.is_set():
                try:
                    state.write_status_snapshot(snapshot_path)
                    next_delay = _next_snapshot_publish_delay(
                        next_delay,
                        success=True,
                        base_interval_seconds=interval_seconds,
                        max_interval_seconds=max_interval_seconds,
                    )
                except Exception as exc:
                    log.warning("Status snapshot write failed: %s", exc)
                    next_delay = _next_snapshot_publish_delay(
                        next_delay,
                        success=False,
                        base_interval_seconds=interval_seconds,
                        max_interval_seconds=max_interval_seconds,
                    )
                stop_event.wait(next_delay)
        finally:
            close_connection = getattr(state, "close_thread_connection", None)
            if close_connection is not None:
                close_connection()

    thread = threading.Thread(
        target=_publish,
        name="dispatcher-status-snapshot",
        daemon=True,
    )
    thread.start()
    return thread


def _start_release_controller(
    cfg: Config,
    state: StateStore,
    stop_event: threading.Event,
) -> threading.Thread | None:
    """Start the opt-in release reconciler after one synchronous gate update."""
    if not cfg.auto_sync_release:
        return None
    controller = ReleaseController(cfg, state)
    first = controller.run_once()
    log.info("Release controller startup state: %s — %s", first["status"], first["message"])
    interval = max(10, int(cfg.release_sync_interval_seconds or 30))

    def _run() -> None:
        try:
            while not stop_event.wait(interval):
                result = controller.run_once()
                log.debug("Release controller: %s — %s", result["status"], result["message"])
        finally:
            close_connection = getattr(state, "close_thread_connection", None)
            if close_connection is not None:
                close_connection()

    thread = threading.Thread(target=_run, name="release-controller", daemon=True)
    thread.start()
    return thread


def _enable_persistent_dispatcher_logging(cfg: Config) -> str:
    """Attach a persistent file handler for the main dispatcher process."""
    log_path = _dispatcher_log_path(cfg)
    root = logging.getLogger()
    abs_log_path = os.path.abspath(log_path)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if os.path.abspath(handler.baseFilename) == abs_log_path:
                    return log_path
            except Exception:
                continue

    file_handler = logging.handlers.WatchedFileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
        delay=True,
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path


def _launch_sidecar(
    name: str,
    cmd: list[str],
    cfg: Config,
    *,
    persistent: bool = False,
) -> subprocess.Popen:
    """Launch a sidecar process (dashboard / tower_proxy) and capture its logs."""
    log_path = _sidecar_log_path(cfg, name)
    log.info("Launching %s: %s", name, " ".join(cmd))
    log.info("%s logs: %s", name, log_path)
    fh = open(log_path, "a", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=fh,
            start_new_session=persistent,
        )
    finally:
        fh.close()
    return proc


def _start_sidecars(cfg: Config, config_path: str) -> list[tuple[str, subprocess.Popen, bool]]:
    """Start dashboard and tower_proxy subprocesses using ports from config.

    Both are optional — if dashboard_port is 0 neither is started.
    If a port is already in use (e.g. from a previous dispatcher instance that
    was killed without taking its children), we reuse the existing process
    instead of launching a new one that would immediately fail to bind.
    Returns ``(name, proc, persistent)`` tuples so shutdown can leave durable
    sidecars running when appropriate.
    """
    disable_sidecars = os.environ.get("MUSSEL_DISABLE_SIDECARS", "").strip().lower()
    if disable_sidecars in {"1", "true", "yes", "on"}:
        log.info("Skipping sidecar startup because MUSSEL_DISABLE_SIDECARS=%s", disable_sidecars)
        return []
    if not cfg.dashboard_port:
        return []

    python = sys.executable
    procs: list[tuple[str, subprocess.Popen, bool]] = []

    if _replace_managed_listener("dashboard", cfg.dashboard_port, "mussel_dispatcher.dashboard"):
        pass
    if _port_in_use(cfg.dashboard_port):
        log.warning(
            "Dashboard port %d already in use — reusing existing process.",
            cfg.dashboard_port,
        )
    else:
        dashboard = _launch_sidecar("dashboard", [
            python, "-m", "mussel_dispatcher.dashboard",
            config_path,
            "--port", str(cfg.dashboard_port),
        ], cfg)
        procs.append(("dashboard", dashboard, False))

    if cfg.tower_proxy_port:
        if _replace_managed_listener("tower_proxy", cfg.tower_proxy_port, "mussel_dispatcher.tower_proxy"):
            pass
        if _port_in_use(cfg.tower_proxy_port):
            log.warning(
                "Tower proxy port %d already in use — reusing existing process.",
                cfg.tower_proxy_port,
            )
        else:
            proxy = _launch_sidecar("tower_proxy", [
                python, "-m", "mussel_dispatcher.tower_proxy",
                "--upstream", f"http://localhost:{cfg.dashboard_port}",
                "--port", str(cfg.tower_proxy_port),
            ], cfg, persistent=True)
            procs.append(("tower_proxy", proxy, True))

    return procs


def _batch_trace_path(log_dir: str, batch_id: str) -> str:
    return os.path.join(log_dir, f"batch_{batch_id}.trace.tsv")


def _batch_nf_log_path(log_dir: str, batch_id: str) -> str:
    return os.path.join(log_dir, f"batch_{batch_id}.nf.log")


def _nf_log_indicates_success(nf_log_path: str | None) -> bool:
    """Return True when a Nextflow debug log clearly shows successful completion."""
    if not nf_log_path:
        return False

    saw_success_stats = False
    saw_execution_complete = False
    try:
        with open(nf_log_path, errors="ignore") as fh:
            for line in fh:
                if (
                    "WorkflowStats[" in line
                    and "failedCount=0" in line
                    and "abortedCount=0" in line
                ):
                    saw_success_stats = True
                elif "Execution complete -- Goodbye" in line:
                    saw_execution_complete = True
    except OSError:
        return False

    return saw_success_stats and saw_execution_complete


def _batch_run_started_at(batch: dict) -> float:
    """Best-effort recovery of the original batch start time as a Unix timestamp."""
    dispatched_at = batch.get("dispatched_at")
    if not dispatched_at:
        return 0.0
    try:
        return datetime.fromisoformat(dispatched_at).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _collect_slurm_job_ids(trace_path: str | None = None, nf_log_path: str | None = None) -> list[str]:
    """Collect SLURM job IDs submitted by a single Nextflow batch."""
    job_ids: set[str] = set()

    if trace_path:
        try:
            with open(trace_path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    name = row.get("name") or ""
                    if not any(proc in name for proc in _SLURM_PROCESSES):
                        continue
                    native_id = (row.get("native_id") or "").strip()
                    if native_id.isdigit():
                        job_ids.add(native_id)
        except OSError:
            pass

    if nf_log_path:
        try:
            with open(nf_log_path) as f:
                for line in f:
                    if "[SLURM] submitted process" not in line:
                        continue
                    if not any(proc in line for proc in _SLURM_PROCESSES):
                        continue
                    match = _SLURM_JOB_ID_RE.search(line)
                    if match:
                        job_ids.add(match.group(1))
        except OSError:
            pass

    return sorted(job_ids, key=int)


def _scancel_slurm_jobs(batch_id: str, job_ids: list[str]) -> int:
    """Cancel submitted SLURM jobs for a batch. Returns number of IDs submitted to scancel."""
    if not job_ids:
        return 0

    cancelled = 0
    chunk_size = 100
    for i in range(0, len(job_ids), chunk_size):
        chunk = job_ids[i:i + chunk_size]
        try:
            result = subprocess.run(
                ["scancel", *chunk],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("Batch %s: failed to scancel %d SLURM job(s): %s", batch_id, len(chunk), exc)
            continue
        if result.returncode == 0:
            cancelled += len(chunk)
        else:
            stderr = (result.stderr or result.stdout or "").strip()
            log.warning(
                "Batch %s: scancel returned %d for %d SLURM job(s): %s",
                batch_id,
                result.returncode,
                len(chunk),
                stderr,
            )
    if cancelled:
        log.info("Batch %s: requested cancellation for %d SLURM job(s)", batch_id, cancelled)
    return cancelled



class BatchScheduler:
    """
    Fires when either:
    - `batch_size` slides have accumulated in the pending deque, OR
    - `max_wait_seconds` have elapsed since the first slide was added.
    """

    def __init__(self, cfg: Config, state: StateStore, run_manager: "RunManager", stop_event: threading.Event):
        self.cfg = cfg
        self.state = state
        self.run_manager = run_manager
        self.stop_event = stop_event
        self._pending: deque = deque()
        self._first_seen_at: Optional[float] = None
        self._lock = threading.Lock()
        self._s3_client = None  # lazily built for pre-dispatch S3 validation
        self._capacity_guard = CapacityGuard(cfg.storage_policy, name="local")
        self._cleanup_thread: threading.Thread | None = None
        # Rotating batch-size index for natural stagger via alternating sizes.
        self._batch_size_idx = 0
        self._db_sweep_interval = self._DB_SWEEP_INTERVAL

    def _next_batch_size(self) -> int:
        """Return the next batch size, rotating through cfg.batch_sizes if set."""
        sizes = getattr(self.cfg, "batch_sizes", None)
        if sizes:
            size = sizes[self._batch_size_idx % len(sizes)]
            self._batch_size_idx += 1
            return size
        return self.cfg.batch_size

    # Number of threads for concurrent S3 existence checks — also sets boto3 pool size.
    _S3_CHECK_WORKERS = 32

    def _get_s3_client(self):
        """Return a boto3 S3 client built from the first watcher that has S3 credentials."""
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError:
            return None
        s3_cfg = BotocoreConfig(
            max_pool_connections=self._S3_CHECK_WORKERS,
            connect_timeout=10,
            read_timeout=10,
            retries={"max_attempts": 1},
        )
        for w in self.cfg.watchers:
            kwargs: dict = {"config": s3_cfg}
            if w.s3_access_key and w.s3_secret_key:
                kwargs["aws_access_key_id"] = w.s3_access_key
                kwargs["aws_secret_access_key"] = w.s3_secret_key
            if w.s3_endpoint:
                kwargs["endpoint_url"] = w.s3_endpoint
            if w.s3_access_key or w.s3_endpoint:
                self._s3_client = boto3.client("s3", **kwargs)
                return self._s3_client
        # Fall back to default credentials
        self._s3_client = boto3.client("s3", config=s3_cfg)
        return self._s3_client

    def _s3_path_exists(self, s3_path: str, s3) -> bool:
        """Return True if the S3 object exists, False on 404/NoSuchKey."""
        from urllib.parse import urlparse
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as exc:
            resp = getattr(exc, "response", None)
            code = (resp or {}).get("Error", {}).get("Code", "") if resp else ""
            if code in ("404", "NoSuchKey") or "404" in str(exc) or "NoSuchKey" in str(exc):
                return False
            # Unexpected error (auth, network) — treat as present to avoid false blacklisting
            log.warning("S3 head_object error for %s: %s (treating as present)", s3_path, exc)
            return True

    def _validate_s3_batch(self, batch: list) -> tuple[list, list]:
        """Check each s3:// slide for existence; return (valid_batch, blacklisted_ids).

        Uses a thread pool to check all paths concurrently.  Non-s3 paths pass through.
        Slides that fail the check are blacklisted in the DB and excluded from the batch.
        """
        s3_slides = [s for s in batch if s.get("slide_path", "").startswith("s3://")]
        if not s3_slides:
            return batch, []

        s3 = self._get_s3_client()
        if s3 is None:
            return batch, []  # boto3 unavailable; skip check

        max_retries = self.cfg.max_slide_retries or 999
        missing: list[str] = []

        def check(slide):
            return slide, self._s3_path_exists(slide["slide_path"], s3)

        with ThreadPoolExecutor(max_workers=self._S3_CHECK_WORKERS, thread_name_prefix="s3-check") as pool:
            for slide, exists in pool.map(check, s3_slides):
                if not exists:
                    missing.append(slide["slide_id"])

        if not missing:
            return batch, []

        blacklisted_ids: list[str] = []
        for slide_id in missing:
            self.state.blacklist_slide(
                slide_id,
                reason="Pre-dispatch S3 check: object not found",
                max_retries=max_retries,
            )
            blacklisted_ids.append(slide_id)

        missing_set = set(missing)
        valid_batch = [s for s in batch if s.get("slide_id") not in missing_set]
        log.warning(
            "Pre-dispatch S3 check: blacklisted %d missing slide(s): %s",
            len(blacklisted_ids), blacklisted_ids,
        )
        return valid_batch, blacklisted_ids

    def enqueue(self, slide: dict):
        with self._lock:
            self._pending.append(slide)
            if self._first_seen_at is None:
                self._first_seen_at = time.monotonic()
            self._db_sweep_interval = self._DB_SWEEP_INTERVAL
            log.debug("Pending queue size: %d", len(self._pending))

    # How often to sweep the DB for PENDING slides that fell out of the deque.
    # This catches slides reset to PENDING by _verify_wds_coverage (or similar)
    # that weren't re-added to the in-memory dispatch queue.
    _DB_SWEEP_INTERVAL = 60  # seconds
    _DB_SWEEP_MAX_INTERVAL = 600  # seconds
    # How often to reset retriable FAILED slides back to PENDING.
    # Complements the startup-time retry so slides failed mid-run (e.g. due to
    # node failures) get re-queued without requiring a dispatcher restart.
    _RETRY_FAILED_INTERVAL = 600  # seconds (10 minutes)
    _RETRY_FINALIZATION_INTERVAL = 60  # seconds

    # How often to check for stuck (hung) NF processes.
    _WATCHDOG_INTERVAL = 600  # seconds (10 minutes)

    # How often to sweep for orphaned work dirs from FAILED batches.
    # Catches dirs left behind when the dispatcher was SIGKILL'd before
    # _cleanup() could run (e.g. during restart churn).
    _FAILED_WORKDIR_CLEANUP_INTERVAL = 3600  # seconds (1 hour)

    def run(self):
        """Scheduler loop — call in the main thread."""
        log.info("BatchScheduler started (batch_size=%d, max_wait=%ds)",
                 self.cfg.batch_size, self.cfg.max_wait_seconds)
        self.run_manager.choose_execution_pool(queued=len(self._pending))
        _last_pool_status = time.monotonic()
        _last_db_sweep = time.monotonic()
        _last_retry_failed = time.monotonic()
        _last_retry_finalization = time.monotonic()
        _last_eta_log = time.monotonic()
        _last_watchdog = time.monotonic()
        _last_failed_workdir_cleanup = time.monotonic()
        _ETA_LOG_INTERVAL = 1800  # log ETA every 30 minutes
        while not self.stop_event.is_set():
            self._maybe_dispatch()
            now = time.monotonic()
            if now - _last_pool_status >= 60:
                _last_pool_status = now
                self.run_manager.choose_execution_pool(queued=len(self._pending))
            if now - _last_db_sweep >= self._db_sweep_interval:
                _last_db_sweep = now
                self._requeue_db_pending()
            if now - _last_retry_failed >= self._RETRY_FAILED_INTERVAL:
                _last_retry_failed = now
                self._retry_failed_slides()
            if now - _last_retry_finalization >= self._RETRY_FINALIZATION_INTERVAL:
                _last_retry_finalization = now
                self._retry_finalizing_batches()
            if now - _last_eta_log >= _ETA_LOG_INTERVAL:
                _last_eta_log = now
                self._log_eta()
            if now - _last_watchdog >= self._WATCHDOG_INTERVAL:
                _last_watchdog = now
                self._watchdog_stuck_batches()
            cleanup_interval = max(
                60, int(self.cfg.storage_policy.cleanup_interval_seconds or 3600)
            )
            if now - _last_failed_workdir_cleanup >= cleanup_interval:
                _last_failed_workdir_cleanup = now
                if self.cfg.cleanup_work_dir:
                    self._cleanup_failed_work_dirs()
            self.stop_event.wait(5)
        # Do NOT force-dispatch on shutdown — pending slides stay in the DB
        # and will be recovered on next startup.
        log.info("BatchScheduler stopped. Pending slides will be recovered on restart.")

    def _log_eta(self):
        """Log current throughput and ETA based on recently completed slides."""
        try:
            stats = self.state.get_throughput_stats(window_hours=6.0)
            tph = stats["throughput_per_hour"]
            remaining = stats["remaining"]
            eta_s = stats["eta_seconds"]
            done_in_window = stats["completed_in_window"]
            if tph is not None:
                if eta_s is not None:
                    h, m = divmod(int(eta_s) // 60, 60)
                    d, h = divmod(h, 24)
                    eta_str = (f"{d}d {h}h {m}m" if d else f"{h}h {m}m") if eta_s >= 60 else "<1m"
                else:
                    eta_str = "unknown"
                log.info(
                    "Progress: %d remaining | %.1f slides/hr (last 6h: %d slides) | ETA: %s",
                    remaining, tph, done_in_window, eta_str,
                )
            else:
                log.info("Progress: %d remaining | throughput not yet available (%d slides in last 6h)",
                         remaining, done_in_window)
        except Exception as exc:
            log.debug("ETA calculation failed: %s", exc)

    def _requeue_db_pending(self):
        """Re-enqueue PENDING slides from the DB that are not currently in the dispatch deque.

        Handles slides reset to PENDING after a batch completion (e.g. by
        _verify_wds_coverage) that were never re-added to the in-memory queue.

        Excludes slides that are in-flight in RunManager (popped from the deque and
        submitted to the thread pool but not yet written as DISPATCHED in the DB).
        Without this exclusion the 60-second sweep would re-enqueue those slides during
        the Nextflow startup window (~30-90 s), causing duplicate dispatches that spiral
        into a runaway batch-creation loop.
        """
        with self._lock:
            in_deque_ids = {s.get("slide_id") for s in self._pending}
            queue_size = len(self._pending)
        if queue_size:
            log.debug(
                "BatchScheduler DB sweep skipped: local queue already has %d slide(s)",
                queue_size,
            )
            self._db_sweep_interval = min(
                self._DB_SWEEP_MAX_INTERVAL,
                max(self._DB_SWEEP_INTERVAL, self._db_sweep_interval * 2),
            )
            return
        in_flight_ids = self.run_manager.in_flight_slide_ids
        excluded = in_deque_ids | in_flight_ids
        max_pending = int(
            os.environ.get(
                "MUSSEL_DB_SWEEP_MAX_PENDING",
                str(_DB_SWEEP_MAX_PENDING),
            )
        )
        fetch_limit = max_pending + 1
        pending_rows = self.state.get_pending_slides(limit=fetch_limit)
        sweep_limited = len(pending_rows) > max_pending
        pending_in_db = pending_rows[:max_pending] if sweep_limited else pending_rows
        newly_enqueued = 0
        for slide in pending_in_db:
            if slide.get("slide_id") not in excluded:
                self.enqueue(slide)
                newly_enqueued += 1
        if sweep_limited:
            log.warning(
                "BatchScheduler DB sweep limited to %d+ pending slide(s); fetched %d to avoid database overload",
                max_pending,
                len(pending_in_db),
            )
        if newly_enqueued:
            self._db_sweep_interval = self._DB_SWEEP_INTERVAL
            log.info("BatchScheduler: re-enqueued %d PENDING slide(s) from DB (missed by watcher)",
                     newly_enqueued)
            return
        self._db_sweep_interval = min(
            self._DB_SWEEP_MAX_INTERVAL,
            max(self._DB_SWEEP_INTERVAL, self._db_sweep_interval * 2),
        )

    def _retry_failed_slides(self):
        """Periodically reset retriable FAILED slides back to PENDING.

        Mirrors the startup-time retry in recover_in_flight() so that slides
        failed mid-run (e.g. due to SLURM node failures with errorStrategy=ignore)
        are re-queued without requiring a dispatcher restart.
        """
        if not self.cfg.retry_failed:
            return
        try:
            n = self.state.reset_failed_to_pending(max_retries=self.cfg.max_slide_retries)
        except sqlite3.OperationalError as exc:
            if StateStore._is_locked_error(exc):
                log.warning("BatchScheduler: skipping periodic retry sweep due to SQLite lock contention: %s", exc)
                return
            raise
        if n:
            log.info("BatchScheduler: reset %d FAILED slide(s) to PENDING (periodic retry sweep)", n)
            self._requeue_db_pending()

    def _retry_finalizing_batches(self):
        """Replay local finalization whose required hooks have not completed."""
        for batch in self.state.get_active_batches():
            if batch.get("backend") == "remote_agent" or batch.get("status") != "FINALIZING":
                continue
            retry_at = batch.get("finalization_retry_at")
            if retry_at:
                try:
                    retry_ts = datetime.fromisoformat(
                        str(retry_at).replace("Z", "+00:00")
                    ).timestamp()
                    if retry_ts > time.time():
                        continue
                except (TypeError, ValueError, OSError):
                    pass
            if self.run_manager.is_running(batch.get("batch_id")):
                continue
            batch_id = batch.get("batch_id")
            if not batch_id or not batch.get("csv_path") or not batch.get("work_dir"):
                continue
            self.run_manager.submit_finalize_recovered_success(
                batch_id,
                batch["csv_path"],
                batch.get("log_path") or "",
                batch["work_dir"],
                run_started_at=_batch_run_started_at(batch),
                nf_log_path=_batch_nf_log_path(
                    os.path.dirname(batch.get("log_path") or ""), batch_id
                ),
            )

    def _watchdog_stuck_batches(self):
        """Kill NF processes whose log file has not been updated recently.

        Uses two signals to distinguish a genuinely dead NF process from one
        that is alive but waiting for queued SLURM jobs:

        1. **NF internal debug log** (``batch_{id}.nf.log``) — Nextflow writes
           to this file every ~5 min while it polls ``squeue``, even when every
           SLURM job is still PENDING.  If this log was updated within the last
           20 minutes the NF JVM is alive; the cluster is simply busy and no
           kill is issued regardless of how old the stdout log is.

        2. **Batch stdout log** (``batch_{id}.log``) — only updated when a task
           starts or finishes.  Used as the staleness signal when the internal
           log does not exist (batches dispatched before the ``-log`` flag was
           added).

        ``stuck_batch_timeout_hours`` is applied against whichever log is used
        as the staleness signal.  Because a healthy NF updates its internal log
        every 5 min, a value of 1 h is sufficient to catch a genuinely crashed
        process while never firing on a congested cluster.  Set to 0 to disable.
        """
        timeout_hours = self.cfg.stuck_batch_timeout_hours
        if not timeout_hours or timeout_hours <= 0:
            return
        cutoff = time.time() - timeout_hours * 3600
        # NF polls squeue every ~5 min and writes to its internal log each time.
        # If the log was touched within this window the JVM is running normally.
        nf_alive_window = 20 * 60  # seconds

        for batch in self.state.get_running_batches():
            batch_id = batch["batch_id"]
            nf_pid = batch.get("nf_pid")
            if not nf_pid:
                continue

            # Primary signal: NF internal debug log (present on newer batches).
            nf_log_path = os.path.join(self.cfg.log_dir, f"batch_{batch_id}.nf.log")
            try:
                nf_log_mtime = os.path.getmtime(nf_log_path)
            except OSError:
                nf_log_mtime = None

            if nf_log_mtime is not None:
                if time.time() - nf_log_mtime < nf_alive_window:
                    # NF is alive and actively polling SLURM.  The cluster is
                    # congested but the batch is not stuck — skip.
                    continue
                stale_mtime = nf_log_mtime
                signal_name = "nf.log"
            else:
                # Fallback for batches dispatched before the -log flag was added.
                # Prefer the log path recorded for the batch over a reconstructed one.
                log_path = batch.get("log_path") or os.path.join(
                    self.cfg.log_dir, f"batch_{batch_id}.log"
                )
                try:
                    stale_mtime = os.path.getmtime(log_path)
                except OSError:
                    continue
                signal_name = "log"

            if stale_mtime < cutoff:
                idle_hours = (time.time() - stale_mtime) / 3600
                log.warning(
                    "Batch %s: %s silent for %.1fh (last update %s) — "
                    "killing stuck NF process PID=%d",
                    batch_id, signal_name, idle_hours,
                    datetime.fromtimestamp(stale_mtime).strftime("%H:%M:%S"),
                    nf_pid,
                )
                _kill_orphaned_nf(
                    batch_id,
                    nf_pid,
                    trace_path=_batch_trace_path(self.cfg.log_dir, batch_id),
                    nf_log_path=_batch_nf_log_path(self.cfg.log_dir, batch_id),
                    run_name=batch.get("run_name"),
                )

    def _cleanup_failed_work_dirs(self):
        """Retry shared execution cleanup without blocking dispatch."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        def _spec(batch):
            work_dir = str(batch.get("work_dir") or "").strip()
            if not work_dir:
                return None
            return ExecutionCleanupSpec(
                batch_id=str(batch["batch_id"]),
                managed_root=self.cfg.work_base_dir,
                run_root=work_dir,
                execution_paths=(work_dir,),
                protected_paths=(self.cfg.outdir,),
            )

        def _bg():
            cleaned = reconcile_execution_cleanup(
                self.state,
                _spec,
                candidates=[
                    batch for batch in self.state.get_execution_cleanup_candidates()
                    if str(batch.get("backend") or "local") != "remote_agent"
                ],
            )
            if cleaned:
                log.info("Periodic cleanup: reconciled %d local execution batch(es)", cleaned)

        self._cleanup_thread = threading.Thread(
            target=_bg, name="execution-cleanup", daemon=True
        )
        self._cleanup_thread.start()

    def _maybe_dispatch(self, force: bool = False):
        # Don't submit a new batch if we're already at the concurrency limit.
        # Uses a BoundedSemaphore (not len(_futures)) to avoid the TOCTOU race where
        # fast-fail resumes release their slot between the check and submit().
        active_batches = self.run_manager.count_active_batches()
        if not self.run_manager.has_fresh_dispatch_capacity(active_batches):
            return
        execution_pool = self.run_manager.choose_execution_pool(queued=len(self._pending))
        if execution_pool is None:
            log.debug("Fresh dispatch blocked: no execution pool has available capacity")
            return
        # Pool capacity and storage were evaluated by choose_execution_pool().
        with self._lock:
            n = len(self._pending)
            if n == 0:
                return
            elapsed = time.monotonic() - self._first_seen_at if self._first_seen_at else 0
            this_batch_size = self._next_batch_size()
            size_trigger = n >= this_batch_size
            time_trigger = elapsed >= self.cfg.max_wait_seconds
            if not force and not size_trigger and not time_trigger:
                # Put the index back if we didn't actually dispatch
                self._batch_size_idx -= 1
                return
            if n < self.cfg.min_batch_size and not force:
                self._batch_size_idx -= 1
                return

            batch = []
            while self._pending and len(batch) < this_batch_size:
                batch.append(self._pending.popleft())
            self._first_seen_at = None if not self._pending else self._first_seen_at

        if batch:
            log.info("Dispatching batch of %d slides (force=%s, size_trigger=%s, time_trigger=%s)",
                     len(batch), force, size_trigger, time_trigger)
            batch, _ = self._validate_s3_batch(batch)
            if not batch:
                log.warning("Batch cancelled: all slides failed pre-dispatch S3 check.")
                return
            batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
            log.info(
                "_maybe_dispatch: generated batch_id=%s for %d slides via pool=%s backend=%s target=%s",
                batch_id,
                len(batch),
                getattr(execution_pool, "name", ""),
                getattr(execution_pool, "backend", ""),
                getattr(execution_pool, "target", ""),
            )
            if not self.run_manager.submit(
                batch_id,
                batch,
                known_active_batches=active_batches,
                execution_pool=execution_pool,
            ):
                with self._lock:
                    self._pending.extendleft(reversed(batch))
                    if self._first_seen_at is None:
                        self._first_seen_at = time.monotonic()
                log.info(
                    "Batch %s was not submitted because capacity disappeared; "
                    "returned %d slide(s) to pending queue",
                    batch_id,
                    len(batch),
                )


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------

def _execution_slot_count(cfg: Config) -> int:
    """Concurrent Nextflow drivers allowed: the summed execution pool limits."""
    return max(1, int(cfg.effective_max_concurrent_runs() or cfg.max_concurrent_runs or 1))


def _split_startup_resumes(cfg: Config, resume_specs: list) -> tuple[list, list]:
    """Split startup resumes into those RunManager has slots for and the rest."""
    cap = _execution_slot_count(cfg)
    return resume_specs[:cap], resume_specs[cap:]


class RunManager:
    _ACTIVE_BATCH_COUNT_CACHE_TTL = 30.0

    def __init__(self, cfg: Config, state: StateStore):
        self.cfg = cfg
        self.state = state
        # One worker per execution slot. Remote-agent drivers (fresh and
        # resumed) wait on this same executor while holding a slot, so size it
        # from the summed pool limits exactly as _slots is; sizing it from
        # cfg.max_concurrent_runs could queue slot-holding futures that never
        # start when the pool limits sum higher.
        max_runs = _execution_slot_count(cfg)
        self._executor = ThreadPoolExecutor(max_workers=max_runs,
                                            thread_name_prefix="nextflow-run")
        # Post-success hooks have their own bounded lane and must not consume
        # Nextflow capacity. Shared WDS/DB locks still protect mutations.
        self._finalization_pool = FinalizationWorkerPool(
            max(1, int(cfg.effective_max_concurrent_finalizations(lane="local"))),
            thread_name_prefix="post-batch-finalize",
        )
        # Retain this compatibility handle for callers/tests that explicitly
        # shut down the historical executor attribute.
        self._finalization_executor = self._finalization_pool.executor
        self._shutdown_event = threading.Event()
        self._futures: dict = {}
        self._finalization_futures: dict = {}
        # batch_id -> consecutive incomplete local finalization attempts.
        self._finalization_attempts: dict[str, int] = {}
        # Maps batch_id → set of slide_ids that have been popped from the dispatch
        # deque and submitted to the thread pool but not yet written as DISPATCHED in
        # the DB.  BatchScheduler._requeue_db_pending() excludes these IDs so that
        # the 60-second sweep never re-enqueues slides that are mid-submission.
        self._in_flight: dict[str, set] = {}
        self._lock = threading.Lock()
        # Semaphore that serialises concurrency decisions atomically.
        # Every submit() / submit_resume() acquires one permit; _on_done() releases
        # it.  Using a semaphore instead of len(_futures) eliminates the TOCTOU race
        # where fast-fail resumes release the future slot between _maybe_dispatch()'s
        # capacity check and the actual submit(), causing overshoot above
        # max_concurrent_runs.
        self._slots = threading.BoundedSemaphore(max_runs)
        self._active_batch_count_cache: int | None = None
        self._active_batch_count_cache_ts = 0.0
        self._release_gate_reasons: dict[str, str] = {}
        self._capacity_guard = CapacityGuard(cfg.storage_policy, name="local")
        self._last_pool_status: list[dict] = []

    def _close_worker_state_connection(self) -> None:
        close_connection = getattr(self.state, "close_thread_connection", None)
        if close_connection is not None:
            close_connection()

    def _run_runner(self, runner: NextflowRunner):
        try:
            return runner.run()
        finally:
            self._close_worker_state_connection()

    def _queue_post_success_finalization(
        self,
        batch_id: str,
        csv_path: str,
        log_path: str,
        work_dir: str,
        run_started_at: float,
    ) -> bool:
        """Move completed NF bookkeeping off the execution worker lane."""
        # Mark this before enqueueing so a capacity read cannot briefly count
        # a completed driver as an active execution batch.
        self.state.set_batch_status(batch_id, "FINALIZING")
        return self.submit_finalize_recovered_success(
            batch_id,
            csv_path,
            log_path,
            work_dir,
            run_started_at=run_started_at,
            nf_log_path=os.path.join(self.cfg.log_dir, f"batch_{batch_id}.nf.log"),
            replace_execution=True,
        )

    def _run_finalize_recovered_success(
        self,
        batch_id: str,
        csv_path: str,
        log_path: str,
        work_dir: str,
        *,
        run_started_at: float = 0.0,
        nf_log_path: str | None = None,
    ):
        try:
            runner = NextflowRunner(
                self.cfg, batch_id, [], self.state, shutdown_event=self._shutdown_event
            )
            return runner.finalize_recovered_success(
                csv_path,
                log_path,
                work_dir,
                run_started_at=run_started_at,
                nf_log_path=nf_log_path,
            )
        finally:
            self._close_worker_state_connection()

    @property
    def in_flight_slide_ids(self) -> set:
        """Return the union of all slide IDs currently in-flight (submitted but not yet DB-confirmed)."""
        with self._lock:
            return set().union(*self._in_flight.values()) if self._in_flight else set()

    def has_capacity(self) -> bool:
        """Return True if a concurrency slot is available (non-blocking check)."""
        acquired = self._slots.acquire(blocking=False)
        if acquired:
            self._slots.release()
        return acquired

    def _invalidate_active_batch_count_cache(self) -> None:
        self._active_batch_count_cache = None
        self._active_batch_count_cache_ts = 0.0

    def count_active_batches(self) -> int:
        now = time.monotonic()
        cached = self._active_batch_count_cache
        if (
            cached is not None
            and (now - self._active_batch_count_cache_ts) < self._ACTIVE_BATCH_COUNT_CACHE_TTL
        ):
            return cached
        active = self.state.count_scheduling_active_batches()
        self._active_batch_count_cache = active
        self._active_batch_count_cache_ts = now
        return active

    def active_batch_counts_by_pool(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for batch in self.state.get_scheduling_active_batches():
            pool_name = self.cfg.infer_batch_pool_name(
                backend=batch.get("backend"),
                target=batch.get("target"),
            )
            if not pool_name:
                continue
            counts[pool_name] = counts.get(pool_name, 0) + 1
        return counts

    def _evaluate_execution_pools(self) -> list[dict]:
        pools = self.cfg.resolved_execution_pools()
        active_counts = self.active_batch_counts_by_pool()
        evaluations = []
        for idx, pool in enumerate(pools):
            active = active_counts.get(pool.name, 0)
            limit = max(0, int(pool.max_concurrent_runs or 0))
            blockers = []
            storage = None
            if pool.backend == "local":
                available = self._capacity_guard.check((self.cfg.work_base_dir, self.cfg.outdir))
                storage = self._capacity_guard.summary()
                if not available:
                    blockers.append({"code": "STORAGE_BELOW_THRESHOLD", "message": "local execution storage is below its dispatch threshold", "details": storage.get("paths", [])})
            elif pool.backend == "remote_agent":
                target_cfg = self.cfg.get_dispatch_target(pool.target)
                if target_cfg is None:
                    blockers.append({"code": "TARGET_NOT_CONFIGURED", "message": f"remote target {pool.target or pool.name} is not configured"})
                else:
                    agent = self.state.read_remote_agent(pool.target) or {}
                    heartbeat = agent.get("heartbeat_at")
                    age = None
                    if heartbeat:
                        try:
                            age = max(0, int(time.time() - datetime.fromisoformat(str(heartbeat)).timestamp()))
                        except Exception:
                            age = None
                    if age is None or age > int(target_cfg.agent_heartbeat_timeout_seconds or 300):
                        blockers.append({"code": "AGENT_HEARTBEAT_STALE", "message": f"remote agent heartbeat is {age if age is not None else "unknown"} seconds old"})
                    storage = agent.get("storage") if isinstance(agent.get("storage"), dict) else None
                    if storage and storage.get("paused"):
                        blockers.append({"code": "STORAGE_BELOW_THRESHOLD", "message": f"remote target {pool.target} capacity guard is paused", "details": storage.get("paths", [])})
                    slurm = agent.get("slurm") if isinstance(agent.get("slurm"), dict) else {}
                    if slurm.get("error"):
                        blockers.append({"code": "SLURM_STATUS_UNAVAILABLE", "message": str(slurm.get("error"))})
                    try:
                        ready, reason = self.state.remote_target_release_ready(pool.target)
                    except Exception as exc:
                        ready, reason = False, f"release readiness check failed: {exc}"
                    if not ready:
                        blockers.append({"code": "RELEASE_NOT_READY", "message": str(reason or "remote release is not ready")})
            else:
                blockers.append({"code": "UNSUPPORTED_BACKEND", "message": f"unsupported execution backend {pool.backend}"})
            if active >= limit:
                blockers.append({"code": "CONCURRENCY_LIMIT", "message": f"{active} active batch(es) at limit {limit}"})
            evaluations.append({
                "name": pool.name,
                "backend": pool.backend,
                "target": pool.target,
                "available": not blockers,
                "state": "AVAILABLE" if not blockers else "BLOCKED",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "active_batches": active,
                "max_batches": limit,
                "storage": storage,
                "blockers": blockers,
                "_order": idx,
            })
        return evaluations

    def execution_pool_status(self) -> list[dict]:
        return [{k: v for k, v in item.items() if k != "_order"} for item in self._evaluate_execution_pools()]

    def choose_execution_pool(self, queued: int | None = None):
        evaluations = self._evaluate_execution_pools()
        self._last_pool_status = [{k: v for k, v in item.items() if k != "_order"} for item in evaluations]
        available = [item for item in evaluations if item["available"]]
        blockers = [
            {"pool": item["name"], **blocker}
            for item in evaluations
            for blocker in item["blockers"]
        ]
        self.state.update_runtime_status(
            execution_pools=self._last_pool_status,
            dispatch_summary={
                "available_pool_count": len(available),
                "queued_work_blocked": bool(blockers and not available),
                "blockers": blockers,
            },
        )
        # Say so (once per change) when queued work cannot go anywhere: a pool
        # held at its limit used to be visible only at debug level.
        block_key = None if available else tuple(sorted((b["pool"], b["code"]) for b in blockers))
        if block_key != getattr(self, "_last_dispatch_block", None):
            self._last_dispatch_block = block_key
            if block_key:
                log.info("Fresh dispatch blocked%s: %s",
                         "" if queued is None else f" ({queued} slide(s) queued)",
                         "; ".join(f"{b['pool']}: {b['code']} ({b['message']})" for b in blockers))
            else:
                log.info("Fresh dispatch unblocked: %s",
                         ", ".join(f"{item['name']} {item['active_batches']}/{item['max_batches']}" for item in available))
        if not available:
            return None
        available.sort(key=lambda item: (item["active_batches"] / item["max_batches"] if item["max_batches"] else 1.0, item["active_batches"], item["_order"]))
        return self.cfg.get_execution_pool(available[0]["name"])

    def has_fresh_dispatch_capacity(self, known_active_batches: int | None = None) -> bool:
        """Return True when both in-process and persisted batch state have room.

        The semaphore only knows about futures owned by this Python process.  If
        the dispatcher is killed after launching Nextflow, the replacement
        process can otherwise start fresh batches while old RUNNING rows and
        their SLURM children still exist.  Treat the DB as authoritative for
        fresh dispatch and let startup recovery decide whether to resume or reset
        existing RUNNING rows.
        """
        running = known_active_batches if known_active_batches is not None else self.count_active_batches()
        limit = self.cfg.effective_max_concurrent_runs()
        blocked = running >= limit
        # INFO once per change, like choose_execution_pool: a silent stall here is
        # as invisible as a silent pool limit.
        if blocked != getattr(self, "_global_dispatch_blocked", False):
            self._global_dispatch_blocked = blocked
            if blocked:
                log.info("Fresh dispatch blocked: %d active batch(es) in DB >= effective max %d", running, limit)
            else:
                log.info("Fresh dispatch no longer at the global limit: %d active batch(es) < %d", running, limit)
        if blocked:
            return False
        return self.has_capacity()

    def submit(
        self,
        batch_id: str,
        slides: list,
        *,
        known_active_batches: int | None = None,
        execution_pool=None,
    ) -> bool:
        """Submit a new batch.  Returns False if at the concurrency limit."""
        # Fail fast if an old batch_id slips through — this should never happen
        # for fresh dispatches but guards against any unexpected code path.
        try:
            from datetime import timezone as _tz
            id_ts = datetime.strptime(batch_id[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_tz.utc)
            age_hours = (datetime.now(_tz.utc) - id_ts).total_seconds() / 3600
            if age_hours > 24:
                import traceback as _tb
                log.error("submit() called with stale batch_id=%s (%.1fh old) — rejecting. Stack:\n%s",
                          batch_id, age_hours, "".join(_tb.format_stack()))
                self.state.reset_dispatched_to_pending(batch_id)
                self.state.complete_batch(batch_id, exit_code=-1)
                return False
        except (ValueError, AttributeError):
            pass

        running = known_active_batches if known_active_batches is not None else self.count_active_batches()
        if running >= self.cfg.effective_max_concurrent_runs():
            log.warning(
                "Refusing fresh batch %s: %d active batch(es) already in DB "
                "(max_concurrent_runs=%d)",
                batch_id,
                running,
                self.cfg.max_concurrent_runs,
            )
            return False
        if not self._slots.acquire(blocking=False):
            return False
        log.debug("submit: batch_id=%s slides=%d", batch_id, len(slides))
        slide_ids = {s["slide_id"] for s in slides if s.get("slide_id")}
        runner_cfg = self.cfg
        if execution_pool is not None:
            runner_cfg = copy.deepcopy(self.cfg)
            runner_cfg.execution_backend = execution_pool.backend
            runner_cfg.default_dispatch_target = (
                execution_pool.target if execution_pool.backend == "remote_agent" else ""
            )
        runner = NextflowRunner(
            runner_cfg,
            batch_id,
            slides,
            self.state,
            shutdown_event=self._shutdown_event,
            finalization_submitter=self._queue_post_success_finalization,
        )
        with self._lock:
            self._in_flight[batch_id] = slide_ids
            future: Future = self._executor.submit(self._run_runner, runner)
            self._futures[batch_id] = future
        future.add_done_callback(lambda f: self._on_done(batch_id, f))
        return True

    def submit_resume(self, batch_id: str, csv_path: str, work_dir: str) -> bool:
        """Re-submit an interrupted batch using the existing work dir and -resume.
        Returns False if at the concurrency limit or if the batch is stale (>24h old)."""
        # Reject batches whose batch_id timestamp is older than 24 hours.
        # This prevents stale May-27 legacy batches from re-entering the
        # pipeline across repeated dispatcher restarts.
        try:
            from datetime import timezone as _tz
            id_ts = datetime.strptime(batch_id[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_tz.utc)
            age_hours = (datetime.now(_tz.utc) - id_ts).total_seconds() / 3600
            if age_hours > 24:
                log.warning("submit_resume: rejecting stale batch %s (created %.1fh ago) — "
                            "resetting slides to PENDING", batch_id, age_hours)
                self.state.reset_dispatched_to_pending(batch_id)
                self.state.complete_batch(batch_id, exit_code=-1)
                return False
        except (ValueError, AttributeError):
            pass  # batch_id doesn't match expected format; proceed

        if not self._slots.acquire(blocking=False):
            return False
        runner_cfg = self.cfg
        try:
            batch = self.state.get_batch(batch_id) or {}
        except Exception:
            batch = {}
        if batch.get("backend") == "remote_agent":
            runner_cfg = copy.deepcopy(self.cfg)
            runner_cfg.execution_backend = "remote_agent"
            runner_cfg.default_dispatch_target = batch.get("target") or self.cfg.default_dispatch_target
        # Resume batches have slides already DISPATCHED in DB — no in-flight tracking needed.
        runner = NextflowRunner(runner_cfg, batch_id, [], self.state,
                                resume=True, existing_csv_path=csv_path,
                                existing_work_dir=work_dir,
                                shutdown_event=self._shutdown_event,
                                finalization_submitter=self._queue_post_success_finalization)
        with self._lock:
            future: Future = self._executor.submit(self._run_runner, runner)
            self._futures[batch_id] = future
        future.add_done_callback(lambda f: self._on_done(batch_id, f))
        return True

    def submit_finalize_recovered_success(
        self,
        batch_id: str,
        csv_path: str,
        log_path: str,
        work_dir: str,
        *,
        run_started_at: float = 0.0,
        nf_log_path: str | None = None,
        replace_execution: bool = False,
    ) -> bool:
        """Submit replay of post-success bookkeeping for a recovered batch."""
        with self._lock:
            if batch_id in self._futures:
                if not replace_execution:
                    return False
                # A live runner calls this before its done callback removes
                # the execution future. Replace the tracking entry so status
                # and future ownership move together without double-submit.
                self._futures.pop(batch_id, None)
        future = self._finalization_pool.submit(
            batch_id,
            self._run_finalize_recovered_success,
            batch_id,
            csv_path,
            log_path,
            work_dir,
            run_started_at=run_started_at,
            nf_log_path=nf_log_path,
        )
        if future is None:
            return False
        with self._lock:
            self._finalization_futures[batch_id] = future
        future.add_done_callback(lambda f: self._on_finalization_done(batch_id, f))
        return True

    def is_running(self, batch_id: str | None) -> bool:
        if not batch_id:
            return False
        with self._lock:
            future = self._futures.get(batch_id) or self._finalization_futures.get(batch_id)
            return bool(future and not future.done()) or self._finalization_pool.is_running(batch_id)

    # Local finalization retry backoff: 60 s doubling per consecutive
    # incomplete attempt, capped at 30 min. The count lives in memory, so a
    # dispatcher restart retries promptly once and then backs off again.
    _FINALIZATION_RETRY_BASE_SECONDS = 60
    _FINALIZATION_RETRY_MAX_SECONDS = 1800

    def _finalization_retry_delay(self, attempts: int) -> int:
        return min(
            self._FINALIZATION_RETRY_MAX_SECONDS,
            self._FINALIZATION_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
        )

    def _on_finalization_done(self, batch_id: str, future: Future) -> None:
        with self._lock:
            self._finalization_futures.pop(batch_id, None)
        error = ""
        result = None
        try:
            result = future.result()
        except Exception as exc:
            # The finalizer leaves the batch in FINALIZING so the periodic
            # retry path can replay it; do not reset its slides as if NF failed.
            log.error("Post-batch finalization for %s failed: %s", batch_id, exc)
            error = f"finalizer raised {type(exc).__name__}: {exc}"
        finally:
            # A required hook failure also returns normally, leaving the batch
            # in FINALIZING. Persist a backoff for both normal and exceptional
            # exits so a missing-output WDS batch cannot monopolize the global
            # finalization lease and starve remote-agent hooks.
            try:
                if not self._shutdown_event.is_set():
                    batch = self.state.get_batch(batch_id) or {}
                    if batch.get("status") == "FINALIZING":
                        with self._lock:
                            attempts = self._finalization_attempts.get(batch_id, 0) + 1
                            self._finalization_attempts[batch_id] = attempts
                        delay = self._finalization_retry_delay(attempts)
                        if not error:
                            error = getattr(result, "detail", "") or (
                                "waiting for a local finalization slot"
                                if result is None
                                else "required finalization incomplete"
                            )
                        defer = getattr(self.state, "defer_local_finalization_retry", None)
                        if defer is not None:
                            defer(
                                batch_id,
                                error=f"attempt {attempts}, next retry in {delay}s: {error}",
                                delay_seconds=delay,
                            )
                    else:
                        with self._lock:
                            self._finalization_attempts.pop(batch_id, None)
            except Exception:
                log.exception(
                    "Batch %s: could not record local finalization retry backoff",
                    batch_id,
                )
            finally:
                # The durable batch row now carries this attempt's result, so
                # consume the pool's outcome record. The pool refuses a new
                # attempt while a batch's outcome is unconsumed; without this,
                # _retry_finalizing_batches() never replays a FINALIZING batch
                # until the dispatcher restarts.
                self._finalization_pool.pop_outcome(batch_id)
                self._close_worker_state_connection()

    def _on_done(self, batch_id: str, future: Future):
        with self._lock:
            self._futures.pop(batch_id, None)
            self._in_flight.pop(batch_id, None)
        self._slots.release()
        self._invalidate_active_batch_count_cache()
        try:
            future.result()
        except Exception as e:
            try:
                active_rows = {
                    row["batch_id"]: row for row in self.state.get_active_batches()
                }
                batch = active_rows.get(batch_id) or {}
                shutdown_recovery = (
                    self._shutdown_event.is_set()
                    and batch.get("status") in {"RECOVERING", "FINALIZING"}
                )
                if batch_id in active_rows and not shutdown_recovery and not self.state.get_batch_nf_pid(batch_id):
                    log.warning(
                        "Batch %s died before Nextflow startup completed; resetting slides to "
                        "PENDING and marking batch FAILED.",
                        batch_id,
                    )
                    self.state.reset_dispatched_to_pending(batch_id)
                    self.state.complete_batch(batch_id, exit_code=-1)
                    self.state.skip_unstarted_batch_hooks(
                        batch_id,
                        reason="driver failed before hooks started",
                    )
            except Exception as cleanup_exc:
                log.error("Batch %s cleanup after startup failure also failed: %s", batch_id, cleanup_exc)
            log.error("Unhandled exception in batch %s: %s", batch_id, e)
        finally:
            self._close_worker_state_connection()

    def shutdown(self, wait: bool = True):
        self._shutdown_event.set()
        log.info("RunManager shutting down (wait=%s)…", wait)
        self._executor.shutdown(wait=wait)
        self._finalization_pool.shutdown(wait=wait)

    def stop_for_restart_recovery(self) -> tuple[int, int]:
        """Stop active batch work while preserving resumable state for restart recovery.

        Local batches with resumable state are left in ``RECOVERING`` (or
        ``FINALIZING`` when the NF log already shows success) so the next
        dispatcher startup routes them through ``recover_in_flight()`` instead of
        resetting their slides back to ``PENDING``. Remote-agent batches are left
        untouched and continue to rely on status-driven recovery.

        Batches without resumable local state still fall back to the historical
        fail/reset behavior.

        Returns ``(preserved, failed)`` counts.
        """
        self._shutdown_event.set()
        with self._lock:
            active = list(self._futures.keys())

        preserved = 0
        failed = 0
        for batch_id in active:
            batch = self.state.get_batch(batch_id) or {"batch_id": batch_id}
            backend = batch.get("backend")
            work_dir = batch.get("work_dir")
            csv_path = batch.get("csv_path")
            log_path = batch.get("log_path")
            batch_log_dir = os.path.dirname(log_path) if log_path else ""
            trace_path = _batch_trace_path(batch_log_dir, batch_id) if batch_log_dir else None
            nf_log_path = _batch_nf_log_path(batch_log_dir, batch_id) if batch_log_dir else None
            has_resume_state = bool(
                backend != "remote_agent"
                and work_dir and os.path.isdir(work_dir)
                and csv_path and os.path.isfile(csv_path)
            )
            # Publish the recovery decision before terminating Nextflow.  The
            # worker can return from proc.wait() immediately after SIGTERM; if
            # it observes RUNNING, its normal exit path can overwrite the
            # preservation decision with FAILED before this method gets there.
            recovery_status = None
            if has_resume_state:
                recovery_status = (
                    "FINALIZING" if _nf_log_indicates_success(nf_log_path)
                    else "RECOVERING"
                )
                self.state.set_batch_status(batch_id, recovery_status)
            elif backend != "remote_agent":
                self.state.reset_dispatched_to_pending(batch_id)
                self.state.complete_batch(batch_id, exit_code=-1)
            nf_pid = self.state.get_batch_nf_pid(batch_id)
            if nf_pid:
                try:
                    _kill_orphaned_nf(
                        batch_id,
                        nf_pid,
                        sigterm_wait=3.0,
                        trace_path=trace_path,
                        nf_log_path=nf_log_path,
                        # Re-read: an auto-resume may have renamed the run.
                        run_name=(self.state.get_batch(batch_id) or {}).get("run_name"),
                    )
                except Exception:
                    pass
            if backend == "remote_agent":
                log.info(
                    "Shutdown: preserving remote batch %s for status-driven recovery on restart",
                    batch_id,
                )
                preserved += 1
                continue
            if has_resume_state:
                if recovery_status == "FINALIZING":
                    log.info(
                        "Shutdown: preserved local batch %s for post-success finalization replay on restart",
                        batch_id,
                    )
                else:
                    log.info(
                        "Shutdown: preserved local batch %s for -resume recovery on restart",
                        batch_id,
                    )
                preserved += 1
                continue
            log.info("Shutdown: marked batch %s FAILED and reset slides to PENDING", batch_id)
            failed += 1

        return preserved, failed

    def running_count(self) -> int:
        with self._lock:
            return len(self._futures)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _nf_pid_alive(
    pid: int,
    *,
    run_name: str | None = None,
    nf_log_path: str | None = None,
) -> bool:
    """Return True only if ``pid`` is alive and is still this batch's Nextflow run.

    A recorded ``nf_pid`` outlives the process: after a reboot (or simply a
    long uptime) the kernel can hand the same PID to an unrelated process —
    possibly another batch's Nextflow. ``os.kill(pid, 0)`` alone would then
    make the dispatcher SIGTERM/SIGKILL whatever now owns that PID.

    Nextflow runs as the ``nextflow`` launcher script or the JVM it execs
    (``nextflow.cli.Launcher``); both keep the original arguments. When the
    batch's run name is known, its ``-name <run_name>`` argument must be in the
    command line; otherwise its ``-log`` file name (``batch_<id>.nf.log``)
    must be. Only a batch with neither falls back to the bare ``nextflow``
    check.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user, so it cannot be a Nextflow run we launched.
        return False
    except OSError:
        return False
    if not os.path.isdir("/proc"):
        # No procfs to verify identity (non-Linux dev host): trust liveness.
        return True
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            argv = [arg for arg in fh.read().split(b"\0") if arg]
    except OSError:
        return False
    if not any(b"nextflow" in arg.lower() for arg in argv):
        return False
    if run_name:
        token = run_name.encode()
        return any(
            arg == token and argv[i - 1] == b"-name"
            for i, arg in enumerate(argv) if i
        )
    if nf_log_path:
        log_name = os.path.basename(nf_log_path).encode()
        return any(arg.endswith(log_name) for arg in argv)
    return True


def _kill_orphaned_nf(
    batch_id: str,
    pid: int,
    sigterm_wait: float = 10.0,
    trace_path: str | None = None,
    nf_log_path: str | None = None,
    run_name: str | None = None,
) -> None:
    """SIGTERM then SIGKILL an orphaned NF process so its run name is freed.

    NF registers the run name in ~/.nextflow/history as 'active' for the lifetime
    of the process.  If the dispatcher crashes while NF is running, NF continues
    as an orphan and keeps the name locked.  Attempting to ``-resume -name <same>``
    from a new dispatcher process fails with "Run name already used".  Killing the
    orphan first releases the lock so the resume succeeds.
    """
    if not _nf_pid_alive(pid, run_name=run_name, nf_log_path=nf_log_path):
        log.debug(
            "Batch %s: orphaned NF PID %d already gone (or reused by another process)",
            batch_id, pid,
        )
        _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
        return

    log.info("Batch %s: terminating orphaned NF process PID=%d", batch_id, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
        return  # Already gone

    # Wait up to sigterm_wait seconds for graceful exit, then SIGKILL.
    deadline = time.monotonic() + sigterm_wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            log.debug("Batch %s: PID %d exited after SIGTERM", batch_id, pid)
            _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
            return
        time.sleep(0.5)

    try:
        os.kill(pid, signal.SIGKILL)
        log.warning("Batch %s: PID %d did not exit after SIGTERM; sent SIGKILL", batch_id, pid)
    except OSError:
        pass
    _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))


def recover_in_flight(state: StateStore, pending_deque: deque, retry_failed: bool = True,
                      max_slide_retries: int = 0, cleanup_work_dir: bool = False,
                      cfg: Config | None = None):
    """
    On restart, recover interrupted batches:
    - If the batch's work dir still exists, schedule a -resume run to skip already-done tasks.
    - If the work dir is gone, reset slides to PENDING for fresh re-dispatch.
    When retry_failed=True (default), also resets FAILED slides back to PENDING, unless
    their fail_count >= max_slide_retries (when max_slide_retries > 0).
    Also reconciles execution artifacts for already terminal drivers.
    Returns `(resume_specs, finalize_specs)` for deferred recovery work.
    """
    if cleanup_work_dir:
        def _spec(batch):
            work_dir = str(batch.get("work_dir") or "").strip()
            if not work_dir:
                return None
            managed_root = cfg.work_base_dir if cfg is not None else os.path.dirname(work_dir)
            return ExecutionCleanupSpec(
                batch_id=str(batch["batch_id"]),
                managed_root=managed_root,
                run_root=work_dir,
                execution_paths=(work_dir,),
                protected_paths=((cfg.outdir,) if cfg is not None else ()),
            )

        def _bg_cleanup():
            cleaned = reconcile_execution_cleanup(
                state,
                _spec,
                candidates=[
                    batch for batch in state.get_execution_cleanup_candidates()
                    if str(batch.get("backend") or "local") != "remote_agent"
                ],
            )
            if cleaned:
                log.info("Startup cleanup: reconciled %d local execution batch(es)", cleaned)

        threading.Thread(
            target=_bg_cleanup, name="startup-execution-cleanup", daemon=True
        ).start()

    resume_specs = []
    remote_resume_specs = []
    finalize_specs = []
    active_batches = state.get_active_batches()
    if active_batches:
        log.info("Recovering %d in-flight batch(es) from previous run…", len(active_batches))
        for batch in active_batches:
            batch_id = batch["batch_id"]
            work_dir = batch.get("work_dir")
            csv_path = batch.get("csv_path")
            log_path = batch.get("log_path")
            if batch.get("backend") == "remote_agent":
                if csv_path and work_dir:
                    state.set_batch_status(batch_id, "RECOVERING")
                    log.info("  Batch %s: remote_agent batch — will resume remote status polling", batch_id)
                    # Passive remote polling must not consume the startup cap
                    # ahead of resumable local Nextflow work.
                    remote_resume_specs.append((batch_id, csv_path, work_dir))
                else:
                    log.info("  Batch %s: remote_agent batch has no local metadata — resetting slides", batch_id)
                    state.reset_dispatched_to_pending(batch_id)
                    state.complete_batch(batch_id, exit_code=-1)
                continue
            batch_log_dir = os.path.dirname(log_path) if log_path else ""
            trace_path = _batch_trace_path(batch_log_dir, batch_id) if batch_log_dir else None
            nf_log_path = _batch_nf_log_path(batch_log_dir, batch_id) if batch_log_dir else None
            run_name = batch.get("run_name")

            # FINALIZING is only written after Nextflow succeeded. Finalization
            # reads the batch CSV and the outdir, never the NF work dir — which
            # cleanup_work_dir deletes as soon as Nextflow exits 0 — so replay
            # it whenever the batch CSV survives instead of treating the
            # missing work dir as a lost run. Replay may re-run a WDS hook
            # whose append finished before its hook row was marked SUCCEEDED;
            # that is safe: append_wds skips slides already indexed with the
            # same content_sha1 and a locked append strips copies left by an
            # interrupted one (tests/test_wds_integrity.py).
            if batch.get("status") == "FINALIZING" and csv_path and os.path.isfile(csv_path):
                if cfg is None:
                    log.warning(
                        "  Batch %s: FINALIZING but no config was provided for recovery "
                        "finalization — leaving it for a full startup replay",
                        batch_id,
                    )
                    continue
                nf_pid = batch.get("nf_pid")
                if nf_pid and _nf_pid_alive(nf_pid, run_name=run_name, nf_log_path=nf_log_path):
                    _kill_orphaned_nf(batch_id, nf_pid, trace_path=trace_path, nf_log_path=nf_log_path,
                                      run_name=run_name)
                finalize_specs.append((
                    batch_id,
                    csv_path,
                    log_path,
                    work_dir,
                    _batch_run_started_at(batch),
                    nf_log_path,
                ))
                log.info(
                    "  Batch %s: FINALIZING after Nextflow success — queued post-success "
                    "finalization replay",
                    batch_id,
                )
                continue

            has_resume_state = bool(
                work_dir and os.path.isdir(work_dir) and csv_path and os.path.isfile(csv_path)
            )
            batch_age_hours = None
            try:
                from datetime import timezone as _tz
                id_ts = datetime.strptime(batch_id[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_tz.utc)
                batch_age_hours = (datetime.now(_tz.utc) - id_ts).total_seconds() / 3600
            except (ValueError, KeyError):
                pass  # batch_id doesn't match expected format; proceed normally

            # If nf_pid is missing or dead but the work dir and batch CSV still
            # exist, prefer a Nextflow -resume over resetting the slides. This
            # covers dispatcher interruptions that happen after a batch has
            # started doing real work but before the runner thread can clean up.
            nf_pid = batch.get("nf_pid")
            if not nf_pid:
                if has_resume_state and _nf_log_indicates_success(nf_log_path):
                    if cfg is None:
                        log.warning(
                            "  Batch %s: NF log shows successful completion but no config was "
                            "provided for recovery finalization — leaving RUNNING for a full startup replay",
                            batch_id,
                        )
                        continue
                    state.set_batch_status(batch_id, "FINALIZING")
                    finalize_specs.append((
                        batch_id,
                        csv_path,
                        log_path,
                        work_dir,
                        _batch_run_started_at(batch),
                        nf_log_path,
                    ))
                    log.info(
                        "  Batch %s: NF log shows successful completion and no NF PID is recorded "
                        "— queued post-success finalization replay",
                        batch_id,
                    )
                    continue
                if batch_age_hours is not None and batch_age_hours > 24:
                    log.info("  Batch %s: stale (created %.1fh ago) — resetting slides to PENDING",
                             batch_id, batch_age_hours)
                    state.reset_dispatched_to_pending(batch_id)
                    state.complete_batch(batch_id, exit_code=-1)
                    continue
                if has_resume_state:
                    state.set_batch_status(batch_id, "RECOVERING")
                    _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
                    log.info("  Batch %s: no nf_pid recorded but work dir is intact — will resume with -resume",
                             batch_id)
                    resume_specs.append((batch_id, csv_path, work_dir))
                    continue
                log.info("  Batch %s: no nf_pid recorded and no resumable state — resetting slides to PENDING",
                         batch_id)
                state.reset_dispatched_to_pending(batch_id)
                state.complete_batch(batch_id, exit_code=-1)
                continue

            nf_alive = _nf_pid_alive(nf_pid, run_name=run_name, nf_log_path=nf_log_path)

            if has_resume_state:
                # Kill any orphaned NF process from the previous dispatcher run.
                # Without this, NF refuses to resume with "-name <same_name>" because the
                # original process still holds the run name in ~/.nextflow/history.
                if nf_alive:
                    _kill_orphaned_nf(batch_id, nf_pid, trace_path=trace_path, nf_log_path=nf_log_path,
                                      run_name=run_name)
                    log.info("  Batch %s: work dir intact and NF PID=%d is alive — will resume with -resume",
                             batch_id, nf_pid)
                else:
                    if _nf_log_indicates_success(nf_log_path):
                        if cfg is None:
                            log.warning(
                                "  Batch %s: NF log shows successful completion but no config was "
                                "provided for recovery finalization — leaving RUNNING for a full startup replay",
                                batch_id,
                            )
                            continue
                        state.set_batch_status(batch_id, "FINALIZING")
                        finalize_specs.append((
                            batch_id,
                            csv_path,
                            log_path,
                            work_dir,
                            _batch_run_started_at(batch),
                            nf_log_path,
                        ))
                        log.info(
                            "  Batch %s: NF PID=%d is dead and NF log shows successful completion "
                            "— queued post-success finalization replay",
                            batch_id, nf_pid,
                        )
                        continue
                    if batch_age_hours is not None and batch_age_hours > 24:
                        log.info("  Batch %s: stale (created %.1fh ago) — resetting slides to PENDING",
                                 batch_id, batch_age_hours)
                        state.reset_dispatched_to_pending(batch_id)
                        state.complete_batch(batch_id, exit_code=-1)
                        continue
                    _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
                    log.info("  Batch %s: NF PID=%d is dead but work dir is intact — will resume with -resume",
                             batch_id, nf_pid)
                state.set_batch_status(batch_id, "RECOVERING")
                resume_specs.append((batch_id, csv_path, work_dir))
                # Leave slides in DISPATCHED state; they'll transition to DONE/FAILED on resume completion
            else:
                if nf_alive:
                    _kill_orphaned_nf(batch_id, nf_pid, trace_path=trace_path, nf_log_path=nf_log_path,
                                      run_name=run_name)
                else:
                    _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
                log.info("  Batch %s: work dir missing — resetting slides to PENDING", batch_id)
                state.reset_dispatched_to_pending(batch_id)
                state.complete_batch(batch_id, exit_code=-1)

    if retry_failed:
        n = state.reset_failed_to_pending(max_retries=max_slide_retries)
        if n:
            log.info("retry_failed: reset %d FAILED slide(s) to PENDING.", n)

    # Re-enqueue pending slides incrementally so startup does not hammer the DB
    # when a very large backlog exists. The periodic DB sweep will continue
    # draining the rest after the scheduler is live.
    startup_limit = int(
        os.environ.get(
            "MUSSEL_STARTUP_REQUEUE_MAX_PENDING",
            str(_STARTUP_REQUEUE_MAX_PENDING),
        )
    )
    pending_rows = state.get_pending_slides(limit=startup_limit + 1)
    startup_limited = len(pending_rows) > startup_limit
    pending = pending_rows[:startup_limit] if startup_limited else pending_rows
    for slide in pending:
        pending_deque.append(slide)
    if pending:
        log.info("Re-enqueued %d slide(s) for dispatch.", len(pending))
    if startup_limited:
        log.warning(
            "Startup requeue limited to %d+ pending slide(s); remaining backlog will be recovered by scheduler sweeps",
            len(pending),
        )

    return resume_specs + remote_resume_specs, finalize_specs


def _pid_is_alive(pid: int) -> bool:
    proc_path = f"/proc/{pid}"
    if os.path.isdir("/proc") and not os.path.exists(proc_path):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_lock_pid(path: str) -> int | None:
    try:
        with open(path) as fh:
            return int(fh.read().strip().split()[0])
    except (FileNotFoundError, ValueError, OSError, IndexError):
        return None


def _resolve_lock_pid(payload: str = "") -> int:
    """Best-effort externally visible PID for lockfile metadata.

    Under systemd, the dispatcher's effective process identity should match the
    process list and dashboard health checks. Prefer the single live dispatcher
    discovered for the same config; otherwise fall back to the current PID.
    """
    env_pid = os.environ.get("MUSSEL_LOCK_PID", "").strip()
    if env_pid.isdigit():
        return int(env_pid)
    if payload:
        try:
            matches = _find_same_config_dispatchers(payload)
            if len(matches) == 1:
                return matches[0][0]
        except Exception:
            pass
    return os.getpid()


def _find_same_config_dispatchers(config_path: str, *, exclude_pid: int | None = None) -> list[tuple[int, str]]:
    """Return live dispatcher-like processes running against the same config."""
    config_path = os.path.realpath(config_path)
    scan = subprocess.run(
        ["pgrep", "-f", "mussel_dispatcher"],
        capture_output=True, text=True,
    )
    pids = [
        int(pid)
        for pid in scan.stdout.split()
        if pid.strip().isdigit() and int(pid) != (exclude_pid or -1)
    ]
    matches: list[tuple[int, str]] = []
    for pid in pids:
        try:
            raw_cmd = Path(f"/proc/{pid}/cmdline").read_text().replace("\x00", " ").strip()
            if "mussel_dispatcher" not in raw_cmd or "tower" in raw_cmd or "dashboard" in raw_cmd:
                continue
            exe = os.path.basename(os.readlink(f"/proc/{pid}/exe"))
            if not exe.startswith("python"):
                continue
            parts = raw_cmd.split()
            if not parts:
                continue
            other_cfg = parts[-1]
            if not os.path.isabs(other_cfg):
                cwd = os.readlink(f"/proc/{pid}/cwd")
                other_cfg = os.path.join(cwd, other_cfg)
            if os.path.realpath(other_cfg) == config_path:
                matches.append((pid, raw_cmd[:160]))
        except OSError:
            continue
    return matches


def _acquire_pid_lock(path: str, *, label: str, payload: str = "") -> None:
    """Atomically acquire a PID lock, removing it only when the PID is stale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                lock_pid = _resolve_lock_pid(payload)
                body = f"{lock_pid} {payload}\n".strip() + "\n"
                os.write(fd, body.encode())
            finally:
                os.close(fd)
            log.info("%s lockfile written: %s (PID=%d)", label, path, lock_pid)
            return
        except FileExistsError:
            old_pid = _read_lock_pid(path)
            if old_pid and _pid_is_alive(old_pid):
                log.error(
                    "%s is already locked by PID %d (%s). Stop that dispatcher first, "
                    "or remove the lockfile if it is stale.",
                    label, old_pid, path,
                )
                sys.exit(1)
            if old_pid:
                log.warning("Stale %s lockfile (PID %d no longer running). Removing: %s",
                            label, old_pid, path)
            else:
                log.warning("Unreadable stale %s lockfile. Removing: %s", label, path)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _release_lock(path: str, *, payload: str = "") -> None:
    try:
        lock_pid = _read_lock_pid(path)
        valid_pids = {os.getpid()}
        resolved_pid = _resolve_lock_pid(payload)
        if resolved_pid:
            valid_pids.add(resolved_pid)
        if lock_pid in valid_pids:
            os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 else 1)

    # Subcommands:
    #   hooks retry|waive <config.yaml> <batch-id> [--index N]
    #   hooks pause|resume <config.yaml> [--target NAME] [--reason TEXT]
    if sys.argv[1] == "hooks":
        if len(sys.argv) < 3 or sys.argv[2] not in {"retry", "waive", "pause", "resume"}:
            print(
                "Usage: mussel-dispatcher hooks retry|waive <config.yaml> <batch-id> "
                "[--index N] [--reason TEXT]\n"
                "       mussel-dispatcher hooks pause|resume <config.yaml> "
                "[--target NAME] [--reason TEXT]",
                file=sys.stderr,
            )
            sys.exit(1)
        if sys.argv[2] in {"pause", "resume"}:
            if len(sys.argv) < 4:
                print("A config path is required", file=sys.stderr)
                sys.exit(1)
            action, config_arg = sys.argv[2:4]
            target = ""
            reason = "operator action"
            args = list(sys.argv[4:])
            while args:
                flag = args.pop(0)
                if flag == "--target" and args:
                    target = args.pop(0)
                elif flag == "--reason" and args:
                    reason = args.pop(0)
                else:
                    print(f"Unknown hooks option: {flag}", file=sys.stderr)
                    sys.exit(1)
            cfg = Config.load(config_arg)
            target = target or cfg.default_dispatch_target or "local"
            state = StateStore.from_config(cfg)
            if action == "pause":
                state.pause_hook_circuit(target, reason=reason)
            else:
                state.begin_hook_canary(target, reason=reason)
            print(f"{action} requested for hook target {target}")
            sys.exit(0)
        if len(sys.argv) < 5:
            print("A config path and batch ID are required", file=sys.stderr)
            sys.exit(1)
        action, config_arg, batch_id = sys.argv[2:5]
        hook_index = None
        reason = "operator action"
        args = list(sys.argv[5:])
        while args:
            flag = args.pop(0)
            if flag == "--index" and args:
                hook_index = int(args.pop(0))
            elif flag == "--reason" and args:
                reason = args.pop(0)
            else:
                print(f"Unknown hooks option: {flag}", file=sys.stderr)
                sys.exit(1)
        cfg = Config.load(config_arg)
        state = StateStore.from_config(cfg)
        rows = state.get_batch_hooks([batch_id]).get(batch_id, [])
        selected = [
            row for row in rows
            if hook_index is None or int(row["hook_index"]) == hook_index
        ]
        if hook_index is not None and not selected:
            print(f"No hook {hook_index} found for batch {batch_id}", file=sys.stderr)
            sys.exit(1)
        if hook_index is None:
            selected = [row for row in selected if row.get("status") == "FAILED"]
        if not selected:
            print(f"No failed hooks selected for batch {batch_id}")
            sys.exit(0)
        for row in selected:
            index = int(row["hook_index"])
            if action == "retry":
                state.retry_batch_hook(batch_id, index)
            else:
                state.waive_batch_hook(batch_id, index, reason=reason)
        state.set_batch_status(batch_id, "FINALIZING")
        batch = state.get_batch(batch_id) or {}
        if batch.get("backend") == "remote_agent":
            state.reopen_remote_finalization(batch_id)
        print(f"{action} requested for {len(selected)} hook(s) on {batch_id}")
        sys.exit(0)

    # Subcommand: collect-manifests <config.yaml>
    if sys.argv[1] == "collect-manifests":
        if len(sys.argv) < 3:
            print("Usage: mussel-dispatcher.py collect-manifests <config.yaml>", file=sys.stderr)
            sys.exit(1)
        cfg = Config.load(sys.argv[2])
        combined_path = cfg.resolved_combined_manifest_path()
        n = collect_manifests(cfg.outdir, combined_path)
        print(f"Combined manifest written to {combined_path} ({n} rows)")
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    config_arg = sys.argv[1]
    config_path = os.path.realpath(config_arg)
    cfg = Config.load(config_arg)
    log.info("Configuration loaded from %s", config_arg)

    # Refuse to run on a pinned (content-addressed) container image that isn't the file
    # its name promises: it would silently change every embedding.
    from .image_pins import verify_pinned_images
    image_problems = verify_pinned_images([cfg.nextflow_config])
    if image_problems:
        for problem in image_problems:
            log.error("Pinned container image mismatch: %s", problem)
        sys.exit(1)

    # Refuse to run a revision that is not on the deployment's release branch: a
    # checkout moved to main would otherwise run unvetted pipeline code here.
    from .release_pin import verify_release_pin
    pin_problem, pin_note = verify_release_pin(cfg)
    if pin_problem:
        log.error("Refusing to start: %s", pin_problem)
        sys.exit(1)
    if pin_note:
        log.info("Release pin: %s", pin_note)

    # Auto-derive tower_endpoint from tower_proxy_port if not explicitly set.
    if not cfg.tower_endpoint and cfg.tower_proxy_port:
        cfg.tower_endpoint = f"http://localhost:{cfg.tower_proxy_port}"
        log.info("Auto-derived tower_endpoint: %s", cfg.tower_endpoint)

    os.makedirs(cfg.work_base_dir, exist_ok=True)
    os.makedirs(cfg.dispatch_dir, exist_ok=True)
    os.makedirs(cfg.state_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    dispatcher_log_path = _enable_persistent_dispatcher_logging(cfg)
    log.info("Persistent dispatcher log: %s", dispatcher_log_path)

    # PID lockfile — prevent two dispatcher instances from running simultaneously.
    # Use O_CREAT|O_EXCL for the initial write to atomically detect races.
    pid_lock_path = os.path.join(cfg.state_dir, "dispatcher.pid")

    rogue = _find_same_config_dispatchers(config_path, exclude_pid=os.getpid())
    if rogue:
        log.error(
            "Found %d existing mussel_dispatcher process(es) already running for config %s: %s",
            len(rogue),
            config_path,
            ", ".join(f"PID {pid} ({cmd}...)" for pid, cmd in rogue),
        )
        sys.exit(1)

    _acquire_pid_lock(pid_lock_path, label="PID", payload=config_path)

    cohort_lock_paths: list[str] = []
    if cfg.allow_shared_cohort:
        log.warning("allow_shared_cohort=true: skipping shared cohort lock")
    else:
        lock_dir = cfg.resolved_cohort_lock_dir()
        for key in cfg.cohort_lock_keys():
            lock_path = os.path.join(lock_dir, Config.lock_filename_for_key(key))
            _acquire_pid_lock(lock_path, label=f"cohort {key}", payload=config_path)
            cohort_lock_paths.append(lock_path)

    # Launch dashboard and tower_proxy sidecars (ports from config).
    sidecar_procs = _start_sidecars(cfg, config_arg)

    # Load manifest for priority dispatch (slides NOT in manifest get priority=1).
    manifest_id_set: frozenset | None = None
    if cfg.manifest_path:
        import csv as _csv
        try:
            _ids: set[str] = set()
            with open(cfg.manifest_path, newline="") as _f:
                for _row in _csv.reader(_f):
                    if _row:
                        _ids.add(_row[cfg.manifest_id_column].strip())
            manifest_id_set = frozenset(_ids)
            log.info("Loaded manifest: %s (%d IDs) — slides not in manifest get priority=1",
                     cfg.manifest_path, len(manifest_id_set))
        except Exception as _e:
            log.warning("Could not load manifest %s: %s — priority dispatch disabled", cfg.manifest_path, _e)

    state = StateStore.from_config(cfg, manifest_id_set=manifest_id_set)
    log.info("State store: %s", state.db_cfg.location())

    if _startup_priority_reconciliation_enabled():
        _apply_manifest_priority_update(state, manifest_id_set)
    elif manifest_id_set is not None:
        log.info(
            "Skipping manifest priority reconciliation on startup; "
            "set MUSSEL_ENABLE_STARTUP_PRIORITY_RECONCILIATION=1 for a planned maintenance run"
        )

    stop_event = threading.Event()
    _start_release_controller(cfg, state, stop_event)
    _start_status_snapshot_publisher(cfg, state, stop_event)

    # Signal handling — first Ctrl+C: graceful stop; second Ctrl+C: immediate exit
    def _handle_signal(signum, frame):
        sys.stderr.write(f"\nReceived signal {signum}, shutting down gracefully…\n")
        sys.stderr.write("Press Ctrl+C again to force exit immediately.\n")
        stop_event.set()
        # Restore default handlers so a second signal exits immediately
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # RunManager
    run_manager = RunManager(cfg, state)

    pending_deque: deque = deque()

    # Recovery
    resume_specs, finalize_specs = recover_in_flight(
        state, pending_deque, cfg.retry_failed, cfg.max_slide_retries,
        cleanup_work_dir=cfg.cleanup_work_dir, cfg=cfg,
    )
    recovery_actions = [
        ("resume", spec) for spec in resume_specs
    ] + [
        ("finalize", spec) for spec in finalize_specs
    ]
    if recovery_actions:
        log.info(
            "Recovery summary: %d resume batch(es), %d finalization batch(es); max_concurrent_runs=%d",
            len(resume_specs), len(finalize_specs), _execution_slot_count(cfg),
        )
    else:
        log.info("Recovery summary: no deferred recovery work found")

    # Nextflow resumes consume execution capacity; post-success finalization
    # uses its own executor and durable lease, so it must not be dropped merely
    # because the resume list is larger than max_concurrent_runs.
    # The cap must match RunManager's slot count (the summed execution pool
    # limits); a larger cap makes submit_resume() refuse the overflow, leaving
    # those batches RECOVERING with no runner.
    resume_cap = _execution_slot_count(cfg)
    capped_resumes, excess_resumes = _split_startup_resumes(cfg, resume_specs)
    for batch_id, csv_path, work_dir in excess_resumes:
        batch = state.get_batch(batch_id) or {}
        if batch.get("backend") == "remote_agent":
            log.warning(
                "Startup cap: leaving remote recovery batch %s active even though it exceeds "
                "max_concurrent_runs (effective)=%d; remote agent heartbeats remain authoritative until a "
                "resume slot is available",
                batch_id,
                resume_cap,
            )
            continue
        log.info(
            "Startup cap: resetting recovery batch %s slides to PENDING "
            "(exceeds max_concurrent_runs (effective)=%d)",
            batch_id, resume_cap,
        )
        state.reset_dispatched_to_pending(batch_id)
        state.complete_batch(batch_id, exit_code=-1)
    submitted_resumes = 0
    submitted_finalizations = 0
    for batch_id, csv_path, work_dir in capped_resumes:
        if run_manager.submit_resume(batch_id, csv_path, work_dir):
            submitted_resumes += 1
        else:
            log.error("Recovery submission failed for resume batch %s", batch_id)
    for batch_id, csv_path, log_path, work_dir, run_started_at, nf_log_path in finalize_specs:
        if run_manager.submit_finalize_recovered_success(
            batch_id,
            csv_path,
            log_path,
            work_dir,
            run_started_at=run_started_at,
            nf_log_path=nf_log_path,
        ):
            submitted_finalizations += 1
        else:
            log.error("Recovery submission failed for finalization batch %s", batch_id)
    if submitted_resumes or submitted_finalizations:
        log.info(
            "Submitted %d recovery resume(s) and %d recovery finalization(s).",
            submitted_resumes, submitted_finalizations,
        )
    if cfg.cleanup_work_dir:
        log.info("Startup cleanup enabled: orphaned finished work dirs will be removed in the background")

    # BatchScheduler
    scheduler = BatchScheduler(cfg, state, run_manager, stop_event)

    # Feed pending_deque into scheduler via a bridge thread
    def _bridge():
        while not stop_event.is_set():
            while pending_deque:
                scheduler.enqueue(pending_deque.popleft())
            time.sleep(1)

    bridge_thread = threading.Thread(target=_bridge, daemon=True, name="bridge")
    bridge_thread.start()

    # Watchers
    watchers = []
    for w_cfg in cfg.watchers:
        if w_cfg.type == "local":
            watcher = LocalWatcher(w_cfg, pending_deque, state, stop_event)
        elif w_cfg.type == "s3":
            watcher = S3Watcher(w_cfg, pending_deque, state, stop_event)
        elif w_cfg.type == "tcga":
            watcher = TcgaWatcher(w_cfg, pending_deque, state, stop_event, cfg.repo_dir, cfg.outdir,
                                   max_slide_retries=cfg.max_slide_retries)
        elif w_cfg.type == "databricks":
            watcher = DatabricksWatcher(w_cfg, pending_deque, state, stop_event)
        else:
            log.warning("Unknown watcher type '%s', skipping.", w_cfg.type)
            continue
        watcher.start()
        watchers.append(watcher)

    if not watchers:
        log.error("No watchers configured. Exiting.")
        for lock_path in cohort_lock_paths:
            _release_lock(lock_path, payload=config_path)
        try:
            os.unlink(pid_lock_path)
        except OSError:
            pass
        sys.exit(1)

    if _startup_priority_reconciliation_enabled():
        _start_startup_priority_sync(cfg, state, stop_event)
    else:
        log.info("Skipping Databricks startup priority sync; live watcher updates remain enabled")
    log.info("mussel-dispatcher running with %d watcher(s). PID=%d", len(watchers), os.getpid())

    # Run scheduler in main thread (blocks until stop_event)
    scheduler.run()

    # Graceful shutdown: stop active work but preserve resumable batches so the
    # next startup drives them through recover_in_flight() instead of wiping
    # local progress.
    n = run_manager.running_count()
    if n:
        log.info("Graceful shutdown: stopping %d active batch worker(s) for restart recovery…", n)
        preserved, failed = run_manager.stop_for_restart_recovery()
        log.info(
            "Graceful shutdown: preserved %d batch(es) for restart recovery; "
            "marked %d batch(es) FAILED and reset slides to PENDING.",
            preserved, failed,
        )
    run_manager.shutdown(wait=False)
    log.info("mussel-dispatcher stopped.")
    for name, proc, persistent in sidecar_procs:
        if persistent:
            log.info("Leaving persistent sidecar running: %s (PID=%d)", name, proc.pid)
            continue
        proc.terminate()
    try:
        _release_lock(pid_lock_path, payload=config_path)
    except OSError:
        pass
    for lock_path in cohort_lock_paths:
        _release_lock(lock_path, payload=config_path)


if __name__ == "__main__":
    main()
