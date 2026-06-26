"""BatchScheduler, RunManager, and main() for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
import csv
import re
import signal
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
from .runner import NextflowRunner, collect_manifests, MANIFEST_HEADER

log = logging.getLogger("mussel-dispatcher")

_SLURM_JOB_ID_RE = re.compile(r"\bjobId:\s*(\d+)\b")
_SLURM_PROCESSES = (
    "MUSSEL:EXTRACT_FEATURES:TESSELLATE",
    "MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH",
)


import socket as _socket


def _port_in_use(port: int) -> bool:
    """Return True if *port* is already bound on localhost."""
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _launch_sidecar(name: str, cmd: list[str]) -> subprocess.Popen:
    """Launch a sidecar process (dashboard / tower_proxy), logging its output."""
    log.info("Launching %s: %s", name, " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _start_sidecars(cfg: Config, config_path: str) -> list[subprocess.Popen]:
    """Start dashboard and tower_proxy subprocesses using ports from config.

    Both are optional — if dashboard_port is 0 neither is started.
    If a port is already in use (e.g. from a previous dispatcher instance that
    was killed without taking its children), we reuse the existing process
    instead of launching a new one that would immediately fail to bind.
    Returns the list of started Popen objects so main() can terminate them on exit.
    """
    if not cfg.dashboard_port:
        return []

    python = sys.executable
    procs: list[subprocess.Popen] = []

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
        ])
        procs.append(dashboard)

    if cfg.tower_proxy_port:
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
            ])
            procs.append(proxy)

    return procs


def _batch_trace_path(log_dir: str, batch_id: str) -> str:
    return os.path.join(log_dir, f"batch_{batch_id}.trace.tsv")


def _batch_nf_log_path(log_dir: str, batch_id: str) -> str:
    return os.path.join(log_dir, f"batch_{batch_id}.nf.log")


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
        # Rotating batch-size index for natural stagger via alternating sizes.
        self._batch_size_idx = 0

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
            log.debug("Pending queue size: %d", len(self._pending))

    # How often to sweep the DB for PENDING slides that fell out of the deque.
    # This catches slides reset to PENDING by _verify_wds_coverage (or similar)
    # that weren't re-added to the in-memory dispatch queue.
    _DB_SWEEP_INTERVAL = 60  # seconds

    # How often to reset retriable FAILED slides back to PENDING.
    # Complements the startup-time retry so slides failed mid-run (e.g. due to
    # node failures) get re-queued without requiring a dispatcher restart.
    _RETRY_FAILED_INTERVAL = 600  # seconds (10 minutes)

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
        _last_db_sweep = time.monotonic()
        _last_retry_failed = time.monotonic()
        _last_eta_log = time.monotonic()
        _last_watchdog = time.monotonic()
        _last_failed_workdir_cleanup = time.monotonic()
        _ETA_LOG_INTERVAL = 1800  # log ETA every 30 minutes
        while not self.stop_event.is_set():
            self._maybe_dispatch()
            now = time.monotonic()
            if now - _last_db_sweep >= self._DB_SWEEP_INTERVAL:
                _last_db_sweep = now
                self._requeue_db_pending()
            if now - _last_retry_failed >= self._RETRY_FAILED_INTERVAL:
                _last_retry_failed = now
                self._retry_failed_slides()
            if now - _last_eta_log >= _ETA_LOG_INTERVAL:
                _last_eta_log = now
                self._log_eta()
            if now - _last_watchdog >= self._WATCHDOG_INTERVAL:
                _last_watchdog = now
                self._watchdog_stuck_batches()
            if now - _last_failed_workdir_cleanup >= self._FAILED_WORKDIR_CLEANUP_INTERVAL:
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
        in_flight_ids = self.run_manager.in_flight_slide_ids
        excluded = in_deque_ids | in_flight_ids
        pending_in_db = self.state.get_pending_slides()
        newly_enqueued = 0
        for slide in pending_in_db:
            if slide.get("slide_id") not in excluded:
                self.enqueue(slide)
                newly_enqueued += 1
        if newly_enqueued:
            log.info("BatchScheduler: re-enqueued %d PENDING slide(s) from DB (missed by watcher)",
                     newly_enqueued)

    def _retry_failed_slides(self):
        """Periodically reset retriable FAILED slides back to PENDING.

        Mirrors the startup-time retry in recover_in_flight() so that slides
        failed mid-run (e.g. due to SLURM node failures with errorStrategy=ignore)
        are re-queued without requiring a dispatcher restart.
        """
        if not self.cfg.retry_failed:
            return
        n = self.state.reset_failed_to_pending(max_retries=self.cfg.max_slide_retries)
        if n:
            log.info("BatchScheduler: reset %d FAILED slide(s) to PENDING (periodic retry sweep)", n)
            self._requeue_db_pending()

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
        if timeout_hours <= 0:
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
            log_path = batch.get("log_path") or os.path.join(self.cfg.log_dir, f"batch_{batch_id}.log")
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
                )

    def _cleanup_failed_work_dirs(self):
        """Periodically remove orphaned work dirs for FAILED batches.

        When the dispatcher is SIGKILL'd during active runs, the runner threads
        never reach _cleanup(), leaving work dirs on disk indefinitely.  This
        sweep catches those orphans once per hour.  Runs in a background daemon
        thread to avoid blocking the scheduler loop on slow GPFS deletes.
        """
        import shutil
        import threading as _threading

        finished = self.state.get_finished_batches_with_work_dirs()
        failed = [b for b in finished if b.get("status") == "FAILED"]
        if not failed:
            return

        def _bg():
            removed = 0
            for batch in failed:
                work_dir = batch.get("work_dir")
                if work_dir and os.path.isdir(work_dir):
                    shutil.rmtree(work_dir, ignore_errors=True)
                    try:
                        os.rmdir(os.path.dirname(work_dir))
                    except OSError:
                        pass
                    removed += 1
            if removed:
                log.info("Periodic cleanup: removed %d orphaned work dir(s) for FAILED batches",
                         removed)

        t = _threading.Thread(target=_bg, name="failed-workdir-cleanup", daemon=True)
        t.start()

    def _maybe_dispatch(self, force: bool = False):
        # Don't submit a new batch if we're already at the concurrency limit.
        # Uses a BoundedSemaphore (not len(_futures)) to avoid the TOCTOU race where
        # fast-fail resumes release their slot between the check and submit().
        if not self.run_manager.has_fresh_dispatch_capacity():
            return
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
            log.info("_maybe_dispatch: generated batch_id=%s for %d slides", batch_id, len(batch))
            if not self.run_manager.submit(batch_id, batch):
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

class RunManager:
    def __init__(self, cfg: Config, state: StateStore):
        self.cfg = cfg
        self.state = state
        self._executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent_runs,
                                            thread_name_prefix="nextflow-run")
        self._shutdown_event = threading.Event()
        self._futures: dict = {}
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
        self._slots = threading.BoundedSemaphore(cfg.max_concurrent_runs)

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

    def has_fresh_dispatch_capacity(self) -> bool:
        """Return True when both in-process and persisted batch state have room.

        The semaphore only knows about futures owned by this Python process.  If
        the dispatcher is killed after launching Nextflow, the replacement
        process can otherwise start fresh batches while old RUNNING rows and
        their SLURM children still exist.  Treat the DB as authoritative for
        fresh dispatch and let startup recovery decide whether to resume or reset
        existing RUNNING rows.
        """
        running = self.state.count_running_batches()
        if running >= self.cfg.max_concurrent_runs:
            log.debug(
                "Fresh dispatch blocked: %d RUNNING batch(es) in DB >= max_concurrent_runs=%d",
                running,
                self.cfg.max_concurrent_runs,
            )
            return False
        return self.has_capacity()

    def submit(self, batch_id: str, slides: list) -> bool:
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

        running = self.state.count_running_batches()
        if running >= self.cfg.max_concurrent_runs:
            log.warning(
                "Refusing fresh batch %s: %d RUNNING batch(es) already in DB "
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
        runner = NextflowRunner(
            self.cfg, batch_id, slides, self.state, shutdown_event=self._shutdown_event
        )
        with self._lock:
            self._in_flight[batch_id] = slide_ids
            future: Future = self._executor.submit(runner.run)
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
        # Resume batches have slides already DISPATCHED in DB — no in-flight tracking needed.
        runner = NextflowRunner(self.cfg, batch_id, [], self.state,
                                resume=True, existing_csv_path=csv_path,
                                existing_work_dir=work_dir,
                                shutdown_event=self._shutdown_event)
        with self._lock:
            future: Future = self._executor.submit(runner.run)
            self._futures[batch_id] = future
        future.add_done_callback(lambda f: self._on_done(batch_id, f))
        return True

    def _on_done(self, batch_id: str, future: Future):
        with self._lock:
            self._futures.pop(batch_id, None)
            self._in_flight.pop(batch_id, None)
        self._slots.release()
        try:
            future.result()
        except Exception as e:
            log.error("Unhandled exception in batch %s: %s", batch_id, e)

    def shutdown(self, wait: bool = True):
        self._shutdown_event.set()
        log.info("RunManager shutting down (wait=%s)…", wait)
        self._executor.shutdown(wait=wait)

    def kill_all_and_mark_failed(self) -> int:
        """Kill all active NF processes and mark their batches FAILED in the DB.

        Called during graceful SIGTERM shutdown so that the DB is left in a
        clean state (no RUNNING entries).  The next dispatcher startup will find
        no in-flight batches and simply re-dispatch from PENDING slides.

        Returns the number of batches that were marked FAILED.
        """
        self._shutdown_event.set()
        with self._lock:
            active = list(self._futures.keys())

        killed = 0
        for batch_id in active:
            nf_pid = self.state.get_batch_nf_pid(batch_id)
            if nf_pid:
                try:
                    _kill_orphaned_nf(
                        batch_id,
                        nf_pid,
                        sigterm_wait=3.0,
                        trace_path=_batch_trace_path(self.cfg.log_dir, batch_id),
                        nf_log_path=_batch_nf_log_path(self.cfg.log_dir, batch_id),
                    )
                except Exception:
                    pass
            self.state.reset_dispatched_to_pending(batch_id)
            self.state.complete_batch(batch_id, exit_code=-1)
            log.info("Shutdown: marked batch %s FAILED and reset slides to PENDING", batch_id)
            killed += 1

        return killed

    def running_count(self) -> int:
        with self._lock:
            return len(self._futures)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _kill_orphaned_nf(
    batch_id: str,
    pid: int,
    sigterm_wait: float = 10.0,
    trace_path: str | None = None,
    nf_log_path: str | None = None,
) -> None:
    """SIGTERM then SIGKILL an orphaned NF process so its run name is freed.

    NF registers the run name in ~/.nextflow/history as 'active' for the lifetime
    of the process.  If the dispatcher crashes while NF is running, NF continues
    as an orphan and keeps the name locked.  Attempting to ``-resume -name <same>``
    from a new dispatcher process fails with "Run name already used".  Killing the
    orphan first releases the lock so the resume succeeds.
    """
    try:
        os.kill(pid, 0)  # Check if process exists
    except OSError:
        log.debug("Batch %s: orphaned NF PID %d already gone", batch_id, pid)
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
                      max_slide_retries: int = 0, cleanup_work_dir: bool = False):
    """
    On restart, recover interrupted batches:
    - If the batch's work dir still exists, schedule a -resume run to skip already-done tasks.
    - If the work dir is gone, reset slides to PENDING for fresh re-dispatch.
    When retry_failed=True (default), also resets FAILED slides back to PENDING, unless
    their fail_count >= max_slide_retries (when max_slide_retries > 0).
    Also cleans up any orphaned work dirs for already-finished batches (can happen when
    shutil.rmtree silently fails because the NF process was still running at cleanup time).
    Returns a list of (batch_id, csv_path, work_dir) tuples to be submitted as resume runs.
    """
    import shutil
    import threading

    # Purge orphaned work dirs for batches that are already SUCCEEDED or FAILED.
    # These are left behind when _cleanup's rmtree runs while NF is still writing to the dir.
    # Run in a background daemon thread so that a slow rmtree on GPFS does not block startup.
    if cleanup_work_dir:
        finished = state.get_finished_batches_with_work_dirs()
        if finished:
            def _bg_cleanup():
                for batch in finished:
                    work_dir = batch.get("work_dir")
                    if work_dir and os.path.isdir(work_dir):
                        log.info("Startup cleanup: removing orphaned work dir for finished batch %s: %s",
                                 batch["batch_id"], work_dir)
                        shutil.rmtree(work_dir, ignore_errors=True)
                        try:
                            os.rmdir(os.path.dirname(work_dir))
                        except OSError:
                            pass
                log.info("Startup cleanup: finished removing %d orphaned work dir(s)", len(finished))
            t = threading.Thread(target=_bg_cleanup, name="startup-cleanup", daemon=True)
            t.start()

    resume_specs = []
    running = state.get_running_batches()
    if running:
        log.info("Recovering %d in-flight batch(es) from previous run…", len(running))
        for batch in running:
            batch_id = batch["batch_id"]
            work_dir = batch.get("work_dir")
            csv_path = batch.get("csv_path")
            log_path = batch.get("log_path")
            batch_log_dir = os.path.dirname(log_path) if log_path else ""
            trace_path = _batch_trace_path(batch_log_dir, batch_id) if batch_log_dir else None
            nf_log_path = _batch_nf_log_path(batch_log_dir, batch_id) if batch_log_dir else None

            # Skip batches whose batch_id encodes a creation time older than 24 hours.
            # batch_ids are formatted as "%Y%m%dT%H%M%S_<uuid>" — parse the prefix.
            # This filters out stale legacy batches that re-appear after repeated
            # restarts because their work dirs and CSVs are recreated by runner threads.
            try:
                from datetime import timezone as _tz
                id_ts = datetime.strptime(batch_id[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_tz.utc)
                age_hours = (datetime.now(_tz.utc) - id_ts).total_seconds() / 3600
                if age_hours > 24:
                    log.info("  Batch %s: stale (created %.1fh ago) — resetting slides to PENDING",
                             batch_id, age_hours)
                    state.reset_dispatched_to_pending(batch_id)
                    state.complete_batch(batch_id, exit_code=-1)
                    continue
            except (ValueError, KeyError):
                pass  # batch_id doesn't match expected format; proceed normally

            has_resume_state = bool(
                work_dir and os.path.isdir(work_dir) and csv_path and os.path.isfile(csv_path)
            )

            # If nf_pid is missing or dead but the work dir and batch CSV still
            # exist, prefer a Nextflow -resume over resetting the slides. This
            # covers dispatcher interruptions that happen after a batch has
            # started doing real work but before the runner thread can clean up.
            nf_pid = batch.get("nf_pid")
            if not nf_pid:
                if has_resume_state:
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

            nf_alive = True
            try:
                os.kill(nf_pid, 0)   # raises OSError if process is dead
            except OSError:
                nf_alive = False

            if has_resume_state:
                # Kill any orphaned NF process from the previous dispatcher run.
                # Without this, NF refuses to resume with "-name <same_name>" because the
                # original process still holds the run name in ~/.nextflow/history.
                if nf_alive:
                    _kill_orphaned_nf(batch_id, nf_pid, trace_path=trace_path, nf_log_path=nf_log_path)
                    log.info("  Batch %s: work dir intact and NF PID=%d is alive — will resume with -resume",
                             batch_id, nf_pid)
                else:
                    _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
                    log.info("  Batch %s: NF PID=%d is dead but work dir is intact — will resume with -resume",
                             batch_id, nf_pid)
                resume_specs.append((batch_id, csv_path, work_dir))
                # Leave slides in DISPATCHED state; they'll transition to DONE/FAILED on resume completion
            else:
                if nf_alive:
                    _kill_orphaned_nf(batch_id, nf_pid, trace_path=trace_path, nf_log_path=nf_log_path)
                else:
                    _scancel_slurm_jobs(batch_id, _collect_slurm_job_ids(trace_path, nf_log_path))
                log.info("  Batch %s: work dir missing — resetting slides to PENDING", batch_id)
                state.reset_dispatched_to_pending(batch_id)
                state.complete_batch(batch_id, exit_code=-1)

    if retry_failed:
        n = state.reset_failed_to_pending(max_retries=max_slide_retries)
        if n:
            log.info("retry_failed: reset %d FAILED slide(s) to PENDING.", n)

    # Re-enqueue all pending slides (recovered fallbacks + previously pending)
    pending = state.get_pending_slides()
    for slide in pending:
        pending_deque.append(slide)
    if pending:
        log.info("Re-enqueued %d slide(s) for dispatch.", len(pending))

    return resume_specs


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


def _acquire_pid_lock(path: str, *, label: str, payload: str = "") -> None:
    """Atomically acquire a PID lock, removing it only when the PID is stale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                body = f"{os.getpid()} {payload}\n".strip() + "\n"
                os.write(fd, body.encode())
            finally:
                os.close(fd)
            log.info("%s lockfile written: %s (PID=%d)", label, path, os.getpid())
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


def _release_lock(path: str) -> None:
    try:
        if _read_lock_pid(path) == os.getpid():
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

    # Auto-derive tower_endpoint from tower_proxy_port if not explicitly set.
    if not cfg.tower_endpoint and cfg.tower_proxy_port:
        cfg.tower_endpoint = f"http://localhost:{cfg.tower_proxy_port}"
        log.info("Auto-derived tower_endpoint: %s", cfg.tower_endpoint)

    os.makedirs(cfg.work_base_dir, exist_ok=True)
    os.makedirs(cfg.dispatch_dir, exist_ok=True)
    os.makedirs(cfg.state_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # PID lockfile — prevent two dispatcher instances from running simultaneously.
    # Use O_CREAT|O_EXCL for the initial write to atomically detect races.
    pid_lock_path = os.path.join(cfg.state_dir, "dispatcher.pid")

    # Belt-and-suspenders: scan for any existing mussel_dispatcher process that
    # might be running silently (e.g. if its lockfile was overwritten by a later
    # restart).  This catches rogue long-running dispatchers even when the lockfile
    # no longer refers to them.
    import subprocess as _sp
    _scan = _sp.run(
        ["pgrep", "-f", "mussel_dispatcher"],
        capture_output=True, text=True,
    )
    _existing_pids = [int(p) for p in _scan.stdout.split() if p.strip().isdigit() and int(p) != os.getpid()]
    if _existing_pids:
        # Filter out our own child processes (sidecars, etc.) and only block
        # a live dispatcher that is using the same config file. Separate
        # dispatchers with isolated configs/state are allowed.
        _rogue = []
        for _pid in _existing_pids:
            try:
                _raw_cmd = open(f"/proc/{_pid}/cmdline").read().replace("\x00", " ").strip()
                if "mussel_dispatcher" not in _raw_cmd or "tower" in _raw_cmd or "dashboard" in _raw_cmd:
                    continue
                _exe = os.path.basename(os.readlink(f"/proc/{_pid}/exe"))
                if not _exe.startswith("python"):
                    continue
                _parts = _raw_cmd.split()
                if not _parts:
                    continue
                _other_cfg = _parts[-1]
                if not os.path.isabs(_other_cfg):
                    _cwd = os.readlink(f"/proc/{_pid}/cwd")
                    _other_cfg = os.path.join(_cwd, _other_cfg)
                if os.path.realpath(_other_cfg) == config_path:
                    _rogue.append((_pid, _raw_cmd[:120]))
            except OSError:
                pass
        if _rogue:
            log.warning(
                "Found %d existing mussel_dispatcher-like process(es) for config %s. "
                "Continuing to authoritative PID/cohort lock checks because process scans can "
                "return stale or transient matches: %s",
                len(_rogue),
                config_path,
                ", ".join(f"PID {p} ({c}...)" for p, c in _rogue),
            )

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

    db_path = os.path.join(cfg.state_dir, "dispatcher.db")

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

    state = StateStore(db_path, manifest_id_set=manifest_id_set)
    log.info("State store: %s", db_path)

    # Apply startup priority update: set priority=1 for existing PENDING slides not in manifest.
    if manifest_id_set is not None:
        import tempfile as _tmp
        try:
            _conn = state._conn()
            _conn.execute("CREATE TEMP TABLE IF NOT EXISTS _manifest_ids (id TEXT PRIMARY KEY)")
            _conn.execute("DELETE FROM _manifest_ids")
            _conn.executemany("INSERT OR IGNORE INTO _manifest_ids VALUES (?)",
                              ((mid,) for mid in manifest_id_set))
            _updated = _conn.execute(
                "UPDATE slides SET priority=1 WHERE priority=0 "
                "AND slide_id NOT IN (SELECT id FROM _manifest_ids)"
            ).rowcount
            _conn.execute("DROP TABLE IF EXISTS _manifest_ids")
            _conn.commit()
            log.info("Priority update: set priority=1 for %d slide(s) not in manifest", _updated)
        except Exception as _e:
            log.warning("Priority startup update failed: %s", _e)

    stop_event = threading.Event()

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

    pending_deque: deque = deque()

    # Recovery
    resume_specs = recover_in_flight(
        state, pending_deque, cfg.retry_failed, cfg.max_slide_retries,
        cleanup_work_dir=cfg.cleanup_work_dir,
    )

    # RunManager
    run_manager = RunManager(cfg, state)

    # Submit resume runs for batches whose work dirs are still intact,
    # but cap at max_concurrent_runs to avoid overshooting the concurrency limit
    # on startup (e.g. if the dispatcher was killed while many batches were running).
    # Excess recovery batches have their slides reset to PENDING for re-dispatch.
    capped = resume_specs[:cfg.max_concurrent_runs]
    excess = resume_specs[cfg.max_concurrent_runs:]
    for batch_id, csv_path, work_dir in excess:
        log.info("Startup cap: resetting batch %s slides to PENDING (exceeds max_concurrent_runs=%d)",
                 batch_id, cfg.max_concurrent_runs)
        state.reset_dispatched_to_pending(batch_id)
        state.complete_batch(batch_id, exit_code=-1)
    for batch_id, csv_path, work_dir in capped:
        run_manager.submit_resume(batch_id, csv_path, work_dir)
    if capped:
        log.info("Submitted %d batch resume(s) with -resume.", len(capped))
    if excess:
        log.info("Reset %d excess recovery batch(es) to PENDING (cap=%d).",
                 len(excess), cfg.max_concurrent_runs)

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
            _release_lock(lock_path)
        try:
            os.unlink(pid_lock_path)
        except OSError:
            pass
        sys.exit(1)

    log.info("mussel-dispatcher running with %d watcher(s). PID=%d", len(watchers), os.getpid())

    # Run scheduler in main thread (blocks until stop_event)
    scheduler.run()

    # Graceful shutdown: kill all active NF processes and mark their batches FAILED
    # so the DB is clean on next startup (no zombie RUNNING entries).
    n = run_manager.running_count()
    if n:
        log.info("Graceful shutdown: killing %d active NF process(es) and marking FAILED…", n)
        killed = run_manager.kill_all_and_mark_failed()
        log.info("Graceful shutdown: marked %d batch(es) FAILED, slides reset to PENDING.", killed)
    run_manager.shutdown(wait=False)
    log.info("mussel-dispatcher stopped.")
    for proc in sidecar_procs:
        proc.terminate()
    try:
        os.unlink(pid_lock_path)
    except OSError:
        pass
    for lock_path in cohort_lock_paths:
        _release_lock(lock_path)


if __name__ == "__main__":
    main()
