"""BatchScheduler, RunManager, and main() for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
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

        A Nextflow process can silently hang — all SLURM tasks finish and write
        their .exitcode files, but NF's monitoring thread never polls the result,
        so the batch stays RUNNING forever.  The log file is updated on every NF
        progress tick, so a stale mtime reliably signals a hung process.

        When a stuck batch is detected its NF process is killed (SIGTERM then
        SIGKILL).  The RunManager's _on_done callback fires, marks the batch
        FAILED, and the periodic retry sweep resets slides to PENDING for
        re-dispatch.

        Controlled by ``stuck_batch_timeout_hours`` in the dispatcher config
        (default 4 h).  Set to 0 to disable.
        """
        timeout_hours = self.cfg.stuck_batch_timeout_hours
        if not timeout_hours:
            return
        cutoff = time.time() - timeout_hours * 3600
        for batch in self.state.get_running_batches():
            batch_id = batch["batch_id"]
            nf_pid = batch.get("nf_pid")
            if not nf_pid:
                continue
            log_path = os.path.join(self.cfg.log_dir, f"batch_{batch_id}.log")
            try:
                mtime = os.path.getmtime(log_path)
            except OSError:
                continue
            if mtime < cutoff:
                idle_hours = (time.time() - mtime) / 3600
                log.warning(
                    "Batch %s: log file silent for %.1fh (last update %s) — "
                    "killing stuck NF process PID=%d",
                    batch_id, idle_hours,
                    datetime.fromtimestamp(mtime).strftime("%H:%M:%S"),
                    nf_pid,
                )
                _kill_orphaned_nf(batch_id, nf_pid)

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
        if not self.run_manager.has_capacity():
            return
        with self._lock:
            n = len(self._pending)
            if n == 0:
                return
            elapsed = time.monotonic() - self._first_seen_at if self._first_seen_at else 0
            size_trigger = n >= self.cfg.batch_size
            time_trigger = elapsed >= self.cfg.max_wait_seconds
            if not force and not size_trigger and not time_trigger:
                return
            if n < self.cfg.min_batch_size and not force:
                return

            batch = []
            while self._pending and len(batch) < self.cfg.batch_size:
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
            self.run_manager.submit(batch_id, batch)


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------

class RunManager:
    def __init__(self, cfg: Config, state: StateStore):
        self.cfg = cfg
        self.state = state
        self._executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent_runs,
                                            thread_name_prefix="nextflow-run")
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

        if not self._slots.acquire(blocking=False):
            return False
        log.debug("submit: batch_id=%s slides=%d", batch_id, len(slides))
        slide_ids = {s["slide_id"] for s in slides if s.get("slide_id")}
        runner = NextflowRunner(self.cfg, batch_id, slides, self.state)
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
                                resume=True, existing_csv_path=csv_path, existing_work_dir=work_dir)
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
        log.info("RunManager shutting down (wait=%s)…", wait)
        self._executor.shutdown(wait=wait)

    def kill_all_and_mark_failed(self) -> int:
        """Kill all active NF processes and mark their batches FAILED in the DB.

        Called during graceful SIGTERM shutdown so that the DB is left in a
        clean state (no RUNNING entries).  The next dispatcher startup will find
        no in-flight batches and simply re-dispatch from PENDING slides.

        Returns the number of batches that were marked FAILED.
        """
        with self._lock:
            active = list(self._futures.keys())

        killed = 0
        for batch_id in active:
            nf_pid = self.state.get_batch_nf_pid(batch_id)
            if nf_pid:
                try:
                    _kill_orphaned_nf(batch_id, nf_pid, sigterm_wait=3.0)
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

def _kill_orphaned_nf(batch_id: str, pid: int, sigterm_wait: float = 10.0) -> None:
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
        return

    log.info("Batch %s: terminating orphaned NF process PID=%d", batch_id, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return  # Already gone

    # Wait up to sigterm_wait seconds for graceful exit, then SIGKILL.
    deadline = time.monotonic() + sigterm_wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            log.debug("Batch %s: PID %d exited after SIGTERM", batch_id, pid)
            return
        time.sleep(0.5)

    try:
        os.kill(pid, signal.SIGKILL)
        log.warning("Batch %s: PID %d did not exit after SIGTERM; sent SIGKILL", batch_id, pid)
    except OSError:
        pass


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

            # If nf_pid is recorded but the process is no longer alive, the batch
            # is a zombie left by a SIGKILL'd dispatcher whose runner thread wrote
            # to the DB but never cleaned up.  Mark it FAILED immediately so its
            # slides return to PENDING without needing a resume attempt.
            # Also treat nf_pid=None as zombie: the runner thread wrote add_batch()
            # but was killed before recording the NF PID, meaning we cannot track
            # or resume the batch reliably.
            nf_pid = batch.get("nf_pid")
            if not nf_pid:
                log.info("  Batch %s: no nf_pid recorded — zombie (runner died before NF start), "
                         "resetting slides to PENDING", batch_id)
                state.reset_dispatched_to_pending(batch_id)
                state.complete_batch(batch_id, exit_code=-1)
                continue
            try:
                os.kill(nf_pid, 0)   # raises OSError if process is dead
            except OSError:
                log.info("  Batch %s: NF process PID=%d is dead — resetting slides to PENDING",
                         batch_id, nf_pid)
                state.reset_dispatched_to_pending(batch_id)
                state.complete_batch(batch_id, exit_code=-1)
                continue

            if work_dir and os.path.isdir(work_dir) and csv_path and os.path.isfile(csv_path):
                # Kill any orphaned NF process from the previous dispatcher run.
                # Without this, NF refuses to resume with "-name <same_name>" because the
                # original process still holds the run name in ~/.nextflow/history.
                nf_pid = batch.get("nf_pid")
                if nf_pid:
                    _kill_orphaned_nf(batch_id, nf_pid)

                log.info("  Batch %s: work dir intact — will resume with -resume", batch_id)
                resume_specs.append((batch_id, csv_path, work_dir))
                # Leave slides in DISPATCHED state; they'll transition to DONE/FAILED on resume completion
            else:
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

    cfg = Config.load(sys.argv[1])
    log.info("Configuration loaded from %s", sys.argv[1])

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

    def _write_pid_lock():
        """Atomically create the lockfile, raising FileExistsError if it already exists."""
        fd = os.open(pid_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)

    while True:
        try:
            _write_pid_lock()
            break  # acquired
        except FileExistsError:
            # Lockfile exists — check if holder is still alive
            try:
                with open(pid_lock_path) as _fh:
                    old_pid = int(_fh.read().strip())
                os.kill(old_pid, 0)
                log.error(
                    "Another dispatcher instance is already running (PID %d, lockfile %s). "
                    "Kill it first or delete the lockfile if it is stale.",
                    old_pid, pid_lock_path,
                )
                sys.exit(1)
            except ProcessLookupError:
                log.warning("Stale PID lockfile (PID %d no longer running). Removing.", old_pid)
                os.unlink(pid_lock_path)
                # Loop back to retry atomic create
            except (FileNotFoundError, ValueError, OSError):
                # Lockfile vanished or unreadable — retry
                pass
    log.info("PID lockfile written: %s (PID=%d)", pid_lock_path, os.getpid())

    # Launch dashboard and tower_proxy sidecars (ports from config).
    sidecar_procs = _start_sidecars(cfg, sys.argv[1])

    db_path = os.path.join(cfg.state_dir, "dispatcher.db")
    state = StateStore(db_path)
    log.info("State store: %s", db_path)

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


if __name__ == "__main__":
    main()
