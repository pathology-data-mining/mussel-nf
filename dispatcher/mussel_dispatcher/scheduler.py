"""BatchScheduler, RunManager, and main() for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
import signal
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

    def run(self):
        """Scheduler loop — call in the main thread."""
        log.info("BatchScheduler started (batch_size=%d, max_wait=%ds)",
                 self.cfg.batch_size, self.cfg.max_wait_seconds)
        _last_db_sweep = time.monotonic()
        while not self.stop_event.is_set():
            self._maybe_dispatch()
            now = time.monotonic()
            if now - _last_db_sweep >= self._DB_SWEEP_INTERVAL:
                _last_db_sweep = now
                self._requeue_db_pending()
            self.stop_event.wait(5)
        # Do NOT force-dispatch on shutdown — pending slides stay in the DB
        # and will be recovered on next startup.
        log.info("BatchScheduler stopped. Pending slides will be recovered on restart.")

    def _requeue_db_pending(self):
        """Re-enqueue PENDING slides from the DB that are not currently in the dispatch deque.

        Handles slides reset to PENDING after a batch completion (e.g. by
        _verify_wds_coverage) that were never re-added to the in-memory queue.
        """
        with self._lock:
            in_deque_ids = {s.get("slide_id") for s in self._pending}
        pending_in_db = self.state.get_pending_slides()
        newly_enqueued = 0
        for slide in pending_in_db:
            if slide.get("slide_id") not in in_deque_ids:
                self.enqueue(slide)
                newly_enqueued += 1
        if newly_enqueued:
            log.info("BatchScheduler: re-enqueued %d PENDING slide(s) from DB (missed by watcher)",
                     newly_enqueued)

    def _maybe_dispatch(self, force: bool = False):
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
        self._lock = threading.Lock()

    def submit(self, batch_id: str, slides: list):
        runner = NextflowRunner(self.cfg, batch_id, slides, self.state)
        future: Future = self._executor.submit(runner.run)
        with self._lock:
            self._futures[batch_id] = future
        future.add_done_callback(lambda f: self._on_done(batch_id, f))

    def submit_resume(self, batch_id: str, csv_path: str, work_dir: str):
        """Re-submit an interrupted batch using the existing work dir and -resume."""
        runner = NextflowRunner(self.cfg, batch_id, [], self.state,
                                resume=True, existing_csv_path=csv_path, existing_work_dir=work_dir)
        future: Future = self._executor.submit(runner.run)
        with self._lock:
            self._futures[batch_id] = future
        future.add_done_callback(lambda f: self._on_done(batch_id, f))

    def _on_done(self, batch_id: str, future: Future):
        with self._lock:
            self._futures.pop(batch_id, None)
        try:
            future.result()
        except Exception as e:
            log.error("Unhandled exception in batch %s: %s", batch_id, e)

    def shutdown(self, wait: bool = True):
        log.info("RunManager shutting down (wait=%s)…", wait)
        self._executor.shutdown(wait=wait)

    def running_count(self) -> int:
        with self._lock:
            return len(self._futures)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

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

    # Purge orphaned work dirs for batches that are already SUCCEEDED or FAILED.
    # These are left behind when _cleanup's rmtree runs while NF is still writing to the dir.
    if cleanup_work_dir:
        for batch in state.get_finished_batches_with_work_dirs():
            work_dir = batch.get("work_dir")
            if work_dir and os.path.isdir(work_dir):
                log.info("Startup cleanup: removing orphaned work dir for finished batch %s: %s",
                         batch["batch_id"], work_dir)
                shutil.rmtree(work_dir, ignore_errors=True)
                try:
                    os.rmdir(os.path.dirname(work_dir))
                except OSError:
                    pass

    resume_specs = []
    running = state.get_running_batches()
    if running:
        log.info("Recovering %d in-flight batch(es) from previous run…", len(running))
        for batch in running:
            batch_id = batch["batch_id"]
            work_dir = batch.get("work_dir")
            csv_path = batch.get("csv_path")

            if work_dir and os.path.isdir(work_dir) and csv_path and os.path.isfile(csv_path):
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

    # Submit resume runs for batches whose work dirs are still intact
    for batch_id, csv_path, work_dir in resume_specs:
        run_manager.submit_resume(batch_id, csv_path, work_dir)
    if resume_specs:
        log.info("Submitted %d batch resume(s) with -resume.", len(resume_specs))

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

    n = run_manager.running_count()
    if n:
        log.info("Waiting for %d in-flight batch(es) to finish… (Ctrl+C to abandon)", n)
    try:
        run_manager.shutdown(wait=True)
    except KeyboardInterrupt:
        log.warning("Forced exit — %d batch(es) may still be running in SLURM. "
                    "They will be recovered on next startup.", run_manager.running_count())
        run_manager.shutdown(wait=False)
    log.info("mussel-dispatcher stopped.")
    try:
        os.unlink(pid_lock_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
