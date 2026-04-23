#!/usr/bin/env python3
"""
mussel-dispatcher.py — Streaming slide dispatcher for mussel-nf.

Watches slide sources for new WSI files, accumulates them into batches,
and dispatches each batch as a parallel `nextflow run` subprocess.
Optionally runs post-batch hooks (e.g. WDS shard append) after each
successful run.

SUBCOMMANDS
-----------
  run (default)
    python mussel-dispatcher.py <config.yaml>

    Start the dispatcher.  Runs until interrupted (SIGINT / SIGTERM).
    Crashed or interrupted runs are automatically recovered on restart —
    in-flight slides are re-enqueued and duplicate work is avoided via the
    SQLite StateStore.

  collect-manifests
    python mussel-dispatcher.py collect-manifests <config.yaml>

    Scan outdir for per-run manifest-*.csv files written by the nextflow
    pipeline and merge them into a single combined manifest CSV at
    <outdir>/manifest-combined.csv (or config.combined_manifest_path).

  help
    python mussel-dispatcher.py --help
    python mussel-dispatcher.py -h

    Show this message and exit.

WATCHERS
--------
  local   — polls a directory for new slide files
  s3      — polls an S3-compatible bucket prefix
  tcga    — syncs TCGA GDC inventory, resolves paths, optionally downloads

  Multiple watchers can run in parallel.  See tcga_dispatcher.yaml and
  dispatcher.yaml for annotated examples.

CONFIG KEYS (top level)
-----------------------
  Required:
    nextflow_profiles   Comma-separated NF profiles, e.g. "cluster,apptainer"
    outdir              Nextflow --outdir (absolute, or relative to config file)

  Optional directory overrides (all default to paths relative to the config file):
    repo_dir            mussel-nf root          (default: ..)
    work_base_dir       Per-batch NF work dirs  (default: work/)
    dispatch_dir        Batch samples CSVs      (default: batches/)
    state_dir           SQLite state DB         (default: state/)
    log_dir             Per-batch NF logs       (default: logs/)

  Batching / parallelism:
    batch_size          Slides per NF run (default 20)
    min_batch_size      Minimum to dispatch; always flushed at shutdown (default 1)
    max_wait_seconds    Time trigger: dispatch a partial batch after N s (default 300)
    max_concurrent_runs Parallel Nextflow jobs (default 2)

  Behaviour:
    retry_failed        Re-enqueue slides from crashed batches on restart (default true)
    cleanup_work_dir    Delete NF work dir after each successful batch (default false)

  Hooks:
    post_batch_hooks    List of {command, args} run after each successful NF run.
                        Template vars: {batch_csv}, {batch_id}, {outdir}, {repo_dir}
                        Auto-generated hooks (from wds_dest on tcga watchers) run first.

  watchers              List of watcher configs (see WATCHERS above)

EXIT CODES
----------
  0   Normal exit after all watchers stop and queues are drained
  1   Fatal configuration or startup error
"""

import csv
import glob as _glob
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mussel-dispatcher")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WatcherConfig:
    type: str                          # "local", "s3", or "tcga"
    # local
    path: Optional[str] = None
    recursive: bool = True
    stability_wait_seconds: int = 30
    min_file_size_mb: float = 10.0
    # s3
    bucket: Optional[str] = None
    prefix: str = ""
    min_file_size_bytes: int = 10_000_000
    aws_profile: Optional[str] = None
    endpoint_url: Optional[str] = None
    # shared
    poll_interval_seconds: int = 60
    extensions: list = field(default_factory=lambda: [".svs", ".tiff", ".tif", ".ndpi", ".scn"])
    # tcga watcher
    inventory_csv: str = ""
    status_csv: str = ""
    results_dir: str = ""
    model: str = "ctranspath"
    local_slides_dir: str = ""
    s3_base: str = ""
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    project: str = ""
    slide_type: str = "DX1"
    download_enabled: bool = False
    download_dir: str = ""
    download_concurrency: int = 4
    gdc_token_file: str = ""
    gdc_max_age_hours: float = 24.0
    scripts_dir: str = ""  # path to scripts/tcga/; defaults to {repo_dir}/scripts/tcga
    # When set, a tcga_append_wds.py post-batch hook is generated automatically.
    # Supports s3:// URIs and local paths.
    wds_dest: str = ""
    wds_staging_dir: str = ""  # required when wds_dest is s3://


@dataclass
class Config:
    # Required
    nextflow_profiles: str
    outdir: str

    # Optional — default to paths relative to the config file's directory.
    # repo_dir defaults to the parent of the config file's directory (i.e. mussel-nf/
    # when the config lives in mussel-nf/dispatcher/).
    repo_dir: str = ""
    work_base_dir: str = ""   # default: <config_dir>/work
    dispatch_dir: str = ""    # default: <config_dir>/batches
    state_dir: str = ""       # default: <config_dir>/state
    log_dir: str = ""         # default: <config_dir>/logs

    max_concurrent_runs: int = 2
    batch_size: int = 20
    min_batch_size: int = 1
    max_wait_seconds: int = 300
    retry_failed: bool = True
    cleanup_work_dir: bool = False
    combined_manifest_path: Optional[str] = None  # defaults to {outdir}/manifest-combined.csv
    post_batch_hooks: list = field(default_factory=list)
    # list of {"command": "...", "args": ["..."]}
    # template vars available in command and args strings:
    #   {batch_csv}  — path to the batch samples CSV
    #   {batch_id}   — unique batch identifier
    #   {outdir}     — nextflow output directory
    #   {repo_dir}   — repository root

    watchers: list = field(default_factory=list)

    def resolved_combined_manifest_path(self) -> str:
        return self.combined_manifest_path or os.path.join(self.outdir, "manifest-combined.csv")

    @classmethod
    def load(cls, path: str) -> "Config":
        config_dir = os.path.dirname(os.path.abspath(path))

        with open(path) as f:
            raw = yaml.safe_load(f)

        # Resolve directory paths relative to the config file's location.
        # Absolute paths in the config are passed through unchanged.
        def _resolve(key: str, default: str) -> str:
            val = raw.get(key, "") or default
            return val if os.path.isabs(val) else os.path.join(config_dir, val)

        raw["repo_dir"]      = _resolve("repo_dir",      "..")
        raw["work_base_dir"] = _resolve("work_base_dir", "work")
        raw["dispatch_dir"]  = _resolve("dispatch_dir",  "batches")
        raw["state_dir"]     = _resolve("state_dir",     "state")
        raw["log_dir"]       = _resolve("log_dir",       "logs")
        # outdir: also resolve relative to config dir if not absolute
        outdir = raw.get("outdir", "")
        if outdir and not os.path.isabs(outdir):
            raw["outdir"] = os.path.join(config_dir, outdir)

        watcher_cfgs = []
        for w in raw.pop("watchers", []):
            watcher_cfgs.append(WatcherConfig(**w))

        raw["watchers"] = watcher_cfgs
        cfg = cls(**raw)
        cfg.post_batch_hooks = cfg._build_auto_hooks() + cfg.post_batch_hooks
        return cfg

    def _build_auto_hooks(self) -> list:
        """Generate post-batch hooks automatically from watcher configuration.

        For each tcga watcher with wds_dest set, a tcga_append_wds.py hook is
        prepended so features are appended to WDS shards after every successful
        Nextflow run without requiring manual hook configuration.

        Explicit post_batch_hooks (e.g. databricks sync) run after auto hooks.
        """
        hooks = []
        for w in self.watchers:
            if w.type != "tcga" or not w.wds_dest:
                continue
            args = [
                "--pt-dir={outdir}/features/" + w.model + "/pt",
                "--h5-dir={outdir}/features/" + w.model + "/tile_h5",
                "--inventory=" + w.inventory_csv,
                "--wds-dest=" + w.wds_dest,
                "--model-type=" + w.model,
                "--slide-ids-csv={batch_csv}",
            ]
            if w.wds_staging_dir:
                args.append("--staging-dir=" + w.wds_staging_dir)
            hooks.append({
                "command": "python {repo_dir}/scripts/tcga/tcga_append_wds.py",
                "args": args,
            })
            log.debug("Auto post-batch hook: tcga_append_wds for model=%s dest=%s", w.model, w.wds_dest)
        return hooks


# ---------------------------------------------------------------------------
# StateStore (SQLite)
# ---------------------------------------------------------------------------

class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS slides (
                slide_path   TEXT PRIMARY KEY,
                slide_id     TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'PENDING',
                batch_id     TEXT,
                first_seen_at TEXT,
                dispatched_at TEXT,
                completed_at  TEXT,
                error_msg    TEXT
            );

            CREATE TABLE IF NOT EXISTS batches (
                batch_id      TEXT PRIMARY KEY,
                csv_path      TEXT,
                status        TEXT NOT NULL DEFAULT 'RUNNING',
                slide_count   INTEGER,
                dispatched_at TEXT,
                completed_at  TEXT,
                nextflow_exit INTEGER,
                log_path      TEXT,
                manifest_path TEXT
            );
        """)
        conn.commit()

    # -- Slides ---------------------------------------------------------------

    def add_slide(self, slide_path: str, slide_id: str):
        conn = self._conn()
        conn.execute(
            """INSERT OR IGNORE INTO slides (slide_path, slide_id, status, first_seen_at)
               VALUES (?, ?, 'PENDING', ?)""",
            (slide_path, slide_id, datetime.utcnow().isoformat()),
        )
        conn.commit()

    def is_known(self, slide_path: str) -> bool:
        row = self._conn().execute(
            "SELECT status FROM slides WHERE slide_path = ?", (slide_path,)
        ).fetchone()
        return row is not None

    def get_pending_slides(self) -> list:
        rows = self._conn().execute(
            "SELECT slide_path, slide_id FROM slides WHERE status = 'PENDING'"
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_dispatched(self, slide_paths: list, batch_id: str):
        conn = self._conn()
        now = datetime.utcnow().isoformat()
        conn.executemany(
            "UPDATE slides SET status='DISPATCHED', batch_id=?, dispatched_at=? WHERE slide_path=?",
            [(batch_id, now, sp) for sp in slide_paths],
        )
        conn.commit()

    def mark_slides_complete(self, batch_id: str, succeeded: bool):
        conn = self._conn()
        status = "SUCCEEDED" if succeeded else "FAILED"
        conn.execute(
            "UPDATE slides SET status=?, completed_at=? WHERE batch_id=?",
            (status, datetime.utcnow().isoformat(), batch_id),
        )
        conn.commit()

    def reset_dispatched_to_pending(self, batch_id: str):
        conn = self._conn()
        conn.execute(
            "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL WHERE batch_id=? AND status='DISPATCHED'",
            (batch_id,),
        )
        conn.commit()

    # -- Batches --------------------------------------------------------------

    def add_batch(self, batch_id: str, csv_path: str, slide_count: int, log_path: str):
        conn = self._conn()
        conn.execute(
            """INSERT INTO batches (batch_id, csv_path, status, slide_count, dispatched_at, log_path)
               VALUES (?, ?, 'RUNNING', ?, ?, ?)""",
            (batch_id, csv_path, slide_count, datetime.utcnow().isoformat(), log_path),
        )
        conn.commit()

    def complete_batch(self, batch_id: str, exit_code: int):
        conn = self._conn()
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        conn.execute(
            "UPDATE batches SET status=?, completed_at=?, nextflow_exit=? WHERE batch_id=?",
            (status, datetime.utcnow().isoformat(), exit_code, batch_id),
        )
        conn.commit()

    def record_batch_manifest(self, batch_id: str, manifest_path: str):
        conn = self._conn()
        conn.execute(
            "UPDATE batches SET manifest_path=? WHERE batch_id=?",
            (manifest_path, batch_id),
        )
        conn.commit()

    def get_all_manifest_paths(self) -> list:
        rows = self._conn().execute(
            "SELECT manifest_path FROM batches WHERE manifest_path IS NOT NULL"
        ).fetchall()
        return [r["manifest_path"] for r in rows]

    def get_running_batches(self) -> list:
        rows = self._conn().execute(
            "SELECT batch_id, csv_path, log_path FROM batches WHERE status='RUNNING'"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Readiness Checker
# ---------------------------------------------------------------------------

class ReadinessChecker:
    """Two-poll size+mtime stability for local files."""

    SKIP_PATTERNS = re.compile(r"(\.part|\.tmp|^\.)$", re.IGNORECASE)

    def __init__(self, stability_wait: int, min_size_bytes: int):
        self.stability_wait = stability_wait
        self.min_size_bytes = min_size_bytes
        self._snapshots: dict = {}  # path -> (size, mtime, timestamp)

    def is_ready(self, path: str) -> bool:
        name = os.path.basename(path)
        if self.SKIP_PATTERNS.search(name) or name.startswith("."):
            return False
        try:
            st = os.stat(path)
        except OSError:
            return False
        size, mtime = st.st_size, st.st_mtime
        if size < self.min_size_bytes:
            return False

        prev = self._snapshots.get(path)
        now = time.monotonic()
        if prev is None or prev[0] != size or prev[1] != mtime:
            self._snapshots[path] = (size, mtime, now)
            return False
        # stable — check elapsed
        return (now - prev[2]) >= self.stability_wait

    def discard(self, path: str) -> None:
        """Remove a path from the snapshot cache once it has been accepted/queued."""
        self._snapshots.pop(path, None)


# ---------------------------------------------------------------------------
# Local Watcher
# ---------------------------------------------------------------------------

class LocalWatcher(threading.Thread):
    def __init__(self, cfg: WatcherConfig, pending: deque, state: StateStore, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"LocalWatcher:{cfg.path}")
        self.cfg = cfg
        self.pending = pending
        self.state = state
        self.stop_event = stop_event
        self.checker = ReadinessChecker(
            cfg.stability_wait_seconds,
            int(cfg.min_file_size_mb * 1024 * 1024),
        )
        self.exts = {e.lower() for e in cfg.extensions}

    def run(self):
        log.info("LocalWatcher started: %s", self.cfg.path)
        while not self.stop_event.is_set():
            try:
                self._scan(self.cfg.path)
            except Exception as e:
                log.error("LocalWatcher scan error: %s", e)
            self.stop_event.wait(self.cfg.poll_interval_seconds)

    def _scan(self, directory: str):
        try:
            entries = list(os.scandir(directory))
        except PermissionError as e:
            log.warning("Cannot scan %s: %s", directory, e)
            return
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and self.cfg.recursive:
                self._scan(entry.path)
            elif entry.is_file(follow_symlinks=False):
                ext = Path(entry.name).suffix.lower()
                if ext not in self.exts:
                    continue
                path = entry.path
                if self.state.is_known(path):
                    continue
                if self.checker.is_ready(path):
                    slide_id = Path(path).stem
                    log.info("New slide ready: %s", path)
                    self.state.add_slide(path, slide_id)
                    self.checker.discard(path)
                    self.pending.append({"slide_path": path, "slide_id": slide_id})


# ---------------------------------------------------------------------------
# S3 Watcher
# ---------------------------------------------------------------------------

class S3Watcher(threading.Thread):
    def __init__(self, cfg: WatcherConfig, pending: deque, state: StateStore, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"S3Watcher:{cfg.bucket}/{cfg.prefix}")
        self.cfg = cfg
        self.pending = pending
        self.state = state
        self.stop_event = stop_event
        self.exts = {e.lower() for e in cfg.extensions}
        self._s3 = None

    def _get_s3(self):
        if self._s3 is None:
            try:
                import boto3
            except ImportError:
                raise RuntimeError("boto3 required for S3 watching: pip install boto3")
            kwargs: dict = {}
            if self.cfg.aws_profile:
                kwargs["profile_name"] = self.cfg.aws_profile
            session = boto3.Session(**kwargs)
            client_kwargs: dict = {}
            if self.cfg.endpoint_url:
                client_kwargs["endpoint_url"] = self.cfg.endpoint_url
            self._s3 = session.client("s3", **client_kwargs)
        return self._s3

    def run(self):
        log.info("S3Watcher started: s3://%s/%s", self.cfg.bucket, self.cfg.prefix)
        while not self.stop_event.is_set():
            try:
                self._scan()
            except Exception as e:
                log.error("S3Watcher error: %s", e)
            self.stop_event.wait(self.cfg.poll_interval_seconds)

    def _in_progress_keys(self) -> set:
        s3 = self._get_s3()
        keys = set()
        paginator = s3.get_paginator("list_multipart_uploads")
        try:
            for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=self.cfg.prefix):
                for upload in page.get("Uploads", []):
                    keys.add(upload["Key"])
        except Exception:
            pass
        return keys

    def _scan(self):
        s3 = self._get_s3()
        in_progress = self._in_progress_keys()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=self.cfg.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = Path(key).suffix.lower()
                if ext not in self.exts:
                    continue
                s3_path = f"s3://{self.cfg.bucket}/{key}"
                if self.state.is_known(s3_path):
                    continue
                if obj["Size"] < self.cfg.min_file_size_bytes:
                    continue
                if key in in_progress:
                    log.debug("S3 object still uploading, skipping: %s", key)
                    continue
                slide_id = Path(key).stem
                log.info("New S3 slide ready: %s", s3_path)
                self.state.add_slide(s3_path, slide_id)
                self.pending.append({"slide_path": s3_path, "slide_id": slide_id})



# ---------------------------------------------------------------------------
# TCGA watcher
# ---------------------------------------------------------------------------

class TcgaWatcher(threading.Thread):
    """
    Polls the TCGA GDC inventory for slides pending feature extraction and
    feeds them into the dispatcher pipeline.

    For each pending slide it:
      1. Syncs the GDC inventory (respects gdc_max_age_hours to avoid
         hammering the API on every poll).
      2. Updates the per-slide status from the local results directory.
      3. Calls tcga_prepare_samples to resolve each slide's path
         (local disk → S3 → needs_download).
      4. Enqueues slides that are already available (local / S3) directly.
      5. Optionally downloads slides via gdc-client (download_enabled=true),
         then enqueues each on completion.

    Because downloads happen in a background thread pool and enqueue slides
    individually as they finish, featurization of already-downloaded slides
    can overlap with downloads of the next batch — the key scheduling win
    over the sequential tcga_run.py loop.
    """

    def __init__(
        self,
        cfg: WatcherConfig,
        pending: deque,
        state: StateStore,
        stop_event: threading.Event,
        repo_dir: str,
    ):
        super().__init__(name="tcga-watcher", daemon=True)
        self.cfg = cfg
        self.pending = pending
        self.state = state
        self.stop_event = stop_event
        self._scripts_dir = cfg.scripts_dir or str(Path(repo_dir) / "scripts" / "tcga")
        self._download_executor = ThreadPoolExecutor(
            max_workers=max(1, cfg.download_concurrency),
            thread_name_prefix="gdc-download",
        )

    def run(self):
        log.info(
            "TcgaWatcher started (poll_interval=%ds, download_enabled=%s)",
            self.cfg.poll_interval_seconds,
            self.cfg.download_enabled,
        )
        self._poll()  # poll immediately on startup
        while not self.stop_event.is_set():
            self.stop_event.wait(self.cfg.poll_interval_seconds)
            if not self.stop_event.is_set():
                self._poll()
        self._download_executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_script(self, script: str, args: list) -> int:
        cmd = [sys.executable, str(Path(self._scripts_dir) / script)] + args
        log.info("TcgaWatcher: $ %s", " ".join(str(a) for a in cmd))
        env = dict(os.environ)
        if self.cfg.s3_access_key:
            env["ECS_ACCESS_KEY"] = self.cfg.s3_access_key
        if self.cfg.s3_secret_key:
            env["ECS_SECRET_KEY"] = self.cfg.s3_secret_key
        if self.cfg.s3_endpoint:
            env.setdefault("ECS_ENDPOINT_URL", self.cfg.s3_endpoint)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode not in (0, 2):
            log.error(
                "TcgaWatcher: %s failed (exit %d):\n%s",
                script, result.returncode, result.stderr[-500:],
            )
        return result.returncode

    def _poll(self):
        log.info("TcgaWatcher: polling GDC inventory…")

        # 1. Sync inventory (skip if fresh)
        sync_args = [
            "--output", self.cfg.inventory_csv,
            "--max-age-hours", str(self.cfg.gdc_max_age_hours),
        ]
        if self.cfg.project:
            sync_args += ["--project", self.cfg.project]
        rc = self._run_script("tcga_sync_inventory.py", sync_args)
        if rc not in (0, 2):
            log.error("TcgaWatcher: inventory sync failed — skipping this poll")
            return

        # 2. Update per-slide status from results directory
        rc = self._run_script("tcga_update_status.py", [
            "--inventory", self.cfg.inventory_csv,
            "--results-dir", self.cfg.results_dir,
            "--output", self.cfg.status_csv,
        ])
        if rc != 0:
            log.error("TcgaWatcher: status update failed — skipping this poll")
            return

        # 3. Resolve slide paths (local / S3 / needs_download)
        samples_csv = str(Path(self.cfg.status_csv).with_suffix("")) + "_dispatcher.csv"
        prepare_args = [
            "--inventory", self.cfg.inventory_csv,
            "--status", self.cfg.status_csv,
            "--output", samples_csv,
            "--model", self.cfg.model,
            "--skip-done",
        ]
        if self.cfg.local_slides_dir:
            prepare_args += ["--local-slides-dir", self.cfg.local_slides_dir]
        if self.cfg.s3_base:
            prepare_args += ["--s3-base", self.cfg.s3_base]
        if self.cfg.s3_endpoint:
            prepare_args += ["--s3-endpoint", self.cfg.s3_endpoint]
        if self.cfg.project:
            prepare_args += ["--project", self.cfg.project]
        if self.cfg.slide_type and self.cfg.slide_type.lower() != "all":
            prepare_args += ["--slide-type", self.cfg.slide_type]

        rc = self._run_script("tcga_prepare_samples.py", prepare_args)
        if rc == 2:
            log.info("TcgaWatcher: no pending slides this poll")
            return
        if rc != 0:
            log.error("TcgaWatcher: prepare_samples failed — skipping this poll")
            return

        # 4. Read sidecar meta CSV (slide_id, slide_path, needs_download, file_id, ...)
        meta_csv = samples_csv.replace(".csv", ".meta.csv")
        if not Path(meta_csv).exists():
            log.warning("TcgaWatcher: meta CSV not found: %s", meta_csv)
            return

        with open(meta_csv, newline="") as f:
            slides = list(csv.DictReader(f))

        ready = [s for s in slides if s.get("needs_download", "").lower() != "true"]
        needs_dl = [s for s in slides if s.get("needs_download", "").lower() == "true"]
        log.info("TcgaWatcher: %d ready, %d need download", len(ready), len(needs_dl))

        # Enqueue slides already on disk / S3
        for s in ready:
            if not self.state.is_known(s["slide_path"]):
                self.state.add_slide(s["slide_path"], s["slide_id"])
                self.pending.append({"slide_id": s["slide_id"], "slide_path": s["slide_path"]})

        # Kick off downloads for slides not yet available
        if self.cfg.download_enabled and needs_dl:
            for s in needs_dl:
                if not self.state.is_known(s["slide_path"]):
                    self.state.add_slide(s["slide_path"], s["slide_id"])
                    self._download_executor.submit(self._download_and_enqueue, s)
        elif needs_dl:
            log.info(
                "TcgaWatcher: %d slides need download but download_enabled=false — skipped",
                len(needs_dl),
            )

    def _download_and_enqueue(self, slide: dict) -> None:
        """Download a single slide via gdc-client, then enqueue it for processing."""
        file_id = slide.get("file_id", "")
        file_name = slide.get("file_name", "")
        slide_id = slide["slide_id"]
        download_root = self.cfg.download_dir or self.cfg.local_slides_dir

        if not file_id or not download_root:
            log.error(
                "TcgaWatcher: cannot download %s — missing file_id or download directory",
                slide_id,
            )
            return

        # gdc-client writes to <download_root>/<file_id>/<file_name>
        dest_path = Path(download_root) / file_id / file_name
        if dest_path.exists():
            log.info("TcgaWatcher: %s already on disk, enqueuing", slide_id)
            self.pending.append({"slide_id": slide_id, "slide_path": str(dest_path)})
            return

        cmd = [
            "gdc-client", "download",
            "--no-related-files",
            "-n", str(max(1, self.cfg.download_concurrency)),
            "-d", str(download_root),
        ]
        if self.cfg.gdc_token_file:
            cmd += ["-t", self.cfg.gdc_token_file]
        cmd.append(file_id)

        log.info("TcgaWatcher: downloading %s (%s)…", slide_id, file_id)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(
                "TcgaWatcher: gdc-client failed for %s (exit %d):\n%s",
                slide_id, result.returncode, result.stderr[-300:],
            )
            return

        if dest_path.exists():
            log.info("TcgaWatcher: downloaded %s → %s", slide_id, dest_path)
            self.pending.append({"slide_id": slide_id, "slide_path": str(dest_path)})
        else:
            log.error("TcgaWatcher: gdc-client exited 0 but %s not found", dest_path)


# ---------------------------------------------------------------------------
# Manifest collection
# ---------------------------------------------------------------------------

MANIFEST_HEADER = ["slide_id", "workflow_id", "key", "value"]


def collect_manifests(outdir: str, combined_path: str) -> int:
    """
    Scan *outdir* for all ``manifest-*.csv`` files produced by individual
    Nextflow runs, merge them into *combined_path*, and return the number of
    unique rows written.

    Each per-run manifest has no header and contains rows of the form::

        slide_id,workflow_id,key,value

    The combined file is written with a header row.  Deduplication is by
    ``(slide_id, key)``; when duplicates exist the row from the *newest*
    manifest file wins (last-write-wins by file mtime).
    """
    pattern = os.path.join(outdir, "manifest-*.csv")
    manifest_files = sorted(_glob.glob(pattern), key=os.path.getmtime)

    if not manifest_files:
        log.info("collect_manifests: no manifest-*.csv files found in %s", outdir)
        return 0

    # (slide_id, key) -> row dict  — later files overwrite earlier ones
    rows: dict = {}
    for mf in manifest_files:
        try:
            with open(mf, newline="") as f:
                reader = csv.reader(f)
                for parts in reader:
                    if not parts:
                        continue
                    if len(parts) != 4:
                        log.warning("collect_manifests: skipping malformed line in %s: %r", mf, parts)
                        continue
                    slide_id, workflow_id, key, value = parts
                    rows[(slide_id, key)] = {
                        "slide_id": slide_id,
                        "workflow_id": workflow_id,
                        "key": key,
                        "value": value,
                    }
        except OSError as e:
            log.warning("collect_manifests: cannot read %s: %s", mf, e)

    os.makedirs(os.path.dirname(combined_path) or ".", exist_ok=True)
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)

    n = len(rows)
    log.info("collect_manifests: wrote %d unique rows to %s (from %d file(s))",
             n, combined_path, len(manifest_files))
    return n


# ---------------------------------------------------------------------------
# NextflowRunner
# ---------------------------------------------------------------------------

class NextflowRunner:
    def __init__(self, cfg: Config, batch_id: str, slides: list, state: StateStore):
        self.cfg = cfg
        self.batch_id = batch_id
        self.slides = slides  # list of {"slide_path": ..., "slide_id": ..., "oncotree_code": ...}
        self.state = state

    def run(self):
        csv_path = self._write_csv()
        work_dir = os.path.join(self.cfg.work_base_dir, f"batch_{self.batch_id}", "work")
        log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)

        self.state.add_batch(self.batch_id, csv_path, len(self.slides), log_path)
        self.state.mark_dispatched([s["slide_path"] for s in self.slides], self.batch_id)

        cmd = [
            "nextflow", "run", self.cfg.repo_dir,
            "-profile", self.cfg.nextflow_profiles,
            "-work-dir", work_dir,
            "--samples_csv", csv_path,
            "--outdir", self.cfg.outdir,
        ]

        log.info("Dispatching batch %s (%d slides): %s", self.batch_id, len(self.slides), " ".join(cmd))

        run_started_at = time.time()
        exit_code = -1
        try:
            with open(log_path, "w") as lf:
                result = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    cwd=self.cfg.repo_dir,
                )
            exit_code = result.returncode
        except Exception as e:
            log.error("Batch %s failed to launch: %s", self.batch_id, e)

        self.state.complete_batch(self.batch_id, exit_code)
        self.state.mark_slides_complete(self.batch_id, exit_code == 0)

        if exit_code == 0:
            log.info("Batch %s completed successfully.", self.batch_id)
            self._collect_manifest(run_started_at)
            self._run_post_batch_hooks(csv_path)
            if self.cfg.cleanup_work_dir:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
        else:
            log.error("Batch %s failed (exit %d). Log: %s", self.batch_id, exit_code, log_path)

        return exit_code

    def _run_post_batch_hooks(self, batch_csv: str) -> None:
        """Run any configured post-batch hooks after a successful NF run."""
        if not self.cfg.post_batch_hooks:
            return

        def _sub(s: str) -> str:
            return s.format(
                batch_csv=batch_csv,
                batch_id=self.batch_id,
                outdir=self.cfg.outdir,
                repo_dir=self.cfg.repo_dir,
            )

        for hook in self.cfg.post_batch_hooks:
            cmd_str = hook.get("command", "")
            if not cmd_str:
                log.warning("Post-batch hook has no 'command' key — skipping")
                continue
            cmd = [_sub(p) for p in cmd_str.split()] + [_sub(a) for a in hook.get("args", [])]
            log.info("Batch %s: running post-batch hook: %s", self.batch_id, " ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.cfg.repo_dir)
                if result.returncode != 0:
                    log.error(
                        "Post-batch hook failed (exit %d):\n%s",
                        result.returncode,
                        result.stderr[-1000:],
                    )
                else:
                    log.info("Post-batch hook succeeded")
            except Exception as exc:
                log.error("Post-batch hook raised: %s", exc)

    def _collect_manifest(self, run_started_at: float):
        """Find the manifest-*.csv written by this batch and update the combined manifest."""
        pattern = os.path.join(self.cfg.outdir, "manifest-*.csv")
        candidates = [
            p for p in _glob.glob(pattern)
            if os.path.getmtime(p) >= run_started_at
        ]
        if candidates:
            # There should be exactly one per run, but take the newest if multiple
            manifest_path = max(candidates, key=os.path.getmtime)
            log.info("Batch %s produced manifest: %s", self.batch_id, manifest_path)
            self.state.record_batch_manifest(self.batch_id, manifest_path)
        else:
            log.warning("Batch %s: no manifest-*.csv found in %s after run start",
                        self.batch_id, self.cfg.outdir)

        combined_path = self.cfg.resolved_combined_manifest_path()
        collect_manifests(self.cfg.outdir, combined_path)

    def _write_csv(self) -> str:
        os.makedirs(self.cfg.dispatch_dir, exist_ok=True)
        csv_path = os.path.join(self.cfg.dispatch_dir, f"batch_{self.batch_id}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["slide_id", "slide_path", "oncotree_code"])
            writer.writeheader()
            for s in self.slides:
                writer.writerow({
                    "slide_id": s["slide_id"],
                    "slide_path": s["slide_path"],
                    "oncotree_code": s.get("oncotree_code", ""),
                })
        log.debug("Wrote batch CSV: %s", csv_path)
        return csv_path


# ---------------------------------------------------------------------------
# BatchScheduler
# ---------------------------------------------------------------------------

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

    def enqueue(self, slide: dict):
        with self._lock:
            self._pending.append(slide)
            if self._first_seen_at is None:
                self._first_seen_at = time.monotonic()
            log.debug("Pending queue size: %d", len(self._pending))

    def run(self):
        """Scheduler loop — call in the main thread."""
        log.info("BatchScheduler started (batch_size=%d, max_wait=%ds)",
                 self.cfg.batch_size, self.cfg.max_wait_seconds)
        while not self.stop_event.is_set():
            self._maybe_dispatch()
            self.stop_event.wait(5)
        # Final drain
        self._maybe_dispatch(force=True)

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
            while self._pending:
                batch.append(self._pending.popleft())
            self._first_seen_at = None

        if batch:
            log.info("Dispatching batch of %d slides (force=%s, size_trigger=%s, time_trigger=%s)",
                     len(batch), force, size_trigger, time_trigger)
            batch_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
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

def recover_in_flight(state: StateStore, pending_deque: deque, retry_failed: bool = True):
    """
    On restart, find batches that were RUNNING and reset their slides to PENDING.
    When retry_failed=True (default), the recovered slides are re-enqueued so
    they will be dispatched again.  Set retry_failed=False to skip re-enqueueing
    (slides remain in the state DB as PENDING but won't be dispatched this run).
    """
    running = state.get_running_batches()
    if not running:
        return
    log.info("Recovering %d in-flight batch(es) from previous run…", len(running))
    for batch in running:
        batch_id = batch["batch_id"]
        log.info("  Resetting DISPATCHED slides for batch %s to PENDING", batch_id)
        state.reset_dispatched_to_pending(batch_id)
        # Mark the batch itself as failed so it won't linger as RUNNING
        state.complete_batch(batch_id, exit_code=-1)

    if retry_failed:
        # Re-enqueue recovered pending slides
        for slide in state.get_pending_slides():
            pending_deque.append(slide)
        log.info("Re-enqueued %d recovered slides.", len(pending_deque))
    else:
        log.info("retry_failed=False — recovered slides left as PENDING; not re-enqueuing.")


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

    cfg = Config.load(sys.argv[1])
    log.info("Configuration loaded from %s", sys.argv[1])

    os.makedirs(cfg.work_base_dir, exist_ok=True)
    os.makedirs(cfg.dispatch_dir, exist_ok=True)
    os.makedirs(cfg.state_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    db_path = os.path.join(cfg.state_dir, "dispatcher.db")
    state = StateStore(db_path)
    log.info("State store: %s", db_path)

    stop_event = threading.Event()

    # Signal handling
    def _handle_signal(signum, frame):
        log.info("Received signal %d, shutting down…", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    pending_deque: deque = deque()

    # Recovery
    recover_in_flight(state, pending_deque, cfg.retry_failed)

    # RunManager
    run_manager = RunManager(cfg, state)

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
            watcher = TcgaWatcher(w_cfg, pending_deque, state, stop_event, cfg.repo_dir)
        else:
            log.warning("Unknown watcher type '%s', skipping.", w_cfg.type)
            continue
        watcher.start()
        watchers.append(watcher)

    if not watchers:
        log.error("No watchers configured. Exiting.")
        sys.exit(1)

    log.info("mussel-dispatcher running with %d watcher(s). PID=%d", len(watchers), os.getpid())

    # Run scheduler in main thread (blocks until stop_event)
    scheduler.run()

    run_manager.shutdown(wait=True)
    log.info("mussel-dispatcher stopped.")


if __name__ == "__main__":
    main()
