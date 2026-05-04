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
    retry_failed             Re-enqueue slides from crashed batches on restart (default true)
    cleanup_work_dir         Delete NF work dir after each batch (success or failure, default false)
    cleanup_downloads        Delete downloaded slides (.svs) after a successful batch (default false)
    cleanup_batch_csv        Delete the per-batch samples CSV after success (default false)
    cleanup_logs_after_days  Delete NF log files for batches older than N days (0 = keep forever)
    cleanup_results          Delete local .pt / .patch.h5 after WDS push succeeds (default false)
    nextflow_config          Path to an extra Nextflow config file passed as -c to every run.
                             Relative paths are resolved relative to the dispatcher config file.
    nextflow_params_file     Path to a YAML/JSON params file passed as -params-file to every run.
                             Relative paths are resolved relative to the dispatcher config file.

  Hooks:
    post_batch_hooks    List of {command, args} run after each successful NF run.
                        Template vars: {batch_csv}, {batch_id}, {outdir}, {repo_dir}
                        Auto-generated hooks run first (WDS append, then Databricks
                        sync), followed by any explicit post_batch_hooks.

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
from datetime import datetime, timedelta, timezone
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
    type: str                          # "local", "s3", "tcga", or "databricks"
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
    # Models to check for skip-done filtering in tcga_prepare_samples.
    # Empty list = auto-read from nextflow.config model_types (default).
    # Set explicitly only to override.
    models: list = field(default_factory=list)
    local_slides_dir: str = ""
    s3_base: str = ""
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    project: str = ""
    slide_type: str = "DX"
    sample_type: str = "Primary Tumor"   # GDC sample_type substring filter; "all" to disable
    gdc_token_file: str = ""
    gdc_max_age_hours: float = 24.0
    scripts_dir: str = ""  # path to scripts/tcga/; defaults to {repo_dir}/scripts/tcga
    # Per-model WDS destinations: {model_type: s3_or_local_path}.
    # A append_wds.py hook is auto-generated for each entry.
    # Example: {ctranspath: s3://bucket/wds, uni2h: s3://bucket/wds}
    # Note: append_wds.py appends /{model_type}/ automatically; do NOT include the model in the path.
    wds_destinations: dict = field(default_factory=dict)
    wds_staging_dir: str = ""  # local staging base for s3:// destinations; each model uses {staging}/{model}/
    wds_s3_max_concurrency: int = 4  # boto3 multipart threads per S3 upload/download (reduce to limit ECS load)
    secrets_env_file: str = ""  # path to a shell env file (KEY=value) with S3/ECS credentials; values are
                                # loaded into s3_access_key / s3_secret_key if not already set in the config
    # When set, a tcga_sync_databricks.py hook is generated automatically.
    # Credentials come from DATABRICKS_HOST / DATABRICKS_TOKEN env vars.
    databricks_volume_folder: str = ""  # UC volume folder; files uploaded as tcga_inventory_<ts>.parquet
    databricks_volume_path: str = ""    # [Legacy] single overwritten file path
    databricks_table: str = ""          # Delta table to MERGE INTO (passed to notebook as target_table)
    databricks_job_id: str = ""         # Databricks job to trigger after upload
    # databricks watcher — joins impact_matched_slides + slide_inventory to get S3 paths.
    # Credentials from DATABRICKS_HOST / DATABRICKS_TOKEN env vars (or ~/.databrickscfg).
    warehouse_id: str = ""              # Databricks SQL warehouse ID (required for type: databricks)
    source_filter: list = field(default_factory=list)  # filter slide_inventory.source (e.g. ['ECS2'])
    additional_where: str = ""          # extra SQL WHERE conditions appended to the query


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
    max_slide_retries: int = 5   # permanently skip slides that fail this many times
    cleanup_work_dir: bool = False
    cleanup_downloads: bool = False       # delete downloaded slides after a successful batch
    cleanup_batch_csv: bool = False       # delete per-batch samples CSV after success
    cleanup_logs_after_days: int = 0      # delete NF log files older than N days (0 = keep forever)
    cleanup_results: bool = False         # delete local .pt / .patch.h5 after WDS push succeeds
    nextflow_config: str = ""             # optional -c <file> passed to every nextflow run
    nextflow_params_file: str = ""        # optional -params-file <file> passed to every nextflow run
    nextflow_version: str = ""            # optional NXF_VER to pin the Nextflow version
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
        # nextflow_config: resolve relative to config dir if not absolute
        nf_cfg = raw.get("nextflow_config", "")
        if nf_cfg and not os.path.isabs(nf_cfg):
            raw["nextflow_config"] = os.path.join(config_dir, nf_cfg)
        # nextflow_params_file: resolve relative to config dir if not absolute
        nf_params = raw.get("nextflow_params_file", "")
        if nf_params and not os.path.isabs(nf_params):
            raw["nextflow_params_file"] = os.path.join(config_dir, nf_params)

        # Resolve path fields in watcher configs relative to the config file.
        # Applies to all string fields that represent filesystem paths.
        _WATCHER_PATH_FIELDS = (
            "path",            # local watcher dir
            "inventory_csv",
            "status_csv",
            "local_slides_dir",
            "wds_staging_dir",
            "scripts_dir",
            "gdc_token_file",
            "secrets_env_file",
        )

        def _resolve_watcher_path(val: str) -> str:
            if not val or os.path.isabs(val):
                return val
            return os.path.join(config_dir, val)

        watcher_cfgs = []
        for w in raw.pop("watchers", []):
            for field in _WATCHER_PATH_FIELDS:
                if field in w:
                    w[field] = _resolve_watcher_path(w[field])
            watcher_cfgs.append(WatcherConfig(**w))

        raw["watchers"] = watcher_cfgs
        cfg = cls(**raw)

        # Load S3 credentials from secrets_env_file if specified and not already set.
        for w in cfg.watchers:
            if w.secrets_env_file and os.path.isfile(w.secrets_env_file):
                _load_secrets_env(w.secrets_env_file, w)

        # Auto-detect model_types from nextflow.config for watchers that didn't
        # specify them explicitly.
        nf_models: list[str] | None = None
        for w in cfg.watchers:
            if not w.models:
                if nf_models is None:
                    nf_models = _read_nf_model_types(cfg.repo_dir)
                    if nf_models:
                        log.info("Auto-detected model_types from nextflow.config: %s",
                                 ", ".join(nf_models))
                w.models = nf_models

        cfg.post_batch_hooks = cfg._build_auto_hooks() + cfg.post_batch_hooks
        return cfg

    def _build_auto_hooks(self) -> list:
        """Generate post-batch hooks automatically from watcher configuration.

        For each tcga watcher:
          - wds_dest set       → prepend a append_wds.py hook (routes via inventory)
          - databricks_volume_path set → append a tcga_sync_databricks.py hook

        For each databricks watcher:
          - wds_dest set       → prepend a append_wds.py hook (routes by oncotree_code
                                 column in the batch CSV, no inventory required)

        Order per watcher: WDS append first, then Databricks sync.
        Explicit post_batch_hooks run after all auto-generated hooks.
        """
        hooks = []
        db_hooks = []
        for w in self.watchers:
            if w.type == "databricks" and w.wds_destinations:
                for model, dest in w.wds_destinations.items():
                    args = [
                        "--pt-dir={outdir}/features/" + model,
                        "--h5-dir={outdir}/tiles",
                        "--wds-dest=" + dest,
                        "--model-type=" + model,
                        "--slide-ids-csv={batch_csv}",
                        "--project-id-column=oncotree_code",
                        "--manifest-csv={outdir}/wds_manifest.csv",
                    ]
                    if w.wds_staging_dir:
                        args.append("--staging-dir=" + w.wds_staging_dir)
                    if w.wds_s3_max_concurrency != 4:
                        args.append(f"--s3-max-concurrency={w.wds_s3_max_concurrency}")
                    if self.cleanup_results:
                        args.append("--delete-local")
                    hooks.append({
                        "command": "python {repo_dir}/scripts/append_wds.py",
                        "args": args,
                    })
                    log.debug("Auto hook: append_wds (databricks) model=%s dest=%s", model, dest)

            if w.type != "tcga":
                continue

            if w.wds_destinations:
                for model, dest in w.wds_destinations.items():
                    args = [
                        "--pt-dir={outdir}/features/" + model,
                        "--h5-dir={outdir}/tiles",
                        "--inventory=" + w.inventory_csv,
                        "--wds-dest=" + dest,
                        "--model-type=" + model,
                        "--slide-ids-csv={batch_csv}",
                        "--manifest-csv={outdir}/wds_manifest.csv",
                    ]
                    if w.wds_staging_dir:
                        args.append("--staging-dir=" + w.wds_staging_dir)
                    if w.wds_s3_max_concurrency != 4:
                        args.append(f"--s3-max-concurrency={w.wds_s3_max_concurrency}")
                    if w.s3_endpoint:
                        args.append("--s3-endpoint=" + w.s3_endpoint)
                    if self.cleanup_results:
                        args.append("--delete-local")
                    hooks.append({
                        "command": "python {repo_dir}/scripts/append_wds.py",
                        "args": args,
                    })
                    log.debug("Auto hook: append_wds model=%s dest=%s", model, dest)

            if w.databricks_volume_folder or w.databricks_volume_path:
                args = [
                    "--inventory=" + w.inventory_csv,
                    "--status=" + w.status_csv,
                ]
                if w.databricks_volume_folder:
                    args.append("--volume-folder=" + w.databricks_volume_folder)
                else:
                    args.append("--volume-path=" + w.databricks_volume_path)
                if w.databricks_table:
                    args.append("--table=" + w.databricks_table)
                if w.databricks_job_id:
                    args.append("--job-id=" + w.databricks_job_id)
                db_hooks.append({
                    "command": "python {repo_dir}/scripts/tcga/tcga_sync_databricks.py",
                    "args": args,
                })
                log.debug(
                    "Auto hook: tcga_sync_databricks folder=%s table=%s",
                    w.databricks_volume_folder or w.databricks_volume_path,
                    w.databricks_table,
                )

        return hooks + db_hooks


def _load_secrets_env(path: str, watcher: "WatcherConfig") -> None:
    """Parse a shell env file (KEY=value or export KEY=value lines) and populate
    watcher S3 credentials if not already set.

    Recognises ECS_ACCESS_KEY / ECS_SECRET_KEY and the standard
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY variants.
    """
    key_map = {
        "ECS_ACCESS_KEY": "s3_access_key",
        "AWS_ACCESS_KEY_ID": "s3_access_key",
        "ECS_SECRET_KEY": "s3_secret_key",
        "AWS_SECRET_ACCESS_KEY": "s3_secret_key",
    }
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip().lstrip("export").strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                attr = key_map.get(k)
                if attr and not getattr(watcher, attr):
                    setattr(watcher, attr, v)
        log.debug("Loaded secrets_env_file: %s", path)
    except OSError as exc:
        log.warning("Could not read secrets_env_file %s: %s", path, exc)


def _read_nf_model_types(repo_dir: str) -> list[str]:
    """Parse model_types from the first matching line in nextflow.config.

    Looks for a line like:
        model_types = ['hoptimus1', 'titan_slide']
    Returns a list of model name strings, or [] if not found / parse error.
    """
    import re
    config_path = os.path.join(repo_dir, "nextflow.config")
    try:
        text = Path(config_path).read_text()
    except OSError:
        log.warning("Could not read %s — model_types not auto-detected", config_path)
        return []
    m = re.search(r"model_types\s*=\s*\[([^\]]*)\]", text)
    if not m:
        return []
    raw = m.group(1)
    models = re.findall(r"['\"]([^'\"]+)['\"]", raw)
    return models


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
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60)
            self._local.conn.row_factory = sqlite3.Row
            # WAL mode allows concurrent reads while a write is in progress,
            # and reduces lock contention across threads.
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=60000")
        return self._local.conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS slides (
                slide_path    TEXT PRIMARY KEY,
                slide_id      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'PENDING',
                batch_id      TEXT,
                download_path TEXT,
                fail_count    INTEGER NOT NULL DEFAULT 0,
                first_seen_at  TEXT,
                dispatched_at  TEXT,
                completed_at   TEXT,
                error_msg      TEXT
            );

            CREATE TABLE IF NOT EXISTS batches (
                batch_id      TEXT PRIMARY KEY,
                csv_path      TEXT,
                work_dir      TEXT,
                status        TEXT NOT NULL DEFAULT 'RUNNING',
                slide_count   INTEGER,
                dispatched_at TEXT,
                completed_at  TEXT,
                nextflow_exit INTEGER,
                log_path      TEXT,
                manifest_path TEXT
            );
        """)
        # Migrate existing databases that pre-date the download_path column.
        try:
            conn.execute("ALTER TABLE slides ADD COLUMN download_path TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migrate existing databases that pre-date the fail_count column.
        try:
            conn.execute("ALTER TABLE slides ADD COLUMN fail_count INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migrate existing databases that pre-date the work_dir column.
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN work_dir TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migrate existing databases that pre-date the nf_session_id column.
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN nf_session_id TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migrate existing databases that pre-date the file_id/file_name/needs_download columns.
        for col_def in (
            "file_id TEXT",
            "file_name TEXT",
            "needs_download INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                conn.execute(f"ALTER TABLE slides ADD COLUMN {col_def}")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    # -- Slides ---------------------------------------------------------------

    def add_slide(self, slide_path: str, slide_id: str, *,
                  file_id: str = "", file_name: str = "", needs_download: bool = False):
        conn = self._conn()
        conn.execute(
            """INSERT OR IGNORE INTO slides
               (slide_path, slide_id, status, file_id, file_name, needs_download, first_seen_at)
               VALUES (?, ?, 'PENDING', ?, ?, ?, ?)""",
            (slide_path, slide_id, file_id, file_name,
             1 if needs_download else 0,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def is_known(self, slide_path: str) -> bool:
        row = self._conn().execute(
            "SELECT status FROM slides WHERE slide_path = ?", (slide_path,)
        ).fetchone()
        return row is not None

    def is_known_by_id(self, slide_id: str) -> bool:
        """Return True if the slide_id already has any record in the DB (any status)."""
        row = self._conn().execute(
            "SELECT status FROM slides WHERE slide_id = ?", (slide_id,)
        ).fetchone()
        return row is not None

    def get_slides_by_id(self, slide_id: str) -> list:
        """Return all DB records for a given slide_id."""
        rows = self._conn().execute(
            "SELECT slide_path, slide_id, status, fail_count FROM slides WHERE slide_id = ?",
            (slide_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_slide(self, slide_path: str):
        """Remove a slide record (used to clear stale PENDING entries before re-download)."""
        conn = self._conn()
        conn.execute("DELETE FROM slides WHERE slide_path = ?", (slide_path,))
        conn.commit()

    def get_pending_slides(self) -> list:
        rows = self._conn().execute(
            "SELECT slide_path, slide_id, file_id, file_name, needs_download FROM slides"
            " WHERE status = 'PENDING' AND slide_path != ''"
        ).fetchall()
        slides = [dict(r) for r in rows]
        # Backfill file_id/file_name from gdc:// URIs for rows migrated from older DB
        # schema that pre-dates those columns (they default to empty string).
        for s in slides:
            if s.get("needs_download") and not s.get("file_id"):
                sp = s.get("slide_path", "")
                if sp.startswith("gdc://"):
                    rest = sp[len("gdc://"):]
                    slash = rest.find("/")
                    if slash > 0:
                        s["file_id"] = rest[:slash]
                        s["file_name"] = rest[slash + 1:]
        return slides

    def mark_dispatched(self, slide_paths: list, batch_id: str):
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "UPDATE slides SET status='DISPATCHED', batch_id=?, dispatched_at=? WHERE slide_path=?",
            [(batch_id, now, sp) for sp in slide_paths],
        )
        conn.commit()

    def mark_slides_complete(self, batch_id: str, succeeded: bool,
                             charge_fail_count: bool = True):
        conn = self._conn()
        status = "SUCCEEDED" if succeeded else "FAILED"
        if succeeded:
            conn.execute(
                "UPDATE slides SET status=?, completed_at=? WHERE batch_id=?",
                (status, datetime.now(timezone.utc).isoformat(), batch_id),
            )
        else:
            if charge_fail_count:
                # Increment fail_count for each slide in this failed batch
                conn.execute(
                    "UPDATE slides SET status=?, completed_at=?, fail_count=fail_count+1 WHERE batch_id=?",
                    (status, datetime.now(timezone.utc).isoformat(), batch_id),
                )
            else:
                # Fast-fail (infra/config error) — reset to PENDING without charging fail_count
                conn.execute(
                    "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL WHERE batch_id=?",
                    (batch_id,),
                )
        conn.commit()

    def reset_dispatched_to_pending(self, batch_id: str):
        conn = self._conn()
        conn.execute(
            "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL WHERE batch_id=? AND status='DISPATCHED'",
            (batch_id,),
        )
        conn.commit()

    def reset_failed_to_pending(self, max_retries: int = 0) -> int:
        """Reset FAILED slides to PENDING so they can be retried.
        Slides with fail_count >= max_retries (when > 0) are left as FAILED.
        Returns count of slides reset."""
        conn = self._conn()
        if max_retries > 0:
            conn.execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL, error_msg=NULL "
                "WHERE status='FAILED' AND fail_count < ?",
                (max_retries,),
            )
            skipped = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status='FAILED' AND fail_count >= ?",
                (max_retries,),
            ).fetchone()[0]
            if skipped:
                log.warning("Permanently skipping %d slide(s) with fail_count >= %d.", skipped, max_retries)
        else:
            conn.execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL, error_msg=NULL WHERE status='FAILED'"
            )
        n = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        return n

    def blacklist_slide(self, slide_id: str, reason: str, max_retries: int = 999):
        """Permanently exclude a slide from future dispatch.

        Sets status=FAILED, error_msg=reason, and fail_count=max_retries so that
        reset_failed_to_pending() will never re-queue it.  The slide stays in the
        DB for audit/tracking purposes.  If the slide is not yet known it is
        inserted first (path stored as empty string so it won't be dispatched).
        """
        conn = self._conn()
        existing = conn.execute(
            "SELECT slide_path FROM slides WHERE slide_id = ?", (slide_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE slides SET status='FAILED', fail_count=?, error_msg=?,
                   completed_at=?, batch_id=NULL
                   WHERE slide_id=?""",
                (max_retries, reason, datetime.now(timezone.utc).isoformat(), slide_id),
            )
        else:
            conn.execute(
                """INSERT INTO slides (slide_path, slide_id, status, fail_count, error_msg, first_seen_at)
                   VALUES ('', ?, 'FAILED', ?, ?, ?)""",
                (slide_id, max_retries, reason, datetime.now(timezone.utc).isoformat()),
            )
        conn.commit()
        log.warning("Blacklisted slide %s: %s", slide_id, reason)

    def set_download_path(self, slide_path: str, download_path: str):
        """Record the local path of a downloaded slide (the file_id directory)."""
        conn = self._conn()
        conn.execute(
            "UPDATE slides SET download_path=? WHERE slide_path=?",
            (download_path, slide_path),
        )
        conn.commit()

    def get_batch_download_paths(self, batch_id: str) -> list:
        """Return distinct non-null download_path values for slides in a batch."""
        rows = self._conn().execute(
            "SELECT DISTINCT download_path FROM slides WHERE batch_id=? AND download_path IS NOT NULL",
            (batch_id,),
        ).fetchall()
        return [r["download_path"] for r in rows]

    def get_old_batch_logs(self, older_than_days: int) -> list:
        """Return (batch_id, log_path) for SUCCEEDED batches with logs older than N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        rows = self._conn().execute(
            """SELECT batch_id, log_path FROM batches
               WHERE status='SUCCEEDED' AND log_path IS NOT NULL AND completed_at < ?""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Batches --------------------------------------------------------------

    def add_batch(self, batch_id: str, csv_path: str, work_dir: str, slide_count: int, log_path: str):
        conn = self._conn()
        conn.execute(
            """INSERT INTO batches (batch_id, csv_path, work_dir, status, slide_count, dispatched_at, log_path)
               VALUES (?, ?, ?, 'RUNNING', ?, ?, ?)""",
            (batch_id, csv_path, work_dir, slide_count, datetime.now(timezone.utc).isoformat(), log_path),
        )
        conn.commit()

    def restart_batch(self, batch_id: str):
        """Re-mark a previously-failed/interrupted batch as RUNNING (for -resume)."""
        conn = self._conn()
        conn.execute(
            "UPDATE batches SET status='RUNNING', completed_at=NULL, nextflow_exit=NULL WHERE batch_id=?",
            (batch_id,),
        )
        conn.commit()

    def complete_batch(self, batch_id: str, exit_code: int):
        conn = self._conn()
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        conn.execute(
            "UPDATE batches SET status=?, completed_at=?, nextflow_exit=? WHERE batch_id=?",
            (status, datetime.now(timezone.utc).isoformat(), exit_code, batch_id),
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
            "SELECT batch_id, csv_path, work_dir, log_path, nf_session_id FROM batches WHERE status='RUNNING'"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_batch_session_id(self, batch_id: str, session_id: str):
        conn = self._conn()
        conn.execute(
            "UPDATE batches SET nf_session_id=? WHERE batch_id=?",
            (session_id, batch_id),
        )
        conn.commit()

    def get_batch_session_id(self, batch_id: str) -> str | None:
        row = self._conn().execute(
            "SELECT nf_session_id FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if row and row["nf_session_id"]:
            return row["nf_session_id"]
        return None

    def get_finished_batches_with_work_dirs(self) -> list:
        """Return SUCCEEDED/FAILED batches whose work_dir column is non-null."""
        rows = self._conn().execute(
            "SELECT batch_id, work_dir FROM batches WHERE status IN ('SUCCEEDED','FAILED') AND work_dir IS NOT NULL"
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
# Databricks watcher
# ---------------------------------------------------------------------------

class DatabricksWatcher(threading.Thread):
    """
    Polls a Databricks SQL warehouse for slides to dispatch.

    Joins:
        cdsi_eng_phi.pdm_base_tables.impact_matched_slides  (m)
        cdsi_eng_phi.pdm_base_tables.slide_inventory        (i)
    on m.image_id = i.image_id

    Produces slide records with:
        slide_id      → m.image_id
        slide_path    → i.path  (S3 URI, e.g. s3://mskmind-bkt/reef-slides/1234.svs)
        oncotree_code → m.ONCOTREE_CODE

    Credentials are resolved by the Databricks SDK in priority order:
        1. DATABRICKS_HOST + DATABRICKS_TOKEN environment variables
        2. ~/.databrickscfg DEFAULT profile

    Config fields (WatcherConfig):
        warehouse_id      Required. SQL warehouse to execute queries against.
        source_filter     Optional list of slide_inventory.source values to include.
                          e.g. ['ECS2'].  Empty list = all sources.
        additional_where  Optional extra SQL WHERE clause appended with AND.
        min_file_size_bytes  Skip slides with i.size < this value (default 10 MB).
        poll_interval_seconds  Seconds between polls. Defaults to 86400 (1 day) if
                               not set in config, since the IMPACT tables update infrequently.
        wds_destinations  Optional {model: s3_or_local_path} dict. When set, a
                          append_wds.py hook is auto-generated for each model
                          that routes slides by oncotree_code into per-cancer-type shards.
        wds_staging_dir   Local staging base dir for s3:// WDS destinations.
        wds_s3_max_concurrency  Boto3 multipart threads per S3 upload (default 4).
    """

    _DEFAULT_POLL_INTERVAL = 86400  # 1 day

    _QUERY_TEMPLATE = """
SELECT
    m.image_id    AS slide_id,
    i.path        AS slide_path,
    m.ONCOTREE_CODE AS oncotree_code
FROM cdsi_eng_phi.pdm_base_tables.impact_matched_slides m
JOIN cdsi_eng_phi.pdm_base_tables.slide_inventory i
  ON m.image_id = i.image_id
WHERE i.path IS NOT NULL
  AND i.size >= {min_size}
{source_clause}{additional_clause}
"""

    def __init__(self, cfg: WatcherConfig, pending: deque, state: StateStore,
                 stop_event: threading.Event):
        super().__init__(daemon=True, name="DatabricksWatcher")
        self.cfg = cfg
        self.pending = pending
        self.state = state
        self.stop_event = stop_event
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from databricks.sdk import WorkspaceClient
            except ImportError:
                raise RuntimeError(
                    "databricks-sdk required for Databricks watching: pip install databricks-sdk"
                )
            self._client = WorkspaceClient()
        return self._client

    def run(self):
        if not self.cfg.warehouse_id:
            log.error("DatabricksWatcher: warehouse_id is required — watcher will not start")
            return
        interval = self.cfg.poll_interval_seconds or self._DEFAULT_POLL_INTERVAL
        log.info("DatabricksWatcher started (warehouse=%s, poll_interval=%ds)",
                 self.cfg.warehouse_id, interval)
        self._poll()
        while not self.stop_event.is_set():
            self.stop_event.wait(interval)
            if not self.stop_event.is_set():
                self._poll()

    def _build_query(self) -> str:
        source_clause = ""
        if self.cfg.source_filter:
            quoted = ", ".join(f"'{s}'" for s in self.cfg.source_filter)
            source_clause = f"  AND i.source IN ({quoted})\n"

        additional_clause = ""
        if self.cfg.additional_where:
            additional_clause = f"  AND ({self.cfg.additional_where})\n"

        return self._QUERY_TEMPLATE.format(
            min_size=self.cfg.min_file_size_bytes,
            source_clause=source_clause,
            additional_clause=additional_clause,
        ).strip()

    def _poll(self):
        log.info("DatabricksWatcher: querying Databricks warehouse %s…", self.cfg.warehouse_id)
        try:
            from databricks.sdk.service.sql import StatementState
            client = self._get_client()
            query = self._build_query()
            log.debug("DatabricksWatcher SQL:\n%s", query)

            resp = client.statement_execution.execute_statement(
                warehouse_id=self.cfg.warehouse_id,
                statement=query,
                wait_timeout="50s",
            )

            # Poll until terminal state if not already done
            while resp.status.state not in (
                StatementState.SUCCEEDED,
                StatementState.FAILED,
                StatementState.CANCELED,
                StatementState.CLOSED,
            ):
                time.sleep(2)
                resp = client.statement_execution.get_statement(resp.statement_id)

            if resp.status.state != StatementState.SUCCEEDED:
                log.error(
                    "DatabricksWatcher: query failed (state=%s): %s",
                    resp.status.state,
                    resp.status.error,
                )
                return

            cols = [c.name for c in resp.manifest.schema.columns]
            n_new = 0

            # Iterate over all result chunks
            chunk = resp.result
            while chunk is not None:
                for row_arr in chunk.data_array or []:
                    row = dict(zip(cols, row_arr))
                    slide_path = row.get("slide_path") or ""
                    slide_id = row.get("slide_id") or ""
                    oncotree_code = row.get("oncotree_code") or ""

                    if not slide_path or not slide_id:
                        continue
                    if self.state.is_known(slide_path):
                        continue

                    log.info("DatabricksWatcher: new slide %s → %s", slide_id, slide_path)
                    self.state.add_slide(slide_path, slide_id)
                    self.pending.append({
                        "slide_id": slide_id,
                        "slide_path": slide_path,
                        "oncotree_code": oncotree_code,
                    })
                    n_new += 1

                # Advance to next chunk (if result was paginated)
                next_chunk_index = getattr(chunk, "next_chunk_index", None)
                if next_chunk_index is None:
                    break
                chunk_resp = client.statement_execution.get_statement_result_chunk_n(
                    resp.statement_id, next_chunk_index
                )
                chunk = chunk_resp

            log.info("DatabricksWatcher: poll complete — %d new slide(s) enqueued", n_new)

        except Exception as exc:
            log.error("DatabricksWatcher: poll error: %s", exc, exc_info=True)


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
      5. Slides not found on S3 are enqueued with needs_download=true; NF's
         DOWNLOAD_SLIDE process fetches them from GDC (storeDir-cached).
    """

    def __init__(
        self,
        cfg: WatcherConfig,
        pending: deque,
        state: StateStore,
        stop_event: threading.Event,
        repo_dir: str,
        outdir: str,
        max_slide_retries: int = 0,
    ):
        super().__init__(name="tcga-watcher", daemon=True)
        self.cfg = cfg
        self.pending = pending
        self.state = state
        self.stop_event = stop_event
        self._outdir = outdir
        self._scripts_dir = cfg.scripts_dir or str(Path(repo_dir) / "scripts" / "tcga")
        self._max_slide_retries = max_slide_retries

    def run(self):
        log.info(
            "TcgaWatcher started (poll_interval=%ds)",
            self.cfg.poll_interval_seconds,
        )
        self._poll()  # poll immediately on startup
        while not self.stop_event.is_set():
            self.stop_event.wait(self.cfg.poll_interval_seconds)
            if not self.stop_event.is_set():
                self._poll()

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

        # 2. Update per-slide status from results directory + WDS manifest
        status_args = [
            "--inventory", self.cfg.inventory_csv,
            "--results-dir", self._outdir,
            "--output", self.cfg.status_csv,
        ]
        if self.cfg.models:
            status_args += ["--model-types", ",".join(self.cfg.models)]
        if self.cfg.slide_type and self.cfg.slide_type.lower() != "all":
            status_args += ["--slide-type", self.cfg.slide_type]
        wds_manifest_path = os.path.join(self._outdir, "wds_manifest.csv")
        if os.path.exists(wds_manifest_path):
            status_args += ["--wds-manifest", wds_manifest_path]
        if self.cfg.wds_destinations:
            wds_base_parts = ",".join(
                f"{model}={dest}" for model, dest in self.cfg.wds_destinations.items()
            )
            status_args += ["--wds-base", wds_base_parts]
        # Export FAILED slides from dispatcher DB to a CSV the status script can read.
        # (Direct sqlite3 import may fail in some conda environments due to libstdc++ version.)
        failed_csv = os.path.join(os.path.dirname(__file__), "tcga_failed_slides.csv")
        dispatcher_db = os.path.join(os.path.dirname(__file__), "state", "dispatcher.db")
        if os.path.exists(dispatcher_db):
            try:
                import sqlite3 as _sqlite3
                conn = _sqlite3.connect(f"file:{dispatcher_db}?mode=ro", uri=True)
                rows = conn.execute(
                    "SELECT slide_id, error_msg FROM slides WHERE status='FAILED'"
                ).fetchall()
                conn.close()
                import csv as _csv
                with open(failed_csv, "w", newline="") as fh:
                    w = _csv.writer(fh)
                    w.writerow(["slide_id", "failure_reason"])
                    for sid, msg in rows:
                        w.writerow([sid, msg or "Failed after max retries"])
                log.info("TcgaWatcher: exported %d FAILED slides to %s", len(rows), failed_csv)
            except Exception as exc:
                log.warning("TcgaWatcher: could not export failed slides from DB: %s", exc)
        if os.path.exists(failed_csv):
            status_args += ["--dispatcher-db", failed_csv]
        slide_mpp_csv = os.path.join(os.path.dirname(__file__), "tcga_slide_mpp.csv")
        if os.path.exists(slide_mpp_csv):
            status_args += ["--slide-mpp", slide_mpp_csv]
        rc = self._run_script("tcga_update_status.py", status_args)
        if rc != 0:
            log.error("TcgaWatcher: status update failed — skipping this poll")
            return

        # 3. Resolve slide paths (local / S3 / needs_download)
        samples_csv = str(Path(self.cfg.status_csv).with_suffix("")) + "_dispatcher.csv"
        prepare_args = [
            "--inventory", self.cfg.inventory_csv,
            "--status", self.cfg.status_csv,
            "--output", samples_csv,
        ]
        # Skip slides already done.  When multiple models are configured, pass the
        # first one; a slide is re-queued if any model still needs extracting.
        if self.cfg.models:
            prepare_args += ["--model", self.cfg.models[0], "--skip-done"]
        elif self.cfg.status_csv:
            # No explicit models — auto-discover; still skip slides marked done
            # for any model found in the status CSV.
            prepare_args += ["--skip-done"]
        if self.cfg.local_slides_dir:
            prepare_args += ["--local-slides-dir", self.cfg.local_slides_dir]
        if self.cfg.s3_base:
            prepare_args += ["--s3-base", self.cfg.s3_base, "--check-s3-exists"]
        if self.cfg.s3_endpoint:
            prepare_args += ["--s3-endpoint", self.cfg.s3_endpoint]
        if self.cfg.project:
            prepare_args += ["--project", self.cfg.project]
        if self.cfg.slide_type and self.cfg.slide_type.lower() != "all":
            prepare_args += ["--slide-type", self.cfg.slide_type]
        if self.cfg.sample_type and self.cfg.sample_type.lower() != "all":
            prepare_args += ["--sample-type", self.cfg.sample_type]

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

        # Enqueue needs-download slides — NF's DOWNLOAD_SLIDE process handles
        # the actual GDC download via storeDir (cached across runs).
        # Use a synthetic gdc:// URI as the DB primary key so no path knowledge
        # is needed in the dispatcher; slide_path in the batch CSV is set to the
        # bare filename (correct extension, NF ignores it for needs_download slides).
        if needs_dl:
            n_new = 0
            for s in needs_dl:
                file_id = s.get("file_id", "")
                file_name = s.get("file_name", "")
                slide_id = s["slide_id"]
                if not file_id or not file_name:
                    log.warning("TcgaWatcher: missing file_id/file_name for %s — skipping", slide_id)
                    continue
                # Synthetic URI used as the unique DB key.
                db_key = f"gdc://{file_id}/{file_name}"
                if self.state.is_known(db_key):
                    continue
                # Check if this slide is permanently blacklisted (fail_count >= max_retries),
                # or already SUCCEEDED/PENDING/DISPATCHED via another path (e.g. s3:// or
                # a local path) — skip in both cases to avoid duplicate slide_id in batches.
                existing = self.state.get_slides_by_id(slide_id)
                if any(r.get("status") == "SUCCEEDED" for r in existing):
                    log.debug(
                        "TcgaWatcher: skipping %s — already SUCCEEDED via another path", slide_id
                    )
                    continue
                # If already PENDING or DISPATCHED via a non-gdc path, don't add a second
                # gdc:// entry — it would produce duplicate slide_ids in the batch CSV.
                if any(
                    r.get("status") in ("PENDING", "DISPATCHED")
                    and not r.get("slide_path", "").startswith("gdc://")
                    for r in existing
                ):
                    log.debug(
                        "TcgaWatcher: skipping gdc:// for %s — already PENDING/DISPATCHED via %s",
                        slide_id, next(
                            r["slide_path"] for r in existing
                            if r.get("status") in ("PENDING", "DISPATCHED")
                            and not r.get("slide_path", "").startswith("gdc://")
                        ),
                    )
                    continue
                max_retries = self._max_slide_retries
                if max_retries > 0 and any(
                    r.get("fail_count", 0) >= max_retries for r in existing
                ):
                    log.debug(
                        "TcgaWatcher: skipping permanently blacklisted slide %s", slide_id
                    )
                    continue
                # Remove any stale S3 record (PENDING or non-permanent FAILED) for the
                # same slide_id.  This happens when a slide was first found on S3 then
                # disappeared — we fall back to GDC download instead.
                for r in existing:
                    if r["slide_path"].startswith("s3://") and r["status"] in ("PENDING", "FAILED"):
                        log.info(
                            "TcgaWatcher: removing stale S3 %s record for %s (%s)",
                            r["status"], slide_id, r["slide_path"],
                        )
                        self.state.remove_slide(r["slide_path"])
                self.state.add_slide(
                    db_key, slide_id,
                    file_id=file_id, file_name=file_name, needs_download=True,
                )
                self.pending.append({
                    "slide_id": slide_id,
                    "slide_path": file_name,   # bare filename; NF ignores for needs_download
                    "file_id": file_id,
                    "file_name": file_name,
                    "needs_download": True,
                    "_db_key": db_key,         # internal: used by mark_dispatched
                })
                n_new += 1
            if n_new:
                log.info("TcgaWatcher: queued %d slide(s) for GDC download via NF", n_new)


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
# NF session ID helpers
# ---------------------------------------------------------------------------

def _parse_run_name_from_log(log_path: str) -> str | None:
    """Parse the Nextflow run name from the batch stdout log file.

    NF prints a line like:
        runName                 : desperate_meucci
    """
    try:
        with open(log_path) as f:
            for line in f:
                if "runName" in line and ":" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _lookup_session_id_in_history(repo_dir: str, run_name: str) -> str | None:
    """Look up a NF session UUID from .nextflow/history by run name.

    History columns (tab-separated):
        date  time  duration  run_name  status  session_uuid  command
    """
    history_file = os.path.join(repo_dir, ".nextflow", "history")
    try:
        with open(history_file) as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 6 and parts[3].strip() == run_name:
                    return parts[5].strip()
    except OSError:
        pass
    return None


def _extract_nf_session_id_from_log(log_path: str, repo_dir: str) -> str | None:
    """Extract the NF session UUID for the batch identified by log_path."""
    run_name = _parse_run_name_from_log(log_path)
    if not run_name:
        return None
    return _lookup_session_id_in_history(repo_dir, run_name)


def _lookup_nf_session_id(repo_dir: str, batch_id: str, log_path: str) -> str | None:
    """Look up the NF session UUID for a batch, trying DB first then log file.

    The session ID is stored in the DB after the first run completes.
    For batches that ran before this feature was added, fall back to parsing
    the run name from the log file and looking it up in .nextflow/history.
    """
    return _extract_nf_session_id_from_log(log_path, repo_dir)


# ---------------------------------------------------------------------------
# NextflowRunner
# ---------------------------------------------------------------------------

class NextflowRunner:
    def __init__(self, cfg: Config, batch_id: str, slides: list, state: StateStore,
                 *, resume: bool = False, existing_csv_path: str = None, existing_work_dir: str = None):
        self.cfg = cfg
        self.batch_id = batch_id
        self.slides = slides  # list of {"slide_path": ..., "slide_id": ..., "oncotree_code": ...}
        self.state = state
        self._resume = resume
        self._existing_csv_path = existing_csv_path
        self._existing_work_dir = existing_work_dir

    def run(self):
        if self._resume:
            csv_path = self._existing_csv_path
            work_dir = self._existing_work_dir
            log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
            self.state.restart_batch(self.batch_id)
        else:
            csv_path = self._write_csv()
            work_dir = os.path.join(self.cfg.work_base_dir, f"batch_{self.batch_id}", "work")
            log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(self.cfg.log_dir, exist_ok=True)

            self.state.add_batch(self.batch_id, csv_path, work_dir, len(self.slides), log_path)
            self.state.mark_dispatched(
                [s.get("_db_key") or s["slide_path"] for s in self.slides],
                self.batch_id,
            )

        cmd = [
            "nextflow", "run", self.cfg.repo_dir,
            "-profile", self.cfg.nextflow_profiles,
            "-work-dir", work_dir,
            "--samples_csv", csv_path,
            "--outdir", self.cfg.outdir,
        ]
        if self.cfg.nextflow_config:
            cmd += ["-c", self.cfg.nextflow_config]
        if self.cfg.nextflow_params_file:
            cmd += ["-params-file", self.cfg.nextflow_params_file]
        # If this batch includes slides that need GDC download, pass the token
        # file to NF if configured (open-access TCGA data needs no token).
        has_download = any(s.get("needs_download") for s in self.slides)
        if has_download:
            for w in self.cfg.watchers:
                if getattr(w, "gdc_token_file", ""):
                    cmd += ["--download.gdc_token_file", w.gdc_token_file]
                    break
        if self._resume:
            # Use the stored session ID to avoid lock conflicts when multiple batches
            # resume concurrently (bare -resume uses the last entry in the shared
            # .nextflow/history, causing all resumes to target the same session).
            session_id = self.state.get_batch_session_id(self.batch_id)
            if not session_id:
                # Fall back to parsing from log + history (for pre-fix batches)
                session_id = _lookup_nf_session_id(
                    self.cfg.repo_dir, self.batch_id, log_path
                )
            if session_id:
                cmd += ["-resume", session_id]
                log.info("Batch %s: resuming NF session %s", self.batch_id, session_id)
            else:
                cmd.append("-resume")
                log.warning("Batch %s: session ID not found, resuming without explicit ID", self.batch_id)

        label = "resume" if self._resume else f"{len(self.slides)} slides"
        log.info("Dispatching batch %s (%s): %s", self.batch_id, label, " ".join(cmd))

        run_started_at = time.time()
        exit_code = -1
        run_env = None
        if self.cfg.nextflow_version:
            run_env = dict(os.environ)
            run_env["NXF_VER"] = self.cfg.nextflow_version
        try:
            with open(log_path, "w") as lf:
                result = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    cwd=self.cfg.repo_dir,
                    env=run_env,
                    start_new_session=True,
                )
            exit_code = result.returncode
        except Exception as e:
            log.error("Batch %s failed to launch: %s", self.batch_id, e)

        # Capture and store the NF session ID so future resumes can use -resume <uuid>
        # instead of bare -resume (which would collide across concurrent batches).
        if not self._resume:
            session_id = _extract_nf_session_id_from_log(log_path, self.cfg.repo_dir)
            if session_id:
                self.state.set_batch_session_id(self.batch_id, session_id)
                log.debug("Batch %s: recorded NF session ID %s", self.batch_id, session_id)

        self.state.complete_batch(self.batch_id, exit_code)

        # Only charge fail_count if the batch ran long enough to have actually
        # attempted slide-level work (≥60 s). Shorter runs indicate an infra or
        # config failure (wrong binary, bad params file, NF launch error) — those
        # should reset slides to PENDING rather than burning a retry slot.
        batch_duration = time.time() - run_started_at
        fast_fail = exit_code != 0 and batch_duration < 60
        if fast_fail:
            log.warning(
                "Batch %s failed in %.0fs — treating as infra failure, resetting slides to PENDING.",
                self.batch_id, batch_duration,
            )
        self.state.mark_slides_complete(self.batch_id, exit_code == 0,
                                        charge_fail_count=not fast_fail)

        if exit_code == 0:
            log.info("Batch %s completed successfully.", self.batch_id)
            self._collect_manifest(run_started_at)
            self._run_post_batch_hooks(csv_path)
            self._verify_wds_coverage(csv_path)
            self._cleanup(csv_path, log_path, work_dir, succeeded=True)
        else:
            log.error("Batch %s failed (exit %d). Log: %s", self.batch_id, exit_code, log_path)
            self._cleanup(csv_path, log_path, work_dir, succeeded=False)

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

        # Build env with ECS credentials injected from any tcga watcher that has them.
        hook_env = os.environ.copy()
        for w in self.cfg.watchers:
            if w.s3_access_key:
                hook_env.setdefault("ECS_ACCESS_KEY", w.s3_access_key)
            if w.s3_secret_key:
                hook_env.setdefault("ECS_SECRET_KEY", w.s3_secret_key)
            if w.s3_endpoint:
                hook_env.setdefault("ECS_ENDPOINT_URL", w.s3_endpoint)

        for hook in self.cfg.post_batch_hooks:
            cmd_str = hook.get("command", "")
            if not cmd_str:
                log.warning("Post-batch hook has no 'command' key — skipping")
                continue
            cmd = [_sub(p) for p in cmd_str.split()] + [_sub(a) for a in hook.get("args", [])]
            log.info("Batch %s: running post-batch hook: %s", self.batch_id, " ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                       cwd=self.cfg.repo_dir, env=hook_env)
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

    def _verify_wds_coverage(self, batch_csv: str) -> None:
        """After append_wds hooks, verify all batch slides appear in WDS for every
        configured model. Slides missing from any model are reset to PENDING so they
        get retried — append_wds.py will skip already-indexed slides on the next run."""
        import csv as _csv
        from collections import defaultdict

        # Collect required models from TcgaWatcher wds_destinations configs
        required_models: set[str] = set()
        for w in self.cfg.watchers:
            if hasattr(w, "wds_destinations") and w.wds_destinations:
                required_models.update(w.wds_destinations.keys())
        if not required_models:
            return

        manifest_path = os.path.join(self.cfg.outdir, "wds_manifest.csv")
        if not os.path.exists(manifest_path):
            log.debug("_verify_wds_coverage: manifest not found at %s, skipping", manifest_path)
            return

        # Read current WDS coverage per model
        wds_by_model: dict[str, set[str]] = defaultdict(set)
        with open(manifest_path) as f:
            for row in _csv.DictReader(f):
                wds_by_model[row["model"]].add(row["slide_id"])

        # Read batch slide IDs
        batch_ids: set[str] = set()
        with open(batch_csv) as f:
            for row in _csv.DictReader(f):
                batch_ids.add(row["slide_id"])

        # Slides missing from at least one required model
        incomplete: set[str] = set()
        for slide_id in batch_ids:
            for model in required_models:
                if slide_id not in wds_by_model.get(model, set()):
                    incomplete.add(slide_id)
                    break

        if not incomplete:
            log.info("Batch %s: WDS coverage verified — all %d slide(s) present for all models.",
                     self.batch_id, len(batch_ids))
            return

        log.warning(
            "Batch %s: %d/%d slide(s) missing from WDS for ≥1 model — resetting to PENDING: %s%s",
            self.batch_id, len(incomplete), len(batch_ids),
            ", ".join(sorted(incomplete)[:5]),
            " …" if len(incomplete) > 5 else "",
        )
        conn = self.state._conn()
        conn.executemany(
            "UPDATE slides SET status='PENDING', fail_count=fail_count+1, "
            "batch_id=NULL, dispatched_at=NULL "
            "WHERE slide_id=? AND status='SUCCEEDED'",
            [(sid,) for sid in incomplete],
        )
        conn.commit()
        log.info("Batch %s: reset %d incomplete slide(s) to PENDING (will retry).",
                 self.batch_id, len(incomplete))

    def _cleanup(self, csv_path: str, log_path: str, work_dir: str, *, succeeded: bool = True):
        """Post-batch cleanup. Work dir is removed on both success and failure when enabled.
        Downloads, batch CSV, and log rotation only run on success."""
        import shutil

        if self.cfg.cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
            log.info("Batch %s: removed work dir %s", self.batch_id, work_dir)
            try:
                os.rmdir(os.path.dirname(work_dir))  # remove empty batch_{id}/ parent
            except OSError:
                pass

        if not succeeded:
            return

        if self.cfg.cleanup_downloads:
            for dl_dir in self.state.get_batch_download_paths(self.batch_id):
                if os.path.isdir(dl_dir):
                    shutil.rmtree(dl_dir, ignore_errors=True)
                    log.info("Batch %s: removed download dir %s", self.batch_id, dl_dir)

        if self.cfg.cleanup_batch_csv and os.path.exists(csv_path):
            os.unlink(csv_path)
            log.info("Batch %s: removed batch CSV %s", self.batch_id, csv_path)

        if self.cfg.cleanup_logs_after_days > 0:
            for row in self.state.get_old_batch_logs(self.cfg.cleanup_logs_after_days):
                old_log = row["log_path"]
                if old_log and os.path.exists(old_log):
                    os.unlink(old_log)
                    log.info("Removed old log (batch %s): %s", row["batch_id"], old_log)

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
        # Deduplicate by slide_id: if the same slide_id appears with multiple paths
        # (e.g. a local/s3 path AND a gdc:// path), keep the non-gdc:// one to avoid
        # MERGE_SAMPLE_FEATURES input file name collisions.
        seen_ids: dict[str, dict] = {}
        for s in self.slides:
            sid = s.get("slide_id", "")
            if sid not in seen_ids:
                seen_ids[sid] = s
            else:
                prev = seen_ids[sid]
                # Prefer the non-gdc:// path (already downloaded local/s3 path)
                if prev.get("slide_path", "").startswith("gdc://") and \
                        not s.get("slide_path", "").startswith("gdc://"):
                    log.warning(
                        "Batch %s: duplicate slide_id %s — replacing gdc:// path with %s",
                        self.batch_id, sid, s["slide_path"],
                    )
                    seen_ids[sid] = s
                else:
                    log.warning(
                        "Batch %s: duplicate slide_id %s — keeping %s, dropping %s",
                        self.batch_id, sid,
                        prev.get("slide_path"), s.get("slide_path"),
                    )
        deduped_slides = list(seen_ids.values())
        if len(deduped_slides) < len(self.slides):
            log.warning(
                "Batch %s: removed %d duplicate slide_id(s) before writing CSV",
                self.batch_id, len(self.slides) - len(deduped_slides),
            )
        # Include GDC download fields if any slide in the batch needs download.
        has_download = any(s.get("needs_download") for s in deduped_slides)
        fieldnames = ["slide_id", "slide_path", "oncotree_code"]
        if has_download:
            fieldnames += ["needs_download", "file_id", "file_name"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in deduped_slides:
                if not s.get("slide_path"):
                    log.warning("Skipping slide %s with empty slide_path", s.get("slide_id"))
                    continue
                row = {
                    "slide_id": s["slide_id"],
                    "slide_path": s["slide_path"],
                    "oncotree_code": s.get("oncotree_code", ""),
                }
                if has_download:
                    file_id = s.get("file_id", "") or ""
                    file_name = s.get("file_name", "") or ""
                    needs_dl = bool(s.get("needs_download"))
                    # Backfill from gdc:// URI if file_id/file_name are missing
                    if not file_id:
                        sp = s.get("slide_path", "")
                        if sp.startswith("gdc://"):
                            rest = sp[len("gdc://"):]
                            slash = rest.find("/")
                            if slash > 0:
                                file_id = rest[:slash]
                                file_name = rest[slash + 1:]
                    if needs_dl and not file_id:
                        log.warning(
                            "Batch %s: slide %s needs_download but has no file_id — skipping",
                            self.batch_id, s.get("slide_id"),
                        )
                        continue
                    row["needs_download"] = "true" if needs_dl else "false"
                    row["file_id"] = file_id
                    row["file_name"] = file_name
                writer.writerow(row)
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

    def run(self):
        """Scheduler loop — call in the main thread."""
        log.info("BatchScheduler started (batch_size=%d, max_wait=%ds)",
                 self.cfg.batch_size, self.cfg.max_wait_seconds)
        while not self.stop_event.is_set():
            self._maybe_dispatch()
            self.stop_event.wait(5)
        # Do NOT force-dispatch on shutdown — pending slides stay in the DB
        # and will be recovered on next startup.
        log.info("BatchScheduler stopped. Pending slides will be recovered on restart.")

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


if __name__ == "__main__":
    main()
