"""Watchers and readiness-checker for mussel-dispatcher."""
from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .config import WatcherConfig
from .state import StateStore

log = logging.getLogger("mussel-dispatcher")


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
        return (now - prev[2]) >= self.stability_wait

    def discard(self, path: str) -> None:
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
# Databricks Watcher
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
                          mussel_dispatcher.wds hook is auto-generated for each model
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
# TCGA Watcher
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
        self._max_slide_retries = max_slide_retries
        from concurrent.futures import ThreadPoolExecutor
        self._download_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tcga-dl")

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

    def _run_script(self, module: str, args: list) -> int:
        cmd = [sys.executable, "-m", f"mussel_dispatcher.tcga.{module}"] + args
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
        rc = self._run_script("sync_inventory", sync_args)
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
        rc = self._run_script("update_status", status_args)
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

        rc = self._run_script("prepare_samples", prepare_args)
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

        # Handle slides that need download.
        # Mode A (download_enabled=True): dispatcher downloads via gdc-client first,
        #   then enqueues to pending once the file is on disk.
        # Mode B (default): pass needs_download=true through to NF DOWNLOAD_SLIDE process.
        if needs_dl and self.cfg.download_enabled:
            for s in needs_dl:
                if not self.state.is_known(s.get("slide_path", "")):
                    self._download_executor.submit(self._download_and_enqueue, s)
        elif needs_dl:
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

    def _download_and_enqueue(self, slide: dict) -> None:
        """Download a slide via gdc-client, then enqueue it for feature extraction.

        Called in a background thread when download_enabled=True.
        """
        slide_path = slide.get("slide_path", "")
        file_id = slide.get("file_id", "")
        file_name = slide.get("file_name", "")
        slide_id = slide.get("slide_id", "")
        download_dir = self.cfg.download_dir

        if not file_id:
            log.warning("TcgaWatcher: missing file_id for %s — cannot download", slide_id)
            return

        token_args = ["-t", self.cfg.gdc_token_file] if self.cfg.gdc_token_file else []
        cmd = ["gdc-client", "download"] + token_args + ["-d", download_dir, file_id]
        log.info("TcgaWatcher: downloading %s (%s)", slide_id, file_id)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("TcgaWatcher: gdc-client failed for %s: %s", slide_id, result.stderr)
            return

        # After download, verify the file is present
        if not Path(slide_path).exists():
            log.error("TcgaWatcher: expected file not found after download: %s", slide_path)
            return

        self.state.add_slide(slide_path, slide_id,
                             file_id=file_id, file_name=file_name)
        self.pending.append({"slide_id": slide_id, "slide_path": slide_path})
        log.info("TcgaWatcher: downloaded and enqueued %s", slide_id)


# ---------------------------------------------------------------------------
# Manifest collection (legacy — kept for backward compat)
# ---------------------------------------------------------------------------

MANIFEST_HEADER = ["slide_id", "sample_id", "workflow_id", "key", "value"]


