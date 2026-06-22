"""Nextflow runner and manifest collection for mussel-dispatcher."""
from __future__ import annotations

import csv
import glob as _glob
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .state import StateStore

MANIFEST_HEADER = ["slide_id", "workflow_id", "key", "value"]

# Nextflow run name must match this pattern (validated before launch to catch
# regressions in name generation before NF rejects them at invocation time).
_NF_RUN_NAME_RE = re.compile(r'^[a-z](?:[a-z\d]|[-_](?=[a-z\d])){0,79}$')
_NF_SESSION_LOCK_ERROR = "Unable to acquire lock on session"
_AUTO_RESUME_LOCK_RETRY_DELAY_SECONDS = 30

log = logging.getLogger("mussel-dispatcher")


def _file_contains(path: str, needle: str) -> bool:
    try:
        with open(path, errors="ignore") as f:
            return needle in f.read()
    except OSError:
        return False

def collect_manifests(outdir: str, combined_path: str) -> int:
    """
    Scan *outdir* for all ``manifest-*.csv`` files produced by individual
    Nextflow runs, merge them into *combined_path*, and return the number of
    unique rows written.

    Each per-run manifest has no header. Supports both 4-column rows::

        slide_id,workflow_id,key,value

    and legacy 5-column rows (``sample_id`` column is discarded)::

        slide_id,sample_id,workflow_id,key,value

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
                    if len(parts) == 5:
                        # Legacy 5-col: slide_id, sample_id, workflow_id, key, value
                        slide_id, _sample_id, workflow_id, key, value = parts
                    elif len(parts) == 4:
                        slide_id, workflow_id, key, value = parts
                    else:
                        log.warning("collect_manifests: skipping malformed line in %s: %r", mf, parts)
                        continue
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


def _query_session_id_via_nf_cli(repo_dir: str, run_name: str) -> str | None:
    """Query NF session UUID via ``nextflow log <run_name> -f session_id``.

    This is the preferred path: it uses NF's own CLI abstraction and avoids
    direct parsing of the internal ``.nextflow/history`` file format.
    Returns None on any failure (NF not on PATH, run not found, etc.).
    """
    try:
        result = subprocess.run(
            ["nextflow", "log", run_name, "-f", "session_id"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=15,
        )
        if result.returncode == 0:
            session_id = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
            if re.match(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", session_id):
                return session_id
    except Exception:
        pass
    return None


def _lookup_session_id_in_history(repo_dir: str, run_name: str) -> str | None:
    """Look up a NF session UUID from .nextflow/history by run name (fallback).

    History columns (tab-separated):
        date  time  duration  run_name  status  session_uuid  command

    Prefer ``_query_session_id_via_nf_cli`` over this function; it reads the
    internal history file directly and is fragile against format changes.
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
    """Extract the NF session UUID for the batch identified by log_path (legacy fallback)."""
    run_name = _parse_run_name_from_log(log_path)
    if not run_name:
        return None
    return _lookup_session_id_in_history(repo_dir, run_name)


def _extract_session_id_from_nf_debug_log(nf_log_path: str) -> str | None:
    """Extract the NF session UUID directly from Nextflow's debug log."""
    try:
        with open(nf_log_path) as f:
            for line in f:
                if "Session UUID:" not in line:
                    continue
                candidate = line.rsplit("Session UUID:", 1)[-1].strip()
                if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", candidate):
                    return candidate
    except OSError:
        pass
    return None


def _lookup_nf_session_id(
    repo_dir: str, batch_id: str, log_path: str, nf_log_path: str | None = None
) -> str | None:
    """Look up the NF session UUID for a batch, preferring the NF debug log.

    The session ID is stored in the DB after the first run completes.
    For batches that ran before this feature was added, fall back to parsing
    the run name from the log file and looking it up in .nextflow/history.
    """
    if nf_log_path:
        session_id = _extract_session_id_from_nf_debug_log(nf_log_path)
        if session_id:
            return session_id
    return _extract_nf_session_id_from_log(log_path, repo_dir)


# ---------------------------------------------------------------------------
# NextflowRunner
# ---------------------------------------------------------------------------

class NextflowRunner:
    def __init__(self, cfg: Config, batch_id: str, slides: list, state: StateStore,
                 *, resume: bool = False, existing_csv_path: str = None,
                 existing_work_dir: str = None, shutdown_event: threading.Event | None = None):
        self.cfg = cfg
        self.batch_id = batch_id
        self.slides = slides  # list of {"slide_path": ..., "slide_id": ..., "oncotree_code": ...}
        self.state = state
        self._resume = resume
        self._existing_csv_path = existing_csv_path
        self._existing_work_dir = existing_work_dir
        self._shutdown_event = shutdown_event

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
            slide_keys = [s.get("_db_key") or s["slide_path"] for s in self.slides]
            claimed = self.state.mark_dispatched(slide_keys, self.batch_id)
            if claimed == 0:
                log.warning(
                    "Batch %s: 0 of %d slides claimed (all already dispatched by another instance). "
                    "Aborting batch.",
                    self.batch_id, len(self.slides),
                )
                self.state.cancel_batch(self.batch_id)
                return
            if claimed < len(self.slides):
                log.warning(
                    "Batch %s: only %d of %d slides claimed; %d were already dispatched "
                    "by another instance and will be skipped.",
                    self.batch_id, claimed, len(self.slides), len(self.slides) - claimed,
                )

        # Use the 8-char UUID hash as the NF run name so it fits within
        # Seqera Platform's 16-char workflow.id field limit.
        # Prefix with "r" to ensure it starts with a letter (Nextflow requirement).
        # For resumes: generate a fresh UUID to avoid NF's "run name already used"
        # rejection — NF ≥ 23.x tracks all run names in .nextflow/history and rejects
        # reuse even when -resume <session_id> is supplied.
        if self._resume:
            nf_run_name = "r" + uuid.uuid4().hex[:8]
        else:
            nf_run_name = "r" + self.batch_id.rsplit("_", 1)[-1]  # e.g. "r410bb3ce"
        if not _NF_RUN_NAME_RE.match(nf_run_name):
            raise ValueError(
                f"Generated NF run name {nf_run_name!r} does not match Nextflow's "
                f"required pattern {_NF_RUN_NAME_RE.pattern!r}. "
                f"This is a bug in run name generation (batch_id={self.batch_id!r})."
            )
        trace_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.trace.tsv")
        nf_log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.nf.log")
        cmd = [
            "nextflow", "-log", nf_log_path,
            "run", self.cfg.repo_dir,
            "-profile", self.cfg.nextflow_profiles,
            "-work-dir", work_dir,
            "--samples_csv", csv_path,
            "--outdir", self.cfg.outdir,
            "-name", nf_run_name,
            "-with-trace", trace_path,
        ]
        if self.cfg.nextflow_config:
            cmd += ["-c", self.cfg.nextflow_config]
        if self.cfg.nextflow_params_file:
            cmd += ["-params-file", self.cfg.nextflow_params_file]
        if self.cfg.tower_endpoint:
            cmd += ["-with-tower"]
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
                    self.cfg.repo_dir, self.batch_id, log_path, nf_log_path=nf_log_path
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
        run_env = dict(os.environ)
        if self.cfg.nextflow_version:
            run_env["NXF_VER"] = self.cfg.nextflow_version
        if self.cfg.tower_endpoint:
            run_env["TOWER_API_ENDPOINT"] = self.cfg.tower_endpoint
            run_env["TOWER_ACCESS_TOKEN"] = run_env.get("TOWER_ACCESS_TOKEN") or "local"
        try:
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    cwd=self.cfg.repo_dir,
                    env=run_env,
                    start_new_session=True,
                )
            # Store PID immediately so recover_in_flight() can kill this process
            # if the dispatcher crashes and restarts before this batch completes.
            self.state.set_batch_nf_pid(self.batch_id, proc.pid)
            log.debug("Batch %s: NF process PID=%d", self.batch_id, proc.pid)
            exit_code = proc.wait()
        except Exception as e:
            log.error("Batch %s failed to launch: %s", self.batch_id, e)

        # Clear the stored PID — this process is done.
        self.state.set_batch_nf_pid(self.batch_id, None)

        # Capture and store the NF session ID so future resumes can use -resume <uuid>
        # instead of bare -resume (which would collide across concurrent batches).
        # Primary path: nextflow log CLI (uses NF's own abstraction, no file parsing).
        # First fallback: read .nextflow/history directly by deterministic run name.
        # Second fallback: parse runName from log + look up in history (legacy batches).
        if not self._resume:
            session_id = _query_session_id_via_nf_cli(self.cfg.repo_dir, nf_run_name)
            if not session_id:
                session_id = _lookup_session_id_in_history(self.cfg.repo_dir, nf_run_name)
            if not session_id:
                session_id = _extract_session_id_from_nf_debug_log(nf_log_path)
            if not session_id:
                session_id = _extract_nf_session_id_from_log(log_path, self.cfg.repo_dir)
            if session_id:
                self.state.set_batch_session_id(self.batch_id, session_id)
                log.debug("Batch %s: recorded NF session ID %s", self.batch_id, session_id)

        # Auto-resume: if the first run failed but did real work, attempt one NF
        # -resume to salvage already-completed tasks (handles cause:null crashes where
        # many featurize tasks finished before NF's finalizer thread died).
        # Only on first failure (not a resume itself), only if the run lasted ≥60 s
        # (real work was attempted), and only if the work dir is still intact.
        run_duration = time.time() - run_started_at
        if exit_code != 0 and not self._resume and run_duration >= 60 and os.path.isdir(work_dir):
            if self._shutdown_event and self._shutdown_event.is_set():
                log.info(
                    "Batch %s: skipping auto-resume because dispatcher shutdown is in progress",
                    self.batch_id,
                )
                self.state.complete_batch(self.batch_id, exit_code)
                self.state.mark_slides_complete(
                    self.batch_id, False, charge_fail_count=False
                )
                log.error("Batch %s failed (exit %d). Log: %s", self.batch_id, exit_code, log_path)
                self._cleanup(csv_path, log_path, work_dir, succeeded=False)
                return exit_code
            resume_session_id = self.state.get_batch_session_id(self.batch_id)
            if not resume_session_id:
                resume_session_id = _lookup_nf_session_id(
                    self.cfg.repo_dir, self.batch_id, log_path, nf_log_path=nf_log_path
                )
            if resume_session_id:
                log.info("Batch %s: auto-resuming NF session %s to salvage completed tasks",
                         self.batch_id, resume_session_id)
            else:
                log.warning("Batch %s: auto-resuming (no session ID found — may conflict with "
                            "concurrent batches)", self.batch_id)
            for resume_attempt in (1, 2):
                try:
                    # Use a fresh run name on every auto-resume launch to avoid
                    # NF's "run name already used" rejection if an earlier
                    # attempt registered the name before failing on a session lock.
                    resume_cmd = list(cmd)
                    try:
                        _ni = resume_cmd.index("-name")
                        resume_cmd[_ni + 1] = "r" + uuid.uuid4().hex[:8]
                    except ValueError:
                        pass  # -name always present in cmd; this branch shouldn't be reached
                    resume_cmd += (["-resume", resume_session_id] if resume_session_id
                                   else ["-resume"])
                    with open(log_path, "a") as lf:
                        lf.write(f"\n\n--- AUTO-RESUME ATTEMPT {resume_attempt} ---\n\n")
                        resume_proc = subprocess.Popen(
                            resume_cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=self.cfg.repo_dir, env=run_env, start_new_session=True,
                        )
                    self.state.set_batch_nf_pid(self.batch_id, resume_proc.pid)
                    resume_started_at = time.time()
                    resume_exit = resume_proc.wait()
                    self.state.set_batch_nf_pid(self.batch_id, None)
                    if resume_exit == 0:
                        log.info("Batch %s: auto-resume succeeded — treating as success", self.batch_id)
                        exit_code = 0
                        run_started_at = resume_started_at  # for manifest collection window
                        break
                    if (
                        resume_attempt == 1
                        and _file_contains(log_path, _NF_SESSION_LOCK_ERROR)
                    ):
                        log.warning(
                            "Batch %s: auto-resume hit NF session lock; retrying once in %ds",
                            self.batch_id,
                            _AUTO_RESUME_LOCK_RETRY_DELAY_SECONDS,
                        )
                        time.sleep(_AUTO_RESUME_LOCK_RETRY_DELAY_SECONDS)
                        continue
                    log.warning("Batch %s: auto-resume also failed (exit %d) — giving up",
                                self.batch_id, resume_exit)
                    break
                except Exception as exc:
                    self.state.set_batch_nf_pid(self.batch_id, None)
                    log.error("Batch %s: auto-resume failed to launch: %s", self.batch_id, exc)
                    break

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
            self._run_post_batch_hooks(csv_path, phase="wds")
            self._cleanup_intermediate_features()
            self._verify_wds_coverage(csv_path)
            self._run_post_batch_hooks(csv_path, phase="db_sync")
            self._cleanup(csv_path, log_path, work_dir, succeeded=True)
        else:
            log.error("Batch %s failed (exit %d). Log: %s", self.batch_id, exit_code, log_path)
            self._cleanup(csv_path, log_path, work_dir, succeeded=False)

        return exit_code

    def _run_post_batch_hooks(self, batch_csv: str, phase: str = "") -> None:
        """Run any configured post-batch hooks after a successful NF run.

        If *phase* is given, only hooks whose ``_phase`` key matches are run.
        Hooks with no ``_phase`` key are treated as phase ``"wds"`` (legacy/user hooks).
        """
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
            hook_phase = hook.get("_phase", "wds")
            if phase and hook_phase != phase:
                continue
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

    def _cleanup_intermediate_features(self) -> None:
        """Remove features subdirectories that are not WDS destinations.

        After WDS push hooks run, the only remaining features dirs are those whose
        model has no wds_destination — i.e. intermediate patch encoders (e.g. conch1_5
        used by titan_slide). This is generic: no per-model mapping is needed.
        """
        if not self.cfg.cleanup_results:
            return

        features_dir = os.path.join(self.cfg.outdir, "features")
        if not os.path.isdir(features_dir):
            return

        wds_models: set[str] = set()
        for w in self.cfg.watchers:
            if hasattr(w, "wds_destinations") and w.wds_destinations:
                wds_models.update(w.wds_destinations.keys())

        if not wds_models:
            return

        import shutil as _shutil
        for entry in os.scandir(features_dir):
            if entry.is_dir() and entry.name not in wds_models:
                log.info("Removing intermediate features directory: %s", entry.path)
                _shutil.rmtree(entry.path, ignore_errors=True)

    def _verify_wds_coverage(self, batch_csv: str) -> None:
        """After append_wds hooks, verify all batch slides appear in WDS for every
        configured model. Slides missing from any model are reset to PENDING for retry.
        Slides whose fail_count+1 would reach max_slide_retries are permanently FAILed
        instead — preventing infinite retry loops for slides that produce 0 features or
        otherwise can never satisfy WDS coverage (e.g. empty/corrupted slides)."""
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
        max_retries = self.cfg.max_slide_retries or 0
        conn = self.state._conn()

        permanently_failed: list[str] = []
        retryable: list[str] = []
        if max_retries > 0:
            for sid in incomplete:
                row = conn.execute(
                    "SELECT fail_count FROM slides WHERE slide_id=?", (sid,)
                ).fetchone()
                current = row[0] if row else 0
                if current + 1 >= max_retries:
                    permanently_failed.append(sid)
                else:
                    retryable.append(sid)
        else:
            retryable = list(incomplete)

        if permanently_failed:
            conn.executemany(
                "UPDATE slides SET status='FAILED', fail_count=fail_count+1, "
                "batch_id=NULL, dispatched_at=NULL, "
                "error_msg='Permanently failed: missing from WDS after max retries' "
                "WHERE slide_id=? AND status='SUCCEEDED'",
                [(sid,) for sid in permanently_failed],
            )
            log.warning(
                "Batch %s: permanently failed %d slide(s) missing from WDS (fail_count >= %d): %s",
                self.batch_id, len(permanently_failed), max_retries,
                ", ".join(sorted(permanently_failed)),
            )

        if retryable:
            conn.executemany(
                "UPDATE slides SET status='PENDING', fail_count=fail_count+1, "
                "batch_id=NULL, dispatched_at=NULL "
                "WHERE slide_id=? AND status='SUCCEEDED'",
                [(sid,) for sid in retryable],
            )
            log.info("Batch %s: reset %d incomplete slide(s) to PENDING (will retry).",
                     self.batch_id, len(retryable))

        conn.commit()

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
                # Also remove the companion trace file written by -with-trace.
                if old_log:
                    from nextflow_turret import trace_path_for_log as _trace_path_for_log
                    old_trace = _trace_path_for_log(old_log)
                    if os.path.exists(old_trace):
                        os.unlink(old_trace)
                        log.info("Removed old trace (batch %s): %s", row["batch_id"], old_trace)

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
