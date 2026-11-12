"""Nextflow runner and manifest collection for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .batch_lifecycle import (
    BatchFinalizer,
    FinalizationContext,
    NextflowRunSpec,
    build_nextflow_command,
    collect_manifests,
    write_batch_csv,
)

# Compatibility exports retained for callers that historically imported these
# helpers from runner.py.
from .batch_lifecycle import MANIFEST_HEADER
from .config import Config
from .contract import build_contract
from .remote import (
    FINALIZATION_QUEUED_STATUS,
    RemoteDispatchClient,
    TERMINAL_REMOTE_STATUSES,
)
from .state import StateStore
from .storage import ExecutionArtifactCleaner, ExecutionCleanupSpec

try:
    import psycopg as _psycopg
    _TRANSIENT_READ_ERRORS: tuple = (_psycopg.OperationalError, _psycopg.InterfaceError,
                                     ConnectionError, TimeoutError, OSError)
except ImportError:  # sqlite-only installs
    _TRANSIENT_READ_ERRORS = (ConnectionError, TimeoutError, OSError)

# Nextflow run name must match this pattern (validated before launch to catch
# regressions in name generation before NF rejects them at invocation time).
_NF_RUN_NAME_RE = re.compile(r'^[a-z](?:[a-z\d]|[-_](?=[a-z\d])){0,79}$')
_NF_SESSION_LOCK_ERROR = "Unable to acquire lock on session"
_AUTO_RESUME_LOCK_RETRY_DELAY_SECONDS = 30
_NF_PID_WRITE_ATTEMPTS = 3
_NF_PID_WRITE_RETRY_SECONDS = 1.0
# A stop that SIGTERMs the whole service (systemd KillMode=control-group) can
# end Nextflow before the dispatcher's own signal handler runs; wait this long
# on a non-zero exit for shutdown to show up before booking a failure.
_SHUTDOWN_EXIT_GRACE_SECONDS = 5.0

log = logging.getLogger("mussel-dispatcher")
# Optional site defaults come from the environment: MUSSEL_JAVA_HOME pins the
# Java runtime, MUSSEL_NEXTFLOW_BIN names a wrapper script and
# MUSSEL_NEXTFLOW_REAL_BIN the Nextflow launcher (default: nextflow on PATH).
_DEFAULT_JAVA_HOME = ""
_DEFAULT_NEXTFLOW_BIN = "nextflow"
_DEFAULT_NEXTFLOW_WRAPPER = ""


def _file_contains(path: str, needle: str) -> bool:
    try:
        with open(path, errors="ignore") as f:
            return needle in f.read()
    except OSError:
        return False


def _resolve_nextflow_java_env(base_env: dict[str, str]) -> dict[str, str]:
    """Pin Nextflow to a known-good Java runtime when available."""
    run_env = dict(base_env)
    java_home = (
        run_env.get("NXF_JAVA_HOME")
        or run_env.get("JAVA_HOME")
        or os.environ.get("MUSSEL_JAVA_HOME")
        or _DEFAULT_JAVA_HOME
    )
    java_bin = os.path.join(java_home, "bin", "java") if java_home else ""
    if java_home and (
        (os.path.isdir(java_home) and os.path.isfile(java_bin) and os.access(java_bin, os.X_OK))
        or (_DEFAULT_JAVA_HOME and java_home == _DEFAULT_JAVA_HOME)
    ):
        run_env["NXF_JAVA_HOME"] = java_home
        run_env["JAVA_HOME"] = java_home
        run_env["JAVA_CMD"] = java_bin
        path_parts = (run_env.get("PATH") or "").split(os.pathsep) if run_env.get("PATH") else []
        if os.path.join(java_home, "bin") not in path_parts:
            run_env["PATH"] = os.pathsep.join([os.path.join(java_home, "bin"), *path_parts]) if path_parts else os.path.join(java_home, "bin")
    return run_env


def _resolve_nextflow_cmd() -> list[str]:
    wrapper = os.environ.get("MUSSEL_NEXTFLOW_BIN", "").strip() or _DEFAULT_NEXTFLOW_WRAPPER
    if wrapper and os.path.isfile(wrapper) and os.access(wrapper, os.X_OK):
        return ["/usr/bin/bash", wrapper]
    return [os.environ.get("MUSSEL_NEXTFLOW_REAL_BIN", "").strip() or _DEFAULT_NEXTFLOW_BIN]


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
            [*_resolve_nextflow_cmd(), "log", run_name, "-f", "session_id"],
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
                 existing_work_dir: str = None, shutdown_event: threading.Event | None = None,
                 finalization_submitter=None):
        self.cfg = cfg
        self.batch_id = batch_id
        self.slides = slides  # list of {"slide_path": ..., "slide_id": ..., "oncotree_code": ...}
        self.state = state
        self._resume = resume
        self._existing_csv_path = existing_csv_path
        self._existing_work_dir = existing_work_dir
        self._shutdown_event = shutdown_event
        # Whether this batch's work may be running right now: a local Nextflow
        # process between Popen and its exit, or a remote order the agent owns. A
        # failed runner's slides are requeued only when this is False (nothing can
        # still be computing them); a remote batch is re-attached instead.
        self.work_may_be_running = False
        # Nextflow exited 0: the outputs exist. A failure after this point must
        # finish the batch (FINALIZING), never requeue it for a full recompute.
        self.driver_succeeded = False
        # RunManager supplies this for live runs so the execution worker can
        # return as soon as Nextflow exits. Direct callers retain synchronous
        # finalization for backwards compatibility and recovery tests.
        self._finalization_submitter = finalization_submitter
        self._lifecycle = BatchFinalizer(
            cfg, batch_id, state, target="local",
            runtime_revision=str(getattr(cfg, "source_revision", "") or ""),
        )
        self._execution_cleaner = ExecutionArtifactCleaner()

    def _cleanup_execution_artifacts(self, work_dir: str, *, record_state: bool = True) -> None:
        if not self.cfg.cleanup_work_dir or not work_dir:
            return
        result = self._execution_cleaner.cleanup(
            ExecutionCleanupSpec(
                batch_id=self.batch_id,
                managed_root=self.cfg.work_base_dir,
                run_root=work_dir,
                execution_paths=(work_dir,),
                protected_paths=(self.cfg.outdir,),
            )
        )
        if record_state:
            self.state.record_execution_cleanup(
                self.batch_id,
                error=result.error,
                retryable=result.retryable,
            )

    def _hook_state_rows(self) -> list[dict]:
        return self._lifecycle.hooks.state_rows()

    def _register_new_batch(
        self,
        csv_path: str,
        work_dir: str,
        log_path: str,
        *,
        nf_run_name: str,
        backend: str = "local",
        target: str | None = None,
        contract: dict | None = None,
        params_path: str | None = None,
    ) -> bool:
        """Create and claim a batch identically for all execution backends."""
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.state.add_batch(
            self.batch_id,
            csv_path,
            work_dir,
            len(self.slides),
            log_path,
            run_name=nf_run_name,
            backend=backend,
            target=target,
            source_revision=(contract or {}).get("source_revision"),
            release_manifest_sha256=(contract or {}).get("release_manifest_sha256"),
            semantic_contract_sha256=(contract or {}).get("semantic_contract_sha256"),
            execution_profile=(contract or {}).get("execution_profile", {}).get("name"),
            execution_profile_sha256=(contract or {}).get("execution_profile_sha256"),
            params_path=params_path,
            effective_params_sha256=(contract or {}).get("effective_params_sha256"),
        )
        self._lifecycle.hooks.runtime_revision = str(
            (contract or {}).get("source_revision") or ""
        )
        self.state.sync_batch_hooks(self.batch_id, self._hook_state_rows())
        from .batch_lifecycle import select_batch_csv_slides

        def _key(slide: dict) -> str:
            return str(slide.get("_db_key") or slide.get("slide_path") or "")

        # Only slides that reach the batch CSV may be claimed: a slide the CSV
        # drops would otherwise sit DISPATCHED and be marked SUCCEEDED with
        # the batch despite producing no output.
        selected, duplicates, invalid = select_batch_csv_slides(
            self.slides, batch_id=self.batch_id
        )
        selected_keys = {_key(slide) for slide in selected}
        for dropped, terminal_status in ((duplicates, "SKIPPED"), (invalid, "FAILED")):
            reasons = {
                _key(slide): f"not dispatched in batch {self.batch_id}: {reason}"
                for slide, reason in dropped
                if _key(slide) and _key(slide) not in selected_keys
            }
            if reasons:
                n_marked = self.state.reject_pending_slides(
                    reasons, terminal_status=terminal_status
                )
                log.warning(
                    "Batch %s: %d slide(s) dropped from the batch CSV were marked %s: %s",
                    self.batch_id, n_marked, terminal_status,
                    "; ".join(f"{k} ({v})" for k, v in reasons.items()),
                )
        claimed = set(self.state.claim_slides([_key(slide) for slide in selected], self.batch_id))
        claimed_slides = [slide for slide in selected if _key(slide) in claimed]
        if not claimed_slides:
            log.warning(
                "Batch %s: 0 of %d slides claimed; aborting",
                self.batch_id, len(self.slides),
            )
            self.state.cancel_batch(self.batch_id)
            try:
                os.unlink(csv_path)
            except OSError:
                pass
            return False
        if len(claimed_slides) < len(self.slides):
            log.warning(
                "Batch %s: running %d of %d slides; the rest were already dispatched "
                "elsewhere or dropped from the CSV",
                self.batch_id, len(claimed_slides), len(self.slides),
            )
            # Restrict everything downstream (CSV, remote order, has_download)
            # to the slides this batch actually owns.
            self.slides = claimed_slides
            write_batch_csv(csv_path, self.slides, batch_id=self.batch_id)
            self.state.set_batch_slide_count(self.batch_id, len(self.slides))
        return True

    def run(self):
        if self.cfg.execution_backend == "remote_agent":
            return self._run_remote_agent()

        # Use the 8-char UUID hash as the NF run name so it fits within
        # Seqera Platform's 16-char workflow.id field limit. Prefix with "r"
        # to satisfy Nextflow's requirement that names begin with a letter.
        nf_run_name = (
            "r" + uuid.uuid4().hex[:8]
            if self._resume
            else "r" + self.batch_id.rsplit("_", 1)[-1]
        )
        if not _NF_RUN_NAME_RE.match(nf_run_name):
            raise ValueError(
                f"Generated NF run name {nf_run_name!r} does not match Nextflow's "
                f"required pattern {_NF_RUN_NAME_RE.pattern!r}. "
                f"This is a bug in run name generation (batch_id={self.batch_id!r})."
            )
        if self._resume:
            csv_path = self._existing_csv_path
            work_dir = self._existing_work_dir
            log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
            self.state.restart_batch(self.batch_id)
            self.state.set_batch_run_name(self.batch_id, nf_run_name)
            self.state.sync_batch_hooks(self.batch_id, self._hook_state_rows())
            params_path = os.path.join(os.path.dirname(work_dir), "params.yaml")
            if not os.path.isfile(params_path):
                params_path = self.cfg.nextflow_params_file
            contract = None
        else:
            csv_path = self._write_csv()
            batch_root = os.path.join(self.cfg.work_base_dir, f"batch_{self.batch_id}")
            work_dir = os.path.join(batch_root, "work")
            params_path = ""
            contract = None
            if self.cfg.nextflow_params_file:
                params_path = os.path.join(batch_root, "params.yaml")
                contract = build_contract(self.cfg)
                Path(params_path).parent.mkdir(parents=True, exist_ok=True)
                Path(params_path).write_text(contract["effective_params_yaml"], encoding="utf-8")
            log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
            if not self._register_new_batch(
                csv_path,
                work_dir,
                log_path,
                nf_run_name=nf_run_name,
                contract=contract,
                params_path=params_path,
            ):
                return

        trace_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.trace.tsv")
        nf_log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.nf.log")
        cmd = build_nextflow_command(NextflowRunSpec(
            executable=_resolve_nextflow_cmd(),
            log_path=nf_log_path,
            repo_dir=self.cfg.repo_dir,
            profiles=self.cfg.nextflow_profiles,
            work_dir=work_dir,
            batch_csv=csv_path,
            outdir=self.cfg.outdir,
            run_name=nf_run_name,
            trace_path=trace_path,
            config_path=self.cfg.nextflow_config,
            params_path=params_path,
            tower_enabled=bool(self.cfg.tower_endpoint),
        ))
        # If this batch includes slides that need GDC download, pass the token
        # file to NF if configured (open-access TCGA data needs no token).
        has_download = any(s.get("needs_download") for s in self.slides)
        if has_download:
            for w in self.cfg.watchers:
                if getattr(w, "gdc_token_file", ""):
                    cmd += ["--download.gdc_token_file", w.gdc_token_file]
                    break
        for warning in self.cfg.publish_mode_safety_warnings():
            log.warning("Batch %s: %s", self.batch_id, warning)
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
        run_env = _resolve_nextflow_java_env(os.environ)
        if self.cfg.nextflow_version:
            run_env["NXF_VER"] = self.cfg.nextflow_version
        if self.cfg.tower_endpoint:
            run_env["TOWER_API_ENDPOINT"] = self.cfg.tower_endpoint
            run_env["TOWER_ACCESS_TOKEN"] = run_env.get("TOWER_ACCESS_TOKEN") or "local"
        log.info(
            "Batch %s Java env: NXF_JAVA_HOME=%s JAVA_HOME=%s JAVA_CMD=%s",
            self.batch_id,
            run_env.get("NXF_JAVA_HOME", ""),
            run_env.get("JAVA_HOME", ""),
            run_env.get("JAVA_CMD", ""),
        )
        proc = None
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
        except Exception as e:
            log.error("Batch %s failed to launch: %s", self.batch_id, e)
        if proc is not None:
            # Once Nextflow is running it must be waited on, whatever the
            # bookkeeping does: giving up here would requeue slides it is still
            # processing. Store PID so recover_in_flight() can kill this process
            # if the dispatcher crashes and restarts before this batch completes.
            self.work_may_be_running = True
            self._record_nf_pid(proc.pid)
            log.debug("Batch %s: NF process PID=%d", self.batch_id, proc.pid)
            exit_code = proc.wait()
            self.work_may_be_running = False

        # Clear the stored PID — this process is done.
        self._record_nf_pid(None)

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

        # A non-zero exit during shutdown belongs to restart recovery, whether
        # or not stop_for_restart_recovery() has published RECOVERING yet: the
        # service stop SIGTERMs every Nextflow at once while shutdown marks the
        # batches one by one.  Booking it here charged fail_count, deleted the
        # work dir and turned a preserved batch FAILED.
        if exit_code != 0 and self._shutdown_stopped_run():
            return exit_code

        # Auto-resume: if the first run failed but did real work, attempt one NF
        # -resume to salvage already-completed tasks (handles cause:null crashes where
        # many featurize tasks finished before NF's finalizer thread died).
        # Only on first failure (not a resume itself), only if the run lasted ≥60 s
        # (real work was attempted), and only if the work dir is still intact.
        run_duration = time.time() - run_started_at
        if exit_code != 0 and not self._resume and run_duration >= 60 and os.path.isdir(work_dir):
            resume_session_id = self.state.get_batch_session_id(self.batch_id)
            if not resume_session_id:
                resume_session_id = _lookup_nf_session_id(
                    self.cfg.repo_dir, self.batch_id, log_path, nf_log_path=nf_log_path
                )
            # Use a fresh run name to avoid NF's "run name already used" rejection
            # (NF ≥ 23.x rejects reusing a name in .nextflow/history, even with -resume).
            resume_nf_run_name = "r" + uuid.uuid4().hex[:8]
            self.state.set_batch_run_name(self.batch_id, resume_nf_run_name)
            resume_cmd = list(cmd)
            try:
                _ni = resume_cmd.index("-name")
                resume_cmd[_ni + 1] = resume_nf_run_name
            except ValueError:
                pass  # -name always present in cmd; this branch shouldn't be reached
            resume_cmd += (["-resume", resume_session_id] if resume_session_id
                           else ["-resume"])
            if resume_session_id:
                log.info("Batch %s: auto-resuming NF session %s to salvage completed tasks",
                         self.batch_id, resume_session_id)
            else:
                log.warning("Batch %s: auto-resuming (no session ID found — may conflict with "
                            "concurrent batches)", self.batch_id)
            for resume_attempt in (1, 2):
                try:
                    with open(log_path, "a") as lf:
                        lf.write(f"\n\n--- AUTO-RESUME ATTEMPT {resume_attempt} ---\n\n")
                        resume_proc = subprocess.Popen(
                            resume_cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=self.cfg.repo_dir, env=run_env, start_new_session=True,
                        )
                except Exception as exc:
                    self._record_nf_pid(None)
                    log.error("Batch %s: auto-resume failed to launch: %s", self.batch_id, exc)
                    break
                self.work_may_be_running = True
                self._record_nf_pid(resume_proc.pid)
                resume_started_at = time.time()
                resume_exit = resume_proc.wait()
                self.work_may_be_running = False
                self._record_nf_pid(None)
                try:
                    # stop_for_restart_recovery() may have marked the batch
                    # RECOVERING/FINALIZING and SIGTERMed this resume run;
                    # its non-zero exit must not be booked as a failure.
                    if resume_exit != 0 and self._shutdown_stopped_run():
                        return resume_exit
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
                        if self._shutdown_event is not None:
                            self._shutdown_event.wait(_AUTO_RESUME_LOCK_RETRY_DELAY_SECONDS)
                        else:
                            time.sleep(_AUTO_RESUME_LOCK_RETRY_DELAY_SECONDS)
                        if self._shutdown_stopped_run(grace=0):
                            # Never launch another Nextflow run during shutdown.
                            return resume_exit
                        continue
                    log.warning("Batch %s: auto-resume also failed (exit %d) — giving up",
                                self.batch_id, resume_exit)
                    break
                except Exception as exc:
                    log.error("Batch %s: auto-resume bookkeeping failed: %s", self.batch_id, exc)
                    break

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
        if exit_code == 0:
            self.driver_succeeded = True
            log.info("Batch %s completed successfully.", self.batch_id)
            self.state.mark_driver_terminal(self.batch_id, exit_code)
            self._cleanup_execution_artifacts(work_dir)
            if self._finalization_submitter is not None:
                try:
                    queued = self._finalization_submitter(
                        self.batch_id,
                        csv_path,
                        log_path,
                        work_dir,
                        run_started_at,
                    )
                    if queued is False:
                        log.warning(
                            "Batch %s post-success finalization was not queued; "
                            "leaving it for retry",
                            self.batch_id,
                        )
                except Exception:
                    # The Nextflow run succeeded. Do not turn a transient
                    # finalizer submission/database error into a batch failure;
                    # the FINALIZING row is retried by the scheduler.
                    log.exception(
                        "Batch %s could not queue post-success finalization",
                        self.batch_id,
                    )
            else:
                self._finalize_success(csv_path, log_path, work_dir, run_started_at)
        else:
            # Conditional: shutdown may have preserved the batch since the
            # check above; it then owns the slides and the work dir.
            if not self.state.complete_batch(self.batch_id, exit_code, unless_preserved=True):
                log.info(
                    "Batch %s: preserved for restart recovery; not booking exit %d",
                    self.batch_id, exit_code,
                )
                return exit_code
            self.state.skip_unstarted_batch_hooks(
                self.batch_id,
                reason=f"driver ended before hooks: exit {exit_code}",
            )
            self.state.mark_slides_complete(
                self.batch_id, False, charge_fail_count=not fast_fail,
                unless_preserved=True,
            )
            log.error("Batch %s failed (exit %d). Log: %s", self.batch_id, exit_code, log_path)
            self._cleanup(csv_path, log_path, work_dir, succeeded=False)

        return exit_code

    def _record_nf_pid(self, pid: int | None) -> None:
        """Record (or clear) the batch's Nextflow PID, retrying a database blip.

        Never raises: the caller is waiting on, or done with, a live process.
        A stale nf_pid is left to restart recovery, which checks the PID is alive.
        """
        for attempt in range(1, _NF_PID_WRITE_ATTEMPTS + 1):
            try:
                self.state.set_batch_nf_pid(self.batch_id, pid)
                return
            except Exception as exc:
                if attempt == _NF_PID_WRITE_ATTEMPTS:
                    log.error(
                        "Batch %s: could not write nf_pid=%s after %d attempts: %s",
                        self.batch_id, pid, attempt, exc,
                    )
                    return
                log.warning(
                    "Batch %s: NF PID write failed (attempt %d/%d): %s",
                    self.batch_id, attempt, _NF_PID_WRITE_ATTEMPTS, exc,
                )
                time.sleep(_NF_PID_WRITE_RETRY_SECONDS * attempt)

    def _shutdown_stopped_run(self, grace: float | None = None) -> bool:
        """Return True when dispatcher shutdown ended this (non-zero) run.

        The batch, its slides and its work dir are then left to
        stop_for_restart_recovery() and the next startup, whatever status the
        row shows right now: shutdown may not have reached this batch yet.
        """
        if self._shutdown_event is None:
            return False
        if grace is None:
            grace = _SHUTDOWN_EXIT_GRACE_SECONDS
        if not self._shutdown_event.wait(grace):
            return False
        log.info(
            "Batch %s: dispatcher shutdown in progress; leaving batch for restart recovery",
            self.batch_id,
        )
        return True

    def finalize_recovered_success(
        self,
        csv_path: str,
        log_path: str,
        work_dir: str,
        *,
        run_started_at: float = 0.0,
        nf_log_path: str | None = None,
    ):
        """Replay post-success bookkeeping for a batch whose NF run already finished.

        Used during dispatcher startup recovery when the underlying Nextflow run
        completed successfully but the dispatcher died before it could run
        manifest collection, post-batch hooks, WDS verification, DB sync, and
        cleanup.
        """
        self.state.set_batch_nf_pid(self.batch_id, None)

        session_id = self.state.get_batch_session_id(self.batch_id)
        if not session_id:
            session_id = _lookup_nf_session_id(
                self.cfg.repo_dir, self.batch_id, log_path, nf_log_path=nf_log_path
            )
            if session_id:
                self.state.set_batch_session_id(self.batch_id, session_id)

        log.info(
            "Batch %s: replaying post-success finalization after dispatcher restart.",
            self.batch_id,
        )
        self.state.mark_driver_terminal(self.batch_id, 0)
        self._cleanup_execution_artifacts(work_dir)
        return self._finalize_success(csv_path, log_path, work_dir, run_started_at)

    def _finalize_success(
        self, csv_path: str, log_path: str, work_dir: str, run_started_at: float
    ) -> None:
        # Drivers may run concurrently. Finalization uses bounded backend
        # slots; WDS destination locks and the shared DB-sync lease protect
        # mutations. If no local slot is available, leave the batch in
        # FINALIZING for the scheduler's durable retry path.
        self.state.set_batch_status(self.batch_id, "FINALIZING")
        owner = f"local-finalizer:{os.getpid()}:{self.batch_id}"
        lease_ttl = max(1, int(getattr(self.cfg, "finalization_lease_ttl_seconds", 300)))
        lease_name = self.state.claim_finalization_slot(
            self.batch_id,
            lane="local",
            max_slots=getattr(self.cfg, "max_concurrent_finalizations", 1),
            owner=owner,
            ttl_seconds=lease_ttl,
        )
        if not lease_name:
            log.info(
                "Batch %s: local finalization slots are busy; queued for retry",
                self.batch_id,
            )
            return None

        stop_heartbeat = threading.Event()

        def heartbeat() -> None:
            interval = max(
                5,
                int(getattr(self.cfg, "finalization_heartbeat_interval_seconds", 60)),
            )
            while not stop_heartbeat.wait(interval):
                try:
                    self.state.heartbeat_finalization_lease(
                        self.batch_id,
                        owner=owner,
                        ttl_seconds=lease_ttl,
                        lease_name=lease_name,
                    )
                except Exception:
                    log.debug(
                        "Batch %s: finalization lease heartbeat failed",
                        self.batch_id,
                        exc_info=True,
                    )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"finalization-heartbeat-{self.batch_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            return self._lifecycle.finalize_success(
                FinalizationContext(
                    batch_csv=csv_path,
                    outdir=self.cfg.outdir,
                    repo_dir=self.cfg.repo_dir,
                    run_started_at=run_started_at,
                ),
                run_hooks=self._run_post_batch_hooks,
                collect_manifest=self._collect_manifest,
                cleanup_intermediate=self._cleanup_intermediate_features,
                verify_wds=self._verify_wds_coverage,
                cleanup=lambda: self._cleanup(
                    csv_path, log_path, work_dir, succeeded=True, execution=False
                ),
            )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=5)
            self.state.release_finalization_lease(
                self.batch_id,
                owner=owner,
                lease_name=lease_name,
            )

    def _run_post_batch_hooks(
        self,
        batch_csv: str,
        phase: str = "",
        *,
        outdir: str | None = None,
        repo_dir: str | None = None,
        path_replacements: dict[str, str] | None = None,
        command_runner=None,
    ) -> bool:
        self._lifecycle.cfg = self.cfg
        hooks = self._lifecycle.hooks
        kwargs = {
            "phase": phase,
            "outdir": outdir,
            "repo_dir": repo_dir,
            "path_replacements": path_replacements,
        }
        if command_runner is not None:
            kwargs["executor"] = command_runner
        return hooks.run(batch_csv, **kwargs)

    def _cleanup_intermediate_features(self) -> None:
        """Remove features subdirectories that are not WDS destinations.

        After WDS push hooks run, the only remaining features dirs are those whose
        model has no wds_destination — i.e. intermediate patch encoders (e.g. conch1_5
        used by titan_slide). This is generic: no per-model mapping is needed.
        """
        self._lifecycle.cfg = self.cfg
        self._lifecycle._cleanup_intermediate_features(self.cfg.outdir)

    def _verify_wds_coverage(self, batch_csv: str) -> None:
        """After append_wds hooks, verify all batch slides appear in WDS for every
        configured model. Slides missing from any model are reset to PENDING for retry.
        Slides whose fail_count+1 would reach max_slide_retries are permanently FAILed
        instead — preventing infinite retry loops for slides that produce 0 features or
        otherwise can never satisfy WDS coverage (e.g. empty/corrupted slides)."""
        self._lifecycle.cfg = self.cfg
        return self._lifecycle._verify_wds_coverage(batch_csv, self.cfg.outdir)

    def _cleanup(
        self,
        csv_path: str,
        log_path: str,
        work_dir: str,
        *,
        succeeded: bool = True,
        execution: bool = True,
        record_state: bool = True,
    ):
        """Post-batch cleanup. Work dir is removed on both success and failure when enabled.
        Downloads, batch CSV, and log rotation only run on success."""
        if execution:
            self._cleanup_execution_artifacts(work_dir, record_state=record_state)

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
        self._lifecycle.cfg = self.cfg
        return self._lifecycle._collect_manifest(self.cfg.outdir, run_started_at)

    def _write_csv(self) -> str:
        os.makedirs(self.cfg.dispatch_dir, exist_ok=True)
        csv_path = os.path.join(self.cfg.dispatch_dir, f"batch_{self.batch_id}.csv")
        write_batch_csv(csv_path, self.slides, batch_id=self.batch_id)
        log.debug("Wrote batch CSV: %s", csv_path)
        return csv_path

    def _run_remote_agent(self):
        if self._resume:
            log.warning("Batch %s: remote resume is status-driven; polling existing remote status", self.batch_id)
            csv_path = self._existing_csv_path
            work_dir = self._existing_work_dir
            log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
        else:
            target = self.cfg.get_dispatch_target()
            if not target:
                raise RuntimeError("execution_backend=remote_agent requires a dispatch target")
            csv_path = self._write_csv()
            work_dir = os.path.join(self.cfg.work_base_dir, f"remote_batch_{self.batch_id}")
            contract = build_contract(self.cfg, target=target)
            params_path = os.path.join(work_dir, "params.yaml")
            log_path = os.path.join(self.cfg.log_dir, f"batch_{self.batch_id}.log")
            nf_run_name = "r" + self.batch_id.rsplit("_", 1)[-1]
            if not _NF_RUN_NAME_RE.match(nf_run_name):
                raise ValueError(f"Generated remote NF run name {nf_run_name!r} is invalid")
            if not self._register_new_batch(
                csv_path,
                work_dir,
                log_path,
                nf_run_name=nf_run_name,
                backend="remote_agent",
                target=target.name,
                contract=contract,
                params_path=params_path,
            ):
                return
            client = RemoteDispatchClient(self.cfg, target)
            order = client.submit(
                batch_id=self.batch_id,
                slides=self.slides,
                csv_path=csv_path,
                log_path=log_path,
                nf_run_name=nf_run_name,
                contract=contract,
            )
            # From here the agent owns the order, even if writing the log below fails.
            self.work_may_be_running = True
            with open(log_path, "w") as lf:
                lf.write(f"Remote batch {self.batch_id} queued for target {target.name}\n")
                lf.write(f"Work order: {order.get('remote_run_dir')}\n")

        # The order exists (just submitted, or this is a status-driven resume):
        # the remote agent owns the batch from here on.
        self.work_may_be_running = True
        target = self.cfg.get_dispatch_target()
        if not target:
            raise RuntimeError("remote batch has no configured dispatch target")
        client = RemoteDispatchClient(self.cfg, target)
        poll_interval = max(1, target.poll_interval_seconds)
        exit_code = -1
        read_failures = 0
        while True:
            if self._shutdown_event and self._shutdown_event.is_set():
                log.info(
                    "Batch %s: dispatcher shutdown while remote batch is active; "
                    "leaving batch state untouched for recovery",
                    self.batch_id,
                )
                return 0
            try:
                status = client.read_status(self.batch_id)
                read_failures = 0
            except _TRANSIENT_READ_ERRORS as exc:
                # A DB or network blip must not end this waiter. Anything else (a bug,
                # a schema mismatch) raises; the scheduler re-attaches a fresh waiter.
                read_failures += 1
                if read_failures == 1 or read_failures % 30 == 0:
                    log.warning("Batch %s: reading remote status failed (%d in a row): %s",
                                self.batch_id, read_failures, exc)
                time.sleep(poll_interval)
                continue
            if status:
                remote_status = status.get("status") or "UNKNOWN"
                if remote_status in {
                    FINALIZATION_QUEUED_STATUS,
                    "HOOKS_RUNNING",
                    "HOOKS_RETRY",
                    # Quarantined until an operator acts; the agent no longer
                    # refreshes its heartbeat, so holding the slot would block
                    # this pool until the stale timeout (or forever).
                    "HOOKS_BLOCKED",
                }:
                    # The remote agent owns post-success finalization from this
                    # point onward. Release this dispatcher execution slot so
                    # another batch can start while hooks drain or retry.
                    log.info(
                        "Batch %s: remote agent owns finalization (%s); releasing dispatcher slot",
                        self.batch_id,
                        remote_status,
                    )
                    self._cleanup(
                        csv_path, log_path, work_dir, succeeded=True, record_state=False
                    )
                    return 0
                if remote_status in TERMINAL_REMOTE_STATUSES:
                    raw_exit_code = status.get("nextflow_exit")
                    exit_code = int(raw_exit_code) if raw_exit_code is not None else (
                        0 if remote_status == "SUCCEEDED" else -1
                    )
                    if remote_status != "SUCCEEDED" and exit_code == 0:
                        # Only SUCCEEDED may clean up as a success, whatever exit
                        # code the agent published (sacct reports 0:0 for jobs Slurm ended).
                        exit_code = -1
                    break
                heartbeat = status.get("heartbeat_at")
                # QUEUED is a normal pre-claim state.  Its heartbeat is the
                # enqueue timestamp, not an agent liveness signal, so applying
                # the active-run timeout here turns a merely backlogged order
                # into a false STALE batch.  Once the agent claims/starts the
                # order, subsequent heartbeats are authoritative for stale
                # detection.
                if remote_status != "QUEUED" and heartbeat and target.stale_timeout_seconds > 0:
                    try:
                        hb_ts = datetime.fromisoformat(heartbeat).timestamp()
                        if time.time() - hb_ts > target.stale_timeout_seconds:
                            # The dispatcher is only a status consumer. Do not
                            # overwrite the agent's DB row here: an agent may be
                            # between polls or recovering on another host.
                            # Leave the row for agent/restart recovery and stop
                            # this local waiter without failing the batch.
                            log.warning(
                                "Batch %s: remote heartbeat stale for target %s; "
                                "leaving DB state unchanged for remote recovery",
                                self.batch_id, target.name,
                            )
                            exit_code = 0
                            break
                    except Exception:
                        pass
            time.sleep(poll_interval)

        # The PostgreSQL-backed remote agent owns execution finalization and
        # slide state. The dispatcher-side worker only waits and cleans its
        # local coordination directory after observing a terminal state.
        self._cleanup(
            csv_path, log_path, work_dir,
            succeeded=(exit_code == 0),
            record_state=False,
        )
        return exit_code


# ---------------------------------------------------------------------------
# BatchScheduler
# ---------------------------------------------------------------------------
