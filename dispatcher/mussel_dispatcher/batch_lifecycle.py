"""Backend-neutral batch construction, hooks, and success finalization."""
from __future__ import annotations

import csv
import glob
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .config import Config
from .hook_hardening import (
    HookFailureCategory,
    classify_hook_failure,
    preflight_hook,
)
from .hook_launcher import launch_command
from .state import DB_SYNC_LEASE_NAME, StateStore

log = logging.getLogger("mussel-dispatcher")
MANIFEST_HEADER = ["slide_id", "workflow_id", "key", "value"]
# Manifest collection builds a large in-memory deduplication map.  Finalizers
# run in parallel, but sharing this merge across them prevents concurrent
# copies from exhausting the remote agent's memory before hooks start.
_MANIFEST_COLLECTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class NextflowRunSpec:
    executable: list[str]
    log_path: str
    repo_dir: str
    profiles: str
    work_dir: str
    batch_csv: str
    outdir: str
    run_name: str
    trace_path: str
    config_path: str = ""
    params_path: str = ""
    tower_enabled: bool = False
    scratch_dir: str = ""
    extra_args: list[str] = field(default_factory=list)


def build_nextflow_command(spec: NextflowRunSpec) -> list[str]:
    """Build the common Nextflow invocation used by every execution backend."""
    cmd = [
        *spec.executable,
        "-log", spec.log_path,
        "run", spec.repo_dir,
        "-profile", spec.profiles,
        "-work-dir", spec.work_dir,
        "--samples_csv", spec.batch_csv,
        "--outdir", spec.outdir,
        "-name", spec.run_name,
        "-with-trace", spec.trace_path,
    ]
    if spec.scratch_dir:
        cmd += ["--scratch_dir", spec.scratch_dir]
    if spec.tower_enabled:
        cmd += ["-with-tower"]
    if spec.config_path:
        cmd += ["-c", spec.config_path]
    if spec.params_path:
        cmd += ["-params-file", spec.params_path]
    return cmd + list(spec.extra_args)


def _deduplicate_slides(slides: list[dict], *, batch_id: str = "") -> list[dict]:
    seen: dict[str, dict] = {}
    for slide in slides:
        slide_id = str(slide.get("slide_id") or "")
        previous = seen.get(slide_id)
        if previous is None:
            seen[slide_id] = slide
            continue
        previous_path = str(previous.get("slide_path") or "")
        new_path = str(slide.get("slide_path") or "")
        if previous_path.startswith("gdc://") and not new_path.startswith("gdc://"):
            seen[slide_id] = slide
            kept, dropped = new_path, previous_path
        else:
            kept, dropped = previous_path, new_path
        log.warning(
            "Batch %s: duplicate slide_id %s - keeping %s, dropping %s",
            batch_id, slide_id, kept, dropped,
        )
    if len(seen) != len(slides):
        log.warning(
            "Batch %s: removed %d duplicate slide_id(s) before writing CSV",
            batch_id, len(slides) - len(seen),
        )
    return list(seen.values())


def _download_file_identity(slide: dict) -> tuple[str, str]:
    """Return (file_id, file_name), deriving both from a gdc:// path if needed."""
    slide_path = str(slide.get("slide_path") or "")
    file_id = str(slide.get("file_id") or "")
    file_name = str(slide.get("file_name") or "")
    if not file_id and slide_path.startswith("gdc://"):
        rest = slide_path[len("gdc://"):]
        file_id, separator, file_name_from_path = rest.partition("/")
        if separator and not file_name:
            file_name = file_name_from_path
    return file_id, file_name


def select_batch_csv_slides(
    slides: list[dict], *, batch_id: str = ""
) -> tuple[list[dict], list[tuple[dict, str]], list[tuple[dict, str]]]:
    """Split ``slides`` into the rows ``write_batch_csv`` writes and those it drops.

    Returns ``(selected, duplicates, invalid)``; dropped rows carry a reason
    so callers can avoid claiming (and later marking SUCCEEDED) slides that
    never reach the pipeline.  ``duplicates`` are other paths for a slide_id
    that is in the CSV (the slide itself will be processed); ``invalid`` rows
    cannot be processed at all.  A dropped duplicate with the very same
    slide_path as the kept row is not reported, since that slide is in the CSV.
    """
    deduped = _deduplicate_slides(slides, batch_id=batch_id)
    kept_by_id = {str(slide.get("slide_id") or ""): slide for slide in deduped}
    duplicates: list[tuple[dict, str]] = []
    invalid: list[tuple[dict, str]] = []
    for slide in slides:
        kept_slide = kept_by_id.get(str(slide.get("slide_id") or ""))
        if kept_slide is None or kept_slide is slide:
            continue
        kept_path = str(kept_slide.get("slide_path") or "")
        if str(slide.get("slide_path") or "") == kept_path:
            continue
        duplicates.append((
            slide,
            f"duplicate slide_id {slide.get('slide_id')} in batch {batch_id}; "
            f"processed via {kept_path}",
        ))
    selected: list[dict] = []
    for slide in deduped:
        if not str(slide.get("slide_path") or ""):
            log.warning("Skipping slide %s with empty slide_path", slide.get("slide_id"))
            invalid.append((slide, "empty slide_path"))
            continue
        if slide.get("needs_download") and not _download_file_identity(slide)[0]:
            log.warning(
                "Batch %s: slide %s needs_download but has no file_id - skipping",
                batch_id, slide.get("slide_id"),
            )
            invalid.append((slide, "needs_download but has no file_id"))
            continue
        selected.append(slide)
    return selected, duplicates, invalid


def write_batch_csv(path: str | Path, slides: list[dict], *, batch_id: str = "") -> str:
    """Serialize a batch identically for local execution and remote transfer."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected, _duplicates, _invalid = select_batch_csv_slides(slides, batch_id=batch_id)
    has_download = any(slide.get("needs_download") for slide in selected)
    fieldnames = ["slide_id", "slide_path", "oncotree_code"]
    if has_download:
        fieldnames += ["needs_download", "file_id", "file_name"]

    with destination.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for slide in selected:
            row = {
                "slide_id": slide["slide_id"],
                "slide_path": str(slide.get("slide_path") or ""),
                "oncotree_code": slide.get("oncotree_code", ""),
            }
            if has_download:
                file_id, file_name = _download_file_identity(slide)
                row.update({
                    "needs_download": "true" if slide.get("needs_download") else "false",
                    "file_id": file_id,
                    "file_name": file_name,
                })
            writer.writerow(row)
    return str(destination)


def collect_manifests(outdir: str, combined_path: str) -> int:
    """Merge per-run manifests using last-write-wins by file modification time."""
    manifest_files = sorted(
        glob.glob(os.path.join(outdir, "manifest-*.csv")), key=os.path.getmtime
    )
    if not manifest_files:
        log.info("collect_manifests: no manifest-*.csv files found in %s", outdir)
        return 0
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for manifest in manifest_files:
        try:
            with open(manifest, newline="") as fh:
                for parts in csv.reader(fh):
                    if not parts:
                        continue
                    if len(parts) == 5:
                        slide_id, _sample_id, workflow_id, key, value = parts
                    elif len(parts) == 4:
                        slide_id, workflow_id, key, value = parts
                    else:
                        log.warning(
                            "collect_manifests: skipping malformed line in %s: %r",
                            manifest, parts,
                        )
                        continue
                    rows[(slide_id, key)] = {
                        "slide_id": slide_id,
                        "workflow_id": workflow_id,
                        "key": key,
                        "value": value,
                    }
        except OSError as exc:
            log.warning("collect_manifests: cannot read %s: %s", manifest, exc)
    os.makedirs(os.path.dirname(combined_path) or ".", exist_ok=True)
    with open(combined_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows.values())
    log.info(
        "collect_manifests: wrote %d unique rows to %s (from %d file(s))",
        len(rows), combined_path, len(manifest_files),
    )
    return len(rows)


class HookExecutor(Protocol):
    def __call__(
        self,
        cmd: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        hook_index: int,
    ) -> subprocess.CompletedProcess: ...


class FinalizationWorkerPool:
    """Bounded, keyed worker pool shared by local and remote finalizers."""

    def __init__(self, max_workers: int, *, thread_name_prefix: str):
        self.max_workers = max(1, int(max_workers or 1))
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._futures: dict[str, Future] = {}
        self._started_at: dict[str, float] = {}
        self._outcomes: dict[str, dict] = {}
        self._lock = threading.RLock()

    def submit(self, batch_id: str, fn: Callable, *args, **kwargs) -> Future | None:
        """Submit one batch if it is not already running and capacity exists."""
        with self._lock:
            current = self._futures.get(batch_id)
            if current and not current.done():
                return None
            # Do not allow a new attempt to hide an unconsumed worker failure.
            # The owner must first reconcile the durable batch and lease.
            if batch_id in self._outcomes:
                # Normally transient: the owner's done callback consumes the
                # outcome right after the pool records it.
                log.debug(
                    "Finalization for batch %s not submitted: previous attempt's "
                    "outcome has not been reconciled by its owner yet",
                    batch_id,
                )
                return None
            active = sum(not future.done() for future in self._futures.values())
            if active >= self.max_workers:
                log.debug(
                    "Finalization for batch %s not submitted: %d/%d worker(s) busy",
                    batch_id, active, self.max_workers,
                )
                return None
            future = self.executor.submit(fn, *args, **kwargs)
            self._futures[batch_id] = future
            self._started_at[batch_id] = time.monotonic()
            future.add_done_callback(
                lambda completed, key=batch_id: self._record_completion(key, completed)
            )
            return future

    def _record_completion(self, batch_id: str, future: Future) -> None:
        with self._lock:
            started_at = self._started_at.pop(batch_id, time.monotonic())
            error = None
            cancelled = future.cancelled()
            if not cancelled:
                try:
                    error = future.exception()
                except CancelledError as exc:
                    cancelled = True
                    error = exc
            self._outcomes[batch_id] = {
                "batch_id": batch_id,
                "started_monotonic": started_at,
                "finished_monotonic": time.monotonic(),
                "cancelled": cancelled,
                "exception": error,
            }
            if self._futures.get(batch_id) is future:
                self._futures.pop(batch_id, None)

    def drain_outcomes(self) -> list[dict]:
        """Return completed worker outcomes exactly once."""
        with self._lock:
            outcomes = list(self._outcomes.values())
            self._outcomes.clear()
            return outcomes

    def pop_outcome(self, batch_id: str) -> dict | None:
        """Consume one batch's completed outcome after its owner reconciled it."""
        with self._lock:
            return self._outcomes.pop(batch_id, None)

    def running_batch_ids(self) -> set[str]:
        with self._lock:
            return {
                batch_id
                for batch_id, future in self._futures.items()
                if not future.done()
            }

    def is_running(self, batch_id: str) -> bool:
        with self._lock:
            future = self._futures.get(batch_id)
            return bool(future and not future.done())

    def running_count(self) -> int:
        with self._lock:
            return sum(not future.done() for future in self._futures.values())

    def has_capacity(self) -> bool:
        return self.running_count() < self.max_workers

    def shutdown(self, *, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)


def local_hook_executor(
    cmd: list[str], *, cwd: str, env: dict[str, str], hook_index: int
) -> subprocess.CompletedProcess:
    del hook_index
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)


def format_hook_name(hook: dict, hook_index: int) -> str:
    if hook.get("_hook_name"):
        return str(hook["_hook_name"])
    phase = hook.get("_phase", "wds")
    command = str(hook.get("command", "")).strip()
    if "mussel_dispatcher.wds" in command:
        model = "wds"
        for arg in hook.get("args", []):
            if str(arg).startswith("--model-type="):
                model = str(arg).split("=", 1)[1] or model
                break
        return f"{phase}:{model}"
    if "sync_databricks" in command:
        return f"{phase}:databricks_sync"
    base = command.split()[-1] if command else f"hook_{hook_index + 1}"
    return f"{phase}:{base}"



def _resolve_hook_runtime(
    command: list[str],
    repo_dir: str,
    runtime_revision: str,
) -> tuple[str, str]:
    """Use the release named by an explicit hook config path when available.

    A historical batch can retain an older source revision while its hook
    command points at an immutable dispatcher config from a newer release.
    Running the old Python package against that newer config is unsafe because
    the config schema may have advanced independently.
    """
    config_path = ""
    for index, raw_part in enumerate(command):
        part = str(raw_part)
        if part == "--config" and index + 1 < len(command):
            config_path = str(command[index + 1])
            break
        if part.startswith("--config="):
            config_path = part.split("=", 1)[1]
            break
    if not config_path or not os.path.isabs(config_path):
        return repo_dir, runtime_revision
    try:
        config_file = Path(config_path).resolve()
    except OSError:
        return repo_dir, runtime_revision
    if config_file.parent.name != "dispatcher":
        return repo_dir, runtime_revision
    candidate = config_file.parent.parent
    if not (candidate / "dispatcher" / "mussel_dispatcher").is_dir():
        return repo_dir, runtime_revision
    revision_result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    effective_revision = revision_result.stdout.strip()
    if revision_result.returncode or not effective_revision:
        return repo_dir, runtime_revision
    if effective_revision != runtime_revision:
        log.info(
            "Hook config selects release %s instead of batch runtime %s",
            effective_revision,
            runtime_revision or "<unknown>",
        )
    return str(candidate), effective_revision


class PostBatchHooks:
    def __init__(
        self,
        cfg: Config,
        batch_id: str,
        state: StateStore,
        *,
        target: str = "",
        runtime_revision: str = "",
    ):
        self.cfg = cfg
        self.batch_id = batch_id
        self.state = state
        self.target = target or getattr(cfg, "default_dispatch_target", "") or "local"
        self.runtime_revision = runtime_revision

    def state_rows(self) -> list[dict]:
        return [
            {
                "hook_index": index,
                "phase": hook.get("_phase", "wds"),
                "hook_name": format_hook_name(hook, index),
                "required": 1 if hook.get("required", True) else 0,
                "command": " ".join(
                    [str(hook.get("command", "")).strip(), *map(str, hook.get("args", []))]
                ).strip(),
                "runtime_revision": self.runtime_revision,
            }
            for index, hook in enumerate(self.cfg.post_batch_hooks)
        ]

    def failed_coverage_only(self, *, phase: str) -> bool:
        """Return whether failed hooks are WDS coverage-only exit-2 failures."""
        rows = self.state.get_batch_hooks([self.batch_id]).get(self.batch_id, [])
        hook_rows = {int(row["hook_index"]): row for row in self.state_rows()}
        failed = [
            row
            for row in rows
            if row.get("status") == "FAILED"
            and hook_rows.get(int(row.get("hook_index", -1)), {}).get("phase", "wds") == phase
        ]
        if not failed:
            return False
        return all(
            int(row.get("exit_code") or -1) == 2
            and "mussel_dispatcher.wds" in str(
                hook_rows.get(int(row.get("hook_index", -1)), {}).get("command") or ""
            )
            for row in failed
        )

    def _failed_required_rows(self, *, phase: str | None) -> list[tuple[dict, dict]]:
        rows = self.state.get_batch_hooks([self.batch_id]).get(self.batch_id, [])
        hook_rows = {int(row["hook_index"]): row for row in self.state_rows()}
        failed = []
        for row in rows:
            hook_row = hook_rows.get(int(row.get("hook_index", -1)), {})
            if (
                row.get("status") == "FAILED"
                and hook_row.get("required", 1)
                and (phase is None or hook_row.get("phase", "wds") == phase)
            ):
                failed.append((row, hook_row))
        return failed

    def _terminal_failure_reason(self, row: dict) -> str:
        """Why a failed hook will not succeed by automatic retry ("" if it may)."""
        category = str(row.get("failure_category") or "")
        attempts = int(row.get("attempt_count") or 0)
        if category == HookFailureCategory.HOOK_BUG and attempts >= 1:
            return f"{category} awaiting manual retry"
        infra_cap = max(1, int(getattr(self.cfg, "hook_max_infra_retries", 3) or 3))
        if category == HookFailureCategory.NODE_INFRA and attempts >= infra_cap:
            return f"{category} retries exhausted ({attempts}/{infra_cap})"
        attempt_cap = int(getattr(self.cfg, "wds_hook_max_attempts", 0) or 0)
        if attempt_cap > 0 and attempts >= attempt_cap:
            return f"{category or 'FAILED'} after attempt cap ({attempts}/{attempt_cap})"
        return ""

    def terminal_failures(self, *, phase: str) -> list[str]:
        """Describe failed required hooks that automatic retry will not fix.

        Returns an empty list unless every failed required hook of the phase
        is either terminal or a coverage-only (exit 2) WDS failure, so a batch
        with any hook still worth retrying keeps waiting for it.
        """
        reasons = []
        for row, hook_row in self._failed_required_rows(phase=phase):
            if (
                int(row.get("exit_code") or -1) == 2
                and "mussel_dispatcher.wds" in str(hook_row.get("command") or "")
            ):
                continue
            reason = self._terminal_failure_reason(row)
            if not reason:
                return []
            reasons.append(f"{hook_row.get('hook_name') or row.get('hook_index')}: {reason}")
        return reasons

    def failure_summary(self, *, phase: str | None = None) -> str:
        """One line per failed required hook: name, category, attempts, detail."""
        parts = []
        for row, hook_row in self._failed_required_rows(phase=phase):
            detail = " ".join(
                str(row.get("failure_detail") or row.get("error_msg") or "").split()
            )[:200]
            parts.append(
                f"{hook_row.get('hook_name') or row.get('hook_index')} "
                f"{row.get('failure_category') or 'FAILED'} "
                f"(exit {row.get('exit_code')}) after {int(row.get('attempt_count') or 0)} "
                f"attempt(s)" + (f": {detail}" if detail else "")
            )
        return "; ".join(parts)

    def waive_failed_coverage_hooks(self, *, phase: str, reason: str) -> int:
        """Record validation-only hook failures as handled after requeueing gaps."""
        rows = self.state.get_batch_hooks([self.batch_id]).get(self.batch_id, [])
        hook_rows = {int(row["hook_index"]): row for row in self.state_rows()}
        waived = 0
        for row in rows:
            index = int(row.get("hook_index", -1))
            if (
                row.get("status") == "FAILED"
                and hook_rows.get(index, {}).get("phase", "wds") == phase
            ):
                self.state.waive_batch_hook(self.batch_id, index, reason=reason)
                waived += 1
        return waived

    def run(
        self,
        batch_csv: str,
        *,
        phase: str = "",
        outdir: str | None = None,
        repo_dir: str | None = None,
        path_replacements: dict[str, str] | None = None,
        executor: HookExecutor = local_hook_executor,
    ) -> bool:
        """Run one hook phase, serializing DB sync across local and remote targets."""
        has_db_sync_hooks = any(
            hook.get("_phase", "wds") == "db_sync"
            for hook in self.cfg.post_batch_hooks
        )
        if phase != "db_sync" or not has_db_sync_hooks:
            return self._run_unlocked(
                batch_csv,
                phase=phase,
                outdir=outdir,
                repo_dir=repo_dir,
                path_replacements=path_replacements,
                executor=executor,
            )

        owner = f"db-sync:{os.getpid()}:{self.batch_id}"
        ttl_seconds = max(
            1, int(getattr(self.cfg, "finalization_lease_ttl_seconds", 300) or 300)
        )
        if not self.state.claim_finalization_lease(
            self.batch_id,
            owner=owner,
            ttl_seconds=ttl_seconds,
            lease_name=DB_SYNC_LEASE_NAME,
        ):
            log.info(
                "Batch %s: shared Databricks sync lease is busy; queued for retry",
                self.batch_id,
            )
            return False

        stop_heartbeat = threading.Event()

        def heartbeat() -> None:
            interval = max(5, min(60, ttl_seconds // 3 or 1))
            while not stop_heartbeat.wait(interval):
                try:
                    self.state.heartbeat_finalization_lease(
                        self.batch_id,
                        owner=owner,
                        ttl_seconds=ttl_seconds,
                        lease_name=DB_SYNC_LEASE_NAME,
                    )
                except Exception:
                    log.debug(
                        "Batch %s: DB sync lease heartbeat failed",
                        self.batch_id,
                        exc_info=True,
                    )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"db-sync-heartbeat-{self.batch_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            return self._run_unlocked(
                batch_csv,
                phase=phase,
                outdir=outdir,
                repo_dir=repo_dir,
                path_replacements=path_replacements,
                executor=executor,
            )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=5)
            self.state.release_finalization_lease(
                self.batch_id,
                owner=owner,
                lease_name=DB_SYNC_LEASE_NAME,
            )

    def _run_unlocked(
        self,
        batch_csv: str,
        *,
        phase: str = "",
        outdir: str | None = None,
        repo_dir: str | None = None,
        path_replacements: dict[str, str] | None = None,
        executor: HookExecutor = local_hook_executor,
    ) -> bool:
        if not self.cfg.post_batch_hooks:
            return True
        self.state.sync_batch_hooks(self.batch_id, self.state_rows())
        existing = self.state.get_batch_hooks([self.batch_id]).get(self.batch_id, [])
        existing_by_index = {int(row["hook_index"]): row for row in existing}
        succeeded = {
            int(row["hook_index"])
            for row in existing
            if row.get("status") in {"SUCCEEDED", "SKIPPED", "WAIVED"}
        }
        hook_outdir = outdir or self.cfg.outdir
        hook_repo_dir = repo_dir or self.cfg.repo_dir

        def substitute(value: str) -> str:
            rendered = value.format(
                batch_csv=batch_csv,
                batch_id=self.batch_id,
                outdir=hook_outdir,
                repo_dir=hook_repo_dir,
            )
            for source, replacement in (path_replacements or {}).items():
                rendered = rendered.replace(source, replacement)
            return rendered

        hook_env = os.environ.copy()
        for watcher in self.cfg.watchers:
            if watcher.s3_access_key:
                hook_env.setdefault("ECS_ACCESS_KEY", watcher.s3_access_key)
            if watcher.s3_secret_key:
                hook_env.setdefault("ECS_SECRET_KEY", watcher.s3_secret_key)
            if watcher.s3_endpoint:
                hook_env.setdefault("ECS_ENDPOINT_URL", watcher.s3_endpoint)
        hook_base_env = hook_env

        phase_ok = True
        canary_claimed = False
        initial_circuit = self.state.hook_circuit_status(self.target) or {}
        if isinstance(initial_circuit, dict) and str(initial_circuit.get("state") or "CLOSED") == "HALF_OPEN":
            if not self.state.claim_hook_canary(self.target, self.batch_id):
                log.error("Post-batch hook canary is already claimed for target %s", self.target)
                return False
            canary_claimed = True
        for index, hook in enumerate(self.cfg.post_batch_hooks):
            hook_phase = hook.get("_phase", "wds")
            if phase and hook_phase != phase:
                continue
            if index in succeeded:
                log.info("Batch %s: hook %d already succeeded; skipping replay", self.batch_id, index)
                continue
            required = bool(hook.get("required", True))
            row = existing_by_index.get(index, {})
            circuit = self.state.hook_circuit_status(self.target) or {}
            if not isinstance(circuit, dict):
                circuit = {}
            canary_mode = str(circuit.get("state") or "CLOSED") == "HALF_OPEN"
            if row.get("status") == "FAILED":
                if not required:
                    continue
                previous_category = str(row.get("failure_category") or "")
                previous_attempts = int(row.get("attempt_count") or 0)
                if (
                    not canary_mode
                    and previous_category == HookFailureCategory.NODE_INFRA
                    and previous_attempts >= max(
                        1, int(getattr(self.cfg, "hook_max_infra_retries", 3) or 3)
                    )
                ):
                    phase_ok = False
                    log.error(
                        "Batch %s: hook %d reached infrastructure retry cap (%d)",
                        self.batch_id, index, previous_attempts,
                    )
                    continue
                if not canary_mode and previous_category == HookFailureCategory.HOOK_BUG and previous_attempts >= 1:
                    phase_ok = False
                    log.error(
                        "Batch %s: hook %d is a hook-logic failure; manual retry required",
                        self.batch_id, index,
                    )
                    continue
                retry_at = str(row.get("next_retry_at") or "")
                if retry_at:
                    try:
                        if not canary_mode and datetime.fromisoformat(retry_at).timestamp() > time.time():
                            if required:
                                phase_ok = False
                            continue
                    except ValueError:
                        pass
            if str(circuit.get("state") or "CLOSED") == "HALF_OPEN":
                if not self.state.claim_hook_canary(self.target, self.batch_id):
                    phase_ok = False
                    log.error(
                        "Post-batch hook canary already claimed for target %s: %s",
                        self.target, circuit.get("canary_batch_id") or "unknown",
                    )
                    continue
                canary_claimed = True
            elif str(circuit.get("state") or "CLOSED") == "OPEN" or int(circuit.get("paused") or 0):
                phase_ok = False
                log.error(
                    "Post-batch hooks paused for target %s: %s",
                    self.target, circuit.get("reason") or "circuit breaker is open",
                )
                continue
            command = str(hook.get("command", ""))
            if not command:
                if required:
                    self.state.start_batch_hook(self.batch_id, index)
                    self.state.complete_batch_hook(
                        self.batch_id, index, exit_code=1,
                        error_msg="required hook has no command",
                    )
                    phase_ok = False
                else:
                    self.state.skip_batch_hook(
                        self.batch_id, index, reason="optional hook has no command"
                    )
                continue
            hook_env = hook_base_env.copy()
            cmd = [substitute(part) for part in shlex.split(command)]
            cmd += [substitute(str(arg)) for arg in hook.get("args", [])]
            execution_repo_dir, hook_runtime_revision = _resolve_hook_runtime(
                cmd, hook_repo_dir, self.runtime_revision
            )
            if cmd and cmd[0] == "python":
                cmd[0] = sys.executable
            cmd = launch_command(cmd, execution_repo_dir)
            if "mussel_dispatcher.wds" in command:
                for key in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS",
                ):
                    hook_env.setdefault(key, "1")
                hook_env.setdefault("TOKENIZERS_PARALLELISM", "false")
            hook_env["MUSSEL_HOOK_TARGET"] = self.target
            hook_env["MUSSEL_RUNTIME_REVISION"] = hook_runtime_revision
            hook_env["MUSSEL_HOOK_IDEMPOTENCY_KEY"] = str(row.get("idempotency_key") or "")
            attempt_number = self.state.start_batch_hook(self.batch_id, index)
            try:
                if getattr(self.cfg, "hook_preflight", True):
                    preflight = preflight_hook(
                        command=cmd,
                        cwd=execution_repo_dir,
                        env=hook_env,
                        paths=(batch_csv, execution_repo_dir),
                        active_revision=hook_runtime_revision,
                    )
                    if preflight is not None:
                        if required and preflight.category != HookFailureCategory.COVERAGE_MISSING:
                            phase_ok = False
                        self.state.complete_batch_hook(
                            self.batch_id,
                            index,
                            exit_code=78,
                            error_msg=preflight.detail,
                            failure_category=preflight.category,
                            failure_detail=preflight.detail,
                            runtime_revision=hook_runtime_revision,
                        )
                        if required:
                            tripped = self.state.record_hook_failure(
                                self.target,
                                preflight.category,
                                preflight.detail,
                                batch_id=self.batch_id,
                                hook_name=format_hook_name(hook, index),
                                attempt_number=attempt_number,
                                threshold=getattr(self.cfg, "hook_circuit_breaker_threshold", 3),
                                window_seconds=getattr(self.cfg, "hook_circuit_breaker_window_seconds", 300),
                            )
                            if tripped:
                                log.error("Hook circuit breaker opened for target %s", self.target)
                        continue
                result = executor(
                    cmd, cwd=execution_repo_dir, env=hook_env, hook_index=index
                )
                if result.returncode == 0:
                    self.state.complete_batch_hook(
                        self.batch_id, index, exit_code=0,
                        runtime_revision=hook_runtime_revision,
                    )
                    log.info("Batch %s: post-batch hook succeeded", self.batch_id)
                else:
                    if required:
                        phase_ok = False
                    error = (result.stderr or result.stdout or "")[-1000:]
                    outcome = classify_hook_failure(
                        result.returncode,
                        error,
                        hook_command=command,
                    )
                    execution_node = ""
                    node_match = re.search(r"(?:hook )?node=([^\s]+)", error, re.IGNORECASE)
                    if node_match:
                        execution_node = node_match.group(1)
                    self.state.complete_batch_hook(
                        self.batch_id,
                        index,
                        exit_code=result.returncode,
                        error_msg=error,
                        failure_category=outcome.category,
                        failure_detail=outcome.detail,
                        runtime_revision=hook_runtime_revision,
                        execution_node=execution_node,
                    )
                    if required and outcome.category != HookFailureCategory.COVERAGE_MISSING:
                        tripped = self.state.record_hook_failure(
                            self.target,
                            outcome.category,
                            error or outcome.detail,
                            batch_id=self.batch_id,
                            hook_name=format_hook_name(hook, index),
                            attempt_number=attempt_number,
                            threshold=getattr(self.cfg, "hook_circuit_breaker_threshold", 3),
                            window_seconds=getattr(self.cfg, "hook_circuit_breaker_window_seconds", 300),
                        )
                        if tripped:
                            log.error("Hook circuit breaker opened for target %s", self.target)
                    log.error("Post-batch hook failed (exit %d):\n%s", result.returncode, error)
            except Exception as exc:
                if required:
                    phase_ok = False
                outcome = classify_hook_failure(None, str(exc), hook_command=command)
                self.state.complete_batch_hook(
                    self.batch_id,
                    index,
                    exit_code=None,
                    error_msg=str(exc)[:1000],
                    failure_category=outcome.category,
                    failure_detail=outcome.detail,
                    runtime_revision=hook_runtime_revision,
                )
                if required:
                    self.state.record_hook_failure(
                        self.target,
                        outcome.category,
                        str(exc),
                        batch_id=self.batch_id,
                        hook_name=format_hook_name(hook, index),
                        attempt_number=attempt_number,
                        threshold=getattr(self.cfg, "hook_circuit_breaker_threshold", 3),
                        window_seconds=getattr(self.cfg, "hook_circuit_breaker_window_seconds", 300),
                    )
                log.error("Post-batch hook raised: %s", exc)
        if canary_claimed and (
            phase in {"", "db_sync"}
            or not any(hook.get("_phase", "wds") == "db_sync" for hook in self.cfg.post_batch_hooks)
        ):
            self.state.finish_hook_canary(
                self.target,
                self.batch_id,
                success=phase_ok,
                reason="canary hook failure" if not phase_ok else "",
            )
        return phase_ok

    def finish_canary(self, *, success: bool, reason: str = "") -> None:
        """Complete a canary spanning multiple finalization phases."""
        self.state.finish_hook_canary(
            self.target,
            self.batch_id,
            success=success,
            reason=reason,
        )


@dataclass(frozen=True)
class FinalizationContext:
    batch_csv: str
    outdir: str
    repo_dir: str
    run_started_at: float = 0.0
    path_replacements: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalizationResult:
    wds_hooks_succeeded: bool
    db_sync_hooks_succeeded: bool
    # Why finalization is incomplete (empty once the batch completed); the
    # owner records it as the batch's finalization_error.
    detail: str = ""


class BatchFinalizer:
    """Single success-finalization contract shared by local and remote runs."""

    def __init__(
        self,
        cfg: Config,
        batch_id: str,
        state: StateStore,
        *,
        target: str = "",
        runtime_revision: str = "",
    ):
        self.cfg = cfg
        self.batch_id = batch_id
        self.state = state
        self.hooks = PostBatchHooks(
            cfg, batch_id, state, target=target, runtime_revision=runtime_revision
        )

    def finalize_success(
        self,
        context: FinalizationContext,
        *,
        hook_executor: HookExecutor = local_hook_executor,
        run_hooks: Callable[..., bool] | None = None,
        collect_manifest: Callable[[float], None] | None = None,
        cleanup_intermediate: Callable[[], None] | None = None,
        verify_wds: Callable[[str], bool | None] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> FinalizationResult:
        self.state.set_batch_status(self.batch_id, "FINALIZING")
        batch = self.state.get_batch(self.batch_id) or {}
        stage = str(batch.get("finalization_stage") or "PENDING").upper()
        if stage in {"PENDING", "PREPARE"}:
            if collect_manifest is None:
                self._collect_manifest(context.outdir, context.run_started_at)
            else:
                collect_manifest(context.run_started_at)
            self.state.set_batch_finalization_stage(self.batch_id, "WDS_HOOKS")
            stage = "WDS_HOOKS"
        hook_kwargs = {
            "outdir": context.outdir,
            "repo_dir": context.repo_dir,
            "path_replacements": context.path_replacements,
        }
        wds_ok = True
        if stage == "WDS_HOOKS":
            if run_hooks is None:
                wds_ok = self.hooks.run(
                    context.batch_csv,
                    phase="wds",
                    executor=hook_executor,
                    **hook_kwargs,
                )
            else:
                outcome = run_hooks(context.batch_csv, phase="wds", **hook_kwargs)
                wds_ok = True if outcome is None else bool(outcome)
            # A WDS hook can append the slides whose outputs exist and still
            # fail validation when Nextflow ignored one or more task failures.
            # Reconcile coverage before returning so those slides are requeued
            # rather than remaining DISPATCHED forever.
            self.state.mark_slides_complete(self.batch_id, succeeded=True)
            if not wds_ok:
                # Coverage reconciliation requeues missing slides (SUCCEEDED ->
                # PENDING, fail_count+1, bounded by max_slide_retries). Run it
                # only when the WDS hooks cannot do better by retrying: a
                # coverage-only failure (exit 2: the hook appended what
                # existed) or a terminal one (hook bug awaiting manual retry,
                # infra retries exhausted, attempt cap reached). A transient
                # failure (S3 error, preflight 78, open breaker, backoff) leaves
                # the slides untouched and the batch FINALIZING for a retry.
                coverage_only = self.hooks.failed_coverage_only(phase="wds")
                terminal = [] if coverage_only else self.hooks.terminal_failures(phase="wds")
                summary = self.hooks.failure_summary(phase="wds")
                if not (coverage_only or terminal):
                    self.hooks.finish_canary(success=False, reason="WDS hook failure")
                    log.warning(
                        "Batch %s: required WDS hook failed; keeping batch in FINALIZING "
                        "with slides untouched for retry: %s",
                        self.batch_id, summary or "no hook result recorded",
                    )
                    return FinalizationResult(
                        False, False,
                        detail=f"WDS hook retry pending: {summary or 'hooks not run'}",
                    )
                coverage_ok = (
                    self._verify_wds_coverage(context.batch_csv, context.outdir)
                    if verify_wds is None
                    else bool(verify_wds(context.batch_csv))
                )
                if not coverage_ok:
                    self.hooks.finish_canary(success=False, reason="WDS coverage verification failed")
                    log.warning(
                        "Batch %s: WDS coverage reconciliation left unresolved slides; "
                        "keeping batch in FINALIZING",
                        self.batch_id,
                    )
                    return FinalizationResult(
                        False, False,
                        detail=f"WDS coverage unresolved after hook failure: {summary}",
                    )
                if terminal:
                    log.error(
                        "Batch %s: WDS hook(s) will not succeed by retry (%s); reconciled "
                        "coverage from the WDS manifest and requeued missing slides",
                        self.batch_id, "; ".join(terminal),
                    )
                    reason = (
                        "terminal WDS hook failure (" + "; ".join(terminal)
                        + "); slides missing from WDS were requeued"
                    )
                else:
                    reason = "WDS appended completed slides; missing outputs were requeued"
                waived = self.hooks.waive_failed_coverage_hooks(phase="wds", reason=reason)
                log.warning(
                    "Batch %s: WDS hook failure handled by coverage reconciliation; "
                    "%d hook(s) waived after requeue",
                    self.batch_id,
                    waived,
                )
                wds_ok = True
            self.state.set_batch_finalization_stage(self.batch_id, "VERIFY_WDS")
            stage = "VERIFY_WDS"
        if stage == "VERIFY_WDS":
            if cleanup_intermediate is None:
                self._cleanup_intermediate_features(context.outdir)
            else:
                cleanup_intermediate()
            if verify_wds is None:
                wds_complete = self._verify_wds_coverage(context.batch_csv, context.outdir)
            else:
                wds_complete = verify_wds(context.batch_csv)
            if wds_complete is False:
                self.hooks.finish_canary(success=False, reason="WDS coverage incomplete")
                log.warning("Batch %s: WDS coverage is incomplete; keeping batch in FINALIZING", self.batch_id)
                return FinalizationResult(True, False, detail="WDS coverage incomplete")
            self.state.set_batch_finalization_stage(self.batch_id, "DB_SYNC")
            stage = "DB_SYNC"
        if run_hooks is None:
            db_sync_ok = self.hooks.run(context.batch_csv, phase="db_sync", executor=hook_executor, **hook_kwargs)
        else:
            outcome = run_hooks(context.batch_csv, phase="db_sync", **hook_kwargs)
            db_sync_ok = True if outcome is None else bool(outcome)
        if not db_sync_ok:
            self.hooks.finish_canary(success=False, reason="database-sync hook failure")
            log.warning("Batch %s: required database-sync hook failed; keeping batch in FINALIZING", self.batch_id)
            summary = self.hooks.failure_summary(phase="db_sync")
            return FinalizationResult(
                True, False,
                detail=f"database-sync hook incomplete: {summary or 'sync lease busy or hook paused'}",
            )
        self.hooks.finish_canary(success=True)
        self.state.complete_batch(self.batch_id, exit_code=0)
        if cleanup is not None:
            cleanup()
        return FinalizationResult(wds_ok, db_sync_ok)

    def _collect_manifest(self, outdir: str, run_started_at: float) -> None:
        candidates = [
            path for path in glob.glob(os.path.join(outdir, "manifest-*.csv"))
            if os.path.getmtime(path) >= run_started_at
        ]
        if candidates:
            manifest = max(candidates, key=os.path.getmtime)
            self.state.record_batch_manifest(self.batch_id, manifest)
            log.info("Batch %s produced manifest: %s", self.batch_id, manifest)
        else:
            log.warning("Batch %s: no manifest-*.csv found in %s after run start", self.batch_id, outdir)
        combined = self.cfg.resolved_combined_manifest_path()
        if outdir != self.cfg.outdir and combined.startswith(self.cfg.outdir.rstrip("/") + "/"):
            combined = outdir.rstrip("/") + combined[len(self.cfg.outdir.rstrip("/")):]
        with _MANIFEST_COLLECTION_LOCK:
            collect_manifests(outdir, combined)

    def _cleanup_intermediate_features(self, outdir: str) -> None:
        if not self.cfg.cleanup_results:
            return
        features_dir = os.path.join(outdir, "features")
        if not os.path.isdir(features_dir):
            return
        wds_models = {
            model
            for watcher in self.cfg.watchers
            for model in getattr(watcher, "wds_destinations", {})
        }
        if not wds_models:
            return
        for entry in os.scandir(features_dir):
            if entry.is_dir() and entry.name not in wds_models:
                shutil.rmtree(entry.path, ignore_errors=True)
                log.info("Removing intermediate features directory: %s", entry.path)

    def _verify_wds_coverage(self, batch_csv: str, outdir: str) -> bool:
        required_models = {
            model
            for watcher in self.cfg.watchers
            for model in getattr(watcher, "wds_destinations", {})
        }
        if not required_models:
            return True
        manifest_path = os.path.join(outdir, "wds_manifest.csv")
        if not os.path.exists(manifest_path):
            log.warning(
                "Batch %s: WDS manifest missing at %s; all batch slides remain incomplete",
                self.batch_id, manifest_path,
            )
            wds_by_model: dict[str, set[str]] = {}
        else:
            wds_by_model = {}
            with open(manifest_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    wds_by_model.setdefault(row["model"], set()).add(row["slide_id"])
        with open(batch_csv, newline="") as fh:
            batch_ids = {row["slide_id"] for row in csv.DictReader(fh)}
        incomplete = sorted(
            slide_id
            for slide_id in batch_ids
            if any(slide_id not in wds_by_model.get(model, set()) for model in required_models)
        )
        if not incomplete:
            log.info(
                "Batch %s: WDS coverage verified - all %d slide(s) present for all models",
                self.batch_id, len(batch_ids),
            )
            return True
        retryable, permanently_failed = self.state.reconcile_incomplete_wds_slides(
            incomplete,
            source_batch_id=self.batch_id,
            max_retries=self.cfg.max_slide_retries or 0,
        )
        statuses = self.state.get_slide_statuses(incomplete)
        handed_off = [
            slide_id
            for slide_id in incomplete
            if (
                statuses.get(slide_id, {}).get("status") in {"DISPATCHED", "SUCCEEDED"}
                and statuses.get(slide_id, {}).get("batch_id") not in {None, self.batch_id}
            )
        ]
        unresolved = [
            slide_id
            for slide_id in incomplete
            if (
                statuses.get(slide_id, {}).get("status") not in {"PENDING", "FAILED"}
                and slide_id not in handed_off
            )
        ]
        log.warning(
            "Batch %s: WDS incomplete for %d/%d slide(s); %d requeued, %d permanently failed, "
            "%d handed off, %d unresolved",
            self.batch_id,
            len(incomplete),
            len(batch_ids),
            len(retryable),
            len(permanently_failed),
            len(handed_off),
            len(unresolved),
        )
        return not unresolved
