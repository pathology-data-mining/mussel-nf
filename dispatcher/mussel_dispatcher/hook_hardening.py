"""Failure classification and preflight checks for post-batch hooks.

Hooks run on two intentionally different execution profiles.  This module
keeps the safety checks independent of the executor so local and Slurm hooks
report the same categories to the dispatcher.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .hook_launcher import is_launched


_PREFLIGHT_IMPORT_CACHE: set[tuple[str, str, str, str, str]] = set()
_PREFLIGHT_IMPORT_LOCK = threading.Lock()


class HookFailureCategory:
    COVERAGE_MISSING = "COVERAGE_MISSING"
    NODE_INFRA = "NODE_INFRA"
    TRANSIENT_SERVICE = "TRANSIENT_SERVICE"
    RELEASE_OR_CONFIG = "RELEASE_OR_CONFIG"
    HOOK_BUG = "HOOK_BUG"


@dataclass(frozen=True)
class HookOutcome:
    category: str
    detail: str


def classify_hook_failure(
    returncode: int | None,
    output: str = "",
    *,
    hook_command: str = "",
    slurm_state: str = "",
) -> HookOutcome:
    """Map an executor result to a bounded recovery policy."""
    text = (output or "").lower()
    state = (slurm_state or "").upper()
    if not state:
        # The Slurm executor reports the job's end state as a line in the output.
        state_match = re.search(r"slurm hook state=([a-z_]+)", text)
        state = state_match.group(1).upper() if state_match else ""
    marker = re.search(r"\[mussel-hook-preflight\]\s+category=([A-Z_]+)", output or "", re.IGNORECASE)
    if marker:
        category = marker.group(1).upper()
        if category in {
            HookFailureCategory.COVERAGE_MISSING,
            HookFailureCategory.NODE_INFRA,
            HookFailureCategory.TRANSIENT_SERVICE,
            HookFailureCategory.RELEASE_OR_CONFIG,
            HookFailureCategory.HOOK_BUG,
        }:
            detail_match = re.search(
                r"\[mussel-hook-preflight\].*?detail=(.*)", output or "", re.IGNORECASE
            )
            return HookOutcome(category, (detail_match.group(1).strip() if detail_match else category))
    if state in {
        "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "PREEMPTED",
        "CANCELLED", "TIMEOUT", "NF", "REVOKED",
    }:
        return HookOutcome(HookFailureCategory.NODE_INFRA, f"Slurm state={state}")
    if any(marker in text for marker in (
        "missing weka", "node_fail", "connection reset", "broken pipe",
        "no space left on device", "input/output error", "stale file handle",
        "out_of_memory", "slurm hook state=timeout", "slurm hook state=cancelled",
    )):
        return HookOutcome(HookFailureCategory.NODE_INFRA, "node or filesystem failure")
    if returncode == 2 and "usage:" in text and ": error:" in text:
        # argparse exits 2 as well (e.g. a release without a newly added flag). That is
        # a release problem for the circuit breaker, not missing slides to retry.
        return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, "hook command-line usage error")
    if returncode == 2 and (re.search(r"can't open file '.*': \[errno \d+\]", text)
                            or "usage: hook_launcher.py" in text):
        # Python exits 2 for a script it cannot open (a removed release's launcher),
        # and so did the launcher's own usage error: a release problem, not coverage.
        return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, "hook launcher is missing or misinvoked")
    if returncode == 2 and "mussel_dispatcher.wds" in hook_command:
        # wds.main's "WDS validation failed for <model>: ... no index entry"; an exit 2
        # with no recognizable marker is conservatively treated the same way.
        return HookOutcome(HookFailureCategory.COVERAGE_MISSING, "WDS coverage is incomplete")
    if returncode == 78:
        # The details the Slurm hook preflight prints, should its marker be cut off.
        if "batch-csv-unreadable" in text or "dispatcher-package-missing" in text:
            return HookOutcome(HookFailureCategory.NODE_INFRA, "compute-node hook preflight path is unavailable")
        if "dispatcher-import-failed" in text:
            return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, "compute-node hook import preflight failed")
        return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, "hook preflight failed")
    # Before the release/config markers: a timeout's traceback names files and
    # paths too, and must stay retryable.
    if any(marker in text for marker in (
        "timed out", "timeout", "connection refused", "temporarily unavailable",
        "too many requests", "deadlock", "lock timeout", "database is locked",
        "wds append lock lost",
    )):
        return HookOutcome(HookFailureCategory.TRANSIENT_SERVICE, "transient service failure")
    if any(marker in text for marker in (
        "modulenotfounderror", "no module named", "source release drift",
        "release manifest drift", "execution profile drift", "schema-v",
        "permission denied", "cannot import", "unsupported work-order",
        "path mapping",
    )):
        return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, "runtime or configuration mismatch")
    return HookOutcome(HookFailureCategory.HOOK_BUG, "hook returned a non-success result")


_PREFLIGHT_MARKER_LINE = re.compile(r"^.*\[mussel-hook-preflight\]\s+category=.*$", re.MULTILINE | re.IGNORECASE)


def hook_failure_text(stdout: str | None, stderr: str | None, limit: int = 1000) -> str:
    """The tail of a failed hook's output, keeping a preflight marker from either stream.

    The Slurm preflight prints its marker on stdout, while stderr usually has
    content too (the Slurm state and node lines, a traceback).
    """
    text = (stderr or stdout or "")[-limit:]
    for stream in (stderr, stdout):
        markers = _PREFLIGHT_MARKER_LINE.findall(stream or "")
        if markers:
            marker = markers[-1].strip()
            if marker not in text:
                text = (text.rstrip() + "\n" + marker)[-limit:]
            break
    return text


def preflight_hook(
    *,
    command: Sequence[str],
    cwd: str,
    env: Mapping[str, str],
    paths: Sequence[str] = (),
    active_revision: str = "",
) -> HookOutcome | None:
    """Validate the runtime before submitting/executing hook application code.

    The returned outcome is suitable for persistence as a failed hook.  A
    ``None`` result means the runtime passed preflight.
    """
    root = Path(cwd)
    has_package = (root / "dispatcher" / "mussel_dispatcher").is_dir()
    # Arbitrary user hooks may intentionally run outside the dispatcher source
    # tree.  Their executor result is still classified, but dispatcher-specific
    # import/path checks apply only to a dispatcher runtime.
    if not has_package:
        return None
    if active_revision:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        actual_revision = revision.stdout.strip()
        if revision.returncode or actual_revision != active_revision:
            return HookOutcome(
                HookFailureCategory.RELEASE_OR_CONFIG,
                f"active release mismatch: expected {active_revision}, got {actual_revision or '<unknown>'}",
            )
    if active_revision and not active_revision.strip():
        return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, "active release revision is empty")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, f"required hook path is missing: {path}")
        required_access = os.R_OK | (os.X_OK if path.is_dir() else 0)
        if not os.access(path, required_access):
            return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, f"hook path is not readable: {path}")
    python = str(command[0]) if command else "python"
    # A launched hook checks where its module comes from in the process that runs it.
    if os.path.basename(python).startswith("python") and not is_launched(list(command)):
        check_env = dict(env)
        package_root = str(root / "dispatcher")
        old_path = check_env.get("PYTHONPATH", "")
        check_env["PYTHONPATH"] = package_root + ((os.pathsep + old_path) if old_path else "")
        module_name = ""
        for index, part in enumerate(command[:-1]):
            if str(part) == "-m":
                module_name = str(command[index + 1])
                break
        import_check = (
            f"import importlib; importlib.import_module({module_name!r})"
            if module_name else "import mussel_dispatcher"
        )
        cache_key = (
            python,
            str(root.resolve()),
            module_name or "mussel_dispatcher",
            active_revision,
            check_env.get("PYTHONPATH", ""),
        )
        # A conda-backed import can be expensive on a shared filesystem. The
        # remote finalizer runs several hooks concurrently, so serialize the
        # first check and reuse a successful result for this immutable release.
        # Failed checks are intentionally not cached: a transient import issue
        # must still be visible to the circuit breaker.
        with _PREFLIGHT_IMPORT_LOCK:
            if cache_key in _PREFLIGHT_IMPORT_CACHE:
                return None
            result = subprocess.run(
                [python, "-c", import_check],
                cwd=str(root), env=check_env, capture_output=True, text=True,
                timeout=30, check=False,
            )
        if result.returncode:
            detail = (result.stderr or result.stdout or "dispatcher import preflight failed")[-1000:]
            return HookOutcome(HookFailureCategory.RELEASE_OR_CONFIG, detail)
        with _PREFLIGHT_IMPORT_LOCK:
            _PREFLIGHT_IMPORT_CACHE.add(cache_key)
    return None
