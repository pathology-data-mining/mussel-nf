"""Shared execution-storage cleanup and capacity checks.

The local and remote execution backends have different path layouts and driver
liveness sources, but their storage safety policy is the same.  This module
keeps deletion and quota handling in one place.
"""
from __future__ import annotations

import errno
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import StoragePolicyConfig

log = logging.getLogger("mussel-dispatcher.storage")

_GIB = 1024 ** 3


@dataclass(frozen=True)
class ExecutionCleanupSpec:
    """Paths that may be deleted after a driver reaches a terminal state."""

    batch_id: str
    managed_root: str
    run_root: str
    execution_paths: tuple[str, ...]
    protected_paths: tuple[str, ...] = ()


@dataclass
class CleanupResult:
    batch_id: str
    removed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str = ""
    retryable: bool = True

    @property
    def succeeded(self) -> bool:
        return not self.error


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, parent: Path, *, allow_equal: bool = False) -> bool:
    if allow_equal and path == parent:
        return True
    return parent in path.parents


class ExecutionArtifactCleaner:
    """Delete only explicitly identified, backend-owned execution artifacts."""

    def cleanup(self, spec: ExecutionCleanupSpec) -> CleanupResult:
        result = CleanupResult(spec.batch_id)
        try:
            if Path(spec.run_root).expanduser().is_symlink():
                raise ValueError(f"refusing symlink run root {spec.run_root}")
            managed = _resolved(spec.managed_root)
            run_root = _resolved(spec.run_root)
            if not _is_within(run_root, managed, allow_equal=True):
                raise ValueError(
                    f"run root {run_root} is outside managed root {managed}"
                )

            protected = [_resolved(path) for path in spec.protected_paths if path]
            for raw_path in spec.execution_paths:
                path = Path(raw_path).expanduser()
                if path.is_symlink():
                    raise ValueError(f"refusing symlink cleanup path {path}")
                resolved = _resolved(path)
                if not _is_within(resolved, run_root, allow_equal=True):
                    raise ValueError(
                        f"cleanup path {resolved} is outside run root {run_root}"
                    )
                if not _is_within(resolved, managed, allow_equal=True):
                    raise ValueError(
                        f"cleanup path {resolved} is outside managed root {managed}"
                    )
                for protected_path in protected:
                    if (
                        resolved == protected_path
                        or _is_within(resolved, protected_path, allow_equal=True)
                        or _is_within(protected_path, resolved, allow_equal=True)
                    ):
                        raise ValueError(
                            f"cleanup path {resolved} overlaps protected path {protected_path}"
                        )
                if not path.exists():
                    result.missing.append(str(path))
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                result.removed.append(str(path))
        except OSError as exc:
            result.error = str(exc)
            log.error("Batch %s execution cleanup failed: %s", spec.batch_id, exc)
        except ValueError as exc:
            result.error = str(exc)
            result.retryable = False
            log.error("Batch %s execution cleanup rejected: %s", spec.batch_id, exc)
        if result.removed:
            log.info(
                "Batch %s removed execution artifacts: %s",
                spec.batch_id,
                ", ".join(result.removed),
            )
        return result


def reconcile_execution_cleanup(
    state,
    spec_factory,
    *,
    cleaner=None,
    candidates=None,
    limit: int | None = None,
    budget_seconds: float | None = None,
) -> int:
    """Retry bounded cleanup for driver-terminal batches with missing cleanup."""
    cleaner = cleaner or ExecutionArtifactCleaner()
    cleaned = 0
    started = time.monotonic()
    processed = 0
    if candidates is None:
        candidates = state.get_execution_cleanup_candidates()
    for batch in candidates:
        if limit is not None and processed >= max(0, int(limit)):
            break
        if (
            budget_seconds is not None
            and processed > 0
            and time.monotonic() - started >= max(0.0, float(budget_seconds))
        ):
            break
        spec = spec_factory(batch)
        if spec is None:
            continue
        processed += 1
        result = cleaner.cleanup(spec)
        state.record_execution_cleanup(
            batch["batch_id"],
            error=result.error,
            retryable=result.retryable,
        )
        if result.succeeded:
            cleaned += 1
    return cleaned


def existing_capacity_path(path: str) -> str:
    """Return an existing directory for stat/probe operations."""

    candidate = Path(path).expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return str(candidate)


@dataclass(frozen=True)
class CapacitySnapshot:
    path: str
    total_bytes: int = 0
    free_bytes: int = 0
    free_percent: float = 0.0
    healthy: bool = True
    reason: str = ""


class CapacityGuard:
    """Apply quota/filesystem thresholds with hysteretic recovery."""

    def __init__(self, policy: StoragePolicyConfig, *, name: str):
        self.policy = policy
        self.name = name
        self.paused = False
        self.healthy_polls = 0
        self.last_snapshots: list[CapacitySnapshot] = []
        self.last_checked_at = 0.0
        self.last_blockers: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return bool(
            float(self.policy.min_free_percent or 0) > 0
            or float(self.policy.min_free_gib or 0) > 0
        )

    def check(self, paths: Iterable[str]) -> bool:
        self.last_checked_at = time.time()
        if not self.enabled:
            self.paused = False
            self.healthy_polls = 0
            self.last_snapshots = []
            self.last_blockers = []
            return True

        snapshots = [self._snapshot(existing_capacity_path(path)) for path in paths]
        self.last_snapshots = snapshots
        thresholds = self._thresholds(resume=self.paused)
        unhealthy = [
            snapshot for snapshot in snapshots
            if not snapshot.healthy
            or (
                thresholds[0] > 0
                and snapshot.free_percent < thresholds[0]
            )
            or (
                thresholds[1] > 0
                and snapshot.free_bytes < thresholds[1] * _GIB
            )
        ]
        if unhealthy:
            self.paused = True
            self.healthy_polls = 0
            reason = "; ".join(
                f"{item.path}: {item.reason or f'{item.free_percent:.1f}%/{item.free_bytes / _GIB:.1f}GiB free'}"
                for item in unhealthy
            )
            log.warning("Dispatch capacity paused for %s: %s", self.name, reason)
            return False

        if self.paused:
            self.healthy_polls += 1
            if self.healthy_polls < 2:
                return False
            self.paused = False
            self.healthy_polls = 0
            log.info("Dispatch capacity resumed for %s", self.name)
        return True

    def _thresholds(self, *, resume: bool) -> tuple[float, float]:
        percent = float(self.policy.min_free_percent or 0)
        gib = float(self.policy.min_free_gib or 0)
        if resume:
            # An unset (0) resume threshold must not resume onto a still-full FS.
            percent = max(percent, float(self.policy.resume_free_percent or 0))
            gib = max(gib, float(self.policy.resume_free_gib or 0))
        return percent, gib

    @staticmethod
    def _snapshot(path: str) -> CapacitySnapshot:
        try:
            stats = os.statvfs(path)
            total = int(stats.f_blocks * stats.f_frsize)
            free = int(stats.f_bavail * stats.f_frsize)
            percent = (free / total * 100.0) if total else 0.0
            if free <= 0:
                return CapacitySnapshot(path, total, free, percent, False, "no free blocks")
            probe = tempfile.NamedTemporaryFile(
                mode="wb", prefix=".mussel-capacity-", dir=path, delete=False
            )
            probe_path = probe.name
            try:
                probe.write(b"1")
                probe.flush()
                os.fsync(probe.fileno())
            finally:
                probe.close()
                try:
                    os.unlink(probe_path)
                except FileNotFoundError:
                    pass
            return CapacitySnapshot(path, total, free, percent)
        except OSError as exc:
            reason = "quota/full" if exc.errno in {errno.ENOSPC, errno.EDQUOT} else str(exc)
            return CapacitySnapshot(path, healthy=False, reason=reason)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paused": self.paused,
            "available": bool(self.enabled and not self.paused) if self.enabled else True,
            "checked_at_epoch": self.last_checked_at or None,
            "thresholds": {"min_free_percent": float(self.policy.min_free_percent or 0), "min_free_gib": float(self.policy.min_free_gib or 0), "resume_free_percent": float(self.policy.resume_free_percent or 0), "resume_free_gib": float(self.policy.resume_free_gib or 0)},
            "blockers": list(self.last_blockers),
            "healthy_polls": self.healthy_polls,
            "paths": [
                {
                    "path": item.path,
                    "total_bytes": item.total_bytes,
                    "free_bytes": item.free_bytes,
                    "free_percent": round(item.free_percent, 2),
                    "healthy": item.healthy,
                    "reason": item.reason,
                }
                for item in self.last_snapshots
            ],
        }
