"""Shared normalization and matching for persisted Tower workflow state."""
from __future__ import annotations

import time
from typing import Iterable, Mapping


_SENTINEL_IGNORED_EXIT = 2147483647


def _coerce_nonnegative_int(value) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def normalize_tower_state(row: Mapping, *, now: float | None = None) -> dict:
    """Add dashboard progress fields without mutating the persisted row."""
    raw_task_counts = row.get("task_counts") or {}
    raw_processes = row.get("processes") or []
    raw_failures = row.get("failures") or []

    dropped_failures_by_process: dict[str, int] = {}
    failures = []
    for failure in raw_failures:
        item = dict(failure)
        try:
            exit_code = int(item.get("exit"))
        except Exception:
            exit_code = None
        if exit_code == _SENTINEL_IGNORED_EXIT:
            proc_name = str(item.get("process") or "")
            dropped_failures_by_process[proc_name] = dropped_failures_by_process.get(proc_name, 0) + 1
            continue
        failures.append(item)

    processes = []
    for process in raw_processes:
        item = dict(process)
        for key in ("pending", "submitted", "running", "succeeded", "cached", "failed", "aborted", "stored", "ignored", "retries"):
            if key in item:
                item[key] = _coerce_nonnegative_int(item.get(key))
        proc_name = str(item.get("name") or item.get("process") or "")
        dropped = dropped_failures_by_process.get(proc_name, 0)
        if dropped:
            item["failed"] = max(0, int(item.get("failed", 0) or 0) - dropped)
        for key in ("loadCpus", "loadMemory", "peakRunning", "peakCpus", "peakMemory"):
            if key in item:
                try:
                    item[key] = max(0, int(item.get(key) or 0))
                except Exception:
                    pass
        processes.append(item)

    task_counts = {
        "succeeded": _coerce_nonnegative_int(raw_task_counts.get("succeeded")),
        "failed": max(0, _coerce_nonnegative_int(raw_task_counts.get("failed")) - sum(dropped_failures_by_process.values())),
        "cached": _coerce_nonnegative_int(raw_task_counts.get("cached")),
        "running": _coerce_nonnegative_int(raw_task_counts.get("running")),
        "pending": _coerce_nonnegative_int(raw_task_counts.get("pending")),
        "submitted": _coerce_nonnegative_int(raw_task_counts.get("submitted")),
        "aborted": _coerce_nonnegative_int(raw_task_counts.get("aborted")),
    }

    done = task_counts["succeeded"] + task_counts["cached"]
    total = done + sum(
        task_counts.get(key, 0)
        for key in ("failed", "running", "pending", "submitted")
    )
    current_time = time.time() if now is None else now
    state = dict(row)
    state["task_counts"] = task_counts
    state["processes"] = processes
    state["failures"] = failures
    # A dispatcher snapshot may already contain an explicitly computed
    # liveness value. Preserve it; otherwise derive it for live registry rows.
    if "stalled" not in state:
        state["stalled"] = (
            not bool(state.get("complete"))
            and current_time - float(state.get("updated_at") or 0.0) > 5 * 60
        )
    state["done"] = done
    state["total"] = total
    state["pct"] = round(done / total * 100) if total else 0
    return state


def newest_tower_state_for_batch(
    batch_id: str,
    tower_states: Iterable[Mapping],
    *,
    active_runs: Mapping[str, str] | None = None,
) -> dict | None:
    """Return the freshest Tower state matched to a dispatcher batch."""
    active_runs = active_runs or {}
    batch_suffix = batch_id.rsplit("_", 1)[-1]
    short_batch_ids = {batch_suffix, f"r{batch_suffix}"}
    matches: list[Mapping] = []
    for state in tower_states:
        if state.get("batch_id") == batch_id:
            matches.append(state)
            continue
        if state.get("batch_id") in short_batch_ids:
            matches.append(state)
            continue
        run_name = state.get("run_name") or ""
        if active_runs.get(run_name) == batch_id or run_name in short_batch_ids:
            matches.append(state)
    if not matches:
        return None
    return dict(max(matches, key=lambda state: state.get("updated_at") or 0.0))
