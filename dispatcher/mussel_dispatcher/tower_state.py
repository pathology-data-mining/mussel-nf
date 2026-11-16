"""Shared normalization and matching for persisted Tower workflow state."""
from __future__ import annotations

from typing import Iterable, Mapping


def _nonnegative_int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def normalize_tower_state(row: Mapping, *, now: float | None = None) -> dict:
    """Add dashboard progress fields without mutating the persisted row.

    Delegates to :func:`nextflow_turret.normalize_workflow_state`, which keeps
    Nextflow's raw ``failed`` count (it includes tasks whose errorStrategy
    ignored them) and derives ``failed_not_ignored`` (``failedNotIgnored`` per
    process) from Nextflow's own ``ignored`` counter.  Exit 2147483647 alone
    says nothing: Nextflow also reports it for Slurm-killed or lost jobs.

    Rows persisted by turret before 0.2.0 carry no task-level ``ignored``;
    derive it from the per-process counters Nextflow reported alongside.
    """
    # Imported here: state.py loads this module where turret is optional.
    from nextflow_turret import normalize_workflow_state

    state = normalize_workflow_state(row, now=now)
    counts = state["task_counts"]
    process_ignored = sum(_nonnegative_int(p.get("ignored")) for p in state["processes"])
    if process_ignored > counts["ignored"]:
        counts["ignored"] = process_ignored
        state["failed_not_ignored"] = max(0, counts["failed"] - process_ignored)
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
