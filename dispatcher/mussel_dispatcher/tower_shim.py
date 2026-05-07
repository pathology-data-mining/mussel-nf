"""In-memory Tower API shim for the mussel-nf dispatcher.

When Nextflow is run with ``-with-tower`` and
``TOWER_API_ENDPOINT=http://localhost:<port>``, it sends structured JSON
events to the Tower trace API.  This module implements an in-memory store
that accumulates those events so the dashboard can read real-time progress
for each running batch without parsing log files.

Tower protocol (endpoints the dispatcher server must handle):

    GET  /user-info                     ← auth check; return stub user
    POST /trace/create                  ← workflow start; return {workflowId}
    PUT  /trace/{workflowId}/begin      ← workflow running
    PUT  /trace/{workflowId}/progress   ← periodic task counts + per-task data
    PUT  /trace/{workflowId}/heartbeat  ← keepalive with progress
    PUT  /trace/{workflowId}/complete   ← workflow finished

The workflowId returned by ``/trace/create`` is set to the batch_id parsed
from the NF run name (``dispatcher_{batch_id}``), making the mapping
transparent without any separate lookup.

The ``progress`` object in each heartbeat/progress payload has these int
fields (all derived from ``WorkflowStats`` in the NF core):

    succeeded, failed, ignored, cached,
    pending, submitted, running, retries, aborted

Thread safety: all mutations go through a single ``threading.Lock``.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# workflowId → WorkflowState dict
_workflows: dict[str, dict] = {}

# Maximum age (seconds) before a completed workflow is eligible for eviction.
_MAX_AGE_SECONDS = 3600  # keep 1 hour of history


class WorkflowState:
    """Holds the latest known state for one NF workflow (= one dispatcher batch)."""

    __slots__ = (
        "workflow_id", "batch_id", "run_name",
        "progress", "task_counts", "complete",
        "started_at", "updated_at",
    )

    def __init__(self, workflow_id: str, batch_id: str, run_name: str):
        self.workflow_id = workflow_id
        self.batch_id = batch_id
        self.run_name = run_name
        self.progress: dict = {}        # latest WorkflowProgress from NF
        self.task_counts: dict = {}     # computed from progress
        self.complete = False
        self.started_at = time.time()
        self.updated_at = time.time()

    def as_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "batch_id": self.batch_id,
            "run_name": self.run_name,
            "progress": self.progress,
            "complete": self.complete,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_workflow(workflow_id: str, batch_id: str, run_name: str) -> None:
    """Called when /trace/create is received.  Creates a new WorkflowState."""
    with _lock:
        _workflows[workflow_id] = WorkflowState(workflow_id, batch_id, run_name)


def update_progress(workflow_id: str, progress: dict) -> None:
    """Called on /trace/progress and /trace/heartbeat.

    ``progress`` is the ``progress`` sub-object from the NF payload, e.g.:
        {"succeeded": 5, "failed": 0, "cached": 2, "running": 3, ...}
    """
    with _lock:
        state = _workflows.get(workflow_id)
        if state is None:
            return
        state.progress = progress
        state.updated_at = time.time()
        # Pre-compute friendly task count dict for the dashboard.
        state.task_counts = {
            "succeeded": progress.get("succeeded", 0),
            "failed":    progress.get("failed", 0),
            "cached":    progress.get("cached", 0),
            "running":   progress.get("running", 0),
            "pending":   progress.get("pending", 0),
            "submitted": progress.get("submitted", 0),
            "aborted":   progress.get("aborted", 0),
        }


def mark_complete(workflow_id: str, progress: dict | None = None) -> None:
    """Called on /trace/complete.  Optionally records final progress."""
    with _lock:
        state = _workflows.get(workflow_id)
        if state is None:
            return
        state.complete = True
        state.updated_at = time.time()
        if progress:
            state.progress = progress
            state.task_counts = {
                "succeeded": progress.get("succeeded", 0),
                "failed":    progress.get("failed", 0),
                "cached":    progress.get("cached", 0),
                "running":   progress.get("running", 0),
                "pending":   progress.get("pending", 0),
                "submitted": progress.get("submitted", 0),
                "aborted":   progress.get("aborted", 0),
            }


def get_progress(batch_id: str) -> Optional[dict]:
    """Return the latest task count dict for a batch, or None if not known."""
    with _lock:
        state = _by_batch_id(batch_id)
        if state is None:
            return None
        return dict(state.task_counts) if state.task_counts else None


def get_state(batch_id: str) -> Optional[dict]:
    """Return the full WorkflowState dict for a batch, or None."""
    with _lock:
        state = _by_batch_id(batch_id)
        return state.as_dict() if state else None


def get_all_states() -> list[dict]:
    """Return a snapshot of all tracked workflow states (for debugging)."""
    with _lock:
        return [s.as_dict() for s in _workflows.values()]


def evict_old(max_age_seconds: float = _MAX_AGE_SECONDS) -> int:
    """Remove completed workflows older than *max_age_seconds*.

    Returns the number of entries removed.
    """
    cutoff = time.time() - max_age_seconds
    with _lock:
        to_remove = [
            wid for wid, s in _workflows.items()
            if s.complete and s.updated_at < cutoff
        ]
        for wid in to_remove:
            del _workflows[wid]
    return len(to_remove)


def workflow_id_for_batch(batch_id: str) -> str:
    """Canonical workflowId for a given batch_id (mirrors what /trace/create returns)."""
    return f"dispatcher_{batch_id}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _by_batch_id(batch_id: str) -> Optional[WorkflowState]:
    """Return the WorkflowState for *batch_id* (must be called under lock)."""
    # Primary lookup: workflowId = dispatcher_{batch_id}
    canonical = workflow_id_for_batch(batch_id)
    state = _workflows.get(canonical)
    if state:
        return state
    # Fallback: linear scan in case of custom workflowId
    for s in _workflows.values():
        if s.batch_id == batch_id:
            return s
    return None


# ---------------------------------------------------------------------------
# HTTP stub responses (used by server.py)
# ---------------------------------------------------------------------------

def user_info_response() -> dict:
    """Stub /user-info response that satisfies NF's auth check."""
    return {
        "user": {
            "id": 1,
            "userName": "dispatcher",
            "email": "dispatcher@localhost",
            "firstName": "Mussel",
            "lastName": "Dispatcher",
            "avatar": None,
            "organization": None,
            "description": None,
        }
    }


def trace_create_response(workflow_id: str) -> dict:
    """Response to POST /trace/create — NF uses the returned workflowId for all
    subsequent calls."""
    return {
        "workflowId": workflow_id,
        "watchUrl": None,
        "message": None,
        "metadata": None,
    }
