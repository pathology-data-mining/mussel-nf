"""In-memory Tower API shim for the mussel-nf dispatcher.

When Nextflow is run with ``-with-tower`` and
``TOWER_API_ENDPOINT=http://localhost:<port>``, it sends structured JSON
events to the Tower trace API.  This module implements an in-memory store
that accumulates those events so the dashboard can read real-time progress
for each running batch without parsing log files.

Tower protocol (endpoints the dispatcher server must handle):

    GET  /user-info                     <- auth check; return stub user
    POST /trace/create                  <- workflow start; return {workflowId}
    PUT  /trace/{workflowId}/begin      <- workflow running
    PUT  /trace/{workflowId}/progress   <- periodic task counts + per-task data
    PUT  /trace/{workflowId}/heartbeat  <- keepalive with progress
    PUT  /trace/{workflowId}/complete   <- workflow finished

The ``progress`` object in each payload:
    succeeded, failed, ignored, cached, pending, submitted, running,
    retries, aborted, loadCpus, loadMemory, peakCpus, peakMemory,
    peakRunning, processes (list of per-process ProgressRecord)

Each element of ``processes``:
    process, pending, submitted, running, succeeded, failed, cached, ...

Each task in ``tasks[]`` (/trace/progress only):
    taskId, status, hash, name, process, tag, exit, start, complete, ...

Thread safety: all mutations go through a single threading.Lock.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
_lock = threading.Lock()
_workflows: dict[str, "WorkflowState"] = {}

_MAX_AGE_SECONDS = 3600
_STALE_SECONDS   = 5 * 60


def _task_counts_from_progress(p: dict) -> dict:
    return {
        "succeeded": p.get("succeeded", 0),
        "failed":    p.get("failed", 0),
        "cached":    p.get("cached", 0),
        "running":   p.get("running", 0),
        "pending":   p.get("pending", 0),
        "submitted": p.get("submitted", 0),
        "aborted":   p.get("aborted", 0),
    }


class WorkflowState:
    __slots__ = (
        "workflow_id", "batch_id", "run_name",
        "task_counts", "processes", "resources",
        "failures", "complete",
        "started_at", "updated_at",
    )

    def __init__(self, workflow_id: str, batch_id: str, run_name: str):
        self.workflow_id  = workflow_id
        self.batch_id     = batch_id
        self.run_name     = run_name
        self.task_counts: dict       = {}
        self.processes:   list[dict] = []   # per-process ProgressRecord list
        self.resources:   dict       = {}   # loadCpus, loadMemory, peakMemory, …
        self.failures:    list[dict] = []   # accumulated failed tasks (max 50)
        self.complete     = False
        self.started_at   = time.time()
        self.updated_at   = time.time()

    def _ingest(self, progress: dict, tasks: list[dict] | None = None) -> None:
        """Update state from a progress payload.  Must be called under _lock."""
        self.task_counts = _task_counts_from_progress(progress)
        self.processes   = progress.get("processes") or []
        self.resources   = {
            k: progress[k]
            for k in ("loadCpus", "loadMemory", "peakCpus", "peakMemory", "peakRunning")
            if progress.get(k) is not None
        }
        if tasks:
            seen = {f["taskId"] for f in self.failures if "taskId" in f}
            for t in tasks:
                if (t.get("status") or "").upper() == "FAILED" and t.get("taskId") not in seen:
                    self.failures.append({
                        "taskId":  t.get("taskId"),
                        "process": t.get("process", ""),
                        "name":    t.get("name", ""),
                        "tag":     t.get("tag"),
                        "exit":    t.get("exit"),
                        "hash":    t.get("hash"),
                    })
            if len(self.failures) > 50:
                self.failures = self.failures[-50:]
        self.updated_at = time.time()

    def is_stalled(self) -> bool:
        return not self.complete and (time.time() - self.updated_at) > _STALE_SECONDS

    def as_dict(self) -> dict:
        done  = self.task_counts.get("succeeded", 0) + self.task_counts.get("cached", 0)
        total = done + sum(
            self.task_counts.get(k, 0)
            for k in ("failed", "running", "pending", "submitted")
        )
        return {
            "workflow_id":  self.workflow_id,
            "batch_id":     self.batch_id,
            "run_name":     self.run_name,
            "task_counts":  self.task_counts,
            "processes":    self.processes,
            "resources":    self.resources,
            "failures":     self.failures,
            "complete":     self.complete,
            "stalled":      self.is_stalled(),
            "done":         done,
            "total":        total,
            "pct":          round(done / total * 100) if total else 0,
            "started_at":   self.started_at,
            "updated_at":   self.updated_at,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_workflow(workflow_id: str, batch_id: str, run_name: str) -> None:
    with _lock:
        _workflows[workflow_id] = WorkflowState(workflow_id, batch_id, run_name)


def update_progress(workflow_id: str, progress: dict,
                    tasks: list[dict] | None = None) -> None:
    with _lock:
        state = _workflows.get(workflow_id)
        if state is None:
            return
        state._ingest(progress, tasks)


def mark_complete(workflow_id: str, progress: dict | None = None) -> None:
    with _lock:
        state = _workflows.get(workflow_id)
        if state is None:
            return
        state.complete = True
        if progress:
            state._ingest(progress)
        else:
            state.updated_at = time.time()


def get_progress(batch_id: str) -> Optional[dict]:
    """Return full state dict for a batch, or None if not known via Tower."""
    with _lock:
        state = _by_batch_id(batch_id)
        return state.as_dict() if state else None


def get_state(batch_id: str) -> Optional[dict]:
    return get_progress(batch_id)


def get_all_states() -> list[dict]:
    with _lock:
        return [s.as_dict() for s in _workflows.values()]


def evict_old(max_age_seconds: float = _MAX_AGE_SECONDS) -> int:
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
    return f"dispatcher_{batch_id}"


# ---------------------------------------------------------------------------

def _by_batch_id(batch_id: str) -> Optional[WorkflowState]:
    canonical = workflow_id_for_batch(batch_id)
    state = _workflows.get(canonical)
    if state:
        return state
    for s in _workflows.values():
        if s.batch_id == batch_id:
            return s
    return None


# ---------------------------------------------------------------------------
# HTTP stub responses
# ---------------------------------------------------------------------------

def user_info_response() -> dict:
    return {
        "user": {
            "id": 1, "userName": "dispatcher",
            "email": "dispatcher@localhost",
            "firstName": "Mussel", "lastName": "Dispatcher",
            "avatar": None, "organization": None, "description": None,
        }
    }


def trace_create_response(workflow_id: str) -> dict:
    return {"workflowId": workflow_id, "watchUrl": None, "message": None, "metadata": None}
