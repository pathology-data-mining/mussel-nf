"""Backward-compatibility shim — implementation moved to nextflow-turret.

All public symbols are re-exported from :mod:`nextflow_turret` so that
existing imports (``from mussel_dispatcher import tower_shim as _tower_shim``)
continue to work unchanged.
"""
from nextflow_turret import (  # noqa: F401  (re-exports)
    WorkflowState,
    WorkflowRegistry,
    default_registry,
    workflow_id_for_batch,
    register_workflow,
    is_registered,
    update_progress,
    mark_complete,
    get_progress,
    get_state,
    get_all_states,
    evict_old,
    user_info_response,
    trace_create_response,
)

# Expose internal lock and dict so existing tests can reset state between runs.
_lock      = default_registry._lock
_workflows = default_registry._workflows
