from mussel_dispatcher.config import StoragePolicyConfig
from mussel_dispatcher.storage import (
    CapacityGuard,
    CapacitySnapshot,
    ExecutionArtifactCleaner,
    ExecutionCleanupSpec,
)
from mussel_dispatcher.state import StateStore


def test_shared_cleaner_removes_only_execution_artifacts(tmp_path):
    managed = tmp_path / "work"
    run = managed / "batch-1"
    disposable = run / "work"
    results = tmp_path / "results"
    disposable.mkdir(parents=True)
    results.mkdir()
    (disposable / "large.bin").write_bytes(b"payload")
    (results / "keep.pt").write_bytes(b"published")

    result = ExecutionArtifactCleaner().cleanup(
        ExecutionCleanupSpec(
            batch_id="batch-1",
            managed_root=str(managed),
            run_root=str(run),
            execution_paths=(str(disposable),),
            protected_paths=(str(results),),
        )
    )

    assert result.succeeded
    assert not disposable.exists()
    assert (results / "keep.pt").exists()


def test_shared_cleaner_rejects_path_escape_and_symlink(tmp_path):
    managed = tmp_path / "work"
    run = managed / "batch-1"
    run.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = run / "link"
    link.symlink_to(outside, target_is_directory=True)

    result = ExecutionArtifactCleaner().cleanup(
        ExecutionCleanupSpec(
            batch_id="batch-1",
            managed_root=str(managed),
            run_root=str(run),
            execution_paths=(str(link),),
        )
    )

    assert not result.succeeded
    assert outside.exists()


def test_capacity_guard_requires_two_healthy_polls_to_resume(monkeypatch, tmp_path):
    guard = CapacityGuard(
        StoragePolicyConfig(
            min_free_percent=10,
            min_free_gib=1,
            resume_free_percent=12,
            resume_free_gib=2,
        ),
        name="test",
    )
    snapshots = iter([
        CapacitySnapshot(str(tmp_path), 100 * 1024**3, 5 * 1024**3, 5, True),
        CapacitySnapshot(str(tmp_path), 100 * 1024**3, 20 * 1024**3, 20, True),
        CapacitySnapshot(str(tmp_path), 100 * 1024**3, 20 * 1024**3, 20, True),
    ])
    monkeypatch.setattr(guard, "_snapshot", lambda path: next(snapshots))

    assert not guard.check([str(tmp_path)])
    assert not guard.check([str(tmp_path)])
    assert guard.check([str(tmp_path)])


def test_cleanup_marker_is_independent_of_finalization(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    work_dir = tmp_path / "work" / "batch-1"
    state.add_batch(
        "batch-1",
        str(tmp_path / "batch.csv"),
        str(work_dir),
        1,
        str(tmp_path / "batch.log"),
    )
    state.mark_driver_terminal("batch-1", 0)
    state.set_batch_status("batch-1", "FINALIZING")

    candidates = state.get_execution_cleanup_candidates()
    assert [row["batch_id"] for row in candidates] == ["batch-1"]
    state.record_execution_cleanup("batch-1")
    assert state.get_execution_cleanup_candidates() == []

def test_remote_execution_cleanup_excludes_finalization_batches(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    statuses = (
        "FINALIZATION_QUEUED",
        "HOOKS_RUNNING",
        "HOOKS_RETRY",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    )
    for status in statuses:
        batch_id = f"batch-{status.lower()}"
        state.add_batch(
            batch_id,
            str(tmp_path / f"{batch_id}.csv"),
            str(tmp_path / "work" / batch_id),
            1,
            str(tmp_path / f"{batch_id}.log"),
            backend="remote_agent",
            target="hpc",
        )
        state.mark_driver_terminal(batch_id, 0)
        state.write_remote_status(
            batch_id,
            {
                "status": status,
                "remote_work_dir": str(tmp_path / "work" / batch_id),
                "remote_outdir": str(tmp_path / "out" / batch_id),
            },
        )

    candidates = state.get_remote_execution_cleanup_batches("hpc")
    assert [row["batch_id"] for row in candidates] == [
        "batch-succeeded",
        "batch-failed",
        "batch-cancelled",
    ]

def test_hook_executor_job_id_is_persisted_for_active_attempt(tmp_path):
    state = StateStore(str(tmp_path / "state.db"))
    state.add_batch(
        "batch-1",
        str(tmp_path / "batch.csv"),
        str(tmp_path / "work"),
        1,
        str(tmp_path / "batch.log"),
        backend="remote_agent",
        target="hpc",
    )
    state.sync_batch_hooks(
        "batch-1",
        [{
            "hook_index": 0,
            "phase": "wds",
            "hook_name": "wds",
            "command": "echo ok",
            "required": True,
        }],
    )
    attempt = state.start_batch_hook("batch-1", 0)
    assert attempt == 1

    state.set_batch_hook_executor(
        "batch-1",
        0,
        job_id="12345",
        state="SUBMITTED",
        expected_attempt_number=attempt,
    )
    row = state.get_batch_hooks(["batch-1"])["batch-1"][0]
    assert row["executor_job_id"] == "12345"
    assert row["executor_state"] == "SUBMITTED"

    state.set_batch_hook_executor(
        "batch-1",
        0,
        job_id="99999",
        state="COMPLETED",
        expected_attempt_number=attempt,
    )
    row = state.get_batch_hooks(["batch-1"])["batch-1"][0]
    assert row["executor_job_id"] == "12345"
    assert row["executor_state"] == "SUBMITTED"

    state.set_batch_hook_executor(
        "batch-1",
        0,
        job_id="12345",
        state="COMPLETED",
        expected_attempt_number=attempt,
    )
    row = state.get_batch_hooks(["batch-1"])["batch-1"][0]
    assert row["executor_job_id"] == "12345"
    assert row["executor_state"] == "COMPLETED"
