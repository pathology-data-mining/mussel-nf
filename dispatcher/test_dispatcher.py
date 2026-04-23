"""Tests for mussel-dispatcher.py"""

import csv
import importlib.util
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the dispatcher module (filename contains a hyphen)
# ---------------------------------------------------------------------------

_DISPATCHER_PY = os.path.join(os.path.dirname(__file__), "mussel-dispatcher.py")
_spec = importlib.util.spec_from_file_location("mussel_dispatcher", _DISPATCHER_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Config = _mod.Config
WatcherConfig = _mod.WatcherConfig
StateStore = _mod.StateStore
ReadinessChecker = _mod.ReadinessChecker
LocalWatcher = _mod.LocalWatcher
collect_manifests = _mod.collect_manifests
BatchScheduler = _mod.BatchScheduler
NextflowRunner = _mod.NextflowRunner
TcgaWatcher = _mod.TcgaWatcher
recover_in_flight = _mod.recover_in_flight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides):
    defaults = dict(
        repo_dir="/tmp/repo",
        nextflow_profiles="standard",
        outdir="/tmp/results",
        work_base_dir="/tmp/work",
        dispatch_dir="/tmp/dispatch",
        state_dir="/tmp/state",
        log_dir="/tmp/logs",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ===========================================================================
# Config
# ===========================================================================

class TestConfig:
    def test_resolved_combined_manifest_path_default(self):
        cfg = make_config(outdir="/results")
        assert cfg.resolved_combined_manifest_path() == "/results/manifest-combined.csv"

    def test_resolved_combined_manifest_path_custom(self):
        cfg = make_config(combined_manifest_path="/custom/combined.csv")
        assert cfg.resolved_combined_manifest_path() == "/custom/combined.csv"

    def test_load_from_yaml(self, tmp_path):
        data = {
            "repo_dir": "/repo",
            "nextflow_profiles": "cluster,conda",
            "outdir": "/out",
            "work_base_dir": "/work",
            "dispatch_dir": "/dispatch",
            "state_dir": "/state",
            "log_dir": "/logs",
            "batch_size": 10,
            "max_wait_seconds": 120,
            "watchers": [
                {
                    "type": "local",
                    "path": "/incoming",
                    "poll_interval_seconds": 30,
                }
            ],
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(data))
        cfg = Config.load(str(yaml_path))
        assert cfg.repo_dir == "/repo"
        assert cfg.batch_size == 10
        assert cfg.max_wait_seconds == 120
        assert len(cfg.watchers) == 1
        assert cfg.watchers[0].type == "local"
        assert cfg.watchers[0].path == "/incoming"


# ===========================================================================
# StateStore
# ===========================================================================

class TestStateStore:
    @pytest.fixture
    def store(self, tmp_path):
        return StateStore(str(tmp_path / "test.db"))

    def test_add_and_is_known(self, store):
        assert store.is_known("/slides/a.svs") is False
        store.add_slide("/slides/a.svs", "a")
        assert store.is_known("/slides/a.svs") is True

    def test_add_slide_idempotent(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.add_slide("/slides/a.svs", "a")  # Should not raise
        assert store.is_known("/slides/a.svs") is True

    def test_get_pending_slides(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.add_slide("/slides/b.svs", "b")
        pending = store.get_pending_slides()
        paths = {r["slide_path"] for r in pending}
        assert paths == {"/slides/a.svs", "/slides/b.svs"}

    def test_mark_dispatched_removes_from_pending(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        assert store.get_pending_slides() == []

    def test_mark_slides_complete_succeeded(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.mark_slides_complete("batch-001", succeeded=True)
        row = store._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"

    def test_mark_slides_complete_failed(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.mark_slides_complete("batch-001", succeeded=False)
        row = store._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "FAILED"

    def test_reset_dispatched_to_pending(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.reset_dispatched_to_pending("batch-001")
        pending = store.get_pending_slides()
        assert len(pending) == 1
        assert pending[0]["slide_path"] == "/slides/a.svs"

    def test_reset_dispatched_only_affects_target_batch(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.add_slide("/slides/b.svs", "b")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.mark_dispatched(["/slides/b.svs"], "batch-002")
        store.reset_dispatched_to_pending("batch-001")
        pending = store.get_pending_slides()
        assert len(pending) == 1
        assert pending[0]["slide_path"] == "/slides/a.svs"

    def test_batch_lifecycle_running_to_succeeded(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", 5, "/logs/1.log")
        assert len(store.get_running_batches()) == 1
        store.complete_batch("batch-001", exit_code=0)
        assert store.get_running_batches() == []
        row = store._conn().execute(
            "SELECT status FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"

    def test_complete_batch_failed_sets_status_and_exit(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", 2, "/logs/1.log")
        store.complete_batch("batch-001", exit_code=1)
        row = store._conn().execute(
            "SELECT status, nextflow_exit FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["nextflow_exit"] == 1

    def test_record_batch_manifest(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", 1, "/logs/1.log")
        store.record_batch_manifest("batch-001", "/results/manifest-001.csv")
        paths = store.get_all_manifest_paths()
        assert "/results/manifest-001.csv" in paths

    def test_get_all_manifest_paths_skips_null(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", 1, "/logs/1.log")
        assert store.get_all_manifest_paths() == []

    def test_get_running_batches_only_returns_running(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", 1, "/logs/1.log")
        store.add_batch("batch-002", "/dispatch/2.csv", 1, "/logs/2.log")
        store.complete_batch("batch-001", 0)
        running = store.get_running_batches()
        assert len(running) == 1
        assert running[0]["batch_id"] == "batch-002"


# ===========================================================================
# ReadinessChecker
# ===========================================================================

class TestReadinessChecker:
    def test_skip_part_extension(self, tmp_path):
        f = tmp_path / "slide.svs.part"
        f.write_bytes(b"x" * 20_000)
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=100)
        assert checker.is_ready(str(f)) is False

    def test_skip_tmp_extension(self, tmp_path):
        f = tmp_path / "slide.tmp"
        f.write_bytes(b"x" * 20_000)
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=100)
        assert checker.is_ready(str(f)) is False

    def test_skip_hidden_file(self, tmp_path):
        f = tmp_path / ".hidden.svs"
        f.write_bytes(b"x" * 20_000)
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=100)
        assert checker.is_ready(str(f)) is False

    def test_skip_too_small(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 50)
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=1_000)
        assert checker.is_ready(str(f)) is False

    def test_first_poll_always_returns_false(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 2_000)
        checker = ReadinessChecker(stability_wait=10, min_size_bytes=100)
        assert checker.is_ready(str(f)) is False

    def test_stable_returns_true_after_wait(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 2_000)
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=100)
        checker.is_ready(str(f))  # first poll: records snapshot
        assert checker.is_ready(str(f)) is True  # second poll: elapsed >= 0

    def test_resets_snapshot_when_size_changes(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 2_000)
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=100)
        checker.is_ready(str(f))
        f.write_bytes(b"x" * 3_000)  # size changed
        assert checker.is_ready(str(f)) is False

    def test_returns_false_for_nonexistent_file(self):
        checker = ReadinessChecker(stability_wait=0, min_size_bytes=100)
        assert checker.is_ready("/nonexistent/slide.svs") is False

    def test_returns_false_before_stability_wait_elapses(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 2_000)
        checker = ReadinessChecker(stability_wait=9_999, min_size_bytes=100)
        checker.is_ready(str(f))  # first poll
        assert checker.is_ready(str(f)) is False  # not enough time elapsed


# ===========================================================================
# LocalWatcher._scan
# ===========================================================================

class TestLocalWatcher:
    def _make_watcher(self, tmp_path, recursive=True, state=None, pending=None):
        if state is None:
            state = StateStore(str(tmp_path / "test.db"))
        if pending is None:
            pending = deque()
        cfg = WatcherConfig(
            type="local",
            path=str(tmp_path),
            recursive=recursive,
            stability_wait_seconds=0,
            min_file_size_mb=0.0,
            extensions=[".svs", ".tiff"],
        )
        watcher = LocalWatcher(cfg, pending, state, threading.Event())
        watcher.checker = ReadinessChecker(stability_wait=0, min_size_bytes=0)
        return watcher, state, pending

    def test_ignores_wrong_extension(self, tmp_path):
        (tmp_path / "slide.jpg").write_bytes(b"x" * 100)
        watcher, _, pending = self._make_watcher(tmp_path)
        watcher._scan(str(tmp_path))
        assert len(pending) == 0

    def test_ignores_known_slides(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 100)
        watcher, state, pending = self._make_watcher(tmp_path)
        state.add_slide(str(f), "slide")
        watcher._scan(str(tmp_path))
        assert len(pending) == 0

    def test_adds_ready_slide_to_pending(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 100)
        watcher, _, pending = self._make_watcher(tmp_path)
        watcher.checker.is_ready(str(f))  # prime snapshot
        watcher._scan(str(tmp_path))
        assert len(pending) == 1
        assert pending[0]["slide_path"] == str(f)
        assert pending[0]["slide_id"] == "slide"

    def test_slide_id_is_stem(self, tmp_path):
        f = tmp_path / "TCGA-AB-1234.svs"
        f.write_bytes(b"x" * 100)
        watcher, _, pending = self._make_watcher(tmp_path)
        watcher.checker.is_ready(str(f))
        watcher._scan(str(tmp_path))
        assert pending[0]["slide_id"] == "TCGA-AB-1234"

    def test_scans_recursively(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        f = subdir / "slide.svs"
        f.write_bytes(b"x" * 100)
        watcher, _, pending = self._make_watcher(tmp_path, recursive=True)
        watcher.checker.is_ready(str(f))
        watcher._scan(str(tmp_path))
        assert len(pending) == 1
        assert pending[0]["slide_path"] == str(f)

    def test_non_recursive_ignores_subdirectories(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        f = subdir / "slide.svs"
        f.write_bytes(b"x" * 100)
        watcher, _, pending = self._make_watcher(tmp_path, recursive=False)
        watcher.checker.is_ready(str(f))
        watcher._scan(str(tmp_path))
        assert len(pending) == 0

    def test_not_ready_slide_stays_off_pending(self, tmp_path):
        f = tmp_path / "slide.svs"
        f.write_bytes(b"x" * 100)
        watcher, _, pending = self._make_watcher(tmp_path)
        # No prior poll — checker will return False
        watcher._scan(str(tmp_path))
        assert len(pending) == 0


# ===========================================================================
# collect_manifests
# ===========================================================================

class TestCollectManifests:
    def test_no_manifests_returns_zero(self, tmp_path):
        n = collect_manifests(str(tmp_path), str(tmp_path / "combined.csv"))
        assert n == 0

    def test_merges_single_manifest(self, tmp_path):
        (tmp_path / "manifest-001.csv").write_text(
            "slide1,wf1,feature_path,/out/slide1.h5\n"
            "slide2,wf1,feature_path,/out/slide2.h5\n"
        )
        out_path = str(tmp_path / "combined.csv")
        n = collect_manifests(str(tmp_path), out_path)
        assert n == 2
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert {r["slide_id"] for r in rows} == {"slide1", "slide2"}

    def test_output_has_correct_header(self, tmp_path):
        (tmp_path / "manifest-001.csv").write_text("slide1,wf1,key,val\n")
        out_path = str(tmp_path / "combined.csv")
        collect_manifests(str(tmp_path), out_path)
        with open(out_path, newline="") as f:
            header = next(csv.reader(f))
        assert header == ["slide_id", "workflow_id", "key", "value"]

    def test_deduplication_last_write_wins(self, tmp_path):
        mf1 = tmp_path / "manifest-001.csv"
        mf2 = tmp_path / "manifest-002.csv"
        mf1.write_text("slide1,wf1,feature_path,/old.h5\n")
        mf2.write_text("slide1,wf2,feature_path,/new.h5\n")
        # Ensure mf2 has a newer mtime
        os.utime(str(mf2), (time.time() + 1, time.time() + 1))
        out_path = str(tmp_path / "combined.csv")
        n = collect_manifests(str(tmp_path), out_path)
        assert n == 1
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["value"] == "/new.h5"

    def test_different_keys_are_not_deduplicated(self, tmp_path):
        (tmp_path / "manifest-001.csv").write_text(
            "slide1,wf1,feature_path,/out/slide1.h5\n"
            "slide1,wf1,tile_path,/out/slide1.patch.h5\n"
        )
        out_path = str(tmp_path / "combined.csv")
        n = collect_manifests(str(tmp_path), out_path)
        assert n == 2

    def test_skips_malformed_lines(self, tmp_path):
        (tmp_path / "manifest-001.csv").write_text(
            "slide1,wf1,feature_path,/out/slide1.h5\n"
            "bad_line\n"
            "slide2,wf1,feature_path,/out/slide2.h5\n"
        )
        out_path = str(tmp_path / "combined.csv")
        n = collect_manifests(str(tmp_path), out_path)
        assert n == 2

    def test_skips_empty_lines(self, tmp_path):
        (tmp_path / "manifest-001.csv").write_text(
            "slide1,wf1,feature_path,/out/slide1.h5\n"
            "\n"
            "slide2,wf1,feature_path,/out/slide2.h5\n"
        )
        out_path = str(tmp_path / "combined.csv")
        n = collect_manifests(str(tmp_path), out_path)
        assert n == 2

    def test_creates_output_directory_if_needed(self, tmp_path):
        nested_out = str(tmp_path / "a" / "b" / "combined.csv")
        (tmp_path / "manifest-001.csv").write_text("s1,wf1,k,v\n")
        collect_manifests(str(tmp_path), nested_out)
        assert os.path.exists(nested_out)


# ===========================================================================
# BatchScheduler
# ===========================================================================

class TestBatchScheduler:
    def _make_scheduler(self, batch_size=3, min_batch_size=1, max_wait_seconds=9_999):
        cfg = make_config(
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            max_wait_seconds=max_wait_seconds,
        )
        run_manager = MagicMock()
        scheduler = BatchScheduler(cfg, MagicMock(), run_manager, threading.Event())
        return scheduler, run_manager

    def _slide(self, name):
        return {"slide_path": f"/slides/{name}.svs", "slide_id": name}

    def test_no_dispatch_when_empty(self):
        scheduler, run_manager = self._make_scheduler()
        scheduler._maybe_dispatch()
        run_manager.submit.assert_not_called()

    def test_size_trigger_dispatches_full_batch(self):
        scheduler, run_manager = self._make_scheduler(batch_size=2)
        scheduler.enqueue(self._slide("a"))
        scheduler.enqueue(self._slide("b"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_called_once()
        batch = run_manager.submit.call_args[0][1]
        assert len(batch) == 2

    def test_no_dispatch_below_batch_size_without_time_trigger(self):
        scheduler, run_manager = self._make_scheduler(batch_size=5, max_wait_seconds=9_999)
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_not_called()

    def test_time_trigger_dispatches(self):
        scheduler, run_manager = self._make_scheduler(batch_size=10, max_wait_seconds=0)
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_called_once()

    def test_force_dispatches_below_min_batch_size(self):
        scheduler, run_manager = self._make_scheduler(batch_size=10, min_batch_size=5, max_wait_seconds=9_999)
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch(force=True)
        run_manager.submit.assert_called_once()

    def test_no_dispatch_below_min_batch_size_without_force(self):
        # batch_size=1 so size trigger fires, but min_batch_size=3 blocks it
        scheduler, run_manager = self._make_scheduler(batch_size=1, min_batch_size=3, max_wait_seconds=9_999)
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch(force=False)
        run_manager.submit.assert_not_called()

    def test_queue_drained_after_dispatch(self):
        scheduler, _ = self._make_scheduler(batch_size=2)
        scheduler.enqueue(self._slide("a"))
        scheduler.enqueue(self._slide("b"))
        scheduler._maybe_dispatch()
        assert len(scheduler._pending) == 0
        assert scheduler._first_seen_at is None

    def test_first_seen_at_set_on_first_enqueue(self):
        scheduler, _ = self._make_scheduler()
        assert scheduler._first_seen_at is None
        scheduler.enqueue(self._slide("a"))
        assert scheduler._first_seen_at is not None

    def test_all_slides_sent_to_run_manager(self):
        scheduler, run_manager = self._make_scheduler(batch_size=3)
        for name in ("a", "b", "c"):
            scheduler.enqueue(self._slide(name))
        scheduler._maybe_dispatch()
        batch = run_manager.submit.call_args[0][1]
        assert {s["slide_id"] for s in batch} == {"a", "b", "c"}


# ===========================================================================
# NextflowRunner._write_csv
# ===========================================================================

class TestNextflowRunnerWriteCsv:
    def test_writes_csv_with_correct_columns(self, tmp_path):
        cfg = make_config(dispatch_dir=str(tmp_path))
        slides = [
            {"slide_path": "/slides/a.svs", "slide_id": "a", "oncotree_code": "BRCA"},
            {"slide_path": "/slides/b.svs", "slide_id": "b"},
        ]
        runner = NextflowRunner(cfg, "batch-001", slides, MagicMock())
        csv_path = runner._write_csv()
        assert os.path.exists(csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0] == {"slide_id": "a", "slide_path": "/slides/a.svs", "oncotree_code": "BRCA"}
        assert rows[1]["oncotree_code"] == ""  # missing key defaults to empty string

    def test_csv_filename_includes_batch_id(self, tmp_path):
        cfg = make_config(dispatch_dir=str(tmp_path))
        runner = NextflowRunner(cfg, "batch-XYZ", [], MagicMock())
        csv_path = runner._write_csv()
        assert "batch_batch-XYZ" in os.path.basename(csv_path)


# ===========================================================================
# recover_in_flight
# ===========================================================================

class TestRecoverInFlight:
    def test_no_running_batches_does_nothing(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        pending = deque()
        recover_in_flight(store, pending)
        assert len(pending) == 0

    def test_resets_dispatched_slides_to_pending(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        recover_in_flight(store, deque())
        assert len(store.get_pending_slides()) == 1

    def test_marks_interrupted_batch_as_failed(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        recover_in_flight(store, deque())
        row = store._conn().execute(
            "SELECT status FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "FAILED"

    def test_re_enqueues_recovered_slides(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_slide("/slides/b.svs", "b")
        store.add_batch("batch-001", "/dispatch/1.csv", 2, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs", "/slides/b.svs"], "batch-001")
        pending = deque()
        recover_in_flight(store, pending)
        assert len(pending) == 2
        assert {s["slide_path"] for s in pending} == {"/slides/a.svs", "/slides/b.svs"}

    def test_does_not_re_enqueue_already_succeeded_slides(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.complete_batch("batch-001", exit_code=0)
        store.mark_slides_complete("batch-001", succeeded=True)
        pending = deque()
        recover_in_flight(store, pending)
        # Batch is not RUNNING, so no recovery occurs
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# TcgaWatcher tests
# ---------------------------------------------------------------------------

class TestTcgaWatcher:
    """TcgaWatcher polls TCGA scripts, resolves paths, enqueues or downloads."""

    def _make_watcher(self, tmp_path, **kwargs):
        pass  # TcgaWatcher, WatcherConfig, StateStore already imported at top
        cfg = WatcherConfig(
            type="tcga",
            inventory_csv=str(tmp_path / "inventory.csv"),
            status_csv=str(tmp_path / "status.csv"),
            scripts_dir=str(tmp_path / "scripts"),
            **kwargs,
        )
        state = StateStore(str(tmp_path / "test.db"))
        stop_event = threading.Event()
        pending: deque = deque()
        watcher = TcgaWatcher(cfg, pending, state, stop_event, repo_dir=str(tmp_path), outdir=str(tmp_path / "results"))
        return watcher, pending, state, stop_event

    def _write_meta_csv(self, path: Path, rows: list[dict]):
        import csv as _csv
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["slide_id", "slide_path", "needs_download", "file_id", "file_name", "project_id"]
        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def test_enqueues_ready_slides(self, tmp_path):
        pass  # TcgaWatcher, WatcherConfig, StateStore already imported at top
        # Patch _run_script to succeed and produce a meta CSV
        meta_csv = tmp_path / "status_dispatcher.meta.csv"
        self._write_meta_csv(meta_csv, [
            {"slide_id": "slide-A", "slide_path": "/slides/A.svs", "needs_download": "false"},
            {"slide_id": "slide-B", "slide_path": "/slides/B.svs", "needs_download": "false"},
        ])
        watcher, pending, state, _ = self._make_watcher(tmp_path)

        def fake_run_script(script, args):
            return 0

        watcher._run_script = fake_run_script
        # Point meta CSV path to what _poll will compute
        import unittest.mock as mock
        with mock.patch.object(watcher, "_run_script", side_effect=fake_run_script):
            # Override the meta_csv path by patching open — instead, just call _poll
            # and monkey-patch the path computation
            orig_poll = watcher._poll

            def patched_poll():
                # fake _run_script side effects: produce the meta CSV
                import csv as _csv
                samples_csv = str(tmp_path / "status_dispatcher.csv")
                # write meta csv that _poll looks for
                self._write_meta_csv(
                    Path(samples_csv.replace(".csv", ".meta.csv")),
                    [
                        {"slide_id": "slide-A", "slide_path": "/slides/A.svs",
                         "needs_download": "false"},
                        {"slide_id": "slide-B", "slide_path": "/slides/B.svs",
                         "needs_download": "false"},
                    ],
                )
                # patch cfg.status_csv so samples_csv resolves correctly
                watcher.cfg.status_csv = str(tmp_path / "status.csv")
                orig_poll()

            watcher._poll = patched_poll
            watcher._poll()

        assert len(pending) == 2
        slide_ids = {s["slide_id"] for s in pending}
        assert slide_ids == {"slide-A", "slide-B"}

    def test_skips_already_known_slides(self, tmp_path):
        pass  # TcgaWatcher, WatcherConfig, StateStore already imported at top
        import unittest.mock as mock

        watcher, pending, state, _ = self._make_watcher(tmp_path)
        state.add_slide("/slides/A.svs", "slide-A")  # already known

        samples_csv = str(tmp_path / "status_dispatcher.csv")
        self._write_meta_csv(
            Path(samples_csv.replace(".csv", ".meta.csv")),
            [{"slide_id": "slide-A", "slide_path": "/slides/A.svs", "needs_download": "false"}],
        )

        with mock.patch.object(watcher, "_run_script", return_value=0):
            watcher.cfg.status_csv = str(tmp_path / "status.csv")
            watcher._poll()

        assert len(pending) == 0  # slide-A is already known, not re-enqueued

    def test_download_enabled_submits_needs_download_slides(self, tmp_path):
        pass  # TcgaWatcher, WatcherConfig, StateStore already imported at top
        import unittest.mock as mock

        watcher, pending, state, _ = self._make_watcher(
            tmp_path, download_enabled=True, download_dir=str(tmp_path / "dl")
        )

        samples_csv = str(tmp_path / "status_dispatcher.csv")
        self._write_meta_csv(
            Path(samples_csv.replace(".csv", ".meta.csv")),
            [{"slide_id": "slide-C", "slide_path": str(tmp_path / "dl/uuid/C.svs"),
              "needs_download": "true", "file_id": "uuid", "file_name": "C.svs"}],
        )

        submitted = []

        def fake_submit(fn, slide):
            submitted.append(slide["slide_id"])

        with mock.patch.object(watcher, "_run_script", return_value=0), \
             mock.patch.object(watcher._download_executor, "submit", side_effect=fake_submit):
            watcher.cfg.status_csv = str(tmp_path / "status.csv")
            watcher._poll()

        assert "slide-C" in submitted

    def test_download_and_enqueue_succeeds(self, tmp_path):
        pass  # TcgaWatcher, WatcherConfig, StateStore already imported at top
        import unittest.mock as mock

        dest = tmp_path / "dl" / "uuid123" / "slide.svs"
        dest.parent.mkdir(parents=True, exist_ok=True)

        watcher, pending, state, _ = self._make_watcher(
            tmp_path, download_enabled=True, download_dir=str(tmp_path / "dl")
        )
        state.add_slide(str(dest), "slide-D")

        slide = {"slide_id": "slide-D", "slide_path": str(dest),
                 "file_id": "uuid123", "file_name": "slide.svs"}

        def fake_run(cmd, **kwargs):
            dest.write_bytes(b"fake")  # simulate gdc-client writing the file
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            watcher._download_and_enqueue(slide)

        assert len(pending) == 1
        assert pending[0]["slide_id"] == "slide-D"

    def test_download_failure_does_not_enqueue(self, tmp_path):
        pass  # TcgaWatcher, WatcherConfig, StateStore already imported at top
        import unittest.mock as mock

        watcher, pending, state, _ = self._make_watcher(
            tmp_path, download_enabled=True, download_dir=str(tmp_path / "dl")
        )
        slide = {"slide_id": "slide-E", "slide_path": str(tmp_path / "dl/u/E.svs"),
                 "file_id": "u", "file_name": "E.svs"}

        fake_result = mock.Mock(returncode=1, stdout="", stderr="network error")
        with mock.patch("subprocess.run", return_value=fake_result):
            watcher._download_and_enqueue(slide)

        assert len(pending) == 0


# ---------------------------------------------------------------------------
# post_batch_hooks tests
# ---------------------------------------------------------------------------

class TestPostBatchHooks:
    """_run_post_batch_hooks runs commands with template substitution."""

    def _make_runner(self, tmp_path, hooks):
        pass  # NextflowRunner, Config, StateStore already imported at top
        cfg = Config(
            repo_dir=str(tmp_path),
            nextflow_profiles="standard",
            outdir=str(tmp_path / "results"),
            work_base_dir=str(tmp_path / "work"),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            post_batch_hooks=hooks,
        )
        state = StateStore(str(tmp_path / "state" / "test.db"))
        return NextflowRunner(cfg, "batch-001", [], state)

    def test_hook_runs_command_with_substitution(self, tmp_path):
        import unittest.mock as mock
        sentinel = tmp_path / "hook_ran.txt"
        hook = {
            "command": sys.executable,
            "args": ["-c", f"open('{sentinel}', 'w').write('{{batch_id}}')"],
        }
        runner = self._make_runner(tmp_path, [hook])
        runner._run_post_batch_hooks("/fake/batch.csv")
        assert sentinel.exists()
        assert sentinel.read_text() == "batch-001"

    def test_hook_substitutes_all_vars(self, tmp_path):
        import unittest.mock as mock
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(returncode=0, stderr="")

        hook = {
            "command": "echo",
            "args": ["{batch_csv}", "{batch_id}", "{outdir}", "{repo_dir}"],
        }
        runner = self._make_runner(tmp_path, [hook])
        with mock.patch("subprocess.run", side_effect=fake_run):
            runner._run_post_batch_hooks("/my/batch.csv")

        assert calls[0] == ["echo", "/my/batch.csv", "batch-001",
                             str(tmp_path / "results"), str(tmp_path)]

    def test_hook_failure_logged_not_raised(self, tmp_path):
        import unittest.mock as mock
        hook = {"command": "false", "args": []}
        runner = self._make_runner(tmp_path, [hook])
        fake_result = mock.Mock(returncode=1, stderr="fail")
        with mock.patch("subprocess.run", return_value=fake_result):
            runner._run_post_batch_hooks("/batch.csv")  # must not raise

    def test_no_hooks_is_noop(self, tmp_path):
        runner = self._make_runner(tmp_path, [])
        runner._run_post_batch_hooks("/batch.csv")  # must not raise or call anything


# ---------------------------------------------------------------------------
# Auto post-batch hook generation
# ---------------------------------------------------------------------------

class TestAutoHooks:
    """Config._build_auto_hooks generates hooks from wds_destinations/databricks_volume_path."""

    def _load_config(self, tmp_path, watcher_extra=None, extra_raw=None):
        import yaml as _yaml
        cfg_data = {
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "watchers": [{
                "type": "tcga",
                "inventory_csv": str(tmp_path / "inventory.csv"),
                "status_csv": str(tmp_path / "status.csv"),
                **(watcher_extra or {}),
            }],
            **(extra_raw or {}),
        }
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(_yaml.dump(cfg_data))
        return Config.load(str(cfg_path))

    def test_no_auto_hook_without_destinations(self, tmp_path):
        cfg = self._load_config(tmp_path)
        assert cfg.post_batch_hooks == []

    def test_auto_hook_generated_when_wds_dest_set(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "wds_destinations": {"ctranspath": "s3://bucket/wds/ctranspath"},
        })
        assert len(cfg.post_batch_hooks) == 1
        hook = cfg.post_batch_hooks[0]
        assert "tcga_append_wds.py" in hook["command"]
        args = " ".join(hook["args"])
        assert "ctranspath" in args
        assert "s3://bucket/wds/ctranspath" in args
        assert "--slide-ids-csv={batch_csv}" in hook["args"]

    def test_auto_hook_includes_staging_dir_when_set(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "wds_destinations": {"ctranspath": "s3://bucket/wds"},
            "wds_staging_dir": "/staging",
        })
        args = " ".join(cfg.post_batch_hooks[0]["args"])
        assert "--staging-dir=/staging" in args

    def test_auto_hook_not_generated_for_local_watcher(self, tmp_path):
        import yaml as _yaml
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "watchers": [{"type": "local", "path": "/slides"}],
        }))
        cfg = Config.load(str(cfg_path))
        assert cfg.post_batch_hooks == []

    def test_auto_hooks_prepend_before_explicit_hooks(self, tmp_path):
        explicit = [{"command": "echo done", "args": []}]
        cfg = self._load_config(
            tmp_path,
            watcher_extra={"wds_destinations": {"ctranspath": "s3://bucket/wds"}},
            extra_raw={"post_batch_hooks": explicit},
        )
        assert len(cfg.post_batch_hooks) == 2
        assert "tcga_append_wds.py" in cfg.post_batch_hooks[0]["command"]
        assert cfg.post_batch_hooks[1] == explicit[0]

    def test_one_auto_hook_per_watcher(self, tmp_path):
        import yaml as _yaml
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "watchers": [
                {"type": "tcga", "inventory_csv": "i.csv", "status_csv": "s.csv",
                 "wds_destinations": {"ctranspath": "s3://b/c"}},
                {"type": "tcga", "inventory_csv": "i.csv", "status_csv": "s.csv",
                 "wds_destinations": {"uni2h": "s3://b/u"}},
            ],
        }))
        cfg = Config.load(str(cfg_path))
        assert len(cfg.post_batch_hooks) == 2
        models = [h["args"][0] for h in cfg.post_batch_hooks]  # --pt-dir contains model name
        assert any("ctranspath" in m for m in models)
        assert any("uni2h" in m for m in models)

    def test_databricks_hook_generated_when_volume_path_set(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "databricks_volume_path": "/Volumes/cat/schema/vol/tcga.parquet",
        })
        assert len(cfg.post_batch_hooks) == 1
        hook = cfg.post_batch_hooks[0]
        assert "tcga_sync_databricks.py" in hook["command"]
        args = " ".join(hook["args"])
        assert "/Volumes/cat/schema/vol/tcga.parquet" in args

    def test_databricks_hook_includes_job_id_when_set(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "databricks_volume_path": "/Volumes/cat/schema/vol/tcga.parquet",
            "databricks_job_id": "99999",
        })
        args = " ".join(cfg.post_batch_hooks[0]["args"])
        assert "--job-id=99999" in args

    def test_cleanup_results_adds_delete_local_flag(self, tmp_path):
        """cleanup_results=True adds --delete-local to the WDS auto-hook args."""
        cfg = self._load_config(
            tmp_path,
            watcher_extra={"wds_destinations": {"ctranspath": "s3://bucket/wds"}},
            extra_raw={"cleanup_results": True},
        )
        args = cfg.post_batch_hooks[0]["args"]
        assert "--delete-local" in args

    def test_cleanup_results_false_does_not_add_delete_local(self, tmp_path):
        """cleanup_results=False (default) does not add --delete-local to hooks."""
        cfg = self._load_config(
            tmp_path,
            watcher_extra={"wds_destinations": {"ctranspath": "s3://bucket/wds"}},
        )
        args = cfg.post_batch_hooks[0]["args"]
        assert "--delete-local" not in args


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests for post-batch cleanup: downloads, batch CSV, old logs."""

    def _make_state(self, tmp_path):
        return StateStore(str(tmp_path / "state" / "dispatcher.db"))

    def test_cleanup_downloads_removes_dir(self, tmp_path):
        """cleanup_downloads=True deletes the download directory recorded for a slide."""
        state = self._make_state(tmp_path)
        slide_path = str(tmp_path / "slides" / "file_abc123" / "slide.svs")
        dl_dir = str(tmp_path / "slides" / "file_abc123")
        os.makedirs(dl_dir)
        Path(slide_path).touch()

        state.add_slide(slide_path, "TCGA-XX-001")
        state.mark_dispatched([slide_path], "batch_001")
        state.set_download_path(slide_path, dl_dir)
        state.mark_slides_complete("batch_001", succeeded=True)

        cfg = make_config(cleanup_downloads=True)
        runner = NextflowRunner(cfg, "batch_001", [], state)
        runner._cleanup(
            csv_path=str(tmp_path / "batch_001.csv"),
            log_path=str(tmp_path / "batch_001.log"),
            work_dir=str(tmp_path / "work"),
        )

        assert not os.path.exists(dl_dir)

    def test_cleanup_downloads_skipped_when_false(self, tmp_path):
        """cleanup_downloads=False (default) leaves download dirs intact."""
        state = self._make_state(tmp_path)
        dl_dir = str(tmp_path / "slides" / "file_xyz")
        os.makedirs(dl_dir)
        slide_path = str(Path(dl_dir) / "slide.svs")
        Path(slide_path).touch()

        state.add_slide(slide_path, "TCGA-XX-002")
        state.mark_dispatched([slide_path], "batch_002")
        state.set_download_path(slide_path, dl_dir)

        cfg = make_config(cleanup_downloads=False)
        runner = NextflowRunner(cfg, "batch_002", [], state)
        runner._cleanup(
            csv_path=str(tmp_path / "batch_002.csv"),
            log_path=str(tmp_path / "batch_002.log"),
            work_dir=str(tmp_path / "work"),
        )

        assert os.path.exists(dl_dir)

    def test_cleanup_batch_csv_removes_file(self, tmp_path):
        """cleanup_batch_csv=True deletes the batch samples CSV."""
        state = self._make_state(tmp_path)
        csv_file = tmp_path / "batch_003.csv"
        csv_file.write_text("slide_id,slide_path\n")

        cfg = make_config(cleanup_batch_csv=True)
        runner = NextflowRunner(cfg, "batch_003", [], state)
        runner._cleanup(
            csv_path=str(csv_file),
            log_path=str(tmp_path / "batch_003.log"),
            work_dir=str(tmp_path / "work"),
        )

        assert not csv_file.exists()

    def test_cleanup_old_logs_removes_stale_log(self, tmp_path):
        """cleanup_logs_after_days removes logs for old succeeded batches."""
        from datetime import datetime, timedelta, timezone

        state = self._make_state(tmp_path)
        log_file = tmp_path / "old_batch.log"
        log_file.write_text("nextflow output\n")

        state.add_batch("old_batch", str(tmp_path / "old.csv"), 5, str(log_file))
        # Manually backdate completed_at to 40 days ago
        old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        conn = state._conn()
        conn.execute(
            "UPDATE batches SET status='SUCCEEDED', completed_at=? WHERE batch_id='old_batch'",
            (old_ts,),
        )
        conn.commit()

        cfg = make_config(cleanup_logs_after_days=30)
        runner = NextflowRunner(cfg, "current_batch", [], state)
        runner._cleanup(
            csv_path=str(tmp_path / "current.csv"),
            log_path=str(tmp_path / "current.log"),
            work_dir=str(tmp_path / "work"),
        )

        assert not log_file.exists()

    def test_cleanup_old_logs_keeps_recent_log(self, tmp_path):
        """cleanup_logs_after_days keeps logs for recently succeeded batches."""
        from datetime import datetime, timedelta, timezone

        state = self._make_state(tmp_path)
        log_file = tmp_path / "recent_batch.log"
        log_file.write_text("nextflow output\n")

        state.add_batch("recent_batch", str(tmp_path / "recent.csv"), 5, str(log_file))
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        conn = state._conn()
        conn.execute(
            "UPDATE batches SET status='SUCCEEDED', completed_at=? WHERE batch_id='recent_batch'",
            (recent_ts,),
        )
        conn.commit()

        cfg = make_config(cleanup_logs_after_days=30)
        runner = NextflowRunner(cfg, "current_batch", [], state)
        runner._cleanup(
            csv_path=str(tmp_path / "current.csv"),
            log_path=str(tmp_path / "current.log"),
            work_dir=str(tmp_path / "work"),
        )

        assert log_file.exists()
