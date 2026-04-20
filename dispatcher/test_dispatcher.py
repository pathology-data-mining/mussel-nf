"""Tests for mussel-dispatcher.py"""

import csv
import importlib.util
import os
import tempfile
import threading
import time
from collections import deque
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
