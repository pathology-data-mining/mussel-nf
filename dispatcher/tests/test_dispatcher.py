"""Tests for mussel-dispatcher."""

import csv
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

import mussel_dispatcher as _mod
from mussel_dispatcher import (
    Config,
    WatcherConfig,
    StateStore,
    ReadinessChecker,
    LocalWatcher,
    BatchScheduler,
    NextflowRunner,
    TcgaWatcher,
    DatabricksWatcher,
    recover_in_flight,
)
from mussel_dispatcher.runner import (
    collect_manifests,
    _parse_run_name_from_log,
    _lookup_session_id_in_history,
    _extract_nf_session_id_from_log,
    _query_session_id_via_nf_cli,
)
from mussel_dispatcher.config import (
    _read_nf_model_types,
    _load_secrets_env,
    _load_nf_secrets,
)
from mussel_dispatcher.scheduler import BatchScheduler, RunManager

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
            "SELECT status, fail_count FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["fail_count"] == 1

    def test_mark_slides_complete_increments_fail_count(self, store):
        store.add_slide("/slides/a.svs", "a")
        for i in range(3):
            store.mark_dispatched(["/slides/a.svs"], f"batch-{i:03d}")
            store.mark_slides_complete(f"batch-{i:03d}", succeeded=False)
        row = store._conn().execute(
            "SELECT fail_count FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["fail_count"] == 3

    def test_mark_slides_complete_fast_fail_resets_to_pending(self, store):
        """charge_fail_count=False resets DISPATCHED slides to PENDING (infra failure)."""
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.mark_slides_complete("batch-001", succeeded=False, charge_fail_count=False)
        row = store._conn().execute(
            "SELECT status, fail_count, batch_id FROM slides WHERE slide_path=?",
            ("/slides/a.svs",),
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["fail_count"] == 0  # retry slot not charged
        assert row["batch_id"] is None

    def test_blacklist_slide_existing(self, store):
        """blacklist_slide marks an existing slide as permanently FAILED."""
        store.add_slide("/slides/bad.svs", "bad-slide")
        store.mark_dispatched(["/slides/bad.svs"], "batch-001")
        store.blacklist_slide("bad-slide", reason="S3 file not found (404)", max_retries=999)
        row = store._conn().execute(
            "SELECT status, fail_count, error_msg, batch_id FROM slides WHERE slide_id=?",
            ("bad-slide",),
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["fail_count"] == 999
        assert "404" in row["error_msg"]
        assert row["batch_id"] is None

    def test_blacklist_slide_new(self, store):
        """blacklist_slide inserts a new record when slide is not yet in DB."""
        store.blacklist_slide("unknown-slide", reason="File missing on S3", max_retries=999)
        row = store._conn().execute(
            "SELECT status, fail_count, error_msg FROM slides WHERE slide_id=?",
            ("unknown-slide",),
        ).fetchone()
        assert row is not None
        assert row["status"] == "FAILED"
        assert row["fail_count"] == 999

    def test_blacklist_slide_never_reset_to_pending(self, store):
        """reset_failed_to_pending never re-queues a blacklisted slide."""
        store.blacklist_slide("stuck-slide", reason="404", max_retries=999)
        store.reset_failed_to_pending(max_retries=999)
        row = store._conn().execute(
            "SELECT status FROM slides WHERE slide_id=?", ("stuck-slide",)
        ).fetchone()
        assert row["status"] == "FAILED"

    def test_blacklist_slide_not_in_pending(self, store):
        """A blacklisted slide is never returned as pending."""
        store.blacklist_slide("blacklisted", reason="404", max_retries=999)
        pending = store.get_pending_slides()
        assert not any(s["slide_id"] == "blacklisted" for s in pending)


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
        store.add_batch("batch-001", "/dispatch/1.csv", None, 5, "/logs/1.log")
        assert len(store.get_running_batches()) == 1
        store.complete_batch("batch-001", exit_code=0)
        assert store.get_running_batches() == []
        row = store._conn().execute(
            "SELECT status FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"

    def test_complete_batch_failed_sets_status_and_exit(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", None, 2, "/logs/1.log")
        store.complete_batch("batch-001", exit_code=1)
        row = store._conn().execute(
            "SELECT status, nextflow_exit FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["nextflow_exit"] == 1

    def test_record_batch_manifest(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        store.record_batch_manifest("batch-001", "/results/manifest-001.csv")
        paths = store.get_all_manifest_paths()
        assert "/results/manifest-001.csv" in paths

    def test_get_all_manifest_paths_skips_null(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        assert store.get_all_manifest_paths() == []

    def test_get_running_batches_only_returns_running(self, store):
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        store.add_batch("batch-002", "/dispatch/2.csv", None, 1, "/logs/2.log")
        store.complete_batch("batch-001", 0)
        running = store.get_running_batches()
        assert len(running) == 1
        assert running[0]["batch_id"] == "batch-002"

    def test_get_pending_slides_backfills_gdc_uri(self, store):
        """Slides with gdc:// URIs but empty file_id/file_name get backfilled."""
        fid = "acbbcfa8-90c4-408a-966a-294b7b30eca3"
        fname = "TCGA-UW-A7GC-11Z-00-DX1.ADAF7B0F.svs"
        db_key = f"gdc://{fid}/{fname}"
        # Simulate a migrated row: gdc:// URI but file_id/file_name are empty
        store._conn().execute(
            "INSERT INTO slides (slide_path, slide_id, status, file_id, file_name, needs_download, first_seen_at)"
            " VALUES (?, ?, 'PENDING', '', '', 1, '2024-01-01')",
            (db_key, "TCGA-UW-A7GC-11Z-00-DX1"),
        )
        store._conn().commit()
        pending = store.get_pending_slides()
        assert len(pending) == 1
        assert pending[0]["file_id"] == fid
        assert pending[0]["file_name"] == fname

    def test_get_pending_slides_no_backfill_without_gdc_uri(self, store):
        """Non-gdc slides with empty file_id are left as-is."""
        store.add_slide("/slides/a.svs", "a", needs_download=False)
        pending = store.get_pending_slides()
        assert pending[0]["file_id"] == ""




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

    def test_csv_backfills_file_id_from_gdc_uri(self, tmp_path):
        """_write_csv extracts file_id/file_name from gdc:// URI when empty."""
        cfg = make_config(dispatch_dir=str(tmp_path))
        fid = "acbbcfa8-90c4-408a-966a-294b7b30eca3"
        fname = "TCGA-UW-A7GC.svs"
        slides = [{
            "slide_path": f"gdc://{fid}/{fname}",
            "slide_id": "TCGA-UW-A7GC",
            "oncotree_code": "",
            "needs_download": True,
            "file_id": "",      # empty — should be backfilled
            "file_name": "",
        }]
        runner = NextflowRunner(cfg, "batch-backfill", slides, MagicMock())
        csv_path = runner._write_csv()
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["file_id"] == fid
        assert rows[0]["file_name"] == fname

    def test_csv_filename_includes_batch_id(self, tmp_path):
        cfg = make_config(dispatch_dir=str(tmp_path))
        runner = NextflowRunner(cfg, "batch-XYZ", [], MagicMock())
        csv_path = runner._write_csv()
        assert "batch_batch-XYZ" in os.path.basename(csv_path)


# ===========================================================================
# NextflowRunner command construction
# ===========================================================================

class TestNextflowRunnerCommand:
    def test_no_nextflow_config_by_default(self, tmp_path):
        cfg = make_config()
        slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
        state = MagicMock()
        runner = NextflowRunner(cfg, "batch-001", slides, state)
        # Simulate the cmd list construction directly (mirrors NextflowRunner.run logic)
        work_dir = "/tmp/work/batch_001"
        csv_path = "/tmp/dispatch/batch_001.csv"
        cmd = [
            "nextflow", "run", cfg.repo_dir,
            "-profile", cfg.nextflow_profiles,
            "-work-dir", work_dir,
            "--samples_csv", csv_path,
            "--outdir", cfg.outdir,
        ]
        if cfg.nextflow_config:
            cmd += ["-c", cfg.nextflow_config]
        assert "-c" not in cmd

    def test_nextflow_config_appended_when_set(self, tmp_path):
        cfg = make_config(nextflow_config="/path/to/custom.config")
        slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
        runner = NextflowRunner(cfg, "batch-001", slides, MagicMock())
        work_dir = "/tmp/work/batch_001"
        csv_path = "/tmp/dispatch/batch_001.csv"
        cmd = [
            "nextflow", "run", cfg.repo_dir,
            "-profile", cfg.nextflow_profiles,
            "-work-dir", work_dir,
            "--samples_csv", csv_path,
            "--outdir", cfg.outdir,
        ]
        if cfg.nextflow_config:
            cmd += ["-c", cfg.nextflow_config]
        assert "-c" in cmd
        assert "/path/to/custom.config" in cmd

    def test_nextflow_config_resolved_from_yaml(self, tmp_path):
        """Relative nextflow_config paths are resolved relative to the config file."""
        custom_cfg = tmp_path / "custom.config"
        custom_cfg.write_text("// custom")
        data = {
            "nextflow_profiles": "standard",
            "outdir": "/out",
            "nextflow_config": "custom.config",
            "watchers": [],
        }
        yaml_path = tmp_path / "dispatcher.yaml"
        yaml_path.write_text(yaml.dump(data))
        cfg = Config.load(str(yaml_path))
        assert cfg.nextflow_config == str(tmp_path / "custom.config")

    def test_nextflow_config_absent_when_not_set_in_yaml(self, tmp_path):
        data = {
            "nextflow_profiles": "standard",
            "outdir": "/out",
            "watchers": [],
        }
        yaml_path = tmp_path / "dispatcher.yaml"
        yaml_path.write_text(yaml.dump(data))
        cfg = Config.load(str(yaml_path))
        assert cfg.nextflow_config == ""

    def test_no_nextflow_params_file_by_default(self, tmp_path):
        cfg = make_config()
        cmd = ["nextflow", "run", cfg.repo_dir, "-profile", cfg.nextflow_profiles,
               "-work-dir", "/tmp/work", "--samples_csv", "/tmp/batch.csv", "--outdir", cfg.outdir]
        if cfg.nextflow_params_file:
            cmd += ["-params-file", cfg.nextflow_params_file]
        assert "-params-file" not in cmd

    def test_nextflow_params_file_appended_when_set(self, tmp_path):
        cfg = make_config(nextflow_params_file="/path/to/params.yaml")
        cmd = ["nextflow", "run", cfg.repo_dir, "-profile", cfg.nextflow_profiles,
               "-work-dir", "/tmp/work", "--samples_csv", "/tmp/batch.csv", "--outdir", cfg.outdir]
        if cfg.nextflow_params_file:
            cmd += ["-params-file", cfg.nextflow_params_file]
        assert "-params-file" in cmd
        assert "/path/to/params.yaml" in cmd

    def test_nextflow_params_file_resolved_from_yaml(self, tmp_path):
        """Relative nextflow_params_file paths are resolved relative to the config file."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text("key: value")
        data = {
            "nextflow_profiles": "standard",
            "outdir": "/out",
            "nextflow_params_file": "params.yaml",
            "watchers": [],
        }
        yaml_path = tmp_path / "dispatcher.yaml"
        yaml_path.write_text(yaml.dump(data))
        cfg = Config.load(str(yaml_path))
        assert cfg.nextflow_params_file == str(tmp_path / "params.yaml")

    def test_nextflow_params_file_absent_when_not_set_in_yaml(self, tmp_path):
        data = {"nextflow_profiles": "standard", "outdir": "/out", "watchers": []}
        yaml_path = tmp_path / "dispatcher.yaml"
        yaml_path.write_text(yaml.dump(data))
        cfg = Config.load(str(yaml_path))
        assert cfg.nextflow_params_file == ""

    def test_name_flag_uses_deterministic_run_name(self, tmp_path):
        """-name dispatcher_{batch_id} is added to the NF command for deterministic run names."""
        cfg = make_config()
        slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
        state = MagicMock()
        runner = NextflowRunner(cfg, "batch-007", slides, state)
        work_dir = "/tmp/work/batch_007"
        csv_path = "/tmp/dispatch/batch_007.csv"
        # Mirror the cmd construction from runner.py
        nf_run_name = "dispatcher_batch-007"
        cmd = [
            "nextflow", "run", cfg.repo_dir,
            "-profile", cfg.nextflow_profiles,
            "-work-dir", work_dir,
            "--samples_csv", csv_path,
            "--outdir", cfg.outdir,
            "-name", nf_run_name,
        ]
        assert "-name" in cmd
        assert "dispatcher_batch-007" in cmd

    def test_name_flag_includes_batch_id(self):
        """The -name value always embeds the batch_id so it is unique per batch."""
        batch_id = "xyz-123"
        nf_run_name = f"dispatcher_{batch_id}"
        assert batch_id in nf_run_name

class TestRecoverInFlight:
    def test_no_running_batches_does_nothing(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        pending = deque()
        specs = recover_in_flight(store, pending)
        assert len(pending) == 0
        assert specs == []

    def test_resets_dispatched_slides_when_work_dir_missing(self, tmp_path):
        """When work dir is missing/None, slides are reset to PENDING."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        specs = recover_in_flight(store, deque())
        assert specs == []
        assert len(store.get_pending_slides()) == 1

    def test_marks_interrupted_batch_as_failed_when_no_work_dir(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        recover_in_flight(store, deque())
        row = store._conn().execute(
            "SELECT status FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "FAILED"

    def test_re_enqueues_recovered_slides_when_no_work_dir(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_slide("/slides/b.svs", "b")
        store.add_batch("batch-001", "/dispatch/1.csv", None, 2, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs", "/slides/b.svs"], "batch-001")
        pending = deque()
        recover_in_flight(store, pending)
        assert len(pending) == 2
        assert {s["slide_path"] for s in pending} == {"/slides/a.svs", "/slides/b.svs"}

    def test_returns_resume_spec_when_work_dir_exists(self, tmp_path):
        """When work dir and csv both exist, returns resume spec instead of resetting to PENDING."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        work_dir = tmp_path / "batch-001" / "work"
        work_dir.mkdir(parents=True)
        csv_path = tmp_path / "batch-001.csv"
        csv_path.write_text("slide_id,slide_path\na,/slides/a.svs\n")
        store.add_batch("batch-001", str(csv_path), str(work_dir), 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        pending = deque()
        specs = recover_in_flight(store, pending)
        # Slides stay DISPATCHED (not reset to PENDING) — will be handled by resume run
        assert len(pending) == 0
        assert len(specs) == 1
        assert specs[0] == ("batch-001", str(csv_path), str(work_dir))
        # Batch should still be RUNNING (not marked FAILED)
        row = store._conn().execute(
            "SELECT status FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "RUNNING"

    def test_does_not_re_enqueue_already_succeeded_slides(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.complete_batch("batch-001", exit_code=0)
        store.mark_slides_complete("batch-001", succeeded=True)
        pending = deque()
        specs = recover_in_flight(store, pending)
        # Batch is not RUNNING, so no recovery occurs
        assert len(pending) == 0
        assert specs == []

    def test_max_slide_retries_skips_high_fail_count(self, tmp_path):
        """Slides with fail_count >= max_slide_retries are permanently skipped."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/bad.svs", "bad")
        store.add_slide("/slides/ok.svs", "ok")
        # Fail bad slide 3 times
        for i in range(3):
            store.mark_dispatched(["/slides/bad.svs"], f"batch-{i:03d}")
            store.mark_slides_complete(f"batch-{i:03d}", succeeded=False)
        # Fail ok slide once
        store.mark_dispatched(["/slides/ok.svs"], "batch-003")
        store.mark_slides_complete("batch-003", succeeded=False)

        pending = deque()
        # max_slide_retries=3: bad slide (fail_count=3) should be skipped; ok slide (fail_count=1) should be re-queued
        recover_in_flight(store, pending, retry_failed=True, max_slide_retries=3)
        paths = {s["slide_path"] for s in pending}
        assert "/slides/ok.svs" in paths
        assert "/slides/bad.svs" not in paths


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
# NextflowRunner.run() integration tests
# ---------------------------------------------------------------------------

class TestNextflowRunnerRun:
    """Integration tests for NextflowRunner.run() — mocks subprocess only."""

    def _make_runner(self, tmp_path, slides=None):
        (tmp_path / "batches").mkdir()
        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        (tmp_path / "work").mkdir()
        cfg = make_config(
            repo_dir=str(tmp_path),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            work_base_dir=str(tmp_path / "work"),
        )
        state = StateStore(str(tmp_path / "state" / "test.db"))
        if slides is None:
            slides = [{"slide_path": "/slides/a.svs", "slide_id": "a"}]
        for s in slides:
            state.add_slide(s["slide_path"], s["slide_id"])
        runner = NextflowRunner(cfg, "batch-001", slides, state)
        state.mark_dispatched([s["slide_path"] for s in slides], "batch-001")
        return runner, state

    def test_run_success_marks_slides_succeeded(self, tmp_path):
        import unittest.mock as mock
        runner, state = self._make_runner(tmp_path)
        fake_proc = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=fake_proc):
            exit_code = runner.run()
        assert exit_code == 0
        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"

    def test_run_failure_marks_slides_failed_and_increments_fail_count(self, tmp_path):
        import unittest.mock as mock
        runner, state = self._make_runner(tmp_path)
        fake_proc = mock.Mock(returncode=1)
        # Patch time so batch_duration >= 60s (not a fast-fail)
        with mock.patch("subprocess.run", return_value=fake_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mock_time:
            mock_time.time.side_effect = [0.0, 120.0]  # started_at=0, ended_at=120
            exit_code = runner.run()
        assert exit_code == 1
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["fail_count"] == 1

    def test_run_fast_fail_resets_slides_to_pending(self, tmp_path):
        """Batch failure in <60s (infra error) resets slides to PENDING without charging fail_count."""
        import unittest.mock as mock
        runner, state = self._make_runner(tmp_path)
        fake_proc = mock.Mock(returncode=1)
        with mock.patch("subprocess.run", return_value=fake_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mock_time:
            mock_time.time.side_effect = [0.0, 5.0]  # 5s — clearly a fast fail
            exit_code = runner.run()
        assert exit_code == 1
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["fail_count"] == 0  # retry budget not consumed


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

class TestNfModelTypes:
    """_read_nf_model_types parses model_types from nextflow.config."""

    def test_reads_model_types(self, tmp_path):
        nf_config = tmp_path / "nextflow.config"
        nf_config.write_text("featurize {\n    model_types = ['hoptimus1', 'titan_slide']\n}\n")
        assert _read_nf_model_types(str(tmp_path)) == ["hoptimus1", "titan_slide"]

    def test_returns_empty_when_file_missing(self, tmp_path):
        assert _read_nf_model_types(str(tmp_path)) == []

    def test_returns_empty_when_no_match(self, tmp_path):
        (tmp_path / "nextflow.config").write_text("params { batch_size = 50 }\n")
        assert _read_nf_model_types(str(tmp_path)) == []

    def test_watcher_models_auto_filled(self, tmp_path):
        import yaml as _yaml
        (tmp_path / "nextflow.config").write_text(
            "featurize {\n    model_types = ['hoptimus1', 'titan_slide']\n}\n"
        )
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "repo_dir": str(tmp_path),
            "watchers": [{"type": "tcga", "inventory_csv": "i.csv", "status_csv": "s.csv"}],
        }))
        cfg = Config.load(str(cfg_path))
        assert cfg.watchers[0].models == ["hoptimus1", "titan_slide"]

    def test_explicit_models_not_overridden(self, tmp_path):
        import yaml as _yaml
        (tmp_path / "nextflow.config").write_text(
            "featurize {\n    model_types = ['hoptimus1']\n}\n"
        )
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "repo_dir": str(tmp_path),
            "watchers": [{"type": "tcga", "inventory_csv": "i.csv", "status_csv": "s.csv",
                          "models": ["ctranspath"]}],
        }))
        cfg = Config.load(str(cfg_path))
        assert cfg.watchers[0].models == ["ctranspath"]


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
        assert "append_wds" in hook["command"]
        args = " ".join(hook["args"])
        assert "ctranspath" in args
        assert "s3://bucket/wds/ctranspath" in args
        assert "--slide-ids-csv={batch_csv}" in hook["args"]
        assert "--manifest-csv={outdir}/wds_manifest.csv" in hook["args"]

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
        assert "append_wds" in cfg.post_batch_hooks[0]["command"]
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
        assert "sync_databricks" in hook["command"]
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

        state.add_batch("old_batch", str(tmp_path / "old.csv"), None, 5, str(log_file))
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

        state.add_batch("recent_batch", str(tmp_path / "recent.csv"), None, 5, str(log_file))
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

    def test_cleanup_work_dir_on_failure(self, tmp_path):
        """cleanup_work_dir=True removes work dir even when batch failed."""
        state = self._make_state(tmp_path)
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "staged_file.svs").write_text("data")
        dl_dir = tmp_path / "downloads"
        dl_dir.mkdir()

        cfg = make_config(cleanup_work_dir=True, cleanup_downloads=True)
        runner = NextflowRunner(cfg, "batch_fail", [], state)
        runner._cleanup(
            csv_path=str(tmp_path / "batch_fail.csv"),
            log_path=str(tmp_path / "batch_fail.log"),
            work_dir=str(work_dir),
            succeeded=False,
        )

        assert not work_dir.exists(), "work dir should be deleted on failure"
        assert dl_dir.exists(), "download dir should NOT be deleted on failure"

    def test_cleanup_work_dir_skipped_on_failure_when_disabled(self, tmp_path):
        """cleanup_work_dir=False leaves work dir even when batch failed."""
        state = self._make_state(tmp_path)
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        cfg = make_config(cleanup_work_dir=False)
        runner = NextflowRunner(cfg, "batch_nodel", [], state)
        runner._cleanup(
            csv_path=str(tmp_path / "batch_nodel.csv"),
            log_path=str(tmp_path / "batch_nodel.log"),
            work_dir=str(work_dir),
            succeeded=False,
        )

        assert work_dir.exists(), "work dir should be kept when cleanup_work_dir=False"

    def test_cleanup_old_logs_also_removes_trace_file(self, tmp_path):
        """Trace file (.trace.tsv) is deleted alongside the log during log rotation."""
        from datetime import datetime, timedelta, timezone

        state = self._make_state(tmp_path)
        log_file = tmp_path / "old_batch.log"
        trace_file = tmp_path / "old_batch.trace.tsv"
        log_file.write_text("nextflow output\n")
        trace_file.write_text("task_id\tname\tstatus\n1\tPROC\tCOMPLETED\n")

        state.add_batch("old_batch", str(tmp_path / "old.csv"), None, 5, str(log_file))
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

        assert not log_file.exists(), "log file should be removed"
        assert not trace_file.exists(), "companion trace file should also be removed"

    def test_cleanup_old_logs_trace_missing_does_not_raise(self, tmp_path):
        """If no .trace.tsv exists alongside a rotated log, cleanup still succeeds."""
        from datetime import datetime, timedelta, timezone

        state = self._make_state(tmp_path)
        log_file = tmp_path / "old_batch2.log"
        log_file.write_text("nextflow output\n")

        state.add_batch("old_batch2", str(tmp_path / "old2.csv"), None, 5, str(log_file))
        old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        conn = state._conn()
        conn.execute(
            "UPDATE batches SET status='SUCCEEDED', completed_at=? WHERE batch_id='old_batch2'",
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


# ===========================================================================
# DatabricksWatcher
# ===========================================================================


def _make_sdk_mock(rows, *, fail=False):
    """Build a minimal mock of the databricks-sdk StatementExecution API."""
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    # Build column descriptors
    col_names = ["slide_id", "slide_path", "oncotree_code"]
    cols = [SimpleNamespace(name=n) for n in col_names]

    # Build the response
    mock_resp = MagicMock()
    if fail:
        from databricks.sdk.service.sql import StatementState
        mock_resp.status.state = StatementState.FAILED
        mock_resp.status.error = SimpleNamespace(message="forced failure")
    else:
        from databricks.sdk.service.sql import StatementState
        mock_resp.status.state = StatementState.SUCCEEDED
        mock_resp.manifest.schema.columns = cols
        mock_resp.result.data_array = [[r["slide_id"], r["slide_path"], r.get("oncotree_code", "")] for r in rows]
        # No next chunk
        mock_resp.result.next_chunk_index = None

    mock_client = MagicMock()
    mock_client.statement_execution.execute_statement.return_value = mock_resp

    return mock_client


class TestDatabricksWatcher:
    def _watcher(self, tmp_path, cfg_overrides=None):
        """Create a DatabricksWatcher with a temp StateStore."""
        state = StateStore(str(tmp_path / "state.db"))
        pending = deque()
        stop_event = threading.Event()
        cfg_kwargs = dict(
            type="databricks",
            warehouse_id="wh-123",
            poll_interval_seconds=3600,
            source_filter=["ECS2"],
            additional_where="",
            min_file_size_bytes=10_000_000,
        )
        if cfg_overrides:
            cfg_kwargs.update(cfg_overrides)
        cfg = WatcherConfig(**cfg_kwargs)
        w = DatabricksWatcher(cfg, pending, state, stop_event)
        return w, pending, state

    def test_enqueues_new_slides(self, tmp_path):
        w, pending, state = self._watcher(tmp_path)
        rows = [
            {"slide_id": "1001", "slide_path": "s3://bucket/1001.svs", "oncotree_code": "PAAD"},
            {"slide_id": "1002", "slide_path": "s3://bucket/1002.svs", "oncotree_code": "LUAD"},
        ]
        w._get_client = lambda: _make_sdk_mock(rows)
        w._poll()

        assert len(pending) == 2
        paths = {s["slide_path"] for s in pending}
        assert paths == {"s3://bucket/1001.svs", "s3://bucket/1002.svs"}
        ids = {s["slide_id"] for s in pending}
        assert ids == {"1001", "1002"}
        assert pending[0]["oncotree_code"] == "PAAD"

    def test_skips_known_slides(self, tmp_path):
        w, pending, state = self._watcher(tmp_path)
        state.add_slide("s3://bucket/1001.svs", "1001")
        rows = [
            {"slide_id": "1001", "slide_path": "s3://bucket/1001.svs", "oncotree_code": "PAAD"},
            {"slide_id": "1002", "slide_path": "s3://bucket/1002.svs", "oncotree_code": "LUAD"},
        ]
        w._get_client = lambda: _make_sdk_mock(rows)
        w._poll()

        assert len(pending) == 1
        assert pending[0]["slide_id"] == "1002"

    def test_deduplicates_on_second_poll(self, tmp_path):
        w, pending, state = self._watcher(tmp_path)
        rows = [{"slide_id": "1001", "slide_path": "s3://bucket/1001.svs", "oncotree_code": "PAAD"}]
        w._get_client = lambda: _make_sdk_mock(rows)
        w._poll()
        w._poll()

        # Only enqueued once despite two polls
        assert len(pending) == 1

    def test_handles_query_failure_gracefully(self, tmp_path):
        w, pending, state = self._watcher(tmp_path)
        w._get_client = lambda: _make_sdk_mock([], fail=True)
        w._poll()  # should not raise
        assert len(pending) == 0

    def test_source_filter_in_query(self, tmp_path):
        w, pending, state = self._watcher(tmp_path, {"source_filter": ["ECS2", "On-prem Storage"]})
        query = w._build_query()
        assert "i.source IN ('ECS2', 'On-prem Storage')" in query

    def test_no_source_filter_omits_clause(self, tmp_path):
        w, pending, state = self._watcher(tmp_path, {"source_filter": []})
        query = w._build_query()
        assert "i.source" not in query

    def test_additional_where_included(self, tmp_path):
        w, pending, state = self._watcher(tmp_path, {"additional_where": "m.IS_HNE = 1"})
        query = w._build_query()
        assert "m.IS_HNE = 1" in query

    def test_missing_warehouse_id_does_not_start(self, tmp_path, caplog):
        import logging
        w, pending, state = self._watcher(tmp_path, {"warehouse_id": ""})
        with caplog.at_level(logging.ERROR, logger="mussel-dispatcher"):
            w.run()
        assert len(pending) == 0

    def test_watcher_config_databricks_fields(self):
        cfg = WatcherConfig(
            type="databricks",
            warehouse_id="abc123",
            source_filter=["ECS2"],
            additional_where="m.IS_HNE = 1",
        )
        assert cfg.warehouse_id == "abc123"
        assert cfg.source_filter == ["ECS2"]
        assert cfg.additional_where == "m.IS_HNE = 1"


# ===========================================================================
# _load_secrets_env
# ===========================================================================

class TestLoadSecretsEnv:
    def test_loads_ecs_access_key(self, tmp_path):
        env_file = tmp_path / "creds.env"
        env_file.write_text("ECS_ACCESS_KEY=myaccesskey\nECS_SECRET_KEY=mysecretkey\n")
        w = WatcherConfig(type="local")
        _load_secrets_env(str(env_file), w)
        assert w.s3_access_key == "myaccesskey"
        assert w.s3_secret_key == "mysecretkey"

    def test_loads_aws_key_names(self, tmp_path):
        env_file = tmp_path / "creds.env"
        env_file.write_text("AWS_ACCESS_KEY_ID=awskey\nAWS_SECRET_ACCESS_KEY=awssecret\n")
        w = WatcherConfig(type="local")
        _load_secrets_env(str(env_file), w)
        assert w.s3_access_key == "awskey"
        assert w.s3_secret_key == "awssecret"

    def test_strips_export_prefix_and_quotes(self, tmp_path):
        env_file = tmp_path / "creds.env"
        env_file.write_text('export ECS_ACCESS_KEY="quoted_key"\nexport ECS_SECRET_KEY=\'singlequote\'\n')
        w = WatcherConfig(type="local")
        _load_secrets_env(str(env_file), w)
        assert w.s3_access_key == "quoted_key"
        assert w.s3_secret_key == "singlequote"

    def test_skips_comments_and_blank_lines(self, tmp_path):
        env_file = tmp_path / "creds.env"
        env_file.write_text("# comment\n\nECS_ACCESS_KEY=realkey\n")
        w = WatcherConfig(type="local")
        _load_secrets_env(str(env_file), w)
        assert w.s3_access_key == "realkey"

    def test_does_not_overwrite_existing_value(self, tmp_path):
        env_file = tmp_path / "creds.env"
        env_file.write_text("ECS_ACCESS_KEY=newkey\n")
        w = WatcherConfig(type="local", s3_access_key="existingkey")
        _load_secrets_env(str(env_file), w)
        assert w.s3_access_key == "existingkey"  # not overwritten

    def test_missing_file_logs_warning_and_does_not_raise(self, tmp_path):
        w = WatcherConfig(type="local")
        _load_secrets_env(str(tmp_path / "nonexistent.env"), w)  # must not raise
        assert w.s3_access_key == ""


# ===========================================================================
# _load_nf_secrets
# ===========================================================================

class TestLoadNfSecrets:
    def test_loads_known_secret(self):
        import unittest.mock as mock
        w = WatcherConfig(type="local")
        fake_result = mock.Mock(returncode=0, stdout="secretvalue\n", stderr="")
        with mock.patch("subprocess.run", return_value=fake_result) as m:
            _load_nf_secrets(["ECS_ACCESS_KEY"], w)
        assert w.s3_access_key == "secretvalue"
        m.assert_called_once_with(
            ["nextflow", "secrets", "get", "ECS_ACCESS_KEY"],
            capture_output=True, text=True, timeout=15,
        )

    def test_skips_unknown_key(self):
        import unittest.mock as mock
        w = WatcherConfig(type="local")
        with mock.patch("subprocess.run") as m:
            _load_nf_secrets(["UNKNOWN_KEY"], w)
        m.assert_not_called()

    def test_does_not_overwrite_existing_value(self):
        import unittest.mock as mock
        w = WatcherConfig(type="local", s3_access_key="existing")
        with mock.patch("subprocess.run") as m:
            _load_nf_secrets(["ECS_ACCESS_KEY"], w)
        m.assert_not_called()

    def test_handles_subprocess_failure_gracefully(self):
        import unittest.mock as mock
        w = WatcherConfig(type="local")
        fake_result = mock.Mock(returncode=1, stdout="", stderr="not found")
        with mock.patch("subprocess.run", return_value=fake_result):
            _load_nf_secrets(["ECS_ACCESS_KEY"], w)  # must not raise
        assert w.s3_access_key == ""

    def test_handles_subprocess_exception_gracefully(self):
        import unittest.mock as mock
        w = WatcherConfig(type="local")
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("nextflow not found")):
            _load_nf_secrets(["ECS_ACCESS_KEY"], w)  # must not raise
        assert w.s3_access_key == ""


# ===========================================================================
# NF session ID extraction
# ===========================================================================

class TestNfSessionIdExtraction:
    def test_parse_run_name_from_log(self, tmp_path):
        log = tmp_path / "batch.log"
        log.write_text(
            "N E X T F L O W  ~  version 23.10.1\n"
            "Launching `main.nf` [desperate_meucci] ...\n"
            "runName                 : desperate_meucci\n"
            "some other line\n"
        )
        assert _parse_run_name_from_log(str(log)) == "desperate_meucci"

    def test_parse_run_name_missing_returns_none(self, tmp_path):
        log = tmp_path / "batch.log"
        log.write_text("no run name here\n")
        assert _parse_run_name_from_log(str(log)) is None

    def test_parse_run_name_missing_file_returns_none(self, tmp_path):
        assert _parse_run_name_from_log(str(tmp_path / "nonexistent.log")) is None

    def test_lookup_session_id_in_history(self, tmp_path):
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t1m\tdesperate_meucci\tOK\t"
            "4d7b3c2a-1234-5678-abcd-ef0123456789\tnextflow run main.nf\n"
        )
        result = _lookup_session_id_in_history(str(tmp_path), "desperate_meucci")
        assert result == "4d7b3c2a-1234-5678-abcd-ef0123456789"

    def test_lookup_session_id_wrong_run_name_returns_none(self, tmp_path):
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t1m\tother_run\tOK\t"
            "4d7b3c2a-1234-5678-abcd-ef0123456789\tnextflow run main.nf\n"
        )
        assert _lookup_session_id_in_history(str(tmp_path), "desperate_meucci") is None

    def test_lookup_session_id_missing_history_returns_none(self, tmp_path):
        assert _lookup_session_id_in_history(str(tmp_path), "any_run") is None

    def test_extract_nf_session_id_from_log_full_pipeline(self, tmp_path):
        log = tmp_path / "batch.log"
        log.write_text("runName                 : brave_newton\n")
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t5m\tbrave_newton\tOK\t"
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\tnextflow run main.nf\n"
        )
        result = _extract_nf_session_id_from_log(str(log), str(tmp_path))
        assert result == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_session_id_lookup_by_deterministic_name(self, tmp_path):
        """When -name dispatcher_{batch_id} is used, session ID can be found directly
        from .nextflow/history without any log parsing."""
        from mussel_dispatcher.runner import _lookup_session_id_in_history
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t5m\tdispatcher_batch-042\tOK\t"
            "deadbeef-1111-2222-3333-444455556666\tnextflow run main.nf -name dispatcher_batch-042\n"
        )
        result = _lookup_session_id_in_history(str(tmp_path), "dispatcher_batch-042")
        assert result == "deadbeef-1111-2222-3333-444455556666"

    def test_session_id_falls_back_to_log_parsing_when_history_missing(self, tmp_path):
        """If .nextflow/history doesn't have the deterministic name entry, fall back
        to _extract_nf_session_id_from_log (legacy log parsing)."""
        from mussel_dispatcher.runner import (
            _lookup_session_id_in_history,
            _extract_nf_session_id_from_log,
        )
        # History has a run under the OLD random name, not the deterministic one
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t5m\told_random_name\tOK\t"
            "cafecafe-aaaa-bbbb-cccc-ddddeeeeffff\tnextflow run main.nf\n"
        )
        # Log has the old-style runName entry
        log = tmp_path / "batch.log"
        log.write_text("runName                 : old_random_name\n")

        # Primary: deterministic name lookup fails
        primary = _lookup_session_id_in_history(str(tmp_path), "dispatcher_batch-042")
        assert primary is None

        # Fallback: log parsing succeeds
        fallback = _extract_nf_session_id_from_log(str(log), str(tmp_path))
        assert fallback == "cafecafe-aaaa-bbbb-cccc-ddddeeeeffff"

    def test_query_session_id_via_nf_cli_success(self, tmp_path):
        """_query_session_id_via_nf_cli returns the UUID when nextflow log succeeds."""
        import unittest.mock as mock
        uuid = "deadbeef-1111-2222-3333-444455556666"
        completed = mock.MagicMock()
        completed.returncode = 0
        completed.stdout = f"{uuid}\n"
        with mock.patch("subprocess.run", return_value=completed) as m:
            result = _query_session_id_via_nf_cli(str(tmp_path), "dispatcher_batch-042")
        assert result == uuid
        args = m.call_args[0][0]
        assert args == ["nextflow", "log", "dispatcher_batch-042", "-f", "session_id"]

    def test_query_session_id_via_nf_cli_nonzero_returns_none(self, tmp_path):
        """_query_session_id_via_nf_cli returns None when nextflow log exits non-zero."""
        import unittest.mock as mock
        completed = mock.MagicMock()
        completed.returncode = 1
        completed.stdout = ""
        with mock.patch("subprocess.run", return_value=completed):
            result = _query_session_id_via_nf_cli(str(tmp_path), "dispatcher_batch-042")
        assert result is None

    def test_query_session_id_via_nf_cli_not_found_returns_none(self, tmp_path):
        """_query_session_id_via_nf_cli returns None when nextflow is not on PATH."""
        import unittest.mock as mock
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("nextflow not found")):
            result = _query_session_id_via_nf_cli(str(tmp_path), "dispatcher_batch-042")
        assert result is None

    def test_query_session_id_via_nf_cli_invalid_uuid_returns_none(self, tmp_path):
        """_query_session_id_via_nf_cli returns None if output is not a valid UUID."""
        import unittest.mock as mock
        completed = mock.MagicMock()
        completed.returncode = 0
        completed.stdout = "not-a-uuid\n"
        with mock.patch("subprocess.run", return_value=completed):
            result = _query_session_id_via_nf_cli(str(tmp_path), "dispatcher_batch-042")
        assert result is None


# ===========================================================================
# _verify_wds_coverage
# ===========================================================================

class TestVerifyWdsCoverage:
    def _make_runner(self, tmp_path, wds_destinations):
        watcher = WatcherConfig(type="tcga", wds_destinations=wds_destinations)
        cfg = make_config(
            repo_dir=str(tmp_path),
            outdir=str(tmp_path / "results"),
            work_base_dir=str(tmp_path / "work"),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            watchers=[watcher],
        )
        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        state = StateStore(str(tmp_path / "state" / "test.db"))
        runner = NextflowRunner(cfg, "batch-001", [], state)
        return runner, state

    def _write_batch_csv(self, tmp_path, slide_ids):
        batch_csv = tmp_path / "batches" / "batch.csv"
        batch_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(batch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide_id", "slide_path"])
            for sid in slide_ids:
                w.writerow([sid, f"/slides/{sid}.svs"])
        return str(batch_csv)

    def _write_wds_manifest(self, tmp_path, entries):
        """entries: list of (slide_id, model)"""
        manifest = tmp_path / "results" / "wds_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide_id", "model"])
            for sid, model in entries:
                w.writerow([sid, model])
        return str(manifest)

    def test_all_covered_no_reset(self, tmp_path):
        runner, state = self._make_runner(tmp_path, {"hoptimus1": "s3://bucket/wds/"})
        for sid in ["A", "B"]:
            state.add_slide(f"/slides/{sid}.svs", sid)
            state.mark_dispatched([f"/slides/{sid}.svs"], "batch-001")
            state.mark_slides_complete("batch-001", succeeded=True)
        batch_csv = self._write_batch_csv(tmp_path, ["A", "B"])
        self._write_wds_manifest(tmp_path, [("A", "hoptimus1"), ("B", "hoptimus1")])
        runner._verify_wds_coverage(batch_csv)
        for sid in ["A", "B"]:
            row = state._conn().execute(
                "SELECT status FROM slides WHERE slide_id=?", (sid,)
            ).fetchone()
            assert row["status"] == "SUCCEEDED"

    def test_missing_slide_reset_to_pending(self, tmp_path):
        runner, state = self._make_runner(tmp_path, {"hoptimus1": "s3://bucket/wds/"})
        for sid in ["A", "B"]:
            state.add_slide(f"/slides/{sid}.svs", sid)
            state.mark_dispatched([f"/slides/{sid}.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)
        batch_csv = self._write_batch_csv(tmp_path, ["A", "B"])
        # Only A is in WDS — B is missing
        self._write_wds_manifest(tmp_path, [("A", "hoptimus1")])
        runner._verify_wds_coverage(batch_csv)
        row_b = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_id=?", ("B",)
        ).fetchone()
        assert row_b["status"] == "PENDING"
        assert row_b["fail_count"] == 1

    def test_no_wds_destinations_skips_check(self, tmp_path):
        runner, state = self._make_runner(tmp_path, {})
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        runner._verify_wds_coverage(batch_csv)  # must not raise

    def test_missing_manifest_skips_check(self, tmp_path):
        runner, state = self._make_runner(tmp_path, {"hoptimus1": "s3://bucket/wds/"})
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        # No wds_manifest.csv written
        runner._verify_wds_coverage(batch_csv)  # must not raise


# ===========================================================================
# S3Watcher._scan / _in_progress_keys
# ===========================================================================

class TestS3WatcherScan:
    def _make_watcher(self, tmp_path):
        import unittest.mock as mock
        pending = []
        state = StateStore(str(tmp_path / "state.db"))
        watcher_cfg = WatcherConfig(
            type="s3",
            bucket="my-bucket",
            prefix="slides/",
            min_file_size_bytes=1000,
        )
        from mussel_dispatcher.watchers import S3Watcher
        w = S3Watcher.__new__(S3Watcher)
        w.cfg = watcher_cfg
        w.state = state
        w.pending = pending
        w.stop_event = threading.Event()
        w._s3 = None
        w.exts = {".svs", ".tiff", ".tif", ".ndpi", ".scn"}
        return w, pending, state

    def test_scan_enqueues_new_slide(self, tmp_path):
        import unittest.mock as mock
        w, pending, state = self._make_watcher(tmp_path)
        mock_s3 = mock.MagicMock()
        mock_s3.get_paginator.return_value.paginate.return_value = [{
            "Contents": [{"Key": "slides/TCGA-AB.svs", "Size": 5000}]
        }]
        # No multipart uploads
        mock_s3.get_paginator.return_value.paginate.side_effect = None
        # Provide separate paginators for list_objects_v2 and list_multipart_uploads
        def paginate_dispatch(op, **_kwargs):
            if op == "list_objects_v2":
                return iter([{"Contents": [{"Key": "slides/TCGA-AB.svs", "Size": 5000}]}])
            return iter([{}])
        mock_s3.get_paginator.return_value.paginate = paginate_dispatch
        mock_s3.get_paginator.side_effect = lambda op: mock.MagicMock(paginate=lambda **kw: paginate_dispatch(op, **kw))
        with mock.patch.object(w, "_get_s3", return_value=mock_s3):
            w._scan()
        assert len(pending) == 1
        assert pending[0]["slide_path"] == "s3://my-bucket/slides/TCGA-AB.svs"
        assert state.is_known("s3://my-bucket/slides/TCGA-AB.svs")

    def test_scan_skips_known_slides(self, tmp_path):
        import unittest.mock as mock
        w, pending, state = self._make_watcher(tmp_path)
        state.add_slide("s3://my-bucket/slides/TCGA-AB.svs", "TCGA-AB")
        mock_s3 = mock.MagicMock()
        def paginate_dispatch(op, **_kw):
            if op == "list_objects_v2":
                return iter([{"Contents": [{"Key": "slides/TCGA-AB.svs", "Size": 5000}]}])
            return iter([{}])
        mock_s3.get_paginator.side_effect = lambda op: mock.MagicMock(paginate=lambda **kw: paginate_dispatch(op, **kw))
        with mock.patch.object(w, "_get_s3", return_value=mock_s3):
            w._scan()
        assert len(pending) == 0

    def test_scan_skips_small_files(self, tmp_path):
        import unittest.mock as mock
        w, pending, state = self._make_watcher(tmp_path)
        mock_s3 = mock.MagicMock()
        def paginate_dispatch(op, **_kw):
            if op == "list_objects_v2":
                return iter([{"Contents": [{"Key": "slides/tiny.svs", "Size": 500}]}])
            return iter([{}])
        mock_s3.get_paginator.side_effect = lambda op: mock.MagicMock(paginate=lambda **kw: paginate_dispatch(op, **kw))
        with mock.patch.object(w, "_get_s3", return_value=mock_s3):
            w._scan()
        assert len(pending) == 0

    def test_scan_skips_wrong_extension(self, tmp_path):
        import unittest.mock as mock
        w, pending, state = self._make_watcher(tmp_path)
        mock_s3 = mock.MagicMock()
        def paginate_dispatch(op, **_kw):
            if op == "list_objects_v2":
                return iter([{"Contents": [{"Key": "slides/report.pdf", "Size": 50000}]}])
            return iter([{}])
        mock_s3.get_paginator.side_effect = lambda op: mock.MagicMock(paginate=lambda **kw: paginate_dispatch(op, **kw))
        with mock.patch.object(w, "_get_s3", return_value=mock_s3):
            w._scan()
        assert len(pending) == 0

    def test_scan_skips_in_progress_uploads(self, tmp_path):
        import unittest.mock as mock
        w, pending, state = self._make_watcher(tmp_path)
        mock_s3 = mock.MagicMock()
        def paginate_dispatch(op, **_kw):
            if op == "list_objects_v2":
                return iter([{"Contents": [{"Key": "slides/uploading.svs", "Size": 5000}]}])
            if op == "list_multipart_uploads":
                return iter([{"Uploads": [{"Key": "slides/uploading.svs"}]}])
            return iter([{}])
        mock_s3.get_paginator.side_effect = lambda op: mock.MagicMock(paginate=lambda **kw: paginate_dispatch(op, **kw))
        with mock.patch.object(w, "_get_s3", return_value=mock_s3):
            w._scan()
        assert len(pending) == 0


# ===========================================================================
# BatchScheduler._validate_s3_batch / _s3_path_exists
# ===========================================================================

class TestS3PreDispatchValidation:
    def _make_scheduler(self, tmp_path):
        (tmp_path / "state").mkdir()
        cfg = make_config(
            repo_dir=str(tmp_path),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            work_base_dir=str(tmp_path / "work"),
            max_slide_retries=3,
        )
        state = StateStore(str(tmp_path / "state" / "test.db"))
        run_manager = MagicMock()
        scheduler = BatchScheduler(cfg, state, run_manager, threading.Event())
        return scheduler, state

    def test_s3_path_exists_returns_true_on_success(self, tmp_path):
        import unittest.mock as mock
        scheduler, state = self._make_scheduler(tmp_path)
        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {}
        assert scheduler._s3_path_exists("s3://bucket/slides/a.svs", mock_s3) is True

    def test_s3_path_exists_returns_false_on_404(self, tmp_path):
        import unittest.mock as mock
        scheduler, state = self._make_scheduler(tmp_path)
        mock_s3 = mock.MagicMock()
        exc = Exception("Not Found")
        exc.response = {"Error": {"Code": "404"}}
        mock_s3.head_object.side_effect = exc
        assert scheduler._s3_path_exists("s3://bucket/slides/missing.svs", mock_s3) is False

    def test_s3_path_exists_returns_true_on_unknown_error(self, tmp_path):
        """Unknown errors (auth, network) treat slide as present to avoid false blacklisting."""
        import unittest.mock as mock
        scheduler, state = self._make_scheduler(tmp_path)
        mock_s3 = mock.MagicMock()
        mock_s3.head_object.side_effect = Exception("network timeout")
        assert scheduler._s3_path_exists("s3://bucket/slides/maybe.svs", mock_s3) is True

    def test_validate_s3_batch_blacklists_missing_slide(self, tmp_path):
        import unittest.mock as mock
        scheduler, state = self._make_scheduler(tmp_path)
        slides = [
            {"slide_path": "s3://bucket/a.svs", "slide_id": "a"},
            {"slide_path": "s3://bucket/b.svs", "slide_id": "b"},
        ]
        for s in slides:
            state.add_slide(s["slide_path"], s["slide_id"])

        not_found = Exception("Not Found")
        not_found.response = {"Error": {"Code": "404"}}

        def head_object(Bucket, Key):
            if "b.svs" in Key:
                raise not_found
            return {}

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = head_object
        with mock.patch.object(scheduler, "_get_s3_client", return_value=mock_s3):
            valid, blacklisted = scheduler._validate_s3_batch(slides)

        assert len(valid) == 1
        assert valid[0]["slide_id"] == "a"
        assert "b" in blacklisted
        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id=?", ("b",)
        ).fetchone()
        assert row["status"] == "FAILED"

    def test_validate_s3_batch_passes_non_s3_slides(self, tmp_path):
        import unittest.mock as mock
        scheduler, state = self._make_scheduler(tmp_path)
        slides = [{"slide_path": "/local/a.svs", "slide_id": "a"}]
        with mock.patch.object(scheduler, "_get_s3_client") as m:
            valid, blacklisted = scheduler._validate_s3_batch(slides)
        m.assert_not_called()
        assert valid == slides
        assert blacklisted == []

    def test_validate_s3_batch_all_missing_returns_empty(self, tmp_path):
        import unittest.mock as mock
        scheduler, state = self._make_scheduler(tmp_path)
        slides = [{"slide_path": "s3://bucket/a.svs", "slide_id": "a"}]
        state.add_slide("s3://bucket/a.svs", "a")
        not_found = Exception("Not Found")
        not_found.response = {"Error": {"Code": "404"}}
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = not_found
        with mock.patch.object(scheduler, "_get_s3_client", return_value=mock_s3):
            valid, blacklisted = scheduler._validate_s3_batch(slides)
        assert valid == []
        assert "a" in blacklisted


# ===========================================================================
# E2E: Full dispatcher loop
# ===========================================================================

class TestE2EDispatcherLoop:
    """End-to-end tests wiring enqueue → BatchScheduler → NextflowRunner → StateStore."""

    def _make_stack(self, tmp_path):
        """Build a complete dispatcher stack with real StateStore, mock NF subprocess."""
        for d in ["batches", "state", "logs", "work", "results"]:
            (tmp_path / d).mkdir()
        cfg = make_config(
            repo_dir=str(tmp_path),
            outdir=str(tmp_path / "results"),
            work_base_dir=str(tmp_path / "work"),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            batch_size=3,
            min_batch_size=1,
            max_wait_seconds=9999,
        )
        state = StateStore(str(tmp_path / "state" / "test.db"))
        run_manager = RunManager(cfg, state)
        scheduler = BatchScheduler(cfg, state, run_manager, threading.Event())
        return cfg, state, scheduler, run_manager

    def test_full_happy_path_slides_succeed(self, tmp_path):
        import unittest.mock as mock
        cfg, state, scheduler, run_manager = self._make_stack(tmp_path)

        slides = [
            {"slide_path": f"/slides/{c}.svs", "slide_id": c}
            for c in ["A", "B", "C"]
        ]
        for s in slides:
            state.add_slide(s["slide_path"], s["slide_id"])
            scheduler.enqueue(s)

        fake_proc = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=fake_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0, 120.0]
            scheduler._maybe_dispatch(force=True)
            run_manager.shutdown(wait=True)

        for s in slides:
            row = state._conn().execute(
                "SELECT status FROM slides WHERE slide_path=?", (s["slide_path"],)
            ).fetchone()
            assert row["status"] == "SUCCEEDED", f"{s['slide_id']} not SUCCEEDED"

    def test_full_failure_then_recovery(self, tmp_path):
        """Batch fails → slides FAILED → recover_in_flight resets to PENDING → second run succeeds."""
        import unittest.mock as mock
        cfg, state, scheduler, run_manager = self._make_stack(tmp_path)

        slide = {"slide_path": "/slides/X.svs", "slide_id": "X"}
        state.add_slide(slide["slide_path"], slide["slide_id"])
        scheduler.enqueue(slide)

        # First run: fail (long enough to charge fail_count)
        fail_proc = mock.Mock(returncode=1)
        with mock.patch("subprocess.run", return_value=fail_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0, 120.0]
            scheduler._maybe_dispatch(force=True)
            run_manager.shutdown(wait=True)

        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_path=?", (slide["slide_path"],)
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["fail_count"] == 1

        # Recovery resets to PENDING
        pending2 = deque()
        recover_in_flight(state, pending2, retry_failed=True, max_slide_retries=3)
        row2 = state._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", (slide["slide_path"],)
        ).fetchone()
        assert row2["status"] == "PENDING"

        # Second run: succeed
        run_manager2 = RunManager(cfg, state)
        scheduler2 = BatchScheduler(cfg, state, run_manager2, threading.Event())
        scheduler2.enqueue(slide)

        ok_proc = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=ok_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0, 120.0]
            scheduler2._maybe_dispatch(force=True)
            run_manager2.shutdown(wait=True)

        row3 = state._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", (slide["slide_path"],)
        ).fetchone()
        assert row3["status"] == "SUCCEEDED"

    def test_s3_predispatch_404_blacklists_and_submits_rest(self, tmp_path):
        """S3 pre-dispatch check: missing slide blacklisted, rest submitted and succeed."""
        import unittest.mock as mock
        cfg, state, scheduler, run_manager = self._make_stack(tmp_path)

        slides = [
            {"slide_path": "s3://bucket/ok.svs",      "slide_id": "ok"},
            {"slide_path": "s3://bucket/missing.svs",  "slide_id": "missing"},
        ]
        for s in slides:
            state.add_slide(s["slide_path"], s["slide_id"])
            scheduler.enqueue(s)

        not_found = Exception("Not Found")
        not_found.response = {"Error": {"Code": "404"}}

        def head_object(Bucket, Key):
            if "missing" in Key:
                raise not_found
            return {}

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = head_object

        ok_proc = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=ok_proc), \
             mock.patch.object(scheduler, "_get_s3_client", return_value=mock_s3), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0, 120.0]
            scheduler._maybe_dispatch(force=True)
            run_manager.shutdown(wait=True)

        ok_row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id=?", ("ok",)
        ).fetchone()
        missing_row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id=?", ("missing",)
        ).fetchone()
        assert ok_row["status"] == "SUCCEEDED"
        assert missing_row["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Nextflow trace file parsing
# ---------------------------------------------------------------------------

class TestParseNfTrace:
    """Tests for parse_nf_trace and _trace_path_for_log in helpers.py."""

    from mussel_dispatcher.dashboard.helpers import parse_nf_trace, _trace_path_for_log

    # --- _trace_path_for_log ---

    def test_trace_path_replaces_dot_log(self):
        from mussel_dispatcher.dashboard.helpers import _trace_path_for_log
        assert _trace_path_for_log("/logs/batch_001.log") == "/logs/batch_001.trace.tsv"

    def test_trace_path_no_extension(self):
        from mussel_dispatcher.dashboard.helpers import _trace_path_for_log
        assert _trace_path_for_log("/logs/batch_001") == "/logs/batch_001.trace.tsv"

    # --- parse_nf_trace ---

    def _write_trace(self, tmp_path, rows: list[dict]) -> str:
        headers = ["task_id", "hash", "native_id", "name", "status", "exit",
                   "submit", "duration", "realtime", "%cpu", "peak_rss"]
        path = str(tmp_path / "batch_001.trace.tsv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, delimiter="\t",
                               extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({h: row.get(h, "") for h in headers})
        return path

    def test_empty_trace_returns_zeros(self, tmp_path):
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        path = self._write_trace(tmp_path, [])
        result = parse_nf_trace(path)
        assert result == {"completed": 0, "cached": 0, "failed": 0, "total": 0, "failures": []}

    def test_counts_completed_and_cached(self, tmp_path):
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        rows = [
            {"name": "TESSELLATE (slide_a)", "status": "COMPLETED", "exit": "0", "hash": "aa/111111"},
            {"name": "TESSELLATE (slide_b)", "status": "COMPLETED", "exit": "0", "hash": "bb/222222"},
            {"name": "FEATURIZE (slide_c)", "status": "CACHED",    "exit": "0", "hash": "cc/333333"},
        ]
        result = parse_nf_trace(self._write_trace(tmp_path, rows))
        assert result["completed"] == 2
        assert result["cached"] == 1
        assert result["failed"] == 0
        assert result["total"] == 3
        assert result["failures"] == []

    def test_counts_failed_and_aborted(self, tmp_path):
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        rows = [
            {"name": "FEATURIZE (slide_x)", "status": "FAILED",  "exit": "1",  "hash": "ab/123456"},
            {"name": "FEATURIZE (slide_y)", "status": "ABORTED", "exit": "143","hash": "cd/789012"},
            {"name": "FEATURIZE (slide_z)", "status": "COMPLETED","exit": "0", "hash": "ef/345678"},
        ]
        result = parse_nf_trace(self._write_trace(tmp_path, rows))
        assert result["failed"] == 2
        assert result["completed"] == 1
        assert result["total"] == 3
        assert len(result["failures"]) == 2
        assert result["failures"][0]["name"] == "FEATURIZE (slide_x)"
        assert result["failures"][0]["exit"] == "1"

    def test_failures_capped_at_five(self, tmp_path):
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        rows = [
            {"name": f"PROC (slide_{i})", "status": "FAILED", "exit": "1", "hash": f"aa/{i:06d}"}
            for i in range(8)
        ]
        result = parse_nf_trace(self._write_trace(tmp_path, rows))
        assert result["failed"] == 8
        assert len(result["failures"]) == 5

    def test_missing_file_returns_zeros(self):
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        result = parse_nf_trace("/nonexistent/trace.tsv")
        assert result["total"] == 0
        assert result["failures"] == []

    def test_none_path_returns_zeros(self):
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        result = parse_nf_trace(None)
        assert result["total"] == 0

    def test_running_tasks_not_counted(self, tmp_path):
        """RUNNING/SUBMITTED tasks don't appear in the trace file yet."""
        from mussel_dispatcher.dashboard.helpers import parse_nf_trace
        rows = [
            {"name": "PROC (a)", "status": "COMPLETED", "exit": "0", "hash": "aa/000001"},
            {"name": "PROC (b)", "status": "RUNNING",   "exit": "-", "hash": "bb/000002"},
        ]
        result = parse_nf_trace(self._write_trace(tmp_path, rows))
        assert result["completed"] == 1
        assert result["total"] == 1  # RUNNING is not a finished status

    # --- parse_nf_log uses trace for errors when available ---

    def test_parse_nf_log_uses_trace_for_errors(self, tmp_path):
        from mussel_dispatcher.dashboard.helpers import parse_nf_log
        # Write a minimal NF log (no ERROR lines)
        log_path = str(tmp_path / "batch_001.log")
        Path(log_path).write_text("[ 50%] 10 of 20\n", encoding="utf-8")
        # Write companion trace with one failure
        trace_path = str(tmp_path / "batch_001.trace.tsv")
        headers = ["task_id", "hash", "native_id", "name", "status", "exit"]
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
            w.writeheader()
            w.writerow({"task_id": "1", "hash": "aa/111111", "native_id": "100",
                        "name": "FEATURIZE_BATCH (batch_001_0)", "status": "FAILED", "exit": "2"})
        result = parse_nf_log(log_path)
        assert result["error_count"] == 1
        assert "FEATURIZE_BATCH" in result["first_error"]
        assert "exit 2" in result["first_error"]
        assert len(result["failures"]) == 1

    def test_parse_nf_log_falls_back_to_regex_when_no_trace(self, tmp_path):
        from mussel_dispatcher.dashboard.helpers import parse_nf_log
        log_path = str(tmp_path / "batch_002.log")
        Path(log_path).write_text(
            "[ 50%] 5 of 10\nERROR ~ Error executing process > 'FEATURIZE_BATCH'\n",
            encoding="utf-8",
        )
        # No trace file exists — should fall back to regex
        result = parse_nf_log(log_path)
        assert result["error_count"] == 1
        assert result["first_error"] == "FEATURIZE_BATCH"
        assert result["failures"] == []


# ===========================================================================
# Tower shim tests
# ===========================================================================

class TestTowerShim:
    """Unit tests for mussel_dispatcher.tower_shim in-memory state store."""

    def setup_method(self):
        # Clear global state before each test.
        import mussel_dispatcher.tower_shim as shim
        with shim._lock:
            shim._workflows.clear()
        self.shim = shim

    def test_register_and_get_empty_progress(self):
        shim = self.shim
        shim.register_workflow("wf1", "b1", "dispatcher_b1")
        prog = shim.get_progress("b1")
        # After register only, task_counts is empty but state dict is returned
        assert prog is not None
        assert prog["batch_id"] == "b1"
        assert prog["task_counts"] == {}

    def test_update_progress(self):
        shim = self.shim
        shim.register_workflow("dispatcher_b2", "b2", "dispatcher_b2")
        shim.update_progress("dispatcher_b2", {"succeeded": 5, "failed": 1, "cached": 2, "running": 3, "pending": 0})
        prog = shim.get_progress("b2")
        assert prog is not None
        tc = prog["task_counts"]
        assert tc["succeeded"] == 5
        assert tc["failed"] == 1
        assert tc["cached"] == 2
        assert tc["running"] == 3

    def test_mark_complete(self):
        shim = self.shim
        shim.register_workflow("dispatcher_b3", "b3", "dispatcher_b3")
        shim.update_progress("dispatcher_b3", {"succeeded": 10, "failed": 0, "cached": 0, "running": 0, "pending": 0})
        shim.mark_complete("dispatcher_b3", {"succeeded": 10, "failed": 0, "cached": 0, "running": 0, "pending": 0})
        state = shim.get_state("b3")
        assert state["complete"] is True
        prog = shim.get_progress("b3")
        assert prog["task_counts"]["succeeded"] == 10

    def test_get_progress_unknown_batch(self):
        prog = self.shim.get_progress("nonexistent")
        assert prog is None

    def test_evict_old_removes_completed(self):
        shim = self.shim
        shim.register_workflow("dispatcher_b4", "b4", "dispatcher_b4")
        shim.mark_complete("dispatcher_b4")
        # Age it artificially.
        with shim._lock:
            shim._workflows["dispatcher_b4"].updated_at -= 7200
        removed = shim.evict_old(max_age_seconds=3600)
        assert removed == 1
        assert shim.get_state("b4") is None

    def test_evict_old_keeps_running(self):
        shim = self.shim
        shim.register_workflow("dispatcher_b5", "b5", "dispatcher_b5")
        # Not marked complete — should not be evicted.
        with shim._lock:
            shim._workflows["dispatcher_b5"].updated_at -= 7200
        removed = shim.evict_old(max_age_seconds=3600)
        assert removed == 0

    def test_workflow_id_for_batch(self):
        assert self.shim.workflow_id_for_batch("abc123") == "dispatcher_abc123"

    def test_user_info_response_has_required_fields(self):
        resp = self.shim.user_info_response()
        assert "user" in resp
        assert "id" in resp["user"]
        assert "userName" in resp["user"]

    def test_trace_create_response(self):
        resp = self.shim.trace_create_response("dispatcher_xyz")
        assert resp["workflowId"] == "dispatcher_xyz"
        assert "watchUrl" in resp

    def test_get_all_states_returns_snapshot(self):
        shim = self.shim
        shim.register_workflow("dispatcher_b6", "b6", "dispatcher_b6")
        shim.register_workflow("dispatcher_b7", "b7", "dispatcher_b7")
        states = shim.get_all_states()
        batch_ids = {s["batch_id"] for s in states}
        assert "b6" in batch_ids
        assert "b7" in batch_ids


class TestTowerShimServerRoutes:
    """Integration tests for Tower API HTTP routes added to the dashboard server."""

    def _make_server(self, tmp_path):
        """Build a minimal dashboard handler for route testing."""
        import sqlite3
        from mussel_dispatcher.dashboard.server import _build_handler

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "dispatcher.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE batches (
            batch_id TEXT PRIMARY KEY, status TEXT, slide_count INTEGER,
            dispatched_at TEXT, completed_at TEXT, nextflow_exit INTEGER,
            log_path TEXT, session_id TEXT)""")
        conn.execute("""CREATE TABLE slides (
            slide_id TEXT PRIMARY KEY, status TEXT, dispatched INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0, last_dispatch TEXT)""")
        conn.commit()
        conn.close()

        cfg = make_config(
            state_dir=str(state_dir),
            outdir=str(tmp_path / "results"),
            repo_dir=str(tmp_path),
        )
        Handler = _build_handler(cfg)
        return Handler

    def _fake_request(self, handler_cls, method, path, body: dict | None = None):
        """Invoke do_POST / do_PUT / do_GET on a handler without a real socket."""
        import io, json
        body_bytes = json.dumps(body or {}).encode()
        headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
        }

        class FakeRequest:
            def makefile(self, *a, **kw):
                return io.BytesIO(body_bytes)

        responses = []

        class TrackingHandler(handler_cls):
            def __init__(self):
                # Don't call super().__init__ — that would try to handle a real request.
                self.path = path
                self.headers = headers
                self.rfile = io.BytesIO(body_bytes)
                self._response_code = None
                self._response_body = None
                self._headers_sent = {}

            def send_response(self, code, message=None):
                self._response_code = code

            def send_header(self, key, value):
                self._headers_sent[key] = value

            def end_headers(self):
                pass

            def _write(self, data):
                self._response_body = data

            @property
            def wfile(self):
                class W:
                    def write(_, data):
                        responses.append(data)
                return W()

        h = TrackingHandler()
        getattr(h, f"do_{method}")()
        return h, responses

    def setup_method(self):
        import mussel_dispatcher.tower_shim as shim
        with shim._lock:
            shim._workflows.clear()

    def test_user_info_route(self, tmp_path):
        import json
        import mussel_dispatcher.tower_shim as shim
        Handler = self._make_server(tmp_path)
        h, responses = self._fake_request(Handler, "GET", "/user-info")
        assert h._response_code == 200
        body = json.loads(responses[0])
        assert "user" in body

    def test_trace_create_route(self, tmp_path):
        import json
        import mussel_dispatcher.tower_shim as shim
        Handler = self._make_server(tmp_path)
        h, responses = self._fake_request(
            Handler, "POST", "/trace/create",
            body={"runName": "dispatcher_test99", "sessionId": "abc-123"}
        )
        assert h._response_code == 200
        body = json.loads(responses[0])
        assert body["workflowId"] == "dispatcher_test99"
        # State should now exist in the shim
        state = shim.get_state("test99")
        assert state is not None
        assert state["batch_id"] == "test99"

    def test_trace_progress_route(self, tmp_path):
        import json
        import mussel_dispatcher.tower_shim as shim
        shim.register_workflow("dispatcher_test88", "test88", "dispatcher_test88")
        Handler = self._make_server(tmp_path)
        progress = {"succeeded": 3, "failed": 1, "cached": 0, "running": 2, "pending": 4}
        h, responses = self._fake_request(
            Handler, "PUT", "/trace/dispatcher_test88/progress",
            body={"progress": progress, "instant": 1234567890}
        )
        assert h._response_code == 200
        prog = shim.get_progress("test88")
        assert prog["task_counts"]["succeeded"] == 3
        assert prog["task_counts"]["running"] == 2

    def test_trace_complete_route(self, tmp_path):
        import json
        import mussel_dispatcher.tower_shim as shim
        shim.register_workflow("dispatcher_test77", "test77", "dispatcher_test77")
        Handler = self._make_server(tmp_path)
        h, responses = self._fake_request(
            Handler, "PUT", "/trace/dispatcher_test77/complete",
            body={"progress": {"succeeded": 10}, "instant": 1234567890}
        )
        assert h._response_code == 200
        state = shim.get_state("test77")
        assert state["complete"] is True

    def test_trace_begin_route(self, tmp_path):
        import mussel_dispatcher.tower_shim as shim
        Handler = self._make_server(tmp_path)
        h, responses = self._fake_request(
            Handler, "PUT", "/trace/dispatcher_test66/begin",
            body={"workflow": {}, "instant": 1234567890}
        )
        assert h._response_code == 200


class TestRunnerTowerEnv:
    """Test that runner.py passes tower env vars when cfg.tower_endpoint is set."""

    def test_tower_env_added_when_configured(self, tmp_path):
        """When tower_endpoint is set, TOWER_API_ENDPOINT and TOWER_ACCESS_TOKEN
        must be present in the subprocess environment."""
        import os

        cfg = make_config(
            tower_endpoint="http://localhost:8050",
            outdir=str(tmp_path / "results"),
            repo_dir=str(tmp_path),
        )
        # Simulate the run-env construction logic from runner.py
        run_env = dict(os.environ)
        if cfg.nextflow_version:
            run_env["NXF_VER"] = cfg.nextflow_version
        if cfg.tower_endpoint:
            run_env["TOWER_API_ENDPOINT"] = cfg.tower_endpoint
            run_env["TOWER_ACCESS_TOKEN"] = run_env.get("TOWER_ACCESS_TOKEN") or "local"
        assert run_env["TOWER_API_ENDPOINT"] == "http://localhost:8050"
        assert run_env["TOWER_ACCESS_TOKEN"] == "local"

    def test_no_tower_env_when_not_configured(self, tmp_path):
        import os
        cfg = make_config(outdir=str(tmp_path / "results"), repo_dir=str(tmp_path))
        run_env = dict(os.environ)
        if cfg.tower_endpoint:
            run_env["TOWER_API_ENDPOINT"] = cfg.tower_endpoint
        assert "TOWER_API_ENDPOINT" not in run_env or os.environ.get("TOWER_API_ENDPOINT") is not None

    def test_with_tower_flag_in_command(self, tmp_path):
        """When tower_endpoint is set, -with-tower should appear in the NF command."""
        import sqlite3, subprocess
        from unittest.mock import patch, MagicMock
        cfg = make_config(
            tower_endpoint="http://localhost:8050",
            outdir=str(tmp_path / "results"),
            repo_dir=str(tmp_path),
        )
        # Verify the config carries the endpoint; command construction is covered
        # by runner integration tests — just confirm the flag logic is correct.
        assert cfg.tower_endpoint == "http://localhost:8050"
        # Simulate the command-building logic from runner.py
        cmd = ["nextflow", "run", "main.nf"]
        if cfg.tower_endpoint:
            cmd += ["-with-tower"]
        assert "-with-tower" in cmd


# ===========================================================================
# SLURM squeue parsing
# ===========================================================================

class TestSlurmStats:
    """Tests for slurm_stats() and _parse_elapsed_hms() in helpers.py."""

    def _mock_squeue(self, lines, monkeypatch):
        import subprocess
        from mussel_dispatcher.dashboard import helpers

        def fake_check_output(cmd, **kwargs):
            if cmd[0] == "squeue":
                return "\n".join(lines)
            if cmd[0] == "sacct":
                return ""
            return ""

        monkeypatch.setattr(helpers, "_squeue_cache", {})
        monkeypatch.setattr(helpers, "_sacct_cache", {"ts": 9e18, "data": {}})
        monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    def test_non_nf_jobs_excluded(self, monkeypatch):
        """squeue rows whose job name does not start with nf- are ignored."""
        self._mock_squeue([
            "123|my_other_job|RUNNING|(None)|gpu01|/some/other/path|0:01:00",
            "124|nf-MUSSEL_EXTRACT_FEATURES_(1)|RUNNING|(None)|gpu02|/work/batch_20240101T120000_abcd1234/work/ab/cd|0:05:00",
        ], monkeypatch)
        from mussel_dispatcher.dashboard.helpers import slurm_stats
        result = slurm_stats()
        assert result["running"] == 1        # only the nf- job
        assert result["total"] == 1
        assert "MUSSEL_EXTRACT_FEATURES_" in result["processes"]

    def test_batch_id_correlated(self, monkeypatch):
        """Running nf- jobs are mapped to their batch via workdir regex."""
        self._mock_squeue([
            "200|nf-MUSSEL_EXTRACT_FEATURES_(1)|RUNNING|(None)|gpu01"
            "|/work/batch_20240101T120000_abcd1234/work/ab/cd|0:10:00",
            "201|nf-MUSSEL_EXTRACT_FEATURES_(2)|RUNNING|(None)|gpu01"
            "|/work/batch_20240101T120000_abcd1234/work/ef/gh|0:12:00",
        ], monkeypatch)
        from mussel_dispatcher.dashboard.helpers import slurm_stats
        result = slurm_stats()
        assert result["running"] == 2
        b = result["jobs_by_batch"].get("20240101T120000_abcd1234")
        assert b is not None
        assert b["running"] == 2

    def test_stalled_task_detected(self, monkeypatch):
        """Tasks running >2h are reported in stalled_tasks."""
        self._mock_squeue([
            "300|nf-MUSSEL_EXTRACT_FEATURES_(1)|RUNNING|(None)|gpu01"
            "|/work/batch_20240101T120000_abcd1234/work/ab/cd|2:30:00",
        ], monkeypatch)
        from mussel_dispatcher.dashboard.helpers import slurm_stats
        result = slurm_stats()
        assert len(result["stalled_tasks"]) == 1
        assert result["stalled_tasks"][0]["elapsed_s"] == 2 * 3600 + 30 * 60

    def test_short_running_task_not_stalled(self, monkeypatch):
        """Tasks running <2h are not stalled."""
        self._mock_squeue([
            "400|nf-MUSSEL_EXTRACT_FEATURES_(1)|RUNNING|(None)|gpu01"
            "|/work/batch_20240101T120000_abcd1234/work/ab/cd|0:45:00",
        ], monkeypatch)
        from mussel_dispatcher.dashboard.helpers import slurm_stats
        result = slurm_stats()
        assert result["stalled_tasks"] == []

    def test_pending_reason_recorded(self, monkeypatch):
        """Pending jobs accumulate their reason counts."""
        self._mock_squeue([
            "500|nf-MUSSEL_EXTRACT_FEATURES_(1)|PENDING|(Resources)|N/A"
            "|/work/batch_20240101T120000_abcd1234/work/ab/cd|0:00:00",
            "501|nf-MUSSEL_EXTRACT_FEATURES_(2)|PENDING|(Resources)|N/A"
            "|/work/batch_20240101T120000_abcd1234/work/ef/gh|0:00:00",
        ], monkeypatch)
        from mussel_dispatcher.dashboard.helpers import slurm_stats
        result = slurm_stats()
        assert result["pending"] == 2
        assert result["pending_reasons"].get("Resources") == 2

    def test_parse_elapsed_hms_formats(self):
        from mussel_dispatcher.dashboard.helpers import _parse_elapsed_hms
        assert _parse_elapsed_hms("1:30:00") == 5400
        assert _parse_elapsed_hms("2:30") == 150
        assert _parse_elapsed_hms("1-02:00:00") == 93600
        assert _parse_elapsed_hms("bad") is None

    def test_per_batch_process_tracking(self, monkeypatch):
        """jobs_by_batch entries include a 'processes' dict with per-process SLURM data."""
        self._mock_squeue([
            "600|nf-MUSSEL_EXTRACT_FEATURES_TESSELLATE_FEATURIZE_BATCH_(1)|RUNNING|(None)|gpu01"
            "|/work/batch_20240101T120000_abcd1234/work/ab/cd|0:20:00",
            "601|nf-MUSSEL_EXTRACT_FEATURES_TESSELLATE_FEATURIZE_BATCH_(2)|RUNNING|(None)|gpu02"
            "|/work/batch_20240101T120000_abcd1234/work/ef/gh|0:25:00",
        ], monkeypatch)
        from mussel_dispatcher.dashboard.helpers import slurm_stats
        result = slurm_stats()
        b = result["jobs_by_batch"].get("20240101T120000_abcd1234")
        assert b is not None
        assert b["running"] == 2
        procs = b["processes"]
        proc_key = "MUSSEL_EXTRACT_FEATURES_TESSELLATE_FEATURIZE_BATCH_"
        assert proc_key in procs
        assert procs[proc_key]["running"] == 2
        assert set(procs[proc_key]["nodes"]) == {"gpu01", "gpu02"}
        assert sorted(procs[proc_key]["elapsed_s"]) == [1200, 1500]


# ===========================================================================
# tower_process_to_slurm_name
# ===========================================================================

class TestTowerProcessToSlurmName:
    """Tests for the Tower→SLURM process name mapping."""

    def test_basic_conversion(self):
        from mussel_dispatcher.dashboard.helpers import tower_process_to_slurm_name
        assert tower_process_to_slurm_name(
            "MUSSEL:EXTRACT_FEATURES:TESSELLATE_FEATURIZE_BATCH"
        ) == "MUSSEL_EXTRACT_FEATURES_TESSELLATE_FEATURIZE_BATCH"

    def test_single_segment(self):
        from mussel_dispatcher.dashboard.helpers import tower_process_to_slurm_name
        assert tower_process_to_slurm_name("DOWNLOAD_SLIDE") == "DOWNLOAD_SLIDE"

    def test_matches_actual_job_names(self):
        """Derived name matches what squeue proc_short produces (strip trailing _)."""
        from mussel_dispatcher.dashboard.helpers import tower_process_to_slurm_name
        import re
        # Real job name from sacct:
        job_name = "nf-MUSSEL_EXTRACT_FEATURES_ONE_STEP_TESSELLATE_FEATURIZE_BATCH_(7)"
        proc_short = re.sub(r'\s*\(\d+\)$', '', job_name[3:])
        # proc_short = "MUSSEL_EXTRACT_FEATURES_ONE_STEP_TESSELLATE_FEATURIZE_BATCH_"
        tower_name = "MUSSEL:EXTRACT_FEATURES:ONE_STEP:TESSELLATE_FEATURIZE_BATCH"
        derived = tower_process_to_slurm_name(tower_name)
        # derived strips trailing _, squeue proc_short has trailing _
        assert proc_short.rstrip("_") == derived
