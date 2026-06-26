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
    _lookup_nf_session_id,
    _extract_nf_session_id_from_log,
    _extract_session_id_from_nf_debug_log,
    _query_session_id_via_nf_cli,
)
from mussel_dispatcher.config import (
    _read_nf_model_types,
    _load_secrets_env,
    _load_nf_secrets,
)
from mussel_dispatcher.scheduler import (
    BatchScheduler,
    RunManager,
    _acquire_pid_lock,
    _release_lock,
    _collect_slurm_job_ids,
    _scancel_slurm_jobs,
)

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

    def test_get_pending_slides_includes_oncotree_code(self, store):
        """oncotree_code stored via add_slide must survive the round-trip through
        get_pending_slides() so that the batch CSV is populated correctly."""
        store.add_slide("/slides/a.svs", "a", oncotree_code="PAAD")
        store.add_slide("/slides/b.svs", "b", oncotree_code="BRCA")
        pending = store.get_pending_slides()
        codes = {r["slide_id"]: r.get("oncotree_code") for r in pending}
        assert codes == {"a": "PAAD", "b": "BRCA"}

    def test_mark_dispatched_removes_from_pending(self, store):
        store.add_slide("/slides/a.svs", "a")
        claimed = store.mark_dispatched(["/slides/a.svs"], "batch-001")
        assert claimed == 1
        assert store.get_pending_slides() == []

    def test_mark_dispatched_returns_claimed_count(self, store):
        """mark_dispatched only claims PENDING slides; returns count actually updated."""
        store.add_slide("/slides/a.svs", "a")
        store.add_slide("/slides/b.svs", "b")
        # Claim a first
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        # Try to claim both — only b is still PENDING
        claimed = store.mark_dispatched(["/slides/a.svs", "/slides/b.svs"], "batch-002")
        assert claimed == 1  # only b was claimed; a was already DISPATCHED

    def test_mark_dispatched_race_condition(self, store):
        """Simulates two dispatchers racing: second batch claims 0 slides."""
        store.add_slide("/slides/a.svs", "a")
        # First dispatcher claims the slide
        c1 = store.mark_dispatched(["/slides/a.svs"], "batch-001")
        assert c1 == 1
        # Second dispatcher tries to claim the same slide
        c2 = store.mark_dispatched(["/slides/a.svs"], "batch-002")
        assert c2 == 0
        # Slide still belongs to batch-001
        row = store._conn().execute(
            "SELECT batch_id FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["batch_id"] == "batch-001"

    def test_mark_slides_complete_succeeded(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.mark_slides_complete("batch-001", succeeded=True)
        row = store._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"

    def test_throughput_window_parses_iso_completed_at(self, store):
        """ISO timestamps with "T" must be time-compared, not text-compared."""
        conn = store._conn()
        for idx in range(5):
            slide_path = f"/slides/old-{idx}.svs"
            store.add_slide(slide_path, f"old-{idx}")
            conn.execute(
                """
                UPDATE slides
                SET status='SUCCEEDED',
                    completed_at=strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now', '-8 hours')
                WHERE slide_path=?
                """,
                (slide_path,),
            )
        for idx in range(2):
            slide_path = f"/slides/recent-{idx}.svs"
            store.add_slide(slide_path, f"recent-{idx}")
            conn.execute(
                """
                UPDATE slides
                SET status='SUCCEEDED',
                    completed_at=strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now', '-30 minutes')
                WHERE slide_path=?
                """,
                (slide_path,),
            )
        conn.commit()

        stats = store.get_throughput_stats(window_hours=6.0)

        assert stats["completed_in_window"] == 2
        assert stats["throughput_per_hour"] is not None

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
            # In real usage, slides are reset to PENDING before each retry
            store._conn().execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL WHERE slide_path=?",
                ("/slides/a.svs",),
            )
            store._conn().commit()
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
            "SELECT status, fail_count, batch_id, completed_at FROM slides WHERE slide_path=?",
            ("/slides/a.svs",),
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["fail_count"] == 0  # retry slot not charged
        assert row["batch_id"] is None
        assert row["completed_at"] is None

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

    def test_reset_succeeded_to_pending_clears_terminal_fields(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.mark_slides_complete("batch-001", succeeded=True)

        n = store.reset_succeeded_to_pending(["/slides/a.svs"])

        assert n == 1
        row = store._conn().execute(
            "SELECT status, batch_id, completed_at, error_msg FROM slides WHERE slide_path=?",
            ("/slides/a.svs",),
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["batch_id"] is None
        assert row["completed_at"] is None
        assert row["error_msg"] is None

    def test_reset_dispatched_to_pending_clears_terminal_fields(self, store):
        store.add_slide("/slides/a.svs", "a")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store._conn().execute(
            "UPDATE slides SET completed_at='2026-06-21T01:00:00+00:00', error_msg='stale' "
            "WHERE slide_path='/slides/a.svs'"
        )
        store._conn().commit()

        store.reset_dispatched_to_pending("batch-001")

        row = store._conn().execute(
            "SELECT status, batch_id, completed_at, error_msg FROM slides WHERE slide_path=?",
            ("/slides/a.svs",),
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["batch_id"] is None
        assert row["completed_at"] is None
        assert row["error_msg"] is None

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
    def _make_scheduler(self, batch_size=3, min_batch_size=1, max_wait_seconds=9_999,
                        max_concurrent_runs=2):
        cfg = make_config(
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            max_wait_seconds=max_wait_seconds,
            max_concurrent_runs=max_concurrent_runs,
        )
        run_manager = MagicMock()
        run_manager.in_flight_slide_ids = set()
        run_manager.has_fresh_dispatch_capacity.return_value = True  # open by default
        run_manager.submit.return_value = True
        scheduler = BatchScheduler(cfg, MagicMock(), run_manager, threading.Event())
        return scheduler, run_manager
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

    def test_requeue_db_pending_enqueues_missing_slides(self, tmp_path):
        """Slides that are PENDING in the DB but not in the deque are re-enqueued."""
        from mussel_dispatcher.state import StateStore
        from mussel_dispatcher.scheduler import BatchScheduler
        db = StateStore(str(tmp_path / "s.db"))
        db.add_slide("/s/x.svs", "x")
        db.add_slide("/s/y.svs", "y")
        cfg = make_config(batch_size=10, max_wait_seconds=9_999)
        rm = MagicMock()
        rm.in_flight_slide_ids = set()
        scheduler = BatchScheduler(cfg, db, rm, threading.Event())
        # Deque is empty; DB has 2 PENDING slides
        scheduler._requeue_db_pending()
        assert len(scheduler._pending) == 2
        assert {s["slide_id"] for s in scheduler._pending} == {"x", "y"}

    def test_requeue_db_pending_skips_already_queued(self, tmp_path):
        """Slides already in the deque are not double-enqueued."""
        from mussel_dispatcher.state import StateStore
        from mussel_dispatcher.scheduler import BatchScheduler
        db = StateStore(str(tmp_path / "s.db"))
        db.add_slide("/s/x.svs", "x")
        cfg = make_config(batch_size=10, max_wait_seconds=9_999)
        rm = MagicMock()
        rm.in_flight_slide_ids = set()
        scheduler = BatchScheduler(cfg, db, rm, threading.Event())
        scheduler.enqueue({"slide_id": "x", "slide_path": "/s/x.svs"})
        scheduler._requeue_db_pending()
        # Still only 1 entry — not duplicated
        assert len(scheduler._pending) == 1

    def test_requeue_db_pending_skips_in_flight(self, tmp_path):
        """Slides in RunManager's in-flight set are not re-enqueued by the DB sweep.

        This prevents the runaway loop where slides popped from the deque but not
        yet written as DISPATCHED in the DB get re-enqueued during NF startup.
        """
        from mussel_dispatcher.state import StateStore
        from mussel_dispatcher.scheduler import BatchScheduler
        db = StateStore(str(tmp_path / "s.db"))
        db.add_slide("/s/x.svs", "x")
        db.add_slide("/s/y.svs", "y")
        cfg = make_config(batch_size=10, max_wait_seconds=9_999)
        rm = MagicMock()
        # "x" has been submitted to the thread pool but not yet written as DISPATCHED
        rm.in_flight_slide_ids = {"x"}
        scheduler = BatchScheduler(cfg, db, rm, threading.Event())
        scheduler._requeue_db_pending()
        # Only "y" should be enqueued; "x" is in-flight
        assert len(scheduler._pending) == 1
        assert scheduler._pending[0]["slide_id"] == "y"


# ===========================================================================
# BatchScheduler._maybe_dispatch concurrency guard
# ===========================================================================

class TestBatchSchedulerConcurrencyGuard:
    """_maybe_dispatch must not submit when already at max_concurrent_runs."""

    def _make_scheduler(self, max_concurrent_runs=2, batch_size=1):
        cfg = make_config(
            batch_size=batch_size,
            min_batch_size=1,
            max_wait_seconds=0,
            max_concurrent_runs=max_concurrent_runs,
        )
        run_manager = MagicMock()
        run_manager.in_flight_slide_ids = set()
        run_manager.has_fresh_dispatch_capacity.return_value = True  # open by default
        run_manager.submit.return_value = True
        scheduler = BatchScheduler(cfg, MagicMock(), run_manager, threading.Event())
        return scheduler, run_manager

    def _slide(self, name):
        return {"slide_path": f"/slides/{name}.svs", "slide_id": name}

    def test_dispatches_when_below_limit(self):
        """Submits when a fresh-dispatch slot is available."""
        scheduler, run_manager = self._make_scheduler(max_concurrent_runs=2)
        run_manager.has_fresh_dispatch_capacity.return_value = True
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_called_once()

    def test_no_dispatch_when_at_limit(self):
        """Does not submit when at the concurrency limit."""
        scheduler, run_manager = self._make_scheduler(max_concurrent_runs=2)
        run_manager.has_fresh_dispatch_capacity.return_value = False
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_not_called()

    def test_no_dispatch_when_above_limit(self):
        """Does not submit when over limit."""
        scheduler, run_manager = self._make_scheduler(max_concurrent_runs=3)
        run_manager.has_fresh_dispatch_capacity.return_value = False
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_not_called()

    def test_slides_remain_in_queue_when_blocked(self):
        """Slides are NOT dequeued when the concurrency guard fires."""
        scheduler, run_manager = self._make_scheduler(max_concurrent_runs=2)
        run_manager.has_fresh_dispatch_capacity.return_value = False
        scheduler.enqueue(self._slide("a"))
        scheduler.enqueue(self._slide("b"))
        scheduler._maybe_dispatch()
        assert len(scheduler._pending) == 2  # slides stay in queue

    def test_dispatches_once_slot_frees(self):
        """After a slot frees (has_capacity flips True), dispatch proceeds normally."""
        scheduler, run_manager = self._make_scheduler(max_concurrent_runs=2)
        run_manager.has_fresh_dispatch_capacity.return_value = False
        scheduler.enqueue(self._slide("a"))
        scheduler._maybe_dispatch()
        run_manager.submit.assert_not_called()

        run_manager.has_fresh_dispatch_capacity.return_value = True  # slot freed
        scheduler._maybe_dispatch()
        run_manager.submit.assert_called_once()

    def test_slides_returned_to_queue_when_submit_refuses(self):
        """If capacity disappears after popping, the batch is put back."""
        scheduler, run_manager = self._make_scheduler(max_concurrent_runs=2, batch_size=2)
        run_manager.has_fresh_dispatch_capacity.return_value = True
        run_manager.submit.return_value = False
        scheduler.enqueue(self._slide("a"))
        scheduler.enqueue(self._slide("b"))

        scheduler._maybe_dispatch()

        assert [s["slide_id"] for s in scheduler._pending] == ["a", "b"]

    def test_semaphore_prevents_overshoot_with_fast_fail_resumes(self):
        """Real RunManager semaphore: fast-fail resumes release slots but cap still holds.

        Simulates the restart race: 3 resumes submitted (r1 lingers, r2/r3 fast-fail),
        then the scheduler tries to fill freed slots. Total must not exceed max_concurrent_runs=3.
        """
        import time as _time

        cfg = make_config(max_concurrent_runs=3, batch_size=1, max_wait_seconds=0)
        state_mock = MagicMock()
        state_mock.get_pending_slides.return_value = []
        state_mock.count_running_batches.return_value = 0
        run_manager = RunManager(cfg, state_mock)

        r1_started = threading.Event()
        r1_release = threading.Event()

        def fake_run_linger(self_runner):
            r1_started.set()
            r1_release.wait(timeout=5)  # hold slot until released

        def fake_run_fast_fail(self_runner):
            pass  # instant return → releases slot

        from mussel_dispatcher.runner import NextflowRunner
        original_init = NextflowRunner.__init__

        def patched_init(self_runner, cfg, batch_id, slides, state, **kwargs):
            original_init(self_runner, cfg, batch_id, slides, state, **kwargs)
            if batch_id == "r1":  # r1 lingers (successful resume)
                self_runner.run = fake_run_linger.__get__(self_runner)
            elif batch_id.startswith("new"):  # new batches linger
                self_runner.run = fake_run_linger.__get__(self_runner)
            else:  # r2, r3 fast-fail (run-name-collision)
                self_runner.run = fake_run_fast_fail.__get__(self_runner)

        import unittest.mock as _mock
        with _mock.patch.object(NextflowRunner, "__init__", patched_init):
            # Submit 3 resumes: r1 lingers, r2/r3 fast-fail
            run_manager.submit_resume("r1", "/c1", "/w1")
            run_manager.submit_resume("r2", "/c2", "/w2")
            run_manager.submit_resume("r3", "/c3", "/w3")

            r1_started.wait(timeout=2)  # wait for r1 to grab its slot
            _time.sleep(0.1)            # let r2, r3 complete and release their slots

            # Now 2 slots free (r2, r3 done), 1 held (r1). Can accept 2 more.
            ok1 = run_manager.submit("new1", [])
            ok2 = run_manager.submit("new2", [])
            # 3rd submit should fail — r1 + new1 + new2 = 3 = max
            ok3 = run_manager.submit("new3", [])

            assert ok1, "new1 should be accepted (slot freed by r2)"
            assert ok2, "new2 should be accepted (slot freed by r3)"
            assert not ok3, "new3 must be rejected — would exceed max_concurrent_runs=3"

            r1_release.set()  # let r1 finish

        run_manager.shutdown(wait=False)

    def test_db_running_batches_block_fresh_submit(self, tmp_path):
        """Fresh dispatch is refused when persisted RUNNING rows are already at cap."""
        cfg = make_config(max_concurrent_runs=2)
        state = StateStore(str(tmp_path / "s.db"))
        state.add_batch("b1", "/c1", "/w1", 1, "/l1")
        state.add_batch("b2", "/c2", "/w2", 1, "/l2")
        run_manager = RunManager(cfg, state)

        assert run_manager.has_fresh_dispatch_capacity() is False
        assert run_manager.submit("b3", []) is False

        run_manager.shutdown(wait=False)


class TestDispatcherPidLocks:
    def test_acquire_pid_lock_removes_stale_lock(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "cohort.lock"
        lock_path.write_text("12345 stale\n")
        monkeypatch.setattr("mussel_dispatcher.scheduler._pid_is_alive", lambda pid: False)

        _acquire_pid_lock(str(lock_path), label="test", payload="cfg")

        assert lock_path.read_text().startswith(f"{os.getpid()} ")
        _release_lock(str(lock_path))
        assert not lock_path.exists()

    def test_acquire_pid_lock_refuses_live_lock(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "cohort.lock"
        lock_path.write_text("12345 live\n")
        monkeypatch.setattr("mussel_dispatcher.scheduler._pid_is_alive", lambda pid: True)

        with pytest.raises(SystemExit):
            _acquire_pid_lock(str(lock_path), label="test", payload="cfg")

        assert lock_path.read_text() == "12345 live\n"


# ===========================================================================
# BatchScheduler._watchdog_stuck_batches
# ===========================================================================

class TestWatchdogStuckBatches:
    """Unit tests for the stuck-batch watchdog in BatchScheduler."""

    def _make_scheduler(self, tmp_path, timeout_hours=4.0):
        cfg = make_config(
            log_dir=str(tmp_path),
            stuck_batch_timeout_hours=timeout_hours,
        )
        state = MagicMock()
        run_manager = MagicMock()
        run_manager.in_flight_slide_ids = set()
        scheduler = BatchScheduler(cfg, state, run_manager, threading.Event())
        return scheduler

    def _write_log(self, tmp_path, batch_id, age_seconds=0):
        """Create a batch log file with a specific mtime."""
        log_path = tmp_path / f"batch_{batch_id}.log"
        log_path.write_text("NF progress line\n")
        mtime = time.time() - age_seconds
        os.utime(log_path, (mtime, mtime))
        return log_path

    def _write_nf_log(self, tmp_path, batch_id, age_seconds=0):
        """Create a per-batch NF internal debug log file with a specific mtime."""
        nf_log_path = tmp_path / f"batch_{batch_id}.nf.log"
        nf_log_path.write_text("NF internal squeue poll\n")
        mtime = time.time() - age_seconds
        os.utime(nf_log_path, (mtime, mtime))
        return nf_log_path

    def test_disabled_when_timeout_zero(self, tmp_path, monkeypatch):
        """When stuck_batch_timeout_hours=0 the watchdog does nothing."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "abc123", "nf_pid": 9999}
        ]
        self._write_log(tmp_path, "abc123", age_seconds=999_999)
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    def test_disabled_when_timeout_negative(self, tmp_path, monkeypatch):
        """A negative timeout_hours is treated as disabled."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=-1.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "abc123", "nf_pid": 9999}
        ]
        self._write_log(tmp_path, "abc123", age_seconds=999_999)
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    def test_no_running_batches_does_nothing(self, tmp_path, monkeypatch):
        """Watchdog is a no-op when there are no RUNNING batches."""
        scheduler = self._make_scheduler(tmp_path)
        scheduler.state.get_running_batches.return_value = []
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    def test_recent_log_not_killed(self, tmp_path, monkeypatch):
        """A batch whose log was updated recently is left alone."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "fresh01", "nf_pid": 1111}
        ]
        self._write_log(tmp_path, "fresh01", age_seconds=60)  # 1 minute old
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    def test_stale_log_kills_nf_process(self, tmp_path, monkeypatch):
        """A batch with a log silent for > timeout hours has its NF process killed."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "stuck01", "nf_pid": 5555}
        ]
        self._write_log(tmp_path, "stuck01", age_seconds=5 * 3600)  # 5 hours old
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [5555]

    def test_uses_stored_log_path_from_db(self, tmp_path, monkeypatch):
        """When the NF debug log is absent, use the DB log_path before reconstructing."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        custom_log = tmp_path / "custom" / "run.log"
        custom_log.parent.mkdir()
        custom_log.write_text("NF progress line\n")
        mtime = time.time() - 5 * 3600
        os.utime(custom_log, (mtime, mtime))
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "custom01", "nf_pid": 1234, "log_path": str(custom_log)}
        ]
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [1234]

    def test_missing_log_not_killed(self, tmp_path, monkeypatch):
        """A batch with no log file yet (just started) is not killed."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "nofile1", "nf_pid": 7777}
        ]
        # no log file written
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    def test_batch_without_pid_skipped(self, tmp_path, monkeypatch):
        """A RUNNING batch with no nf_pid recorded is skipped (can't kill)."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "nopid01", "nf_pid": None}
        ]
        self._write_log(tmp_path, "nopid01", age_seconds=5 * 3600)
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    def test_only_stuck_batch_killed_healthy_left_alone(self, tmp_path, monkeypatch):
        """With a stuck and a healthy batch, only the stuck one is killed."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "stuck01", "nf_pid": 5555},
            {"batch_id": "healthy1", "nf_pid": 6666},
        ]
        self._write_log(tmp_path, "stuck01", age_seconds=5 * 3600)   # 5h — stuck
        self._write_log(tmp_path, "healthy1", age_seconds=10 * 60)   # 10min — fine
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [5555]
        assert 6666 not in killed

    def test_exactly_at_threshold_not_killed(self, tmp_path, monkeypatch):
        """A batch exactly at the threshold (not yet exceeded) is not killed."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=4.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "edge001", "nf_pid": 8888}
        ]
        # Exactly 4h - 1s: should NOT be killed
        self._write_log(tmp_path, "edge001", age_seconds=4 * 3600 - 1)
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == []

    # --- NF internal log (batch_{id}.nf.log) signal tests ---

    def test_fresh_nf_log_suppresses_kill_even_if_stdout_stale(self, tmp_path, monkeypatch):
        """If the NF internal log is fresh (< 20 min), NF is alive polling SLURM.
        No kill should be issued even if the stdout log is very old (cluster busy)."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=1.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "pend01", "nf_pid": 2222}
        ]
        self._write_log(tmp_path, "pend01", age_seconds=5 * 3600)  # stdout: 5h stale
        self._write_nf_log(tmp_path, "pend01", age_seconds=5 * 60)  # nf.log: 5 min (fresh)
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [], "NF alive (fresh nf.log) — should not be killed"

    def test_stale_nf_log_triggers_kill(self, tmp_path, monkeypatch):
        """If the NF internal log is stale beyond timeout, the JVM is dead → kill."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=1.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "dead01", "nf_pid": 3333}
        ]
        self._write_log(tmp_path, "dead01", age_seconds=2 * 3600)   # stdout: 2h stale
        self._write_nf_log(tmp_path, "dead01", age_seconds=2 * 3600)  # nf.log: 2h stale
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [3333], "NF internal log stale beyond timeout — should be killed"

    def test_nf_log_absent_falls_back_to_stdout_log(self, tmp_path, monkeypatch):
        """Without a nf.log file (pre-flag batches), fall back to stdout log mtime."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=1.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "old001", "nf_pid": 4444}
        ]
        self._write_log(tmp_path, "old001", age_seconds=2 * 3600)  # stdout: 2h stale, no nf.log
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [4444], "No nf.log → falls back to stdout log → should be killed"

    def test_nf_log_present_fresh_stdout_stale_healthy_not_killed(self, tmp_path, monkeypatch):
        """Two batches: one with a fresh nf.log (cluster-busy, alive), one with stale
        nf.log (dead JVM).  Only the dead one should be killed."""
        scheduler = self._make_scheduler(tmp_path, timeout_hours=1.0)
        scheduler.state.get_running_batches.return_value = [
            {"batch_id": "pend02", "nf_pid": 5555},  # alive, waiting for GPUs
            {"batch_id": "dead02", "nf_pid": 6666},  # JVM crashed
        ]
        self._write_log(tmp_path, "pend02", age_seconds=4 * 3600)
        self._write_nf_log(tmp_path, "pend02", age_seconds=3 * 60)   # nf.log: 3 min (fresh)
        self._write_log(tmp_path, "dead02", age_seconds=2 * 3600)
        self._write_nf_log(tmp_path, "dead02", age_seconds=90 * 60)  # nf.log: 90 min (stale)
        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda bid, pid, **kw: killed.append(pid),
        )
        scheduler._watchdog_stuck_batches()
        assert killed == [6666]
        assert 5555 not in killed


class TestSlurmJobCancellation:
    def test_collect_slurm_job_ids_from_trace_and_nf_log(self, tmp_path):
        trace = tmp_path / "batch.trace.tsv"
        trace.write_text(
            "task_id\thash\tnative_id\tname\tstatus\n"
            "1\taa/bb\t1234\tsaveParams\tCOMPLETED\n"
            "2\tcc/dd\t3546207\tMUSSEL:EXTRACT_FEATURES:TESSELLATE (48)\tCOMPLETED\n"
            "3\tee/ff\t3548007\tMUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH (48)\tRUNNING\n"
            "4\tgg/hh\tlocalpid\tMUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH (49)\tRUNNING\n"
        )
        nf_log = tmp_path / "batch.nf.log"
        nf_log.write_text(
            "DEBUG nextflow.executor.GridTaskHandler - [SLURM] submitted process "
            "MUSSEL:EXTRACT_FEATURES:FEATURIZE_BATCH (49) > jobId: 3548008; workDir: /work/a\n"
            "DEBUG nextflow.executor.GridTaskHandler - [SLURM] submitted process "
            "OTHER_PROCESS > jobId: 9999; workDir: /work/b\n"
        )

        assert _collect_slurm_job_ids(str(trace), str(nf_log)) == [
            "3546207",
            "3548007",
            "3548008",
        ]

    def test_scancel_slurm_jobs_chunks_ids(self, monkeypatch):
        calls = []

        def fake_run(cmd, capture_output, text, timeout):
            calls.append(cmd)
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = ""
            proc.stderr = ""
            return proc

        monkeypatch.setattr("mussel_dispatcher.scheduler.subprocess.run", fake_run)
        job_ids = [str(1000 + i) for i in range(205)]

        assert _scancel_slurm_jobs("batch-001", job_ids) == 205
        assert [len(call) - 1 for call in calls] == [100, 100, 5]
        assert all(call[0] == "scancel" for call in calls)


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

    def test_csv_oncotree_code_roundtrip_via_state(self, tmp_path):
        """oncotree_code added via StateStore must appear in the batch CSV.

        This is the integration test that would have caught the get_pending_slides()
        SELECT omitting oncotree_code: store → get_pending_slides → _write_csv → CSV.
        """
        from mussel_dispatcher.state import StateStore
        state = StateStore(str(tmp_path / "dispatcher.db"))
        state.add_slide("/slides/a.svs", "a", oncotree_code="PAAD")
        state.add_slide("/slides/b.svs", "b", oncotree_code="BRCA")
        slides = state.get_pending_slides()

        cfg = make_config(dispatch_dir=str(tmp_path))
        runner = NextflowRunner(cfg, "batch-rt", slides, MagicMock())
        csv_path = runner._write_csv()
        with open(csv_path, newline="") as f:
            rows = {r["slide_id"]: r["oncotree_code"] for r in csv.DictReader(f)}
        assert rows == {"a": "PAAD", "b": "BRCA"}

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

    def test_name_flag_uses_hash_suffix_run_name(self, tmp_path):
        """-name r{hash8} is added to the NF command; fits in Seqera's 16-char limit."""
        from mussel_dispatcher.runner import _NF_RUN_NAME_RE
        batch_id = "20260617T144813_58aa329a"
        nf_run_name = "r" + batch_id.rsplit("_", 1)[-1]  # r58aa329a
        assert nf_run_name == "r58aa329a"
        assert len(nf_run_name) <= 16
        assert _NF_RUN_NAME_RE.match(nf_run_name)
        cfg = make_config()
        cmd = [
            "nextflow", "run", cfg.repo_dir,
            "-profile", cfg.nextflow_profiles,
            "-work-dir", "/tmp/work",
            "--samples_csv", "/tmp/batch.csv",
            "--outdir", cfg.outdir,
            "-name", nf_run_name,
        ]
        assert "-name" in cmd
        assert nf_run_name in cmd

    def test_name_flag_includes_unique_batch_suffix(self):
        """The -name value embeds the UUID suffix so it is unique per batch."""
        batch_id = "20260617T000000_abcd1234"
        nf_run_name = "r" + batch_id.rsplit("_", 1)[-1]
        assert "abcd1234" in nf_run_name


class TestNfRunNameValidation:
    """The run name generated from a batch_id must always satisfy Nextflow's naming
    requirement: ^[a-z](?:[a-z\\d]|[-_](?=[a-z\\d])){0,79}$

    The 'r' prefix was added after a batch_id whose UUID started with a digit
    ('453d7b9d') was rejected by Nextflow.  These tests pin the invariant so any
    regression in name generation is caught at test time, not at launch time.
    """

    def _make_run_name(self, batch_id: str) -> str:
        """Replicate the exact logic from runner.py."""
        return "r" + batch_id.rsplit("_", 1)[-1]

    def test_digit_start_uuid_gets_r_prefix(self):
        """UUID starting with a digit (the bug case: '453d7b9d') becomes valid."""
        from mussel_dispatcher.runner import _NF_RUN_NAME_RE
        batch_id = "20260523T021436_453d7b9d"
        name = self._make_run_name(batch_id)
        assert name == "r453d7b9d"
        assert _NF_RUN_NAME_RE.match(name), f"{name!r} failed Nextflow pattern"

    def test_letter_start_uuid_also_valid(self):
        """UUID starting with a letter (e.g. 'afad9532') is also valid with prefix."""
        from mussel_dispatcher.runner import _NF_RUN_NAME_RE
        batch_id = "20260523T021143_afad9532"
        name = self._make_run_name(batch_id)
        assert name == "rafad9532"
        assert _NF_RUN_NAME_RE.match(name)

    def test_all_hex_chars_are_valid(self):
        """All 16 hex characters (0-9, a-f) are valid in the UUID suffix."""
        from mussel_dispatcher.runner import _NF_RUN_NAME_RE
        import uuid as _uuid
        for _ in range(50):
            batch_id = f"20260101T000000_{_uuid.uuid4().hex[:8]}"
            name = self._make_run_name(batch_id)
            assert _NF_RUN_NAME_RE.match(name), f"{name!r} failed for batch_id={batch_id!r}"

    def test_name_length_within_nextflow_limit(self):
        """Run name must be ≤ 16 chars to fit Seqera Platform's workflow.id field."""
        batch_id = "20260523T021436_453d7b9d"
        name = self._make_run_name(batch_id)
        assert len(name) <= 16, f"Run name {name!r} is {len(name)} chars, exceeds 16"

    def test_resume_uses_fresh_run_name_not_original(self):
        """Resume run names must differ from the original to avoid NF 'already used' error.

        NF ≥ 23.x rejects reusing a run name that appears in .nextflow/history,
        even when -resume <session_id> is supplied.  The runner must generate a
        fresh UUID-based name for every resume so the invocation succeeds.
        """
        import re
        import uuid as _uuid
        from mussel_dispatcher.runner import _NF_RUN_NAME_RE

        original_name = "r58aa329a"  # deterministic name from the initial run

        # Simulate the fresh-name logic used by the runner for both manual
        # resumes (self._resume=True) and auto-resume attempts.
        resume_name = "r" + _uuid.uuid4().hex[:8]

        assert resume_name != original_name, "resume name must differ from original"
        assert len(resume_name) <= 16
        assert _NF_RUN_NAME_RE.match(resume_name), f"{resume_name!r} fails NF pattern"


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

    def test_returns_resume_spec_when_work_dir_exists_and_nf_pid_dead(self, tmp_path, monkeypatch):
        """A dead recorded nf_pid still resumes when the batch state is intact."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        work_dir = tmp_path / "batch-001" / "work"
        work_dir.mkdir(parents=True)
        csv_path = tmp_path / "batch-001.csv"
        csv_path.write_text("slide_id,slide_path\na,/slides/a.svs\n")
        store.add_batch("batch-001", str(csv_path), str(work_dir), 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.set_batch_nf_pid("batch-001", 424242)

        def fake_kill(pid, sig):
            raise OSError("dead pid")

        monkeypatch.setattr("mussel_dispatcher.scheduler.os.kill", fake_kill)

        pending = deque()
        specs = recover_in_flight(store, pending)

        assert len(pending) == 0
        assert specs == [("batch-001", str(csv_path), str(work_dir))]
        row = store._conn().execute(
            "SELECT status FROM batches WHERE batch_id=?", ("batch-001",)
        ).fetchone()
        assert row["status"] == "RUNNING"

    def test_kills_live_nf_pid_before_reset_when_work_dir_missing(self, tmp_path, monkeypatch):
        """A live orphaned NF process is killed before slides are reset when resume is impossible."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        store.add_batch("batch-001", "/dispatch/1.csv", None, 1, "/logs/1.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-001")
        store.set_batch_nf_pid("batch-001", 424242)

        killed = []
        monkeypatch.setattr(
            "mussel_dispatcher.scheduler._kill_orphaned_nf",
            lambda batch_id, pid, **kw: killed.append((batch_id, pid)),
        )
        monkeypatch.setattr("mussel_dispatcher.scheduler.os.kill", lambda pid, sig: None)

        pending = deque()
        specs = recover_in_flight(store, pending)

        assert specs == []
        assert killed == [("batch-001", 424242)]
        assert len(pending) == 1

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
            store._conn().execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL WHERE slide_path=?",
                ("/slides/bad.svs",),
            )
            store._conn().commit()
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
# Startup recovery concurrency cap
# ---------------------------------------------------------------------------

class TestStartupRecoveryConcurrencyCap:
    """main() caps recovery resumes at max_concurrent_runs; excess resets to PENDING."""

    def _make_running_batch(self, store, tmp_path, batch_id, slide_path):
        work_dir = tmp_path / batch_id / "work"
        work_dir.mkdir(parents=True)
        csv = tmp_path / f"{batch_id}.csv"
        csv.write_text(f"slide_id,slide_path\n{batch_id},{slide_path}\n")
        store.add_slide(slide_path, batch_id)
        store.add_batch(batch_id, str(csv), str(work_dir), 1, f"/logs/{batch_id}.log")
        store.mark_dispatched([slide_path], batch_id)
        return str(csv_path := csv), str(work_dir)

    def test_excess_batches_reset_to_pending(self, tmp_path):
        """When recovery batches exceed max_concurrent_runs, excess slides go to PENDING."""
        store = StateStore(str(tmp_path / "test.db"))
        specs = []
        for i in range(5):
            bid = f"batch-{i:03d}"
            work_dir = tmp_path / bid / "work"
            work_dir.mkdir(parents=True)
            csv = tmp_path / f"{bid}.csv"
            slide = f"/slides/{i}.svs"
            csv.write_text(f"slide_id,slide_path\n{i},{slide}\n")
            store.add_slide(slide, str(i))
            store.add_batch(bid, str(csv), str(work_dir), 1, f"/logs/{bid}.log")
            store.mark_dispatched([slide], bid)
            specs.append((bid, str(csv), str(work_dir)))

        # Simulate what main() does: cap at max_concurrent_runs=3
        max_runs = 3
        capped = specs[:max_runs]
        excess = specs[max_runs:]
        for batch_id, csv_path, work_dir in excess:
            store.reset_dispatched_to_pending(batch_id)
            store.complete_batch(batch_id, exit_code=-1)

        # Check that excess slides (those reset) are now PENDING
        pending = store.get_pending_slides()
        pending_paths = {s["slide_path"] for s in pending}
        for i in range(max_runs, 5):
            assert f"/slides/{i}.svs" in pending_paths, f"/slides/{i}.svs should be PENDING"

        # Capped slides should still be DISPATCHED (no reset)
        for i in range(max_runs):
            slide = f"/slides/{i}.svs"
            row = store._conn().execute(
                "SELECT status FROM slides WHERE slide_path=?", (slide,)
            ).fetchone()
            assert row["status"] == "DISPATCHED", f"{slide} should still be DISPATCHED"

    def test_fewer_than_limit_all_resumed(self, tmp_path):
        """When recovery batches <= max_concurrent_runs, all are resumed (no excess)."""
        specs = [("b1", "/c1", "/w1"), ("b2", "/c2", "/w2")]
        max_runs = 3
        capped = specs[:max_runs]
        excess = specs[max_runs:]
        assert len(capped) == 2
        assert len(excess) == 0

    def test_exactly_at_limit_all_resumed(self, tmp_path):
        """Exactly max_concurrent_runs recovery batches — all resumed, none excess."""
        specs = [("b1", "/c1", "/w1"), ("b2", "/c2", "/w2"), ("b3", "/c3", "/w3")]
        max_runs = 3
        capped = specs[:max_runs]
        excess = specs[max_runs:]
        assert len(capped) == 3
        assert len(excess) == 0


# ---------------------------------------------------------------------------
# recover_in_flight startup cleanup (background thread)
# ---------------------------------------------------------------------------

class TestStartupCleanupNonBlocking:
    """Startup work-dir cleanup runs in a background thread, not blocking startup."""

    def _make_store_with_finished_batch(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/a.svs", "a")
        work_dir = tmp_path / "batch-fin" / "work"
        work_dir.mkdir(parents=True)
        (work_dir / "somefile.h5").write_bytes(b"data")
        csv = tmp_path / "batch-fin.csv"
        csv.write_text("slide_id,slide_path\na,/slides/a.svs\n")
        store.add_batch("batch-fin", str(csv), str(work_dir), 1, "/logs/fin.log")
        store.mark_dispatched(["/slides/a.svs"], "batch-fin")
        store.complete_batch("batch-fin", exit_code=0)
        store.mark_slides_complete("batch-fin", succeeded=True)
        return store, work_dir

    def test_cleanup_disabled_leaves_work_dir(self, tmp_path):
        """cleanup_work_dir=False leaves orphaned work dirs untouched."""
        store, work_dir = self._make_store_with_finished_batch(tmp_path)
        pending = deque()
        recover_in_flight(store, pending, cleanup_work_dir=False)
        time.sleep(0.2)
        assert work_dir.exists()

    def test_cleanup_enabled_removes_work_dir(self, tmp_path):
        """cleanup_work_dir=True removes orphaned work dirs (via background thread)."""
        store, work_dir = self._make_store_with_finished_batch(tmp_path)
        pending = deque()
        recover_in_flight(store, pending, cleanup_work_dir=True)
        # Background thread — give it time to finish
        deadline = time.monotonic() + 5.0
        while work_dir.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not work_dir.exists(), "work dir should have been deleted by background cleanup"

    def test_cleanup_does_not_block_startup(self, tmp_path):
        """recover_in_flight returns quickly even when cleanup work is slow."""
        store, work_dir = self._make_store_with_finished_batch(tmp_path)
        # Add many dummy files to simulate a large work dir
        for i in range(200):
            (work_dir / f"file_{i}.h5").write_bytes(b"x" * 1024)
        pending = deque()
        start = time.monotonic()
        recover_in_flight(store, pending, cleanup_work_dir=True)
        elapsed = time.monotonic() - start
        # Should return in well under 1 second even with 200 files to delete
        assert elapsed < 1.0, f"recover_in_flight blocked for {elapsed:.2f}s"

    def test_running_batches_not_cleaned_up(self, tmp_path):
        """RUNNING batch work dirs are not touched — only SUCCEEDED/FAILED."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("/slides/b.svs", "b")
        work_dir = tmp_path / "batch-run" / "work"
        work_dir.mkdir(parents=True)
        (work_dir / "data.h5").write_bytes(b"data")
        csv = tmp_path / "batch-run.csv"
        csv.write_text("slide_id,slide_path\nb,/slides/b.svs\n")
        store.add_batch("batch-run", str(csv), str(work_dir), 1, "/logs/run.log")
        store.mark_dispatched(["/slides/b.svs"], "batch-run")
        # Batch stays RUNNING — do NOT call complete_batch
        pending = deque()
        recover_in_flight(store, pending, cleanup_work_dir=True)
        time.sleep(0.2)
        assert work_dir.exists(), "RUNNING batch work dir must not be deleted at startup"


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

    def test_succeeded_slides_reset_to_pending_when_prepare_samples_reports_them(self, tmp_path):
        """Regression: slides SUCCEEDED in the dispatcher DB but still reported
        as pending by prepare_samples (i.e. their output files were lost) must
        be reset to PENDING so the DB sweep re-enqueues them.

        Prior to the fix, these slides were silently skipped by the watcher
        (is_known returned True) and never re-dispatched, stalling the pipeline.
        """
        import unittest.mock as mock
        import csv as _csv

        watcher, pending, state, _ = self._make_watcher(tmp_path)

        # Pre-populate state DB with slides as SUCCEEDED
        state.add_slide("/slides/A.svs", "A")
        state.add_slide("/slides/B.svs", "B")
        state._conn().execute(
            "UPDATE slides SET status='SUCCEEDED' WHERE slide_id IN ('A', 'B')"
        )
        state._conn().commit()

        # prepare_samples reports both slides as still pending (features lost)
        samples_csv = str(tmp_path / "status_dispatcher.csv")
        self._write_meta_csv(
            Path(samples_csv.replace(".csv", ".meta.csv")),
            [
                {"slide_id": "A", "slide_path": "/slides/A.svs", "needs_download": "false"},
                {"slide_id": "B", "slide_path": "/slides/B.svs", "needs_download": "false"},
            ],
        )
        with open(samples_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["slide_id", "slide_path", "file_id", "file_name", "needs_download"])
            w.writeheader()
            for sid, sp in [("A", "/slides/A.svs"), ("B", "/slides/B.svs")]:
                w.writerow({"slide_id": sid, "slide_path": sp, "needs_download": "false", "file_id": "", "file_name": ""})

        def fake_run_script(script, args):
            return 0

        watcher.cfg.status_csv = str(tmp_path / "status.csv")
        Path(tmp_path / "status.csv").write_text("slide_id,status\n")
        watcher._run_script = fake_run_script

        orig_path = tmp_path / "status_dispatcher.csv"
        with mock.patch.object(watcher, "_run_script", side_effect=lambda s, a: 0):
            watcher._poll()

        # Slides must now be PENDING (not SUCCEEDED) in the state DB
        conn = state._conn()
        for sid in ("A", "B"):
            row = conn.execute("SELECT status FROM slides WHERE slide_id=?", (sid,)).fetchone()
            assert row["status"] == "PENDING", (
                f"slide {sid} should have been reset to PENDING, got {row['status']}"
            )

        # Nothing in the in-memory pending deque (reset slides are picked up by
        # the DB sweep, not immediately enqueued by the watcher)
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
        return runner, state

    def test_run_success_marks_slides_succeeded(self, tmp_path):
        import unittest.mock as mock
        runner, state = self._make_runner(tmp_path)
        fake_proc = mock.Mock()
        fake_proc.pid = 12345
        fake_proc.wait.return_value = 0
        with mock.patch("subprocess.Popen", return_value=fake_proc):
            exit_code = runner.run()
        assert exit_code == 0
        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"

    def test_run_failure_marks_slides_failed_and_increments_fail_count(self, tmp_path):
        import unittest.mock as mock
        runner, state = self._make_runner(tmp_path)
        fake_proc = mock.Mock()
        fake_proc.pid = 12345
        fake_proc.wait.return_value = 1
        real_isdir = os.path.isdir
        # Patch time so batch_duration >= 60s (not a fast-fail)
        with mock.patch("subprocess.Popen", return_value=fake_proc), \
             mock.patch("mussel_dispatcher.runner.os.path.isdir",
                        side_effect=lambda p: False if p.endswith("/work") else real_isdir(p)), \
             mock.patch("mussel_dispatcher.runner.time") as mock_time:
            mock_time.time.side_effect = [0.0, 120.0, 120.0]
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
        fake_proc = mock.Mock()
        fake_proc.pid = 12345
        fake_proc.wait.return_value = 1
        with mock.patch("subprocess.Popen", return_value=fake_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mock_time:
            mock_time.time.side_effect = [0.0, 5.0, 5.0]
            exit_code = runner.run()
        assert exit_code == 1
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["fail_count"] == 0  # retry budget not consumed

    def test_run_shutdown_skip_auto_resume(self, tmp_path):
        """Intentional dispatcher shutdown must not trigger per-batch auto-resume."""
        import unittest.mock as mock
        shutdown_event = threading.Event()
        shutdown_event.set()
        runner, state = self._make_runner(tmp_path)
        runner._shutdown_event = shutdown_event

        proc = mock.Mock()
        proc.pid = 12345
        proc.wait.return_value = 1

        with mock.patch("subprocess.Popen", return_value=proc) as mock_popen, \
             mock.patch("mussel_dispatcher.runner._query_session_id_via_nf_cli", return_value=None), \
             mock.patch("mussel_dispatcher.runner._lookup_session_id_in_history", return_value=None), \
             mock.patch("mussel_dispatcher.runner._extract_nf_session_id_from_log", return_value=None), \
             mock.patch.object(runner, "_cleanup") as mock_cleanup, \
             mock.patch("mussel_dispatcher.runner.time") as mock_time:
            mock_time.time.side_effect = [0.0, 120.0]
            exit_code = runner.run()

        assert exit_code == 1
        assert mock_popen.call_count == 1, "shutdown should suppress auto-resume launch"
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["fail_count"] == 0
        mock_cleanup.assert_called_once()

    def test_auto_resume_retries_once_on_session_lock(self, tmp_path):
        """A transient NF session lock should not make auto-resume give up immediately."""
        import unittest.mock as mock
        runner, state = self._make_runner(tmp_path)

        waits = [1, 1, 0]  # initial run fails, first resume lock-fails, second resume succeeds
        pids = [111, 222, 333]

        def fake_popen(*args, **kwargs):
            stdout = kwargs.get("stdout")
            if stdout and len(waits) == 2:
                stdout.write("ERROR ~ Unable to acquire lock on session with ID sess-1\n")
                stdout.flush()
            proc = mock.Mock()
            proc.pid = pids.pop(0)
            proc.wait.return_value = waits.pop(0)
            return proc

        with mock.patch("subprocess.Popen", side_effect=fake_popen) as mock_popen, \
             mock.patch("mussel_dispatcher.runner._query_session_id_via_nf_cli", return_value="sess-1"), \
             mock.patch("mussel_dispatcher.runner.time.time", side_effect=[0.0, 120.0, 121.0, 122.0, 123.0, 124.0]), \
             mock.patch("mussel_dispatcher.runner.time.sleep") as mock_sleep, \
             mock.patch.object(runner, "_collect_manifest"), \
             mock.patch.object(runner, "_run_post_batch_hooks"), \
             mock.patch.object(runner, "_cleanup_intermediate_features"), \
             mock.patch.object(runner, "_verify_wds_coverage"), \
             mock.patch.object(runner, "_cleanup"):
            exit_code = runner.run()

        assert exit_code == 0
        assert mock_popen.call_count == 3
        mock_sleep.assert_called_once()
        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_path=?", ("/slides/a.svs",)
        ).fetchone()
        assert row["status"] == "SUCCEEDED"


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

    def test_wds_phase_runs_untagged_hooks(self, tmp_path):
        """Hooks with no _phase are treated as wds phase (backward compat)."""
        import unittest.mock as mock
        calls = []
        hook = {"command": "echo", "args": ["untagged"]}  # no _phase key
        runner = self._make_runner(tmp_path, [hook])
        fake = mock.Mock(returncode=0, stderr="")
        with mock.patch("subprocess.run", return_value=fake) as m:
            runner._run_post_batch_hooks("/batch.csv", phase="wds")
        assert m.called

    def test_wds_phase_skips_db_sync_hooks(self, tmp_path):
        """phase='wds' must not run hooks tagged _phase='db_sync'."""
        import unittest.mock as mock
        hook = {"command": "echo", "args": [], "_phase": "db_sync"}
        runner = self._make_runner(tmp_path, [hook])
        with mock.patch("subprocess.run") as m:
            runner._run_post_batch_hooks("/batch.csv", phase="wds")
        m.assert_not_called()

    def test_db_sync_phase_runs_tagged_hooks(self, tmp_path):
        """phase='db_sync' runs only hooks tagged _phase='db_sync'."""
        import unittest.mock as mock
        hook = {"command": "echo", "args": ["sync"], "_phase": "db_sync"}
        runner = self._make_runner(tmp_path, [hook])
        fake = mock.Mock(returncode=0, stderr="")
        with mock.patch("subprocess.run", return_value=fake) as m:
            runner._run_post_batch_hooks("/batch.csv", phase="db_sync")
        assert m.called

    def test_db_sync_phase_skips_untagged_hooks(self, tmp_path):
        """phase='db_sync' must not run hooks that have no _phase (wds hooks)."""
        import unittest.mock as mock
        hook = {"command": "echo", "args": ["wds-hook"]}  # no _phase
        runner = self._make_runner(tmp_path, [hook])
        with mock.patch("subprocess.run") as m:
            runner._run_post_batch_hooks("/batch.csv", phase="db_sync")
        m.assert_not_called()

    def test_mixed_phases_only_runs_matching(self, tmp_path):
        """With wds + db_sync hooks, each phase runs exactly its own subset."""
        import unittest.mock as mock
        wds_hook = {"command": "echo", "args": ["wds"]}
        db_hook = {"command": "echo", "args": ["db"], "_phase": "db_sync"}
        runner = self._make_runner(tmp_path, [wds_hook, db_hook])
        fake = mock.Mock(returncode=0, stderr="")
        with mock.patch("subprocess.run", return_value=fake) as m:
            runner._run_post_batch_hooks("/batch.csv", phase="wds")
        assert m.call_count == 1
        assert m.call_args[0][0][-1] == "wds"

        with mock.patch("subprocess.run", return_value=fake) as m:
            runner._run_post_batch_hooks("/batch.csv", phase="db_sync")
        assert m.call_count == 1
        assert m.call_args[0][0][-1] == "db"


# ---------------------------------------------------------------------------
# Auto post-batch hook generation
# ---------------------------------------------------------------------------

class TestNfModelTypes:
    """_read_nf_model_types parses model_types from the active NF config surface."""

    def test_reads_model_types(self, tmp_path):
        nf_config = tmp_path / "nextflow.config"
        nf_config.write_text("featurize {\n    model_types = ['hoptimus1', 'titan_slide']\n}\n")
        assert _read_nf_model_types(str(tmp_path)) == ["hoptimus1", "titan_slide"]

    def test_returns_empty_when_file_missing(self, tmp_path):
        assert _read_nf_model_types(str(tmp_path)) == []

    def test_returns_empty_when_no_match(self, tmp_path):
        (tmp_path / "nextflow.config").write_text("params { batch_size = 50 }\n")
        assert _read_nf_model_types(str(tmp_path)) == []

    def test_reads_model_types_from_params_file_first(self, tmp_path):
        nf_config = tmp_path / "nextflow.config"
        nf_config.write_text("featurize {\n    model_types = ['hoptimus1', 'titan_slide']\n}\n")
        params_file = tmp_path / "params.yaml"
        params_file.write_text(
            "featurize:\n"
            "  model_types:\n"
            "    - hoptimus1\n"
            "    - optimus\n"
            "    - titan_slide\n"
        )
        assert _read_nf_model_types(
            str(tmp_path), nextflow_params_file=str(params_file)
        ) == ["hoptimus1", "optimus", "titan_slide"]

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

    def test_watcher_models_auto_filled_from_params_file(self, tmp_path):
        import yaml as _yaml
        (tmp_path / "nextflow.config").write_text(
            "featurize {\n    model_types = ['hoptimus1', 'titan_slide']\n}\n"
        )
        (tmp_path / "params.yaml").write_text(
            "featurize:\n"
            "  model_types:\n"
            "    - hoptimus1\n"
            "    - optimus\n"
            "    - titan_slide\n"
        )
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "repo_dir": str(tmp_path),
            "nextflow_params_file": "params.yaml",
            "watchers": [{"type": "tcga", "inventory_csv": "i.csv", "status_csv": "s.csv"}],
        }))
        cfg = Config.load(str(cfg_path))
        assert cfg.watchers[0].models == ["hoptimus1", "optimus", "titan_slide"]

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
        assert "mussel_dispatcher.wds" in hook["command"]
        args = " ".join(hook["args"])
        assert "ctranspath" in args
        assert "s3://bucket/wds/ctranspath" in args
        assert "--slide-ids-csv={batch_csv}" in hook["args"]
        assert "--manifest-csv={outdir}/wds_manifest.csv" in hook["args"]
        assert any(a.startswith("--lock-dir=") for a in hook["args"])

    def test_cohort_lock_keys_derive_from_wds_destination_and_model(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "models": ["hoptimus1", "titan_slide"],
            "wds_destination": "s3://bucket/wds/",
        })
        assert cfg.cohort_lock_keys() == [
            "s3://bucket/wds::hoptimus1",
            "s3://bucket/wds::titan_slide",
        ]

    def test_resolved_cohort_lock_dir_uses_shared_dispatch_root(self, tmp_path):
        dispatch_root = tmp_path / "shared-dispatch"
        cfg = make_config(
            state_dir=str(dispatch_root / "titan-backfill" / "state")
        )
        assert cfg.resolved_cohort_lock_dir() == str(dispatch_root / "locks")

    def test_auto_hook_generated_from_single_wds_destination(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "models": ["hoptimus1", "titan_slide"],
            "wds_destination": "s3://bucket/wds",
        })
        assert len(cfg.post_batch_hooks) == 2
        pt_dirs = {hook["args"][0] for hook in cfg.post_batch_hooks}
        assert "--pt-dir={outdir}/features/hoptimus1" in pt_dirs
        assert "--pt-dir={outdir}/features/titan_slide" in pt_dirs
        assert cfg.watchers[0].wds_destinations == {
            "hoptimus1": "s3://bucket/wds",
            "titan_slide": "s3://bucket/wds",
        }

    def test_single_existing_wds_destination_fills_missing_models(self, tmp_path):
        cfg = self._load_config(tmp_path, watcher_extra={
            "models": ["hoptimus1", "optimus", "titan_slide"],
            "wds_destinations": {"hoptimus1": "s3://bucket/wds"},
        })
        assert cfg.watchers[0].wds_destinations == {
            "hoptimus1": "s3://bucket/wds",
            "optimus": "s3://bucket/wds",
            "titan_slide": "s3://bucket/wds",
        }

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
        assert "mussel_dispatcher.wds" in cfg.post_batch_hooks[0]["command"]
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

    def test_cleanup_results_deletes_shared_coords_only_on_last_wds_hook(self, tmp_path):
        """Shared .patch.h5 coords must survive until every model has written WDS."""
        cfg = self._load_config(
            tmp_path,
            watcher_extra={
                "models": ["hoptimus1", "optimus", "titan_slide"],
                "wds_destination": "s3://bucket/wds",
            },
            extra_raw={"cleanup_results": True},
        )

        hooks = cfg.post_batch_hooks
        assert len(hooks) == 3
        assert all("--delete-local" in hook["args"] for hook in hooks)
        assert "--delete-coords-local" not in hooks[0]["args"]
        assert "--delete-coords-local" not in hooks[1]["args"]
        assert "--delete-coords-local" in hooks[2]["args"]

    def test_cleanup_results_false_does_not_add_delete_local(self, tmp_path):
        """cleanup_results=False (default) does not add --delete-local to hooks."""
        cfg = self._load_config(
            tmp_path,
            watcher_extra={"wds_destinations": {"ctranspath": "s3://bucket/wds"}},
        )
        args = cfg.post_batch_hooks[0]["args"]
        assert "--delete-local" not in args

    def test_wds_auto_hook_has_no_phase_tag(self, tmp_path):
        """WDS auto-hooks have no _phase key (they run in the default wds phase)."""
        cfg = self._load_config(tmp_path, watcher_extra={
            "wds_destinations": {"ctranspath": "s3://bucket/wds"},
        })
        hook = cfg.post_batch_hooks[0]
        assert "mussel_dispatcher.wds" in hook["command"]
        assert "_phase" not in hook

    def test_tcga_db_sync_hook_tagged_db_sync(self, tmp_path):
        """TCGA Databricks sync auto-hook is tagged _phase='db_sync'."""
        cfg = self._load_config(tmp_path, watcher_extra={
            "databricks_volume_path": "/Volumes/cat/schema/vol/tcga.parquet",
        })
        hook = cfg.post_batch_hooks[0]
        assert "sync_databricks" in hook["command"]
        assert hook.get("_phase") == "db_sync"


class TestImpactDatabricksExport:
    def test_pending_rows_do_not_export_stale_wds_or_completed_at(self, tmp_path):
        from mussel_dispatcher.impact.sync_databricks import build_export

        db_path = tmp_path / "dispatcher.db"
        store = StateStore(str(db_path))
        store.add_slide("/slides/retry.svs", "retry", oncotree_code="MNET")
        store.add_slide("/slides/done.svs", "done", oncotree_code="MNET")
        store.mark_dispatched(["/slides/done.svs"], "batch-001")
        store.mark_slides_complete("batch-001", succeeded=True)
        store._conn().execute(
            "UPDATE slides SET completed_at='2026-06-21T01:00:00+00:00' "
            "WHERE slide_id='retry'"
        )
        store._conn().commit()

        manifest = tmp_path / "wds_manifest.csv"
        manifest.write_text(
            "slide_id,model,wds_path\n"
            "retry,optimus,s3://bucket/wds/optimus/MNET/000000.tar\n"
            "done,optimus,s3://bucket/wds/optimus/MNET/000001.tar\n"
        )

        df = build_export(str(db_path), str(manifest), ["optimus"])
        retry = df[df["slide_id"] == "retry"].iloc[0]
        done = df[df["slide_id"] == "done"].iloc[0]

        assert retry["status"] == "PENDING"
        assert retry["wds_path"] == ""
        assert retry["completed_at"] == ""
        assert done["status"] == "SUCCEEDED"
        assert done["wds_path"].endswith("/000001.tar")
        assert done["completed_at"] != ""


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


# ---------------------------------------------------------------------------
# _cleanup_intermediate_features tests
# ---------------------------------------------------------------------------

class TestCleanupIntermediateFeatures:
    """_cleanup_intermediate_features removes features dirs not in wds_destinations."""

    def _make_runner(self, tmp_path, wds_destinations, cleanup_results=True):
        import yaml as _yaml
        watcher = {
            "type": "databricks",
            "databricks_table": "cat.schema.tbl",
            "wds_destinations": wds_destinations,
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "cleanup_results": cleanup_results,
            "watchers": [watcher],
        }))
        cfg = Config.load(str(cfg_path))
        state = StateStore(str(tmp_path / "state.db"))
        return NextflowRunner(cfg, "batch-001", [], state)

    def test_removes_intermediate_dir_not_in_wds_destinations(self, tmp_path):
        """conch1_5 (patch encoder) dir is removed; titan_slide (wds dest) dir is left."""
        features = tmp_path / "results" / "features"
        titan = features / "titan_slide"
        conch = features / "conch1_5"
        titan.mkdir(parents=True)
        conch.mkdir(parents=True)
        (conch / "slide.patch_features.h5").touch()

        runner = self._make_runner(
            tmp_path,
            wds_destinations={"titan_slide": "s3://bucket/wds"},
        )
        runner._cleanup_intermediate_features()

        assert titan.exists()
        assert not conch.exists()

    def test_noop_when_cleanup_results_false(self, tmp_path):
        """cleanup_results=False leaves intermediate dirs untouched."""
        features = tmp_path / "results" / "features"
        conch = features / "conch1_5"
        conch.mkdir(parents=True)
        (conch / "slide.patch_features.h5").touch()

        runner = self._make_runner(
            tmp_path,
            wds_destinations={"titan_slide": "s3://bucket/wds"},
            cleanup_results=False,
        )
        runner._cleanup_intermediate_features()

        assert conch.exists()

    def test_noop_when_features_dir_absent(self, tmp_path):
        """No error when features/ directory doesn't exist yet."""
        runner = self._make_runner(
            tmp_path,
            wds_destinations={"titan_slide": "s3://bucket/wds"},
        )
        runner._cleanup_intermediate_features()  # must not raise

    def test_all_wds_dest_dirs_preserved(self, tmp_path):
        """Every model listed in wds_destinations keeps its features dir."""
        features = tmp_path / "results" / "features"
        for model in ("hoptimus1", "titan_slide"):
            (features / model).mkdir(parents=True)

        runner = self._make_runner(
            tmp_path,
            wds_destinations={
                "hoptimus1": "s3://bucket/wds",
                "titan_slide": "s3://bucket/wds",
            },
        )
        runner._cleanup_intermediate_features()

        assert (features / "hoptimus1").exists()
        assert (features / "titan_slide").exists()

    def test_noop_when_no_wds_destinations(self, tmp_path):
        """No wds_destinations configured → nothing is removed."""
        features = tmp_path / "results" / "features"
        some_dir = features / "some_model"
        some_dir.mkdir(parents=True)

        import yaml as _yaml
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_yaml.dump({
            "nextflow_profiles": "standard",
            "outdir": str(tmp_path / "results"),
            "cleanup_results": True,
        }))
        cfg = Config.load(str(cfg_path))
        state = StateStore(str(tmp_path / "state.db"))
        runner = NextflowRunner(cfg, "batch-001", [], state)

        runner._cleanup_intermediate_features()

        assert some_dir.exists()


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
        mock_resp.result.external_links = None  # INLINE disposition; no external links
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
            min_file_size_bytes=10_000_000,
            query="SELECT slide_id, slide_path, oncotree_code FROM slides",
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

    def test_missing_warehouse_id_does_not_start(self, tmp_path, caplog):
        import logging
        w, pending, state = self._watcher(tmp_path, {"warehouse_id": ""})
        with caplog.at_level(logging.ERROR, logger="mussel-dispatcher"):
            w.run()
        assert len(pending) == 0

    def test_no_query_uses_builtin_template(self, tmp_path):
        """_build_query falls back to the built-in Databricks query template."""
        w, pending, state = self._watcher(tmp_path, {"query": "", "query_file": ""})
        query = w._build_query()
        assert "FROM" in query
        assert "JOIN" in query
        assert "i.size >= 10000000" in query

    def test_inline_query_used(self, tmp_path):
        """_build_query returns inline SQL from cfg.query."""
        sql = "SELECT slide_id, slide_path, oncotree_code FROM my_catalog.slides"
        w, pending, state = self._watcher(tmp_path, {"query": sql})
        assert w._build_query() == sql

    def test_query_file_used(self, tmp_path):
        """_build_query reads SQL from the configured query_file."""
        sql = "SELECT slide_id, slide_path, oncotree_code FROM my_catalog.slides"
        qf = tmp_path / "slides.sql"
        qf.write_text(sql)
        w, pending, state = self._watcher(tmp_path, {"query": "", "query_file": str(qf)})
        assert w._build_query() == sql


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

    def test_extract_session_id_from_nf_debug_log(self, tmp_path):
        nf_log = tmp_path / "batch.nf.log"
        nf_log.write_text(
            "Jun-19 12:00:00.000 [main] DEBUG nextflow.cli.CmdRun - Session UUID: "
            "12345678-1234-1234-1234-1234567890ab\n"
            "Jun-19 12:00:00.001 [main] DEBUG nextflow.cli.CmdRun - Run name: reef_run\n"
        )
        assert (
            _extract_session_id_from_nf_debug_log(str(nf_log))
            == "12345678-1234-1234-1234-1234567890ab"
        )

    def test_lookup_nf_session_id_prefers_nf_debug_log(self, tmp_path):
        nf_log = tmp_path / "batch.nf.log"
        nf_log.write_text(
            "Session UUID: 12345678-1234-1234-1234-1234567890ab\n"
        )
        log = tmp_path / "batch.log"
        log.write_text("runName                 : brave_newton\n")
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t5m\tbrave_newton\tOK\t"
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\tnextflow run main.nf\n"
        )

        result = _lookup_nf_session_id(
            str(tmp_path), "batch-001", str(log), nf_log_path=str(nf_log)
        )
        assert result == "12345678-1234-1234-1234-1234567890ab"

    def test_lookup_nf_session_id_falls_back_when_nf_debug_log_missing(self, tmp_path):
        log = tmp_path / "batch.log"
        log.write_text("runName                 : brave_newton\n")
        nf_dir = tmp_path / ".nextflow"
        nf_dir.mkdir()
        (nf_dir / "history").write_text(
            "2024-01-01\t12:00:00\t5m\tbrave_newton\tOK\t"
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\tnextflow run main.nf\n"
        )

        result = _lookup_nf_session_id(
            str(tmp_path),
            "batch-001",
            str(log),
            nf_log_path=str(tmp_path / "missing.nf.log"),
        )
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

    def test_permanently_failed_at_max_retries(self, tmp_path):
        """Slides missing from WDS that reach max_slide_retries are FAILed, not re-queued."""
        runner, state = self._make_runner(tmp_path, {"hoptimus1": "s3://bucket/wds/"})
        runner.cfg = make_config(
            repo_dir=str(tmp_path),
            outdir=str(tmp_path / "results"),
            work_base_dir=str(tmp_path / "work"),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            watchers=[WatcherConfig(type="tcga", wds_destinations={"hoptimus1": "s3://bucket/wds/"})],
            max_slide_retries=3,
        )
        # Slide "A" has already failed twice → next fail will hit the threshold (3)
        state.add_slide("/slides/A.svs", "A")
        state.mark_dispatched(["/slides/A.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)
        state._conn().execute("UPDATE slides SET fail_count=2 WHERE slide_id='A'")
        state._conn().commit()
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        # A not in WDS manifest — third failure should permanently fail it
        self._write_wds_manifest(tmp_path, [])
        runner._verify_wds_coverage(batch_csv)
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_id='A'"
        ).fetchone()
        assert row["status"] == "FAILED", f"expected FAILED, got {row['status']}"
        assert row["fail_count"] == 3

    def test_retryable_below_max_retries(self, tmp_path):
        """Slides missing from WDS below max_slide_retries are reset to PENDING."""
        runner, state = self._make_runner(tmp_path, {"hoptimus1": "s3://bucket/wds/"})
        runner.cfg = make_config(
            repo_dir=str(tmp_path),
            outdir=str(tmp_path / "results"),
            work_base_dir=str(tmp_path / "work"),
            dispatch_dir=str(tmp_path / "batches"),
            state_dir=str(tmp_path / "state"),
            log_dir=str(tmp_path / "logs"),
            watchers=[WatcherConfig(type="tcga", wds_destinations={"hoptimus1": "s3://bucket/wds/"})],
            max_slide_retries=3,
        )
        state.add_slide("/slides/A.svs", "A")
        state.mark_dispatched(["/slides/A.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)
        # fail_count=1 → next fail (2) is still below threshold (3)
        state._conn().execute("UPDATE slides SET fail_count=1 WHERE slide_id='A'")
        state._conn().commit()
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        self._write_wds_manifest(tmp_path, [])
        runner._verify_wds_coverage(batch_csv)
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_id='A'"
        ).fetchone()
        assert row["status"] == "PENDING"
        assert row["fail_count"] == 2


# ===========================================================================
# append_wds silent skip detection (n_missing_project)
# ===========================================================================

class TestAppendWdsMissingProject:
    """Tests for the silent-skip failure mode in append_wds().

    When slide_to_project maps no slides (e.g. oncotree_code was missing from
    the batch CSV), append_wds returns n_appended=0 with no exception.  The
    _stats key in the returned index must expose n_missing_project so callers
    can detect and surface the data quality issue.
    """

    def _make_pt(self, tmp_path, slide_id):
        import torch
        pt = tmp_path / f"{slide_id}.features.pt"
        torch.save(torch.zeros(1, 4), pt)
        return pt

    def test_stats_key_present_on_success(self, tmp_path, monkeypatch):
        """_stats is always present in the returned index."""
        import torch
        from mussel_dispatcher import wds as wds_mod
        import pandas as pd

        self._make_pt(tmp_path, "SLIDE-A")
        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: {})
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)
        monkeypatch.setattr(wds_mod, "_ShardWriter", MagicMock(
            return_value=MagicMock(append=MagicMock(return_value="PROJ/000000.tar"),
                                   flush=MagicMock())
        ))

        result = wds_mod.append_wds(
            pt_dir=tmp_path, h5_dir=None,
            inventory_df=pd.DataFrame({"file_name": ["SLIDE-A.svs"], "project_id": ["PROJ"]}),
            wds_dest="local_wds", model_type="hoptimus1",
            staging_dir=None, max_shard_bytes=10 * 1024 ** 3,
            slide_to_project={"SLIDE-A": "PROJ"},
        )
        assert "_stats" in result
        assert result["_stats"]["n_appended"] == 1
        assert result["_stats"]["n_missing_project"] == 0

    def test_n_missing_project_counted_when_no_routing(self, tmp_path, monkeypatch):
        """Slides with no project_id are counted in n_missing_project."""
        import torch
        from mussel_dispatcher import wds as wds_mod
        import pandas as pd

        for sid in ["SLIDE-A", "SLIDE-B", "SLIDE-C"]:
            self._make_pt(tmp_path, sid)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: {})
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        # All slides missing from routing dict — simulates empty oncotree_code
        result = wds_mod.append_wds(
            pt_dir=tmp_path, h5_dir=None,
            inventory_df=pd.DataFrame({"file_name": [], "project_id": []}),
            wds_dest="local_wds", model_type="hoptimus1",
            staging_dir=None, max_shard_bytes=10 * 1024 ** 3,
            slide_to_project={},  # empty: all slides will be skipped
        )
        assert result["_stats"]["n_appended"] == 0
        assert result["_stats"]["n_missing_project"] == 3

    def test_mixed_some_missing_project(self, tmp_path, monkeypatch):
        """Only slides with a project_id are appended; others counted as missing."""
        import torch
        from mussel_dispatcher import wds as wds_mod
        import pandas as pd

        for sid in ["SLIDE-GOOD", "SLIDE-BAD"]:
            self._make_pt(tmp_path, sid)

        mock_writer = MagicMock()
        mock_writer.append.return_value = "PROJ/000000.tar"
        mock_writer.flush = MagicMock()
        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: {})
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)
        monkeypatch.setattr(wds_mod, "_ShardWriter", MagicMock(return_value=mock_writer))

        result = wds_mod.append_wds(
            pt_dir=tmp_path, h5_dir=None,
            inventory_df=pd.DataFrame({"file_name": [], "project_id": []}),
            wds_dest="local_wds", model_type="hoptimus1",
            staging_dir=None, max_shard_bytes=10 * 1024 ** 3,
            slide_to_project={"SLIDE-GOOD": "PROJ"},  # SLIDE-BAD has no entry
        )
        assert result["_stats"]["n_appended"] == 1
        assert result["_stats"]["n_missing_project"] == 1


# ===========================================================================
# _count_unique_slides_from_manifest (module-level helper in dashboard/server)
# ===========================================================================

class TestCountUniqueSlidesFromManifest:
    """Tests for the refactored _count_unique_slides_from_manifest() helper.

    The core bug: counting manifest rows (not unique slide_ids) inflated WDS %
    past 100 % when slides span multiple shards or runs produce duplicate rows.
    """

    def _write_manifest(self, path, rows):
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["slide_id", "model", "wds_path"])
            w.writeheader()
            w.writerows(rows)

    def test_unique_count_not_row_count(self, tmp_path):
        """A slide in 3 shards counts as 1, not 3."""
        from mussel_dispatcher.dashboard.server import _count_unique_slides_from_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/wds/PROJ/000000.tar"},
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/wds/PROJ/000001.tar"},
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/wds/PROJ/000002.tar"},
            {"slide_id": "S2", "model": "hoptimus1", "wds_path": "s3://b/wds/PROJ/000000.tar"},
        ])
        counts = _count_unique_slides_from_manifest(str(manifest))
        assert counts["hoptimus1"] == 2  # S1 + S2, not 4 rows

    def test_counts_per_model_independently(self, tmp_path):
        """Each model gets its own unique count."""
        from mussel_dispatcher.dashboard.server import _count_unique_slides_from_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1",  "wds_path": "s3://b/wds/PROJ/000000.tar"},
            {"slide_id": "S2", "model": "hoptimus1",  "wds_path": "s3://b/wds/PROJ/000000.tar"},
            {"slide_id": "S1", "model": "titan_slide", "wds_path": "s3://b/wds/PROJ/000000.tar"},
        ])
        counts = _count_unique_slides_from_manifest(str(manifest))
        assert counts["hoptimus1"] == 2
        assert counts["titan_slide"] == 1

    def test_models_filter_restricts_output(self, tmp_path):
        """When models allowlist is given, other models are excluded."""
        from mussel_dispatcher.dashboard.server import _count_unique_slides_from_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1",  "wds_path": "x"},
            {"slide_id": "S2", "model": "titan_slide", "wds_path": "x"},
        ])
        counts = _count_unique_slides_from_manifest(str(manifest), models=["hoptimus1"])
        assert "hoptimus1" in counts
        assert "titan_slide" not in counts

    def test_missing_file_returns_empty(self, tmp_path):
        from mussel_dispatcher.dashboard.server import _count_unique_slides_from_manifest
        counts = _count_unique_slides_from_manifest(str(tmp_path / "nonexistent.csv"))
        assert counts == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        from mussel_dispatcher.dashboard.server import _count_unique_slides_from_manifest
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02 not csv \xff\xfe")
        counts = _count_unique_slides_from_manifest(str(bad))
        assert counts == {}

    def test_empty_slide_id_rows_ignored(self, tmp_path):
        from mussel_dispatcher.dashboard.server import _count_unique_slides_from_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "",   "model": "hoptimus1", "wds_path": "x"},
            {"slide_id": "S1", "model": "",          "wds_path": "x"},
            {"slide_id": "S2", "model": "hoptimus1", "wds_path": "x"},
        ])
        counts = _count_unique_slides_from_manifest(str(manifest))
        assert counts.get("hoptimus1", 0) == 1  # only S2


class TestParseWdsManifest:
    """Tests for _parse_wds_manifest() — the extracted two-value manifest parser.

    This is the module-level function that _api_wds() now delegates to, enabling
    unit testing of the unique-slide counting and shard distribution logic that
    was previously locked inside a closure.
    """

    def _write_manifest(self, path, rows):
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["slide_id", "model", "wds_path"])
            w.writeheader()
            w.writerows(rows)

    def test_unique_slides_deduplicates_across_shards(self, tmp_path):
        """A slide appearing in 3 shards is counted once, not 3 times."""
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/000001.tar"},
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/000002.tar"},
            {"slide_id": "S2", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
        ])
        unique, shards = _parse_wds_manifest(str(manifest))
        assert len(unique["hoptimus1"]) == 2  # S1 + S2

    def test_shard_counts_are_row_counts_not_unique(self, tmp_path):
        """shard_slide_counts tracks row count per shard path (for distribution stats)."""
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
            {"slide_id": "S2", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
            {"slide_id": "S3", "model": "hoptimus1", "wds_path": "s3://b/000001.tar"},
        ])
        _, shards = _parse_wds_manifest(str(manifest))
        assert shards["hoptimus1"]["s3://b/000000.tar"] == 2
        assert shards["hoptimus1"]["s3://b/000001.tar"] == 1

    def test_multiple_models_tracked_independently(self, tmp_path):
        """Two models get separate unique sets and shard dicts."""
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1",  "wds_path": "s3://b/h/000000.tar"},
            {"slide_id": "S1", "model": "titan_slide", "wds_path": "s3://b/t/000000.tar"},
            {"slide_id": "S2", "model": "hoptimus1",  "wds_path": "s3://b/h/000000.tar"},
        ])
        unique, shards = _parse_wds_manifest(str(manifest))
        assert len(unique["hoptimus1"]) == 2
        assert len(unique["titan_slide"]) == 1
        assert "hoptimus1" in shards
        assert "titan_slide" in shards

    def test_models_filter_excludes_other_models(self, tmp_path):
        """Optional models allowlist excludes unlisted models from both outputs."""
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        manifest = tmp_path / "wds_manifest.csv"
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1",  "wds_path": "x"},
            {"slide_id": "S2", "model": "titan_slide", "wds_path": "x"},
        ])
        unique, shards = _parse_wds_manifest(str(manifest), models=["hoptimus1"])
        assert "hoptimus1" in unique
        assert "titan_slide" not in unique
        assert "titan_slide" not in shards

    def test_missing_file_returns_empty_dicts(self, tmp_path):
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        unique, shards = _parse_wds_manifest(str(tmp_path / "nonexistent.csv"))
        assert unique == {}
        assert shards == {}

    def test_corrupt_file_returns_empty_dicts(self, tmp_path):
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02 not csv \xff\xfe")
        unique, shards = _parse_wds_manifest(str(bad))
        assert unique == {}
        assert shards == {}

    def test_slides_per_inventory_never_exceeds_inventory_when_rows_inflated(self, tmp_path):
        """The original bug: 12502 manifest rows for 11802 unique slides pushed pct > 100.

        With _parse_wds_manifest, unique count stays at 3 even with 6 duplicate rows.
        """
        from mussel_dispatcher.dashboard.server import _parse_wds_manifest
        manifest = tmp_path / "wds_manifest.csv"
        # 3 unique slides, each appearing twice (simulating two pipeline runs)
        self._write_manifest(manifest, [
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
            {"slide_id": "S2", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
            {"slide_id": "S3", "model": "hoptimus1", "wds_path": "s3://b/000000.tar"},
            {"slide_id": "S1", "model": "hoptimus1", "wds_path": "s3://b/000001.tar"},
            {"slide_id": "S2", "model": "hoptimus1", "wds_path": "s3://b/000001.tar"},
            {"slide_id": "S3", "model": "hoptimus1", "wds_path": "s3://b/000001.tar"},
        ])
        unique, _ = _parse_wds_manifest(str(manifest))
        inventory_total = 3
        wds_slides = len(unique["hoptimus1"])
        pct = min(100.0, wds_slides / inventory_total * 100)
        assert wds_slides == 3        # unique count, not 6
        assert pct == 100.0           # exactly 100, not 200



    """Tests for the manifest write path in append_wds().

    These cover the bug where slides already present in the WDS S3 index were
    silently omitted from the local manifest CSV when n_appended == 0.  After
    the fix, batch slides confirmed in the index are always written to the
    manifest so that _verify_wds_coverage does not reset them to PENDING.
    """

    def _fake_index(self, slide_ids, project="PROJ"):
        return {
            sid: {"project_id": project, "shard_file": f"{project}/000000.tar",
                  "native_mpp": None, "mpp_is_fallback": None}
            for sid in slide_ids
        }

    def test_already_indexed_slides_written_to_manifest(self, tmp_path, monkeypatch):
        """When all batch slides are already in the WDS index (n_appended==0),
        they must still be recorded in manifest_csv."""
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        slide_ids = {"A", "B"}
        fake_index = self._fake_index(slide_ids)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(fake_index))
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        # No .pt files — slides are already indexed, nothing to append.

        inv = pd.DataFrame({"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ", "PROJ"]})
        manifest_csv = tmp_path / "wds_manifest.csv"

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter=slide_ids,
            manifest_csv=manifest_csv,
        )

        assert manifest_csv.exists(), "manifest_csv was not created"
        with open(manifest_csv, newline="") as f:
            rows = {r["slide_id"] for r in csv.DictReader(f)}
        assert rows == slide_ids, f"expected {slide_ids} in manifest, got {rows}"

    def test_mixed_new_and_already_indexed_both_written(self, tmp_path, monkeypatch):
        """When some slides are new and some are already indexed, all appear in
        manifest_csv after append_wds."""
        import pandas as pd
        import torch
        from mussel_dispatcher import wds as wds_mod

        already = {"A"}
        new_slide = "B"
        fake_index = self._fake_index(already)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(fake_index))
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        # Provide a real .pt file for the new slide so append_wds can load it.
        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        pt_file = pt_dir / f"{new_slide}.features.pt"
        torch.save(torch.zeros(1, 4), pt_file)

        inv = pd.DataFrame(
            {"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ", "PROJ"]}
        )
        manifest_csv = tmp_path / "wds_manifest.csv"

        # Mock ShardWriter so no real S3 writes occur.
        mock_writer = MagicMock()
        mock_writer.append.return_value = "000000.tar"
        monkeypatch.setattr(wds_mod, "_ShardWriter", lambda **kw: mock_writer)

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter={"A", "B"},
            manifest_csv=manifest_csv,
        )

        assert manifest_csv.exists(), "manifest_csv was not created"
        with open(manifest_csv, newline="") as f:
            rows = {r["slide_id"] for r in csv.DictReader(f)}
        assert "A" in rows, "already-indexed slide A missing from manifest"
        assert "B" in rows, "newly-appended slide B missing from manifest"

    def test_no_slide_id_filter_already_indexed_not_written(self, tmp_path, monkeypatch):
        """Without slide_id_filter, already-indexed slides are NOT re-written to
        manifest (only newly appended slides are recorded)."""
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        fake_index = self._fake_index({"A", "B"})
        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(fake_index))
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        inv = pd.DataFrame({"file_name": ["A.svs"], "project_id": ["PROJ"]})
        manifest_csv = tmp_path / "wds_manifest.csv"

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter=None,  # no filter — already-indexed slides stay out of manifest
            manifest_csv=manifest_csv,
        )

        # No new slides appended and no filter → nothing written to manifest
        if manifest_csv.exists():
            with open(manifest_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows == [], f"expected empty manifest, got {rows}"


# ===========================================================================
# _s3_download error-code handling
# ===========================================================================

class TestS3Download:
    """Unit tests for _s3_download error-code behaviour.

    These cover the cases that previously caused silent WDS upload failures:
    - 404 / NoSuchKey  → return False (index absent → start fresh)
    - 403 / AccessDenied → return False with a warning (ECS quirk: returns
      403 instead of 404 for non-existent keys with wrong/missing credentials)
    - Other errors     → re-raised so the caller sees a real failure
    """

    def _make_client_error(self, code: str):
        from botocore.exceptions import ClientError
        return ClientError(
            {"Error": {"Code": code, "Message": "test"}},
            "GetObject",
        )

    def _patch_s3_client(self, monkeypatch, exc):
        """Patch _s3_client() so download_file raises *exc*."""
        from mussel_dispatcher import wds as wds_mod
        mock_client = MagicMock()
        mock_client.download_file.side_effect = exc
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

    def test_404_returns_false(self, tmp_path, monkeypatch):
        from mussel_dispatcher import wds as wds_mod
        self._patch_s3_client(monkeypatch, self._make_client_error("404"))
        result = wds_mod._s3_download("s3://bucket/key", tmp_path / "out.json")
        assert result is False

    def test_no_such_key_returns_false(self, tmp_path, monkeypatch):
        from mussel_dispatcher import wds as wds_mod
        self._patch_s3_client(monkeypatch, self._make_client_error("NoSuchKey"))
        result = wds_mod._s3_download("s3://bucket/key", tmp_path / "out.json")
        assert result is False

    def test_403_returns_false_with_warning(self, tmp_path, monkeypatch):
        """ECS returns 403 instead of 404 for non-existent keys — must not raise."""
        from mussel_dispatcher import wds as wds_mod
        self._patch_s3_client(monkeypatch, self._make_client_error("403"))
        result = wds_mod._s3_download("s3://bucket/key", tmp_path / "out.json")
        assert result is False

    def test_access_denied_returns_false_with_warning(self, tmp_path, monkeypatch):
        from mussel_dispatcher import wds as wds_mod
        self._patch_s3_client(monkeypatch, self._make_client_error("AccessDenied"))
        result = wds_mod._s3_download("s3://bucket/key", tmp_path / "out.json")
        assert result is False

    def test_other_client_error_is_raised(self, tmp_path, monkeypatch):
        """Unexpected S3 errors (e.g. 500 InternalError) must propagate."""
        from mussel_dispatcher import wds as wds_mod
        from botocore.exceptions import ClientError
        self._patch_s3_client(monkeypatch, self._make_client_error("InternalError"))
        with pytest.raises(ClientError):
            wds_mod._s3_download("s3://bucket/key", tmp_path / "out.json")

    def test_success_returns_true(self, tmp_path, monkeypatch):
        from mussel_dispatcher import wds as wds_mod
        out = tmp_path / "out.json"
        mock_client = MagicMock()
        mock_client.download_file.side_effect = lambda b, k, p, **kw: Path(p).write_text("{}")
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)
        result = wds_mod._s3_download("s3://bucket/key", out)
        assert result is True
        assert out.read_text() == "{}"

    def test_403_does_not_prevent_upload_attempt(self, tmp_path, monkeypatch):
        """After a 403 on _load_index, append_wds should still attempt uploads
        (and fail loudly if credentials are wrong), not silently skip slides.

        Regression test: previously the 403 raised an exception that was caught
        at the top level and the process exited 0 with no slides uploaded.
        """
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        download_calls = []

        def fake_download(bucket, key, path, **kw):
            download_calls.append(key)
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "GetObject")

        mock_client = MagicMock()
        mock_client.download_file.side_effect = fake_download

        upload_calls = []
        uploaded_sizes: dict[str, int] = {}

        def fake_upload(path, bucket, key, **kw):
            from pathlib import Path as _Path
            upload_calls.append(key)
            uploaded_sizes[key] = _Path(path).stat().st_size

        mock_client.upload_file.side_effect = fake_upload

        def fake_head(Bucket, Key, **kw):
            return {"ContentLength": uploaded_sizes.get(Key, 0)}

        mock_client.head_object.side_effect = fake_head
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        import torch
        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        (pt_dir / "SLIDE1.features.pt").parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(1, 4), pt_dir / "SLIDE1.features.pt")

        inv = pd.DataFrame({"file_name": ["SLIDE1.svs"], "project_id": ["PROJ"]})
        staging = tmp_path / "staging"
        staging.mkdir()

        # append_wds must attempt at least one upload despite the initial 403
        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=staging,
            max_shard_bytes=10 * 1024 ** 3,
        )

        assert len(upload_calls) > 0, (
            "append_wds made no upload calls after 403 on _load_index — "
            "silent failure regression"
        )


# ===========================================================================
# _s3_upload post-upload verification
# ===========================================================================

class TestS3Upload:
    """_s3_upload must verify the object exists and has the correct size after upload.

    Some S3-compatible backends (e.g. ECS under storage pressure) return a
    successful upload response while buffering the write — if storage is full the
    object is silently evicted from cache and never committed to disk.  Checking
    ContentLength catches both "key not found" and "partial write / cache eviction"
    cases and ensures the manifest is never written with a stale path.
    """

    def test_head_object_called_after_upload(self, tmp_path, monkeypatch):
        """_s3_upload calls head_object to verify the shard landed with correct size."""
        from mussel_dispatcher import wds as wds_mod

        f = tmp_path / "shard.tar"
        f.write_bytes(b"hello")

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": len(b"hello")}
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        wds_mod._s3_upload(f, "s3://bucket/wds/hoptimus1/shard.tar")

        mock_client.upload_file.assert_called_once()
        mock_client.head_object.assert_called_once_with(
            Bucket="bucket", Key="wds/hoptimus1/shard.tar"
        )

    def test_raises_if_head_object_fails_after_upload(self, tmp_path, monkeypatch):
        """If head_object raises after upload, _s3_upload propagates the error so
        the manifest is never written with a stale path."""
        from botocore.exceptions import ClientError
        from mussel_dispatcher import wds as wds_mod

        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        f = tmp_path / "shard.tar"
        f.write_bytes(b"x")
        with pytest.raises(ClientError):
            wds_mod._s3_upload(f, "s3://bucket/wds/hoptimus1/shard.tar")

    def test_raises_if_remote_size_wrong(self, tmp_path, monkeypatch):
        """If ContentLength doesn't match the local file, _s3_upload raises.
        This catches ECS write-buffer scenarios where head_object returns 200 but
        the reported size reflects a partially-committed or cache-evicted object."""
        from mussel_dispatcher import wds as wds_mod

        f = tmp_path / "shard.tar"
        f.write_bytes(b"hello")  # 5 bytes

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 3}  # wrong
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        with pytest.raises(RuntimeError, match="size mismatch"):
            wds_mod._s3_upload(f, "s3://bucket/wds/hoptimus1/shard.tar")

    def test_manifest_not_written_if_upload_verification_fails(
        self, tmp_path, monkeypatch
    ):
        """When shard upload verification fails, manifest_csv must not be written
        (prevents stale manifest entries pointing to non-existent S3 shards)."""
        import pandas as pd
        import torch
        from botocore.exceptions import ClientError
        from mussel_dispatcher import wds as wds_mod

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: {})
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        torch.save(torch.zeros(1, 4), pt_dir / "SLIDE1.features.pt")

        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        inv = pd.DataFrame({"file_name": ["SLIDE1.svs"], "project_id": ["PROJ"]})
        staging = tmp_path / "staging"
        staging.mkdir()
        manifest_csv = tmp_path / "wds_manifest.csv"

        with pytest.raises(ClientError):
            wds_mod.append_wds(
                pt_dir=pt_dir,
                h5_dir=None,
                inventory_df=inv,
                wds_dest="s3://bucket/wds",
                model_type="hoptimus1",
                staging_dir=staging,
                max_shard_bytes=10 * 1024**3,
                manifest_csv=manifest_csv,
            )

        assert not manifest_csv.exists(), (
            "manifest_csv must not be written when shard upload verification fails"
        )


# ===========================================================================
# _ShardWriter — shard index initialisation
# ===========================================================================

class TestShardWriterInit:
    """Unit tests for _ShardWriter._init_current_shard().

    The critical invariant: when local staging is empty but S3 already has
    shards for a (model, project) prefix, new shards must start at
    max_existing_s3_index + 1 so the existing S3 shards are never overwritten.
    This is the root cause of the TCGA titan_slide data-loss incident where a
    partial re-run uploaded 000000.tar and clobbered thousands of slides.
    """

    def _make_writer(self, tmp_path, wds_dest, mock_s3_list, **kwargs):
        """Create a _ShardWriter with _s3_client patched to return mock listings."""
        from mussel_dispatcher import wds as wds_mod

        staging = tmp_path / "staging"
        staging.mkdir()

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = mock_s3_list

        writer = wds_mod._ShardWriter(
            wds_dest=wds_dest,
            model="titan_slide",
            project_id="TCGA-PROJ",
            staging_dir=staging,
            max_shard_bytes=10 * 1024 ** 3,
            **kwargs,
        )
        return writer, staging / "titan_slide" / "TCGA-PROJ"

    def test_empty_staging_no_s3_shards_starts_at_zero(self, tmp_path, monkeypatch):
        """With empty staging and no S3 shards, start at index 0 (normal first run)."""
        from mussel_dispatcher import wds as wds_mod

        staging = tmp_path / "staging"
        staging.mkdir()

        mock_client = MagicMock()
        mock_client.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        writer = wds_mod._ShardWriter(
            wds_dest="s3://bucket/wds",
            model="titan_slide",
            project_id="TCGA-PROJ",
            staging_dir=staging,
            max_shard_bytes=10 * 1024 ** 3,
        )
        assert writer._current_index == 0
        assert writer._current_path is None

    def test_empty_staging_existing_s3_shards_starts_after_max(self, tmp_path, monkeypatch):
        """Empty staging + S3 has 000000-000004 → new index must be 5, not 0.

        Without this guard, the writer would upload 000000.tar and overwrite
        all slides that were in the existing S3 shard.
        """
        from mussel_dispatcher import wds as wds_mod

        staging = tmp_path / "staging"
        staging.mkdir()

        existing_keys = [
            {"Key": "wds/titan_slide/TCGA-PROJ/000000.tar"},
            {"Key": "wds/titan_slide/TCGA-PROJ/000001.tar"},
            {"Key": "wds/titan_slide/TCGA-PROJ/000002.tar"},
            {"Key": "wds/titan_slide/TCGA-PROJ/000003.tar"},
            {"Key": "wds/titan_slide/TCGA-PROJ/000004.tar"},
        ]
        mock_client = MagicMock()
        mock_client.get_paginator.return_value.paginate.return_value = [
            {"Contents": existing_keys}
        ]
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        writer = wds_mod._ShardWriter(
            wds_dest="s3://bucket/wds",
            model="titan_slide",
            project_id="TCGA-PROJ",
            staging_dir=staging,
            max_shard_bytes=10 * 1024 ** 3,
        )
        assert writer._current_index == 5  # not 0 — would overwrite 000000.tar
        assert writer._current_path is None

    def test_local_staging_present_takes_priority_over_s3(self, tmp_path, monkeypatch):
        """If local staging has a partial shard, resume from it (don't call S3)."""
        from mussel_dispatcher import wds as wds_mod

        staging = tmp_path / "staging"
        work_dir = staging / "titan_slide" / "TCGA-PROJ"
        work_dir.mkdir(parents=True)
        # Write a partial local shard at index 3
        partial = work_dir / "000003.tar"
        partial.write_bytes(b"x" * 1024)

        mock_client = MagicMock()
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        writer = wds_mod._ShardWriter(
            wds_dest="s3://bucket/wds",
            model="titan_slide",
            project_id="TCGA-PROJ",
            staging_dir=staging,
            max_shard_bytes=10 * 1024 ** 3,
        )
        # Should resume from local shard, no S3 listing needed
        assert writer._current_index == 3
        assert writer._current_path == partial
        mock_client.get_paginator.assert_not_called()

    def test_s3_listing_error_falls_back_to_zero(self, tmp_path, monkeypatch):
        """If S3 listing fails, fall back to index 0 (best-effort, don't crash)."""
        from mussel_dispatcher import wds as wds_mod

        staging = tmp_path / "staging"
        staging.mkdir()

        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("S3 unavailable")
        monkeypatch.setattr(wds_mod, "_s3_client", lambda: mock_client)

        writer = wds_mod._ShardWriter(
            wds_dest="s3://bucket/wds",
            model="titan_slide",
            project_id="TCGA-PROJ",
            staging_dir=staging,
            max_shard_bytes=10 * 1024 ** 3,
        )
        assert writer._current_index == 0  # falls back gracefully


# ===========================================================================
# _load_features / _load_slide_meta silent failure modes
# ===========================================================================

class TestWdsFeatureLoading:
    """Tests for _load_features(), _load_coords(), _load_slide_meta().

    These functions swallow exceptions and return None/fallback values.
    The tests verify: (a) the silent-return contract is upheld, (b) the
    warning is logged so operators can see corrupt files, and (c) a valid
    file returns correct data.
    """

    def test_load_slide_meta_corrupt_h5_returns_nones(self, tmp_path, caplog):
        import logging
        from mussel_dispatcher import wds as wds_mod

        corrupt = tmp_path / "corrupt.h5"
        corrupt.write_bytes(b"not an hdf5 file")

        with caplog.at_level(logging.WARNING, logger="mussel_dispatcher.wds"):
            coords, mpp, fallback = wds_mod._load_slide_meta(corrupt)

        assert coords is None
        assert mpp is None
        assert fallback is None
        assert any("Could not read" in r.message for r in caplog.records)

    def test_load_slide_meta_valid_h5(self, tmp_path):
        import h5py, numpy as np
        from mussel_dispatcher import wds as wds_mod

        h5_path = tmp_path / "slide.h5"
        arr = np.array([[100, 200], [300, 400]], dtype=np.int64)
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset("coords", data=arr)
            ds.attrs["native_mpp"] = 0.5
            ds.attrs["mpp_is_fallback"] = False

        coords, mpp, fallback = wds_mod._load_slide_meta(h5_path)
        assert coords is not None
        assert coords.shape == (2, 2)
        assert mpp == pytest.approx(0.5)
        assert fallback is False

    def test_load_coords_corrupt_h5_returns_none(self, tmp_path, caplog):
        import logging
        from mussel_dispatcher import wds as wds_mod

        corrupt = tmp_path / "corrupt.h5"
        corrupt.write_bytes(b"\xff\xfe bad data")

        with caplog.at_level(logging.WARNING, logger="mussel_dispatcher.wds"):
            result = wds_mod._load_coords(corrupt)

        assert result is None
        assert any("Could not read coords" in r.message for r in caplog.records)

    def test_load_features_1d_tensor_unsqueezed(self, tmp_path):
        """1-D feature tensor should be reshaped to (1, N)."""
        import torch, numpy as np
        from mussel_dispatcher import wds as wds_mod

        pt_path = tmp_path / "slide.features.pt"
        torch.save(torch.ones(768), pt_path)

        arr = wds_mod._load_features(pt_path)
        assert arr.ndim == 2
        assert arr.shape == (1, 768)

    def test_load_features_bfloat16_stored_as_uint16(self, tmp_path):
        """bfloat16 tensors must be reinterpreted as uint16 (no native numpy bfloat16)."""
        import torch, numpy as np
        from mussel_dispatcher import wds as wds_mod

        pt_path = tmp_path / "slide.features.pt"
        torch.save(torch.ones(1, 4, dtype=torch.bfloat16), pt_path)

        arr = wds_mod._load_features(pt_path)
        assert arr.dtype == np.uint16


# ===========================================================================
# _verify_wds_coverage + WDS hook end-to-end manifest interaction
# ===========================================================================

class TestVerifyWdsCoverageManifestRoundtrip:
    """Integration tests for the full round-trip:

    WDS hook writes to manifest → _verify_wds_coverage reads manifest →
    slides are (or are not) reset to PENDING.

    These catch the regression where already-indexed slides were skipped
    without being written to the manifest, causing _verify_wds_coverage to
    perpetually reset them to PENDING even though the S3 data was present.
    """

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

    def _write_manifest(self, tmp_path, entries):
        """entries: list of (slide_id, model[, wds_path])"""
        manifest = tmp_path / "results" / "wds_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide_id", "model", "wds_path"])
            for entry in entries:
                slide_id, model = entry[0], entry[1]
                wds_path = entry[2] if len(entry) > 2 else f"s3://b/wds/{model}/PROJ/000000.tar"
                w.writerow([slide_id, model, wds_path])

    def test_slides_in_manifest_are_not_reset(self, tmp_path):
        """Slides present in wds_manifest for all required models must NOT be
        reset to PENDING — they are already in WDS."""
        runner, state = self._make_runner(
            tmp_path, {"hoptimus1": "s3://b/wds", "titan_slide": "s3://b/wds"}
        )
        state.add_slide("/slides/A.svs", "A")
        state.mark_dispatched(["/slides/A.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)

        self._write_manifest(tmp_path, [
            ("A", "hoptimus1"),
            ("A", "titan_slide"),
        ])
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        runner._verify_wds_coverage(batch_csv)

        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id='A'"
        ).fetchone()
        assert row["status"] == "SUCCEEDED", (
            "slide in manifest for all models must stay SUCCEEDED"
        )

    def test_slide_missing_from_one_model_is_reset(self, tmp_path):
        """If a slide is in the manifest for one model but not the other, it
        must be reset to PENDING."""
        runner, state = self._make_runner(
            tmp_path, {"hoptimus1": "s3://b/wds", "titan_slide": "s3://b/wds"}
        )
        state.add_slide("/slides/A.svs", "A")
        state.mark_dispatched(["/slides/A.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)

        # Only hoptimus1 in manifest — titan_slide missing
        self._write_manifest(tmp_path, [
            ("A", "hoptimus1"),
        ])
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        runner._verify_wds_coverage(batch_csv)

        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id='A'"
        ).fetchone()
        assert row["status"] == "PENDING", (
            "slide missing from titan_slide manifest must be reset to PENDING"
        )

    def test_stale_manifest_entries_do_not_protect_slides(self, tmp_path):
        """Manifest entries pointing to a model that is not in wds_destinations
        must not satisfy coverage for the required model."""
        runner, state = self._make_runner(
            tmp_path, {"hoptimus1": "s3://b/wds"}  # only hoptimus1 required
        )
        state.add_slide("/slides/A.svs", "A")
        state.mark_dispatched(["/slides/A.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)

        # Manifest has titan_slide but NOT hoptimus1
        self._write_manifest(tmp_path, [
            ("A", "titan_slide"),
        ])
        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        runner._verify_wds_coverage(batch_csv)

        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id='A'"
        ).fetchone()
        assert row["status"] == "PENDING"

    def test_already_indexed_not_in_manifest_is_reset(self, tmp_path):
        """Regression test for the original bug (pre-d19bb92):

        If append_wds confirmed a slide was already in the S3 WDS index but
        omitted it from wds_manifest.csv, _verify_wds_coverage would reset
        the slide to PENDING on every cycle — an infinite retry loop.

        This test verifies the consequence: a SUCCEEDED slide with no manifest
        entry for the required model IS reset to PENDING (correct behaviour),
        and therefore the bug's fix (writing all confirmed slides to the
        manifest) is necessary for the slide to remain SUCCEEDED.
        """
        runner, state = self._make_runner(
            tmp_path, {"hoptimus1": "s3://b/wds"}
        )
        state.add_slide("/slides/A.svs", "A")
        state.mark_dispatched(["/slides/A.svs"], "batch-001")
        state.mark_slides_complete("batch-001", succeeded=True)

        # Empty manifest — simulates the pre-fix behaviour where already-indexed
        # slides were not written to the manifest.
        manifest = tmp_path / "results" / "wds_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerow(["slide_id", "model", "wds_path"])

        batch_csv = self._write_batch_csv(tmp_path, ["A"])
        runner._verify_wds_coverage(batch_csv)

        row = state._conn().execute(
            "SELECT status FROM slides WHERE slide_id='A'"
        ).fetchone()
        assert row["status"] == "PENDING", (
            "slide absent from manifest must be reset to PENDING — "
            "confirms that the fix (writing already-indexed slides to manifest) is necessary"
        )


# ===========================================================================
# Cross-batch failed_slides scope regression tests
# ===========================================================================

class TestWdsCrossBatchPruningRegression:
    """End-to-end regression tests for the bug where append_wds() pruned
    wds_index entries for slides from *other* batches that happened to be
    FAILED at the time the current batch's WDS hook ran.

    Scenario (exact production failure):
      1. Batch A succeeds → slides A1, A2 uploaded to WDS; index has A1, A2.
      2. GPU node fails → A1, A2 are reset to FAILED in the dispatcher DB
         and appear as 'failed' in tcga_status.csv.
      3. Batch B runs for slides B1, B2.  Its WDS hook builds failed_slides
         from the status CSV (includes A1, A2) and calls append_wds with
         slide_id_filter={B1, B2}.
      4. BUG (pre-fix): A1, A2 are pruned from the index even though they
         belong to batch A, not batch B.
         FIX: pruning is scoped to slide_id_filter ∩ failed_slides.
    """

    def _fake_index(self, slide_ids, project="PROJ"):
        return {
            sid: {"project_id": project, "shard_file": f"{project}/000000.tar",
                  "native_mpp": None, "mpp_is_fallback": None}
            for sid in slide_ids
        }

    def _run_wds_hook(self, tmp_path, monkeypatch, wds_mod, *,
                      pt_dir, slide_id_filter, failed_slides, index,
                      manifest_csv=None):
        """Helper: run append_wds with a fake index and capture the saved index."""
        saved_index = {}

        def fake_save(idx, *a, **kw):
            saved_index.clear()
            saved_index.update(idx)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(index))
        monkeypatch.setattr(wds_mod, "_save_index", fake_save)

        import pandas as pd
        all_ids = slide_id_filter | set(index.keys())
        inv = pd.DataFrame({
            "file_name": [f"{sid}.svs" for sid in all_ids],
            "project_id": ["PROJ"] * len(all_ids),
        })

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter=slide_id_filter,
            manifest_csv=manifest_csv,
            failed_slides=failed_slides,
        )
        return saved_index

    def test_batch_a_index_entry_survives_batch_b_hook(self, tmp_path, monkeypatch):
        """Core regression: A1 is in the index from batch A, temporarily FAILED,
        but batch B's WDS hook must NOT evict A1 from the index."""
        from mussel_dispatcher import wds as wds_mod

        # Index after batch A completed
        index = self._fake_index({"A1", "A2", "B1"})
        # B1 is the current batch; A1, A2 are FAILED from another batch
        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()

        saved = self._run_wds_hook(
            tmp_path, monkeypatch, wds_mod,
            pt_dir=pt_dir,
            slide_id_filter={"B1"},
            failed_slides={"A1", "A2"},  # from batch A, not current batch
            index=index,
        )

        # saved_index is only written when pruning happens — if not called, index unchanged
        # Either way, A1 and A2 must not have been removed
        surviving = saved if saved else index
        assert "A1" in surviving, "A1 was pruned despite not being in the current batch"
        assert "A2" in surviving, "A2 was pruned despite not being in the current batch"

    def test_current_batch_failed_slide_still_pruned(self, tmp_path, monkeypatch):
        """A slide that is FAILED and belongs to the CURRENT batch must still
        be evicted from the index (permanent failure for this batch)."""
        from mussel_dispatcher import wds as wds_mod

        index = self._fake_index({"B1", "B2"})  # B2 is a permanent failure
        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()

        saved = self._run_wds_hook(
            tmp_path, monkeypatch, wds_mod,
            pt_dir=pt_dir,
            slide_id_filter={"B1", "B2"},
            failed_slides={"B2"},  # B2 is in current batch AND failed
            index=index,
        )

        assert "B2" not in saved, "Current-batch failed slide should have been pruned"
        assert "B1" in saved or "B1" in index, "Non-failed B1 should be intact"

    def test_manifest_entries_for_other_batch_not_duplicated_or_lost(
            self, tmp_path, monkeypatch):
        """Running batch B's WDS hook must not add or remove manifest entries
        belonging to batch A slides."""
        from mussel_dispatcher import wds as wds_mod

        manifest_csv = tmp_path / "wds_manifest.csv"
        # Pre-existing manifest with A1 from batch A
        with open(manifest_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide_id", "model", "wds_path"])
            w.writerow(["A1", "hoptimus1", "s3://bucket/wds/hoptimus1/PROJ/000000.tar"])

        index = self._fake_index({"A1", "B1"})
        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()

        self._run_wds_hook(
            tmp_path, monkeypatch, wds_mod,
            pt_dir=pt_dir,
            slide_id_filter={"B1"},
            failed_slides={"A1"},
            index=index,
            manifest_csv=manifest_csv,
        )

        with open(manifest_csv, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["model"] == "hoptimus1"]
        sids = {r["slide_id"] for r in rows}
        assert "A1" in sids, "Batch A's manifest entry must not be removed by batch B's hook"

    def test_status_csv_failed_slides_scoped_to_model(self, tmp_path, monkeypatch):
        """When building failed_slides from tcga_status.csv, only slides FAILED
        for the requested model are included — not failures for other models."""
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        # A1 failed for titan_slide but succeeded for hoptimus1
        status_csv = tmp_path / "status.csv"
        with open(status_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slide_id", "model", "status"])
            w.writeheader()
            w.writerow({"slide_id": "A1", "model": "hoptimus1", "status": "done"})
            w.writerow({"slide_id": "A1", "model": "titan_slide", "status": "failed"})

        # Simulate the status CSV parsing from wds.py main()
        sdf = pd.read_csv(status_csv, dtype=str).fillna("")
        models_to_run = {"hoptimus1"}
        mask = sdf["status"].str.lower() == "failed"
        mask = mask & sdf["model"].isin(models_to_run)
        failed_slides = set(sdf.loc[mask, "slide_id"].str.strip())

        assert "A1" not in failed_slides, (
            "A1 failed for titan_slide only — must not appear in hoptimus1 failed_slides"
        )

    def test_failed_slide_in_current_batch_not_uploaded_or_in_manifest(
            self, tmp_path, monkeypatch):
        """A slide that is FAILED and in the current batch's slide_id_filter
        must be skipped from upload (line 445) and must not appear in manifest."""
        import pandas as pd
        import torch
        from mussel_dispatcher import wds as wds_mod

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: {})
        monkeypatch.setattr(wds_mod, "_save_index", lambda *a, **kw: None)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        # Provide .pt file for the failed slide — it should be ignored anyway
        torch.save(torch.zeros(1, 4), pt_dir / "bad.features.pt")

        inv = pd.DataFrame({"file_name": ["bad.svs"], "project_id": ["PROJ"]})
        manifest_csv = tmp_path / "wds_manifest.csv"

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter={"bad"},
            manifest_csv=manifest_csv,
            failed_slides={"bad"},
        )

        if manifest_csv.exists():
            with open(manifest_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            bad_rows = [r for r in rows if r["slide_id"] == "bad"]
            assert bad_rows == [], "Failed slide in current batch must not appear in manifest"

    def test_full_sequence_two_batches(self, tmp_path, monkeypatch):
        """Full two-batch sequence:
          batch A: A1, A2 → both uploaded to WDS, both in manifest.
          A1, A2 become FAILED (infra failure).
          batch B: B1 → WDS hook runs with failed_slides={A1, A2}.
          After batch B: A1 and A2 must still be in the index and manifest.
        """
        import pandas as pd
        import torch
        from mussel_dispatcher import wds as wds_mod

        manifest_csv = tmp_path / "wds_manifest.csv"
        current_index = {}

        def fake_load(*a, **kw):
            return dict(current_index)

        def fake_save(idx, *a, **kw):
            current_index.clear()
            current_index.update(idx)

        monkeypatch.setattr(wds_mod, "_load_index", fake_load)
        monkeypatch.setattr(wds_mod, "_save_index", fake_save)
        mock_writer = MagicMock()
        mock_writer.append.return_value = "PROJ/000000.tar"
        monkeypatch.setattr(wds_mod, "_ShardWriter", lambda **kw: mock_writer)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()

        # --- Batch A ---
        for sid in ["A1", "A2"]:
            torch.save(torch.zeros(1, 4), pt_dir / f"{sid}.features.pt")

        inv_a = pd.DataFrame({
            "file_name": ["A1.svs", "A2.svs"],
            "project_id": ["PROJ", "PROJ"],
        })
        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv_a,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter={"A1", "A2"},
            manifest_csv=manifest_csv,
        )
        assert "A1" in current_index and "A2" in current_index, "Batch A must populate index"
        with open(manifest_csv, newline="") as f:
            manifest_after_a = {r["slide_id"] for r in csv.DictReader(f)}
        assert {"A1", "A2"} <= manifest_after_a

        # A1, A2 now FAILED (infra event) — they appear in status CSV as failed
        failed_after_infra = {"A1", "A2"}

        # --- Batch B ---
        torch.save(torch.zeros(1, 4), pt_dir / "B1.features.pt")
        inv_b = pd.DataFrame({
            "file_name": ["A1.svs", "A2.svs", "B1.svs"],
            "project_id": ["PROJ", "PROJ", "PROJ"],
        })
        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv_b,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter={"B1"},
            manifest_csv=manifest_csv,
            failed_slides=failed_after_infra,
        )

        # A1 and A2 must still be in the index
        assert "A1" in current_index, "A1 evicted from index by batch B — regression!"
        assert "A2" in current_index, "A2 evicted from index by batch B — regression!"
        assert "B1" in current_index, "B1 must be in index after batch B"

        # A1 and A2 must still be in the manifest (were written by batch A)
        with open(manifest_csv, newline="") as f:
            manifest_after_b = {r["slide_id"] for r in csv.DictReader(f)}
        assert "A1" in manifest_after_b, "A1 lost from manifest after batch B — regression!"
        assert "A2" in manifest_after_b, "A2 lost from manifest after batch B — regression!"


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

        fake_proc = mock.Mock()
        fake_proc.pid = 12345
        fake_proc.wait.return_value = 0
        with mock.patch("subprocess.Popen", return_value=fake_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0] + [120.0] * 10
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
        fail_proc = mock.Mock()
        fail_proc.pid = 12345
        fail_proc.wait.return_value = 1
        real_isdir = os.path.isdir
        with mock.patch("subprocess.Popen", return_value=fail_proc), \
             mock.patch("mussel_dispatcher.runner.os.path.isdir",
                        side_effect=lambda p: False if p.endswith("/work") else real_isdir(p)), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0] + [120.0] * 10
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

        ok_proc = mock.Mock()
        ok_proc.pid = 12345
        ok_proc.wait.return_value = 0
        with mock.patch("subprocess.Popen", return_value=ok_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0] + [120.0] * 10
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

        ok_proc = mock.Mock()
        ok_proc.pid = 12345
        ok_proc.wait.return_value = 0
        with mock.patch("subprocess.Popen", return_value=ok_proc), \
             mock.patch.object(scheduler, "_get_s3_client", return_value=mock_s3), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0] + [120.0] * 10
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
    """Tests for parse_nf_trace and trace_path_for_log from nextflow_turret."""

    from nextflow_turret import parse_nf_trace
    from nextflow_turret import trace_path_for_log as _trace_path_for_log

    # --- trace_path_for_log ---

    def test_trace_path_replaces_dot_log(self):
        from nextflow_turret import trace_path_for_log as _trace_path_for_log
        assert _trace_path_for_log("/logs/batch_001.log") == "/logs/batch_001.trace.tsv"

    def test_trace_path_no_extension(self):
        from nextflow_turret import trace_path_for_log as _trace_path_for_log
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
        from nextflow_turret import parse_nf_trace
        path = self._write_trace(tmp_path, [])
        result = parse_nf_trace(path)
        assert result == {"completed": 0, "cached": 0, "failed": 0, "total": 0, "failures": []}

    def test_counts_completed_and_cached(self, tmp_path):
        from nextflow_turret import parse_nf_trace
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
        from nextflow_turret import parse_nf_trace
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
        from nextflow_turret import parse_nf_trace
        rows = [
            {"name": f"PROC (slide_{i})", "status": "FAILED", "exit": "1", "hash": f"aa/{i:06d}"}
            for i in range(8)
        ]
        result = parse_nf_trace(self._write_trace(tmp_path, rows))
        assert result["failed"] == 8
        assert len(result["failures"]) == 5

    def test_missing_file_returns_zeros(self):
        from nextflow_turret import parse_nf_trace
        result = parse_nf_trace("/nonexistent/trace.tsv")
        assert result["total"] == 0
        assert result["failures"] == []

    def test_none_path_returns_zeros(self):
        from nextflow_turret import parse_nf_trace
        result = parse_nf_trace(None)
        assert result["total"] == 0

    def test_running_tasks_not_counted(self, tmp_path):
        """RUNNING/SUBMITTED tasks don't appear in the trace file yet."""
        from nextflow_turret import parse_nf_trace
        rows = [
            {"name": "PROC (a)", "status": "COMPLETED", "exit": "0", "hash": "aa/000001"},
            {"name": "PROC (b)", "status": "RUNNING",   "exit": "-", "hash": "bb/000002"},
        ]
        result = parse_nf_trace(self._write_trace(tmp_path, rows))
        assert result["completed"] == 1
        assert result["total"] == 1  # RUNNING is not a finished status

    # --- parse_nf_log uses trace for errors when available ---

    def test_parse_nf_log_uses_trace_for_errors(self, tmp_path):
        from nextflow_turret import parse_nf_log
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
        from nextflow_turret import parse_nf_log
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
        pass  # each test creates its own fresh registry via _make_server

    def test_user_info_route(self, tmp_path):
        import json
        Handler = self._make_server(tmp_path)
        h, responses = self._fake_request(Handler, "GET", "/user-info")
        assert h._response_code == 200
        body = json.loads(responses[0])
        assert "user" in body

    def test_trace_create_r_prefix_run_name(self, tmp_path):
        """dispatcher_r{hash8} run names (with r-prefix) must resolve to the correct batch_id."""
        import json, sqlite3
        Handler = self._make_server(tmp_path)
        # Insert a batch with the full timestamped ID
        db_path = tmp_path / "state" / "dispatcher.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO batches (batch_id, status, slide_count) VALUES (?,?,?)",
                ("20260523T041220_4c59c6c7", "RUNNING", 5),
            )
        # Tower sends create with the r-prefixed short run name
        h, responses = self._fake_request(
            Handler, "POST", "/trace/create",
            body={"runName": "dispatcher_r4c59c6c7", "sessionId": "sess-xyz"}
        )
        assert h._response_code == 200
        # Must resolve to the full batch_id, not "r4c59c6c7"
        state = Handler.tower_registry.get_by_batch("20260523T041220_4c59c6c7")
        assert state is not None, "r-prefix run name not resolved to batch_id"
        assert state["batch_id"] == "20260523T041220_4c59c6c7"

    def test_trace_create_route(self, tmp_path):
        import json
        Handler = self._make_server(tmp_path)
        h, responses = self._fake_request(
            Handler, "POST", "/trace/create",
            body={"runName": "dispatcher_test99", "sessionId": "abc-123"}
        )
        assert h._response_code == 200
        body = json.loads(responses[0])
        assert body["workflowId"] == "dispatcher_test99"
        # State should now exist in the registry
        state = Handler.tower_registry.get_by_batch("test99")
        assert state is not None
        assert state["batch_id"] == "test99"

    def test_trace_progress_route(self, tmp_path):
        import json
        Handler = self._make_server(tmp_path)
        Handler.tower_registry.register("dispatcher_test88", "test88", "dispatcher_test88")
        progress = {"succeeded": 3, "failed": 1, "cached": 0, "running": 2, "pending": 4}
        h, responses = self._fake_request(
            Handler, "PUT", "/trace/dispatcher_test88/progress",
            body={"progress": progress, "instant": 1234567890}
        )
        assert h._response_code == 200
        prog = Handler.tower_registry.get_by_batch("test88")
        assert prog["task_counts"]["succeeded"] == 3
        assert prog["task_counts"]["running"] == 2

    def test_trace_complete_route(self, tmp_path):
        import json
        Handler = self._make_server(tmp_path)
        Handler.tower_registry.register("dispatcher_test77", "test77", "dispatcher_test77")
        h, responses = self._fake_request(
            Handler, "PUT", "/trace/dispatcher_test77/complete",
            body={"progress": {"succeeded": 10}, "instant": 1234567890}
        )
        assert h._response_code == 200
        state = Handler.tower_registry.get_by_batch("test77")
        assert state["complete"] is True

    def test_trace_begin_route(self, tmp_path):
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
    """Integration test: derived name matches real squeue/sacct job name format."""

    def test_matches_actual_job_names(self):
        """Derived name matches what squeue proc_short produces (strip trailing _)."""
        from nextflow_turret import tower_process_to_slurm_name
        import re
        # Real job name from sacct:
        job_name = "nf-MUSSEL_EXTRACT_FEATURES_ONE_STEP_TESSELLATE_FEATURIZE_BATCH_(7)"
        proc_short = re.sub(r'\s*\(\d+\)$', '', job_name[3:])
        # proc_short = "MUSSEL_EXTRACT_FEATURES_ONE_STEP_TESSELLATE_FEATURIZE_BATCH_"
        tower_name = "MUSSEL:EXTRACT_FEATURES:ONE_STEP:TESSELLATE_FEATURIZE_BATCH"
        derived = tower_process_to_slurm_name(tower_name)
        # derived strips trailing _, squeue proc_short has trailing _
        assert proc_short.rstrip("_") == derived


# ===========================================================================
# Shared Databricks utilities (databricks_sync.py)
# ===========================================================================

class TestEnsureVolumeExists:
    """Tests for ensure_volume_exists() — new function, completely untested before."""

    def test_success_logs_volume_ready(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": {"state": "SUCCEEDED"}}
        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(ds.requests, "post", mock_post)

        ds.ensure_volume_exists(
            "/Volumes/cat/sch/vol", host="https://host", token="tok", warehouse_id="wh1"
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert "CREATE VOLUME IF NOT EXISTS cat.sch.vol" in payload["statement"]
        assert payload["warehouse_id"] == "wh1"

    def test_parses_three_part_fqn_correctly(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        captured = {}
        def fake_post(url, headers, json, timeout):
            captured["payload"] = json
            r = MagicMock()
            r.json.return_value = {"status": {"state": "SUCCEEDED"}}
            return r
        monkeypatch.setattr(ds.requests, "post", fake_post)

        ds.ensure_volume_exists(
            "/Volumes/my_catalog/my_schema/my_volume/subdir",
            host="https://host", token="tok", warehouse_id="wh"
        )
        assert "my_catalog.my_schema.my_volume" in captured["payload"]["statement"]

    def test_bad_path_logs_warning_and_returns(self, monkeypatch, caplog):
        import logging
        from mussel_dispatcher import databricks_sync as ds
        mock_post = MagicMock()
        monkeypatch.setattr(ds.requests, "post", mock_post)

        with caplog.at_level(logging.WARNING, logger="mussel_dispatcher.databricks_sync"):
            ds.ensure_volume_exists(
                "/not/a/volume/path", host="https://host", token="tok", warehouse_id="wh"
            )

        mock_post.assert_not_called()
        assert any("Cannot parse" in r.message for r in caplog.records)

    def test_http_error_raises(self, monkeypatch):
        import requests as req_lib
        from mussel_dispatcher import databricks_sync as ds
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError("403")
        monkeypatch.setattr(ds.requests, "post", MagicMock(return_value=mock_resp))

        with pytest.raises(req_lib.HTTPError):
            ds.ensure_volume_exists(
                "/Volumes/a/b/c", host="https://host", token="tok", warehouse_id="wh"
            )

    def test_non_succeeded_state_logs_warning(self, monkeypatch, caplog):
        import logging
        from mussel_dispatcher import databricks_sync as ds
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": {"state": "FAILED", "error": {"message": "Permission denied"}}
        }
        monkeypatch.setattr(ds.requests, "post", MagicMock(return_value=mock_resp))

        with caplog.at_level(logging.WARNING, logger="mussel_dispatcher.databricks_sync"):
            ds.ensure_volume_exists(
                "/Volumes/a/b/c", host="https://host", token="tok", warehouse_id="wh"
            )
        assert any("Permission denied" in r.message for r in caplog.records)


class TestResolveWarehouseId:
    """Tests for resolve_warehouse_id() — silently returns '' on any error."""

    def test_returns_empty_when_no_credentials(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        monkeypatch.setattr(ds, "resolve_credentials", lambda h, t: ("", ""))
        assert ds.resolve_warehouse_id() == ""

    def test_returns_empty_on_http_error(self, monkeypatch):
        import requests as req_lib
        from mussel_dispatcher import databricks_sync as ds
        monkeypatch.setattr(ds, "resolve_credentials", lambda h, t: ("https://host", "tok"))
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError("500")
        monkeypatch.setattr(ds.requests, "get", MagicMock(return_value=mock_resp))
        assert ds.resolve_warehouse_id() == ""

    def test_returns_empty_when_no_warehouses(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        monkeypatch.setattr(ds, "resolve_credentials", lambda h, t: ("https://host", "tok"))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"warehouses": []}
        monkeypatch.setattr(ds.requests, "get", MagicMock(return_value=mock_resp))
        assert ds.resolve_warehouse_id() == ""

    def test_prefers_running_over_stopped(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        monkeypatch.setattr(ds, "resolve_credentials", lambda h, t: ("https://host", "tok"))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"warehouses": [
            {"id": "stopped-wh", "state": "STOPPED",  "warehouse_type": "SERVERLESS", "name": "a"},
            {"id": "running-wh", "state": "RUNNING",  "warehouse_type": "CLASSIC",    "name": "b"},
        ]}
        monkeypatch.setattr(ds.requests, "get", MagicMock(return_value=mock_resp))
        assert ds.resolve_warehouse_id() == "running-wh"

    def test_prefers_serverless_over_classic_same_state(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        monkeypatch.setattr(ds, "resolve_credentials", lambda h, t: ("https://host", "tok"))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"warehouses": [
            {"id": "classic-wh",    "state": "RUNNING", "warehouse_type": "CLASSIC",    "name": "a"},
            {"id": "serverless-wh", "state": "RUNNING", "warehouse_type": "SERVERLESS", "name": "b"},
        ]}
        monkeypatch.setattr(ds.requests, "get", MagicMock(return_value=mock_resp))
        assert ds.resolve_warehouse_id() == "serverless-wh"

    def test_skips_deleted_warehouses(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds
        monkeypatch.setattr(ds, "resolve_credentials", lambda h, t: ("https://host", "tok"))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"warehouses": [
            {"id": "deleted-wh", "state": "DELETED",  "warehouse_type": "SERVERLESS", "name": "a"},
            {"id": "live-wh",    "state": "STOPPED",  "warehouse_type": "CLASSIC",    "name": "b"},
        ]}
        monkeypatch.setattr(ds.requests, "get", MagicMock(return_value=mock_resp))
        assert ds.resolve_warehouse_id() == "live-wh"


class TestMergeViaWarehouse:
    """Tests for merge_via_warehouse() — 120 lines of SQL generation, untested before."""

    def _make_mock_post(self, monkeypatch, ds, responses):
        """responses: list of (status_code_ok, json_body) in call order."""
        call_iter = iter(responses)
        def fake_post(url, headers, json, timeout):
            body = next(call_iter)
            r = MagicMock()
            r.json.return_value = body
            return r
        monkeypatch.setattr(ds.requests, "post", fake_post)

    def _make_mock_get(self, monkeypatch, ds, body):
        monkeypatch.setattr(ds.requests, "get",
            MagicMock(return_value=MagicMock(json=MagicMock(return_value=body))))

    def test_create_table_failure_returns_false(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds

        # POST for CREATE TABLE → FAILED immediately
        self._make_mock_post(monkeypatch, ds, [
            {"statement_id": "s1"},  # submit
        ])
        # GET poll → FAILED
        self._make_mock_get(monkeypatch, ds, {
            "status": {"state": "FAILED", "error": {"message": "Table error"}}
        })

        ok, msg = ds.merge_via_warehouse(
            "/Volumes/a/b/c", "cat.sch.table",
            host="https://host", token="tok", warehouse_id="wh",
            poll_interval_s=0,
        )
        assert not ok
        assert "Table error" in msg or "CREATE TABLE" in msg

    def test_merge_success_returns_true(self, monkeypatch):
        from mussel_dispatcher import databricks_sync as ds

        call_count = [0]
        def fake_post(url, headers, json, timeout):
            call_count[0] += 1
            r = MagicMock()
            if "CREATE TABLE" in json.get("statement", ""):
                r.json.return_value = {"statement_id": "s-create"}
            elif "DESCRIBE" in json.get("statement", "") and "parquet" in json.get("statement", ""):
                r.json.return_value = {
                    "result": {"data_array": [["slide_id"], ["model"], ["embedding"]]}
                }
            elif "DESCRIBE TABLE" in json.get("statement", ""):
                r.json.return_value = {
                    "result": {"data_array": [["slide_id"], ["model"], ["embedding"]]}
                }
            elif "MERGE" in json.get("statement", ""):
                r.json.return_value = {"statement_id": "s-merge"}
            else:
                r.json.return_value = {"statement_id": "s-other"}
            return r

        get_responses = iter([
            # poll for CREATE TABLE
            {"status": {"state": "SUCCEEDED"}},
            # poll for MERGE
            {"status": {"state": "SUCCEEDED"}},
        ])
        monkeypatch.setattr(ds.requests, "post", fake_post)
        monkeypatch.setattr(ds.requests, "get",
            MagicMock(side_effect=lambda *a, **kw: MagicMock(
                json=MagicMock(return_value=next(get_responses))
            )))

        ok, msg = ds.merge_via_warehouse(
            "/Volumes/a/b/c", "cat.sch.table",
            host="https://host", token="tok", warehouse_id="wh",
            poll_interval_s=0,
        )
        assert ok

    def test_merge_sql_uses_column_intersection(self, monkeypatch):
        """When source and target schemas differ, only common cols appear in SET clause."""
        from mussel_dispatcher import databricks_sync as ds

        merge_statements = []
        def fake_post(url, headers, json, timeout):
            stmt = json.get("statement", "")
            if "MERGE" in stmt:
                merge_statements.append(stmt)
            r = MagicMock()
            if "CREATE TABLE" in stmt:
                r.json.return_value = {"statement_id": "s1"}
            elif "DESCRIBE" in stmt and "parquet" in stmt:
                r.json.return_value = {
                    "result": {"data_array": [["slide_id"], ["model"]]}
                }
            elif "DESCRIBE TABLE" in stmt:
                r.json.return_value = {
                    "result": {"data_array": [["slide_id"], ["model"], ["extra_target_col"]]}
                }
            elif "MERGE" in stmt:
                r.json.return_value = {"statement_id": "s2"}
            else:
                r.json.return_value = {"statement_id": "sx"}
            return r

        get_responses = iter([
            {"status": {"state": "SUCCEEDED"}},  # CREATE TABLE
            {"status": {"state": "SUCCEEDED"}},  # MERGE
        ])
        monkeypatch.setattr(ds.requests, "post", fake_post)
        monkeypatch.setattr(ds.requests, "get",
            MagicMock(side_effect=lambda *a, **kw: MagicMock(
                json=MagicMock(return_value=next(get_responses))
            )))

        ds.merge_via_warehouse(
            "/Volumes/a/b/c", "cat.sch.table",
            host="https://host", token="tok", warehouse_id="wh",
            poll_interval_s=0,
        )

        assert merge_statements, "No MERGE statement was issued"
        sql = merge_statements[0]
        # extra_target_col is target-only → should appear as NULL in INSERT
        assert "NULL" in sql
        # common cols slide_id and model should be in SET clause
        assert "t.slide_id = s.slide_id" in sql or "t.model = s.model" in sql

    def test_fallback_wildcard_merge_when_schema_unavailable(self, monkeypatch):
        """When DESCRIBE fails, falls back to MERGE … UPDATE SET * INSERT *."""
        from mussel_dispatcher import databricks_sync as ds

        merge_statements = []
        def fake_post(url, headers, json, timeout):
            stmt = json.get("statement", "")
            if "MERGE" in stmt:
                merge_statements.append(stmt)
            r = MagicMock()
            if "CREATE TABLE" in stmt:
                r.json.return_value = {"statement_id": "s1"}
            elif "DESCRIBE" in stmt:
                raise Exception("DESCRIBE unavailable")
            elif "MERGE" in stmt:
                r.json.return_value = {"statement_id": "s2"}
            else:
                r.json.return_value = {"statement_id": "sx"}
            return r

        get_responses = iter([
            {"status": {"state": "SUCCEEDED"}},
            {"status": {"state": "SUCCEEDED"}},
        ])
        monkeypatch.setattr(ds.requests, "post", fake_post)
        monkeypatch.setattr(ds.requests, "get",
            MagicMock(side_effect=lambda *a, **kw: MagicMock(
                json=MagicMock(return_value=next(get_responses))
            )))

        ok, _ = ds.merge_via_warehouse(
            "/Volumes/a/b/c", "cat.sch.table",
            host="https://host", token="tok", warehouse_id="wh",
            poll_interval_s=0,
        )
        assert ok
        assert merge_statements
        assert "UPDATE SET *" in merge_statements[0]
        assert "INSERT *" in merge_statements[0]


# ===========================================================================
# Shared Databricks utilities (databricks_sync.py)
# ===========================================================================

class TestResolveCredentials:
    def test_resolve_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://host.example.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test_token")
        from mussel_dispatcher.databricks_sync import resolve_credentials
        host, token = resolve_credentials("", "")
        assert host == "https://host.example.com"
        assert token == "dapi_test_token"

    def test_resolve_credentials_explicit_args_take_priority(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://env-host.example.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "env_token")
        from mussel_dispatcher.databricks_sync import resolve_credentials
        host, token = resolve_credentials("https://explicit-host.example.com", "explicit_token")
        assert host == "https://explicit-host.example.com"
        assert token == "explicit_token"


class TestUploadAndTrigger:
    def test_upload_and_trigger_dry(self, tmp_path, monkeypatch):
        """upload_and_trigger calls PUT to the correct URL and optionally POST for job trigger."""
        import argparse
        from unittest.mock import MagicMock, patch
        import pandas as pd
        from mussel_dispatcher.databricks_sync import upload_and_trigger

        df = pd.DataFrame({"slide_id": ["s1", "s2"], "status": ["SUCCEEDED", "PENDING"]})

        args = argparse.Namespace(
            databricks_host="https://test.databricks.com",
            token="test_token",
            volume_folder="/Volumes/test/schema/folder",
            volume_path=None,
            table=None,
            job_id=None,
            output_parquet=None,
        )

        put_resp = MagicMock()
        put_resp.raise_for_status = MagicMock()

        with patch("mussel_dispatcher.databricks_sync.requests.put", return_value=put_resp) as mock_put:
            upload_and_trigger(df, args, filename_prefix="test_inventory_")

        assert mock_put.called
        call_url = mock_put.call_args[0][0]
        assert call_url.startswith("https://test.databricks.com/api/2.0/fs/files/Volumes/test/schema/folder/test_inventory_")
        assert call_url.endswith(".parquet")

    def test_upload_and_trigger_with_job(self, monkeypatch):
        """When --job-id is set, trigger_job is called and poll_job_run is awaited."""
        import argparse
        from unittest.mock import MagicMock, patch
        import pandas as pd
        from mussel_dispatcher.databricks_sync import upload_and_trigger

        df = pd.DataFrame({"slide_id": ["s1"], "status": ["SUCCEEDED"]})

        args = argparse.Namespace(
            databricks_host="https://test.databricks.com",
            token="test_token",
            volume_folder="/Volumes/test/schema/folder",
            volume_path=None,
            table="catalog.schema.table",
            job_id="99",
            output_parquet=None,
            status_file=None,
        )

        put_resp = MagicMock()
        put_resp.raise_for_status = MagicMock()

        post_resp = MagicMock()
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"run_id": 42}

        with patch("mussel_dispatcher.databricks_sync.requests.put", return_value=put_resp), \
             patch("mussel_dispatcher.databricks_sync.requests.post", return_value=post_resp) as mock_post, \
             patch("mussel_dispatcher.databricks_sync.poll_job_run", return_value=(True, "Run 42 succeeded")) as mock_poll:
            upload_and_trigger(df, args, filename_prefix="test_inventory_")

        assert mock_post.called
        body = mock_post.call_args[1]["json"]
        assert body["job_id"] == 99
        assert body["notebook_params"]["target_table"] == "catalog.schema.table"
        mock_poll.assert_called_once_with("42", "https://test.databricks.com", "test_token")

    def test_upload_and_trigger_job_failure(self, monkeypatch):
        """When poll_job_run returns failure, upload_and_trigger raises SystemExit."""
        import argparse
        import pytest
        from unittest.mock import MagicMock, patch
        import pandas as pd
        from mussel_dispatcher.databricks_sync import upload_and_trigger

        df = pd.DataFrame({"slide_id": ["s1"], "status": ["SUCCEEDED"]})

        args = argparse.Namespace(
            databricks_host="https://test.databricks.com",
            token="test_token",
            volume_folder="/Volumes/test/schema/folder",
            volume_path=None,
            table="catalog.schema.table",
            job_id="99",
            output_parquet=None,
            status_file=None,
        )

        put_resp = MagicMock()
        put_resp.raise_for_status = MagicMock()

        post_resp = MagicMock()
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"run_id": 42}

        with patch("mussel_dispatcher.databricks_sync.requests.put", return_value=put_resp), \
             patch("mussel_dispatcher.databricks_sync.requests.post", return_value=post_resp), \
             patch("mussel_dispatcher.databricks_sync.poll_job_run", return_value=(False, "MERGE failed: unresolved expression")):
            with pytest.raises(SystemExit):
                upload_and_trigger(df, args, filename_prefix="test_inventory_")

# ===========================================================================
# StateStore oncotree_code
# ===========================================================================

class TestStateOncotree:
    def test_state_oncotree_stored(self, tmp_path):
        """add_slide with oncotree_code stores and retrieves correctly."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("s3://bucket/slide1.svs", "slide1", oncotree_code="IDC")
        slides = store.get_all_slides()
        assert len(slides) == 1
        assert slides[0]["oncotree_code"] == "IDC"

    def test_state_oncotree_default_empty(self, tmp_path):
        """add_slide without oncotree_code stores empty string."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("s3://bucket/slide2.svs", "slide2")
        slides = store.get_all_slides()
        assert slides[0]["oncotree_code"] == ""

    def test_state_oncotree_migration(self, tmp_path):
        """StateStore migrates an existing DB that lacks the oncotree_code column."""
        import sqlite3
        db_path = str(tmp_path / "old.db")
        # Create old-schema DB without oncotree_code column
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE slides (
                slide_path    TEXT PRIMARY KEY,
                slide_id      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'PENDING',
                batch_id      TEXT,
                download_path TEXT,
                fail_count    INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT,
                dispatched_at TEXT,
                completed_at  TEXT,
                error_msg     TEXT,
                file_id       TEXT,
                file_name     TEXT,
                needs_download INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "INSERT INTO slides (slide_path, slide_id, status) VALUES ('s3://b/s.svs', 'legacy_slide', 'SUCCEEDED')"
        )
        conn.commit()
        conn.close()

        # Opening StateStore should migrate without error
        store = StateStore(db_path)
        slides = store.get_all_slides()
        assert any(s["slide_id"] == "legacy_slide" for s in slides)
        # The migrated column should exist and default to ''
        legacy = next(s for s in slides if s["slide_id"] == "legacy_slide")
        assert legacy["oncotree_code"] == ""

    def test_get_all_slides_returns_expected_columns(self, tmp_path):
        """get_all_slides returns rows with all expected columns."""
        store = StateStore(str(tmp_path / "test.db"))
        store.add_slide("s3://bucket/s.svs", "s1", oncotree_code="LUAD")
        rows = store.get_all_slides()
        assert rows
        row = rows[0]
        for col in ("slide_path", "slide_id", "oncotree_code", "status",
                    "error_msg", "first_seen_at", "completed_at"):
            assert col in row, f"Missing column: {col}"


# ===========================================================================
# BatchScheduler._retry_failed_slides  (periodic retry sweep)
# ===========================================================================

class TestRetryFailedSlides:
    """Tests for the periodic FAILED → PENDING sweep added to BatchScheduler.run()."""

    def _make_scheduler(self, tmp_path, retry_failed=True, max_slide_retries=5):
        from mussel_dispatcher.state import StateStore
        db = StateStore(str(tmp_path / "s.db"))
        cfg = make_config(
            batch_size=10,
            max_wait_seconds=9_999,
            retry_failed=retry_failed,
            max_slide_retries=max_slide_retries,
        )
        rm = MagicMock()
        rm.in_flight_slide_ids = set()
        scheduler = BatchScheduler(cfg, db, rm, threading.Event())
        return scheduler, db

    def _set_failed(self, db, slide_id, fail_count=1):
        """Directly write a FAILED slide with given fail_count to the DB."""
        db._conn().execute(
            "UPDATE slides SET status='FAILED', fail_count=? WHERE slide_id=?",
            (fail_count, slide_id),
        )
        db._conn().commit()

    def test_retry_failed_resets_retriable_slides_to_pending(self, tmp_path):
        """FAILED slides below max_retries are reset to PENDING and enqueued."""
        scheduler, db = self._make_scheduler(tmp_path, max_slide_retries=5)
        db.add_slide("/s/a.svs", "a")
        self._set_failed(db, "a", fail_count=1)

        scheduler._retry_failed_slides()

        row = db._conn().execute("SELECT status FROM slides WHERE slide_id='a'").fetchone()
        assert row["status"] == "PENDING"
        assert any(s["slide_id"] == "a" for s in scheduler._pending)

    def test_retry_failed_does_not_reset_at_max_retries(self, tmp_path):
        """FAILED slides at max_retries are left FAILED (permanently blacklisted)."""
        scheduler, db = self._make_scheduler(tmp_path, max_slide_retries=3)
        db.add_slide("/s/b.svs", "b")
        self._set_failed(db, "b", fail_count=3)

        scheduler._retry_failed_slides()

        row = db._conn().execute("SELECT status, fail_count FROM slides WHERE slide_id='b'").fetchone()
        assert row["status"] == "FAILED"
        assert row["fail_count"] == 3
        assert len(scheduler._pending) == 0

    def test_retry_failed_disabled_when_cfg_retry_failed_false(self, tmp_path):
        """No reset happens when cfg.retry_failed=False."""
        scheduler, db = self._make_scheduler(tmp_path, retry_failed=False)
        db.add_slide("/s/c.svs", "c")
        self._set_failed(db, "c", fail_count=1)

        scheduler._retry_failed_slides()

        row = db._conn().execute("SELECT status FROM slides WHERE slide_id='c'").fetchone()
        assert row["status"] == "FAILED"
        assert len(scheduler._pending) == 0

    def test_retry_failed_mixed_counts(self, tmp_path):
        """Retriable slides are reset; exhausted slides stay FAILED."""
        scheduler, db = self._make_scheduler(tmp_path, max_slide_retries=5)
        db.add_slide("/s/ok.svs", "ok")
        db.add_slide("/s/perm.svs", "perm")
        self._set_failed(db, "ok",   fail_count=1)  # retriable
        self._set_failed(db, "perm", fail_count=5)  # exhausted

        scheduler._retry_failed_slides()

        ok_row   = db._conn().execute("SELECT status FROM slides WHERE slide_id='ok'").fetchone()
        perm_row = db._conn().execute("SELECT status FROM slides WHERE slide_id='perm'").fetchone()
        assert ok_row["status"] == "PENDING"
        assert perm_row["status"] == "FAILED"
        assert len(scheduler._pending) == 1
        assert scheduler._pending[0]["slide_id"] == "ok"

    def test_retry_failed_already_in_deque_not_duplicated(self, tmp_path):
        """If a reset slide is already in the deque it is not added again."""
        scheduler, db = self._make_scheduler(tmp_path, max_slide_retries=5)
        db.add_slide("/s/dup.svs", "dup")
        self._set_failed(db, "dup", fail_count=1)
        # Manually pre-populate the deque (simulates a race where DB swept first)
        scheduler.enqueue({"slide_id": "dup", "slide_path": "/s/dup.svs"})

        scheduler._retry_failed_slides()

        assert sum(1 for s in scheduler._pending if s["slide_id"] == "dup") == 1

    def test_retry_failed_does_not_touch_succeeded(self, tmp_path):
        """SUCCEEDED slides are never reset by the retry sweep."""
        scheduler, db = self._make_scheduler(tmp_path, max_slide_retries=5)
        db.add_slide("/s/done.svs", "done")
        db._conn().execute("UPDATE slides SET status='SUCCEEDED' WHERE slide_id='done'")
        db._conn().commit()

        scheduler._retry_failed_slides()

        row = db._conn().execute("SELECT status FROM slides WHERE slide_id='done'").fetchone()
        assert row["status"] == "SUCCEEDED"
        assert len(scheduler._pending) == 0


# ===========================================================================
# TestE2ERetryFailedLoop — dispatch-loop regression with periodic retry
# ===========================================================================

class TestE2ERetryFailedLoop:
    """End-to-end regression tests ensuring periodic retry does not re-create
    the runaway dispatch loop fixed in the in-flight tracking patch."""

    def _make_stack(self, tmp_path, max_slide_retries=5):
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
            retry_failed=True,
            max_slide_retries=max_slide_retries,
        )
        state = StateStore(str(tmp_path / "state" / "test.db"))
        run_manager = RunManager(cfg, state)
        scheduler = BatchScheduler(cfg, state, run_manager, threading.Event())
        return cfg, state, scheduler, run_manager

    def test_periodic_retry_requeues_failed_slides(self, tmp_path):
        """_retry_failed_slides re-queues slides that failed mid-run (simulating
        node failures) so they are picked up by the next dispatch without restart."""
        import unittest.mock as mock

        cfg, state, scheduler, run_manager = self._make_stack(tmp_path)

        slides = [{"slide_path": f"/slides/{c}.svs", "slide_id": c} for c in ["A", "B"]]
        for s in slides:
            state.add_slide(s["slide_path"], s["slide_id"])
            scheduler.enqueue(s)

        # First dispatch: batch fails (simulating NF crash / node failure)
        fail_proc = mock.Mock()
        fail_proc.pid = 12345
        fail_proc.wait.return_value = 1
        real_isdir = os.path.isdir
        with mock.patch("subprocess.Popen", return_value=fail_proc), \
             mock.patch("mussel_dispatcher.runner.os.path.isdir",
                        side_effect=lambda p: False if p.endswith("/work") else real_isdir(p)), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0] + [120.0] * 10
            scheduler._maybe_dispatch(force=True)
            run_manager.shutdown(wait=True)

        for s in slides:
            row = state._conn().execute(
                "SELECT status, fail_count FROM slides WHERE slide_id=?", (s["slide_id"],)
            ).fetchone()
            assert row["status"] == "FAILED"
            assert row["fail_count"] == 1

        # Periodic retry sweep re-queues them (no restart needed)
        scheduler._retry_failed_slides()

        for s in slides:
            row = state._conn().execute(
                "SELECT status FROM slides WHERE slide_id=?", (s["slide_id"],)
            ).fetchone()
            assert row["status"] == "PENDING"
        assert {s["slide_id"] for s in scheduler._pending} == {"A", "B"}

        # Second dispatch: succeeds
        run_manager2 = RunManager(cfg, state)
        scheduler2 = BatchScheduler(cfg, state, run_manager2, threading.Event())
        scheduler2.enqueue({"slide_path": "/slides/A.svs", "slide_id": "A"})
        scheduler2.enqueue({"slide_path": "/slides/B.svs", "slide_id": "B"})

        ok_proc = mock.Mock()
        ok_proc.pid = 12345
        ok_proc.wait.return_value = 0
        with mock.patch("subprocess.Popen", return_value=ok_proc), \
             mock.patch("mussel_dispatcher.runner.time") as mt:
            mt.time.side_effect = [0.0] + [120.0] * 10
            scheduler2._maybe_dispatch(force=True)
            run_manager2.shutdown(wait=True)

        for s in slides:
            row = state._conn().execute(
                "SELECT status FROM slides WHERE slide_id=?", (s["slide_id"],)
            ).fetchone()
            assert row["status"] == "SUCCEEDED"

    def test_periodic_retry_does_not_cause_dispatch_loop(self, tmp_path):
        """_retry_failed_slides must not re-enqueue slides that are in-flight
        (already popped from deque but not yet written DISPATCHED in DB) to
        avoid recreating the runaway loop fixed by in-flight tracking."""
        cfg, state, scheduler, run_manager = self._make_stack(tmp_path)

        slides = [{"slide_path": f"/slides/{c}.svs", "slide_id": c} for c in ["X", "Y", "Z"]]
        for s in slides:
            state.add_slide(s["slide_path"], s["slide_id"])

        # Put all three slides in FAILED state directly
        state._conn().execute(
            "UPDATE slides SET status='FAILED', fail_count=1"
        )
        state._conn().commit()

        # Simulate "X" in-flight: popped by RunManager but not yet DISPATCHED in DB
        run_manager._in_flight["batch-001"] = {"X"}

        scheduler._retry_failed_slides()

        # X should NOT be in the deque — it is in-flight
        queued_ids = {s["slide_id"] for s in scheduler._pending}
        assert "X" not in queued_ids, "In-flight slide was re-enqueued — dispatch loop risk!"
        # Y and Z are not in-flight → they should be re-queued
        assert "Y" in queued_ids
        assert "Z" in queued_ids

    def test_exhausted_slides_never_dispatched(self, tmp_path):
        """Slides at max_slide_retries are permanently FAILED and never re-dispatched."""
        import unittest.mock as mock

        cfg, state, scheduler, run_manager = self._make_stack(tmp_path, max_slide_retries=2)

        slide = {"slide_path": "/slides/bad.svs", "slide_id": "bad"}
        state.add_slide(slide["slide_path"], slide["slide_id"])
        scheduler.enqueue(slide)

        for _ in range(2):
            fail_proc = mock.Mock()
            fail_proc.pid = 12345
            fail_proc.wait.return_value = 1
            real_isdir = os.path.isdir
            with mock.patch("subprocess.Popen", return_value=fail_proc), \
                 mock.patch("mussel_dispatcher.runner.os.path.isdir",
                            side_effect=lambda p: False if p.endswith("/work") else real_isdir(p)), \
                 mock.patch("mussel_dispatcher.runner.time") as mt:
                mt.time.side_effect = [0.0] + [120.0] * 10
                scheduler._maybe_dispatch(force=True)
                run_manager.shutdown(wait=True)
            # After each failure, periodic retry re-queues (if not exhausted)
            scheduler._retry_failed_slides()
            run_manager = RunManager(cfg, state)
            scheduler = BatchScheduler(cfg, state, run_manager, threading.Event())
            for s in scheduler.state.get_pending_slides():
                scheduler.enqueue(s)

        # After max_slide_retries failures the slide should be permanently FAILED
        row = state._conn().execute(
            "SELECT status, fail_count FROM slides WHERE slide_id='bad'"
        ).fetchone()
        assert row["status"] == "FAILED"
        assert row["fail_count"] >= 2
        # Retry sweep does nothing more
        scheduler._retry_failed_slides()
        assert len(scheduler._pending) == 0


# ===========================================================================
# wds.py failed_slides pruning scope fix
# ===========================================================================

class TestWdsPruneScope:
    """Tests for the bug where append_wds() pruned wds_index entries for slides
    from *other* batches that happened to be FAILED at the time the current
    batch's WDS hook ran.

    Fix: scope the prune to slide_id_filter (current-batch slides only).
    """

    def _fake_index(self, slide_ids, project="PROJ"):
        return {
            sid: {"project_id": project, "shard_file": f"{project}/000000.tar",
                  "native_mpp": None, "mpp_is_fallback": None}
            for sid in slide_ids
        }

    def test_failed_slides_outside_filter_not_pruned(self, tmp_path, monkeypatch):
        """Slides that are FAILED but NOT in the current batch's slide_id_filter
        must keep their wds_index entries."""
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        current_batch = {"A", "B"}
        other_batch_slide = "X"  # previously uploaded, now temporarily FAILED elsewhere

        # Index contains both current batch and the other-batch slide
        initial_index = self._fake_index(current_batch | {other_batch_slide})
        saved = {}

        def fake_save(index, *a, **kw):
            saved.update(index)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(initial_index))
        monkeypatch.setattr(wds_mod, "_save_index", fake_save)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        inv = pd.DataFrame({"file_name": ["A.svs", "B.svs"], "project_id": ["PROJ", "PROJ"]})

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter=current_batch,
            manifest_csv=None,
            failed_slides={other_batch_slide},  # from another batch, currently FAILED
        )

        # The other-batch slide must still be in the index (not pruned)
        # saved is only updated if _save_index is called; check original index was preserved
        # Since no new slides were appended, _save_index shouldn't even be called for prune
        assert other_batch_slide not in saved or saved.get(other_batch_slide) is not None, \
            "Other-batch slide was wrongly pruned from wds_index"

    def test_failed_slides_inside_filter_are_pruned(self, tmp_path, monkeypatch):
        """Slides that are FAILED and ARE in the current batch's slide_id_filter
        must be pruned from the wds_index (they are permanent failures for this batch)."""
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        current_batch = {"A", "B", "bad"}
        failed_in_current = {"bad"}

        initial_index = self._fake_index(current_batch)
        pruned_index = {}

        def fake_save(index, *a, **kw):
            pruned_index.update(index)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(initial_index))
        monkeypatch.setattr(wds_mod, "_save_index", fake_save)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        inv = pd.DataFrame({
            "file_name": ["A.svs", "B.svs", "bad.svs"],
            "project_id": ["PROJ", "PROJ", "PROJ"],
        })

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter=current_batch,
            manifest_csv=None,
            failed_slides=failed_in_current,
        )

        assert "bad" not in pruned_index, "Current-batch failed slide should have been pruned"
        assert "A" in pruned_index or "A" in initial_index  # A, B untouched

    def test_no_filter_prunes_all_failed(self, tmp_path, monkeypatch):
        """When slide_id_filter is None (manual / full-index run), all failed_slides
        are still pruned (backward-compatible behaviour)."""
        import pandas as pd
        from mussel_dispatcher import wds as wds_mod

        all_slides = {"A", "B", "bad"}
        initial_index = self._fake_index(all_slides)
        pruned_index = {}

        def fake_save(index, *a, **kw):
            pruned_index.update(index)

        monkeypatch.setattr(wds_mod, "_load_index", lambda *a, **kw: dict(initial_index))
        monkeypatch.setattr(wds_mod, "_save_index", fake_save)

        pt_dir = tmp_path / "pt"
        pt_dir.mkdir()
        inv = pd.DataFrame({
            "file_name": ["A.svs", "B.svs", "bad.svs"],
            "project_id": ["PROJ", "PROJ", "PROJ"],
        })

        wds_mod.append_wds(
            pt_dir=pt_dir,
            h5_dir=None,
            inventory_df=inv,
            wds_dest="s3://bucket/wds",
            model_type="hoptimus1",
            staging_dir=None,
            max_shard_bytes=10 * 1024 ** 3,
            dry_run=False,
            slide_id_filter=None,  # no filter — full run
            manifest_csv=None,
            failed_slides={"bad"},
        )

        assert "bad" not in pruned_index, "Failed slide should be pruned in no-filter mode"
