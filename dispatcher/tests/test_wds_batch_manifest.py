"""WDS coverage is verified from a per-batch manifest, not the cumulative one.

``{outdir}/wds_manifest.csv`` is appended to by every WDS hook run and never
pruned (the Databricks sync reads it), so a slide that was appended once, later
lost from the WDS index, and re-run without output still has rows there. Batch
finalization must not count those stale rows as this batch's coverage.
"""
from __future__ import annotations

import csv
import subprocess

from mussel_dispatcher.batch_lifecycle import BatchFinalizer, FinalizationContext, PostBatchHooks
from mussel_dispatcher.config import batch_wds_manifest_path
from tests.test_review_followups import _real_wds_batch, _run_wds

MODELS = ("model-a", "model-b")


def _wds_executor(ran):
    def executor(cmd, **kwargs):
        if "mussel_dispatcher.wds" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        model, rc = _run_wds(cmd)
        ran.append((model, rc))
        return subprocess.CompletedProcess(cmd, rc, "", "")
    return executor


def _write_stale_cumulative_rows(outdir, slide_id):
    """Rows an earlier run appended for ``slide_id`` before it left the WDS index."""
    with open(outdir / "wds_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "model", "wds_path"])
        for model in MODELS:
            w.writerow([slide_id, model, f"s3://old/{model}/shard-000000.tar"])


def _rows(path):
    with open(path, newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _slides(state):
    return {r["slide_id"]: dict(r) for r in state._conn().execute(
        "SELECT slide_id, status, fail_count FROM slides")}


def _finalize(cfg, state, csv_path, outdir, ran):
    return BatchFinalizer(cfg, "batch-1", state).finalize_success(
        FinalizationContext(batch_csv=str(csv_path), outdir=str(outdir), repo_dir=cfg.repo_dir),
        hook_executor=_wds_executor(ran), collect_manifest=lambda started: None)


def test_stale_cumulative_rows_do_not_cover_a_slide_rerun_without_output(tmp_path):
    # b was appended by an earlier run (stale cumulative rows), is no longer in
    # the WDS index, and this run produced no output for it.
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {"model-a": "a", "model-b": "a"})
    _write_stale_cumulative_rows(outdir, "b")
    ran = []

    _finalize(cfg, state, csv_path, outdir, ran)

    assert ran == [("model-a", 2), ("model-b", 2)]          # both hooks lack b
    slides = _slides(state)
    assert slides["b"]["status"] == "PENDING" and slides["b"]["fail_count"] == 1
    assert slides["a"]["status"] == "SUCCEEDED" and slides["a"]["fail_count"] == 0


def test_per_batch_manifest_holds_only_this_runs_rows(tmp_path):
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {"model-a": "ab", "model-b": "a"})
    _write_stale_cumulative_rows(outdir, "zz-other-batch")
    ran = []

    _finalize(cfg, state, csv_path, outdir, ran)

    per_batch = batch_wds_manifest_path(str(outdir), "batch-1")
    rows = _rows(per_batch)
    assert sorted((r["slide_id"], r["model"]) for r in rows) == [
        ("a", "model-a"), ("a", "model-b"), ("b", "model-a")]
    assert all(r["wds_path"].startswith(str(tmp_path / "wds")) for r in rows)
    with open(per_batch) as fh:                              # one header across both hooks
        assert fh.read().count("slide_id,model,wds_path") == 1
    # The cumulative manifest keeps the earlier rows and gains this run's.
    cumulative = {(r["slide_id"], r["model"]) for r in _rows(outdir / "wds_manifest.csv")}
    assert {("zz-other-batch", "model-a"), ("a", "model-a"), ("b", "model-a")} <= cumulative
    assert _slides(state)["b"]["status"] == "PENDING"        # model-b lacks b


def test_already_indexed_batch_slides_are_confirmed_in_the_per_batch_manifest(tmp_path):
    # Batch 1 appends a and b. Batch 2 re-runs both with no new output: they
    # are still in the index, so batch 2's manifest confirms them.
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {"model-a": "ab", "model-b": "ab"})
    _finalize(cfg, state, csv_path, outdir, [])
    hook = next(h for h in cfg.post_batch_hooks if "--model-type=model-a" in h["args"])
    cmd = ["python", "-m", "mussel_dispatcher.wds", *(
        a.format(outdir=outdir, batch_csv=csv_path, batch_id="batch-2") for a in hook["args"])]

    _, rc = _run_wds(cmd)

    assert rc == 0
    assert sorted((r["slide_id"], r["model"]) for r in _rows(
        batch_wds_manifest_path(str(outdir), "batch-2"))) == [("a", "model-a"), ("b", "model-a")]


def test_hooks_with_a_per_batch_manifest_never_fall_back_to_the_cumulative_one(tmp_path):
    # The hooks were given a per-batch manifest but did not write one: the
    # cumulative rows for a and b must not count.
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {})
    _write_stale_cumulative_rows(outdir, "a")
    with open(outdir / "wds_manifest.csv", "a", newline="") as fh:
        csv.writer(fh).writerows([["b", m, "s3://old/x.tar"] for m in MODELS])
    state.mark_slides_complete("batch-1", succeeded=True)
    state.sync_batch_hooks("batch-1", PostBatchHooks(cfg, "batch-1", state).state_rows())

    BatchFinalizer(cfg, "batch-1", state)._verify_wds_coverage(str(csv_path), str(outdir))

    assert {s["status"] for s in _slides(state).values()} == {"PENDING"}


def test_batches_whose_hooks_predate_the_per_batch_manifest_use_the_cumulative_one(tmp_path):
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {})
    _write_stale_cumulative_rows(outdir, "a")
    with open(outdir / "wds_manifest.csv", "a", newline="") as fh:
        csv.writer(fh).writerows([["b", m, "s3://new/x.tar"] for m in MODELS])
    state.mark_slides_complete("batch-1", succeeded=True)
    for hook in cfg.post_batch_hooks:                        # what older code recorded
        hook["args"] = [a for a in hook.get("args", []) if not a.startswith("--batch-manifest-csv")]
    state.sync_batch_hooks("batch-1", PostBatchHooks(cfg, "batch-1", state).state_rows())

    assert BatchFinalizer(cfg, "batch-1", state)._verify_wds_coverage(str(csv_path), str(outdir))
    assert {s["status"] for s in _slides(state).values()} == {"SUCCEEDED"}


def test_every_wds_hook_writes_the_per_batch_manifest_coverage_reads(tmp_path):
    from mussel_dispatcher.config import WatcherConfig
    from tests.test_batch_lifecycle import make_config

    cfg = make_config(tmp_path, watchers=[
        WatcherConfig(type="databricks", wds_destinations={"m1": "s3://wds"}),
        WatcherConfig(type="tcga", inventory_csv="inv.csv", status_csv="st.csv",
                      wds_destinations={"m2": "s3://wds"}),
    ])
    wds_hooks = [h for h in cfg._build_auto_hooks() if "mussel_dispatcher.wds" in h["command"]]
    assert len(wds_hooks) == 2
    for hook in wds_hooks:
        arg = next(a for a in hook["args"] if a.startswith("--batch-manifest-csv="))
        rendered = arg.split("=", 1)[1].format(outdir="/out", batch_id="B1")
        assert rendered == batch_wds_manifest_path("/out", "B1")


def test_a_hook_that_appends_nothing_still_leaves_a_header_only_per_batch_manifest(tmp_path):
    # Neither model produced output: the file still records that this batch's
    # hooks ran with a per-batch manifest (and an empty one covers nothing).
    cfg, state, csv_path, outdir = _real_wds_batch(tmp_path, {})
    ran = []

    _finalize(cfg, state, csv_path, outdir, ran)

    per_batch = batch_wds_manifest_path(str(outdir), "batch-1")
    with open(per_batch) as fh:
        assert fh.read().strip() == "slide_id,model,wds_path"
    assert {s["status"] for s in _slides(state).values()} == {"PENDING"}
