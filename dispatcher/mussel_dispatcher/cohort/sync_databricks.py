#!/usr/bin/env python3
"""Export a cohort's feature inventory to a Databricks Unity Catalog volume.

Reads the dispatcher state DB and each model's WDS index
(``<wds_destination>/<model>/wds_index.json``), builds a Parquet export (one
row per slide × model), and uploads it to a Databricks Unity Catalog volume.
Optionally triggers a Databricks job to MERGE it into a Delta table.

``wds_path`` comes from the WDS index, never from ``wds_manifest.csv``.  The
export holds every (slide, model) and the MERGE overwrites every row, so its
paths must be the authoritative ones: the cumulative manifest is append-only
and goes stale when re-runs, repairs or dedupes rewrite shards, and a partial
manifest (e.g. an out-of-band run's) would blank every other wds_path.
``--wds-manifest`` is kept only to infer the model list when neither
``--model-types`` nor a WDS destination names it.  If a model's index cannot
be loaded while the DB has SUCCEEDED slides, the sync fails instead of
exporting blank paths.  The MERGE is also checked by the wds_path guard in
:mod:`mussel_dispatcher.databricks_sync` (``--allow-wds-path-loss`` skips it).

Export columns:
    slide_id       slide identifier from the Databricks watcher's query
    oncotree_code  cancer type code
    slide_path     original slide path
    model          feature model name (e.g. titan_slide)
    status         PENDING | SUCCEEDED | FAILED
    failure_reason non-empty when status == FAILED
    wds_path       path to the WDS shard (null when not yet complete)
    first_seen_at  ISO timestamp when the slide was first enqueued
    completed_at   ISO timestamp when the slide completed (or null)

Credentials resolved in order:
1. CLI flags --databricks-host / --token
2. Environment variables DATABRICKS_HOST / DATABRICKS_TOKEN
3. ~/.databrickscfg [DEFAULT] section

Usage
-----
    python -m mussel_dispatcher.cohort.sync_databricks \\
        --config cohort_dispatcher.yaml \\
        --model-types titan_slide,hoptimus1 \\
        --volume-folder /Volumes/catalog/schema/cohort_dispatcher \\
        [--table your_catalog.your_schema.slide_embeddings] \\
        [--job-id 12345]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from mussel_dispatcher import wds as wds_mod
from mussel_dispatcher.databricks_sync import add_upload_args, upload_and_trigger
from mussel_dispatcher.config import Config
from mussel_dispatcher.state import StateStore

log = logging.getLogger(__name__)

EXPORT_COLUMNS = [
    "slide_id", "oncotree_code", "slide_path", "model", "status",
    "failure_reason", "wds_path", "first_seen_at", "completed_at",
]

# (wds_dest, model) -> {slide_id: wds_path}.  A production index is ~100-160 MB
# of JSON; the path map kept here is far smaller (paths are interned per shard),
# and a process that builds several exports downloads each index once.
_INDEX_PATH_CACHE: dict[tuple[str, str], dict[str, str]] = {}


class WdsIndexUnavailable(RuntimeError):
    """A model's WDS index could not be loaded, so its wds_paths are unknown."""


def clear_index_cache() -> None:
    _INDEX_PATH_CACHE.clear()


def _index_paths(index: dict, wds_dest: str, model: str) -> dict[str, str]:
    """Map slide_id -> ``<wds_dest>/<model>/<shard_file>`` (what append_wds writes)."""
    prefix = f"{wds_dest.rstrip('/')}/{model}/"
    by_shard: dict[str, str] = {}
    paths: dict[str, str] = {}
    for slide_id, entry in index.items():
        if str(slide_id).startswith("_") or not isinstance(entry, dict):
            continue
        shard_file = str(entry.get("shard_file") or "").strip()
        if not shard_file:
            continue
        path = by_shard.get(shard_file)
        if path is None:
            path = by_shard[shard_file] = prefix + shard_file
        paths[str(slide_id)] = path
    return paths


def load_index_paths(
    wds_dest: str,
    model: str,
    *,
    staging_dir: str | Path | None = None,
    s3_max_concurrency: int = 4,
) -> dict[str, str]:
    """Return {slide_id: wds_path} from ``<wds_dest>/<model>/wds_index.json``.

    S3 indexes are downloaded into a private temporary directory (under
    *staging_dir* when it exists), never into the WDS hook's own
    ``<staging_dir>/<model>/wds_index.json``: an append hook writes that file
    and then uploads it, so replacing it in between would publish this stale
    copy as the index.  Cached per process.  Raises WdsIndexUnavailable when
    the index is missing or empty.
    """
    key = (wds_dest.rstrip("/"), model)
    if key in _INDEX_PATH_CACHE:
        return _INDEX_PATH_CACHE[key]
    index_uri = f"{wds_dest.rstrip('/')}/{model}/wds_index.json"
    tmp_parent = None
    if staging_dir and os.path.isdir(staging_dir) and os.access(staging_dir, os.W_OK):
        tmp_parent = str(staging_dir)
    try:
        with tempfile.TemporaryDirectory(prefix=".databricks-sync-", dir=tmp_parent) as tmp:
            index = wds_mod._load_index(wds_dest, model, Path(tmp), s3_max_concurrency)
            paths = _index_paths(index, wds_dest, model)
            del index
    except Exception as exc:  # S3/botocore errors, unreadable or corrupt JSON
        raise WdsIndexUnavailable(f"could not read WDS index {index_uri}: {exc}") from exc
    if not paths:
        raise WdsIndexUnavailable(f"WDS index {index_uri} is missing or has no slides")
    log.info("Loaded WDS index %s: %d slides", index_uri, len(paths))
    _INDEX_PATH_CACHE[key] = paths
    return paths


def _wds_settings_from_config(cfg: Config) -> tuple[dict[str, str], str, object | None]:
    """(model -> wds_dest, staging_dir, watcher) from the Databricks watcher(s)."""
    watchers = [w for w in cfg.watchers if w.type == "databricks" and w.wds_destinations]
    if not watchers:
        watchers = [w for w in cfg.watchers if w.wds_destinations]
    dests: dict[str, str] = {}
    for w in watchers:
        for model, dest in w.wds_destinations.items():
            if not dest:
                continue
            if model in dests and dests[model].rstrip("/") != dest.rstrip("/"):
                raise WdsIndexUnavailable(
                    f"watchers disagree on the WDS destination of {model}: "
                    f"{dests[model]} vs {dest}")
            dests[model] = dest
    staging = next((w.wds_staging_dir for w in watchers if w.wds_staging_dir), "")
    return dests, staging, (watchers[0] if watchers else None)


def _configure_s3_client(watcher) -> None:
    """Point wds's S3 client at the watcher's endpoint/credentials (as the WDS hook does)."""
    if watcher is None or wds_mod._s3_client_instance is not None:
        return
    wds_mod._s3_client_instance = wds_mod._make_s3_client(
        endpoint_url=(watcher.s3_endpoint or watcher.endpoint_url or None),
        access_key=watcher.s3_access_key or None,
        secret_key=watcher.s3_secret_key or None,
        aws_profile=watcher.aws_profile,
    )


def _parse_wds_dest_args(values: list[str] | None, model_types: list[str] | None) -> dict[str, str]:
    """``--wds-dest`` values: ``model=dest`` pairs, or a root applied to every model."""
    dests: dict[str, str] = {}
    roots: list[str] = []
    for value in values or []:
        model, sep, dest = value.partition("=")
        if sep and not model.startswith("s3:") and "/" not in model:
            dests[model.strip()] = dest.strip()
        else:
            roots.append(value.strip())
    if len(roots) > 1:
        raise SystemExit("--wds-dest: give one root, or model=dest pairs")
    if roots:
        for model in model_types or []:
            dests.setdefault(model, roots[0])
    return dests


def _representative_rows(slides: list[dict]) -> dict[str, dict]:
    """One state row per slide_id: its latest SUCCEEDED row, else its latest row.

    A slide can have several rows, one per slide path (e.g. re-imported from another
    bucket). Keeping whichever row came last reported a slide FAILED, with no
    wds_path, when its old path failed and its new path succeeded, although its
    features are in WDS.

    "Latest" is by completed_at, then first_seen_at, then slide_path, so the choice
    never depends on row order. A row that has not finished has no completed_at and
    ranks below any finished one: a slide that failed and was re-queued under a new
    path stays FAILED until the new run finishes.

    A SKIPPED row is a duplicate path whose slide_id is processed through
    another row, so it ranks below every non-SKIPPED row of that slide_id and
    never stands in for its twin.  If a slide_id somehow has only SKIPPED rows,
    the latest one is used and build_export reports it PENDING (not done, no
    failure), so "SKIPPED" is never exported.
    """
    def rank(row: dict) -> tuple:
        return (row.get("status") != "SKIPPED", row.get("status") == "SUCCEEDED",
                row.get("completed_at") or "", row.get("first_seen_at") or "",
                row.get("slide_path") or "")

    best: dict[str, dict] = {}
    for row in slides:
        sid = row.get("slide_id") or ""
        if sid and (sid not in best or rank(row) >= rank(best[sid])):
            best[sid] = row
    return best


def build_export(
    db_path: str | None,
    wds_manifest_path: str | None,
    model_types: list[str] | None,
    *,
    config_path: str | None = None,
    wds_destinations: dict[str, str] | None = None,
    staging_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build a flat export DataFrame from the state DB and the WDS indexes.

    One row per (slide_id, model).  A SUCCEEDED row's ``wds_path`` is the
    slide's shard in ``<wds_dest>/<model>/wds_index.json``; a SUCCEEDED slide
    absent from the index gets an empty path and is logged as a discrepancy.

    Parameters
    ----------
    db_path:
        Legacy path to the dispatcher state DB when using sqlite.
    wds_manifest_path:
        Path to ``wds_manifest.csv``.  Only used to infer the model list when
        *model_types* is empty and no WDS destination names the models; its
        wds_path column is ignored (see the module docstring).
    model_types:
        Models to fan out to.  If None/empty: the configured WDS destinations'
        models, else the manifest's models, else ``["unknown"]``.
    config_path:
        Dispatcher YAML; supplies the state backend, the watcher's
        ``wds_destination(s)``, ``wds_staging_dir`` and S3 settings.
    wds_destinations:
        model -> WDS root (``s3://bucket/wds`` or a local dir); overrides the
        config's.

    Raises WdsIndexUnavailable when a model has no destination or its index
    cannot be loaded while the DB has SUCCEEDED slides.
    """
    cfg = Config.load(config_path) if config_path else None
    if cfg is not None:
        store = StateStore.from_config(cfg)
        cfg_dests, cfg_staging, watcher = _wds_settings_from_config(cfg)
    else:
        store = StateStore(db_path or "dispatcher.db")
        cfg_dests, cfg_staging, watcher = {}, "", None
    dests = {**cfg_dests, **(wds_destinations or {})}
    staging_dir = staging_dir or cfg_staging or None
    slide_lookup = _representative_rows(store.get_all_slides())

    # Resolve model list
    if not model_types:
        if dests:
            model_types = sorted(dests)
        else:
            manifest_models: list[str] = []
            if wds_manifest_path and Path(wds_manifest_path).exists():
                mdf = pd.read_csv(wds_manifest_path, dtype=str, usecols=lambda c: c == "model")
                if "model" in mdf.columns:
                    manifest_models = sorted(mdf["model"].dropna().unique().tolist())
            model_types = manifest_models or ["unknown"]
    if wds_manifest_path:
        log.info("--wds-manifest %s is not a wds_path source; paths come from the WDS indexes",
                 wds_manifest_path)

    n_succeeded = sum(1 for s in slide_lookup.values() if s.get("status") == "SUCCEEDED")
    index_paths: dict[str, dict[str, str]] = {}
    if n_succeeded:
        _configure_s3_client(watcher)
        for model in model_types:
            dest = dests.get(model)
            if not dest:
                raise WdsIndexUnavailable(
                    f"no WDS destination for model {model!r}: wds_path cannot be derived for "
                    f"{n_succeeded} SUCCEEDED slides (pass --config with the watcher's "
                    "wds_destination, or --wds-dest)")
            index_paths[model] = load_index_paths(dest, model, staging_dir=staging_dir)

    missing: dict[str, list[str]] = {m: [] for m in model_types}

    # Fan out: one row per (slide_id, model)
    rows = []
    for slide_id, slide in slide_lookup.items():
        db_status = slide.get("status") or "PENDING"
        # Every other dispatcher status (DISPATCHED, a lone SKIPPED) is not
        # finished and not failed, so it is exported as PENDING.
        if db_status == "SUCCEEDED":
            status = "SUCCEEDED"
        elif db_status == "FAILED":
            status = "FAILED"
        else:
            status = "PENDING"

        failure_reason = slide.get("error_msg") or ""
        if status != "FAILED":
            failure_reason = ""

        for model in model_types:
            completed_at = slide.get("completed_at") or ""
            wds_path = ""
            if status == "SUCCEEDED":
                wds_path = index_paths[model].get(str(slide_id), "")
                if not wds_path:
                    missing[model].append(str(slide_id))
            else:
                completed_at = ""
            rows.append({
                "slide_id": str(slide_id),
                "oncotree_code": slide.get("oncotree_code") or "",
                "slide_path": slide.get("slide_path") or "",
                "model": model,
                "status": status,
                "failure_reason": failure_reason,
                "wds_path": wds_path,
                "first_seen_at": slide.get("first_seen_at") or "",
                "completed_at": completed_at,
            })

    for model, sids in missing.items():
        if sids:
            log.warning(
                "WDS index discrepancy: %d SUCCEEDED slides have no %s entry in the index "
                "and are exported with an empty wds_path (e.g. %s)",
                len(sids), model, ", ".join(sorted(sids)[:10]))

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=EXPORT_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default="dispatcher.db",
                        help="Legacy path to the dispatcher SQLite database")
    parser.add_argument("--config", default=None,
                        help="Dispatcher YAML config path; preferred for backend-aware state "
                             "access, and the source of the WDS destinations and S3 settings")
    parser.add_argument("--wds-manifest", default=None,
                        help="Path to wds_manifest.csv.  Only used to infer --model-types when "
                             "no WDS destination is known; wds_path comes from the WDS indexes")
    parser.add_argument("--wds-dest", action="append", default=None,
                        help="WDS root (s3://bucket/wds) for every model, or model=root; "
                             "repeatable.  Overrides the config's wds_destination(s)")
    parser.add_argument("--staging-dir", default=None,
                        help="Parent dir for temporary index downloads "
                             "(default: the watcher's wds_staging_dir)")
    parser.add_argument("--model-types", default=None,
                        help="Comma-separated model types, e.g. titan_slide,hoptimus1. "
                             "Inferred from the WDS destinations (or wds_manifest) if omitted.")
    add_upload_args(parser)
    args = parser.parse_args(argv)

    if not args.volume_folder and not args.volume_path:
        parser.error("One of --volume-folder or --volume-path is required")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    model_types = (
        [m.strip() for m in args.model_types.split(",") if m.strip()]
        if args.model_types
        else None
    )

    try:
        export_df = build_export(
            args.db, args.wds_manifest, model_types, config_path=args.config,
            wds_destinations=_parse_wds_dest_args(args.wds_dest, model_types),
            staging_dir=args.staging_dir,
        )
    except WdsIndexUnavailable as exc:
        log.error("Refusing to sync: %s", exc)
        raise SystemExit(f"Databricks sync aborted: {exc}") from exc
    log.info("Built export: %d rows", len(export_df))

    upload_and_trigger(export_df, args, filename_prefix="cohort_inventory_",
                       wds_path_guard=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
