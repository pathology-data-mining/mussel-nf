"""Shared WDS status helpers used by the dispatcher snapshot and dashboard."""
from __future__ import annotations

import csv
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config, WatcherConfig


def parse_wds_manifest(
    manifest_path: str,
    models: list[str] | None = None,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Parse a WDS manifest CSV and return (unique_slides, shard_slide_counts)."""
    unique_slides: dict[str, set[str]] = {}
    shard_slide_counts: dict[str, dict[str, int]] = {}
    if not os.path.exists(manifest_path):
        return unique_slides, shard_slide_counts
    try:
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                model = row.get("model", "unknown")
                slide_id = row.get("slide_id", "")
                shard = row.get("wds_path", "")
                if models is not None and model not in models:
                    continue
                if slide_id:
                    unique_slides.setdefault(model, set()).add(slide_id)
                model_shards = shard_slide_counts.setdefault(model, {})
                model_shards[shard] = model_shards.get(shard, 0) + 1
    except Exception:
        return {}, {}
    return unique_slides, shard_slide_counts


def parse_wds_index(index_path: str) -> tuple[set[str], dict[str, str]]:
    """Parse a WDS index JSON and return (slide_ids, slide_to_shard_file)."""
    slide_ids: set[str] = set()
    slide_to_shard: dict[str, str] = {}
    if not os.path.exists(index_path):
        return slide_ids, slide_to_shard
    try:
        with open(index_path) as f:
            payload = json.load(f)
    except Exception:
        return set(), {}

    for slide_id, info in payload.items():
        if slide_id.startswith("_") or not isinstance(info, dict):
            continue
        shard_file = info.get("shard_file", "")
        if not shard_file:
            continue
        slide_ids.add(slide_id)
        slide_to_shard[slide_id] = shard_file
    return slide_ids, slide_to_shard


def inventory_total_for_watcher(cfg: "Config", watcher: "WatcherConfig | None") -> int:
    if watcher is None or not getattr(watcher, "inventory_csv", ""):
        return 0
    inv_path = watcher.inventory_csv
    slide_type_prefix = getattr(watcher, "slide_type", "") or ""
    if not os.path.exists(inv_path):
        return 0
    try:
        with open(inv_path, newline="") as f:
            reader = csv.DictReader(f)
            if slide_type_prefix:
                return sum(
                    1
                    for row in reader
                    if row.get("slide_type", "").startswith(slide_type_prefix)
                )
            return sum(1 for _ in reader)
    except Exception:
        return 0


def local_pt_counts(outdir: str, models: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    features_root = os.path.join(outdir, "features")
    for model in models:
        model_dir = os.path.join(features_root, model)
        n = 0
        if os.path.isdir(model_dir):
            for _root, _dirs, files in os.walk(model_dir):
                n += sum(1 for name in files if name.endswith(".features.pt"))
        counts[model] = n
    return counts


def wds_index_data(staging_dir: str, models: set[str]) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for model in models:
        index_path = os.path.join(staging_dir, model, "wds_index.json")
        slide_ids, slide_to_shard = parse_wds_index(index_path)
        data[model] = {
            "slides": slide_ids,
            "slide_to_shard": slide_to_shard,
        }
    return data


def build_wds_snapshot(cfg: "Config") -> dict:
    """Build a dashboard-compatible WDS summary payload without DB reads."""
    watcher = next((w for w in cfg.watchers if getattr(w, "wds_destinations", None)), None)
    inventory_total = inventory_total_for_watcher(cfg, watcher)
    if watcher is None or not watcher.wds_destinations:
        return {
            "models": {},
            "total": 0,
            "inventory_total": inventory_total,
            "error": None,
            "source": "dispatcher_snapshot",
        }

    models = set(watcher.wds_destinations.keys())
    wds_staging_dir = getattr(watcher, "wds_staging_dir", "") or os.path.join(cfg.state_dir, "wds-staging")
    manifest_path = os.path.join(cfg.outdir, "wds_manifest.csv")

    error: str | None = None
    try:
        index_data = wds_index_data(wds_staging_dir, models)
        unique_slides, shard_slide_counts = parse_wds_manifest(manifest_path, models=sorted(models))
        pt_counts = local_pt_counts(cfg.outdir, models)
    except Exception as exc:  # pragma: no cover - defensive guard around fs churn
        index_data = {}
        unique_slides = {}
        shard_slide_counts = {}
        pt_counts = {}
        error = str(exc)

    try:
        from .dashboard import helpers as dashboard_helpers

        s3_data = dashboard_helpers.s3_stats(watcher, "wds")
    except Exception as exc:  # pragma: no cover - network/env dependent
        s3_data = {}
        error = error or str(exc)

    model_payloads: dict[str, dict] = {}
    for model in sorted(models | set(unique_slides)):
        model_index = index_data.get(model, {})
        indexed_slides = model_index.get("slides", set())
        slide_to_shard = model_index.get("slide_to_shard", {})
        indexed_shard_counts: dict[str, int] = {}
        for slide_id in indexed_slides:
            shard = slide_to_shard.get(slide_id)
            if shard:
                indexed_shard_counts[shard] = indexed_shard_counts.get(shard, 0) + 1
        all_index_shards = {
            slide_to_shard[sid] for sid in indexed_slides if slide_to_shard.get(sid)
        }
        per_shard = list(indexed_shard_counts.values())
        shard_stats = {}
        if per_shard:
            shard_stats = {
                "avg": round(sum(per_shard) / len(per_shard), 1),
                "min": min(per_shard),
                "max": max(per_shard),
                "count": len(per_shard),
            }
        model_s3 = s3_data.get(model, {})
        model_payloads[model] = {
            "slides": len(indexed_slides),
            "gap": 0,
            "shards": model_s3.get("shards", 0),
            "index_shards": len(all_index_shards),
            "current_index_shards": len(per_shard),
            "manifest_shards": len((shard_slide_counts.get(model) or {}).values()),
            "shard_stats": shard_stats,
            "local_pt": pt_counts.get(model, 0),
            "error": model_s3.get("error"),
            "indexed_total": len(indexed_slides),
        }

    total_rows = sum(len(info.get("slides", set())) for info in index_data.values())
    return {
        "models": model_payloads,
        "total": total_rows,
        "inventory_total": inventory_total,
        "error": error,
        "source": "dispatcher_snapshot",
    }
