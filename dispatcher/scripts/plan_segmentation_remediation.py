#!/usr/bin/env python3
"""Join a segmentation metadata audit to dispatcher state without mutating it."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from mussel_dispatcher.config import Config, load_runtime_env
from mussel_dispatcher.state import StateStore


def load_affected_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return sorted(
            {
                str(row.get("slide_id") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("status") or "") in {"mismatch", "error"}
                and str(row.get("slide_id") or "").strip()
            }
        )


def query_rows(state: StateStore, slide_ids: list[str], chunk_size: int = 500) -> list[dict]:
    rows = []
    conn = state._conn()
    for offset in range(0, len(slide_ids), chunk_size):
        chunk = slide_ids[offset : offset + chunk_size]
        placeholders = state._placeholders(len(chunk))
        result = conn.execute(
            state._sql(
                f"SELECT slide_id, slide_path, status, fail_count, batch_id "
                f"FROM slides WHERE slide_id IN ({placeholders}) "
                "ORDER BY slide_id, slide_path"
            ),
            chunk,
        ).fetchall()
        rows.extend(dict(row) for row in result)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slide_id", "slide_path", "status", "fail_count", "batch_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    slide_ids = load_affected_ids(args.audit_manifest)
    cfg = Config.load(args.config)
    load_runtime_env(cfg)
    state = StateStore.from_config(cfg)
    try:
        rows = query_rows(state, slide_ids)
    finally:
        state.close_thread_connection()
    write_rows(args.output, rows)

    rows_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_id[str(row["slide_id"])].append(row)
    per_id = Counter({slide_id: len(group) for slide_id, group in rows_by_id.items()})
    duplicate_status_patterns = Counter(
        "+".join(sorted(str(row["status"]) for row in group))
        for group in rows_by_id.values()
        if len(group) > 1
    )
    summary = {
        "affected_ids": len(slide_ids),
        "database_rows": len(rows),
        "database_ids": len(per_id),
        "missing_ids": len(set(slide_ids) - set(per_id)),
        "duplicate_ids": sum(count > 1 for count in per_id.values()),
        "duplicate_status_patterns": dict(duplicate_status_patterns),
        "ids_by_status": {
            status: sum(
                any(str(row["status"]) == status for row in group)
                for group in rows_by_id.values()
            )
            for status in ("PENDING", "DISPATCHED", "SUCCEEDED", "FAILED")
        },
        "statuses": dict(Counter(str(row["status"]) for row in rows)),
    }
    print(json.dumps(summary, sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
