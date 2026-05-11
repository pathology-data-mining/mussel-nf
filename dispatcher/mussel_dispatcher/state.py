"""SQLite-backed state store for mussel-dispatcher."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("mussel-dispatcher")


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=60000")
        return self._local.conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS slides (
                slide_path    TEXT PRIMARY KEY,
                slide_id      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'PENDING',
                batch_id      TEXT,
                download_path TEXT,
                fail_count    INTEGER NOT NULL DEFAULT 0,
                first_seen_at  TEXT,
                dispatched_at  TEXT,
                completed_at   TEXT,
                error_msg      TEXT,
                file_id        TEXT,
                file_name      TEXT,
                needs_download INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id      TEXT PRIMARY KEY,
                status        TEXT NOT NULL DEFAULT 'RUNNING',
                slide_count   INTEGER,
                dispatched_at TEXT,
                completed_at  TEXT,
                nextflow_exit INTEGER,
                csv_path      TEXT,
                work_dir      TEXT,
                log_path      TEXT,
                manifest_path TEXT,
                session_id    TEXT,
                tasks_done    INTEGER,
                tasks_total   INTEGER,
                tasks_failed  INTEGER
            );
            -- migrations for databases created before tasks_* columns existed
            CREATE TABLE IF NOT EXISTS _schema_migrations (name TEXT PRIMARY KEY);
        """)
        for col, coltype in [("tasks_done", "INTEGER"), ("tasks_total", "INTEGER"), ("tasks_failed", "INTEGER")]:
            try:
                conn.execute(f"ALTER TABLE batches ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists
        conn.commit()

    # -----------------------------------------------------------------------
    # Slide operations
    # -----------------------------------------------------------------------

    def add_slide(self, slide_path: str, slide_id: str, *,
                  file_id: str = "", file_name: str = "",
                  needs_download: bool = False) -> bool:
        """Insert a new slide. Returns True if inserted, False if already known."""
        try:
            self._conn().execute(
                """INSERT INTO slides (slide_path, slide_id, status, first_seen_at,
                   file_id, file_name, needs_download)
                   VALUES (?, ?, 'PENDING', ?, ?, ?, ?)""",
                (slide_path, slide_id,
                 datetime.now(timezone.utc).isoformat(),
                 file_id, file_name, int(needs_download)),
            )
            self._conn().commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def is_known(self, slide_path: str) -> bool:
        return bool(self._conn().execute(
            "SELECT 1 FROM slides WHERE slide_path=?", (slide_path,)
        ).fetchone())

    def is_known_by_id(self, slide_id: str) -> bool:
        return bool(self._conn().execute(
            "SELECT 1 FROM slides WHERE slide_id=?", (slide_id,)
        ).fetchone())

    def get_slides_by_id(self, slide_id: str) -> list:
        rows = self._conn().execute(
            "SELECT * FROM slides WHERE slide_id=?", (slide_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_slide(self, slide_path: str):
        self._conn().execute("DELETE FROM slides WHERE slide_path=?", (slide_path,))
        self._conn().commit()

    def get_pending_slides(self) -> list:
        rows = self._conn().execute(
            """SELECT slide_path, slide_id, download_path, file_id, file_name, needs_download
               FROM slides WHERE status='PENDING' AND slide_path != ''"""
        ).fetchall()
        slides = [dict(r) for r in rows]
        # Backfill file_id/file_name from gdc:// URIs for rows from older DB schema
        for s in slides:
            if s.get("needs_download") and not s.get("file_id"):
                sp = s.get("slide_path", "")
                if sp.startswith("gdc://"):
                    rest = sp[len("gdc://"):]
                    slash = rest.find("/")
                    if slash > 0:
                        s["file_id"] = rest[:slash]
                        s["file_name"] = rest[slash + 1:]
        return slides

    def mark_dispatched(self, slide_paths: list, batch_id: str) -> int:
        """Atomically claim PENDING slides for a batch.

        Only slides still in PENDING status are updated.  Returns the number
        of slides actually claimed.  If this is less than len(slide_paths),
        another dispatcher instance raced and claimed some of them first.
        """
        now = datetime.now(timezone.utc).isoformat()
        claimed = 0
        conn = self._conn()
        for p in slide_paths:
            cur = conn.execute(
                "UPDATE slides SET status='DISPATCHED', batch_id=?, dispatched_at=?"
                " WHERE slide_path=? AND status='PENDING'",
                (batch_id, now, p),
            )
            claimed += cur.rowcount
        conn.commit()
        return claimed

    def mark_slides_complete(self, batch_id: str, succeeded: bool,
                             per_slide_status: dict | None = None,
                             charge_fail_count: bool = True):
        """Mark all slides in a batch as SUCCEEDED or FAILED.

        When charge_fail_count=False (fast infra failure), slides are reset to
        PENDING instead of FAILED so they don't burn a retry slot.
        """
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        if per_slide_status:
            for slide_path, ok in per_slide_status.items():
                status = "SUCCEEDED" if ok else "FAILED"
                conn.execute(
                    "UPDATE slides SET status=?, completed_at=? WHERE slide_path=? AND batch_id=?",
                    (status, now, slide_path, batch_id),
                )
        elif not succeeded and not charge_fail_count:
            # Infra/config failure — reset to PENDING without charging fail_count
            conn.execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL WHERE batch_id=? AND status='DISPATCHED'",
                (batch_id,),
            )
        else:
            status = "SUCCEEDED" if succeeded else "FAILED"
            conn.execute(
                """UPDATE slides SET status=?, completed_at=?
                   WHERE batch_id=? AND status='DISPATCHED'""",
                (status, now, batch_id),
            )
            if not succeeded:
                conn.execute(
                    "UPDATE slides SET fail_count=fail_count+1 WHERE batch_id=? AND status='FAILED'",
                    (batch_id,),
                )
        conn.commit()

    def reset_dispatched_to_pending(self, batch_id: str):
        self._conn().execute(
            "UPDATE slides SET status='PENDING', batch_id=NULL WHERE batch_id=? AND status='DISPATCHED'",
            (batch_id,),
        )
        self._conn().commit()

    def reset_failed_to_pending(self, max_retries: int = 0) -> int:
        """Reset FAILED slides to PENDING. Slides at/above max_retries are left FAILED."""
        conn = self._conn()
        if max_retries > 0:
            conn.execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL, error_msg=NULL "
                "WHERE status='FAILED' AND fail_count < ?",
                (max_retries,),
            )
            skipped = conn.execute(
                "SELECT COUNT(*) FROM slides WHERE status='FAILED' AND fail_count >= ?",
                (max_retries,),
            ).fetchone()[0]
            if skipped:
                log.warning("Permanently skipping %d slide(s) with fail_count >= %d.",
                            skipped, max_retries)
        else:
            conn.execute(
                "UPDATE slides SET status='PENDING', batch_id=NULL, dispatched_at=NULL, "
                "error_msg=NULL WHERE status='FAILED'"
            )
        conn.commit()
        return conn.execute(
            "SELECT COUNT(*) FROM slides WHERE status='PENDING'"
        ).fetchone()[0]

    def blacklist_slide(self, slide_id: str, reason: str, max_retries: int = 999):
        """Permanently exclude a slide from future dispatch.

        Sets status=FAILED and fail_count=max_retries so reset_failed_to_pending()
        never re-queues it. Inserts a tombstone row if the slide is not yet known.
        """
        conn = self._conn()
        existing = conn.execute(
            "SELECT slide_path FROM slides WHERE slide_id=?", (slide_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE slides SET status='FAILED', fail_count=?, error_msg=?,
                   completed_at=?, batch_id=NULL WHERE slide_id=?""",
                (max_retries, reason, datetime.now(timezone.utc).isoformat(), slide_id),
            )
        else:
            conn.execute(
                """INSERT INTO slides (slide_path, slide_id, status, fail_count, error_msg, first_seen_at)
                   VALUES ('', ?, 'FAILED', ?, ?, ?)""",
                (slide_id, max_retries, reason, datetime.now(timezone.utc).isoformat()),
            )
        conn.commit()
        log.warning("Blacklisted slide %s: %s", slide_id, reason)

    def set_download_path(self, slide_path: str, download_path: str):
        self._conn().execute(
            "UPDATE slides SET download_path=? WHERE slide_path=?", (download_path, slide_path)
        )
        self._conn().commit()

    def get_batch_download_paths(self, batch_id: str) -> list:
        rows = self._conn().execute(
            "SELECT download_path FROM slides WHERE batch_id=? AND download_path IS NOT NULL",
            (batch_id,),
        ).fetchall()
        return [r["download_path"] for r in rows]

    def get_old_batch_logs(self, older_than_days: int) -> list:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        rows = self._conn().execute(
            """SELECT batch_id, log_path FROM batches
               WHERE status='SUCCEEDED' AND log_path IS NOT NULL AND completed_at < ?""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Batch operations
    # -----------------------------------------------------------------------

    def add_batch(self, batch_id: str, csv_path: str, work_dir: str,
                  slide_count: int, log_path: str):
        self._conn().execute(
            """INSERT OR REPLACE INTO batches
               (batch_id, status, slide_count, dispatched_at, csv_path, work_dir, log_path)
               VALUES (?, 'RUNNING', ?, ?, ?, ?, ?)""",
            (batch_id, slide_count, datetime.now(timezone.utc).isoformat(),
             csv_path, work_dir, log_path),
        )
        self._conn().commit()

    def restart_batch(self, batch_id: str):
        """Mark a recovered batch as RUNNING again (for -resume runs)."""
        self._conn().execute(
            "UPDATE batches SET status='RUNNING', completed_at=NULL WHERE batch_id=?",
            (batch_id,),
        )
        self._conn().commit()

    def cancel_batch(self, batch_id: str):
        """Remove a batch record that was never actually started (race-condition abort)."""
        self._conn().execute("DELETE FROM batches WHERE batch_id=?", (batch_id,))
        self._conn().commit()

    def complete_batch(self, batch_id: str, exit_code: int):
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        self._conn().execute(
            """UPDATE batches SET status=?, completed_at=?, nextflow_exit=?
               WHERE batch_id=?""",
            (status, datetime.now(timezone.utc).isoformat(), exit_code, batch_id),
        )
        self._conn().commit()

    def record_batch_task_counts(self, batch_id: str, done: int, total: int, failed: int):
        """Persist final task counts from the Tower complete event."""
        self._conn().execute(
            "UPDATE batches SET tasks_done=?, tasks_total=?, tasks_failed=? WHERE batch_id=?",
            (done, total, failed, batch_id),
        )
        self._conn().commit()

    def record_batch_manifest(self, batch_id: str, manifest_path: str):
        self._conn().execute(
            "UPDATE batches SET manifest_path=? WHERE batch_id=?", (manifest_path, batch_id)
        )
        self._conn().commit()

    def get_all_manifest_paths(self) -> list:
        rows = self._conn().execute(
            "SELECT manifest_path FROM batches WHERE manifest_path IS NOT NULL"
        ).fetchall()
        return [r["manifest_path"] for r in rows]

    def get_running_batches(self) -> list:
        rows = self._conn().execute(
            "SELECT * FROM batches WHERE status='RUNNING'"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_batch_session_id(self, batch_id: str, session_id: str):
        self._conn().execute(
            "UPDATE batches SET session_id=? WHERE batch_id=?", (session_id, batch_id)
        )
        self._conn().commit()

    def get_batch_session_id(self, batch_id: str) -> str | None:
        row = self._conn().execute(
            "SELECT session_id FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        return row["session_id"] if row else None

    def get_finished_batches_with_work_dirs(self) -> list:
        rows = self._conn().execute(
            "SELECT batch_id, work_dir FROM batches WHERE status IN ('SUCCEEDED','FAILED') AND work_dir IS NOT NULL"
        ).fetchall()
        return [dict(r) for r in rows]
