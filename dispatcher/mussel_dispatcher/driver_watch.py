"""Watch a remote Nextflow driver: lost Slurm jobs and stalled runs.

Stdlib only (Python >= 3.9): the remote driver runs this file as a script
next to Nextflow (``<python> driver_watch.py --run-dir ...``, with the
agent's interpreter or ``driver_watch_python``), and the remote agent
imports it to read the result.

Lost jobs: sbatch returned a job id while slurmctld was
failing, but the job never ran and its id was later reused. Nextflow waits
on a SUBMITTED task forever (its missing-exit-file timeout applies only to
started tasks). The watcher fails such a task the way a real failure would,
.command.begin then a nonzero .exitcode, so the process's errorStrategy
applies. It never touches a task whose job is queued under our user and
name, a task that started, or one with an exit code; it fails a task only
when two consecutive polls find its job lost, and first scancels the job
(by id, our user and the task's job name) so a misjudged one cannot run
beside the retry.

Stalls: ``assess_stall`` reads nextflow.log for a run that has nothing to
run ("No more task to compute" from every executor) or has neither
submitted nor completed a task for a long time while no executor has tasks
in flight. ``nextflow_abort_cause`` tells a run Nextflow aborted because
Slurm would not take or keep its jobs (an infrastructure failure) from one
that failed on its own.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

STATUS_FILE = "driver_watch.json"
MARKER_FILE = ".mussel_lost_job.json"
# EX_TEMPFAIL: tells a lost job apart from a task's own exit code in trace.tsv.
LOST_JOB_EXIT_CODE = 75

_TS = re.compile(r"^([A-Z][a-z]{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) ")
_SUBMITTED = re.compile(
    r"\[SLURM\] submitted process (?P<name>.+?) > jobId: (?P<job>\d+); workDir: (?P<wd>\S+)"
)
_PROGRESS = re.compile(r"Submitted process > |Task completed > |Cached process > ")
_DUMP = re.compile(r"!! executor (?P<ex>\S+) > (?P<what>No more task to compute|tasks to be completed)")
# Why Nextflow aborted the session (logged once, at DEBUG, when it terminates).
_ABORTED = re.compile(r"Session aborted -- Cause: (?P<cause>.*)")
_SUBMIT_FAILED = re.compile(r"^Error submitting process '(?P<task>.+)' for execution")
_EXIT_STATUS = re.compile(r"^Process `(?P<task>.+)` terminated with an error exit status \((?P<code>\d+)\)")
# sacct states after which a job can no longer start.
_ENDED = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OOM", "REVOKED",
})

Runner = Callable[[list[str]], str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime | None) -> str | None:
    return when.astimezone(timezone.utc).isoformat() if when else None


def parse_log_time(stamp: str, now: datetime) -> datetime | None:
    """Nextflow log stamps (``Jan-01 12:00:00.000``) are local time without a year."""
    try:
        naive = datetime.strptime(f"{now.astimezone().year}-{stamp}", "%Y-%b-%d %H:%M:%S.%f")
    except ValueError:
        return None
    local = naive.astimezone()
    if local - now > timedelta(days=1):   # December log read in January
        local = local.replace(year=local.year - 1)
    return local.astimezone(timezone.utc)


def assess_stall(
    text: str,
    now: datetime,
    *,
    idle_seconds: float,
    no_progress_seconds: float,
) -> dict[str, Any] | None:
    """Return a stall finding for a nextflow.log (tail), or None.

    ``idle_with_active_nodes``: every executor's latest status dump says it has
    nothing to run, after the last task event, for ``idle_seconds``: the
    dataflow is waiting on a channel that will never close.
    ``no_task_progress``: no task submitted, cached or completed for
    ``no_progress_seconds`` (0 disables) while no executor's latest status
    dump has tasks in flight. A busy executor is waiting on its queue (GPU
    jobs can pend for a day) or on a task the lost-job watch will fail, so
    it never counts. Only ``idle_with_active_nodes`` is ``cancellable``:
    nothing can make that run progress again.
    """
    first = last_progress = None
    dumps: dict[str, tuple[datetime, bool]] = {}
    current = None
    for line in text.splitlines():
        m = _TS.match(line)
        if not m:
            continue
        current = parse_log_time(m.group(1), now) or current
        if current is None:
            continue
        first = first or current
        if _PROGRESS.search(line):
            last_progress = current
            continue
        d = _DUMP.search(line)
        if d:
            dumps[d.group("ex")] = (current, d.group("what").startswith("No more"))
    if first is None:
        return None
    # No task event in the tail: progress is at least as old as its first line.
    since = last_progress or first
    quiet = (now - since).total_seconds()
    idle = bool(dumps) and all(
        idle_now and at >= since for at, idle_now in dumps.values()
    )
    busy = any(not idle_now for _, idle_now in dumps.values())
    finding = {
        "failure_class": "DRIVER_STALLED",
        "last_task_event_at": _iso(last_progress),
        "quiet_seconds": int(quiet),
        "idle_executors": sorted(ex for ex, (_, idle_now) in dumps.items() if idle_now),
        "busy_executors": sorted(ex for ex, (_, idle_now) in dumps.items() if not idle_now),
    }
    if idle and idle_seconds > 0 and quiet >= idle_seconds:
        return {**finding, "reason": "idle_with_active_nodes", "cancellable": True}
    if not busy and no_progress_seconds > 0 and quiet >= no_progress_seconds:
        return {**finding, "reason": "no_task_progress", "cancellable": False}
    return None


def nextflow_abort_cause(text: str) -> dict[str, Any] | None:
    """The infrastructure failure that made Nextflow abort this run, or None.

    ``submission_failed``: a task's sbatch kept failing past every retry
    (slurmctld down). ``lost_job``: a task the watch failed as lost (exit
    ``LOST_JOB_EXIT_CODE``) was lost again on its last attempt. Any other
    abort cause is the run's own failure.
    """
    cause = None
    for line in text.splitlines():
        m = _ABORTED.search(line)
        if m:
            cause = m.group("cause").strip()
    if cause is None:
        return None
    m = _SUBMIT_FAILED.match(cause)
    if m:
        return {"reason": "submission_failed", "task": m.group("task"), "cause": cause}
    m = _EXIT_STATUS.match(cause)
    if m and int(m.group("code")) == LOST_JOB_EXIT_CODE:
        return {"reason": "lost_job", "task": m.group("task"), "cause": cause}
    return None


def read_tail(path: Path, max_bytes: int) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read()
    if size > max_bytes:
        data = data.split(b"\n", 1)[-1]
    return data.decode("utf-8", errors="replace")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout


def slurm_job_name(work_dir: Path) -> str | None:
    """The job name Nextflow gave this task (``#SBATCH -J`` in .command.run)."""
    try:
        with open(work_dir / ".command.run", "r", errors="replace") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                m = re.match(r"#SBATCH\s+(?:-J|--job-name[= ])\s*(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


def classify(
    job_id: str,
    name: str,
    user: str,
    queued: dict[str, tuple[str, str]],
    sacct_rows: Iterable[dict[str, str]] | None,
) -> str | None:
    """Why a submitted, unstarted task's job is lost, or None if it may still run.

    ``queued``: our squeue rows, job id -> (name, user). ``sacct_rows``: every
    accounting record for the id, any user (None: sacct did not answer).
    """
    row = queued.get(job_id)
    if row is not None:
        return None if row[0] == name else "id_reused_by_other_job"
    if sacct_rows is None:
        return None
    rows = [r for r in sacct_rows if r.get("job_id") == job_id]
    ours = [r for r in rows if r.get("user") == user and r.get("name") == name]
    if not ours:
        return "id_reused_by_other_job" if rows else "no_accounting_record"
    if any(r.get("state") not in _ENDED for r in ours):
        return None   # accounting says it may still run; squeue disagreeing is lag
    if any(r.get("start") not in ("", "None", "Unknown", None) for r in ours):
        # It got a node but the task never began (BOOT_FAIL, NODE_FAIL, killed in
        # the prolog): with no .command.begin Nextflow's exit-file timeout never
        # applies, and an ended job can no longer write one.
        return "ended_before_task_began"
    return "never_started"


def parse_squeue(out: str) -> dict[str, tuple[str, str]]:
    rows = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[0].strip().isdigit():
            rows[parts[0].strip()] = (parts[1].strip(), parts[2].strip())
    return rows


def parse_sacct(out: str) -> list[dict[str, str]]:
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and parts[0].isdigit():
            rows.append({
                "job_id": parts[0], "name": parts[1], "user": parts[2],
                "state": parts[3].split()[0].rstrip("+") if parts[3] else "", "start": parts[4],
            })
    return rows


class LostJobWatcher:
    """Find and fail Nextflow tasks whose Slurm job is gone without having run."""

    def __init__(
        self,
        run_dir: Path,
        nf_log: Path,
        *,
        grace_seconds: float = 1800,
        exit_code: int = LOST_JOB_EXIT_CODE,
        user: str | None = None,
        runner: Runner = _run,
        clock: Callable[[], datetime] = utc_now,
        log: Callable[[str], None] = lambda msg: print(msg, flush=True),
    ):
        self.run_dir = Path(run_dir)
        self.nf_log = Path(nf_log)
        self.grace = float(grace_seconds)
        self.exit_code = int(exit_code)
        self.user = user or getpass.getuser()
        self.runner = runner
        self.clock = clock
        self.log = log
        self._offset = 0
        self._partial = b""
        # work dir -> task; tasks leave once they have an exit code.
        self.submitted: dict[str, dict[str, Any]] = {}
        self.missing_since: dict[str, datetime] = {}
        # work dir -> consecutive polls that classified its job lost.
        self.lost_polls: dict[str, int] = {}
        self.lost: list[dict[str, Any]] = []
        self.last_error: str | None = None

    def _read_log(self) -> None:
        try:
            size = self.nf_log.stat().st_size
        except OSError:
            return
        if size < self._offset:   # replaced (a new run with the same log path)
            self._offset, self._partial = 0, b""
        with open(self.nf_log, "rb") as fh:
            fh.seek(self._offset)
            data = self._partial + fh.read()
            self._offset = fh.tell()
        lines = data.split(b"\n")
        self._partial = lines.pop()
        now = self.clock()
        for raw in lines:
            line = raw.decode("utf-8", errors="replace")
            m = _SUBMITTED.search(line)
            if not m:
                continue
            ts = _TS.match(line)
            self.submitted[m.group("wd")] = {
                "task": m.group("name"), "job_id": m.group("job"), "work_dir": m.group("wd"),
                "submitted_at": (parse_log_time(ts.group(1), now) if ts else None) or now,
            }

    def poll(self) -> list[dict[str, Any]]:
        """One pass; returns the tasks marked lost in it."""
        self._read_log()
        now = self.clock()
        waiting = []
        for wd, task in list(self.submitted.items()):
            path = Path(wd)
            if (path / ".exitcode").exists():
                self.submitted.pop(wd, None)
                self.missing_since.pop(wd, None)
                self.lost_polls.pop(wd, None)
            elif not (path / ".command.begin").exists():
                waiting.append(task)
        if not waiting:
            return []
        try:
            queued = parse_squeue(self.runner(["squeue", "--me", "-h", "-o", "%i|%j|%u"]))
        except Exception as exc:   # controller down: nothing is known, nothing ages
            self.last_error = f"squeue: {exc}"
            return []
        self.last_error = None
        marked = []
        for task in waiting:
            wd = task["work_dir"]
            name = slurm_job_name(Path(wd))
            if name is None:
                continue
            row = queued.get(task["job_id"])
            if row is not None and row[0] == name:
                self.missing_since.pop(wd, None)
                self.lost_polls.pop(wd, None)
                continue
            first = self.missing_since.setdefault(wd, now)
            if (now - first).total_seconds() < self.grace or (now - task["submitted_at"]).total_seconds() < self.grace:
                continue
            try:
                rows = parse_sacct(self.runner([
                    "sacct", "-a", "-D", "-X", "-n", "-P", "-j", task["job_id"],
                    "--format=JobID,JobName,User,State,Start",
                ]))
            except Exception as exc:
                self.last_error = f"sacct: {exc}"
                continue
            reason = classify(task["job_id"], name, self.user, queued, rows)
            if reason is None:
                self.lost_polls.pop(wd, None)
                continue
            # squeue and slurmdbd can both lag a fresh job: act on a second look.
            self.lost_polls[wd] = self.lost_polls.get(wd, 0) + 1
            if self.lost_polls[wd] < 2:
                continue
            record = self._mark(task, name, reason, rows, now)
            if record:
                marked.append(record)
        return marked

    def _mark(self, task, name, reason, rows, now) -> dict[str, Any] | None:
        wd = Path(task["work_dir"])
        # Re-check under the final state: never overwrite a real start or exit.
        if (wd / ".exitcode").exists() or (wd / ".command.begin").exists():
            return None
        if reason != "id_reused_by_other_job":
            # If the judgement was wrong (a job Slurm had yet to report), it must not
            # run beside the retry. Only our job with the task's name is cancelled.
            try:
                self.runner(["scancel", "--user", self.user, "--name", name, task["job_id"]])
            except Exception as exc:
                self.log(f"[driver-watch] scancel {task['job_id']} failed (marking anyway): {exc}")
        record = {
            "task": task["task"], "job_id": task["job_id"], "slurm_job_name": name,
            "work_dir": str(wd), "reason": reason, "exit_code": self.exit_code,
            "submitted_at": _iso(task["submitted_at"]),
            "missing_since": _iso(self.missing_since.get(str(wd))), "marked_at": _iso(now),
            "sacct": rows[:10],
        }
        begin = wd / ".command.begin"
        try:
            begin.touch(exist_ok=False)
        except FileExistsError:
            return None   # the task began after all
        except OSError as exc:
            self.last_error = f"marking {wd}: {exc}"
            return None
        try:
            (wd / MARKER_FILE).write_text(json.dumps(record, indent=1) + "\n")
            msg = (f"mussel driver watch: Slurm job {task['job_id']} ({name}) is gone without "
                   f"having run ({reason}); failing the task with exit {self.exit_code}\n")
            with open(wd / ".command.err", "a") as fh:
                fh.write(msg)
            if not _create_exclusive(wd / ".exitcode", str(self.exit_code)):
                return None   # it has an exit code after all
        except OSError as exc:
            self.last_error = f"marking {wd}: {exc}"
            # Not marked: take back the begin file, or the task would look started and
            # never be looked at again.
            try:
                begin.unlink()
            except OSError:
                pass
            return None
        self.lost.append(record)
        self.log(f"[driver-watch] {_iso(now)} LOST SLURM JOB {task['job_id']} task={task['task']!r} "
                 f"reason={reason} work_dir={wd}: marked failed (exit {self.exit_code})")
        return record

    def status(self) -> dict[str, Any]:
        return {
            "version": 1,
            "pid": os.getpid(),
            "heartbeat_at": _iso(self.clock()),
            "grace_seconds": self.grace,
            "waiting_tasks": sum(
                1 for wd in self.submitted if not (Path(wd) / ".command.begin").exists()
            ),
            "missing_from_queue": len(self.missing_since),
            "lost_jobs": self.lost[-50:],
            "last_error": self.last_error,
        }


def _create_exclusive(path: Path, text: str) -> bool:
    """Create ``path`` holding ``text``, all at once; False if it already exists.

    Nextflow reads .exitcode as soon as it exists: an O_EXCL create followed by
    a write shows it an empty file in between. Written aside, then hard-linked
    into place (no overwrite, unlike a rename).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)
    except FileExistsError:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return True


def write_status(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, path)


def read_status(run_dir: Path, max_bytes: int = 1 << 20) -> dict[str, Any] | None:
    path = Path(run_dir) / STATUS_FILE
    try:
        if path.stat().st_size > max_bytes:
            return None
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--nf-log", required=True)
    ap.add_argument("--grace-seconds", type=float, default=1800)
    ap.add_argument("--interval-seconds", type=float, default=120)
    ap.add_argument("--exit-code", type=int, default=LOST_JOB_EXIT_CODE)
    ap.add_argument("--parent-pid", type=int, default=os.getppid())
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    watcher = LostJobWatcher(run_dir, Path(args.nf_log), grace_seconds=args.grace_seconds,
                             exit_code=args.exit_code)
    status_path = run_dir / STATUS_FILE
    started_ppid = os.getppid()
    print(f"[driver-watch] started pid={os.getpid()} grace={args.grace_seconds}s", flush=True)
    while True:
        try:
            watcher.poll()
        except Exception as exc:   # keep watching; the status shows the error
            watcher.last_error = f"poll: {exc}"
        try:
            write_status(status_path, watcher.status())
        except OSError as exc:
            print(f"[driver-watch] status write failed: {exc}", flush=True)
        try:
            os.kill(args.parent_pid, 0)
        except OSError:
            return 0   # the driver is gone
        if started_ppid == args.parent_pid and os.getppid() != started_ppid:
            return 0   # reparented: the driver shell exited
        time.sleep(max(1.0, args.interval_seconds))


if __name__ == "__main__":
    sys.exit(main())
