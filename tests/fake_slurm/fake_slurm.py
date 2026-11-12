#!/usr/bin/env python3
"""Fake Slurm (sbatch/squeue/sacct/scancel) that runs jobs as local processes and can
misbehave the way a failing Slurm controller can.

Invoked through the sibling symlinks; dispatches on argv[0]. Each run has a root:
the directory of $FAKE_SLURM_CONF, else the nearest directory at or above the working
directory that holds a fake_slurm.conf (Nextflow runs squeue from its launch dir and
sbatch from the task's work dir, both inside the run root an nf-test or pytest sets
up). State (one JSON per job, submit-failure counters) lives in $FAKE_SLURM_DIR,
else <root>/.fake_slurm, else <tmp>/fake_slurm-<uid>, so runs never share counters.
Misbehaviour is read at submit time from the conf, as KEY=VALUE lines:

  FAIL_SUBMIT=<regex>     job names whose sbatch fails ...
  FAIL_TIMES=<n>          ... the first n times per name (default: always)
  LOSE=<regex>            job names that get an id but never run ...
  LOSE_MODE=never_started|reused|no_record|boot_fail
                          ... sacct shows ours never started / only another user's
                          job under that id / nothing / ours got a node (Start set)
                          and ended BOOT_FAIL before the task began
  LOSE_TIMES=<n>          ... the first n times per name (default: always)
  EXEC_FAIL=<regex>       job names that start and exit EXEC_EXIT (default 1) without
                          running the script ...
  EXEC_FAIL_TIMES=<n>     ... the first n times per name (default: always)
  WATCHDOG=<seconds>      SIGTERM the submitter (Nextflow) after this long: a hung run
                          fails instead of blocking CI

$FAKE_SLURM_DOWN=1 makes every command fail as if slurmctld were unreachable.
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

USER = os.environ.get("USER") or "fakeuser"
OTHER_USER = "otheruser"
CONF_NAME = "fake_slurm.conf"


def _find_conf():
    path = os.environ.get("FAKE_SLURM_CONF")
    if path:
        return os.path.abspath(path)
    here = os.getcwd()
    while True:
        candidate = os.path.join(here, CONF_NAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


CONF = _find_conf()
ROOT = os.path.dirname(CONF) if CONF else None
STATE = Path(
    os.environ.get("FAKE_SLURM_DIR")
    or (os.path.join(ROOT, ".fake_slurm") if ROOT else None)
    or Path(tempfile.gettempdir()) / f"fake_slurm-{os.getuid()}"
)


def _lock():
    import fcntl
    STATE.mkdir(parents=True, exist_ok=True)
    fh = open(STATE / ".lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _jobs():
    out = []
    for path in STATE.glob("job_*.json"):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda j: (j["id"], j.get("user") != USER))


def _save(job):
    suffix = "" if job.get("user") == USER else "_" + job["user"]
    tmp = STATE / f".job_{job['id']}{suffix}.tmp"
    tmp.write_text(json.dumps(job))
    tmp.replace(STATE / f"job_{job['id']}{suffix}.json")


def _conf():
    conf = {}
    if CONF and os.path.isfile(CONF):
        for line in Path(CONF).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                conf[key.strip()] = value.strip()
    return conf


def _alive(job):
    pid = job.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _state(job):
    if job.get("state") in ("PENDING", "RUNNING") and job.get("pid") and not _alive(job):
        return "COMPLETED"
    return job.get("state")


def _watchdog(conf):
    seconds = conf.get("WATCHDOG")
    marker = STATE / f"watchdog_{os.getppid()}"
    if not seconds or marker.exists():
        return
    marker.write_text(str(time.time()))
    root = conf.get("_root")
    subprocess.Popen(
        ["bash", "-c", f"sleep {int(float(seconds))}; kill -0 {os.getppid()} 2>/dev/null || exit 0; "
                       f"{'echo fired > ' + repr(root + '/fake_slurm_watchdog_fired') + '; ' if root else ''}"
                       f"kill -TERM {os.getppid()}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _count(key, limit):
    """True (and counted) while ``key`` has happened fewer than ``limit`` times."""
    path = STATE / re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    done = int(path.read_text()) if path.exists() else 0
    if done >= int(limit or 1 << 30):
        return False
    path.write_text(str(done + 1))
    return True


def sbatch(argv):
    script = os.path.abspath(argv[-1])
    text = Path(script).read_text()
    m = re.search(r"^#SBATCH\s+(?:-J|--job-name[= ])\s*(\S+)", text, re.M)
    name = m.group(1) if m else os.path.basename(script)
    m = re.search(r"^#SBATCH\s+(?:-o|--output[= ])\s*(\S+)", text, re.M)
    out_path = m.group(1) if m else os.devnull
    conf = _conf()
    if ROOT:
        conf["_root"] = ROOT
    with _lock():
        _watchdog(conf)
        counter = STATE / "next_id"
        job_id = int(counter.read_text()) + 1 if counter.exists() else 8_560_000
        counter.write_text(str(job_id))
        if conf.get("_root"):
            with open(os.path.join(conf["_root"], "fake_slurm.log"), "a") as fh:
                fh.write(f"{job_id} {name}\n")
        if conf.get("FAIL_SUBMIT") and re.search(conf["FAIL_SUBMIT"], name) and _count(
                f"submit_failures_{name}", conf.get("FAIL_TIMES")):
            print("sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)", file=sys.stderr)
            return 1
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        job = {"id": job_id, "name": name, "user": USER, "state": "PENDING", "submit": now,
               "start": None, "end": None, "exit_code": None, "script": script, "pid": None}
        if conf.get("LOSE") and re.search(conf["LOSE"], name) and _count(
                f"lost_{name}", conf.get("LOSE_TIMES")):
            mode = conf.get("LOSE_MODE") or "never_started"
            if mode == "never_started":
                job.update(state="CANCELLED", end=now, exit_code=0)
                _save(job)
            elif mode == "boot_fail":
                job.update(state="BOOT_FAIL", start=now, end=now, exit_code=0)
                _save(job)
            elif mode == "reused":
                _save({"id": job_id, "name": "other-array_7", "user": OTHER_USER, "state": "COMPLETED",
                       "submit": now, "start": now, "end": now, "exit_code": 0, "script": None, "pid": None})
            print(f"Submitted batch job {job_id}")
            return 0
        workdir = os.path.dirname(script)
        if conf.get("EXEC_FAIL") and re.search(conf["EXEC_FAIL"], name) and _count(
                f"exec_failures_{name}", conf.get("EXEC_FAIL_TIMES")):
            cmd = f"touch .command.begin; echo {int(conf.get('EXEC_EXIT') or 1)} > .exitcode"
        else:
            cmd = f"exec bash {script!r}"
        proc = subprocess.Popen(
            ["bash", "-c", cmd], cwd=workdir, stdin=subprocess.DEVNULL,
            stdout=open(out_path, "ab"), stderr=subprocess.STDOUT, start_new_session=True,
        )
        job.update(state="RUNNING", start=now, pid=proc.pid)
        _save(job)
    print(f"Submitted batch job {job_id}")
    return 0


def _fmt(job, spec):
    fields = {"i": str(job["id"]), "j": job["name"], "u": job["user"],
              "t": {"RUNNING": "R", "PENDING": "PD"}.get(_state(job), "CD"), "T": _state(job)}
    return re.sub(r"%(\w)", lambda m: fields.get(m.group(1), ""), spec)


def squeue(argv):
    spec, ids, user, header = "%i %t", None, None, True
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--format"):
            spec = argv[i + 1]; i += 1
        elif a.startswith("--format="):
            spec = a.split("=", 1)[1]
        elif a in ("-j", "--jobs"):
            ids = {int(x) for x in argv[i + 1].split(",") if x}; i += 1
        elif a in ("-u", "--user"):
            user = argv[i + 1]; i += 1
        elif a == "--me":
            user = USER
        elif a in ("-h", "--noheader"):
            header = False
        elif a in ("-t", "--states", "-p", "--partition", "-n", "--name"):
            i += 1
        i += 1
    if not header:
        pass
    for job in _jobs():
        if _state(job) not in ("PENDING", "RUNNING") or not job.get("pid"):
            continue
        if (ids and job["id"] not in ids) or (user and job["user"] != user):
            continue
        print(_fmt(job, spec))
    return 0


_SACCT = {"jobid": "id", "jobname": "name", "user": "user", "state": "state", "start": "start",
          "end": "end", "submit": "submit", "exitcode": "exit_code"}


def sacct(argv):
    fields, ids, name, all_users, dups = ["JobID", "State", "ExitCode"], None, None, False, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--format="):
            fields = a.split("=", 1)[1].split(",")
        elif a in ("-o", "--format"):
            fields = argv[i + 1].split(","); i += 1
        elif a in ("-j", "--jobs"):
            ids = {int(x.split(".")[0]) for x in argv[i + 1].split(",") if x}; i += 1
        elif a.startswith("--name="):
            name = a.split("=", 1)[1]
        elif a == "--name":
            name = argv[i + 1]; i += 1
        elif a in ("-a", "--allusers"):
            all_users = True
        elif a in ("-D", "--duplicates"):
            dups = True
        elif a in ("-S", "--starttime", "-E", "--endtime", "-u", "--user"):
            i += 1
        i += 1
    rows = []
    for job in _jobs():
        if ids and job["id"] not in ids:
            continue
        if name and job["name"] != name:
            continue
        if not all_users and job["user"] != USER:
            continue
        rows.append(job)
    if not dups:
        latest = {}
        for job in rows:
            if job["id"] not in latest or job["user"] != USER:
                latest[job["id"]] = job
        rows = list(latest.values())
    for job in rows:
        out = []
        for f in fields:
            key = _SACCT.get(f.lower().split("%")[0], "")
            value = _state(job) if key == "state" else job.get(key)
            if key == "exit_code":
                value = f"{value or 0}:0"
            if key in ("start", "end") and value is None:
                value = "None" if key == "start" else "Unknown"
            out.append(str(value))
        print("|".join(out))
    return 0


def scancel(argv):
    name, user = None, USER
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-n", "--name", "--jobname", "-u", "--user"):
            value = argv[i + 1]; i += 1
            if a in ("-u", "--user"):
                user = value
            else:
                name = value
        elif a.startswith("--name=") or a.startswith("--jobname="):
            name = a.split("=", 1)[1]
        elif a.startswith("--user="):
            user = a.split("=", 1)[1]
        i += 1
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "scancel.log", "a") as fh:
        fh.write(" ".join(argv) + "\n")
    for a in argv:
        if a.isdigit():
            for job in _jobs():
                # Filters restrict, as in Slurm: only a job matching every one is signalled.
                if (job["id"] == int(a) and job["user"] == USER == user
                        and (name is None or job["name"] == name)
                        and _state(job) in ("PENDING", "RUNNING")):
                    if _alive(job):
                        try:
                            os.killpg(job["pid"], signal.SIGTERM)
                        except OSError:
                            pass
                    job["state"] = "CANCELLED"
                    _save(job)
    return 0


def main():
    cmd = os.path.basename(sys.argv[0])
    handler = {"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "scancel": scancel}.get(cmd)
    if handler is None:
        print(f"fake_slurm: unknown command {cmd}", file=sys.stderr)
        return 2
    if os.environ.get("FAKE_SLURM_DOWN") == "1":
        print(f"{cmd}: error: Unable to contact slurm controller (connect failure)", file=sys.stderr)
        return 1
    return handler(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
