"""PostgreSQL-coordinated Slurm dispatch for non-shared filesystem clusters."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .batch_lifecycle import (
    BatchFinalizer,
    FinalizationContext,
    FinalizationWorkerPool,
    NextflowRunSpec,
    build_nextflow_command,
    write_batch_csv,
)
from .config import Config, RemoteTargetConfig
from .contract import (
    build_contract,
    execution_profile_hash,
    git_revision,
    release_manifest_hash,
    tracked_source_manifest,
    validate_semantic_params,
)
from .hook_launcher import is_launched
from .state import REMOTE_FINALIZATION_LEASE_NAME, StateStore
from .storage import (
    CapacityGuard,
    ExecutionArtifactCleaner,
    ExecutionCleanupSpec,
    reconcile_execution_cleanup,
)

TERMINAL_REMOTE_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
FINALIZATION_QUEUED_STATUS = "FINALIZATION_QUEUED"
HOOKS_BLOCKED_STATUS = "HOOKS_BLOCKED"
# Per-batch WDS staging, mapped to ``{remote_run_dir}/wds-staging`` by
# remote_hook_path_replacements.
WDS_STAGING_SUBDIR = "wds-staging"
REMOTE_BLOCKED_STATUS = "BLOCKED"
_CLAIM_REPLAY_DELAY_SECONDS = 90
_BLOCKED_RETRY_SECONDS = 300
_RUNTIME_ROOT_FILES = ("main.nf", "nextflow.config")
_RUNTIME_SOURCE_DIRS = ("modules", "lib", "bin")
_RUNTIME_OPTIONAL_FILES = (
    # Optional site config for the remote target (-c), staged with the release.
    "dispatcher/remote_nextflow.config",
    "mussel_env_fastattn.yaml",
)
_REMOTE_LOG_TAIL_MAX_BYTES = 64 * 1024
_REMOTE_CORE_LOG_TAIL_BYTES = 16 * 1024
_REMOTE_HOOK_LOG_TAIL_BYTES = 4 * 1024
_REMOTE_HOOK_LOG_FILE_LIMIT = 4
_DRIVER_JOB_PREFIX = "mussel-driver-"
# A stalled main loop exits non-zero (EX_SOFTWARE). Death by SIGTERM counts as
# a clean stop for systemd's Restart=on-failure, so the agent would stay down.
_WATCHDOG_EXIT_CODE = 70
_SLURM_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OOM", "REVOKED",
})
# Terminal states that say nothing against the slides themselves: their slides are
# released without being charged a retry. One set for every place that classifies.
_SLURM_INFRASTRUCTURE_FAILURES = frozenset({
    "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
    "DEADLINE", "OOM", "REVOKED",
})
log = logging.getLogger("mussel-dispatcher")


_ORDER_FAILURE_BACKOFF_BASE_SECONDS = 30
_ORDER_FAILURE_BACKOFF_MAX_SECONDS = 900
# A staged release that is genuinely missing (not a transient git timeout) is
# retried this many times before the order is retired (driver) or the batch is
# quarantined as HOOKS_BLOCKED (hooks).
_STAGE_MISSING_MAX_ATTEMPTS = 3
_STAGE_RETRY_BASE_SECONDS = 60
_STAGE_RETRY_MAX_SECONDS = 900


class StagedReleaseError(RuntimeError):
    """The order's staged release is unusable; ``transient`` if worth retrying."""

    def __init__(self, message: str, *, transient: bool):
        super().__init__(message)
        self.transient = transient


class _RefreshCycle:
    """Lazily computed values shared by all orders in one refresh pass."""

    def __init__(self, agent: "RemoteAgent"):
        self._owner = agent
        self._capacity_ok: bool | None = None
        self.agent: dict[str, Any] | None = None

    def capacity_ok(self) -> bool:
        if self._capacity_ok is None:
            self._capacity_ok = self._owner._capacity_available()
        return self._capacity_ok


def _hard_exit(code: int) -> None:
    """Flush logs and terminate immediately, from any thread."""
    for handler in logging.getLogger().handlers + log.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bounded_log_tail(path: Path, max_bytes: int) -> tuple[str, bool, int]:
    """Read at most ``max_bytes`` from the end of a log, tolerating bad UTF-8."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        read_size = min(size, max(0, int(max_bytes)))
        handle.seek(max(0, size - read_size))
        return (
            handle.read(read_size).decode("utf-8", errors="replace"),
            size > read_size,
            read_size,
        )


def _canonical_params_snapshot(cfg: Config) -> tuple[str | None, str]:
    """Return the exact scientific params sent to every remote backend."""
    if not cfg.nextflow_params_file:
        return None, ""
    path = Path(cfg.nextflow_params_file)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Could not read canonical params file {path}: {exc}") from exc
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Canonical params file is not UTF-8: {path}") from exc
    return content, _sha256_bytes(data)


def _tracked_runtime_source_files(root: Path) -> list[Path] | None:
    """Return git-tracked files under the runtime source dirs, or None outside git.

    Only tracked files are part of a release: an untracked or ignored file in
    the dispatching checkout must not enter the fingerprint, because the remote
    release (a clean worktree of the same revision) can never contain it.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *_RUNTIME_SOURCE_DIRS],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return sorted(
        root / item
        for item in result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        if item
    )


def _runtime_sha256_manifest(repo_dir: str) -> dict[str, str]:
    """Fingerprint source files that can change scientific pipeline behavior."""
    root = Path(repo_dir)
    candidates = [root / relative for relative in _RUNTIME_ROOT_FILES]
    tracked = _tracked_runtime_source_files(root)
    if tracked is not None:
        candidates.extend(
            path for path in tracked
            if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        )
    else:
        # Not a git checkout (unit fixtures, ad-hoc runs): fingerprint the tree.
        for directory in _RUNTIME_SOURCE_DIRS:
            source_root = root / directory
            if source_root.is_dir():
                candidates.extend(
                    sorted(
                        path
                        for path in source_root.rglob("*")
                        if path.is_file()
                        and "__pycache__" not in path.parts
                        and path.suffix not in {".pyc", ".pyo"}
                    )
                )
    candidates.extend(
        path for relative in _RUNTIME_OPTIONAL_FILES if (path := root / relative).is_file()
    )

    manifest: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            raise RuntimeError(f"Required pipeline runtime file is missing: {path}")
        manifest[path.relative_to(root).as_posix()] = _sha256_bytes(path.read_bytes())
    return manifest


def _verify_order_runtime(order: dict[str, Any]) -> None:
    """Fail closed when the remote checkout differs from the dispatching runtime."""
    manifest = order.get("runtime_sha256") or {}
    if int(order.get("schema_version") or 1) >= 2 and not manifest:
        raise RuntimeError("Schema-v2 remote order is missing its runtime fingerprint")
    repo_dir = Path(str(order["remote_repo_dir"]))
    for relative, expected in sorted(manifest.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Invalid remote runtime path in fingerprint: {relative}")
        path = repo_dir / relative
        try:
            actual = _sha256_bytes(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"Remote runtime file is unavailable: {path}: {exc}") from exc
        if actual != expected:
            raise RuntimeError(
                f"Remote runtime drift for {relative}: expected {expected}, got {actual}"
            )


def _verify_order_release(order: dict[str, Any]) -> None:
    """Fail closed when a schema-v3 order targets a different release."""
    if int(order.get("schema_version") or 1) < 3:
        return
    expected_revision = str(order.get("source_revision") or "").strip()
    expected_manifest = str(order.get("release_manifest_sha256") or "").strip()
    repo_dir = str(order.get("remote_repo_dir") or "").strip()
    if not expected_revision or not expected_manifest or not repo_dir:
        raise RuntimeError("Schema-v3 remote order is missing its release identity")
    actual_revision = git_revision(repo_dir)
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Remote source release drift: expected {expected_revision}, got {actual_revision}"
        )
    dirty = subprocess.run(
        ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Remote source checkout has tracked changes: {repo_dir}")
    actual_manifest = release_manifest_hash(tracked_source_manifest(repo_dir))
    if actual_manifest != expected_manifest:
        raise RuntimeError(
            f"Remote release manifest drift: expected {expected_manifest}, got {actual_manifest}"
        )


def _verify_order_profile(
    order: dict[str, Any], cfg: Config, target: RemoteTargetConfig
) -> None:
    """Ensure the agent is running the target profile used to create the order."""
    if int(order.get("schema_version") or 1) < 3:
        return
    expected = str(order.get("execution_profile_sha256") or "").strip()
    if not expected:
        raise RuntimeError("Schema-v3 remote order is missing its execution profile identity")
    actual = execution_profile_hash(cfg, target=target)
    if actual != expected:
        raise RuntimeError(
            f"Remote execution profile drift: expected {expected}, got {actual}"
        )


def _materialize_order_params(order: dict[str, Any]) -> None:
    """Write and verify the dispatcher-owned params snapshot for one remote run."""
    content = order.get("nextflow_params_content")
    expected = str(order.get("nextflow_params_sha256") or "")
    if content is None:
        if int(order.get("schema_version") or 1) >= 2 and expected:
            raise RuntimeError("Schema-v2 remote order is missing its params snapshot")
        return
    data = str(content).encode("utf-8")
    actual = _sha256_bytes(data)
    if not expected or actual != expected:
        raise RuntimeError(
            f"Remote params payload checksum mismatch: expected {expected or '<missing>'}, got {actual}"
        )
    if int(order.get("schema_version") or 1) >= 3:
        try:
            params = yaml.safe_load(content) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Remote params payload is invalid YAML: {exc}") from exc
        if not isinstance(params, dict):
            raise RuntimeError("Remote params payload must be a YAML mapping")
        validate_semantic_params(params)
        featurize = params.get("featurize") or {}
        if featurize.get("slide_batch_size") is None:
            raise RuntimeError("Remote params payload is missing featurize.slide_batch_size")
        effective_expected = str(order.get("effective_params_sha256") or "").strip()
        if effective_expected and effective_expected != actual:
            raise RuntimeError(
                "Remote effective params checksum does not match the materialized payload"
            )
    path = Path(str(order["nextflow_params_file"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _render_runtime_checks(order: dict[str, Any]) -> str:
    checks: list[tuple[str, str]] = []
    repo_dir = str(order["remote_repo_dir"])
    for relative, digest in sorted((order.get("runtime_sha256") or {}).items()):
        checks.append((str(digest), os.path.join(repo_dir, relative)))
    params_digest = str(order.get("nextflow_params_sha256") or "")
    params_path = str(order.get("nextflow_params_file") or "")
    if params_digest and params_path:
        checks.append((params_digest, params_path))
    if not checks:
        return ""
    commands = [
        f"printf '%s  %s\\n' {shlex.quote(digest)} {shlex.quote(path)} | sha256sum --check --strict -"
        for digest, path in checks
    ]
    return "echo '[mussel-driver] verifying runtime and params fingerprints'\n" + "\n".join(commands)


def _render_release_check(order: dict[str, Any]) -> str:
    """Ensure a remote driver uses the exact immutable source revision."""
    revision = str(order.get("source_revision") or "").strip()
    repo_dir = str(order.get("remote_repo_dir") or "").strip()
    if not revision or not repo_dir:
        return ""
    quoted_repo = shlex.quote(repo_dir)
    quoted_revision = shlex.quote(revision)
    return (
        "echo '[mussel-driver] verifying source release'\n"
        f"test \"$(git -C {quoted_repo} rev-parse HEAD)\" = {quoted_revision}\n"
        f"test -z \"$(git -C {quoted_repo} status --porcelain --untracked-files=no)\"\n"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dispatcher_host(target: RemoteTargetConfig) -> str:
    return (target.dispatcher_host or "").strip()


def _remote_tower_endpoint(cfg: Config, target: RemoteTargetConfig) -> str:
    explicit = (target.tower_endpoint or "").strip()
    if explicit:
        return explicit
    host = _dispatcher_host(target)
    if host and cfg.tower_proxy_port:
        return f"http://{host}:{cfg.tower_proxy_port}"
    if host and cfg.dashboard_port:
        return f"http://{host}:{cfg.dashboard_port}"
    return ""


def render_driver_script(order: dict[str, Any], target: RemoteTargetConfig) -> str:
    """Render the Slurm driver script that runs Nextflow inside an allocation."""
    batch_id = order["batch_id"]
    remote_run_dir = order["remote_run_dir"]
    remote_work_dir = order.get("remote_work_dir") or os.path.join(remote_run_dir, "work")
    remote_outdir = order.get("remote_outdir") or target.remote_outdir
    remote_csv = order.get("remote_batch_csv") or os.path.join(remote_run_dir, "batch.csv")
    nf_log = order.get("remote_nf_log") or os.path.join(remote_run_dir, "nextflow.log")
    trace = order.get("remote_trace") or os.path.join(remote_run_dir, "trace.tsv")
    repo_dir = order["remote_repo_dir"]
    profiles = order["nextflow_profiles"]
    params_file = order.get("nextflow_params_file") or ""
    config_file = order.get("nextflow_config") or ""
    nf_version = order.get("nextflow_version") or target.nextflow_version
    tower_endpoint = order.get("tower_endpoint") or ""
    run_name = order["nextflow_run_name"]
    nextflow_bin = (
        os.environ.get("MUSSEL_REMOTE_NEXTFLOW_BIN", "").strip()
        or shutil.which("nextflow")
        or "nextflow"
    )
    java_home = (
        os.environ.get("MUSSEL_REMOTE_JAVA_HOME", "").strip()
        or os.environ.get("JAVA_HOME", "").strip()
    )
    conda_exe = (
        os.environ.get("MUSSEL_REMOTE_CONDA_EXE", "").strip()
        or os.environ.get("CONDA_EXE", "").strip()
        or shutil.which("conda")
        or ""
    )
    conda_bin_dir = os.path.dirname(conda_exe) if conda_exe else ""
    conda_root = os.path.dirname(conda_bin_dir) if conda_bin_dir else ""
    env_lines = []
    for key, value in (target.env or {}).items():
        env_lines.append(f"export {key}={shlex.quote(str(value))}")
    if target.secrets_env_file:
        env_lines.append(f"set -a; source {shlex.quote(target.secrets_env_file)}; set +a")
    if java_home:
        env_lines.append(f"export JAVA_HOME={shlex.quote(java_home)}")
        env_lines.append(f"export NXF_JAVA_HOME={shlex.quote(java_home)}")
        env_lines.append(f"export JAVA_CMD={shlex.quote(os.path.join(java_home, 'bin', 'java'))}")
        env_lines.append(f"export PATH={shlex.quote(os.path.join(java_home, 'bin'))}:$PATH")
    if target.conda_env_path:
        env_lines.append(f"export MUSSEL_VENV={shlex.quote(target.conda_env_path)}")
        env_lines.append(f"export PATH={shlex.quote(os.path.join(target.conda_env_path, 'bin'))}:$PATH")
    if conda_exe:
        env_lines.append(f"export CONDA_EXE={shlex.quote(conda_exe)}")
    if conda_root:
        env_lines.append(f"export CONDA_PYTHON_EXE={shlex.quote(os.path.join(conda_root, 'bin', 'python'))}")
        env_lines.append(f"export PATH={shlex.quote(os.path.join(conda_root, 'condabin'))}:{shlex.quote(conda_bin_dir)}:$PATH")
    if nextflow_bin:
        env_lines.append(f"export MUSSEL_REMOTE_NEXTFLOW_BIN={shlex.quote(nextflow_bin)}")
        env_lines.append(f"export PATH={shlex.quote(os.path.dirname(nextflow_bin))}:$PATH")
    if nf_version:
        env_lines.append(f"export NXF_VER={shlex.quote(nf_version)}")
    if tower_endpoint:
        env_lines.append(f"export TOWER_API_ENDPOINT={shlex.quote(tower_endpoint)}")
        env_lines.append('export TOWER_ACCESS_TOKEN="${TOWER_ACCESS_TOKEN:-local}"')
    cmd = build_nextflow_command(NextflowRunSpec(
        executable=[nextflow_bin],
        log_path=nf_log,
        repo_dir=repo_dir,
        profiles=profiles,
        work_dir=remote_work_dir,
        batch_csv=remote_csv,
        outdir=remote_outdir,
        run_name=run_name,
        trace_path=trace,
        config_path=config_file,
        params_path=params_file,
        tower_enabled=bool(tower_endpoint),
        scratch_dir=os.path.join(remote_run_dir, target.task_scratch_subdir),
        extra_args=[
            str(part)
            for part in (order.get("nextflow_cli_args") or target.nextflow_cli_args)
        ],
    ))
    quoted_cmd = " ".join(shlex.quote(str(part)) for part in cmd)
    runtime_checks = _render_runtime_checks(order)
    release_check = _render_release_check(order)
    # Computed here: a backslash inside an f-string expression is a SyntaxError before 3.12.
    nxf_dist_line = (
        f"export NXF_DIST={shlex.quote(target.nextflow_framework_dir)}"
        if target.nextflow_framework_dir
        else 'export NXF_DIST="${NXF_DIST:-$HOME/.nextflow/framework}"'
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=mussel-driver-{batch_id}
#SBATCH --partition={target.driver_partition}
#SBATCH --time={target.driver_time}
#SBATCH --cpus-per-task={target.driver_cpus}
#SBATCH --mem={target.driver_mem}
{f"#SBATCH --account={target.driver_account}" if target.driver_account else ""}
{f"#SBATCH --qos={target.driver_qos}" if target.driver_qos else ""}
#SBATCH --output={remote_run_dir}/driver.out
#SBATCH --error={remote_run_dir}/driver.err

set -euo pipefail
mkdir -p {shlex.quote(remote_work_dir)} {shlex.quote(remote_outdir)} {shlex.quote(os.path.join(remote_run_dir, target.task_scratch_subdir))}
cd {shlex.quote(remote_run_dir)}
export NXF_HOME={shlex.quote(os.path.join(remote_run_dir, ".nextflow"))}
{nxf_dist_line}
export NXF_WORK={shlex.quote(remote_work_dir)}
export XDG_CACHE_HOME={shlex.quote(os.path.join(remote_run_dir, ".cache"))}
{"".join(line + chr(10) for line in env_lines)}
{release_check}{runtime_checks}
if [[ -n "${{HF_TOKEN:-}}" ]]; then
  mkdir -p "${{NXF_HOME}}"
  {shlex.quote(nextflow_bin)} secrets set HF_TOKEN "${{HF_TOKEN}}" >/dev/null
fi
if [[ -n "${{ECS_ACCESS_KEY:-}}" ]]; then
  mkdir -p "${{NXF_HOME}}"
  {shlex.quote(nextflow_bin)} secrets set ECS_ACCESS_KEY "${{ECS_ACCESS_KEY}}" >/dev/null
fi
if [[ -n "${{ECS_SECRET_KEY:-}}" ]]; then
  mkdir -p "${{NXF_HOME}}"
  {shlex.quote(nextflow_bin)} secrets set ECS_SECRET_KEY "${{ECS_SECRET_KEY}}" >/dev/null
fi
echo "[mussel-driver] $(date -Is) starting batch {batch_id}"
echo "[mussel-driver] command: {quoted_cmd}"
{quoted_cmd}
echo "[mussel-driver] $(date -Is) completed batch {batch_id}"
"""


def _squeue_job_ids_by_name(name: str) -> list[str]:
    """Return our queued/running job ids with this exact name, oldest first.

    Raises RuntimeError when squeue cannot answer.
    """
    try:
        sq = subprocess.run(
            ["squeue", "--me", "-h", "-n", name, "-o", "%i"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"squeue failed for {name}: {exc}") from exc
    if sq.returncode != 0:
        raise RuntimeError(
            f"squeue failed for {name}: {(sq.stderr or sq.stdout).strip()[:300]}"
        )
    return sorted(
        {item.strip() for item in sq.stdout.split() if item.strip().isdigit()}, key=int
    )


def _sacct_since(timestamp: object) -> str:
    """sacct ``-S`` value for an ISO timestamp, with a margin for local time."""
    try:
        when = datetime.fromisoformat(str(timestamp or ""))
    except ValueError:
        return "now-30days"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    # sacct reads local time; a one-day margin covers any UTC offset.
    return (when - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")


def _sacct_jobs_by_name(name: str, since: str) -> list[dict[str, Any]]:
    """Return accounting rows for our jobs with this name, oldest first.

    Each row has ``job_id``, ``state`` and ``exit_code``. Raises RuntimeError
    when sacct cannot answer.
    """
    try:
        sa = subprocess.run(
            ["sacct", "-n", "-X", "--name", name, "-S", since,
             "--format=JobID,State,ExitCode", "--parsable2"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"sacct failed for {name}: {exc}") from exc
    if sa.returncode != 0:
        raise RuntimeError(
            f"sacct failed for {name}: {(sa.stderr or sa.stdout).strip()[:300]}"
        )
    jobs: dict[str, dict[str, Any]] = {}
    for line in sa.stdout.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if not fields or not fields[0].isdigit():
            continue
        state = fields[1].split()[0].rstrip("+") if len(fields) > 1 and fields[1] else ""
        exit_code = None
        if len(fields) > 2:
            try:
                exit_code = int(fields[2].split(":", 1)[0])
            except ValueError:
                pass
        jobs[fields[0]] = {"job_id": fields[0], "state": state, "exit_code": exit_code}
    return [jobs[key] for key in sorted(jobs, key=int)]


class RemoteDispatchClient:
    """Dispatcher-side helper for submitting and polling remote work orders."""

    def __init__(
        self, cfg: Config, target: RemoteTargetConfig, *, state: StateStore | None = None
    ):
        self.cfg = cfg
        self.target = target
        if (cfg.state_backend or "").strip().lower() != "postgres":
            raise RuntimeError("remote_agent requires state_backend=postgres")
        self.state = state or StateStore.from_config(cfg)

    def submit(
        self,
        *,
        batch_id: str,
        slides: list[dict[str, Any]],
        csv_path: str,
        log_path: str,
        nf_run_name: str,
        contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        remote_run_dir = os.path.join(self.target.remote_scratch_root, batch_id)
        contract = contract or build_contract(self.cfg, target=self.target)
        params_content = str(contract["effective_params_yaml"])
        params_sha256 = str(contract["effective_params_sha256"])
        params_path = (
            os.path.join(remote_run_dir, "params.yaml")
            if params_content is not None
            else ""
        )
        order = {
            "schema_version": 3,
            "batch_id": batch_id,
            "created_at": utc_now(),
            "target": self.target.name,
            "slides": slides,
            "local_batch_csv": csv_path,
            "local_log_path": log_path,
            "nextflow_run_name": nf_run_name,
            "remote_scratch_root": self.target.remote_scratch_root,
            "remote_run_dir": remote_run_dir,
            "remote_work_dir": os.path.join(remote_run_dir, "work"),
            "remote_batch_csv": os.path.join(remote_run_dir, "batch.csv"),
            "remote_driver_script": os.path.join(remote_run_dir, "driver.sbatch"),
            "remote_driver_log": os.path.join(remote_run_dir, "driver.out"),
            "remote_driver_err": os.path.join(remote_run_dir, "driver.err"),
            "remote_nf_log": os.path.join(remote_run_dir, "nextflow.log"),
            "remote_trace": os.path.join(remote_run_dir, "trace.tsv"),
            "remote_repo_dir": self.target.remote_repo_dir or self.cfg.repo_dir,
            "remote_outdir": self.target.remote_outdir or self.cfg.outdir,
            "nextflow_profiles": self.target.nextflow_profiles or self.cfg.nextflow_profiles,
            "nextflow_config": self.target.nextflow_config,
            "nextflow_params_file": params_path,
            "nextflow_params_content": params_content,
            "nextflow_params_sha256": params_sha256,
            "runtime_sha256": _runtime_sha256_manifest(self.cfg.repo_dir),
            "source_revision": contract["source_revision"],
            "release_manifest_sha256": contract["release_manifest_sha256"],
            "semantic_contract_sha256": contract["semantic_contract_sha256"],
            "execution_profile": contract["execution_profile"],
            "execution_profile_sha256": contract["execution_profile_sha256"],
            "effective_params_sha256": contract["effective_params_sha256"],
            "nextflow_version": self.target.nextflow_version or self.cfg.nextflow_version,
            "tower_endpoint": _remote_tower_endpoint(self.cfg, self.target),
            "nextflow_cli_args": list(self.target.nextflow_cli_args or []),
            "conda_env_path": self.target.conda_env_path,
            "driver_sbatch": {
                "partition": self.target.driver_partition,
                "cpus": self.target.driver_cpus,
                "mem": self.target.driver_mem,
                "time": self.target.driver_time,
                "account": self.target.driver_account,
                "qos": self.target.driver_qos,
            },
            "hook_path_replacements": dict(self.target.remote_hook_path_replacements or {}),
        }
        # The remote agent has no shared filesystem with the dispatcher. Keep
        # the complete work order in the PostgreSQL payload so the agent can
        # reconstruct the remote batch CSV after claiming it.
        queued_order = dict(order)
        queued_order.update({
            "status": "QUEUED",
            "updated_at": utc_now(),
            "heartbeat_at": utc_now(),
            "remote_run_id": order["nextflow_run_name"],
            "remote_log_path": order["remote_driver_log"],
        })
        self.write_status(batch_id, queued_order)
        return order

    def read_status(self, batch_id: str) -> dict[str, Any] | None:
        return self.state.read_remote_status(batch_id)

    def write_status(self, batch_id: str, payload: dict[str, Any]) -> bool:
        return self.state.write_remote_status(batch_id, payload)

    def request_cancel(self, batch_id: str) -> None:
        self.state.request_remote_cancel(batch_id)


class RemoteAgent:
    """Remote-side worker that submits Nextflow driver jobs to Slurm."""

    def __init__(self, cfg: Config, target: RemoteTargetConfig, *, once: bool = False, dry_run: bool = False):
        self.cfg = cfg
        self.target = target
        self.once = once
        self.dry_run = dry_run
        # Wall-clock start of this incarnation; claims older than this with no
        # driver job were orphaned by a previous agent process.
        self._started_at = time.time()
        self.state = self._build_state_store()
        self._finalization_threads: dict[str, threading.Thread] = {}
        self._finalization_lease_names: dict[str, str] = {}
        self._finalization_pool = FinalizationWorkerPool(
            self._max_concurrent_finalizations(cfg, target),
            thread_name_prefix="remote-post-batch-finalize",
        )
        self._finalization_lock = threading.RLock()
        self._finalization_owner = (
            f"remote-agent:{self.target.name}:{socket.gethostname()}:{os.getpid()}"
        )
        self._next_poll_delay = max(1, int(self.target.poll_interval_seconds or 60))
        self._last_agent_summary_at = 0.0
        self._last_hook_reconcile_at = 0.0
        self._last_finalization_reconcile_at = 0.0
        self._finalization_reconcile_failures = 0
        self._finalization_reconcile_health: dict[str, Any] = {
            "state": "STARTING",
            "updated_at": utc_now(),
            "durable_slots": 0,
            "process_active": 0,
            "repaired": [],
            "terminal_released": [],
            "expired": [],
            "foreign_live": [],
            "errors": [],
        }
        self._hook_executor_missing_polls: dict[tuple[str, int], int] = {}
        self._hook_executor_health: dict[str, Any] = {}
        self._finalization_heartbeat_health: dict[str, dict[str, Any]] = {}
        self._status_writes_suppressed = 0
        self._driver_slurm_queries = 0
        self._agent_snapshot_writes = 0
        self._execution_cleaner = ExecutionArtifactCleaner()
        self._capacity_guard = CapacityGuard(
            self.target.storage_policy, name=f"remote:{self.target.name}"
        )
        self._last_storage_cleanup_at = 0.0
        self._snapshot_lock = threading.Lock()
        self._main_loop_progress_lock = threading.Lock()
        self._main_loop_progress_monotonic = time.monotonic()
        self._main_loop_progress_at = utc_now()
        self._agent_heartbeat_stop = threading.Event()
        self._agent_heartbeat_thread: threading.Thread | None = None
        self._agent_watchdog_thread: threading.Thread | None = None
        try:
            self._agent_revision = git_revision(str(Path(__file__).resolve().parents[2]))
        except Exception:
            self._agent_revision = ""

    @staticmethod
    def _max_concurrent_finalizations(cfg: Config, target: RemoteTargetConfig) -> int:
        target_limit = int(getattr(target, "max_concurrent_finalizations", 0) or 0)
        if target_limit > 0:
            return target_limit
        resolver = getattr(cfg, "effective_max_concurrent_finalizations", None)
        if resolver is not None:
            return resolver(lane="remote", target=target.name)
        return max(1, int(getattr(cfg, "max_concurrent_finalizations", 1) or 1))

    def _finalization_limit(self) -> int:
        return self._max_concurrent_finalizations(self.cfg, self.target)

    def _mark_main_loop_progress(self) -> None:
        with self._main_loop_progress_lock:
            self._main_loop_progress_monotonic = time.monotonic()
            self._main_loop_progress_at = utc_now()

    def _main_loop_progress(self) -> tuple[str, float]:
        with self._main_loop_progress_lock:
            return self._main_loop_progress_at, self._main_loop_progress_monotonic

    def _start_agent_heartbeat(self) -> None:
        if self._agent_heartbeat_thread and self._agent_heartbeat_thread.is_alive():
            return
        self._agent_heartbeat_stop.clear()
        interval = max(5, min(60, int(self.target.agent_summary_interval_seconds or 60)))
        watchdog_timeout = max(
            300, int(getattr(self.target, "agent_heartbeat_timeout_seconds", 300) or 300)
        )

        def _heartbeat() -> None:
            while not self._agent_heartbeat_stop.wait(interval):
                try:
                    self._publish_agent_snapshot(force=True)
                except Exception as exc:
                    log.warning("Agent heartbeat snapshot failed: %s", exc)

        def _watchdog() -> None:
            check_interval = max(5, min(30, interval))
            while not self._agent_heartbeat_stop.wait(check_interval):
                progress_at, progress_mono = self._main_loop_progress()
                stalled = time.monotonic() - progress_mono
                if stalled <= watchdog_timeout:
                    continue
                log.error(
                    "Remote agent main loop stalled for %.0fs (last progress %s); "
                    "exiting %d for supervisor recovery",
                    stalled,
                    progress_at,
                    _WATCHDOG_EXIT_CODE,
                )
                _hard_exit(_WATCHDOG_EXIT_CODE)
                return

        self._agent_heartbeat_thread = threading.Thread(
            target=_heartbeat, name="remote-agent-heartbeat", daemon=True
        )
        self._agent_watchdog_thread = threading.Thread(
            target=_watchdog, name="remote-agent-watchdog", daemon=True
        )
        self._agent_heartbeat_thread.start()
        self._agent_watchdog_thread.start()

    def _stop_agent_heartbeat(self) -> None:
        self._agent_heartbeat_stop.set()
        for thread in (self._agent_heartbeat_thread, self._agent_watchdog_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5)

    def run(self) -> int:
        # Same pinned-image check as the dispatcher, for this site's images.
        from .image_pins import verify_pinned_images
        image_problems = verify_pinned_images(
            [self.target.nextflow_config, self.cfg.nextflow_config])
        if image_problems:
            for problem in image_problems:
                log.error("Pinned container image mismatch: %s", problem)
            return 1
        self._start_agent_heartbeat()
        self._mark_main_loop_progress()
        # Advertise the loaded release before startup cleanup or recovery can
        # spend a long time in filesystem I/O. Release activation must not
        # interpret a large stale work directory as a dead worker.
        self._publish_agent_snapshot(force=True)
        while True:
            self._reconcile_finalization_workers()
            self._mark_main_loop_progress()
            self._reconcile_finalization_state()
            self._reconcile_execution_cleanup()
            self._reconcile_hook_executors()
            try:
                has_active_orders = self._refresh_claimed_orders()
            except Exception:
                # Per-order failures are isolated inside the refresh; this is
                # the order listing itself failing. Keep polling rather than
                # crash-looping every batch on this target.
                log.exception("Remote order refresh cycle failed; retrying next poll")
                has_active_orders = True
            did_work = self._claim_and_process_one()
            self._publish_agent_snapshot(force=did_work)
            if self.once:
                self._stop_agent_heartbeat()
                return 0
            if did_work or has_active_orders:
                self._next_poll_delay = max(1, int(self.target.poll_interval_seconds or 60))
            else:
                self._next_poll_delay = min(
                    max(
                        self._next_poll_delay,
                        max(1, int(self.target.poll_interval_seconds or 60)),
                    ) * 2,
                    max(
                        self._next_poll_delay,
                        int(self.target.idle_poll_max_interval_seconds or 300),
                    ),
                )
            self._idle_sleep(max(1, self._next_poll_delay))

    def _idle_sleep(self, seconds: float, chunk: float = 30.0) -> None:
        """Sleep between polls, marking main-loop progress as it goes.

        The idle backoff (idle_poll_max_interval_seconds, 300 s by default) reaches the
        watchdog's stall limit (max(300, agent_heartbeat_timeout_seconds)), so one
        uninterrupted sleep made an idle agent kill itself. Waiting is not a stall.
        """
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(chunk, remaining))
            self._mark_main_loop_progress()

    def _release_finalization_lease_safely(
        self, batch_id: str, lease_name: str
    ) -> bool:
        owner = getattr(self, "_finalization_owner", "remote-agent")
        last_error = ""
        for attempt in range(1, 4):
            try:
                if self.state.release_finalization_lease(
                    batch_id, owner=owner, lease_name=lease_name
                ):
                    return True
                current = self.state.finalization_lease_status(lease_name)
                if not current or (
                    str(current.get("owner") or "") != owner
                    or str(current.get("batch_id") or "") != batch_id
                ):
                    return True
                last_error = "exact owned lease still exists"
            except Exception as exc:
                last_error = str(exc)
            if attempt < 3:
                time.sleep(0.2 * attempt)
        if last_error:
            log.error(
                "Batch %s: could not release finalization lease %s after retries: %s",
                batch_id, lease_name, last_error,
            )
        return False

    def _reconcile_finalization_workers(self) -> None:
        """Turn unexpected worker exits into durable, retryable state."""
        pool = getattr(self, "_finalization_pool", None)
        if pool is None:
            return
        outcomes = pool.drain_outcomes()
        if not outcomes:
            return
        health_state = getattr(self, "_finalization_reconcile_health", {})
        errors = list(health_state.get("errors") or [])
        for outcome in outcomes:
            batch_id = str(outcome.get("batch_id") or "")
            exc = outcome.get("exception")
            if not batch_id or exc is None and not outcome.get("cancelled"):
                continue
            message = (
                "finalization worker was cancelled"
                if outcome.get("cancelled")
                else f"finalization worker raised {type(exc).__name__}: {exc}"
            )
            log.error("Batch %s: %s; returning it to durable retry", batch_id, message)
            lease_name = self._finalization_lease_names.get(
                batch_id, REMOTE_FINALIZATION_LEASE_NAME
            )
            try:
                self.state.mark_remote_finalization_retry(
                    batch_id, error=message, delay_seconds=1,
                    owner=getattr(self, "_finalization_owner", "remote-agent"),
                    lease_name=lease_name,
                )
            except Exception as retry_exc:
                errors.append(f"{batch_id}: retry transition failed: {retry_exc}")
            if not self._release_finalization_lease_safely(batch_id, lease_name):
                errors.append(f"{batch_id}: lease release failed ({lease_name})")
            self._finalization_lease_names.pop(batch_id, None)
        health_state["errors"] = errors[-10:]
        health_state["last_worker_outcome_at"] = utc_now()
        self._finalization_reconcile_health = health_state

    def _reconcile_finalization_state(self) -> None:
        interval = max(
            5,
            int(
                getattr(
                    self.cfg,
                    "finalization_reconcile_interval_seconds",
                    self.target.finalization_poll_interval_seconds or 60,
                )
                or 60
            ),
        )
        now_mono = time.monotonic()
        if now_mono - getattr(self, "_last_finalization_reconcile_at", 0.0) < interval:
            return
        self._last_finalization_reconcile_at = now_mono
        pool = getattr(self, "_finalization_pool", None)
        active_batch_ids = pool.running_batch_ids() if pool is not None else set()
        try:
            health = self.state.reconcile_finalization_slots(
                lane="remote",
                max_slots=self._finalization_limit(),
                owner=self._finalization_owner,
                active_batch_ids=active_batch_ids,
            )
            self._finalization_reconcile_failures = 0
        except Exception as exc:
            self._finalization_reconcile_failures += 1
            health = dict(self._finalization_reconcile_health)
            health.update({
                "state": "DEGRADED",
                "updated_at": utc_now(),
                "errors": [*(health.get("errors") or []), str(exc)][-10:],
                "failure_count": self._finalization_reconcile_failures,
            })
            self._finalization_reconcile_health = health
            threshold = max(
                1,
                int(getattr(self.cfg, "finalization_reconcile_failure_threshold", 3) or 3),
            )
            log.error(
                "Finalization reconciliation failed (%d/%d): %s",
                self._finalization_reconcile_failures,
                threshold,
                exc,
            )
            if self._finalization_reconcile_failures >= threshold:
                raise RuntimeError(
                    "finalization reconciliation failed repeatedly; "
                    "stopping for systemd recovery"
                ) from exc
            return
        repaired = list(health.get("repaired") or [])
        for item in repaired:
            batch_id = str(item.get("batch_id") or "")
            self._finalization_lease_names.pop(batch_id, None)
        if repaired or health.get("terminal_released"):
            log.warning(
                "Finalization reconciliation repaired=%d terminal_released=%d "
                "expired=%d foreign_live=%d",
                len(repaired),
                len(health.get("terminal_released") or []),
                len(health.get("expired") or []),
                len(health.get("foreign_live") or []),
            )
        health.update({
            "state": str(
                health.get("health_state")
                or (
                    "DEGRADED"
                    if int(health.get("durable_slots", 0) or 0) < len(active_batch_ids)
                    else "HEALTHY"
                )
            ),
            "updated_at": utc_now(),
            "process_batch_ids": sorted(active_batch_ids),
            "errors": list(getattr(self, "_finalization_reconcile_health", {}).get("errors") or [])[-10:],
        })
        previous_heartbeat = getattr(self, "_finalization_reconcile_health", {}).get(
            "last_successful_heartbeat_at"
        )
        if previous_heartbeat:
            health["last_successful_heartbeat_at"] = previous_heartbeat
        try:
            queue_health = self.state.remote_finalization_queue_health(self.target.name)
        except Exception as exc:
            queue_health = {
                "queued": None,
                "oldest_queued_at": None,
                "oldest_queued_age_s": None,
                "error": str(exc),
            }
        health["queue"] = queue_health
        self._finalization_reconcile_health = health

    def _hook_slurm_state(self, job_id: str) -> dict[str, Any]:
        try:
            sq = subprocess.run(
                ["squeue", "-h", "-j", job_id, "-o", "%T"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            )
            states = [line.strip() for line in sq.stdout.splitlines() if line.strip()]
            if states:
                return {"state": states[0].split()[0].rstrip("+"), "source": "squeue"}
            # A finished job makes squeue exit non-zero with "Invalid job id"; any
            # other failure (controller down, timeout) says nothing about the job.
            if sq.returncode and "invalid job id" not in (sq.stderr or "").lower():
                return {"state": "UNKNOWN", "source": "squeue", "error": (sq.stderr or "").strip()[-300:]}
        except FileNotFoundError as exc:
            return {"state": "UNKNOWN", "source": "squeue", "error": str(exc)}
        except Exception as exc:
            return {"state": "UNKNOWN", "source": "squeue", "error": str(exc)}
        try:
            sa = subprocess.run(
                ["sacct", "-n", "-X", "-j", job_id,
                 "--format=State,ExitCode,NodeList", "--parsable2"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            )
            if sa.returncode:
                # slurmdbd unavailable: no rows is not evidence the job is gone.
                return {"state": "UNKNOWN", "source": "sacct", "error": (sa.stderr or "").strip()[-300:]}
            for line in sa.stdout.splitlines():
                fields = [part.strip() for part in line.split("|")]
                if not fields or not fields[0]:
                    continue
                state = fields[0].split()[0].rstrip("+")
                exit_code = None
                if len(fields) > 1:
                    try:
                        exit_code = int(fields[1].split(":", 1)[0])
                    except ValueError:
                        pass
                node = fields[2] if len(fields) > 2 else ""
                return {"state": state, "exit_code": exit_code, "node": node, "source": "sacct"}
        except Exception as exc:
            return {"state": "UNKNOWN", "source": "sacct", "error": str(exc)}
        return {"state": "MISSING", "source": "sacct"}

    def _reconcile_hook_executors(self) -> None:
        interval = max(5, int(getattr(self.cfg, "hook_executor_reconcile_interval_seconds", 60) or 60))
        now_mono = time.monotonic()
        if now_mono - self._last_hook_reconcile_at < interval:
            return
        self._last_hook_reconcile_at = now_mono
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        health: dict[str, Any] = {
            "updated_at": now_iso,
            "active": 0,
            "active_jobs": 0,
            "completed": 0,
            "missing": 0,
            "unknown": 0,
            "stale_results": 0,
            "errors": [],
        }
        try:
            candidates = self.state.get_active_hook_executors(self.target.name)
        except Exception as exc:
            health["errors"] = [f"candidate query failed: {exc}"]
            self._hook_executor_health = health
            return
        health["active"] = len(candidates)
        health["active_jobs"] = len(candidates)
        active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
        terminal_states = _SLURM_TERMINAL_STATES
        missing_grace = max(30, int(getattr(self.cfg, "hook_executor_missing_grace_seconds", 900) or 900))
        missing_polls_limit = max(1, int(getattr(self.cfg, "hook_executor_missing_polls", 3) or 3))
        for row in candidates:
            batch_id = str(row.get("batch_id") or "")
            hook_index = int(row.get("hook_index") or 0)
            job_id = str(row.get("executor_job_id") or "")
            attempt_number = int(row.get("active_attempt_number") or 0)
            if not batch_id or not job_id.isdigit() or attempt_number <= 0:
                health["errors"].append(f"invalid executor row {batch_id}/{hook_index}")
                continue
            result = self._hook_slurm_state(job_id)
            state = str(result.get("state") or "UNKNOWN").upper().rstrip("+")
            key = (batch_id, hook_index)
            if state == "UNKNOWN":
                health["unknown"] += 1
                self.state.mark_batch_hook_executor_polled(
                    batch_id, hook_index, job_id=job_id, attempt_number=attempt_number,
                    state="UNKNOWN", polled_at=now_iso,
                    missing_since=row.get("executor_missing_since"),
                )
                continue
            if state in active_states:
                self._hook_executor_missing_polls.pop(key, None)
                self.state.mark_batch_hook_executor_polled(
                    batch_id, hook_index, job_id=job_id, attempt_number=attempt_number,
                    state=state, polled_at=now_iso, missing_since=None,
                )
                continue
            if state == "MISSING":
                health["missing"] += 1
                missing_since = str(row.get("executor_missing_since") or now_iso)
                polls = self._hook_executor_missing_polls.get(key, 0) + 1
                self._hook_executor_missing_polls[key] = polls
                self.state.mark_batch_hook_executor_polled(
                    batch_id, hook_index, job_id=job_id, attempt_number=attempt_number,
                    state="MISSING", polled_at=now_iso, missing_since=missing_since,
                )
                try:
                    missing_at = datetime.fromisoformat(missing_since.replace("Z", "+00:00"))
                    missing_age = max(0.0, (now - missing_at).total_seconds())
                except ValueError:
                    missing_age = 0.0
                # Fail the hook only after BOTH the grace period and the poll count:
                # a job that briefly drops out of squeue/sacct must not be re-run
                # while it may still be appending.
                if missing_age < missing_grace or polls < missing_polls_limit:
                    continue
                completed = self.state.complete_batch_hook(
                    batch_id, hook_index, exit_code=1,
                    error_msg=f"Slurm hook job {job_id} disappeared from squeue and sacct",
                    failure_category="EXECUTOR_JOB_MISSING",
                    failure_detail=f"job_id={job_id} missing_age_seconds={int(missing_age)} polls={polls}",
                    runtime_revision=self._agent_revision,
                    expected_executor_job_id=job_id, expected_attempt_number=attempt_number,
                    executor_polled_at=now_iso,
                )
                if completed:
                    health["completed"] += 1
                else:
                    health["stale_results"] += 1
                continue
            if state not in terminal_states:
                health["unknown"] += 1
                continue
            self._hook_executor_missing_polls.pop(key, None)
            exit_code = result.get("exit_code")
            success = state == "COMPLETED" and (exit_code in (None, 0))
            if success:
                exit_code = 0
                error_msg = ""
                category = ""
            else:
                exit_code = int(exit_code or 1)
                error_msg = f"Slurm hook job {job_id} ended in {state}"
                category = ("SLURM_INFRASTRUCTURE" if state in _SLURM_INFRASTRUCTURE_FAILURES
                            else "SLURM_EXECUTION")
            completed = self.state.complete_batch_hook(
                batch_id, hook_index, exit_code=exit_code, error_msg=error_msg,
                failure_category=category, failure_detail=error_msg,
                runtime_revision=self._agent_revision, execution_node=str(result.get("node") or ""),
                expected_executor_job_id=job_id, expected_attempt_number=attempt_number,
                executor_polled_at=now_iso,
            )
            if completed:
                health["completed"] += 1
            else:
                health["stale_results"] += 1
        health["errors"] = health["errors"][-10:]
        progress_times = [
            str(row.get("executor_last_polled_at") or "")
            for row in candidates
            if row.get("executor_last_polled_at")
        ]
        if progress_times:
            health["last_progress_at"] = max(progress_times)
        self._hook_executor_health = health


    def _publish_agent_snapshot(self, *, force: bool = False) -> None:
        if not self._snapshot_lock.acquire(blocking=False):
            return
        try:
            self._publish_agent_snapshot_locked(force=force)
        finally:
            self._snapshot_lock.release()

    def _publish_agent_snapshot_locked(self, *, force: bool = False) -> None:
        now = time.monotonic()
        interval = max(1, int(self.target.agent_summary_interval_seconds or 60))
        if not force and now - self._last_agent_summary_at < interval:
            return
        self._last_agent_summary_at = now
        progress_at, progress_mono = self._main_loop_progress()
        stalled_for = max(0.0, time.monotonic() - progress_mono)
        payload = {
            "target": self.target.name,
            "host": socket.gethostname(),
            "status": "ALIVE",
            "heartbeat_at": utc_now(),
            "agent_revision": self._agent_revision,
            "slurm": self._collect_target_slurm_summary(),
            "coordination": {
                "status_writes_suppressed": self._status_writes_suppressed,
                "driver_slurm_queries": self._driver_slurm_queries,
                "agent_snapshot_writes": self._agent_snapshot_writes,
            },
            "storage": self._capacity_guard.summary(),
            "hook_executor_health": dict(self._hook_executor_health),
            "main_loop": {
                "last_progress_at": progress_at,
                "stalled_for_s": stalled_for,
            },
            "finalization": {
                "active": self._finalization_pool.running_count(),
                "limit": self._finalization_limit(),
                "available": max(
                    0,
                    self._finalization_limit() - self._finalization_pool.running_count(),
                ),
                "process_batch_ids": sorted(self._finalization_pool.running_batch_ids()),
                "heartbeats": dict(self._finalization_heartbeat_health),
                "reconciliation": dict(getattr(self, "_finalization_reconcile_health", {})),
            },
        }
        self.state.write_remote_agent_snapshot(self.target.name, payload)
        self._agent_snapshot_writes += 1

    def _claim_and_process_one(self) -> bool:
        if not self._capacity_available():
            return False
        order = self.state.claim_remote_batch(
            self.target.name,
            claimer_host=socket.gethostname(),
        )
        if not order:
            return False
        self._process_order(order)
        return True

    def _capacity_available(self) -> bool:
        return self._capacity_guard.check(
            (self.target.remote_scratch_root, self.target.remote_outdir)
        )

    def _find_existing_driver_job(self, order: dict[str, Any]) -> str | None:
        """Return the Slurm job id of this batch's driver if one was submitted.

        Drivers are named ``mussel-driver-<batch_id>`` and a batch never gets
        a second driver, so any job with that name (live in squeue, or
        finished in sacct) is this batch's driver. Raises when Slurm cannot
        answer: submitting blind could start a duplicate driver.
        """
        name = f"{_DRIVER_JOB_PREFIX}{order['batch_id']}"
        live = _squeue_job_ids_by_name(name)
        if live:
            if len(live) > 1:
                log.warning(
                    "Batch %s: %d live driver jobs named %s; adopting newest %s, "
                    "leaving duplicates %s running (not cancelled automatically)",
                    order["batch_id"], len(live), name, live[-1], ", ".join(live[:-1]),
                )
            return live[-1]
        finished = _sacct_jobs_by_name(name, _sacct_since(order.get("created_at")))
        return finished[-1]["job_id"] if finished else None

    def _remote_cleanup_spec(self, order: dict[str, Any]) -> ExecutionCleanupSpec | None:
        work_hint = str(order.get("remote_work_dir") or "").strip()
        run_dir = str(order.get("remote_run_dir") or "").strip()
        if not run_dir and work_hint:
            run_dir = str(Path(work_hint).parent)
        if not run_dir:
            return None
        work_dir = work_hint or os.path.join(run_dir, "work")
        task_scratch = os.path.join(run_dir, self.target.task_scratch_subdir)
        execution_paths = tuple(dict.fromkeys((work_dir, task_scratch,
                                                os.path.join(run_dir, ".nextflow"),
                                                os.path.join(run_dir, ".cache"))))
        return ExecutionCleanupSpec(
            batch_id=str(order.get("batch_id") or "unknown"),
            managed_root=self.target.remote_scratch_root,
            run_root=run_dir,
            execution_paths=execution_paths,
            protected_paths=(str(order.get("remote_outdir") or self.target.remote_outdir),),
        )

    def _cleanup_remote_execution(self, order: dict[str, Any]) -> None:
        if not self.cfg.cleanup_work_dir:
            return
        spec = self._remote_cleanup_spec(order)
        if spec is None:
            return
        cleaner = getattr(self, "_execution_cleaner", None)
        if cleaner is None:
            cleaner = self._execution_cleaner = ExecutionArtifactCleaner()
        result = cleaner.cleanup(spec)
        batch_id = str(order.get("batch_id") or "")
        if batch_id:
            self.state.record_execution_cleanup(
                batch_id,
                error=result.error,
                retryable=result.retryable,
            )

    def _reconcile_execution_cleanup(self) -> None:
        if not self.cfg.cleanup_work_dir:
            return
        interval = max(
            60, int(self.target.storage_policy.cleanup_interval_seconds or 3600)
        )
        now = time.monotonic()
        if now - self._last_storage_cleanup_at < interval:
            return
        self._last_storage_cleanup_at = now

        def _spec(batch):
            if str(batch.get("target") or self.target.name) != self.target.name:
                return None
            order = dict(batch)
            order.setdefault("batch_id", batch.get("batch_id"))
            return self._remote_cleanup_spec(order)

        cleaned = reconcile_execution_cleanup(
            self.state,
            _spec,
            cleaner=self._execution_cleaner,
            candidates=self.state.get_remote_execution_cleanup_batches(self.target.name),
            limit=int(self.target.storage_policy.cleanup_batch_size or 50),
            budget_seconds=float(self.target.storage_policy.cleanup_budget_seconds or 15.0),
        )
        if cleaned:
            print(f"[mussel-remote-agent] reconciled {cleaned} execution batch(es)")
        staged = self._reconcile_wds_staging_cleanup(
            limit=int(self.target.storage_policy.cleanup_batch_size or 50),
            budget_seconds=float(self.target.storage_policy.cleanup_budget_seconds or 15.0),
        )
        if staged:
            print(f"[mussel-remote-agent] removed WDS staging for {staged} succeeded batch(es)")

    def _wds_staging_cleanup_spec(self, order: dict[str, Any]) -> ExecutionCleanupSpec | None:
        """Per-batch WDS staging is disposable only after finalization succeeds.

        WDS hooks upload every sealed and partial staging shard to S3 with a
        size check, so once the batch is SUCCEEDED the local copy is
        redundant. Until then hook retries reuse it, which is why execution
        cleanup (run at driver exit, before hooks) must not touch it.
        """
        run_dir = str(order.get("remote_run_dir") or "").strip()
        if not run_dir:
            return None
        return ExecutionCleanupSpec(
            batch_id=str(order.get("batch_id") or "unknown"),
            managed_root=self.target.remote_scratch_root,
            run_root=run_dir,
            execution_paths=(os.path.join(run_dir, WDS_STAGING_SUBDIR),),
            protected_paths=(str(order.get("remote_outdir") or self.target.remote_outdir),),
        )

    def _cleanup_remote_wds_staging(self, order: dict[str, Any]) -> bool:
        if not self.cfg.cleanup_work_dir:
            return False
        spec = self._wds_staging_cleanup_spec(order)
        if spec is None:
            return False
        cleaner = getattr(self, "_execution_cleaner", None)
        if cleaner is None:
            cleaner = self._execution_cleaner = ExecutionArtifactCleaner()
        return bool(cleaner.cleanup(spec).removed)

    def _reconcile_wds_staging_cleanup(self, *, limit: int, budget_seconds: float) -> int:
        """Remove WDS staging left behind by batches that already SUCCEEDED.

        Covers agent restarts between the SUCCEEDED transition and cleanup,
        and batches finalized before staging cleanup existed.
        """
        if not self.cfg.cleanup_work_dir:
            return 0
        scratch_root = str(self.target.remote_scratch_root or "").strip()
        if not scratch_root:
            return 0
        try:
            run_dirs = sorted(
                path.parent for path in Path(scratch_root).glob(f"*/{WDS_STAGING_SUBDIR}")
            )
        except OSError as exc:
            log.warning("WDS staging scan under %s failed: %s", scratch_root, exc)
            return 0
        started = time.monotonic()
        cleaned = 0
        processed = 0
        for run_dir in run_dirs:
            if cleaned >= max(0, int(limit)):
                break
            if processed > 0 and time.monotonic() - started >= max(0.0, float(budget_seconds)):
                break
            processed += 1
            batch = self.state.get_batch(run_dir.name)
            if (
                not batch
                or batch.get("backend") != "remote_agent"
                or str(batch.get("target") or "") != self.target.name
                or batch.get("remote_status") != "SUCCEEDED"
            ):
                continue
            if self._cleanup_remote_wds_staging({
                "batch_id": run_dir.name,
                "remote_run_dir": str(run_dir),
                "remote_outdir": batch.get("remote_outdir"),
            }):
                cleaned += 1
        return cleaned

    def _process_order(
        self,
        order: dict[str, Any],
        *,
        replay_capacity: Callable[[], bool] | None = None,
    ) -> None:
        """Prepare and submit one order's driver.

        ``replay_capacity`` marks a replay (orphaned CLAIMED order or BLOCKED
        retry). Only those can follow an sbatch whose job id was lost, so only
        they look for an existing driver to adopt -- before the storage gate,
        which then applies to an actual new submission. A fresh claim has
        already passed the gate and cannot have a driver yet.
        """
        order = self._normalize_order(order)
        batch_id = order["batch_id"]
        client = RemoteDispatchClient(self.cfg, self.target, state=self.state)
        remote_run_dir = Path(order["remote_run_dir"])
        driver_path = Path(order["remote_driver_script"])
        status = {
            "status": "SUBMITTED" if self.dry_run else "STARTING",
            "batch_id": batch_id,
            "target": self.target.name,
            "updated_at": utc_now(),
            "heartbeat_at": utc_now(),
            "remote_run_dir": str(remote_run_dir),
            "remote_driver_script": str(driver_path),
            "remote_work_dir": order.get("remote_work_dir"),
            "remote_outdir": order.get("remote_outdir"),
            "remote_log_path": order.get("remote_driver_log") or str(driver_path.with_name("driver.out")),
        }
        try:
            if replay_capacity is not None and not self.dry_run:
                # A replay may follow an sbatch that succeeded before its job
                # id was recorded. Adopt that job instead of starting a second
                # driver in the same work dir.
                existing_job_id = self._find_existing_driver_job(order)
                if existing_job_id:
                    log.warning(
                        "Batch %s: adopting existing Slurm driver job %s instead of resubmitting",
                        batch_id, existing_job_id,
                    )
                    status["driver_job_id"] = existing_job_id
                    status["status"] = "RUNNING"
                    status["adopted_driver_job"] = True
                    client.write_status(batch_id, status)
                    return
                if not replay_capacity():
                    return
            order = self._pin_order_to_staged_release(order)
            remote_run_dir.mkdir(parents=True, exist_ok=True)
            _materialize_order_params(order)
            _verify_order_release(order)
            _verify_order_profile(order, self.cfg, self.target)
            _verify_order_runtime(order)
            write_batch_csv(
                remote_run_dir / "batch.csv", order.get("slides", []), batch_id=batch_id
            )
            driver = render_driver_script(order, self.target)
            driver_path.write_text(driver)
            driver_path.chmod(0o755)
            if self.dry_run:
                status["driver_job_id"] = None
                client.write_status(batch_id, status)
                return
            result = subprocess.run(
                ["sbatch", "--parsable", str(driver_path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            job_id = result.stdout.strip().split(";")[0]
            status["driver_job_id"] = job_id
            status["status"] = "RUNNING"
            client.write_status(batch_id, status)
            self._stage_attempt_state().pop(batch_id, None)
        except Exception as exc:
            error_text = str(exc)
            stage_exhausted = False
            if isinstance(exc, StagedReleaseError) and not exc.transient:
                attempts = self._stage_attempt_state().get(batch_id, 0) + 1
                self._stage_attempt_state()[batch_id] = attempts
                stage_exhausted = attempts >= _STAGE_MISSING_MAX_ATTEMPTS
                if stage_exhausted:
                    # The order's release will not appear; return its slides
                    # so they are re-dispatched under the current release.
                    self._stage_attempt_state().pop(batch_id, None)
            stale_order = stage_exhausted or any(
                marker in error_text
                for marker in (
                    "source release drift",
                    "release manifest drift",
                    "execution profile drift",
                    "Schema-v3 remote order is missing",
                )
            )
            status["status"] = REMOTE_BLOCKED_STATUS
            status["error"] = f"remote preparation or submission failed: {error_text}"
            status["failure_class"] = (
                "STALE_ORDER" if stale_order else "REMOTE_PREPARATION"
            )
            status["nextflow_exit"] = None
            status["retry_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=_BLOCKED_RETRY_SECONDS)
            ).isoformat()
            client.write_status(batch_id, status)
            if stale_order:
                self.state.requeue_blocked_remote_batch(batch_id, reason=status["error"])

    def _capture_remote_log_tail(
        self,
        order: dict[str, Any],
        previous: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Build a bounded mirror of remote driver and recent hook logs."""
        run_dir = Path(str(order.get("remote_run_dir") or ""))
        if not str(order.get("remote_run_dir") or "").strip():
            return previous
        candidates: list[tuple[str, Path, int]] = [
            (
                "driver.out",
                Path(str(order.get("remote_driver_log") or run_dir / "driver.out")),
                _REMOTE_CORE_LOG_TAIL_BYTES,
            ),
            (
                "driver.err",
                Path(str(order.get("remote_driver_err") or run_dir / "driver.err")),
                _REMOTE_CORE_LOG_TAIL_BYTES,
            ),
            (
                "nextflow.log",
                Path(str(order.get("remote_nf_log") or run_dir / "nextflow.log")),
                _REMOTE_CORE_LOG_TAIL_BYTES,
            ),
        ]
        hook_dir = run_dir / "hooks"
        try:
            hook_paths = sorted(
                (
                    path
                    for path in hook_dir.glob("hook_*.*")
                    if path.is_file() and path.suffix in {".out", ".err"}
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )[:_REMOTE_HOOK_LOG_FILE_LIMIT]
        except OSError:
            hook_paths = []
        candidates.extend(
            (f"hooks/{path.name}", path, _REMOTE_HOOK_LOG_TAIL_BYTES)
            for path in hook_paths
        )

        remaining = _REMOTE_LOG_TAIL_MAX_BYTES
        files: list[dict[str, Any]] = []
        had_read_error = False
        for name, path, file_limit in candidates:
            if remaining <= 0:
                break
            try:
                text, truncated, consumed = _read_bounded_log_tail(
                    path, min(file_limit, remaining)
                )
            except FileNotFoundError:
                continue
            except OSError:
                had_read_error = True
                continue
            if not text:
                continue
            remaining -= consumed
            files.append({"name": name, "text": text, "truncated": truncated})

        if not files:
            return previous if had_read_error or previous else None
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        content_sha256 = _sha256_bytes(canonical)
        if previous and previous.get("content_sha256") == content_sha256:
            return previous
        return {
            "version": 1,
            "captured_at": utc_now(),
            "content_sha256": content_sha256,
            "files": files,
        }

    def _attach_remote_log_tail(
        self,
        order: dict[str, Any],
        current_status: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        tail = self._capture_remote_log_tail(order, current_status.get("log_tail"))
        if tail:
            payload["log_tail"] = tail

    def _publish_remote_log_tail(
        self,
        client: RemoteDispatchClient,
        order: dict[str, Any],
        status: dict[str, Any],
    ) -> None:
        payload: dict[str, Any] = {"batch_id": order["batch_id"]}
        self._attach_remote_log_tail(order, status, payload)
        tail = payload.get("log_tail")
        if tail and tail != status.get("log_tail"):
            self._write_status_if_changed(client, order["batch_id"], payload)

    def _refresh_claimed_orders(self) -> bool:
        client = RemoteDispatchClient(self.cfg, self.target, state=self.state)
        order_iter = self.state.get_remote_batches_for_target(self.target.name)
        has_active_orders = False
        cycle = _RefreshCycle(self)
        for order in order_iter:
            if not order:
                continue
            has_active_orders = True
            batch_id = str(order.get("batch_id") or "")
            if self._order_in_backoff(batch_id):
                continue
            # One bad order (a persistent DB or filesystem error while
            # recovering it) must not halt every other batch on this target.
            try:
                self._refresh_order(order, client, cycle)
            except Exception as exc:
                self._record_order_failure(batch_id, exc)
            else:
                self._order_failure_state().pop(batch_id, None)
        return has_active_orders

    def _order_failure_state(self) -> dict[str, tuple[int, float]]:
        return self.__dict__.setdefault("_order_failures", {})

    def _order_in_backoff(self, batch_id: str) -> bool:
        failure = self._order_failure_state().get(batch_id)
        return bool(failure) and time.monotonic() < failure[1]

    def _record_order_failure(self, batch_id: str, exc: Exception) -> None:
        failures, _ = self._order_failure_state().get(batch_id, (0, 0.0))
        failures += 1
        delay = min(
            _ORDER_FAILURE_BACKOFF_MAX_SECONDS,
            _ORDER_FAILURE_BACKOFF_BASE_SECONDS * 2 ** (failures - 1),
        )
        self._order_failure_state()[batch_id] = (failures, time.monotonic() + delay)
        log.error(
            "Batch %s: refresh failed (%d consecutive); skipping it for %ds: %s",
            batch_id or "<unknown>", failures, delay, exc,
            exc_info=exc,
        )

    def _refresh_order(
        self, order: dict[str, Any], client: RemoteDispatchClient, cycle: "_RefreshCycle"
    ) -> None:
        # Orders persisted before schema-v3 path mappings were added may
        # not contain hook_path_replacements.  Normalize every order
        # before status-driven recovery/finalization, not only newly
        # claimed orders, so legacy hook replays use the target's remote
        # paths instead of the local dispatcher's paths.
        order = self._normalize_order(order)
        batch_id = order["batch_id"]
        status = client.read_status(batch_id) or {}
        if status.get("status") in TERMINAL_REMOTE_STATUSES:
            return
        if status.get("status") == HOOKS_BLOCKED_STATUS:
            return
        if status.get("status") == REMOTE_BLOCKED_STATUS:
            if self._blocked_retry_due(status):
                self._process_order(order, replay_capacity=cycle.capacity_ok)
            return
        if status.get("status") == "HOOKS_RUNNING" and not self._finalization_is_stale(status):
            self._publish_remote_log_tail(client, order, status)
            return
        if status.get("status") == "HOOKS_RETRY" and not self._finalization_retry_due(status):
            self._publish_remote_log_tail(client, order, status)
            return
        job_id = status.get("driver_job_id")
        recoverable_status = str(status.get("status") or "").upper()
        if recoverable_status in {"HOOKS_RETRY", "FINALIZATION_QUEUED", "HOOKS_RUNNING"}:
            if self._recover_persisted_finalization(order, status, client):
                return
        if not job_id:
            if cycle.agent is None:
                cycle.agent = self.state.read_remote_agent(self.target.name) or {}
        if not job_id and self._should_replay_claimed_order(order, status, cycle.agent):
            self._process_order(order, replay_capacity=cycle.capacity_ok)
            return
        if status.get("cancel_requested_at") and job_id:
            try:
                subprocess.run(
                    ["scancel", str(job_id)],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            except Exception:
                pass
        if not job_id:
            return
        slurm_state = self._driver_slurm_state(str(job_id))
        update = {
            "batch_id": batch_id,
            "target": self.target.name,
            "heartbeat_at": utc_now(),
            "updated_at": utc_now(),
            "driver_job_id": job_id,
            "remote_run_dir": order.get("remote_run_dir"),
            "remote_work_dir": order.get("remote_work_dir"),
            "remote_outdir": order.get("remote_outdir"),
            "remote_log_path": order.get("remote_driver_log"),
            "slurm": slurm_state,
        }
        self._attach_remote_log_tail(order, status, update)
        raw_state = str(slurm_state.get("state") or "").split()
        state = raw_state[0].rstrip("+") if raw_state else None   # "CANCELLED by 123", "CANCELLED+"
        if state in {"COMPLETED"}:
            self.state.mark_driver_terminal(batch_id, 0)
            self._cleanup_remote_work_dir(order)
            # The database lease, not this process-local thread list, is
            # the serialization boundary. Publish queued state while a
            # different batch owns the lease so completed drivers do not
            # remain misleadingly RUNNING until that hook finishes.
            max_finalizations = max(
                1, self._finalization_limit(),
            )
            pool = getattr(self, "_finalization_pool", None)
            active_finalizations = (
                pool.running_count()
                if pool is not None
                else sum(
                    thread.is_alive()
                    for thread in getattr(self, "_finalization_threads", {}).values()
                )
            )
            if active_finalizations >= max_finalizations:
                update["status"] = FINALIZATION_QUEUED_STATUS
                update["nextflow_exit"] = 0
                client.write_status(batch_id, update)
                return
            lease_name = self.state.claim_finalization_slot(
                batch_id,
                lane="remote",
                max_slots=max_finalizations,
                owner=getattr(self, "_finalization_owner", "remote-agent"),
                ttl_seconds=max(
                    1,
                    int(getattr(self.target, "finalization_lease_ttl_seconds", 300) or 300),
                ),
            )
            if not lease_name:
                update["status"] = FINALIZATION_QUEUED_STATUS
                update["nextflow_exit"] = 0
                client.write_status(batch_id, update)
                return
            if not self.state.claim_remote_finalization(
                batch_id,
                stale_after_seconds=max(
                    1, int(getattr(self.target, "finalization_lease_ttl_seconds", 300) or 300)
                ),
                owner=getattr(self, "_finalization_owner", "remote-agent"),
                lease_name=lease_name,
            ):
                self._release_finalization_lease_safely(batch_id, lease_name)
                update["status"] = FINALIZATION_QUEUED_STATUS
                update["nextflow_exit"] = 0
                client.write_status(batch_id, update)
                return
            update["status"] = "HOOKS_RUNNING"
            update["nextflow_exit"] = 0
            if not hasattr(self, "_finalization_lease_names"):
                self._finalization_lease_names = {}
            self._finalization_lease_names[batch_id] = lease_name
        elif state in _SLURM_TERMINAL_STATES - {"COMPLETED"}:
            # Every terminal state, as for hook jobs: DEADLINE, OOM or REVOKED used
            # to fall through to RUNNING and leave the batch running forever.
            update["status"] = "CANCELLED" if state == "CANCELLED" else "FAILED"
            # sacct often reports 0:0 for a job Slurm ended (CANCELLED, TIMEOUT,
            # DEADLINE, REVOKED). A failed batch must never publish exit 0: the
            # dispatcher's waiter would clean it up as a success.
            update["nextflow_exit"] = slurm_state.get("exit_code") or -1
            update["failure_class"] = (
                "SLURM_INFRASTRUCTURE" if state in _SLURM_INFRASTRUCTURE_FAILURES
                else "SLURM_EXECUTION"
            )
            update["error"] = f"driver job ended with Slurm state {state}"
        elif status.get("cancel_requested_at"):
            update["status"] = "CANCEL_REQUESTED"
        else:
            update["status"] = "RUNNING"
        # -1 is reserved for a terminal failure whose exit code could not
        # be recovered.  A live Slurm job has no exit code yet; publishing
        # -1 here makes an active batch look failed in the dashboard.
        if update["status"] in {"RUNNING", "CANCEL_REQUESTED"}:
            update["nextflow_exit"] = None
        if update["status"] not in TERMINAL_REMOTE_STATUSES:
            self._write_status_if_changed(client, batch_id, update)
        if update["status"] == "HOOKS_RUNNING":
            self._launch_finalization(order, job_id)
        elif update["status"] in TERMINAL_REMOTE_STATUSES:
            # Release the batch and its slides BEFORE publishing the
            # terminal remote_status: terminal rows are no longer listed
            # for this agent and the dispatcher defers to it, so a crash
            # after a terminal write but before these steps would strand
            # the batch RUNNING and its slides DISPATCHED forever. Each
            # step is idempotent and is simply redone on the next poll.
            exit_code = int(update.get("nextflow_exit") or -1)
            self.state.complete_batch(batch_id, exit_code)
            self.state.skip_unstarted_batch_hooks(
                batch_id,
                reason=f"driver ended before hooks: Slurm state {state}",
            )
            slurm_failure = str(state or "").upper()
            infrastructure_failure = slurm_failure in _SLURM_INFRASTRUCTURE_FAILURES
            # mark_slides_complete charges fail_count on every call, so a
            # replay after a crash must not run it for already-released
            # slides.
            if self._batch_still_owns_dispatched_slides(order):
                self.state.mark_slides_complete(
                    batch_id,
                    False,
                    charge_fail_count=not infrastructure_failure,
                )
            self._write_status_if_changed(client, batch_id, update)
            self._cleanup_remote_work_dir(order)

    def _batch_still_owns_dispatched_slides(self, order: dict[str, Any]) -> bool:
        """True unless every slide of the order is known to be released."""
        batch_id = str(order.get("batch_id") or "")
        slide_ids = [
            str(slide.get("slide_id"))
            for slide in (order.get("slides") or [])
            if isinstance(slide, dict) and slide.get("slide_id")
        ]
        if not slide_ids:
            return True
        statuses = self.state.get_slide_statuses(slide_ids)
        return any(
            str(row.get("status") or "") == "DISPATCHED"
            and str(row.get("batch_id") or "") == batch_id
            for row in statuses.values()
        )

    def _write_status_if_changed(
        self, client: RemoteDispatchClient, batch_id: str, payload: dict[str, Any]
    ) -> None:
        """Avoid rewriting unchanged running-state payloads every poll."""
        if not client.write_status(batch_id, payload):
            self._status_writes_suppressed = getattr(self, "_status_writes_suppressed", 0) + 1

    def _finalization_is_stale(self, status: dict[str, Any]) -> bool:
        heartbeat = status.get("heartbeat_at") or status.get("updated_at")
        if not heartbeat:
            return True
        try:
            age = time.time() - datetime.fromisoformat(str(heartbeat)).timestamp()
        except (TypeError, ValueError, OSError):
            return True
        return age >= max(1, int(self.target.finalization_lease_ttl_seconds or 300))

    @staticmethod
    def _finalization_retry_due(status: dict[str, Any]) -> bool:
        retry_at = status.get("finalization_retry_at")
        if not retry_at:
            return True
        try:
            return datetime.fromisoformat(str(retry_at)).timestamp() <= time.time()
        except (TypeError, ValueError, OSError):
            return True

    @staticmethod
    def _blocked_retry_due(status: dict[str, Any]) -> bool:
        retry_at = status.get("retry_at")
        if not retry_at:
            return True
        try:
            return datetime.fromisoformat(str(retry_at)).timestamp() <= time.time()
        except (TypeError, ValueError, OSError):
            return True

    def _launch_finalization(
        self,
        order: dict[str, Any],
        job_id: str,
        *,
        lease_name: str | None = None,
    ) -> None:
        batch_id = order["batch_id"]
        lease_name = lease_name or getattr(self, "_finalization_lease_names", {}).get(
            batch_id, REMOTE_FINALIZATION_LEASE_NAME
        )
        if not hasattr(self, "_finalization_lock"):
            self._finalization_lock = threading.RLock()
            self._finalization_threads = {}
        with self._finalization_lock:
            current = self._finalization_threads.get(batch_id)
            if current and current.is_alive():
                return
            pool = getattr(self, "_finalization_pool", None)
            max_finalizations = max(
                1, self._finalization_limit(),
            )
            active = (
                pool.running_count()
                if pool is not None
                else sum(thread.is_alive() for thread in self._finalization_threads.values())
            )
            if active >= max_finalizations:
                return
            if self.once:
                self._finalize_and_publish(order, job_id, lease_name)
                return
            if pool is not None:
                future = pool.submit(
                    batch_id,
                    self._finalize_and_publish,
                    order,
                    job_id,
                    lease_name,
                )
                if future is None:
                    self.state.mark_remote_finalization_retry(
                        batch_id,
                        error="finalization worker pool became full",
                        delay_seconds=1,
                        owner=getattr(self, "_finalization_owner", "remote-agent"),
                        lease_name=lease_name,
                    )
                    self._release_finalization_lease_safely(batch_id, lease_name)
                    getattr(self, "_finalization_lease_names", {}).pop(batch_id, None)
                    return
                return
            if pool is None:
                self._finalize_and_publish(order, job_id, lease_name)
                return
            thread = threading.Thread(
                target=self._finalize_and_publish,
                args=(order, job_id, lease_name),
                name=f"remote-finalize-{batch_id}",
                daemon=True,
            )
            self._finalization_threads[batch_id] = thread
            thread.start()

    def _recover_persisted_finalization(
        self,
        order: dict[str, Any],
        status: dict[str, Any],
        client: RemoteDispatchClient,
    ) -> bool:
        """Reclaim a queued finalization after its worker or agent disappeared."""
        batch_id = str(order.get("batch_id") or "")
        if not batch_id:
            return False
        max_finalizations = max(
            1, self._finalization_limit(),
        )
        pool = getattr(self, "_finalization_pool", None)
        active_finalizations = (
            pool.running_count()
            if pool is not None
            else sum(
                thread.is_alive()
                for thread in getattr(self, "_finalization_threads", {}).values()
            )
        )
        if active_finalizations >= max_finalizations:
            return False
        owner = getattr(self, "_finalization_owner", "remote-agent")
        ttl_seconds = max(
            1,
            int(getattr(self.target, "finalization_lease_ttl_seconds", 300) or 300),
        )
        lease_name = self.state.claim_finalization_slot(
            batch_id,
            lane="remote",
            max_slots=max_finalizations,
            owner=owner,
            ttl_seconds=ttl_seconds,
        )
        if not lease_name:
            return False
        if not self.state.claim_remote_finalization(
            batch_id,
            stale_after_seconds=ttl_seconds,
            owner=owner,
            lease_name=lease_name,
        ):
            self._release_finalization_lease_safely(batch_id, lease_name)
            return False
        driver_job_id = str(
            status.get("driver_job_id")
            or order.get("remote_driver_job_id")
            or "recovered"
        )
        update = {
            "batch_id": batch_id,
            "target": self.target.name,
            "status": "HOOKS_RUNNING",
            "heartbeat_at": utc_now(),
            "updated_at": utc_now(),
            "driver_job_id": driver_job_id,
            "nextflow_exit": 0,
        }
        self._attach_remote_log_tail(order, status, update)
        self._write_status_if_changed(client, batch_id, update)
        if not hasattr(self, "_finalization_lease_names"):
            self._finalization_lease_names = {}
        self._finalization_lease_names[batch_id] = lease_name
        self._launch_finalization(order, driver_job_id, lease_name=lease_name)
        return True

    def _finalize_and_publish(
        self,
        order: dict[str, Any],
        job_id: str,
        lease_name: str | None = None,
    ) -> None:
        batch_id = order["batch_id"]
        lease_name = lease_name or getattr(self, "_finalization_lease_names", {}).get(
            batch_id, REMOTE_FINALIZATION_LEASE_NAME
        )
        try:
            try:
                finalization = self._finalize_success(order)
            except StagedReleaseError as exc:
                self._defer_finalization_for_stage(batch_id, exc)
                return
            except Exception as exc:
                status = self.state.read_remote_status(batch_id) or {}
                self._publish_remote_log_tail(
                    RemoteDispatchClient(self.cfg, self.target, state=self.state),
                    order,
                    status,
                )
                self.state.mark_remote_finalization_retry(
                    batch_id, error=str(exc), delay_seconds=60,
                    owner=getattr(self, "_finalization_owner", "remote-agent"),
                    lease_name=lease_name,
                )
                return
            self._stage_attempt_state().pop(batch_id, None)
            if not (
                finalization.wds_hooks_succeeded
                and finalization.db_sync_hooks_succeeded
            ):
                status = self.state.read_remote_status(batch_id) or {}
                self._publish_remote_log_tail(
                    RemoteDispatchClient(self.cfg, self.target, state=self.state),
                    order,
                    status,
                )
                hooks = self.state.get_batch_hooks([batch_id]).get(batch_id, [])
                failed = [
                    f"{row.get('hook_name') or row.get('hook_index')}: {row.get('error_msg') or 'failed'}"
                    for row in hooks
                    if row.get("status") == "FAILED" and row.get("required", 1)
                ]
                blocked_hooks = [
                    {
                        "hook_name": row.get("hook_name") or row.get("hook_index"),
                        "hook_index": row.get("hook_index"),
                        "failure_category": row.get("failure_category"),
                        "error_msg": row.get("error_msg") or "failed",
                    }
                    for row in hooks
                    if row.get("status") == "FAILED"
                    and row.get("required", 1)
                    and row.get("failure_category") in {"COVERAGE_MISSING", "HOOK_BUG"}
                ]
                if blocked_hooks:
                    blocked_error = "; ".join(
                        f"{item['hook_name']}: {item['error_msg']}"
                        for item in blocked_hooks
                    )
                    self.state.mark_remote_finalization_blocked(
                        batch_id,
                        error=blocked_error[:2000] or "manual intervention required",
                        payload={
                            "blocked_reason": "required hook needs operator intervention",
                            "blocked_hooks": blocked_hooks,
                        },
                        owner=getattr(self, "_finalization_owner", "remote-agent"),
                        lease_name=lease_name,
                    )
                    return
                self.state.mark_remote_finalization_retry(
                    batch_id,
                    error="; ".join(failed)[:2000] or "required finalization incomplete",
                    delay_seconds=60,
                    owner=getattr(self, "_finalization_owner", "remote-agent"),
                    lease_name=lease_name,
                )
                return
            hooks = self.state.get_batch_hooks([batch_id]).get(batch_id, [])
            status = self.state.read_remote_status(batch_id) or {}
            terminal_payload = {
                "batch_id": batch_id,
                "target": self.target.name,
                "driver_job_id": job_id,
                "hooks": hooks,
                "wds_hooks_succeeded": True,
                "db_sync_hooks_succeeded": True,
            }
            self._attach_remote_log_tail(order, status, terminal_payload)
            if not self.state.transition_remote_status(
                batch_id,
                expected_statuses={"HOOKS_RUNNING"},
                status="SUCCEEDED",
                payload=terminal_payload,
            ):
                return
            self._cleanup_remote_wds_staging(order)
        finally:
            self._release_finalization_lease_safely(batch_id, lease_name)
            with self._finalization_lock:
                self._finalization_threads.pop(batch_id, None)
                getattr(self, "_finalization_lease_names", {}).pop(batch_id, None)

    def _defer_finalization_for_stage(self, batch_id: str, exc: StagedReleaseError) -> None:
        """Back off on an unusable staged release; quarantine if it stays missing."""
        attempts = self._stage_attempt_state().get(batch_id, 0) + 1
        self._stage_attempt_state()[batch_id] = attempts
        if not exc.transient and attempts >= _STAGE_MISSING_MAX_ATTEMPTS:
            self._stage_attempt_state().pop(batch_id, None)
            error = f"post-batch hooks need operator action: {exc}"
            log.error("Batch %s: %s", batch_id, error)
            self.state.mark_remote_finalization_blocked(
                batch_id,
                error=error[:2000],
                payload={
                    "blocked_reason": "staged release missing",
                    "staged_release_error": str(exc)[:2000],
                },
            )
            return
        delay = min(
            _STAGE_RETRY_MAX_SECONDS, _STAGE_RETRY_BASE_SECONDS * 2 ** (attempts - 1)
        )
        log.warning(
            "Batch %s: deferring hooks %ds (%s attempt %d): %s",
            batch_id, delay, "transient" if exc.transient else "missing stage",
            attempts, exc,
        )
        self.state.mark_remote_finalization_retry(
            batch_id, error=str(exc)[:2000], delay_seconds=delay
        )

    def _hook_repo_dir(self, order: dict[str, Any]) -> str:
        """Resolve a batch hook repo to its immutable recorded release.

        Raises :class:`StagedReleaseError` when the order's staged release is
        configured but unusable: the active link may already point at a newer
        release, and running this batch's hooks from it would mix revisions.
        """
        staged = self._resolve_staged_release(order)
        if staged is not None:
            return str(staged)
        # Legacy order or a target without release staging.
        return str(
            order.get("remote_repo_dir")
            or self.target.remote_repo_dir
            or self.cfg.repo_dir
            or ""
        )

    def _staged_release_path(self, order: dict[str, Any]) -> Path | None:
        """Return ``release_stage_root/<deployment>/<order revision>`` if configured."""
        revision = str(order.get("source_revision") or "").strip()
        stage_root = str(getattr(self.target, "release_stage_root", "") or "").strip()
        deployment_id = str(
            getattr(self.cfg, "release_deployment_id", "") or "default"
        ).strip()
        if not (revision and stage_root and deployment_id):
            return None
        return Path(stage_root) / deployment_id / revision

    def _resolve_staged_release(self, order: dict[str, Any]) -> Path | None:
        """Return the order's staged release dir, or None if staging is not configured.

        One policy for drivers and hooks: when staging is configured, the
        order runs from its staged release or not at all -- never silently
        from the active link. A git timeout is transient; a missing or
        mismatched stage is not.
        """
        candidate = self._staged_release_path(order)
        if candidate is None:
            return None
        revision = str(order["source_revision"]).strip()
        label = f"staged release {candidate} for revision {revision[:12]}"
        if not (candidate / "dispatcher" / "mussel_dispatcher").is_dir():
            raise StagedReleaseError(f"{label} is missing", transient=False)
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise StagedReleaseError(
                f"could not verify {label}: {exc}", transient=True
            ) from exc
        actual = result.stdout.strip()
        if result.returncode != 0 or actual != revision:
            raise StagedReleaseError(
                f"{label} is invalid (HEAD={actual or '<none>'}: "
                f"{result.stderr.strip()[:200]})",
                transient=False,
            )
        return candidate

    def _stage_attempt_state(self) -> dict[str, int]:
        return self.__dict__.setdefault("_stage_missing_attempts", {})

    def _pin_order_to_staged_release(self, order: dict[str, Any]) -> dict[str, Any]:
        """Run a driver from its immutable staged release, not the active link.

        ``remote_repo_dir`` is normalized to the target's active release link,
        which the release agent re-points on activation. A queued or running
        driver reading code through that link would fail its revision check
        (pending) or load a different release mid-run (running). The staged
        worktree for the order's ``source_revision`` never changes, so paths
        under the active link are rebased onto it. Orders and targets without
        release staging keep using ``remote_repo_dir``.
        """
        candidate = self._resolve_staged_release(order)
        if candidate is None:
            return order
        active_repo = str(order.get("remote_repo_dir") or "").rstrip("/")
        pinned = dict(order)
        pinned["remote_repo_dir"] = str(candidate)
        if active_repo:
            for key in ("nextflow_config", "nextflow_params_file"):
                value = str(pinned.get(key) or "")
                if value.startswith(active_repo + "/"):
                    pinned[key] = str(candidate) + value[len(active_repo):]
        return pinned

    def _finalize_success(
        self,
        order: dict[str, Any],
        *,
        lease_name: str | None = None,
    ):
        batch_id = order["batch_id"]
        lease_name = lease_name or getattr(self, "_finalization_lease_names", {}).get(
            batch_id, REMOTE_FINALIZATION_LEASE_NAME
        )
        created_at = str(order.get("created_at") or "")
        try:
            run_started_at = datetime.fromisoformat(created_at).timestamp()
        except (TypeError, ValueError):
            run_started_at = 0.0

        stop_heartbeat = threading.Event()
        ownership_lost = threading.Event()
        ttl_seconds = max(
            1, int(self.target.finalization_lease_ttl_seconds or 300)
        )
        safety_margin = min(60, max(1, ttl_seconds // 5))
        heartbeat_state = {
            "last_success_at": utc_now(),
            "last_success_monotonic": time.monotonic(),
            "lease_name": lease_name,
            "ownership_lost": False,
            "last_error": "",
        }
        if not hasattr(self, "_finalization_heartbeat_health"):
            self._finalization_heartbeat_health = {}
        self._finalization_heartbeat_health[batch_id] = heartbeat_state

        def lose_ownership(reason: str) -> None:
            if ownership_lost.is_set():
                return
            ownership_lost.set()
            heartbeat_state["ownership_lost"] = True
            heartbeat_state["last_error"] = reason
            heartbeat_state["lost_at"] = utc_now()
            log.error(
                "Batch %s: finalization lease %s was lost; "
                "no further hooks will be launched (%s)",
                batch_id,
                lease_name,
                reason,
            )

        def heartbeat() -> None:
            interval = max(
                1,
                min(60, int(self.target.finalization_poll_interval_seconds or 60)),
            )
            while True:
                deadline = (
                    float(heartbeat_state["last_success_monotonic"])
                    + ttl_seconds
                    - safety_margin
                )
                wait_seconds = min(
                    float(interval), max(0.1, deadline - time.monotonic())
                )
                if stop_heartbeat.wait(wait_seconds):
                    return
                try:
                    renewed = self.state.heartbeat_remote_finalization(
                        batch_id,
                        owner=self._finalization_owner,
                        lease_ttl_seconds=ttl_seconds,
                        lease_name=lease_name,
                    )
                    if not renewed:
                        lose_ownership("lease renewal rejected")
                        return
                    heartbeat_state["last_success_at"] = utc_now()
                    heartbeat_state["last_success_monotonic"] = time.monotonic()
                    heartbeat_state["last_error"] = ""
                except Exception as exc:
                    heartbeat_state["last_error"] = str(exc)
                    log.warning(
                        "Batch %s: finalization lease heartbeat failed: %s",
                        batch_id,
                        exc,
                    )
                    safety_deadline = (
                        float(heartbeat_state["last_success_monotonic"])
                        + ttl_seconds
                        - safety_margin
                    )
                    if time.monotonic() >= safety_deadline:
                        lose_ownership(
                            "heartbeat unavailable through lease safety deadline"
                        )
                        return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"remote-finalize-heartbeat-{batch_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            def hook_executor(cmd, **kwargs):
                if self.target.hook_use_slurm:
                    return self._run_hook_slurm(
                        order,
                        cmd,
                        ownership_lost=ownership_lost,
                        **kwargs,
                    )
                try:
                    owns_lease = self.state.owns_finalization_lease(
                        batch_id,
                        owner=self._finalization_owner,
                        lease_name=lease_name,
                    )
                except Exception as exc:
                    heartbeat_state["last_error"] = str(exc)
                    owns_lease = False
                if ownership_lost.is_set() or not owns_lease:
                    lose_ownership("pre-submission ownership check failed")
                    return subprocess.CompletedProcess(
                        cmd,
                        78,
                        "",
                        "[mussel-hook-preflight] category=TRANSIENT_SERVICE "
                        "detail=finalization-lease-lost",
                    )
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=kwargs["cwd"],
                    env=kwargs["env"],
                )

            return BatchFinalizer(
                self.cfg,
                batch_id,
                self.state,
                target=self.target.name,
                runtime_revision=str(
                    order.get("source_revision") or getattr(self, "_agent_revision", "") or ""
                ),
            ).finalize_success(
                FinalizationContext(
                    batch_csv=str(order.get("remote_batch_csv") or os.path.join(order["remote_run_dir"], "batch.csv")),
                    outdir=str(order.get("remote_outdir") or self.target.remote_outdir or self.cfg.outdir),
                    repo_dir=self._hook_repo_dir(order),
                    run_started_at=run_started_at,
                    path_replacements=self._remote_hook_path_replacements(order),
                ),
                hook_executor=hook_executor,
            )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=5)
            reconcile_health = getattr(self, "_finalization_reconcile_health", {})
            reconcile_health["last_successful_heartbeat_at"] = heartbeat_state.get(
                "last_success_at"
            )
            self._finalization_reconcile_health = reconcile_health
            self._finalization_heartbeat_health.pop(batch_id, None)

    def _run_hook_slurm(self, order: dict[str, Any], cmd: list[str], *, cwd: str,
                        env: dict[str, str], hook_index: int,
                        ownership_lost: threading.Event | None = None) -> subprocess.CompletedProcess:
        """Run one remote post-batch hook in a dedicated Slurm allocation."""
        run_dir = Path(str(order.get("remote_run_dir") or self.target.remote_scratch_root))
        log_dir = run_dir / "hooks"
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = log_dir / f"hook_{hook_index}"
        rows = self.state.get_batch_hooks([order["batch_id"]]).get(order["batch_id"], [])
        existing = next(
            (row for row in rows if int(row.get("hook_index", -1)) == hook_index),
            {},
        )
        existing_job_id = str(existing.get("executor_job_id") or "")
        hook_attempt_number = int(existing.get("active_attempt_number") or 0) or None
        if existing_job_id.isdigit():
            return self._poll_hook_slurm(
                order, cmd, hook_index=hook_index, job_id=existing_job_id, stem=stem, attempt_number=hook_attempt_number
            )
        owner = getattr(self, "_finalization_owner", "remote-agent")
        lease_name = getattr(self, "_finalization_lease_names", {}).get(
            order["batch_id"], REMOTE_FINALIZATION_LEASE_NAME
        )
        try:
            owns_lease = self.state.owns_finalization_lease(
                order["batch_id"], owner=owner, lease_name=lease_name
            )
        except Exception as exc:
            log.warning(
                "Batch %s: lease pre-submission check failed: %s",
                order["batch_id"], exc,
            )
            owns_lease = False
        if (ownership_lost is not None and ownership_lost.is_set()) or not owns_lease:
            if ownership_lost is not None:
                ownership_lost.set()
            return subprocess.CompletedProcess(
                cmd,
                78,
                "",
                "[mussel-hook-preflight] category=TRANSIENT_SERVICE "
                "detail=finalization-lease-lost",
            )
        # Do not rely on the login shell's PYTHONPATH. The remote source tree
        # contains the package under ``dispatcher/`` and Slurm starts with a
        # clean environment; omitting this export causes opaque 2-second
        # ModuleNotFoundError hook failures.
        package_root = str(Path(cwd) / "dispatcher")
        hook_python = ""
        if cmd and Path(str(cmd[0])).name.startswith("python"):
            hook_python = str(cmd[0])
        preflight = ""
        if hook_python and any("mussel_dispatcher" in str(part) for part in cmd):
            module_name = ""
            for position, part in enumerate(cmd[:-1]):
                if str(part) == "-m":
                    module_name = str(cmd[position + 1])
                    break
            import_check = (
                f"import importlib; importlib.import_module({module_name!r})"
                if module_name else "import mussel_dispatcher"
            )
            preflight = (
                "echo '[mussel-hook] preflight node='$(hostname)\n"
                "test -r %s || { echo '[mussel-hook-preflight] category=NODE_INFRA detail=batch-csv-unreadable'; exit 78; }\n"
                "test -d %s || { echo '[mussel-hook-preflight] category=NODE_INFRA detail=dispatcher-package-missing'; exit 78; }\n"
            ) % (
                shlex.quote(str(order.get("remote_batch_csv") or run_dir / "batch.csv")),
                shlex.quote(str(Path(cwd) / "dispatcher" / "mussel_dispatcher")),
            )
            # A launched hook checks where its module comes from in the process that runs it.
            if not is_launched(cmd):
                preflight += (
                    "if ! %s -c %s; then echo '[mussel-hook-preflight] category=RELEASE_OR_CONFIG "
                    "detail=dispatcher-import-failed'; exit 78; fi\n"
                ) % (shlex.quote(hook_python), shlex.quote(import_check))
        script = (
            "#!/bin/bash\nset -euo pipefail\n"
            "export PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH}\n"
            "cd %s\n%sexec %s\n"
        ) % (shlex.quote(package_root), shlex.quote(cwd), preflight, shlex.join(cmd))
        job_name = f"mussel-hook-{order['batch_id'][-8:]}-{hook_index}"
        # A previous attempt may have submitted this hook and then lost the
        # job id (sbatch timeout, executor write failure, agent crash). Adopt
        # that job instead of running the hook (e.g. a WDS append) twice: a
        # live one is polled; a finished one is adopted only if it succeeded.
        try:
            adopted = self._find_orphaned_hook_job(order, job_name)
        except RuntimeError as exc:
            return subprocess.CompletedProcess(
                cmd, 78, "",
                "[mussel-hook-preflight] category=TRANSIENT_SERVICE "
                f"detail=slurm-lookup-unavailable {exc}",
            )
        if adopted:
            log.warning(
                "Batch %s: adopting Slurm hook job %s (%s) instead of resubmitting",
                order["batch_id"], adopted, job_name,
            )
            self._record_hook_executor_submitted(
                order["batch_id"], hook_index, adopted, hook_attempt_number
            )
            return self._poll_hook_slurm(
                order, cmd, hook_index=hook_index, job_id=adopted, stem=stem,
                attempt_number=hook_attempt_number,
            )
        args = [
            "sbatch", "--parsable", "--export=ALL",
            "--job-name", job_name,
            "--partition", self.target.hook_partition,
            "--cpus-per-task", str(self.target.hook_cpus),
            "--mem", self.target.hook_mem,
            "--time", self.target.hook_time,
            "--output", str(stem) + ".out",
            "--error", str(stem) + ".err",
        ]
        if self.target.hook_account:
            args += ["--account", self.target.hook_account]
        if self.target.hook_qos:
            args += ["--qos", self.target.hook_qos]
        excludes = [str(self.target.hook_exclude).strip()] if self.target.hook_exclude else []
        if (
            str(existing.get("failure_category") or "") == "NODE_INFRA"
            and str(existing.get("execution_node") or "").strip()
        ):
            excludes.append(str(existing["execution_node"]).strip())
        if excludes:
            args += ["--exclude", ",".join(item for item in excludes if item)]
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=f".hook_{hook_index}_", suffix=".sbatch",
            dir=log_dir, delete=False,
        ) as script_file:
            script_file.write(script)
            script_path = script_file.name
        try:
            submitted = subprocess.run(
                args + [script_path], text=True, env=env,
                capture_output=True, check=False, timeout=60,
            )
        except subprocess.TimeoutExpired:
            # The submission may still have gone through; the retry adopts
            # the job by name rather than submitting a duplicate.
            return subprocess.CompletedProcess(
                args, 78, "",
                "[mussel-hook-preflight] category=TRANSIENT_SERVICE "
                f"detail=sbatch-timeout job_name={job_name}",
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
        if submitted.returncode != 0:
            return submitted
        lines = submitted.stdout.strip().splitlines()
        job_id = lines[-1].split(";", 1)[0].strip() if lines else ""
        if not job_id.isdigit():
            return subprocess.CompletedProcess(args, 1, submitted.stdout,
                                               "sbatch returned no job id: " + submitted.stderr)
        self._record_hook_executor_submitted(
            order["batch_id"], hook_index, job_id, hook_attempt_number
        )
        return self._poll_hook_slurm(
            order, cmd, hook_index=hook_index, job_id=job_id, stem=stem, attempt_number=hook_attempt_number
        )

    def _record_hook_executor_submitted(
        self, batch_id: str, hook_index: int, job_id: str, attempt_number: int | None
    ) -> None:
        """Persist a submitted hook job id; never abandon a live job on failure.

        If this write fails the job is still running. Raising would mark the
        hook FAILED and let the retry submit a second copy, so keep polling the
        job we know about; ``_poll_hook_slurm`` records its final state.
        """
        executor_kwargs: dict[str, Any] = {"job_id": job_id, "state": "SUBMITTED"}
        if attempt_number is not None:
            executor_kwargs["expected_attempt_number"] = attempt_number
        try:
            self.state.set_batch_hook_executor(batch_id, hook_index, **executor_kwargs)
        except Exception as exc:
            log.error(
                "Batch %s: could not record hook %d Slurm job %s (%s); polling it anyway",
                batch_id, hook_index, job_id, exc,
            )

    @staticmethod
    def _find_orphaned_hook_job(order: dict[str, Any], job_name: str) -> str | None:
        """Return a hook job of this name to adopt instead of resubmitting.

        A queued/running job is always adopted. Otherwise the newest job of
        this name since the batch was created is adopted if it is still
        active in accounting or COMPLETED with exit 0 (its success is then
        recorded by polling); a failed one is left alone so the hook reruns.
        Raises RuntimeError when Slurm cannot answer.
        """
        live = _squeue_job_ids_by_name(job_name)
        if live:
            if len(live) > 1:
                log.warning(
                    "Batch %s: %d live hook jobs named %s; adopting newest %s, "
                    "leaving %s running",
                    order["batch_id"], len(live), job_name, live[-1], ", ".join(live[:-1]),
                )
            return live[-1]
        finished = _sacct_jobs_by_name(job_name, _sacct_since(order.get("created_at")))
        if not finished:
            return None
        newest = finished[-1]
        if newest["state"] not in _SLURM_TERMINAL_STATES:
            return newest["job_id"]
        if newest["state"] == "COMPLETED" and newest["exit_code"] in (None, 0):
            return newest["job_id"]
        return None

    def _poll_hook_slurm(
        self,
        order: dict[str, Any],
        cmd: list[str],
        *,
        hook_index: int,
        job_id: str,
        attempt_number: int | None = None,
        stem: Path,
    ) -> subprocess.CompletedProcess:
        """Poll a persisted hook allocation, including after agent restart."""
        state = "UNKNOWN"
        exit_code = None
        node = ""
        deadline = time.time() + 24 * 3600
        while time.time() < deadline:
            current_rows = self.state.get_batch_hooks([order["batch_id"]]).get(order["batch_id"], [])
            current = next((row for row in current_rows if int(row.get("hook_index", -1)) == hook_index), None)
            current_state = str(current.get("status") or "") if current else ""
            current_job = str(current.get("executor_job_id") or "") if current else ""
            if current and current_job == job_id and current_state != "RUNNING":
                current_code = int(current.get("exit_code") or (0 if current_state in {"SUCCEEDED", "WAIVED"} else 1))
                return subprocess.CompletedProcess(cmd, current_code, "", str(current.get("error_msg") or ""))
            poll_sleep = max(1, int(self.target.finalization_poll_interval_seconds or 60))
            try:
                sq = subprocess.run(
                    ["squeue", "-h", "-j", job_id, "-o", "%T"],
                    text=True, capture_output=True, check=False, timeout=30,
                )
                queued = sq.returncode == 0 and bool(sq.stdout.strip())
            except subprocess.TimeoutExpired:
                queued = False
            if queued:
                time.sleep(poll_sleep)
                continue
            # squeue exits non-zero both when slurmctld is unreachable and
            # when a finished job was purged ("Invalid job id specified"), so
            # its failure proves nothing either way: ask sacct, which only
            # ends polling on a terminal state.
            try:
                sa = subprocess.run(
                    ["sacct", "-n", "-X", "-j", job_id,
                     "--format=State,ExitCode,NodeList", "--parsable2"],
                    text=True, capture_output=True, check=False, timeout=30,
                )
            except subprocess.TimeoutExpired:
                time.sleep(poll_sleep)
                continue
            line = next((x.strip() for x in sa.stdout.splitlines() if x.strip()), "")
            if sa.returncode == 0 and line:
                fields = line.split("|")
                sacct_state = fields[0].split()[0].rstrip("+") if fields[0].split() else ""
                # Only a terminal accounting state is final; sacct also
                # reports PENDING/RUNNING jobs that squeue momentarily missed.
                if sacct_state in _SLURM_TERMINAL_STATES:
                    state = sacct_state
                    exit_code = None
                    if len(fields) > 1:
                        try:
                            exit_code = int(fields[1].split(":", 1)[0])
                        except ValueError:
                            pass
                    if len(fields) > 2 and fields[2].strip():
                        node = fields[2].strip()
                    else:
                        node = ""
                    break
            time.sleep(poll_sleep)
        else:
            state = "TIMEOUT"
            exit_code = 124

        stdout = stem.with_suffix(".out").read_text(errors="replace") if stem.with_suffix(".out").exists() else ""
        stderr = stem.with_suffix(".err").read_text(errors="replace") if stem.with_suffix(".err").exists() else ""
        if state != "COMPLETED" and not stderr:
            stderr = f"Slurm hook state={state} job_id={job_id}\n"
        if node:
            stderr = (stderr.rstrip() + f"\nSlurm hook node={node}\n").strip() + "\n"
        executor_kwargs = {"job_id": job_id, "state": state}
        if attempt_number is not None:
            executor_kwargs["expected_attempt_number"] = attempt_number
        self.state.set_batch_hook_executor(order["batch_id"], hook_index, **executor_kwargs)
        return subprocess.CompletedProcess(cmd, exit_code or (0 if state == "COMPLETED" else 1), stdout, stderr)

    @staticmethod
    def _remote_hook_path_replacements(order: dict[str, Any]) -> dict[str, str]:
        run_dir = str(order.get("remote_run_dir") or "")
        replacements: dict[str, str] = {}
        for source, replacement in (order.get("hook_path_replacements") or {}).items():
            if not source or not replacement:
                continue
            template = str(replacement)
            try:
                resolved = template.format(remote_run_dir=run_dir)
            except (KeyError, ValueError):
                resolved = template
            # Tolerate payloads written by older code that escaped the
            # placeholder as ``{{remote_run_dir}}``.
            replacements[str(source)] = resolved.replace("{remote_run_dir}", run_dir)
        return replacements

    def _should_replay_claimed_order(
        self,
        order: dict[str, Any],
        status: dict[str, Any],
        agent: dict[str, Any] | None,
    ) -> bool:
        if (status.get("driver_job_id") or "").strip():
            return False
        order_status = str(order.get("status") or status.get("status") or "").upper()
        if order_status not in {"CLAIMED", "STARTING"}:
            return False
        claimed_at = order.get("claimed_at") or status.get("claimed_at") or order.get("heartbeat_at")
        if not claimed_at:
            return False
        try:
            claimed_ts = datetime.fromisoformat(str(claimed_at)).timestamp()
        except Exception:
            return False
        age_s = max(0.0, time.time() - claimed_ts)
        if age_s < _CLAIM_REPLAY_DELAY_SECONDS:
            return False
        # The target heartbeat is written by this very process, so it is
        # always fresh here and cannot reveal an order orphaned by a previous
        # agent incarnation (crash/restart between claim and the RUNNING
        # write). This process claims and submits synchronously, so a claim
        # older than our start with no driver job belongs to a dead agent.
        started_at = getattr(self, "_started_at", None)
        if started_at is not None and claimed_ts < started_at:
            print(
                f"[mussel-remote-agent] replaying batch {order.get('batch_id')} orphaned "
                f"by a previous agent (claimed_age_s={age_s:.0f})"
            )
            return True
        heartbeat = str((agent or {}).get("heartbeat_at") or "")
        if heartbeat:
            try:
                hb_age_s = max(0.0, time.time() - datetime.fromisoformat(heartbeat).timestamp())
            except Exception:
                hb_age_s = self.target.agent_heartbeat_timeout_seconds + 1
        else:
            hb_age_s = self.target.agent_heartbeat_timeout_seconds + 1
        if hb_age_s <= self.target.agent_heartbeat_timeout_seconds:
            return False
        print(
            f"[mussel-remote-agent] replaying orphaned claimed batch {order.get('batch_id')} "
            f"(claimed_age_s={age_s:.0f}, agent_heartbeat_age_s={hb_age_s:.0f})"
        )
        return True

    def _build_state_store(self) -> StateStore:
        if (self.cfg.state_backend or "").strip().lower() != "postgres":
            raise RuntimeError("remote agent requires state_backend=postgres")
        self._load_db_env()
        return StateStore.from_config(self.cfg)

    def _cleanup_remote_work_dir(self, order: dict[str, Any]) -> None:
        # Compatibility name retained for callers and older tests.  The shared
        # cleaner now removes all disposable execution artifacts.
        self._cleanup_remote_execution(order)

    def _load_db_env(self) -> None:
        for env_path in (self.cfg.secrets_env_file, self.target.secrets_env_file):
            self._source_env_file(env_path)

    @staticmethod
    def _source_env_file(path: str) -> None:
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            return

    def _normalize_order(self, order: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(order)
        batch_id = str(normalized.get("batch_id") or "")
        if not normalized.get("nextflow_run_name") and batch_id:
            # Older claimed orders may predate persistence of the NF run name.
            # Replays must still be recoverable instead of crashing the agent.
            normalized["nextflow_run_name"] = "r" + batch_id.rsplit("_", 1)[-1]
        remote_run_dir = normalized["remote_run_dir"]
        normalized.setdefault("remote_work_dir", os.path.join(remote_run_dir, "work"))
        normalized.setdefault("remote_batch_csv", os.path.join(remote_run_dir, "batch.csv"))
        normalized.setdefault("remote_driver_script", os.path.join(remote_run_dir, "driver.sbatch"))
        normalized.setdefault("remote_driver_log", os.path.join(remote_run_dir, "driver.out"))
        normalized.setdefault("remote_driver_err", os.path.join(remote_run_dir, "driver.err"))
        normalized.setdefault("remote_nf_log", os.path.join(remote_run_dir, "nextflow.log"))
        normalized.setdefault("remote_trace", os.path.join(remote_run_dir, "trace.tsv"))
        default_overrides = {
            "remote_outdir": self.target.remote_outdir or self.cfg.outdir,
            "remote_repo_dir": self.target.remote_repo_dir or self.cfg.repo_dir,
            "nextflow_profiles": self.target.nextflow_profiles or self.cfg.nextflow_profiles,
            "nextflow_config": self.target.nextflow_config,
            "nextflow_params_file": (
                self.target.nextflow_params_file or self.cfg.nextflow_params_file
            ),
            "nextflow_version": self.target.nextflow_version or self.cfg.nextflow_version,
            "tower_endpoint": _remote_tower_endpoint(self.cfg, self.target),
            "hook_path_replacements": dict(self.target.remote_hook_path_replacements or {}),
        }
        for key, value in default_overrides.items():
            if key == "nextflow_params_file" and int(normalized.get("schema_version") or 1) >= 2:
                continue
            if key == "hook_path_replacements" and not normalized.get(key):
                normalized[key] = value
            elif normalized.get(key) in (None, ""):
                normalized[key] = value
        # Legacy orders can point at the pre-release source checkout.  That
        # checkout may have a different dispatcher schema than the agent that
        # is replaying the hooks (for example, an older Config parser).  Hook
        # replays must use the active target release; the persisted batch CSV,
        # outputs, and remote paths remain batch-specific above.
        active_repo = self.target.remote_repo_dir or self.cfg.repo_dir
        if active_repo:
            normalized["remote_repo_dir"] = active_repo
        return normalized

    def _driver_slurm_state(self, job_id: str) -> dict[str, Any]:
        self._driver_slurm_queries += 1
        try:
            sq = subprocess.run(
                ["squeue", "-h", "-j", job_id, "-o", "%T"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            states = [ln.strip() for ln in sq.stdout.splitlines() if ln.strip()]
            if states:
                return {"state": states[0], "source": "squeue"}
        except Exception as exc:
            return {"state": "UNKNOWN", "source": "squeue", "error": str(exc)}
        try:
            sa = subprocess.run(
                ["sacct", "-n", "-X", "-j", job_id, "--format=State,ExitCode", "--parsable2"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            for line in sa.stdout.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if not parts or not parts[0]:
                    continue
                exit_code = None
                if len(parts) > 1 and ":" in parts[1]:
                    try:
                        exit_code = int(parts[1].split(":", 1)[0])
                    except ValueError:
                        pass
                return {"state": parts[0].split()[0], "exit_code": exit_code, "source": "sacct"}
        except Exception as exc:
            return {"state": "UNKNOWN", "source": "sacct", "error": str(exc)}
        return {"state": "UNKNOWN", "source": "sacct"}

    def _collect_target_slurm_summary(self) -> dict[str, Any]:
        summary = {
            "target": self.target.name,
            "updated_at": utc_now(),
            "running": 0,
            "pending": 0,
            "completing": 0,
            "total": 0,
            "nodes": [],
            "driver_running": 0,
            "driver_pending": 0,
            "driver_completing": 0,
            "driver_total": 0,
            "error": None,
        }
        scratch_root = str(self.target.remote_scratch_root or "").rstrip("/")
        nodes: set[str] = set()
        try:
            out = subprocess.check_output(
                ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%R|%N|%Z|%M"],
                timeout=20,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            summary["error"] = "squeue not found"
            return summary
        except subprocess.TimeoutExpired:
            summary["error"] = "squeue timed out"
            return summary
        except Exception as exc:
            summary["error"] = str(exc)[:160]
            return summary

        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 7:
                continue
            _job_id, job_name, state, _reason, node, workdir, _elapsed = parts[:7]
            is_driver = job_name.startswith("mussel-driver-")
            is_task = job_name.startswith("nf-")
            in_target_tree = bool(scratch_root and workdir and workdir.startswith(scratch_root))
            if not is_driver and not (is_task and in_target_tree):
                continue
            summary["total"] += 1
            if is_driver:
                summary["driver_total"] += 1
            if state == "RUNNING":
                summary["running"] += 1
                if is_driver:
                    summary["driver_running"] += 1
                if node and node != "N/A":
                    nodes.add(node)
            elif state == "PENDING":
                summary["pending"] += 1
                if is_driver:
                    summary["driver_pending"] += 1
            elif state == "COMPLETING":
                summary["completing"] += 1
                if is_driver:
                    summary["driver_completing"] += 1

        summary["nodes"] = sorted(nodes)
        return summary

def _load_target(config_path: str, target_name: str | None) -> tuple[Config, RemoteTargetConfig]:
    cfg = Config.load(config_path)
    target = cfg.get_dispatch_target(target_name)
    if not target:
        raise SystemExit(f"Remote target not found: {target_name or cfg.default_dispatch_target}")
    return cfg, target


def agent_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mussel remote Slurm agent")
    parser.add_argument("config")
    parser.add_argument("--target", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg, target = _load_target(args.config, args.target or None)
    return RemoteAgent(cfg, target, once=args.once, dry_run=args.dry_run).run()


if __name__ == "__main__":
    raise SystemExit(agent_main())
