"""Database-coordinated release publication and status commands.

Release intent is written to the same PostgreSQL database used for dispatch.
Hosts stage and report releases through :mod:`release_agent`; no host-to-host
SSH or dispatcher-side remote shell is required.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import Config
from .contract import build_contract, git_revision
from .release_pin import on_release_branch
from .state import StateStore


log = logging.getLogger("mussel-dispatcher.release")


class ReleaseController:
    """Reconcile a clean local checkout into the shared deployment release."""

    def __init__(self, cfg: Config, state: StateStore):
        self.cfg = cfg
        self.state = state
        self.deployment_id = cfg.release_deployment_id or "default"
        try:
            self.loaded_revision = git_revision(str(Path(__file__).resolve().parents[2]))
        except Exception:
            self.loaded_revision = ""

    def _report(self, status: str, message: str, revision: str, *, error: str | None = None) -> None:
        self.state.report_release_controller(
            self.deployment_id,
            status=status,
            message=message,
            revision=revision,
            error=error,
        )

    def _tracked_dirty(self) -> str:
        result = subprocess.run(
            ["git", "-C", self.cfg.repo_dir, "status", "--porcelain", "--untracked-files=no"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        return result.stdout.strip()

    # A revision refused by the release-branch pin is not re-fetched on every sync
    # cycle (e.g. every 15 s): the refusal is reused for this long, or until HEAD moves.
    _PIN_RECHECK_SECONDS = 300

    def _pin_refusal(self, revision: str) -> str:
        """Why ``revision`` may not be published under ``release_branch`` ('' if it may)."""
        cached = getattr(self, "_pin_block", None)
        if cached and cached[0] == revision and time.monotonic() < cached[1]:
            return cached[2]
        pinned, why = on_release_branch(
            self.cfg.repo_dir, revision,
            self.cfg.release_origin_remote or "origin", self.cfg.release_branch,
        )
        if pinned:
            self._pin_block = None
            return ""
        message = f"Not publishing: {why}"
        self._pin_block = (revision, time.monotonic() + self._PIN_RECHECK_SECONDS, message)
        return message

    def _available_from_origin(self, revision: str) -> bool:
        remote = self.cfg.release_origin_remote or "origin"
        subprocess.run(
            ["git", "-C", self.cfg.repo_dir, "fetch", "--quiet", "--prune", remote],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        result = subprocess.run(
            ["git", "-C", self.cfg.repo_dir, "branch", "-r", "--contains", revision],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        prefix = remote + "/"
        return any(line.strip().lstrip("* ").startswith(prefix) for line in result.stdout.splitlines())

    def _required_targets(self) -> list:
        return [
            target for target in self.cfg.dispatch_targets
            if str(target.backend or "remote_agent") == "remote_agent"
        ]

    @staticmethod
    def _fresh(heartbeat: object, timeout_seconds: int) -> bool:
        try:
            timestamp = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - timestamp).total_seconds() <= timeout_seconds
        except (TypeError, ValueError, OSError):
            return False

    def _target_staged(self, target, desired: str, nodes: list[dict]) -> tuple[bool, str]:
        candidates = [node for node in nodes if str(node.get("target") or "") == target.name]
        if not candidates:
            return False, f"{target.name}: waiting for release agent registration"
        for node in candidates:
            if str(node.get("status") or "").upper() == "ERROR":
                return False, f"{target.name}: {node.get('error') or 'release agent error'}"
            if not self._fresh(node.get("heartbeat_at"), target.agent_heartbeat_timeout_seconds):
                return False, f"{target.name}: release agent heartbeat stale"
            if str(node.get("staged_revision") or "") != desired:
                return False, f"{target.name}: staging {desired[:12]}"
        return True, ""

    def _target_active(self, target, desired: str, nodes: list[dict]) -> tuple[bool, str]:
        staged, reason = self._target_staged(target, desired, nodes)
        if not staged:
            return False, reason
        candidates = [node for node in nodes if str(node.get("target") or "") == target.name]
        if any(str(node.get("active_revision") or "") != desired for node in candidates):
            return False, f"{target.name}: activating {desired[:12]}"
        agent = self.state.read_remote_agent(target.name) or {}
        if not self._fresh(agent.get("heartbeat_at"), target.agent_heartbeat_timeout_seconds):
            return False, f"{target.name}: waiting for restarted worker heartbeat"
        loaded = str(agent.get("agent_revision") or "")
        if loaded != desired:
            return False, (
                f"{target.name}: active {desired[:12]}, worker loaded "
                f"{loaded[:12] if loaded else '<unknown>'}"
            )
        return True, ""

    def run_once(self) -> dict:
        revision = ""
        try:
            revision = git_revision(self.cfg.repo_dir)
            dirty = self._tracked_dirty()
            if dirty:
                message = "Tracked changes in the production checkout; remote dispatch is paused"
                self._report("BLOCKED_LOCAL", message, revision)
                return {"status": "BLOCKED_LOCAL", "message": message, "revision": revision}
            if not self.loaded_revision:
                message = "Cannot verify the revision loaded by the local dispatcher"
                self._report("BLOCKED_LOCAL", message, revision)
                return {"status": "BLOCKED_LOCAL", "message": message, "revision": revision}
            if self.loaded_revision != revision:
                message = (
                    f"Local checkout is {revision[:12]} but the dispatcher loaded "
                    f"{self.loaded_revision[:12]}; restart the local dispatcher"
                )
                self._report("BLOCKED_LOCAL", message, revision)
                return {"status": "BLOCKED_LOCAL", "message": message, "revision": revision}

            release = self.state.get_deployment_release(self.deployment_id)
            if not release or str(release.get("desired_revision") or "") != revision:
                if self.cfg.release_branch:
                    message = self._pin_refusal(revision)
                    if message:
                        self._report("BLOCKED_LOCAL", message, revision)
                        return {"status": "BLOCKED_LOCAL", "message": message, "revision": revision}
                elif self.cfg.release_require_origin and not self._available_from_origin(revision):
                    message = f"Revision {revision[:12]} is not available from {self.cfg.release_origin_remote}"
                    self._report("BLOCKED_LOCAL", message, revision)
                    return {"status": "BLOCKED_LOCAL", "message": message, "revision": revision}
                contract = build_contract(self.cfg)
                self.state.publish_deployment_release(
                    self.deployment_id,
                    desired_revision=revision,
                    desired_release_manifest_sha256=str(contract["release_manifest_sha256"]),
                    desired_semantic_contract_sha256=str(contract["semantic_contract_sha256"]),
                    requested_by=f"auto:{socket.gethostname()}:{os.getpid()}",
                )
                message = f"Published {revision[:12]}; waiting for remote staging"
                return {"status": "SYNCING", "message": message, "revision": revision}

            desired = str(release.get("desired_revision") or "")
            phase = str(release.get("phase") or "STAGED").upper()
            nodes = self.state.list_deployment_nodes(self.deployment_id)
            targets = self._required_targets()
            if not targets:
                self.state.set_deployment_phase(
                    self.deployment_id,
                    phase="ACTIVE",
                    expected_phases=(phase,),
                    expected_revision=desired,
                )
                self._report("ACTIVE", "No managed remote targets", revision)
                return {"status": "ACTIVE", "message": "No managed remote targets", "revision": revision}

            if phase == "STAGED":
                checks = [self._target_staged(target, desired, nodes) for target in targets]
                failure = next((reason for ready, reason in checks if not ready), "")
                if failure:
                    status = "ERROR" if "error" in failure.lower() else "SYNCING"
                    self._report(status, failure, revision, error=failure if status == "ERROR" else None)
                    return {"status": status, "message": failure, "revision": revision}
                self.state.set_deployment_phase(
                    self.deployment_id,
                    phase="ACTIVATING",
                    expected_phases=("STAGED",),
                    expected_revision=desired,
                )
                message = f"All targets staged {desired[:12]}; activating"
                self._report("SYNCING", message, revision)
                return {"status": "SYNCING", "message": message, "revision": revision}

            checks = [self._target_active(target, desired, nodes) for target in targets]
            failure = next((reason for ready, reason in checks if not ready), "")
            if failure:
                if phase == "ACTIVE":
                    self.state.set_deployment_phase(
                        self.deployment_id,
                        phase="ACTIVATING",
                        expected_phases=("ACTIVE",),
                        expected_revision=desired,
                    )
                status = "ERROR" if "error" in failure.lower() else "SYNCING"
                self._report(status, failure, revision, error=failure if status == "ERROR" else None)
                return {"status": status, "message": failure, "revision": revision}

            self.state.set_deployment_phase(
                self.deployment_id,
                phase="ACTIVE",
                expected_phases=("ACTIVATING", "ACTIVE"),
                expected_revision=desired,
            )
            message = f"All targets active on {desired[:12]}"
            self._report("ACTIVE", message, revision)
            return {"status": "ACTIVE", "message": message, "revision": revision}
        except Exception as exc:
            message = f"Release synchronization failed: {exc}"
            log.exception(message)
            self._report("ERROR", message, revision, error=message)
            return {"status": "ERROR", "message": message, "revision": revision}


def _ensure_clean_release(cfg: Config) -> None:
    result = subprocess.run(
        ["git", "-C", cfg.repo_dir, "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    if result.stdout.strip():
        raise RuntimeError(
            f"Refusing to publish a release from a tracked-dirty checkout: {cfg.repo_dir}"
        )


def publish(cfg: Config, deployment_id: str, requested_by: str = "") -> dict:
    _ensure_clean_release(cfg)
    contract = build_contract(cfg)
    # A manual publish is held to the same pin as the release controller.
    if cfg.release_branch:
        pinned, why = on_release_branch(cfg.repo_dir, contract["source_revision"],
                                        cfg.release_origin_remote or "origin", cfg.release_branch)
        if not pinned:
            raise RuntimeError(f"Refusing to publish: {why}; deploy by updating {cfg.release_branch}")
    state = StateStore.from_config(cfg)
    state.publish_deployment_release(
        deployment_id,
        desired_revision=contract["source_revision"],
        desired_release_manifest_sha256=contract["release_manifest_sha256"],
        desired_semantic_contract_sha256=contract["semantic_contract_sha256"],
        requested_by=requested_by or f"{socket.gethostname()}:{os.getpid()}",
    )
    return state.get_deployment_release(deployment_id) or {}


def status(cfg: Config, deployment_id: str) -> dict:
    state = StateStore.from_config(cfg)
    return {
        "release": state.get_deployment_release(deployment_id),
        "nodes": state.list_deployment_nodes(deployment_id),
    }


def activate(cfg: Config, deployment_id: str) -> dict:
    """Begin activation only after every registered node is staged."""
    state = StateStore.from_config(cfg)
    release = state.get_deployment_release(deployment_id)
    if not release:
        raise RuntimeError(f"No published release for deployment {deployment_id}")
    nodes = state.list_deployment_nodes(deployment_id)
    desired = str(release["desired_revision"])
    not_ready = [
        node["node_id"]
        for node in nodes
        if node.get("status") not in {"READY", "ACTIVE"}
        or str(node.get("staged_revision") or "") != desired
    ]
    if not nodes:
        raise RuntimeError("Cannot activate a release with no registered nodes")
    if not_ready:
        raise RuntimeError("Release is not staged on: " + ", ".join(not_ready))
    if not state.set_deployment_phase(
        deployment_id,
        phase="ACTIVATING",
        expected_phases=("STAGED", "READY"),
        expected_revision=desired,
    ):
        raise RuntimeError("Release phase changed while activation was being requested")
    return state.get_deployment_release(deployment_id) or {}


def migrate(cfg: Config) -> dict:
    """Run the idempotent StateStore schema migration explicitly.

    Strict mode makes DDL lock/statement timeouts fatal and verifies every
    required column afterwards, so a partial migration raises (non-zero exit)
    instead of reporting MIGRATED.
    """
    previous = os.environ.pop("MUSSEL_POSTGRES_SKIP_SCHEMA_INIT", None)
    try:
        StateStore.from_config(cfg, strict_schema=True)
    finally:
        if previous is not None:
            os.environ["MUSSEL_POSTGRES_SKIP_SCHEMA_INIT"] = previous
    return {"status": "MIGRATED", "database": cfg.postgres_database}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--requested-by", default="")
    parser.add_argument("command", choices=("publish", "status", "activate", "migrate"))
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command == "publish":
        result = publish(cfg, args.deployment_id, args.requested_by)
    elif args.command == "activate":
        result = activate(cfg, args.deployment_id)
    elif args.command == "migrate":
        result = migrate(cfg)
    else:
        result = status(cfg, args.deployment_id)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
