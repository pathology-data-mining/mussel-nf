"""Self-updating release agent coordinated through the dispatcher database."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .config import Config, RemoteTargetConfig
from .contract import build_contract, git_revision, release_manifest_hash, tracked_source_manifest
from .state import StateStore


# The release tree this process's code was loaded from (.../<tree>/dispatcher/
# mussel_dispatcher/release_agent.py).  Unlike cfg.repo_dir, which goes through
# mussel-nf.active, it does not move when the link is switched.
_CODE_TREE = Path(__file__).resolve().parents[2]

# Run inside a release tree, with that tree's own mussel_dispatcher on
# PYTHONPATH (argv: config path, target name).  Prints, after _TREE_MARKER, the
# tree's contract identity, the target's restart units and the dispatcher
# modules its post-batch hooks run, after checking that the import really comes
# from the tree and that those hook modules exist there.
_TREE_MARKER = "MUSSEL_RELEASE_TREE "
_TREE_SCRIPT = """
import importlib.util, json, shlex, sys
from pathlib import Path
root = (Path.cwd() / "dispatcher").resolve()
def in_tree(name, path):
    # An editable install's import finder can supply a module the tree lacks
    # from another checkout, so check every module's origin, not just the package.
    if not path or not Path(path).resolve().is_relative_to(root):
        raise SystemExit(f"{name} is not loaded from the release tree: {path}")
import mussel_dispatcher
import mussel_dispatcher.config, mussel_dispatcher.contract
for module in (mussel_dispatcher, mussel_dispatcher.config, mussel_dispatcher.contract):
    in_tree(module.__name__, module.__file__)
from mussel_dispatcher.config import Config
from mussel_dispatcher.contract import build_contract
cfg = Config.load(sys.argv[1])
target = cfg.get_dispatch_target(sys.argv[2])
if target is None:
    raise SystemExit("release tree has no target " + sys.argv[2])
contract = build_contract(cfg, target=target)
hook_modules = set()
for hook in cfg.post_batch_hooks or []:
    try:
        parts = shlex.split(str(hook.get("command") or "")) + [str(a) for a in hook.get("args", [])]
    except ValueError:
        continue
    for index, part in enumerate(parts[:-1]):
        if part == "-m" and parts[index + 1].startswith("mussel_dispatcher."):
            hook_modules.add(parts[index + 1])
specs = {name: importlib.util.find_spec(name) for name in hook_modules}
missing = sorted(name for name, spec in specs.items() if spec is None)
if missing:
    raise SystemExit("missing hook modules: " + ", ".join(missing))
for name, spec in specs.items():
    in_tree(name, spec.origin)
keys = ("source_revision", "release_manifest_sha256",
        "semantic_contract_sha256", "execution_profile_sha256")
out = {key: contract[key] for key in keys}
out["release_restart_units"] = [str(unit) for unit in (target.release_restart_units or [])]
out["hook_modules"] = sorted(hook_modules)
print("MUSSEL_RELEASE_TREE " + json.dumps(out))
"""


class ReleaseAgent:
    """Stage and optionally activate a desired release without SSH.

    The active checkout is never overwritten.  By default this agent stages a
    detached worktree and reports READY.  Activation is enabled only when the
    target supplies ``active_release_link``; the service launcher must already
    be configured to execute through that symlink.
    """

    def __init__(
        self,
        cfg: Config,
        target: RemoteTargetConfig,
        *,
        deployment_id: str,
        once: bool = False,
    ):
        self.cfg = cfg
        self.target = target
        self.deployment_id = deployment_id
        self.once = once
        self.state = StateStore.from_config(cfg)
        self.node_id = os.environ.get("MUSSEL_RELEASE_NODE_ID", "") or (
            f"{target.name}:{socket.gethostname()}"
        )
        self.repo_dir = target.remote_repo_dir or cfg.repo_dir
        stage_root = (
            os.environ.get("MUSSEL_RELEASE_STAGE_ROOT", "").strip()
            or getattr(target, "release_stage_root", "")
            or f"{self.repo_dir}.releases"
        )
        self.stage_root = Path(stage_root)
        self.active_release_link = (
            os.environ.get("MUSSEL_ACTIVE_RELEASE_LINK", "").strip()
            or getattr(target, "active_release_link", "")
        )

    def _report(self, **fields: Any) -> None:
        self.state.write_deployment_node(
            self.deployment_id,
            self.node_id,
            target=self.target.name,
            host=socket.gethostname(),
            heartbeat_at=fields.pop("heartbeat_at", None),
            **fields,
        )

    def _relative_config(self) -> str:
        return os.path.relpath(self.cfg.config_path, self.cfg.repo_dir)

    def _tree_contract(self, tree: Path) -> dict[str, Any]:
        """Describe a release tree using that tree's own dispatcher code.

        The running agent's code may predate the tree's rules: a release that
        adds a semantic key would hash differently here than on the publishing
        host, and one that adds a config field would not even parse.  So the
        tree is read in a subprocess that imports ``mussel_dispatcher`` from the
        tree itself (see ``_TREE_SCRIPT`` for what it returns).
        """
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tree / "dispatcher")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _TREE_SCRIPT,
                 str(tree / self._relative_config()), self.target.name],
                cwd=str(tree), env=env, capture_output=True, text=True,
                timeout=120, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"cannot read release tree {tree}: timed out after {exc.timeout} s") from exc
        lines = [line for line in result.stdout.splitlines() if line.startswith(_TREE_MARKER)]
        if result.returncode or not lines:
            detail = (result.stderr or result.stdout or "no output")[-1000:]
            raise RuntimeError(f"cannot read release tree {tree}: {detail}")
        return json.loads(lines[-1][len(_TREE_MARKER):])

    def _active_tree(self) -> Path | None:
        """The tree ``active_release_link`` serves, or None before the first activation."""
        if not self.active_release_link:
            return None
        link = Path(self.active_release_link)
        if not link.exists():
            if link.is_symlink():
                raise RuntimeError(f"active release link is dangling: {link} -> {os.readlink(link)}")
            return None
        tree = link.resolve()
        if not tree.is_dir():
            raise RuntimeError(f"active release link does not point at a directory: {link} -> {tree}")
        return tree

    def _current_identity(self) -> tuple[str, str, str, str]:
        # Describe what the active link serves, not the checkout this process
        # was started from: an agent launched from a staged tree would
        # otherwise report that tree as active without switching the link.
        # In-process only when this process's code is the active tree's code;
        # after an activation whose restart failed, the old code is still
        # running and must not judge the new tree.
        tree = self._active_tree()
        if tree is None or tree == _CODE_TREE:
            contract = build_contract(self.cfg, target=self.target)
        else:
            contract = self._tree_contract(tree)
        return (
            str(contract["source_revision"]),
            str(contract["release_manifest_sha256"]),
            str(contract["semantic_contract_sha256"]),
            str(contract["execution_profile_sha256"]),
        )

    def _stage(self, revision: str, expected_manifest: str) -> Path:
        self.stage_root.mkdir(parents=True, exist_ok=True)
        stage = self.stage_root / self.deployment_id / revision
        if stage.exists():
            if git_revision(str(stage)) != revision:
                raise RuntimeError(f"staging path contains another revision: {stage}")
        else:
            stage.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "-C", self.repo_dir, "fetch", "--all", "--prune"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            subprocess.run(
                ["git", "-C", self.repo_dir, "worktree", "add", "--detach", str(stage), revision],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        actual_manifest = release_manifest_hash(tracked_source_manifest(str(stage)))
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                f"staged release manifest mismatch: expected {expected_manifest}, got {actual_manifest}"
            )
        return stage

    def _activate(self, stage: Path) -> None:
        if not self.active_release_link:
            raise RuntimeError(
                "release phase is ACTIVATING but active_release_link is not configured"
            )
        link = Path(self.active_release_link)
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        temporary.symlink_to(stage, target_is_directory=True)
        os.replace(temporary, link)

    def _staged_contract(self, stage: Path) -> dict[str, Any]:
        """The staged tree's contract, restart units and hook modules, read by its own code."""
        return self._tree_contract(stage)

    def _restart_units(self, units: Sequence[str] | None = None) -> None:
        if units is None:
            units = getattr(self, "_restart_target_units", None)
        if units is None:
            units = self.target.release_restart_units or []
        for unit in units:
            unit_name = str(unit).strip()
            if not unit_name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@_.-" for ch in unit_name):
                raise RuntimeError(f"Invalid release restart unit: {unit_name!r}")
            subprocess.run(
                ["systemctl", "--user", "restart", unit_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

    def _existing_status(self) -> str:
        return next(
            (
                str(node.get("status") or "")
                for node in self.state.list_deployment_nodes(self.deployment_id)
                if str(node.get("node_id") or "") == self.node_id
            ),
            "",
        ).upper()

    def run_once(self) -> bool:
        """Reconcile one release and return True after an activation switch."""
        desired = self.state.get_deployment_release(self.deployment_id)
        try:
            (
                current_revision,
                current_manifest,
                current_semantic,
                current_profile,
            ) = self._current_identity()
        except Exception as exc:
            self._report(status="ERROR", error=str(exc))
            return False

        if not desired:
            self._report(
                status="OBSERVING",
                active_revision=current_revision,
                staged_revision=current_revision,
                semantic_contract_sha256=current_semantic,
                execution_profile_sha256=current_profile,
            )
            return False

        desired_revision = str(desired["desired_revision"])
        desired_manifest = str(desired["desired_release_manifest_sha256"])
        desired_semantic = str(desired.get("desired_semantic_contract_sha256") or "")
        if current_revision == desired_revision and current_manifest == desired_manifest:
            restarted = False
            try:
                if desired_semantic and current_semantic != desired_semantic:
                    raise RuntimeError(
                        "active semantic contract differs from the published release: "
                        f"expected {desired_semantic}, got {current_semantic}"
                    )
                if self._existing_status() in {"ACTIVATING", "ERROR"}:
                    self._restart_units()
                    restarted = True
                self._report(
                    status="ACTIVE",
                    active_revision=current_revision,
                    staged_revision=current_revision,
                    semantic_contract_sha256=current_semantic,
                    execution_profile_sha256=current_profile,
                )
            except Exception as exc:
                self._report(
                    status="ERROR",
                    active_revision=current_revision,
                    staged_revision=current_revision,
                    semantic_contract_sha256=current_semantic,
                    execution_profile_sha256=current_profile,
                    error=str(exc),
                )
                return False
            return restarted

        activated = False
        try:
            stage = self._stage(desired_revision, desired_manifest)
            staged_contract = self._staged_contract(stage)
            staged_semantic = str(staged_contract["semantic_contract_sha256"])
            staged_profile = str(staged_contract["execution_profile_sha256"])
            if desired_semantic and staged_semantic != desired_semantic:
                raise RuntimeError(
                    "staged semantic contract differs from the published release: "
                    f"expected {desired_semantic}, got {staged_semantic}"
                )
            phase = str(desired.get("phase") or "STAGED").upper()
            if phase == "ACTIVATING":
                latest = self.state.get_deployment_release(self.deployment_id) or {}
                if (
                    str(latest.get("desired_revision") or "") != desired_revision
                    or str(latest.get("desired_release_manifest_sha256") or "") != desired_manifest
                    or str(latest.get("phase") or "").upper() != "ACTIVATING"
                ):
                    self._report(
                        status="READY",
                        staged_revision=desired_revision,
                        active_revision=current_revision,
                        semantic_contract_sha256=staged_semantic,
                        execution_profile_sha256=staged_profile,
                    )
                    return False
                self._report(
                    status="ACTIVATING",
                    staged_revision=desired_revision,
                    active_revision=current_revision,
                    semantic_contract_sha256=staged_semantic,
                    execution_profile_sha256=staged_profile,
                )
                self._activate(stage)
                activated = True
                # Keep the staged units in memory so a failed first restart can
                # be retried with the newly activated unit configuration.
                self._restart_target_units = list(staged_contract.get("release_restart_units") or [])
                self._restart_units(self._restart_target_units)
            self._report(
                status="ACTIVE" if phase == "ACTIVATING" else "READY",
                staged_revision=desired_revision,
                active_revision=(desired_revision if phase == "ACTIVATING" else current_revision),
                semantic_contract_sha256=staged_semantic,
                execution_profile_sha256=staged_profile,
            )
            return activated
        except Exception as exc:
            self._report(
                status="ERROR",
                staged_revision=desired_revision,
                active_revision=(desired_revision if activated else current_revision),
                semantic_contract_sha256=current_semantic,
                execution_profile_sha256=current_profile,
                error=str(exc),
            )
            return False

    def run(self) -> int:
        interval = max(10, int(os.environ.get("MUSSEL_RELEASE_POLL_SECONDS", "60")))
        while True:
            activated = self.run_once()
            if self.once:
                return 0
            if activated:
                # systemd Restart=on-failure reloads this release-agent process
                # through the newly activated symlink.
                return 75
            time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--target", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    target = cfg.get_dispatch_target(args.target)
    if target is None:
        raise SystemExit(f"Unknown dispatch target: {args.target}")
    return ReleaseAgent(
        cfg, target, deployment_id=args.deployment_id, once=args.once
    ).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
