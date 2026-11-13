import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from mussel_dispatcher.config import Config, RemoteTargetConfig
from mussel_dispatcher.release import ReleaseController
from mussel_dispatcher.release_agent import ReleaseAgent
from mussel_dispatcher.state import StateStore


def _release_cfg(tmp_path) -> Config:
    return Config(
        nextflow_profiles="standard",
        outdir=str(tmp_path / "out"),
        repo_dir=str(tmp_path / "repo"),
        auto_sync_release=True,
        release_deployment_id="default",
        dispatch_targets=[
            RemoteTargetConfig(
                name="hpc",
                agent_heartbeat_timeout_seconds=300,
            )
        ],
    )


def _release_tree(root, revision: str, *, hooks=(), units=("agent.service",), modules=()):
    """A release tree whose own mussel_dispatcher describes itself with ``revision``."""
    package = root / "dispatcher" / "mussel_dispatcher"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n")
    (package / "config.py").write_text(
        "class Target:\n"
        f"    release_restart_units = {list(units)!r}\n"
        "class Config:\n"
        f"    post_batch_hooks = {list(hooks)!r}\n"
        "    @classmethod\n"
        "    def load(cls, path):\n"
        "        return cls()\n"
        "    def get_dispatch_target(self, name):\n"
        "        return Target() if name == 'hpc' else None\n"
    )
    (package / "contract.py").write_text(
        "def build_contract(cfg, target=None):\n"
        f"    return {{key: '{revision}-' + key for key in ("
        "'source_revision', 'release_manifest_sha256', "
        "'semantic_contract_sha256', 'execution_profile_sha256')}\n"
    )
    for module in modules:
        (package / f"{module}.py").write_text("\n")
    (root / "dispatcher" / "dispatcher.yaml").write_text("{}\n")
    return root


def _tree_agent(tmp_path, active_link: str = "") -> ReleaseAgent:
    agent = object.__new__(ReleaseAgent)
    agent.cfg = MagicMock(
        config_path=str(tmp_path / "running" / "dispatcher" / "dispatcher.yaml"),
        repo_dir=str(tmp_path / "running"),
    )
    agent.target = RemoteTargetConfig(name="hpc")
    agent.active_release_link = active_link
    return agent


def test_release_tree_rejects_import_from_outside_the_tree(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    old_package = tmp_path / "old" / "dispatcher" / "mussel_dispatcher"
    old_package.mkdir(parents=True)
    (old_package / "__init__.py").write_text("\n")
    (old_package / "config.py").write_text("class Config: pass\n")
    monkeypatch.setenv("PYTHONPATH", str(old_package.parent))

    with pytest.raises(RuntimeError, match="cannot read release tree"):
        _tree_agent(tmp_path)._tree_contract(stage)


def test_staged_release_is_read_by_the_staged_code(tmp_path):
    # The running agent's code must not judge a release whose rules it predates:
    # a newly added semantic key would hash differently, and a newly added
    # config field would not even parse.  Nothing is parsed in-process.
    stage = _release_tree(tmp_path / "stage", "staged", units=("new-agent.service",))
    agent = _tree_agent(tmp_path)
    with patch("mussel_dispatcher.release_agent.Config.load", side_effect=TypeError("unknown key")), \
            patch("mussel_dispatcher.release_agent.build_contract") as running_build:
        contract = agent._staged_contract(stage)
    running_build.assert_not_called()
    assert contract["semantic_contract_sha256"] == "staged-semantic_contract_sha256"
    assert contract["release_restart_units"] == ["new-agent.service"]


def test_staged_release_hook_modules_must_exist_in_the_stage(tmp_path):
    hooks = [{"command": "python -m mussel_dispatcher.wds", "args": ["append"]}]
    stage = _release_tree(tmp_path / "ok", "ok", hooks=hooks, modules=("wds",))
    assert _tree_agent(tmp_path)._staged_contract(stage)["hook_modules"] == ["mussel_dispatcher.wds"]

    # Absent from the stage: whether the interpreter finds nothing or (with an
    # editable install of the dispatcher) the same module in another checkout,
    # the stage must be rejected.
    stage = _release_tree(tmp_path / "missing", "missing", hooks=hooks)
    with pytest.raises(
        RuntimeError,
        match="missing hook modules: mussel_dispatcher.wds|mussel_dispatcher.wds is not loaded from the release tree",
    ):
        _tree_agent(tmp_path)._staged_contract(stage)


def test_release_tree_must_supply_its_own_contract_module(tmp_path):
    # An installed or editable dispatcher elsewhere must not stand in for it.
    stage = _release_tree(tmp_path / "stage", "staged")
    (stage / "dispatcher" / "mussel_dispatcher" / "contract.py").unlink()
    with pytest.raises(RuntimeError, match="cannot read release tree"):
        _tree_agent(tmp_path)._staged_contract(stage)


def test_release_tree_without_the_target_is_an_error(tmp_path):
    stage = _release_tree(tmp_path / "stage", "staged")
    agent = _tree_agent(tmp_path)
    agent.target = RemoteTargetConfig(name="local-hpc")
    with pytest.raises(RuntimeError, match="release tree has no target local-hpc"):
        agent._staged_contract(stage)


def test_release_tree_failure_and_timeout_are_errors(tmp_path):
    tree = _release_tree(tmp_path / "broken", "broken")
    (tree / "dispatcher" / "mussel_dispatcher" / "contract.py").write_text("raise ImportError('boom')\n")
    agent = _tree_agent(tmp_path)
    with pytest.raises(RuntimeError, match=r"(?s)cannot read release tree.*boom"):
        agent._tree_contract(tree)
    with patch(
        "mussel_dispatcher.release_agent.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=120),
    ), pytest.raises(RuntimeError, match="timed out after 120"):
        agent._tree_contract(tree)


def test_current_identity_describes_the_active_link_not_the_running_checkout(tmp_path):
    active = _release_tree(tmp_path / "release-a", "active")
    link = tmp_path / "mussel-nf.active"
    link.symlink_to(active, target_is_directory=True)
    agent = _tree_agent(tmp_path, active_link=str(link))
    with patch("mussel_dispatcher.release_agent.build_contract") as running_build:
        identity = agent._current_identity()
    running_build.assert_not_called()
    assert identity[0] == "active-source_revision"
    assert identity[2] == "active-semantic_contract_sha256"


@pytest.mark.parametrize("link_state", ["unconfigured", "not-yet-created", "running-checkout"])
def test_current_identity_builds_in_process_when_the_running_checkout_is_active(tmp_path, link_state):
    (tmp_path / "running").mkdir()
    link = tmp_path / "mussel-nf.active"
    if link_state == "running-checkout":
        link.symlink_to(tmp_path / "running", target_is_directory=True)
    agent = _tree_agent(tmp_path, active_link="" if link_state == "unconfigured" else str(link))
    agent._tree_contract = MagicMock()
    contract = {key: "running" for key in (
        "source_revision", "release_manifest_sha256",
        "semantic_contract_sha256", "execution_profile_sha256")}
    with patch("mussel_dispatcher.release_agent._CODE_TREE", (tmp_path / "running").resolve()), \
            patch("mussel_dispatcher.release_agent.build_contract", return_value=contract):
        assert agent._current_identity() == ("running",) * 4
    agent._tree_contract.assert_not_called()


def test_old_code_does_not_judge_the_new_tree_after_activation(tmp_path):
    # The link now serves the new tree, and cfg.repo_dir (which goes through the
    # link) resolves there too, but this process still runs the old tree's code
    # (e.g. the restart after activation failed): the new tree is read by its own code.
    old_code = _release_tree(tmp_path / "release-a", "old")
    new_tree = _release_tree(tmp_path / "release-b", "new")
    link = tmp_path / "mussel-nf.active"
    link.symlink_to(new_tree, target_is_directory=True)
    agent = _tree_agent(tmp_path, active_link=str(link))
    agent.cfg.repo_dir = str(link)
    agent.cfg.config_path = str(link / "dispatcher" / "dispatcher.yaml")
    with patch("mussel_dispatcher.release_agent._CODE_TREE", old_code.resolve()), \
            patch("mussel_dispatcher.release_agent.build_contract") as running_build:
        identity = agent._current_identity()
    running_build.assert_not_called()
    assert identity[2] == "new-semantic_contract_sha256"


def test_dangling_active_link_is_an_error(tmp_path):
    link = tmp_path / "mussel-nf.active"
    link.symlink_to(tmp_path / "deleted-release", target_is_directory=True)
    agent = _tree_agent(tmp_path, active_link=str(link))
    with patch("mussel_dispatcher.release_agent.build_contract") as running_build, \
            pytest.raises(RuntimeError, match="active release link is dangling"):
        agent._current_identity()
    running_build.assert_not_called()


def test_stale_remote_order_is_retired_without_charging_slide_retry(tmp_path):
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.add_slide("s3://slides/a.svs", "a")
    state.add_batch(
        "batch-1",
        str(tmp_path / "batch.csv"),
        str(tmp_path / "work"),
        1,
        str(tmp_path / "batch.log"),
        backend="remote_agent",
        target="hpc",
    )
    assert state.mark_dispatched(["s3://slides/a.svs"], "batch-1") == 1
    state.write_remote_status(
        "batch-1",
        {"status": "BLOCKED", "failure_class": "STALE_ORDER"},
    )

    assert state.requeue_blocked_remote_batch("batch-1", reason="release drift") == 1
    slide = state.get_slides_by_id("a")[0]
    batch = state.get_batch("batch-1")
    remote = state.read_remote_status("batch-1")
    assert slide["status"] == "PENDING"
    assert slide["fail_count"] == 0
    assert batch["status"] == "FAILED"
    assert remote["status"] == "FAILED"
    assert remote["requeued"] is True


def test_release_state_is_cas_updated(tmp_path):
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.publish_deployment_release(
        "default",
        desired_revision="abc123",
        desired_release_manifest_sha256="manifest",
        desired_semantic_contract_sha256="semantic",
    )
    state.write_deployment_node(
        "default",
        "hpc:node-1",
        target="hpc",
        status="READY",
        staged_revision="abc123",
    )
    assert state.set_deployment_phase(
        "default", phase="ACTIVATING", expected_phases=("STAGED",)
    )
    assert not state.set_deployment_phase(
        "default", phase="ACTIVE", expected_phases=("STAGED",)
    )
    assert state.get_deployment_release("default")["phase"] == "ACTIVATING"
    assert state.list_deployment_nodes("default")[0]["status"] == "READY"


def test_release_phase_cas_also_guards_revision(tmp_path):
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.publish_deployment_release(
        "default",
        desired_revision="new-release",
        desired_release_manifest_sha256="manifest",
    )

    assert not state.set_deployment_phase(
        "default",
        phase="ACTIVATING",
        expected_phases=("STAGED",),
        expected_revision="stale-release",
    )
    assert state.get_deployment_release("default")["phase"] == "STAGED"


def test_managed_release_gate_is_closed_before_first_publish(tmp_path):
    cfg = _release_cfg(tmp_path)
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.cfg = cfg

    ready, reason = state.remote_target_release_ready("hpc")

    assert not ready
    assert "first managed release" in reason
    snapshot = state.release_sync_snapshot()
    assert snapshot["managed"] is True
    assert snapshot["state"] == "SYNCING"


def test_release_controller_blocks_tracked_dirty_checkout(tmp_path):
    cfg = _release_cfg(tmp_path)
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.cfg = cfg
    state.publish_deployment_release(
        "default",
        desired_revision="clean-release",
        desired_release_manifest_sha256="manifest",
    )
    controller = ReleaseController(cfg, state)

    with patch("mussel_dispatcher.release.git_revision", return_value="dirty-release"), patch.object(
        controller, "_tracked_dirty", return_value=" M dispatcher/file.py"
    ):
        result = controller.run_once()

    assert result["status"] == "BLOCKED_LOCAL"
    release = state.get_deployment_release("default")
    assert release["controller_status"] == "BLOCKED_LOCAL"
    ready, reason = state.remote_target_release_ready("hpc")
    assert not ready
    assert "Tracked changes" in reason


def test_release_controller_blocks_until_local_dispatcher_reloads(tmp_path):
    cfg = _release_cfg(tmp_path)
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.cfg = cfg
    state.publish_deployment_release(
        "default",
        desired_revision="old-release",
        desired_release_manifest_sha256="manifest",
    )
    controller = ReleaseController(cfg, state)
    controller.loaded_revision = "old-release"

    with patch("mussel_dispatcher.release.git_revision", return_value="new-release"), patch.object(
        controller, "_tracked_dirty", return_value=""
    ):
        result = controller.run_once()

    assert result["status"] == "BLOCKED_LOCAL"
    assert "restart the local dispatcher" in result["message"]
    release = state.get_deployment_release("default")
    assert release["desired_revision"] == "old-release"
    assert release["controller_revision"] == "new-release"


def test_release_controller_stages_activates_and_waits_for_loaded_agent(tmp_path):
    cfg = _release_cfg(tmp_path)
    state = StateStore(str(tmp_path / "dispatcher.db"))
    state.cfg = cfg
    state.publish_deployment_release(
        "default",
        desired_revision="release-a",
        desired_release_manifest_sha256="manifest",
    )
    now = datetime.now(timezone.utc).isoformat()
    state.write_deployment_node(
        "default",
        "hpc:node-1",
        target="hpc",
        status="READY",
        staged_revision="release-a",
        active_revision="old-release",
        heartbeat_at=now,
    )
    controller = ReleaseController(cfg, state)
    controller.loaded_revision = "release-a"

    with patch("mussel_dispatcher.release.git_revision", return_value="release-a"), patch.object(
        controller, "_tracked_dirty", return_value=""
    ):
        first = controller.run_once()
        assert first["status"] == "SYNCING"
        assert state.get_deployment_release("default")["phase"] == "ACTIVATING"

        state.write_deployment_node(
            "default",
            "hpc:node-1",
            target="hpc",
            status="ACTIVE",
            staged_revision="release-a",
            active_revision="release-a",
            heartbeat_at=now,
        )
        waiting = controller.run_once()
        assert waiting["status"] == "SYNCING"
        assert "worker" in waiting["message"]

        state.write_remote_agent(
            "hpc",
            {
                "status": "ALIVE",
                "heartbeat_at": now,
                "agent_revision": "release-a",
            },
        )
        active = controller.run_once()

    assert active["status"] == "ACTIVE"
    assert state.get_deployment_release("default")["phase"] == "ACTIVE"
    assert state.remote_target_release_ready("hpc") == (True, "")
    snapshot = state.release_sync_snapshot()
    assert snapshot["state"] == "ACTIVE"
    assert snapshot["targets"][0]["agent_revision"] == "release-a"


def test_release_agent_validates_staged_contract_before_ready(tmp_path):
    agent = object.__new__(ReleaseAgent)
    agent.deployment_id = "default"
    agent.node_id = "hpc:node-1"
    agent.target = RemoteTargetConfig(name="hpc")
    agent.state = MagicMock()
    agent.state.get_deployment_release.return_value = {
        "desired_revision": "release-b",
        "desired_release_manifest_sha256": "manifest-b",
        "desired_semantic_contract_sha256": "semantic-b",
        "phase": "STAGED",
    }
    agent._current_identity = MagicMock(
        return_value=("release-a", "manifest-a", "semantic-a", "profile-a")
    )
    agent._stage = MagicMock(return_value=tmp_path / "release-b")
    agent._staged_contract = MagicMock(
        return_value={
            "semantic_contract_sha256": "semantic-b",
            "execution_profile_sha256": "profile-b",
        }
    )
    agent._report = MagicMock()

    assert agent.run_once() is False
    report = agent._report.call_args.kwargs
    assert report["status"] == "READY"
    assert report["semantic_contract_sha256"] == "semantic-b"
    assert not report.get("error")


def test_release_agent_rechecks_revision_before_activation(tmp_path):
    agent = object.__new__(ReleaseAgent)
    agent.deployment_id = "default"
    agent.node_id = "hpc:node-1"
    agent.target = RemoteTargetConfig(name="hpc")
    desired = {
        "desired_revision": "release-b",
        "desired_release_manifest_sha256": "manifest-b",
        "desired_semantic_contract_sha256": "semantic-b",
        "phase": "ACTIVATING",
    }
    agent.state = MagicMock()
    agent.state.get_deployment_release.side_effect = [
        desired,
        {**desired, "desired_revision": "release-c"},
    ]
    agent._current_identity = MagicMock(
        return_value=("release-a", "manifest-a", "semantic-a", "profile-a")
    )
    agent._stage = MagicMock(return_value=tmp_path / "release-b")
    agent._staged_contract = MagicMock(
        return_value={
            "semantic_contract_sha256": "semantic-b",
            "execution_profile_sha256": "profile-b",
        }
    )
    agent._activate = MagicMock()
    agent._report = MagicMock()

    assert agent.run_once() is False
    agent._activate.assert_not_called()
    assert agent._report.call_args.kwargs["status"] == "READY"


def test_release_agent_activation_restarts_units_from_staged_target(tmp_path):
    agent = object.__new__(ReleaseAgent)
    agent.deployment_id = "default"
    agent.node_id = "hpc:node-1"
    agent.target = RemoteTargetConfig(
        name="hpc", release_restart_units=["old-agent.service"]
    )
    desired = {
        "desired_revision": "release-b",
        "desired_release_manifest_sha256": "manifest-b",
        "desired_semantic_contract_sha256": "semantic-b",
        "phase": "ACTIVATING",
    }
    agent.state = MagicMock()
    agent.state.get_deployment_release.side_effect = [desired, desired]
    agent._current_identity = MagicMock(
        return_value=("release-a", "manifest-a", "semantic-a", "profile-a")
    )
    agent._stage = MagicMock(return_value=tmp_path / "release-b")
    agent._staged_contract = MagicMock(
        return_value={
            "semantic_contract_sha256": "semantic-b",
            "execution_profile_sha256": "profile-b",
            "release_restart_units": ["new-agent.service"],
        }
    )
    agent._activate = MagicMock()
    agent._restart_units = MagicMock()
    agent._report = MagicMock()

    assert agent.run_once() is True
    agent._activate.assert_called_once_with(tmp_path / "release-b")
    agent._restart_units.assert_called_once_with(["new-agent.service"])
    assert agent._restart_target_units == ["new-agent.service"]
    assert agent._report.call_args.kwargs["status"] == "ACTIVE"


def test_release_agent_retries_partial_activation_and_requests_self_reload():
    agent = object.__new__(ReleaseAgent)
    agent.deployment_id = "default"
    agent.node_id = "hpc:node-1"
    agent.target = RemoteTargetConfig(name="hpc")
    agent.state = MagicMock()
    agent.state.get_deployment_release.return_value = {
        "desired_revision": "release-b",
        "desired_release_manifest_sha256": "manifest-b",
        "desired_semantic_contract_sha256": "semantic-b",
        "phase": "ACTIVATING",
    }
    agent._current_identity = MagicMock(
        return_value=("release-b", "manifest-b", "semantic-b", "profile-b")
    )
    agent._existing_status = MagicMock(return_value="ERROR")
    agent._restart_units = MagicMock()
    agent._report = MagicMock()

    assert agent.run_once() is True
    agent._restart_units.assert_called_once_with()
    assert agent._report.call_args.kwargs["status"] == "ACTIVE"
