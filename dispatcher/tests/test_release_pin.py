from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

from mussel_dispatcher.release_pin import on_release_branch, verify_release_pin
from mussel_dispatcher.release import ReleaseController


def _git(cwd, *args) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _repos(tmp_path):
    """origin (bare) with release/x at commit A; main at B (A plus one commit); a clone."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for k, v in (("user.email", "t@example.org"), ("user.name", "t")):
        _git(work, "config", k, v)
    (work / "f").write_text("a\n")
    _git(work, "add", "f"); _git(work, "commit", "-qm", "A")
    a = _git(work, "rev-parse", "HEAD")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "origin", "main", f"{a}:refs/heads/release/x")
    (work / "f").write_text("b\n")
    _git(work, "commit", "-qam", "B")
    b = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", "origin", "main")
    return work, a, b


def test_on_release_branch_accepts_the_pinned_line_only(tmp_path):
    work, a, b = _repos(tmp_path)
    assert on_release_branch(str(work), a, "origin", "release/x") == (True, "")
    ok, why = on_release_branch(str(work), b, "origin", "release/x")
    assert not ok and "is not on origin/release/x" in why       # on main, not on the release branch
    ok, why = on_release_branch(str(work), a, "origin", "release/none")
    assert not ok and "does not exist" in why


def test_on_release_branch_survives_an_unreachable_remote(tmp_path):
    work, a, b = _repos(tmp_path)
    on_release_branch(str(work), a, "origin", "release/x")       # fetch once
    _git(work, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    ok, why = on_release_branch(str(work), a, "origin", "release/x")
    assert ok and "fetch failed" in why                          # last fetched ref still decides
    ok, _ = on_release_branch(str(work), b, "origin", "release/x")
    assert not ok


def test_release_controller_does_not_publish_an_unpinned_revision(tmp_path):
    work, a, b = _repos(tmp_path)
    _git(work, "checkout", "-q", "main")                        # HEAD = B, not on release/x
    cfg = SimpleNamespace(repo_dir=str(work), release_branch="release/x", release_origin_remote="origin",
                          release_require_origin=True, release_deployment_id="d")
    controller = ReleaseController.__new__(ReleaseController)
    controller.cfg, controller.state, controller.deployment_id = cfg, MagicMock(), "d"
    controller.loaded_revision = b
    controller.state.get_deployment_release.return_value = None

    result = controller.run_once()

    assert result["status"] == "BLOCKED_LOCAL" and "is not on origin/release/x" in result["message"]
    controller.state.publish_deployment_release.assert_not_called()


def test_verify_release_pin_is_the_startup_guard(tmp_path):
    work, a, b = _repos(tmp_path)
    cfg = SimpleNamespace(repo_dir=str(work), release_branch="", release_origin_remote="origin")
    assert verify_release_pin(cfg) == (None, "")                 # unpinned deployment: no check

    cfg.release_branch = "release/x"
    _git(work, "checkout", "-q", a)
    problem, note = verify_release_pin(cfg)
    assert problem is None and note == f"{a[:12]} is on origin/release/x"

    _git(work, "checkout", "-q", "main")                         # someone moved the checkout to main
    problem, _ = verify_release_pin(cfg)
    assert "is not on origin/release/x" in problem and "deploy by updating release/x, not main" in problem


def test_release_controller_reuses_a_pin_refusal_until_head_moves(tmp_path, monkeypatch):
    calls = []

    def fake_pin(repo_dir, revision, remote, branch):
        calls.append(revision)
        return False, f"revision {revision[:12]} is not on {remote}/{branch}"

    monkeypatch.setattr("mussel_dispatcher.release.on_release_branch", fake_pin)
    controller = ReleaseController.__new__(ReleaseController)
    controller.cfg = SimpleNamespace(repo_dir="/x", release_branch="release/x", release_origin_remote="origin")

    first = controller._pin_refusal("b" * 40)
    assert first and controller._pin_refusal("b" * 40) == first      # within the window: no new fetch
    assert calls == ["b" * 40]
    controller._pin_refusal("c" * 40)                                # HEAD moved: checked at once
    assert calls == ["b" * 40, "c" * 40]
    controller._pin_block = ("c" * 40, 0.0, "old")                   # window expired
    controller._pin_refusal("c" * 40)
    assert calls == ["b" * 40, "c" * 40, "c" * 40]
