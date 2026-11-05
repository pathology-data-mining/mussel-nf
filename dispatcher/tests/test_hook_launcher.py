from __future__ import annotations

import json
import subprocess
import sys

from mussel_dispatcher.hook_hardening import HookFailureCategory, classify_hook_failure
from mussel_dispatcher.hook_launcher import LAUNCHER, is_launched, launch_command

PROBE = "import json, sys\nprint(json.dumps({'file': __file__, 'argv': sys.argv[1:], 'name': __name__}))\n"


def _package(root, probe=PROBE):
    pkg = root / "mussel_dispatcher"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("\n")
    (pkg / "probe.py").write_text(probe)
    return pkg


def _run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env or {})


def test_launch_command_rewrites_dispatcher_modules_only(tmp_path):
    _package(tmp_path / "repo" / "dispatcher")
    repo = str(tmp_path / "repo")
    head = [LAUNCHER, "--release", f"{repo}/dispatcher", "mussel_dispatcher.wds", "--x", "1"]
    cases = {
        ("python3", "-u", "-m", "mussel_dispatcher.wds", "--x", "1"): ["python3", "-u", *head],
        # Options that take a value must not hide the -m that follows them.
        ("python", "-X", "importtime", "-W", "ignore", "-m", "mussel_dispatcher.wds", "--x", "1"):
            ["python", "-X", "importtime", "-W", "ignore", *head],
        ("python", "-mmussel_dispatcher.wds", "--x", "1"): ["python", *head],
    }
    for cmd, want in cases.items():
        assert launch_command(list(cmd), repo) == want
        assert is_launched(want) and not is_launched(list(cmd))
    for other in (["python", "-m", "example"], ["python", "script.py"], ["bash", "-c", "x"], [],
                  ["python", "script.py", "-m", "mussel_dispatcher.wds"],      # the script's argument
                  ["python", "-c", "pass", "-m", "mussel_dispatcher.wds"],     # -c ends options
                  ["python", "-X", "-m", "mussel_dispatcher.wds"]):            # -m is -X's value
        assert launch_command(other, repo) == other


def test_launcher_refuses_a_tree_without_the_package(tmp_path):
    # Fail closed: a misconfigured repo_dir must not fall back to an installed copy.
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.probe"], str(tmp_path))
    assert is_launched(cmd)
    out = _run(cmd, tmp_path)
    assert out.returncode == 78 and "not a dispatcher release" in out.stderr
    assert classify_hook_failure(out.returncode, out.stderr).category == HookFailureCategory.RELEASE_OR_CONFIG


def test_launcher_runs_the_release_module_as_main(tmp_path):
    release = _package(tmp_path / "repo" / "dispatcher").parent
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.probe", "a", "--b=c"], str(tmp_path / "repo"))
    out = _run(cmd, tmp_path / "repo")
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got == {"file": str(release / "mussel_dispatcher" / "probe.py"), "argv": ["a", "--b=c"],
                   "name": "__main__"}


def test_launcher_refuses_a_copy_from_another_checkout(tmp_path):
    # An installed copy that wins the import the way an editable install's finder can.
    _package(tmp_path / "repo" / "dispatcher")
    stale = _package(tmp_path / "stale").parent
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import sys\nfrom importlib.machinery import PathFinder\n"
        "class Stale:\n"
        "    @staticmethod\n"
        "    def find_spec(name, path=None, target=None):\n"
        f"        return PathFinder.find_spec(name, [{str(stale)!r}]) if name == 'mussel_dispatcher' else None\n"
        "sys.meta_path.insert(0, Stale)\n"
    )
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.probe"], str(tmp_path / "repo"))
    out = _run(cmd, tmp_path / "repo", env={"PYTHONPATH": str(site)})
    assert out.returncode == 78 and out.stdout == ""            # refused before running anything
    assert str(stale) in out.stderr
    assert classify_hook_failure(out.returncode, out.stderr).category == HookFailureCategory.RELEASE_OR_CONFIG


def test_launcher_refuses_a_missing_module(tmp_path):
    _package(tmp_path / "repo" / "dispatcher")
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.absent"], str(tmp_path / "repo"))
    out = _run(cmd, tmp_path / "repo")
    assert out.returncode == 78 and "not found" in out.stderr


def test_launcher_passes_through_the_module_exit_code(tmp_path):
    _package(tmp_path / "repo" / "dispatcher", probe="import sys\nsys.exit(3)\n")
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.probe"], str(tmp_path / "repo"))
    assert _run(cmd, tmp_path / "repo").returncode == 3


def test_launcher_keeps_the_stdlib_under_safe_path(tmp_path):
    # With PYTHONSAFEPATH, sys.path[0] is a stdlib entry, not the launcher's directory:
    # the release goes in front and every original entry stays.
    release = _package(tmp_path / "repo" / "dispatcher",
                       probe="import json, sys\nprint(json.dumps(sys.path))\n").parent
    env = {"PYTHONSAFEPATH": "1"}
    plain = json.loads(_run([sys.executable, "-c", "import json, sys; print(json.dumps(sys.path))"],
                            tmp_path, env=env).stdout)
    cmd = launch_command([sys.executable, "-m", "mussel_dispatcher.probe"], str(tmp_path / "repo"))
    out = _run(cmd, tmp_path / "repo", env=env)
    assert out.returncode == 0, out.stderr
    launched = json.loads(out.stdout)
    assert launched[0] == str(release)
    assert [p for p in launched if p in plain] == plain
