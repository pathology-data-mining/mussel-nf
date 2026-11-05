"""Run a dispatcher hook module from one release, and refuse any other copy of it.

    python hook_launcher.py --release <repo>/dispatcher mussel_dispatcher.wds [args...]

Hook commands name modules (``python -m mussel_dispatcher.wds ...``) and run from the
repository root, where Python does not see ``dispatcher/`` and imports whatever copy the
interpreter has installed, e.g. a stale editable install of an older checkout,
which every local WDS append would then run. The dispatcher therefore rewrites such
commands (:func:`launch_command`) to run through this file, which puts the release first
on ``sys.path``, checks that the module resolves inside it and only then runs it, in
the same process: the code that was checked is the code that runs.

A refusal exits 78 with the ``[mussel-hook-preflight]`` marker, so it is classified
RELEASE_OR_CONFIG like any other preflight failure. The launcher is invoked by path
from the running dispatcher's own package, so it also serves hooks whose ``--config``
selects a different release tree.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path

LAUNCHER = os.path.abspath(__file__)
PACKAGE = "mussel_dispatcher"
REFUSED = 78


# Interpreter options whose value is the next argument (``-W ignore``, ``-X importtime``).
_VALUE_OPTIONS = {"-W", "-X", "--check-hash-based-pycs"}


def _module_argument(cmd: list[str]) -> tuple[int, str] | None:
    """(index, module) of ``-m MODULE`` / ``-mMODULE`` in interpreter-option position, as
    Python parses its command line: options (and their values) first, then -m, -c or a
    script, which ends option parsing."""
    i = 1
    while i < len(cmd):
        token = str(cmd[i])
        if token == "-m":
            return (i, str(cmd[i + 1])) if i + 1 < len(cmd) else None
        if token.startswith("-m") and not token.startswith("--"):
            return i, token[2:]
        if token in _VALUE_OPTIONS:
            i += 2
            continue
        if token in ("-c", "-") or not token.startswith("-"):
            return None
        i += 1
    return None


def launch_command(cmd: list[str], repo_dir: str) -> list[str]:
    """``[python, *opts, -m, mussel_dispatcher.X, *args]`` run through the launcher.

    Every such command is rewritten, even when ``repo_dir`` has no
    ``dispatcher/mussel_dispatcher``: the launcher then refuses it rather than
    letting an installed copy run. Other commands are returned unchanged.
    """
    if not cmd or not os.path.basename(str(cmd[0])).startswith("python"):
        return cmd
    found = _module_argument(cmd)
    if found is None or found[1].split(".")[0] != PACKAGE:
        return cmd
    i, module = found
    end = i + 2 if str(cmd[i]) == "-m" else i + 1
    return [*cmd[:i], LAUNCHER, "--release", str(Path(repo_dir) / "dispatcher"), module, *cmd[end:]]


def is_launched(cmd: list[str]) -> bool:
    """True for a command :func:`launch_command` rewrote: the launcher checks its own import."""
    return any(str(part) == LAUNCHER for part in cmd)


def _refuse(detail: str) -> int:
    print(f"[mussel-hook-preflight] category=RELEASE_OR_CONFIG detail={detail}", file=sys.stderr)
    return REFUSED


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "--release":
        print("usage: hook_launcher.py --release DIR MODULE [ARGS...]", file=sys.stderr)
        return 2
    release, module, args = os.path.realpath(argv[1]), argv[2], argv[3:]
    if not os.path.isdir(os.path.join(release, PACKAGE)):
        return _refuse(f"{release} is not a dispatcher release (no {PACKAGE}/)")
    # Run as a file, sys.path[0] is this package's directory, whose modules (config,
    # storage, ...) would shadow top-level names: the release takes its place. Under
    # -P / -I / PYTHONSAFEPATH there is no such entry, and sys.path[0] is the stdlib.
    here = os.path.dirname(os.path.realpath(LAUNCHER))
    if sys.path and sys.path[0] and os.path.realpath(sys.path[0]) == here:
        sys.path[0] = release
    else:
        sys.path.insert(0, release)
    try:
        spec = importlib.util.find_spec(module)
    except ImportError as exc:
        return _refuse(f"cannot import {module}: {exc}")
    if spec is None or not spec.origin:
        return _refuse(f"{module} not found in {release}")
    origin = os.path.realpath(spec.origin)
    if not origin.startswith(release + os.sep):
        return _refuse(f"{module} imports from {origin}, not the release {release}")
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
