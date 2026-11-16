"""Release-branch pin: a deployment runs and publishes only commits on one origin branch.

A production deployment may need to stay off whatever lands on main (e.g. a pipeline
change for a newer Mussel). With ``release_branch`` set, the dispatcher refuses to start on a checkout
whose HEAD is not on ``<release_origin_remote>/<release_branch>``
(:func:`verify_release_pin`), and the release controller publishes nothing else to
remote targets. Deploy by updating that branch, then pulling it in production.
"""
from __future__ import annotations

import subprocess

from .contract import git_revision


def on_release_branch(repo_dir: str, revision: str, remote: str, branch: str) -> tuple[bool, str]:
    """Whether ``revision`` is on ``<remote>/<branch>``, and a message saying why not.

    Fetches first; if the fetch fails (network), checks the last fetched ref and says so,
    so an unreachable remote alone never blocks a pinned revision.
    """
    ref = f"{remote}/{branch}"
    note = ""
    fetch = subprocess.run(
        ["git", "-C", repo_dir, "fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{ref}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    if fetch.returncode:
        note = f" (fetch failed, checked the last fetched {ref}: {fetch.stderr.strip()[-200:]})"
    known = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "--verify", "--quiet", f"refs/remotes/{ref}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
    )
    if known.returncode:
        return False, f"{ref} does not exist{note}"
    ancestor = subprocess.run(
        ["git", "-C", repo_dir, "merge-base", "--is-ancestor", revision, f"refs/remotes/{ref}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
    )
    if ancestor.returncode == 0:
        return True, note.strip()
    return False, f"revision {revision[:12]} is not on {ref}{note}"


def verify_release_pin(cfg) -> tuple[str | None, str]:
    """(problem, note) for this checkout against ``cfg.release_branch``.

    ``problem`` is None when the deployment is unpinned or HEAD is on the branch;
    ``note`` carries the revision checked and any fetch caveat, for the log.
    """
    if not cfg.release_branch:
        return None, ""
    remote = cfg.release_origin_remote or "origin"
    revision = git_revision(cfg.repo_dir)
    pinned, why = on_release_branch(cfg.repo_dir, revision, remote, cfg.release_branch)
    if not pinned:
        return f"{why}; deploy by updating {cfg.release_branch}, not main", revision
    return None, f"{revision[:12]} is on {remote}/{cfg.release_branch}" + (f" {why}" if why else "")
