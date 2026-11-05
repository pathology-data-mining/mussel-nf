"""Content-addressed container images: verify that pinned images are the files they claim.

A Nextflow config pins an image by naming the file after its content hash,
``<name>-sha256-<64 hex>.sif`` (or ``.img``). That name is only a promise: a rebuilt or
overwritten file would keep it and silently change every embedding. At startup the
dispatcher and the remote agent check each pinned image their config references that
exists on this host (the other site's images are skipped), and refuse to start on a
mismatch. A file is hashed once and the result is cached by (path, size, mtime), so a
restart costs a stat per image.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

PIN_RE = re.compile(r"(/[^'\"\s]*-sha256-([0-9a-f]{64})\.(?:sif|img))")
DEFAULT_CACHE = Path(os.environ.get(
    "MUSSEL_IMAGE_PIN_CACHE", os.path.expanduser("~/.cache/mussel-dispatcher/image_pins.json")))


def pinned_images(config_paths: Iterable[str]) -> list[tuple[str, str]]:
    """(path, expected sha256) of every content-addressed image named in these configs."""
    seen: dict[str, str] = {}
    for config in config_paths:
        if config and os.path.isfile(config):
            for path, digest in PIN_RE.findall(Path(config).read_text()):
                seen.setdefault(path, digest)
    return sorted(seen.items())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(64 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def verify_pinned_images(config_paths: Iterable[str], cache_path: Path = DEFAULT_CACHE) -> list[str]:
    """Problems with the pinned images present on this host (empty when all match)."""
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    problems, changed = [], False
    for path, expected in pinned_images(config_paths):
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue  # another site's image
        key = f"{st.st_size}:{st.st_mtime_ns}"
        entry = cache.get(path) or {}
        actual = entry.get("sha256") if entry.get("key") == key else None
        if actual is None:
            actual = _sha256(path)
            cache[path] = {"key": key, "sha256": actual}
            changed = True
        if actual != expected:
            problems.append(f"{path}: sha256 is {actual}, the name pins {expected}")
    if changed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
            tmp.replace(cache_path)
        except OSError:
            pass  # the check still ran; it just re-hashes next time
    return problems
