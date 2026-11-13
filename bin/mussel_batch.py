#!/usr/bin/env python3
"""Run a Mussel batch command and settle the batch's outputs (Mussel >= 1.6.0).

Mussel's batch extract_features / tessellate_extract_features try every slide:
when some fail, they write the others' outputs and then exit non-zero with
"N of M slide(s) failed (the others' outputs were written): <id>: <reason>; ..."
(mussel.utils.feature_extract.raise_if_slides_failed). Slides without tiles are
skipped with a warning; extract_features (identity) still writes an empty (0, D)
.features.h5 and an empty .pt for them.

  run     Run the command, copying its stdout and stderr to --log as well as
          to this script's stdout / stderr; exit with the command's status.

  finish  After the command (and any renaming of its outputs):
          1. Rename empty (0-row) feature files in --scan dirs to <name>.empty,
             so no *.features.{pt,h5} / *.patch_features.* glob (publishDir,
             task outputs, the dispatcher's WDS append) picks them up.
          2. Write --report (TSV: slide_id, model, status) listing each expected
             (slide, model) without an output: status "failed" (named in Mussel's
             failure message), "no_tiles" (empty outputs moved aside), or
             "missing" (no output, e.g. skipped for having no tissue).
             Written only when it has rows.
          3. Exit 0 if the command succeeded, or if it failed with Mussel's
             partial-failure message and every model still has at least one
             slide's output (a partial success: the failed slides are in the
             report). Otherwise exit with the command's status, so a real
             failure (crash, OOM kill, a model failing on every slide, "none of
             the N slide(s) had tiles") still fails the task.

--expect MODEL=TEMPLATE names the primary output of each (slide, model), with
"{sid}" standing for the slide ID, e.g. optimus={sid}.features.pt or
titan_slide=pt/{sid}.features.pt. Paths are relative to the working directory.
"""
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

PARTIAL_FAILURE_MARKER = "failed (the others' outputs were written)"
FEATURE_SUFFIXES = (".features.h5", ".features.pt", ".patch_features.h5", ".patch_features.pt")
EMPTY_SUFFIX = ".empty"


# ── run ──────────────────────────────────────────────────────────────────────

def _pump(src, dst, log, lock):
    for line in iter(src.readline, b""):
        dst.buffer.write(line)
        dst.flush()
        with lock:
            log.write(line)
            log.flush()
    src.close()


def run(log_path: str, command: list[str]) -> int:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("mussel_batch.py run: no command given", file=sys.stderr)
        return 2
    lock = threading.Lock()
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def forward(signum, _frame):
            proc.send_signal(signum)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2):
            signal.signal(sig, forward)
        threads = [
            threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, log, lock)),
            threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, log, lock)),
        ]
        for t in threads:
            t.start()
        rc = proc.wait()
        for t in threads:
            t.join()
    # A command killed by a signal: report it as the shell would (128 + signal).
    return 128 - rc if rc < 0 else rc


# ── finish ───────────────────────────────────────────────────────────────────

def _h5_rows(path: Path) -> int | None:
    """Rows in a features H5 (coords, else features); None if unreadable."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            if "coords" in f:
                return len(f["coords"])
            if "features" in f:
                return len(f["features"])
    except Exception:
        return None
    return None


def _pt_rows(path: Path) -> int | None:
    """Rows of a saved feature tensor; None if it can't be read (or torch is missing)."""
    try:
        import torch

        t = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not hasattr(t, "shape"):
        return None
    if t.numel() == 0:
        return 0
    return int(t.shape[0]) if len(t.shape) else 1


def _feature_stem(name: str) -> str | None:
    for suffix in FEATURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def quarantine_empty(scan_dirs: list[str]) -> list[Path]:
    """Rename 0-row feature files to <name>.empty; returns the original paths.

    A .pt pairs with the .h5 of the same name (in the same dir, or in the sibling
    h5/ dir of a pt/ dir) and is empty when that .h5 is; a .pt without an .h5 is
    loaded to count its rows. Unreadable files are kept.
    """
    files = []
    for d in scan_dirs:
        p = Path(d)
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.is_file() and _feature_stem(f.name) is not None))
    h5_rows = {}  # (dir holding the pt/h5 pair, name without .h5) -> rows
    for f in files:
        if f.name.endswith(".h5"):
            pair_root = f.parent.parent if f.parent.name == "h5" else f.parent
            h5_rows[(str(pair_root), f.name[: -len(".h5")])] = _h5_rows(f)
    moved = []
    for f in files:
        if f.name.endswith(".h5"):
            pair_root = f.parent.parent if f.parent.name == "h5" else f.parent
            rows = h5_rows[(str(pair_root), f.name[: -len(".h5")])]
        else:
            pair_root = f.parent.parent if f.parent.name == "pt" else f.parent
            key = (str(pair_root), f.name[: -len(".pt")])
            rows = h5_rows[key] if key in h5_rows else _pt_rows(f)
        if rows == 0:
            moved.append(f)
    for f in moved:
        f.rename(f.with_name(f.name + EMPTY_SUFFIX))
    return moved


def _failure_listing(log_text: str) -> str:
    """The "<id>: <reason>; ..." part of every partial-failure message in the log."""
    parts = []
    for line in log_text.splitlines():
        i = line.find(PARTIAL_FAILURE_MARKER)
        if i >= 0:
            parts.append(line[i + len(PARTIAL_FAILURE_MARKER):])
    return "\n".join(parts)


def _named_as_failed(sid: str, listing: str) -> bool:
    # extract_features names "<id>: reason"; multi-model passes "<id> (<MODEL>): reason".
    # Match the whole ID: "1001234" must not match inside "11001234: ...".
    return re.search(rf"(?:^|[\s;:]){re.escape(sid)}(?:: | \()", listing) is not None


def finish(rc: int, log_path: str, slide_ids: list[str], expects: list[str], report: str,
           scan_dirs: list[str]) -> int:
    try:
        log_text = Path(log_path).read_text(errors="replace")
    except OSError:
        log_text = ""
    moved = quarantine_empty(scan_dirs)
    empty_by_sid = {_feature_stem(f.name) for f in moved}
    listing = _failure_listing(log_text)

    rows = []
    present_by_model = {}
    for item in expects:
        model, _, template = item.partition("=")
        if not template:
            print(f"mussel_batch.py finish: --expect needs MODEL=TEMPLATE, got {item!r}", file=sys.stderr)
            return 2
        present_by_model[model] = 0
        for sid in slide_ids:
            if Path(template.replace("{sid}", sid)).is_file():
                present_by_model[model] += 1
                continue
            if _named_as_failed(sid, listing):
                status = "failed"
            elif sid in empty_by_sid:
                status = "no_tiles"
            else:
                status = "missing"
            rows.append((sid, model, status))

    if rows:
        with open(report, "w") as fh:
            fh.write("slide_id\tmodel\tstatus\n")
            for row in rows:
                fh.write("\t".join(row) + "\n")
    for f in moved:
        print(f"WARNING: {f} has no rows (slide without tiles); moved to {f.name}{EMPTY_SUFFIX}", file=sys.stderr)

    if rc == 0:
        return 0
    partial = PARTIAL_FAILURE_MARKER in log_text
    if partial and present_by_model and all(n > 0 for n in present_by_model.values()):
        failed = [f"{sid} ({model}): {status}" for sid, model, status in rows]
        print(f"WARNING: Mussel exited {rc} after writing the other slides' outputs; "
              f"continuing with them. Without outputs ({report}): {', '.join(failed)}", file=sys.stderr)
        return 0
    if partial:
        empty_models = [m for m, n in present_by_model.items() if n == 0]
        print(f"ERROR: Mussel exited {rc}; no outputs for {empty_models or 'any model'}, so this is "
              "treated as a failure, not a partial success", file=sys.stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run a command, copying its output to a log")
    p_run.add_argument("--log", required=True)
    p_run.add_argument("command", nargs=argparse.REMAINDER)
    p_fin = sub.add_parser("finish", help="settle a batch's outputs after the command")
    p_fin.add_argument("--rc", type=int, required=True)
    p_fin.add_argument("--log", required=True)
    p_fin.add_argument("--slide-ids", required=True, help="comma-separated slide IDs of the batch")
    p_fin.add_argument("--expect", action="append", default=[], metavar="MODEL=TEMPLATE")
    p_fin.add_argument("--scan", action="append", default=[], metavar="DIR",
                       help="directory whose empty feature files are moved aside (repeatable)")
    p_fin.add_argument("--report", default="failed_slides.tsv")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return run(args.log, args.command)
    slide_ids = [s for s in args.slide_ids.split(",") if s]
    return finish(args.rc, args.log, slide_ids, args.expect, args.report, args.scan)


if __name__ == "__main__":
    raise SystemExit(main())
