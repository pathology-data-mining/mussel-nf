#!/usr/bin/env python3
"""
mussel-dispatcher.py — Streaming slide dispatcher for mussel-nf.

Watches slide sources for new WSI files, accumulates them into batches,
and dispatches each batch as a parallel `nextflow run` subprocess.
Optionally runs post-batch hooks (e.g. WDS shard append) after each
successful run.

SUBCOMMANDS
-----------
  run (default)
    python mussel-dispatcher.py <config.yaml>

  collect-manifests
    python mussel-dispatcher.py collect-manifests <config.yaml>

  help
    python mussel-dispatcher.py --help
"""
# Re-export all public symbols for backward compatibility with
# importlib-based importers (test_dispatcher.py, dashboard.py).
from dispatcher_config import (  # noqa: F401
    WatcherConfig,
    Config,
    _read_nf_model_types,
)
from dispatcher_state import StateStore  # noqa: F401
from dispatcher_watchers import (  # noqa: F401
    ReadinessChecker,
    LocalWatcher,
    S3Watcher,
    DatabricksWatcher,
    TcgaWatcher,
)
from dispatcher_runner import (  # noqa: F401
    NextflowRunner,
    collect_manifests,
    MANIFEST_HEADER,
)
from dispatcher_scheduler import (  # noqa: F401
    BatchScheduler,
    RunManager,
    recover_in_flight,
    main,
)

if __name__ == "__main__":
    main()
