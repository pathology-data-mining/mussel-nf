"""mussel-dispatcher package exports."""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "Config", "RemoteTargetConfig", "WatcherConfig", "StoragePolicyConfig",
    "StateStore",
    "ReadinessChecker", "LocalWatcher", "S3Watcher", "DatabricksWatcher", "TcgaWatcher",
    "NextflowRunner", "collect_manifests", "MANIFEST_HEADER",
    "BatchScheduler", "RunManager", "recover_in_flight", "main",
]

_EXPORTS = {
    "Config": ("mussel_dispatcher.config", "Config"),
    "RemoteTargetConfig": ("mussel_dispatcher.config", "RemoteTargetConfig"),
    "WatcherConfig": ("mussel_dispatcher.config", "WatcherConfig"),
    "StoragePolicyConfig": ("mussel_dispatcher.config", "StoragePolicyConfig"),
    "StateStore": ("mussel_dispatcher.state", "StateStore"),
    "ReadinessChecker": ("mussel_dispatcher.watchers", "ReadinessChecker"),
    "LocalWatcher": ("mussel_dispatcher.watchers", "LocalWatcher"),
    "S3Watcher": ("mussel_dispatcher.watchers", "S3Watcher"),
    "DatabricksWatcher": ("mussel_dispatcher.watchers", "DatabricksWatcher"),
    "TcgaWatcher": ("mussel_dispatcher.watchers", "TcgaWatcher"),
    "NextflowRunner": ("mussel_dispatcher.runner", "NextflowRunner"),
    "collect_manifests": ("mussel_dispatcher.runner", "collect_manifests"),
    "MANIFEST_HEADER": ("mussel_dispatcher.runner", "MANIFEST_HEADER"),
    "BatchScheduler": ("mussel_dispatcher.scheduler", "BatchScheduler"),
    "RunManager": ("mussel_dispatcher.scheduler", "RunManager"),
    "recover_in_flight": ("mussel_dispatcher.scheduler", "recover_in_flight"),
    "main": ("mussel_dispatcher.scheduler", "main"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
