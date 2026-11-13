from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from mussel_dispatcher import scheduler


def test_startup_guards_log_why_they_refuse(monkeypatch, caplog):
    cfg = MagicMock()
    caplog.set_level("INFO", logger="mussel-dispatcher")
    monkeypatch.setattr("mussel_dispatcher.image_pins.verify_pinned_images", lambda configs: [])
    monkeypatch.setattr("mussel_dispatcher.release_pin.verify_release_pin",
                        lambda c: (None, "abc is on origin/release/x"))
    assert scheduler._startup_guards(cfg) is True
    assert "Release pin: abc is on origin/release/x" in caplog.text

    monkeypatch.setattr("mussel_dispatcher.release_pin.verify_release_pin",
                        lambda c: ("revision abc is not on origin/release/x; deploy by updating release/x", ""))
    assert scheduler._startup_guards(cfg) is False
    assert "Refusing to start: revision abc is not on origin/release/x" in caplog.text

    monkeypatch.setattr("mussel_dispatcher.image_pins.verify_pinned_images", lambda configs: ["x.sif: sha256 differs"])
    assert scheduler._startup_guards(cfg) is False
    assert "Pinned container image mismatch: x.sif: sha256 differs" in caplog.text


def test_guards_run_after_the_log_file_is_attached_and_before_the_pid_lock(tmp_path, monkeypatch):
    # A refusal must reach dispatcher.log (not only the service journal) and must
    # stop the dispatcher before it takes its PID lock.
    order = []
    cfg = MagicMock(tower_endpoint="x", work_base_dir=str(tmp_path / "w"), dispatch_dir=str(tmp_path / "d"),
                    state_dir=str(tmp_path / "s"), log_dir=str(tmp_path / "l"))
    monkeypatch.setattr(sys, "argv", ["mussel_dispatcher", str(tmp_path / "cfg.yaml")])
    monkeypatch.setattr(scheduler.Config, "load", staticmethod(lambda path: cfg))
    monkeypatch.setattr(scheduler, "_enable_persistent_dispatcher_logging",
                        lambda c: order.append("log file") or str(tmp_path / "dispatcher.log"))
    monkeypatch.setattr(scheduler, "_startup_guards", lambda c: order.append("guards") or False)
    monkeypatch.setattr(scheduler, "_acquire_pid_lock", lambda *a, **k: order.append("pid lock"))
    monkeypatch.setattr(scheduler, "_find_same_config_dispatchers", lambda *a, **k: order.append("rogue check") or [])

    with pytest.raises(SystemExit) as exit_info:
        scheduler.main()

    assert exit_info.value.code == 1
    assert order == ["log file", "guards"]
