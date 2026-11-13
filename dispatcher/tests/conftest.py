import pytest

from mussel_dispatcher import runner as runner_mod


@pytest.fixture(autouse=True)
def _short_shutdown_exit_grace(monkeypatch):
    """Failing fake runs would otherwise each wait the full grace for a shutdown."""
    monkeypatch.setattr(runner_mod, "_SHUTDOWN_EXIT_GRACE_SECONDS", 0.05)
