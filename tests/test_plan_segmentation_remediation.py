import csv
import importlib.util
import json
import os
from pathlib import Path

from mussel_dispatcher import StateStore


SCRIPT = (
    Path(__file__).parents[1]
    / "dispatcher"
    / "scripts"
    / "plan_segmentation_remediation.py"
)
SPEC = importlib.util.spec_from_file_location("plan_segmentation_remediation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_loads_unique_affected_ids_and_reports_all_database_rows(tmp_path):
    manifest = tmp_path / "affected.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "status"])
        writer.writeheader()
        writer.writerows(
            [
                {"slide_id": "A", "status": "mismatch"},
                {"slide_id": "A", "status": "mismatch"},
                {"slide_id": "B", "status": "error"},
                {"slide_id": "C", "status": "matching"},
            ]
        )
    state = StateStore(str(tmp_path / "state.db"))
    state.add_slide("/one/A.svs", "A")
    state.add_slide("/two/A.svs", "A")
    state.add_slide("/one/B.svs", "B")

    ids = MODULE.load_affected_ids(manifest)
    rows = MODULE.query_rows(state, ids)

    assert ids == ["A", "B"]
    assert [(row["slide_id"], row["slide_path"]) for row in rows] == [
        ("A", "/one/A.svs"),
        ("A", "/two/A.svs"),
        ("B", "/one/B.svs"),
    ]


def test_main_loads_configured_database_secret_without_overriding_shell_value(
    tmp_path, monkeypatch, capsys
):
    secret_file = tmp_path / "database.env"
    secret_file.write_text("POSTGRES_APP_PASSWORD=from-file\n")
    config_file = tmp_path / "dispatcher.yaml"
    config_file.write_text(
        "nextflow_profiles: test\n"
        "outdir: results\n"
        "secrets_env_file: database.env\n"
    )
    manifest = tmp_path / "affected.csv"
    manifest.write_text("slide_id,status\nA,mismatch\n")
    output = tmp_path / "rows.csv"
    captured = {}

    class FakeState:
        def close_thread_connection(self):
            captured["closed"] = True

    def fake_from_config(cfg):
        captured["password"] = os.environ.get("POSTGRES_APP_PASSWORD")
        return FakeState()

    monkeypatch.setenv("POSTGRES_APP_PASSWORD", "from-shell")
    monkeypatch.setattr(MODULE.StateStore, "from_config", fake_from_config)
    monkeypatch.setattr(MODULE, "query_rows", lambda state, slide_ids: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--config",
            str(config_file),
            "--audit-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    assert MODULE.main() == 0
    assert captured == {"password": "from-shell", "closed": True}
    summary = json.loads(capsys.readouterr().out.splitlines()[0])
    assert summary["affected_ids"] == 1
    assert summary["duplicate_status_patterns"] == {}
    assert summary["ids_by_status"] == {
        "DISPATCHED": 0,
        "FAILED": 0,
        "PENDING": 0,
        "SUCCEEDED": 0,
    }
