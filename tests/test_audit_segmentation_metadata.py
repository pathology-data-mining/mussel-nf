import csv
import subprocess
import sys
from pathlib import Path

import h5py


SCRIPT = (
    Path(__file__).parents[1]
    / "dispatcher"
    / "scripts"
    / "audit_segmentation_metadata.py"
)


def _write(path: Path, model=None):
    with h5py.File(path, "w") as handle:
        coords = handle.create_dataset("coords", data=[[0, 0]])
        if model is not None:
            coords.attrs["seg_model"] = model


def test_reports_only_mismatches_and_errors(tmp_path):
    root = tmp_path / "tiles"
    root.mkdir()
    _write(root / "neural.patch.h5", "neural")
    _write(root / "classic.patch.h5", b"classic")
    _write(root / "missing.patch.h5")
    (root / "broken.patch.h5").write_text("broken")
    output = tmp_path / "affected.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--h5-root={root}",
            f"--output={output}",
            "--expected-seg-model=neural",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["slide_id"] for row in rows} == {"broken", "classic", "missing"}
    assert '"matching": 1' in result.stdout
    assert '"mismatch": 2' in result.stdout
    assert '"error": 1' in result.stdout
