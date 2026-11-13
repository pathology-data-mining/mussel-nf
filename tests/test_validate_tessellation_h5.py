import subprocess
import sys
from pathlib import Path

import h5py


SCRIPT = Path(__file__).parents[1] / "bin" / "validate_tessellation_h5.py"


def _write_h5(path: Path, seg_model=None):
    with h5py.File(path, "w") as handle:
        coords = handle.create_dataset("coords", data=[[0, 0]])
        if seg_model is not None:
            coords.attrs["seg_model"] = seg_model


def _run(expected: str, *paths: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--expected-seg-model={expected}",
            *(str(path) for path in paths),
        ],
        capture_output=True,
        text=True,
    )


def test_accepts_matching_segmentation_metadata(tmp_path):
    output = tmp_path / "neural.patch.h5"
    _write_h5(output, "neural")

    result = _run("NEURAL", output)

    assert result.returncode == 0, result.stderr


def test_rejects_wrong_or_missing_segmentation_metadata(tmp_path):
    classic = tmp_path / "classic.patch.h5"
    missing = tmp_path / "missing.patch.h5"
    _write_h5(classic, "classic")
    _write_h5(missing)

    result = _run("neural", classic, missing)

    assert result.returncode == 1
    assert "seg_model=classic, expected neural" in result.stderr
    assert "seg_model=<missing>, expected neural" in result.stderr


def test_rejects_missing_or_malformed_h5(tmp_path):
    malformed = tmp_path / "malformed.patch.h5"
    malformed.write_text("not hdf5")

    result = _run("neural", malformed, tmp_path / "absent.patch.h5")

    assert result.returncode == 1
    assert result.stderr.count("unreadable H5") == 2
