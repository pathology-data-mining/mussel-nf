import shutil
import subprocess
from pathlib import Path

import pytest


def test_tessellate_batch_skips_failed_slide(tmp_path):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is not installed")

    repo = Path(__file__).resolve().parents[1]
    slide = repo / "tests" / "testdata" / "948176.svs"
    if not slide.exists():
        pytest.skip("stub slide fixture is missing")

    slide_a = tmp_path / "slide_a.svs"
    slide_b = tmp_path / "slide_b.svs"
    slide_c = tmp_path / "slide_c.svs"
    slide_a.symlink_to(slide)
    slide_b.symlink_to(slide)
    slide_c.symlink_to(slide)

    samples_csv = tmp_path / "samples.csv"
    samples_csv.write_text(
        "slide_id,slide_path\n"
        f"948176,{slide_a}\n"
        f"FAIL001,{slide_b}\n"
        f"948178,{slide_c}\n"
    )

    outdir = tmp_path / "out"
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text(
        f"samples_csv: {samples_csv}\n"
        "tiling:\n"
        "  workflow_batch_size: 3\n"
        f"outdir: {outdir}\n"
    )

    result = subprocess.run(
        [
            "nextflow",
            "run",
            ".",
            "-profile",
            "test_stub_two_step",
            "-stub-run",
            "-params-file",
            str(params_yaml),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    assert (outdir / "tiles" / "9481" / "948176.patch.h5").exists()
    assert not (outdir / "tiles" / "FAIL" / "FAIL001.patch.h5").exists()
    assert (outdir / "tiles" / "9481" / "948178.patch.h5").exists()

    assert (outdir / "features" / "resnet50" / "948176.features.pt").exists()
    assert not (outdir / "features" / "resnet50" / "FAIL001.features.pt").exists()
    assert (outdir / "features" / "resnet50" / "948178.features.pt").exists()

    manifest_text = "\n".join(p.read_text() for p in outdir.glob("manifest-*.csv"))
    assert "948176,948176,reef,tiles_h5_path,tiles/9481/948176.patch.h5" in manifest_text
    assert "FAIL001" not in manifest_text
    assert "948178,948178,reef,tiles_h5_path,tiles/9481/948178.patch.h5" in manifest_text
