"""Regression tests for the non-stub tessellation task templates."""

import re
import subprocess
from pathlib import Path


TESSELLATION_NF = Path(__file__).parents[1] / "modules" / "tessellation.nf"


def _extract_batch_script() -> str:
    text = TESSELLATION_NF.read_text()
    process = re.search(
        r"process TESSELLATE_BATCH\s*\{(.*?)\n\}\s*\nprocess ",
        text,
        re.DOTALL,
    )
    assert process, "TESSELLATE_BATCH process was not found"
    script = re.search(r"\n\s*script:\s*(.*?)\n\s*stub:", process.group(1), re.DOTALL)
    assert script, "TESSELLATE_BATCH script block was not found"
    template = re.search(r'"""(.*?)"""', script.group(1), re.DOTALL)
    assert template, "TESSELLATE_BATCH script template was not found"
    return template.group(1)


def _render(template: str) -> str:
    values = {
        "seg_config_str": "seg_config=biopsy",
        "seg_params_str": "seg_config.mpp=0.5 seg_config.seg_model=neural",
        "task.cpus": "1",
        "slide_paths_arg": '"1001234.svs","1001235.svs"',
        "slide_ids_arg": '"1001234","1001235"',
        "output_h5_paths_arg": '"1001234.patch.h5","1001235.patch.h5"',
        "neural_seg_max_tiles": "0",
        "expected_seg_model": "neural",
    }
    rendered = template
    for expression, value in sorted(values.items(), key=lambda item: -len(item[0])):
        rendered = rendered.replace("${" + expression + "}", value)
    return rendered


def test_tessellate_batch_script_is_valid_bash(tmp_path):
    script = _render(_extract_batch_script())
    script_path = tmp_path / "tessellate_batch.sh"
    script_path.write_text("#!/usr/bin/env bash\n" + script)

    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "export MUSSEL_NEURAL_SEG_MAX_TILES=0" in script
    assert "seg_config=biopsy" in script
    assert "seg_config.seg_model=neural" in script
    assert "--expected-seg-model=neural" in script
    assert "slide_paths=[\"1001234.svs\",\"1001235.svs\"]" in script
    assert "output_h5_paths=[\"1001234.patch.h5\",\"1001235.patch.h5\"]" in script


def test_tessellation_exports_configured_large_slide_limit():
    text = TESSELLATION_NF.read_text()
    assert text.count("export MUSSEL_NEURAL_SEG_MAX_TILES=") == 2
    assert "params.tiling.neural_seg_max_tiles != null" in text


def test_segmentation_model_overrides_named_presets():
    text = TESSELLATION_NF.read_text()
    assert text.count("if (params.tiling.seg_model != null)") == 2
    assert "!params.tiling.seg_config_group && params.tiling.seg_model" not in text
    assert text.count("validate_tessellation_h5.py") == 2
    combined = (TESSELLATION_NF.parent / "tessellate_featurize.nf").read_text()
    assert "if (params.tiling.seg_model != null)" in combined
    assert "validate_tessellation_h5.py" in combined


def test_tessellate_batch_multi_size_script_is_valid_bash(tmp_path):
    """Several tile sets: one segmentation, Mussel's {patch_size} placeholder in every output path."""
    template = _extract_batch_script()
    values = {
        "seg_config_str": "seg_config=biopsy",
        "seg_params_str": "seg_config.mpp=0.5 'seg_config.patch_sizes=[224,512]'",
        "task.cpus": "1",
        "slide_paths_arg": '"1001234.svs","1001235.svs"',
        "slide_ids_arg": '"1001234","1001235"',
        "output_h5_paths_arg": '"1001234.{patch_size}px.patch.h5","1001235.{patch_size}px.patch.h5"',
        "neural_seg_max_tiles": "0",
        "expected_seg_model": "neural",
    }
    script = template
    for expression, value in sorted(values.items(), key=lambda item: -len(item[0])):
        script = script.replace("${" + expression + "}", value)
    path = tmp_path / "tessellate_batch_multi.sh"
    path.write_text("#!/usr/bin/env bash\n" + script)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "'seg_config.patch_sizes=[224,512]'" in script
    assert 'output_h5_paths=["1001234.{patch_size}px.patch.h5","1001235.{patch_size}px.patch.h5"]' in script


def test_multi_size_helpers_present():
    text = TESSELLATION_NF.read_text()
    assert "seg_config.patch_sizes=[" in text           # one segmentation for all sizes
    assert ".{patch_size}px" in text                   # per-size output placeholder
    assert "seg_config.patch_mpps=[" in text           # per-set resolution, same segmentation
    assert ".{patch_size}px_{mpp}mpp" in text          # per-set placeholder when resolutions differ
    assert "def tilePublishPath" in text and "def tileSetOf" in text
