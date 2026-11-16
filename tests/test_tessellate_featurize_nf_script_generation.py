"""Run TESSELLATE_FEATURIZE_BATCH's real task script with a fake tessellate_extract_features.

Nextflow stub tests bypass ``script:`` blocks. These tests render the GString
(escaped ``\\$`` become ``$``; other ``${...}`` take the values below, or "")
and run it with ``bash -ue`` as Nextflow does, checking how it handles Mussel
>= 1.6.0 output layouts and partial batch failures.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NF = REPO / "modules" / "tessellate_featurize.nf"


def _script_template() -> str:
    text = NF.read_text()
    script = re.search(r"\n\s*script:\s*(.*?)\n\s*stub:", text, re.DOTALL)
    assert script, "TESSELLATE_FEATURIZE_BATCH script block was not found"
    template = re.search(r'"""(.*?)"""', script.group(1), re.DOTALL)
    assert template, "TESSELLATE_FEATURIZE_BATCH script template was not found"
    return template.group(1)


def _render(template: str, values: dict) -> str:
    out = template.replace("\\$", "\x00")
    for expr, value in sorted(values.items(), key=lambda kv: -len(kv[0])):
        out = out.replace("${" + expr + "}", value)
    out = re.sub(r"\$\{[^}]*\}", "", out)  # options this test leaves unset
    return out.replace("\x00", "$")


def _values(slide_model: bool, expected_seg_model: str = "") -> dict:
    return {
        "slide_ids_str": "S1,S2",
        "slide_paths_str": "S1.svs,S2.svs",
        "task.cpus": "1",
        "neural_seg_max_tiles": "0",
        "model_type.toUpperCase()": "CTRANSPATH",
        "model_type_name.toUpperCase()": "CHIEF_SLIDE" if slide_model else "CTRANSPATH",
        "patch_encoder_name.toUpperCase()": "CTRANSPATH",
        "is_slide_model": "true" if slide_model else "false",
        "model_type_input": "chief_slide" if slide_model else "ctranspath",
        "expected_seg_model": expected_seg_model,
    }


def _run(tmp_path, values, fake_body):
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    fake = fake_bin / "tessellate_extract_features"
    fake.write_text("#!/usr/bin/env bash\nset -e\n" + fake_body)
    fake.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    script = work / ".command.sh"
    script.write_text(_render(_script_template(), values))
    return work, subprocess.run(
        ["bash", "-ue", str(script)],
        cwd=work,
        env={**os.environ, "PATH": f"{fake_bin}:{REPO / 'bin'}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )


# Mussel 1.6.0, chief_slide (multi-model mode): CHIEF's tiles (256 px at 1.0 µm/px)
# differ from CTransPath's own (224 px at 0.5), so CHIEF's patch features go to
# CTRANSPATH_256px_1mpp/, and CTRANSPATH/ holds the standalone CTransPath run.
def _chief_layout(standalone: bool) -> str:
    body = "mkdir -p CHIEF_SLIDE/pt CHIEF_SLIDE/h5 CTRANSPATH_256px_1mpp/pt CTRANSPATH_256px_1mpp/h5 CTRANSPATH_256px_1mpp/tile_h5\n"
    if standalone:
        body += "mkdir -p CTRANSPATH/pt CTRANSPATH/h5 CTRANSPATH/tile_h5\n"
    body += "for s in S1 S2; do\n"
    body += "  echo slide > CHIEF_SLIDE/pt/$s.features.pt; echo slide > CHIEF_SLIDE/h5/$s.features.h5\n"
    body += "  for k in pt h5; do echo chief > CTRANSPATH_256px_1mpp/$k/$s.features.$k; done\n"
    body += "  echo chief > CTRANSPATH_256px_1mpp/tile_h5/$s.patch.h5\n"
    if standalone:
        body += "  for k in pt h5; do echo standalone > CTRANSPATH/$k/$s.features.$k; done\n"
        body += "  echo standalone > CTRANSPATH/tile_h5/$s.patch.h5\n"
    body += "done\n"
    return body


@pytest.mark.parametrize("standalone", [True, False])
def test_slide_encoder_patch_features_from_size_tagged_dir(tmp_path, standalone):
    work, result = _run(tmp_path, _values(slide_model=True), _chief_layout(standalone))
    assert result.returncode == 0, result.stderr
    for s in ("S1", "S2"):
        assert (work / "pt" / f"{s}.features.pt").read_text().strip() == "slide"
        assert (work / "pt" / f"{s}.patch_features.pt").read_text().strip() == "chief"
        assert (work / "h5" / f"{s}.patch_features.h5").read_text().strip() == "chief"
        assert (work / "tile_h5" / f"{s}.patch.h5").read_text().strip() == "chief"
    assert not (work / "failed_slides.tsv").exists()


def test_validator_skipped_without_tile_files(tmp_path):
    """Only slides with tiles have a tile H5: with none, don't validate a literal glob."""
    body = "mkdir -p pt h5\nfor s in S1 S2; do touch pt/$s.features.pt h5/$s.features.h5; done\n"
    work, result = _run(tmp_path, _values(slide_model=False, expected_seg_model="neural"), body)
    assert result.returncode == 0, result.stderr
    assert "Invalid tessellation metadata" not in result.stderr


def test_partial_batch_failure_keeps_the_other_slides(tmp_path):
    body = (
        "mkdir -p pt h5 tile_h5\ntouch pt/S1.features.pt h5/S1.features.h5\n"
        "echo \"RuntimeError: 1 of 2 slide(s) failed (the others' outputs were written): "
        "S2: tessellation failed: boom\" >&2\nexit 1\n"
    )
    work, result = _run(tmp_path, _values(slide_model=False), body)
    assert result.returncode == 0, result.stderr
    assert (work / "failed_slides.tsv").read_text().splitlines() == [
        "slide_id\tmodel\tstatus", "S2\tctranspath\tfailed"]


def test_slide_model_failing_on_every_slide_fails_the_task(tmp_path):
    """Multi-model mode: the patch encoder's outputs exist, but none of the slide encoder's."""
    body = (
        "mkdir -p CHIEF_SLIDE CTRANSPATH/pt\ntouch CTRANSPATH/pt/S1.features.pt CTRANSPATH/pt/S2.features.pt\n"
        "echo \"RuntimeError: 1 of 2 model(s) failed (the others' outputs were written): "
        "CHIEF_SLIDE: CUDA out of memory\" >&2\nexit 1\n"
    )
    _work, result = _run(tmp_path, _values(slide_model=True), body)
    assert result.returncode == 1
