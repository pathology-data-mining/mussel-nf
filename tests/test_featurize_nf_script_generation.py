"""Validate the real FEATURIZE_BATCH task script, not only its stub path.

Nextflow stub tests bypass ``script:`` blocks. These tests render the GString
with representative values, check Bash syntax, and run the rendered script with a
stand-in extract_features to check the arguments it gets.
"""
import re
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

FEATURIZE_NF = Path(__file__).parent.parent / "modules" / "featurize.nf"
REPO_BIN = Path(__file__).parent.parent / "bin"

# ---------------------------------------------------------------------------
# Groovy GString simulation
# ---------------------------------------------------------------------------
# Test values that approximate real pipeline variable values for a titan_slide batch
TEST_VARS = {
    "patch_h5_paths_arg":        '"1001234.patch.h5","1001235.patch.h5"',
    "slide_paths_arg":           '"1001234.svs","1001235.svs"',
    "slide_ids_arg":             '"1001234","1001235"',
    "slide_ids_csv":             "1001234,1001235",
    "model_type_input":          "titan_slide",
    "max_slide_patches_str":     "max_slide_patches=2",
    "tile_size_check_str":       "",
    "slide_batch_size":          "4",
    "batch_size":                "64",
    "mtype.toUpperCase()":       "CONCH1_5",
    "mpath_str":                 "",
    "slide_model_str":           "slide_model_type=TITAN_SLIDE",
    "aggregation_str":           "aggregation_method=model",
    "embedding_precision_str":   "embedding_precision=bfloat16",
    "TaskProfile.script(params.profile_dir, task)": "",
    "TaskProfile.timingsArg(params.profile_dir)": "'++timings_path=.task_timings.jsonl'",
    "params.featurize.use_gpu ? \"true\" : \"false\"": "true",
    # task meta
    "task.cpus":                 "8",
}


def _simulate_groovy_gstring(template: str, variables: dict[str, str]) -> str:
    """Simulate Groovy GString evaluation.

    Rules:
    - ``\\$``      → literal ``$`` in bash (escaped from Groovy: ``\\${var}``,
                    ``\\$(cmd)``, ``\\$?``, ...)
    - ``${expr}``  → replaced with variables[expr] or empty string
    """
    result = template

    # Step 1: protect escaped sequences by replacing them with placeholders
    result = result.replace(r"\${", "\x00ESCAPED_BRACE\x00")
    result = result.replace(r"\$(", "\x00ESCAPED_DOLLAR_PAREN\x00")
    result = result.replace(r"\$", "\x00ESCAPED_DOLLAR\x00")

    # Step 2: replace ${expr} with test values (longest match first to avoid
    # partial substitution of nested expressions)
    for expr, value in sorted(variables.items(), key=lambda kv: -len(kv[0])):
        result = result.replace("${" + expr + "}", value)

    # Step 3: restore escaped sequences as their bash equivalents
    result = result.replace("\x00ESCAPED_BRACE\x00", "${")
    result = result.replace("\x00ESCAPED_DOLLAR_PAREN\x00", "$(")
    result = result.replace("\x00ESCAPED_DOLLAR\x00", "$")

    return result


def _extract_script_block(nf_text: str) -> str | None:
    """Extract the bash template (triple-quoted string) from the script: section."""
    # The script: block has Groovy variable assignments before the triple-quoted bash string.
    # Pattern: find the triple-quoted string that comes after `script:` and before `stub:`.
    script_section = re.search(r'\bscript:(.*?)\bstub:', nf_text, re.DOTALL)
    if not script_section:
        return None
    section = script_section.group(1)
    # Extract the triple-quoted string from within the section
    match = re.search(r'"""(.*?)"""', section, re.DOTALL)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeaturizeNfScriptGeneration:
    """Validate that featurize.nf script: block generates valid bash."""

    def _get_script(self) -> str:
        nf_text = FEATURIZE_NF.read_text()
        script = _extract_script_block(nf_text)
        assert script is not None, "Could not extract script: block triple-quoted string from featurize.nf"
        return script

    def _bash_syntax_ok(self, script: str) -> tuple[bool, str]:
        """Run bash -n on the script; return (ok, stderr)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="test_featurize_"
        ) as f:
            f.write("#!/bin/bash -ue\n")
            f.write(script)
            tmppath = f.name
        result = subprocess.run(
            ["bash", "-n", tmppath], capture_output=True, text=True
        )
        Path(tmppath).unlink(missing_ok=True)
        return result.returncode == 0, result.stderr

    def test_patch_encoder_script_is_valid_bash(self):
        """Patch encoders skip the slide-level subsampling branch."""
        script_template = self._get_script()
        vars_ = {
            **TEST_VARS,
            "max_slide_patches_str": "",
            "slide_model_str": "",
            "mtype.toUpperCase()": "HOPTIMUS1",
            "aggregation_str": "aggregation_method=identity",
            "embedding_precision_str": "",
        }
        script = _simulate_groovy_gstring(script_template, vars_)
        ok, stderr = self._bash_syntax_ok(script)
        assert ok, f"bash -n failed for patch encoder (hoptimus1) script:\n{stderr}\n\nGenerated script:\n{script[:500]}"

    def test_slide_encoder_titan_script_is_valid_bash(self):
        """TITAN path renders the subsampling and precision controls."""
        script_template = self._get_script()
        vars_ = {**TEST_VARS, "slide_model_str": "slide_model_type=TITAN_SLIDE",
                 "mtype.toUpperCase()": "CONCH1_5", "aggregation_str": "aggregation_method=model"}
        script = _simulate_groovy_gstring(script_template, vars_)
        ok, stderr = self._bash_syntax_ok(script)
        assert ok, (
            f"bash -n failed for titan_slide (CONCH1_5) script:\n{stderr}\n\n"
            f"Generated script:\n{script[:800]}"
        )

    def _run_task(self, tmp_path, vars_, fake_body):
        """Run the rendered script with a fake extract_features (bash `fake_body`)."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        extract_features = fake_bin / "extract_features"
        extract_features.write_text("#!/usr/bin/env bash\n" + fake_body)
        extract_features.chmod(0o755)
        script_path = tmp_path / "task.sh"
        script_path.write_text(_simulate_groovy_gstring(self._get_script(), vars_))
        return subprocess.run(
            ["bash", "-ue", str(script_path)],  # Nextflow's default task shell
            cwd=tmp_path,
            env={"PATH": f"{fake_bin}:{REPO_BIN}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
        )

    def _run_with_stub_extract(self, tmp_path, vars_):
        result = self._run_task(
            tmp_path, vars_,
            "printf '%s\\n' \"$*\" > extract_args.txt\n"
            "touch 1001234.features.pt 1001235.features.pt\n",
        )
        assert result.returncode == 0, result.stderr
        return (tmp_path / "extract_args.txt").read_text()

    # Mussel >= 1.6.0: some slides fail, the others' outputs are written, exit 1.
    PARTIAL = (
        "touch 1001234.features.pt 1001234.features.h5\n"
        "echo \"RuntimeError: 1 of 2 slide(s) failed (the others' outputs were written): "
        "1001235: CUDA out of memory\" >&2\n"
        "exit 1\n"
    )

    def test_partial_batch_failure_keeps_the_other_slides(self, tmp_path):
        result = self._run_task(tmp_path, TEST_VARS, self.PARTIAL)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "1001234.features.pt").exists()
        assert (tmp_path / "failed_slides.tsv").read_text().splitlines() == [
            "slide_id\tmodel\tstatus", "1001235\ttitan_slide\tfailed"]
        assert "continuing with them" in result.stderr
        # Mussel's own log still reaches the task's stderr.
        assert "1001235: CUDA out of memory" in result.stderr

    def test_failure_without_outputs_fails_the_task(self, tmp_path):
        result = self._run_task(
            tmp_path, TEST_VARS,
            "echo \"RuntimeError: 2 of 2 slide(s) failed (the others' outputs were written): x\" >&2\nexit 1\n",
        )
        assert result.returncode == 1

    @pytest.mark.parametrize("fake, rc", [
        ("touch 1001234.features.pt\nexit 137\n", 137),  # e.g. OOM kill after one slide
        ("echo 'RuntimeError: none of the 2 slide(s) had tiles, so no output files were written' >&2\n"
         "exit 1\n", 1),
    ])
    def test_other_failures_keep_their_exit_status(self, tmp_path, fake, rc):
        assert self._run_task(tmp_path, TEST_VARS, fake).returncode == rc

    def test_titan_cap_is_passed_to_mussel(self, tmp_path):
        """slide_max_patches reaches Mussel's extract_features as max_slide_patches."""
        args = self._run_with_stub_extract(tmp_path, TEST_VARS)
        assert 'patch_h5_paths=["1001234.patch.h5","1001235.patch.h5"]' in args
        assert "max_slide_patches=2" in args
        assert "embedding_precision=bfloat16" in args

    def test_no_unescaped_dollar_paren_in_script_block(self):
        """Ensure Bash command substitution is escaped from Groovy.

        Any Bash command substitution inside a Groovy GString MUST be written as
        \\$(cmd) to escape the ``$`` from Groovy's GString parser.
        """
        nf_text = FEATURIZE_NF.read_text()
        block = _extract_script_block(nf_text)
        assert block is not None, "Could not find script: block in featurize.nf"
        # Find any $(  that is NOT preceded by a backslash
        unescaped = re.findall(r'(?<!\\)\$\(', block)
        assert not unescaped, (
            f"Found {len(unescaped)} unescaped '$(' in featurize.nf script: block. "
            "Groovy GStrings interpret $(cmd) as a Groovy expression, corrupting "
            "all subsequent variable interpolations. Use \\\\$(cmd) instead."
        )
