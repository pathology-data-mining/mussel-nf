"""
Test that featurize.nf Groovy GString script blocks generate valid bash.

Motivation
----------
NF stub tests bypass the `script:` block entirely — Groovy GString
template bugs are invisible to stub-mode CI. This test simulates Groovy's
GString evaluation and checks the result with `bash -n` (syntax check only,
no execution).

The specific regression caught: `$(cat .sub_patch_h5_paths)` in a Groovy
GString was parsed as a Groovy expression instead of bash command substitution,
corrupting all subsequent variable interpolations and producing bash syntax
errors (exit:2) in every FEATURIZE_BATCH task.

The fix was `\\$(cat ...)` (escaped `$`), which passes through to bash as
the literal `$(cat ...)` command substitution.
"""
import re
import subprocess
import tempfile
from pathlib import Path

FEATURIZE_NF = Path(__file__).parent.parent / "modules" / "featurize.nf"

# ---------------------------------------------------------------------------
# Groovy GString simulation
# ---------------------------------------------------------------------------
# Test values that approximate real pipeline variable values for a titan_slide batch
TEST_VARS = {
    # Groovy variables in the script: block
    "patch_h5_paths_str":        "1001234.patch.h5,1001235.patch.h5",
    "slide_paths_str":           "1001234.svs,1001235.svs",
    "slide_ids_str":             "1001234,1001235",
    "slide_batch_size":          "4",
    "batch_size":                "64",
    "mtype.toUpperCase()":       "CONCH1_5",
    "mpath_str":                 "",
    "slide_model_str":           "slide_model_type=TITAN_SLIDE",
    "aggregation_str":           "aggregation_method=model",
    "embedding_precision_str":   "",
    "params.featurize.use_gpu ? \"true\" : \"false\"": "true",
    "params.featurize.slide_max_patches ?: 0":         "4096",
    # task meta
    "task.cpus":                 "8",
}


def _simulate_groovy_gstring(template: str, variables: dict[str, str]) -> str:
    """Simulate Groovy GString evaluation.

    Rules:
    - ``\\${var}`` → literal ``${var}`` in bash  (escaped from Groovy)
    - ``\\$(``     → literal ``$(``  in bash      (escaped bash command sub)
    - ``${expr}``  → replaced with variables[expr] or empty string
    """
    result = template

    # Step 1: protect escaped sequences by replacing them with placeholders
    result = result.replace(r"\${", "\x00ESCAPED_BRACE\x00")
    result = result.replace(r"\$(", "\x00ESCAPED_DOLLAR_PAREN\x00")

    # Step 2: replace ${expr} with test values (longest match first to avoid
    # partial substitution of nested expressions)
    for expr, value in sorted(variables.items(), key=lambda kv: -len(kv[0])):
        result = result.replace("${" + expr + "}", value)

    # Step 3: restore escaped sequences as their bash equivalents
    result = result.replace("\x00ESCAPED_BRACE\x00", "${")
    result = result.replace("\x00ESCAPED_DOLLAR_PAREN\x00", "$(")

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
        """hoptimus1 / optimus path: slide_model_str is empty, subsampling block skipped."""
        script_template = self._get_script()
        vars_ = {**TEST_VARS, "slide_model_str": "", "mtype.toUpperCase()": "HOPTIMUS1",
                 "aggregation_str": "aggregation_method=identity"}
        script = _simulate_groovy_gstring(script_template, vars_)
        ok, stderr = self._bash_syntax_ok(script)
        assert ok, f"bash -n failed for patch encoder (hoptimus1) script:\n{stderr}\n\nGenerated script:\n{script[:500]}"

    def test_slide_encoder_titan_script_is_valid_bash(self):
        """titan_slide path: slide_model_str is set, H5 subsampling block is entered."""
        script_template = self._get_script()
        vars_ = {**TEST_VARS, "slide_model_str": "slide_model_type=TITAN_SLIDE",
                 "mtype.toUpperCase()": "CONCH1_5", "aggregation_str": "aggregation_method=model"}
        script = _simulate_groovy_gstring(script_template, vars_)
        ok, stderr = self._bash_syntax_ok(script)
        assert ok, (
            f"bash -n failed for titan_slide (CONCH1_5) script — likely Groovy GString "
            f"corruption (e.g. unescaped $(cmd) in template):\n{stderr}\n\n"
            f"Generated script:\n{script[:800]}"
        )

    def test_no_unescaped_dollar_paren_in_script_block(self):
        """Ensure the script: block has no bare $(cmd) patterns that Groovy would corrupt.

        Any bash command substitution inside a Groovy GString MUST be written as
        \\$(cmd) to escape the $ from Groovy's GString parser.
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
