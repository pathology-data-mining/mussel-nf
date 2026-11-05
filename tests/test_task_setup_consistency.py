"""Task environment setup lives in nextflow.config, not in the modules.

The model caches (Hugging Face, Matplotlib, XDG) and HDF5 locking are set in the
config's ``env`` scope, which Nextflow exports in every task and passes into
Apptainer/Docker containers. A process-level ``beforeScript`` would duplicate it
and also shadow a profile's ``process.beforeScript`` (e.g. the one that puts
$MUSSEL_VENV/bin on PATH), because a directive in the process definition takes
precedence over the generic ``process`` scope of the config.
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODULES = ROOT / "modules"
CONFIG = ROOT / "nextflow.config"


def _processes(text):
    """(name, body) of each top-level process in a module."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^process (\w+) \{", text, re.M)]
    for k, (pos, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(text)
        yield name, text[pos:end]


def _env_scope(text):
    """Variables set in the top-level ``env { ... }`` block of the config."""
    match = re.search(r"^env \{\n(.*?)^\}", text, re.M | re.S)
    assert match, "nextflow.config has no top-level env { } block"
    return dict(re.findall(r"^\s+(\w+) = (.+)$", match.group(1), re.M))


def test_no_module_process_sets_up_its_own_environment():
    offenders = [
        f"{nf.name}:{name}"
        for nf in sorted(MODULES.glob("*.nf"))
        for name, body in _processes(nf.read_text())
        if re.search(r"^    (beforeScript|afterScript)\b", body, re.M)
        or "APPTAINERENV_" in body
    ]
    assert not offenders, offenders


def test_config_env_scope_keeps_task_caches_in_the_work_dir():
    text = CONFIG.read_text()
    env = _env_scope(text)
    assert env["HDF5_USE_FILE_LOCKING"] == "'FALSE'"
    # Single-quoted so $PWD expands in the task (its work dir), not at config time.
    for name in ("HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        assert env[name].startswith("'$PWD"), (name, env[name])
    # Hugging Face: the shared params.hf_cache_dir if set, else the work dir.
    for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        assert env[name].startswith("params.hf_cache_dir ?"), (name, env[name])
        assert re.search(r": '\$PWD/\.task-cache/hf\b", env[name]), (name, env[name])
    # After the profiles, so a profile's params.hf_cache_dir reaches it.
    assert text.index("\nenv {") > text.index("\nprofiles {")
    # PYTHONPATH before them, so the test profiles' override (projectDir is tests/) wins.
    assert "PYTHONPATH" not in env
    assert text.index('\nenv.PYTHONPATH = "${projectDir}/lib"') < text.index("\nprofiles {")


def test_mussel_model_tasks_are_profiled():
    """Every process running a Mussel model CLI carries the opt-in profiling (lib/TaskProfile.groovy)."""
    found = {}
    for nf in sorted(MODULES.glob("*.nf")):
        for name, body in _processes(nf.read_text()):
            script = body.split("\n    stub:")[0]
            # The CLI runs directly or under bin/mussel_batch.py (Mussel >= 1.6.0 partial batches).
            if re.search(r"^\s+(?:mussel_batch\.py run --log \S+ -- )?(tessellate|extract_features|tessellate_extract_features) \\", script, re.M):
                found[f"{nf.name}:{name}"] = (
                    "${TaskProfile.script(params.profile_dir, task)}" in script,
                    "${TaskProfile.timingsArg(params.profile_dir)}" in script,
                )
    assert len(found) >= 5, found  # TESSELLATE, TESSELLATE_BATCH, FEATURIZE_BATCH, FEATURIZE_PATCH_MULTI, TESSELLATE_FEATURIZE_BATCH
    assert all(all(v) for v in found.values()), found
