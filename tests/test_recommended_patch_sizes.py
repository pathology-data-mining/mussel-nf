"""assets/recommended_patch_sizes.json must match Mussel's MODEL_PATCH_SIZES /
MODEL_TARGET_MPP / SLIDE_ENCODER_COMPATIBILITY.

mussel-nf tiles every model at this size unless explicitly overridden, so a
stale copy would silently tile a model at the wrong size; a stale slide-encoder
pairing would feed a slide encoder the wrong patch features.
"""
import json
import re
from pathlib import Path

import pytest

ASSET = Path(__file__).resolve().parents[1] / "assets" / "recommended_patch_sizes.json"
SCHEMA = Path(__file__).resolve().parents[1] / "nextflow_schema.json"


def _sizes():
    return json.loads(ASSET.read_text())["sizes"]


def test_every_selectable_model_has_a_size():
    schema = json.loads(SCHEMA.read_text())
    enum = schema["properties"]["featurize"]["properties"]["model_types"]["items"]["enum"]
    missing = sorted(set(enum) - set(_sizes()))
    assert not missing, f"models without a recommended tile size: {missing}"


def test_matches_pinned_mussel_ref():
    """The asset is bin/sync_patch_sizes.py's output for its _mussel_ref.

    Fetches Mussel's model_factory.py at that ref. Offline it skips locally,
    but fails in CI (CI=true) so a stale table can't pass unnoticed.
    """
    import importlib.util
    import os
    import urllib.error

    spec = importlib.util.spec_from_file_location("sync_patch_sizes", ASSET.parents[1] / "bin" / "sync_patch_sizes.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    ref = json.loads(ASSET.read_text())["_mussel_ref"]
    try:
        source = sync.fetch(ref)
    except (urllib.error.URLError, OSError) as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"cannot fetch Mussel {ref}: {exc}")
    assert sync.render(sync.parse(source), ref) == ASSET.read_text(), (
        f"assets/recommended_patch_sizes.json differs from Mussel {ref}; run bin/sync_patch_sizes.py --ref {ref}")


def test_parser_reads_model_factory_tables():
    import importlib.util

    spec = importlib.util.spec_from_file_location("sync_patch_sizes", ASSET.parents[1] / "bin" / "sync_patch_sizes.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    src = """
class ModelType(Enum):
    def __init__(self, id, code, path): pass
    UNI = 11, "uni", "hf-hub:MahmoodLab/UNI"
    TITAN_SLIDE = 30, "titan_slide", "MahmoodLab/TITAN"
MODEL_PATCH_SIZES = {ModelType.UNI: 256, ModelType.TITAN_SLIDE: 512}
MODEL_TARGET_MPP = {ModelType.TITAN_SLIDE: 0.5}
SLIDE_ENCODER_COMPATIBILITY = {ModelType.TITAN_SLIDE: ModelType.UNI}
"""
    assert sync.parse(src) == {"sizes": {"titan_slide": 512, "uni": 256}, "target_mpp": {"titan_slide": 0.5},
                               "slide_encoders": {"titan_slide": "uni"}}


def test_slide_to_patch_mapping_matches_mussel():
    """nextflow.config's slide_to_patch_mapping (and the schema defaults) pair each
    slide encoder with the patch encoder Mussel pairs it with."""
    mussel = json.loads(ASSET.read_text())["slide_encoders"]
    text = (ASSET.parents[1] / "nextflow.config").read_text()
    block = re.search(r"slide_to_patch_mapping\s*\{(.*?)\}", text, re.S).group(1)
    config = dict(re.findall(r"^\s*(\w+)\s*=\s*'(\w+)'", block, re.M))
    assert config == mussel
    schema = json.loads(SCHEMA.read_text())
    props = schema["properties"]["featurize"]["properties"]["slide_to_patch_mapping"]["properties"]
    assert {k: v["default"] for k, v in props.items()} == mussel
