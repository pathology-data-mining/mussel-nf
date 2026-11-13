import hashlib
from unittest.mock import patch

from mussel_dispatcher import image_pins
from mussel_dispatcher.image_pins import pinned_images, verify_pinned_images


def _image(tmp_path, content: bytes, *, lie: bool = False):
    digest = hashlib.sha256(b"something else" if lie else content).hexdigest()
    path = tmp_path / f"mussel-1.6.0-sha256-{digest}.sif"
    path.write_bytes(content)
    return path, digest


def _config(tmp_path, *images):
    cfg = tmp_path / "site.config"
    cfg.write_text("\n".join(f"process.container = '{p}'" for p in images)
                   + "\nprocess.container = 'docker://mskmind/mussel@sha256:" + "a" * 64 + "'\n")
    return cfg


def test_pinned_images_finds_content_addressed_files_only(tmp_path):
    path, digest = _image(tmp_path, b"image")
    assert pinned_images([str(_config(tmp_path, path))]) == [(str(path), digest)]


def test_matching_image_passes_and_is_cached(tmp_path):
    path, _ = _image(tmp_path, b"image")
    cfg, cache = _config(tmp_path, path), tmp_path / "cache.json"
    assert verify_pinned_images([str(cfg)], cache) == []
    # A second start uses the cache: no re-hash of an unchanged file.
    with patch.object(image_pins, "_sha256", side_effect=AssertionError("re-hashed")):
        assert verify_pinned_images([str(cfg)], cache) == []


def test_mismatched_image_is_reported(tmp_path):
    path, digest = _image(tmp_path, b"rebuilt image", lie=True)
    problems = verify_pinned_images([str(_config(tmp_path, path))], tmp_path / "cache.json")
    assert len(problems) == 1 and digest in problems[0]


def test_changed_file_is_rehashed(tmp_path):
    path, _ = _image(tmp_path, b"image")
    cfg, cache = _config(tmp_path, path), tmp_path / "cache.json"
    assert verify_pinned_images([str(cfg)], cache) == []
    path.write_bytes(b"overwritten in place")
    assert len(verify_pinned_images([str(cfg)], cache)) == 1


def test_other_sites_images_and_missing_configs_are_skipped(tmp_path):
    cfg = tmp_path / "other_site.config"
    cfg.write_text(f"container = '/elsewhere/images/mussel-sha256-{'b' * 64}.sif'\n")
    assert verify_pinned_images([str(cfg), str(tmp_path / "absent.config"), ""], tmp_path / "c.json") == []
