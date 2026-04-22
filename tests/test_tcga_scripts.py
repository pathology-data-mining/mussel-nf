"""End-to-end tests for the scripts/tcga/ pipeline.

Covers the three core data-processing steps executed by tcga_run.py:

    1. tcga_update_status  — scan a results directory → status CSV
    2. tcga_prepare_samples — resolve paths (local → S3 → needs_download)
    3. tcga_append_wds      — write per-cancer WDS shards from .pt outputs

Each test uses only in-memory / tmp-dir data; no network calls are made.
S3 listing in tcga_prepare_samples is monkey-patched to return a fixed set
of file_ids so the tests are fully hermetic.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

INVENTORY_ROWS = [
    # file_id, file_name, project_id, slide_type, file_size, md5sum
    (
        "aaaa0000-0000-0000-0000-000000000001",
        "TCGA-BR-A44T-01Z-00-DX1.1A2B3C4D-0000-0000-0000-000000000000.svs",
        "TCGA-BRCA",
        "DX1",
        1_000_000,
        "abc123",
    ),
    (
        "bbbb0000-0000-0000-0000-000000000002",
        "TCGA-BR-A44U-01Z-00-DX1.2B3C4D5E-0000-0000-0000-000000000000.svs",
        "TCGA-BRCA",
        "DX1",
        2_000_000,
        "def456",
    ),
    (
        "cccc0000-0000-0000-0000-000000000003",
        "TCGA-LU-A5YX-01Z-00-DX1.3C4D5E6F-0000-0000-0000-000000000000.svs",
        "TCGA-LUAD",
        "DX1",
        3_000_000,
        "ghi789",
    ),
    (
        "dddd0000-0000-0000-0000-000000000004",
        "TCGA-LU-A5YY-01Z-00-DX1.4D5E6F7G-0000-0000-0000-000000000000.svs",
        "TCGA-LUAD",
        "DX1",
        4_000_000,
        "jkl012",
    ),
]

INVENTORY_COLUMNS = [
    "file_id", "file_name", "project_id", "slide_type", "file_size", "md5sum",
]


def make_inventory() -> pd.DataFrame:
    return pd.DataFrame(INVENTORY_ROWS, columns=INVENTORY_COLUMNS).astype(str)


def make_pt_file(path: Path, n_patches: int = 8, dim: int = 512) -> None:
    """Write a minimal .features.pt file that tcga_append_wds can load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.randn(n_patches, dim)
    torch.save(tensor, path)


def make_h5_file(path: Path, n_patches: int = 8) -> None:
    """Write a minimal .patch.h5 file with a 'coords' dataset."""
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=np.random.randint(0, 1000, (n_patches, 2)))


# ---------------------------------------------------------------------------
# 1. tcga_update_status — build_status()
# ---------------------------------------------------------------------------

class TestUpdateStatus:
    """build_status() scans a results dir and emits pending/done per slide×model."""

    def test_marks_slides_done_when_pt_exists(self, tmp_path):
        from scripts.tcga.tcga_update_status import build_status

        # Slides 1 & 3 have completed; slides 2 & 4 are pending
        model = "ctranspath"
        pt_dir = tmp_path / "features" / model / "pt"
        make_pt_file(pt_dir / "TCGA-BR-A44T-01Z-00-DX1.features.pt")
        make_pt_file(pt_dir / "TCGA-LU-A5YX-01Z-00-DX1.features.pt")

        inventory = make_inventory()
        status = build_status(inventory, results_dir=tmp_path, model_types=[model])

        done = status[status["status"] == "done"]
        pending = status[status["status"] == "pending"]

        assert set(done["slide_id"]) == {"TCGA-BR-A44T-01Z-00-DX1", "TCGA-LU-A5YX-01Z-00-DX1"}
        assert set(pending["slide_id"]) == {"TCGA-BR-A44U-01Z-00-DX1", "TCGA-LU-A5YY-01Z-00-DX1"}

    def test_all_pending_when_no_results(self, tmp_path):
        from scripts.tcga.tcga_update_status import build_status

        inventory = make_inventory()
        status = build_status(inventory, results_dir=tmp_path, model_types=["ctranspath"])

        assert (status["status"] == "pending").all()
        assert len(status) == len(INVENTORY_ROWS)

    def test_multiple_models_independent(self, tmp_path):
        from scripts.tcga.tcga_update_status import build_status

        for model in ("ctranspath", "uni2h"):
            pt_dir = tmp_path / "features" / model / "pt"
            make_pt_file(pt_dir / "TCGA-BR-A44T-01Z-00-DX1.features.pt")

        inventory = make_inventory()
        status = build_status(
            inventory, results_dir=tmp_path, model_types=["ctranspath", "uni2h"]
        )

        assert len(status) == len(INVENTORY_ROWS) * 2  # 4 slides × 2 models
        # Each model sees exactly 1 done slide
        for model in ("ctranspath", "uni2h"):
            sub = status[status["model"] == model]
            assert (sub["status"] == "done").sum() == 1

    def test_pt_path_recorded(self, tmp_path):
        from scripts.tcga.tcga_update_status import build_status

        model = "ctranspath"
        pt_file = tmp_path / "features" / model / "pt" / "TCGA-BR-A44T-01Z-00-DX1.features.pt"
        make_pt_file(pt_file)

        inventory = make_inventory()
        status = build_status(inventory, results_dir=tmp_path, model_types=[model])

        row = status[status["slide_id"] == "TCGA-BR-A44T-01Z-00-DX1"].iloc[0]
        assert Path(row["pt_path"]) == pt_file


# ---------------------------------------------------------------------------
# 2. tcga_prepare_samples — prepare_samples()
# ---------------------------------------------------------------------------

# File_ids present on S3 (slides 2 & 3; slide 1 is local, slide 4 needs download)
S3_FILE_IDS = {
    "bbbb0000-0000-0000-0000-000000000002",
    "cccc0000-0000-0000-0000-000000000003",
}


class TestPrepareSamples:
    """prepare_samples() resolves paths: local → S3 → needs_download."""

    @pytest.fixture()
    def local_slides_dir(self, tmp_path):
        """Create a local slides dir with slide 1 already present (correct size)."""
        file_id = "aaaa0000-0000-0000-0000-000000000001"
        file_name = "TCGA-BR-A44T-01Z-00-DX1.1A2B3C4D-0000-0000-0000-000000000000.svs"
        slide_path = tmp_path / file_id / file_name
        slide_path.parent.mkdir(parents=True)
        # Write exactly file_size bytes so partial-download detection passes
        slide_path.write_bytes(b"\x00" * 1_000_000)
        return tmp_path

    def test_local_path_resolved(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        inventory = make_inventory()
        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=S3_FILE_IDS,
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                local_slides_dir=local_slides_dir,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
            )

        local_row = result[result["slide_id"] == "TCGA-BR-A44T-01Z-00-DX1"].iloc[0]
        assert not local_row["needs_download"]
        expected = str(
            local_slides_dir
            / "aaaa0000-0000-0000-0000-000000000001"
            / "TCGA-BR-A44T-01Z-00-DX1.1A2B3C4D-0000-0000-0000-000000000000.svs"
        )
        assert local_row["slide_path"] == expected

    def test_s3_path_for_available_slides(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        inventory = make_inventory()
        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=S3_FILE_IDS,
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                local_slides_dir=local_slides_dir,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
            )

        # Slides 2 & 3 are on S3 but not local → should get s3:// paths
        for slide_id, file_id, file_name in [
            ("TCGA-BR-A44U-01Z-00-DX1", "bbbb0000-0000-0000-0000-000000000002",
             "TCGA-BR-A44U-01Z-00-DX1.2B3C4D5E-0000-0000-0000-000000000000.svs"),
            ("TCGA-LU-A5YX-01Z-00-DX1", "cccc0000-0000-0000-0000-000000000003",
             "TCGA-LU-A5YX-01Z-00-DX1.3C4D5E6F-0000-0000-0000-000000000000.svs"),
        ]:
            row = result[result["slide_id"] == slide_id].iloc[0]
            assert not row["needs_download"], f"{slide_id} should not need download"
            assert row["slide_path"] == f"s3://pathology/TCGA/{file_id}/{file_name}"

    def test_needs_download_for_missing_slides(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        inventory = make_inventory()
        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=S3_FILE_IDS,
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                local_slides_dir=local_slides_dir,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
            )

        # Slide 4 is neither local nor on S3
        row = result[result["slide_id"] == "TCGA-LU-A5YY-01Z-00-DX1"].iloc[0]
        assert row["needs_download"]

    def test_skip_done_excludes_completed_slides(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        inventory = make_inventory()
        status = pd.DataFrame([
            {"slide_id": "TCGA-BR-A44T-01Z-00-DX1", "model": "ctranspath", "status": "done"},
        ])

        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=S3_FILE_IDS,
        ):
            result = prepare_samples(
                inventory,
                status_df=status,
                model="ctranspath",
                local_slides_dir=local_slides_dir,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
            )

        assert "TCGA-BR-A44T-01Z-00-DX1" not in result["slide_id"].values
        assert len(result) == 3

    def test_partial_download_flagged_as_needs_download(self, tmp_path):
        """A local file whose size doesn't match inventory is re-flagged."""
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        file_id = "aaaa0000-0000-0000-0000-000000000001"
        file_name = "TCGA-BR-A44T-01Z-00-DX1.1A2B3C4D-0000-0000-0000-000000000000.svs"
        slide_path = tmp_path / file_id / file_name
        slide_path.parent.mkdir(parents=True)
        slide_path.write_bytes(b"\x00" * 512)  # partial: only 512 bytes, expected 1_000_000

        inventory = make_inventory()
        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=set(),  # not on S3 either
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                local_slides_dir=tmp_path,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
            )

        row = result[result["slide_id"] == "TCGA-BR-A44T-01Z-00-DX1"].iloc[0]
        assert row["needs_download"], "partial download should be flagged"

    def test_limit_respected(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        inventory = make_inventory()
        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=S3_FILE_IDS,
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                local_slides_dir=local_slides_dir,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
                limit=2,
            )

        assert len(result) == 2

    def test_slide_type_filter(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        # Add a TS1 slide to inventory
        extra = pd.DataFrame([{
            "file_id": "eeee0000-0000-0000-0000-000000000005",
            "file_name": "TCGA-BR-A44V-01Z-00-TS1.FFFFFFFF.svs",
            "project_id": "TCGA-BRCA",
            "slide_type": "TS1",
            "file_size": "500000",
            "md5sum": "xyz",
        }])
        inventory = pd.concat([make_inventory(), extra], ignore_index=True).astype(str)

        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=set(),
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                slide_type_filter="DX",
            )

        assert all(
            row["slide_id"] != "TCGA-BR-A44V-01Z-00-TS1"
            for _, row in result.iterrows()
        ), "TS1 slide should be filtered out by DX prefix filter"


# ---------------------------------------------------------------------------
# 3. tcga_append_wds — append_wds()
# ---------------------------------------------------------------------------

class TestAppendWds:
    """append_wds() writes per-cancer-type WDS tar shards and an index JSON."""

    @pytest.fixture()
    def results_dir(self, tmp_path):
        """Create a results dir with .pt (and optional .h5) files for 3 slides."""
        model = "ctranspath"
        pt_dir = tmp_path / "features" / model / "pt"
        h5_dir = tmp_path / "features" / model / "tile_h5"

        # TCGA-BRCA: 2 slides
        make_pt_file(pt_dir / "TCGA-BR-A44T-01Z-00-DX1.features.pt")
        make_h5_file(h5_dir / "TCGA-BR-A44T-01Z-00-DX1.patch.h5")
        make_pt_file(pt_dir / "TCGA-BR-A44U-01Z-00-DX1.features.pt")

        # TCGA-LUAD: 1 slide (no h5)
        make_pt_file(pt_dir / "TCGA-LU-A5YX-01Z-00-DX1.features.pt")

        return tmp_path, model, pt_dir, h5_dir

    def test_shards_created_per_cancer_type(self, tmp_path, results_dir):
        from scripts.tcga.tcga_append_wds import append_wds

        res_dir, model, pt_dir, h5_dir = results_dir
        wds_dest = str(tmp_path / "wds")

        index = append_wds(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=make_inventory(),
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        # One shard per cancer type
        brca_shard = Path(wds_dest) / model / "TCGA-BRCA" / "000000.tar"
        luad_shard = Path(wds_dest) / model / "TCGA-LUAD" / "000000.tar"
        assert brca_shard.exists(), "BRCA shard not created"
        assert luad_shard.exists(), "LUAD shard not created"

    def test_index_has_correct_entries(self, tmp_path, results_dir):
        from scripts.tcga.tcga_append_wds import append_wds

        res_dir, model, pt_dir, h5_dir = results_dir
        wds_dest = str(tmp_path / "wds")

        index = append_wds(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=make_inventory(),
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        assert set(index.keys()) == {
            "TCGA-BR-A44T-01Z-00-DX1", "TCGA-BR-A44U-01Z-00-DX1", "TCGA-LU-A5YX-01Z-00-DX1"
        }
        assert index["TCGA-BR-A44T-01Z-00-DX1"]["project_id"] == "TCGA-BRCA"
        assert index["TCGA-LU-A5YX-01Z-00-DX1"]["project_id"] == "TCGA-LUAD"

    def test_index_json_persisted(self, tmp_path, results_dir):
        from scripts.tcga.tcga_append_wds import append_wds

        res_dir, model, pt_dir, h5_dir = results_dir
        wds_dest = str(tmp_path / "wds")

        append_wds(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=make_inventory(),
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        index_path = Path(wds_dest) / model / "wds_index.json"
        assert index_path.exists()
        saved = json.loads(index_path.read_text())
        assert len(saved) == 3

    def test_shard_contains_features_npy(self, tmp_path, results_dir):
        from scripts.tcga.tcga_append_wds import append_wds

        res_dir, model, pt_dir, h5_dir = results_dir
        wds_dest = str(tmp_path / "wds")

        append_wds(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=make_inventory(),
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        brca_shard = Path(wds_dest) / model / "TCGA-BRCA" / "000000.tar"
        with tarfile.open(brca_shard) as tf:
            members = [m.name for m in tf.getmembers()]

        feature_entries = [m for m in members if m.endswith(".features.npy")]
        assert len(feature_entries) == 2  # 2 BRCA slides

    def test_h5_coords_embedded_in_shard(self, tmp_path, results_dir):
        from scripts.tcga.tcga_append_wds import append_wds

        res_dir, model, pt_dir, h5_dir = results_dir
        wds_dest = str(tmp_path / "wds")

        append_wds(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=make_inventory(),
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        brca_shard = Path(wds_dest) / model / "TCGA-BRCA" / "000000.tar"
        with tarfile.open(brca_shard) as tf:
            members = [m.name for m in tf.getmembers()]

        coords_entries = [m for m in members if m.endswith(".coords.npy")]
        # Only TCGA-BR-A44T-01 has a .h5 file → 1 coords entry
        assert len(coords_entries) == 1

    def test_idempotent_second_run(self, tmp_path, results_dir):
        """Running append_wds twice must not duplicate entries."""
        from scripts.tcga.tcga_append_wds import append_wds

        res_dir, model, pt_dir, h5_dir = results_dir
        wds_dest = str(tmp_path / "wds")
        kwargs = dict(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=make_inventory(),
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        index1 = append_wds(**kwargs)
        index2 = append_wds(**kwargs)

        assert len(index2) == len(index1) == 3


# ---------------------------------------------------------------------------
# 4. End-to-end pipeline: status → prepare → append
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Drive all three stages in sequence with shared fixtures."""

    def test_full_pipeline(self, tmp_path):
        from scripts.tcga.tcga_update_status import build_status
        from scripts.tcga.tcga_prepare_samples import prepare_samples
        from scripts.tcga.tcga_append_wds import append_wds

        model = "ctranspath"
        inventory = make_inventory()

        # --- Stage 1: simulate partial results (2 of 4 slides done) ---
        pt_dir = tmp_path / "results" / "features" / model / "pt"
        h5_dir = tmp_path / "results" / "features" / model / "tile_h5"
        make_pt_file(pt_dir / "TCGA-BR-A44T-01Z-00-DX1.features.pt")
        make_h5_file(h5_dir / "TCGA-BR-A44T-01Z-00-DX1.patch.h5")
        make_pt_file(pt_dir / "TCGA-LU-A5YX-01Z-00-DX1.features.pt")

        status = build_status(
            inventory, results_dir=tmp_path / "results", model_types=[model]
        )
        assert (status["status"] == "done").sum() == 2
        assert (status["status"] == "pending").sum() == 2

        # --- Stage 2: prepare samples for the 2 pending slides ---
        # Slide 2 (BRCA) is on S3; slide 4 (LUAD) needs download
        s3_file_ids = {"bbbb0000-0000-0000-0000-000000000002"}
        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=s3_file_ids,
        ):
            samples = prepare_samples(
                inventory,
                status_df=status,
                model=model,
                s3_base="s3://pathology/TCGA",
                check_s3_exists=True,
            )

        assert len(samples) == 2
        s3_row = samples[samples["slide_id"] == "TCGA-BR-A44U-01Z-00-DX1"].iloc[0]
        dl_row = samples[samples["slide_id"] == "TCGA-LU-A5YY-01Z-00-DX1"].iloc[0]
        assert not s3_row["needs_download"]
        assert s3_row["slide_path"].startswith("s3://")
        assert dl_row["needs_download"]

        # --- Stage 3: append the 2 completed slides to WDS ---
        wds_dest = str(tmp_path / "wds")
        index = append_wds(
            pt_dir=pt_dir,
            h5_dir=h5_dir,
            inventory_df=inventory,
            wds_dest=wds_dest,
            model_type=model,
            staging_dir=None,
            max_shard_bytes=500 * 1024 * 1024,
        )

        assert len(index) == 2
        assert Path(wds_dest, model, "wds_index.json").exists()
        # Each completed slide lands in its correct cancer-type shard
        assert index["TCGA-BR-A44T-01Z-00-DX1"]["project_id"] == "TCGA-BRCA"
        assert index["TCGA-LU-A5YX-01Z-00-DX1"]["project_id"] == "TCGA-LUAD"
