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
    # file_id, file_name, project_id, slide_type, file_size, md5sum, primary_site, disease_type,
    # gender, age_at_index, vital_status, primary_diagnosis, ajcc_pathologic_stage,
    # sample_type, percent_tumor_cells, first_seen_date, removed_date
    (
        "aaaa0000-0000-0000-0000-000000000001",
        "TCGA-BR-A44T-01Z-00-DX1.1A2B3C4D-0000-0000-0000-000000000000.svs",
        "TCGA-BRCA", "DX1", 1_000_000, "abc123",
        "Breast", "Breast Invasive Carcinoma",
        "female", "52", "Alive", "Infiltrating duct carcinoma, NOS", "Stage IIA",
        "Primary Tumor", "60.0",
        "2024-01-01", "",
    ),
    (
        "bbbb0000-0000-0000-0000-000000000002",
        "TCGA-BR-A44U-01Z-00-DX1.2B3C4D5E-0000-0000-0000-000000000000.svs",
        "TCGA-BRCA", "DX1", 2_000_000, "def456",
        "Breast", "Breast Invasive Carcinoma",
        "female", "67", "Dead", "Lobular carcinoma, NOS", "Stage IIIA",
        "Primary Tumor", "75.0",
        "2024-01-01", "",
    ),
    (
        "cccc0000-0000-0000-0000-000000000003",
        "TCGA-LU-A5YX-01Z-00-DX1.3C4D5E6F-0000-0000-0000-000000000000.svs",
        "TCGA-LUAD", "DX1", 3_000_000, "ghi789",
        "Lung", "Lung Adenocarcinoma",
        "male", "71", "Dead", "Adenocarcinoma, NOS", "Stage IB",
        "Primary Tumor", "55.0",
        "2024-01-01", "",
    ),
    (
        "dddd0000-0000-0000-0000-000000000004",
        "TCGA-LU-A5YY-01Z-00-DX1.4D5E6F7G-0000-0000-0000-000000000000.svs",
        "TCGA-LUAD", "DX1", 4_000_000, "jkl012",
        "Lung", "Lung Adenocarcinoma",
        "male", "58", "Alive", "Adenocarcinoma, NOS", "Stage IA",
        "Primary Tumor", "70.0",
        "2024-01-01", "",
    ),
]

INVENTORY_COLUMNS = [
    "file_id", "file_name", "project_id", "slide_type", "file_size", "md5sum",
    "primary_site", "disease_type",
    "gender", "age_at_index", "vital_status", "primary_diagnosis", "ajcc_pathologic_stage",
    "sample_type", "percent_tumor_cells",
    "first_seen_date", "removed_date",
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
# 0. tcga_sync_inventory — _parse_hit()
# ---------------------------------------------------------------------------

class TestParseHit:
    """_parse_hit() correctly extracts all metadata fields from a GDC API hit."""

    def _make_hit(self, **overrides):
        hit = {
            "file_id": "aaaa-0001",
            "file_name": "TCGA-BR-A44T-01Z-00-DX1.uuid.svs",
            "file_size": 500_000_000,
            "md5sum": "abc123",
            "updated_datetime": "2021-10-14T18:00:00",
            "cases": [{
                "submitter_id": "TCGA-BR-A44T",
                "project": {
                    "project_id": "TCGA-BRCA",
                    "primary_site": "Breast",
                    "disease_type": "Breast Invasive Carcinoma",
                },
                "demographic": {
                    "gender": "female",
                    "age_at_index": 63,
                    "vital_status": "Alive",
                    "race": "white",
                    "ethnicity": "not hispanic or latino",
                },
                "diagnoses": [{
                    "primary_diagnosis": "Infiltrating duct carcinoma, NOS",
                    "morphology": "8500/3",
                    "ajcc_pathologic_stage": "Stage IA",
                    "tumor_grade": None,
                }],
                "samples": [{
                    "sample_type": "Primary Tumor",
                    "tissue_type": "Tumor",
                    "tumor_descriptor": "Primary",
                    "portions": [{"slides": [{
                        "section_location": "TOP",
                        "percent_tumor_cells": 50.0,
                        "percent_stromal_cells": 35.0,
                        "percent_necrosis": 10.0,
                        "percent_normal_cells": 5.0,
                    }]}],
                }],
            }],
        }
        hit.update(overrides)
        return hit

    def test_all_columns_present(self):
        from scripts.tcga.tcga_sync_inventory import _parse_hit, GDC_COLUMNS
        row = _parse_hit(self._make_hit())
        assert set(row.keys()) == set(GDC_COLUMNS)

    def test_core_fields(self):
        from scripts.tcga.tcga_sync_inventory import _parse_hit
        row = _parse_hit(self._make_hit())
        assert row["file_id"] == "aaaa-0001"
        assert row["project_id"] == "TCGA-BRCA"
        assert row["slide_type"] == "DX1"
        assert row["file_size"] == 500_000_000

    def test_metadata_fields(self):
        from scripts.tcga.tcga_sync_inventory import _parse_hit
        row = _parse_hit(self._make_hit())
        assert row["primary_site"] == "Breast"
        assert row["disease_type"] == "Breast Invasive Carcinoma"
        assert row["gender"] == "female"
        assert row["age_at_index"] == 63
        assert row["vital_status"] == "Alive"
        assert row["primary_diagnosis"] == "Infiltrating duct carcinoma, NOS"
        assert row["ajcc_pathologic_stage"] == "Stage IA"
        assert row["sample_type"] == "Primary Tumor"
        assert row["percent_tumor_cells"] == 50.0
        assert row["percent_stromal_cells"] == 35.0

    def test_missing_optional_fields_default_empty(self):
        from scripts.tcga.tcga_sync_inventory import _parse_hit
        # Minimal hit — no demographic, diagnoses, samples
        hit = {
            "file_id": "x",
            "file_name": "TCGA-XX-0001-01Z-00-DX1.uuid.svs",
            "file_size": 0,
            "md5sum": "",
            "updated_datetime": "",
            "cases": [{"submitter_id": "TCGA-XX-0001",
                        "project": {"project_id": "TCGA-TEST"}}],
        }
        row = _parse_hit(hit)
        assert row["gender"] == ""
        assert row["ajcc_pathologic_stage"] == ""
        assert row["percent_tumor_cells"] == ""



# ---------------------------------------------------------------------------
# 0b. tcga_sync_inventory — merge_inventory()
# ---------------------------------------------------------------------------

def _make_gdc_row(file_id: str, project_id: str = "TCGA-BRCA",
                  updated: str = "2024-01-01T00:00:00") -> dict:
    """Minimal GDC-column row (no temporal columns) for merge tests."""
    from scripts.tcga.tcga_sync_inventory import GDC_COLUMNS
    base = {c: "" for c in GDC_COLUMNS}
    base.update({"file_id": file_id, "project_id": project_id, "updated_datetime": updated})
    return base


class TestMergeInventory:
    """merge_inventory() correctly tracks new, removed, and re-appearing slides."""

    def _old(self, rows: list[dict]) -> pd.DataFrame:
        from scripts.tcga.tcga_sync_inventory import INVENTORY_COLUMNS
        return pd.DataFrame(rows, columns=INVENTORY_COLUMNS).astype(str).fillna("")

    def _new(self, rows: list[dict]) -> pd.DataFrame:
        from scripts.tcga.tcga_sync_inventory import GDC_COLUMNS
        return pd.DataFrame(rows, columns=GDC_COLUMNS).astype(str).fillna("")

    def test_new_slide_gets_first_seen_date(self):
        from scripts.tcga.tcga_sync_inventory import merge_inventory
        old = self._old([{**_make_gdc_row("aaa"), "first_seen_date": "2024-01-01", "removed_date": ""}])
        new = self._new([_make_gdc_row("aaa"), _make_gdc_row("bbb")])
        merged, added, _, _ = merge_inventory(old, new, "2024-06-01")
        bbb = merged[merged["file_id"] == "bbb"].iloc[0]
        assert bbb["first_seen_date"] == "2024-06-01"
        assert bbb["removed_date"] == ""
        assert len(added) == 1 and added.iloc[0]["file_id"] == "bbb"

    def test_removed_slide_gets_removed_date(self):
        from scripts.tcga.tcga_sync_inventory import merge_inventory
        old = self._old([
            {**_make_gdc_row("aaa"), "first_seen_date": "2024-01-01", "removed_date": ""},
            {**_make_gdc_row("bbb"), "first_seen_date": "2024-01-01", "removed_date": ""},
        ])
        new = self._new([_make_gdc_row("aaa")])  # bbb is gone
        merged, _, removed, _ = merge_inventory(old, new, "2024-06-01")
        bbb = merged[merged["file_id"] == "bbb"].iloc[0]
        assert bbb["removed_date"] == "2024-06-01"
        assert len(removed) == 1 and removed.iloc[0]["file_id"] == "bbb"

    def test_already_removed_not_double_stamped(self):
        from scripts.tcga.tcga_sync_inventory import merge_inventory
        old = self._old([
            {**_make_gdc_row("aaa"), "first_seen_date": "2024-01-01", "removed_date": ""},
            {**_make_gdc_row("bbb"), "first_seen_date": "2024-01-01", "removed_date": "2024-03-01"},
        ])
        new = self._new([_make_gdc_row("aaa")])  # bbb still absent
        merged, _, removed, _ = merge_inventory(old, new, "2024-06-01")
        bbb = merged[merged["file_id"] == "bbb"].iloc[0]
        assert bbb["removed_date"] == "2024-03-01"  # original stamp preserved
        assert len(removed) == 0  # not newly removed this run

    def test_reappearing_slide_clears_removed_date(self):
        from scripts.tcga.tcga_sync_inventory import merge_inventory
        old = self._old([
            {**_make_gdc_row("aaa"), "first_seen_date": "2024-01-01", "removed_date": "2024-03-01"},
        ])
        new = self._new([_make_gdc_row("aaa")])  # aaa is back
        merged, added, removed, _ = merge_inventory(old, new, "2024-06-01")
        aaa = merged[merged["file_id"] == "aaa"].iloc[0]
        assert aaa["removed_date"] == ""
        assert aaa["first_seen_date"] == "2024-01-01"  # preserved
        assert len(added) == 0  # not counted as brand-new
        assert len(removed) == 0

    def test_first_seen_date_preserved_on_update(self):
        from scripts.tcga.tcga_sync_inventory import merge_inventory
        old = self._old([
            {**_make_gdc_row("aaa", updated="2024-01-01T00:00:00"),
             "first_seen_date": "2023-11-01", "removed_date": ""},
        ])
        new = self._new([_make_gdc_row("aaa", updated="2024-05-01T00:00:00")])
        merged, _, _, updated = merge_inventory(old, new, "2024-06-01")
        aaa = merged[merged["file_id"] == "aaa"].iloc[0]
        assert aaa["first_seen_date"] == "2023-11-01"
        assert len(updated) == 1

    def test_snapshot_query(self):
        """Simulate point-in-time snapshot: first_seen_date <= T AND removed_date empty or > T."""
        from scripts.tcga.tcga_sync_inventory import merge_inventory
        old = self._old([
            {**_make_gdc_row("aaa"), "first_seen_date": "2024-01-01", "removed_date": ""},
            {**_make_gdc_row("bbb"), "first_seen_date": "2024-01-01", "removed_date": "2024-04-01"},
        ])
        new = self._new([_make_gdc_row("aaa"), _make_gdc_row("ccc")])
        merged, _, _, _ = merge_inventory(old, new, "2024-06-01")

        # Snapshot at 2024-03-01: aaa + bbb (bbb removed in April)
        T = "2024-03-01"
        snap = merged[(merged["first_seen_date"] <= T) &
                      ((merged["removed_date"] == "") | (merged["removed_date"] > T))]
        assert set(snap["file_id"]) == {"aaa", "bbb"}

        # Snapshot at today: aaa + ccc
        T = "2024-06-01"
        snap = merged[(merged["first_seen_date"] <= T) &
                      ((merged["removed_date"] == "") | (merged["removed_date"] > T))]
        assert set(snap["file_id"]) == {"aaa", "ccc"}


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

    def test_sample_type_filter_excludes_normal(self, local_slides_dir):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        extra = pd.DataFrame([{
            "file_id": "ffff0000-0000-0000-0000-000000000006",
            "file_name": "TCGA-HV-AA8V-11Z-00-DX1.AAAAAAAA.svs",
            "project_id": "TCGA-PAAD",
            "slide_type": "DX1",
            "sample_type": "Solid Tissue Normal",
            "file_size": "500000",
            "md5sum": "zzz",
        }])
        inventory = pd.concat([make_inventory(), extra], ignore_index=True).fillna("").astype(str)

        with patch(
            "scripts.tcga.tcga_prepare_samples._list_s3_file_ids",
            return_value=set(),
        ):
            result = prepare_samples(
                inventory,
                status_df=None,
                sample_type_filter="Primary Tumor",
            )

        assert all(
            row["slide_id"] != "TCGA-HV-AA8V-11Z-00-DX1"
            for _, row in result.iterrows()
        ), "Normal tissue slide should be excluded by sample_type filter"

    def test_sample_type_filter_tumor_substring(self):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        rows = [
            {"file_id": "a1", "file_name": "TCGA-XX-0001-01Z-00-DX1.A.svs",
             "slide_type": "DX1", "sample_type": "Primary Tumor",
             "project_id": "TCGA-BRCA", "file_size": "100", "md5sum": "a"},
            {"file_id": "a2", "file_name": "TCGA-XX-0002-06Z-00-DX1.B.svs",
             "slide_type": "DX1", "sample_type": "Metastatic",
             "project_id": "TCGA-BRCA", "file_size": "100", "md5sum": "b"},
            {"file_id": "a3", "file_name": "TCGA-XX-0003-11Z-00-DX1.C.svs",
             "slide_type": "DX1", "sample_type": "Solid Tissue Normal",
             "project_id": "TCGA-BRCA", "file_size": "100", "md5sum": "c"},
        ]
        inventory = pd.DataFrame(rows).fillna("").astype(str)

        result = prepare_samples(inventory, status_df=None,
                                 sample_type_filter="tumor,metastatic")

        ids = set(result["slide_id"])
        assert "TCGA-XX-0001-01Z-00-DX1" in ids
        assert "TCGA-XX-0002-06Z-00-DX1" in ids
        assert "TCGA-XX-0003-11Z-00-DX1" not in ids, "Normal should be excluded"

    def test_sample_type_filter_all_disables_filter(self):
        from scripts.tcga.tcga_prepare_samples import prepare_samples

        rows = [
            {"file_id": "b1", "file_name": "TCGA-YY-0001-01Z-00-DX1.A.svs",
             "slide_type": "DX1", "sample_type": "Primary Tumor",
             "project_id": "TCGA-BRCA", "file_size": "100", "md5sum": "a"},
            {"file_id": "b2", "file_name": "TCGA-YY-0002-11Z-00-DX1.B.svs",
             "slide_type": "DX1", "sample_type": "Solid Tissue Normal",
             "project_id": "TCGA-BRCA", "file_size": "100", "md5sum": "b"},
        ]
        inventory = pd.DataFrame(rows).fillna("").astype(str)

        result = prepare_samples(inventory, status_df=None, sample_type_filter="all")
        assert len(result) == 2, "sample_type='all' should not filter anything"


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
