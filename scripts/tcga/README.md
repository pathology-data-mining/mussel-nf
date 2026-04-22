# TCGA Feature Extraction Pipeline

Automates end-to-end feature extraction for all ~30,000 TCGA whole-slide images
using the [mussel-nf](../../README.md) Nextflow pipeline.

## Contents

- [Overview](#overview)
- [TCGA Slide Reference](#tcga-slide-reference)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
- [Configuration Reference](#configuration-reference)
- [WDS Output Format](#wds-output-format)
- [Design Notes](#design-notes)

---

## Overview

The pipeline handles ~30,326 TCGA slide images across all projects. For each
slide it:

1. Discovers slides from the GDC API
2. Resolves paths: local disk → S3 (ECS) → downloads via `gdc-client`
3. Runs mussel-nf to extract patch-level features (`.patch.h5`) and
   slide-level embeddings (`.features.pt`)
4. Appends outputs to per-cancer-type [WebDataset](https://webdataset.github.io/webdataset/)
   shards for use in training

The orchestrator (`tcga_run.py`) is cron-friendly: re-running it picks up
where it left off. A `--initial-run` flag loops in configurable chunks
(default 500 slides) until all pending slides are processed.

---

## TCGA Slide Reference

### Slide Types

TCGA encodes the slide type and sequence number in the filename barcode.
The barcode structure is:

```
TCGA-{TSS}-{Patient}-{SampleType}{Vial}-{Portion}-{SlideType}{Seq}.<UUID>.svs
  e.g.  TCGA-BR-A44T   -01Z          -00       -DX1        .<uuid>.svs
```

| Prefix | Full name       | Description                                         | Count  | On S3 |
|--------|-----------------|-----------------------------------------------------|--------|-------|
| `DX`   | Diagnostic      | H&E-stained diagnostic glass slide; primary imaging modality for computational pathology | 11,848 | ✅ Yes |
| `TS`   | Top Section     | Frozen section from the top of the tissue block     | 8,997  | ❌ No |
| `BS`   | Bottom Section  | Frozen section from the bottom of the tissue block  | 4,941  | ❌ No |
| `MS`   | Middle Section  | Frozen section from the middle of the tissue block  | 77     | ❌ No |

DX slides are the standard for computational pathology. TS/BS/MS slides are
frozen sections used for intraoperative diagnosis and have more artifacts.

Within each type, the sequence number (DX1, DX2, …, DX17) distinguishes
multiple physical slides prepared from the same tumor block. Many patients
have only DX1; some have DX1+DX2; a small number have up to 17 slides.

> **Important:** 882 patient-samples in DX slides alone have 2+ slides
> (DX1 and DX2+). Using only the 4-part barcode `TCGA-BR-A44T-01` as the
> slide identifier would silently collapse these into one record. All scripts
> use the **full slide barcode** (`TCGA-BR-A44T-01Z-00-DX1`) as `slide_id`.

### Sample Type Codes

The two-digit number after the patient ID encodes the tissue type:

| Code | Sample type                  | Count  |
|------|------------------------------|--------|
| `01` | Primary Solid Tumor          | 26,854 |
| `11` | Solid Tissue Normal          | 2,789  |
| `06` | Metastatic                   | 566    |
| `02` | Recurrent Solid Tumor        | 89     |
| `05` | Additional New Primary       | 26     |
| `07` | Additional Metastatic        | 2      |

Normal adjacent tissue (`11`) is available for a subset of projects and can
be useful for self-supervised or contrastive learning tasks.

### Slide Type Filter

The `slides.slide_type` config key (and `--slide-type` CLI arg) controls
which slide types are included. Supports prefix matching and comma-separated
values:

```yaml
slide_type: all      # all 30,326 slides (default)
slide_type: DX       # all DX slides (DX1–DX17); these are on S3
slide_type: DX1      # primary diagnostic only (most common)
slide_type: DX1,DX2  # first two diagnostic slides per sample
```

---

## Architecture

```
tcga_run.py              ← orchestrator; runs phases 1–7 in order
  │
  ├─ Phase 1: tcga_sync_inventory.py   ← GDC API → inventory CSV (cached, TTL 24h)
  ├─ Phase 2: tcga_update_status.py    ← scan results dir → status CSV
  ├─ Phase 3: tcga_prepare_samples.py  ← path resolution → samples CSV
  ├─ Phase 4a: nextflow (ready slides) ← slides on disk or S3 run immediately
  ├─ Phase 4b: gdc-client download     ← download slides not on S3/disk
  ├─ Phase 4c: nextflow (downloaded)   ← run nextflow on freshly downloaded slides
  ├─ Phase 5: tcga_append_wds.py       ← append .pt → per-cancer WDS shards
  └─ Phase 6: tcga_sync_databricks.py  ← upload status table to Databricks (optional)
```

### Path Resolution (Phase 3)

For each pending slide, `tcga_prepare_samples.py` resolves `slide_path`
using this priority chain:

```
1. Local disk   <local_slides_dir>/<file_id>/<file_name>   (size-validated)
2. S3           s3://<s3_base>/<file_id>/<file_name>       (batch listing)
3. Download     needs_download = True  → Phase 4b
```

S3 availability is determined by a single paginated `ListObjectsV2` call
(not per-file `HeadObject`), so checking 11,741 DX slides costs one API
call. Local files are size-validated against the GDC inventory; partially
downloaded files are re-flagged for download.

### Credentials

S3 credentials for the ECS endpoint are resolved in this order:

1. `slides.s3_access_key / s3_secret_key` in config
2. `ECS_ACCESS_KEY` / `ECS_SECRET_KEY` environment variables
3. `nextflow secrets get ECS_ACCESS_KEY` / `ECS_SECRET_KEY`

The S3 endpoint URL is resolved from:

1. `slides.s3_endpoint` in config
2. `ECS_ENDPOINT_URL` environment variable
3. `aws.client.endpoint` from `nextflow config -flat` ← **recommended**

Setting the endpoint in `nextflow.config` means both the pipeline and the
TCGA scripts share a single source of truth.

### Initial Run vs Incremental

| Mode | Usage | Behaviour |
|------|-------|-----------|
| **Incremental** (default) | cron, `tcga_run.py --config cfg.yaml` | One pass: sync inventory, update status, prepare up to `chunk_size` slides, run nextflow, append WDS |
| **Initial run** | `--initial-run` | Loops phases 3–5 with `--chunk-size` slides per iteration until all pending slides are done. After first chunk, `--skip-sync` is set automatically. Databricks sync runs once at the end. |

---

## Quick Start

```bash
# 1. Copy and edit the config
cp scripts/tcga/tcga_run_config.yaml /data/tcga/config.yaml
$EDITOR /data/tcga/config.yaml

# 2. Store ECS credentials in the nextflow secrets store (once)
nextflow secrets set ECS_ACCESS_KEY <key>
nextflow secrets set ECS_SECRET_KEY <secret>

# 3. Dry-run to verify commands
python scripts/tcga/tcga_run.py --config /data/tcga/config.yaml --dry-run

# 4. Initial load of all TCGA DX slides, 500 at a time
python scripts/tcga/tcga_run.py --config /data/tcga/config.yaml \
    --initial-run --chunk-size 500

# 5. Incremental updates (add to cron)
python scripts/tcga/tcga_run.py --config /data/tcga/config.yaml

# Run a single project first (e.g. to validate end-to-end)
python scripts/tcga/tcga_run.py --config /data/tcga/config.yaml \
    --initial-run --project TCGA-BRCA --chunk-size 50

# Skip specific phases
python scripts/tcga/tcga_run.py --config /data/tcga/config.yaml \
    --skip-sync --skip-download
```

---

## Scripts

### `tcga_sync_inventory.py` — GDC Inventory

Fetches the full TCGA slide inventory from the GDC API and writes
`tcga_inventory.csv`. Subsequent calls within `max_age_hours` (default 24h)
return exit code 2 without re-fetching (cached).

```
Columns: file_id, file_name, case_submitter_id, project_id,
         slide_type, file_size, md5sum, updated_datetime
```

```bash
python tcga_sync_inventory.py \
    --output tcga_inventory.csv \
    [--project TCGA-BRCA]       \  # filter to one project
    [--max-age-hours 24]        \  # cache TTL (0 = always fetch)
    [--force]                      # bypass cache
```

Exit codes: `0` = updated, `2` = no changes (cache still fresh), `1` = error.

---

### `tcga_update_status.py` — Status Tracking

Scans a nextflow results directory for completed outputs (`.features.pt` and
`.patch.h5` files) and writes `tcga_status.csv`.

```
Columns: file_id, slide_id, project_id, slide_type, model,
         status (pending|done), pt_path, h5_path, last_updated
```

```bash
python tcga_update_status.py \
    --inventory tcga_inventory.csv \
    --results-dir /data/tcga-results \
    --output tcga_status.csv \
    [--model-types ctranspath,uni2h]  # auto-discovered if omitted
```

---

### `tcga_prepare_samples.py` — Path Resolution

Resolves the filesystem or S3 path for each pending slide and writes a
nextflow-compatible `samples_to_run.csv`. Also writes a `.meta.csv` sidecar
with `needs_download` flags used by the orchestrator.

```bash
python tcga_prepare_samples.py \
    --inventory tcga_inventory.csv \
    --status tcga_status.csv \
    --output samples_to_run.csv \
    --s3-base s3://pathology/TCGA \
    --local-slides-dir /data/tcga-slides \
    [--model ctranspath]           \  # skip slides done for this model
    [--slide-type DX]              \  # filter by type prefix
    [--project TCGA-BRCA]          \  # filter by project
    [--limit 500]                  \  # cap number of output rows
    [--check-s3-exists]               # verify S3 paths via ListObjectsV2
```

Exit codes: `0` = success, `2` = no pending slides, `1` = error.

---

### `tcga_append_wds.py` — WDS Shard Building

Appends `.features.pt` (and optionally `.patch.h5` coords) to
[WebDataset](https://webdataset.github.io/webdataset/) tar shards, grouped
by cancer type. Maintains a `wds_index.json` for idempotency — re-running
never duplicates entries.

```bash
# Via results dir (auto-discovers models):
python tcga_append_wds.py \
    --results-dir /data/tcga-results \
    --inventory tcga_inventory.csv \
    --wds-dest s3://pathology/tcga-features/wds \
    --staging-dir /data/wds-staging

# Explicit model and dirs:
python tcga_append_wds.py \
    --pt-dir /data/tcga-results/features/ctranspath/pt \
    --h5-dir /data/tcga-results/features/ctranspath/tile_h5 \
    --model-type ctranspath \
    --inventory tcga_inventory.csv \
    --wds-dest /data/wds \
    [--slide-ids-csv samples_to_run.csv]  # restrict to current chunk's slides
    [--max-shard-bytes 2147483648]        # 2 GB per shard (default)
```

---

### `tcga_run.py` — Orchestrator

Chains all phases end-to-end. See [Quick Start](#quick-start) above.

```
Flags:
  --config PATH            Required. Path to tcga_run_config.yaml
  --initial-run            Loop phases 3–5 until no slides remain
  --chunk-size N           Slides per chunk (default: 500 from config)
  --project PROJ           Comma-separated project filter override
  --delete-slides          Delete downloaded SVS after each chunk
  --dry-run                Print commands without executing
  --force-sync             Re-fetch GDC inventory even if cache is fresh
  --skip-sync/status/prepare/download/run/append-wds/databricks
                           Skip individual phases
  -v / --verbose           Debug logging
```

---

### `tcga_sync_databricks.py` — Databricks Sync

Uploads the status CSV as Parquet to a Databricks Unity Catalog volume and
optionally triggers a Databricks job to refresh a Delta table.

```bash
python tcga_sync_databricks.py \
    --status tcga_status.csv \
    --inventory tcga_inventory.csv \
    --volume-path /Volumes/catalog/schema/vol/tcga_status.parquet \
    --host https://<workspace>.azuredatabricks.net \
    --token $DATABRICKS_TOKEN \
    [--job-id 12345]
```

---

## Configuration Reference

See [`tcga_run_config.yaml`](tcga_run_config.yaml) for a fully annotated
example. Key sections:

```yaml
inventory_csv: /data/tcga/tcga_inventory.csv
status_csv:    /data/tcga/tcga_status.csv
samples_csv:   /data/tcga/samples_to_run.csv

gdc:
  token_file: ~/.gdc-token      # required for controlled-access data
  client_bin: gdc-client
  max_age_hours: 24             # inventory cache TTL

download:
  local_dir: /data/tcga-slides

slides:
  slide_type: DX                # 'all', 'DX', 'DX1', 'DX1,TS1', …
  local_slides_dir: /data/tcga-slides
  s3_base: s3://pathology/TCGA
  # Endpoint auto-read from nextflow.config aws.client.endpoint if omitted
  s3_endpoint: http://pmindecs.mskcc.org:9020
  check_s3_exists: true

initial_run:
  chunk_size: 500
  delete_slides_after_chunk: true

nextflow:
  profile: cluster,slurm,apptainer
  outdir: /data/tcga-results
  params_file: /data/tcga/params.yaml   # source of truth for model_types

wds:
  dest: s3://pathology/tcga-features/wds
  staging_dir: /data/wds-staging
  max_shard_bytes: 2147483648
```

---

## WDS Output Format

Shards are written under `<wds_dest>/<model>/<project_id>/`:

```
wds/
  ctranspath/
    TCGA-BRCA/
      000000.tar     ← up to max_shard_bytes of slide entries
      000001.tar
    TCGA-LUAD/
      000000.tar
    wds_index.json   ← {slide_id: {project_id, shard_file}} for O(1) lookup
  uni2h/
    ...
```

Each tar entry contains:

| File | Description |
|------|-------------|
| `{slide_id}.features.npy` | Slide-level feature vector (shape: `[D]` or `[N, D]`) |
| `{slide_id}.coords.npy` | Patch coordinates (shape: `[N, 2]`), if `.patch.h5` exists |

### Reading with WebDataset

```python
import webdataset as wds
import numpy as np

ds = wds.WebDataset("s3://pathology/tcga-features/wds/ctranspath/TCGA-BRCA/000000.tar")
for sample in ds:
    features = np.load(io.BytesIO(sample["features.npy"]))
    coords   = np.load(io.BytesIO(sample["coords.npy"])) if "coords.npy" in sample else None
    slide_id = sample["__key__"]
```

---

## Design Notes

**Full slide barcode as `slide_id`.** TCGA filenames encode a full slide
barcode before the UUID: `TCGA-BR-A44T-01Z-00-DX1.<uuid>.svs`. All scripts
extract this as `slide_id` via `file_name.split('.')[0]`. This is essential
because 882+ patient-samples have multiple DX slides (DX1, DX2, …) that
share the same 4-part barcode `TCGA-BR-A44T-01`; using the short barcode
silently loses thousands of slides.

**Per-cancer-type WDS shards.** Splitting shards by `project_id`
(e.g. `TCGA-BRCA`, `TCGA-LUAD`) allows training jobs to load a single
cancer type without reading irrelevant data, enables per-cancer-type
stratified sampling in WebDataset's `ResampledShards`, and makes partial
ingestion safe (incomplete cancer types are easy to identify).

**Launch before downloads complete.** Slides available on S3 or local disk
run through nextflow immediately (Phase 4a) while slides that need
downloading via `gdc-client` are handled in parallel (Phase 4b → 4c). This
avoids waiting hours for downloads before any GPU work starts.

**Single nextflow params source of truth.** `model_types` and all tiling
parameters come from the user's `params_file` (a standard nextflow params
YAML). `tcga_run.py` deep-merges TCGA-specific overrides
(`samples_csv`, `outdir`, `wds.enabled=false`) on top of that file before
passing it to nextflow as a single `-params-file` argument.

**S3 existence check via batch listing.** Rather than one `HeadObject` call
per slide, `tcga_prepare_samples.py` does a single paginated
`ListObjectsV2` with `Delimiter='/'` to get all 11,741 file_id prefixes in
one shot, then does O(1) set lookups per slide.
