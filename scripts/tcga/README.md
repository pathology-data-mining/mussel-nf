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

The **dispatcher** (`dispatcher/mussel-dispatcher.py`) is the primary
orchestrator. It watches for slides, batches them, runs Nextflow concurrently,
pushes results to WDS shards, and resumes automatically after restarts.

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
| `TS`   | Top Section     | Frozen section from the top of the tissue block     | 8,997  | b�� No |
| `BS`   | Bottom Section  | Frozen section from the bottom of the tissue block  | 4,941  | ❌ No |
| `MS`   | Middle Section  | Frozen section from the middle of the tissue block  | 77     | ❌ No |

DX slides are the standard for computational pathology. TS/BS/MS slides are
frozen sections used for intraoperative diagnosis and have more artifacts.

Within each type, the sequence number (DX1, DX2, …, DX17) distinguishes
multiple physical slides prepared from the same tumor block. Many patients
have only DX1; a subset have DX1+DX2 or more.

### Sample Types

TCGA encodes tissue sample origin in the barcode's sample-type field:

| Code | Description                       |
|------|-----------------------------------|
| `01` | Primary Solid Tumor               |
| `02` | Recurrent Solid Tumor             |
| `06` | Metastatic                        |
| `11` | Solid Tissue Normal               |
| …    | Others (blood, cell-line, etc.)   |

The `sample_type` config key (and `--sample-type` CLI arg) filters which
samples are included. Use `all` (default) or a comma-separated list of codes:

```yaml
sample_type: "01"       # primary solid tumors only
sample_type: "01,06"    # primary + metastatic
sample_type: all        # no filtering (default)
```

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
dispatcher/mussel-dispatcher.py        ← primary orchestrator (streaming)
  │
  ├─ TcgaWatcher:
  │    ├─ tcga_sync_inventory.py       ← GDC API → inventory CSV (cached, TTL 24h)
  │    ├─ tcga_update_status.py        ← scan results dir → status CSV
  │    └─ tcga_prepare_samples.py      ← path resolution, S3 check, gdc-client download
  │
  ├─ BatchScheduler → nextflow run     ← parallel Nextflow batches (SLURM)
  │
  └─ Post-batch hooks:
       └─ scripts/append_wds.py        ← append .pt/.h5 → per-group WDS shards (S3)
```

### Path Resolution

For each pending slide, `tcga_prepare_samples.py` resolves `slide_path`
using this priority chain:

```
1. Local disk   <local_slides_dir>/<file_id>/<file_name>   (size-validated)
2. S3           s3://<s3_base>/<file_id>/<file_name>       (batch listing)
3. Download     needs_download = True  → gdc-client
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

---

## Quick Start

```bash
# 1. Copy and edit the dispatcher config
cp dispatcher/tcga_dispatcher.yaml /data/tcga/tcga_dispatcher.yaml
$EDITOR /data/tcga/tcga_dispatcher.yaml

# 2. Store ECS credentials in the nextflow secrets store (once)
nextflow secrets set ECS_ACCESS_KEY <key>
nextflow secrets set ECS_SECRET_KEY <secret>

# 3. Run the dispatcher (streams slides, dispatches parallel Nextflow batches)
python dispatcher/mussel-dispatcher.py /data/tcga/tcga_dispatcher.yaml

# Ctrl+C to gracefully stop (waits for in-flight batches to finish)
# Ctrl+C again to force-exit immediately
# Restart: automatically resumes interrupted batches with -resume
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
with `needs_download` flags used by the dispatcher.

```bash
python tcga_prepare_samples.py \
    --inventory tcga_inventory.csv \
    --status tcga_status.csv \
    --output samples_to_run.csv \
    --s3-base s3://pathology/TCGA \
    --local-slides-dir /data/tcga-slides \
    [--model ctranspath]           \  # skip slides done for this model
    [--slide-type DX]              \  # filter by type prefix
    [--sample-type 01]             \  # filter by sample type code
    [--project TCGA-BRCA]          \  # filter by project
    [--limit 500]                  \  # cap number of output rows
    [--check-s3-exists]               # verify S3 paths via ListObjectsV2
```

Exit codes: `0` = success, `2` = no pending slides, `1` = error.

---

### `scripts/append_wds.py` — WDS Shard Building

Appends `.features.pt` (and optionally `.patch.h5` coords) to
[WebDataset](https://webdataset.github.io/webdataset/) tar shards, grouped
by a routing key. Maintains a `wds_index.json` for idempotency — re-running
never duplicates entries.

```bash
# Via results dir (auto-discovers models), routing by TCGA inventory:
python scripts/append_wds.py \
    --results-dir /data/tcga-results \
    --inventory tcga_inventory.csv \
    --wds-dest s3://pathology/tcga-features/wds \
    --staging-dir /data/wds-staging

# Explicit model and dirs:
python scripts/append_wds.py \
    --pt-dir /data/tcga-results/features/ctranspath/pt \
    --h5-dir /data/tcga-results/features/ctranspath/tile_h5 \
    --model-type ctranspath \
    --inventory tcga_inventory.csv \
    --wds-dest /data/wds \
    [--slide-ids-csv samples_to_run.csv]  # restrict to current chunk's slides
    [--max-shard-bytes 2147483648]        # 2 GB per shard (default)
    [--s3-max-concurrency 4]              # boto3 multipart thread limit
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

See [`dispatcher/tcga_dispatcher.yaml`](../../dispatcher/tcga_dispatcher.yaml)
for a fully annotated example. Key sections:

```yaml
watchers:
  - type: tcga
    inventory_csv: /data/tcga/tcga_inventory.csv
    status_csv: /data/tcga/tcga_status.csv
    slide_type: DX
    sample_type: "01"
    s3_base: s3://pathology/TCGA
    check_s3_exists: true
    wds_destinations:
      hoptimus1: s3://pathology/tcga-features/wds/hoptimus1
      titan_slide: s3://pathology/tcga-features/wds/titan_slide
    wds_staging_dir: /data/wds-staging
    wds_s3_max_concurrency: 4
    cleanup_results: true

batch_size: 50
max_concurrent_runs: 2
outdir: /data/tcga-results
nextflow_profiles: cluster,slurm,conda
```

---

## WDS Output Format

Shards are written under `<wds_dest>/<project_id>/`:

```
wds/
  hoptimus1/
    TCGA-BRCA/
      000000.tar     ← up to max_shard_bytes of slide entries
      000001.tar
    TCGA-LUAD/
      000000.tar
    wds_index.json   ← {slide_id: {project_id, shard_file}} for O(1) lookup
  titan_slide/
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

ds = wds.WebDataset("s3://pathology/tcga-features/wds/hoptimus1/TCGA-BRCA/000000.tar")
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

**Dispatcher resumes on restart.** When the dispatcher restarts, if a
batch's Nextflow work dir is still on disk the run is re-submitted with
`-resume` so already-completed SLURM tasks are skipped (no wasted GPU
recompute). Batches whose work dirs were deleted fall back to full re-dispatch.

**S3 existence check via batch listing.** Rather than one `HeadObject` call
per slide, `tcga_prepare_samples.py` does a single paginated
`ListObjectsV2` with `Delimiter='/'` to get all 11,741 file_id prefixes in
one shot, then does O(1) set lookups per slide.
