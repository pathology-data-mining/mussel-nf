# Copilot Instructions for mussel-nf-internal

> **This is the MSK-internal fork** of the public [`pathology-data-mining/mussel-nf`](https://github.com/pathology-data-mining/mussel-nf) repository.
> It extends the public pipeline with MSK-specific watchers (TCGA GDC, IMPACT), internal dispatcher configs, and internal infrastructure wiring.
>
> **Source of truth is the public repo.** Sync periodically:
> ```bash
> git fetch public && git merge public/main && git push origin main
> ```
> Keep MSK-internal files in directories never touched by the public repo (e.g. `dispatcher/mussel_dispatcher/impact/`, `dispatcher/mussel_dispatcher/tcga/`, `dispatcher/queries/` for internal SQL) to avoid merge conflicts.

---

## Overview

`mussel-nf` is a [Nextflow](https://www.nextflow.io/) pipeline for computational pathology. It tessellates whole-slide images (WSIs) and extracts deep learning features using the [Mussel](https://github.com/pathology-data-mining/Mussel) library. Outputs are patch-level and slide-level feature tensors (`.h5`, `.pt`) used for downstream tasks like CLIP annotation and linear probe benchmarking.

## Running the Pipeline

```bash
# Minimal run (local, docker)
nextflow run main.nf -profile standard,docker --samples_csv samples.csv

# MSK cluster (SLURM + Apptainer)
nextflow run main.nf -profile cluster,slurm,apptainer --samples_csv samples.csv

# Resume a previous run
nextflow run main.nf -profile standard,docker --samples_csv samples.csv -resume

# Override params from a YAML file
nextflow run main.nf -params-file filter_workflow.yaml --samples_csv samples.csv

# See all parameters
nextflow run main.nf --help
nextflow run main.nf --helpFull
```

### Validate outputs after a run

```bash
nextflow run validate.nf --manifest_csv results/manifest-*.csv --results_dir results
```

## Architecture

```
main.nf                        # Entry point: validates params (nf-schema), streams manifest, calls MUSSEL
modules/
  mussel.nf                    # Top-level MUSSEL workflow + EXTRACT_FEATURES workflow
  tessellation.nf              # TESSELLATE, FILTER_TILES processes
  featurize.nf                 # FEATURIZE, FEATURIZE_BATCH processes
  tessellate_featurize.nf      # TESSELLATE_FEATURIZE_BATCH (combined one-step process)
  clip/main.nf                 # CLIP annotation workflow (e.g., QuiltNet)
  linear_probe/main.nf         # Linear probe benchmarking workflow
assets/
  schema_input.json            # Validates samples_csv columns (slide_id, slide_path, oncotree_code)
  schema_annotations.json      # Validates annotations_csv for linear probe
```

### Workflow execution paths

- **`params.use_one_step_workflow = true`** (default): Uses `TESSELLATE_FEATURIZE_BATCH` — tessellation and feature extraction happen in a single Nextflow task per batch.
- **`params.use_one_step_workflow = false`**: Separate `TESSELLATE` → `FEATURIZE_BATCH` steps. Useful when reusing pre-tessellated tiles.
- **Tile filtering** (`params.tiling.filter_tiles = true`): Adds a filter step between tessellation and feature extraction using a small classifier (ctranspath + a `.pkl` model).
- **CLIP workflow**: Triggered when `params.clip.model_types` is non-empty (e.g., `quiltnet`). Requires `oncotree_code` in the sample sheet or `params.clip.default_classes`.
- **Linear probe**: Triggered when `params.linear_probe.annotations_csv` is set.

### Manifest generation

`main.nf` uses a Nextflow **topic channel** (`topic: 'slide_meta'`) to stream `(slide_id, key, value)` tuples from all processes into a rolling `manifest-{timestamp}.csv` in `params.outdir`. Each process emits metadata via `topic: meta_out` in its output block.

## Key Conventions

### Process resource labels

Processes are tagged with labels that map to executor-specific resources in `nextflow.config`:

| Label | Typical use |
|---|---|
| `gpuTask` | Feature extraction (GPU required) |
| `cpuTask` | Tessellation, CLIP annotation |
| `bigTask` | Merge/stack operations (~30 GB RAM) |
| `hugeTask` | Large aggregation (up to 1 TB RAM) |
| `smallTask` | Lightweight processes |
| `parallelTask` | Multi-threaded (uses `task.cpus`) |
| `localTask` | Always runs on head node (not dispatched) |

### Batching — three independent dimensions

1. **`featurize.workflow_batch_size`** (default: 8): Number of slides grouped into a single Nextflow task. Reduces scheduling overhead.
2. **`featurize.batch_size`** (default: 64): Tiles per forward pass through the patch encoder.
3. **`featurize.slide_batch_size`** (default: 8): Slides aggregated together in slide-level encoders (e.g., `gigapath_slide`).

### Slide-level models auto-resolve patch encoder

Specifying a slide encoder (e.g., `gigapath_slide`) in `featurize.model_types` automatically uses the required patch encoder (`gigapath`). The mapping is in `params.featurize.slide_to_patch_mapping`.

### Publish directory structure

Output is published under `params.outdir` (default: `results`):
```
results/
  features/{model_type}/h5/       # Patch-level features (HDF5)
  features/{model_type}/pt/       # Slide-level feature tensors
  features/{model_type}/tile_h5/  # Patch coordinates
  tiles/                          # Raw tessellation output
  filter_tiles/                   # Post-filter patch HDF5
  annotate/{model_type}/          # CLIP annotation CSVs
  annotation_features/{model_type}/ # Linear probe features
  manifest-{timestamp}.csv        # Auto-generated result manifest
  params.json                     # Saved tiling/workflow params
```

When `params.publish_slide_prefix = true` (default), published files are nested under a 4-character prefix of the slide ID (e.g., `features/optimus/h5/TCGA/slide.features.h5`).

### Sample sheet format

`samples_csv` must have columns `slide_id` and `slide_path` (required), plus optional `oncotree_code`. Accepted slide extensions: `.svs`, `.tiff`, `.tif`, `.ndpi`, `.scn`.

### Segmentation config groups

`params.tiling.seg_config_group` selects preset segmentation parameters (`biopsy`, `resection`, `tcga`, or `default`). Individual parameters like `patch_size` and `mpp` override the group defaults when set explicitly.

## Supported Models

**Patch encoders:** `resnet50`, `ctranspath`, `gigapath`, `virchow`, `virchow2`, `optimus`, `uni`, `uni2h`, `conch1_5`, `clip`, `googlepath`

**Slide encoders:** `gigapath_slide` (requires `gigapath`), `titan_slide` (requires `conch1_5`)

Models are downloaded from HuggingFace automatically unless a local path is set in `params.featurize.model_paths.{model_type}`.

## Execution Profiles

Combine a runtime profile and an executor profile:

```bash
# Local + Docker
-profile standard,docker

# HPC cluster + Apptainer
-profile cluster,slurm,apptainer

# Azure Batch
-profile docker,cloud
```

Profiles: `standard`, `docker`, `apptainer`, `conda`, `cluster`, `condor`, `slurm`, `cloud`

## Environment / Dependencies

- **Conda env**: `mussel_env.yaml` (Python 3.12 + Mussel from GitHub). Set `MUSSEL_VENV` env var to the path of a pre-built venv.
- **Container**: `mskmind/mussel:current`
- **Nextflow plugin**: `nf-schema@2.2.0` (parameter validation + samplesheet parsing)

### HPC-specific cache locations

On shared HPC systems, redirect caches to avoid filling home directories:

```bash
export UV_CACHE_DIR=/path/to/large/mount/.uv
export HF_HOME=/path/to/large/mount/.hf
```

### Pre-download models before parallel runs

Running many jobs simultaneously against HuggingFace causes race conditions. Run a single-slide dry run first to cache models before launching at scale.

## Azure Batch Notes

Azure Batch nodes have poor disk space management — running many jobs will eventually exhaust disk space and put nodes into an **unusable** state (they linger and continue incurring cost until deleted). Mitigate by mounting Azure file shares with large capacity to the batch nodes via `params.azure.storage.fileShares`. A PowerShell script to automatically delete unusable nodes is also recommended regardless, as nodes enter the unusable state for various other reasons.

## Dispatcher

`dispatcher/` is a Python daemon (`mussel-dispatcher`) that runs the pipeline continuously as new slides become available. It watches one or more slide sources, batches slides, launches Nextflow runs, and runs post-batch hooks (e.g. sync results to Databricks or push WDS shards to S3).

```
dispatcher/
  mussel_dispatcher/
    config.py          # Config / WatcherConfig dataclasses; YAML parsing
    scheduler.py       # BatchScheduler: buffers slides, fires when batch ready
    runner.py          # RunManager: launches NF runs up to max_concurrent_runs
    watchers.py        # LocalWatcher, S3Watcher, TcgaWatcher, DatabricksWatcher
    state.py           # SQLite StateStore for crash recovery
    wds.py             # append_wds() — WDS shard builder / S3 uploader
    databricks_sync.py # Generic Databricks Parquet → Delta MERGE sync
    tcga/              # MSK-INTERNAL: TCGA GDC watcher support scripts
    impact/            # MSK-INTERNAL: IMPACT-specific sync scripts
  queries/             # SQL files for DatabricksWatcher (internal queries here)
  tests/
  dispatcher.yaml               # Default config (edit/copy before running)
  tcga_dispatcher.yaml          # MSK-INTERNAL: TCGA run config
  reef_v2_dispatcher.yaml       # MSK-INTERNAL: REEF v2 / IMPACT run config
```

### Running the dispatcher

```bash
cd dispatcher
pip install -e .

# Start (reads dispatcher.yaml by default)
python -m mussel_dispatcher dispatcher.yaml

# Collect and merge all per-run manifests
python -m mussel_dispatcher collect-manifests dispatcher.yaml

# Run tests
pytest tests/
```

### Watcher types

| Type | Config key `type:` | Discovers slides from |
|---|---|---|
| `local` | `local` | Filesystem directory poll |
| `s3` | `s3` | S3/ECS bucket prefix listing |
| `tcga` | `tcga` | TCGA GDC API + local download |
| `databricks` | `databricks` | Databricks SQL Warehouse query |

**DatabricksWatcher** requires `warehouse_id` + either `query` (inline SQL) or `query_file` (path to `.sql`). The query must return `slide_id`, `slide_path`, and optionally `oncotree_code`.

### Post-batch hooks

After each successful Nextflow run the scheduler calls hooks in `post_batch_hooks`. Auto-generated hooks (from watcher config):
- **WDS hook** — invokes `python -m mussel_dispatcher.wds` to push `.pt`/`.h5` feature files to an S3/ECS WDS destination.
- **Databricks sync hook** — invokes `python -m mussel_dispatcher.databricks_sync` (generic) or `python -m mussel_dispatcher.tcga.sync_databricks` / `python -m mussel_dispatcher.impact.sync_databricks` (MSK-internal) to upload a Parquet inventory snapshot to a Unity Catalog volume, then optionally triggers a Databricks job to MERGE it into a Delta table.

### Key config fields (`Config`)

| Field | Default | Description |
|---|---|---|
| `max_concurrent_runs` | 2 | Max parallel NF runs |
| `batch_size` | 20 | Slides per NF run |
| `max_wait_seconds` | 300 | Max time to wait before firing a partial batch |
| `retry_failed` | true | Re-enqueue failed slides |
| `max_slide_retries` | 5 | Give up after N retries |
| `nextflow_config` | — | Path to a custom `nextflow.config` |
| `nextflow_params_file` | — | Path to a params YAML override |

## Databricks Integration

`databricks/` contains a parameterized Databricks notebook and job JSON for syncing feature inventory into a Delta table.

```
databricks/
  notebooks/metadata_sync.py   # Databricks notebook: Parquet → Delta MERGE via MERGE INTO
  jobs/metadata_sync_job.json  # Databricks Jobs API definition (i3.xlarge single-node)
```

**Notebook parameters** (set as Databricks job task values or widget defaults):

| Parameter | Default | Description |
|---|---|---|
| `volume_folder` | — | UC Volume path containing Parquet files |
| `target_table` | — | Delta table to MERGE INTO |
| `merge_key` | `slide_id` | Column used as row identifier in MERGE |
| `filename_prefix` | `""` | Only use Parquet files with this filename prefix |

The MERGE uses explicit column intersection (`source_cols ∩ target_cols`) to avoid `DELTA_MERGE_UNRESOLVED_EXPRESSION` errors when the source Parquet has fewer columns than the target Delta table.

## Utility Scripts

- **`scripts/create_manifest.py`**: Manually rebuild manifest from a results directory. Scans for `*.features.pt`, `*.patch.h5` across `features/`, `tiles/`, `filter_tiles/`.
- **`validate.nf`**: Validates `.h5` (checks for `coords` key) and `.pt` files from a manifest CSV.

---

## MSK-Internal: TCGA Dispatcher

The `tcga/` module handles TCGA GDC inventory management and syncing.

```
dispatcher/mussel_dispatcher/tcga/
  prepare_samples.py    # Build samples CSV from tcga_inventory.csv + status
  sync_databricks.py    # Upload TCGA inventory Parquet to Databricks UC volume
  sync_inventory.py     # Query GDC API; write/update tcga_inventory.csv
  update_status.py      # Update status CSV after a batch completes
```

**Config files:**
- `dispatcher/tcga_dispatcher.yaml` — full TCGA run config with GDC token, S3 ECS endpoint, inventory CSV paths, WDS destinations, Databricks volume/table

**Key watcher fields for TCGA (`type: tcga`):**

| Field | Description |
|---|---|
| `inventory_csv` | Path to `tcga_inventory.csv` (GDC metadata) |
| `status_csv` | Path to `tcga_status.csv` (per-slide run status) |
| `local_slides_dir` | Local directory to download slides into |
| `s3_base` / `s3_endpoint` | ECS S3 endpoint for WDS upload |
| `project` | TCGA project filter (e.g. `TCGA-BRCA` or blank for all) |
| `slide_type` | `DX` (diagnostic) or other GDC slide type code |
| `download_enabled` | If true, dispatcher downloads slides before NF run |
| `gdc_token_file` | Path to GDC data access token |
| `wds_destinations` | `{model_type: s3://bucket/prefix}` map |
| `databricks_volume_folder` | UC Volume path for Parquet export |
| `databricks_table` | Delta table name for MERGE |
| `databricks_job_id` | Databricks job ID to trigger after upload |

## MSK-Internal: IMPACT Dispatcher

The `impact/` module handles the IMPACT cohort (MSK-specific internal slide library).

```
dispatcher/mussel_dispatcher/impact/
  sync_databricks.py    # Export IMPACT feature inventory Parquet to Databricks UC volume
```

**Config files:**
- `dispatcher/reef_v2_dispatcher.yaml` — full IMPACT/REEF v2 run config

The IMPACT sync script reads the dispatcher SQLite database + WDS manifest, builds a Parquet export (one row per slide × model), and uploads it to a Databricks Unity Catalog volume. It optionally triggers a Databricks job to MERGE it into a Delta table.

**IMPACT-specific Parquet columns:** `slide_id` (= `image_id`), `oncotree_code`, `slide_path` (ECS S3 path), `model`, `status`, `failure_reason`, `wds_path`, `first_seen_at`, `completed_at`.

## Public/Private Sync Workflow

```bash
# In mussel-nf-internal:
git remote -v
# origin  git@github.com:pathology-data-mining/mussel-nf-internal.git
# public  git@github.com:pathology-data-mining/mussel-nf.git

# Pull upstream public changes
git fetch public
git merge public/main
git push origin main

# Or to preview what's coming
git log HEAD..public/main --oneline
```

**Rules to avoid merge conflicts:**
- Never modify files that exist in the public repo from this repo — only add new files in `impact/`, `tcga/`, or new query files.
- Internal config files (`reef_v2_dispatcher.yaml`, `tcga_dispatcher.yaml`) are fine to keep here as they don't exist in the public repo.
- If the public repo adds a new file that collides with an internal file, rename the internal one and update imports.
