# Copilot Instructions for mussel-nf

## Overview

`mussel-nf` is a [Nextflow](https://www.nextflow.io/) pipeline for computational pathology. It tessellates whole-slide images (WSIs) and extracts deep learning features using the [Mussel](https://github.com/pathology-data-mining/Mussel) library. Outputs are patch-level and slide-level feature tensors (`.h5`, `.pt`) used for downstream tasks like CLIP annotation and linear probe benchmarking.

## Running the Pipeline

```bash
# Minimal run (local, docker)
nextflow run main.nf -profile standard,docker --samples_csv samples.csv

# Cluster (SLURM + Apptainer)
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

## Utility Scripts

- **`scripts/create_manifest.py`**: Manually rebuild manifest from a results directory. Scans for `*.features.pt`, `*.patch.h5` across `features/`, `tiles/`, `filter_tiles/`.
- **`validate.nf`**: Validates `.h5` (checks for `coords` key) and `.pt` files from a manifest CSV.
- **`create_muv_manifest.py`**: Creates MUV-format manifests.
