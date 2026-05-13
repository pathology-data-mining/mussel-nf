# Mussel-NF pipeline

A pipeline for running [Mussel](https://github.com/pathology-data-mining/Mussel).

## Requirements

* Unix-like operating system
* Java 17+
* Nextflow ≥ 24.x (uses [nf-schema](https://nextflow-io.github.io/nf-schema/) plugin)
* One of: Docker, Apptainer, or a Conda/Mamba environment

### Conda environment

Two environment files are provided depending on which models you need:

| File | Extra | Supported models |
|---|---|---|
| `mussel_env.yaml` | `torch-gpu` | All models **except** `googlepath`, `gigapath`, `gigapath_slide` |
| `mussel_env_tf.yaml` | `tensorflow-gpu` | `googlepath` (Google Path Foundation, TensorFlow-based) |
| `mussel_env_fastattn.yaml` | `fastattn` | `gigapath`, `gigapath_slide` (requires flash-attn) |

The `torch-gpu`, `tensorflow-gpu`, and `fastattn` extras are mutually exclusive — they cannot be installed together. Use `-profile conda` with the default `mussel_env.yaml` for most workflows, and switch to the appropriate env file when running `googlepath` or `gigapath`.

## Quickstart

1. Install Nextflow:
    ```bash
    curl -s https://get.nextflow.io | bash
    ```

2. Launch the pipeline:
    ```bash
    # Local with Docker
    nextflow run main.nf -profile standard,docker --samples_csv samples.csv

    # HPC cluster with Apptainer
    nextflow run main.nf -profile cluster,slurm,apptainer --samples_csv samples.csv

    # Resume a previous run
    nextflow run main.nf -profile standard,docker --samples_csv samples.csv -resume
    ```

    `samples.csv` must have columns `slide_id` and `slide_path` (required), plus optional
    `oncotree_code` and `sample_id`.
    Accepted slide extensions: `.svs`, `.tiff`, `.tif`, `.ndpi`, `.scn`.

3. When the execution completes, results will be in `params.outdir` (default: `results/`).

## Continuous Processing (Dispatcher)

For large-scale cohorts (e.g. all TCGA slides, MSK IMPACT), use the **mussel-dispatcher** rather than invoking Nextflow directly. The dispatcher:

- Streams slides from multiple sources (TCGA GDC API, Databricks SQL warehouse, local directories, S3 buckets)
- Batches slides and dispatches parallel Nextflow runs up to a configurable concurrency limit
- Tracks all slides and batches in SQLite — safe to kill and restart at any time
- Runs post-batch hooks to append outputs to [WebDataset](https://github.com/webdataset/webdataset) shards after each batch

```bash
# Start the dispatcher (streams TCGA slides → Nextflow → WDS shards)
cd dispatcher/
python -m mussel_dispatcher tcga_dispatcher.yaml

# Monitor via dashboard
python -m mussel_dispatcher.dashboard tcga_dispatcher.yaml --port 8050
```

See [`dispatcher/README.md`](dispatcher/README.md) for full configuration reference, watcher types, and deployment notes. For TCGA-specific details (slide types, GDC inventory, path resolution) see [`dispatcher/docs/tcga.md`](dispatcher/docs/tcga.md).

## Supported Models

**Patch encoders** (`params.featurize.model_types`):
`resnet50`, `ctranspath`, `gigapath`, `virchow`, `virchow2`, `optimus`, `hoptimus1`, `h0mini`, `uni`, `uni2h`, `conch1_5`, `conch_v1`, `clip`, `googlepath`, `phikon`, `phikon_v2`, `midnight12k`, `gpfm`, `hibou_l`, `openmidnight`, `genbio_pathfm`, `kaiko_vits8`, `kaiko_vits16`, `kaiko_vitb8`, `kaiko_vitb16`, `kaiko_vitl14`, `lunit_vits8`, `lunit_vits16`

**Slide encoders** (specified in `model_types`; patch encoder auto-resolved):

| Model key | Patch encoder |
|---|---|
| `gigapath_slide` | `gigapath` |
| `titan_slide` | `conch1_5` |
| `prism_slide` | `virchow` |
| `feather_slide` | `conch1_5` |
| `chief_slide` | `ctranspath` |
| `madeleine_slide` | `clip` |

See [SLIDE_MODELS.md](SLIDE_MODELS.md) for slide encoder configuration details.

## Misc Notes

* See full parameters with `--help` or `--helpFull`.
* To use gated HuggingFace models (e.g. UNI, Virchow), set the `HF_TOKEN` Nextflow secret:
  `nextflow secrets set HF_TOKEN <token>`
* To use models from local paths instead of downloading, set `params.featurize.model_paths.{model_type}`.
* Set `params.publish_slide_prefix = true` to nest published files under a 4-char slide ID prefix.
* The pipeline auto-generates a manifest CSV in `params.outdir`. It can also be rebuilt manually
  with `scripts/create_manifest.py`.
* On shared HPC systems, redirect caches to avoid filling home directories:
  ```bash
  export UV_CACHE_DIR=/path/to/large/mount/.uv
  export HF_HOME=/path/to/large/mount/.hf
  ```
* Pre-download models before launching at scale — parallel jobs all hitting HuggingFace simultaneously
  causes race conditions. Run a single-slide dry-run first.
* `params.featurize.workflow_batch_size` (default: 8) controls how many slides are grouped into a
  single Nextflow task, reducing scheduler overhead.

## Workflows

The standard workflow tessellates slides and extracts features for `params.featurize.model_types`.

### One-step vs two-step execution

* **One-step** (`params.use_one_step_workflow = true`, default): tessellation and feature extraction
  happen in a single task via `tessellate_extract_features`. More efficient for most use cases.
* **Two-step** (`params.use_one_step_workflow = false`): separate `TESSELLATE` → `FEATURIZE_BATCH`
  tasks. Useful when reusing pre-tessellated tiles.

### Tile filtering

There are two independent mechanisms for filtering out non-tissue or low-quality tiles:

#### Legacy classifier-based filtering (`params.tiling.filter_tiles`)

A post-tessellation step that uses a pre-trained sklearn classifier to score tiles:

1. Tessellation
2. Feature extraction using `params.tiling.filter_model_type` (default: `ctranspath`)
3. Classify tiles with the `.pkl` model at `params.tiling.filter_model_path`; discard tiles below `params.tiling.filter_threshold`
4. Feature extraction for `params.featurize.model_types` on surviving tiles

Requires a pre-trained `.pkl` classifier. Set `filter_tiles = true` to enable.

#### Segmentation-integrated artifact removal (`params.tiling.remove_artifacts` / `remove_penmarks`)

These options remove artifacts **during tessellation** by refining the tissue mask before patches are extracted. No separate classifier model is needed. Powered by [GrandQC](https://www.nature.com/articles/s41467-024-54769-y) (U-Net, EfficientNet-B0 encoder), which classifies each pixel of the slide thumbnail into 8 tissue categories.

| Parameter | Default | Description |
|---|---|---|
| `remove_penmarks` | `true` | Remove pen markings (class 4) and background (class 7). Conservative — safe for all tissue types. |
| `remove_artifacts` | `false` | **Aggressive mode** — also remove blood (2), necrosis (3), folds (5), and holes (6). May over-remove tissue in CNS or sarcoma slides. |
| `artifact_exclude_classes` | _(unset)_ | **Override**: explicit list of GrandQC class IDs to remove. Takes precedence over the flags above. |
| `seg_model` | `'neural'` | Tissue segmentation backend: `'classic'` (HSV + fixed threshold), `'otsu'` (HSV + Otsu), or `'neural'` (DeepLabV3). |
| `min_tissue_proportion` | `0.0` | Drop patches where less than this fraction of pixels are tissue. |
| `overlap` | `0` | Extract overlapping patches (pixels). |
| `slide_mpp_override` | _(unset)_ | Override the slide's MPP when metadata is missing or incorrect. |

**GrandQC class IDs** (for `artifact_exclude_classes`):

| ID | Class | Removed by `remove_penmarks` | Removed by `remove_artifacts` |
|---|---|---|---|
| 0 | Glass / clear-slide background | ✗ | ✗ |
| 1 | Normal tissue | ✗ | ✗ |
| 2 | Blood / haemorrhage | ✗ | ✓ |
| 3 | Necrosis | ✗ | ✓ |
| 4 | Pen marking | ✓ | ✓ |
| 5 | Fold | ✗ | ✓ |
| 6 | Hole / physical damage | ✗ | ✓ |
| 7 | Background | ✓ | ✓ |

**Preset examples** — set in `nextflow.config` or a `--params-file` YAML:

```yaml
tiling:
  seg_model: neural
  remove_penmarks: true            # conservative (default): pen marks + background only

  # remove_artifacts: true         # aggressive: all non-normal-tissue classes

  # artifact_exclude_classes:      # custom: pen marks, folds, holes, background (moderate)
  #   - 4
  #   - 5
  #   - 6
  #   - 7
```

**Resilience**: if GrandQC removes more than 90% of tissue (e.g. out-of-distribution slides), the pre-removal mask is used automatically and a warning is logged.

These options can be combined with legacy tile filtering or used independently.

### CLIP-based annotation

When `params.clip.model_types` is non-empty (e.g. `['quiltnet']`), the standard feature extraction
workflow runs, then tile annotation and tile caching are performed.

Default annotation classes are set via `params.clip.default_classes`, or per oncotree code via
`params.clip.oncotree_class_csv` (two columns: `oncotree_code`, `class`). The sample sheet
`oncotree_code` column is used to look up classes per slide.

### Linear probe benchmarking

Linear probe benchmarking trains logistic regression classifiers on top of frozen patch-level
features and measures how well each encoder separates annotated tissue classes. It runs
automatically when `params.linear_probe.annotations_csv` is set.

#### Required inputs

| Parameter | Description |
|---|---|
| `params.linear_probe.annotations_csv` | CSV with `slide_id` and `annotation_bmp_path` columns. Each row maps a slide to a BMP mask where pixel values are annotation class IDs. |

#### Optional inputs

| Parameter | Description |
|---|---|
| `params.linear_probe.annotation_class_mapping_yaml` | YAML mapping BMP pixel values (integers) to remapped class IDs. When omitted, raw non-zero BMP pixel values are used directly as class labels — sufficient for most multiclass cases where BMP values already represent the desired class IDs. |

#### Annotation CSV format

```csv
slide_id,annotation_bmp_path
TCGA-XX-1234-01Z-00-DX1,/path/to/TCGA-XX-1234.bmp
```

#### Class mapping YAML format

Use this when you need to remap BMP pixel values (e.g. collapse multiple classes, exclude specific
values, or create binary labels). Background pixels (value 0) are always excluded regardless.

```yaml
# Binary example: remap BMP values 1→class 0, 2→class 1
1: 0   # annotation ID 1 → class 0 (negative / background)
2: 1   # annotation ID 2 → class 1 (positive / tumour)
```

For multiclass with identity mapping (BMP values used as-is), omit the YAML and set
`params.linear_probe.multiclass = true`:

```bash
nextflow run main.nf ... \
  --linear_probe.annotations_csv annotations.csv \
  --linear_probe.multiclass
```

Or provide an explicit mapping if you need to remap values:

```yaml
# Multiclass example with explicit mapping
1: 1   # tumour
2: 2   # stroma
3: 3   # necrosis
```

#### Workflow steps

1. **Tessellation + feature extraction** — normal pipeline produces per-slide `.h5` patch features.
2. **`MERGE_ANNOTATION_FEATURES`** — for each slide, maps tile centre coordinates into the BMP mask
   to assign each tile a class label, producing a per-slide `annotation_features.parquet`.
   Tiles with fewer than the expected annotated-pixel fraction are dropped.
3. **`STACK_ANNOTATION_FEATURES`** — concatenates all per-slide parquets into one
   `annotation_features.parquet` per model type.
4. **`LINEAR_PROBE_BENCHMARK`** — runs grid-searched logistic regression (scikit-learn) and
   emits plots and metrics to `results/linear_probe_benchmark/{model_type}/`.
5. **`SUMMARIZE_LINEAR_PROBE`** — collects `results.json` from all model types and writes
   `results/linear_probe_benchmark/summary.csv` + `summary.png` for cross-model comparison.

#### Outputs

```
results/
  annotation_features/{model_type}/         # Per-slide annotation_features.parquet files
  linear_probe_benchmark/{model_type}/
    classification_report.csv               # Per-class precision/recall/F1
    classification_report_test.csv
    confusion_matrix.png
    confusion_matrix_test.png
    roc_curve.png
    pr_curve.png
    grid_search_heatmap.png
    feature_importance.png
    calibration_curve.png
    cv_results.csv                          # Full grid search results
    results.json                            # Numeric metrics (AUC-ROC, F1, AP, 95% CI)
  linear_probe_benchmark/
    summary.csv                             # All models side-by-side
    summary.png                             # Bar chart with 95% CI error bars
```

#### Hyperparameters

All configurable via `params.linear_probe.*`:

| Parameter | Default | Description |
|---|---|---|
| `cv` | `5` | Cross-validation folds for grid search |
| `C_values` | `[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0]` | Regularisation strengths to search over |
| `penalties` | `["l2"]` | Penalty types to search over (`"l1"`, `"l2"`, `"elasticnet"`) |
| `n_seeds` | `5` | Number of random seeds for bootstrap variance estimation |
| `n_bootstrap` | `1000` | Bootstrap resamples for 95% CI on test metrics |
| `random_state` | `42` | Global random seed |
| `positive_annotation_label` | `1` | Class ID treated as positive for binary AUC/AP |
| `multiclass` | `false` | Enable OvR macro AUC-ROC / macro F1 for ≥ 3 classes |

#### Example invocation

```bash
nextflow run main.nf \
  -profile cluster,apptainer \
  --samples_csv samples.csv \
  --linear_probe.annotations_csv annotations.csv \
  --linear_probe.annotation_class_mapping_yaml class_mapping.yaml \
  --featurize.model_types "optimus,uni"
```

### WebDataset sharding

After feature extraction, completed `.pt` slide-feature files (and optionally `.h5` patch-feature
files) are packed into [paladin](https://github.com/pathology-data-mining/paladin)-compatible
[WebDataset](https://github.com/webdataset/webdataset) `.tar` shards.

Each tar entry contains:
- `{slide_id}.features.npy` — float32 feature array (converted from `.pt`)
- `{slide_id}.coords.npy` — int64 tile coordinates from `.h5` (when `wds.shard_h5=true`)

Shards are directly readable by the `webdataset` Python library:

```python
import io, numpy as np, torch, webdataset as wds
ds = wds.WebDataset("results/wds/optimus/all/{000000..000004}.tar")
for sample in ds:
    slide_id = sample["__key__"]
    features = torch.from_numpy(np.load(io.BytesIO(sample["features.npy"])))
```

Enable and configure sharding via `params.wds`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wds.enabled` | `false` | Enable WDS shard output |
| `wds.group_by_oncotree` | `false` | Separate shard sets per `oncotree_code`; requires that column in `samples_csv` |
| `wds.shard_h5` | `false` | Extract tile coords from `.h5` and store as `.coords.npy` |
| `wds.max_shard_size` | `1000` | Max slides per `.tar` shard |
| `wds.shard_prefix` | `""` | Filename prefix (`""` → `000000.tar`) |

**Example — shard optimus features grouped by cancer type:**

```bash
nextflow run main.nf \
  -profile cluster,slurm,apptainer \
  --samples_csv samples.csv \
  --featurize.model_types='["optimus"]' \
  --wds.enabled=true \
  --wds.group_by_oncotree=true \
  --wds.max_shard_size=500
```

Output layout:

```
results/wds/optimus/BRCA/000000.tar
results/wds/optimus/BRCA/000001.tar
results/wds/optimus/LUAD/000000.tar
...
```

Without grouping (`group_by_oncotree=false`):

```
results/wds/optimus/all/000000.tar
results/wds/optimus/all/000001.tar
...
```


### Multi-slide sample aggregation

When a `samples_csv` contains multiple rows with the same `sample_id`, the pipeline produces
both per-slide features and a merged sample-level output.

Add an optional `sample_id` column to your samples CSV:

```csv
slide_id,slide_path,sample_id
biopsy_A,/data/biopsy_A.svs,PATIENT_001
biopsy_B,/data/biopsy_B.svs,PATIENT_001
resection,/data/resection.svs,PATIENT_002
```

Slides sharing a `sample_id` are processed individually through tessellation and feature extraction,
then their per-slide feature H5 files are concatenated into one sample-level H5 and PT by
`aggregate_sample_features` (CPU-only — no re-inference). When `sample_id` is omitted it defaults
to `slide_id`, so existing CSVs continue to work without modification.

Requires `params.featurize.save_features_to_h5 = true` (per-slide H5 files are the aggregation input).

**Output** (in addition to per-slide outputs):
```
results/features/{model_type}/{sample_id}.features.h5
results/features/{model_type}/{sample_id}.features.pt
```

**Subsampling** (when total tiles exceed a budget):

| Parameter | Default | Description |
|---|---|---|
| `featurize.max_tiles_per_sample` | `null` | Max tiles per sample after concatenation |
| `featurize.subsampling_strategy` | `"random"` | `"random"`, `"proportional"`, or `"equal"` |
| `featurize.subsampling_seed` | `42` | Random seed for reproducibility |

**Example:**
```bash
nextflow run main.nf \
  --samples_csv multi_patient.csv \
  --featurize.model_types='["optimus"]' \
  --featurize.save_features_to_h5=true \
  --featurize.max_tiles_per_sample=20000 \
  --featurize.subsampling_strategy=proportional
```


## Integration tests

Integration tests use [nf-test](https://www.nf-test.com) and are run via `make`:

```bash
make test                  # run all tests (requires GPU)
make test-standard         # one-step workflow
make test-two-step         # two-step workflow
make test-wds              # WebDataset flat sharding
make test-wds-grouped      # WebDataset grouped sharding (by oncotree_code)
make test-multi-slide      # multi-slide sample aggregation
```

Stub tests run without a GPU and are used in CI:

```bash
make test-stub             # one-step stub
make test-stub-two-step    # two-step stub
make test-stub-filter      # two-step + tile filtering stub
make test-stub-wds         # WebDataset stub
make test-stub-wds-grouped # WebDataset grouped stub
make test-stub-clip        # CLIP annotation stub
make test-stub-multi-slide # multi-slide aggregation stub
make test-stub-all         # all stubs in one pass
```

Extra Nextflow profiles (e.g. `conda`, `slurm,cluster`) can be composed:

```bash
make test PROFILES=conda
make test PROFILES=slurm,cluster
make test-standard NXF_ARGS=-resume
```

The test slide (`tests/testdata/948176.svs`) is vendored in the repository.
Override the slide used for the standard/two-step/WDS tests:

```bash
make test MUSSEL_TEST_SLIDE=/path/to/other.svs
```

## Azure Batch support

1. Create an Azure storage account and batch account.

2. Modify the nextflow configuration files as necessary (see <https://www.nextflow.io/docs/edge/azure.html>)

3. Set the necessary secrets using `nextflow secrets set`. At a minimum set `AZURE_BATCH_KEY` and `AZURE_STORAGE_KEY`.

4. Launch nextflow:
    ```
    nextflow -Dcom.amazonaws.sdk.disableCertChecking=true run main.nf -bucket-dir {azure_bucket_dir} -profile docker,cloud
    ```
    where `{azure_bucket_dir}` is an azure path like `az://test/nftest`.

5. When the execution completes, results will be in the `results` directory

### Azure notes

#### Disk management and unusable nodes

The Azure Batch nodes have poor disk space management such that if you run a
lot of jobs, they will inevitably run out of disk space, putting the node into
the unusable state. One possible solution is to delete the unusable nodes
which can be done automatically with a powershell script. A better solution is
to mount Azure file shares with large files to the batch nodes using
`params.azure.storage.fileShares`. It's a good idea to periodically run the
powershell script either way as nodes end up in the unusable state for a variety of
reasons and will linger (costing $) until deleted.

## Troubleshooting

### AttributeError

Error: 
```
AttributeError: 'Attention' object has no attribute 'norm'
```

Solution:

When this occurs it means that a dependency has a version mismatch between what 
was loaded into the pickle file and what Mussel is using. It is best to not use the 
pickle file and instead use this Mussel 
[feauture](https://github.com/pathology-data-mining/Mussel/blob/main/README-commands.md#save_model) 
to automatically download the models from huggingface.

### Cache filling up

Error:

On our on-prem machines, the uv and huggingface caches are by default set to user's home directory. 
This mount fills up quickly so it is best to move this cache elsewhere.

Solution:

Move the uv and huggingface cache directory to a different mount by setting the environment
variables `UV_CACHE_DIR` and `HF_HOME`. 

### Conflicting Huggingface Downloads

Error:

Launching multiple jobs without the models already downloaded in parallel to slurm can cause 
job failures since they will all try to save the models at the same time.

Solution:

Run your workflow for a single slide first as a 'dry-run' to properly download the models, then 
re-run with multiple slides afterwards.
