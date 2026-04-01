# Mussel-NF pipeline

A pipeline for running [Mussel](https://github.com/pathology-data-mining/Mussel) (pinned to v1.3.0).

## Requirements

* Unix-like operating system
* Java 17+
* Nextflow ≥ 24.x (uses [nf-schema](https://nextflow-io.github.io/nf-schema/) plugin)
* One of: Docker, Apptainer, or a Conda/Mamba environment

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

    `samples.csv` must have columns `slide_id` and `slide_path` (required), plus optional `oncotree_code`.
    Accepted slide extensions: `.svs`, `.tiff`, `.tif`, `.ndpi`, `.scn`.

3. When the execution completes, results will be in `params.outdir` (default: `results/`).

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
| `abmil_slide` | (encoder-agnostic — specify patch encoder separately) |

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

These options remove artifacts **during tessellation** by refining the tissue mask before patches are extracted. No separate classifier model is needed.

- **`remove_artifacts`**: runs the GrandQC neural artifact remover to exclude ink, air bubbles, and other slide artifacts from the tissue mask
- **`remove_penmarks`**: detects and excludes pen mark regions before tessellation
- **`seg_model`**: tissue segmentation backend — `'classic'` (HSV + fixed threshold, default), `'otsu'` (HSV + Otsu automatic threshold), or `'neural'` (DeepLabV3 neural network)
- **`min_tissue_proportion`**: drop patches where less than this fraction of pixels are tissue (default `0.0`)
- **`overlap`**: extract overlapping patches (pixels, default `0`)
- **`slide_mpp_override`**: override the slide's MPP value when metadata is missing or incorrect

These options can be combined with legacy tile filtering or used independently.

### CLIP-based annotation

When `params.clip.model_types` is non-empty (e.g. `['quiltnet']`), the standard feature extraction
workflow runs, then tile annotation and tile caching are performed.

Default annotation classes are set via `params.clip.default_classes`, or per oncotree code via
`params.clip.oncotree_class_csv` (two columns: `oncotree_code`, `class`). The sample sheet
`oncotree_code` column is used to look up classes per slide.

### Linear probe benchmarking

If `params.linear_probe.annotations_csv` is specified, the linear probe benchmarking workflow runs.
The CSV must have columns `slide_id` and `annotation_bmp_path`.

1. Tessellation
2. Feature extraction
3. Map tile coordinates to annotation classes using the BMP mask and `params.linear_probe.annotation_class_mapping_yaml`
4. Combine per-slide annotation mappings
5. Benchmark logistic regression classifiers

### WebDataset sharding

After feature extraction, completed `.pt` slide-feature files (and optionally `.h5` patch-feature
files) can be packed into [WebDataset](https://github.com/webdataset/webdataset)-compatible `.tar`
shards.  Shards are directly readable by the `webdataset` Python library via:

```python
import webdataset as wds
ds = wds.WebDataset("results/wds/optimus/all/shard-{000000..000004}.tar")
for sample in ds:
    slide_id = sample["__key__"]
    features = torch.load(io.BytesIO(sample["pt"]))
```

Enable and configure sharding via `params.wds`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wds.enabled` | `false` | Enable WDS shard output |
| `wds.group_by_oncotree` | `false` | Separate shard sets per `oncotree_code`; requires that column in `samples_csv` |
| `wds.shard_h5` | `false` | Also bundle `.h5` patch-feature files (large) |
| `wds.max_shard_size` | `1000` | Max slides per `.tar` shard |
| `wds.shard_prefix` | `"shard-"` | Filename prefix, e.g. `shard-000000.tar` |

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
results/wds/optimus/BRCA/shard-000000.tar
results/wds/optimus/BRCA/shard-000001.tar
results/wds/optimus/LUAD/shard-000000.tar
...
```

Without grouping (`group_by_oncotree=false`):

```
results/wds/optimus/all/shard-000000.tar
results/wds/optimus/all/shard-000001.tar
...
```


## Integration tests

A single-slide integration test is included for both workflow modes:

```bash
bash tests/run_integration_test.sh
```

Requires `tests/data/1079807.svs` to be present (not tracked by git — copy or symlink a real slide file).

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
