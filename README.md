# Mussel-NF pipeline

A pipeline for running [Mussel](https://github.com/pathology-data-mining/Mussel).

## Requirements

* Unix-like operating system
* Java 11

## Quickstart

1. Install docker

2. Install nextflow:
    ```
    curl -s https://get.nextflow.io | bash
    ```

3. Launch the pipeline execution using docker
    ```
    ./nextflow run pathology-data-mining/mussel-nf -profile standard,docker --samples_csv samples.csv
    ```
    `samples.csv` is a csv file with three named columns: `slide_id`, `slide_path`, and `oncotree_code`.
        * `slide_id`: slide ID
        * `slide_path`: path to slide
        * `oncotree_code`: Oncotree code. Optional column for the QuiltNet workflow. If not specified, quiltnet uses `params.quiltnet_default_classes`.

4. When the execution completes, results will be in the `params.outdir` directory

## Misc Notes

* See full parameters with `--help` or `--helpFull` option.
* Set `params.publish_slide_prefix` to true to use a slide prefix in the publish directory.
* The pipeline outputs a manifest automatically in `params.outdir`, but it can
  also be manually built with `scripts/create_manifest.py`. Partial manifest
  results can be found in `{params.outdir}/tmp`.
* If using docker, it's best to keep the models in the docker container, despite how large they can be.

## Workflows

The standard workflow tessellates and extracts features for the specified `params.model_types`.

### Tile filtering

1. Tessellation

2. Feature extraction for `params.tiling.filter_model_type`

3. Filter tiles using `mussel.cli.filter_features`

4. Feature extraction for `params.featurize.model_types`

### CLIP-based models

When a clip-based model is specified (for now, only `quiltnet` is supported),
the standard workflow runs in addition to tile annotation, and tile caching.
Default annotation classes can be specified (`params.clip.default_classes`) or the
classes can be determined from `params.clip.oncotree_class_csv` that maps oncotree codes to
classes and `oncotree_code` in the sample sheet. The optional
`params.clip.oncotree_class_csv` has the format of two columns: oncotree code
and class.

### Linear probe benchmarking

If `params.linear_probe.annotations_csv` is specified, the linear probe benchmarking workflow will
run. The csv must have two named columns: `slide_id` and `annotation_bmp_path`.

1. Tessellation

2. Feature extraction

3. Map feature tiles to annotation classes using the annotation bmp file (in
   `params.linear_probe.annotations_csv`) and `params.linear_probe.annotation_class_mapping_yaml`.

4. Combine annotation tile mappings

5. Benchmark linear models


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
