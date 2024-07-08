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

## Azure Batch Support

1. Create an Azure storage account and batch account.

2. Modify the nextflow configuration files as necessary (see <https://www.nextflow.io/docs/edge/azure.html>)

3. Set the necessary secrets using `nextflow secrets set`. At a minimum set `AZURE_BATCH_KEY` and `AZURE_STORAGE_KEY`.

4. Launch nextflow:
    ```
    nextflow -Dcom.amazonaws.sdk.disableCertChecking=true run main.nf -bucket-dir {azure_bucket_dir} -profile docker,cloud
    ```
    where `{azure_bucket_dir}` is an azure path like `az://test/nftest`.

5. When the execution completes, results will be in the `results` directory
