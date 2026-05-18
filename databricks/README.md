# Databricks Integration

Sync TCGA slide metadata and feature extraction status to Databricks for downstream ML workloads.

## Architecture

```
[mussel-dispatcher: batch succeeds]
    → post_batch_hook: tcga_sync_databricks.py
        1. joins tcga_status.csv + tcga_inventory.csv → Parquet
        2. uploads  /Volumes/your_catalog/your_schema/tcga_dispatcher/
                        tcga_inventory_<YYYYMMDDTHHMMSS>.parquet
        3. triggers Databricks job → notebooks/tcga_metadata_sync.py
                MERGE INTO your_catalog.your_schema.tcga_slide_embeddings_v2
```

## Delta Table

**Target:** `your_catalog.your_schema.tcga_slide_embeddings_v2`  
**MERGE key:** `(file_id, model)` — one row per slide × feature model (e.g. `hoptimus1`, `titan_slide`)

Schema is inferred from the Parquet on first run (table auto-created if absent). Key columns:

| Column | Source | Description |
|---|---|---|
| `file_id` | inventory | GDC file UUID (primary key part 1) |
| `model` | status | Feature model name (primary key part 2) |
| `slide_id` | status | TCGA slide identifier |
| `project_id` | inventory | e.g. `TCGA-BRCA` |
| `slide_type` | inventory | `DX1`, `DX2`, `BS`, `TS`, … |
| `status` | status | `pending` / `done` / `failed` |
| `pt_path` | status | S3 path to `.features.pt` slide-level embedding |
| `h5_path` | status | S3 path to `.patch.h5` patch coordinates |
| `last_updated` | status | ISO timestamp of last status update |
| `primary_site` | inventory | Tissue of origin |
| `primary_diagnosis` | inventory | Cancer diagnosis |
| `ajcc_pathologic_stage` | inventory | Pathologic stage |
| `sample_type` | inventory | `Primary Tumor`, `Solid Tissue Normal`, … |
| *(+ other clinical columns)* | inventory | gender, age, vital_status, morphology, … |

An audit table `your_catalog.your_schema.tcga_sync_audit` is automatically maintained with one row per sync run (source file, row counts, timestamp).

## Setup

### 1. Create the Unity Catalog volume folder

In Databricks SQL or a notebook:

```sql
CREATE VOLUME IF NOT EXISTS your_catalog.your_schema.tcga_dispatcher;
```

### 2. Import the notebook

Upload `databricks/notebooks/tcga_metadata_sync.py` to your Databricks workspace, e.g.:

```bash
databricks workspace import \
  databricks/notebooks/tcga_metadata_sync.py \
  /Repos/mussel-nf/databricks/notebooks/tcga_metadata_sync \
  --language PYTHON
```

Or use the Repos integration if the repo is connected to your workspace.

### 3. Create the Workflow job

```bash
databricks jobs create --json @databricks/jobs/tcga_metadata_sync_job.json
```

Note the job ID printed in the output — you'll need it for the dispatcher config.

Update `notebook_path` in the JSON if your workspace import path differs.

### 4. Configure the dispatcher

In `dispatcher/tcga_dispatcher.yaml`, set:

```yaml
databricks_volume_path: /Volumes/your_catalog/your_schema/tcga_dispatcher
databricks_job_id: "<job-id-from-step-3>"
databricks_table: your_catalog.your_schema.tcga_slide_embeddings_v2
```

Export credentials before starting the dispatcher:

```bash
export DATABRICKS_HOST=https://<workspace>.azuredatabricks.net
export DATABRICKS_TOKEN=<personal-access-token>
```

Alternatively, the script reads credentials from `~/.databrickscfg` (the standard Databricks CLI config file) if the environment variables are not set. The `[DEFAULT]` profile is used.

### 5. Verify

After the next successful batch, check the table in Databricks SQL:

```sql
SELECT status, COUNT(*) AS n
FROM your_catalog.your_schema.tcga_slide_embeddings_v2
GROUP BY status
ORDER BY status;
```

## Running Manually

```bash
python scripts/tcga/tcga_sync_databricks.py \
    --status  dispatcher/tcga_status.csv \
    --inventory dispatcher/tcga_inventory.csv \
    --volume-folder /Volumes/your_catalog/your_schema/tcga_dispatcher \
    --table your_catalog.your_schema.tcga_slide_embeddings_v2 \
    --job-id <job-id> \
    --verbose
```

## Notebook Parameters

The Databricks notebook (`notebooks/tcga_metadata_sync.py`) accepts two parameters
(passed automatically by the job trigger, or set manually via widgets):

| Parameter | Default | Description |
|---|---|---|
| `volume_folder` | `/Volumes/your_catalog/your_schema/tcga_dispatcher` | UC volume folder containing Parquet files |
| `target_table` | `your_catalog.your_schema.tcga_slide_embeddings_v2` | Delta table to MERGE INTO |

The notebook always picks the **lexicographically latest** `tcga_inventory_*.parquet` file in the folder.
