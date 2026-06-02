# Databricks Integration

Sync slide metadata and feature extraction status to Databricks for downstream ML workloads.

## Architecture

```
[mussel-dispatcher: batch succeeds]
    → post_batch_hook: dispatcher sync script
        1. joins status.csv + inventory.csv → Parquet
        2. uploads  /Volumes/your_catalog/your_schema/your_volume/
                        inventory_<YYYYMMDDTHHMMSS>.parquet
        3. triggers Databricks job → notebooks/metadata_sync.py
                MERGE INTO your_catalog.your_schema.slide_embeddings
```

## Delta Table

**Target:** `your_catalog.your_schema.slide_embeddings`  
**MERGE key:** `(slide_id, model)` — one row per slide × feature model (e.g. `hoptimus1`, `titan_slide`)

Schema is inferred from the Parquet on first run (table auto-created if absent). Key columns:

| Column | Source | Description |
|---|---|---|
| `slide_id` | status | Slide identifier (primary key part 1) |
| `model` | status | Feature model name (primary key part 2) |
| `status` | status | `pending` / `done` / `failed` |
| `pt_path` | status | Path to `.features.pt` slide-level embedding |
| `h5_path` | status | Path to `.patch.h5` patch coordinates |
| `last_updated` | status | ISO timestamp of last status update |
| *(+ other clinical/metadata columns)* | inventory | From the source inventory CSV |

An audit table `your_catalog.your_schema.sync_audit` is automatically maintained with one row per sync run (source file, row counts, timestamp).

## Setup

### 1. Create the Unity Catalog volume folder

In Databricks SQL or a notebook:

```sql
CREATE VOLUME IF NOT EXISTS your_catalog.your_schema.mussel_dispatcher;
```

### 2. Import the notebook

Upload `databricks/notebooks/metadata_sync.py` to your Databricks workspace, e.g.:

```bash
databricks workspace import \
  databricks/notebooks/metadata_sync.py \
  /Repos/mussel-nf/databricks/notebooks/metadata_sync \
  --language PYTHON
```

Or use the Repos integration if the repo is connected to your workspace.

### 3. Create the Workflow job

```bash
databricks jobs create --json @databricks/jobs/metadata_sync_job.json
```

Note the job ID printed in the output — you'll need it for the dispatcher config.

Update `notebook_path` in the JSON if your workspace import path differs.

### 4. Configure the dispatcher

In your dispatcher YAML, set:

```yaml
databricks_volume_path: /Volumes/your_catalog/your_schema/mussel_dispatcher
databricks_job_id: "<job-id-from-step-3>"
databricks_table: your_catalog.your_schema.slide_embeddings
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
FROM your_catalog.your_schema.slide_embeddings
GROUP BY status
ORDER BY status;
```

## Notebook Parameters

The Databricks notebook (`notebooks/metadata_sync.py`) accepts parameters
(passed automatically by the job trigger, or set manually via widgets):

| Parameter | Default | Description |
|---|---|---|
| `volume_folder` | `/Volumes/your_catalog/your_schema/mussel_dispatcher` | UC volume folder containing Parquet files |
| `target_table` | `your_catalog.your_schema.slide_embeddings` | Delta table to MERGE INTO |
| `merge_key` | `slide_id` | Column used as the row identifier in the MERGE condition |
| `filename_prefix` | `""` | Only pick Parquet files whose name starts with this prefix (empty = any `.parquet` file) |

The notebook always picks the **lexicographically latest** matching `.parquet` file in the volume folder.
