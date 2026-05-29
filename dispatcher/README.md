# mussel-dispatcher

A streaming, crash-tolerant dispatcher that runs the `mussel-nf` Nextflow pipeline continuously as new slides become available — including live TCGA GDC data.

## Overview

The dispatcher solves the scheduling problem of processing large slide collections (e.g., all TCGA DX1 slides) where:
- Slides trickle in over time (GDC inventory changes, downloads complete)
- Each Nextflow run has overhead, so batching is desirable
- Downloads and featurization should overlap (not run sequentially)
- Multiple NF runs should run in parallel up to cluster capacity
- Crashes should not lose progress or duplicate work

```
┌─────────────────────────────────────────────────────────────────────┐
│                        mussel-dispatcher                            │
│                                                                     │
│  Watcher(s)                  BatchScheduler       RunManager        │
│  ──────────                  ─────────────        ──────────        │
│  TcgaWatcher ──────┐         fires when:          up to N concurrent│
│  DatabricksWatcher─┼──queue─▶ • N slides ready   NF runs; each run │
│  LocalWatcher ─────┤         • timeout elapsed   calls post_batch  │
│  S3Watcher ────────┘                              hooks on success  │
│                                                                     │
│  StateStore (SQLite) tracks all slides/batches for crash recovery   │
└─────────────────────────────────────────────────────────────────────┘
```

## Usage

```bash
# Start the dispatcher with a YAML config
python dispatcher/mussel-dispatcher.py dispatcher/tcga_dispatcher.yaml

# Collect and merge all per-run manifests into one file
python dispatcher/mussel-dispatcher.py collect-manifests dispatcher/tcga_dispatcher.yaml

# Launch the monitoring dashboard (default port 8050)
python dispatcher/dashboard.py dispatcher/tcga_dispatcher.yaml --port 8050
```

## How It Works

### 1. Watchers

Watchers run in background threads and push slides onto a shared queue.

| Watcher | Source |
|---|---|
| `local` | Directory on disk — polls for new `.svs`/`.tiff` files |
| `s3` | S3-compatible bucket — polls for new objects by prefix |
| `tcga` | GDC API — syncs inventory, resolves paths, downloads missing slides |
| `databricks` | Databricks SQL warehouse — queries IMPACT-matched slide inventory |

Multiple watchers can run simultaneously (e.g., local + tcga).

> **TCGA deep-dive:** see [docs/tcga.md](docs/tcga.md) for slide types, sample types, script reference, WDS format, and design notes.

#### TcgaWatcher (streaming TCGA processing)

On every poll cycle (`poll_interval_seconds`, default 3600 s):

1. **Sync inventory** — calls `mussel_dispatcher.tcga.sync_inventory` to fetch the latest GDC file listing. Skips re-fetch if the existing CSV is younger than `gdc_max_age_hours` (default 24 h).

2. **Update status** — calls `mussel_dispatcher.tcga.update_status` to scan the results directory and mark which slides already have features.

3. **Resolve paths** — calls `mussel_dispatcher.tcga.prepare_samples --skip-done --model <model>`, which writes two files:
   - `*_dispatcher.csv` — sample sheet (`slide_id, slide_path, oncotree_code`)
   - `*_dispatcher.meta.csv` — resolution details including `needs_download` flag

4. **Enqueue ready slides** — slides with a local or S3 path go directly onto the pending queue. Already-known slides (in StateStore) are skipped.

5. **Download missing slides** — if `download_enabled: true`, slides with `needs_download: true` are submitted to a `ThreadPoolExecutor`. Each download runs `gdc-client download` and enqueues the slide as soon as it completes.

**Key throughput property:** downloads for batch N+1 overlap with featurization of batch N, and up to `max_concurrent_runs` Nextflow jobs run simultaneously.

#### DatabricksWatcher (MSK IMPACT slides)

Queries a Databricks SQL warehouse to discover slides from the MSK IMPACT-matched cohort. On every poll cycle (`poll_interval_seconds`, default 86400 s = 1 day, since IMPACT tables update infrequently):

1. **Query warehouse** — executes a SQL join of `impact_matched_slides` and `slide_inventory` to retrieve `(slide_id, slide_path, oncotree_code)` tuples for all slides with a valid S3 path.
2. **Enqueue new slides** — slides not already in the StateStore are added and pushed to the dispatch queue. Already-known slides are skipped.

Credentials are resolved by the Databricks SDK in priority order:
1. `DATABRICKS_HOST` + `DATABRICKS_TOKEN` environment variables
2. `~/.databrickscfg` DEFAULT profile

### 2. BatchScheduler

Accumulates slides from the queue and fires a batch when either:
- **Size trigger**: `batch_size` slides are ready (default 20)
- **Time trigger**: `max_wait_seconds` have elapsed since the first slide arrived (default 300 s)

This prevents holding slides hostage waiting for a full batch while also avoiding per-slide NF invocations.

### 3. RunManager

Maintains a pool of up to `max_concurrent_runs` Nextflow runner threads. Each runner:

1. Writes a `batch-{id}.csv` samples file to `dispatch_dir/`
2. Runs `nextflow run main.nf` with the configured profiles and params
3. On **success**: calls `post_batch_hooks` (see below), marks slides complete in StateStore
4. On **failure**: marks batch failed; if `retry_failed: true`, slides are re-enqueued on the next restart

### 4. post_batch_hooks

After each successful Nextflow run, the dispatcher executes a list of commands with template substitution:

| Variable | Value |
|---|---|
| `{batch_csv}` | Path to the batch samples CSV (`slide_id, slide_path, oncotree_code`) |
| `{batch_id}` | Unique batch identifier (timestamp + random suffix) |
| `{outdir}` | Nextflow output directory |
| `{repo_dir}` | Repository root |

Example use: append newly extracted features to WebDataset shards immediately after each batch, rather than in a separate post-processing job:

```yaml
post_batch_hooks:
  - command: "python {repo_dir}/scripts/append_wds.py"
    args:
      - "--pt-dir={outdir}/features/ctranspath/pt"
      - "--h5-dir={outdir}/features/ctranspath/tile_h5"
      - "--wds-dest=s3://bucket/wds/ctranspath"
      - "--slide-ids-csv={batch_csv}"   # restrict to this batch's slides
```

Hook failures are logged but do **not** abort the dispatcher or affect slide status.

### 5. StateStore (crash recovery)

All slides and batches are tracked in a SQLite database (`state_dir/state.db`). On restart:

- Batches that were `RUNNING` when the process died are reset to `FAILED`
- Their slides are reset to `PENDING` and re-enqueued for the next dispatch cycle
- Slides already marked `COMPLETE` are never re-processed

This means the dispatcher is safe to kill/restart at any time without losing progress or duplicating work.

## Configuration

The config is a YAML file. See [`tcga_dispatcher.yaml`](tcga_dispatcher.yaml) for a fully annotated TCGA example and [`dispatcher.yaml`](dispatcher.yaml) for a local/S3 example.

### Top-level fields

| Field | Default | Description |
|---|---|---|
| `repo_dir` | required | Path to `mussel-nf` checkout |
| `nextflow_profiles` | required | Comma-separated NF profiles (e.g. `cluster,apptainer`) |
| `outdir` | required | Nextflow `--outdir` |
| `work_base_dir` | required | Nextflow work directory root |
| `dispatch_dir` | required | Where batch CSVs are written |
| `state_dir` | required | SQLite state DB location |
| `log_dir` | required | Per-batch NF log files |
| `batch_size` | `20` | Slides per NF run |
| `min_batch_size` | `1` | Minimum to dispatch (at shutdown, any size is flushed) |
| `max_wait_seconds` | `300` | Time trigger for dispatching a partial batch |
| `max_concurrent_runs` | `2` | Parallel Nextflow jobs |
| `retry_failed` | `true` | Re-enqueue slides from crashed batches on restart |
| `cleanup_work_dir` | `false` | Delete NF work dir after successful run |
| `post_batch_hooks` | `[]` | Commands to run after each successful batch |
| `watchers` | required | List of watcher configs (see below) |

### TcgaWatcher fields (`type: tcga`)

| Field | Default | Description |
|---|---|---|
| `inventory_csv` | required | Path to TCGA inventory CSV (managed by `sync_inventory`) |
| `status_csv` | required | Path to per-slide status CSV |
| `results_dir` | required | NF `outdir` to scan for completed features |
| `model` | `ctranspath` | Feature model — used to determine which slides are already done |
| `local_slides_dir` | `""` | Directory to check for slides already on disk |
| `s3_base` | `""` | S3 URI prefix to check for slides (e.g. `s3://bucket/slides`) |
| `s3_endpoint` | `""` | S3-compatible endpoint URL |
| `s3_access_key` | `""` | S3 access key (or set `ECS_ACCESS_KEY` env var) |
| `s3_secret_key` | `""` | S3 secret key (or set `ECS_SECRET_KEY` env var) |
| `project` | `""` | Restrict to one TCGA project (e.g. `TCGA-BRCA`); empty = all |
| `slide_type` | `DX1` | Restrict to slide type (`DX1`, `DX2`, `all`) |
| `poll_interval_seconds` | `3600` | How often to poll GDC |
| `gdc_max_age_hours` | `24.0` | Skip inventory re-fetch if CSV is younger than this |
| `download_enabled` | `false` | Automatically download slides via `gdc-client` |
| `download_dir` | `""` | Where `gdc-client` writes files (defaults to `local_slides_dir`) |
| `download_concurrency` | `4` | Parallel download threads |
| `gdc_token_file` | `""` | Path to GDC user token for controlled-access data |
| `wds_destinations` | `{}` | `{model: s3_or_local_path}` — auto-generates `append_wds` post-batch hook per model |
| `wds_staging_dir` | `""` | Local staging dir for S3 WDS destinations |
| `wds_s3_max_concurrency` | `4` | Boto3 multipart upload threads per S3 write |
| `databricks_volume_folder` | `""` | Databricks volume folder to sync status Parquet to (e.g. `/Volumes/cat/schema/vol/`) |
| `databricks_volume_path` | `""` | Full Databricks volume path for the Parquet file (alternative to `volume_folder`) |
| `databricks_table` | `""` | Delta table name to refresh after Parquet sync |
| `databricks_job_id` | `""` | Databricks job ID to trigger after Parquet sync |

### DatabricksWatcher fields (`type: databricks`)

| Field | Default | Description |
|---|---|---|
| `warehouse_id` | required | Databricks SQL warehouse ID to run queries against |
| `source_filter` | `[]` | List of `slide_inventory.source` values to include (e.g. `['ECS2']`); empty = all |
| `additional_where` | `""` | Extra SQL `WHERE` clause appended with `AND` |
| `min_file_size_mb` | `10.0` | Skip slides smaller than this (MB) |
| `poll_interval_seconds` | `86400` | How often to poll (default 1 day — IMPACT tables update infrequently) |
| `wds_destinations` | `{}` | `{model: s3_or_local_path}` — auto-generates `append_wds` post-batch hook per model |
| `wds_staging_dir` | `""` | Local staging dir for S3 WDS destinations |
| `wds_s3_max_concurrency` | `4` | Boto3 multipart upload threads per S3 write |

**Requires:** `pip install databricks-sdk`

**Example config:**

```yaml
watchers:
  - type: databricks
    warehouse_id: abc123def456
    source_filter: [ECS2]
    min_file_size_mb: 50
    poll_interval_seconds: 86400
    wds_destinations:
      hoptimus1: s3://my-bucket/wds/hoptimus1
    wds_staging_dir: /data/wds-staging
```

## Credentials & Secrets

S3 credentials must be provided to both the **dispatcher** (for its boto3 S3 watcher / WDS push operations) and to **Nextflow itself** (for pipeline tasks that read/write S3). These are configured independently.

### Nextflow secrets (recommended)

Nextflow has a built-in encrypted secret store (`~/.nextflow/secrets/`). The dispatcher can read from it — set credentials once and both the dispatcher and the pipeline will use them.

```bash
nextflow secrets set AWS_ACCESS_KEY_ID     your-access-key
nextflow secrets set AWS_SECRET_ACCESS_KEY your-secret-key
```

Then reference them in the dispatcher YAML:

```yaml
nextflow_secrets:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
```

The dispatcher maps these to its `s3_access_key`/`s3_secret_key` fields automatically. Nextflow uses them directly via `secrets.AWS_ACCESS_KEY_ID` in `nextflow.config`.

If your S3-compatible endpoint requires a custom URL, also add it to the watcher config:

```yaml
watchers:
  - type: tcga          # or reef_v2
    s3_endpoint: https://your-s3-compatible-endpoint:9000
    # s3_access_key / s3_secret_key not needed — loaded from nextflow_secrets above
```

### Shell env file (alternative)

Create a plain env file (do **not** commit it):

```bash
# secrets.env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

Reference it in the dispatcher YAML:

```yaml
secrets_env_file: /path/to/secrets.env
```

### Standard AWS credential chain

For real AWS S3 (no custom endpoint), no dispatcher config is needed if credentials are already available via the [standard AWS chain](https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html): `~/.aws/credentials`, instance profile, or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` environment variables.

You still need to set the Nextflow secrets so the pipeline can authenticate:

```bash
nextflow secrets set AWS_ACCESS_KEY_ID     your-access-key
nextflow secrets set AWS_SECRET_ACCESS_KEY your-secret-key
```

### HuggingFace gated models

```bash
nextflow secrets set HF_TOKEN your-hf-token
```

### Azure Batch

1. Create an Azure storage account and batch account.

2. Modify the Nextflow configuration files as necessary (see <https://www.nextflow.io/docs/latest/azure.html>).

3. Set the required secrets:
   ```bash
   nextflow secrets set AZURE_STORAGE_KEY your-storage-key
   nextflow secrets set AZURE_BATCH_KEY   your-batch-key
   ```

4. Launch Nextflow:
   ```bash
   nextflow run main.nf -bucket-dir az://your-container/nfwork -profile docker,cloud
   ```

#### Disk management and unusable nodes

Azure Batch nodes have poor disk space management — after many jobs they can run out of disk
and enter an unusable state. Options:

- Delete unusable nodes automatically with a PowerShell script.
- Mount Azure File Shares for large files via `params.azure.storage.fileShares` to avoid local
  disk pressure.

It's a good idea to run the cleanup script periodically regardless, as nodes can end up unusable
for various reasons and will linger (costing money) until deleted.

---

## Slide Path Resolution

`mussel_dispatcher.tcga.prepare_samples` resolves each TCGA slide in this priority order:

1. **Local disk** (`local_slides_dir`) — if the `.svs` file exists locally
2. **S3** (`s3_base`) — if the object exists in the configured bucket
3. **Needs download** — sets `needs_download=true` in the meta CSV

The dispatcher uses this to enqueue ready slides immediately and, if `download_enabled`, kick off background downloads for the rest.

## Running Tests

```bash
python -m pytest dispatcher/test_dispatcher.py -v
```

Tests cover: StateStore CRUD + lifecycle, BatchScheduler triggers, RunManager concurrency, crash recovery, TcgaWatcher enqueue/download/skip logic, and post_batch_hooks template substitution.

---

## Monitoring Dashboard

`dashboard.py` is a self-contained real-time monitoring dashboard (Python stdlib HTTP server + embedded HTML/JS). It reads the same YAML config as the dispatcher and connects to the same SQLite state DB.

```bash
# Launch (runs on port 8050 by default)
python dispatcher/dashboard.py dispatcher/tcga_dispatcher.yaml --port 8050
# Then open http://localhost:8050/ (or SSH tunnel)
```

### Panels

#### Slide Funnel
- **Pending** / **Dispatched** / **Done** slide counts from the StateStore
- Global % complete, estimated from completed batches + fractional progress of in-flight batches (parsed from NF logs)

#### Batch Table
Live table of all active and recently-completed batches, updated every 10 s:

| Column | Description |
|---|---|
| Batch ID | Timestamp + random suffix; links to log viewer |
| Status | `RUNNING` / `SUCCEEDED` / `FAILED` |
| Slides | Number of slides in batch |
| Duration | Elapsed time (⏳ indicator for running batches) |
| Tasks | NF task progress: `done/total (%)` parsed from log |
| SLURM | Running ▶ / Pending ⏳ SLURM jobs cross-referenced by work dir |
| Alerts | ⚠ WARN count / 🚨 ERROR details / 🔥 infra-stop events |

Click any batch row to open the live log viewer (streams the last 200 lines of the NF log).

#### SLURM Panel
Live `squeue` data (refreshed every 15 s):
- Running / pending job counts, unique node list
- Pending reasons breakdown (e.g. `Resources`, `Priority`)

`sacct` history for the last 24 h:
- Completed / failed / cancelled / timeout counts + average elapsed time
- **Failure type breakdown** — classified per job by reading `.command.err` from the NF task work directory:

| Type | Signal |
|---|---|
| `oom_gpu` | "CUDA out of memory" in stderr |
| `oom_host` | exit 137 or "oom-kill" in stderr |
| `sigterm` | exit 143, or SLURM CANCELLED state (infra restart / dispatcher stop) |
| `disk_full` | "no space left" in stderr |
| `s3_error` | S3 URL + exception in stderr |
| `python_error` | Python traceback in stderr |
| `timeout` | SLURM TIMEOUT state |
| `error_exit1` | exit 1 with no specific pattern |
| `unknown` | any other non-zero exit |

#### S3 / WDS Panel
If an S3 watcher is configured, shows per-model shard counts and total objects in the WDS destination prefix.

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/status` | Slide counts, pct_done, S3 stats |
| `GET /api/batches` | All batch rows with NF log info + SLURM cross-ref |
| `GET /api/logs/{batch_id}` | Last 200 lines of the NF log for a batch |
| `GET /api/wds` | WDS shard counts from S3 |
| `GET /api/slurm` | squeue + sacct stats with failure type breakdown |
