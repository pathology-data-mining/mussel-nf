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
│  Watcher(s)              BatchScheduler       RunManager            │
│  ──────────              ─────────────        ──────────            │
│  TcgaWatcher ──┐         fires when:          up to N concurrent    │
│  LocalWatcher ─┼──queue─▶ • N slides ready   NF runs; each run     │
│  S3Watcher ────┘         • timeout elapsed   calls post_batch_hooks │
│                                               on success            │
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
| `inventory_csv` | required | Path to TCGA inventory CSV (managed by `tcga_sync_inventory.py`) |
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
| `scripts_dir` | `""` | Path to `scripts/tcga/`; defaults to `{repo_dir}/scripts/tcga` |

## Slide Path Resolution

`tcga_prepare_samples.py` resolves each slide in this priority order:

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
