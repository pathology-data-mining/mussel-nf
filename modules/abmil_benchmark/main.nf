/**
 * ABMIL Benchmark module
 *
 * Trains a Gated-ABMIL classifier on per-slide H5 feature files and reports
 * AUROC across multiple seeds with 95% bootstrap CIs.  Designed as a drop-in
 * downstream benchmark for the precision-benchmarking workflow.
 *
 * Required params (under params.abmil_benchmark):
 *   labels_parquet  – parquet with columns: slide_id, <target_col>, [split_col]
 *   target_col      – binary label column (integer 0/1)
 *
 * Optional params mirror AbmilBenchmarkConfig defaults; see nextflow.config.
 */

process ABMIL_BENCHMARK {
    label "bigTask"

    publishDir path: { "${params.outdir}/abmil_benchmark/${model_type}/" }, mode: "${params.publish_mode}"

    input:
    tuple val(model_type), path(h5_files, stageAs: 'features_dir/*')
    path(labels_parquet)

    output:
    tuple val(model_type), path("results.json"), emit: results_json

    script:
    def n_seeds      = params.abmil_benchmark.n_seeds      ?: 3
    def n_bootstrap  = params.abmil_benchmark.n_bootstrap  ?: 1000
    def random_state = params.abmil_benchmark.random_state ?: 42
    def n_epochs     = params.abmil_benchmark.n_epochs     ?: 20
    def batch_size   = params.abmil_benchmark.batch_size   ?: 8
    def lr           = params.abmil_benchmark.lr           ?: 1e-4
    def head_dim     = params.abmil_benchmark.head_dim     ?: 256
    def n_heads      = params.abmil_benchmark.n_heads      ?: 8
    def target_col   = params.abmil_benchmark.target_col
    def dtype        = params.abmil_benchmark.dtype        ?: "float32"
    def split_col    = params.abmil_benchmark.split_col    ? "split_col=${params.abmil_benchmark.split_col}" : ""
    def test_size    = params.abmil_benchmark.test_size    ?: 0.2
    def val_size     = params.abmil_benchmark.val_size     ?: 0.1
    """
    abmil_benchmark \
        features_dir=features_dir \
        labels_parquet=${labels_parquet} \
        target_col=${target_col} \
        dtype=${dtype} \
        n_seeds=${n_seeds} \
        n_bootstrap=${n_bootstrap} \
        random_state=${random_state} \
        n_epochs=${n_epochs} \
        batch_size=${batch_size} \
        lr=${lr} \
        head_dim=${head_dim} \
        n_heads=${n_heads} \
        test_size=${test_size} \
        val_size=${val_size} \
        ${split_col} \
        output_summary_json=results.json
    """
}

process SUMMARIZE_ABMIL_BENCHMARK {
    publishDir "${params.outdir}/abmil_benchmark/", mode: "${params.publish_mode}"

    input:
    val(model_data)  // list of [model_type, results_json_path]

    output:
    path "abmil_benchmark_summary.csv"
    path "abmil_benchmark_summary.json"

    script:
    def pairs_str = model_data.collect { model, json -> "${model}:${json}" }.join(" ")
    """
    #!/usr/bin/env python3
    import json, pathlib, math
    import pandas as pd

    pairs_str = "${pairs_str}"

    rows = []
    combined = {}
    for token in pairs_str.split():
        model_name, json_path = token.split(":", 1)
        data = json.loads(pathlib.Path(json_path).read_text())
        combined[model_name] = data

        def _safe(v):
            return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v

        row = {"model": model_name}
        for split in ("val", "test"):
            for metric in ("auroc",):
                if metric in data.get(split, {}):
                    row[f"{split}_{metric}_mean"] = _safe(data[split][metric].get("mean"))
                    row[f"{split}_{metric}_std"]  = _safe(data[split][metric].get("std"))
                    if split == "test" and "bootstrap_ci_95" in data[split].get(metric, {}):
                        ci = data[split][metric]["bootstrap_ci_95"]
                        row[f"test_{metric}_ci95_lo"] = _safe(ci[0])
                        row[f"test_{metric}_ci95_hi"] = _safe(ci[1])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("test_auroc_mean", ascending=False)
    df.to_csv("abmil_benchmark_summary.csv", index=False)

    pathlib.Path("abmil_benchmark_summary.json").write_text(
        json.dumps(combined, indent=2)
    )
    """
}

workflow ABMIL_BENCHMARK_WORKFLOW {
    take:
        ch_h5_features  // tuple val(meta), val(model_type), path(h5)

    main:
        if (params.abmil_benchmark.labels_parquet && params.abmil_benchmark.target_col) {
            ch_labels = Channel.value(file(params.abmil_benchmark.labels_parquet))

            // Collect all per-slide H5 files for each model_type into a staging directory.
            // Each process invocation receives one directory containing all slides for that
            // model_type, so abmil_benchmark can discover slides by glob (*.h5).
            ch_features_by_model = ch_h5_features
                .map { meta, model_type, h5 -> tuple(model_type, h5) }
                .groupTuple()

            ABMIL_BENCHMARK(ch_features_by_model, ch_labels)

            ABMIL_BENCHMARK.out.results_json
                .collect(flat: false)
                | SUMMARIZE_ABMIL_BENCHMARK
        }
}
