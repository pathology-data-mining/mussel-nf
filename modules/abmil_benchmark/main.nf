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
    def split_col = params.abmil_benchmark.split_col ? "split_col=${params.abmil_benchmark.split_col}" : ""
    """
    abmil_benchmark \
        features_dir=features_dir \
        labels_parquet=${labels_parquet} \
        target_col=${params.abmil_benchmark.target_col} \
        dtype=${params.abmil_benchmark.dtype} \
        n_seeds=${params.abmil_benchmark.n_seeds} \
        n_bootstrap=${params.abmil_benchmark.n_bootstrap} \
        random_state=${params.abmil_benchmark.random_state} \
        n_epochs=${params.abmil_benchmark.n_epochs} \
        batch_size=${params.abmil_benchmark.batch_size} \
        lr=${params.abmil_benchmark.lr} \
        head_dim=${params.abmil_benchmark.head_dim} \
        n_heads=${params.abmil_benchmark.n_heads} \
        test_size=${params.abmil_benchmark.test_size} \
        val_size=${params.abmil_benchmark.val_size} \
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
    summarize_abmil_benchmark.py ${pairs_str}
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
