/**
 * ABMIL precision benchmarking via paladin ABMIL training.
 *
 * Runs paladin ABMIL training on WDS feature shards at each target feature_dtype
 * (float32 / float16 / bfloat16) with precision: 32-true (no AMP), so the model
 * sees features in the loaded dtype — giving a clean side-by-side comparison of
 * how feature precision affects ABMIL performance vs the linear probe baseline.
 *
 * The training is triggered by mussel-nf's WDS_SHARD output; the same float32 shards
 * are reused for all dtype variants (paladin casts at load time via feature_dtype).
 *
 * Required params (under params.abmil_benchmark):
 *   enabled        : true / false
 *   paladin_dir    : absolute path to paladin checkout
 *   conda_env      : conda env name with paladin installed
 *   dataset_config : paladin dataset config (e.g. clf_LUAD_Primary_STK11_ONCOGENIC)
 *   train_config   : paladin train config (default: precision_benchmark)
 *   nn_config      : paladin nn config    (default: jointbb)
 *   data_config    : paladin data config  (default: joint_optimus_wds)
 *   dtypes         : list of feature dtypes to benchmark
 *   seed           : random seed index
 */

process PALADIN_ABMIL_BENCHMARK {
    label "gpuTask"
    conda params.abmil_benchmark.conda_env

    publishDir path: { "${params.outdir}/abmil_benchmark/${model_type}/${group_name}/${feature_dtype}" },
               mode: "${params.publish_mode}",
               pattern: "test_metrics.json"

    input:
    tuple val(group_name), val(model_type), path(shard_files), val(feature_dtype)

    output:
    tuple val(group_name), val(model_type), val(feature_dtype), path("test_metrics.json"), emit: metrics

    script:
    def cfg         = params.abmil_benchmark
    def train_cfg   = cfg.train_config  ?: 'precision_benchmark'
    def nn_cfg      = cfg.nn_config     ?: 'jointbb'
    def data_cfg    = cfg.data_config   ?: 'joint_optimus_wds'
    def dataset_cfg = cfg.dataset_config
    def seed        = cfg.seed          ?: 0
    def shard_list  = shard_files instanceof List
                        ? shard_files.collect { it.name }.join(' ')
                        : shard_files.name
    """
    WORK_DIR=\$(pwd)

    # Flat shard layout: symlink all shards into a single subdirectory.
    # paladin's PaladinWDSDataset detects *.tar in the root and uses flat-layout discovery.
    mkdir -p \${WORK_DIR}/shards
    for f in ${shard_list}; do
        ln -sf "\$(realpath "\$f")" \${WORK_DIR}/shards/
    done

    cd ${cfg.paladin_dir}

    # PALADIN_METRICS_OUTPUT is read by TestMetricsCallback.__init__ and takes precedence
    # over the yaml output_path, ensuring the JSON lands in the Nextflow work directory.
    export PALADIN_METRICS_OUTPUT=\${WORK_DIR}/test_metrics.json

    python src/paladin/run.py \\
        train=${train_cfg} \\
        nn=${nn_cfg} \\
        "nn/data=${data_cfg}" \\
        "nn/data/dataset=${dataset_cfg}" \\
        nn.data.wds_shard_dir=\${WORK_DIR}/shards \\
        nn.data.dataset.feature_dtype=${feature_dtype} \\
        train.seed_index=${seed} \\
        "core.tags=[precision-benchmark,${feature_dtype},${dataset_cfg},abmil]"

    # Confirm the callback wrote the file
    if [ ! -f "\${WORK_DIR}/test_metrics.json" ]; then
        echo "WARNING: test_metrics.json not written by TestMetricsCallback — writing empty stub" >&2
        echo '{}' > "\${WORK_DIR}/test_metrics.json"
    fi
    """

    stub:
    """
    echo '{"auroc/test": 0.75}' > test_metrics.json
    """
}

process SUMMARIZE_ABMIL_BENCHMARK {
    label "smallTask"

    publishDir "${params.outdir}/abmil_benchmark/", mode: "${params.publish_mode}"

    input:
    val(model_data)  // list of "group:model_type:feature_dtype:json_path" tokens

    output:
    path "abmil_summary.csv",        emit: summary_csv
    path "abmil_precision_delta.csv", emit: delta_csv
    path "abmil_report.html",        emit: report

    script:
    def tokens_str = model_data.collect { it }.join(" ")
    """
    #!/usr/bin/env python3
    import json, pathlib, math
    import pandas as pd

    tokens_str = "${tokens_str}"

    rows = []
    for token in tokens_str.split():
        group, model_type, feature_dtype, json_path = token.split(":", 3)
        data = json.loads(pathlib.Path(json_path).read_text())
        row = {
            "group":         group,
            "model_type":    model_type,
            "feature_dtype": feature_dtype,
        }
        for k, v in data.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("abmil_summary.csv", index=False)

    # Compute deltas vs float32 baseline
    delta_rows = []
    for (group, model_type), grp in df.groupby(["group", "model_type"]):
        baseline = grp[grp["feature_dtype"] == "float32"]
        if baseline.empty:
            continue
        b = baseline.iloc[0]
        for _, row in grp.iterrows():
            r = {"group": group, "model_type": model_type, "feature_dtype": row["feature_dtype"]}
            for col in ["auroc/test", "pearson/test"]:
                if col in row and col in b:
                    try:
                        r[f"delta_{col}"] = float(row[col]) - float(b[col])
                    except (TypeError, ValueError):
                        pass
            delta_rows.append(r)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv("abmil_precision_delta.csv", index=False)

    # Simple HTML report
    with open("abmil_report.html", "w") as f:
        f.write("<html><head><title>ABMIL Precision Benchmark</title>")
        f.write("<style>body{font-family:sans-serif;margin:2em}")
        f.write("table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 12px}")
        f.write("th{background:#f0f0f0}</style></head><body>")
        f.write("<h1>ABMIL Precision Benchmark</h1>")
        f.write("<h2>Summary</h2>")
        f.write(df.to_html(index=False))
        f.write("<h2>Delta vs float32</h2>")
        f.write(delta_df.to_html(index=False))
        f.write("</body></html>")

    print("ABMIL benchmark summary written.")
    """

    stub:
    """
    echo "group,model_type,feature_dtype,auroc/test" > abmil_summary.csv
    echo "group,model_type,feature_dtype,delta_auroc/test" > abmil_precision_delta.csv
    echo "<html><body>stub</body></html>" > abmil_report.html
    """
}
