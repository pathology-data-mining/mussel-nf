
process MERGE_ANNOTATION_FEATURES {
    label "hugeTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_h5), path(annotation_bmp) 
    path(class_mapping_yaml)

    output:
    tuple val(model_type), path("${meta.slide_id}.annotation_features.parquet"), optional: true

    script:
    """
    merge_annotation_features \
        features_h5_path=${features_h5} \
        annotation_bmp_path=${annotation_bmp} \
        output_parquet_path=${meta.slide_id}.annotation_features.parquet \
        class_mapping_yaml_path='${class_mapping_yaml}' \
        slide_id=${meta.slide_id}
    """
}

process STACK_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(model_type), path(annotation_features)

    output:
    tuple val(model_type), path("annotation_features.parquet")

    script:
    """
    #!/usr/bin/env python
    import pandas as pd
    files = "${annotation_features}".split()
    dfs = [pd.read_parquet(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet("annotation_features.parquet")
    """

}

process LINEAR_PROBE_BENCHMARK {
    label "bigTask"

    publishDir "${params.outdir}/linear_probe_benchmark/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(model_type), path(annotation_features)

    output:
    path "classification_report.csv"
    path "confusion_matrix.png"
    path "classification_report_test.csv"
    path "confusion_matrix_test.png"
    path "roc_curve.png"
    path "pr_curve.png"
    path "grid_search_heatmap.png"
    path "feature_importance.png"
    path "calibration_curve.png"
    path "cv_results.csv"
    tuple val(model_type), path("results.json"), emit: results_json

    script:
    def cv           = params.linear_probe.cv ?: 5
    def C_values     = (params.linear_probe.C_values ?: [0.001, 0.01, 0.1, 1.0, 10.0]).join(",")
    def penalties    = (params.linear_probe.penalties ?: ["l2"]).join(",")
    def n_seeds      = params.linear_probe.n_seeds ?: 5
    def n_bootstrap  = params.linear_probe.n_bootstrap ?: 1000
    def random_state = params.linear_probe.random_state ?: 42
    def pos_label    = params.linear_probe.positive_annotation_label ?: 1
    def multiclass   = params.linear_probe.multiclass ? "true" : "false"
    """
    linear_probe_benchmark \
        features_annotation_parquet_path=${annotation_features} \
        cv=${cv} \
        'C_values=[${C_values}]' \
        'penalties=[${penalties}]' \
        n_seeds=${n_seeds} \
        n_bootstrap=${n_bootstrap} \
        random_state=${random_state} \
        positive_annotation_label=${pos_label} \
        multiclass=${multiclass}
    """

}

workflow LINEAR_PROBE {
    take:
        ch_annotations // meta, annotation_bmp_path
        ch_h5_features // meta, model_type, h5_features

    main:
        if (params.linear_probe.annotation_class_mapping_yaml) {
            ch_features_ann = ch_h5_features.map{tuple(it[0].slide_id, *it)}.combine(ch_annotations.map{tuple(it[0].slide_id, *it.tail())}, by: 0).map{it.tail()}
            ch_benchmark = MERGE_ANNOTATION_FEATURES(ch_features_ann, file(params.linear_probe.annotation_class_mapping_yaml)) | \
                groupTuple | \
                STACK_ANNOTATION_FEATURES | \
                LINEAR_PROBE_BENCHMARK

            ch_benchmark.results_json
                | map { model_type, json -> [model_type, json] }
                | collect(flat: false)
                | SUMMARIZE_LINEAR_PROBE
        }

}

process SUMMARIZE_LINEAR_PROBE {
    label "smallTask"

    publishDir "${params.outdir}/linear_probe_benchmark/", mode: "${params.publish_mode}"

    input:
    val(model_json_pairs)  // list of [model_type, results.json path] tuples

    output:
    path "summary.csv"
    path "summary.png"

    script:
    // Build a space-separated list of  "model_type:path"  pairs to pass to the script
    def pairs_str = model_json_pairs.collect { model, json -> "${model}:${json}" }.join(" ")
    """
    #!/usr/bin/env python
    import json, pathlib, math
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pairs_str = """" + pairs_str + """"

    rows = []
    for token in pairs_str.split():
        model_name, json_path = token.split(":", 1)
        data = json.loads(pathlib.Path(json_path).read_text())

        def _safe(v):
            return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v

        row = {"model": model_name}
        for split in ("val", "test"):
            for metric in ("tile_auc_roc", "tile_f1", "tile_average_precision"):
                if metric in data.get(split, {}):
                    row[f"{split}_{metric}_mean"] = _safe(data[split][metric].get("mean"))
                    row[f"{split}_{metric}_std"]  = _safe(data[split][metric].get("std"))
                    if split == "test" and "bootstrap_ci_95" in data[split].get(metric, {}):
                        ci = data[split][metric]["bootstrap_ci_95"]
                        row[f"test_{metric}_ci95_lo"] = _safe(ci[0])
                        row[f"test_{metric}_ci95_hi"] = _safe(ci[1])
        row["best_cv_auc"]  = _safe(data.get("best_cv_auc"))
        row["best_C"]       = data.get("best_params", {}).get("C")
        row["best_penalty"] = data.get("best_params", {}).get("penalty")
        rows.append(row)

    df = pd.DataFrame(rows)
    primary = next(
        (c for c in ["test_tile_auc_roc_mean", "test_tile_average_precision_mean"] if c in df.columns),
        df.columns[1]
    )
    df = df.sort_values(primary, ascending=False)
    df.to_csv("summary.csv", index=False)

    # --- Bar chart with 95% CI error bars ---
    models = df["model"].tolist()
    means  = df[primary].tolist()
    ci_lo  = primary.replace("_mean", "_ci95_lo")
    ci_hi  = primary.replace("_mean", "_ci95_hi")
    yerr = None
    if ci_lo in df.columns and ci_hi in df.columns:
        lo_vals = df[ci_lo].tolist()
        hi_vals = df[ci_hi].tolist()
        yerr = [
            [max(0, (m or 0) - (l or 0)) for m, l in zip(means, lo_vals)],
            [max(0, (h or 0) - (m or 0)) for m, h in zip(means, hi_vals)],
        ]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 5))
    colors = plt.cm.tab10.colors[:len(models)]
    bars = ax.bar(x, [m or 0 for m in means], yerr=yerr, capsize=6,
                  color=colors, alpha=0.85, ecolor="black", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel(primary.replace("_", " ").replace(" mean", "").title(), fontsize=11)
    ax.set_title("Linear Probe Benchmark — Test Set Comparison", fontsize=13)
    ax.set_ylim(0, 1.08)
    for bar, val in zip(bars, means):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Summary written: summary.csv, summary.png")
    """
}

