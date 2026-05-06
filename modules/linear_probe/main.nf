
process MERGE_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir path: { "${params.outdir}/annotation_features/${model_type}/" }, mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_h5), path(annotation_bmp)
    path(class_mapping_yaml)

    output:
    tuple val(model_type), path("${meta.slide_id}.annotation_features.parquet"), optional: true, emit: parquet
    tuple val(meta), val("${model_type}_annotation_features_path"), val("annotation_features/${model_type}/${meta.slide_id}.annotation_features.parquet"), path("${meta.slide_id}.annotation_features.parquet"), optional: true, topic: slide_meta

    script:
    class_mapping_str = class_mapping_yaml.name != 'NO_FILE' ? "class_mapping_yaml_path='${class_mapping_yaml}'" : ""
    """
    merge_annotation_features \
        features_h5_path=${features_h5} \
        annotation_bmp_path=${annotation_bmp} \
        output_parquet_path=${meta.slide_id}.annotation_features.parquet \
        ${class_mapping_str} \
        slide_id=${meta.slide_id}
    """
}

process STACK_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir path: { "${params.outdir}/annotation_features/${model_type}/" }, mode: "${params.publish_mode}"

    input:
    tuple val(model_type), path(annotation_features, stageAs: '?/*')

    output:
    tuple val(model_type), path("annotation_features.parquet")

    script:
    """
    #!/usr/bin/env python3
    import glob, pandas as pd
    files = sorted(glob.glob("*/*.annotation_features.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate in case the same slide appears more than once (e.g. from --resume)
    df = df.drop_duplicates()
    df.to_parquet("annotation_features.parquet")
    """

}

process LINEAR_PROBE_BENCHMARK {
    label "bigTask"

    publishDir path: { "${params.outdir}/linear_probe_benchmark/${model_type}/" }, mode: "${params.publish_mode}"

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
    tuple val(model_type), path("results.json"),                    emit: results_json
    tuple val(model_type), path("classification_report_test.csv"), emit: clf_report_test

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
        ch_annotations // tuple val(meta), file(annotation_bmp_path)
        ch_h5_features // tuple val(meta), val(model_type), path(h5_features)

    main:
        if (params.linear_probe.annotations_csv) {
            // Broadcast each annotation BMP to all model types for that slide using combine(by: slide_id).
            // combine (not join) is used because ch_h5_features has one row per (slide, model_type).
            ch_features_ann = ch_h5_features
                .map { meta, model_type, h5 -> tuple(meta.slide_id, meta, model_type, h5) }
                .combine(
                    ch_annotations.map { meta, bmp -> tuple(meta.slide_id, bmp) },
                    by: 0
                )
                .map { _id, meta, model_type, h5, bmp -> tuple(meta, model_type, h5, bmp) }

            MERGE_ANNOTATION_FEATURES(
                ch_features_ann,
                params.linear_probe.annotation_class_mapping_yaml
                    ? file(params.linear_probe.annotation_class_mapping_yaml)
                    : file("NO_FILE", checkIfExists: false)
            )

            MERGE_ANNOTATION_FEATURES.out.parquet \
                | groupTuple \
                | STACK_ANNOTATION_FEATURES \
                | LINEAR_PROBE_BENCHMARK

            LINEAR_PROBE_BENCHMARK.out.results_json
                .join(LINEAR_PROBE_BENCHMARK.out.clf_report_test, by: 0)
                .collect(flat: false)
                | SUMMARIZE_LINEAR_PROBE
        }

}

process SUMMARIZE_LINEAR_PROBE {
    label "smallTask"

    publishDir "${params.outdir}/linear_probe_benchmark/", mode: "${params.publish_mode}"

    input:
    val(model_data)  // list of [model_type, results.json path, classification_report_test.csv path]

    output:
    path "summary.csv"
    path "summary.png"
    path "per_class_f1.csv"
    path "per_class_heatmap.png"
    path "precision_delta.csv"

    script:
    def triples_str = model_data.collect { model, json, csv -> "${model}:${json}:${csv}" }.join(" ")
    """
    #!/usr/bin/env python3
    import json, pathlib, math, re
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    triples_str = "${triples_str}"

    # ── Parse inputs ──────────────────────────────────────────────────────────
    summary_rows = []
    clf_dfs = {}
    for token in triples_str.split():
        model_name, json_path, csv_path = token.split(":", 2)
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
        row["best_C"]       = data.get("best_params", {}).get("clf__C") or data.get("best_params", {}).get("C")
        row["best_penalty"] = data.get("best_params", {}).get("clf__penalty") or data.get("best_params", {}).get("penalty")
        summary_rows.append(row)

        cr = pd.read_csv(csv_path, index_col=0)
        clf_dfs[model_name] = cr

    # ── summary.csv + summary.png ─────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows)
    primary = next(
        (c for c in ["test_tile_auc_roc_mean", "test_tile_average_precision_mean"] if c in df_sum.columns),
        df_sum.columns[1]
    )
    df_sum = df_sum.sort_values(primary, ascending=False)
    df_sum.to_csv("summary.csv", index=False)

    models   = df_sum["model"].tolist()
    means    = df_sum[primary].tolist()
    ci_lo_c  = primary.replace("_mean", "_ci95_lo")
    ci_hi_c  = primary.replace("_mean", "_ci95_hi")
    yerr = None
    if ci_lo_c in df_sum.columns and ci_hi_c in df_sum.columns:
        lo_vals = df_sum[ci_lo_c].tolist()
        hi_vals = df_sum[ci_hi_c].tolist()
        yerr = [
            [max(0, (m or 0) - (l or 0)) for m, l in zip(means, lo_vals)],
            [max(0, (h or 0) - (m or 0)) for m, h in zip(means, hi_vals)],
        ]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.8), 5))
    colors = plt.cm.tab10.colors[:len(models)]
    bars = ax.bar(x, [m or 0 for m in means], yerr=yerr, capsize=6,
                  color=colors, alpha=0.85, ecolor="black", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(primary.replace("_", " ").replace(" mean", "").title(), fontsize=11)
    ax.set_title("Linear Probe Benchmark — Test Set Comparison", fontsize=13)
    ax.set_ylim(0, 1.08)
    for bar, val in zip(bars, means):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig("summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── per_class_f1.csv + per_class_heatmap.png ──────────────────────────────
    meta_cols = {"accuracy", "macro avg", "weighted avg"}
    class_cols = None
    per_class_rows = []
    for model_name in models:
        if model_name not in clf_dfs:
            continue
        cr = clf_dfs[model_name]
        if class_cols is None:
            class_cols = [c for c in cr.columns if c not in meta_cols]
        row = {"model": model_name}
        if "f1-score" in cr.index:
            for cls in class_cols:
                if cls in cr.columns:
                    row[f"class_{cls}_f1"] = cr.loc["f1-score", cls]
        if "accuracy" in cr.index and "accuracy" in cr.columns:
            row["accuracy"] = cr.loc["accuracy", "accuracy"]
        if "auc_roc" in cr.index and "weighted avg" in cr.columns:
            row["weighted_auc"] = cr.loc["auc_roc", "weighted avg"]
        per_class_rows.append(row)

    df_pc = pd.DataFrame(per_class_rows)
    df_pc.to_csv("per_class_f1.csv", index=False)

    # Heatmap: rows = models, cols = classes
    f1_cols = [c for c in df_pc.columns if c.startswith("class_")]
    class_labels = [c.replace("class_", "").replace("_f1", "") for c in f1_cols]
    heatmap_data = df_pc[f1_cols].values.astype(float)

    n_models, n_classes = heatmap_data.shape
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.9), max(4, n_models * 0.7)))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(df_pc["model"].tolist(), fontsize=9)
    ax.set_xlabel("Annotation Class", fontsize=10)
    ax.set_title("Per-Class F1 Score by Model and Precision", fontsize=12)
    for i in range(n_models):
        for j in range(n_classes):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="F1 Score")
    fig.tight_layout()
    fig.savefig("per_class_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── precision_delta.csv ───────────────────────────────────────────────────
    # For each base model, compute delta vs float16 / bfloat16 variants
    precision_suffixes = ["_float16", "_bfloat16"]
    delta_rows = []
    processed_bases = set()
    for model_name in models:
        base = model_name
        for sfx in precision_suffixes:
            base = base.replace(sfx, "")
        if base in processed_bases or base not in {r["model"] for r in summary_rows}:
            continue
        processed_bases.add(base)
        base_row = next(r for r in summary_rows if r["model"] == base)
        for sfx in precision_suffixes:
            variant = base + sfx
            variant_row = next((r for r in summary_rows if r["model"] == variant), None)
            if variant_row is None:
                continue
            delta = {"base_model": base, "precision": sfx.lstrip("_")}
            for col in [c for c in base_row if c not in ("model", "best_C", "best_penalty")]:
                bv = base_row.get(col)
                vv = variant_row.get(col)
                if bv is not None and vv is not None:
                    try:
                        delta[f"delta_{col}"] = round(vv - bv, 6)
                    except TypeError:
                        pass
            # Per-class F1 deltas
            if base in clf_dfs and variant in clf_dfs:
                cr_base    = clf_dfs[base]
                cr_variant = clf_dfs[variant]
                if "f1-score" in cr_base.index and "f1-score" in cr_variant.index:
                    for cls in [c for c in cr_base.columns if c not in meta_cols]:
                        if cls in cr_variant.columns:
                            delta[f"delta_class_{cls}_f1"] = round(
                                float(cr_variant.loc["f1-score", cls]) - float(cr_base.loc["f1-score", cls]), 6
                            )
            delta_rows.append(delta)

    pd.DataFrame(delta_rows).to_csv("precision_delta.csv", index=False)
    print("Summary written: summary.csv, summary.png, per_class_f1.csv, per_class_heatmap.png, precision_delta.csv")
    """
}
