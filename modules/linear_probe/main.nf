
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
    def multiclass = params.linear_probe.multiclass ? "true" : "false"
    """
    linear_probe_benchmark \
        features_annotation_parquet_path=${annotation_features} \
        cv=${params.linear_probe.cv} \
        'C_values=[${params.linear_probe.C_values.join(",")}]' \
        'penalties=[${params.linear_probe.penalties.join(",")}]' \
        n_seeds=${params.linear_probe.n_seeds} \
        n_bootstrap=${params.linear_probe.n_bootstrap} \
        random_state=${params.linear_probe.random_state} \
        positive_annotation_label=${params.linear_probe.positive_annotation_label} \
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
    path "report.html"

    script:
    def triples_str = model_data.collect { model, json, csv -> "${model}:${json}:${csv}" }.join(" ")
    """
    summarize_linear_probe.py ${triples_str}
    """
}
