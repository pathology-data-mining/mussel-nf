
process MERGE_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

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
    label "cpuTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(model_type), path(annotation_features)

    output:
    tuple val(model_type), path("annotation_features.parquet")

    script:
    """
    #!/usr/bin/env python3
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
    path "results.json"

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
        if (!params.linear_probe.annotation_class_mapping_yaml) {
            log.warn "params.linear_probe.annotation_class_mapping_yaml is not set — linear probe benchmarking will be skipped"
        } else {
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
        }

}
