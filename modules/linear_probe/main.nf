
process MERGE_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_h5), path(annotation_bmp)
    path(class_mapping_yaml)

    output:
    tuple val(model_type), path("${meta.slide_id}.annotation_features.parquet"), emit: parquet
    tuple val(meta), val("${model_type}_annotation_features_path"), val("annotation_features/${model_type}/${meta.slide_id}.annotation_features.parquet"), path("${meta.slide_id}.annotation_features.parquet"), topic: slide_meta

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
    import geopandas as gpd
    import pandas as pd
    files = "${annotation_features}".split()
    dfs = [gpd.read_parquet(f) for f in files]
    pd.concat(dfs).to_parquet("annotation_features.parquet")
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

    script:
    penalty_str = params.linear_probe.penalty ? "penalty=${params.linear_probe.penalty}" : ""
    """
    linear_probe_benchmark \
        features_annotation_parquet_path=${annotation_features} \
        test_size=${params.linear_probe.test_size} \
        val_size=${params.linear_probe.val_size} \
        random_state=${params.linear_probe.random_state} \
        annotation_percent_filter_threshold=${params.linear_probe.annotation_percent_filter_threshold} \
        C=${params.linear_probe.C} \
        max_iter=${params.linear_probe.max_iter} \
        ${penalty_str}
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
