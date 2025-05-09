
process MERGE_ANNOTATION_FEATURES {
    label "hugeTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_h5), path(annotation_bmp) 
    path(class_mapping_yaml)

    output:
    tuple val(model_type), path("${meta.slide_id}.annotation_features.parquet")

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
    import geopandas as gpd
    import pandas as pd
    files = "${annotation_features}".split()
    dfs = [gpd.read_parquet(file) for file in files]
    df = pd.concat(dfs)
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

    script:
    """
    linear_probe_benchmark \
        features_annotation_parquet_path=${annotation_features}
    """

}

workflow LINEAR_PROBE {
    take:
        ch_annotations // meta, annotation_bmp_path
        ch_h5_features // meta, model_type, h5_features

    main:
        if (params.linear_probe.annotation_class_mapping_yaml) {
            ch_features_ann = ch_h5_features.map{tuple(it[0].slide_id, *it)}.combine(ch_annotations.map{tuple(it[0].slide_id, *it.tail())}, by: 0).map{it.tail()}
            MERGE_ANNOTATION_FEATURES(ch_features_ann, file(params.linear_probe.annotation_class_mapping_yaml)) | \
                groupTuple | \
                STACK_ANNOTATION_FEATURES | \
                LINEAR_PROBE_BENCHMARK
        }

}

