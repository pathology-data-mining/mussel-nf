params.annotation_class_mapping_yaml = "class_mapping.yaml"

process MERGE_ANNOTATION_FEATURES {
    label "hugeTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(slide_id), path(annotation_bmp), val(model_type), path(features_h5)
    path(class_mapping_yaml)

    output:
    tuple val(model_type), path("${slide_id}.annotation_features.parquet")

    script:
    """
    merge_annotation_features \
        features_h5_path=${features_h5} \
        annotation_bmp_path=${annotation_bmp} \
        output_parquet_path=${slide_id}.annotation_features.parquet \
        class_mapping_yaml_path='${class_mapping_yaml}' \
        slide_id=${slide_id}
    """
}

process STACK_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/", mode: "${params.publish_mode}"

    input:
    tuple val(model_type), path(annotation_features)

    output:
    path "annotation_features.parquet"

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
    input:
    path annotation_features

    script:
    """
    linear_probe_benchmark \
        features_annotation_parquet_path=${annotation_features}
    """

}

workflow LINEAR_PROBE {
    take:
        ch_annotations // slide_id, annotation_bmp_path
        ch_h5_features // slide_id, model_type, h5_features

    main:
        ch_ann_features = ch_annotations.join(ch_h5_features)
        MERGE_ANNOTATION_FEATURES(ch_ann_features, file(params.annotation_class_mapping_yaml)) | \
            groupTuple | \
            STACK_ANNOTATION_FEATURES | \
            LINEAR_PROBE_BENCHMARK

}

