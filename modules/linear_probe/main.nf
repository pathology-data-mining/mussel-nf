params.annotation_class_mapping = "{ 1 : 1, 2 : 1, 3 : 1, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 11 : 0 }"

process MERGE_ANNOTATION_FEATURES {
    label "hugeTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/"

    input:
    tuple val(slide_id), path(annotation_bmp), val(model_type), path(features_h5)

    output:
    path("${slide_id}.annotation_features.parquet")

    script:
    """
    merge_annotation_features \
        features_h5_path=${features_h5} \
        annotation_bmp_path=${annotation_bmp} \
        output_parquet_path=${slide_id}.annotation_features.parquet \
        class_mapping='${params.annotation_class_mapping}' \
        slide_id=${slide_id}
    """
}

process STACK_ANNOTATION_FEATURES {
    label "bigTask"

    publishDir "${params.outdir}/annotation_features/${model_type}/"

    input:
    path annotation_features

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
        ch_annotations.join(ch.h5_features) | \
            MERGE_ANNOTATION_FEATURES | \
            collect | \
            STACK_ANNOTATION_FEATURES | \
            LINEAR_PROBE_BENCHMARK

}

