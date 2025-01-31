
process TESSELLATE {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide)

    output:
    tuple val(meta), path("${meta.slide_id}.patch.h5"), optional: true, emit: h5
    tuple val(meta), val("tiles_h5_urlpath"), val("${publish_path}/${meta.slide_id}.patch.h5"), path("${meta.slide_id}.patch.h5"), optional: true, topic: slide_meta
    path "${meta.slide_id}_png/*.png", optional: true, emit: png
    path "${meta.slide_id}.thumbnail.png", optional: true, emit: thumbnail_png
    tuple val(meta), val("thumbnail_urlpath"), val("${publish_path}/${meta.slide_id}.thumbnail.png"), path("${meta.slide_id}.thumbnail.png"), optional: true, topic: slide_meta
    tuple val(meta), val("tile_png_urlpath"), val("${publish_path}/${meta.slide_id}_png"), path("${meta.slide_id}_png/*.png"), optional: true, topic: slide_meta

    script:
    publish_path = "tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    save_tile_param = ""
    if (params.tiling.save_tile_png)
        save_tile_param = "output_png_dir=${meta.slide_id}_png"
    save_thumbnail_param = ""
    if (params.tiling.save_slide_thumbnail)
        save_thumbnail_param = "output_thumbnail_path=${meta.slide_id}.thumbnail.png"
    """
    tessellate \
        patch_config.mpp=${params.tiling.mpp} \
        patch_config.patch_size=${params.tiling.patch_size} \
        seg_config.segment_threshold=${params.tiling.segment_threshold} \
        seg_config.median_blur_ksize=${params.tiling.median_blur_ksize} \
        seg_config.morphology_ex_kernel=${params.tiling.morphology_ex_kernel} \
        filter_config.tissue_area_threshold=${params.tiling.tissue_area_threshold} \
        filter_config.hole_area_threshold=${params.tiling.hole_area_threshold} \
        filter_config.max_num_holes=${params.tiling.max_num_holes} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        output_h5_path=${meta.slide_id}.patch.h5 \
        ${save_thumbnail_param} \
        ${save_tile_param}
    """
}

process FILTER_TILES {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_h5)

    output:
    tuple val(meta), path("${meta.slide_id}.filtered_features.h5"), emit: h5
    tuple val(meta), val("filtered_features_h5_urlpath"), val("${publish_path}/${meta.slide_id}.filtered_features.h5"), topic: slide_meta

    script:
    publish_path = "filter_tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    """
    filter_features \
        features_h5_path=${features_h5} \
        output_h5_path=${meta.slide_id}.filtered_features.h5 \
        classifier_threshold=${params.tiling.filter_threshold} \
        classifier_pkl=${params.tiling.filter_model_path}
    """
}


process STITCH_TILES {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide), path(tiles_h5)

    output:
    path "${meta.slide_id}.stitch.jpg"
    tuple val(meta), val("thumbnail_urlpath"), val("${publish_path}/${meta.slide_id}.stitch.jpg"), topic: slide_meta

    script:
    publish_path = "stitch_tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    """
    stitch_tiles slide_path=${slide} h5_path=${tiles_h5} output_jpeg_path=${meta.slide_id}.stitch.jpg
    """
}
