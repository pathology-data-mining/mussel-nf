
process TESSELLATE {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide)

    output:
    tuple val(meta), path("${meta.slide_id}.patch.h5"), optional: true, emit: h5
    tuple val(meta), val("tiles_h5_path"), val("${publish_path}/${meta.slide_id}.patch.h5"), path("${meta.slide_id}.patch.h5"), optional: true, topic: slide_meta
    path "${meta.slide_id}_png/*.png", optional: true, emit: png
    path "${meta.slide_id}.*.png", optional: true, emit: thumbnail_png
    tuple val(meta), val("thumbnail_path"), val("${publish_path}/${meta.slide_id}.thumbnail.png"), path("${meta.slide_id}.thumbnail.png"), optional: true, topic: slide_meta
    tuple val(meta), val("grid_mask_path"), val("${publish_path}/${meta.slide_id}.grid_mask.png"), path("${meta.slide_id}.grid_mask.png"), optional: true, topic: slide_meta
    tuple val(meta), val("mask_path"), val("${publish_path}/${meta.slide_id}.mask.png"), path("${meta.slide_id}.mask.png"), optional: true, topic: slide_meta
    tuple val(meta), val("tile_png_path"), val("${publish_path}/${meta.slide_id}_png"), path("${meta.slide_id}_png/*.png"), optional: true, topic: slide_meta

    script:
    publish_path = "tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    save_tile_param = ""
    if (params.tiling.save_tile_png)
        save_tile_param = "output_png_dir=${meta.slide_id}_png"
    stitch_tile_param = ""
    if (params.tiling.stitch_tiles)
        stitch_tile_param = "output_grid_mask_path=${meta.slide_id}.grid_mask.png"
        stitch_tile_param += " output_mask_path=${meta.slide_id}.mask.png"
    save_thumbnail_param = ""
    if (params.tiling.save_slide_thumbnail)
        save_thumbnail_param = "output_thumbnail_path=${meta.slide_id}.thumbnail.png"

    // Use seg_config group if specified, otherwise use individual parameters
    seg_config_str = params.tiling.seg_config_group ? "seg_config=${params.tiling.seg_config_group}" : ""

    """
    tessellate \
        ${seg_config_str} \
        seg_config.mpp=${params.tiling.mpp} \
        seg_config.patch_size=${params.tiling.patch_size} \
        seg_config.segment_threshold=${params.tiling.segment_threshold} \
        seg_config.median_blur_ksize=${params.tiling.median_blur_ksize} \
        seg_config.morphology_ex_kernel=${params.tiling.morphology_ex_kernel} \
        seg_config.tissue_area_threshold=${params.tiling.tissue_area_threshold} \
        seg_config.hole_area_threshold=${params.tiling.hole_area_threshold} \
        seg_config.max_num_holes=${params.tiling.max_num_holes} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        output_h5_path=${meta.slide_id}.patch.h5 \
        ${save_thumbnail_param} \
        ${save_tile_param} \
        ${stitch_tile_param}
    """
}

process FILTER_TILES {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/${tiles_publish_path}", mode: "${params.publish_mode}", pattern: "*.h5"
    publishDir path: "${params.outdir}/${pt_publish_path}", mode: "${params.publish_mode}", pattern: "*.pt"

    input:
    tuple val(meta), val(model_type), path(features_h5)

    output:
    tuple val(meta), path("${meta.slide_id}.patch.h5"), emit: h5
    tuple val(meta), path("${meta.slide_id}.features.pt"), emit: pt
    tuple val(meta), val("filtered_tiles_h5_path"), val("${tiles_publish_path}/${meta.slide_id}.patch.h5"), topic: slide_meta
    tuple val(meta), val("${model_type}_features_tensor_path"), val("${pt_publish_path}/${meta.slide_id}.features.pt"), topic: slide_meta

    script:
    tiles_publish_path = "filter_tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    pt_publish_path = "features/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    """
    filter_features \
        features_h5_path=${features_h5} \
        output_h5_path=${meta.slide_id}.patch.h5 \
        output_pt_path=${meta.slide_id}.features.pt \
        classifier_threshold=${params.tiling.filter_threshold} \
        classifier_pkl=${params.tiling.filter_model_path}
    """
}

