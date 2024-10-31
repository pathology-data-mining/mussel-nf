params.filter_model_path = "/gpfs/mskmind_ess/limr/repos/Mussel/model-1727990346535.pkl"
params.filter_threshold = 0.75

params.patch_size = 224
params.mpp = 0.5
params.segment_threshold = 15
params.median_blur_ksize = 11
params.morphology_ex_kernel = 2
params.tissue_area_threshold = 1
params.hole_area_threshold = 1
params.max_num_holes = 2

params.save_slide_thumbnail = false
params.save_tile_png = false

process TESSELLATE {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/tiles/${slide_id[0..3]}/", mode: "${params.publish_mode}"

    input:
    tuple val(slide_id), path(slide)

    output:
    tuple val(slide_id), path("${slide_id}.patch.h5"), optional: true, emit: h5
    tuple val(slide_id), val("tiles_h5_urlpath"), val("${task.publishDir[0].path}/${slide_id}.patch.h5"), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("patch_size"), val(params.patch_size), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("mpp"), val(params.mpp), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("segment_threshold"), val(params.segment_threshold), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("median_blur_ksize"), val(params.median_blur_ksize), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("morphology_ex_kernel"), val(params.morphology_ex_kernel), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("tissue_area_threshold"), val(params.tissue_area_threshold), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("hole_area_threshold"), val(params.hole_area_threshold), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("max_num_holes"), val(params.max_num_holes), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    path "${slide_id}_png/*.png", optional: true, emit: png
    path "${slide_id}.thumbnail.png", optional: true, emit: thumbnail_png
    tuple val(slide_id), val("thumbnail_urlpath"), val("${task.publishDir[0].path}/${slide_id}.thumbnail.png"), path("${slide_id}.thumbnail.png"), optional: true, topic: meta_out
    tuple val(slide_id), val("tile_png_urlpath"), val("${task.publishDir[0].path}/${slide_id}_png"), path("${slide_id}_png/*.png"), optional: true, topic: meta_out

    script:
    save_tile_param = ""
    if (params.save_tile_png)
        save_tile_param = "output_png_dir=${slide_id}_png"
    save_thumbnail_param = ""
    if (params.save_slide_thumbnail)
        save_thumbnail_param = "output_thumbnail_path=${slide_id}.thumbnail.png"
    """
    tessellate \
        patch_config.mpp=${params.mpp} \
        patch_config.patch_size=${params.patch_size} \
        seg_config.segment_threshold=${params.segment_threshold} \
        seg_config.median_blur_ksize=${params.median_blur_ksize} \
        seg_config.morphology_ex_kernel=${params.morphology_ex_kernel} \
        filter_config.tissue_area_threshold=${params.tissue_area_threshold} \
        filter_config.hole_area_threshold=${params.hole_area_threshold} \
        filter_config.max_num_holes=${params.max_num_holes} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        output_h5_path=${slide_id}.patch.h5 \
        ${save_thumbnail_param} \
        ${save_tile_param}
    """
}

process FILTER_TILES {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/filter_tiles/${slide_id[0..3]}/", mode: "${params.publish_mode}"

    input:
    tuple val(slide_id), val(model_type), path(features_h5)

    output:
    tuple val(slide_id), path("${slide_id}.filtered_features.h5"), emit: h5
    tuple val(slide_id), val("filtered_features_urlpath"), val("${task.publishDir[0].path}/${slide_id}.filtered_features.h5"), topic: meta_out
    tuple val(slide_id), val("filter_threshold"), val(params.filter_threshold), topic: meta_out

    script:
    """
    filter_features \
        features_h5_path=${features_h5} \
        output_h5_path=${slide_id}.filtered_features.h5 \
        classifier_threshold=${params.filter_threshold} \
        classifier_pkl=${params.filter_model_path}
    """
}


process STITCH_TILES {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/stitch_tiles/${slide_id[0..3]}/", mode: "${params.publish_mode}"

    input:
    tuple val(slide_id), path(slide), path(tiles_h5)

    output:
    path "${slide_id}.stitch.jpg"
    tuple val(slide_id), val("thumbnail_urlpath"), val("${task.publishDir[0].path}/${slide_id}.stitch.jpg"), topic: meta_out


    script:
    """
    stitch_tiles slide_path=${slide} h5_path=${tiles_h5} output_jpeg_path=${slide_id}.stitch.jpg
    """
}
