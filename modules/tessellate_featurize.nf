
process TESSELLATE_FEATURIZE {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide)
    each model_config

    output:
    tuple val(meta), val(model_type), path("${meta.slide_id}.features.pt"), emit: pt
    tuple val(meta), val(model_type), path("${meta.slide_id}.features.h5"), optional: true, emit: h5
    tuple val(meta), val(model_type), path("${meta.slide_id}.patch.h5"), emit: patch_h5
    tuple val(meta), val("${model_type_name}_features_tensor_path"), val("${publish_path}/${meta.slide_id}.features.pt"), topic: slide_meta
    tuple val(meta), val("${model_type_name}_features_h5_path"), val("${publish_path}/${meta.slide_id}.features.h5"), optional: true, topic: slide_meta
    tuple val(meta), val("${model_type_name}_tiles_h5_path"), val("${publish_path}/${meta.slide_id}.patch.h5"), topic: slide_meta
    path "${meta.slide_id}_png/*.png", optional: true, emit: png
    path "${meta.slide_id}.*.png", optional: true, emit: thumbnail_png
    tuple val(meta), val("thumbnail_path"), val("${publish_path}/${meta.slide_id}.thumbnail.png"), path("${meta.slide_id}.thumbnail.png"), optional: true, topic: slide_meta
    tuple val(meta), val("grid_mask_path"), val("${publish_path}/${meta.slide_id}.grid_mask.png"), path("${meta.slide_id}.grid_mask.png"), optional: true, topic: slide_meta
    tuple val(meta), val("mask_path"), val("${publish_path}/${meta.slide_id}.mask.png"), path("${meta.slide_id}.mask.png"), optional: true, topic: slide_meta

    script:
    model_type = model_config[0]
    model_path = model_config[1]
    slide_model_type = model_config.size() > 2 ? model_config[2] : null
    slide_model_path = model_config.size() > 3 ? model_config[3] : null
    prefilter_model_type = model_config.size() > 4 ? model_config[4] : null
    prefilter_model_path = model_config.size() > 5 ? model_config[5] : null

    model_type_name = slide_model_type ?: model_type
    publish_path = "features/${model_type_name}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    
    mpath_str = ""
    if (model_path)
        mpath_str = "model_path=${model_path}"

    slide_model_str = ""
    aggregation_str = ""
    if (slide_model_type) {
        slide_model_str = "slide_model_type=${slide_model_type.toUpperCase()}"
        if (slide_model_path)
            slide_model_str += " slide_model_path=${slide_model_path}"
        aggregation_str = "aggregation_method=model"
    }

    // Filter model configuration (for extracting features before filtering)
    prefilter_model_str = ""
    prefilter_model_path_str = ""
    classifier_pkl_str = ""
    if (params.tiling.filter_tiles) {
        if (prefilter_model_type) {
            prefilter_model_str = "prefilter_model_type=${prefilter_model_type.toUpperCase()}"
            if (prefilter_model_path)
                prefilter_model_path_str = "prefilter_model_path=${prefilter_model_path}"
        }
        if (params.tiling.filter_model_path)
            classifier_pkl_str = "classifier_pkl=${params.tiling.filter_model_path}"
    }

    save_tile_param = ""
    if (params.tiling.save_tile_png)
        save_tile_param = "output_png_dir=${meta.slide_id}_png"

    stitch_tile_param = ""
    if (params.tiling.stitch_tiles) {
        stitch_tile_param = "output_grid_mask_path=${meta.slide_id}.grid_mask.png"
        stitch_tile_param += " output_mask_path=${meta.slide_id}.mask.png"
    }

    save_thumbnail_param = ""
    if (params.tiling.save_slide_thumbnail)
        save_thumbnail_param = "output_thumbnail_path=${meta.slide_id}.thumbnail.png"

    save_h5_param = "save_features_to_h5=true"  // Always save features in one-step workflow

    // Use seg_config group if specified, otherwise use individual parameters
    seg_config_str = params.tiling.seg_config_group ? "seg_config=${params.tiling.seg_config_group}" : ""

    // Build individual parameter overrides (only if not using group, or to override group defaults)
    seg_params = []
    if (params.tiling.mpp != null) seg_params << "seg_config.mpp=${params.tiling.mpp}"
    if (params.tiling.patch_size != null) seg_params << "seg_config.patch_size=${params.tiling.patch_size}"
    if (params.tiling.segment_threshold != null) seg_params << "seg_config.segment_threshold=${params.tiling.segment_threshold}"
    if (params.tiling.median_blur_ksize != null) seg_params << "seg_config.median_blur_ksize=${params.tiling.median_blur_ksize}"
    if (params.tiling.morphology_ex_kernel != null) seg_params << "seg_config.morphology_ex_kernel=${params.tiling.morphology_ex_kernel}"
    if (params.tiling.tissue_area_threshold != null) seg_params << "seg_config.tissue_area_threshold=${params.tiling.tissue_area_threshold}"
    if (params.tiling.hole_area_threshold != null) seg_params << "seg_config.hole_area_threshold=${params.tiling.hole_area_threshold}"
    if (params.tiling.max_num_holes != null) seg_params << "seg_config.max_num_holes=${params.tiling.max_num_holes}"
    seg_params_str = seg_params.join(' \\\n        ')

    """
    tessellate_extract_features \
        ${seg_config_str} \
        ${seg_params_str} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        slide_id=${meta.slide_id} \
        output_h5_path=${meta.slide_id}.features.h5 \
        output_pt_path=${meta.slide_id}.features.pt \
        intermediate_h5_path=${meta.slide_id}.patch.h5 \
        model_type=${model_type.toUpperCase()} ${mpath_str} \
        use_gpu=${params.featurize.use_gpu ? "true" : "false"} \
        batch_size=${params.featurize.batch_size} \
        ${slide_model_str} \
        ${aggregation_str} \
        ${prefilter_model_str} \
        ${prefilter_model_path_str} \
        ${classifier_pkl_str} \
        classifier_threshold=${params.tiling.filter_threshold} \
        ${save_thumbnail_param} \
        ${save_tile_param} \
        ${stitch_tile_param} \
        ${save_h5_param}
    """
}

process TESSELLATE_FEATURIZE_BATCH {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/${publish_path_base}", mode: "${params.publish_mode}", pattern: "pt/*.pt", saveAs: { fn -> fn.replaceFirst("pt/", "") }
    publishDir path: "${params.outdir}/${publish_path_base}", mode: "${params.publish_mode}", pattern: "h5/*.h5", saveAs: { fn -> fn.replaceFirst("h5/", "") }
    publishDir path: "${params.outdir}/tiles", mode: "${params.publish_mode}", pattern: "tile_h5/*.h5", saveAs: { fn -> fn.replaceFirst("tile_h5/", "") }

    input:
    tuple val(slide_batch), path(slides) // batch with slide files
    each model_config // tuple of [model_type, model_path, slide_model_type, slide_model_path, prefilter_model_type, prefilter_model_path]

    output:
    tuple val(batch_metadata), val(model_type), path("pt/*.features.pt"), emit: pt
    tuple val(batch_metadata), val(model_type), path("h5/*.features.h5"), optional: true, emit: h5
    tuple val(batch_metadata), path("tile_h5/*.patch.h5"), emit: patch_h5

    script:
    model_type = model_config[0]
    model_path = model_config[1]
    slide_model_type = model_config.size() > 2 ? model_config[2] : null
    slide_model_path = model_config.size() > 3 ? model_config[3] : null
    prefilter_model_type = model_config.size() > 4 ? model_config[4] : null
    prefilter_model_path = model_config.size() > 5 ? model_config[5] : null

    // Extract metadata for all slides in batch
    batch_metadata = slide_batch.collect { meta, slide -> meta }

    model_type_name = slide_model_type ?: model_type
    publish_path_base = "features/${model_type_name}"

    mpath_str = ""
    if (model_path)
        mpath_str = "model_path=${model_path}"

    slide_model_str = ""
    aggregation_str = ""
    if (slide_model_type) {
        slide_model_str = "slide_model_type=${slide_model_type.toUpperCase()}"
        if (slide_model_path)
            slide_model_str += " slide_model_path=${slide_model_path}"
        aggregation_str = "aggregation_method=model"
    } else {
        aggregation_str = "aggregation_method=identity"
    }

    // Filter model configuration (for extracting features before filtering)
    prefilter_model_str = ""
    prefilter_model_path_str = ""
    classifier_pkl_str = ""
    if (params.tiling.filter_tiles) {
        if (prefilter_model_type) {
            prefilter_model_str = "prefilter_model_type=${prefilter_model_type.toUpperCase()}"
            if (prefilter_model_path)
                prefilter_model_path_str = "prefilter_model_path=${prefilter_model_path}"
        }
        if (params.tiling.filter_model_path)
            classifier_pkl_str = "classifier_pkl=${params.tiling.filter_model_path}"
    }

    save_h5_param = "save_features_to_h5=true"  // Always save features in one-step workflow

    // Slides are staged with original filenames (no stageAs), so we can use their names directly
    slide_paths_str = slides.collect { it.name }.join(',')
    slide_ids_str = slide_batch.collect { meta, path -> meta.slide_id }.join(',')

    output_mask_suffix_str = params.tiling.stitch_tiles ? "output_mask_suffix=mask.png" : ""
    output_grid_mask_suffix_str = params.tiling.stitch_tiles ? "output_grid_mask_suffix=grid_mask.png" : ""
    output_thumbnail_suffix_str = params.tiling.save_slide_thumbnail ? "output_thumbnail_suffix=thumbnail.png" : ""
    output_png_dir_suffix_str = params.tiling.save_tile_png ? "output_png_dir_suffix=png" : ""

    // Use slide_batch_size for both:
    // 1. How many slides to process together in this Nextflow task (Type 2 batching)
    // 2. How many slides to aggregate together during slide-level aggregation (Type 3 batching)
    slide_batch_size = params.featurize.slide_batch_size ?: 8

    // Build model_batch_sizes dict string for Hydra (e.g., "model_batch_sizes={CTRANSPATH:32,OPTIMUS:64}")
    model_batch_sizes_str = ""
    if (params.featurize.model_batch_sizes) {
        def batch_sizes_entries = params.featurize.model_batch_sizes.collect { k, v -> "${k.toUpperCase()}:${v}" }.join(',')
        model_batch_sizes_str = "model_batch_sizes={${batch_sizes_entries}}"
    }

    // Use seg_config group if specified, otherwise use individual parameters
    seg_config_str = params.tiling.seg_config_group ? "seg_config=${params.tiling.seg_config_group}" : ""

    // Build individual parameter overrides (only if not using group, or to override group defaults)
    seg_params = []
    if (params.tiling.mpp != null) seg_params << "seg_config.mpp=${params.tiling.mpp}"
    if (params.tiling.patch_size != null) seg_params << "seg_config.patch_size=${params.tiling.patch_size}"
    if (params.tiling.segment_threshold != null) seg_params << "seg_config.segment_threshold=${params.tiling.segment_threshold}"
    if (params.tiling.median_blur_ksize != null) seg_params << "seg_config.median_blur_ksize=${params.tiling.median_blur_ksize}"
    if (params.tiling.morphology_ex_kernel != null) seg_params << "seg_config.morphology_ex_kernel=${params.tiling.morphology_ex_kernel}"
    if (params.tiling.tissue_area_threshold != null) seg_params << "seg_config.tissue_area_threshold=${params.tiling.tissue_area_threshold}"
    if (params.tiling.hole_area_threshold != null) seg_params << "seg_config.hole_area_threshold=${params.tiling.hole_area_threshold}"
    if (params.tiling.max_num_holes != null) seg_params << "seg_config.max_num_holes=${params.tiling.max_num_holes}"
    seg_params_str = seg_params.join(' \\\n        ')

    """
    tessellate_extract_features \
        ${seg_config_str} \
        ${seg_params_str} \
        num_workers=${task.cpus} \
        slide_paths='[${slide_paths_str}]' \
        slide_ids='[${slide_ids_str}]' \
        output_dir=. \
        output_h5_suffix=features.h5 \
        output_pt_suffix=features.pt \
        model_type=${model_type.toUpperCase()} ${mpath_str} \
        use_gpu=${params.featurize.use_gpu ? "true" : "false"} \
        batch_size=${params.featurize.batch_size} \
        ${model_batch_sizes_str} \
        slide_batch_size=${slide_batch_size} \
        ${slide_model_str} \
        ${aggregation_str} \
        ${prefilter_model_str} \
        ${prefilter_model_path_str} \
        ${classifier_pkl_str} \
        classifier_threshold=${params.tiling.filter_threshold} \
        ${output_mask_suffix_str} \
        ${output_grid_mask_suffix_str} \
        ${output_thumbnail_suffix_str} \
        ${output_png_dir_suffix_str} \
        ${save_h5_param}
    """
}
