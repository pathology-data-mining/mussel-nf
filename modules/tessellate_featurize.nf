process TESSELLATE_FEATURIZE_BATCH {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    // Publish slide encoder features (always created)
    publishDir path: { "${params.outdir}/features/${model_type_input}" }, mode: "${params.publish_mode}", pattern: "pt/*.features.pt", saveAs: { fn -> fn.replaceFirst("pt/", "") }
    publishDir path: { "${params.outdir}/features/${model_type_input}" }, mode: "${params.publish_mode}", pattern: "h5/*.features.h5", saveAs: { fn -> fn.replaceFirst("h5/", "") }
    // Publish patch encoder features (only created when using slide-level model)
    publishDir path: { def sm = params.featurize.slide_to_patch_mapping; def mt = (sm && sm[model_type_input]) ? sm[model_type_input] : model_type_input; "${params.outdir}/features/${mt}" }, mode: "${params.publish_mode}", pattern: "pt/*.patch_features.pt", saveAs: { fn -> fn.replaceFirst("pt/", "") }
    publishDir path: { def sm = params.featurize.slide_to_patch_mapping; def mt = (sm && sm[model_type_input]) ? sm[model_type_input] : model_type_input; "${params.outdir}/features/${mt}" }, mode: "${params.publish_mode}", pattern: "h5/*.patch_features.h5", saveAs: { fn -> fn.replaceFirst("h5/", "") }
    publishDir path: "${params.outdir}/tiles", mode: "${params.publish_mode}", pattern: "tile_h5/*.h5", saveAs: { fn -> fn.replaceFirst("tile_h5/", "") }

    input:
    tuple val(slide_batch), path(slides) // batch with slide files
    each model_type_input // Model type string - can be:
                          // - Patch encoder: 'resnet50', 'ctranspath', 'gigapath', 'virchow', 'virchow2', 'optimus', 'uni', 'uni2h', 'conch1_5', 'clip', 'googlepath'
                          // - Slide encoder: 'gigapath_slide', 'titan_slide'
                          // When using slide encoders, the required patch encoder is automatically inferred from params.featurize.slide_to_patch_mapping

    output:
    tuple val(batch_metadata), val(model_type_input), path("pt/*.features.pt"), optional: true, emit: pt
    tuple val(batch_metadata), val(model_type_input), path("h5/*.features.h5"), optional: true, emit: h5
    tuple val(batch_metadata), val(model_type), path("pt/*.patch_features.pt"), optional: true, emit: patch_pt
    tuple val(batch_metadata), val(model_type), path("h5/*.patch_features.h5"), optional: true, emit: patch_h5
    tuple val(batch_metadata), path("tile_h5/*.patch.h5"), optional: true, emit: tile_h5

    script:
    // Determine if this is a slide-level model and infer the patch encoder
    slide_model_type = params.featurize.slide_to_patch_mapping && params.featurize.slide_to_patch_mapping[model_type_input] ? model_type_input : null
    model_type = slide_model_type ? params.featurize.slide_to_patch_mapping[model_type_input] : model_type_input

    // Look up paths from params if available, otherwise models will be downloaded from HF hub
    model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
    slide_model_path = slide_model_type && params.featurize.slide_model_paths && params.featurize.slide_model_paths[slide_model_type] ? params.featurize.slide_model_paths[slide_model_type] : null

    // For tessellate, also check if we need a prefilter model
    prefilter_model_type = params.tiling.filter_tiles ? params.tiling.filter_model_type : null
    prefilter_model_path = prefilter_model_type && params.featurize.model_paths && params.featurize.model_paths[prefilter_model_type] ? params.featurize.model_paths[prefilter_model_type] : null

    // Extract metadata for all slides in batch
    batch_metadata = slide_batch.collect { meta, slide -> meta }

    model_type_name = slide_model_type ?: model_type
    patch_encoder_name = model_type

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
    classifier_threshold_str = ""
    if (params.tiling.filter_tiles) {
        if (prefilter_model_type) {
            prefilter_model_str = "prefilter_model_type=${prefilter_model_type.toUpperCase()}"
            if (prefilter_model_path)
                prefilter_model_path_str = "prefilter_model_path=${prefilter_model_path}"
        }
        if (params.tiling.filter_model_path)
            classifier_pkl_str = "classifier_pkl=${params.tiling.filter_model_path}"
        classifier_threshold_str = "classifier_threshold=${params.tiling.filter_threshold}"
    }

    save_h5_param = "save_features_to_h5=true"  // Always save features in one-step workflow

    // Slides are staged with original filenames (no stageAs), so we can use their names directly
    slide_paths_str = slides.collect { it.name }.join(',')
    slide_ids_str = slide_batch.collect { meta, path -> meta.slide_id }.join(',')

    output_mask_suffix_str = params.tiling.stitch_tiles ? "output_mask_suffix=mask.png" : ""
    output_grid_mask_suffix_str = params.tiling.stitch_tiles ? "output_grid_mask_suffix=grid_mask.png" : ""
    output_thumbnail_suffix_str = params.tiling.save_slide_thumbnail ? "output_thumbnail_suffix=thumbnail.png" : ""
    output_png_dir_suffix_str = params.tiling.save_tile_png ? "output_png_dir_suffix=png" : ""

    // SLIDE ENCODER BATCHING: When using slide-level models (e.g., gigapath_slide),
    // this controls how many slide-level feature aggregations are computed together.
    // This is the batch size passed to the slide encoder model.
    slide_batch_size = params.featurize.slide_batch_size ?: 8

    // SLIDE PATCH LIMIT: Subsample patches before slide-level aggregation if a slide
    // exceeds this count. Prevents CUDA OOM for TITAN_SLIDE (O(N²) alibi attention).
    // Default null = no limit. Recommended: 4096 for V100 GPUs, 8192 for A100.
    slide_max_patches_str = (params.featurize.slide_max_patches != null)
        ? "max_slide_patches=${params.featurize.slide_max_patches}"
        : ""

    // Resolve per-model batch size override, falling back to global default
    batch_size = (params.featurize.model_batch_sizes && params.featurize.model_batch_sizes[model_type.toUpperCase()])
        ? params.featurize.model_batch_sizes[model_type.toUpperCase()]
        : (params.featurize.batch_size ?: 64)

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
    if (params.tiling.overlap != null) seg_params << "seg_config.overlap=${params.tiling.overlap}"
    if (params.tiling.min_tissue_proportion != null) seg_params << "seg_config.min_tissue_proportion=${params.tiling.min_tissue_proportion}"
    if (params.tiling.remove_artifacts) seg_params << "seg_config.remove_artifacts=true"
    if (params.tiling.remove_penmarks) seg_params << "seg_config.remove_penmarks=true"
    if (params.tiling.seg_model != null) seg_params << "seg_config.seg_model=${params.tiling.seg_model}"
    if (params.tiling.slide_mpp_override != null) seg_params << "seg_config.slide_mpp_override=${params.tiling.slide_mpp_override}"
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
        batch_size=${batch_size} \
        slide_batch_size=${slide_batch_size} \
        ${slide_max_patches_str} \
        ${slide_model_str} \
        ${aggregation_str} \
        ${prefilter_model_str} \
        ${prefilter_model_path_str} \
        ${classifier_pkl_str} \
        ${classifier_threshold_str} \
        ${output_mask_suffix_str} \
        ${output_grid_mask_suffix_str} \
        ${output_thumbnail_suffix_str} \
        ${output_png_dir_suffix_str} \
        ${save_h5_param}

    # When a slide encoder is used, the Mussel CLI writes outputs into model-named
    # subdirs (e.g. TITAN_SLIDE/pt/, CONCH1_5/pt/) instead of flat pt/.
    # Normalize to the flat structure that NF publishDir patterns expect.
    if [[ -d "${model_type_name.toUpperCase()}" ]]; then
        mkdir -p pt h5 tile_h5
        # Slide-level features → flat pt/ and h5/
        [[ -d "${model_type_name.toUpperCase()}/pt" ]] && mv ${model_type_name.toUpperCase()}/pt/*.features.pt pt/ 2>/dev/null || true
        [[ -d "${model_type_name.toUpperCase()}/h5" ]] && mv ${model_type_name.toUpperCase()}/h5/*.features.h5 h5/ 2>/dev/null || true
        # Patch encoder features → rename to *.patch_features.* so NF can tell them apart
        if [[ -d "${patch_encoder_name.toUpperCase()}/pt" ]]; then
            for f in ${patch_encoder_name.toUpperCase()}/pt/*.features.pt; do
                [[ -f "\$f" ]] && mv "\$f" pt/\$(basename "\${f%.features.pt}").patch_features.pt
            done
        fi
        if [[ -d "${patch_encoder_name.toUpperCase()}/h5" ]]; then
            for f in ${patch_encoder_name.toUpperCase()}/h5/*.features.h5; do
                [[ -f "\$f" ]] && mv "\$f" h5/\$(basename "\${f%.features.h5}").patch_features.h5
            done
        fi
        # Tile coordinates live under patch encoder subdir
        [[ -d "${patch_encoder_name.toUpperCase()}/tile_h5" ]] && mv ${patch_encoder_name.toUpperCase()}/tile_h5/* tile_h5/ 2>/dev/null || true
    fi
    """

    stub:
    stub_slide_ids = slide_batch.collect { meta, path -> meta.slide_id }.join(',')
    batch_metadata = slide_batch.collect { meta, path -> meta }
    model_type = (params.featurize.slide_to_patch_mapping && params.featurize.slide_to_patch_mapping[model_type_input]) ? params.featurize.slide_to_patch_mapping[model_type_input] : model_type_input
    model_type_name = model_type_input
    patch_encoder_name = model_type
    is_slide_model = model_type_input != model_type
    """
    #!/usr/bin/env python3
    import os, torch, h5py, numpy as np
    os.makedirs("pt", exist_ok=True)
    os.makedirs("h5", exist_ok=True)
    os.makedirs("tile_h5", exist_ok=True)
    n_feat = 8
    is_slide_model = ${is_slide_model ? "True" : "False"}
    for sid in "${stub_slide_ids}".split(","):
        torch.save(torch.zeros(1, n_feat), f"pt/{sid}.features.pt")
        with h5py.File(f"h5/{sid}.features.h5", "w") as f:
            f.create_dataset("features", data=np.zeros((1, n_feat), dtype="float32"))
        if is_slide_model:
            torch.save(torch.zeros(1, n_feat), f"pt/{sid}.patch_features.pt")
            with h5py.File(f"h5/{sid}.patch_features.h5", "w") as f:
                f.create_dataset("features", data=np.zeros((1, n_feat), dtype="float32"))
        with h5py.File(f"tile_h5/{sid}.patch.h5", "w") as f:
            f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
    """
}
