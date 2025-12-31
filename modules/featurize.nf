process FEATURIZE_BATCH {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/${publish_path_base}", mode: "${params.publish_mode}", pattern: "*.{pt,h5}"

    input:
    tuple val(slide_batch), path(slides, stageAs: 'slide_*'), path(patch_h5s, stageAs: '*.patch.h5')
    each model_config // tuple of [model_type, model_path, slide_model_type, slide_model_path]
    val post_filter // true if coords in h5 are post filter (set to true if no filtering)

    output:
    tuple val(batch_metadata), val(model_type), path("*.features.pt"), emit: pt
    tuple val(batch_metadata), val(model_type), path("*.features.h5"), emit: h5

    script:
    model_type = model_config[0]
    model_path = model_config[1]
    slide_model_type = model_config.size() > 2 ? model_config[2] : null
    slide_model_path = model_config.size() > 3 ? model_config[3] : null
    prefilter_model_type = model_config.size() > 4 ? model_config[4] : null
    prefilter_model_path = model_config.size() > 5 ? model_config[5] : null

    // Extract metadata for all slides in batch
    batch_metadata = slide_batch.collect { meta, slide, patch_h5 -> meta }

    mtype = model_type
    if (model_type in params.clip.model_types)
        mtype = "CLIP"
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

    model_type_name = "${post_filter ? '' : 'prefilter_'}${slide_model_type ?: model_type}"
    publish_path_base = "features/${model_type_name}"

    // Build lists of paths for batch processing - use staged file basenames
    // Nextflow stages files in work directory, so we can use basenames directly
    patch_h5_paths_str = patch_h5s.collect { it.name }.join(',')
    slide_paths_str = slides.collect { it.name }.join(',')
    slide_ids_str = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')

    // Use slide_batch_size for both:
    // 1. How many slides to process together in this Nextflow task (Type 2 batching)
    // 2. How many slides to aggregate together during slide-level aggregation (Type 3 batching)
    slide_batch_size = params.featurize.slide_batch_size ?: 8

    """
    extract_features \
        patch_h5_paths='[${patch_h5_paths_str}]' \
        slide_paths='[${slide_paths_str}]' \
        slide_ids='[${slide_ids_str}]' \
        output_dir=. \
        num_workers=${task.cpus} \
        model_type=${mtype.toUpperCase()} ${mpath_str} \
        use_gpu=${params.featurize.use_gpu ? "true" : "false"} \
        batch_size=${params.featurize.batch_size ?: 64} \
        slide_batch_size=${slide_batch_size} \
        ${slide_model_str} \
        ${aggregation_str}
    """
}
