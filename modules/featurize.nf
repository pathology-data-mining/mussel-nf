include { resolvePrecision } from './utils'

process FEATURIZE_BATCH {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    // HDF5 file locking is incompatible with GPFS/Lustre/NFS: multiple DataLoader
    // workers opening the same .patch.h5 concurrently get EAGAIN (errno=11).
    beforeScript 'export HDF5_USE_FILE_LOCKING=FALSE'

    secret 'HF_TOKEN'

    // Publish slide encoder features (always created)
    publishDir path: { def prefix = post_filter ? '' : 'prefilter_'; "${params.outdir}/features/${prefix}${model_type_input}" }, mode: "${params.publish_mode}", pattern: "*.features.{pt,h5}"
    // Publish patch encoder features (only created when using slide-level model)
    publishDir path: { def sm = params.featurize.slide_to_patch_mapping; def mt = (sm && sm[model_type_input]) ? sm[model_type_input] : model_type_input; def prefix = post_filter ? '' : 'prefilter_'; "${params.outdir}/features/${prefix}${mt}" }, mode: "${params.publish_mode}", pattern: "*.patch_features.{pt,h5}"

    input:
    tuple val(slide_batch), path(slides), path(patch_h5s)
    each model_type_input // Model type string - can be:
                          // - Patch encoder: 'resnet50', 'ctranspath', 'gigapath', 'virchow', 'virchow2', 'optimus', 'uni', 'uni2h', 'conch1_5', 'clip', 'googlepath'
                          // - Slide encoder: 'gigapath_slide', 'titan_slide'
                          // When using slide encoders, the required patch encoder is automatically inferred from params.featurize.slide_to_patch_mapping
    val post_filter // true if coords in h5 are post filter (set to true if no filtering)

    output:
    tuple val(batch_metadata), val(model_type_input), path("*.features.pt"), emit: pt
    tuple val(batch_metadata), val(model_type_input), path("*.features.h5"), emit: h5
    // Patch-level features are only produced when using slide-level encoders (optional)
    tuple val(batch_metadata), val(model_type), path("*.patch_features.pt"), optional: true, emit: patch_pt
    tuple val(batch_metadata), val(model_type), path("*.patch_features.h5"), optional: true, emit: patch_h5

    script:
    // Determine if this is a slide-level model and infer the patch encoder
    slide_model_type = params.featurize.slide_to_patch_mapping && params.featurize.slide_to_patch_mapping[model_type_input] ? model_type_input : null
    model_type = slide_model_type ? params.featurize.slide_to_patch_mapping[model_type_input] : model_type_input

    // Look up paths from params if available, otherwise models will be downloaded from HF hub
    model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
    slide_model_path = slide_model_type && params.featurize.slide_model_paths && params.featurize.slide_model_paths[slide_model_type] ? params.featurize.slide_model_paths[slide_model_type] : null

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
    patch_encoder_name = "${post_filter ? '' : 'prefilter_'}${model_type}"

    // Build lists of paths for batch processing - use staged file basenames
    // Nextflow stages files in work directory, so we can use basenames directly
    patch_h5_paths_str = patch_h5s.collect { it.name }.join(',')
    slide_paths_str = slides.collect { it.name }.join(',')
    slide_ids_str = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')

    // SLIDE ENCODER BATCHING: When using slide-level models (e.g., gigapath_slide),
    // this controls how many slide-level feature aggregations are computed together.
    // This is the batch size passed to the slide encoder model.
    slide_batch_size = params.featurize.slide_batch_size ?: 8

    // Resolve per-model batch size override, falling back to global default
    batch_size = (params.featurize.model_batch_sizes && params.featurize.model_batch_sizes[mtype.toUpperCase()])
        ? params.featurize.model_batch_sizes[mtype.toUpperCase()]
        : (params.featurize.batch_size ?: 64)

    // Resolve embedding precision: per-model override takes precedence over global default
    precision = resolvePrecision(model_type)
    embedding_precision_str = (precision != 'float32') ? "embedding_precision=${precision}" : ""

    """
    extract_features \
        patch_h5_paths='[${patch_h5_paths_str}]' \
        slide_paths='[${slide_paths_str}]' \
        slide_ids='[${slide_ids_str}]' \
        output_dir=. \
        num_workers=${task.cpus} \
        model_type=${mtype.toUpperCase()} ${mpath_str} \
        use_gpu=${params.featurize.use_gpu ? "true" : "false"} \
        batch_size=${batch_size} \
        slide_batch_size=${slide_batch_size} \
        ${slide_model_str} \
        ${aggregation_str} \
        ${embedding_precision_str}
    """

    stub:
    stub_slide_ids = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')
    batch_metadata = slide_batch.collect { meta, slide, patch_h5 -> meta }
    model_type = (params.featurize.slide_to_patch_mapping && params.featurize.slide_to_patch_mapping[model_type_input]) ? params.featurize.slide_to_patch_mapping[model_type_input] : model_type_input
    model_type_name = "${post_filter ? '' : 'prefilter_'}${model_type}"
    patch_encoder_name = model_type_name
    """
    #!/usr/bin/env python3
    import os, torch, h5py, numpy as np
    n_feat = 8
    for sid in "${stub_slide_ids}".split(","):
        torch.save(torch.zeros(1, n_feat), f"{sid}.features.pt")
        with h5py.File(f"{sid}.features.h5", "w") as f:
            f.create_dataset("features", data=np.zeros((1, n_feat), dtype="float32"))
    """
}
