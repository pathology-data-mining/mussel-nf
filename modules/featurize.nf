include { resolvePrecision } from './utils'

process FEATURIZE_BATCH {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    // Publish slide encoder features (always created)
    publishDir path: { def prefix = post_filter ? '' : 'prefilter_'; "${params.outdir}/features/${prefix}${model_type_input}" }, mode: "${params.publish_mode}", pattern: "*.features.{pt,h5}"
    // Two-step slide-level extraction writes intermediate patch features only as HDF5.
    publishDir path: { def sm = params.featurize.slide_to_patch_mapping; def mt = (sm && sm[model_type_input]) ? sm[model_type_input] : model_type_input; def prefix = post_filter ? '' : 'prefilter_'; "${params.outdir}/features/${prefix}${mt}" }, mode: "${params.publish_mode}", pattern: "*.patch_features.h5"

    input:
    // One batch of slides tiled at one size, paired with one model that uses that
    // size (EXTRACT_FEATURES combines each tile set only with its own models).
    tuple val(slide_batch), path(slides), path(patch_h5s), val(model_type_input), val(tile_size_check) // model_type_input can be:
                          // - Patch encoder: 'resnet50', 'ctranspath', 'gigapath', 'virchow', 'virchow2', 'optimus', 'uni', 'uni2h', 'conch1_5', 'clip', 'googlepath'
                          // - Slide encoder: 'gigapath_slide', 'titan_slide'
                          // When using slide encoders, the required patch encoder is automatically inferred from params.featurize.slide_to_patch_mapping
    val post_filter // true if coords in h5 are post filter (set to true if no filtering)

    output:
    tuple val(batch_metadata), val(model_type_input), path("*.features.pt"), emit: pt
    tuple val(batch_metadata), val(model_type_input), path("*.features.h5"), emit: h5
    tuple val(batch_metadata), val(model_type), path("*.patch_features.h5"), optional: true, emit: patch_h5
    // Slides of the batch without outputs (bin/mussel_batch.py finish); not published.
    tuple val(batch_metadata), val(model_type_input), path("failed_slides.tsv"), optional: true, emit: failed

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

    // Staged basenames and slide IDs contain no shell metacharacters, so the Hydra
    // lists are built in Groovy.
    patch_h5_paths_arg = patch_h5s.collect { "\"${it.name}\"" }.join(',')
    slide_paths_arg = slides.collect { "\"${it.name}\"" }.join(',')
    slide_ids_arg = slide_batch.collect { meta, slide, patch_h5 -> "\"${meta.slide_id}\"" }.join(',')
    slide_ids_csv = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')
    slide_batch_size = params.featurize.slide_batch_size ?: 8

    // Resolve per-model batch size override, falling back to global default
    batch_size = (params.featurize.model_batch_sizes && params.featurize.model_batch_sizes[mtype.toUpperCase()])
        ? params.featurize.model_batch_sizes[mtype.toUpperCase()]
        : (params.featurize.batch_size ?: 64)

    // Resolve embedding precision: per-model override takes precedence over global default.
    // Mussel checks the tiles match the model's recommended size / MPP; the workflow
    // picks the mode (TileSizes.checkMode): "error", "warn" if overridden, or null.
    tile_size_check_str = tile_size_check ? "tile_size_check=${tile_size_check}" : ""

    precision = resolvePrecision(model_type)
    embedding_precision_str = (precision != 'float32') ? "embedding_precision=${precision}" : ""

    // TITAN attention is quadratic in the number of tiles: with a limit, Mussel
    // aggregates at most slide_max_patches tiles per slide (a subset seeded by the
    // slide's file name; Mussel #151). Patch features are still written for every tile.
    slide_max_patches = params.featurize.slide_max_patches ?: 0
    max_slide_patches_str = (slide_model_type && slide_max_patches > 0) ? "max_slide_patches=${slide_max_patches}" : ""

    // Mussel >= 1.6.0 writes the other slides' outputs when some fail, then exits
    // non-zero: bin/mussel_batch.py keeps those (the failed slides go to
    // failed_slides.tsv) and moves empty outputs of slides without tiles aside.
    """
    ${TaskProfile.script(params.profile_dir, task)}
    rc=0
    mussel_batch.py run --log mussel.log -- extract_features \
        'patch_h5_paths=[${patch_h5_paths_arg}]' \
        'slide_paths=[${slide_paths_arg}]' \
        'slide_ids=[${slide_ids_arg}]' \
        output_dir=. \
        output_h5_suffix=features.h5 \
        output_pt_suffix=features.pt \
        num_workers=${task.cpus} \
        model_type=${mtype.toUpperCase()} ${mpath_str} \
        ${slide_model_str} \
        ${aggregation_str} \
        use_gpu=${params.featurize.use_gpu ? "true" : "false"} \
        batch_size=${batch_size} \
        slide_batch_size=${slide_batch_size} \
        ${embedding_precision_str} \
        ${tile_size_check_str} \
        ${max_slide_patches_str} \
        ${TaskProfile.timingsArg(params.profile_dir)} || rc=\$?
    mussel_batch.py finish --rc \$rc --log mussel.log --slide-ids '${slide_ids_csv}' \
        --expect '${model_type_input}={sid}.features.pt' --scan . --report failed_slides.tsv
    """

    stub:
    stub_slide_ids = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')
    batch_metadata = slide_batch.collect { meta, slide, patch_h5 -> meta }
    slide_model_type = (params.featurize.slide_to_patch_mapping && params.featurize.slide_to_patch_mapping[model_type_input]) ? model_type_input : null
    model_type = slide_model_type ? params.featurize.slide_to_patch_mapping[model_type_input] : model_type_input
    is_slide_model = slide_model_type ? "True" : "False"
    """
    #!/usr/bin/env python3
    import os, torch, h5py, numpy as np
    n_feat = 8
    is_slide_model = ${is_slide_model}
    for sid in "${stub_slide_ids}".split(","):
        torch.save(torch.zeros(1, n_feat), f"{sid}.features.pt")
        with h5py.File(f"{sid}.features.h5", "w") as f:
            f.create_dataset("features", data=np.zeros((1, n_feat), dtype="float32"))
        if is_slide_model:
            with h5py.File(f"{sid}.patch_features.h5", "w") as f:
                f.create_dataset("features", data=np.zeros((1, n_feat), dtype="float32"))
                ds = f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
                ds.attrs["patch_size"] = 512
    """
}

// Several patch models (no slide encoder, not CLIP) on one tile set in one task:
// Mussel's extract_features model_types=[...] reads each tile once and feeds every
// model with the same preprocessing (e.g. H-optimus-0 and H-optimus-1). Needs a
// Mussel with model_types support (pathology-data-mining/Mussel#144). Outputs land
// in <model>/<slide_id>.features.{pt,h5} and publish to features/<model>/.
process FEATURIZE_PATCH_MULTI {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/features", mode: "${params.publish_mode}", pattern: "*/*.features.{pt,h5}"

    input:
    tuple val(slide_batch), path(slides), path(patch_h5s), val(model_types), val(tile_size_check) // model_types: list of patch models

    output:
    tuple val(batch_metadata), path("*/*.features.pt"), emit: pt
    tuple val(batch_metadata), path("*/*.features.h5"), emit: h5
    tuple val(batch_metadata), path("failed_slides.tsv"), optional: true, emit: failed

    script:
    batch_metadata = slide_batch.collect { meta, slide, patch_h5 -> meta }
    model_types_arg = model_types.collect { it.toUpperCase() }.join(',')
    precisions = model_types.collectEntries { m -> [m, resolvePrecision(m)] }.findAll { m, prec -> prec != 'float32' }
    // ++: Hydra's structured config is in struct mode, so dict entries can't be set with plain key=value.
    precisions_arg = precisions ? "'++model_precisions={" + precisions.collect { m, prec -> "${m}:${prec}" }.join(',') + "}'" : ""
    paths = model_types.findAll { params.featurize.model_paths && params.featurize.model_paths[it] }
    // Paths double-quoted: unquoted Hydra strings can't hold commas, spaces or braces.
    paths_arg = paths ? "'++model_paths={" + paths.collect { m -> "${m}:\"${params.featurize.model_paths[m]}\"" }.join(',') + "}'" : ""
    sizes = model_types.collect { m -> (params.featurize.model_batch_sizes && params.featurize.model_batch_sizes[m.toUpperCase()]) ?: (params.featurize.batch_size ?: 64) }
    batch_size = sizes.min()
    patch_h5_paths_arg = patch_h5s.collect { "\"${it.name}\"" }.join(',')
    slide_paths_arg = slides.collect { "\"${it.name}\"" }.join(',')
    slide_ids_arg = slide_batch.collect { meta, slide, patch_h5 -> "\"${meta.slide_id}\"" }.join(',')
    tile_size_check_str = tile_size_check ? "tile_size_check=${tile_size_check}" : ""
    // Outputs land in <model>/ (lower-case code, as perModel in modules/mussel.nf expects).
    slide_ids_csv = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')
    expect_args = model_types.collect { m -> "--expect '${m}=${m}/{sid}.features.pt' --scan '${m}'" }.join(' ')
    // Partial failures and slides without tiles: see FEATURIZE_BATCH.
    """
    ${TaskProfile.script(params.profile_dir, task)}
    rc=0
    mussel_batch.py run --log mussel.log -- extract_features \\
        'patch_h5_paths=[${patch_h5_paths_arg}]' \\
        'slide_paths=[${slide_paths_arg}]' \\
        'slide_ids=[${slide_ids_arg}]' \\
        output_dir=. \\
        output_h5_suffix=features.h5 \\
        output_pt_suffix=features.pt \\
        num_workers=${task.cpus} \\
        'model_types=[${model_types_arg}]' ${precisions_arg} ${paths_arg} \\
        aggregation_method=identity \\
        use_gpu=${params.featurize.use_gpu ? "true" : "false"} \\
        batch_size=${batch_size} \\
        ${tile_size_check_str} \\
        ${TaskProfile.timingsArg(params.profile_dir)} || rc=\$?
    mussel_batch.py finish --rc \$rc --log mussel.log --slide-ids '${slide_ids_csv}' \\
        ${expect_args} --report failed_slides.tsv
    """

    stub:
    batch_metadata = slide_batch.collect { meta, slide, patch_h5 -> meta }
    stub_slide_ids = slide_batch.collect { meta, slide, patch_h5 -> meta.slide_id }.join(',')
    stub_models = model_types.join(',')
    """
    #!/usr/bin/env python3
    import os, torch, h5py, numpy as np
    for model in "${stub_models}".split(","):
        os.makedirs(model, exist_ok=True)
        for sid in "${stub_slide_ids}".split(","):
            torch.save(torch.zeros(1, 8), f"{model}/{sid}.features.pt")
            with h5py.File(f"{model}/{sid}.features.h5", "w") as f:
                f.create_dataset("features", data=np.zeros((1, 8), dtype="float32"))
    """
}
