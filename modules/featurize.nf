process FEATURIZE {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/${publish_path}", pattern: "*.features.pt", mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide), path(patch_h5)
    each model_config // tuple of [model_type, model_path]
    val post_filter // true if coords in h5 are post filter (set to true if no filtering)

    output:
    tuple val(meta), val(model_type), path("${meta.slide_id}.features.pt"), emit: pt
    tuple val(meta), val(model_type), path("${meta.slide_id}.features.h5"), emit: h5
    tuple val(meta), val("${model_type_name}_features_tensor_path"), val("${publish_path}/${meta.slide_id}.features.pt"), topic: slide_meta

    script:
    model_type = model_config[0]
    model_path = model_config[1]
    
    mtype = model_type
    if (model_type in params.clip.model_types)
        mtype = "CLIP"
    mpath_str = ""
    if (model_path)
        mpath_str = "model_path=${model_path}"

    model_type_name = "${post_filter ? '' : 'prefilter_'}${model_type}"
    publish_path = "features/${model_type_name}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    """
    extract_features \
        slide_path=${slide} \
        patch_h5_path=${patch_h5} \
        output_h5_path=${meta.slide_id}.features.h5 \
        output_pt_path=${meta.slide_id}.features.pt \
        num_workers=${task.cpus} \
        model_type=${mtype.toUpperCase()} ${mpath_str} \
        use_gpu=${params.featurize.use_gpu ? "true" : "false"}
    """
}
