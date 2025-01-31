process FEATURIZE {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/${publish_path}", pattern: "*.features.pt", mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide), path(patch_h5)
    each model_type

    output:
    tuple val(meta), val(model_type), path("${meta.slide_id}.features.pt"), emit: pt
    tuple val(meta), val(model_type), path("${meta.slide_id}.features.h5"), emit: h5
    tuple val(meta), val("${model_type}_features_tensor_urlpath"), val("${publish_path}/${meta.slide_id}.features.pt"), topic: slide_meta

    script:
    publish_path = "features/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    mtype = model_type
    if (model_type in params.featurize.clip_model_types)
        mtype = "CLIP"
    mpath_str = ""
    if (params.featurize.model_paths && params.featurize.model_paths[model_type])
        mpath_str = "model_path=${params.featurize.model_paths[model_type]}"
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
