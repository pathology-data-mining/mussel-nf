for (model_type in params.all_model_types) {
    params[model_type] = [:]
}
params.ctranspath.model_path = "/gpfs/mskmind_ess/limr/repos/TransPath/ctranspath.pth"
params.optimus.model_path = "/gpfs/mskmind_ess/limr/repos/mussel-nf/optimus.pkl"
params.resnet50.model_path = "/gpfs/mskmind_ess/limr/repos/Mussel/resnet50.pkl"

params.use_gpu = true


process FEATURIZE {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir path: "${params.outdir}/features/${model_type}/", pattern: "*.features.pt", mode: "${params.publish_mode}"

    input:
    tuple val(slide_id), path(slide), path(patch_h5)
    each model_type

    output:
    tuple val(slide_id), val(model_type), path("${slide_id}.features.pt"), emit: pt
    tuple val(slide_id), val(model_type), path("${slide_id}.features.h5"), emit: h5
    tuple val(slide_id), val("${model_type}_features_tensor_urlpath"), val("${task.publishDir[0].path}/${slide_id}.features.pt"), topic: meta_out

    script:
    mtype = model_type
    if (model_type in params.clip_model_types)
        mtype = "CLIP"
    """
    extract_features \
        slide_path=${slide} \
        patch_h5_path=${patch_h5} \
        output_h5_path=${slide_id}.features.h5 \
        output_pt_path=${slide_id}.features.pt \
        num_workers=${task.cpus} \
        model_path=${params[model_type].model_path} \
        model_type=${mtype.toUpperCase()} \
        use_gpu=${params.use_gpu ? "true" : "false"}
    """
}
