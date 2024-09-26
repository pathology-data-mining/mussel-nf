
params.outdir = 'results'

params.model_types = ['optimus', 'virchow']
params.all_model_types = ['quiltnet', 'gigapath', 'ctranspath', 'resnet50', 'virchow', 'optimus']
params.clip_model_types = ['quiltnet']

for (model_type in params.all_model_types) {
    params[model_type] = [:]
}
params.ctranspath.model_path = "/gpfs/mskmind_ess/limr/repos/TransPath/ctranspath.pth"
params.optimus.model_path = "/gpfs/mskmind_ess/limr/repos/mussel-nf/optimus.pkl"

params.patch_size = 224
params.mpp = 0.5
params.tissue_area_threshold = 100

params.save_tile_png = false

include { CLIP } from './clip'


process TESSELLATE {
    label "bigTask"
    label "cpuTask"

    publishDir "${params.outdir}/tiles/"

    input:
    tuple val(slide_id), path(slide)

    output:
    tuple val(slide_id), path("${slide_id}.patch.h5"), optional: true, emit: h5
    tuple val(slide_id), val("tiles_h5_urlpath"), val("${task.publishDir[0].path}/${slide_id}.patch.h5"), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("patch_size"), val(params.patch_size), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("mpp"), val(params.mpp), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    tuple val(slide_id), val("tissue_area_threshold"), val(params.tissue_area_threshold), path("${slide_id}.patch.h5"), optional: true, topic: meta_out
    path "${slide_id}_png/*.png", optional: true, emit: png

    script:
    save_tile_param = ""
    if (params.save_tile_png)
        save_tile_param = "output_png_dir=${slide_id}_png"
    """
    tessellate \
        patch_config.mpp=${params.mpp} \
        patch_config.patch_size=${params.patch_size} \
        filter_config.tissue_area_threshold=${params.tissue_area_threshold} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        output_h5_path=${slide_id}.patch.h5 \
        ${save_tile_param}
    """
}

params.use_gpu = true

process FEATURIZE {
    label "bigTask"
    label "gpuTask"
    label "parallelTask"

    secret 'HF_TOKEN'

    publishDir "${params.outdir}/features/${model_type}/"

    input:
    tuple val(slide_id), path(slide), path(patch_h5)
    each model_type

    output:
    tuple val(slide_id), val(model_type), path("${slide_id}.features.pt"), emit: pt
    tuple val(slide_id), val(model_type), path("${slide_id}.features.h5"), emit: h5
    tuple val(slide_id), val("${model_type}_features_tensor_urlpath"), val("${task.publishDir[0].path}/${slide_id}.features.pt"), topic: meta_out
    tuple val(slide_id), val("${model_type}_features_h5_urlpath"), val("${task.publishDir[0].path}/${slide_id}.features.h5"), topic: meta_out

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

workflow EXTRACT_FEATURES {
    take:
        ch_slides // slide_id, slide

    main:
        ch_patches = TESSELLATE(ch_slides)
        FEATURIZE(ch_slides.combine(ch_patches.h5, by: 0), params.model_types)

    emit:
        patches = ch_patches.h5
        features = FEATURIZE.out.pt
}

workflow MUSSEL {
    take:
        ch_samples // slide_id, slide_path, oncotree_code

    main:
        ch_slides = ch_samples.map { [it.slide_id, it.slide_path] }

        ch = EXTRACT_FEATURES(ch_slides)


        ch_features = EXTRACT_FEATURES.out.features.branch {
            slide_id, model_type, features ->
                clip: params.clip_model_types.contains(model_type)
        }.clip
        ch_patches = EXTRACT_FEATURES.out.patches

        ch_oncotree_slide = ch_samples.map {
            oncotree_code = "default"
            if ("oncotree_code" in it) {
                oncotree_code = it["oncotree_code"]
            }
            [oncotree_code, it.slide_id]
        }

        ch_clip = CLIP(ch_oncotree_slide,
            ch_slides,
            ch_features,
            ch_patches)

}

