
params.outdir = 'results'

params.model_types = ['quiltnet', 'ctranspath', 'resnet50']
params.all_model_types = ['quiltnet', 'gigapath', 'ctranspath', 'resnet50']
params.clip_model_types = ['quiltnet']

for (model_type in params.all_model_types) {
    params[model_type] = [:]
}

params.quiltnet.patch_size = 256
params.quiltnet.step_size = 256
params.quiltnet.model_path = "hf-hub:wisdomik/QuiltNet-B-16-PMB"

params.resnet50.patch_size = 256
params.resnet50.step_size = 256

params.ctranspath.patch_size = 224
params.ctranspath.step_size = 224
params.ctranspath.model_path = "/gpfs/mskmind_ess/limr/repos/TransPath/ctranspath_model.pt"

params.quiltnet.patch_size = 256
params.quiltnet.step_size = 256
params.quiltnet.model_path = "hf-hub:wisdomik/QuiltNet-B-16-PMB"

params.gigapath.patch_size = 256
params.gigapath.step_size = 256
params.gigapath.model_path = "/gpfs/mskmind_ess/limr/repos/hf/prov-gigapath"

params.mpp = 0.5
params.tissue_area_threshold = 100

include { CLIP } from './clip'

process TESSELLATE {
    label "bigTask"
    label "cpuTask"

    publishDir "${params.outdir}/tiles/${model_type}/"

    errorStrategy "ignore"

    input:
    tuple val(slide_id), path(slide)
    each model_type

    output:
    tuple val(slide_id), val(model_type), path("${slide_id}.patch.h5")

    script:
    """
    tessellate \
        patch_config.patch_size=${params[model_type].patch_size} \
        patch_config.step_size=${params[model_type].step_size} \
        patch_config.mpp=${params.mpp} \
        filter_config.tissue_area_threshold=${params.tissue_area_threshold} \
        patch_config.num_workers=${task.cpus} \
        slide_path=${slide} \
        output_h5_path=${slide_id}.patch.h5 \
        stitch_jpeg_path=${slide_id}.stitch.jpeg
    """
}

params.use_gpu = true
params.gpu_device_ids = [0]

process FEATURIZE {
    label "bigTask"
    label "gpuTask"

    secret 'HF_TOKEN'

    publishDir "${params.outdir}/features/${model_type}/"

    input:
    tuple val(slide_id), path(slide), val(model_type), path(patch_h5)

    output:
    tuple val(slide_id), val(model_type), path("${slide_id}.features.pt")

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
        use_gpu=${params.use_gpu ? "true" : "false"} \
        gpu_device_ids="${params.gpu_device_ids}"
    """
}

workflow EXTRACT_FEATURES {
    take:
        ch_slides // slide_id, slide

    main:
        ch_patches = TESSELLATE(ch_slides, params.model_types)
        FEATURIZE(ch_slides.combine(ch_patches, by: 0))

    emit:
        patches = TESSELLATE.out
        features = FEATURIZE.out
}

workflow MUSSEL {
    take:
        ch_samples // slide_id, slide, oncotree_code

    main:
        ch_slides = ch_samples.map { [it.slide_id, it.slide_path] }

        ch = EXTRACT_FEATURES(ch_slides)

        ch_features = EXTRACT_FEATURES.out.features.branch {
            slide_id, model_type, features ->
                clip: params.clip_model_types.contains(model_type)
        }.clip
        ch_patches = EXTRACT_FEATURES.out.patches.branch {
            slide_id, model_type, patches ->
                clip: params.clip_model_types.contains(model_type)
        }.clip

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

