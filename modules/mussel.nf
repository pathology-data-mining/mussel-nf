params.outdir = 'results'

params.model_types = ['quiltnet', 'ctranspath', 'resnet50']

for (model_type in params.model_types) {
    params[model_type] = [:]
}

params.resnet50.patch_size = 256
params.resnet50.step_size = 256

params.ctranspath.patch_size = 224
params.ctranspath.step_size = 224

params.quiltnet.patch_size = 256
params.quiltnet.step_size = 256

params.ctranspath_repo_dir = ""
params.ctranspath_model_path = ""
params.quiltnet_model_path = "hf-hub:wisdomik/QuiltNet-B-16-PMB"

params.mpp = 0.5
params.tissue_area_threshold = 100

include { QUILTNET } from './quiltnet'

process TESSELLATE {
    label "bigTask"

    publishDir "$params.outdir/tiles/$model_type/"

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

    publishDir "$params.outdir/features/$model_type/"

    input:
    tuple val(slide_id), path(slide), val(model_type), path(patch_h5)

    output:
    tuple val(slide_id), val(model_type), path("${slide_id}.features.pt")

    script:
    """
    extract_features \
        slide_path=${slide} \
        patch_h5_path=${patch_h5} \
        output_h5_path=${slide_id}.features.h5 \
        output_pt_path=${slide_id}.features.pt \
        num_workers=${task.cpus} \
        transpath_dir=${params.ctranspath_repo_dir} \
        transpath_model_path=${params.ctranspath_model_path} \
        quiltnet_model_path=${params.quiltnet_model_path} \
        model=${model_type.toUpperCase()} \
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
        ch_slides = ch_samples.map { [it.slide_id, it.slide] }
        ch = EXTRACT_FEATURES(ch_slides)

        ch_quiltnet = Channel.empty()
        if ("quiltnet" in params.model_types) {
            ch_features = EXTRACT_FEATURES.out.features.branch {
                slide_id, model_type, features ->
                    quiltnet: model_type == "quiltnet"
            }.quiltnet
            ch_patches = EXTRACT_FEATURES.out.patches.branch {
                slide_id, model_type, patches ->
                    quiltnet: model_type == "quiltnet"
            }.quiltnet
            ch_oncotree_slide = ch_samples.map { [it.oncotree_code, it.slide_id] }
            ch_quiltnet = QUILTNET(ch_oncotree_slide,
                ch_slides,
                ch_features,
                ch_patches)
        }

}

