params.outdir = 'results'

params.filter_model_type = 'ctranspath'

params.model_types = ['optimus', 'virchow']
params.all_model_types = ['quiltnet', 'gigapath', 'ctranspath', 'resnet50', 'virchow', 'optimus']
params.clip_model_types = ['quiltnet']

params.stitch_tiles = false

params.filter_tiles = false

include { CLIP } from './clip'
include { LINEAR_PROBE } from './linear_probe'

include { FEATURIZE; FEATURIZE as FILTER_FEATURIZE } from './featurize'

include { TESSELLATE; STITCH_TILES; FILTER_TILES } from './tessellation'


workflow EXTRACT_FEATURES {
    take:
        ch_slides // slide_id, slide

    main:
        ch_patches = TESSELLATE(ch_slides)

        if (params.stitch_tiles) {
            ch_slides.join(ch_patches.h5) | STITCH_TILES
        }

        if (params.filter_tiles) {
            ch_filter_features = FILTER_FEATURIZE(ch_slides.combine(ch_patches.h5, by: 0), params.filter_model_type)
            ch_filter_patches = FILTER_TILES(ch_filter_features.h5)
            FEATURIZE(ch_slides.combine(ch_filter_patches.h5, by: 0), params.model_types)
        } else {
            FEATURIZE(ch_slides.combine(ch_patches.h5, by: 0), params.model_types)
        }


    emit:
        patches_h5 = ch_patches.h5
        pt = FEATURIZE.out.pt
        h5 = FEATURIZE.out.h5
}

workflow MUSSEL {
    take:
        ch_samples // slide_id, slide_path, oncotree_code
        ch_annotations // slide_id, annotation_bmp_path

    main:
        ch_slides = ch_samples.map { [it.slide_id, it.slide_path] }

        ch_extract_feat = EXTRACT_FEATURES(ch_slides)

        LINEAR_PROBE(ch_annotations, ch_extract_feat.h5)

        ch_features = ch_extract_feat.pt.branch {
            slide_id, model_type, features ->
                clip: params.clip_model_types.contains(model_type)
        }.clip
        ch_patches = EXTRACT_FEATURES.out.patches_h5

        ch_oncotree_slide = ch_samples.map {
            oncotree_code = "default"
            if ("oncotree_code" in it) {
                oncotree_code = it["oncotree_code"]
            }
            [oncotree_code, it.slide_id]
        }

        if (!params.model_types.disjoint(params.clip_model_types)) {
            ch_clip = CLIP(ch_oncotree_slide,
                ch_slides,
                ch_features,
                ch_patches)
        }
}
