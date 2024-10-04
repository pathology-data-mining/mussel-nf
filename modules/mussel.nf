params.outdir = 'results'

params.filter_model_type = 'ctranspath'

params.model_types = ['optimus', 'virchow']
params.all_model_types = ['quiltnet', 'gigapath', 'ctranspath', 'resnet50', 'virchow', 'optimus']
params.clip_model_types = ['quiltnet']

include { CLIP } from './clip'

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


        FEATURIZE(ch_slides.combine(ch_patches.h5, by: 0), params.model_types)

    emit:
        patches = ch_patches.h5
        features = FEATURIZE.out.pt
}

workflow EXTRACT_FEATURES2 {
    take:
        ch_slides // slide_id, slide

    main:
        ch_patches = TESSELLATE(ch_slides)

        ch_filter_features = FILTER_FEATURIZE(ch_slides.combine(ch_patches.h5, by: 0), params.filter_model_type)
        ch_filtered_tiles = FILTER_TILES(ch_filter_features.h5)

        if (params.stitch_tiles) {
            ch_slides.join(ch_filtered_tiles.h5) | STITCH_TILES
        }

        ch_features = FEATURIZE(ch_slides.combine(ch_filtered_tiles.h5, by: 0), params.model_types)

    emit:
        filtered_h5 = ch_filtered_tiles.h5
        pt = ch_features.pt
        h5 = ch_features.h5
}


workflow MUSSEL {
    take:
        ch_samples // slide_id, slide_path, oncotree_code

    main:
        ch_slides = ch_samples.map { [it.slide_id, it.slide_path] }

        EXTRACT_FEATURES(ch_slides)

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

        if (!params.model_types.disjoint(params.clip_model_types)) {
            ch_clip = CLIP(ch_oncotree_slide,
                ch_slides,
                ch_features,
                ch_patches)
        }


}

workflow MUSSEL2 {
    take:
        ch_samples // slide_id, slide_path, oncotree_code

    main:
        ch_slides = ch_samples.map { [it.slide_id, it.slide_path] }

        EXTRACT_FEATURES2(ch_slides)
}

