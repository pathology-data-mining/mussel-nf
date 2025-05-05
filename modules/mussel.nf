include { CLIP } from './clip'
include { LINEAR_PROBE } from './linear_probe'

include { FEATURIZE; FEATURIZE as FILTER_FEATURIZE } from './featurize'

include { TESSELLATE; FILTER_TILES } from './tessellation'



workflow EXTRACT_FEATURES {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        ch_patches = TESSELLATE(ch_samples)

        if (params.tiling.filter_tiles) {
            ch_filter_features = FILTER_FEATURIZE(ch_samples.combine(ch_patches.h5, by: 0), params.tiling.filter_model_type, false)
            ch_filter_patches = FILTER_TILES(ch_filter_features.h5)
            FEATURIZE(ch_samples.combine(ch_filter_patches.h5, by: 0), params.featurize.model_types, true)
        } else {
            FEATURIZE(ch_samples.combine(ch_patches.h5, by: 0), params.featurize.model_types, true)
        }


    emit:
        patches_h5 = ch_patches.h5
        pt = FEATURIZE.out.pt
        h5 = FEATURIZE.out.h5
}



workflow MUSSEL {
    take:
        ch_samples // tuple val(meta), file(slide_path)
        ch_annotations // tuple val(meta), file(annotation_bmp_path)

    main:

        ch_extract_feat = EXTRACT_FEATURES(ch_samples)

        LINEAR_PROBE(ch_annotations, ch_extract_feat.h5)

        ch_features = ch_extract_feat.pt.branch {
            slide_id, model_type, features ->
                clip: params.clip.model_types.contains(model_type)
        }.clip
        ch_patches = EXTRACT_FEATURES.out.patches_h5


        if (!params.featurize.model_types.disjoint(params.clip.model_types)) {
            ch_clip = CLIP(
                ch_samples,
                ch_features,
                ch_patches)
        }
}

