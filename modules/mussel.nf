include { CLIP } from './clip'
include { LINEAR_PROBE } from './linear_probe'

include { FEATURIZE_BATCH; FEATURIZE_BATCH as FILTER_FEATURIZE } from './featurize'

include { TESSELLATE; FILTER_TILES } from './tessellation'

include { TESSELLATE_FEATURIZE_BATCH } from './tessellate_featurize'



workflow EXTRACT_FEATURES {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        ch_patches = TESSELLATE(ch_samples)

        if (params.tiling.filter_tiles) {
            // Batch slides for filtering feature extraction
            // Extract features using the feature extraction model (e.g., ctranspath)
            // Then FILTER_TILES will use the classifier_pkl to filter tiles
            filter_model_type = params.tiling.filter_model_type
            filter_model_path = params.featurize.model_paths ? params.featurize.model_paths[filter_model_type] : null
            filter_model_config = Channel.of([filter_model_type, filter_model_path, null, null])

            ch_filter_slide_batches = ch_samples.combine(ch_patches.h5, by: 0)
                .collate(params.featurize.slide_batch_size ?: 8)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s)
                }

            FILTER_FEATURIZE(ch_filter_slide_batches, filter_model_config, false)

            // Flatten filter results
            ch_filter_h5 = FILTER_FEATURIZE.out.h5
                .flatMap { batch_meta, model_type, h5_files ->
                    h5_files.collect { h5_file ->
                        def filename = h5_file.name.replaceAll('.features.h5$', '')
                        def meta = batch_meta.find { it.slide_id == filename }
                        tuple(meta, model_type, h5_file)
                    }
                }

            ch_filter_patches = FILTER_TILES(ch_filter_h5)

            // Create a channel with model configs (model_type, model_path, slide_model_type, slide_model_path)
            ch_model_configs = Channel.fromList(params.featurize.model_types).map { model_type ->
                model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
                slide_model_type = params.featurize.slide_model_types ? params.featurize.slide_model_types[0] : null
                slide_model_path = slide_model_type && params.featurize.slide_model_paths ? params.featurize.slide_model_paths[slide_model_type] : null
                [model_type, model_path, slide_model_type, slide_model_path]
            }

            // Batch slides together for processing
            ch_slide_batches = ch_samples.combine(ch_filter_patches.h5, by: 0)
                .collate(params.featurize.slide_batch_size ?: 8)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s)
                }

            FEATURIZE_BATCH(
                ch_slide_batches,
                ch_model_configs,
                true
            )

            // Flatten batch results back to individual slides
            // Use flatMap to emit individual items from batch
            ch_pt_out = FEATURIZE_BATCH.out.pt
                .flatMap { batch_meta, model_type, pt_files ->
                    // Match each file to its metadata by extracting slide_id from filename
                    pt_files.collect { pt_file ->
                        def filename = pt_file.name.replaceAll('.features.pt$', '')
                        def meta = batch_meta.find { it.slide_id == filename }
                        tuple(meta, model_type, pt_file)
                    }
                }

            ch_h5_out = FEATURIZE_BATCH.out.h5
                .flatMap { batch_meta, model_type, h5_files ->
                    h5_files.collect { h5_file ->
                        def filename = h5_file.name.replaceAll('.features.h5$', '')
                        def meta = batch_meta.find { it.slide_id == filename }
                        tuple(meta, model_type, h5_file)
                    }
                }
        } else {
            // Create a channel with model configs (model_type, model_path, slide_model_type, slide_model_path)
            ch_model_configs = Channel.fromList(params.featurize.model_types).map { model_type ->
                model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
                slide_model_type = params.featurize.slide_model_types ? params.featurize.slide_model_types[0] : null
                slide_model_path = slide_model_type && params.featurize.slide_model_paths ? params.featurize.slide_model_paths[slide_model_type] : null
                [model_type, model_path, slide_model_type, slide_model_path]
            }

            // Batch slides together for processing
            ch_slide_batches = ch_samples.combine(ch_patches.h5, by: 0)
                .collate(params.featurize.slide_batch_size ?: 8)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s)
                }

            FEATURIZE_BATCH(
                ch_slide_batches,
                ch_model_configs,
                true
            )

            // Flatten batch results back to individual slides
            ch_pt_out = FEATURIZE_BATCH.out.pt
                .flatMap { batch_meta, model_type, pt_files ->
                    pt_files.collect { pt_file ->
                        def filename = pt_file.name.replaceAll('.features.pt$', '')
                        def meta = batch_meta.find { it.slide_id == filename }
                        tuple(meta, model_type, pt_file)
                    }
                }

            ch_h5_out = FEATURIZE_BATCH.out.h5
                .flatMap { batch_meta, model_type, h5_files ->
                    h5_files.collect { h5_file ->
                        def filename = h5_file.name.replaceAll('.features.h5$', '')
                        def meta = batch_meta.find { it.slide_id == filename }
                        tuple(meta, model_type, h5_file)
                    }
                }
        }


    emit:
        patches_h5 = ch_patches.h5
        pt = ch_pt_out
        h5 = ch_h5_out
}


workflow EXTRACT_FEATURES_ONE_STEP {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        // Create a channel with model configs (model_type, model_path, slide_model_type, slide_model_path)
        ch_model_configs = Channel.fromList(params.featurize.model_types).map { model_type ->
            model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
            slide_model_type = params.featurize.slide_model_types ? params.featurize.slide_model_types[0] : null
            slide_model_path = slide_model_type && params.featurize.slide_model_paths ? params.featurize.slide_model_paths[slide_model_type] : null
            [model_type, model_path, slide_model_type, slide_model_path]
        }

        // Batch slides together for one-step tessellate + featurize
        ch_slide_batches = ch_samples
            .collate(params.featurize.slide_batch_size ?: 8)
            .map { batch ->
                def slides = batch.collect { meta, slide -> slide }
                tuple(batch, slides)
            }

        TESSELLATE_FEATURIZE_BATCH(
            ch_slide_batches,
            ch_model_configs
        )

        // Flatten batch results back to individual slides
        ch_pt_out = TESSELLATE_FEATURIZE_BATCH.out.pt
            .flatMap { batch_meta, model_type, pt_files ->
                pt_files.collect { pt_file ->
                    def filename = pt_file.name.replaceAll('.features.pt$', '')
                    def meta = batch_meta.find { it.slide_id == filename }
                    tuple(meta, model_type, pt_file)
                }
            }

        ch_h5_out = TESSELLATE_FEATURIZE_BATCH.out.h5
            .flatMap { batch_meta, model_type, h5_files ->
                h5_files.collect { h5_file ->
                    def filename = h5_file.name.replaceAll('.features.h5$', '')
                    def meta = batch_meta.find { it.slide_id == filename }
                    tuple(meta, model_type, h5_file)
                }
            }

        ch_patches_out = TESSELLATE_FEATURIZE_BATCH.out.patch_h5
            .flatMap { batch_meta, patch_h5_files ->
                patch_h5_files.collect { patch_h5_file ->
                    def filename = patch_h5_file.name.replaceAll('.patch.h5$', '')
                    def meta = batch_meta.find { it.slide_id == filename }
                    tuple(meta, patch_h5_file)
                }
            }

    emit:
        patches_h5 = ch_patches_out
        pt = ch_pt_out
        h5 = ch_h5_out
}


workflow MUSSEL {
    take:
        ch_samples // tuple val(meta), file(slide_path)
        ch_annotations // tuple val(meta), file(annotation_bmp_path)

    main:
        // Choose between one-step or two-step workflow
        if (params.use_one_step_workflow) {
            ch_extract_feat = EXTRACT_FEATURES_ONE_STEP(ch_samples)
        } else {
            ch_extract_feat = EXTRACT_FEATURES(ch_samples)
        }

        LINEAR_PROBE(ch_annotations, ch_extract_feat.h5)

        ch_features = ch_extract_feat.pt.branch {
            meta, model_type, features ->
                clip: params.clip.model_types.contains(model_type)
        }.clip
        ch_patches = ch_extract_feat.patches_h5


        if (!params.featurize.model_types.disjoint(params.clip.model_types)) {
            ch_clip = CLIP(
                ch_samples,
                ch_features,
                ch_patches)
        }
}

