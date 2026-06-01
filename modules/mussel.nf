include { CLIP } from './clip'
include { LINEAR_PROBE } from './linear_probe'
include { ABMIL_BENCHMARK_WORKFLOW } from './abmil_benchmark/main'

include { FEATURIZE_BATCH; FEATURIZE_BATCH as FILTER_FEATURIZE } from './featurize'

include { TESSELLATE; FILTER_TILES; EMIT_MPP_META } from './tessellation'

include { TESSELLATE_FEATURIZE_BATCH } from './tessellate_featurize'

include { CONVERT_FEATURES_PRECISION } from './convert_precision'

include { WDS_SHARD } from './wds'
include { MERGE_SAMPLE_FEATURES } from './sample_merge'
include { resolvePrecision } from './utils'


workflow EXTRACT_FEATURES {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        ch_patches = TESSELLATE(ch_samples)
        EMIT_MPP_META(ch_patches.h5)

        if (params.tiling.filter_tiles) {
            // Batch slides for filtering feature extraction
            // Extract features using the feature extraction model (e.g., ctranspath)
            // Then FILTER_TILES will use the classifier_pkl to filter tiles
            filter_model_type = Channel.of(params.tiling.filter_model_type)

            ch_filter_slide_batches = ch_samples.combine(ch_patches.h5, by: 0)
                .collate(params.featurize.workflow_batch_size ?: params.featurize.slide_batch_size ?: 8)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s)
                }

            FILTER_FEATURIZE(ch_filter_slide_batches, filter_model_type, false)

            // Flatten filter results
            // Match files to metadata by index since filenames may use staged names
            ch_filter_h5 = FILTER_FEATURIZE.out.h5
                .map { batch_meta, model_type, h5_files ->
                    def files_list = h5_files instanceof List ? h5_files : [h5_files]
                    // Sort both lists by slide_id so positional pairing is correct
                    // regardless of channel arrival order (which is nondeterministic).
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, model_type, sorted_files]
                }
                .transpose(by: [0, 2])
                .map { meta, model_type, h5_file ->
                    tuple(meta, model_type, h5_file)
                }

            ch_filter_patches = FILTER_TILES(ch_filter_h5)

            // Just pass model type strings - processes will look up paths from params
            ch_model_types = Channel.fromList(params.featurize.model_types)

            // Batch slides together for processing (workflow batching)
            ch_slide_batches = ch_samples.combine(ch_filter_patches.h5, by: 0)
                .collate(params.featurize.workflow_batch_size ?: params.featurize.slide_batch_size ?: 8)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s)
                }

            FEATURIZE_BATCH(
                ch_slide_batches,
                ch_model_types,
                true
            )

            // Flatten batch results back to individual slides
            // Use indexed matching since output files may use staged names
            ch_pt_out = FEATURIZE_BATCH.out.pt
                .flatMap { batch_meta, model_type, pt_files ->
                    def files_list = pt_files instanceof List ? pt_files : [pt_files]
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, sorted_files].transpose().collect { meta, pt_file ->
                        tuple(meta, model_type, pt_file)
                    }
                }

            ch_h5_out = FEATURIZE_BATCH.out.h5
                .flatMap { batch_meta, model_type, h5_files ->
                    def files_list = h5_files instanceof List ? h5_files : [h5_files]
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, sorted_files].transpose().collect { meta, h5_file ->
                        tuple(meta, model_type, h5_file)
                    }
                }

            ch_patch_h5_out = FEATURIZE_BATCH.out.patch_h5
                .flatMap { batch_meta, model_type, patch_h5_files ->
                    def files_list = patch_h5_files instanceof List ? patch_h5_files : [patch_h5_files]
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, sorted_files].transpose().collect { meta, patch_h5_file ->
                        tuple(meta, model_type, patch_h5_file)
                    }
                }
        } else {
            // Just pass model type strings - processes will look up paths from params
            ch_model_types = Channel.fromList(params.featurize.model_types)

            // Batch slides together for processing (workflow batching)
            ch_slide_batches = ch_samples.combine(ch_patches.h5, by: 0)
                .collate(params.featurize.workflow_batch_size ?: params.featurize.slide_batch_size ?: 8)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s)
                }

            FEATURIZE_BATCH(
                ch_slide_batches,
                ch_model_types,
                true
            )

            // Flatten batch results back to individual slides
            // Use indexed matching since output files may use staged names
            ch_pt_out = FEATURIZE_BATCH.out.pt
                .flatMap { batch_meta, model_type, pt_files ->
                    def files_list = pt_files instanceof List ? pt_files : [pt_files]
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, sorted_files].transpose().collect { meta, pt_file ->
                        tuple(meta, model_type, pt_file)
                    }
                }

            ch_h5_out = FEATURIZE_BATCH.out.h5
                .flatMap { batch_meta, model_type, h5_files ->
                    def files_list = h5_files instanceof List ? h5_files : [h5_files]
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, sorted_files].transpose().collect { meta, h5_file ->
                        tuple(meta, model_type, h5_file)
                    }
                }

            ch_patch_h5_out = FEATURIZE_BATCH.out.patch_h5
                .flatMap { batch_meta, model_type, patch_h5_files ->
                    def files_list = patch_h5_files instanceof List ? patch_h5_files : [patch_h5_files]
                    def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                    def sorted_files = files_list.sort { it.name }
                    [sorted_meta, sorted_files].transpose().collect { meta, patch_h5_file ->
                        tuple(meta, model_type, patch_h5_file)
                    }
                }
        }


    emit:
        patches_h5 = ch_patches.h5
        pt = ch_pt_out
        h5 = ch_h5_out
        patch_h5 = ch_patch_h5_out
}


workflow EXTRACT_FEATURES_ONE_STEP {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        // Just pass model type strings - processes will look up paths from params
        ch_model_types = Channel.fromList(params.featurize.model_types)

        // Batch slides together for one-step tessellate + featurize (workflow batching)
        ch_slide_batches = ch_samples
            .collate(params.featurize.workflow_batch_size ?: params.featurize.slide_batch_size ?: 8)
            .map { batch ->
                def slides = batch.collect { meta, slide -> slide }
                tuple(batch, slides)
            }

        TESSELLATE_FEATURIZE_BATCH(
            ch_slide_batches,
            ch_model_types
        )

        // Flatten batch results back to individual slides
        // Use indexed matching since output files may use staged names
        ch_pt_out = TESSELLATE_FEATURIZE_BATCH.out.pt
            .flatMap { batch_meta, model_type, pt_files ->
                def files_list = pt_files instanceof List ? pt_files : [pt_files]
                def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                def sorted_files = files_list.sort { it.name }
                [sorted_meta, sorted_files].transpose().collect { meta, pt_file ->
                    tuple(meta, model_type, pt_file)
                }
            }

        ch_h5_out = TESSELLATE_FEATURIZE_BATCH.out.h5
            .flatMap { batch_meta, model_type, h5_files ->
                def files_list = h5_files instanceof List ? h5_files : [h5_files]
                def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                def sorted_files = files_list.sort { it.name }
                [sorted_meta, sorted_files].transpose().collect { meta, h5_file ->
                    tuple(meta, model_type, h5_file)
                }
            }

        ch_patches_out = TESSELLATE_FEATURIZE_BATCH.out.tile_h5
            .flatMap { batch_meta, patch_h5_files ->
                def files_list = patch_h5_files instanceof List ? patch_h5_files : [patch_h5_files]
                def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                def sorted_files = files_list.sort { it.name }
                [sorted_meta, sorted_files].transpose().collect { meta, patch_h5_file ->
                    tuple(meta, patch_h5_file)
                }
            }

        ch_patch_h5_out = TESSELLATE_FEATURIZE_BATCH.out.patch_h5
            .flatMap { batch_meta, model_type, patch_h5_files ->
                def files_list = patch_h5_files instanceof List ? patch_h5_files : [patch_h5_files]
                def sorted_meta  = batch_meta.toSorted { it.slide_id.toString() }
                def sorted_files = files_list.sort { it.name }
                [sorted_meta, sorted_files].transpose().collect { meta, patch_h5_file ->
                    tuple(meta, model_type, patch_h5_file)
                }
            }

        EMIT_MPP_META(ch_patches_out)

    emit:
        patches_h5 = ch_patches_out
        pt = ch_pt_out
        h5 = ch_h5_out
        patch_h5 = ch_patch_h5_out
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

        // ── Multi-slide sample aggregation ───────────────────────────────────────
        // When multiple slides share the same sample_id, aggregate their per-slide
        // feature H5 files (already computed above) into one per-sample output.
        // Groups by (sample_id, model_type) so each invocation handles one model.
        ch_sample_feat_h5 = ch_extract_feat.h5
            .filter { meta, model_type, h5 -> (meta.n_slides ?: 1) > 1 }
            .map { meta, model_type, h5 ->
                tuple(groupKey([sample_id: meta.sample_id, model_type: model_type], meta.n_slides), meta, h5)
            }
            .groupTuple(remainder: true)
            .map { key, metas, h5s ->
                tuple(key.sample_id, metas, key.model_type, h5s)
            }

        MERGE_SAMPLE_FEATURES(ch_sample_feat_h5)
        // ─────────────────────────────────────────────────────────────────────────

        // ── Precision benchmarking ────────────────────────────────────────────
        // When benchmark_precisions is set, cast float32 features to each listed
        // precision for side-by-side linear probe comparison (no GPU needed).

        // Build patch-level h5 channel for linear probe:
        // - Patch encoders: *.features.h5 already has coords+features
        // - Slide encoders: *.patch_features.h5 (emitted as patch_h5) has coords+features
        //   with model_type = patch encoder name (e.g. conch1_5 for titan_slide)
        ch_lp_h5 = ch_extract_feat.h5
            .filter { meta, model_type, h5 ->
                def sm = params.featurize.slide_to_patch_mapping
                !(sm && sm.containsKey(model_type))
            }
            .mix(ch_extract_feat.patch_h5)

        if (params.featurize.benchmark_precisions) {
            ch_benchmark_precisions = Channel.fromList(params.featurize.benchmark_precisions)
            CONVERT_FEATURES_PRECISION(
                ch_lp_h5.combine(ch_benchmark_precisions)
                    .filter { meta, model_type, h5, target_precision ->
                        // Skip if target equals the source precision (no-op conversion)
                        def source_precision = resolvePrecision(model_type)
                        target_precision != source_precision
                    }
            )
            def ch_all_h5 = ch_lp_h5.mix(CONVERT_FEATURES_PRECISION.out.h5)
            LINEAR_PROBE(ch_annotations, ch_all_h5)
            ABMIL_BENCHMARK_WORKFLOW(ch_all_h5)
        } else {
            LINEAR_PROBE(ch_annotations, ch_lp_h5)
            ABMIL_BENCHMARK_WORKFLOW(ch_lp_h5)
        }
        // ─────────────────────────────────────────────────────────────────────

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

        // ── WebDataset sharding (opt-in) ──────────────────────────────────────
        ch_wds_shards = Channel.empty()
        if (params.wds.enabled) {
            // Determine group key per slide: oncotree_code or the fixed string "all"
            ch_pt_keyed = ch_extract_feat.pt.map { meta, model_type, pt_file ->
                def group = (params.wds.group_by_oncotree && meta.oncotree_code)
                    ? meta.oncotree_code
                    : "all"
                tuple(group, model_type, meta.slide_id, pt_file)
            }

            if (params.wds.shard_h5) {
                // Join pt and h5 channels by (group, model_type, slide_id)
                ch_h5_keyed = ch_extract_feat.h5.map { meta, model_type, h5_file ->
                    def group = (params.wds.group_by_oncotree && meta.oncotree_code)
                        ? meta.oncotree_code
                        : "all"
                    tuple(group, model_type, meta.slide_id, h5_file)
                }

                // Collect per (group, model_type) — wait for all slides in each group
                ch_wds_input = ch_pt_keyed
                    .join(ch_h5_keyed, by: [0, 1, 2])         // key: [group, model_type, slide_id]
                    .groupTuple(by: [0, 1])                    // group by (group, model_type)
                    .map { group, model_type, slide_ids, pt_files, h5_files ->
                        tuple(group, model_type, slide_ids, pt_files, h5_files)
                    }
            } else {
                // pt only — collect per (group, model_type)
                ch_wds_input = ch_pt_keyed
                    .groupTuple(by: [0, 1])                    // group by (group, model_type)
                    .map { group, model_type, slide_ids, pt_files ->
                        // Pass an empty list for h5_files so WDS_SHARD input tuple is consistent
                        tuple(group, model_type, slide_ids, pt_files, [])
                    }
            }

            WDS_SHARD(ch_wds_input)
            ch_wds_shards = WDS_SHARD.out.shards
        }

    emit:
        pt         = ch_extract_feat.pt
        h5         = ch_extract_feat.h5
        wds_shards = ch_wds_shards
}

