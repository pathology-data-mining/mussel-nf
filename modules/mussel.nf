include { CLIP } from './clip'
include { LINEAR_PROBE } from './linear_probe'
include { ABMIL_BENCHMARK_WORKFLOW } from './abmil_benchmark/main'

include { FEATURIZE_BATCH as FEATURIZE_BATCH_PATCH; FEATURIZE_BATCH as FEATURIZE_BATCH_SLIDE; FEATURIZE_BATCH as FILTER_FEATURIZE; FEATURIZE_PATCH_MULTI } from './featurize'

include { TESSELLATE; TESSELLATE_BATCH; DERIVE_TILE_GRID; EMIT_TESSELLATE_PATH_META; FILTER_TILES; EMIT_MPP_META; tileSetOf; tileSetByTag; tilePublishPath } from './tessellation'

include { TESSELLATE_FEATURIZE_BATCH } from './tessellate_featurize'

include { CONVERT_FEATURES_PRECISION } from './convert_precision'

include { WDS_SHARD } from './wds'
include { MERGE_SAMPLE_FEATURES } from './sample_merge'
include { resolvePrecision } from './utils'


// ── Tile sets ─────────────────────────────────────────────────────────────────
// Every model is tiled at its recommended size and resolution unless explicitly
// overridden (see lib/TileSizes.groovy). Models that resolve to the same size and
// resolution share a "tile set"; all tile sets come from one tissue segmentation
// per slide. A tile set is [size: px, mpp: µm/px, key: channel key,
// dir: publish subdir, suffix: manifest key suffix] (TileSizes.tileSets).

def resolveTileSets() {
    def model_specs = TileSizes.resolve(params, projectDir, log)
    def tile_sets = TileSizes.tileSets(model_specs, TileSizes.derivedModels(params))
    def summary = tile_sets.collect { ts ->
        def models = model_specs.findAll { m, spec -> TileSizes.tileSetFor(tile_sets, spec).key == ts.key }.keySet()
        "${ts.size} px at ${ts.mpp} µm/px${ts.derived ? ' (derived from the primary tiles)' : ''} -> ${ts.dir}/ (${models.join(', ')})"
    }
    log.info "Tile sets: ${summary.join('; ')}"
    return [model_specs, tile_sets]
}

// Pair each slide batch (tile set key, batch, slides, patch_h5s) with the models
// that use that tile set -> FEATURIZE_BATCH inputs.
// Each job carries the model's Mussel tile_size_check mode (TileSizes.checkMode).
def featurizeJobs(ch_slide_batches, models, model_specs) {
    return ch_slide_batches
        .combine(Channel.fromList(models.collect { m -> tuple(TileSizes.key(model_specs[m]), m) }), by: 0)
        .map { key, batch, slides, patch_h5s, model ->
            tuple(batch, slides, patch_h5s, model, TileSizes.checkMode(params, projectDir, model, model_specs[model]))
        }
}

// Patch models to run together in FEATURIZE_PATCH_MULTI: with
// featurize.multi_model_patch_pass, every tile set used by two or more plain patch
// models (no slide encoder, not CLIP). Returns [tile set key: [models]].
def multiModelGroups(patch_models, model_specs) {
    if (!params.featurize.multi_model_patch_pass) {
        return [:]
    }
    def eligible = patch_models.findAll { !(it in (params.clip.model_types ?: [])) }
    return eligible.groupBy { TileSizes.key(model_specs[it]) }.findAll { key, models -> models.size() > 1 }
}

// Mussel applies one tile_size_check to every model in a multi-model pass: use the
// most lenient of the group's modes (off < warn < error; null if the check is unset),
// and say so when that relaxes a model's own check.
def groupCheckMode(models, model_specs) {
    def modes = models.collect { m -> TileSizes.checkMode(params, projectDir, m, model_specs[m]) }
    def mode = ['off', 'warn', 'error'].find { it in modes }
    if (modes.unique(false).size() > 1) {
        log.warn "featurize.multi_model_patch_pass: ${models} share one task, so their tile size check is " +
                 "'${mode ?: 'unset'}' (per model: ${[models, modes].transpose().collect { m, c -> "${m}=${c ?: 'unset'}" }.join(', ')})"
    }
    return mode
}

// FEATURIZE_PATCH_MULTI emits all models' files together; split them into the
// per-model (batch_meta, model, files) tuples FEATURIZE_BATCH emits.
def perModel(ch) {
    return ch.flatMap { batch_meta, files ->
        def files_list = files instanceof List ? files : [files]
        files_list.groupBy { it.parent.name }.collect { model, fs -> tuple(batch_meta, model, fs) }
    }
}

// Slide ID in a Mussel output file name: <slide_id>.features.{pt,h5},
// <slide_id>.patch_features.{pt,h5} or <slide_id>.patch.h5; null for any other name.
def slideIdOf(String name) {
    def m = (name =~ /^(.+)\.(features\.pt|features\.h5|patch_features\.pt|patch_features\.h5|patch\.h5)$/)
    return m ? m[0][1] : null
}

// Pair a batch task's output files with their slides' metas by the slide ID in each
// file name, never by position: a batch task writes nothing for a slide that failed
// or had no tiles (Mussel >= 1.6.0 goes on with the others). A file naming no slide
// of the batch is dropped with a warning. Returns [[meta, file], ...].
def bySlide(batch_meta, files, String what) {
    def files_list = files instanceof List ? files : [files]
    def meta_by_slide = batch_meta.collectEntries { meta -> [meta.slide_id.toString(), meta] }
    def pairs = []
    files_list.each { f ->
        def meta = meta_by_slide[slideIdOf(f.name.toString())]
        if (meta == null) {
            log.warn "${what}: ${f.name} is not an output of a slide in its batch (${meta_by_slide.keySet().join(', ')}); dropped"
        } else {
            pairs << [meta, f]
        }
    }
    def missing = meta_by_slide.keySet() - pairs.collect { meta, f -> meta.slide_id.toString() }
    if (missing) {
        log.info "${what}: no output for ${missing.join(', ')} (failed or without tiles; see the task's failed_slides.tsv)"
    }
    return pairs
}

// Split batch task outputs (one task per batch x model) back into per-slide tuples.
def perSlide(ch) {
    return ch.flatMap { batch_meta, model_type, files ->
        bySlide(batch_meta, files, model_type.toString()).collect { meta, f -> tuple(meta, model_type, f) }
    }
}


workflow TESSELLATE_PATCHES {
    take:
        ch_samples // tuple val(meta), file(slide)
        tile_sets  // list of tile sets (resolveTileSets), primary first

    main:
        // One tessellation task per slide (or batch) writes every tessellated tile set
        // from a single tissue segmentation; outputs are split back into per-set tuples.
        // Derived sets (featurize.grid_from_primary) are built from the primary tiles
        // afterwards (DERIVE_TILE_GRID).
        def tess_sets = tile_sets.findAll { !it.derived }
        def derived_sets = tile_sets.findAll { it.derived }
        tessellation_batch_size = params.tiling.workflow_batch_size ?: 1
        if (tessellation_batch_size > 1) {
            ch_tessellation_batches = ch_samples
                .collate(tessellation_batch_size)
                .map { batch ->
                    def metas = batch.collect { meta, slide -> meta }
                    def slides = batch.collect { meta, slide -> slide }
                    tuple(metas, slides, tess_sets)
                }

            TESSELLATE_BATCH(ch_tessellation_batches)

            ch_patches_h5 = TESSELLATE_BATCH.out.h5
                .flatMap { batch_meta, patch_h5_files ->
                    def files_list = patch_h5_files instanceof List ? patch_h5_files : [patch_h5_files]
                    def meta_by_slide = batch_meta.collectEntries { meta -> [meta.slide_id.toString(), meta] }
                    files_list.collect { patch_h5_file ->
                        def (slide_id, tset) = tileSetOf(patch_h5_file.name, tile_sets)
                        tuple(meta_by_slide[slide_id], tset, patch_h5_file)
                    }.findAll { meta, tset, patch_h5_file -> meta != null && tset != null }
                }
        } else {
            TESSELLATE(ch_samples.map { meta, slide -> tuple(meta, slide, tess_sets) })
            ch_patches_h5 = TESSELLATE.out.h5
                .flatMap { meta, patch_h5_files ->
                    def files_list = patch_h5_files instanceof List ? patch_h5_files : [patch_h5_files]
                    files_list.collect { patch_h5_file -> tuple(meta, tileSetOf(patch_h5_file.name, tile_sets)[1], patch_h5_file) }
                }
            // Per-size grid masks and PNG tiles: manifest entries under each tile set's dir.
            ch_sized_meta = TESSELLATE.out.grid_mask
                .flatMap { meta, files -> (files instanceof List ? files : [files]).collect { f ->
                    def tset = tileSetOf(f.name.replaceFirst(/\.grid_mask\.png$/, '.patch.h5'), tile_sets)[1]
                    tuple(meta, "grid_mask_path${tset.suffix}", tilePublishPath(f.name, tile_sets))
                } }
                .mix(TESSELLATE.out.png.flatMap { meta, files ->
                    (files instanceof List ? files : [files]).collect { f -> f.parent.name }.unique().collect { d ->
                        def m = (d =~ /_(\d+)px(?:_([0-9.]+)mpp)?_png$/)
                        def tset = m ? (tileSetByTag(tile_sets, m[0][1], m[0][2]) ?: tile_sets[0]) : tile_sets[0]
                        tuple(meta, "tile_png_path${tset.suffix}", tilePublishPath(d, tile_sets))
                    }
                })
        }

        if (derived_sets) {
            // Batches of 64 slides' primary tile files (derivation is cheap; batching saves
            // task start-up), each paired with every derived tile set.
            ch_derive_batches = ch_patches_h5
                .filter { meta, tset, h5 -> tset.suffix == '' }
                .collate(64, true)
                .map { batch -> tuple(batch.collect { it[0] }, batch.collect { it[2] }) }
                .combine(Channel.fromList(derived_sets))
                .map { metas, h5s, tset -> tuple(metas, h5s, tset, tile_sets) }
            DERIVE_TILE_GRID(ch_derive_batches)
            ch_patches_h5 = ch_patches_h5.mix(DERIVE_TILE_GRID.out.h5.flatMap { batch_meta, tset, files ->
                def meta_by_slide = batch_meta.collectEntries { meta -> [meta.slide_id.toString(), meta] }
                (files instanceof List ? files : [files]).collect { f -> tuple(meta_by_slide[tileSetOf(f.name, tile_sets)[0]], tset, f) }
            })
        }

        // Manifest: one tiles_h5_path[_<size>px] entry per slide and tile set.
        ch_path_meta = ch_patches_h5.map { meta, tset, h5 ->
            tuple(meta, "tiles_h5_path${tset.suffix}", "${tset.dir}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.patch.h5")
        }
        if (tessellation_batch_size <= 1) {
            ch_path_meta = ch_path_meta.mix(ch_sized_meta)
        }
        EMIT_TESSELLATE_PATH_META(ch_path_meta)

        // Native MPP is a property of the slide; record it once (primary tile set).
        EMIT_MPP_META(ch_patches_h5.filter { meta, tset, h5 -> tset.suffix == '' }.map { meta, tset, h5 -> tuple(meta, h5) })

    emit:
        patches_h5 = ch_patches_h5 // tuple val(meta), val(tset), path(patch_h5)
}


workflow EXTRACT_FEATURES {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        def resolved = resolveTileSets()
        def model_specs = resolved[0]
        def tile_sets = resolved[1]
        slide_encoder_mapping = params.featurize.slide_to_patch_mapping ?: [:]
        slide_model_types = params.featurize.model_types.findAll { slide_encoder_mapping.containsKey(it) }
        patch_model_types = params.featurize.model_types.findAll { !slide_encoder_mapping.containsKey(it) }
        featurize_batch_size = params.featurize.workflow_batch_size ?: params.featurize.slide_batch_size ?: 8

        ch_tessellation = TESSELLATE_PATCHES(ch_samples, tile_sets)
        ch_patches_h5 = ch_tessellation.patches_h5

        if (params.tiling.filter_tiles) {
            if (tile_sets.size() > 1) {
                error "tiling.filter_tiles supports a single tile set, but the requested models need " +
                      "${model_specs}. Set featurize.model_patch_sizes / model_mpps or tiling.patch_size / mpp to use one."
            }
            def tset = tile_sets[0]
            // Extract features with the filter model (e.g. ctranspath), then
            // FILTER_TILES applies classifier_pkl to keep tissue tiles.
            ch_filter_slide_batches = ch_samples.combine(ch_patches_h5.map { meta, ts, h5 -> tuple(meta, h5) }, by: 0)
                .collate(featurize_batch_size)
                .map { batch ->
                    def slides = batch.collect { meta, slide, patch_h5 -> slide }
                    def patch_h5s = batch.collect { meta, slide, patch_h5 -> patch_h5 }
                    tuple(batch, slides, patch_h5s, params.tiling.filter_model_type,
                          TileSizes.checkMode(params, projectDir, params.tiling.filter_model_type.toString(), tset))
                }

            FILTER_FEATURIZE(ch_filter_slide_batches, false)

            ch_filter_patches = FILTER_TILES(perSlide(FILTER_FEATURIZE.out.h5))
            ch_feature_tiles = ch_filter_patches.h5.map { meta, h5 -> tuple(meta, tset, h5) }
        } else {
            ch_feature_tiles = ch_patches_h5
        }

        // Batch slides per tile set (workflow batching), then pair each batch
        // only with the models that use that tile set.
        ch_slide_batches = ch_feature_tiles
            .combine(ch_samples, by: 0)
            .map { meta, tset, patch_h5, slide -> tuple(tset.key, [meta, slide, patch_h5]) }
            .groupTuple(by: 0, size: featurize_batch_size, remainder: true)
            .map { key, batch ->
                tuple(key, batch, batch.collect { meta, slide, patch_h5 -> slide }, batch.collect { meta, slide, patch_h5 -> patch_h5 })
            }

        ch_patch_pt_out = Channel.empty()
        ch_patch_h5_feature_out = Channel.empty()
        def multi_groups = multiModelGroups(patch_model_types, model_specs)
        def single_patch_models = patch_model_types - multi_groups.values().flatten()
        if (multi_groups) {
            def ch_multi_jobs = ch_slide_batches
                .combine(Channel.fromList(multi_groups.collect { key, models -> tuple(key, models) }), by: 0)
                .map { key, batch, slides, patch_h5s, models ->
                    tuple(batch, slides, patch_h5s, models, groupCheckMode(models, model_specs))
                }
            FEATURIZE_PATCH_MULTI(ch_multi_jobs)
            ch_patch_pt_out = ch_patch_pt_out.mix(perSlide(perModel(FEATURIZE_PATCH_MULTI.out.pt)))
            ch_patch_h5_feature_out = ch_patch_h5_feature_out.mix(perSlide(perModel(FEATURIZE_PATCH_MULTI.out.h5)))
        }
        if (single_patch_models) {
            FEATURIZE_BATCH_PATCH(featurizeJobs(ch_slide_batches, single_patch_models, model_specs), true)
            ch_patch_pt_out = ch_patch_pt_out.mix(perSlide(FEATURIZE_BATCH_PATCH.out.pt))
            ch_patch_h5_feature_out = ch_patch_h5_feature_out.mix(perSlide(FEATURIZE_BATCH_PATCH.out.h5))
        }

        ch_slide_pt_out = Channel.empty()
        ch_slide_h5_feature_out = Channel.empty()
        ch_slide_patch_h5_out = Channel.empty()
        if (slide_model_types) {
            FEATURIZE_BATCH_SLIDE(featurizeJobs(ch_slide_batches, slide_model_types, model_specs), true)
            ch_slide_pt_out = perSlide(FEATURIZE_BATCH_SLIDE.out.pt)
            ch_slide_h5_feature_out = perSlide(FEATURIZE_BATCH_SLIDE.out.h5)
            ch_slide_patch_h5_out = perSlide(FEATURIZE_BATCH_SLIDE.out.patch_h5)
        }

        ch_pt_out = ch_patch_pt_out.mix(ch_slide_pt_out)
        ch_h5_out = ch_patch_h5_feature_out.mix(ch_slide_h5_feature_out)
        ch_patch_h5_out = ch_slide_patch_h5_out

    emit:
        // Primary tile set only, for consumers that expect one tiling per slide (CLIP).
        patches_h5 = ch_patches_h5.filter { meta, tset, h5 -> tset.suffix == '' }.map { meta, tset, h5 -> tuple(meta, h5) }
        pt = ch_pt_out
        h5 = ch_h5_out
        patch_h5 = ch_patch_h5_out
}


workflow TESSELLATE_ONLY {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        def resolved = resolveTileSets()
        def model_specs = resolved[0]
        def tile_sets = resolved[1]
        ch_tessellation = TESSELLATE_PATCHES(ch_samples, tile_sets)

    emit:
        patches_h5 = ch_tessellation.patches_h5.filter { meta, tset, h5 -> tset.suffix == '' }.map { meta, tset, h5 -> tuple(meta, h5) }
        pt = Channel.empty()
        h5 = Channel.empty()
        patch_h5 = Channel.empty()
}


workflow EXTRACT_FEATURES_ONE_STEP {
    take:
        ch_samples // tuple val(meta), file(slide)

    main:
        // Each model runs with the tile set for its resolved tile size and resolution.
        def resolved = resolveTileSets()
        def model_specs = resolved[0]
        def tile_sets = resolved[1]
        if (tile_sets.any { it.derived }) {
            error "featurize.grid_from_primary needs the two-step workflow (use_one_step_workflow = false): " +
                  "the one-step workflow tessellates each model separately."
        }
        def one_step_conflicts = TileSizes.oneStepConflicts(model_specs, (params.featurize.slide_to_patch_mapping ?: [:]).keySet(), projectDir)
        if (one_step_conflicts) {
            error "The one-step workflow can't tile ${one_step_conflicts.join('; ')}: each slide encoder runs in " +
                  "Mussel's multi-model tessellate_extract_features, which would silently use other tiles. " +
                  "Use the two-step workflow (use_one_step_workflow = false) for this override, or remove it " +
                  "(featurize.model_patch_sizes / model_mpps, tiling.patch_size / mpp)."
        }
        ch_model_types = Channel.fromList(params.featurize.model_types.collect { m ->
            tuple(m, TileSizes.tileSetFor(tile_sets, model_specs[m]))
        })

        // Batch slides together for one-step tessellate + featurize (workflow batching)
        ch_slide_batches = ch_samples
            .collate(params.featurize.workflow_batch_size ?: params.featurize.slide_batch_size ?: 8)
            .map { batch ->
                def slides = batch.collect { meta, slide -> slide }
                tuple(batch, slides)
            }

        TESSELLATE_FEATURIZE_BATCH(ch_slide_batches.combine(ch_model_types))

        // Split batch results back into per-slide tuples, by the slide ID in each file name.
        ch_pt_out = perSlide(TESSELLATE_FEATURIZE_BATCH.out.pt)
        ch_h5_out = perSlide(TESSELLATE_FEATURIZE_BATCH.out.h5)

        // Every model task re-emits its tiles; keep the primary tile set only.
        ch_patches_out = TESSELLATE_FEATURIZE_BATCH.out.tile_h5
            .filter { batch_meta, tset, patch_h5_files -> tset.suffix == '' }
            .flatMap { batch_meta, tset, patch_h5_files ->
                bySlide(batch_meta, patch_h5_files, "tiles").collect { meta, patch_h5_file -> tuple(meta, patch_h5_file) }
            }

        ch_patch_h5_out = perSlide(TESSELLATE_FEATURIZE_BATCH.out.patch_h5)

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

        if (params.featurize.merge_sample_features) {
            // ── Multi-slide sample aggregation ───────────────────────────────────
            // When multiple slides share the same sample_id, aggregate their
            // per-slide feature H5 files into one per-sample output.
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
            // ─────────────────────────────────────────────────────────────────────
        }

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
            // CLIP annotation reads the primary tile set (patches_h5), so its
            // models must use that tile set.
            def model_specs = TileSizes.resolve(params, projectDir)
            def primary = TileSizes.tileSets(model_specs, TileSizes.derivedModels(params))[0]
            def off_primary = params.clip.model_types.findAll { model_specs.containsKey(it) && TileSizes.key(model_specs[it]) != primary.key }
            if (off_primary) {
                error "CLIP models ${off_primary} use a tile set other than the primary ${primary.size} px at ${primary.mpp} µm/px " +
                      "(first entry of featurize.model_types); list a CLIP model first or set featurize.model_patch_sizes / model_mpps."
            }
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
