process CREATE_CLASS_EMBEDDINGS {
    label "cpuTask"

    input:
    val oncotree_code
    val class_map
    each model_config // tuple of [model_type, model_path]

    output:
    tuple val(oncotree_code), val(model_type), path("${oncotree_code}.${model_type}.class_embedding.pt")

    script:
    model_type = model_config[0]
    model_path = model_config[1]
    
    classes = class_map[oncotree_code] ?: params.clip.default_classes
    mpath_str = ""
    if (model_path)
        mpath_str = "model_path=${model_path}"
    """
    create_class_embeddings ${mpath_str} \
        output_pt_path=${oncotree_code}.${model_type}.class_embedding.pt \
        classes="${classes}"
    """

    stub:
    model_type = model_config[0]
    """
    #!/usr/bin/env python3
    import torch
    torch.save(torch.zeros(1, 8), "${oncotree_code}.${model_type}.class_embedding.pt")
    """
}

process ANNOTATE {
    label "cpuTask"

    publishDir path: { "${params.outdir}/annotate/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}" }, mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_pt), val(oncotree_code), path(class_embedding)
    val class_map

    output:
    tuple val(meta), val(model_type), val(oncotree_code), path("${meta.slide_id}.annotation.csv"), emit: csv
    tuple val(meta), val("${model_type}_annotation_csv_path"), val("annotate/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.annotation.csv"), topic: meta_out

    script:
    def classes = class_map[oncotree_code] ?: params.clip.default_classes
    """
    annotate \
        features_pt_path=$features_pt \
        classes="${classes}" \
        class_embedding_pt_path=$class_embedding \
        output_csv_path=${meta.slide_id}.annotation.csv
    """

    stub:
    """
    echo "class,score" > ${meta.slide_id}.annotation.csv
    echo "stub,1.0"   >> ${meta.slide_id}.annotation.csv
    """
}

process CACHE_TILES {
    label "cpuTask"
    label "parallelTask"

    publishDir path: { "${params.outdir}/cache_tiles/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}" }, mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), val(oncotree_code), path(annotation_csv), path(slide), path(patch_h5)
    val class_map

    output:
    path "${meta.slide_id}.indices.json"
    path "${meta.slide_id}.cache.pt"
    tuple val(meta), val("${model_type}_tile_cache_indices_json_path"), val("cache_tiles/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.indices.json"), topic: meta_out
    tuple val(meta), val("${model_type}_tile_cache_tensor_path"), val("cache_tiles/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.cache.pt"), topic: meta_out

    script:
    def classes = class_map[oncotree_code] ?: params.clip.default_classes
    """
    cache_tiles \
        patch_h5_path=${patch_h5} \
        slide_path=${slide} \
        annotation_csv_path=${annotation_csv} \
        output_indices_json_path=${meta.slide_id}.indices.json \
        output_pt_path=${meta.slide_id}.cache.pt \
        num_workers=${task.cpus} \
        limit_to_class="${classes}"
    """

    stub:
    """
    #!/usr/bin/env python3
    import json, torch
    with open("${meta.slide_id}.indices.json", "w") as f:
        json.dump({}, f)
    torch.save(torch.zeros(1, 8), "${meta.slide_id}.cache.pt")
    """
}


workflow CLIP {
    take:
        ch_samples // meta, slide
        ch_features // meta, features_pt
        ch_patches // meta, patch_pt

    main:
        ch_oncotree_slide = ch_samples.map {
            meta, slide ->
                def oncotree_code = "default"
                if ("oncotree_code" in meta) {
                    oncotree_code = meta["oncotree_code"]
                }
                [oncotree_code, meta.slide_id]
        }

        def oncotree_class_map = [:].withDefault {[]}
        if (params.clip.oncotree_class_csv) {
            new File(params.clip.oncotree_class_csv).readLines().eachWithIndex {
                row, row_index ->
                    if (row_index != 0) {
                        def cells = row.split(',')
                        if (cells.size() == 2) {
                            oncotree_class_map[cells[0]].add(cells[1])
                        }
                    }
            }
        }

        ch_oncotree_codes = ch_oncotree_slide.map { it[0] }.unique()
        
        // Create a channel with model types and their paths as tuples
        ch_clip_model_configs = Channel.fromList(params.clip.model_types).map { model_type ->
            def model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
            [model_type, model_path]
        }
        
        ch_class_embeddings = CREATE_CLASS_EMBEDDINGS(
            ch_oncotree_codes,
            oncotree_class_map,
            ch_clip_model_configs)

        ch_slide_oncotree = ch_oncotree_slide.combine(ch_class_embeddings, by: 0).map {[it[1], it[2], it[0], it[3]]} // slide_id and model type first
        // Join on slide_id (string) — remap ch_features to use slide_id as key before joining
        ch_annot_input = ch_features
            .map { meta, model_type, pt -> [meta.slide_id, model_type, meta, pt] }
            .join(ch_slide_oncotree, by: [0, 1])
            .map { slide_id, model_type, meta, pt, oncotree_code, class_embedding ->
                tuple(meta, model_type, pt, oncotree_code, class_embedding)
            }
        ch_annotations = ANNOTATE(ch_annot_input, oncotree_class_map).csv

        ch_tile_cache = params.clip.skip_tile_caching ? Channel.empty() : CACHE_TILES(ch_annotations.join(ch_samples).join(ch_patches), oncotree_class_map)

    emit:
        annotations = ch_annotations
        cached_tiles = ch_tile_cache
}

