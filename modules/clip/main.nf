process CREATE_CLASS_EMBEDDINGS {
    label "cpuTask"

    input:
    val oncotree_code
    val class_map
    each model_type

    output:
    tuple val(oncotree_code), val(model_type), path("${oncotree_code}.${model_type}.class_embedding.pt")

    script:
    classes = class_map[oncotree_code] ?: params.clip.default_classes
    """
    create_class_embeddings \
        model_path=${params.featurize[model_type].model_path} \
        output_pt_path=${oncotree_code}.${model_type}.class_embedding.pt \
        classes="${classes}"
    """
}

process ANNOTATE {
    label "cpuTask"

    publishDir path: "${params.outdir}/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), path(features_pt), val(oncotree_code), path(class_embedding)
    val class_map

    output:
    tuple val(meta), val(model_type), val(oncotree_code), path("${meta.slide_id}.annotation.csv"), emit: csv
    tuple val(meta), val("${model_type}_annotation_csv_urlpath"), val("${publish_path}/${meta.slide_id}.annotation.csv"), topic: meta_out

    script:
    publish_path = "annotate/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    classes = class_map[oncotree_code] ?: params.clip.default_classes
    """
    annotate \
        features_pt_path=$features_pt \
        classes="${classes}" \
        class_embedding_pt_path=$class_embedding \
        output_csv_path=${meta.slide_id}.annotation.csv
    """
}

process CACHE_TILES {
    label "cpuTask"
    label "parallelTask"

    publishDir path: "${params.outdir}/cache_tiles/${publish_path}", mode: "${params.publish_mode}"

    input:
    tuple val(meta), val(model_type), val(oncotree_code), path(annotation_csv), path(slide), path(patch_h5)
    val class_map

    output:
    path "${meta.slide_id}.indices.json"
    path "${meta.slide_id}.cache.pt"
    tuple val(meta), val("${model_type}_tile_cache_indices_json_urlpath"), val("${publish_path}/${meta.slide_id}.indices.json"), topic: meta_out
    tuple val(meta), val("${model_type}_tile_cache_tensor_urlpath"), val("${publish_path}/${meta.slide_id}.cache.pt"), topic: meta_out

    script:
    publish_path = "cache_tiles/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}"
    classes = class_map[oncotree_code] ?: params.clip.default_classes
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
}


workflow CLIP {
    take:
        ch_samples // meta, slide
        ch_features // meta, features_pt
        ch_patches // meta, patch_pt

    main:
        ch_oncotree_slide = ch_samples.map {
            meta, slide ->
                oncotree_code = "default"
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

        ch_oncotree_codes = ch_samples.map { meta, slide -> meta.oncotree_code }.unique()
        ch_class_embeddings = CREATE_CLASS_EMBEDDINGS(
            ch_oncotree_codes,
            oncotree_class_map,
            Channel.fromList(params.clip.model_types))

        ch_slide_oncotree = ch_oncotree_slide.combine(ch_class_embeddings, by: 0).map {[it[1], it[2], it[0], it[3]]} // slide_id and model type first
        ch_annotations = ANNOTATE(ch_features.join(ch_slide_oncotree, by: [0,1]), oncotree_class_map).csv

        ch_tile_cache = params.clip.skip_tile_caching ? Channel.empty() : CACHE_TILES(ch_annotations.join(ch_slides).join(ch_patches), oncotree_class_map)

    emit:
        annotations = ch_annotations
        cached_tiles = ch_tile_cache
}

