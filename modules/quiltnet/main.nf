params.quiltnet_default_classes = [
        "carcinoma in situ",
        "invasive carcinoma",
        "collagenous stroma",
        "adipose",
        "vessel",
        "necrosis",
        "invasive adenocarcinoma",
        "sarcoma"]


process CREATE_CLASS_EMBEDDINGS {
    publishDir "$params.outdir/class_embeddings"

    input:
    val oncotree_code
    val class_map

    output:
    tuple val(oncotree_code), path("${oncotree_code}.class_embedding.pt")

    script:
    classes = class_map[oncotree_code] ?: params.quiltnet_default_classes
    """
    create_class_embeddings \
        quiltnet_model_path=${params.quiltnet_model_path} \
        output_pt_path=${oncotree_code}.class_embedding.pt \
        classes="${classes}"
    """
}

process ANNOTATE {
    publishDir "$params.outdir/annotate"

    input:
    tuple val(slide_id), val(model_type), val(features_pt), val(oncotree_code), path(class_embedding)
    val class_map

    output:
    tuple val(slide_id), val(oncotree_code), path("${slide_id}.annotation.csv")

    script:
    classes = class_map[oncotree_code] ?: params.quiltnet_default_classes
    """
    annotate \
        features_pt_path=$features_pt \
        classes="${classes}" \
        class_embedding_pt_path=$class_embedding \
        output_csv_path=${slide_id}.annotation.csv
    """
}

process CACHE_TILES {
    publishDir "$params.outdir/cache_tiles"

    input:
    tuple val(slide_id), val(oncotree_code), path(annotation_csv), path(slide), val(model_type), path(patch_h5)
    val class_map

    output:
    tuple val(slide_id), val(oncotree_code), path("${slide_id}.indices.json"), path("${slide_id}.cache.pt")

    script:
    classes = class_map[oncotree_code] ?: params.quiltnet_default_classes
    """
    cache_tiles \
        patch_h5_path=${patch_h5} \
        slide_path=${slide} \
        annotation_csv_path=${annotation_csv} \
        output_indices_json_path=${slide_id}.indices.json \
        output_pt_path=${slide_id}.cache.pt \
        num_workers=${task.cpus} \
        limit_to_class="${classes}"
    """
}

params.skip_tile_caching = true

workflow QUILTNET {
    take:
        ch_oncotree_slide // oncotree_code, slide_id
        ch_slides // slide_id, slide
        ch_features // slide_id, features_pt
        ch_patches // slide_id, patch_pt

    main:
        def oncotree_class_map = [:].withDefault {[]}
        if (params.oncotree_class_csv) {
            new File(params.oncotree_class_csv).readLines().eachWithIndex { row, row_index ->
                if (row_index != 0) {
                    def cells = row.split(',')
                    if (cells.size() == 2) {
                        oncotree_class_map[cells[0]].add(cells[1])
                    }
                }
            }
        }

        ch_oncotree_codes = ch_oncotree_slide.map {it[0]}.unique()
        ch_class_embeddings = CREATE_CLASS_EMBEDDINGS(ch_oncotree_codes, oncotree_class_map)

        ch_slide_oncotree = ch_oncotree_slide.combine(ch_class_embeddings, by: 0).map {[it[1], it[0], it[2]]} // slide_id first
        ch_annotations = ANNOTATE(ch_features.join(ch_slide_oncotree), oncotree_class_map)
        ch_tile_cache = params.skip_tile_caching ? Channel.empty() : CACHE_TILES(ch_annotations.join(ch_slides).join(ch_patches), oncotree_class_map)

    emit:
        annotations = ANNOTATE.out
        cached_tiles = ch_tile_cache
}

