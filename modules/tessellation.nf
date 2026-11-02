// All tile sets come from one tissue segmentation per slide: with several, Mussel's
// tessellate gets seg_config.patch_sizes (and seg_config.patch_mpps when their
// resolutions differ; Mussel >= #142) and writes <slide_id>.<size>px[_<mpp>mpp].patch.h5
// (grid masks / PNG tiles likewise per set); with one it runs exactly as before
// (seg_config.patch_size, <slide_id>.patch.h5). tilePublishPath routes a task file
// to its tile set's publish dir under the usual name.

// Mussel output-path placeholder naming a tile set ('' with a single set).
def sizeTag(tsets) {
    if (tsets.size() == 1) return ''
    return TileSizes.multiMpp(tsets) ? '.{patch_size}px_{mpp}mpp' : '.{patch_size}px'
}

// The patch.h5 name Mussel writes for one tile set (sizeTag expanded).
def tileFileName(slide_id, tsets, tset) {
    def tag = sizeTag(tsets).replace('{patch_size}', "${tset.size}").replace('{mpp}', TileSizes.mppTag(tset.mpp))
    return "${slide_id}${tag}.patch.h5".toString()
}

// The tile set named by a <size>px[_<mpp>mpp] tag.
def tileSetByTag(tsets, size, mpp) {
    return tsets.find { it.size == (size as Integer) && (mpp == null || TileSizes.sameMpp(it.mpp, mpp as Double)) }
}

def tilePublishPath(String filename, tsets) {
    def m = (filename =~ /^(.*?)([._])(\d+)px(?:_([0-9.]+)mpp)?([._].*)$/)
    def tset = m ? (tileSetByTag(tsets, m[0][3], m[0][4]) ?: tsets[0]) : tsets[0]
    def name = m ? m[0][1] + m[0][5] : filename
    def sid = name.tokenize('/')[0].replaceFirst(/(_png|\.patch\.h5|\.(grid_mask|mask|thumbnail)\.png)$/, '')
    def prefix = params.publish_slide_prefix ? "${sid.take(4)}/" : ''
    return "${tset.dir}/${prefix}${name}"
}

// Slide id and tile set of a tessellation output file: <id>.<size>px[_<mpp>mpp].patch.h5,
// or <id>.patch.h5 for the only tile set.
def tileSetOf(String filename, tsets) {
    def m = (filename =~ /^(.*)\.(\d+)px(?:_([0-9.]+)mpp)?\.patch\.h5$/)
    if (m) {
        return [m[0][1], tileSetByTag(tsets, m[0][2], m[0][3])]
    }
    return [filename.replaceFirst(/\.patch\.h5$/, ''), tsets[0]]
}

// tessellate output path argument: "@" becomes the tile-set placeholder (sizeTag,
// quoted for Hydra), or is dropped with a single tile set.
def sizedOutputArg(String key, slide_id, String rest, tsets) {
    def tag = sizeTag(tsets)
    return tag ? "'${key}=\"${slide_id}${rest.replace('@', tag)}\"'" : "${key}=${slide_id}${rest.replace('@', '')}"
}

// tessellate arguments for the tile sets: sizes, and resolutions (seg_config.mpp,
// or seg_config.patch_mpps per size when they differ).
def tessellateSizeArgs(tsets) {
    if (tsets.size() == 1) {
        return "seg_config.patch_size=${tsets[0].size} seg_config.mpp=${tsets[0].mpp}"
    }
    def args = ["'seg_config.patch_sizes=[${tsets.collect { it.size }.join(',')}]'", "seg_config.mpp=${tsets[0].mpp}"]
    if (TileSizes.multiMpp(tsets)) {
        args << "'seg_config.patch_mpps=[${tsets.collect { it.mpp }.join(',')}]'"
    }
    return args.join(' ')
}


process TESSELLATE {
    label "medTask"
    // Note: no cpuTask/gpuTask label here — GPU requirement is determined at
    // runtime based on params.tiling.seg_model.  When seg_model='neural' (the
    // default), tessellation uses a neural segmentation model (GrandQC) that
    // requires a GPU.  The executor profiles (condor, slurm, cloud) handle this
    // via withName:TESSELLATE directives in nextflow.config.

    scratch params.scratch_dir ?: false

    publishDir path: "${params.outdir}", mode: "${params.publish_mode}", saveAs: { fn -> tilePublishPath(fn, tsets) }

    input:
    tuple val(meta), path(slide), val(tsets) // tsets: all tile sets (TileSizes.tileSets), primary first

    output:
    tuple val(meta), path("*.patch.h5"), optional: true, emit: h5
    tuple val(meta), path("${meta.slide_id}*_png/*.png"), optional: true, emit: png
    path "${meta.slide_id}.*.png", optional: true, emit: thumbnail_png
    // Size-independent images (tissue mask, thumbnail) are recorded with the primary tile set.
    tuple val(meta), val("thumbnail_path"), val("${tsets[0].dir}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.thumbnail.png"), path("${meta.slide_id}.thumbnail.png"), optional: true, topic: slide_meta
    tuple val(meta), val("mask_path"), val("${tsets[0].dir}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.mask.png"), path("${meta.slide_id}.mask.png"), optional: true, topic: slide_meta
    tuple val(meta), path("${meta.slide_id}*.grid_mask.png"), optional: true, emit: grid_mask

    script:
    // Several tile sets: per-set outputs carry Mussel's {patch_size} (and {mpp}) placeholders.
    def tag = sizeTag(tsets)
    def save_tile_param = ""
    if (params.tiling.save_tile_png)
        save_tile_param = tag ? "'output_png_dir=\"${meta.slide_id}_${tag.substring(1)}_png\"'" : "output_png_dir=${meta.slide_id}_png"
    stitch_tile_param = ""
    if (params.tiling.stitch_tiles)
        stitch_tile_param = sizedOutputArg("output_grid_mask_path", meta.slide_id, "@.grid_mask.png", tsets)
        stitch_tile_param += " output_mask_path=${meta.slide_id}.mask.png"
    save_thumbnail_param = ""
    if (params.tiling.save_slide_thumbnail)
        save_thumbnail_param = "output_thumbnail_path=${meta.slide_id}.thumbnail.png"
    neural_seg_max_tiles = params.tiling.neural_seg_max_tiles != null
        ? params.tiling.neural_seg_max_tiles
        : 4096
    expected_seg_model = params.tiling.seg_model?.toString()?.trim()?.toLowerCase()

    // Use seg_config group if specified, otherwise use individual parameters
    seg_config_str = params.tiling.seg_config_group ? "seg_config=${params.tiling.seg_config_group}" : ""

    // Build individual parameter overrides (only if not using group, or to override group defaults)
    seg_params = []
    seg_params << tessellateSizeArgs(tsets)  // all tile sets, one segmentation (TileSizes)
    if (params.tiling.segment_threshold != null) seg_params << "seg_config.segment_threshold=${params.tiling.segment_threshold}"
    if (params.tiling.median_blur_ksize != null) seg_params << "seg_config.median_blur_ksize=${params.tiling.median_blur_ksize}"
    if (params.tiling.morphology_ex_kernel != null) seg_params << "seg_config.morphology_ex_kernel=${params.tiling.morphology_ex_kernel}"
    if (params.tiling.tissue_area_threshold != null) seg_params << "seg_config.tissue_area_threshold=${params.tiling.tissue_area_threshold}"
    if (params.tiling.hole_area_threshold != null) seg_params << "seg_config.hole_area_threshold=${params.tiling.hole_area_threshold}"
    if (params.tiling.max_num_holes != null) seg_params << "seg_config.max_num_holes=${params.tiling.max_num_holes}"
    if (params.tiling.overlap != null) seg_params << "seg_config.overlap=${params.tiling.overlap}"
    if (params.tiling.min_tissue_proportion != null) seg_params << "seg_config.min_tissue_proportion=${params.tiling.min_tissue_proportion}"
    if (params.tiling.remove_artifacts) seg_params << "seg_config.remove_artifacts=true"
    if (params.tiling.remove_penmarks) seg_params << "seg_config.remove_penmarks=true"
    if (params.tiling.seg_model != null) seg_params << "seg_config.seg_model=${params.tiling.seg_model}"
    if (params.tiling.slide_mpp_override != null) seg_params << "seg_config.slide_mpp_override=${params.tiling.slide_mpp_override}"
    seg_params_str = seg_params.join(' \\\n        ')

    """
    ${TaskProfile.script(params.profile_dir, task)}
    export MUSSEL_NEURAL_SEG_MAX_TILES=${neural_seg_max_tiles}
    tessellate \
        ${seg_config_str} \
        ${seg_params_str} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        ${sizedOutputArg("output_h5_path", meta.slide_id, "@.patch.h5", tsets)} \
        ${save_thumbnail_param} \
        ${save_tile_param} \
        ${stitch_tile_param} \
        ${TaskProfile.timingsArg(params.profile_dir)}
    if [ -n "${expected_seg_model}" ]; then
        validate_tessellation_h5.py \
            --expected-seg-model=${expected_seg_model} \
            ./*.patch.h5
    fi
    """

    stub:
    stub_files = tsets.collect { "${tileFileName(meta.slide_id, tsets, it)}:${it.size}:${it.mpp}" }.join(',')
    """
    #!/usr/bin/env python3
    import h5py, numpy as np
    for item in "${stub_files}".split(","):
        name, size, mpp = item.rsplit(":", 2)
        with h5py.File(name, "w") as f:
            ds = f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
            ds.attrs["native_mpp"] = 0.5
            ds.attrs["mpp_is_fallback"] = False
            ds.attrs["patch_size_to_resize_to_for_desired_mpp"] = int(size)
            ds.attrs["mpp"] = float(mpp)
    """
}

process TESSELLATE_BATCH {
    label "medTask"

    scratch params.scratch_dir ?: false

    publishDir path: "${params.outdir}", mode: "${params.publish_mode}", pattern: "*.patch.h5", saveAs: { fn -> tilePublishPath(fn, tsets) }

    input:
    tuple val(batch_meta), path(slides), val(tsets) // tsets: all tile sets [size, dir, suffix], primary first

    output:
    tuple val(batch_meta), path("*.patch.h5"), optional: true, emit: h5

    script:
    // Use seg_config group if specified, otherwise use individual parameters.
    seg_config_str = params.tiling.seg_config_group ? "seg_config=${params.tiling.seg_config_group}" : ""

    seg_params = []
    seg_params << tessellateSizeArgs(tsets)  // all tile sets, one segmentation (TileSizes)
    if (params.tiling.segment_threshold != null) seg_params << "seg_config.segment_threshold=${params.tiling.segment_threshold}"
    if (params.tiling.median_blur_ksize != null) seg_params << "seg_config.median_blur_ksize=${params.tiling.median_blur_ksize}"
    if (params.tiling.morphology_ex_kernel != null) seg_params << "seg_config.morphology_ex_kernel=${params.tiling.morphology_ex_kernel}"
    if (params.tiling.tissue_area_threshold != null) seg_params << "seg_config.tissue_area_threshold=${params.tiling.tissue_area_threshold}"
    if (params.tiling.hole_area_threshold != null) seg_params << "seg_config.hole_area_threshold=${params.tiling.hole_area_threshold}"
    if (params.tiling.max_num_holes != null) seg_params << "seg_config.max_num_holes=${params.tiling.max_num_holes}"
    if (params.tiling.overlap != null) seg_params << "seg_config.overlap=${params.tiling.overlap}"
    if (params.tiling.min_tissue_proportion != null) seg_params << "seg_config.min_tissue_proportion=${params.tiling.min_tissue_proportion}"
    if (params.tiling.remove_artifacts) seg_params << "seg_config.remove_artifacts=true"
    if (params.tiling.remove_penmarks) seg_params << "seg_config.remove_penmarks=true"
    if (params.tiling.seg_model != null) seg_params << "seg_config.seg_model=${params.tiling.seg_model}"
    if (params.tiling.slide_mpp_override != null) seg_params << "seg_config.slide_mpp_override=${params.tiling.slide_mpp_override}"
    seg_params_str = seg_params.join(' \\\n            ')

    // Staged basenames and slide IDs contain no shell metacharacters. Build the
    // Hydra lists in Groovy so the task script does not need a Python heredoc.
    slide_ids_arg = batch_meta.collect { "\"${it.slide_id}\"" }.join(',')
    slide_paths_arg = slides.collect { "\"${it.name}\"" }.join(',')
    // Several tile sets: Mussel writes one file per set from the sizeTag placeholders.
    output_h5_paths_arg = batch_meta.collect { "\"${it.slide_id}${sizeTag(tsets)}.patch.h5\"" }.join(',')
    neural_seg_max_tiles = params.tiling.neural_seg_max_tiles != null
        ? params.tiling.neural_seg_max_tiles
        : 4096
    expected_seg_model = params.tiling.seg_model?.toString()?.trim()?.toLowerCase()

    """
    ${TaskProfile.script(params.profile_dir, task)}
    export MUSSEL_NEURAL_SEG_MAX_TILES=${neural_seg_max_tiles}
    tessellate \\
        ${seg_config_str} \\
        ${seg_params_str} \\
        num_workers=${task.cpus} \\
        'slide_paths=[${slide_paths_arg}]' \\
        'slide_ids=[${slide_ids_arg}]' \\
        'output_h5_paths=[${output_h5_paths_arg}]' \\
        continue_on_error=true \\
        failures_tsv_path=tessellate_batch_failures.tsv \\
        ${TaskProfile.timingsArg(params.profile_dir)}
    if [ -n "${expected_seg_model}" ]; then
        validate_tessellation_h5.py \\
            --expected-seg-model=${expected_seg_model} \\
            ./*.patch.h5
    fi
    """

    stub:
    slide_ids_str = batch_meta.collect { it.slide_id }.join('\n')
    // "{sid}" stands for each slide id in the tile file names.
    stub_sets = tsets.collect { "${tileFileName('{sid}', tsets, it)}:${it.size}:${it.mpp}" }.join(',')
"""#!/usr/bin/env python3
import h5py, numpy as np
failures = []
for slide_id in '''${slide_ids_str}'''.splitlines():
    if slide_id.startswith("FAIL"):
        failures.append(slide_id)
        continue
    for item in "${stub_sets}".split(","):
        name, size, mpp = item.rsplit(":", 2)
        with h5py.File(name.replace("{sid}", slide_id), "w") as f:
            ds = f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
            ds.attrs["native_mpp"] = 0.5
            ds.attrs["mpp_is_fallback"] = False
            ds.attrs["patch_size_to_resize_to_for_desired_mpp"] = int(size)
            ds.attrs["mpp"] = float(mpp)
if failures:
    with open("tessellate_batch_failures.tsv", "w") as fh:
        for slide_id in failures:
            fh.write(f"{slide_id}\\t1\\n")
"""
}

// Derived tile sets (featurize.grid_from_primary): every cell of the set's size that
// contains a primary tile centre, from a batch of slides' primary tiles, with Mussel's
// derive_tile_grid (Mussel #151). It only reads and writes coordinate files, but it runs
// a batch of slides in the Mussel image, so it is a cluster job rather than a local task:
// inside a memory-limited Slurm-hosted driver it can OOM-kill the driver. Give it the
// Mussel environment of the models that use the set (withName: DERIVE_TILE_GRID), e.g.
// the TITAN image.
process DERIVE_TILE_GRID {
    label "medTask"

    publishDir path: "${params.outdir}", mode: "${params.publish_mode}", saveAs: { fn -> tilePublishPath(fn, tsets) }

    input:
    // Primary tile files of a batch of slides (metas in the same order), the derived
    // tile set to write, and all tile sets (for file names and publishing).
    tuple val(batch_meta), path(primary_h5s), val(tset), val(tsets)

    output:
    tuple val(batch_meta), val(tset), path("*.patch.h5"), emit: h5

    script:
    def files = primary_h5s instanceof List ? primary_h5s : [primary_h5s]
    def sources = files.collect { "\"${it.name}\"" }.join(',')
    def outputs = batch_meta.collect { "\"${tileFileName(it.slide_id, tsets, tset)}\"" }.join(',')
    """
    derive_tile_grid \
        'source_h5_paths=[${sources}]' \
        'output_h5_paths=[${outputs}]' \
        patch_size=${tset.size}
    """

    stub:
    def outputs = batch_meta.collect { tileFileName(it.slide_id, tsets, tset) }.join(',')
    """
    python3 - <<'PYEOF'
import h5py, numpy as np
for name in "${outputs}".split(","):
    with h5py.File(name, "w") as f:
        ds = f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
        ds.attrs["native_mpp"] = 0.5
        ds.attrs["mpp_is_fallback"] = False
        ds.attrs["patch_size_to_resize_to_for_desired_mpp"] = ${tset.size}
        ds.attrs["mpp"] = ${tset.mpp}
        ds.attrs["patch_size"] = ${tset.size}
        ds.attrs["patch_level"] = 0
        ds.attrs["derived_grid_method"] = "cells_containing_source_tile_centres"
PYEOF
    """
}

process EMIT_TESSELLATE_PATH_META {
    label "localTask"

    // One slide_meta manifest entry: (meta, key, path relative to outdir).
    input:
    tuple val(meta), val(key), val(relpath)

    output:
    tuple val(meta), val(key), val(relpath), topic: slide_meta

    script:
    """
    true
    """

    stub:
    """
    true
    """
}

process EMIT_MPP_META {
    label "localTask"

    input:
    tuple val(meta), path(tile_h5)

    output:
    tuple val(meta), val("mpp_is_fallback"), env('MPP_IS_FALLBACK'), topic: slide_meta
    tuple val(meta), val("native_mpp"),      env('NATIVE_MPP'),      topic: slide_meta

    script:
    """
    python3 -c 'import h5py; f=h5py.File("${tile_h5}", "r"); attrs=f["coords"].attrs; flag=attrs.get("mpp_is_fallback", None); mpp=attrs.get("native_mpp", None); print("MPP_IS_FALLBACK=" + ("true" if flag else "false")); print("NATIVE_MPP=" + (str(mpp) if mpp is not None else "unknown")); f.close()' > mpp_meta.env || printf '%s\n' MPP_IS_FALLBACK=unknown NATIVE_MPP=unknown > mpp_meta.env
    source mpp_meta.env
    export MPP_IS_FALLBACK
    export NATIVE_MPP
    """

    stub:
    """
    export MPP_IS_FALLBACK=false
    export NATIVE_MPP=0.5
    """
}

process FILTER_TILES {
    label "bigTask"
    label "cpuTask"

    publishDir path: { "${params.outdir}/filter_tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}" }, mode: "${params.publish_mode}", pattern: "*.h5"
    publishDir path: { "${params.outdir}/features/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}" }, mode: "${params.publish_mode}", pattern: "*.pt"

    input:
    tuple val(meta), val(model_type), path(features_h5)

    output:
    tuple val(meta), path("${meta.slide_id}.patch.h5"), emit: h5
    tuple val(meta), path("${meta.slide_id}.features.pt"), emit: pt
    tuple val(meta), val("filtered_tiles_h5_path"), val("filter_tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.patch.h5"), topic: slide_meta
    tuple val(meta), val("${model_type}_features_tensor_path"), val("features/${model_type}/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.features.pt"), topic: slide_meta

    script:
    """
    filter_features \
        features_h5_path=${features_h5} \
        output_h5_path=${meta.slide_id}.patch.h5 \
        output_pt_path=${meta.slide_id}.features.pt \
        classifier_threshold=${params.tiling.filter_threshold} \
        classifier_pkl=${params.tiling.filter_model_path}
    """

    stub:
    """
    #!/usr/bin/env python3
    import h5py, numpy as np, torch
    with h5py.File("${meta.slide_id}.patch.h5", "w") as f:
        f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
    torch.save(torch.zeros(1, 8), "${meta.slide_id}.features.pt")
    """
}
