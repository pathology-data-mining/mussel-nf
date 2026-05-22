
process TESSELLATE {
    label "bigTask"
    // Note: no cpuTask/gpuTask label here — GPU requirement is determined at
    // runtime based on params.tiling.seg_model.  When seg_model='neural' (the
    // default), tessellation uses a neural segmentation model (GrandQC) that
    // requires a GPU.  The executor profiles (condor, slurm, cloud) handle this
    // via withName:TESSELLATE directives in nextflow.config.

    scratch params.scratch_dir ?: false

    publishDir path: { "${params.outdir}/tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}" }, mode: "${params.publish_mode}"

    input:
    tuple val(meta), path(slide)

    output:
    tuple val(meta), path("${meta.slide_id}.patch.h5"), optional: true, emit: h5
    tuple val(meta), val("tiles_h5_path"), val("tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.patch.h5"), path("${meta.slide_id}.patch.h5"), optional: true, topic: slide_meta
    path "${meta.slide_id}_png/*.png", optional: true, emit: png
    path "${meta.slide_id}.*.png", optional: true, emit: thumbnail_png
    tuple val(meta), val("thumbnail_path"), val("tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.thumbnail.png"), path("${meta.slide_id}.thumbnail.png"), optional: true, topic: slide_meta
    tuple val(meta), val("grid_mask_path"), val("tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.grid_mask.png"), path("${meta.slide_id}.grid_mask.png"), optional: true, topic: slide_meta
    tuple val(meta), val("mask_path"), val("tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}.mask.png"), path("${meta.slide_id}.mask.png"), optional: true, topic: slide_meta
    tuple val(meta), val("tile_png_path"), val("tiles/${params.publish_slide_prefix ? meta.slide_id.toString()[0..3] : ''}/${meta.slide_id}_png"), path("${meta.slide_id}_png/*.png"), optional: true, topic: slide_meta

    script:
    def save_tile_param = ""
    if (params.tiling.save_tile_png)
        save_tile_param = "output_png_dir=${meta.slide_id}_png"
    stitch_tile_param = ""
    if (params.tiling.stitch_tiles)
        stitch_tile_param = "output_grid_mask_path=${meta.slide_id}.grid_mask.png"
        stitch_tile_param += " output_mask_path=${meta.slide_id}.mask.png"
    save_thumbnail_param = ""
    if (params.tiling.save_slide_thumbnail)
        save_thumbnail_param = "output_thumbnail_path=${meta.slide_id}.thumbnail.png"

    // Use seg_config group if specified, otherwise use individual parameters
    seg_config_str = params.tiling.seg_config_group ? "seg_config=${params.tiling.seg_config_group}" : ""

    // Build individual parameter overrides (only if not using group, or to override group defaults)
    seg_params = []
    if (params.tiling.mpp != null) seg_params << "seg_config.mpp=${params.tiling.mpp}"
    if (params.tiling.patch_size != null) seg_params << "seg_config.patch_size=${params.tiling.patch_size}"
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
    tessellate \
        ${seg_config_str} \
        ${seg_params_str} \
        num_workers=${task.cpus} \
        slide_path=${slide} \
        output_h5_path=${meta.slide_id}.patch.h5 \
        ${save_thumbnail_param} \
        ${save_tile_param} \
        ${stitch_tile_param}
    """

    stub:
    """
    #!/usr/bin/env python3
    import h5py, numpy as np
    with h5py.File("${meta.slide_id}.patch.h5", "w") as f:
        ds = f.create_dataset("coords", data=np.array([[0, 0]], dtype="int64"))
        ds.attrs["native_mpp"] = 0.5
        ds.attrs["mpp_is_fallback"] = False
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
    python3 - > mpp_meta.env <<'PYEOF'
import h5py, sys
try:
    with h5py.File("${tile_h5}", "r") as f:
        attrs = f["coords"].attrs
        flag  = attrs.get("mpp_is_fallback", None)
        mpp   = attrs.get("native_mpp", None)
    print(f"MPP_IS_FALLBACK={'true' if flag else 'false'}")
    print(f"NATIVE_MPP={mpp if mpp is not None else 'unknown'}")
except Exception as e:
    print("MPP_IS_FALLBACK=unknown")
    print("NATIVE_MPP=unknown")
PYEOF
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

