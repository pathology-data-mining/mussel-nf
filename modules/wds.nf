/**
 * WDS_SHARD — pack per-slide feature files into paladin-compatible WebDataset tar shards.
 *
 * Inputs
 * ------
 *   group_name  : string key for the shard group, e.g. an oncotree code or "all"
 *   model_type  : the encoder name used to produce the features, e.g. "optimus"
 *   slide_ids   : list of slide ID strings (same order as files)
 *   pt_files    : collected list of *.features.pt paths
 *   h5_files    : collected list of *.patch.h5 paths (empty list when wds.shard_h5=false)
 *
 * Output
 * ------
 *   NNNNNN.tar files published to
 *     ${params.outdir}/wds/${model_type}/${group_name}/
 *
 * WDS format (paladin-compatible)
 * --------------------------------
 *   Each tar entry consists of:
 *     {slide_id}.features.npy  — float32 feature array (converted from .pt)
 *     {slide_id}.coords.npy    — int64 tile coords extracted from .h5 (when shard_h5=true)
 *   Shards are deterministically ordered by slide_id within each group.
 *   Directly consumable by paladin and the `webdataset` Python library.
 */
process WDS_SHARD {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/wds/${model_type}/${group_name}",
               mode: "${params.publish_mode}",
               pattern: "*.tar"

    input:
    tuple val(group_name), val(model_type), val(slide_ids), path(pt_files), path(h5_files)

    output:
    tuple val(group_name), val(model_type), path("*.tar"), emit: shards

    script:
    max_shard_size = params.wds.max_shard_size ?: 1000
    prefix         = params.wds.shard_prefix != null ? params.wds.shard_prefix : ""
    slide_ids_str  = slide_ids.join(",")

    // Build the pt file list in the work directory (staged names)
    pt_list  = pt_files instanceof List  ? pt_files.collect  { it.name } : [pt_files.name]
    h5_list  = h5_files instanceof List  ? h5_files.collect  { it.name } : [h5_files.name]

    // Only pass --h5_files when the caller actually provided h5 paths
    h5_arg = (params.wds.shard_h5 && h5_list.any { it != 'NO_FILE' && it != '' })
        ? "--h5_files ${h5_list.join(' ')}"
        : ""

    """
    wds_shard.py \\
        --pt_files ${pt_list.join(' ')} \\
        --slide_ids '${slide_ids_str}' \\
        --output_dir . \\
        --max_shard_size ${max_shard_size} \\
        --prefix '${prefix}' \\
        ${h5_arg}
    """

    stub:
    stub_prefix = params.wds.shard_prefix != null ? params.wds.shard_prefix : ""
    """
    #!/usr/bin/env python3
    import tarfile, io
    with tarfile.open("${stub_prefix}000000.tar", "w") as t:
        data = b""
        info = tarfile.TarInfo(name="stub.txt")
        info.size = 0
        t.addfile(info, io.BytesIO(data))
    """
}
