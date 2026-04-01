/**
 * WDS_SHARD — pack per-slide feature files into WebDataset tar shards.
 *
 * Inputs
 * ------
 *   group_name  : string key for the shard group, e.g. an oncotree code or "all"
 *   model_type  : the encoder name used to produce the features, e.g. "optimus"
 *   slide_ids   : list of slide ID strings (same order as files)
 *   pt_files    : collected list of *.features.pt paths
 *   h5_files    : collected list of *.features.h5 paths (empty list when not sharding h5)
 *
 * Output
 * ------
 *   shard-NNNNNN.tar files published to
 *     ${params.outdir}/wds/${model_type}/${group_name}/
 *
 * WDS format
 * ----------
 *   Each tar entry is named  {slide_id}.pt  (and optionally {slide_id}.features.h5).
 *   Shards are deterministically ordered by slide_id within each group.
 *   This is directly readable by the `webdataset` Python library.
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
    prefix         = params.wds.shard_prefix   ?: "shard-"
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
}
