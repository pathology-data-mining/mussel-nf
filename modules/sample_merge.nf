// Process to aggregate per-slide patch-level features into one sample-level output.
// Uses aggregate_sample_features (pure concatenation — no GPU inference needed):
//   aggregate_sample_features patch_features_h5_paths=[...] sample_ids=[...] output_dir=.
//
// Input format: tuple(sample_id, [metas], model_type, [feature_h5s])
//   feature_h5s — *.features.h5 files from TESSELLATE_FEATURIZE_BATCH / FEATURIZE_BATCH
//
// Output: one .features.h5 and .features.pt per (sample_id, model_type).
// NOTE: requires save_features_to_h5 = true in the featurize params so that
//       per-slide feature H5 files are produced upstream.

process MERGE_SAMPLE_FEATURES {
    label "bigTask"
    label "cpuTask"

    publishDir path: "${params.outdir}/features/${model_type_name}", mode: "${params.publish_mode}", pattern: "*.features.pt"
    publishDir path: "${params.outdir}/features/${model_type_name}", mode: "${params.publish_mode}", pattern: "*.features.h5"

    input:
    tuple val(sample_id), val(slide_metas), val(model_type_input), path(feature_h5s)

    output:
    tuple val(sample_meta), val(model_type_input), path("*.features.pt"), emit: pt
    tuple val(sample_meta), val(model_type_input), path("*.features.h5"), emit: h5

    script:
    // Determine publish directory name (slide-level models keep their name)
    model_type_name = model_type_input

    // Construct sample-level meta (slide_id == sample_id for downstream compatibility)
    sample_oncotree = slide_metas[0].containsKey('oncotree_code') ? slide_metas[0].oncotree_code : null
    sample_meta = [slide_id: sample_id, sample_id: sample_id] + (sample_oncotree ? [oncotree_code: sample_oncotree] : [:])

    // Build Hydra list arguments: [path1,path2,...] using staged basenames
    h5_names     = feature_h5s instanceof List ? feature_h5s.collect { it.name } : [feature_h5s.name]
    h5_paths_arg = "[${h5_names.join(',')}]"
    sample_ids_arg = "[${h5_names.collect { sample_id }.join(',')}]"

    max_tiles_str = (params.featurize.max_tiles_per_sample != null) ? "max_tiles=${params.featurize.max_tiles_per_sample}" : ""

    """
    aggregate_sample_features \\
        'patch_features_h5_paths=${h5_paths_arg}' \\
        'sample_ids=${sample_ids_arg}' \\
        output_dir=. \\
        ${max_tiles_str}

    python3 -c "
import h5py, torch, numpy as np
with h5py.File('${sample_id}.features.h5', 'r') as f:
    features = torch.from_numpy(np.array(f['features']))
torch.save(features, '${sample_id}.features.pt')
"
    """
}

