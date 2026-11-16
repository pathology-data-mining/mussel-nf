nextflow.enable.dsl=2

include { FEATURIZE_PATCH_MULTI } from '../modules/featurize'

workflow {
    slide = file("${projectDir}/testdata/948176.svs")
    patch_h5 = file("${projectDir}/fixtures/1001234.patch.h5")
    batch = [[[slide_id: '1001234', sample_id: '1001234'], slide, patch_h5]]
    // The fake extract_features validates the real command line (model_types,
    // ++model_precisions) and writes <model>/<slide_id>.features.{pt,h5}.
    FEATURIZE_PATCH_MULTI(Channel.of(tuple(batch, slide, patch_h5, ['hoptimus1', 'optimus'], 'error')))
}
