nextflow.enable.dsl=2

include { FEATURIZE_BATCH } from '../modules/featurize'

workflow {
    meta = [slide_id: '1001234', sample_id: '1001234']
    slide = file("${projectDir}/testdata/948176.svs")
    patch_h5 = file("${projectDir}/fixtures/1001234.patch.h5")

    // The fake command validates the generated script's real command-line
    // arguments and creates the declared feature outputs.
    batch = [[meta, slide, patch_h5]]
    input_ch = Channel.of(tuple(batch, slide, patch_h5, 'optimus', 'error'))
    FEATURIZE_BATCH(input_ch, true)
}
