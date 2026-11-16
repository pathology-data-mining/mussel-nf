nextflow.enable.dsl=2

include { FEATURIZE_BATCH } from '../modules/featurize'
include { perSlide } from '../modules/mussel'

// One batch of four slides through FEATURIZE_BATCH's real script and the fake
// extract_features (tests/fixtures/bin): FAIL0001 fails, EMPTY0003 has no tiles,
// 1001234 and SLIDE0002 succeed. Mussel >= 1.6.0 writes the good slides' outputs
// and exits 1; the task must keep them, and perSlide must pair each file with its
// own slide (sorted positionally, SLIDE0002's file would go to EMPTY0003).
workflow {
    def patch_h5 = file("${projectDir}/fixtures/1001234.patch.h5")
    def ids = params.ids ? params.ids.tokenize(',') : ['1001234', 'FAIL0001', 'SLIDE0002', 'EMPTY0003']
    // Distinct staged names: one slide file linked under each slide's name.
    def batch = ids.collect { sid ->
        def slide = file("${params.outdir}/slides/${sid}.svs")
        if (!slide.exists()) {
            slide.parent.mkdirs()
            java.nio.file.Files.createSymbolicLink(slide, file("${projectDir}/testdata/948176.svs"))
        }
        def h5 = file("${params.outdir}/slides/${sid}.patch.h5")
        if (!h5.exists()) {
            java.nio.file.Files.createSymbolicLink(h5, patch_h5)
        }
        [[slide_id: sid, sample_id: sid], slide, h5]
    }
    FEATURIZE_BATCH(Channel.of(tuple(batch, batch.collect { it[1] }, batch.collect { it[2] }, params.model, 'error')), true)

    perSlide(FEATURIZE_BATCH.out.pt.mix(FEATURIZE_BATCH.out.h5))
        .map { meta, model, f -> "${meta.slide_id}\t${f.name}" }
        .collectFile(name: 'pairs.tsv', storeDir: params.outdir, newLine: true, sort: true)
    FEATURIZE_BATCH.out.failed
        .map { metas, model, tsv -> tsv }
        .collectFile(name: 'failed_slides.tsv', storeDir: params.outdir)
}
