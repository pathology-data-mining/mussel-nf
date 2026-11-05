nextflow.enable.dsl=2

include { ABMIL_BENCHMARK } from '../modules/abmil_benchmark/main'

def featureFiles(dir, ids, suffix) {
    return ids.collect { sid ->
        def f = file("${dir}/${sid}.${suffix}")
        f.text = ''
        f
    }
}

// ABMIL_BENCHMARK's real script with the fake abmil_benchmark (tests/fixtures/bin):
// a patch encoder's {slide_id}.features.h5 and a slide encoder's
// {slide_id}.patch_features.h5 (its patch-level features) must both be found.
workflow {
    def dir = file("${params.outdir}/inputs")
    dir.mkdirs()
    def ids = ['S1', 'S2', 'S3']
    def labels = file("${dir}/labels.txt")
    labels.text = ids.join('\n') + '\n'
    ch = Channel.of(
        tuple('optimus', featureFiles(dir, ids, 'features.h5')),
        tuple('conch1_5', featureFiles(dir, ids, 'patch_features.h5')),
    )
    ABMIL_BENCHMARK(ch, labels)
}
