params.outdir = "results"
params.samples_csv = null
params.oncotree_class_csv = null

params.test = false

include { MUSSEL } from './modules/mussel'

workflow {
    ch_samples = Channel.empty()

    if (params.samples_csv) {
        ch_samples = Channel.fromPath(params.samples_csv) \
            .splitCsv(header: true)
            .filter { file(it.slide_path).exists() }

    }

    if (params.test) {
        ch_samples = ch_samples.take(1)
    }

    MUSSEL(ch_samples)

    Channel.topic('meta_out')
        .map { it[0..3] }
        .collectFile(storeDir: params.outdir) {
            slide_id, model_type, type, path ->
            ["manifest.csv", "${slide_id},${model_type},${type},${path}\n"]
        }

}

