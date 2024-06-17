params.outdir = "results"
params.samples_csv = null
params.oncotree_class_csv = null

include { MUSSEL } from './modules/mussel'

workflow {
    ch_samples = Channel.empty()

    if (params.samples_csv) {
        ch_samples = Channel.fromPath(params.samples_csv) \
            .splitCsv(header: true)
    }

    MUSSEL(ch_samples)
}

