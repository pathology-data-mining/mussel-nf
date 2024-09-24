params.outdir = "results"
params.samples_csv = null
params.annotations_csv = null
params.oncotree_class_csv = null

nextflow.preview.topic = true

params.test = false

include { MUSSEL } from './modules/mussel'

workflow {
    ch_samples = Channel.empty()
    ch_annotations = Channel.empty()

    if (params.samples_csv) {
        ch_samples = Channel.fromPath(params.samples_csv) \
            .splitCsv(header: true)
            .filter { file(it.slide_path).exists() }
    }

    if (params.annotations_csv) {
        ch_annotations = Channel.fromPath(params.annotations_csv) \
            .splitCsv(header: true)
            .map { [it.slide_id, it.annotation_bmp_path] }
    }

    if (params.test) {
        ch_samples = ch_samples.take(1)
    }

    MUSSEL(ch_samples, ch_annotations)

    Channel.topic('meta_out')
        .map { it[0..2] }
        .collectFile(storeDir: params.outdir) {
            slide_id, type, path ->
            ["manifest.csv", "${slide_id},${type},${path}\n"]
        }

}

