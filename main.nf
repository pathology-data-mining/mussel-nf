import org.apache.commons.io.FilenameUtils

params.outdir = "results"
params.publish_mode = "copy"
params.samples_csv = null
params.annotations_csv = null
params.oncotree_class_csv = null

params.samples_csv_watch_path = null
params.watch_path = null

nextflow.preview.topic = true

params.test = false

include { MUSSEL } from './modules/mussel'

timestamp = new Date().getTime()

workflow {
    ch_samples = Channel.empty()
    ch_annotations = Channel.empty()

    if (params.samples_csv) {
        ch_samples = Channel.fromPath(params.samples_csv)
            .splitCsv(header: true)
            .filter { file(it.slide_path).exists() }
    }

    if (params.annotations_csv) {
        ch_annotations = Channel.fromPath(params.annotations_csv)
            .splitCsv(header: true)
            .map { [it.slide_id, it.annotation_bmp_path] }
    }

    if (params.samples_csv_watch_path) {
        ch_samples = Channel.watchPath("${params.samples_csv_watch_path}/*.csv", 'create,modify')
            .splitCsv(header: true)
    }

    if (params.watch_path) {
        ch_samples = Channel.watchPath(params.watch_path).map { [ slide_id: FilenameUtils.removeExtension(it.name), slide_path: it ] }
    }

    if (params.test) {
        ch_samples = ch_samples.take(1)
    }


    Channel.topic('meta_out')
        .map { it[0..2] }
        .collectFile(storeDir: params.outdir) {
            slide_id, type, path ->
            ["manifest-${timestamp}.csv", "${slide_id},${type},${path}\n"]
        }


    MUSSEL(ch_samples, ch_annotations)

}

