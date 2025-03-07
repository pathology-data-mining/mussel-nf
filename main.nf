include { validateParameters; paramsSummaryLog; samplesheetToList } from 'plugin/nf-schema'

// Validate input parameters
validateParameters()

// Print summary of supplied parameters
log.info paramsSummaryLog(workflow)


import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput

timestamp = new Date().getTime()

nextflow.preview.topic = true

include { MUSSEL } from './modules/mussel'


process saveParams {
    publishDir params.outdir, mode: 'copy'

    output:
        path "params.json"

    script:
        params_out = params.subMap(['workflow_id', 'tiling'])
        params_out["outdir"] = new File(params.outdir).absolutePath
        "echo '${JsonOutput.toJson(params_out)}' > params.json"
}

workflow {
    ch_samples = Channel.empty()
    ch_annotations = Channel.empty()

    if (params.samples_csv) {
        ch_samples = Channel.fromList(samplesheetToList(params.samples_csv, "assets/schema_input.json"))
    }

    if (params.linear_probe.annotations_csv) {
        ch_annotations = Channel.fromList(samplesheetToList(params.linear_probe.annotations_csv, "assets/schema_annotations.json"))
    }

    if (params.samples_csv_watch_path) {
        ch_samples = Channel.watchPath("${params.samples_csv_watch_path}/*.csv", 'create,modify')
            .flatMap { samplesheetToList(it, "assets/schema_input.json") }
    }


    tmpdir = "${params.outdir}/tmp"
    new File(tmpdir).mkdirs()
    Channel.topic('slide_meta')
        .map { it[0..2] }
        .collectFile(storeDir: params.outdir, tempDir: tmpdir, sort: false, cache: true) {
            meta, key, value ->
            ["manifest-${timestamp}.csv", "${meta.slide_id},${params.workflow_id},${key},${value}\n"]
        }


    MUSSEL(ch_samples, ch_annotations)

    saveParams()

}

