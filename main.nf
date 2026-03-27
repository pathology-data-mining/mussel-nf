include { validateParameters; paramsSummaryLog; samplesheetToList } from 'plugin/nf-schema'

// Validate input parameters
validateParameters()

// Print summary of supplied parameters
log.info paramsSummaryLog(workflow)


import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput

timestamp = new Date().getTime()

// nextflow.preview.topic = true

include { MUSSEL } from './modules/mussel'


// ── Pre-flight validation (replaces workflow.onStart removed in NF 25.x) ─────

// Validate model paths exist on disk
if (params.featurize?.model_paths) {
    params.featurize.model_paths.each { model_name, model_path ->
        if (model_path && !file(model_path).exists()) {
            log.error "Model path for '${model_name}' does not exist: ${model_path}"
            System.exit(1)
        }
    }
    log.info "✓ All model paths validated"
}

// Warn if HF_TOKEN is likely needed but not set
def gated_models = ['gigapath', 'virchow', 'virchow2', 'uni', 'uni2h', 'conch1_5',
                    'googlepath', 'hoptimus0', 'hoptimus1', 'prism_slide', 'feather_slide']
def requested_models = []
if (params.featurize?.model_types) {
    requested_models = params.featurize.model_types instanceof List
        ? params.featurize.model_types
        : [params.featurize.model_types]
}
def needs_hf_token = requested_models.any { it in gated_models }
if (needs_hf_token) {
    try {
        def token = secrets.HF_TOKEN
        log.info "✓ HF_TOKEN secret is available"
    } catch (Exception e) {
        log.warn "⚠ HF_TOKEN secret not found. Gated HuggingFace models (${requested_models.findAll { it in gated_models }}) may fail to download."
    }
}

// Validate output directory parent is writable
if (params.outdir) {
    def outdir = file(params.outdir)
    def parent = outdir.parent ?: file('.')
    if (parent.exists() && !parent.canWrite()) {
        log.error "Output directory parent is not writable: ${parent}"
        System.exit(1)
    }
}

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

