include { validateParameters; paramsSummaryLog; samplesheetToList } from 'plugin/nf-schema'

// nextflow.preview.topic = true

include { MUSSEL } from './modules/mussel'
include { DOWNLOAD_SLIDE } from './modules/download'


process saveParams {
    executor 'local'
    publishDir params.outdir, mode: 'copy'

    output:
        path "params.json"

    script:
        params_out = params.subMap(['workflow_id', 'tiling'])
        params_out["outdir"] = new File(params.outdir).absolutePath
        "echo '${groovy.json.JsonOutput.toJson(params_out)}' > params.json"
}

workflow {
    // Validate input parameters
    validateParameters()

    // Print summary of supplied parameters
    log.info paramsSummaryLog(workflow)

    def timestamp = new Date().getTime()

    // ── Pre-flight validation ──────────────────────────────────────────────

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
        def outdir_path = file(params.outdir)
        def parent = outdir_path.parent ?: file('.')
        if (parent.exists() && !parent.canWrite()) {
            log.error "Output directory parent is not writable: ${parent}"
            System.exit(1)
        }
    }

    ch_samples = Channel.empty()
    ch_annotations = Channel.empty()

    if (params.samples_csv) {
        // Load samplesheet and enrich meta with sample_id (defaults to slide_id) and n_slides
        def raw_list = samplesheetToList(params.samples_csv, "assets/schema_input.json")
        // Count how many slides belong to each sample_id
        def sample_counts = [:]
        raw_list.each { meta, slide ->
            def sid = meta.sample_id ?: meta.slide_id.toString()
            sample_counts[sid] = (sample_counts[sid] ?: 0) + 1
        }
        def enriched = raw_list.collect { meta, slide ->
            def sid = meta.sample_id ?: meta.slide_id.toString()
            def enriched_meta = meta + [sample_id: sid, n_slides: sample_counts[sid]]
            // For GDC download slides, nf-schema may return [] for empty optional fields.
            // Extract file_id and file_name from the gdc:// slide_path URI when missing.
            if (enriched_meta.needs_download) {
                def fid = enriched_meta.file_id
                def fname = enriched_meta.file_name
                def fid_missing = !fid || (fid instanceof List && fid.isEmpty())
                def fname_missing = !fname || (fname instanceof List && fname.isEmpty())
                if (fid_missing || fname_missing) {
                    def sp = slide?.toUriString() ?: slide?.toString() ?: ""
                    def m = sp =~ /^gdc:\/\/([^\/]+)\/(.+)$/
                    if (m) {
                        enriched_meta = enriched_meta + [
                            file_id:   fid_missing   ? m[0][1] : fid,
                            file_name: fname_missing ? m[0][2] : fname,
                        ]
                    }
                }
            }
            tuple(enriched_meta, slide)
        }
        ch_samples = Channel.fromList(enriched)

        // Route slides that need GDC download through DOWNLOAD_SLIDE.
        // Slides with needs_download=true have a future local path in slide_path
        // that doesn't exist yet; we never stage that file — instead we pass only
        // val(meta) to DOWNLOAD_SLIDE which fetches and caches the file via storeDir.
        ch_samples.branch { meta, slide ->
            needs_download: meta.needs_download == true
            ready: true
        }.set { ch_branched }

        ch_downloaded = DOWNLOAD_SLIDE(
            ch_branched.needs_download.map { meta, _slide -> meta }
        )
        ch_samples = ch_branched.ready.mix(ch_downloaded)
    }

    if (params.linear_probe.annotations_csv) {
        ch_annotations = Channel.fromList(samplesheetToList(params.linear_probe.annotations_csv, "assets/schema_annotations.json"))
    }

    if (params.samples_csv_watch_path) {
        ch_samples = Channel.watchPath("${params.samples_csv_watch_path}/*.csv", 'create,modify')
            .flatMap { csv ->
                def raw_list = samplesheetToList(csv, "assets/schema_input.json")
                def sample_counts = [:]
                raw_list.each { meta, slide ->
                    def sid = meta.sample_id ?: meta.slide_id.toString()
                    sample_counts[sid] = (sample_counts[sid] ?: 0) + 1
                }
                raw_list.collect { meta, slide ->
                    def sid = meta.sample_id ?: meta.slide_id.toString()
                    tuple(meta + [sample_id: sid, n_slides: sample_counts[sid]], slide)
                }
            }
    }


    tmpdir = "${params.outdir}/tmp/${timestamp}"
    new File(tmpdir).mkdirs()
    Channel.topic('slide_meta')
        .map { it[0..2] }
        // Manifest columns: slide_id, sample_id, workflow_id, key, value
        .collectFile(storeDir: params.outdir, tempDir: tmpdir, sort: false, cache: true) {
            meta, key, value ->
            ["manifest-${timestamp}.csv", "${meta.slide_id},${meta.sample_id},${params.workflow_id},${key},${value}\n"]
        }


    MUSSEL(ch_samples, ch_annotations)

    saveParams()

}

