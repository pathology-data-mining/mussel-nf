// Standalone integration test for multi-slide sample aggregation.
//
// Handles its own test-data setup in Groovy (no shell pre-processing needed),
// so it composes cleanly with any additional Nextflow profile:
//
//   nextflow run tests/multi_slide.nf -profile test_multi_slide
//   nextflow run tests/multi_slide.nf -profile test_multi_slide,conda
//   nextflow run tests/multi_slide.nf -profile test_multi_slide,slurm,cluster
//
// The only prerequisite is that tests/testdata/948176.svs exists (vendored in the repo).

include { MUSSEL } from '../modules/mussel'

import groovy.json.JsonOutput

timestamp = new Date().getTime()

workflow {

    // ── Test-data setup (pure Groovy — runs before any process is submitted) ──

    // Resolve data dir relative to the script location (tests/data/), not via params,
    // so this works regardless of launch directory or how projectDir resolves.
    def dataDir = file("${workflow.projectDir}/testdata")
    dataDir.mkdirs()

    def svs_a = dataDir.resolve('948176.svs')
    def svs_b = dataDir.resolve('948176_b.svs')

    if (!svs_a.exists()) {
        log.error """\
            Test SVS not found: ${svs_a}
            The test slide should be vendored at tests/testdata/948176.svs.
            Re-clone or restore it from the repository.
            """.stripIndent()
        System.exit(1)
    }

    if (!svs_b.exists()) {
        java.nio.file.Files.createSymbolicLink(svs_b, svs_a.toRealPath())
        log.info "Created test symlink: ${svs_b} -> ${svs_a.name}"
    }

    // ── Sample channel — two slides belonging to one sample ───────────────────
    // Constructed directly; no CSV file required.

    ch_samples = Channel.fromList([
        [ [slide_id: '948176_A', sample_id: 'PATIENT_TEST', n_slides: 2], svs_a ],
        [ [slide_id: '948176_B', sample_id: 'PATIENT_TEST', n_slides: 2], svs_b ],
    ])

    ch_annotations = Channel.empty()

    // ── Manifest (mirrors main.nf) ────────────────────────────────────────────

    tmpdir = "${params.outdir}/tmp"
    new File(tmpdir).mkdirs()
    Channel.topic('slide_meta')
        .map { it[0..2] }
        .collectFile(storeDir: params.outdir, tempDir: tmpdir, sort: false, cache: true) {
            meta, key, value ->
            ["manifest-${timestamp}.csv", "${meta.slide_id},${meta.sample_id},${params.workflow_id},${key},${value}\n"]
        }

    MUSSEL(ch_samples, ch_annotations)
}
