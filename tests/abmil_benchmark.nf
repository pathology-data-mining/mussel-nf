// Standalone integration test for the ABMIL_BENCHMARK_WORKFLOW.
//
// Uses pre-generated fixtures in tests/fixtures/abmil_benchmark/ (20 synthetic
// slides, variable tiles, 8 features, binary labels parquet).
// Runs entirely on the local executor — no GPU, no real slides.
//
// Calls ABMIL_BENCHMARK / SUMMARIZE_ABMIL_BENCHMARK directly so fixture paths
// are resolved relative to this script.
//
// Usage:
//   nf-test test tests/abmil_benchmark.nf.test

include { ABMIL_BENCHMARK        } from '../modules/abmil_benchmark/main'
include { SUMMARIZE_ABMIL_BENCHMARK } from '../modules/abmil_benchmark/main'

workflow {

    def fixtureDir = file("${workflow.projectDir}/fixtures/abmil_benchmark")
    def labelsParquet = file("${workflow.projectDir}/fixtures/abmil_benchmark/labels.parquet")

    if (!fixtureDir.exists()) {
        error """\
            ABMIL benchmark test fixtures not found: ${fixtureDir}
            Regenerate them with:
              python3 tests/fixtures/abmil_benchmark/generate.py
            """.stripIndent()
    }

    ch_h5_files = Channel.fromList(
        (1..20).collect { i ->
            fixtureDir.resolve("abmil_slide_${String.format('%04d', i)}.h5")
        }
    )

    ch_features_by_model = ch_h5_files
        .map { h5 -> tuple('resnet50', h5) }
        .groupTuple()

    ABMIL_BENCHMARK(ch_features_by_model, labelsParquet)

    ABMIL_BENCHMARK.out.results_json
        .collect(flat: false)
        | SUMMARIZE_ABMIL_BENCHMARK
}
