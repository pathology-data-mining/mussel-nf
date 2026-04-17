// Standalone integration test for the LINEAR_PROBE workflow.
//
// Uses pre-generated fixtures in tests/fixtures/linear_probe/ (10 synthetic
// slides, 10 tiles each, 8 features, binary tumour/non-tumour BMP labels).
// Runs entirely on the local executor — no GPU, no real slides.
//
// Calls MERGE_ANNOTATION_FEATURES / STACK_ANNOTATION_FEATURES /
// LINEAR_PROBE_BENCHMARK directly so the class_mapping YAML path is
// resolved relative to this script (workflow.projectDir = tests/) rather
// than relying on params, which resolves inconsistently under nf-test.
//
// Usage:
//   nf-test test tests/linear_probe.nf.test

include { MERGE_ANNOTATION_FEATURES  } from '../modules/linear_probe/main'
include { STACK_ANNOTATION_FEATURES  } from '../modules/linear_probe/main'
include { LINEAR_PROBE_BENCHMARK     } from '../modules/linear_probe/main'

workflow {

    // workflow.projectDir == tests/ when this script is the entry point
    def fixtureDir    = file("${workflow.projectDir}/fixtures/linear_probe")
    def classMappingYaml = file("${workflow.projectDir}/fixtures/linear_probe/class_mapping.yaml")

    if (!fixtureDir.exists()) {
        error """\
            Linear probe test fixtures not found: ${fixtureDir}
            Regenerate them with:
              python3 tests/fixtures/linear_probe/generate.py
            """.stripIndent()
    }

    ch_h5_features = Channel.fromList(
        (1..20).collect { i ->
            def slide_id = "lp_slide_${String.format('%04d', i)}"
            [ [slide_id: slide_id], 'resnet50', fixtureDir.resolve("${slide_id}.h5") ]
        }
    )

    ch_annotations = Channel.fromList(
        (1..20).collect { i ->
            def slide_id = "lp_slide_${String.format('%04d', i)}"
            [ [slide_id: slide_id], fixtureDir.resolve("${slide_id}.bmp") ]
        }
    )

    // Join on slide_id → (meta, model_type, h5, bmp) for MERGE_ANNOTATION_FEATURES
    ch_joined = ch_h5_features
        .map { meta, model_type, h5 -> tuple(meta.slide_id, meta, model_type, h5) }
        .combine(
            ch_annotations.map { meta, bmp -> tuple(meta.slide_id, bmp) },
            by: 0
        )
        .map { _sid, meta, model_type, h5, bmp -> tuple(meta, model_type, h5, bmp) }

    MERGE_ANNOTATION_FEATURES(ch_joined, classMappingYaml)

    MERGE_ANNOTATION_FEATURES.out
        | map { model_type, parquet -> tuple(model_type, parquet) }
        | groupTuple
        | STACK_ANNOTATION_FEATURES
        | LINEAR_PROBE_BENCHMARK
}
