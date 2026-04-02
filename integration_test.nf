/**
 * integration_test.nf — end-to-end integration test for mussel-nf.
 *
 * Runs the full MUSSEL pipeline (tessellation + feature extraction + optional
 * WDS sharding) and immediately validates the outputs through the same
 * Nextflow channels — no filesystem scanning required.
 *
 * Because all configuration comes from params (set by profiles), tests can be
 * composed freely with any executor or resource profile:
 *
 *   nextflow run integration_test.nf -profile test_wds
 *   nextflow run integration_test.nf -profile test_wds,cluster
 *   nextflow run integration_test.nf -profile test_wds_grouped,slurm,apptainer
 *
 * The only required runtime argument is --samples_csv, pointing to a CSV
 * with columns slide_id, slide_path (and optionally oncotree_code).
 *
 * See tests/run_integration_test.sh for an example driver that generates the
 * CSV at runtime and runs all four test profiles in sequence.
 */

include { validateParameters; samplesheetToList } from 'plugin/nf-schema'

include { MUSSEL } from './modules/mussel'

// Reuse the validate processes already defined in validate.nf.
// They now exit 1 on any error, making the pipeline task fail immediately.
include { VALIDATE_PT; VALIDATE_H5; VALIDATE_WDS_SHARDS } from './validate'

validateParameters()

// Batch size for validation tasks (independent of featurize.batch_size)
params.validation_batch_size = 500

workflow {
    ch_samples = Channel.fromList(
        samplesheetToList(params.samples_csv, "assets/schema_input.json")
    )

    ch_annotations = Channel.empty()
    if (params.linear_probe?.annotations_csv) {
        ch_annotations = Channel.fromList(
            samplesheetToList(params.linear_probe.annotations_csv, "assets/schema_annotations.json")
        )
    }

    // ── Run the full pipeline ─────────────────────────────────────────────────
    MUSSEL(ch_samples, ch_annotations)

    // ── Validate .pt slide-feature files ─────────────────────────────────────
    VALIDATE_PT(
        MUSSEL.out.pt
            .map { meta, model_type, f -> f }
            .buffer(size: params.validation_batch_size, remainder: true)
    )

    // ── Validate WDS shards (only when wds.enabled) ───────────────────────────
    if (params.wds?.enabled) {
        VALIDATE_WDS_SHARDS(
            MUSSEL.out.wds_shards
                .flatMap { group, model_type, tars ->
                    tars instanceof List ? tars : [tars]
                }
                .buffer(size: params.validation_batch_size, remainder: true)
        )
    }
}
