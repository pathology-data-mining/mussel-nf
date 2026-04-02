#!/usr/bin/env bash
# Integration test for mussel-nf.
# Runs the pipeline with both workflow modes and validates outputs.
# Also tests WebDataset shard output (flat and oncotree-grouped).
#
# Usage:
#   ./tests/run_integration_test.sh
#
# Environment variables:
#   MUSSEL_REPO       Path to the Mussel source repository.
#                     Default: /gpfs/mskmind_ess/limr/repos/Mussel
#   MUSSEL_TEST_SLIDE Absolute path to the test WSI.
#                     Default: ${MUSSEL_REPO}/tests/testdata/948176.svs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Resolve test slide ────────────────────────────────────────────────────────

MUSSEL_REPO="${MUSSEL_REPO:-/gpfs/mskmind_ess/limr/repos/Mussel}"
SLIDE="${MUSSEL_TEST_SLIDE:-${MUSSEL_REPO}/tests/testdata/948176.svs}"
SLIDE_ID="$(basename "${SLIDE%.*}")"   # e.g. 948176

if [[ ! -f "$SLIDE" ]]; then
    echo "ERROR: Test slide not found at $SLIDE"
    echo "Override with:  MUSSEL_TEST_SLIDE=/path/to/slide.svs $0"
    echo "Or set:         MUSSEL_REPO=/path/to/Mussel  (expects tests/testdata/948176.svs)"
    exit 1
fi

# ── Generate test CSV files at runtime ───────────────────────────────────────
# CSVs contain absolute paths so nf-schema can validate them on startup.
# Generated into a temp directory so no absolute path lives in the repo.

TMPDIR_CSV="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_CSV"' EXIT

CSV_PLAIN="${TMPDIR_CSV}/test.csv"
CSV_ONCOTREE="${TMPDIR_CSV}/test_oncotree.csv"

printf 'slide_id,slide_path\n%s,%s\n'                     "$SLIDE_ID" "$SLIDE"        > "$CSV_PLAIN"
printf 'slide_id,slide_path,oncotree_code\n%s,%s,BRCA\n'  "$SLIDE_ID" "$SLIDE"        > "$CSV_ONCOTREE"

echo "==> Test slide : $SLIDE  (id: $SLIDE_ID)"

cd "$PROJECT_DIR"

# ── Helper: run pipeline then validate with validate.nf ───────────────────────
# Usage: run_and_validate <nf_profile> <results_dir> <samples_csv>

run_and_validate() {
    local profile="$1"
    local results_dir="$2"
    local samples_csv="$3"

    echo ""
    echo "==> Running pipeline  (profile: ${profile})"
    nextflow run main.nf \
        -profile "${profile}" \
        --samples_csv "$samples_csv"

    local manifest
    manifest=$(find "$results_dir" -maxdepth 1 -name "manifest-*.csv" 2>/dev/null | sort | tail -1)

    if [[ -z "$manifest" ]]; then
        echo "FAIL [${profile}]: No manifest CSV found in $results_dir"
        return 1
    fi

    echo "==> Validating outputs (profile: ${profile})"
    nextflow run validate.nf \
        --manifest_csv "$manifest" \
        --results_dir  "$results_dir"

    echo "PASS [${profile}]"
}

# ── Helper: run WDS pipeline then validate shards via validate.nf ─────────────
# All validation logic lives in validate.nf — profiles supply outdir + wds params.
# Usage: run_and_validate_wds <nf_profile> <samples_csv>

run_and_validate_wds() {
    local profile="$1"
    local samples_csv="$2"

    echo ""
    echo "==> Running pipeline  (profile: ${profile})"
    nextflow run main.nf \
        -profile "${profile}" \
        --samples_csv "$samples_csv"

    echo "==> Validating WDS shards (profile: ${profile})"
    nextflow run validate.nf \
        -profile "${profile}"

    echo "PASS [${profile}]"
}

# ── Run both standard workflow modes ─────────────────────────────────────────

run_and_validate test          "$SCRIPT_DIR/results"          "$CSV_PLAIN"
run_and_validate test_two_step "$SCRIPT_DIR/results_two_step" "$CSV_PLAIN"

# ── Run WDS shard tests ───────────────────────────────────────────────────────

run_and_validate_wds test_wds         "$CSV_PLAIN"
run_and_validate_wds test_wds_grouped "$CSV_ONCOTREE"

echo ""
echo "PASS: All integration tests succeeded."
