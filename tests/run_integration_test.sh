#!/usr/bin/env bash
# Integration test for mussel-nf.
# Runs the pipeline with both workflow modes and validates outputs.
#
# Usage: ./tests/run_integration_test.sh [extra nextflow args]
#
# Prerequisites:
#   - tests/data/1079807.svs must exist (copy with:
#       cp /gpfs/mskmind_emc/data_large/pathology/BR_16-512/slides/1079807.svs tests/data/)
#   - conda environment or container with Mussel must be available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SLIDE="$SCRIPT_DIR/data/1079807.svs"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if [[ ! -f "$SLIDE" ]]; then
    echo "ERROR: Test slide not found at $SLIDE"
    echo "Copy it with:"
    echo "  cp /gpfs/mskmind_emc/data_large/pathology/BR_16-512/slides/1079807.svs $SCRIPT_DIR/data/"
    exit 1
fi

cd "$PROJECT_DIR"

# ── Helper: run one profile and validate its outputs ─────────────────────────

run_and_validate() {
    local profile="$1"
    local results_dir="$2"

    echo ""
    echo "==> Running mussel-nf integration test (profile: ${profile})"
    nextflow run main.nf -profile "${profile}" "$@"

    echo ""
    echo "==> Validating outputs for profile: ${profile}"

    local manifest
    manifest=$(find "$results_dir" -maxdepth 1 -name "manifest-*.csv" 2>/dev/null | sort | tail -1)

    if [[ -z "$manifest" ]]; then
        echo "FAIL [${profile}]: No manifest CSV found in $results_dir"
        return 1
    fi

    local pt_file
    pt_file=$(find "$results_dir/features" -name "*.features.pt" 2>/dev/null | head -1)

    if [[ -z "$pt_file" ]]; then
        echo "FAIL [${profile}]: No .features.pt files found under $results_dir/features"
        return 1
    fi

    echo "==> Running validate.nf (profile: ${profile}, manifest: $manifest)"
    nextflow run validate.nf \
        --manifest_csv "$manifest" \
        --results_dir  "$results_dir"

    echo "PASS [${profile}]"
}

# ── Run both workflow modes ───────────────────────────────────────────────────

run_and_validate test          "$SCRIPT_DIR/results"
run_and_validate test_two_step "$SCRIPT_DIR/results_two_step"

echo ""
echo "PASS: All integration tests succeeded."
