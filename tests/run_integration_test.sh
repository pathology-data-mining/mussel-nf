#!/usr/bin/env bash
# Driver for the mussel-nf integration tests.
#
# Generates sample-sheet CSVs at runtime (so no absolute path lives in the
# repository), then runs integration_test.nf for each test profile.
# All pipeline logic and validation live in Nextflow -- this script only
# handles the slide pre-flight check and CSV generation.
#
# Usage:
#   ./tests/run_integration_test.sh
#
# Environment variables:
#   MUSSEL_REPO       Path to the Mussel source repository.
#                     Default: /gpfs/mskmind_ess/limr/repos/Mussel
#   MUSSEL_TEST_SLIDE Absolute path to the test WSI (overrides MUSSEL_REPO).
#                     Default: ${MUSSEL_REPO}/tests/testdata/948176.svs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# -- Resolve test slide -------------------------------------------------------

MUSSEL_REPO="${MUSSEL_REPO:-/gpfs/mskmind_ess/limr/repos/Mussel}"
SLIDE="${MUSSEL_TEST_SLIDE:-${MUSSEL_REPO}/tests/testdata/948176.svs}"
SLIDE_ID="$(basename "$SLIDE" .svs)"

if [[ ! -f "$SLIDE" ]]; then
    echo "ERROR: Test slide not found at $SLIDE"
    echo "Override with:  MUSSEL_TEST_SLIDE=/path/to/slide.svs $0"
    echo "Or set:         MUSSEL_REPO=/path/to/Mussel  (expects tests/testdata/948176.svs)"
    exit 1
fi

# -- Generate sample-sheet CSVs ----------------------------------------------

TMPDIR_CSV="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_CSV"' EXIT

CSV_PLAIN="${TMPDIR_CSV}/test.csv"
CSV_ONCOTREE="${TMPDIR_CSV}/test_oncotree.csv"

printf 'slide_id,slide_path\n%s,%s\n'                    "$SLIDE_ID" "$SLIDE" > "$CSV_PLAIN"
printf 'slide_id,slide_path,oncotree_code\n%s,%s,BRCA\n' "$SLIDE_ID" "$SLIDE" > "$CSV_ONCOTREE"

echo "==> Test slide : $SLIDE  (id: $SLIDE_ID)"

cd "$PROJECT_DIR"

# -- Run all test profiles ---------------------------------------------------
# All logic (pipeline + validation) lives in integration_test.nf.
# Profiles supply outdir, model settings, and wds options.

nextflow run integration_test.nf -profile test          --samples_csv "$CSV_PLAIN"
nextflow run integration_test.nf -profile test_two_step --samples_csv "$CSV_PLAIN"
nextflow run integration_test.nf -profile test_wds      --samples_csv "$CSV_PLAIN"
nextflow run integration_test.nf -profile test_wds_grouped --samples_csv "$CSV_ONCOTREE"

echo ""
echo "PASS: All integration tests succeeded."
