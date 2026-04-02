#!/usr/bin/env bash
# Driver for the mussel-nf integration tests (uses nf-test).
#
# Usage:
#   ./tests/run_integration_test.sh [nf-test options, e.g. --debug or --tag foo]
#
# Environment variables:
#   MUSSEL_REPO       Path to the Mussel source repository.
#                     Default: /gpfs/mskmind_ess/limr/repos/Mussel
#   MUSSEL_TEST_SLIDE Absolute path to the test WSI (overrides MUSSEL_REPO).
#                     Default: ${MUSSEL_REPO}/tests/testdata/948176.svs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MUSSEL_REPO="${MUSSEL_REPO:-/gpfs/mskmind_ess/limr/repos/Mussel}"
export MUSSEL_REPO

MUSSEL_TEST_SLIDE="${MUSSEL_TEST_SLIDE:-${MUSSEL_REPO}/tests/testdata/948176.svs}"
export MUSSEL_TEST_SLIDE

if [[ ! -f "$MUSSEL_TEST_SLIDE" ]]; then
    echo "ERROR: Test slide not found at $MUSSEL_TEST_SLIDE"
    echo "Override with:  MUSSEL_TEST_SLIDE=/path/to/slide.svs $0"
    echo "Or set:         MUSSEL_REPO=/path/to/Mussel  (expects tests/testdata/948176.svs)"
    exit 1
fi

echo "==> Test slide : $MUSSEL_TEST_SLIDE"

cd "$PROJECT_DIR"
nf-test test tests/main.nf.test "$@"
