#!/usr/bin/env bash
# Integration test for mussel-nf.
# Runs the pipeline with both workflow modes and validates outputs.
# Also tests WebDataset shard output (flat and oncotree-grouped).
#
# Usage:
#   ./tests/run_integration_test.sh [extra nextflow args]
#
# Environment variables:
#   MUSSEL_REPO       Path to the Mussel source repository.
#                     Default: /gpfs/mskmind_ess/limr/repos/Mussel
#   MUSSEL_TEST_SLIDE Absolute path to the test WSI.
#                     Default: ${MUSSEL_REPO}/tests/testdata/948176.svs
#
# The test CSVs are generated at runtime into a temp directory so no
# absolute path is baked into committed files.

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

# ── Generate test CSV files into a temp directory ─────────────────────────────
# Nextflow nf-schema validates file-path columns at startup, so the CSVs must
# contain real paths — we produce them at run-time to avoid committing absolute
# paths into the repository.

TMPDIR_CSV="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_CSV"' EXIT

CSV_PLAIN="${TMPDIR_CSV}/test.csv"
CSV_ONCOTREE="${TMPDIR_CSV}/test_oncotree.csv"

printf 'slide_id,slide_path\n%s,%s\n' "$SLIDE_ID" "$SLIDE" > "$CSV_PLAIN"
printf 'slide_id,slide_path,oncotree_code\n%s,%s,BRCA\n' "$SLIDE_ID" "$SLIDE" > "$CSV_ONCOTREE"

echo "==> Test slide:  $SLIDE"
echo "==> Slide ID:    $SLIDE_ID"
echo "==> Plain CSV:   $CSV_PLAIN"
echo "==> Oncotree CSV:$CSV_ONCOTREE"

cd "$PROJECT_DIR"

# ── Helper: run one profile and validate its outputs ─────────────────────────

run_and_validate() {
    local profile="$1"
    local results_dir="$2"
    local samples_csv="$3"

    echo ""
    echo "==> Running mussel-nf integration test (profile: ${profile})"
    nextflow run main.nf \
        -profile "${profile}" \
        --samples_csv "$samples_csv"

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

# ── Helper: validate WebDataset shards ───────────────────────────────────────
#
# validate_wds_shards <results_dir> <model_type> <group_name> <slide_id>

validate_wds_shards() {
    local results_dir="$1"
    local model_type="$2"
    local group_name="$3"
    local slide_id="$4"

    local shard_dir="${results_dir}/wds/${model_type}/${group_name}"

    echo ""
    echo "==> Validating WDS shards in ${shard_dir} (slide: ${slide_id})"

    # 1. At least one shard tar must exist
    local shards=()
    while IFS= read -r -d '' f; do
        shards+=("$f")
    done < <(find "$shard_dir" -maxdepth 1 -name "shard-*.tar" -print0 2>/dev/null)

    if [[ ${#shards[@]} -eq 0 ]]; then
        echo "FAIL: No shard-*.tar files found in ${shard_dir}"
        return 1
    fi
    echo "  Found ${#shards[@]} shard(s)"

    # 2. The expected slide must be present in the shards (as <slide_id>.pt)
    local found=0
    for shard in "${shards[@]}"; do
        if python3 - "$shard" "$slide_id" <<'PYEOF'
import sys, tarfile
shard_path, slide_id = sys.argv[1], sys.argv[2]
with tarfile.open(shard_path) as t:
    names = t.getnames()
if f"{slide_id}.pt" not in names:
    print(f"  {shard_path}: members={names}")
    sys.exit(1)
print(f"  {shard_path}: found {slide_id}.pt  OK")
PYEOF
        then
            found=1
            break
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "FAIL: ${slide_id}.pt not found in any shard under ${shard_dir}"
        return 1
    fi

    echo "PASS [WDS shards: ${model_type}/${group_name}]"
}

# ── Run both workflow modes ───────────────────────────────────────────────────

run_and_validate test          "$SCRIPT_DIR/results"          "$CSV_PLAIN"
run_and_validate test_two_step "$SCRIPT_DIR/results_two_step" "$CSV_PLAIN"

# ── Run WDS shard tests ───────────────────────────────────────────────────────

echo ""
echo "==> Running WDS flat-shard test (group_by_oncotree=false)"
nextflow run main.nf \
    -profile test_wds \
    --samples_csv "$CSV_PLAIN"

validate_wds_shards \
    "$SCRIPT_DIR/results_wds" \
    "resnet50" \
    "all" \
    "$SLIDE_ID"

echo ""
echo "==> Running WDS grouped-shard test (group_by_oncotree=true, oncotree_code=BRCA)"
nextflow run main.nf \
    -profile test_wds_grouped \
    --samples_csv "$CSV_ONCOTREE"

validate_wds_shards \
    "$SCRIPT_DIR/results_wds_grouped" \
    "resnet50" \
    "BRCA" \
    "$SLIDE_ID"

echo ""
echo "PASS: All integration tests succeeded."
