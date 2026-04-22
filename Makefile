# mussel-nf integration tests
#
# Usage:
#   make test                          # run all tests (local executor)
#   make test-standard                 # one-step workflow test only
#   make test-two-step                 # two-step workflow test only
#   make test-wds                      # WDS flat-sharding test only
#   make test-wds-grouped              # WDS oncotree-grouped sharding test only
#   make test-multi-slide              # multi-slide sample aggregation test only
#   make test-stub                     # stub workflow test (no GPU, CI-friendly)
#   make test-stub-all                 # all stub tests (no GPU)
#
#   make test          PROFILES=conda            # add extra Nextflow profiles
#   make test          PROFILES=slurm,cluster
#   make test-standard NXF_ARGS="-resume"        # extra Nextflow CLI args
#
# The test samplesheets (tests/test.csv, tests/test_oncotree.csv) are gitignored
# and generated from MUSSEL_TEST_SLIDE before each run.
# Override the slide path via the env var:
#   MUSSEL_TEST_SLIDE=/path/to/slide.svs make test

PROFILES  ?=
NXF_ARGS  ?=

# Path to the test SVS slide. Defaults to the slide vendored in tests/testdata/.
# Override if needed: MUSSEL_TEST_SLIDE=/path/to/other.svs make test
MUSSEL_TEST_SLIDE ?= $(CURDIR)/tests/testdata/948176.svs
SLIDE_ID          := 948176

nf_test   := nf-test
# Pass extra profiles with the '+' prefix so they compose with the test profile.
nf_flags  := $(if $(PROFILES),--profile +$(PROFILES),) $(NXF_ARGS)

# Generated test samplesheets — created from MUSSEL_TEST_SLIDE at test time.
tests/test.csv:
	@printf 'slide_id,slide_path\n$(SLIDE_ID),$(MUSSEL_TEST_SLIDE)\n' > $@

tests/test_oncotree.csv:
	@printf 'slide_id,slide_path,oncotree_code\n$(SLIDE_ID),$(MUSSEL_TEST_SLIDE),BRCA\n' > $@

MUSSEL_TEST_SLIDE_B ?= $(CURDIR)/tests/testdata/948176_b.svs
tests/test_multi_slide.csv:
	@printf 'slide_id,slide_path,sample_id\n948176_A,$(MUSSEL_TEST_SLIDE),PATIENT_STUB\n948176_B,$(MUSSEL_TEST_SLIDE_B),PATIENT_STUB\n' > $@

.PHONY: test test-standard test-two-step test-wds test-wds-grouped test-multi-slide \
        test-stub test-stub-two-step test-stub-filter test-stub-wds test-stub-wds-grouped \
        test-stub-clip test-stub-multi-slide test-stub-all help

test: test-standard test-two-step test-wds test-wds-grouped test-multi-slide

test-standard: tests/test.csv
	$(nf_test) test tests/pipeline.nf.test $(nf_flags)

test-two-step: tests/test.csv
	$(nf_test) test tests/pipeline_two_step.nf.test $(nf_flags)

test-wds: tests/test.csv
	$(nf_test) test tests/pipeline_wds.nf.test $(nf_flags)

test-wds-grouped: tests/test_oncotree.csv
	$(nf_test) test tests/pipeline_wds_grouped.nf.test $(nf_flags)

test-multi-slide:
	$(nf_test) test tests/multi_slide.nf.test $(nf_flags)

test-stub: tests/test.csv
	$(nf_test) test tests/pipeline_stub.nf.test $(nf_flags)

test-stub-two-step: tests/test.csv
	$(nf_test) test tests/pipeline_stub_two_step.nf.test $(nf_flags)

test-stub-filter: tests/test.csv
	$(nf_test) test tests/pipeline_stub_filter.nf.test $(nf_flags)

test-stub-wds: tests/test.csv
	$(nf_test) test tests/pipeline_stub_wds.nf.test $(nf_flags)

test-stub-wds-grouped: tests/test_oncotree.csv
	$(nf_test) test tests/pipeline_stub_wds_grouped.nf.test $(nf_flags)

test-stub-clip: tests/test_oncotree.csv
	$(nf_test) test tests/pipeline_stub_clip.nf.test $(nf_flags)

test-stub-multi-slide: tests/test_multi_slide.csv
	$(nf_test) test tests/pipeline_stub_multi_slide.nf.test $(nf_flags)

test-stub-all: tests/test.csv tests/test_oncotree.csv tests/test_multi_slide.csv
	$(nf_test) test \
	  tests/pipeline_stub.nf.test \
	  tests/pipeline_stub_two_step.nf.test \
	  tests/pipeline_stub_filter.nf.test \
	  tests/pipeline_stub_wds.nf.test \
	  tests/pipeline_stub_wds_grouped.nf.test \
	  tests/pipeline_stub_clip.nf.test \
	  tests/pipeline_stub_multi_slide.nf.test \
	  $(nf_flags)

help:
	@echo "mussel-nf integration tests"
	@echo ""
	@echo "Targets:"
	@echo "  make test               run all integration tests"
	@echo "  make test-standard      one-step workflow (main.nf -profile test)"
	@echo "  make test-two-step      two-step workflow (main.nf -profile test_two_step)"
	@echo "  make test-wds           WDS flat sharding (main.nf -profile test_wds)"
	@echo "  make test-wds-grouped   WDS per-oncotree sharding (main.nf -profile test_wds_grouped)"
	@echo "  make test-multi-slide   multi-slide aggregation (main.nf -profile test_multi_slide)"
	@echo "  make test-stub              one-step stub (no GPU)"
	@echo "  make test-stub-two-step     two-step stub"
	@echo "  make test-stub-filter       two-step + filter-tiles stub"
	@echo "  make test-stub-wds          one-step + WDS sharding stub"
	@echo "  make test-stub-wds-grouped  one-step + WDS grouped stub"
	@echo "  make test-stub-clip         one-step + CLIP annotation stub"
	@echo "  make test-stub-multi-slide  multi-slide aggregation stub"
	@echo "  make test-stub-all          all stub tests"
	@echo ""
	@echo "Variables:"
	@echo "  MUSSEL_TEST_SLIDE=<path>  path to a test SVS slide (default: tests/testdata/948176.svs)"
	@echo "  PROFILES=<profiles>       extra Nextflow profiles, comma-separated"
	@echo "  NXF_ARGS=<args>           extra Nextflow CLI arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make test MUSSEL_TEST_SLIDE=/path/to/slide.svs"
	@echo "  make test PROFILES=conda"
	@echo "  make test PROFILES=slurm,cluster"
	@echo "  make test-wds NXF_ARGS=-resume"
