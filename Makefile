# mussel-nf integration tests
#
# Usage:
#   make test                          # run all tests (local executor)
#   make test-standard                 # one-step workflow test only
#   make test-two-step                 # two-step workflow test only
#   make test-wds                      # WDS flat-sharding test only
#   make test-wds-grouped              # WDS oncotree-grouped sharding test only
#
#   make test          PROFILES=conda            # add extra Nextflow profiles
#   make test          PROFILES=slurm,cluster
#   make test-standard NXF_ARGS="-resume"        # extra Nextflow CLI args

PROFILES  ?=
NXF_ARGS  ?=

nf_test   := bin/nf-test
# Pass extra profiles with the '+' prefix so they compose with the test profile.
nf_flags  := $(if $(PROFILES),--profile +$(PROFILES),) $(NXF_ARGS)

.PHONY: test test-standard test-two-step test-wds test-wds-grouped help

test: test-standard test-two-step test-wds test-wds-grouped

test-standard:
	$(nf_test) test tests/pipeline.nf.test $(nf_flags)

test-two-step:
	$(nf_test) test tests/pipeline_two_step.nf.test $(nf_flags)

test-wds:
	$(nf_test) test tests/pipeline_wds.nf.test $(nf_flags)

test-wds-grouped:
	$(nf_test) test tests/pipeline_wds_grouped.nf.test $(nf_flags)

help:
	@echo "mussel-nf integration tests"
	@echo ""
	@echo "Targets:"
	@echo "  make test               run all integration tests"
	@echo "  make test-standard      one-step workflow (main.nf -profile test)"
	@echo "  make test-two-step      two-step workflow (main.nf -profile test_two_step)"
	@echo "  make test-wds           WDS flat sharding (main.nf -profile test_wds)"
	@echo "  make test-wds-grouped   WDS per-oncotree sharding (main.nf -profile test_wds_grouped)"
	@echo ""
	@echo "Variables:"
	@echo "  PROFILES=<profiles>   extra Nextflow profiles, comma-separated"
	@echo "  NXF_ARGS=<args>       extra Nextflow CLI arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make test PROFILES=conda"
	@echo "  make test PROFILES=slurm,cluster"
	@echo "  make test-wds NXF_ARGS=-resume"
