from dataclasses import dataclass

import pytest

from mussel_dispatcher.contract import (
    ContractError,
    effective_params,
    execution_profile,
    semantic_contract_hash,
)


def _params(*, slide_batch_size=1):
    return {
        "tiling": {"seg_model": "neural", "remove_penmarks": True},
        "featurize": {
            "model_types": ["hoptimus1", "optimus", "titan_slide"],
            "slide_batch_size": slide_batch_size,
            "slide_max_patches": 4096,
            "model_precision_overrides": {"hoptimus1": "bfloat16"},
        },
        "use_one_step_workflow": False,
        "publish_mode": "copy",
    }


def test_semantic_hash_ignores_target_slide_batch_tuning():
    assert semantic_contract_hash(_params(slide_batch_size=1)) == semantic_contract_hash(
        _params(slide_batch_size=8)
    )


def test_effective_params_applies_only_target_tuning():
    effective = effective_params(_params(slide_batch_size=1), {"featurize.slide_batch_size": 8})
    assert effective["featurize"]["slide_batch_size"] == 8
    with pytest.raises(ContractError, match="not allowlisted"):
        effective_params(_params(), {"tiling.seg_model": "preset"})


def test_effective_params_requires_explicit_batch_size():
    params = _params()
    del params["featurize"]["slide_batch_size"]
    with pytest.raises(ContractError, match="slide_batch_size"):
        effective_params(params)
    assert effective_params(params, {"featurize.slide_batch_size": 1})["featurize"][
        "slide_batch_size"
    ] == 1


@dataclass
class Target:
    name: str = "hpc"
    profile_name: str = "hpc"
    remote_repo_dir: str = "/remote/runtime/mussel-nf"
    remote_outdir: str = "/remote/results"
    nextflow_profiles: str = "cluster,remote_slurm,conda"
    nextflow_config: str = "/remote/runtime/mussel-nf/dispatcher/remote_nextflow.config"
    driver_partition: str = "cpu"
    conda_env_path: str = "/remote/conda/mussel"
    parameter_overrides: dict = None
    nextflow_params_file: str = ""
    secrets_env_file: str = ""
    nextflow_cli_args: list = None


def test_execution_profiles_retain_host_runtime_differences():
    target = Target(parameter_overrides={"featurize.slide_batch_size": 1}, nextflow_cli_args=[])
    profile = execution_profile(object(), target=target)
    assert profile["profile_name"] == "hpc"
    assert profile["nextflow_profiles"].endswith("conda")
    assert profile["driver_partition"] == "cpu"
    assert "parameter_overrides" not in profile
    assert "nextflow_params_file" not in profile


def test_semantic_hash_covers_the_derived_tile_grid():
    # featurize.grid_from_primary changes which tiles TITAN reads (e.g. TITAN's 512 px grid).
    params = _params()
    derived = _params()
    derived["featurize"]["grid_from_primary"] = ["titan_slide"]
    assert semantic_contract_hash(params) != semantic_contract_hash(derived)
