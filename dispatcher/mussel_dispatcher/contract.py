"""Portable scientific contracts and target-specific execution profiles.

The dispatcher runs the same scientific workflow on hosts that intentionally
have different paths, containers, Conda environments, and Slurm resources.
This module keeps those concerns separate and gives every batch a durable
identity for both parts of the configuration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


SEMANTIC_FEATURIZE_KEYS = (
    "model_types",
    "slide_max_patches",
    "model_precision_overrides",
    "grid_from_primary",
)
SEMANTIC_TOP_LEVEL_KEYS = (
    "tiling",
    "featurize",
    "use_one_step_workflow",
    "publish_mode",
)
TARGET_PARAMETER_OVERRIDES = {"featurize.slide_batch_size"}


class ContractError(ValueError):
    """Raised when a deployment contract is incomplete or malformed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_params_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ContractError(f"Parameters must be a YAML mapping: {path}")
    return data


def _copy_semantic_params(params: dict[str, Any]) -> dict[str, Any]:
    semantic = {
        key: copy.deepcopy(params[key])
        for key in SEMANTIC_TOP_LEVEL_KEYS
        if key in params
    }
    featurize = semantic.get("featurize")
    if isinstance(featurize, dict):
        semantic["featurize"] = {
            key: copy.deepcopy(featurize[key])
            for key in SEMANTIC_FEATURIZE_KEYS
            if key in featurize
        }
    return semantic


def validate_semantic_params(params: dict[str, Any]) -> None:
    tiling = params.get("tiling")
    featurize = params.get("featurize")
    if not isinstance(tiling, dict) or not tiling.get("seg_model"):
        raise ContractError("semantic contract requires tiling.seg_model")
    if not isinstance(featurize, dict):
        raise ContractError("semantic contract requires featurize")
    if not featurize.get("model_types"):
        raise ContractError("semantic contract requires featurize.model_types")
    if featurize.get("slide_max_patches") is None:
        raise ContractError(
            "semantic contract requires an explicit featurize.slide_max_patches limit"
        )


def semantic_contract(params: dict[str, Any]) -> dict[str, Any]:
    validate_semantic_params(params)
    return _copy_semantic_params(params)


def semantic_contract_hash(params: dict[str, Any]) -> str:
    return sha256_json(semantic_contract(params))


def _set_dotted(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    pieces = dotted_key.split(".")
    current = mapping
    for piece in pieces[:-1]:
        child = current.setdefault(piece, {})
        if not isinstance(child, dict):
            raise ContractError(f"Cannot override {dotted_key}: {piece} is not a mapping")
        current = child
    current[pieces[-1]] = value


def effective_params(
    params: dict[str, Any], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return canonical params plus allowlisted target tuning overrides."""
    validate_semantic_params(params)
    result = copy.deepcopy(params)
    for key, value in (overrides or {}).items():
        if key not in TARGET_PARAMETER_OVERRIDES:
            raise ContractError(f"Parameter override is not allowlisted: {key}")
        _set_dotted(result, key, value)
    if "featurize.slide_batch_size" not in (overrides or {}):
        # The canonical file must still make the target tuning explicit. This
        # prevents an omitted field from silently falling back to nextflow.config.
        if not isinstance(result.get("featurize"), dict) or result["featurize"].get("slide_batch_size") is None:
            raise ContractError("effective params require featurize.slide_batch_size")
    return result


def params_yaml(params: dict[str, Any]) -> str:
    return yaml.safe_dump(params, sort_keys=False, default_flow_style=False)


def _profile_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _profile_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_profile_value(v) for v in value]
    return value


def execution_profile(config: Any, target: Any | None = None) -> dict[str, Any]:
    """Build a hashable profile while excluding secrets and live state."""
    if target is not None:
        raw = _profile_value(target)
        name = getattr(target, "profile_name", "") or getattr(target, "name", "remote")
        raw["name"] = name
        raw.pop("secrets_env_file", None)
        raw.pop("nextflow_params_file", None)
        raw.pop("nextflow_cli_args", None)
        raw.pop("parameter_overrides", None)
        if isinstance(raw.get("env"), dict):
            raw["env"] = {
                key: value
                for key, value in raw["env"].items()
                if not any(
                    marker in str(key).upper()
                    for marker in ("SECRET", "TOKEN", "PASSWORD", "ACCESS_KEY", "PRIVATE_KEY")
                )
            }
        return raw

    raw = _profile_value(config)
    for key in (
        "config_path", "nextflow_params_file", "secrets_env_file", "postgres",
        "watchers", "post_batch_hooks", "dispatch_targets", "execution_pools",
        "parameter_overrides", "s3_access_key", "s3_secret_key", "nextflow_secrets",
    ):
        raw.pop(key, None)
    raw["name"] = getattr(config, "profile_name", "") or "local"
    return raw


def execution_profile_hash(config: Any, target: Any | None = None) -> str:
    return sha256_json(execution_profile(config, target))


def git_revision(repo_dir: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    return result.stdout.strip()


def tracked_source_manifest(repo_dir: str) -> dict[str, str]:
    """Hash all tracked files so dispatcher and agent execute one release."""
    result = subprocess.run(
        ["git", "-C", repo_dir, "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    paths = [item for item in result.stdout.decode().split("\0") if item]
    manifest: dict[str, str] = {}
    root = Path(repo_dir)
    for relative in paths:
        path = root / relative
        if path.is_file():
            manifest[relative] = sha256_bytes(path.read_bytes())
    return manifest


def release_manifest_hash(manifest: dict[str, str]) -> str:
    return sha256_json(manifest)


def build_contract(config: Any, target: Any | None = None) -> dict[str, Any]:
    params = load_params_file(config.nextflow_params_file)
    overrides = (
        getattr(target, "parameter_overrides", None)
        if target is not None
        else getattr(config, "parameter_overrides", None)
    ) or {}
    effective = effective_params(params, overrides)
    manifest = tracked_source_manifest(config.repo_dir)
    profile = execution_profile(config, target)
    return {
        "source_revision": git_revision(config.repo_dir),
        "release_manifest": manifest,
        "release_manifest_sha256": release_manifest_hash(manifest),
        "semantic_contract": semantic_contract(params),
        "semantic_contract_sha256": semantic_contract_hash(params),
        "execution_profile": profile,
        "execution_profile_sha256": sha256_json(profile),
        "effective_params": effective,
        "effective_params_yaml": params_yaml(effective),
        "effective_params_sha256": sha256_bytes(params_yaml(effective).encode()),
    }
