"""Backend and binary compatibility policy for the Neural Foundry local worker."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


MANIFEST_SCHEMA_VERSION = "neural-forge-worker-manifest.v2"
NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION = "ida-native-execution-request.v1"
BINARY_HASH_ALGORITHM = "sha256"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TRAINING_MODES = {"from_scratch", "fine_tune", "resume"}


@dataclass(frozen=True)
class BackendPolicy:
    backend_id: str
    v3_binary_name: str
    worker_binary_name: str
    artifact_id: str
    supported_precisions: tuple[str, ...]
    supported_optimizers: tuple[str, ...]
    supported_attention: tuple[str, ...]
    smoke_only: bool


BACKEND_POLICIES: dict[str, BackendPolicy] = {
    "cuda": BackendPolicy(
        "cuda", "ida_native_train", "ida_native_train", "canopy-foundry-cuda-v1",
        ("fp32", "legacy_bf16", "legacy_fp8"),
        ("lion",), ("scalar_flash",), False,
    ),
    "opencl": BackendPolicy(
        "opencl", "ida_native_opencl_train", "ida_native_opencl_train", "canopy-foundry-opencl-smoke-v1",
        ("fp32",), (), ("scalar_flash",), True,
    ),
    "cpu": BackendPolicy(
        "cpu", "ida_native_cpu_train", "ida_native_cpu_train", "canopy-foundry-cpu-smoke-v1",
        ("fp32",), (), ("scalar_flash",), True,
    ),
}


# This is a private deployment contract, not an extension of the public
# Foundry catalog.  It records the exact settings that a deployment has
# elected to run from the V3 native engine.  The public catalog may continue
# to expose only the stable scalar/Lion route.
V3_NATIVE_PROFILE_CONTRACTS: dict[str, dict[str, str]] = {
    "edge-full": {
        "backend": "cuda",
        "precision_profile": "legacy_bf16",
        "optimizer_type": "adamw",
        "attention_backend": "hopper_wgmma_packed_fp4",
    },
    "ai": {
        "backend": "cuda",
        "precision_profile": "legacy_fp8",
        "optimizer_type": "adamw",
        "attention_backend": "hopper_wgmma_packed_fp4",
    },
    "edge-swift": {
        "backend": "cuda",
        "precision_profile": "legacy_bf16",
        "optimizer_type": "adamw",
        "attention_backend": "scalar_flash",
    },
    "moe": {
        "backend": "cuda",
        "precision_profile": "legacy_fp8",
        "optimizer_type": "adamw",
        "attention_backend": "scalar_flash",
    },
    "opencl-smoke": {
        "backend": "opencl",
        "precision_profile": "fp32",
        "optimizer_type": "adamw",
        "attention_backend": "scalar_flash",
    },
    "cpu-smoke": {
        "backend": "cpu",
        "precision_profile": "fp32",
        "optimizer_type": "adamw",
        "attention_backend": "scalar_flash",
    },
}


def _opaque_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value) or ".." in value:
        raise ValueError(f"{label} is invalid")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} is invalid")
    return value.lower()


def validate_v3_profile_contract(
    value: Any,
    profile_id: str,
    label: str = "deployment map native_execution",
) -> dict[str, str]:
    """Validate the deployment-owned V3 settings without widening public policy."""

    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    required = {
        "schema_version", "profile_id", "backend", "precision_profile",
        "optimizer_type", "attention_backend",
    }
    if set(value) != required or value.get("schema_version") != NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION:
        raise ValueError(f"{label} fields are invalid")
    profile_id = _opaque_id(profile_id, f"{label}.profile_id")
    if value.get("profile_id") != profile_id:
        raise ValueError(f"{label}.profile_id does not match Hub profile")
    expected = V3_NATIVE_PROFILE_CONTRACTS.get(profile_id)
    if expected is None:
        raise ValueError(f"{label}.profile_id is not a V3 profile")
    normalized = {
        "schema_version": NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION,
        "profile_id": profile_id,
        "backend": _opaque_id(value.get("backend"), f"{label}.backend"),
        "precision_profile": _opaque_id(value.get("precision_profile"), f"{label}.precision_profile"),
        "optimizer_type": _opaque_id(value.get("optimizer_type"), f"{label}.optimizer_type"),
        "attention_backend": _opaque_id(value.get("attention_backend"), f"{label}.attention_backend"),
    }
    if any(normalized[key] != expected[key] for key in expected):
        raise ValueError(f"{label} does not match the V3 profile contract")
    return normalized


def validate_v3_native_execution_request(
    value: Any,
    *,
    expected_run_id: str | None = None,
    expected_profile_id: str | None = None,
) -> dict[str, Any]:
    """Validate the private descriptor produced after canonical manifest validation."""

    if not isinstance(value, dict):
        raise ValueError("native execution request is invalid")
    required = {
        "schema_version", "run_id", "profile_id", "backend", "artifact_id",
        "binary_name", "binary_sha256", "trainer_version", "policy_version",
        "precision_profile", "optimizer_type", "attention_backend",
        "training_mode", "dataset_id", "resource_class", "max_steps",
        "timeout_seconds",
    }
    optional = {"base_model_id", "checkpoint_id", "evaluation_request_id", "model_contract_id"}
    if set(value) - required - optional or not required.issubset(value):
        raise ValueError("native execution request fields are invalid")
    if value["schema_version"] != NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION:
        raise ValueError("native execution request schema_version is invalid")
    run_id = _opaque_id(value["run_id"], "native execution request.run_id")
    if expected_run_id is not None and run_id != _opaque_id(expected_run_id, "expected run_id"):
        raise ValueError("native execution request run_id does not match job")
    profile_id = _opaque_id(value["profile_id"], "native execution request.profile_id")
    if expected_profile_id is not None and profile_id != _opaque_id(expected_profile_id, "expected profile_id"):
        raise ValueError("native execution request profile_id does not match deployment")
    contract = validate_v3_profile_contract({
        "schema_version": NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION,
        "profile_id": profile_id,
        "backend": value["backend"],
        "precision_profile": value["precision_profile"],
        "optimizer_type": value["optimizer_type"],
        "attention_backend": value["attention_backend"],
    }, profile_id)
    normalized: dict[str, Any] = {
        **contract,
        "run_id": run_id,
        "artifact_id": _opaque_id(value["artifact_id"], "native execution request.artifact_id"),
        "binary_name": _opaque_id(value["binary_name"], "native execution request.binary_name"),
        "binary_sha256": _hash(value["binary_sha256"], "native execution request.binary_sha256"),
        "trainer_version": _opaque_id(value["trainer_version"], "native execution request.trainer_version"),
        "policy_version": _opaque_id(value["policy_version"], "native execution request.policy_version"),
        "training_mode": value["training_mode"],
        "dataset_id": _opaque_id(value["dataset_id"], "native execution request.dataset_id"),
        "resource_class": _opaque_id(value["resource_class"], "native execution request.resource_class"),
        "max_steps": value["max_steps"],
        "timeout_seconds": value["timeout_seconds"],
    }
    if normalized["training_mode"] not in _TRAINING_MODES:
        raise ValueError("native execution request training_mode is invalid")
    if (
        isinstance(normalized["max_steps"], bool)
        or not isinstance(normalized["max_steps"], int)
        or normalized["max_steps"] < 1
        or isinstance(normalized["timeout_seconds"], bool)
        or not isinstance(normalized["timeout_seconds"], int)
        or normalized["timeout_seconds"] < 1
    ):
        raise ValueError("native execution request limits are invalid")
    for key in optional:
        if value.get(key) is not None:
            normalized[key] = _opaque_id(value[key], f"native execution request.{key}")
    return normalized


def project_v3_native_execution_request(
    manifest: dict[str, Any],
    *,
    profile_id: str,
    execution: dict[str, str],
    native_contract: dict[str, str],
) -> dict[str, Any]:
    """Project a validated canonical Hub manifest into V3's private descriptor."""

    request = manifest.get("request")
    policy = manifest.get("policy")
    if not isinstance(request, dict) or not isinstance(policy, dict):
        raise ValueError("canonical manifest request or policy is invalid")
    contract = validate_v3_profile_contract(native_contract, profile_id)
    if execution.get("backend") != contract["backend"]:
        raise ValueError("native execution backend does not match deployment contract")
    descriptor: dict[str, Any] = {
        **contract,
        "run_id": manifest.get("run_id"),
        "artifact_id": execution.get("artifact_id"),
        "binary_name": execution.get("binary_name"),
        "binary_sha256": execution.get("binary_sha256"),
        "trainer_version": policy.get("trainer_version"),
        "policy_version": policy.get("policy_version"),
        "training_mode": request.get("training_mode"),
        "dataset_id": request.get("dataset_id"),
        "resource_class": request.get("resource_class"),
        "max_steps": policy.get("max_steps"),
        "timeout_seconds": policy.get("timeout_seconds"),
    }
    for key in ("base_model_id", "checkpoint_id", "evaluation_request_id", "model_contract_id"):
        if request.get(key) is not None:
            descriptor[key] = request[key]
    return validate_v3_native_execution_request(
        descriptor,
        expected_run_id=manifest.get("run_id"),
        expected_profile_id=profile_id,
    )


def validate_binary_attestation(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("execution attestation must be an object")
    required = {"backend", "artifact_id", "binary_name", "binary_sha256"}
    if set(value) != required:
        raise ValueError("execution attestation fields are invalid")
    backend = value["backend"]
    if not isinstance(backend, str) or backend not in BACKEND_POLICIES:
        raise ValueError("execution backend is invalid")
    policy = BACKEND_POLICIES[backend]
    if value["artifact_id"] != policy.artifact_id:
        raise ValueError("execution artifact_id does not match backend")
    if value["binary_name"] not in {policy.v3_binary_name, policy.worker_binary_name}:
        raise ValueError("execution binary_name is not approved")
    digest = value["binary_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in digest
    ):
        raise ValueError("execution binary_sha256 is invalid")
    return {
        "backend": backend,
        "artifact_id": value["artifact_id"],
        "binary_name": value["binary_name"],
        "binary_sha256": digest.lower(),
    }
