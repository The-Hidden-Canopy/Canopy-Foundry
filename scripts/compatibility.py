"""Backend and binary compatibility policy for the Neural Foundry local worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MANIFEST_SCHEMA_VERSION = "neural-forge-worker-manifest.v2"
BINARY_HASH_ALGORITHM = "sha256"


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
