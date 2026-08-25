"""Validate the deployment-owned Hub-to-local Neural Foundry mapping.

The public capability catalog remains neutral.  A deployment map contains
only opaque IDs, approved external execution identities, trainer-version
allowlists, and local device numbers.  It must not contain paths, commands,
model bytes, credentials, or caller-controlled training settings.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

try:
    from scripts.compatibility import BACKEND_POLICIES
    from scripts.model_contracts import validate_model_contract_id
    from scripts.private_runtime import validate_optional_private_runtime
except ModuleNotFoundError:
    from compatibility import BACKEND_POLICIES
    from model_contracts import validate_model_contract_id
    from private_runtime import validate_optional_private_runtime


DEPLOYMENT_MAP_SCHEMA_VERSION = "neural-foundry-deployment-map.v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_RESERVED_IDS = {"__proto__", "prototype", "constructor"}


class DeploymentMapError(ValueError):
    """A deployment-owned mapping failed closed."""


def _record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentMapError(f"{label} is invalid")
    return value


def _id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _ID_RE.fullmatch(value)
        or ".." in value
        or value.lower() in _RESERVED_IDS
    ):
        raise DeploymentMapError(f"{label} is invalid")
    return value


def _id_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DeploymentMapError(f"{label} is invalid")
    result = [_id(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise DeploymentMapError(f"{label} contains duplicates")
    return result


def _execution(value: Any, label: str) -> dict[str, Any]:
    entry = _record(value, label)
    required = {"backend", "artifact_id", "binary_names"}
    if set(entry) != required:
        raise DeploymentMapError(f"{label} fields are invalid")
    backend = _id(entry["backend"], f"{label}.backend")
    if backend not in BACKEND_POLICIES:
        raise DeploymentMapError(f"{label}.backend is invalid")
    return {
        "backend": backend,
        "artifact_id": _id(entry["artifact_id"], f"{label}.artifact_id"),
        "binary_names": _id_list(entry["binary_names"], f"{label}.binary_names"),
    }


def load_deployment_map(path: Path) -> dict[str, Any]:
    """Load and normalize a deployment-only external-to-local mapping."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentMapError("deployment map is unavailable") from exc
    payload = _record(payload, "deployment map")
    required = {"schema_version", "hub_profiles", "resource_classes"}
    if set(payload) != required or payload["schema_version"] != DEPLOYMENT_MAP_SCHEMA_VERSION:
        raise DeploymentMapError("deployment map fields are invalid")

    profiles = _record(payload["hub_profiles"], "deployment map hub_profiles")
    if not profiles:
        raise DeploymentMapError("deployment map hub_profiles is empty")
    normalized_profiles: dict[str, Any] = {}
    local_profile_ids: set[str] = set()
    for profile_id, raw in profiles.items():
        profile_id = _id(profile_id, "Hub profile id")
        entry = _record(raw, f"Hub profile {profile_id}")
        required_profile = {"local_profile_id", "trainer_versions", "execution"}
        allowed_profile = required_profile | {"model_contract_ids", "private_runtime"}
        if not required_profile.issubset(entry) or set(entry) - allowed_profile:
            raise DeploymentMapError(f"Hub profile {profile_id} fields are invalid")
        local_profile_id = _id(
            entry["local_profile_id"],
            f"Hub profile {profile_id}.local_profile_id",
        )
        if local_profile_id in local_profile_ids:
            raise DeploymentMapError("deployment map local profiles must be distinct")
        local_profile_ids.add(local_profile_id)
        normalized = {
            "local_profile_id": local_profile_id,
            "trainer_versions": _id_list(
                entry["trainer_versions"],
                f"Hub profile {profile_id}.trainer_versions",
            ),
            "execution": _execution(entry["execution"], f"Hub profile {profile_id}.execution"),
        }
        if "model_contract_ids" in entry:
            try:
                contracts = [
                    validate_model_contract_id(item, f"Hub profile {profile_id}.model_contract_ids[{index}]")
                    for index, item in enumerate(_id_list(entry["model_contract_ids"], f"Hub profile {profile_id}.model_contract_ids"))
                ]
            except ValueError as exc:
                raise DeploymentMapError(str(exc)) from exc
            normalized["model_contract_ids"] = contracts
        if "private_runtime" in entry:
            try:
                normalized["private_runtime"] = validate_optional_private_runtime(
                    entry["private_runtime"], f"Hub profile {profile_id}.private_runtime"
                )
            except ValueError as exc:
                raise DeploymentMapError(str(exc)) from exc
        normalized_profiles[profile_id] = normalized

    resources = _record(payload["resource_classes"], "deployment map resource_classes")
    if not resources:
        raise DeploymentMapError("deployment map resource_classes is empty")
    normalized_resources: dict[str, Any] = {}
    for resource_id, raw in resources.items():
        resource_id = _id(resource_id, "resource class id")
        entry = _record(raw, f"resource class {resource_id}")
        if set(entry) != {"device"}:
            raise DeploymentMapError(f"resource class {resource_id} fields are invalid")
        device = entry["device"]
        if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device <= 255:
            raise DeploymentMapError(f"resource class {resource_id}.device is invalid")
        normalized_resources[resource_id] = {"device": device}

    return {
        "schema_version": DEPLOYMENT_MAP_SCHEMA_VERSION,
        "hub_profiles": normalized_profiles,
        "resource_classes": normalized_resources,
    }


def validate_model_contract_binding(
    contract_id: Any,
    profile_id: str,
    profile: dict[str, Any],
) -> str | None:
    """Require a Hub-selected contract to be allowed by the local map."""

    if contract_id is None:
        return None
    try:
        normalized = validate_model_contract_id(contract_id, "worker manifest model_contract_id")
    except ValueError as exc:
        raise DeploymentMapError(str(exc)) from exc
    allowed = profile.get("model_contract_ids")
    if allowed is not None and normalized not in allowed:
        raise DeploymentMapError(f"model contract does not match profile {profile_id}")
    return normalized


def validate_private_runtime_binding(
    value: Any,
    profile_id: str,
    profile: dict[str, Any],
) -> dict[str, str] | None:
    """Match an opaque Hub package reference to the deployment-owned map."""

    try:
        requested = validate_optional_private_runtime(value, "worker manifest private runtime")
        mapped = validate_optional_private_runtime(profile.get("private_runtime"), f"Hub profile {profile_id}.private_runtime")
    except ValueError as exc:
        raise DeploymentMapError(str(exc)) from exc
    if requested != mapped:
        raise DeploymentMapError(f"private runtime does not match profile {profile_id}")
    return requested


def validate_execution_attestation(
    value: Any,
    profile_id: str,
    profile: dict[str, Any],
) -> dict[str, str]:
    """Validate the Hub attestation against one mapped profile."""

    entry = _record(value, "worker manifest execution")
    required = {"backend", "artifact_id", "binary_name", "binary_sha256"}
    if set(entry) != required:
        raise DeploymentMapError("worker manifest execution fields are invalid")
    backend = _id(entry["backend"], "worker manifest execution.backend")
    artifact_id = _id(entry["artifact_id"], "worker manifest execution.artifact_id")
    binary_name = _id(entry["binary_name"], "worker manifest execution.binary_name")
    digest = entry["binary_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise DeploymentMapError("worker manifest execution.binary_sha256 is invalid")
    expected = profile["execution"]
    if (
        backend != expected["backend"]
        or artifact_id != expected["artifact_id"]
        or binary_name not in expected["binary_names"]
    ):
        raise DeploymentMapError(f"worker manifest execution does not match profile {profile_id}")
    return {
        "backend": backend,
        "artifact_id": artifact_id,
        "binary_name": binary_name,
        "binary_sha256": digest.lower(),
    }
