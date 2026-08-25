"""Public capability catalog and private local asset bindings.

The public catalog contains no filesystem paths.  The local binding file is
deployment-owned and must remain outside Git; it maps Hub-owned opaque IDs to
relative local references.  The binding also pins the exact catalog bytes so
a worker fails closed when its public capability list drifts from the one
approved for that installation.  The worker verifies the referenced content
hashes after resolving them under its configured roots.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from pathlib import PurePosixPath
from typing import Any

try:
    from scripts.model_contracts import validate_model_contract_id
except ModuleNotFoundError:
    from model_contracts import validate_model_contract_id


CATALOG_SCHEMA_VERSION = "neural-foundry-capability-catalog.v1"
BINDING_SCHEMA_VERSION = "neural-foundry-local-binding.v2"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REF_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_RESERVED_IDS = {"__proto__", "prototype", "constructor"}


class BindingError(ValueError):
    """A public catalog or private local binding failed closed."""


def _record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BindingError(f"{label} is invalid")
    return value


def _id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _ID_RE.fullmatch(value)
        or ".." in value
        or value.lower() in _RESERVED_IDS
    ):
        raise BindingError(f"{label} is invalid")
    return value


def _relative_ref(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or "\\" in value
        or value.startswith(("/", "//"))
        or _REF_SCHEME_RE.match(value) is not None
        or any(part in {"", ".", ".."} for part in PurePosixPath(value).parts)
    ):
        raise BindingError(f"{label} is invalid")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BindingError(f"{label} is invalid")
    return value.lower()


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BindingError(f"{label} is unavailable") from exc
    return _record(value, label)


def _entry(value: Any, label: str, *, allow_contract: bool = False) -> dict[str, str]:
    entry = _record(value, label)
    allowed = {"ref", "sha256"} | ({"contract_id"} if allow_contract else set())
    if set(entry) - allowed or not {"ref", "sha256"}.issubset(entry):
        raise BindingError(f"{label} fields are invalid")
    ref = entry["ref"]
    ref = _relative_ref(entry["ref"], f"{label}.ref")
    result = {
        "ref": ref,
        "sha256": _sha256(entry["sha256"], f"{label}.sha256"),
    }
    if allow_contract:
        result["contract_id"] = _id(entry["contract_id"], f"{label}.contract_id") if "contract_id" in entry else None
    return result


def _model_entry(value: Any, label: str) -> dict[str, str]:
    result = _entry(value, label, allow_contract=True)
    if result.get("contract_id") is None:
        result.pop("contract_id", None)
    return result


def _private_runtime_entry(value: Any, label: str) -> dict[str, str]:
    entry = _record(value, label)
    allowed = {"ref", "sha256", "binary_sha256"}
    if set(entry) != allowed:
        raise BindingError(f"{label} fields are invalid")
    return {
        "ref": _relative_ref(entry["ref"], f"{label}.ref"),
        "sha256": _sha256(entry["sha256"], f"{label}.sha256"),
        "binary_sha256": _sha256(entry["binary_sha256"], f"{label}.binary_sha256"),
    }


def _profile_entry(value: Any, label: str) -> dict[str, str]:
    entry = _record(value, label)
    allowed = {"ref", "sha256", "precision", "optimizer", "kernel_ref"}
    if set(entry) - allowed or "ref" not in entry:
        raise BindingError(f"{label} fields are invalid")
    result = _entry({key: entry[key] for key in ("ref", "sha256") if key in entry}, label)
    for field in ("precision", "optimizer"):
        if field in entry:
            result[field] = _id(entry[field], f"{label}.{field}")
    if "kernel_ref" in entry:
        result["kernel_ref"] = _relative_ref(entry["kernel_ref"], f"{label}.kernel_ref")
    return result


def load_capability_catalog(path: Path) -> dict[str, Any]:
    payload = _json_file(path, "capability catalog")
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise BindingError("capability catalog schema_version is invalid")
    _id(payload.get("catalog_version"), "catalog_version")
    profiles = _record(payload.get("profiles"), "catalog profiles")
    normalized: dict[str, Any] = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": payload["catalog_version"],
        "profiles": {},
    }
    for profile_id, raw in profiles.items():
        profile_id = _id(profile_id, "profile_id")
        entry = _record(raw, f"profile {profile_id}")
        required = {
            "channel", "backend", "artifact_id", "binary_names",
            "supported_precisions", "supported_optimizers", "supported_attention",
        }
        allowed = required | {"kernel_sha256", "enabled", "model_contract_ids"}
        if set(entry) - allowed or not required.issubset(entry):
            raise BindingError(f"profile {profile_id} fields are invalid")
        channel = entry["channel"]
        if channel not in {"stable", "experimental"}:
            raise BindingError(f"profile {profile_id}.channel is invalid")
        if not isinstance(entry.get("enabled", True), bool):
            raise BindingError(f"profile {profile_id}.enabled is invalid")
        backend = _id(entry["backend"], f"profile {profile_id}.backend")
        artifact_id = _id(entry["artifact_id"], f"profile {profile_id}.artifact_id")
        lists: dict[str, list[str]] = {}
        for field in ("binary_names", "supported_precisions", "supported_optimizers", "supported_attention"):
            value = entry[field]
            allow_disabled_smoke_optimizer_gap = (
                field == "supported_optimizers"
                and entry.get("enabled", True) is False
                and backend in {"cpu", "opencl"}
            )
            if (
                not isinstance(value, list)
                or (not value and not allow_disabled_smoke_optimizer_gap)
            ):
                raise BindingError(f"profile {profile_id}.{field} is invalid")
            lists[field] = [_id(item, f"profile {profile_id}.{field}") for item in value]
        model_contract_ids: list[str] = []
        if "model_contract_ids" in entry:
            raw_contracts = entry["model_contract_ids"]
            if not isinstance(raw_contracts, list) or not raw_contracts:
                raise BindingError(f"profile {profile_id}.model_contract_ids is invalid")
            try:
                model_contract_ids = [
                    validate_model_contract_id(item, f"profile {profile_id}.model_contract_ids[{index}]")
                    for index, item in enumerate(raw_contracts)
                ]
            except ValueError as exc:
                raise BindingError(str(exc)) from exc
            if len(model_contract_ids) != len(set(model_contract_ids)):
                raise BindingError(f"profile {profile_id}.model_contract_ids contains duplicates")
        if backend == "opencl" and "kernel_sha256" not in entry:
            raise BindingError(f"profile {profile_id}.kernel_sha256 is required")
        profile = {
            "enabled": entry.get("enabled", True),
            "channel": channel,
            "backend": backend,
            "artifact_id": artifact_id,
            **lists,
            "model_contract_ids": model_contract_ids,
        }
        if "kernel_sha256" in entry:
            profile["kernel_sha256"] = _sha256(
                entry["kernel_sha256"], f"profile {profile_id}.kernel_sha256"
            )
        normalized["profiles"][profile_id] = profile
    return normalized


def load_local_binding(path: Path) -> dict[str, Any]:
    payload = _json_file(path, "local binding")
    if payload.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise BindingError("local binding schema_version is invalid")
    _id(payload.get("catalog_version"), "binding catalog_version")
    required = {
        "schema_version", "catalog_version", "catalog_sha256", "profiles",
        "datasets", "models", "checkpoints", "binaries",
    }
    allowed = required | {"hardware_probes", "private_runtimes"}
    if set(payload) - allowed:
        raise BindingError("local binding fields are invalid")
    result = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "catalog_version": payload["catalog_version"],
        "catalog_sha256": _sha256(payload["catalog_sha256"], "binding catalog_sha256"),
    }
    for collection in ("profiles", "datasets", "models", "checkpoints", "binaries"):
        raw = _record(payload[collection], f"local binding {collection}")
        entry_loader = _profile_entry if collection == "profiles" else _model_entry if collection in {"models", "checkpoints"} else _entry
        result[collection] = {
            _id(key, f"{collection} id"): entry_loader(value, f"{collection}.{key}")
            for key, value in raw.items()
        }
    raw_private_runtimes = _record(payload.get("private_runtimes", {}), "local binding private_runtimes")
    result["private_runtimes"] = {
        _id(key, f"private_runtimes id"): _private_runtime_entry(value, f"private_runtimes.{key}")
        for key, value in raw_private_runtimes.items()
    }
    if "hardware_probes" in payload:
        raw_probes = _record(payload["hardware_probes"], "local binding hardware_probes")
        if set(raw_probes) - {"nvidia", "amd"}:
            raise BindingError("local binding hardware_probes fields are invalid")
        result["hardware_probes"] = {
            vendor: _entry(value, f"hardware_probes.{vendor}")
            for vendor, value in raw_probes.items()
        }
    return result


def file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise BindingError("bound local asset cannot be a symbolic link")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BindingError("bound local asset is unavailable") from exc
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    """Hash a file or a deterministic relative-path/file-content tree."""

    if path.is_symlink():
        raise BindingError("bound local asset cannot be a symbolic link")
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise BindingError("bound local asset is unavailable")
    root = path.resolve()
    digest = hashlib.sha256()
    try:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(directories)
            files[:] = sorted(files)
            for name in [*directories, *files]:
                child = current_path / name
                if child.is_symlink():
                    raise BindingError("bound local asset tree contains a symbolic link")
                try:
                    child.resolve().relative_to(root)
                except ValueError as exc:
                    raise BindingError("bound local asset escapes its root") from exc
            for name in files:
                child = current_path / name
                relative = child.relative_to(root).as_posix().encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(bytes.fromhex(file_sha256(child)))
    except OSError as exc:
        raise BindingError("bound local asset is unavailable") from exc
    return digest.hexdigest()
