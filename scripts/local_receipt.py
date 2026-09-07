"""Machine-local receipts and private evidence storage.

The Hub receives only a hash-bound receipt reference.  Detailed native
execution evidence remains in the deployment-owned local evidence directory
and is never serialized into a Hub request or public projection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from collections.abc import Mapping
from typing import Any

try:
    from scripts.private_adapter import PrivateAdapterError, load_private_adapter
except ModuleNotFoundError:
    from private_adapter import PrivateAdapterError, load_private_adapter


class LocalReceiptError(RuntimeError):
    """A local run receipt is invalid or conflicts with the requested run."""


RECEIPT_SCHEMA_VERSION = "neural-foundry-run-receipt.v2"
LEGACY_RECEIPT_SCHEMA_VERSION = "neural-foundry-run-receipt.v1"
NATIVE_EXECUTION_EVIDENCE_SCHEMA_VERSION = "neural-foundry-native-execution-evidence.v1"
PRIVATE_EVIDENCE_SCHEMA_VERSION = "neural-foundry-private-evidence.v1"
EXECUTION_RECEIPT_REF_SCHEMA_VERSION = "neural-foundry-run-receipt-ref.v2"
RUN_ID_RE = re.compile(r"^nf-[a-z0-9-]{8,80}$", re.IGNORECASE)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{16,64}$")
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")

RECEIPT_REF_FIELDS = frozenset({
    "schema_version", "manifest_sha256", "status", "metrics_available",
    "evidence_ref", "evidence_sha256", "native_evidence_sha256", "receipt_sha256",
})
DESCRIPTOR_RECEIPT_REF_FIELDS = RECEIPT_REF_FIELDS | frozenset({"descriptor_sha256"})
LOCAL_RECEIPT_FIELDS = frozenset({
    "schema_version", "run_id", "manifest_sha256", "status", "metrics_available", "attempt",
    "evidence_ref", "evidence_sha256", "native_evidence_sha256",
})


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise LocalReceiptError("run_id is invalid")
    return run_id


def _opaque_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not OPAQUE_ID_RE.fullmatch(value) or ".." in value:
        raise LocalReceiptError(f"{label} is invalid")
    return value


def _evidence_text(value: Any, label: str, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(character) < 0x20 for character in value)
        or "/" in value
        or "\\" in value
        or "://" in value
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise LocalReceiptError(f"{label} is invalid")
    return value.strip()


def _hash(value: Any, label: str, *, content: bool = False) -> str:
    pattern = CONTENT_HASH_RE if content else HASH_RE
    if not isinstance(value, str) or not pattern.fullmatch(value.lower()):
        raise LocalReceiptError(f"{label} is invalid")
    return value.lower()


def _validate_native_execution_evidence_shape(value: Any) -> dict[str, Any]:
    """Validate detailed evidence before it enters the private evidence store."""

    if not isinstance(value, dict) or set(value) != NATIVE_EVIDENCE_FIELDS:
        raise LocalReceiptError("native execution evidence is invalid")
    if value.get("schema_version") != NATIVE_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise LocalReceiptError("native execution evidence schema_version is invalid")
    normalized: dict[str, Any] = {
        "schema_version": NATIVE_EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "profile_id": _opaque_id(value.get("profile_id"), "native execution evidence.profile_id"),
        "artifact_id": _opaque_id(value.get("artifact_id"), "native execution evidence.artifact_id"),
        "request_identity_sha256": _hash(value.get("request_identity_sha256"), "native execution evidence.request_identity_sha256"),
        "binary_sha256": _hash(value.get("binary_sha256"), "native execution evidence.binary_sha256"),
        "source_sha256": _hash(value.get("source_sha256"), "native execution evidence.source_sha256"),
        "source_manifest_version": _opaque_id(value.get("source_manifest_version"), "native execution evidence.source_manifest_version"),
        "source_file_count": value.get("source_file_count"),
        "recipe_contract_version": _opaque_id(value.get("recipe_contract_version"), "native execution evidence.recipe_contract_version"),
        "recipe_id": _opaque_id(value.get("recipe_id"), "native execution evidence.recipe_id"),
        "recipe_card_sha256": _hash(value.get("recipe_card_sha256"), "native execution evidence.recipe_card_sha256"),
        "source_recipe_spec_hash": _hash(value.get("source_recipe_spec_hash"), "native execution evidence.source_recipe_spec_hash", content=True),
        "recipe_resolution_hash": _hash(value.get("recipe_resolution_hash"), "native execution evidence.recipe_resolution_hash"),
        "backend": _opaque_id(value.get("backend"), "native execution evidence.backend"),
        "runtime": _opaque_id(value.get("runtime"), "native execution evidence.runtime"),
        "required_arch": _opaque_id(value.get("required_arch"), "native execution evidence.required_arch"),
        "architecture": _opaque_id(value.get("architecture"), "native execution evidence.architecture"),
        "architecture_source": _opaque_id(value.get("architecture_source"), "native execution evidence.architecture_source"),
        "device": _evidence_text(value.get("device"), "native execution evidence.device"),
        "steps": value.get("steps"),
        "tokens": value.get("tokens"),
        "loss": value.get("loss"),
        "tokens_per_second": value.get("tokens_per_second"),
        "checkpoint_written": value.get("checkpoint_written"),
        "parameters_changed": value.get("parameters_changed"),
        "optimizer": _opaque_id(value.get("optimizer"), "native execution evidence.optimizer"),
    }
    if normalized["backend"] not in {"opencl", "hip"} or normalized["runtime"] != normalized["backend"]:
        raise LocalReceiptError("native execution evidence runtime is invalid")
    if normalized["source_manifest_version"] != "native_source_manifest_v1":
        raise LocalReceiptError("native execution evidence source manifest is invalid")
    if normalized["architecture"].lower() != normalized["required_arch"].lower():
        raise LocalReceiptError("native execution evidence architecture does not match required_arch")
    if normalized["architecture_source"] not in {"device_metadata", "runtime_device_properties"}:
        raise LocalReceiptError("native execution evidence architecture source is not observed")
    if (
        isinstance(normalized["source_file_count"], bool)
        or not isinstance(normalized["source_file_count"], int)
        or normalized["source_file_count"] < 1
    ):
        raise LocalReceiptError("native execution evidence source file count is invalid")
    for key in ("steps", "tokens"):
        if isinstance(normalized[key], bool) or not isinstance(normalized[key], int) or normalized[key] < 1:
            raise LocalReceiptError(f"native execution evidence {key} is invalid")
    for key in ("loss", "tokens_per_second"):
        item = normalized[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise LocalReceiptError(f"native execution evidence {key} is invalid")
    if normalized["checkpoint_written"] is not False or not isinstance(normalized["parameters_changed"], bool):
        raise LocalReceiptError("native execution evidence completion state is invalid")
    return normalized


def _descriptor_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build a structural trust anchor from a descriptor already deployment-validated."""

    recipe = value.get("recipe")
    source = value.get("source")
    return {
        "schema_version": value.get("schema_version"),
        "profile_id": value.get("profile_id"),
        "backend": value.get("backend"),
        "runtime": value.get("runtime"),
        "precision_profile": value.get("precision_profile"),
        "optimizer_type": value.get("optimizer_type"),
        "attention_backend": value.get("attention_backend"),
        "hardware_profile": value.get("hardware_profile"),
        "hardware_architecture": value.get("hardware_architecture"),
        "required_arch": value.get("required_arch"),
        "execution_lane": value.get("execution_lane"),
        "recipe": recipe,
        "source": source,
        "required_native_settings": value.get("native_settings"),
        "artifact_id": value.get("artifact_id"),
        "binary_names": [value.get("binary_name")],
        "model_contract_id": value.get("model_contract_id"),
        "fresh_only": False,
    }


def _validate_native_execution_descriptor(value: Any, *, run_id: str) -> dict[str, Any]:
    del value, run_id
    raise LocalReceiptError("private execution evidence requires an adapter")


def validate_native_execution_evidence(
    value: Any,
    *,
    expected_descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate detailed evidence and bind it to the descriptor that launched it."""

    normalized = _validate_native_execution_evidence_shape(value)
    if expected_descriptor is None:
        raise LocalReceiptError("native execution evidence requires an approved descriptor")
    descriptor = _validate_native_execution_descriptor(
        expected_descriptor,
        run_id=str(expected_descriptor.get("run_id", "")),
    )
    expected = {
        "profile_id": descriptor["profile_id"],
        "artifact_id": descriptor["artifact_id"],
        "request_identity_sha256": descriptor["request_identity_sha256"],
        "binary_sha256": descriptor["binary_sha256"],
        "source_sha256": descriptor["source"]["source_sha256"],
        "source_manifest_version": descriptor["source"]["manifest_version"],
        "source_file_count": descriptor["source"]["file_count"],
        "recipe_contract_version": descriptor["recipe"]["contract_version"],
        "recipe_id": descriptor["recipe"]["recipe_id"],
        "recipe_card_sha256": descriptor["recipe"]["recipe_card_sha256"],
        "source_recipe_spec_hash": descriptor["recipe"]["source_recipe_spec_hash"],
        "recipe_resolution_hash": descriptor["recipe"]["resolution_hash"],
        "backend": descriptor["backend"],
        "runtime": descriptor["runtime"],
        "required_arch": descriptor["required_arch"],
    }
    if any(normalized[field] != expected[field] for field in NATIVE_EVIDENCE_IDENTITY_FIELDS):
        raise LocalReceiptError("native execution evidence does not match the approved descriptor")
    if normalized["optimizer"] != descriptor["optimizer_type"]:
        raise LocalReceiptError("native execution evidence optimizer does not match descriptor")
    return normalized


def _validate_receipt_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalReceiptError("execution receipt reference fields are invalid")
    fields = DESCRIPTOR_RECEIPT_REF_FIELDS if "descriptor_sha256" in value else RECEIPT_REF_FIELDS
    if set(value) != fields:
        raise LocalReceiptError("execution receipt reference fields are invalid")
    if value["schema_version"] != EXECUTION_RECEIPT_REF_SCHEMA_VERSION:
        raise LocalReceiptError("execution receipt reference schema is invalid")
    if value["status"] not in {"succeeded", "failed", "cancelled"}:
        raise LocalReceiptError("receipt reference status is invalid")
    if not isinstance(value["metrics_available"], bool):
        raise LocalReceiptError("receipt reference metrics flag is invalid")
    evidence_ref = value["evidence_ref"]
    evidence_sha256 = value["evidence_sha256"]
    native_evidence_sha256 = value["native_evidence_sha256"]
    descriptor_sha256 = value.get("descriptor_sha256")
    if evidence_ref is not None:
        _opaque_id(evidence_ref, "receipt reference evidence_ref")
    if evidence_sha256 is not None:
        _hash(evidence_sha256, "receipt reference evidence_sha256")
    if native_evidence_sha256 is not None:
        _hash(native_evidence_sha256, "receipt reference native_evidence_sha256")
    if descriptor_sha256 is not None:
        _hash(descriptor_sha256, "receipt reference descriptor_sha256")
    if (evidence_ref is None) != (evidence_sha256 is None):
        raise LocalReceiptError("receipt reference evidence binding is incomplete")
    if native_evidence_sha256 is not None and evidence_ref is None:
        raise LocalReceiptError("receipt reference native evidence requires private evidence")
    if evidence_ref is not None and (value["status"] != "succeeded" or value["metrics_available"] is not True):
        raise LocalReceiptError("receipt reference evidence requires a successful metrics receipt")
    identity = {key: value[key] for key in fields if key != "receipt_sha256"}
    if value["receipt_sha256"] != canonical_fingerprint(identity):
        raise LocalReceiptError("receipt reference fingerprint is invalid")
    return dict(value)


def build_execution_receipt_ref(
    receipt_identity: Mapping[str, Any],
    *,
    status: str,
    metrics_available: bool,
    evidence_ref: str | None = None,
    evidence_sha256: str | None = None,
    native_evidence_sha256: str | None = None,
    descriptor_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the only receipt shape permitted to cross into Hub."""

    if not isinstance(receipt_identity, Mapping):
        raise LocalReceiptError("receipt identity is invalid")
    identity = {
        "schema_version": EXECUTION_RECEIPT_REF_SCHEMA_VERSION,
        "manifest_sha256": canonical_fingerprint(receipt_identity),
        "status": status,
        "metrics_available": metrics_available,
        "evidence_ref": evidence_ref,
        "evidence_sha256": evidence_sha256,
        "native_evidence_sha256": native_evidence_sha256,
        **({"descriptor_sha256": descriptor_sha256} if descriptor_sha256 is not None else {}),
    }
    receipt = {**identity, "receipt_sha256": canonical_fingerprint(identity)}
    return _validate_receipt_reference(receipt)


class LocalRunReceiptStore:
    """Persist bounded retry state and keep detailed evidence owner-local."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.receipt_root = self.root / ".receipts"
        self.private_evidence_root = self.root / ".private-evidence"

    def _path(self, run_id: str) -> Path:
        return self.receipt_root / f"{_safe_run_id(run_id)}.json"

    def _evidence_path(self, evidence_ref: str) -> Path:
        return self.private_evidence_root / f"{_opaque_id(evidence_ref, 'evidence_ref')}.json"

    @staticmethod
    def _ensure_directory(path: Path, label: str) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            mode = os.lstat(path).st_mode
        except OSError as exc:
            raise LocalReceiptError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise LocalReceiptError(f"{label} cannot be a symbolic link")

    def _ensure_receipt_root(self) -> None:
        self._ensure_directory(self.receipt_root, "local receipt directory")

    def _ensure_private_evidence_root(self) -> None:
        self._ensure_directory(self.private_evidence_root, "private evidence directory")

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            if path.is_symlink():
                raise LocalReceiptError("local receipt cannot be a symbolic link")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LocalReceiptError("local receipt is invalid") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") not in {
            LEGACY_RECEIPT_SCHEMA_VERSION,
            RECEIPT_SCHEMA_VERSION,
        }:
            raise LocalReceiptError("local receipt is invalid")
        if set(payload) - LOCAL_RECEIPT_FIELDS:
            raise LocalReceiptError("local receipt contains unsupported or private fields")
        allowed_statuses = {"in_progress", "succeeded", "failed"}
        if payload.get("schema_version") == RECEIPT_SCHEMA_VERSION:
            allowed_statuses.add("cancelled")
        if payload.get("status") not in allowed_statuses:
            raise LocalReceiptError("local receipt status is invalid")
        _safe_run_id(payload.get("run_id"))
        _hash(payload.get("manifest_sha256"), "local receipt fingerprint")
        if not isinstance(payload.get("metrics_available"), bool):
            raise LocalReceiptError("local receipt metrics flag is invalid")
        if isinstance(payload.get("attempt"), bool) or not isinstance(payload.get("attempt"), int) or payload["attempt"] < 1:
            raise LocalReceiptError("local receipt attempt is invalid")
        if "evidence_ref" in payload or "evidence_sha256" in payload or "native_evidence_sha256" in payload:
            if payload.get("status") != "succeeded" or payload.get("metrics_available") is not True:
                raise LocalReceiptError("local receipt evidence is not terminal")
            if "evidence_ref" not in payload or "evidence_sha256" not in payload:
                raise LocalReceiptError("local receipt evidence binding is incomplete")
            _opaque_id(payload["evidence_ref"], "local receipt evidence_ref")
            _hash(payload["evidence_sha256"], "local receipt evidence_sha256")
            if payload.get("native_evidence_sha256") is not None:
                _hash(payload["native_evidence_sha256"], "local receipt native_evidence_sha256")
        return payload

    def claim(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        run_id = _safe_run_id(run_id)
        if not isinstance(manifest, dict):
            raise LocalReceiptError("manifest is invalid")
        fingerprint = canonical_fingerprint(manifest)
        path = self._path(run_id)
        self._ensure_receipt_root()
        record = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_sha256": fingerprint,
            "status": "in_progress",
            "metrics_available": False,
            "attempt": 1,
        }
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = self._read(path)
            if existing is None:
                raise LocalReceiptError("local receipt could not be read")
            if existing["manifest_sha256"] != fingerprint:
                raise LocalReceiptError("run identity was reused for a different manifest")
            if existing["status"] in {"succeeded", "cancelled"}:
                return {"decision": "terminal", **existing}
            if existing["status"] == "in_progress":
                return {"decision": "busy", **existing}
            record["attempt"] = int(existing.get("attempt", 1)) + 1
            self._write(path, {**record, "decision": "retry"})
            return {"decision": "retry", **record}
        else:
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(record, stream, sort_keys=True)
                    stream.write("\n")
            except OSError as exc:
                raise LocalReceiptError("local receipt could not be created") from exc
            return {"decision": "new", **record}

    def prepare_private_evidence(
        self,
        run_id: str,
        manifest: dict[str, Any],
        *,
        native_execution: Mapping[str, Any],
        native_descriptor: Mapping[str, Any],
    ) -> dict[str, str]:
        """Validate and atomically store detailed native evidence privately."""

        run_id = _safe_run_id(run_id)
        try:
            adapter = load_private_adapter()
        except PrivateAdapterError as exc:
            raise LocalReceiptError("private adapter is unavailable") from exc
        if adapter is None or not adapter.is_request_schema(native_descriptor.get("schema_version")):
            raise LocalReceiptError("private execution evidence requires an adapter")
        try:
            evidence = adapter.validate_execution_evidence(native_execution, native_descriptor)
        except ValueError as exc:
            raise LocalReceiptError("private execution evidence is invalid") from exc
        evidence_sha256 = canonical_fingerprint(evidence)
        evidence_ref = f"evidence-{evidence_sha256[:32]}"
        metadata = {
            "schema_version": PRIVATE_EVIDENCE_SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_sha256": canonical_fingerprint(manifest),
            "evidence_ref": evidence_ref,
            "evidence_sha256": evidence_sha256,
            "native_evidence_sha256": evidence_sha256,
            "native_execution": evidence,
        }
        self._ensure_private_evidence_root()
        path = self._evidence_path(evidence_ref)
        if path.is_symlink():
            raise LocalReceiptError("private evidence cannot be a symbolic link")
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise LocalReceiptError("private evidence is invalid") from exc
            if not isinstance(existing, dict) or existing.get("evidence_sha256") != evidence_sha256:
                raise LocalReceiptError("private evidence reference conflicts")
            return {
                "evidence_ref": evidence_ref,
                "evidence_sha256": evidence_sha256,
                "native_evidence_sha256": evidence_sha256,
            }
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.private_evidence_root,
                prefix=f".{evidence_ref}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                json.dump(metadata, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), 0o600)
                else:
                    os.chmod(stream.name, 0o600)
            os.replace(temporary, path)
        except OSError as exc:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
            raise LocalReceiptError("private evidence could not be stored") from exc
        return {
            "evidence_ref": evidence_ref,
            "evidence_sha256": evidence_sha256,
            "native_evidence_sha256": evidence_sha256,
        }

    def complete(
        self,
        run_id: str,
        manifest: dict[str, Any],
        *,
        status: str,
        metrics_available: bool,
        evidence_ref: str | None = None,
        evidence_sha256: str | None = None,
        native_evidence_sha256: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise LocalReceiptError("terminal receipt status is invalid")
        if not isinstance(metrics_available, bool):
            raise LocalReceiptError("local receipt metrics flag is invalid")
        path = self._path(_safe_run_id(run_id))
        existing = self._read(path)
        fingerprint = canonical_fingerprint(manifest)
        if existing is None or existing["manifest_sha256"] != fingerprint:
            raise LocalReceiptError("local receipt does not match manifest")
        if existing["status"] in {"succeeded", "cancelled"} and existing["status"] != status:
            raise LocalReceiptError("local receipt is already terminal")
        evidence_values = (evidence_ref, evidence_sha256, native_evidence_sha256)
        if any(value is not None for value in evidence_values):
            if status != "succeeded" or not metrics_available:
                raise LocalReceiptError("evidence requires a successful metrics receipt")
            if evidence_ref is None or evidence_sha256 is None:
                raise LocalReceiptError("evidence binding is incomplete")
            _opaque_id(evidence_ref, "local receipt evidence_ref")
            _hash(evidence_sha256, "local receipt evidence_sha256")
            if native_evidence_sha256 is not None:
                _hash(native_evidence_sha256, "local receipt native_evidence_sha256")
            evidence_path = self._evidence_path(evidence_ref)
            if evidence_path.is_symlink():
                raise LocalReceiptError("private evidence cannot be a symbolic link")
            try:
                evidence_record = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise LocalReceiptError("private evidence is unavailable") from exc
            if not isinstance(evidence_record, dict) or evidence_record.get("evidence_sha256") != evidence_sha256:
                raise LocalReceiptError("private evidence fingerprint does not match receipt")
        updated = {
            **existing,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": status,
            "metrics_available": metrics_available,
        }
        if any(value is not None for value in evidence_values):
            updated.update({
                "evidence_ref": evidence_ref,
                "evidence_sha256": evidence_sha256,
                "native_evidence_sha256": native_evidence_sha256,
            })
        self._write(path, updated)
        return updated

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        self._ensure_receipt_root()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.receipt_root,
                prefix=f".{path.stem}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                json.dump(payload, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), 0o600)
                else:
                    os.chmod(stream.name, 0o600)
            os.replace(temporary, path)
        except OSError as exc:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
            raise LocalReceiptError("local receipt could not be updated") from exc
