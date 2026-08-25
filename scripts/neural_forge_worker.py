#!/usr/bin/env python3
"""Local-only Neural Foundry worker.

The Hub owns a sanitized job queue. This worker owns all sensitive training
inputs and execution: configs, datasets, checkpoints, the native binary, and
the local output directory. Only bounded progress/status records leave this
machine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from scripts.compatibility import (
        BACKEND_POLICIES,
        MANIFEST_SCHEMA_VERSION,
    )
    from scripts.deployment_map import (
        DeploymentMapError,
        load_deployment_map,
        validate_model_contract_binding,
        validate_private_runtime_binding,
        validate_execution_attestation,
    )
    from scripts.model_contracts import (
        ModelContractError,
        validate_model_contract_id,
        validate_model_shape,
    )
    from scripts.private_runtime import PrivateRuntimeError, validate_optional_private_runtime
    from scripts.local_receipt import LocalReceiptError, LocalRunReceiptStore, canonical_fingerprint
    from scripts.hardware_identity import probe_hardware
    from scripts.local_binding import (
        BindingError,
        file_sha256,
        load_capability_catalog,
        load_local_binding,
        path_sha256,
    )
except ModuleNotFoundError:
    # Keep the documented `python scripts/neural_forge_worker.py` form
    # executable when Python places only the scripts directory on sys.path.
    from compatibility import BACKEND_POLICIES, MANIFEST_SCHEMA_VERSION
    from deployment_map import (
        DeploymentMapError,
        load_deployment_map,
        validate_model_contract_binding,
        validate_private_runtime_binding,
        validate_execution_attestation,
    )
    from model_contracts import ModelContractError, validate_model_contract_id, validate_model_shape
    from private_runtime import PrivateRuntimeError, validate_optional_private_runtime
    from local_receipt import LocalReceiptError, LocalRunReceiptStore, canonical_fingerprint
    from hardware_identity import probe_hardware
    from local_binding import BindingError, file_sha256, load_capability_catalog, load_local_binding, path_sha256


RUN_ID_RE = re.compile(r"^nf-[a-z0-9-]{8,80}$", re.IGNORECASE)
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*$"
)
DEFAULT_HUB_HOSTS = ("hidden-canopy-hub-api.azurewebsites.net",)
EVENT_FIELDS = {
    "type", "status", "step", "micro_step", "loss", "learning_rate",
    "tokens_per_second", "grad_norm", "checkpoint_written", "exit_code",
    "message",
}
NATIVE_TOP_LEVEL_FIELDS = {
    "attention_backend", "precision_profile", "optimizer_state_precision",
    "optimizer_type", "gradient_buffer_precision", "gemm_accumulator_precision",
    "family", "seat", "version", "device", "model", "training", "input",
    "grad_accum_override", "max_samples", "seed", "pss_pred_rank",
    "pss_pred_lr_scale", "pss_pred_aux_weight", "pss_conditioning_mode",
    "pss_aux_normalize_mode", "pss_pred_state_out", "pss_pred_state_init_path",
    "global_clip_override", "act_row_clip_override", "spec_hash",
    "lion_lr_scale_override", "lion_wd_scale_override", "lion_beta1_override",
    "lion_beta2_override", "lion_trust_ratio_enabled_override",
    "lion_trust_ratio_lo_override", "lion_trust_ratio_hi_override",
    "architecture_compatibility", "architecture_contract", "expected_terminal_phase",
    "job_id",
}
NATIVE_DEVICE_FIELDS = {
    "runtime", "required_arch", "precision", "model_parallel_devices", "pipeline_split_layer"
}
NATIVE_MODEL_FIELDS = {
    "architecture_contract", "hidden_size", "intermediate_size", "layers", "heads", "kv_heads", "vocab_size",
    "num_cognitive_routes", "top_k_routes", "num_personality_experts",
    "personality_residual_expert_width", "expert_balancing_loss_coef",
    "top_k_experts", "use_personality_residual_experts", "local_attention_window",
    "moe_native_fp4", "rope_theta", "generic_moe_num_experts",
    "generic_moe_top_k", "generic_moe_expert_width", "generic_moe_shared_expert_width",
    "generic_moe_normalize_topk", "qkv_bias", "normalization_type", "activation_type",
    "position_embedding_type", "norm_eps", "max_position_embeddings",
    "projection_bias", "tied_embeddings",
}
NATIVE_TRAINING_FIELDS = {
    "microbatch", "per_device_train_batch_size", "grad_accumulation",
    "gradient_accumulation_steps", "learning_rate", "max_steps",
}
NATIVE_INPUT_FIELDS = {"token_blocks", "label_blocks", "seg_blocks", "batch_size", "sequence_length", "shape"}
TRAINING_MODES = {"from_scratch", "fine_tune", "resume"}
DISABLED_OPTIMIZERS = {"adam", "adamw"}
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
RESERVED_IDS = {"__proto__", "prototype", "constructor"}

# These are the fields the Hub/native adapter reserves for local execution or
# private diagnostics.  They must never be accepted inside a Hub manifest,
# including when an upstream service accidentally nests them in lineage.
PRIVATE_MANIFEST_FIELDS = {
    "config_ref", "dataset_ref", "optimizer", "precision", "gpu",
    "init_from_model_ref", "resume_from_model_ref", "resume_from_ref",
    "output_dir", "binary", "binary_path", "device", "local_path", "path",
    "command", "token_blocks", "label_blocks", "seg_blocks", "status_file",
    "kernel_source", "kernel_source_ref", "kernel_name", "run_dir",
    "settings", "stdout", "stderr", "raw_stdout", "raw_stderr", "log",
    "logs", "credential", "credential_value", "password", "token",
    "bearer_token", "private_key", "secret", "secret_value", "url", "uri",
    "download_url", "artifact_url", "package_url", "source_url",
}
PRIVATE_DIAGNOSTIC_FIELDS = {
    "telemetry", "raw_telemetry", "ontology", "evidence", "kernel_telemetry",
    "kernel_profile", "trace", "profiler", "debug_dump", "raw_source", "source_code",
}


def opaque_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not OPAQUE_ID_RE.fullmatch(value)
        or ".." in value
        or value.lower() in RESERVED_IDS
    ):
        raise WorkerError(f"{label} is invalid")
    return value


def opaque_id_list(value: Any, label: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise WorkerError(f"{label} is invalid")
    result = [opaque_id(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise WorkerError(f"{label} contains duplicates")
    return result


def sha256_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise WorkerError(f"{label} is invalid")
    return value.lower()


def reject_private_manifest_fields(value: Any, label: str = "manifest") -> None:
    """Reject private execution and diagnostic keys at every nesting level."""

    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_private_manifest_fields(item, f"{label}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        if not isinstance(key, str):
            raise WorkerError(f"{label} contains an invalid field")
        normalized = key.lower()
        if normalized in PRIVATE_MANIFEST_FIELDS:
            raise WorkerError(f"{label} contains a local execution field")
        if normalized in PRIVATE_DIAGNOSTIC_FIELDS:
            raise WorkerError(f"{label} contains a private diagnostic field")
        reject_private_manifest_fields(child, f"{label}.{key}")


def validate_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "subject", "role", "capability_scopes", "justification_hash"
    }:
        raise WorkerError("worker manifest authority is invalid")
    return {
        "subject": opaque_id(value["subject"], "authority subject"),
        "role": opaque_id(value["role"], "authority role"),
        "capability_scopes": opaque_id_list(
            value["capability_scopes"], "authority capability_scopes"
        ),
        "justification_hash": sha256_id(
            value["justification_hash"], "authority justification_hash"
        ),
    }


LINEAGE_ENTRY_FIELDS = {"id", "revision", "lineage_id", "content_hash", "parent_id"}
REPRODUCIBILITY_FIELDS = {
    "policy_version", "training_profile_id", "trainer_version", "training_mode",
    "dataset_id", "resource_class", "base_model_id", "checkpoint_id", "model_contract_id",
}


def validate_lineage_entry(value: Any, label: str, *, allow_kind: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerError(f"{label} is invalid")
    allowed = LINEAGE_ENTRY_FIELDS | ({"kind"} if allow_kind else set())
    if set(value) - allowed or "id" not in value or (allow_kind and "kind" not in value):
        raise WorkerError(f"{label} is invalid")
    result: dict[str, Any] = {"id": opaque_id(value["id"], f"{label}.id")}
    if allow_kind:
        result["kind"] = opaque_id(value["kind"], f"{label}.kind")
    for field in ("revision", "lineage_id", "parent_id"):
        if field in value:
            result[field] = opaque_id(value[field], f"{label}.{field}")
    if "content_hash" in value:
        result["content_hash"] = sha256_id(value["content_hash"], f"{label}.content_hash")
    return result


def validate_lineage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or "dataset" not in value:
        raise WorkerError("worker manifest lineage is invalid")
    allowed = {"dataset", "source", "reproducibility"}
    if set(value) - allowed:
        raise WorkerError("worker manifest lineage is invalid")
    result: dict[str, Any] = {
        "dataset": validate_lineage_entry(value["dataset"], "lineage.dataset")
    }
    if "source" in value:
        result["source"] = validate_lineage_entry(
            value["source"], "lineage.source", allow_kind=True
        )
    if "reproducibility" not in value:
        raise WorkerError("lineage.reproducibility is required")
    reproducibility = value["reproducibility"]
    if not isinstance(reproducibility, dict) or set(reproducibility) - REPRODUCIBILITY_FIELDS:
        raise WorkerError("lineage.reproducibility is invalid")
    required = {
        "policy_version", "training_profile_id", "trainer_version", "training_mode",
        "dataset_id", "resource_class",
    }
    if not required.issubset(reproducibility):
        raise WorkerError("lineage.reproducibility is incomplete")
    result["reproducibility"] = {
        key: opaque_id(item, f"lineage.reproducibility.{key}")
        for key, item in reproducibility.items()
    }
    return result


class WorkerError(RuntimeError):
    """A local worker request or execution failed without exposing local paths."""


MAX_HUB_RESPONSE_BYTES = 1024 * 1024
MAX_NATIVE_STDOUT_BYTES = 8 * 1024 * 1024
MAX_NATIVE_STDERR_BYTES = 8 * 1024 * 1024
MAX_NATIVE_EVENTS = 4096
DEFAULT_MAX_NATIVE_OUTPUT_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MAX_NATIVE_OUTPUT_FILES = 100_000


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise WorkerError("Hub redirects are rejected")


HUB_OPENER = build_opener(_NoRedirectHandler)


def sanitized_environment() -> dict[str, str]:
    """Pass only deployment-neutral process basics to the native child.

    Loader paths and accelerator selection are deployment-owned concerns. They
    must not be inherited from a caller, shell, or previous tenant.
    """
    allowed = {
        "SystemRoot", "WINDIR", "TEMP", "TMP",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key in allowed and value
    }


def validate_training_mode(request: dict[str, Any]) -> str:
    has_base_model = request.get("base_model_id") is not None
    has_checkpoint = request.get("checkpoint_id") is not None
    if has_base_model and has_checkpoint:
        raise WorkerError("parent inputs are mutually exclusive")
    inferred = "fine_tune" if has_base_model else "resume" if has_checkpoint else "from_scratch"
    mode = request.get("training_mode", inferred)
    if not isinstance(mode, str) or mode not in TRAINING_MODES or mode != inferred:
        raise WorkerError("training_mode does not match parent inputs")
    return mode


def output_usage(path: Path) -> tuple[int, int]:
    """Return bounded output bytes and file count without following links."""

    total_bytes = 0
    file_count = 0
    try:
        for current, directories, files in os.walk(path, topdown=True, followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(directories)
            files[:] = sorted(files)
            for name in [*directories, *files]:
                child = current_path / name
                if child.is_symlink():
                    raise WorkerError("native output contains an unsupported symbolic link")
            for name in files:
                child = current_path / name
                total_bytes += os.lstat(child).st_size
                file_count += 1
    except OSError as exc:
        raise WorkerError("native output could not be inspected") from exc
    return total_bytes, file_count


def binary_sha256(path: Path) -> str:
    try:
        return file_sha256(path)
    except BindingError as exc:
        raise WorkerError("native binary is unavailable") from exc


def relative_ref(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise WorkerError(f"{label} is invalid")
    if "\\" in value:
        raise WorkerError(f"{label} is invalid")
    candidate = Path(value.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WorkerError(f"{label} is invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise WorkerError(f"{label} is invalid") from exc
    return resolved


def run_output(root: Path, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(str(run_id)):
        raise WorkerError("run_id is invalid")
    return relative_ref(root, run_id, "run_id")


def bounded_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "worker_error", "message": "native event was not an object"}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in EVENT_FIELDS:
            continue
        if isinstance(item, str):
            result[key] = item[:256]
        elif isinstance(item, bool):
            result[key] = item
        elif isinstance(item, (int, float)) and abs(item) <= 1e15:
            result[key] = item
    if "type" not in result and "status" not in result:
        result["type"] = "native_event"
    return result


def public_event(value: Any) -> dict[str, Any]:
    """Project native output before it crosses the machine boundary."""
    result = bounded_event(value)
    result.pop("message", None)
    for key in ("type", "status"):
        if key in result and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(result[key])):
            result.pop(key)
    if "type" not in result and "status" not in result:
        result["type"] = "native_event"
    return result


def load_local_config(config_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError("local config is unavailable") from exc
    if not isinstance(payload, dict):
        raise WorkerError("local config is invalid")
    return payload


def local_config(config_path: Path, output_path: Path, max_steps: Any) -> Path:
    if max_steps is None:
        return config_path
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100000:
        raise WorkerError("max_steps is invalid")
    payload = load_local_config(config_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("training"), dict):
        raise WorkerError("local config is invalid")
    payload["training"] = dict(payload["training"])
    payload["training"]["max_steps"] = max_steps
    output_path.mkdir(parents=True, exist_ok=True)
    derived = output_path / "worker-config.json"
    try:
        derived.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise WorkerError("local config could not be prepared") from exc
    return derived


def project_native_section(payload: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise WorkerError(f"local {name} section is invalid")
    if set(value) - allowed:
        raise WorkerError(f"local {name} section contains unsupported fields")
    return dict(value)


def dataset_file(dataset_path: Path, value: Any, default: str, label: str) -> Path:
    reference = default if value in (None, "") else value
    resolved = relative_ref(dataset_path, reference, label)
    if not resolved.is_file():
        raise WorkerError("local dataset input is unavailable")
    return resolved


def bound_asset(root: Path, entry: dict[str, str], label: str) -> Path:
    """Resolve one private binding and verify its required local digest."""

    resolved = relative_ref(root, entry.get("ref"), label)
    if not resolved.exists():
        raise WorkerError(f"{label} is unavailable")
    expected = entry.get("sha256")
    if not expected:
        raise WorkerError(f"{label} integrity pin is required")
    try:
        actual = path_sha256(resolved)
    except BindingError as exc:
        raise WorkerError(f"{label} is unavailable") from exc
    if actual != expected:
        raise WorkerError(f"{label} integrity check failed")
    return resolved


def utc_expiry(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise WorkerError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkerError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise WorkerError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def verified_local_binding(config: "WorkerConfig") -> tuple[dict[str, Any], dict[str, Any]]:
    if config.binding_path is None:
        raise WorkerError("local capability binding is unavailable")
    try:
        catalog = load_capability_catalog(config.catalog_path)
        binding = load_local_binding(config.binding_path)
        catalog_hash = file_sha256(config.catalog_path)
    except BindingError as exc:
        raise WorkerError(str(exc)) from exc
    if binding["catalog_version"] != catalog["catalog_version"]:
        raise WorkerError("local capability binding is unavailable or stale")
    if binding["catalog_sha256"] != catalog_hash:
        raise WorkerError("public capability catalog integrity check failed")
    return catalog, binding


def hardware_probe_tools(binding: dict[str, Any], config: "WorkerConfig") -> dict[str, Path]:
    """Resolve only hash-pinned local SMI tools for optional attestation."""

    tools: dict[str, Path] = {}
    for vendor, entry in binding.get("hardware_probes", {}).items():
        if vendor not in {"nvidia", "amd"}:
            continue
        try:
            candidate = bound_asset(
                config.hardware_probe_root,
                entry,
                f"{vendor} hardware probe",
            )
        except (WorkerError, KeyError):
            continue
        expected_name = "nvidia-smi" if vendor == "nvidia" else "rocm-smi"
        if candidate.stem.lower() != expected_name:
            continue
        tools[vendor] = candidate
    return tools


def manifest_local_request(
    job: dict[str, Any],
    config: "WorkerConfig",
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    """Project a Hub manifest into private local references and policy."""

    manifest = job.get("worker_manifest") or job.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise WorkerError("worker manifest is invalid")
    reject_private_manifest_fields(manifest)
    required_manifest_fields = {
        "schema_version", "run_id", "issued_at", "org", "worker_subject",
        "execution", "request", "authority", "policy", "lineage",
    }
    if not required_manifest_fields.issubset(manifest):
        raise WorkerError("worker manifest is incomplete")
    if set(manifest) - {
        "schema_version", "run_id", "issued_at", "claimed_at", "deadline_at",
        "org", "worker_subject", "execution", "request", "authority", "policy",
        "lineage",
    }:
        raise WorkerError("worker manifest contains local or unsupported fields")
    run_id = job.get("run_id")
    if manifest.get("run_id") != run_id:
        raise WorkerError("worker manifest run_id does not match job")
    opaque_id(manifest.get("org"), "worker manifest organization")
    if manifest.get("org") != config.org:
        raise WorkerError("worker manifest organization is invalid")
    if config.worker_id is None or opaque_id(manifest.get("worker_subject"), "worker subject") != config.worker_id:
        raise WorkerError("worker manifest worker identity is invalid")
    issued_at = utc_expiry(manifest["issued_at"], "worker manifest issued_at")
    claimed_at = manifest.get("claimed_at")
    deadline_at = manifest.get("deadline_at")
    if claimed_at is None or deadline_at is None:
        raise WorkerError("worker manifest claim window is incomplete")
    claimed_at = utc_expiry(claimed_at, "worker manifest claimed_at")
    deadline_at = utc_expiry(deadline_at, "worker manifest deadline")
    if issued_at > claimed_at or claimed_at > deadline_at:
        raise WorkerError("worker manifest claim window is invalid")
    if deadline_at <= datetime.now(timezone.utc):
        raise WorkerError("worker manifest deadline has elapsed")
    validate_authority(manifest["authority"])

    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise WorkerError("worker manifest policy is invalid")
    if set(policy) - {
        "policy_version", "training_profile_id", "trainer_version", "resource_class",
        "quota_id", "max_steps", "timeout_seconds", "max_concurrency",
        "required_evaluation_gates", "credential_policy_id", "secret_ref_ids",
        "evaluation_request_id", "model_contract_id", "private_runtime",
    }:
        raise WorkerError("worker manifest policy contains local or unsupported fields")
    for field in ("policy_version", "training_profile_id", "trainer_version", "resource_class"):
        opaque_id(policy.get(field), f"worker manifest policy.{field}")
    if "quota_id" in policy:
        opaque_id(policy["quota_id"], "worker manifest policy.quota_id")
    if "credential_policy_id" in policy:
        opaque_id(policy["credential_policy_id"], "worker manifest policy.credential_policy_id")
    if "secret_ref_ids" in policy:
        opaque_id_list(policy["secret_ref_ids"], "worker manifest policy.secret_ref_ids")
    if "required_evaluation_gates" in policy:
        opaque_id_list(
            policy["required_evaluation_gates"],
            "worker manifest policy.required_evaluation_gates",
        )
    if "evaluation_request_id" in policy:
        opaque_id(policy["evaluation_request_id"], "worker manifest policy.evaluation_request_id")
    policy_contract_id = None
    if "model_contract_id" in policy:
        try:
            policy_contract_id = validate_model_contract_id(
                policy["model_contract_id"], "worker manifest policy.model_contract_id"
            )
        except ModelContractError as exc:
            raise WorkerError(str(exc)) from exc
    try:
        private_runtime = validate_optional_private_runtime(
            policy.get("private_runtime"), "worker manifest policy.private_runtime"
        )
    except PrivateRuntimeError as exc:
        raise WorkerError(str(exc)) from exc
    max_steps = policy.get("max_steps")
    timeout_seconds = policy.get("timeout_seconds")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100000:
        raise WorkerError("worker manifest max_steps is invalid")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 7 * 24 * 60 * 60:
        raise WorkerError("worker manifest timeout is invalid")
    if deadline_at > claimed_at + timedelta(seconds=timeout_seconds + 60):
        raise WorkerError("worker manifest deadline exceeds policy window")

    request = manifest.get("request")
    if not isinstance(request, dict):
        raise WorkerError("worker manifest request is invalid")
    allowed = {
        "training_profile_id", "dataset_id", "base_model_id", "checkpoint_id",
        "resource_class", "evaluation_request_id", "training_mode", "model_contract_id",
    }
    if set(request) - allowed:
        raise WorkerError("worker manifest request contains local or unsupported fields")
    profile_id = request.get("training_profile_id")
    dataset_id = request.get("dataset_id")
    resource_class = request.get("resource_class")
    opaque_id(profile_id, "worker manifest request.training_profile_id")
    opaque_id(dataset_id, "worker manifest request.dataset_id")
    opaque_id(resource_class, "worker manifest request.resource_class")
    for field in ("base_model_id", "checkpoint_id", "evaluation_request_id"):
        if field in request:
            opaque_id(request[field], f"worker manifest request.{field}")
    request_contract_id = None
    if "model_contract_id" in request:
        try:
            request_contract_id = validate_model_contract_id(
                request["model_contract_id"], "worker manifest request.model_contract_id"
            )
        except ModelContractError as exc:
            raise WorkerError(str(exc)) from exc
    if policy_contract_id is not None and request_contract_id not in {None, policy_contract_id}:
        raise WorkerError("worker manifest model contract does not match policy")
    model_contract_id = request_contract_id or policy_contract_id
    if profile_id != policy["training_profile_id"] or resource_class != policy["resource_class"]:
        raise WorkerError("worker manifest policy does not match request")
    if (
        ("evaluation_request_id" in request) != ("evaluation_request_id" in policy)
        or request.get("evaluation_request_id") != policy.get("evaluation_request_id")
    ):
        raise WorkerError("worker manifest evaluation request does not match policy")
    training_mode = validate_training_mode(request)
    lineage = validate_lineage(manifest["lineage"])
    if lineage["dataset"]["id"] != dataset_id:
        raise WorkerError("worker manifest lineage dataset does not match request")
    reproducibility = lineage["reproducibility"]
    expected_reproducibility = {
        "policy_version": policy["policy_version"],
        "training_profile_id": profile_id,
        "trainer_version": policy["trainer_version"],
        "training_mode": training_mode,
        "dataset_id": dataset_id,
        "resource_class": resource_class,
    }
    if model_contract_id is not None:
        expected_reproducibility["model_contract_id"] = model_contract_id
    if any(reproducibility.get(key) != value for key, value in expected_reproducibility.items()):
        raise WorkerError("worker manifest reproducibility does not match policy")
    for parent_key, source_kind in (("base_model_id", "base_model"), ("checkpoint_id", "checkpoint")):
        if parent_key in request:
            if reproducibility.get(parent_key) != request[parent_key]:
                raise WorkerError("worker manifest reproducibility does not match request")
            source = lineage.get("source")
            if not isinstance(source, dict) or source.get("kind") != source_kind or source.get("id") != request[parent_key]:
                raise WorkerError("worker manifest lineage source does not match request")
        elif parent_key in reproducibility:
            raise WorkerError("worker manifest reproducibility contains an unexpected parent")
    if not any(key in request for key in ("base_model_id", "checkpoint_id")) and "source" in lineage:
        raise WorkerError("worker manifest lineage source is unexpected")

    try:
        deployment = load_deployment_map(config.deployment_map_path)
        mapping = deployment["hub_profiles"].get(profile_id)
        resource_mapping = deployment["resource_classes"].get(resource_class)
        if mapping is None or resource_mapping is None:
            raise DeploymentMapError("deployment mapping is missing")
        execution = validate_execution_attestation(
            manifest.get("execution"), profile_id, mapping
        )
        validate_model_contract_binding(model_contract_id, profile_id, mapping)
        validate_private_runtime_binding(private_runtime, profile_id, mapping)
    except (DeploymentMapError, KeyError) as exc:
        raise WorkerError(str(exc)) from exc
    if policy["trainer_version"] not in mapping["trainer_versions"]:
        raise WorkerError("worker manifest trainer version is not deployment-approved")

    catalog, binding = verified_local_binding(config)
    local_profile_id = mapping["local_profile_id"]
    profile = catalog["profiles"].get(local_profile_id)
    profile_binding = binding["profiles"].get(local_profile_id)
    if profile is None or profile_binding is None:
        raise WorkerError("training profile is not locally approved")
    if profile.get("enabled", True) is False:
        raise WorkerError("training profile is disabled")
    if profile["channel"] == "experimental" and not config.allow_experimental:
        raise WorkerError("experimental profile requires explicit local acknowledgement")
    if execution["backend"] != profile["backend"]:
        raise WorkerError("execution attestation does not match profile")
    approved_contracts = profile.get("model_contract_ids", [])
    if model_contract_id is not None and approved_contracts and model_contract_id not in approved_contracts:
        raise WorkerError("model contract is not approved by the local profile")
    precision = profile_binding.get("precision")
    optimizer = profile_binding.get("optimizer")
    if not isinstance(optimizer, str) or optimizer.lower() in DISABLED_OPTIMIZERS:
        raise WorkerError("Adam and AdamW optimizers are disabled")
    if precision not in profile["supported_precisions"] or optimizer not in profile["supported_optimizers"]:
        raise WorkerError("local profile settings are not approved")

    config_path = bound_asset(config.config_root, profile_binding, "profile config")
    dataset_entry = binding["datasets"].get(dataset_id)
    if dataset_entry is None:
        raise WorkerError("dataset is not locally approved")
    dataset_path = bound_asset(config.dataset_root, dataset_entry, "dataset")
    local_request = dict(request)
    local_request.update({
        "backend": execution["backend"],
        "precision": precision,
        "optimizer": optimizer,
        "training_mode": training_mode,
    })
    kernel_source: Path | None = None
    if execution["backend"] == "opencl":
        kernel_ref = profile_binding.get("kernel_ref")
        if kernel_ref is None:
            raise WorkerError("OpenCL profile is missing a private kernel binding")
        else:
            kernel_root = config.kernel_root or config.artifact_root
            try:
                kernel_source = bound_asset(
                    kernel_root,
                    {"ref": kernel_ref, "sha256": profile["kernel_sha256"]},
                    "OpenCL kernel",
                )
            except KeyError as exc:
                raise WorkerError("OpenCL profile is missing kernel attestation") from exc
            local_request["kernel_source_ref"] = kernel_ref
    if "base_model_id" in request:
        entry = binding["models"].get(request["base_model_id"])
        if entry is None:
            raise WorkerError("base model is not locally approved")
        if model_contract_id is not None and entry.get("contract_id") not in {None, model_contract_id}:
            raise WorkerError("base model contract does not match manifest")
        local_request["init_from_model_ref"] = bound_asset(
            config.artifact_root, entry, "base model"
        ).relative_to(config.artifact_root).as_posix()
    if "checkpoint_id" in request:
        entry = binding["checkpoints"].get(request["checkpoint_id"])
        if entry is None:
            raise WorkerError("checkpoint is not locally approved")
        if model_contract_id is not None and entry.get("contract_id") not in {None, model_contract_id}:
            raise WorkerError("checkpoint contract does not match manifest")
        local_request["resume_from_ref"] = bound_asset(
            config.artifact_root, entry, "checkpoint"
        ).relative_to(config.artifact_root).as_posix()
    local_request["config_ref"] = config_path.relative_to(config.config_root).as_posix()
    local_request["dataset_ref"] = dataset_path.relative_to(config.dataset_root).as_posix()
    if model_contract_id is not None:
        local_request["model_contract_id"] = model_contract_id
    return local_request, config_path, dataset_path, {
        "execution": execution,
        "policy": policy,
        "profile": profile,
        "binding": binding,
        "device": resource_mapping["device"],
        "hardware_probe_tools": hardware_probe_tools(binding, config),
        "kernel_sha256": profile.get("kernel_sha256"),
        "kernel_source": kernel_source,
        "model_contract_id": model_contract_id,
        "private_runtime": private_runtime,
    }


def native_request_file(
    config_path: Path,
    dataset_path: Path,
    artifact_root: Path,
    output_path: Path,
    request: dict[str, Any],
    *,
    run_id: str,
    approved_max_steps: int | None = None,
    approved_attention: list[str] | None = None,
) -> Path:
    config = load_local_config(config_path)
    unknown = set(config) - NATIVE_TOP_LEVEL_FIELDS - {"required_arch"}
    if unknown:
        raise WorkerError("local config contains unsupported fields")
    native: dict[str, Any] = {
        key: config[key] for key in NATIVE_TOP_LEVEL_FIELDS if key in config
    }
    native.setdefault("attention_backend", "scalar_flash")
    device = project_native_section(config, "device", NATIVE_DEVICE_FIELDS)
    if "required_arch" in config and "required_arch" not in device:
        device["required_arch"] = config["required_arch"]
    native["device"] = device
    native["model"] = project_native_section(config, "model", NATIVE_MODEL_FIELDS)
    native["training"] = project_native_section(config, "training", NATIVE_TRAINING_FIELDS)
    native["input"] = project_native_section(config, "input", NATIVE_INPUT_FIELDS)
    validate_training_mode(request)
    model_contract_id = request.get("model_contract_id") or native["model"].get("architecture_contract")
    if model_contract_id is not None:
        try:
            validate_model_shape(model_contract_id, native["model"])
        except ModelContractError as exc:
            raise WorkerError(str(exc)) from exc
    if "settings" in request:
        raise WorkerError("worker manifest settings are not accepted")

    if not dataset_path.is_dir():
        raise WorkerError("local dataset is unavailable")
    native["input"]["token_blocks"] = str(dataset_file(
        dataset_path, native["input"].get("token_blocks"), "tokens.u32", "token_blocks"
    ))
    native["input"]["label_blocks"] = str(dataset_file(
        dataset_path, native["input"].get("label_blocks"), "labels.i32", "label_blocks"
    ))
    if native["input"].get("seg_blocks") not in (None, ""):
        native["input"]["seg_blocks"] = str(dataset_file(
            dataset_path, native["input"]["seg_blocks"], "segs.u16", "seg_blocks"
        ))
    else:
        native["input"].pop("seg_blocks", None)

    backend = request.get("backend")
    if not isinstance(backend, str) or backend not in BACKEND_POLICIES:
        raise WorkerError("native backend is invalid")
    policy = BACKEND_POLICIES[backend]
    precision = request.get("precision")
    optimizer = request.get("optimizer")
    if not isinstance(optimizer, str) or optimizer.lower() in DISABLED_OPTIMIZERS:
        raise WorkerError("Adam and AdamW optimizers are disabled")
    if not isinstance(precision, str) or precision.strip() not in policy.supported_precisions:
        raise WorkerError("precision is invalid")
    if optimizer not in policy.supported_optimizers:
        raise WorkerError("native optimizer is invalid")
    if native["attention_backend"] not in policy.supported_attention:
        raise WorkerError("attention backend is invalid for selected backend")
    if approved_attention is not None and native["attention_backend"] not in approved_attention:
        raise WorkerError("attention backend is not approved for selected profile")
    native["device"]["runtime"] = backend
    default_arch = "gfx1036" if backend == "opencl" else "host" if backend == "cpu" else "sm_90"
    native["device"].setdefault("required_arch", default_arch)

    if policy.smoke_only:
        device = native["device"]
        model = native["model"]
        training = native["training"]
        input_section = native["input"]
        required_arch = device.get(
            "required_arch", "gfx1036" if backend == "opencl" else "host"
        )
        valid_arches = {"gfx1036", "auto"} if backend == "opencl" else {"host", "auto"}
        if required_arch not in valid_arches:
            raise WorkerError("required architecture is invalid for selected backend")
        if (
            model.get("layers") != 1
            or model.get("heads") != 1
            or not 0 < model.get("hidden_size", 0) <= 256
            or not 0 < model.get("intermediate_size", 0) <= 1024
            or not 1 < model.get("vocab_size", 0) <= 65536
        ):
            raise WorkerError("selected smoke backend model shape is unsupported")
        if input_section.get("batch_size") != 1 or input_section.get("sequence_length") != 2048:
            raise WorkerError("selected smoke backend requires batch_size=1 and sequence_length=2048")
        if device.get("model_parallel_devices"):
            raise WorkerError("selected smoke backend does not support model parallelism")
        if any(
            model.get(field, 0) not in (0, False)
            for field in (
                "num_cognitive_routes", "top_k_routes", "num_personality_experts",
                "personality_residual_expert_width", "top_k_experts",
                "expert_balancing_loss_coef", "local_attention_window",
                "moe_native_fp4", "rope_theta", "generic_moe_num_experts",
                "generic_moe_top_k", "generic_moe_expert_width",
                "generic_moe_shared_expert_width", "qkv_bias", "max_position_embeddings",
                "projection_bias", "tied_embeddings",
            )
        ) or native.get("pss_pred_rank", -1) > 0:
            raise WorkerError("selected smoke backend does not support optional model blocks")
        if any(
            field in model and model[field] not in allowed
            for field, allowed in (
                ("architecture_contract", {"ida_lattice_native_v1"}),
                ("normalization_type", {"rmsnorm"}),
                ("activation_type", {"swiglu"}),
                ("position_embedding_type", {"none"}),
            )
        ):
            raise WorkerError("selected smoke backend does not support the requested model contract")
        if request.get("init_from_model_ref") is not None or request.get("resume_from_ref") is not None:
            raise WorkerError("selected smoke backend is fresh-training-only")
        if training.get("max_steps", 1) <= 0 or training.get("grad_accumulation", 1) <= 0:
            raise WorkerError("training limits are invalid")
    native["backend"] = "native"
    native["job_id"] = run_id
    native["precision_profile"] = precision.strip()
    native["optimizer_type"] = optimizer
    native["output_dir"] = str(output_path)
    # The canonical CUDA binary reports completion through its status file;
    # keep all engine-owned evidence inside this run's private directory.
    native["status_file"] = str(output_path / "status.json")
    native["repo_root"] = str(output_path)
    native.setdefault("expected_terminal_phase", "native_smoke_complete")

    max_steps = request.get("max_steps")
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100000:
            raise WorkerError("max_steps is invalid")
        native["training"]["max_steps"] = max_steps
    configured_steps = native["training"].get("max_steps", 1)
    if isinstance(configured_steps, bool) or not isinstance(configured_steps, int) or configured_steps < 1:
        raise WorkerError("local max_steps is invalid")
    if approved_max_steps is not None and configured_steps > approved_max_steps:
        raise WorkerError("local max_steps exceeds approved policy")

    native.pop("init_from_model", None)
    native.pop("resume_from_checkpoint", None)
    for field, native_field in (
        ("init_from_model_ref", "init_from_model"),
        ("resume_from_ref", "resume_from_checkpoint"),
    ):
        if request.get(field) is not None:
            native[native_field] = str(relative_ref(artifact_root, request[field], field))

    output_path.mkdir(parents=True, exist_ok=True)
    request_path = output_path / "native-request.json"
    try:
        request_path.write_text(json.dumps(native, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise WorkerError("native request could not be prepared") from exc
    return request_path


def default_binary(backend: str) -> Path:
    try:
        binary_name = BACKEND_POLICIES[backend].worker_binary_name
    except KeyError as exc:
        raise WorkerError("native backend is invalid") from exc
    binary = Path(__file__).parents[1] / "bin" / binary_name
    if os.name == "nt" and binary.with_suffix(".exe").is_file():
        return binary.with_suffix(".exe")
    return binary


def approved_binary(
    attestation: dict[str, Any],
    binding: dict[str, Any],
    config: "WorkerConfig",
    allowed_binary_names: list[str],
) -> Path:
    backend = attestation["backend"]
    entry = binding.get("binaries", {}).get(backend)
    if not isinstance(entry, dict):
        raise WorkerError("native binary is not locally approved")
    binary = bound_asset(config.binary_root, entry, "native binary")
    if not binary.is_file():
        raise WorkerError("configured native binary is unavailable")
    if config.binary is not None and config.binary != binary:
        raise WorkerError("configured native binary does not match local binding")
    binary_name = binary.name.lower()
    if binary_name.endswith(".exe"):
        binary_name = binary_name[:-4]
    local_binary_names = {
        str(name).lower().removesuffix(".exe") for name in allowed_binary_names
    }
    if binary_name not in local_binary_names:
        raise WorkerError("native binary name is not approved by the local profile")
    if binary_sha256(binary) != attestation["binary_sha256"]:
        raise WorkerError("native binary attestation failed")
    return binary


@dataclass(frozen=True)
class WorkerConfig:
    hub_url: str
    worker_token: str
    org: str
    config_root: Path
    dataset_root: Path
    artifact_root: Path
    run_root: Path
    kernel_root: Path | None = None
    binary: Path | None = None
    binary_root: Path | None = None
    binding_path: Path | None = None
    catalog_path: Path = Path("configs/public/capabilities.json")
    deployment_map_path: Path = Path("configs/local/neural-forge-deployment-map.json")
    hardware_probe_root: Path = Path("vendor/hardware-tools")
    allowed_hub_hosts: tuple[str, ...] = DEFAULT_HUB_HOSTS
    worker_id: str | None = None
    function_key: str | None = None
    gpu_fingerprint_key: str | None = None
    leaderboard_attestation_key: str | None = None
    allow_experimental: bool = False
    poll_seconds: float = 5.0
    max_output_bytes: int = DEFAULT_MAX_NATIVE_OUTPUT_BYTES
    max_output_files: int = DEFAULT_MAX_NATIVE_OUTPUT_FILES

    def __post_init__(self) -> None:
        parsed = urlsplit(self.hub_url)
        try:
            parsed.port
        except ValueError as exc:
            raise WorkerError("hub_url must be an HTTPS URL without embedded credentials") from exc
        if (
            parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment
        ):
            raise WorkerError("hub_url must be an HTTPS URL without embedded credentials")
        if not isinstance(self.allowed_hub_hosts, (tuple, list)) or not self.allowed_hub_hosts:
            raise WorkerError("allowed_hub_hosts must be a non-empty sequence")
        if any(not isinstance(host, str) for host in self.allowed_hub_hosts):
            raise WorkerError("allowed_hub_hosts contains an invalid hostname")
        normalized_hub_hosts = tuple(host.strip().lower() for host in self.allowed_hub_hosts)
        if any(not HOSTNAME_RE.fullmatch(host) for host in normalized_hub_hosts):
            raise WorkerError("allowed_hub_hosts contains an invalid hostname")
        if parsed.hostname.lower() not in normalized_hub_hosts:
            raise WorkerError("hub_url host is not approved")
        object.__setattr__(self, "allowed_hub_hosts", normalized_hub_hosts)
        if (
            not isinstance(self.worker_token, str) or not self.worker_token
            or len(self.worker_token) > 4096
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.worker_token)
            or not isinstance(self.org, str) or not OPAQUE_ID_RE.fullmatch(self.org)
        ):
            raise WorkerError("worker token is required")
        if self.worker_id is not None and (
            not isinstance(self.worker_id, str) or not OPAQUE_ID_RE.fullmatch(self.worker_id)
        ):
            raise WorkerError("worker id is invalid")
        if self.function_key is not None and (
            not isinstance(self.function_key, str) or not self.function_key.strip() or len(self.function_key) > 512
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.function_key)
        ):
            raise WorkerError("function key is invalid")
        if self.gpu_fingerprint_key is not None and (
            not isinstance(self.gpu_fingerprint_key, str) or len(self.gpu_fingerprint_key) < 16 or len(self.gpu_fingerprint_key) > 4096
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.gpu_fingerprint_key)
        ):
            raise WorkerError("GPU fingerprint key is invalid")
        if self.leaderboard_attestation_key is not None and (
            not isinstance(self.leaderboard_attestation_key, str)
            or len(self.leaderboard_attestation_key) < 32
            or len(self.leaderboard_attestation_key) > 4096
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.leaderboard_attestation_key)
        ):
            raise WorkerError("leaderboard attestation key is invalid")
        if (
            self.gpu_fingerprint_key is not None
            and self.leaderboard_attestation_key is not None
            and self.gpu_fingerprint_key == self.leaderboard_attestation_key
        ):
            raise WorkerError("leaderboard secrets must be distinct")
        object.__setattr__(self, "hub_url", self.hub_url.rstrip("/") + "/")
        for field in ("config_root", "dataset_root", "artifact_root", "run_root"):
            object.__setattr__(self, field, getattr(self, field).resolve())
        if self.kernel_root is not None:
            object.__setattr__(self, "kernel_root", self.kernel_root.resolve())
        if self.binary is not None:
            object.__setattr__(self, "binary", self.binary.resolve())
        binary_root = self.binary_root
        if binary_root is None:
            binary_root = self.binary.parent if self.binary is not None else Path(__file__).parents[1] / "bin"
        object.__setattr__(self, "binary_root", binary_root.resolve())
        if self.binding_path is not None:
            object.__setattr__(self, "binding_path", self.binding_path.resolve())
        object.__setattr__(self, "catalog_path", self.catalog_path.resolve())
        object.__setattr__(self, "deployment_map_path", self.deployment_map_path.resolve())
        object.__setattr__(self, "hardware_probe_root", self.hardware_probe_root.resolve())
        if self.poll_seconds <= 0 or self.poll_seconds > 3600:
            raise WorkerError("poll_seconds is invalid")
        if not isinstance(self.max_output_bytes, int) or not 1 <= self.max_output_bytes <= 1 << 40:
            raise WorkerError("max_output_bytes is invalid")
        if not isinstance(self.max_output_files, int) or not 1 <= self.max_output_files <= 1_000_000:
            raise WorkerError("max_output_files is invalid")


class HubClient:
    def __init__(self, config: WorkerConfig):
        self.config = config

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = urljoin(self.config.hub_url, path.lstrip("/"))
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        request = Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.worker_token}",
                "X-Neural-Forge-Org": self.config.org,
                "Accept": "application/json",
                **({"x-functions-key": self.config.function_key} if self.config.function_key else {}),
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
            method=method,
        )
        try:
            with HUB_OPENER.open(request, timeout=30) as response:
                raw = response.read(MAX_HUB_RESPONSE_BYTES + 1)
                if len(raw) > MAX_HUB_RESPONSE_BYTES:
                    raise WorkerError("Hub worker response is too large")
                payload = json.loads(raw.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise WorkerError("Hub worker request failed") from exc
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise WorkerError("Hub worker request was rejected")
        return payload

    def claim(self) -> dict[str, Any] | None:
        payload = self._request("GET", "/api/neural-forge/worker/claim")
        job = payload.get("job")
        if job is None:
            return None
        if not isinstance(job, dict) or not isinstance(payload.get("worker_manifest"), dict):
            raise WorkerError("Hub worker claim is incomplete")
        claimed = dict(job)
        # Hub intentionally keeps the public job projection separate from the
        # internal worker manifest.  Preserve both without exposing the
        # manifest through public status updates.
        claimed["worker_manifest"] = payload["worker_manifest"]
        return claimed

    def update(
        self,
        run_id: str,
        *,
        status: str,
        event: dict[str, Any] | None = None,
        metrics_available: bool = False,
        leaderboard_attestation: dict[str, str] | None = None,
        operation_id: str | None = None,
    ) -> None:
        if not RUN_ID_RE.fullmatch(str(run_id)):
            raise WorkerError("run_id is invalid")
        body: dict[str, Any] = {
            "operation_id": operation_id or stable_operation_id(run_id, status, event, metrics_available, leaderboard_attestation),
            "status": status,
            "metrics_available": metrics_available,
        }
        if event is not None:
            body["event"] = public_event(event)
        if leaderboard_attestation is not None:
            body["leaderboard_attestation"] = dict(leaderboard_attestation)
        self._request("POST", f"/api/neural-forge/worker/runs/{run_id}", body)


def stable_operation_id(
    run_id: str,
    status: str,
    event: dict[str, Any] | None,
    metrics_available: bool,
    leaderboard_attestation: dict[str, str] | None,
) -> str:
    """Derive a retry-stable Hub operation ID without storing local details."""

    payload = {
        "run_id": run_id,
        "status": status,
        "event": event,
        "metrics_available": bool(metrics_available),
        "leaderboard_attestation": leaderboard_attestation,
    }
    return f"nfu-{canonical_fingerprint(payload)}"


def receipt_manifest(job: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, private-field-free manifest identity for local dedup."""

    manifest = job.get("worker_manifest") or job.get("manifest")
    if not isinstance(manifest, dict):
        raise WorkerError("worker manifest is required")
    reject_private_manifest_fields(manifest)
    identity = json.loads(json.dumps(manifest, sort_keys=True))
    for field in ("claimed_at", "deadline_at", "worker_subject"):
        identity.pop(field, None)
    return identity


def build_command(job: dict[str, Any], config: WorkerConfig) -> tuple[list[str], Path]:
    if not isinstance(job, dict):
        raise WorkerError("job is invalid")
    run_id = job.get("run_id")
    manifest = job.get("manifest") or job.get("worker_manifest")
    approved_max_steps: int | None = None
    approved_attention: list[str] | None = None
    approved_kernel_sha256: str | None = None
    approved_kernel_source: Path | None = None
    binding: dict[str, Any]
    if manifest is None:
        raise WorkerError("worker manifest is required")
    request, config_path, dataset_path, resolved = manifest_local_request(job, config)
    attestation = resolved["execution"]
    approved_max_steps = resolved["policy"]["max_steps"]
    approved_attention = resolved["profile"]["supported_attention"]
    approved_kernel_sha256 = resolved["kernel_sha256"]
    approved_kernel_source = resolved["kernel_source"]
    binding = resolved["binding"]
    local_device = resolved["device"]

    output_path = run_output(config.run_root, run_id)
    if output_path.exists():
        try:
            if any(output_path.iterdir()):
                raise WorkerError("local run output already exists")
        except OSError as exc:
            raise WorkerError("local run output is unavailable") from exc
    binary = approved_binary(
        attestation,
        binding,
        config,
        resolved["profile"]["binary_names"],
    )
    policy = BACKEND_POLICIES[attestation["backend"]]
    kernel_source: Path | None = None
    if attestation["backend"] == "opencl" and approved_kernel_sha256 is not None:
        kernel_source = approved_kernel_source or (binary.parent / "opencl_smoke.cl")
        try:
            if not kernel_source.is_file() or file_sha256(kernel_source) != approved_kernel_sha256:
                raise WorkerError("OpenCL kernel attestation failed")
        except BindingError as exc:
            raise WorkerError("OpenCL kernel is unavailable") from exc
    request_path = native_request_file(
        config_path,
        dataset_path,
        config.artifact_root,
        output_path,
        request,
        run_id=run_id,
        approved_max_steps=approved_max_steps,
        approved_attention=approved_attention,
    )
    command = [
        str(binary),
        "--request-json", str(request_path),
    ]
    if kernel_source is not None:
        command.extend(["--kernel-source", str(kernel_source)])
    if attestation["backend"] in {"cuda", "opencl"}:
        command.extend(["--device", str(local_device)])
    return command, output_path


def completed_metrics(
    output_path: Path,
    *,
    run_id: str,
    expected_phase: str,
    backend: str,
) -> bool:
    metric_backend = {"cpu": "native_cpu", "opencl": "native_opencl"}.get(backend)
    for filename in ("metrics.json", "status.json"):
        try:
            payload = json.loads((output_path / filename).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            filename == "metrics.json"
            and metric_backend is not None
            and payload.get("backend") == metric_backend
            and payload.get("run_id") == run_id
            and payload.get("expected_terminal_phase") == expected_phase
            and payload.get("type") == "complete"
            and payload.get("status") == "complete"
        ):
            return True
        if (
            filename == "status.json"
            and backend == "cuda"
            and payload.get("backend") == "native"
            and payload.get("job_id") == run_id
            and payload.get("phase") == expected_phase
            and (output_path / "model.safetensors").is_file()
        ):
            return True
    return False


def optional_leaderboard_attestation(job: dict[str, Any], config: WorkerConfig) -> dict[str, str] | None:
    """Probe only when the Hub explicitly marked the run as opted in.

    Probe failures are local eligibility failures, not training failures. The
    helper returns no raw SMI identity, path, command output, or diagnostic.
    """

    if (
        job.get("leaderboard_requested") is not True
        or config.gpu_fingerprint_key is None
        or config.leaderboard_attestation_key is None
    ):
        return None
    try:
        _request, _config_path, _dataset_path, resolved = manifest_local_request(job, config)
        backend = resolved["execution"]["backend"]
        if backend not in {"cuda", "opencl"}:
            return None
        expected_vendor = "nvidia" if backend == "cuda" else "amd"
        attestation = probe_hardware(
            resolved["device"],
            config.gpu_fingerprint_key,
            expected_vendor=expected_vendor,
            tools=resolved.get("hardware_probe_tools"),
        )
        if attestation is None:
            return None
        signing_key = config.leaderboard_attestation_key
        run_id = job.get("run_id")
        if not isinstance(run_id, str):
            return None
        signature_payload = "\u0000".join([
            run_id,
            attestation["vendor"],
            attestation["model"],
            attestation["fingerprint"],
        ]).encode("utf-8")
        return {
            **attestation,
            "signature": hmac.new(signing_key.encode("utf-8"), signature_payload, hashlib.sha256).hexdigest(),
        }
    except (WorkerError, OSError, ValueError, TypeError):
        return None


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        try:
            system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
            taskkill = str(Path(system_root) / "System32" / "taskkill.exe")
            subprocess.run(
                [taskkill, "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                env=sanitized_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
        return
    try:
        import signal
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


def _capture_native_stream(
    stream: Any,
    chunks: list[bytes],
    limit: int,
    overflow: threading.Event,
) -> None:
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if total < limit:
            chunks.append(chunk[: max(0, limit - total)])
        total += len(chunk)
        if total > limit:
            overflow.set()


def execute_job(job: dict[str, Any], config: WorkerConfig, client: HubClient) -> None:
    run_id = job.get("run_id")
    receipt_store = LocalRunReceiptStore(config.run_root)
    try:
        receipt_identity = receipt_manifest(job)
        receipt = receipt_store.claim(run_id, receipt_identity)
    except (WorkerError, LocalReceiptError) as exc:
        client.update(run_id, status="failed", event={"type": "worker_error", "message": str(exc)})
        return
    if receipt["decision"] == "terminal":
        client.update(
            run_id,
            status=receipt["status"],
            metrics_available=bool(receipt.get("metrics_available")),
            event={"type": "worker_replay", "status": receipt["status"]},
        )
        return
    if receipt["decision"] == "busy":
        client.update(
            run_id,
            status="running",
            event={"type": "worker_replay_busy", "status": "running"},
        )
        return
    try:
        command, output_path = build_command(job, config)
    except WorkerError as exc:
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": str(exc)})
        return

    manifest = job.get("worker_manifest") or job.get("manifest")
    policy = manifest.get("policy") if isinstance(manifest, dict) else job.get("policy")
    if not isinstance(policy, dict):
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "execution policy is missing"})
        return
    timeout_seconds = policy.get("timeout_seconds")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 7 * 24 * 60 * 60:
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "timeout policy is invalid"})
        return
    try:
        native_request = json.loads(
            (output_path / "native-request.json").read_text(encoding="utf-8")
        )
        expected_phase = native_request["expected_terminal_phase"]
        backend = native_request["device"]["runtime"]
        if not isinstance(expected_phase, str) or not isinstance(backend, str):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native execution contract is invalid"})
        return
    client.update(run_id, status="running", event={"type": "worker_started", "status": "running"})
    process: subprocess.Popen[bytes] | None = None
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    output_overflow = threading.Event()
    output_limit = threading.Event()
    timed_out = False
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=sanitized_environment(),
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(
                target=_capture_native_stream,
                args=(process.stdout, stdout_chunks, MAX_NATIVE_STDOUT_BYTES, output_overflow),
                daemon=True,
            ),
            threading.Thread(
                target=_capture_native_stream,
                args=(process.stderr, stderr_chunks, MAX_NATIVE_STDERR_BYTES, output_overflow),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + timeout_seconds
        next_output_check = 0.0
        while process.poll() is None:
            if output_overflow.is_set():
                terminate_process_tree(process)
                break
            now_monotonic = time.monotonic()
            if now_monotonic >= next_output_check:
                try:
                    used_bytes, used_files = output_usage(output_path)
                except WorkerError:
                    output_limit.set()
                    terminate_process_tree(process)
                    break
                if used_bytes > config.max_output_bytes or used_files > config.max_output_files:
                    output_limit.set()
                    terminate_process_tree(process)
                    break
                next_output_check = now_monotonic + 0.25
            if now_monotonic >= deadline:
                timed_out = True
                terminate_process_tree(process)
                break
            time.sleep(0.05)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process)
            process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=10)
    except OSError:
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native process could not start"})
        return

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    try:
        (output_path / "stderr.log").write_text(stderr, encoding="utf-8")
    except OSError:
        pass
    if timed_out:
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native process timed out"})
        return
    if output_overflow.is_set():
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native process output exceeded the safety limit"})
        return
    if output_limit.is_set():
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native process output exceeded the local quota"})
        return

    try:
        used_bytes, used_files = output_usage(output_path)
    except WorkerError:
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native output failed the local boundary check"})
        return
    if used_bytes > config.max_output_bytes or used_files > config.max_output_files:
        try:
            receipt_store.complete(run_id, receipt_identity, status="failed", metrics_available=False)
        except LocalReceiptError:
            pass
        client.update(run_id, status="failed", event={"type": "worker_error", "message": "native process output exceeded the local quota"})
        return

    event_count = 0
    for line in stdout.splitlines():
        if event_count >= MAX_NATIVE_EVENTS:
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            client.update(run_id, status="running", event=event)
            event_count += 1
    metrics_available = completed_metrics(
        output_path,
        run_id=run_id,
        expected_phase=expected_phase,
        backend=backend,
    )
    checkpoint_written = (output_path / "model.safetensors").is_file()
    final_status = "succeeded" if process.returncode == 0 and metrics_available else "failed"
    leaderboard_attestation = optional_leaderboard_attestation(job, config) if final_status == "succeeded" else None
    try:
        receipt_store.complete(
            run_id, receipt_identity, status=final_status, metrics_available=metrics_available
        )
    except LocalReceiptError:
        final_status = "failed"
        metrics_available = False
    client.update(
        run_id,
        status=final_status,
        metrics_available=metrics_available,
        leaderboard_attestation=leaderboard_attestation,
        event={
            "type": "process_exit",
            "status": final_status,
            "exit_code": process.returncode,
            "checkpoint_written": checkpoint_written,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-url", default=os.environ.get("NEURAL_FORGE_HUB_URL", ""))
    parser.add_argument("--hub-allowed-host", dest="hub_allowed_hosts", action="append", default=None)
    parser.add_argument("--worker-token", default=os.environ.get("NEURAL_FORGE_WORKER_TOKEN", ""))
    parser.add_argument("--org", default=os.environ.get("NEURAL_FORGE_ORG", ""))
    parser.add_argument("--config-root", type=Path, default=Path(os.environ.get("NEURAL_FORGE_CONFIG_ROOT", "configs")))
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("NEURAL_FORGE_DATASET_ROOT", "datasets")))
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("NEURAL_FORGE_ARTIFACT_ROOT", "artifacts")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("NEURAL_FORGE_RUN_ROOT", "runs")))
    parser.add_argument("--kernel-root", type=Path, default=Path(os.environ.get("NEURAL_FORGE_KERNEL_ROOT", "artifacts/kernels")))
    parser.add_argument("--binding", type=Path, default=Path(os.environ.get("NEURAL_FORGE_LOCAL_BINDING", "configs/local/neural-forge-binding.json")))
    parser.add_argument("--catalog", type=Path, default=Path(os.environ.get("NEURAL_FORGE_CAPABILITY_CATALOG", "configs/public/capabilities.json")))
    parser.add_argument("--deployment-map", type=Path, default=Path(os.environ.get("NEURAL_FORGE_DEPLOYMENT_MAP", "configs/local/neural-forge-deployment-map.json")))
    parser.add_argument("--hardware-probe-root", type=Path, default=Path(os.environ.get("NEURAL_FORGE_HARDWARE_PROBE_ROOT", "vendor/hardware-tools")))
    parser.add_argument("--worker-id", default=os.environ.get("NEURAL_FORGE_WORKER_ID"))
    parser.add_argument("--function-key", default=os.environ.get("NEURAL_FORGE_FUNCTION_KEY"))
    parser.add_argument("--gpu-fingerprint-key", default=os.environ.get("NEURAL_FORGE_GPU_FINGERPRINT_KEY"))
    parser.add_argument("--leaderboard-attestation-key", default=os.environ.get("NEURAL_FORGE_LEADERBOARD_ATTESTATION_KEY"))
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help="acknowledge execution of catalogued experimental profiles",
    )
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("NEURAL_FORGE_POLL_SECONDS", "5")))
    parser.add_argument("--max-output-bytes", type=int, default=int(os.environ.get("NEURAL_FORGE_MAX_OUTPUT_BYTES", str(DEFAULT_MAX_NATIVE_OUTPUT_BYTES))))
    parser.add_argument("--max-output-files", type=int, default=int(os.environ.get("NEURAL_FORGE_MAX_OUTPUT_FILES", str(DEFAULT_MAX_NATIVE_OUTPUT_FILES))))
    parser.add_argument("--once", action="store_true", help="claim at most one job and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        configured_hub_hosts = tuple(args.hub_allowed_hosts or (
            item.strip()
            for item in os.environ.get("NEURAL_FORGE_HUB_ALLOWED_HOSTS", "").split(",")
            if item.strip()
        )) or DEFAULT_HUB_HOSTS
        config = WorkerConfig(
            hub_url=args.hub_url,
            worker_token=args.worker_token,
            org=args.org,
            config_root=args.config_root,
            dataset_root=args.dataset_root,
            artifact_root=args.artifact_root,
            run_root=args.run_root,
            kernel_root=args.kernel_root,
            binary=args.binary,
            binding_path=args.binding,
            catalog_path=args.catalog,
            deployment_map_path=args.deployment_map,
            hardware_probe_root=args.hardware_probe_root,
            allowed_hub_hosts=configured_hub_hosts,
            worker_id=args.worker_id,
            function_key=args.function_key,
            gpu_fingerprint_key=args.gpu_fingerprint_key,
            leaderboard_attestation_key=args.leaderboard_attestation_key,
            allow_experimental=args.allow_experimental,
            poll_seconds=args.poll_seconds,
            max_output_bytes=args.max_output_bytes,
            max_output_files=args.max_output_files,
        )
        client = HubClient(config)
        while True:
            job = client.claim()
            if job is not None:
                execute_job(job, config, client)
            if args.once:
                return 0
            time.sleep(config.poll_seconds)
    except WorkerError as exc:
        raise SystemExit(f"neural-forge-worker: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
