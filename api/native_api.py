#!/usr/bin/env python3
"""Pure native request admission for the governed external wrapper.

This module intentionally has no HTTP server, bearer-token listener, child
process launcher, or telemetry endpoint. The external governed wrapper owns
authority, supervision, and execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

_AUTHORITY_KEYS = {
    "org", "organization", "role", "role_tier", "justification",
    "transition", "audit", "audit_event", "domain_event",
    "execute_transition", "append_domain_event", "promotion",
    "promotion_enabled", "telemetry", "ontology", "evidence", "socket",
    "serve", "queue", "worker", "repo_root", "status_file",
}
_REQUEST_KEYS = {
    "backend", "attention_backend", "precision_profile",
    "optimizer_state_precision", "optimizer_type", "gradient_buffer_precision",
    "gemm_accumulator_precision", "family", "seat", "version", "device",
    "model", "training", "input", "init_from_model",
    "resume_from_checkpoint", "grad_accum_override", "max_samples", "seed",
    "pss_pred_rank", "pss_pred_lr_scale", "pss_pred_aux_weight",
    "pss_conditioning_mode", "pss_aux_normalize_mode", "global_clip_override",
    "act_row_clip_override", "spec_hash", "lion_lr_scale_override",
    "lion_wd_scale_override", "lion_beta1_override", "lion_beta2_override",
    "lion_trust_ratio_enabled_override", "lion_trust_ratio_lo_override",
    "lion_trust_ratio_hi_override",
}
_DEVICE_KEYS = {
    "runtime", "required_arch", "precision", "model_parallel_devices",
    "pipeline_split_layer",
}
_MODEL_KEYS = {
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
_TRAINING_KEYS = {
    "microbatch", "per_device_train_batch_size", "grad_accumulation",
    "gradient_accumulation_steps", "learning_rate", "max_steps",
}
_INPUT_KEYS = {
    "token_blocks", "label_blocks", "seg_blocks", "batch_size",
    "sequence_length", "shape",
}


class RequestFailure(ValueError):
    """A request failed closed without exposing sensitive details."""


def _reject_authority_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _AUTHORITY_KEYS:
                raise RequestFailure("authority fields are not accepted")
            _reject_authority_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_authority_fields(child)


def _require_keys(value: Any, allowed: set[str]) -> None:
    if not isinstance(value, dict) or set(value) - allowed:
        raise RequestFailure("request section is invalid")


def _relative_path(root: Path, value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RequestFailure("request path is invalid")
    candidate = Path(value)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise RequestFailure("request path is invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RequestFailure("request path is invalid") from exc
    return str(resolved)


def _require_regular(root: Path, value: Any) -> str:
    resolved = _relative_path(root, value)
    if not Path(resolved).is_file():
        raise RequestFailure("required input is unavailable")
    return resolved


def _prepare_request(
    payload: Any,
    dataset_root: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], int]:
    if not isinstance(payload, dict):
        raise RequestFailure("body is invalid")
    _reject_authority_fields(payload)
    if set(payload) != {"device", "request"}:
        raise RequestFailure("body shape is invalid")

    device = payload["device"]
    if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device <= 255:
        raise RequestFailure("device is invalid")

    request = payload["request"]
    _require_keys(request, _REQUEST_KEYS)
    if request.get("backend") not in {None, "native"}:
        raise RequestFailure("backend is invalid")
    if "output_dir" in request:
        raise RequestFailure("output ownership is wrapper-controlled")

    normalized = dict(request)
    if "device" in normalized:
        _require_keys(normalized["device"], _DEVICE_KEYS)
        normalized["device"] = dict(normalized["device"])
        if normalized["device"].get("runtime") not in {None, "cuda", "opencl", "cpu"}:
            raise RequestFailure("runtime is invalid")
        if "model_parallel_devices" in normalized["device"]:
            raise RequestFailure("model parallel requests are not supported")
    for section, allowed in (
        ("model", _MODEL_KEYS),
        ("training", _TRAINING_KEYS),
        ("input", _INPUT_KEYS),
    ):
        if section in normalized:
            _require_keys(normalized[section], allowed)
            normalized[section] = dict(normalized[section])

    input_section = normalized.setdefault("input", {})
    if not isinstance(input_section, dict):
        raise RequestFailure("input section is invalid")
    for field in ("token_blocks", "label_blocks"):
        if field not in input_section:
            raise RequestFailure("required input is missing")
        input_section[field] = _require_regular(dataset_root, input_section[field])
    if input_section.get("seg_blocks"):
        input_section["seg_blocks"] = _require_regular(dataset_root, input_section["seg_blocks"])
    elif "seg_blocks" in input_section:
        input_section["seg_blocks"] = ""

    if normalized.get("init_from_model"):
        normalized["init_from_model"] = _relative_path(artifact_root, normalized["init_from_model"])
    if normalized.get("resume_from_checkpoint"):
        normalized["resume_from_checkpoint"] = _relative_path(
            artifact_root, normalized["resume_from_checkpoint"]
        )
    if normalized.get("init_from_model") and normalized.get("resume_from_checkpoint"):
        raise RequestFailure("parent inputs are mutually exclusive")
    return normalized, device


@dataclass
class ApiConfig:
    dataset_root: Path
    artifact_root: Path
    run_root: Path
    binary: Path

    def __post_init__(self) -> None:
        self.dataset_root = self.dataset_root.resolve()
        self.artifact_root = self.artifact_root.resolve()
        self.run_root = self.run_root.resolve()
        self.binary = self.binary.resolve()


class NativeApiService:
    """In-process request admission only; execution remains external."""

    def __init__(self, config: ApiConfig):
        self.config = config

    def validate(self, payload: Any) -> dict[str, Any]:
        _prepare_request(payload, self.config.dataset_root, self.config.artifact_root)
        return {"valid": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        result = NativeApiService(ApiConfig(
            dataset_root=args.dataset_root,
            artifact_root=args.artifact_root,
            run_root=args.run_root,
            binary=args.binary,
        )).validate(payload)
    except (OSError, json.JSONDecodeError, RequestFailure) as exc:
        raise SystemExit(f"native request rejected: {exc}") from exc
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
