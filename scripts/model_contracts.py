"""Evidence-tiered model contracts for the local Neural Foundry runtime.

The contract names describe validated tensor/layout adapters.  They are not
model downloads, authority, or a promise that every upstream checkpoint is
compatible.  Model bytes and conversion tooling remain local to the operator.
"""

from __future__ import annotations

from typing import Any
import re


class ModelContractError(ValueError):
    """A model contract or shape was not admitted."""


OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# These tiers deliberately describe evidence posture, not performance or
# promotion.  The generic MoE contract is portability evidence only.
MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "hf_gpt2_native_v1": {
        "tier": "observed",
        "families": ["gpt2-compatible"],
        "shared_expert": False,
    },
    "ida_lattice_native_v1": {
        "tier": "observed",
        "families": ["smollm2-shaped", "qwen2.5-shaped-dense"],
        "shared_expert": False,
    },
    "generic_moe_native_v1": {
        "tier": "experimental",
        "families": ["mixtral-shaped"],
        "shared_expert": False,
    },
}


def validate_model_contract_id(value: Any, label: str = "model_contract_id") -> str:
    if not isinstance(value, str) or not OPAQUE_ID_RE.fullmatch(value):
        raise ModelContractError(f"{label} is invalid")
    if value not in MODEL_CONTRACTS:
        raise ModelContractError(f"{label} is unsupported")
    return value


def validate_model_shape(
    contract_id: Any,
    model: Any,
    *,
    label: str = "model",
) -> dict[str, Any]:
    """Validate public shape metadata without inspecting model bytes.

    Shared-expert Qwen-MoE layouts are intentionally rejected.  The generic
    MoE contract covers the observed Mixtral-shaped portability surface only.
    """

    contract = validate_model_contract_id(contract_id, f"{label}.contract_id")
    if not isinstance(model, dict):
        raise ModelContractError(f"{label} is invalid")
    declared = model.get("architecture_contract")
    if declared is not None and declared != contract:
        raise ModelContractError(f"{label}.architecture_contract does not match contract")
    shared_width = model.get("generic_moe_shared_expert_width", 0)
    if isinstance(shared_width, bool) or not isinstance(shared_width, int) or shared_width < 0:
        raise ModelContractError(f"{label}.generic_moe_shared_expert_width is invalid")
    if contract == "generic_moe_native_v1" and shared_width:
        raise ModelContractError("Qwen-MoE shared-expert layouts are not supported")
    if contract != "generic_moe_native_v1" and any(
        model.get(field, 0) not in (0, False, None)
        for field in (
            "generic_moe_num_experts",
            "generic_moe_top_k",
            "generic_moe_expert_width",
            "generic_moe_shared_expert_width",
        )
    ):
        raise ModelContractError(f"{label} contains MoE fields for a dense contract")
    return {"model_contract_id": contract, "tier": MODEL_CONTRACTS[contract]["tier"]}


def contract_for_model(model: Any, *, default: str | None = None) -> str | None:
    if isinstance(model, dict) and model.get("architecture_contract") is not None:
        return validate_model_contract_id(model["architecture_contract"], "model.architecture_contract")
    if default is None:
        return None
    return validate_model_contract_id(default, "default model contract")


def contract_is_experimental(contract_id: str) -> bool:
    return MODEL_CONTRACTS[validate_model_contract_id(contract_id)]["tier"] == "experimental"
