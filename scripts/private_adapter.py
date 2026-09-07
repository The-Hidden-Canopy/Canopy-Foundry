"""Optional deployment-owned adapter seam for non-public native runtimes.

This public worker never selects an adapter from a Hub manifest or deployment
map.  An operator may provide one only through the local environment, and the
adapter must expose the small, path-free validation/projection interface below.
Without it, private execution is unavailable and fails closed.
"""

from __future__ import annotations

import importlib
import os
import re
from types import ModuleType


_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_REQUIRED = frozenset({
    "is_request_schema",
    "validate_profile_contract",
    "validate_descriptor",
    "validate_source_manifest",
    "build_native_request",
    "valid_metrics",
    "build_execution_evidence",
    "validate_execution_evidence",
    "prepare_local_execution",
})


class PrivateAdapterError(RuntimeError):
    """The deployment-owned native adapter is unavailable or incomplete."""


def load_private_adapter() -> ModuleType | None:
    """Load only a local operator-configured adapter, never a caller value."""

    module_name = os.environ.get("NEURAL_FORGE_PRIVATE_ADAPTER_MODULE", "").strip()
    if not module_name:
        return None
    if not _MODULE_NAME.fullmatch(module_name):
        raise PrivateAdapterError("private adapter module name is invalid")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ValueError) as exc:
        raise PrivateAdapterError("private adapter is unavailable") from exc
    if any(not callable(getattr(module, name, None)) for name in _REQUIRED):
        raise PrivateAdapterError("private adapter interface is incomplete")
    return module
