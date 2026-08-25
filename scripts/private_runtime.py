"""Opaque references to deployment-owned advanced runtime packages.

This module intentionally validates metadata only.  It never downloads,
serves, opens, or names a private source package.  A deployment worker resolves
the reference through its ignored local bindings and verifies the package
digest before execution.
"""

from __future__ import annotations

from typing import Any
import re


class PrivateRuntimeError(ValueError):
    """A private runtime reference is malformed or unsafe."""


PRIVATE_RUNTIME_SCHEMA_VERSION = "neural-foundry-private-runtime.v1"
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PRIVATE_RUNTIME_FIELDS = {"package_id", "package_sha256", "tier"}
PRIVATE_RUNTIME_TIERS = {"private_advanced"}


def _opaque(value: Any, label: str) -> str:
    if not isinstance(value, str) or not OPAQUE_ID_RE.fullmatch(value) or ".." in value:
        raise PrivateRuntimeError(f"{label} is invalid")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PrivateRuntimeError(f"{label} is invalid")
    return value.lower()


def validate_private_runtime(value: Any, label: str = "private runtime") -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != PRIVATE_RUNTIME_FIELDS:
        raise PrivateRuntimeError(f"{label} is invalid")
    tier = _opaque(value["tier"], f"{label}.tier")
    if tier not in PRIVATE_RUNTIME_TIERS:
        raise PrivateRuntimeError(f"{label}.tier is invalid")
    return {
        "package_id": _opaque(value["package_id"], f"{label}.package_id"),
        "package_sha256": _sha256(value["package_sha256"], f"{label}.package_sha256"),
        "tier": tier,
    }


def validate_optional_private_runtime(value: Any, label: str = "private runtime") -> dict[str, str] | None:
    if value is None:
        return None
    return validate_private_runtime(value, label)
