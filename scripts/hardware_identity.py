"""Private GPU identity probe for optional Neural Foundry leaderboard entries.

Only the sanitized vendor/model and an HMAC fingerprint leave the worker.
Raw SMI UUIDs/serials are parsed and discarded in this module.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping


MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/-]{0,127}$")
MAX_DEVICE = 255
_EXPECTED_TOOLS = {"nvidia": "nvidia-smi", "amd": "rocm-smi"}
_ENVIRONMENT_KEYS = {"SystemRoot", "WINDIR", "TEMP", "TMP"}


def _clean_model(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    model = " ".join(value.strip().split())
    if not model or len(model) > 128 or not MODEL_RE.fullmatch(model):
        return None
    return model


def _clean_raw_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw_id = value.strip()
    if not raw_id or len(raw_id) > 512 or any(ord(char) < 0x20 or ord(char) == 0x7F for char in raw_id):
        return None
    if raw_id.lower() in {"n/a", "unknown", "not supported", "[not supported]"}:
        return None
    return raw_id


def _csv_rows(output: str) -> list[list[str]]:
    try:
        return [[item.strip() for item in row] for row in csv.reader(output.splitlines()) if any(item.strip() for item in row)]
    except (csv.Error, TypeError):
        return []


def _identity_from_rows(rows: list[list[str]], vendor: str) -> tuple[str, str] | None:
    if not rows:
        return None
    header_index = None
    model_index = None
    id_index = None
    for index, row in enumerate(rows[:3]):
        lowered = [cell.lower() for cell in row]
        possible_model = next((column for column, cell in enumerate(lowered) if any(token in cell for token in ("name", "product", "series", "model"))), None)
        possible_id = next((column for column, cell in enumerate(lowered) if any(token in cell for token in ("uuid", "unique", "serial", "id"))), None)
        if possible_model is not None and possible_id is not None:
            header_index = index
            model_index = possible_model
            id_index = possible_id
            break
    if header_index is not None:
        for row in rows[header_index + 1:]:
            if max(model_index, id_index) < len(row):
                model = _clean_model(row[model_index])
                raw_id = _clean_raw_id(row[id_index])
                if model and raw_id:
                    return model, raw_id
    row = rows[0]
    if vendor == "nvidia" and len(row) >= 2:
        model = _clean_model(row[0])
        raw_id = _clean_raw_id(row[1])
        if model and raw_id:
            return model, raw_id
    if vendor == "amd" and len(row) >= 3:
        model = _clean_model(row[-2])
        raw_id = _clean_raw_id(row[-1])
        if model and raw_id:
            return model, raw_id
    return None


def _fingerprint(key: str, vendor: str, model: str, raw_id: str) -> str:
    message = f"{vendor}\x00{model}\x00{raw_id}".encode("utf-8")
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def sanitized_probe_environment() -> dict[str, str]:
    """Return the only environment allowed for a deployment-owned probe."""

    return {
        name: value
        for name, value in os.environ.items()
        if name in _ENVIRONMENT_KEYS and value
    }


def _trusted_tool(tools: Mapping[str, str | Path] | None, vendor: str) -> Path | None:
    if not isinstance(tools, Mapping):
        return None
    value = tools.get(vendor)
    if not isinstance(value, (str, Path)):
        return None
    tool = Path(value)
    expected = _EXPECTED_TOOLS[vendor]
    if (
        not tool.is_absolute()
        or tool.is_symlink()
        or not tool.is_file()
        or tool.stem.lower() != expected
    ):
        return None
    return tool


def probe_hardware(
    device: int,
    key: str | None,
    *,
    expected_vendor: str | None = None,
    tools: Mapping[str, str | Path] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, str] | None:
    """Return a public-safe identity for one local GPU, or ``None``.

    The worker supplies absolute, hash-verified executable paths from its
    ignored local binding. A missing driver, missing key, malformed output, or
    unsupported device simply makes the run ineligible for the optional
    leaderboard; it does not fail training. Ambient PATH lookup is never used.
    """

    if expected_vendor not in {None, "nvidia", "amd"}:
        return None
    if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device <= MAX_DEVICE:
        return None
    if not isinstance(key, str) or len(key) < 16 or any(ord(char) < 0x20 or ord(char) == 0x7F for char in key):
        return None
    commands = (
        ("nvidia", [f"--id={device}", "--query-gpu=name,uuid", "--format=csv,noheader,nounits"]),
        ("amd", ["--showproductname", "--showuniqueid", "--csv", "--device", str(device)]),
    )
    for vendor, arguments in commands:
        if expected_vendor is not None and vendor != expected_vendor:
            continue
        tool = _trusted_tool(tools, vendor)
        if tool is None:
            continue
        command = [str(tool), *arguments]
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=sanitized_probe_environment(),
            )
        except (OSError, subprocess.SubprocessError, TypeError):
            continue
        if getattr(result, "returncode", 1) != 0:
            continue
        parsed = _identity_from_rows(_csv_rows(getattr(result, "stdout", "")), vendor)
        if parsed is None:
            continue
        model, raw_id = parsed
        return {"vendor": vendor, "model": model, "fingerprint": _fingerprint(key, vendor, model, raw_id)}
    return None


__all__ = ["probe_hardware", "sanitized_probe_environment"]
