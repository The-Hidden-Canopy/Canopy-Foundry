"""Machine-local, path-free run receipts for native retry deduplication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


class LocalReceiptError(RuntimeError):
    """A local run receipt is invalid or conflicts with the requested run."""


RECEIPT_SCHEMA_VERSION = "neural-foundry-run-receipt.v2"
LEGACY_RECEIPT_SCHEMA_VERSION = "neural-foundry-run-receipt.v1"
RUN_ID_RE = re.compile(r"^nf-[a-z0-9-]{8,80}$", re.IGNORECASE)


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise LocalReceiptError("run_id is invalid")
    return run_id


class LocalRunReceiptStore:
    """Persist only opaque run identity, fingerprints, and bounded state.

    The receipt directory is expected to be ignored and machine-local.  It
    contains no command, filesystem path, model name, dataset bytes, logs, or
    telemetry.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.receipt_root = self.root / ".receipts"

    def _path(self, run_id: str) -> Path:
        return self.receipt_root / f"{_safe_run_id(run_id)}.json"

    def _ensure_receipt_root(self) -> None:
        try:
            self.receipt_root.mkdir(parents=True, exist_ok=True)
            mode = os.lstat(self.receipt_root).st_mode
        except OSError as exc:
            raise LocalReceiptError("local receipt directory is unavailable") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise LocalReceiptError("local receipt directory cannot be a symbolic link")

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
        allowed_statuses = {"in_progress", "succeeded", "failed"}
        if payload.get("schema_version") == RECEIPT_SCHEMA_VERSION:
            allowed_statuses.add("cancelled")
        if payload.get("status") not in allowed_statuses:
            raise LocalReceiptError("local receipt status is invalid")
        if not isinstance(payload.get("manifest_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", payload["manifest_sha256"]):
            raise LocalReceiptError("local receipt fingerprint is invalid")
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

    def complete(self, run_id: str, manifest: dict[str, Any], *, status: str, metrics_available: bool) -> dict[str, Any]:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise LocalReceiptError("terminal receipt status is invalid")
        path = self._path(_safe_run_id(run_id))
        existing = self._read(path)
        fingerprint = canonical_fingerprint(manifest)
        if existing is None or existing["manifest_sha256"] != fingerprint:
            raise LocalReceiptError("local receipt does not match manifest")
        if existing["status"] in {"succeeded", "cancelled"} and existing["status"] != status:
            raise LocalReceiptError("local receipt is already terminal")
        updated = {
            **existing,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": status,
            "metrics_available": bool(metrics_available),
        }
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
            os.replace(temporary, path)
        except OSError as exc:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
            raise LocalReceiptError("local receipt could not be updated") from exc
