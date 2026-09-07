import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.local_receipt import (
    RECEIPT_SCHEMA_VERSION,
    LocalReceiptError,
    LocalRunReceiptStore,
    build_execution_receipt_ref,
    canonical_fingerprint,
)
from scripts.model_contracts import ModelContractError, validate_model_shape
from scripts.private_runtime import PrivateRuntimeError, validate_private_runtime


class _PrivateEvidenceAdapter:
    """A minimal deployment adapter double; private schema semantics stay external."""

    @staticmethod
    def is_request_schema(schema_version: object) -> bool:
        return schema_version == "opaque-private-schema.v1"

    @staticmethod
    def validate_execution_evidence(evidence: object, descriptor: object) -> dict[str, object]:
        if not isinstance(evidence, dict) or not isinstance(descriptor, dict):
            raise ValueError("invalid private evidence")
        if set(evidence) != {"evidence_id", "evidence_sha256"}:
            raise ValueError("private evidence shape is invalid")
        if descriptor.get("request_sha256") != evidence["evidence_sha256"]:
            raise ValueError("private evidence does not match descriptor")
        return dict(evidence)


class PrivateBoundaryTests(unittest.TestCase):
    def test_evidence_tiered_model_contracts_reject_shared_expert_moe(self) -> None:
        self.assertEqual(
            validate_model_shape("hf_gpt2_native_v1", {"architecture_contract": "hf_gpt2_native_v1"})["tier"],
            "observed",
        )
        self.assertEqual(
            validate_model_shape("generic_moe_native_v1", {"architecture_contract": "generic_moe_native_v1"})["tier"],
            "experimental",
        )
        with self.assertRaisesRegex(ModelContractError, "shared-expert"):
            validate_model_shape(
                "generic_moe_native_v1",
                {
                    "architecture_contract": "generic_moe_native_v1",
                    "generic_moe_shared_expert_width": 128,
                },
            )

    def test_private_runtime_reference_has_no_source_or_url_surface(self) -> None:
        valid = {
            "package_id": "nf-private-kernel-v1",
            "package_sha256": "a" * 64,
            "tier": "private_advanced",
        }
        self.assertEqual(validate_private_runtime(valid), valid)
        for field, value in (
            ("url", "https://private.example/package"),
            ("source", "private/math.cu"),
            ("path", "C:/private/math"),
            ("command", "native-private-runner"),
        ):
            candidate = dict(valid)
            candidate[field] = value
            with self.subTest(field=field), self.assertRaises(PrivateRuntimeError):
                validate_private_runtime(candidate)

    def test_local_receipt_deduplicates_and_rejects_manifest_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalRunReceiptStore(Path(temporary) / "runs")
            manifest = {"run_id": "nf-receipt12345678", "request": {"profile": "public-local"}}
            first = store.claim(manifest["run_id"], manifest)
            self.assertEqual(first["decision"], "new")
            self.assertEqual(store.claim(manifest["run_id"], manifest)["decision"], "busy")
            store.complete(manifest["run_id"], manifest, status="succeeded", metrics_available=True)
            replay = store.claim(manifest["run_id"], manifest)
            self.assertEqual(replay["decision"], "terminal")
            cancelled_manifest = {"run_id": "nf-cancelled123456", "request": {"profile": "public-local"}}
            store.claim(cancelled_manifest["run_id"], cancelled_manifest)
            store.complete(cancelled_manifest["run_id"], cancelled_manifest, status="cancelled", metrics_available=False)
            cancelled_replay = store.claim(cancelled_manifest["run_id"], cancelled_manifest)
            self.assertEqual(cancelled_replay["decision"], "terminal")
            changed = {**manifest, "request": {"profile": "different"}}
            with self.assertRaises(LocalReceiptError):
                store.claim(manifest["run_id"], changed)
            with self.assertRaisesRegex(LocalReceiptError, "already terminal"):
                store.complete(manifest["run_id"], manifest, status="failed", metrics_available=False)
            legacy_manifest = {"run_id": "nf-legacy123456", "request": {"profile": "public-local"}}
            store.claim(legacy_manifest["run_id"], legacy_manifest)
            legacy_path = Path(temporary) / "runs" / ".receipts" / f"{legacy_manifest['run_id']}.json"
            legacy_receipt = json.loads(legacy_path.read_text())
            legacy_receipt["schema_version"] = "neural-foundry-run-receipt.v1"
            legacy_path.write_text(json.dumps(legacy_receipt), encoding="utf-8")
            store.complete(legacy_manifest["run_id"], legacy_manifest, status="cancelled", metrics_available=False)
            upgraded = json.loads(legacy_path.read_text())
            self.assertEqual(upgraded["schema_version"], RECEIPT_SCHEMA_VERSION)
            receipt = json.loads((Path(temporary) / "runs" / ".receipts" / "nf-receipt12345678.json").read_text())
            self.assertNotIn("command", receipt)
            self.assertNotIn("path", receipt)
            self.assertNotIn("source", receipt)

    def test_local_receipt_rejects_a_symlinked_receipt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            target = Path(temporary) / "redirected"
            target.mkdir()
            root.mkdir()
            try:
                (root / ".receipts").symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic-link creation is unavailable")
            store = LocalRunReceiptStore(root)
            with self.assertRaisesRegex(LocalReceiptError, "receipt directory"):
                store.claim("nf-symlink12345678", {"request": {"profile": "public-local"}})

    def test_private_adapter_evidence_stays_private_and_is_pinned(self) -> None:
        evidence = {
            "evidence_id": "private-evidence-a",
            "evidence_sha256": "a" * 64,
        }
        descriptor = {
            "schema_version": "opaque-private-schema.v1",
            "request_sha256": evidence["evidence_sha256"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalRunReceiptStore(Path(temporary) / "runs")
            manifest = {"run_id": "nf-native-evidence123", "request": {"profile": "private"}}
            store.claim(manifest["run_id"], manifest)
            with self.assertRaisesRegex(LocalReceiptError, "requires an adapter"):
                store.prepare_private_evidence(
                    manifest["run_id"], manifest,
                    native_execution=evidence,
                    native_descriptor=descriptor,
                )
            with patch("scripts.local_receipt.load_private_adapter", return_value=_PrivateEvidenceAdapter()):
                private_ref = store.prepare_private_evidence(
                    manifest["run_id"],
                    manifest,
                    native_execution=evidence,
                    native_descriptor=descriptor,
                )
            stored = store.complete(
                manifest["run_id"],
                manifest,
                status="succeeded",
                metrics_available=True,
                **private_ref,
            )
            self.assertNotIn("native_execution", stored)
            private_record = json.loads(
                (Path(temporary) / "runs" / ".private-evidence" / f"{private_ref['evidence_ref']}.json").read_text()
            )
            self.assertEqual(private_record["native_execution"], evidence)
            receipt_ref = build_execution_receipt_ref(
                manifest,
                status="succeeded",
                metrics_available=True,
                **private_ref,
            )
            self.assertNotIn("native_execution", json.dumps(receipt_ref))
            self.assertNotIn("path", json.dumps(receipt_ref).lower())
            self.assertEqual(
                receipt_ref["receipt_sha256"],
                canonical_fingerprint({key: value for key, value in receipt_ref.items() if key != "receipt_sha256"}),
            )
            with self.assertRaisesRegex(LocalReceiptError, "private evidence is unavailable"):
                store.complete(
                    "nf-native-evidence123",
                    manifest,
                    status="succeeded",
                    metrics_available=True,
                    evidence_ref="evidence-missing",
                    evidence_sha256="1" * 64,
                )

        with tempfile.TemporaryDirectory() as temporary, \
                patch("scripts.local_receipt.load_private_adapter", return_value=_PrivateEvidenceAdapter()):
            with self.assertRaisesRegex(LocalReceiptError, "is invalid"):
                store = LocalRunReceiptStore(Path(temporary) / "runs")
                store.prepare_private_evidence(
                    "nf-native-evidence123",
                    {"run_id": "nf-native-evidence123"},
                    native_execution={**evidence, "path": "C:/private/output"},
                    native_descriptor=descriptor,
                )

    def test_legacy_full_native_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            store = LocalRunReceiptStore(root)
            manifest = {"run_id": "nf-legacy-native123", "request": {"profile": "private"}}
            store.claim(manifest["run_id"], manifest)
            path = root / ".receipts" / f"{manifest['run_id']}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["native_execution"] = {"profile_id": "private-profile-a"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LocalReceiptError, "unsupported or private fields"):
                store.claim(manifest["run_id"], manifest)


if __name__ == "__main__":
    unittest.main()
