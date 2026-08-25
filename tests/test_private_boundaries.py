from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.local_receipt import LocalReceiptError, LocalRunReceiptStore
from scripts.model_contracts import ModelContractError, validate_model_shape
from scripts.private_runtime import PrivateRuntimeError, validate_private_runtime


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
            ("command", "nvcc private.cu"),
        ):
            candidate = dict(valid)
            candidate[field] = value
            with self.subTest(field=field), self.assertRaises(PrivateRuntimeError):
                validate_private_runtime(candidate)

    def test_local_receipt_deduplicates_and_rejects_manifest_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalRunReceiptStore(Path(temporary) / "runs")
            manifest = {"run_id": "nf-receipt12345678", "request": {"profile": "cuda-local"}}
            first = store.claim(manifest["run_id"], manifest)
            self.assertEqual(first["decision"], "new")
            self.assertEqual(store.claim(manifest["run_id"], manifest)["decision"], "busy")
            store.complete(manifest["run_id"], manifest, status="succeeded", metrics_available=True)
            replay = store.claim(manifest["run_id"], manifest)
            self.assertEqual(replay["decision"], "terminal")
            changed = {**manifest, "request": {"profile": "different"}}
            with self.assertRaises(LocalReceiptError):
                store.claim(manifest["run_id"], changed)
            receipt = json.loads((Path(temporary) / "runs" / ".receipts" / "nf-receipt12345678.json").read_text())
            self.assertNotIn("command", receipt)
            self.assertNotIn("path", receipt)
            self.assertNotIn("source", receipt)


if __name__ == "__main__":
    unittest.main()
