from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from scripts.neural_forge_worker import completed_metrics


class PrivateCompletionTests(unittest.TestCase):
    def completed(self, root, adapter, *, backend="cuda", device=None):
        return completed_metrics(root, run_id="nf-private1234", expected_phase="complete", backend=backend,
            private_adapter_native=True, adapter=adapter, native_descriptor={"contract": "opaque"},
            expected_device=device)

    def test_private_rejection_cannot_fall_back_to_public_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"fixture")
            (root / "status.json").write_text(json.dumps({"backend": "native", "job_id": "nf-private1234", "phase": "complete"}))
            (root / "metrics.json").write_text('{}')
            adapter = SimpleNamespace(valid_metrics=Mock(return_value=False))
            self.assertFalse(self.completed(root, adapter))
            adapter.valid_metrics.assert_called_once_with({}, {"contract": "opaque"})

    def test_public_shaped_metrics_cannot_override_private_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "model.safetensors").write_bytes(b"fixture")
            metrics = {"backend": "native_cuda", "type": "complete", "status": "complete", "checkpoint_written": True,
                "step": 1, "optimizer_steps": 1, "tokens": 8, "loss": 1., "learning_rate": .01,
                "tokens_per_second": 3., "grad_norm": .2, "output": str(root)}
            (root / "metrics.json").write_text(json.dumps(metrics))
            self.assertFalse(self.completed(root, SimpleNamespace(valid_metrics=lambda *args: False)))

    def test_missing_adapter_or_metrics_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(self.completed(root, None))
            (root / "metrics.json").write_text('{}')
            self.assertFalse(self.completed(root, None))

    def test_only_explicit_adapter_true_accepts_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "metrics.json").write_text('{}')
            self.assertTrue(self.completed(root, SimpleNamespace(valid_metrics=lambda *args: True)))
            self.assertFalse(self.completed(root, SimpleNamespace(valid_metrics=lambda *args: "true")))
            self.assertFalse(self.completed(root, SimpleNamespace(valid_metrics=Mock(side_effect=ValueError))))

    def test_hip_metrics_must_match_locally_selected_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter = SimpleNamespace(valid_metrics=lambda *args: True)
            for observed in (0, True, "3", None):
                (root / "metrics.json").write_text(json.dumps({"device_index": observed}))
                with self.subTest(observed=observed):
                    self.assertFalse(self.completed(root, adapter, backend="hip", device=3))
            (root / "metrics.json").write_text('{"device_index":3}')
            self.assertTrue(self.completed(root, adapter, backend="hip", device=3))


if __name__ == "__main__":
    unittest.main()
