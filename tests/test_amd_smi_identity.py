"""ROCm 7.2 CLI contract fixtures; no driver or GPU is exercised."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.hardware_identity import probe_hardware, supported_probe_name
from scripts.neural_forge_worker import (
    hardware_probe_tools, optional_leaderboard_attestation,
)


class AmdSmiIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tool = Path(self.temp.name) / "amd-smi"
        self.tool.write_bytes(b"fixture only")
        self.calls = []

    def static(self, device=3):
        return {"gpu_data": [{
            "gpu": device,
            "asic": {"market_name": "AMD Instinct MI300X", "vendor_id": "0x1002",
                     "asic_serial": "0xfixture-private-serial"},
            "bus": {"bdf": "0000:41:00.0"},
        }]}

    def run_probe(self, outputs, *, hip=False, device=3):
        queue = list(outputs)
        def runner(command, **kwargs):
            self.calls.append((command, kwargs))
            value = queue.pop(0)
            if isinstance(value, Exception):
                raise value
            return SimpleNamespace(returncode=0, stdout=value if isinstance(value, str) else json.dumps(value))
        return probe_hardware(device, "local-fixture-key-1234", expected_vendor="amd",
                              tools={"amd": self.tool}, runner=runner, hip_device=hip)

    def test_static_is_hmac_only_and_uses_fixed_read_only_arguments(self):
        result = self.run_probe([self.static()])
        self.assertEqual(result["model"], "AMD Instinct MI300X")
        self.assertEqual(set(result), {"vendor", "model", "fingerprint"})
        self.assertNotIn("fixture-private-serial", json.dumps(result))
        self.assertRegex(result["fingerprint"], "^[a-f0-9]{64}$")
        self.assertEqual(self.calls[0][0][1:], ["static", "--gpu", "3", "--asic", "--bus", "--json"])
        self.assertNotIn("PATH", self.calls[0][1]["env"])
        self.assertEqual(self.calls[0][1]["timeout"], 10)

    def test_hip_ordinal_maps_to_different_smi_index(self):
        result = self.run_probe([
            [{"gpu": 3, "hip_id": 0, "bdf": "0000:41:00.0"}], self.static(),
        ], hip=True, device=0)
        self.assertIsNotNone(result)
        self.assertEqual(self.calls[0][0][1:], ["list", "-e", "--json"])
        self.assertEqual(self.calls[1][0][3], "3")

    def test_missing_ambiguous_or_bool_hip_mapping_is_rejected(self):
        cases = [
            [{"gpu": 3, "hip_id": "N/A", "bdf": "0000:41:00.0"}],
            [{"gpu": 3, "hip_id": False, "bdf": "0000:41:00.0"}],
            [{"gpu": 3, "hip_id": 0, "bdf": "0000:41:00.0"}] * 2,
            [{"gpu": 3, "hip_id": 0, "bdf": "not-a-bdf"}],
        ]
        for records in cases:
            with self.subTest(records=records):
                self.assertIsNone(self.run_probe([records], hip=True, device=0))

    def test_mapping_to_duplicate_smi_index_is_rejected(self):
        records = [{"gpu": 3, "hip_id": 0, "bdf": "0000:41:00.0"},
                   {"gpu": 3, "hip_id": 1, "bdf": "0000:41:00.0"}]
        self.assertIsNone(self.run_probe([records], hip=True, device=0))

    def test_changed_bdf_between_mapping_and_static_is_rejected(self):
        self.assertIsNone(self.run_probe([
            [{"gpu": 3, "hip_id": 0, "bdf": "0000:42:00.0"}], self.static(),
        ], hip=True, device=0))

    def test_wrong_selected_gpu_and_multiple_records_are_rejected(self):
        for value in [self.static(0), {"gpu_data": self.static()["gpu_data"] * 2}]:
            with self.subTest(value=value):
                self.assertIsNone(self.run_probe([value]))

    def test_invalid_vendor_or_identity_is_rejected(self):
        for field, value in [("vendor_id", "0x10de"), ("market_name", "N/A"),
                             ("asic_serial", "N/A"), ("asic_serial", "0x0")]:
            with self.subTest(field=field, value=value):
                data = self.static()
                data["gpu_data"][0]["asic"][field] = value
                self.assertIsNone(self.run_probe([data]))

    def test_duplicate_keys_nonfinite_and_diagnostics_are_rejected(self):
        for value in ['{"gpu_data": [], "gpu_data": []}', '[NaN]',
                      'warning\n' + json.dumps(self.static()), 'x' * 65537]:
            with self.subTest(value=value[:50]):
                self.assertIsNone(self.run_probe([value]))

    def test_driver_error_is_not_training_failure(self):
        self.assertIsNone(self.run_probe([OSError("fixture driver unavailable")]))

    def test_legacy_rocm_smi_is_not_used_for_unmapped_hip_ordinal(self):
        tool = Path(self.temp.name) / "rocm-smi"
        tool.write_bytes(b"fixture only")
        with patch("scripts.hardware_identity.subprocess.run") as runner:
            self.assertIsNone(probe_hardware(0, "local-fixture-key-1234", expected_vendor="amd",
                                            tools={"amd": tool}, hip_device=True, runner=runner))
            runner.assert_not_called()

    def test_amd_tool_allowlist_is_shared_with_worker_binding(self):
        self.assertTrue(supported_probe_name("amd", "amd-smi"))
        self.assertTrue(supported_probe_name("amd", "rocm-smi"))
        self.assertFalse(supported_probe_name("amd", "nvidia-smi"))
        config = SimpleNamespace(hardware_probe_root=Path(self.temp.name))
        binding = {"hardware_probes": {"amd": {"ref": "opaque"}}}
        with patch("scripts.neural_forge_worker.bound_asset", return_value=self.tool) as resolve:
            self.assertEqual(hardware_probe_tools(binding, config), {"amd": self.tool})
            resolve.assert_called_once()

    def test_opted_in_hip_worker_requests_hip_mapping(self):
        config = SimpleNamespace(gpu_fingerprint_key="local-fixture-key-1234",
                                 leaderboard_attestation_key="local-signing-key-1234")
        job = {"run_id": "opaque-run", "leaderboard_requested": True}
        resolved = {"execution": {"backend": "hip"}, "device": 0,
                    "hardware_probe_tools": {"amd": self.tool}}
        with patch("scripts.neural_forge_worker.manifest_local_request",
                   return_value=({}, Path("config"), Path("dataset"), resolved)), \
             patch("scripts.neural_forge_worker.probe_hardware", return_value={
                 "vendor": "amd", "model": "AMD Instinct MI300X", "fingerprint": "a" * 64,
             }) as probe:
            result = optional_leaderboard_attestation(job, config)
        self.assertRegex(result["signature"], "^[a-f0-9]{64}$")
        probe.assert_called_once_with(0, "local-fixture-key-1234", expected_vendor="amd",
                                      tools={"amd": self.tool}, hip_device=True)

    def test_hip_probe_remains_opt_in(self):
        config = SimpleNamespace(gpu_fingerprint_key="local-fixture-key-1234",
                                 leaderboard_attestation_key="local-signing-key-1234")
        with patch("scripts.neural_forge_worker.manifest_local_request") as manifest:
            self.assertIsNone(optional_leaderboard_attestation({}, config))
            manifest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
