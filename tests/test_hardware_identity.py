import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.hardware_identity import probe_hardware


class HardwareIdentityTests(unittest.TestCase):
    def test_nvidia_probe_returns_hmac_only(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "nvidia-smi.exe"
            tool.write_bytes(b"trusted probe")

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(returncode=0, stdout="NVIDIA RTX 5070, GPU-raw-uuid\n")

            result = probe_hardware(0, "local-fingerprint-secret", tools={"nvidia": tool}, runner=runner)
        self.assertEqual(result["vendor"], "nvidia")
        self.assertEqual(result["model"], "NVIDIA RTX 5070")
        self.assertNotIn("GPU-raw-uuid", json.dumps(result))
        self.assertEqual(len(result["fingerprint"]), 64)
        self.assertEqual(Path(calls[0][0][0]).name, "nvidia-smi.exe")
        self.assertEqual(calls[0][0][1:], [
            "--id=0", "--query-gpu=name,uuid", "--format=csv,noheader,nounits"
        ])
        self.assertNotIn("PATH", calls[0][1]["env"])
        self.assertNotIn("local-fingerprint-secret", json.dumps(calls[0][1]["env"]))

    def test_amd_probe_is_used_when_nvidia_is_unavailable(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            nvidia_tool = Path(directory) / "nvidia-smi.exe"
            amd_tool = Path(directory) / "rocm-smi.exe"
            nvidia_tool.write_bytes(b"trusted nvidia probe")
            amd_tool.write_bytes(b"trusted amd probe")

            def runner(command, **kwargs):
                calls.append(command)
                if Path(command[0]).stem.lower() == "nvidia-smi":
                    return SimpleNamespace(returncode=1, stdout="")
                return SimpleNamespace(
                    returncode=0,
                    stdout="device,Card series,Unique ID\n0,AMD Radeon RX 7900 XTX,0xraw-serial\n",
                )

            result = probe_hardware(
                2,
                "local-fingerprint-secret",
                tools={"nvidia": nvidia_tool, "amd": amd_tool},
                runner=runner,
            )
        self.assertEqual(result["vendor"], "amd")
        self.assertEqual(result["model"], "AMD Radeon RX 7900 XTX")
        self.assertNotIn("0xraw-serial", json.dumps(result))
        self.assertEqual(Path(calls[1][0]).name, "rocm-smi.exe")
        self.assertEqual(calls[1][1:], [
            "--showproductname", "--showuniqueid", "--csv", "--device", "2"
        ])

    def test_missing_key_or_malformed_identity_is_ineligible(self) -> None:
        self.assertIsNone(probe_hardware(0, "too-short", tools={}, runner=lambda *args, **kwargs: None))
        self.assertIsNone(probe_hardware(
            0,
            "local-fingerprint-secret",
            tools={},
            runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="NVIDIA RTX 5070, [Not Supported]\n"),
        ))
        self.assertIsNone(probe_hardware(256, "local-fingerprint-secret"))

    def test_probe_requires_an_absolute_expected_tool(self) -> None:
        self.assertIsNone(probe_hardware(0, "local-fingerprint-secret", tools={"nvidia": "nvidia-smi"}))

    def test_expected_vendor_prevents_cross_backend_probe(self) -> None:
        self.assertIsNone(probe_hardware(
            0,
            "local-fingerprint-secret",
            expected_vendor="amd",
            tools={"nvidia": "nvidia-smi"},
        ))
        self.assertIsNone(probe_hardware(0, "local-fingerprint-secret", expected_vendor="cuda"))


if __name__ == "__main__":
    unittest.main()
