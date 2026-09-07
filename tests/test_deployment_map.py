import json
from pathlib import Path
import tempfile
import unittest

from scripts.deployment_map import (
    DEPLOYMENT_MAP_SCHEMA_VERSION,
    DeploymentMapError,
    load_deployment_map,
    validate_execution_attestation,
)


class DeploymentMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def valid_payload(self) -> dict[str, object]:
        return {
            "schema_version": DEPLOYMENT_MAP_SCHEMA_VERSION,
            "hub_profiles": {
                profile_id: {
                    "local_profile_id": f"public-{profile_id}",
                    "trainer_versions": ["trainer-ref-v3"],
                    "execution": {
                        "backend": "cuda",
                        "artifact_id": "artifact-ref-v3",
                        "binary_names": ["ida_native_train", "canopy_foundry_train"],
                    },
                }
                for profile_id in ("edge-full", "ai", "edge-swift", "moe")
            },
            "resource_classes": {"gpu-standard": {"device": 0}},
        }

    def write(self, payload: dict[str, object]) -> Path:
        path = self.root / "deployment-map.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_or_invalid_map_fails_closed(self) -> None:
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.root / "missing.json")
        invalid = self.root / "invalid.json"
        invalid.write_text("not-json", encoding="utf-8")
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(invalid)

    def test_duplicate_trainer_versions_and_unknown_backend_fail_closed(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["trainer_versions"] = [
            "trainer-ref-v3", "trainer-ref-v3"
        ]
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["execution"]["backend"] = "python"
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

    def test_four_profiles_are_explicit_and_distinct(self) -> None:
        deployment = load_deployment_map(self.write(self.valid_payload()))
        profiles = deployment["hub_profiles"]
        self.assertEqual(set(profiles), {"edge-full", "ai", "edge-swift", "moe"})
        self.assertEqual(
            len({entry["local_profile_id"] for entry in profiles.values()}),
            4,
        )

    def test_duplicate_local_profile_mapping_fails_closed(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["ai"]["local_profile_id"] = "public-edge-full"
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

    def test_reserved_and_prototype_ids_fail_closed(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["__proto__"] = payload["hub_profiles"]["edge-full"]
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["local_profile_id"] = "constructor"
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

    def test_paths_urls_and_unknown_mapping_fields_fail_closed(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["execution"]["path"] = "C:/private"
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["local_profile_id"] = "https://private.example"
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

    def test_invalid_resource_device_fails_closed(self) -> None:
        for device in (-1, 256, True, "0"):
            payload = self.valid_payload()
            payload["resource_classes"]["gpu-standard"]["device"] = device
            with self.subTest(device=device), self.assertRaises(DeploymentMapError):
                load_deployment_map(self.write(payload))

    def test_execution_attestation_must_match_mapped_profile(self) -> None:
        deployment = load_deployment_map(self.write(self.valid_payload()))
        profile = deployment["hub_profiles"]["edge-full"]
        valid = {
            "backend": "cuda",
            "artifact_id": "artifact-ref-v3",
            "binary_name": "ida_native_train",
            "binary_sha256": "a" * 64,
        }
        self.assertEqual(validate_execution_attestation(valid, "edge-full", profile), valid)
        for field, value in (
            ("artifact_id", "canopy-foundry-cuda-v1"),
            ("binary_name", "other-trainer"),
            ("binary_sha256", "not-a-hash"),
        ):
            candidate = dict(valid)
            candidate[field] = value
            with self.subTest(field=field), self.assertRaises(DeploymentMapError):
                validate_execution_attestation(candidate, "edge-full", profile)

    def test_v3_native_contract_is_explicit_and_binary_digest_is_pinned(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["native_execution"] = {
            "schema_version": "ida-native-execution-request.v1",
            "profile_id": "edge-full",
            "backend": "cuda",
            "precision_profile": "precision-ref-v3",
            "optimizer_type": "optimizer-ref-v3",
            "attention_backend": "attention-ref-v3",
        }
        with self.assertRaisesRegex(DeploymentMapError, "binary_sha256 is required"):
            load_deployment_map(self.write(payload))
        payload["hub_profiles"]["edge-full"]["execution"]["binary_sha256"] = "a" * 64
        deployment = load_deployment_map(self.write(payload))
        profile = deployment["hub_profiles"]["edge-full"]
        self.assertEqual(profile["native_execution"]["profile_id"], "edge-full")
        valid = {
            "backend": "cuda",
            "artifact_id": "artifact-ref-v3",
            "binary_name": "ida_native_train",
            "binary_sha256": "a" * 64,
        }
        self.assertEqual(validate_execution_attestation(valid, "edge-full", profile), valid)
        candidate = dict(valid)
        candidate["binary_sha256"] = "b" * 64
        with self.assertRaises(DeploymentMapError):
            validate_execution_attestation(candidate, "edge-full", profile)

    def test_v3_native_contract_is_external_and_rejects_local_fields(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["native_execution"] = {
            "schema_version": "ida-native-execution-request.v1",
            "profile_id": "edge-full",
            "backend": "cuda",
            "precision_profile": "precision-ref-v3",
            "optimizer_type": "optimizer-ref-v3",
            "attention_backend": "attention-ref-v3",
        }
        payload["hub_profiles"]["edge-full"]["execution"]["binary_sha256"] = "a" * 64
        deployment = load_deployment_map(self.write(payload))
        self.assertEqual(
            deployment["hub_profiles"]["edge-full"]["native_execution"]["optimizer_type"],
            "optimizer-ref-v3",
        )

        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["native_execution"] = {
            "schema_version": "ida-native-execution-request.v1",
            "profile_id": "edge-full",
            "backend": "cuda",
            "precision_profile": "precision-ref-v3",
            "optimizer_type": "optimizer-ref-v3",
            "attention_backend": "attention-ref-v3",
            "output_dir": "private",
        }
        with self.assertRaises(DeploymentMapError):
            load_deployment_map(self.write(payload))

    def test_model_contract_and_private_runtime_bindings_are_opaque_and_exact(self) -> None:
        payload = self.valid_payload()
        payload["hub_profiles"]["edge-full"]["model_contract_ids"] = [
            "hf_gpt2_native_v1", "ida_lattice_native_v1"
        ]
        payload["hub_profiles"]["edge-full"]["private_runtime"] = {
            "package_id": "nf-private-kernel-v1",
            "package_sha256": "b" * 64,
            "tier": "private_advanced",
        }
        deployment = load_deployment_map(self.write(payload))
        profile = deployment["hub_profiles"]["edge-full"]
        from scripts.deployment_map import validate_model_contract_binding, validate_private_runtime_binding

        self.assertEqual(validate_model_contract_binding("hf_gpt2_native_v1", "edge-full", profile), "hf_gpt2_native_v1")
        self.assertEqual(
            validate_private_runtime_binding(profile["private_runtime"], "edge-full", profile),
            profile["private_runtime"],
        )
        with self.assertRaises(DeploymentMapError):
            validate_model_contract_binding("generic_moe_native_v1", "edge-full", profile)
        with self.assertRaises(DeploymentMapError):
            validate_private_runtime_binding({
                "package_id": "nf-other-package",
                "package_sha256": "c" * 64,
                "tier": "private_advanced",
            }, "edge-full", profile)

    def test_private_runtime_paths_urls_and_unknown_fields_fail_closed(self) -> None:
        for candidate in (
            {"package_id": "nf-private", "package_sha256": "a" * 64, "tier": "private_advanced", "url": "https://private"},
            {"package_id": "nf-private", "package_sha256": "a" * 64, "tier": "private_advanced", "path": "C:/math"},
            {"package_id": "nf-private", "package_sha256": "a" * 64, "tier": "private_advanced", "source": "math.cu"},
        ):
            payload = self.valid_payload()
            payload["hub_profiles"]["edge-full"]["private_runtime"] = candidate
            with self.subTest(candidate=candidate), self.assertRaises(DeploymentMapError):
                load_deployment_map(self.write(payload))


if __name__ == "__main__":
    unittest.main()
