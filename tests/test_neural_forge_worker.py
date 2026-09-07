from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.neural_forge_worker import (
    HubClient,
    WorkerConfig,
    WorkerError,
    bounded_event,
    build_command,
    completed_metrics,
    execute_job,
    local_config,
    manifest_local_request,
    public_event,
    relative_ref,
    run_output,
    sanitized_environment,
    output_usage,
    optional_leaderboard_attestation,
    publish_terminal_update,
    wait_for_hub_evaluation,
)
from scripts.compatibility import (
    BACKEND_POLICIES,
    MANIFEST_SCHEMA_VERSION,
    validate_v3_native_execution_request,
)
from scripts.deployment_map import DEPLOYMENT_MAP_SCHEMA_VERSION, DeploymentMapError, load_deployment_map
from scripts.local_binding import BindingError, file_sha256, load_local_binding, path_sha256


class NeuralForgeWorkerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config_path = self.root / "configs" / "examples" / "small.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        source_config = Path(__file__).resolve().parents[1] / "configs" / "examples" / "small.json"
        config_path.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
        dataset = self.root / "datasets" / "demo"
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "tokens.u32").write_bytes(b"tokens")
        (dataset / "labels.i32").write_bytes(b"labels")
        self.binary_path = self.root / "ida_native_train"
        self.binary_path.write_bytes(b"test-binary")
        self.catalog_path = Path(__file__).resolve().parents[1] / "configs" / "public" / "capabilities.json"
        self.deployment_map_path = self.root / "deployment-map.json"
        self.deployment_map_path.write_text(json.dumps({
            "schema_version": DEPLOYMENT_MAP_SCHEMA_VERSION,
            "hub_profiles": {
                profile_id: {
                    "local_profile_id": profile_id,
                        "trainer_versions": ["trainer-ref-v3"],
                    "execution": {
                        "backend": backend,
                        "artifact_id": BACKEND_POLICIES[backend].artifact_id,
                        "binary_names": [BACKEND_POLICIES[backend].worker_binary_name],
                    },
                }
                for profile_id, backend in (
                    ("cuda-local", "cuda"),
                    ("opencl-smoke", "opencl"),
                    ("cpu-smoke", "cpu"),
                )
            },
            "resource_classes": {"gpu-standard": {"device": 0}},
        }), encoding="utf-8")
        self.binding_path = self.root / "binding.json"
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": catalog["catalog_version"],
            "catalog_sha256": file_sha256(self.catalog_path),
            "profiles": {
                "cuda-local": {
                    "ref": "examples/small.json",
                    "sha256": file_sha256(config_path),
                    "precision": "legacy_bf16",
                    "optimizer": "lion",
                },
            },
            "datasets": {
                "dataset-001": {
                    "ref": "demo",
                    "sha256": path_sha256(dataset),
                },
            },
            "models": {}, "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        self.config = WorkerConfig(
            hub_url="https://hidden-canopy-hub-api.azurewebsites.net/",
            worker_token="worker-secret",
            org="org.example",
            config_root=self.root / "configs",
            dataset_root=self.root / "datasets",
            artifact_root=self.root / "artifacts",
            run_root=self.root / "runs",
            binary=self.binary_path,
            binding_path=self.binding_path,
            catalog_path=self.catalog_path,
            deployment_map_path=self.deployment_map_path,
            worker_id="worker",
        )

    def execution(self, backend: str = "cuda", binary_path: Path | None = None) -> dict[str, str]:
        policy = BACKEND_POLICIES[backend]
        import hashlib
        path = binary_path or self.binary_path
        return {
            "backend": backend,
            "artifact_id": policy.artifact_id,
            "binary_name": policy.worker_binary_name,
            "binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def manifest_window(self) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        return {
            "issued_at": (now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
            "claimed_at": now.isoformat().replace("+00:00", "Z"),
            "deadline_at": (now + timedelta(seconds=90)).isoformat().replace("+00:00", "Z"),
        }

    def manifest_job(
        self,
        *,
        run_id: str = "nf-12345678",
        profile_id: str = "cuda-local",
        dataset_id: str = "dataset-001",
        resource_class: str = "gpu-standard",
        backend: str = "cuda",
        binary_path: Path | None = None,
        trainer_version: str = "trainer-ref-v3",
        base_model_id: str | None = None,
        checkpoint_id: str | None = None,
        request_overrides: dict[str, object] | None = None,
        policy_overrides: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "training_profile_id": profile_id,
            "dataset_id": dataset_id,
            "resource_class": resource_class,
        }
        if base_model_id is not None:
            request["base_model_id"] = base_model_id
            request["training_mode"] = "fine_tune"
        elif checkpoint_id is not None:
            request["checkpoint_id"] = checkpoint_id
            request["training_mode"] = "resume"
        else:
            request["training_mode"] = "from_scratch"
        if request_overrides:
            request.update(request_overrides)
        policy: dict[str, object] = {
            "policy_version": "policy-1",
            "training_profile_id": profile_id,
            "trainer_version": trainer_version,
            "resource_class": resource_class,
            "quota_id": "quota-gpu-standard",
            "max_steps": 1,
            "timeout_seconds": 60,
            "max_concurrency": 1,
            "required_evaluation_gates": [],
        }
        if policy_overrides:
            policy.update(policy_overrides)
        lineage: dict[str, object] = {"dataset": {"id": dataset_id}}
        if base_model_id is not None:
            lineage["source"] = {"kind": "base_model", "id": base_model_id}
        elif checkpoint_id is not None:
            lineage["source"] = {"kind": "checkpoint", "id": checkpoint_id}
        lineage["reproducibility"] = {
            "policy_version": policy["policy_version"],
            "training_profile_id": profile_id,
            "trainer_version": policy["trainer_version"],
            "training_mode": request["training_mode"],
            "dataset_id": dataset_id,
            "resource_class": resource_class,
            **({"base_model_id": base_model_id} if base_model_id is not None else {}),
            **({"checkpoint_id": checkpoint_id} if checkpoint_id is not None else {}),
        }
        return {
            "run_id": run_id,
            "request": dict(request),
            "worker_manifest": {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                **self.manifest_window(),
                "org": "org.example",
                "worker_subject": "worker",
                "authority": {
                    "subject": "user-001",
                    "role": "operator",
                    "capability_scopes": ["neural_forge.run.submit"],
                    "justification_hash": "a" * 64,
                },
                "execution": self.execution(backend, binary_path),
                "request": request,
                "policy": policy,
                "lineage": lineage,
            },
        }

    def v3_native_job(self) -> dict[str, object]:
        """Build a Hub-shaped job with an explicit private V3 deployment map."""

        config_path = self.config.config_root / "examples" / "small.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["attention_backend"] = "attention-ref-v3"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["profiles"]["cuda-local"].update({
            "sha256": file_sha256(config_path),
            "precision": "precision-ref-v3",
            "optimizer": "optimizer-ref-v3",
        })
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")

        binary_sha256 = file_sha256(self.binary_path)
        self.deployment_map_path.write_text(json.dumps({
            "schema_version": DEPLOYMENT_MAP_SCHEMA_VERSION,
            "hub_profiles": {
                "edge-full": {
                    "local_profile_id": "cuda-local",
                    "trainer_versions": ["trainer-ref-v3"],
                    "execution": {
                        "backend": "cuda",
                        "artifact_id": "artifact-ref-v3",
                        "binary_names": ["ida_native_train"],
                        "binary_sha256": binary_sha256,
                    },
                    "native_execution": {
                        "schema_version": "ida-native-execution-request.v1",
                        "profile_id": "edge-full",
                        "backend": "cuda",
                        "precision_profile": "precision-ref-v3",
                        "optimizer_type": "optimizer-ref-v3",
                        "attention_backend": "attention-ref-v3",
                    },
                },
            },
            "resource_classes": {"gpu-standard": {"device": 0}},
        }), encoding="utf-8")

        job = self.manifest_job(profile_id="edge-full", trainer_version="trainer-ref-v3")
        job["worker_manifest"]["execution"] = {
            "backend": "cuda",
            "artifact_id": "artifact-ref-v3",
            "binary_name": "ida_native_train",
            "binary_sha256": binary_sha256,
        }
        return job

    def test_v3_native_contract_reaches_foundry_request_without_public_translation(self) -> None:
        job = self.v3_native_job()
        local_request, _, _, resolved = manifest_local_request(job, self.config)
        self.assertEqual(resolved["native_execution"]["schema_version"], "ida-native-execution-request.v1")
        self.assertEqual(resolved["native_execution"]["optimizer_type"], "optimizer-ref-v3")
        self.assertEqual(resolved["native_execution"]["attention_backend"], "attention-ref-v3")
        command, output_path = build_command(job, self.config)
        self.assertTrue(command)
        descriptor = json.loads((output_path / "native-execution-request.json").read_text(encoding="utf-8"))
        self.assertEqual(descriptor["profile_id"], "edge-full")
        self.assertEqual(descriptor["optimizer_type"], "optimizer-ref-v3")
        self.assertNotIn("authority", descriptor)
        native_request = json.loads((output_path / "native-request.json").read_text(encoding="utf-8"))
        self.assertEqual(native_request["optimizer_type"], "optimizer-ref-v3")
        self.assertEqual(native_request["attention_backend"], "attention-ref-v3")
        for field in ("repo_root", "status_file", "job_id", "expected_terminal_phase"):
            self.assertNotIn(field, native_request)

    def test_v3_native_contract_rejects_lion_or_scalar_translation(self) -> None:
        job = self.v3_native_job()
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["profiles"]["cuda-local"]["optimizer"] = "lion"
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        with self.assertRaisesRegex(WorkerError, "native contract"):
            manifest_local_request(job, self.config)

    def test_v3_native_descriptor_rejects_public_artifact_identity(self) -> None:
        job = self.v3_native_job()
        deployment = load_deployment_map(self.deployment_map_path)
        contract = deployment["hub_profiles"]["edge-full"]["native_execution"]
        execution = job["worker_manifest"]["execution"]
        descriptor = {
            "schema_version": "ida-native-execution-request.v1",
            "run_id": "nf-12345678",
            "profile_id": "edge-full",
            "backend": "cuda",
            "artifact_id": "public-artifact-ref",
            "binary_name": "canopy_foundry_train",
            "binary_sha256": "a" * 64,
            "trainer_version": "trainer-ref-v3",
            "policy_version": "policy-1",
            "precision_profile": "precision-ref-v3",
            "optimizer_type": "optimizer-ref-v3",
            "attention_backend": "attention-ref-v3",
            "training_mode": "from_scratch",
            "dataset_id": "dataset-001",
            "resource_class": "gpu-standard",
            "max_steps": 1,
            "timeout_seconds": 60,
        }
        with self.assertRaisesRegex(ValueError, "artifact_id"):
            validate_v3_native_execution_request(
                descriptor,
                expected_contract=contract,
                expected_execution=execution,
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def set_profile_architecture_contract(self, contract_id: str) -> None:
        config_path = self.config.config_root / "examples" / "small.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload.setdefault("model", {})["architecture_contract"] = contract_id
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["profiles"]["cuda-local"]["sha256"] = file_sha256(config_path)
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")

    def test_local_refs_reject_absolute_and_traversal_paths(self) -> None:
        with self.assertRaises(WorkerError):
            relative_ref(self.root, "../../secrets", "dataset_ref")
        with self.assertRaises(WorkerError):
            relative_ref(self.root, str(self.root / "secret"), "dataset_ref")
        with self.assertRaises(WorkerError):
            relative_ref(self.root, "datasets\\demo", "dataset_ref")

    def test_bound_asset_hashing_rejects_symlink_trees(self) -> None:
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        link = self.root / "linked"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links are unavailable in this environment")
        with self.assertRaises(BindingError):
            path_sha256(link)
        tree = self.root / "tree"
        tree.mkdir()
        os.symlink(outside, tree / "outside")
        with self.assertRaises(BindingError):
            path_sha256(tree)

    def test_native_output_quota_is_bounded_and_link_safe(self) -> None:
        output = self.root / "output"
        output.mkdir()
        (output / "metrics.json").write_bytes(b"1234")
        self.assertEqual(output_usage(output), (4, 1))
        with self.assertRaises(WorkerError):
            WorkerConfig(
                hub_url=self.config.hub_url, worker_token=self.config.worker_token,
                org=self.config.org, config_root=self.config.config_root,
                dataset_root=self.config.dataset_root, artifact_root=self.config.artifact_root,
                run_root=self.config.run_root, max_output_bytes=0,
            )

    def test_run_id_is_bounded_to_local_run_root(self) -> None:
        self.assertEqual(run_output(self.config.run_root, "nf-12345678"), (self.config.run_root / "nf-12345678").resolve())
        with self.assertRaises(WorkerError):
            run_output(self.config.run_root, "../../outside")

    def test_worker_command_contains_local_paths_only(self) -> None:
        command, output = build_command(self.manifest_job(), self.config)

        self.assertIn("--request-json", command)
        request_path = Path(command[command.index("--request-json") + 1])
        native_request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(native_request["backend"], "native")
        self.assertEqual(native_request["precision_profile"], "legacy_bf16")
        self.assertEqual(native_request["optimizer_type"], "lion")
        self.assertEqual(native_request["input"]["token_blocks"], str((self.config.dataset_root / "demo" / "tokens.u32").resolve()))
        self.assertEqual(native_request["input"]["label_blocks"], str((self.config.dataset_root / "demo" / "labels.i32").resolve()))
        self.assertIn("--device", command)
        self.assertEqual(command[command.index("--device") + 1], "0")
        self.assertEqual(output, (self.config.run_root / "nf-12345678").resolve())
        self.assertNotIn("train.py", command)
        self.assertIn("output_dir", native_request)
        self.assertEqual(native_request["status_file"], str(output / "status.json"))
        self.assertEqual(native_request["repo_root"], str(output))
        self.assertEqual(native_request["job_id"], "nf-12345678")
        self.assertEqual(native_request["expected_terminal_phase"], "native_smoke_complete")

        for field in ("settings", "gpu", "backend", "config_ref", "dataset_ref"):
            with self.assertRaises(WorkerError):
                build_command(self.manifest_job(
                    run_id=f"nf-invalid-{field.replace('_', '')}1234",
                    request_overrides={field: "adam" if field == "settings" else "C:/private"},
                ), self.config)

    def test_legacy_direct_request_jobs_are_rejected(self) -> None:
        with self.assertRaisesRegex(WorkerError, "worker manifest is required"):
            build_command({
                "run_id": "nf-legacy12345678",
                "execution": self.execution(),
                "request": {
                    "backend": "cuda",
                    "config_ref": "examples/small.json",
                    "dataset_ref": "demo",
                    "precision": "legacy_bf16",
                    "optimizer": "lion",
                },
            }, self.config)

    def test_user_model_config_is_forwarded_without_using_a_personal_profile(self) -> None:
        config_path = self.config.config_root / "examples" / "gpt2-local.json"
        config_path.write_text(json.dumps({
            "required_arch": "sm_90",
            "attention_backend": "scalar_flash",
            "precision_profile": "legacy_bf16",
            "model": {
                "architecture_contract": "hf_gpt2_native_v1",
                "hidden_size": 128,
                "intermediate_size": 512,
                "layers": 2,
                "heads": 4,
                "kv_heads": 4,
                "vocab_size": 4096,
                "normalization_type": "layernorm",
                "activation_type": "gelu_new",
                "position_embedding_type": "learned_absolute",
                "norm_eps": 1e-5,
                "max_position_embeddings": 2048,
                "qkv_bias": True,
                "projection_bias": True,
                "tied_embeddings": True,
            },
            "training": {"microbatch": 1, "grad_accumulation": 1, "max_steps": 1},
            "input": {"batch_size": 1, "sequence_length": 2048},
        }), encoding="utf-8")
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["profiles"]["cuda-local"]["ref"] = "examples/gpt2-local.json"
        binding["profiles"]["cuda-local"]["sha256"] = file_sha256(config_path)
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        command, _ = build_command(
            self.manifest_job(run_id="nf-gpt2local1234"),
            self.config,
        )
        request_path = Path(command[command.index("--request-json") + 1])
        native_request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(native_request["model"]["architecture_contract"], "hf_gpt2_native_v1")
        self.assertEqual(native_request["model"]["normalization_type"], "layernorm")
        self.assertTrue(native_request["model"]["qkv_bias"])
        self.assertTrue(native_request["model"]["projection_bias"])
        self.assertNotIn("edge-full", json.dumps(native_request))

    def test_portable_backends_select_runtime_and_device_arguments(self) -> None:
        for backend, config_name, binary_name in (
            ("opencl", "opencl.json", "ida_native_opencl_train"),
            ("cpu", "cpu.json", "ida_native_cpu_train"),
        ):
            portable_binary = self.root / binary_name
            portable_binary.write_bytes(b"test-portable-binary")
            portable_binding = self.root / f"{backend}-binding.json"
            portable_binding.write_text(json.dumps({
                "schema_version": "neural-foundry-local-binding.v2",
                "catalog_version": json.loads(self.catalog_path.read_text(encoding="utf-8"))["catalog_version"],
                "catalog_sha256": file_sha256(self.catalog_path),
                "profiles": {}, "datasets": {}, "models": {}, "checkpoints": {},
                "binaries": {backend: {"ref": portable_binary.name, "sha256": file_sha256(portable_binary)}},
            }), encoding="utf-8")
            portable_config = WorkerConfig(
                hub_url=self.config.hub_url,
                worker_token=self.config.worker_token,
                org=self.config.org,
                config_root=self.config.config_root,
                dataset_root=self.config.dataset_root,
                artifact_root=self.config.artifact_root,
                run_root=self.config.run_root,
                binary=portable_binary,
                binding_path=portable_binding,
                catalog_path=self.catalog_path,
            )
            config_path = self.root / "configs" / "examples" / config_name
            config_path.write_text(json.dumps({
                "required_arch": "gfx1036" if backend == "opencl" else "host",
                "attention_backend": "scalar_flash",
                "model": {
                    "hidden_size": 32,
                    "intermediate_size": 128,
                    "layers": 1,
                    "heads": 1,
                    "vocab_size": 256,
                },
                "training": {"microbatch": 1, "grad_accumulation": 1, "max_steps": 1},
                "input": {"batch_size": 1, "sequence_length": 2048},
            }), encoding="utf-8")
            with self.assertRaises(WorkerError):
                build_command(self.manifest_job(
                    run_id=f"nf-{backend}12345678",
                    profile_id=f"{backend}-smoke",
                    backend=backend,
                    binary_path=portable_binary,
                ), portable_config)

    def test_manifest_attestation_is_authoritative(self) -> None:
        job = self.manifest_job()
        job["execution"] = {**self.execution(), "binary_sha256": "00" * 32}
        command, _ = build_command(job, self.config)
        self.assertIn("--request-json", command)

    def test_manifest_requires_hub_authority_lineage_and_timestamp(self) -> None:
        for field in ("issued_at", "authority", "lineage"):
            job = json.loads(json.dumps(self.manifest_job()))
            job["worker_manifest"].pop(field)
            with self.subTest(field=field), self.assertRaises(WorkerError):
                build_command(job, self.config)

    def test_manifest_requires_canonical_authority_and_policy_controls(self) -> None:
        invalid_jobs = []
        role = json.loads(json.dumps(self.manifest_job()))
        role["worker_manifest"]["authority"]["role"] = "viewer"
        invalid_jobs.append(("authority role", role))
        scopes = json.loads(json.dumps(self.manifest_job()))
        scopes["worker_manifest"]["authority"]["capability_scopes"] = []
        invalid_jobs.append(("authority scopes", scopes))
        for field in ("quota_id", "max_concurrency", "required_evaluation_gates"):
            missing = json.loads(json.dumps(self.manifest_job()))
            missing["worker_manifest"]["policy"].pop(field)
            invalid_jobs.append((f"policy {field}", missing))
        for label, job in invalid_jobs:
            with self.subTest(label=label), self.assertRaises(WorkerError):
                build_command(job, self.config)

    def test_legacy_v3_same_name_manifest_is_rejected_explicitly(self) -> None:
        legacy = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": "nf-12345678",
            "org": "org.example",
            "role_tier": "operator",
            "transition_id": "transition-01",
            "request": {
                "profile_id": "edge-full",
                "dataset_id": "dataset-001",
            },
        }
        with self.assertRaisesRegex(WorkerError, "canonical Hub/Foundry envelope"):
            build_command({"run_id": legacy["run_id"], "worker_manifest": legacy}, self.config)

    def test_manifest_rejects_nested_private_fields_without_echoing_values(self) -> None:
        for field, value in (
            ("path", "C:/private/checkpoint"),
            ("telemetry", {"raw": "private"}),
            ("token", "bearer-secret"),
        ):
            job = json.loads(json.dumps(self.manifest_job()))
            job["worker_manifest"]["lineage"]["dataset"][field] = value
            with self.subTest(field=field), self.assertRaises(WorkerError) as raised:
                build_command(job, self.config)
            self.assertNotIn(str(value), str(raised.exception))

    def test_lineage_source_requires_an_opaque_kind_and_id(self) -> None:
        for source in (
            {"id": "model-001"},
            {"kind": "base/model", "id": "model-001"},
            {"kind": "base_model", "id": "C:/private"},
        ):
            job = json.loads(json.dumps(self.manifest_job()))
            job["worker_manifest"]["lineage"]["source"] = source
            with self.subTest(source=source), self.assertRaises(WorkerError):
                build_command(job, self.config)

    def test_missing_deployment_map_fails_closed(self) -> None:
        config = WorkerConfig(
            hub_url=self.config.hub_url,
            worker_token=self.config.worker_token,
            org=self.config.org,
            config_root=self.config.config_root,
            dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root,
            run_root=self.config.run_root,
            binary=self.binary_path,
            binding_path=self.binding_path,
            catalog_path=self.catalog_path,
            deployment_map_path=self.root / "missing-deployment-map.json",
            worker_id="worker",
        )
        with self.assertRaisesRegex(WorkerError, "deployment map"):
            build_command(self.manifest_job(), config)

    def test_hub_claim_manifest_resolves_opaque_ids_locally(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "public" / "capabilities.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        binding_path = self.root / "configs" / "local" / "neural-forge-binding.json"
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": catalog["catalog_version"],
            "catalog_sha256": file_sha256(catalog_path),
            "profiles": {
                "cuda-local": {
                    "ref": "examples/small.json",
                    "sha256": file_sha256(self.config.config_root / "examples" / "small.json"),
                    "precision": "legacy_bf16",
                    "optimizer": "lion",
                }
            },
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {},
            "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        config = WorkerConfig(
            hub_url=self.config.hub_url,
            worker_token=self.config.worker_token,
            org=self.config.org,
            config_root=self.config.config_root,
            dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root,
            run_root=self.config.run_root,
            binary=self.binary_path,
            binding_path=binding_path,
            catalog_path=catalog_path,
            deployment_map_path=self.deployment_map_path,
            worker_id="worker",
        )
        job = {
            "run_id": "nf-12345678",
            "request": {"training_profile_id": "cuda-local", "dataset_id": "dataset-001", "resource_class": "gpu-standard"},
            "worker_manifest": {
                "schema_version": "neural-forge-worker-manifest.v2",
                "run_id": "nf-12345678",
                **self.manifest_window(),
                "org": "org.example",
                "worker_subject": "worker",
                "authority": {
                    "subject": "user-001",
                    "role": "operator",
                    "capability_scopes": ["neural_forge.run.submit"],
                    "justification_hash": "a" * 64,
                },
                "execution": self.execution(),
                "request": {"training_profile_id": "cuda-local", "dataset_id": "dataset-001", "resource_class": "gpu-standard"},
                "policy": {
                    "policy_version": "policy-1",
                    "training_profile_id": "cuda-local",
                    "trainer_version": "trainer-ref-v3",
                    "resource_class": "gpu-standard",
                    "quota_id": "quota-gpu-standard",
                    "max_steps": 1,
                    "timeout_seconds": 60,
                    "max_concurrency": 1,
                    "required_evaluation_gates": [],
                },
                "lineage": {
                    "dataset": {"id": "dataset-001"},
                    "reproducibility": {
                        "policy_version": "policy-1", "training_profile_id": "cuda-local",
                        "trainer_version": "trainer-ref-v3", "training_mode": "from_scratch",
                        "dataset_id": "dataset-001", "resource_class": "gpu-standard",
                    },
                },
            },
        }
        command, _ = build_command(job, config)
        native_request = json.loads(Path(command[command.index("--request-json") + 1]).read_text(encoding="utf-8"))
        self.assertEqual(native_request["backend"], "native")
        self.assertEqual(native_request["precision_profile"], "legacy_bf16")
        self.assertEqual(native_request["optimizer_type"], "lion")
        self.assertEqual(native_request["device"]["runtime"], "cuda")

    def test_v3_public_profile_maps_to_distinct_local_profile_and_device(self) -> None:
        catalog_path = self.root / "public-model-capabilities.json"
        catalog = {
            "schema_version": "neural-foundry-capability-catalog.v1",
            "catalog_version": "catalog-public-models-v1",
            "profiles": {
                "public-edge-full": {
                    "channel": "stable",
                    "backend": "cuda",
                    "artifact_id": "canopy-foundry-cuda-v1",
                    "binary_names": ["ida_native_train"],
                    "supported_precisions": ["legacy_bf16"],
                    "supported_optimizers": ["lion"],
                    "supported_attention": ["scalar_flash"],
                },
            },
        }
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        binding_path = self.root / "public-model-binding.json"
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": catalog["catalog_version"],
            "catalog_sha256": file_sha256(catalog_path),
            "profiles": {
                "public-edge-full": {
                    "ref": "examples/small.json",
                    "sha256": file_sha256(self.config.config_root / "examples" / "small.json"),
                    "precision": "legacy_bf16",
                    "optimizer": "lion",
                },
            },
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {},
            "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        deployment_map_path = self.root / "public-model-deployment-map.json"
        deployment_map_path.write_text(json.dumps({
            "schema_version": DEPLOYMENT_MAP_SCHEMA_VERSION,
            "hub_profiles": {
                "edge-full": {
                    "local_profile_id": "public-edge-full",
                    "trainer_versions": ["trainer-ref-v3"],
                    "execution": {
                        "backend": "cuda",
                        "artifact_id": "artifact-ref-v3",
                        "binary_names": ["ida_native_train", "canopy_foundry_train"],
                    },
                },
            },
            "resource_classes": {"gpu-standard": {"device": 3}},
        }), encoding="utf-8")
        config = WorkerConfig(
            hub_url=self.config.hub_url,
            worker_token=self.config.worker_token,
            org=self.config.org,
            config_root=self.config.config_root,
            dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root,
            run_root=self.config.run_root,
            binary=self.binary_path,
            binding_path=binding_path,
            catalog_path=catalog_path,
            deployment_map_path=deployment_map_path,
            worker_id="worker",
        )
        job = self.manifest_job(profile_id="edge-full")
        job["worker_manifest"]["execution"] = {
            "backend": "cuda",
            "artifact_id": "artifact-ref-v3",
            "binary_name": "ida_native_train",
            "binary_sha256": file_sha256(self.binary_path),
        }
        command, _ = build_command(job, config)
        self.assertEqual(command[command.index("--device") + 1], "3")
        native_request = json.loads(Path(command[command.index("--request-json") + 1]).read_text(encoding="utf-8"))
        self.assertEqual(native_request["precision_profile"], "legacy_bf16")
        self.assertEqual(native_request["optimizer_type"], "lion")
        self.assertNotIn("edge-full", json.dumps(native_request))
        self.assertNotIn("public-edge-full", json.dumps(native_request))

        mismatched_trainer = self.manifest_job(profile_id="edge-full")
        mismatched_trainer["worker_manifest"]["execution"] = job["worker_manifest"]["execution"]
        mismatched_trainer["worker_manifest"]["policy"]["trainer_version"] = "ida-native-v2"
        mismatched_trainer["worker_manifest"]["lineage"]["reproducibility"]["trainer_version"] = "ida-native-v2"
        with self.assertRaisesRegex(WorkerError, "trainer version"):
            build_command(mismatched_trainer, config)

        missing_profile = self.manifest_job(profile_id="ai")
        missing_profile["worker_manifest"]["execution"] = job["worker_manifest"]["execution"]
        with self.assertRaisesRegex(WorkerError, "deployment mapping"):
            build_command(missing_profile, config)

        mismatched_hash = self.manifest_job(
            profile_id="edge-full", run_id="nf-hash12345678"
        )
        mismatched_hash["worker_manifest"]["execution"] = {
            **job["worker_manifest"]["execution"],
            "binary_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(WorkerError, "native binary attestation"):
            build_command(mismatched_hash, config)

    def test_user_owned_models_and_checkpoints_resolve_only_locally(self) -> None:
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        profile_path = self.config.config_root / "examples" / "small.json"
        model_path = self.config.artifact_root / "models" / "my-model"
        checkpoint_path = self.config.artifact_root / "checkpoints" / "my-checkpoint"
        model_path.mkdir(parents=True)
        checkpoint_path.mkdir(parents=True)
        (model_path / "weights.safetensors").write_bytes(b"model")
        (checkpoint_path / "model.safetensors").write_bytes(b"checkpoint")
        (checkpoint_path / "optimizer_state.safetensors").write_bytes(b"optimizer")
        binding_path = self.root / "user-assets-binding.json"
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": catalog["catalog_version"],
            "catalog_sha256": file_sha256(self.catalog_path),
            "profiles": {
                "cuda-local": {
                    "ref": "examples/small.json",
                    "sha256": file_sha256(profile_path),
                    "precision": "legacy_bf16",
                    "optimizer": "lion",
                }
            },
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {"model-my-lab": {"ref": "models/my-model", "sha256": path_sha256(model_path)}},
            "checkpoints": {"checkpoint-my-lab": {"ref": "checkpoints/my-checkpoint", "sha256": path_sha256(checkpoint_path)}},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        config = WorkerConfig(
            hub_url=self.config.hub_url, worker_token=self.config.worker_token,
            org=self.config.org, config_root=self.config.config_root,
            dataset_root=self.config.dataset_root, artifact_root=self.config.artifact_root,
            run_root=self.config.run_root, binary=self.binary_path,
            binding_path=binding_path, catalog_path=self.catalog_path,
            deployment_map_path=self.deployment_map_path, worker_id="worker",
        )
        base_manifest = {
            "schema_version": "neural-forge-worker-manifest.v2",
            "run_id": "nf-12345678", **self.manifest_window(),
            "org": "org.example", "worker_subject": "worker",
            "authority": {
                "subject": "user-001", "role": "operator",
                "capability_scopes": ["neural_forge.run.submit"],
                "justification_hash": "a" * 64,
            },
            "execution": self.execution(),
            "policy": {
                "policy_version": "policy-1", "training_profile_id": "cuda-local",
                "trainer_version": "trainer-ref-v3", "resource_class": "gpu-standard",
                "quota_id": "quota-gpu-standard",
                "max_steps": 1, "timeout_seconds": 60,
                "max_concurrency": 1, "required_evaluation_gates": [],
            },
            "lineage": {
                "dataset": {"id": "dataset-001"},
                "reproducibility": {
                    "policy_version": "policy-1", "training_profile_id": "cuda-local",
                    "trainer_version": "trainer-ref-v3", "training_mode": "from_scratch",
                    "dataset_id": "dataset-001", "resource_class": "gpu-standard",
                },
            },
        }
        model_job = {
            "run_id": "nf-12345678",
            "worker_manifest": {
                **base_manifest,
                "request": {
                    "training_profile_id": "cuda-local", "dataset_id": "dataset-001",
                    "resource_class": "gpu-standard", "base_model_id": "model-my-lab",
                    "training_mode": "fine_tune",
                },
                "lineage": {
                    **base_manifest["lineage"],
                    "source": {"kind": "base_model", "id": "model-my-lab"},
                    "reproducibility": {
                        **base_manifest["lineage"]["reproducibility"],
                        "training_mode": "fine_tune", "base_model_id": "model-my-lab",
                    },
                },
            },
        }
        local_request, _, _, _ = manifest_local_request(model_job, config)
        self.assertEqual(local_request["init_from_model_ref"], "models/my-model")
        self.assertEqual(local_request["training_mode"], "fine_tune")
        self.assertNotIn(str(model_path.resolve()), json.dumps(local_request))
        command, _ = build_command(model_job, config)
        native_request = json.loads(Path(command[command.index("--request-json") + 1]).read_text(encoding="utf-8"))
        self.assertEqual(native_request["precision_profile"], "legacy_bf16")
        self.assertEqual(native_request["optimizer_type"], "lion")
        self.assertEqual(native_request["attention_backend"], "scalar_flash")
        self.assertEqual(native_request["training"]["microbatch"], 1)
        self.assertEqual(native_request["training"]["grad_accumulation"], 1)
        self.assertEqual(native_request["training"]["learning_rate"], 0.0003)
        self.assertEqual(native_request["training"]["max_steps"], 1)
        self.assertNotIn("seed", native_request)
        checkpoint_job = {
            "run_id": "nf-87654321",
            "worker_manifest": {
                **base_manifest,
                "run_id": "nf-87654321",
                "request": {
                    "training_profile_id": "cuda-local", "dataset_id": "dataset-001",
                    "resource_class": "gpu-standard", "checkpoint_id": "checkpoint-my-lab",
                    "training_mode": "resume",
                },
                "lineage": {
                    **base_manifest["lineage"],
                    "source": {"kind": "checkpoint", "id": "checkpoint-my-lab"},
                    "reproducibility": {
                        **base_manifest["lineage"]["reproducibility"],
                        "training_mode": "resume", "checkpoint_id": "checkpoint-my-lab",
                    },
                },
            },
        }
        local_request, _, _, _ = manifest_local_request(checkpoint_job, config)
        self.assertEqual(local_request["resume_from_ref"], "checkpoints/my-checkpoint")
        self.assertEqual(local_request["training_mode"], "resume")
        with self.assertRaises(WorkerError):
            manifest_local_request({
                **model_job,
                "worker_manifest": {
                    **model_job["worker_manifest"],
                    "request": {
                        **model_job["worker_manifest"]["request"],
                        "settings": {"config_ref": "../../private"},
                    },
                },
            }, config)

    def test_manifest_org_and_policy_limits_fail_closed(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "public" / "capabilities.json"
        binding_path = self.root / "binding.json"
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": "catalog-2026-08-23-public-blank",
            "catalog_sha256": file_sha256(catalog_path),
            "profiles": {"cuda-local": {"ref": "examples/small.json", "sha256": file_sha256(self.config.config_root / "examples" / "small.json"), "precision": "legacy_bf16", "optimizer": "lion"}},
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {},
            "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        config = WorkerConfig(
            hub_url=self.config.hub_url, worker_token=self.config.worker_token, org=self.config.org,
            config_root=self.config.config_root, dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root, run_root=self.config.run_root,
            binary=self.binary_path, binding_path=binding_path, catalog_path=catalog_path,
            deployment_map_path=self.deployment_map_path,
            worker_id="worker",
        )
        base = {
            "run_id": "nf-12345678", "request": {}, "worker_manifest": {
                "schema_version": "neural-forge-worker-manifest.v2", "run_id": "nf-12345678",
                **self.manifest_window(),
                "org": "org.example", "worker_subject": "worker", "execution": self.execution(),
                "authority": {
                    "subject": "user-001", "role": "operator",
                    "capability_scopes": ["neural_forge.run.submit"],
                    "justification_hash": "a" * 64,
                },
                "request": {"training_profile_id": "cuda-local", "dataset_id": "dataset-001", "resource_class": "gpu-standard"},
                "policy": {"policy_version": "policy-1", "training_profile_id": "cuda-local", "trainer_version": "trainer-ref-v3", "resource_class": "gpu-standard", "quota_id": "quota-gpu-standard", "max_steps": 1, "timeout_seconds": 60, "max_concurrency": 1, "required_evaluation_gates": []},
                "lineage": {},
            }
        }
        base["worker_manifest"]["org"] = "other-org"
        with self.assertRaises(WorkerError):
            build_command(base, config)
        base["worker_manifest"]["org"] = "org.example"
        base["worker_manifest"]["policy"]["max_steps"] = 0
        with self.assertRaises(WorkerError):
            build_command(base, config)

    def test_hub_claim_preserves_private_manifest(self) -> None:
        client = object.__new__(HubClient)
        client._request = lambda *_args, **_kwargs: {
            "ok": True,
            "job": {"run_id": "nf-12345678", "request": {"training_profile_id": "cuda-local"}},
            "worker_manifest": {"schema_version": "neural-forge-worker-manifest.v2", "org": "org.example"},
        }
        claimed = client.claim()
        self.assertEqual(claimed["worker_manifest"]["schema_version"], "neural-forge-worker-manifest.v2")

    def test_hub_status_preserves_evaluation_and_cancellation_state(self) -> None:
        client = object.__new__(HubClient)
        client._request = lambda *_args, **_kwargs: {
            "ok": True,
            "run": {
                "run_id": "nf-12345678",
                "status": "running",
                "evaluation": {"complete": False},
            },
            "cancel_requested": False,
        }
        state = client.get_worker_run("nf-12345678")
        self.assertEqual(state["run"]["status"], "running")
        self.assertFalse(state["run"]["evaluation"]["complete"])
        self.assertFalse(state["cancel_requested"])

        client._request = lambda *_args, **_kwargs: {
            "ok": True,
            "run": {
                "run_id": "nf-12345678",
                "status": "cancel_requested",
                "evaluation": {"complete": False},
            },
            "cancel_requested": True,
        }
        state = client.get_worker_run("nf-12345678")
        self.assertTrue(state["cancel_requested"])

    def test_worker_waits_for_hub_evaluation_before_terminal_success(self) -> None:
        from unittest.mock import patch

        job = self.manifest_job()

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def get_worker_run(self, _run_id: str) -> dict[str, object]:
                self.calls += 1
                return {
                    "run": {
                        "run_id": "nf-12345678",
                        "status": "running",
                        "evaluation": {"complete": self.calls > 1},
                    },
                    "cancel_requested": False,
                }

        client = FakeClient()
        with patch("scripts.neural_forge_worker.time.sleep") as sleep:
            result = wait_for_hub_evaluation(job, replace(self.config, poll_seconds=0.05), client)
        self.assertEqual(result, ("succeeded", False))
        self.assertEqual(client.calls, 2)
        sleep.assert_called_once()

    def test_worker_wait_for_evaluation_honors_controller_cancellation(self) -> None:
        job = self.manifest_job()

        class FakeClient:
            def get_worker_run(self, _run_id: str) -> dict[str, object]:
                return {
                    "run": {
                        "run_id": "nf-12345678",
                        "status": "cancel_requested",
                        "evaluation": {"complete": False},
                    },
                    "cancel_requested": True,
                }

        self.assertEqual(
            wait_for_hub_evaluation(job, self.config, FakeClient()),
            ("cancelled", False),
        )

    def test_worker_wait_for_evaluation_fails_closed_at_deadline(self) -> None:
        job = self.manifest_job()
        job["worker_manifest"]["deadline_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        ).isoformat().replace("+00:00", "Z")

        class UnavailableClient:
            def get_worker_run(self, _run_id: str) -> dict[str, object]:
                raise WorkerError("Hub unavailable")

        self.assertEqual(
            wait_for_hub_evaluation(job, self.config, UnavailableClient()),
            ("failed", False),
        )

    def test_terminal_update_commits_hub_before_local_receipt(self) -> None:
        calls: list[str] = []
        receipt_refs: list[dict[str, object]] = []

        class FakeClient:
            def update(self, *_args: object, **_kwargs: object) -> None:
                calls.append("hub")
                receipt = _kwargs.get("execution_receipt")
                if isinstance(receipt, dict):
                    receipt_refs.append(receipt)

        class FakeReceiptStore:
            def complete(self, *_args: object, **_kwargs: object) -> None:
                calls.append("local")

        publish_terminal_update(
            FakeClient(),  # type: ignore[arg-type]
            FakeReceiptStore(),  # type: ignore[arg-type]
            "nf-12345678",
            {"run_id": "nf-12345678"},
            status="succeeded",
            metrics_available=True,
            event={"type": "process_exit", "status": "succeeded"},
        )
        self.assertEqual(calls, ["hub", "local"])
        self.assertEqual(receipt_refs[0]["schema_version"], "neural-foundry-run-receipt-ref.v2")
        self.assertNotIn("path", json.dumps(receipt_refs[0]).lower())

    def test_terminal_update_reconciles_a_lost_hub_response(self) -> None:
        calls: list[str] = []

        class FakeClient:
            def __init__(self) -> None:
                self.receipt_sha256 = None

            def update(self, *_args: object, **_kwargs: object) -> None:
                calls.append("hub")
                receipt = _kwargs.get("execution_receipt")
                if isinstance(receipt, dict):
                    self.receipt_sha256 = receipt.get("receipt_sha256")
                raise WorkerError("response lost")

            def get_worker_run(self, _run_id: str) -> dict[str, object]:
                calls.append("status")
                return {
                    "run": {
                        "run_id": "nf-12345678",
                        "status": "succeeded",
                        "evaluation": {"complete": True},
                        "execution_receipt_sha256": self.receipt_sha256,
                    },
                    "cancel_requested": False,
                }

        class FakeReceiptStore:
            def complete(self, *_args: object, **_kwargs: object) -> None:
                calls.append("local")

        publish_terminal_update(
            FakeClient(),  # type: ignore[arg-type]
            FakeReceiptStore(),  # type: ignore[arg-type]
            "nf-12345678",
            {"run_id": "nf-12345678"},
            status="succeeded",
            metrics_available=True,
            event={"type": "process_exit", "status": "succeeded"},
        )
        self.assertEqual(calls, ["hub", "status", "local"])

    def test_execute_job_waits_for_evaluation_before_local_success_receipt(self) -> None:
        from unittest.mock import patch

        run_id = "nf-execute123456"
        output = self.config.run_root / run_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "native-request.json").write_text(
            json.dumps({
                "device": {"runtime": "cuda"},
                "expected_terminal_phase": "native_smoke_complete",
            }), encoding="utf-8"
        )
        (output / "native-execution-request.json").write_text(json.dumps({
            "schema_version": "ida-native-execution-request.v1",
            "run_id": run_id,
            "profile_id": "edge-full",
            "backend": "cuda",
            "artifact_id": "artifact-ref-v3",
            "binary_name": "ida_native_train",
            "binary_sha256": "a" * 64,
            "trainer_version": "trainer-ref-v3",
            "policy_version": "policy-1",
            "precision_profile": "precision-ref-v3",
            "optimizer_type": "optimizer-ref-v3",
            "attention_backend": "attention-ref-v3",
            "training_mode": "from_scratch",
            "dataset_id": "dataset-001",
            "resource_class": "gpu-standard",
            "max_steps": 1,
            "timeout_seconds": 60,
            "model_contract_id": "hf_gpt2_native_v1",
        }, indent=2), encoding="utf-8")
        job = {
            "run_id": run_id,
            "worker_manifest": {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "deadline_at": self.manifest_window()["deadline_at"],
                "policy": {"timeout_seconds": 60},
            },
        }
        child_code = (
            "import json,pathlib,sys; "
            "p=pathlib.Path(sys.argv[1]); "
            "(p/'model.safetensors').write_bytes(b'checkpoint'); "
            "(p/'metrics.json').write_text(json.dumps({"
            "'type':'complete','status':'complete','backend':'native',"
            "'step':1,'optimizer_steps':1,'loss':0.5,'learning_rate':0.0003,"
            "'tokens':2048,'tokens_per_second':100.0,'grad_norm':1.0,"
            "'checkpoint_written':True,'output':str(p.resolve())}))"
        )

        class FakeClient:
            def __init__(self) -> None:
                self.status_calls = 0
                self.updates: list[tuple[str, dict[str, object] | None]] = []
                self.execution_receipts: list[dict[str, object]] = []

            def get_worker_run(self, _run_id: str) -> dict[str, object]:
                self.status_calls += 1
                return {
                    "run": {
                        "run_id": run_id,
                        "status": "running",
                        "evaluation": {"complete": self.status_calls > 1},
                    },
                    "cancel_requested": False,
                }

            def update(self, _run_id: str, *, status: str, event: dict[str, object] | None = None,
                       **kwargs: object) -> None:
                self.updates.append((status, event))
                receipt = kwargs.get("execution_receipt")
                if isinstance(receipt, dict):
                    self.execution_receipts.append(receipt)

        client = FakeClient()
        v3_contract = {
            "schema_version": "ida-native-execution-request.v1",
            "profile_id": "edge-full",
            "backend": "cuda",
            "precision_profile": "precision-ref-v3",
            "optimizer_type": "optimizer-ref-v3",
            "attention_backend": "attention-ref-v3",
        }
        v3_execution = {
            "backend": "cuda",
            "artifact_id": "artifact-ref-v3",
            "binary_name": "ida_native_train",
            "binary_sha256": "a" * 64,
        }
        with patch(
            "scripts.neural_forge_worker.build_command",
            return_value=([sys.executable, "-c", child_code, str(output)], output),
        ), patch(
            "scripts.neural_forge_worker.manifest_local_request",
            return_value=({}, Path(), Path(), {
                "native_contract": v3_contract,
                "execution": v3_execution,
            }),
        ), patch("scripts.neural_forge_worker.time.sleep"):
            execute_job(job, replace(self.config, poll_seconds=0.05), client)  # type: ignore[arg-type]

        self.assertEqual(client.updates[0][0], "running")
        self.assertEqual(client.updates[-1][0], "succeeded")
        self.assertEqual([status for status, _ in client.updates].count("succeeded"), 1)
        self.assertGreaterEqual(len(client.execution_receipts), 2)
        self.assertEqual(
            {receipt["receipt_sha256"] for receipt in client.execution_receipts},
            {client.execution_receipts[0]["receipt_sha256"]},
        )
        receipt = json.loads((self.config.run_root / ".receipts" / f"{run_id}.json").read_text())
        self.assertEqual(receipt["status"], "succeeded")
        self.assertGreaterEqual(client.status_calls, 2)

    def test_execute_job_honors_controller_cancellation(self) -> None:
        from unittest.mock import patch

        run_id = "nf-cancel123456"
        output = self.config.run_root / run_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "native-request.json").write_text(
            json.dumps({
                "device": {"runtime": "cuda"},
                "expected_terminal_phase": "native_smoke_complete",
            }), encoding="utf-8"
        )
        job = {
            "run_id": run_id,
            "worker_manifest": {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "deadline_at": self.manifest_window()["deadline_at"],
                "policy": {"timeout_seconds": 60},
            },
        }

        class FakeClient:
            def __init__(self) -> None:
                self.updates: list[str] = []

            def get_worker_run(self, _run_id: str) -> dict[str, object]:
                return {
                    "run": {
                        "run_id": run_id,
                        "status": "cancel_requested",
                        "evaluation": {"complete": False},
                    },
                    "cancel_requested": True,
                }

            def update(self, _run_id: str, *, status: str, **_kwargs: object) -> None:
                self.updates.append(status)

        client = FakeClient()
        with patch(
            "scripts.neural_forge_worker.build_command",
            return_value=([sys.executable, "-c", "import time; time.sleep(60)"], output),
        ), patch(
            "scripts.neural_forge_worker.terminate_process_tree",
            side_effect=lambda process: process.kill(),
        ), patch(
            "scripts.neural_forge_worker.manifest_local_request",
            return_value=({}, Path(), Path(), {"native_contract": None}),
        ), patch("scripts.neural_forge_worker.time.sleep"):
            execute_job(job, replace(self.config, poll_seconds=0.05), client)  # type: ignore[arg-type]

        self.assertEqual(client.updates[-1], "cancelled")
        receipt = json.loads((self.config.run_root / ".receipts" / f"{run_id}.json").read_text())
        self.assertEqual(receipt["status"], "cancelled")

    def test_hub_update_carries_only_sanitized_leaderboard_attestation(self) -> None:
        client = object.__new__(HubClient)
        captured = {}
        client._request = lambda method, path, body: captured.update({"method": method, "path": path, "body": body}) or {"ok": True}
        client.update(
            "nf-12345678",
            status="succeeded",
            metrics_available=True,
            leaderboard_attestation={"vendor": "nvidia", "model": "NVIDIA RTX 5070", "fingerprint": "a" * 64},
            operation_id="nfu-test-operation",
        )
        self.assertEqual(captured["body"]["operation_id"], "nfu-test-operation")
        self.assertEqual(captured["body"]["leaderboard_attestation"]["model"], "NVIDIA RTX 5070")
        self.assertNotIn("uuid", captured["body"])
        self.assertNotIn("path", captured["body"])

    def test_leaderboard_probe_is_opt_in_and_uses_private_resource_device(self) -> None:
        from unittest.mock import patch
        from dataclasses import replace

        config = replace(
            self.config,
            gpu_fingerprint_key="local-fingerprint-secret",
            leaderboard_attestation_key="local-leaderboard-attestation-secret-32",
        )
        resolved = {"execution": {"backend": "cuda"}, "device": 2}
        job = {"run_id": "nf-12345678", "leaderboard_requested": True}
        with patch("scripts.neural_forge_worker.manifest_local_request", return_value=({}, Path("config"), Path("dataset"), resolved)) as manifest, \
             patch("scripts.neural_forge_worker.probe_hardware", return_value={"vendor": "nvidia", "model": "NVIDIA RTX 5070", "fingerprint": "b" * 64}) as probe:
            result = optional_leaderboard_attestation(job, config)
        self.assertEqual(result["fingerprint"], "b" * 64)
        self.assertRegex(result["signature"], r"^[a-f0-9]{64}$")
        manifest.assert_called_once_with(job, config)
        probe.assert_called_once_with(2, "local-fingerprint-secret", expected_vendor="nvidia", tools=None)
        self.assertIsNone(optional_leaderboard_attestation({"leaderboard_requested": False}, config))

    def test_leaderboard_secrets_must_not_be_reused(self) -> None:
        with self.assertRaises(WorkerError):
            replace(
                self.config,
                gpu_fingerprint_key="same-secret-value-1234567890123456",
                leaderboard_attestation_key="same-secret-value-1234567890123456",
            )

    def test_hub_url_requires_tls(self) -> None:
        for hub_url in (
            "http://hub.example.test/",
            "https://hub.example.test/?token=leak",
            "https://hub.example.test/#token-leak",
        ):
            with self.assertRaises(WorkerError):
                WorkerConfig(
                    hub_url=hub_url, worker_token="token", org="org.example",
                    config_root=self.config.config_root, dataset_root=self.config.dataset_root,
                    artifact_root=self.config.artifact_root, run_root=self.config.run_root,
                )

        with self.assertRaises(WorkerError):
            replace(self.config, hub_url="https://collector.invalid/")

        with self.assertRaises(WorkerError):
            replace(self.config, allowed_hub_hosts=("bad host",))

    def test_manifest_requires_evaluation_request_id_parity(self) -> None:
        valid = self.manifest_job()

        request_only = copy.deepcopy(valid)
        request_only["worker_manifest"]["request"]["evaluation_request_id"] = "eval-001"
        with self.assertRaises(WorkerError):
            manifest_local_request(request_only, self.config)

        policy_only = copy.deepcopy(valid)
        policy_only["worker_manifest"]["policy"]["evaluation_request_id"] = "eval-001"
        with self.assertRaises(WorkerError):
            manifest_local_request(policy_only, self.config)

        mismatched = copy.deepcopy(valid)
        mismatched["worker_manifest"]["request"]["evaluation_request_id"] = "eval-001"
        mismatched["worker_manifest"]["policy"]["evaluation_request_id"] = "eval-002"
        with self.assertRaises(WorkerError):
            manifest_local_request(mismatched, self.config)

    def test_worker_headers_reject_control_and_non_opaque_values(self) -> None:
        for field, value in (
            ("worker_token", "token\r\nX-Leak: value"),
            ("org", "org/other"),
            ("worker_id", "worker\nother"),
            ("function_key", "key\r\nX-Leak: value"),
        ):
            with self.subTest(field=field), self.assertRaises(WorkerError):
                replace(self.config, **{field: value})

    def test_manifest_requires_worker_identity_and_catalog_pin(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "public" / "capabilities.json"
        binding_path = self.root / "binding.json"
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": "catalog-2026-08-23-public-blank",
            "catalog_sha256": file_sha256(catalog_path),
            "profiles": {"cuda-local": {"ref": "examples/small.json", "sha256": file_sha256(self.config.config_root / "examples" / "small.json"), "precision": "legacy_bf16", "optimizer": "lion"}},
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {},
            "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        config = WorkerConfig(
            hub_url=self.config.hub_url, worker_token=self.config.worker_token, org=self.config.org,
            config_root=self.config.config_root, dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root, run_root=self.config.run_root,
            binary=self.binary_path, binding_path=binding_path, catalog_path=catalog_path,
        )
        job = {
            "run_id": "nf-12345678",
            "worker_manifest": {
                "schema_version": "neural-forge-worker-manifest.v2",
                "run_id": "nf-12345678",
                **self.manifest_window(),
                "org": "org.example",
                "execution": self.execution(),
                "authority": {
                "subject": "user-001", "role": "operator",
                "capability_scopes": ["neural_forge.run.submit"],
                    "justification_hash": "a" * 64,
                },
                "request": {"training_profile_id": "cuda-local", "dataset_id": "dataset-001", "resource_class": "gpu-standard"},
                "policy": {"policy_version": "policy-1", "training_profile_id": "cuda-local", "trainer_version": "trainer-ref-v3", "resource_class": "gpu-standard", "quota_id": "quota-gpu-standard", "max_steps": 1, "timeout_seconds": 60, "max_concurrency": 1, "required_evaluation_gates": []},
                "lineage": {
                    "dataset": {"id": "dataset-001"},
                    "reproducibility": {
                        "policy_version": "policy-1", "training_profile_id": "cuda-local",
                        "trainer_version": "trainer-ref-v3", "training_mode": "from_scratch",
                        "dataset_id": "dataset-001", "resource_class": "gpu-standard",
                    },
                },
            },
        }
        with self.assertRaises(WorkerError):
            build_command(job, config)
        job["worker_manifest"]["worker_subject"] = "worker"
        with self.assertRaises(WorkerError):
            build_command(job, config)
        config_with_id = WorkerConfig(
            hub_url=self.config.hub_url, worker_token=self.config.worker_token, org=self.config.org,
            config_root=self.config.config_root, dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root, run_root=self.config.run_root,
            binary=self.binary_path, binding_path=binding_path, catalog_path=catalog_path,
            deployment_map_path=self.deployment_map_path,
            worker_id="worker",
        )
        build_command(job, config_with_id)
        pinned = json.loads(binding_path.read_text(encoding="utf-8"))
        pinned["catalog_sha256"] = "0" * 64
        binding_path.write_text(json.dumps(pinned), encoding="utf-8")
        job["run_id"] = "nf-87654321"
        job["worker_manifest"]["run_id"] = "nf-87654321"
        with self.assertRaises(WorkerError):
            build_command(job, config_with_id)

    def test_manifest_policy_caps_local_config_and_rejects_naive_deadline(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "public" / "capabilities.json"
        binding_path = self.root / "binding.json"
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": "catalog-2026-08-23-public-blank",
            "catalog_sha256": file_sha256(catalog_path),
            "profiles": {"cuda-local": {"ref": "examples/small.json", "sha256": file_sha256(self.config.config_root / "examples" / "small.json"), "precision": "legacy_bf16", "optimizer": "lion"}},
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {},
            "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name, "sha256": file_sha256(self.binary_path)}},
        }), encoding="utf-8")
        config_path = self.config.config_root / "examples" / "small.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["training"]["max_steps"] = 2
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")
        config = WorkerConfig(
            hub_url=self.config.hub_url, worker_token=self.config.worker_token, org=self.config.org,
            config_root=self.config.config_root, dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root, run_root=self.config.run_root,
            binary=self.binary_path, binding_path=binding_path, catalog_path=catalog_path,
            worker_id="worker",
        )
        job = {
            "run_id": "nf-12345678",
            "worker_manifest": {
                "schema_version": "neural-forge-worker-manifest.v2",
                "run_id": "nf-12345678", **self.manifest_window(),
                "org": "org.example", "worker_subject": "worker",
                "authority": {
                    "subject": "user-001", "role": "operator",
                    "capability_scopes": ["neural_forge.run.submit"],
                    "justification_hash": "a" * 64,
                },
                "execution": self.execution(),
                "request": {"training_profile_id": "cuda-local", "dataset_id": "dataset-001", "resource_class": "gpu-standard"},
                "policy": {"policy_version": "policy-1", "training_profile_id": "cuda-local", "trainer_version": "v3", "resource_class": "gpu-standard", "quota_id": "quota-gpu-standard", "max_steps": 1, "timeout_seconds": 60, "max_concurrency": 1, "required_evaluation_gates": []},
                "lineage": {
                    "dataset": {"id": "dataset-001"},
                    "reproducibility": {
                        "policy_version": "policy-1", "training_profile_id": "cuda-local",
                        "trainer_version": "v3", "training_mode": "from_scratch",
                        "dataset_id": "dataset-001", "resource_class": "gpu-standard",
                    },
                },
                **self.manifest_window(),
                "deadline_at": "2026-08-23T12:00:00",
            },
        }
        with self.assertRaises(WorkerError):
            build_command(job, config)
        job["worker_manifest"].pop("deadline_at")
        with self.assertRaises(WorkerError):
            build_command(job, config)

    def test_completion_requires_a_complete_metrics_object(self) -> None:
        output = self.config.run_root / "nf-12345678"
        output.mkdir(parents=True, exist_ok=True)
        metrics = output / "metrics.json"
        metrics.write_text("{}", encoding="utf-8")
        self.assertFalse(completed_metrics(output, run_id="nf-12345678", expected_phase="native_smoke_complete", backend="cpu"))
        metrics.write_text(
            '{"type":"complete","status":"complete","backend":"native_cpu",'
            '"run_id":"nf-12345678","expected_terminal_phase":"native_smoke_complete"}',
            encoding="utf-8",
        )
        self.assertTrue(completed_metrics(output, run_id="nf-12345678", expected_phase="native_smoke_complete", backend="cpu"))

    def test_completion_accepts_canonical_cuda_terminal_status(self) -> None:
        output = self.config.run_root / "nf-12345678"
        output.mkdir(parents=True, exist_ok=True)
        (output / "status.json").write_text(
            '{"backend":"native","phase":"native_smoke_complete","job_id":"nf-12345678",'
            '"promotion_eligible":false}',
            encoding="utf-8",
        )
        (output / "model.safetensors").write_bytes(b"checkpoint")
        self.assertTrue(completed_metrics(output, run_id="nf-12345678", expected_phase="native_smoke_complete", backend="cuda"))

    def test_completion_accepts_v3_native_receipts_for_each_backend(self) -> None:
        for backend, metric_backend in (
            ("cuda", "native"),
            ("cpu", "native_cpu"),
            ("opencl", "native_opencl"),
        ):
            with self.subTest(backend=backend):
                output = self.config.run_root / f"nf-v3-{backend}123456"
                output.mkdir(parents=True, exist_ok=True)
                payload = {
                    "type": "complete",
                    "status": "complete",
                    "backend": metric_backend,
                }
                if backend == "cuda":
                    payload.update({
                        "step": 1,
                        "optimizer_steps": 1,
                        "loss": 0.5,
                        "learning_rate": 0.0003,
                        "tokens": 2048,
                        "tokens_per_second": 100.0,
                        "grad_norm": 1.0,
                        "checkpoint_written": True,
                    })
                    payload["output"] = str(output.resolve())
                    (output / "model.safetensors").write_bytes(b"checkpoint")
                else:
                    payload.update({
                        "device": "cpu" if backend == "cpu" else "AMD test device",
                        "steps": 1,
                        "loss": 0.5,
                        "tokens": 2048,
                        "tokens_per_second": 100.0,
                        "parameters_changed": True,
                        "checkpoint_written": False,
                    })
                    if backend == "cpu":
                        payload.update({"kernel_variant": "scalar", "threads": 2})
                (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
                self.assertTrue(
                    completed_metrics(
                        output,
                        run_id=f"nf-v3-{backend}123456",
                        expected_phase="native_smoke_complete",
                        backend=backend,
                        v3_native=True,
                    )
                )

    def test_v3_native_receipt_rejects_a_fabricated_completion_marker(self) -> None:
        output = self.config.run_root / "nf-v3-fake123456"
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps({
            "type": "complete",
            "status": "complete",
            "backend": "native",
            "checkpoint_written": True,
            "output": str(output.resolve()),
        }), encoding="utf-8")
        self.assertFalse(
            completed_metrics(
                output,
                run_id="nf-v3-fake123456",
                expected_phase="native_smoke_complete",
                backend="cuda",
                v3_native=True,
            )
        )

    def test_v3_cuda_receipt_requires_worker_owned_output_and_checkpoint(self) -> None:
        output = self.config.run_root / "nf-v3-cuda123456"
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "complete",
            "status": "complete",
            "backend": "native",
            "checkpoint_written": True,
            "output": str(self.root / "different-output"),
        }
        (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        (output / "model.safetensors").write_bytes(b"checkpoint")
        self.assertFalse(
            completed_metrics(
                output,
                run_id="nf-v3-cuda123456",
                expected_phase="native_smoke_complete",
                backend="cuda",
                v3_native=True,
            )
        )
        payload["output"] = str(output.resolve())
        (output / "model.safetensors").unlink()
        (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(
            completed_metrics(
                output,
                run_id="nf-v3-cuda123456",
                expected_phase="native_smoke_complete",
                backend="cuda",
                v3_native=True,
            )
        )

    def test_completion_rejects_a_different_run_identity(self) -> None:
        output = self.config.run_root / "nf-12345678"
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            '{"type":"complete","status":"complete","backend":"native_cpu",'
            '"run_id":"nf-other12345678","expected_terminal_phase":"native_smoke_complete"}',
            encoding="utf-8",
        )
        self.assertFalse(completed_metrics(output, run_id="nf-12345678", expected_phase="native_smoke_complete", backend="cpu"))

    def test_local_binding_requires_v2_and_integrity_pins(self) -> None:
        invalid = self.root / "invalid-binding.json"
        invalid.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v1",
            "catalog_version": "catalog-2026-08-23-public-blank",
            "catalog_sha256": "0" * 64,
            "profiles": {}, "datasets": {}, "models": {}, "checkpoints": {},
            "binaries": {"cuda": {"ref": self.binary_path.name}},
        }), encoding="utf-8")
        with self.assertRaises(BindingError):
            load_local_binding(invalid)

    def test_local_binding_rejects_reserved_ids_and_path_refs(self) -> None:
        base = json.loads(self.binding_path.read_text(encoding="utf-8"))
        base["profiles"]["__proto__"] = base["profiles"]["cuda-local"]
        reserved = self.root / "reserved-binding.json"
        reserved.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(BindingError):
            load_local_binding(reserved)

        base = json.loads(self.binding_path.read_text(encoding="utf-8"))
        base["datasets"]["dataset-001"]["ref"] = "C:/private/dataset"
        absolute = self.root / "absolute-binding.json"
        absolute.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(BindingError):
            load_local_binding(absolute)

        base = json.loads(self.binding_path.read_text(encoding="utf-8"))
        base["datasets"]["dataset-001"]["ref"] = "../private/dataset"
        traversal = self.root / "traversal-binding.json"
        traversal.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(BindingError):
            load_local_binding(traversal)

    def test_experimental_opencl_requires_acknowledgement_and_kernel_pin(self) -> None:
        catalog_path = Path(__file__).resolve().parents[1] / "configs" / "public" / "capabilities.json"
        opencl_config = self.config.config_root / "examples" / "opencl.json"
        opencl_config.write_text(
            (Path(__file__).resolve().parents[1] / "configs" / "examples" / "opencl_smoke.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        portable_binary = self.root / "ida_native_opencl_train"
        portable_binary.write_bytes(b"test-opencl-binary")
        kernel = portable_binary.parent / "opencl_smoke.cl"
        kernel_bytes = (Path(__file__).resolve().parents[1] / "kernels" / "opencl_smoke.cl").read_bytes()
        kernel.write_bytes(kernel_bytes)
        private_kernel_root = self.root / "private-kernels"
        private_kernel_root.mkdir()
        private_kernel = private_kernel_root / "opencl_smoke.cl"
        private_kernel.write_bytes(kernel_bytes)
        binding_path = self.root / "binding.json"
        binding_path.write_text(json.dumps({
            "schema_version": "neural-foundry-local-binding.v2",
            "catalog_version": "catalog-2026-08-23-public-blank",
            "catalog_sha256": file_sha256(catalog_path),
            "profiles": {"opencl-smoke": {"ref": "examples/opencl.json", "sha256": file_sha256(opencl_config), "precision": "fp32", "optimizer": "lion", "kernel_ref": "opencl_smoke.cl"}},
            "datasets": {"dataset-001": {"ref": "demo", "sha256": path_sha256(self.config.dataset_root / "demo")}},
            "models": {},
            "checkpoints": {},
            "binaries": {"opencl": {"ref": portable_binary.name, "sha256": file_sha256(portable_binary)}},
        }), encoding="utf-8")
        base_config = dict(
            hub_url=self.config.hub_url, worker_token=self.config.worker_token, org=self.config.org,
            config_root=self.config.config_root, dataset_root=self.config.dataset_root,
            artifact_root=self.config.artifact_root, run_root=self.config.run_root,
            kernel_root=private_kernel_root,
            binary=portable_binary, binding_path=binding_path, catalog_path=catalog_path,
            deployment_map_path=self.deployment_map_path,
            worker_id="worker",
        )
        job = {
            "run_id": "nf-opencl12345678",
            "worker_manifest": {
                "schema_version": "neural-forge-worker-manifest.v2",
                "run_id": "nf-opencl12345678", **self.manifest_window(),
                "org": "org.example", "worker_subject": "worker",
                "authority": {
                    "subject": "user-001", "role": "operator",
                    "capability_scopes": ["neural_forge.run.submit"],
                    "justification_hash": "a" * 64,
                },
                "lineage": {
                    "dataset": {"id": "dataset-001"},
                    "reproducibility": {
                        "policy_version": "policy-1", "training_profile_id": "opencl-smoke",
                        "trainer_version": "v3", "training_mode": "from_scratch",
                        "dataset_id": "dataset-001", "resource_class": "gpu-standard",
                    },
                },
                "execution": self.execution("opencl", portable_binary),
                "request": {"training_profile_id": "opencl-smoke", "dataset_id": "dataset-001", "resource_class": "gpu-standard"},
                "policy": {"policy_version": "policy-1", "training_profile_id": "opencl-smoke", "trainer_version": "v3", "resource_class": "gpu-standard", "quota_id": "quota-gpu-standard", "max_steps": 3, "timeout_seconds": 60, "max_concurrency": 1, "required_evaluation_gates": []},
            },
        }
        with self.assertRaises(WorkerError):
            build_command(job, WorkerConfig(**base_config))
        with self.assertRaises(WorkerError):
            build_command(job, WorkerConfig(**base_config, allow_experimental=True))

    def test_attestation_cannot_label_a_different_local_binary(self) -> None:
        with self.assertRaises(WorkerError):
            build_command({
                "run_id": "nf-opencl12345678",
                "execution": self.execution("opencl"),
                "request": {
                    "backend": "opencl",
                    "config_ref": "examples/small.json",
                    "dataset_ref": "demo",
                    "precision": "fp32",
                    "optimizer": "lion",
                },
            }, self.config)

    def test_binary_attestation_is_required_and_pinned(self) -> None:
        job = {
            "run_id": "nf-12345678",
            "request": {
                "backend": "cuda",
                "config_ref": "examples/small.json",
                "dataset_ref": "demo",
                "precision": "legacy_bf16",
                "optimizer": "lion",
            },
        }
        with self.assertRaises(WorkerError):
            build_command(job, self.config)
        job["execution"] = self.execution()
        job["execution"]["binary_sha256"] = "00" * 32
        with self.assertRaises(WorkerError):
            build_command(job, self.config)

    def test_child_environment_excludes_worker_credentials(self) -> None:
        environment = sanitized_environment()
        self.assertNotIn("NEURAL_FORGE_WORKER_TOKEN", environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        for loader_variable in ("PATH", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "OCL_ICD_VENDORS"):
            self.assertNotIn(loader_variable, environment)

    def test_event_projection_drops_private_fields_and_bounds_text(self) -> None:
        event = bounded_event({
            "type": "step",
            "loss": 0.5,
            "kernel_name": "private",
            "message": "x" * 400,
        })
        self.assertEqual(event["type"], "step")
        self.assertEqual(event["loss"], 0.5)
        self.assertNotIn("kernel_name", event)
        self.assertEqual(len(event["message"]), 256)
        public = public_event({"type": "step", "message": str(self.root / "private" / "checkpoint")})
        self.assertEqual(public, {"type": "step"})
        self.assertEqual(public_event({"type": "../private"}), {"type": "native_event"})

    def test_max_steps_override_is_written_only_to_local_output(self) -> None:
        config = self.root / "tests" / "worker-config-fixture.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('{"training": {"max_steps": 99}, "model": {}}', encoding="utf-8")
        try:
            output = self.root / ".pytest_worker_output"
            derived = local_config(config, output, 3)
            self.assertEqual(derived.parent, output)
            self.assertIn('"max_steps": 3', derived.read_text(encoding="utf-8"))
            self.assertIn('"max_steps": 99', config.read_text(encoding="utf-8"))
        finally:
            config.unlink(missing_ok=True)
            for path in (self.root / ".pytest_worker_output").glob("*"):
                path.unlink(missing_ok=True)
            (self.root / ".pytest_worker_output").rmdir()

    def test_existing_run_output_is_rejected(self) -> None:
        output = self.config.run_root / "nf-12345678"
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "metrics.json"
        marker.write_text("{}", encoding="utf-8")
        try:
            with self.assertRaises(WorkerError):
                build_command({
                    "run_id": "nf-12345678",
                    "execution": self.execution(),
                    "request": {
                        "backend": "cuda",
                        "config_ref": "examples/cpu_smoke.json",
                        "dataset_ref": "demo",
                        "precision": "legacy_bf16",
                        "optimizer": "lion",
                    },
                }, self.config)
        finally:
            marker.unlink(missing_ok=True)
            output.rmdir()

    def test_model_contract_is_bound_to_the_deployment_map(self) -> None:
        self.set_profile_architecture_contract("hf_gpt2_native_v1")
        deployment = json.loads(self.deployment_map_path.read_text(encoding="utf-8"))
        deployment["hub_profiles"]["cuda-local"]["model_contract_ids"] = ["hf_gpt2_native_v1"]
        self.deployment_map_path.write_text(json.dumps(deployment), encoding="utf-8")
        job = self.manifest_job(
            run_id="nf-contract12345678",
            request_overrides={"model_contract_id": "hf_gpt2_native_v1"},
        )
        job["worker_manifest"]["policy"]["model_contract_id"] = "hf_gpt2_native_v1"
        job["worker_manifest"]["lineage"]["reproducibility"]["model_contract_id"] = "hf_gpt2_native_v1"
        command, _ = build_command(job, self.config)
        self.assertIn("--request-json", command)

        deployment["hub_profiles"]["cuda-local"]["model_contract_ids"] = ["ida_lattice_native_v1"]
        self.deployment_map_path.write_text(json.dumps(deployment), encoding="utf-8")
        with self.assertRaises(WorkerError):
            build_command(job, self.config)

    def test_model_contract_is_required_on_local_parent_bindings(self) -> None:
        self.set_profile_architecture_contract("hf_gpt2_native_v1")
        model_path = self.config.artifact_root / "models" / "model-001.bin"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"model-bytes")
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["models"] = {
            "model-001": {
                "ref": "models/model-001.bin",
                "sha256": file_sha256(model_path),
            }
        }
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        job = self.manifest_job(
            run_id="nf-parentcontract1234",
            base_model_id="model-001",
            request_overrides={"model_contract_id": "hf_gpt2_native_v1"},
        )
        job["worker_manifest"]["policy"]["model_contract_id"] = "hf_gpt2_native_v1"
        job["worker_manifest"]["lineage"]["reproducibility"]["model_contract_id"] = "hf_gpt2_native_v1"
        with self.assertRaisesRegex(WorkerError, "base model contract"):
            build_command(job, self.config)

        binding["models"]["model-001"]["contract_id"] = "hf_gpt2_native_v1"
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        command, _ = build_command(job, self.config)
        self.assertIn("--request-json", command)

    def test_private_runtime_is_an_opaque_local_binding_not_a_download(self) -> None:
        deployment = json.loads(self.deployment_map_path.read_text(encoding="utf-8"))
        private_root = self.root / "private-runtime"
        private_root.mkdir(parents=True, exist_ok=True)
        package_path = private_root / "nf-private-kernel-v1.bundle"
        package_path.write_bytes(b"deployment-owned-private-package")
        package_sha256 = file_sha256(package_path)
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["private_runtimes"] = {
            "nf-private-kernel-v1": {
                "ref": package_path.name,
                "sha256": package_sha256,
                "binary_sha256": file_sha256(self.binary_path),
            }
        }
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        private_runtime = {
            "package_id": "nf-private-kernel-v1",
            "package_sha256": package_sha256,
            "tier": "private_advanced",
        }
        deployment["hub_profiles"]["cuda-local"]["private_runtime"] = private_runtime
        self.deployment_map_path.write_text(json.dumps(deployment), encoding="utf-8")
        job = self.manifest_job(
            run_id="nf-private12345678",
            policy_overrides={"private_runtime": private_runtime},
        )
        config = replace(self.config, private_runtime_root=private_root)
        command, _ = build_command(job, config)
        self.assertIn("--request-json", command)

        job["worker_manifest"]["policy"]["private_runtime"] = {
            **private_runtime,
            "package_sha256": "c" * 64,
        }
        with self.assertRaises(WorkerError):
            build_command(job, config)

        binding["private_runtimes"] = {}
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        job["worker_manifest"]["policy"]["private_runtime"] = private_runtime
        with self.assertRaisesRegex(WorkerError, "not locally approved"):
            build_command(job, config)

    def test_experimental_model_contract_requires_explicit_local_acknowledgement(self) -> None:
        config_path = self.config.config_root / "examples" / "experimental-moe.json"
        config_path.write_text(json.dumps({
            "required_arch": "sm_90",
            "attention_backend": "scalar_flash",
            "model": {
                "architecture_contract": "generic_moe_native_v1",
                "hidden_size": 128,
                "intermediate_size": 512,
                "layers": 2,
                "heads": 4,
                "vocab_size": 4096,
            },
            "training": {"microbatch": 1, "grad_accumulation": 1, "max_steps": 1},
            "input": {"batch_size": 1, "sequence_length": 2048},
        }), encoding="utf-8")
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["profiles"]["cuda-local"]["ref"] = "examples/experimental-moe.json"
        binding["profiles"]["cuda-local"]["sha256"] = file_sha256(config_path)
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        deployment = json.loads(self.deployment_map_path.read_text(encoding="utf-8"))
        deployment["hub_profiles"]["cuda-local"]["model_contract_ids"] = ["generic_moe_native_v1"]
        self.deployment_map_path.write_text(json.dumps(deployment), encoding="utf-8")
        job = self.manifest_job(
            run_id="nf-experimental123456",
            request_overrides={"model_contract_id": "generic_moe_native_v1"},
        )
        job["worker_manifest"]["policy"]["model_contract_id"] = "generic_moe_native_v1"
        job["worker_manifest"]["lineage"]["reproducibility"]["model_contract_id"] = "generic_moe_native_v1"
        with self.assertRaisesRegex(WorkerError, "experimental model contract"):
            build_command(job, self.config)
        command, _ = build_command(job, replace(self.config, allow_experimental=True))
        self.assertIn("--request-json", command)

    def test_qwen_shared_expert_shape_is_not_claimed_by_generic_moe_contract(self) -> None:
        config_path = self.config.config_root / "examples" / "shared-moe.json"
        config_path.write_text(json.dumps({
            "required_arch": "sm_90",
            "attention_backend": "scalar_flash",
            "model": {
                "architecture_contract": "generic_moe_native_v1",
                "hidden_size": 128,
                "intermediate_size": 512,
                "layers": 2,
                "heads": 4,
                "vocab_size": 4096,
                "generic_moe_num_experts": 4,
                "generic_moe_top_k": 2,
                "generic_moe_expert_width": 512,
                "generic_moe_shared_expert_width": 128,
            },
            "training": {"microbatch": 1, "grad_accumulation": 1, "max_steps": 1},
            "input": {"batch_size": 1, "sequence_length": 2048},
        }), encoding="utf-8")
        binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        binding["profiles"]["cuda-local"]["ref"] = "examples/shared-moe.json"
        binding["profiles"]["cuda-local"]["sha256"] = file_sha256(config_path)
        self.binding_path.write_text(json.dumps(binding), encoding="utf-8")
        deployment = json.loads(self.deployment_map_path.read_text(encoding="utf-8"))
        deployment["hub_profiles"]["cuda-local"]["model_contract_ids"] = ["generic_moe_native_v1"]
        self.deployment_map_path.write_text(json.dumps(deployment), encoding="utf-8")
        job = self.manifest_job(
            run_id="nf-sharedmoe123456",
            request_overrides={"model_contract_id": "generic_moe_native_v1"},
        )
        job["worker_manifest"]["policy"]["model_contract_id"] = "generic_moe_native_v1"
        job["worker_manifest"]["lineage"]["reproducibility"]["model_contract_id"] = "generic_moe_native_v1"
        with self.assertRaisesRegex(WorkerError, "shared-expert"):
            build_command(job, replace(self.config, allow_experimental=True))


if __name__ == "__main__":
    unittest.main()
