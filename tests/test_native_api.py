from __future__ import annotations

import json
from pathlib import Path
import unittest


from api.native_api import ApiConfig, NativeApiService, RequestFailure


class NativeApiBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = self._temporary_directory()
        root = Path(self.temp)
        self.dataset_root = root / "datasets"
        self.artifact_root = root / "artifacts"
        self.run_root = root / "runs"
        self.dataset_root.mkdir()
        self.artifact_root.mkdir()
        (self.dataset_root / "tokens.u32").write_bytes(b"\0" * 8192)
        (self.dataset_root / "labels.i32").write_bytes(b"\0" * 8192)
        self.service = NativeApiService(ApiConfig(
            dataset_root=self.dataset_root,
            artifact_root=self.artifact_root,
            run_root=self.run_root,
            binary=root / "ida_native_train",
        ))

    def tearDown(self) -> None:
        self._cleanup(self.temp)

    @staticmethod
    def _temporary_directory():
        import tempfile
        return tempfile.mkdtemp()

    @staticmethod
    def _cleanup(path: str) -> None:
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def valid_body() -> dict:
        return {
            "device": 0,
            "request": {
                "input": {"token_blocks": "tokens.u32", "label_blocks": "labels.i32"},
            },
        }

    def test_in_process_validation_works(self) -> None:
        self.assertEqual(self.service.validate(self.valid_body()), {"valid": True})
        self.assertFalse(self.run_root.exists())

    def test_absolute_path_is_rejected_without_echoing_it(self) -> None:
        body = self.valid_body()
        body["request"]["input"]["token_blocks"] = str(self.dataset_root / "tokens.u32")
        with self.assertRaises(RequestFailure) as error:
            self.service.validate(body)
        self.assertNotIn(str(self.dataset_root), str(error.exception))

    def test_authority_fields_are_rejected(self) -> None:
        body = self.valid_body()
        body["request"]["role"] = "admin"
        with self.assertRaises(RequestFailure):
            self.service.validate(body)

    def test_runtime_is_explicit_and_bounded(self) -> None:
        body = self.valid_body()
        body["request"]["device"] = {"runtime": "cpu"}
        self.assertEqual(self.service.validate(body), {"valid": True})
        body["request"]["device"] = {"runtime": "pytorch"}
        with self.assertRaises(RequestFailure):
            self.service.validate(body)

    def test_device_ordinal_is_bounded(self) -> None:
        body = self.valid_body()
        body["device"] = 256
        with self.assertRaises(RequestFailure):
            self.service.validate(body)

    def test_user_owned_external_model_contract_fields_are_admitted(self) -> None:
        body = self.valid_body()
        body["request"]["model"] = {
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
            "generic_moe_num_experts": 0,
            "generic_moe_top_k": 0,
            "generic_moe_expert_width": 0,
            "generic_moe_shared_expert_width": 0,
            "generic_moe_normalize_topk": True,
        }
        self.assertEqual(self.service.validate(body), {"valid": True})

    def test_model_contract_cannot_smuggle_authority(self) -> None:
        body = self.valid_body()
        body["request"]["model"] = {"architecture_contract": "hf_gpt2_native_v1", "role": "admin"}
        with self.assertRaises(RequestFailure):
            self.service.validate(body)

    def test_no_server_or_process_surface_remains(self) -> None:
        source = Path(__file__).parents[1].joinpath("api", "native_api.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "ThreadingHTTPServer",
            "BaseHTTPRequestHandler",
            "serve_forever",
            "subprocess.Popen",
            "urlopen",
            "/v1/",
        ):
            self.assertNotIn(forbidden, source)
        self.assertFalse(hasattr(self.service, "start"))


if __name__ == "__main__":
    unittest.main()
