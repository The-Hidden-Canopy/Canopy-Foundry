from __future__ import annotations

from pathlib import Path
import unittest

from api.native_api import _MODEL_KEYS
from scripts.neural_forge_worker import NATIVE_MODEL_FIELDS


ROOT = Path(__file__).resolve().parents[1]


class ModelPortabilitySurfaceTests(unittest.TestCase):
    def test_api_and_worker_accept_the_same_user_model_fields(self) -> None:
        self.assertEqual(_MODEL_KEYS, NATIVE_MODEL_FIELDS)
        for field in (
            "architecture_contract", "kv_heads", "generic_moe_num_experts",
            "generic_moe_top_k", "generic_moe_expert_width",
            "generic_moe_shared_expert_width", "generic_moe_normalize_topk",
            "qkv_bias", "normalization_type", "activation_type",
            "position_embedding_type", "norm_eps", "max_position_embeddings",
            "projection_bias", "tied_embeddings",
        ):
            self.assertIn(field, _MODEL_KEYS)

    def test_public_native_tree_contains_model_forward_backward_and_checkpoint_paths(self) -> None:
        trainer = (ROOT / "src" / "trainer.cu").read_text(encoding="utf-8")
        checkpoint = (ROOT / "src" / "checkpoint.cpp").read_text(encoding="utf-8")
        kernels = (ROOT / "kernels" / "layernorm.cu").read_text(encoding="utf-8")
        for symbol in (
            "hf_gpt2_native_v1", "layernorm_forward", "layernorm_backward",
            "gelu_new_forward", "gelu_new_backward", "moe_generic_forward",
            "moe_generic_backward", "position_embedding_forward",
            "position_embedding_backward",
        ):
            self.assertIn(symbol, trainer)
        self.assertIn("position_embeddings.weight", checkpoint)
        self.assertIn("ffn_in.bias", checkpoint)
        self.assertIn("__global__ void k_layernorm_forward", kernels)

    def test_public_checkpoint_surface_excludes_private_manifest_paths(self) -> None:
        checkpoint = (ROOT / "src" / "checkpoint.cpp").read_text(encoding="utf-8")
        self.assertNotIn('"ontology_path"', checkpoint)
        self.assertNotIn('"analytics_path"', checkpoint)
        self.assertIn("#if !IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY", checkpoint)

    def test_native_runtime_and_worker_do_not_import_pytorch(self) -> None:
        paths = [
            ROOT / "CMakeLists.txt",
            ROOT / "api" / "native_api.py",
            ROOT / "scripts" / "neural_forge_worker.py",
            *sorted((ROOT / "src").glob("*")),
            *sorted((ROOT / "include").glob("ida_native/*")),
            *sorted((ROOT / "kernels").glob("*")),
        ]
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in paths
            if path.is_file() and path.suffix in {".c", ".cpp", ".cu", ".h", ".hpp", ".py", ""}
        ).lower()
        self.assertNotIn("import torch", source)
        self.assertNotIn("import pytorch", source)
        self.assertNotIn("safetensors.torch", source)

    def test_public_build_does_not_link_the_legacy_adamw_kernel(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("IDA_NATIVE_ENABLE_ADAMW=0", cmake)
        self.assertNotIn("  kernels/adamw.cu", cmake)
        self.assertFalse((ROOT / "kernels" / "adamw.cu").exists())


if __name__ == "__main__":
    unittest.main()
