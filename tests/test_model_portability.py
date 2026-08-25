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

    def test_public_build_cannot_inherit_private_observability(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        presets = (ROOT / "CMakePresets.json").read_text(encoding="utf-8")
        self.assertIn("option(CANOPY_FOUNDRY_PUBLIC_BUILD", cmake)
        self.assertIn(
            "if(CANOPY_FOUNDRY_PUBLIC_BUILD AND IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY)",
            cmake,
        )
        self.assertIn('"CANOPY_FOUNDRY_PUBLIC_BUILD": "ON"', presets)
        self.assertIn('"IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY": "OFF"', presets)

    def test_public_build_has_a_pinned_nlohmann_fallback(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        amd_cmake = (ROOT / "amd" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("find_package(nlohmann_json 3.11 CONFIG QUIET)", cmake)
        self.assertIn("FetchContent_Declare(nlohmann_json", cmake)
        self.assertIn(
            "URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d",
            cmake,
        )
        self.assertIn(
            "if(NOT TARGET nlohmann_json::nlohmann_json)",
            amd_cmake,
        )

    def test_generated_run_and_dependency_trees_are_ignored(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for path in ("/run-output/", "/smoke-run/", "/output/", "/outputs/", "/.deps/"):
            self.assertIn(path, gitignore)

    def test_public_build_does_not_compile_dead_adam_constants(self) -> None:
        trainer = (ROOT / "src" / "trainer.cu").read_text(encoding="utf-8")
        self.assertIn(
            "#if IDA_NATIVE_ENABLE_ADAMW\n    const float adam_b1",
            trainer,
        )

    def test_public_cuda_sources_have_no_known_compile_boundary_regressions(self) -> None:
        trainer = (ROOT / "src" / "trainer.cu").read_text(encoding="utf-8")
        fp8 = (ROOT / "kernels" / "fp8.cu").read_text(encoding="utf-8")
        self.assertNotIn("if (weight == nullptr) return;", trainer)
        self.assertIn('#include "ida_native/pack_trace.hpp"', fp8)

    def test_public_cuda_target_contains_the_declared_kernel_definitions(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        activations = (ROOT / "kernels" / "activations.cu").read_text(encoding="utf-8")
        positions = (ROOT / "kernels" / "position_embeddings.cu").read_text(encoding="utf-8")
        attention = (ROOT / "kernels" / "attention.cu").read_text(encoding="utf-8")
        for source in ("kernels/activations.cu", "kernels/position_embeddings.cu"):
            self.assertIn(source, cmake)
        for symbol in ("void swiglu_forward", "void swiglu_backward"):
            self.assertIn(symbol, activations)
        for symbol in ("void position_embedding_forward", "void position_embedding_backward"):
            self.assertIn(symbol, positions)
        self.assertIn("void attention_forward", attention)
        self.assertIn("void attention_backward", attention)
        self.assertIn("cudaStream_t stream, int nKVH", attention)

    def test_native_artifacts_require_explicit_promotion_enablement(self) -> None:
        main = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")
        checkpoint = (ROOT / "src" / "checkpoint.cpp").read_text(encoding="utf-8")
        self.assertIn("request.promotion_enabled && merged_weights", main)
        self.assertIn("request.promotion_enabled &&", main)
        self.assertIn(
            "const bool promotion_eligible = request.promotion_enabled && real_weights;",
            checkpoint,
        )
        self.assertNotIn("const bool promotion_eligible = real_weights;", checkpoint)
        self.assertNotIn(
            '(real_weights ? "true" : "false")',
            checkpoint,
        )


if __name__ == "__main__":
    unittest.main()
