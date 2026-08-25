from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_public_boundary import check_boundary  # noqa: E402


class PublicBoundaryTests(unittest.TestCase):
    def test_public_boundary_is_clean(self) -> None:
        self.assertEqual(check_boundary(ROOT), [])

    def test_nested_repository_is_not_present(self) -> None:
        self.assertFalse((ROOT / "neural-foundry" / ".git").exists())

    def test_public_catalog_optimizer_policy(self) -> None:
        catalog = json.loads(
            (ROOT / "configs" / "public" / "capabilities.json").read_text(encoding="utf-8")
        )
        self.assertTrue(catalog["profiles"])
        self.assertEqual(
            set(catalog["profiles"]),
            {"cuda-local", "opencl-smoke", "cpu-smoke"},
        )
        self.assertNotIn("edge-full", catalog["profiles"])
        self.assertNotIn("edge-swift", catalog["profiles"])
        self.assertNotIn("ai", catalog["profiles"])
        self.assertNotIn("moe", catalog["profiles"])
        for profile_id, profile in catalog["profiles"].items():
            expected = [] if profile_id in {"opencl-smoke", "cpu-smoke"} else ["lion"]
            self.assertEqual(profile["supported_optimizers"], expected)
            self.assertFalse(set(profile) & {"model", "training", "input", "dataset", "checkpoint"})

    def test_cli_welcome_explains_the_local_power_path(self) -> None:
        welcome = (ROOT / "include" / "ida_native" / "cli_welcome.hpp").read_text(encoding="utf-8")
        self.assertIn("Power path: local config -> local data/checkpoint -> native trainer.", welcome)
        self.assertIn("Control path:", welcome)
        self.assertIn("docs/hf-git-cache.md", welcome)
        self.assertIn("Leaderboard: opt in only; normal training never submits leaderboard data.", welcome)
        self.assertIn("Publication requires a separate confirmation", welcome)
        self.assertIn("credentials", welcome)
        self.assertIn("private telemetry stay local.", welcome)
        self.assertNotIn("worker-token", welcome.lower())

    def test_optional_hf_git_cache_is_ignored_and_documented(self) -> None:
        ignored_cache_path = ".local-cache/hf-git/huggingface/token"
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT}",
                "-C",
                str(ROOT),
                "check-ignore",
                "--no-index",
                "-q",
                "--",
                ignored_cache_path,
            ],
            check=False,
        )
        self.assertEqual(result.returncode, 0)

        guide = (ROOT / "docs" / "hf-git-cache.md").read_text(encoding="utf-8")
        self.assertIn("hf auth login", guide)
        self.assertIn("credential.helper store", guide)
        self.assertIn("does not add an HF or PyTorch dependency", guide)
        self.assertTrue((ROOT / "scripts" / "enable_hf_git_cache.ps1").is_file())

    def test_boundary_scans_the_staged_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs" / "public").mkdir(parents=True)
            (root / "configs" / "public" / "capabilities.json").write_text(
                json.dumps(
                    {
                        "profiles": {
                            "cpu-smoke": {
                                "backend": "cpu",
                                "enabled": False,
                                "supported_optimizers": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "-C",
                    str(root),
                    "add",
                    "--",
                    "configs/public/capabilities.json",
                    "tracked.txt",
                ],
                check=True,
            )
            tracked.write_text(f"hf_{'A' * 24}\n", encoding="utf-8")
            subprocess.run(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "-C",
                    str(root),
                    "add",
                    "--",
                    "tracked.txt",
                ],
                check=True,
            )
            tracked.write_text("clean\n", encoding="utf-8")

            findings = check_boundary(root)

        self.assertIn(
            "token-shaped Hugging Face credential in tracked.txt (staged index)",
            findings,
        )

    def test_boundary_rejects_private_kernel_sources_and_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs" / "public").mkdir(parents=True)
            (root / "configs" / "public" / "capabilities.json").write_text(
                json.dumps(
                    {
                        "profiles": {
                            "cpu-smoke": {
                                "backend": "cpu",
                                "enabled": False,
                                "supported_optimizers": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            private_source = root / "kernels" / "fp4.cu"
            private_source.parent.mkdir(parents=True)
            private_source.write_text(
                'asm volatile("' + "wgmma." + "mma_async.sync.aligned.m64n64k32" + '");\n',
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)

            findings = check_boundary(root)

        self.assertTrue(any("private/generated path is publishable" in item for item in findings))
        self.assertTrue(any("private kernel instruction" in item for item in findings))

    def test_ignored_cache_git_metadata_does_not_fail_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs" / "public").mkdir(parents=True)
            (root / "configs" / "public" / "capabilities.json").write_text(
                json.dumps(
                    {
                        "profiles": {
                            "cpu-smoke": {
                                "backend": "cpu",
                                "enabled": False,
                                "supported_optimizers": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / ".gitignore").write_text("/.local-cache/hf-git/*\n", encoding="utf-8")
            (root / ".local-cache" / "hf-git" / "repos" / "model" / ".git").mkdir(parents=True)
            (root / "outside" / ".git").mkdir(parents=True)
            subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)

            findings = check_boundary(root)

        self.assertNotIn(
            "nested Git metadata: .local-cache/hf-git/repos/model/.git",
            findings,
        )
        self.assertIn("nested Git metadata: outside/.git", findings)


if __name__ == "__main__":
    unittest.main()
