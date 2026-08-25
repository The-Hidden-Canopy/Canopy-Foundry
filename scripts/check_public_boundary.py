#!/usr/bin/env python3
"""Fail-closed public-boundary checks for the Canopy Foundry repository."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN_PATH = (
    re.compile(r"(^|/)(?:build|bin|runs|data|datasets|artifacts|worker-state|secrets)(?:/|$)"),
    re.compile(r"(^|/)configs/(?:local|private)(?:/|$)"),
    re.compile(r"(^|/)(?:\.git|\.pytest_cache|__pycache__)(?:/|$)"),
    re.compile(
        r"(^|/)(?:kernels/(?:attention_wgmma|fp4|nvfp4_pack)\.cu|"
        r"include/ida_native/(?:mxf4|nvfp4|nvfp4_pack|wgmma)\.cuh|"
        r"probes/(?:mxf4_gemm|nvfp4_gemm|nvfp4_pack|wgmma_flash|wgmma_qk)_probe\.cu|"
        r"benchmarks/(?:ampere_fp4_fp8_fp4_braid|nvfp4_pack)_bench\.cu)$"
    ),
    re.compile(r"(?:\.bak[^/]*|\.orig|\.rej|\.log|\.jsonl|\.u32|\.i32|\.u16)$"),
    re.compile(r"(?:\.safetensors|\.pt|\.pth|\.ckpt|\.onnx|\.sqlite3?|\.db)$"),
)
_FORBIDDEN_CONTENT = (
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "token-shaped Hugging Face credential"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "token-shaped GitHub credential"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "token-shaped GitHub credential"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key material"),
    (re.compile(r"(?:E:|C:)\\(?:HiddenCanopy|Users)\\"), "machine-specific Windows path"),
    (re.compile(r"/(?:home/ubuntu|workspace)/"), "machine-specific Unix path"),
    (re.compile(r"IDA[-_]TRAIN[-_]V2"), "blackboard dependency reference"),
    (
        re.compile(
            r"wgmma\." + r"mma_async|mma\.sync\.aligned\.m16n8k64"
        ),
        "private kernel instruction",
    ),
    (
        re.compile(
            r"kind::" + r"mxf4|cvt\.rn\.satfinite\.e2m1x2"
        ),
        "private low-precision encoding implementation",
    ),
)
_PRIVATE_OVERLAY_PATHS = {
    "include/ida_native/ontology.hpp",
    "include/ida_native/gemm_trace.hpp",
    "include/ida_native/pack_trace.hpp",
    "amd/include/ida_native/opencl_ontology.hpp",
}
_LOCAL_CACHE_ROOT = ".local-cache/hf-git"

_PERSONAL_PROFILE_IDS = {"edge-full", "edge-swift", "ai", "moe"}


def _git_files(root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=*",
            "-C",
            str(root),
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return [item for item in result.stdout.decode("utf-8").split("\0") if item]


def _git_index_files(root: Path) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=*",
            "-C",
            str(root),
            "ls-files",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return {
        item.replace("\\", "/")
        for item in result.stdout.decode("utf-8").split("\0")
        if item
    }


def _git_history_paths(root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=*",
            "-C",
            str(root),
            "rev-list",
            "--objects",
            "--all",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        _object_id, separator, path = line.partition(" ")
        if separator and path:
            paths.append(path.replace("\\", "/"))
    return paths


def _decode_text(data: bytes) -> str | None:
    if b"\0" in data or len(data) > 8 * 1024 * 1024:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return _decode_text(data)


def _read_index_text(root: Path, relative: str) -> str | None:
    result = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=*",
            "-C",
            str(root),
            "show",
            f":{relative}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return _decode_text(result.stdout)


def _is_ignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=*",
            "-C",
            str(root),
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            relative,
        ],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _is_ignored_local_cache_path(root: Path, relative: str) -> bool:
    return (
        relative == _LOCAL_CACHE_ROOT
        or relative.startswith(f"{_LOCAL_CACHE_ROOT}/")
    ) and _is_ignored(root, relative)


def _nested_git_metadata(root: Path) -> list[str]:
    nested_git: list[str] = []
    for current, directories, _files in os.walk(root):
        current_path = Path(current)
        for directory in list(directories):
            path = current_path / directory
            relative = path.relative_to(root).as_posix()
            if _is_ignored_local_cache_path(root, relative):
                directories.remove(directory)
                continue
            if directory == ".git":
                if path != root / ".git":
                    nested_git.append(relative)
                directories.remove(directory)
    return sorted(nested_git)


def _content_findings(relative: str, text: str, source: str = "") -> list[str]:
    location = f"{relative} ({source})" if source else relative
    return [
        f"{label} in {location}"
        for pattern, label in _FORBIDDEN_CONTENT
        if pattern.search(text)
    ]


def check_boundary(root: Path = ROOT, *, include_history: bool = False) -> list[str]:
    findings: list[str] = []
    files = _git_files(root)
    index_files = _git_index_files(root)
    normalized = {item.replace("\\", "/") for item in files}

    for relative in sorted(normalized):
        if relative in _PRIVATE_OVERLAY_PATHS:
            findings.append(f"private overlay is tracked: {relative}")
        if any(pattern.search(relative) for pattern in _FORBIDDEN_PATH):
            findings.append(f"private/generated path is publishable: {relative}")

        path = root / Path(relative)
        text = _read_text(path)
        if text is not None:
            findings.extend(_content_findings(relative, text))
        if relative in index_files:
            index_text = _read_index_text(root, relative)
            if index_text is not None:
                findings.extend(_content_findings(relative, index_text, "staged index"))

    findings.extend(f"nested Git metadata: {path}" for path in _nested_git_metadata(root))

    if include_history:
        for relative in sorted(set(_git_history_paths(root))):
            if any(pattern.search(relative) for pattern in _FORBIDDEN_PATH):
                findings.append(f"private/generated path is reachable in Git history: {relative}")

    catalog_path = root / "configs" / "public" / "capabilities.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        profiles = catalog["profiles"]
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("profiles must be a non-empty object")
        for name, profile in profiles.items():
            if name in _PERSONAL_PROFILE_IDS:
                findings.append(f"personal profile identity is publishable: {name}")
            expected_optimizers = [] if (
                profile.get("enabled", True) is False
                and profile.get("backend") in {"cpu", "opencl"}
            ) else ["lion"]
            if profile.get("supported_optimizers") != expected_optimizers:
                findings.append(f"optimizer catalog does not match enabled state: {name}")
            if set(profile) & {"model", "training", "input", "dataset", "checkpoint"}:
                findings.append(f"public profile contains user run configuration: {name}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        findings.append(f"public capability catalog is invalid: {exc}")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also inspect paths reachable from all local Git refs",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    findings = check_boundary(root, include_history=args.history)
    if findings:
        print("PUBLIC BOUNDARY: FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("PUBLIC BOUNDARY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
