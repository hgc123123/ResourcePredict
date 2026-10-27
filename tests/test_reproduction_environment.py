from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.capture_environment import (
    PAPER_CUDA_BUILD_VERSION,
    PAPER_PACKAGE_VERSIONS,
    PAPER_PYTHON_VERSION,
    _model_inventory,
    _validate_gpus,
    _validate_paper_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT_ROOT / "SOURCE_SHA256SUMS.txt"


def test_model_inventory_hashes_every_file_deterministically(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_bytes(b"{}\n")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"model-bytes")

    first = _model_inventory(tmp_path, hash_files=True)
    second = _model_inventory(tmp_path, hash_files=True)

    assert first == second
    assert first["files_hashed"] is True
    assert first["file_count"] == 2
    assert first["total_bytes"] == 3 + len(b"model-bytes")
    assert [record["path"] for record in first["files"]] == [
        "config.json",
        "weights/model.safetensors",
    ]
    assert all(len(record["sha256"]) == 64 for record in first["files"])
    assert len(first["inventory_sha256"]) == 64


def test_gpu_contract_requires_count_and_normalized_model_name() -> None:
    a800 = {
        "index": 0,
        "name": "NVIDIA A800-SXM4-80GB",
        "uuid": "GPU-0",
        "driver_version": "0",
        "memory_total_mib": 81920,
    }
    _validate_gpus([dict(a800, index=index) for index in range(6)], 6, "A800")

    with pytest.raises(ValueError, match="at least 6 visible GPUs"):
        _validate_gpus([a800], 6, "A800")
    with pytest.raises(ValueError, match="do not all match"):
        _validate_gpus([dict(a800, name="NVIDIA GeForce RTX 4090D")], 1, "A800")


def test_paper_runtime_contract_fails_before_training_on_version_drift() -> None:
    payload = {
        "python": {"version": PAPER_PYTHON_VERSION},
        "packages": dict(PAPER_PACKAGE_VERSIONS),
        "torch_runtime": {
            "available": True,
            "cuda_available": True,
            "cuda_build_version": PAPER_CUDA_BUILD_VERSION,
        },
        "gpus": [{"driver_version": "550.120"}],
    }
    _validate_paper_runtime(payload, "550.120")

    payload["packages"]["transformers"] = "0.0.0"
    with pytest.raises(ValueError, match="transformers"):
        _validate_paper_runtime(payload, "550.120")


def test_source_manifest_covers_and_matches_reproduction_entry_points() -> None:
    entries: dict[str, str] = {}
    for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64
        assert relative not in entries
        entries[relative] = digest

    required = {
        "README.md",
        "docs/EXPERIMENTS.md",
        "SOURCE_MAP.md",
        "configs/paper.local.yaml",
        "requirements-paper.txt",
        "scripts/report_publication_results.py",
        "scripts/capture_environment.py",
        "scripts/run_primary.slurm",
        "scripts/run_online.slurm",
        "scripts/run_publication_experiments.py",
    }
    assert required <= entries.keys()
    for relative, expected in entries.items():
        path = PROJECT_ROOT / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative

    excluded_directories = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".test-deps",
        ".venv",
        "qa_tmp",
        "outputs",
        "logs",
    }
    excluded_directory_prefixes = (".pytest-tmp-", ".verification-", "pytest-cache-files-")
    excluded_suffixes = {".pyc", ".pyo", ".out", ".err", ".zip"}
    eligible = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and path != SOURCE_MANIFEST
        and not any(
            part in excluded_directories
            or part.startswith(excluded_directory_prefixes)
            for part in path.relative_to(PROJECT_ROOT).parts
        )
        and path.suffix.lower() not in excluded_suffixes
    }
    assert set(entries) == eligible
