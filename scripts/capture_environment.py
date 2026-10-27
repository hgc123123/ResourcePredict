#!/usr/bin/env python3
"""Capture and validate the execution environment without exposing secrets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "joblib",
    "xgboost",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
)
SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_CPUS_ON_NODE",
)
PAPER_PYTHON_VERSION = "3.12.9"
PAPER_CUDA_BUILD_VERSION = "12.6"
PAPER_PACKAGE_VERSIONS = {
    "numpy": "2.1.3",
    "pandas": "2.3.0",
    "scikit-learn": "1.7.0",
    "xgboost": "3.0.2",
    "torch": "2.7.1",
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "accelerate": "1.7.0",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=("training", "ablation", "latency", "resume"),
    )
    parser.add_argument("--data-csv")
    parser.add_argument("--model-dir")
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-gpu-count", type=int)
    parser.add_argument("--expected-gpu-name-substring")
    parser.add_argument("--expected-driver-version")
    parser.add_argument(
        "--enforce-paper-runtime",
        action="store_true",
        help="Fail before writing when recorded Python/package/CUDA versions differ",
    )
    parser.add_argument(
        "--hash-model-files",
        action="store_true",
        help="Hash every regular file in the base-model directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "torch_version": str(torch.__version__),
        "cuda_build_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _run_text(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "stdout": completed.stdout.strip()}


def _visible_gpus() -> list[dict[str, Any]]:
    result = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result["available"]:
        return []
    rows: list[dict[str, Any]] = []
    for line in str(result["stdout"]).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": int(parts[4]),
            }
        )
    return rows


def _normalized_device_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _validate_gpus(
    gpus: list[dict[str, Any]],
    expected_count: int | None,
    expected_name: str | None,
) -> None:
    if expected_count is not None:
        if expected_count <= 0:
            raise ValueError("--expected-gpu-count must be positive")
        if len(gpus) < expected_count:
            raise ValueError(
                f"Expected at least {expected_count} visible GPUs, observed {len(gpus)}"
            )
    if expected_name:
        needle = _normalized_device_name(expected_name)
        if not needle:
            raise ValueError("--expected-gpu-name-substring is empty after normalization")
        mismatched = [
            str(gpu["name"])
            for gpu in gpus
            if needle not in _normalized_device_name(str(gpu["name"]))
        ]
        if not gpus or mismatched:
            raise ValueError(
                f"Visible GPUs do not all match {expected_name!r}: "
                f"observed={[gpu['name'] for gpu in gpus]}"
            )


def _validate_paper_runtime(
    payload: dict[str, Any], expected_driver_version: str | None = None
) -> None:
    python = payload.get("python") or {}
    if str(python.get("version")) != PAPER_PYTHON_VERSION:
        raise ValueError(
            f"Paper runtime requires Python {PAPER_PYTHON_VERSION}; "
            f"observed {python.get('version')!r}"
        )
    packages = payload.get("packages") or {}
    for package, expected in PAPER_PACKAGE_VERSIONS.items():
        observed = packages.get(package)
        if str(observed) != expected:
            raise ValueError(
                f"Paper runtime requires {package}={expected}; observed {observed!r}"
            )
    torch_runtime = payload.get("torch_runtime") or {}
    if torch_runtime.get("available") is not True:
        raise ValueError("Paper runtime requires an importable PyTorch installation")
    if torch_runtime.get("cuda_available") is not True:
        raise ValueError("Paper runtime requires CUDA to be available to PyTorch")
    if str(torch_runtime.get("cuda_build_version")) != PAPER_CUDA_BUILD_VERSION:
        raise ValueError(
            f"Paper runtime requires CUDA build {PAPER_CUDA_BUILD_VERSION}; "
            f"observed {torch_runtime.get('cuda_build_version')!r}"
        )
    if expected_driver_version:
        mismatched = [
            str(gpu.get("driver_version"))
            for gpu in payload.get("gpus", [])
            if str(gpu.get("driver_version")) != expected_driver_version
        ]
        if mismatched:
            raise ValueError(
                f"Expected NVIDIA driver {expected_driver_version}; "
                f"observed {mismatched}"
            )
def _model_inventory(model_dir: Path, hash_files: bool) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Base-model directory not found: {model_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        record: dict[str, Any] = {
            "path": path.relative_to(model_dir).as_posix(),
            "bytes": int(path.stat().st_size),
        }
        if hash_files:
            record["sha256"] = sha256_file(path)
        records.append(record)
    if not records:
        raise ValueError(f"Base-model directory contains no files: {model_dir}")
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "path": str(model_dir),
        "file_count": len(records),
        "total_bytes": int(sum(record["bytes"] for record in records)),
        "files_hashed": bool(hash_files),
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": records,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Environment artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    gpus = _visible_gpus()
    _validate_gpus(gpus, args.expected_gpu_count, args.expected_gpu_name_substring)

    payload: dict[str, Any] = {
        "schema": "paper-experiment-environment/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": _package_versions(),
        "torch_runtime": _torch_runtime(),
        "gpus": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm": {key: os.environ.get(key) for key in SLURM_KEYS},
        "module_list": _run_text(["bash", "-lc", "module -t list 2>&1"]),
        "source_manifest": str(PROJECT_ROOT / "SOURCE_SHA256SUMS.txt"),
        "source_manifest_sha256": sha256_file(PROJECT_ROOT / "SOURCE_SHA256SUMS.txt"),
    }

    if args.data_csv:
        data_path = Path(args.data_csv).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {data_path}")
        data_hash = sha256_file(data_path)
        expected_hash = (
            str(args.expected_data_sha256).strip().lower()
            if args.expected_data_sha256
            else None
        )
        if expected_hash and data_hash != expected_hash:
            raise ValueError(
                f"Dataset SHA-256 mismatch: expected={expected_hash}, actual={data_hash}"
            )
        payload["dataset"] = {
            "path": str(data_path),
            "bytes": int(data_path.stat().st_size),
            "sha256": data_hash,
            "expected_sha256": expected_hash,
        }

    if args.model_dir:
        payload["base_model"] = _model_inventory(
            Path(args.model_dir).expanduser().resolve(), bool(args.hash_model_files)
        )

    if args.enforce_paper_runtime:
        _validate_paper_runtime(payload, args.expected_driver_version)

    _atomic_write_json(output, payload, bool(args.overwrite))
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print(f"Environment artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
