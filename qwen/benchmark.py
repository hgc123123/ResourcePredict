#!/usr/bin/env python3
"""Measure honest batch latency and throughput for a saved Qwen checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.data import load_dataset  # noqa: E402
from qwen.contract import (  # noqa: E402
    validate_and_restore_contract,
    validate_base_model_override,
)
from qwen.dataset import (  # noqa: E402
    QwenBatchCollator,
    QwenJobDataset,
    QwenPreprocessors,
    ensure_tokenizer_padding,
    split_fingerprint,
    validate_qwen_bundle,
)
from qwen.model import GatedQwenRegressor  # noqa: E402


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    versions = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    return ",".join(versions) or None


def _runtime_environment(device: torch.device) -> Dict[str, Any]:
    hardware: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        hardware.update(
            {
                "gpu_index": int(index),
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_compute_capability": (
                    f"{properties.major}.{properties.minor}"
                ),
                "nvidia_driver": _nvidia_driver_version(),
            }
        )
    software = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "peft": _package_version("peft"),
        "safetensors": _package_version("safetensors"),
    }
    return {"hardware": hardware, "software": software}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batch latency and throughput for a saved Qwen checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Defaults to <run-dir>/latency_benchmark.json")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-model", help="Override the base model for a LoRA checkpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument(
        "--include-tokenization",
        action="store_true",
        help="Include tokenizer collation in each recorded batch time",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_device(request: str) -> torch.device:
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dtype(config: Dict[str, Any], device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    precision = str(get_value(config, "qwen.precision", "bf16")).lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_and_predict(
    model: GatedQwenRegressor,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    user_ids = batch["user_ids"].to(device, non_blocking=True)
    metadata = batch["metadata"].to(device, non_blocking=True)
    with _autocast(device, dtype), torch.inference_mode():
        model(input_ids, attention_mask, user_ids, metadata)


def _evenly_spaced_spans(
    dataset_size: int, batch_size: int, requested_batches: int
) -> list[tuple[int, int]]:
    """Select deterministic batches across the full chronological split."""
    if dataset_size <= 0 or batch_size <= 0 or requested_batches <= 0:
        return []
    all_spans = [
        (start, min(start + batch_size, dataset_size))
        for start in range(0, dataset_size, batch_size)
    ]
    count = min(requested_batches, len(all_spans))
    if count == len(all_spans):
        return all_spans
    indices = np.linspace(0, len(all_spans) - 1, num=count)
    selected = np.rint(indices).astype(np.int64)
    if len(np.unique(selected)) != count:
        raise RuntimeError("Evenly spaced benchmark selection produced duplicate batches")
    return [all_spans[int(index)] for index in selected]


def main() -> None:
    args = parse_args()
    if args.warmup_batches < 0 or args.measure_batches <= 0:
        raise ValueError("warmup-batches must be non-negative and measure-batches positive")

    config = load_config(args.config, args.overrides)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = _load_json(checkpoint / "training_manifest.json")
    saved_contract = validate_and_restore_contract(config, manifest)
    base_model_override = validate_base_model_override(saved_contract, args.base_model)

    bundle = load_dataset(config)
    validate_qwen_bundle(bundle)
    if bundle.data_sha256 != manifest["data_sha256"]:
        raise ValueError("Benchmark CSV hash does not match the checkpoint training data")
    if split_fingerprint(bundle.frame) != manifest["split_fingerprint"]:
        raise ValueError("Benchmark split assignments do not match the checkpoint")
    if list(bundle.numeric_features) != list(manifest["numeric_features"]):
        raise ValueError("Benchmark metadata feature order does not match training")

    preprocessors = QwenPreprocessors.load(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "tokenizer",
        trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)),
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    tokenizer.padding_side = "right"

    device = _resolve_device(args.device)
    dtype = _model_dtype(config, device)
    attention_implementation = get_value(config, "qwen.attention_implementation")
    if not attention_implementation:
        attention_implementation = "eager" if device.type == "cpu" else "sdpa"
    model = GatedQwenRegressor.from_checkpoint(
        checkpoint,
        torch_dtype=dtype,
        model_name_override=base_model_override,
        attention_implementation=str(attention_implementation),
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    frame = bundle.frame.loc[bundle.frame["split"].eq(args.split)].reset_index(drop=True)
    dataset = QwenJobDataset(frame, preprocessors, training=False)
    effective = dict(manifest.get("effective_parameters") or {})
    max_tokens = int(effective.get("max_tokens", get_value(config, "qwen.max_tokens", 2048)))
    head_fraction = float(effective.get("head_fraction", 0.60))
    prompt_prefix = str(effective.get("prompt_prefix", ""))
    prompt_suffix = str(effective.get("prompt_suffix", ""))
    batch_size = int(
        args.batch_size
        or get_value(config, "qwen.eval_batch_size", get_value(config, "qwen.batch_size", 16))
    )
    if max_tokens <= 0 or batch_size <= 0:
        raise ValueError("max_tokens and batch_size must be positive")
    collator = QwenBatchCollator(
        tokenizer,
        max_tokens=max_tokens,
        head_fraction=head_fraction,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )

    if len(dataset) == 0:
        raise ValueError(f"The {args.split} split is empty")
    warmup_spans = _evenly_spaced_spans(
        len(dataset), batch_size, args.warmup_batches
    )
    measurement_spans = _evenly_spaced_spans(
        len(dataset), batch_size, args.measure_batches
    )
    warmup_count = len(warmup_spans)
    measurement_count = len(measurement_spans)

    latencies: list[float] = []
    measured_jobs = 0
    measured_batch_sizes: list[int] = []
    scheduled_spans = [(False, span) for span in warmup_spans] + [
        (True, span) for span in measurement_spans
    ]
    for measured, (start, end) in scheduled_spans:
        examples = [dataset[index] for index in range(start, end)]
        if args.include_tokenization:
            _synchronize(device)
            started = time.perf_counter()
            batch = collator(examples)
            _move_and_predict(model, batch, device, dtype)
            _synchronize(device)
            elapsed = time.perf_counter() - started
        else:
            batch = collator(examples)
            _synchronize(device)
            started = time.perf_counter()
            _move_and_predict(model, batch, device, dtype)
            _synchronize(device)
            elapsed = time.perf_counter() - started

        if measured:
            latencies.append(elapsed)
            measured_jobs += end - start
            measured_batch_sizes.append(end - start)

    latency_ms = np.asarray(latencies, dtype=np.float64) * 1000.0
    total_seconds = float(np.sum(latencies))
    scope = (
        "tokenizer_collation+host_to_device+model_forward"
        if args.include_tokenization
        else "host_to_device+model_forward;tokenizer_collation_excluded"
    )
    result: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "variant": manifest["variant"],
        "experiment_contract_sha256": saved_contract["sha256"],
        "base_model_override": base_model_override,
        "split": args.split,
        "device": str(device),
        "dtype": str(dtype),
        "measurement_unit": "one evaluated batch",
        "measurement_scope": scope,
        "model_and_tokenizer_load_time_included": False,
        "cuda_synchronized": device.type == "cuda",
        "configured_batch_size": batch_size,
        "batch_selection": "deterministic_even_spacing_across_full_split",
        "split_jobs": int(len(dataset)),
        "warmup_batches": warmup_count,
        "measured_batches": measurement_count,
        "measured_jobs": measured_jobs,
        "measured_batch_size_min": int(min(measured_batch_sizes)),
        "measured_batch_size_max": int(max(measured_batch_sizes)),
        "batch_latency_ms_mean": float(np.mean(latency_ms)),
        "batch_latency_ms_median": float(np.median(latency_ms)),
        "batch_latency_ms_p95": float(np.percentile(latency_ms, 95)),
        "batch_latency_ms_p99": float(np.percentile(latency_ms, 99)),
        "throughput_jobs_per_second": float(measured_jobs / total_seconds),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        **_runtime_environment(device),
        "note": "Batch latency is not divided by batch size or reported as request latency.",
    }

    output = (
        resolve_path(config, args.output)
        if args.output
        else checkpoint.parent / "latency_benchmark.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"Benchmark artifact: {output}")


if __name__ == "__main__":
    main()
