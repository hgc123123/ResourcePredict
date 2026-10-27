"""Measure real conditional inference in a serial in-process event-time replay.

Unlike offline routing diagnostics, this command invokes the saved encoder on
every required cache miss. A saved prediction CSV is optional and is used only
after timing as a numerical parity check. This is not a scheduler experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from common.config import get_value, load_config
from common.data import load_dataset, sha256_file
from qwen.base_selection import verify_base_training_selection
from qwen.benchmark import _autocast, _model_dtype, _resolve_device, _runtime_environment, _synchronize
from qwen.contract import validate_and_restore_contract, validate_base_model_override
from qwen.dataset import QwenBatchCollator, QwenPreprocessors, ensure_tokenizer_padding, script_encoder_input, split_fingerprint
from qwen.model import GatedQwenRegressor
from reliability_cascade.integrity import canonical_sha256, verify_run_directory
from reliability_cascade.online_runtime import ExactLRU, OnlineHistory, post_fuse, pre_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="The frozen base Qwen checkpoint")
    parser.add_argument("--cascade-dir", required=True, help="A completed verified cascade run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-capacity", type=int, default=1024, help="LRU entry count; zero disables reuse")
    parser.add_argument("--cache-start", choices=("test", "validation"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--base-model")
    parser.add_argument("--warmup-jobs", type=int, default=10)
    parser.add_argument("--max-test-jobs", type=int, help="Debugging only; omit for the paper benchmark")
    parser.add_argument("--parity-predictions", help="Optional offline cascade predictions, checked only outside timing")
    parser.add_argument("--parity-atol", type=float, default=0.005, help="Allows batch-one vs batched BF16 variation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def latency_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0}
    output: dict[str, Any] = {"n": int(len(frame))}
    for name in ("total_ms", "history_activation_ms", "history_features_ms", "pre_gate_ms", "model_path_ms", "post_fusion_ms", "cache_commit_ms"):
        values = frame[name].to_numpy(float)
        output[name] = {
            "mean": float(np.mean(values)), "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        }
    seconds = float(frame["total_ms"].sum() / 1000)
    output["serial_service_throughput_jobs_per_second"] = len(frame) / seconds if seconds else None
    output["encoder_forward_rate"] = float(frame["encoder_forward"].mean())
    output["cache_hit_rate_all_jobs"] = float(frame["cache_hit"].mean())
    output["history_bypass_job_rate"] = float(frame["all_applicable_resources_bypassed"].mean())
    return output


class LiveBasePredictor:
    def __init__(self, model: Any, preprocessors: Any, collator: Any, device: Any, dtype: Any):
        self.model, self.preprocessors, self.collator = model, preprocessors, collator
        self.device, self.dtype = device, dtype

    def predict(self, row: pd.Series, cached: Any | None, *, retain_embedding: bool = True) -> tuple[np.ndarray, np.ndarray, bool, Any | None, bool]:
        # Scaling and vocabulary lookup are deliberately inside the timed path.
        job = row.to_frame().T
        metadata_np = self.preprocessors.scaler.transform(job)
        user_np = self.preprocessors.user_vocabulary.encode([str(row["user_id"])])
        user_ids = torch.as_tensor(user_np, dtype=torch.long, device=self.device)
        metadata = torch.as_tensor(metadata_np, dtype=torch.float32, device=self.device)
        code_enabled = "code" in self.model.enabled_modalities
        created = None
        forward = False
        with _autocast(self.device, self.dtype), torch.inference_mode():
            if cached is not None:
                code_hidden = cached.to(self.device)
            elif code_enabled:
                example = {
                    "code": script_encoder_input(row["code"]), "user_id": int(user_np[0]),
                    "metadata": metadata_np[0], "targets": np.zeros(2, np.float32),
                    "target_mask": np.zeros(2, bool), "row_index": 0,
                }
                batch = self.collator([example])
                code_hidden = self.model.encode_script(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
                if retain_embedding:
                    created = code_hidden.detach().to("cpu").clone()
                forward = True
            else:
                # Strict No-Code requires no tokenization or encoder allocation.
                code_hidden = self.model._zero_branch(1, self.device)
            result = self.model.predict_from_code_hidden(code_hidden, user_ids, metadata)
        _synchronize(self.device)
        return (
            result["predictions"][0].detach().float().cpu().numpy(),
            result["gate_weights"][0].detach().float().cpu().numpy(),
            bool(user_np[0] != 0), created, forward,
        )


def _parity_check(records: pd.DataFrame, path: Path, atol: float) -> dict[str, Any]:
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("Parity tolerance must be finite and non-negative")
    reference = pd.read_csv(path, dtype={"sample_id": str})
    reference = reference.loc[reference["split"].eq("test"), ["sample_id", "resource", "prediction", "history_bypass"]]
    if reference.duplicated(["sample_id", "resource"]).any():
        raise ValueError("Parity reference contains duplicate jobs/resources")
    live = pd.concat([
        records[["sample_id", f"prediction_{resource}", f"bypass_{resource}"]].rename(columns={f"prediction_{resource}": "live_prediction", f"bypass_{resource}": "live_bypass"}).assign(resource=resource)
        for resource in ("memory", "vram")
    ], ignore_index=True)
    merged = live.merge(reference, on=["sample_id", "resource"], how="left", validate="one_to_one")
    if merged["prediction"].isna().any():
        raise ValueError("Parity reference lacks measured jobs/resources")
    reference_bypass = merged["history_bypass"].astype(str).str.lower().isin(["true", "1"])
    route_mismatch = int(np.sum(merged["live_bypass"].astype(bool).to_numpy() != reference_bypass.to_numpy()))
    difference = np.abs(merged["live_prediction"].to_numpy(float) - merged["prediction"].to_numpy(float))
    # CPU-only jobs have no applicable VRAM path, so an arbitrary sentinel is
    # excluded from value parity. Route decisions are still checked above.
    applicable = merged["resource"].eq("memory").to_numpy() | np.isfinite(merged["live_prediction"].to_numpy(float))
    finite = applicable & np.isfinite(difference)
    max_abs = float(np.max(difference[finite])) if finite.any() else 0.0
    result = {"rows": len(merged), "route_mismatch_rows": route_mismatch, "max_prediction_abs_difference": max_abs, "atol": atol, "passed": route_mismatch == 0 and max_abs <= atol}
    if not result["passed"]:
        raise ValueError(f"Live/offline numerical parity failed: {result}")
    return result


def main() -> int:
    args = parse_args()
    if args.warmup_jobs < 0 or (args.max_test_jobs is not None and args.max_test_jobs <= 0):
        raise ValueError("Warmup must be non-negative and max-test-jobs positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Benchmark output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.overrides)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cascade = Path(args.cascade_dir).expanduser().resolve()
    cascade_manifest = verify_run_directory(cascade)
    training_manifest = json.loads((checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
    contract = validate_and_restore_contract(config, training_manifest)
    base_override = validate_base_model_override(contract, args.base_model)
    bundle = load_dataset(config)
    if bundle.data_sha256 != training_manifest["data_sha256"] or split_fingerprint(bundle.frame) != training_manifest["split_fingerprint"]:
        raise ValueError("Live benchmark data/splits differ from the base checkpoint")
    details = cascade_manifest.get("details") or {}
    if cascade_manifest["data_sha256"] != bundle.data_sha256 or cascade_manifest.get("seed") != training_manifest.get("seed"):
        raise ValueError("Live benchmark base and cascade do not share data/seed")
    component = details.get("component_contract") or {}
    eval_manifest_path = checkpoint.parent / "run_manifest.json"
    if not eval_manifest_path.is_file() or sha256_file(eval_manifest_path) != component.get("qwen_run_manifest_sha256"):
        raise ValueError("Base checkpoint run does not match the cascade component manifest")
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if (eval_manifest.get("details") or {}).get("experiment_contract_sha256") != contract["sha256"]:
        raise ValueError("Base evaluation and checkpoint experiment contracts differ")
    selection = training_manifest.get("base_training_selection")
    if selection:
        verify_base_training_selection(bundle, config, selection)
    models = joblib.load(cascade / "checkpoint" / "cascade_models.joblib")
    if set(models) != {"memory", "vram"}:
        raise ValueError("Saved cascade lacks exactly two resource gates")
    preprocessors = QwenPreprocessors.load(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True, trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)))
    ensure_tokenizer_padding(tokenizer)
    tokenizer.padding_side = "right"
    effective = training_manifest.get("effective_parameters") or {}
    device = _resolve_device(args.device)
    dtype = _model_dtype(config, device)
    model = GatedQwenRegressor.from_checkpoint(checkpoint, torch_dtype=dtype, model_name_override=base_override, attention_implementation="sdpa" if device.type == "cuda" else "eager").to(device).eval()
    collator = QwenBatchCollator(tokenizer, max_tokens=int(effective.get("max_tokens", 4096)), head_fraction=float(effective.get("head_fraction", 0.6)), prompt_prefix=str(effective.get("prompt_prefix", "")), prompt_suffix=str(effective.get("prompt_suffix", "")), pad_to_multiple_of=8 if device.type == "cuda" else None)
    live_base = LiveBasePredictor(model, preprocessors, collator, device, dtype)
    frame = bundle.frame.loc[bundle.frame["split"].isin(["test"] if args.cache_start == "test" else ["validation", "test"])].copy()
    frame = frame.sort_values(["submit_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    if args.max_test_jobs is not None:
        test_ids = set(frame.loc[frame["split"].eq("test"), "sample_id"].astype(str).iloc[:args.max_test_jobs])
        frame = frame.loc[~frame["split"].eq("test") | frame["sample_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No validation/test submissions available for replay")
    for index in range(args.warmup_jobs):
        live_base.predict(frame.iloc[index % len(frame)], None, retain_embedding=False)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_hashes = {str(path.relative_to(checkpoint).as_posix()): sha256_file(path) for path in sorted(checkpoint.rglob("*")) if path.is_file()}
    namespace = canonical_sha256({"experiment_contract": contract["sha256"], "checkpoint_files": checkpoint_hashes})
    cache = ExactLRU(args.cache_capacity, namespace)
    started = time.perf_counter()
    history = OnlineHistory(bundle, history_k=int(get_value(config, "reliability_cascade.history_k", 5)), half_life_hours=float(get_value(config, "reliability_cascade.history_half_life_hours", 168.0)))
    history.activate_before(int(pd.Timestamp(frame.iloc[0]["submit_time"]).value))
    initialization_seconds = time.perf_counter() - started
    records = []
    prefix_jobs = 0
    for _, row in frame.iterrows():
        _synchronize(device)
        total_start = stage_start = time.perf_counter()
        submit_ns = int(pd.Timestamp(row["submit_time"]).value)
        activated = history.activate_before(submit_ns)
        activation_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        features = history.query(row)
        identity = str(features.frame.iloc[0]["script_sha256"])
        observed = bool(features.frame.iloc[0]["script_observed"])
        features_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        routes = [pre_route(models[resource], features, index) for index, resource in enumerate(("memory", "vram"))]
        required = any(route["applicable"] and not route["bypass"] for route in routes)
        pre_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        cache.advance(submit_ns)
        cache_commit_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        cache_hit = False
        encoder_forward = False
        qwen = np.zeros(2)
        gates = np.zeros(3)
        user_known = False
        if required:
            cached = cache.get(identity, observed) if "code" in model.enabled_modalities else None
            cache_hit = cached is not None
            qwen, gates, user_known, created, encoder_forward = live_base.predict(row, cached, retain_embedding=cache.capacity > 0 and observed)
            if created is not None:
                cache.put_after_group(identity, created, observed)
        _synchronize(device)
        model_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        predictions = []
        for index, resource in enumerate(("memory", "vram")):
            route = routes[index]
            if not route["applicable"]:
                predictions.append(float("nan"))
            elif route["bypass"]:
                predictions.append(float(np.clip(route["history_prediction"], 0.0, 1.0)))
            else:
                predictions.append(post_fuse(models[resource], features, index, qwen, gates, user_known))
        post_ms = (time.perf_counter() - stage_start) * 1000
        elapsed_ms = (time.perf_counter() - total_start) * 1000
        if str(row["split"]) != "test":
            prefix_jobs += 1
            continue
        records.append({
            "sample_id": str(row["sample_id"]), "submit_time": row["submit_time"],
            "total_ms": elapsed_ms, "history_activation_ms": activation_ms,
            "history_features_ms": features_ms, "pre_gate_ms": pre_ms,
            "cache_commit_ms": cache_commit_ms, "model_path_ms": model_ms,
            "post_fusion_ms": post_ms, "activated_outcomes": activated,
            "model_path_required": required, "cache_hit": cache_hit,
            "encoder_forward": encoder_forward,
            "all_applicable_resources_bypassed": not required,
            "prediction_memory": predictions[0], "prediction_vram": predictions[1],
            "bypass_memory": routes[0]["bypass"], "bypass_vram": routes[1]["bypass"],
        })
    cache.flush()
    measured = pd.DataFrame(records)
    if measured.empty:
        raise ValueError("No test requests measured")
    parity = _parity_check(measured, Path(args.parity_predictions).expanduser().resolve(), args.parity_atol) if args.parity_predictions else {"performed": False}
    report = {
        "schema": "reliability-cascade-serial-live-path/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": training_manifest.get("seed"), "variant": training_manifest.get("variant"),
        "data_sha256": bundle.data_sha256, "split_fingerprint": split_fingerprint(bundle.frame),
        "checkpoint_namespace_sha256": namespace,
        "checkpoint_files_sha256": checkpoint_hashes,
        "cascade_models_sha256": sha256_file(cascade / "checkpoint" / "cascade_models.joblib"),
        "cache_capacity_entries": args.cache_capacity, "cache_start": args.cache_start,
        "cache_same_timestamp_policy": "insert after timestamp group; earlier hits update LRU order",
        "cache_evictions_total_including_prefix": cache.evictions,
        "cache_final_entries": len(cache.values), "cache_storage": "CPU projected code-branch tensor; heads recomputed per job",
        "history_initialization_seconds_excluded": initialization_seconds,
        "prefix_jobs_processed_not_measured": prefix_jobs,
        "warmup_model_calls_excluded": args.warmup_jobs,
        "complete_test_period": args.max_test_jobs is None,
        "cuda_synchronized": device.type == "cuda",
        "torch_num_threads": torch.get_num_threads(),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "resource_policies": {resource: saved.policy() for resource, saved in models.items()},
        "measurement_scope": "serial in-process request from standardized/redacted job object; incremental completed-label activation + history features + pre-gate + LRU + conditional numeric/user preprocessing + tokenization/encoder/fusion + post-gate/trust region",
        "excluded_scope": ["CSV import, Slurm request parsing and script redaction", "process/model/tokenizer startup", "initial historical-state preload", "IPC/RPC", "concurrent request queueing", "timeouts/retries/failures", "scheduler placement and utilization"],
        "warning": "This measures a live local inference path with actual conditional forwards, not an integrated scheduler service or a measured utilization gain. Validation cache prefix, if requested, replays the final frozen policy retrospectively.",
        "all_requests": latency_summary(measured),
        "by_route": {name: latency_summary(measured.loc[mask]) for name, mask in {
            "history_bypass": ~measured["model_path_required"],
            "exact_cache_hit": measured["cache_hit"],
            "encoder_miss": measured["encoder_forward"],
            "no_code_model_path": measured["model_path_required"] & ~measured["cache_hit"] & ~measured["encoder_forward"],
        }.items()},
        "offline_parity": parity, **_runtime_environment(device),
    }
    measured.to_csv(output / "request_timings.csv", index=False)
    (output / "online_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["all_requests"], indent=2))
    print(f"Measured live serial path: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
