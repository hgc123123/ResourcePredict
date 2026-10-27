"""Measure a frozen cascade's actual serial, in-process online prediction path.

Run via scripts/run_online_study.py. No training or test-based selection occurs.
Saved predictions are used outside timed regions for reference verification;
the measured path never looks up a saved job prediction to replace inference.
"""
from __future__ import annotations

import argparse
import json
import time
from copy import copy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from common.config import get_value, load_config, set_value
from common.data import load_dataset
from qwen.base_selection import verify_base_training_selection
from qwen.benchmark import _autocast, _model_dtype, _resolve_device, _runtime_environment, _synchronize
from qwen.contract import validate_and_restore_contract, validate_base_model_override
from qwen.dataset import QwenBatchCollator, QwenPreprocessors, ensure_tokenizer_padding, script_encoder_input, split_fingerprint
from qwen.model import GatedQwenRegressor
from reliability_cascade.integrity import canonical_sha256, verify_run_directory
from reliability_cascade.online_runtime import OnlineHistory
from online_study.measurement import OPERATIONS, STAGES, cache_parity_check, frozen_reference, latency_summary, parity_check, sha256, source_contract
from online_study.runtime import ServingLRU, fuse_frozen, route_frozen
from online_study.reference import load_or_build_base_reference
from online_study.offline_reference import (archived_base_jobs, base_parity, canonical_predictions,
    comparison_details, drift_summary, inject_live_inputs, structural_archive_check)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cascade-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--operation", choices=OPERATIONS, default="full")
    p.add_argument("--cache-capacity", type=int, default=1024)
    p.add_argument("--cache-start", choices=("validation", "test"), default="validation")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    p.add_argument("--base-model")
    p.add_argument("--data-csv")
    p.add_argument("--warmup-jobs", type=int, default=10)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--max-test-jobs", type=int, help="Debug only: such a run is not publishable")
    p.add_argument("--parity-predictions", help="Defaults to the sealed cascade predictions.csv")
    p.add_argument("--parity-atol", type=float, default=0.005)
    p.add_argument("--reference-dir", help="Shared seed-specific independent batch-one reference directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--set", action="append", default=[], dest="overrides")
    return p.parse_args()


class TimedBasePredictor:
    def __init__(self, model, preprocessors, collator, device, dtype):
        self.model, self.preprocessors, self.collator = model, preprocessors, collator
        self.device, self.dtype = device, dtype

    def predict(self, row, cached=None, *, retain_embedding=True):
        times = {name: 0.0 for name in ("numeric_user_ms", "tokenization_ms", "code_path_ms", "fusion_heads_ms")}
        t = time.perf_counter()
        metadata_np = self.preprocessors.scaler.transform(row.to_frame().T)
        user_np = self.preprocessors.user_vocabulary.encode([str(row.user_id)])
        user_ids = torch.as_tensor(user_np, dtype=torch.long, device=self.device)
        metadata = torch.as_tensor(metadata_np, dtype=torch.float32, device=self.device)
        _synchronize(self.device)
        times["numeric_user_ms"] = (time.perf_counter() - t) * 1000
        created, forward = None, False
        with _autocast(self.device, self.dtype), torch.inference_mode():
            if cached is None and "code" in self.model.enabled_modalities:
                t = time.perf_counter()
                example = {"code": script_encoder_input(row.code), "user_id": int(user_np[0]),
                           "metadata": metadata_np[0], "targets": np.zeros(2, np.float32),
                           "target_mask": np.zeros(2, bool), "row_index": 0}
                batch = self.collator([example])
                times["tokenization_ms"] = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                code_hidden = self.model.encode_script(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
                if retain_embedding:
                    created = code_hidden.detach().to("cpu").clone()
                _synchronize(self.device)
                times["code_path_ms"] = (time.perf_counter() - t) * 1000
                forward = True
            else:
                t = time.perf_counter()
                code_hidden = cached.to(self.device) if cached is not None else self.model._zero_branch(1, self.device)
                _synchronize(self.device)
                times["code_path_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            result = self.model.predict_from_code_hidden(code_hidden, user_ids, metadata)
            predictions = result["predictions"][0].detach().float().cpu().numpy()
            gates = result["gate_weights"][0].detach().float().cpu().numpy()
            _synchronize(self.device)
            times["fusion_heads_ms"] = (time.perf_counter() - t) * 1000
        if not np.isfinite(predictions).all() or not np.isfinite(gates).all():
            raise ValueError("Non-finite live model outputs")
        return predictions, gates, bool(user_np[0] != 0), created, forward, times


def _history_settings(checkpoint_contract, base_config):
    h = checkpoint_contract["history_contract"]
    if h.get("strict_completion_relation") != "completion_time < submit_time":
        raise ValueError("Unsupported saved history eligibility rule")
    k, half = int(h["history_k"]), float(h["decay_half_life_hours"])
    for key, value in (("history_k", k), ("history_half_life_hours", half)):
        configured = get_value(base_config, f"reliability_cascade.{key}")
        if configured is not None and float(configured) != float(value):
            raise ValueError(f"Configuration conflicts with frozen cascade history: {key}")
    return k, half


def _validate_base_architecture(model, preprocessors, bundle, manifest, contract):
    if preprocessors.scaler.columns != bundle.numeric_features:
        raise ValueError("Saved scaler columns differ from live metadata")
    if list(manifest["numeric_features"]) != bundle.numeric_features:
        raise ValueError("Base manifest metadata differs from live metadata")
    if model.num_users != len(preprocessors.user_vocabulary) or model.metadata_dim != len(bundle.numeric_features):
        raise ValueError("Base model architecture differs from saved preprocessors")
    e = contract["payload"]["qwen_semantics"]
    checks = {name: (getattr(model, name), e[name]) for name in (
        "fusion_dim", "user_embedding_dim", "head_hidden_dim", "fusion_mode")}
    checks.update(enabled_modalities=(list(model.enabled_modalities), e["enabled_modalities"]),
                  dropout=(model.dropout_probability, e["dropout"]),
                  lora_enabled=(model.lora_enabled, e["lora"]["enabled"]))
    changed = [name for name, (observed, expected) in checks.items() if observed != expected]
    if changed:
        raise ValueError(f"Base architecture differs from saved contract: {changed}")


def _accuracy(records, reference):
    rows = []
    for resource in ("memory", "vram"):
        ref = reference.loc[reference.split.eq("test") & reference.resource.eq(resource)].copy()
        ref["sample_id"] = ref.sample_id.astype(str)
        merged = records[["sample_id", f"prediction_{resource}"]].merge(ref, on="sample_id", validate="one_to_one")
        valid = merged.valid.astype(str).str.lower().isin(("true", "1"))
        y, p = merged.loc[valid, "target"].to_numpy(float), merged.loc[valid, f"prediction_{resource}"].to_numpy(float)
        if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("Cannot calculate finite live accuracy")
        error = p - y
        rows.append({"resource": resource, "n": len(y), "mae": float(abs(error).mean()),
                     "mse": float(np.square(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                     "acc_at_10": float((abs(error) <= .10).mean())})
    return rows


def main():
    args = parse_args()
    if args.warmup_jobs < 0 or args.num_threads < 1 or args.repeat < 1 or args.cache_capacity < 0:
        raise ValueError("Invalid warmup, threads, repeat or cache capacity")
    if args.max_test_jobs is not None and args.max_test_jobs < 1:
        raise ValueError("max-test-jobs must be positive")
    torch.set_num_threads(args.num_threads)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Nonempty output directory: {output}; inspect it before re-running")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, cascade = (Path(p).expanduser().resolve() for p in (args.checkpoint, args.cascade_dir))
    cascade_manifest = verify_run_directory(cascade)  # includes original source and every sealed artifact
    cp_contract = json.loads((cascade / "checkpoint/checkpoint_contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
    config = load_config(args.config, args.overrides)
    if args.data_csv:
        set_value(config, "data.path", str(Path(args.data_csv).expanduser().resolve()))
    contract = validate_and_restore_contract(config, manifest)
    base_override = validate_base_model_override(contract, args.base_model)
    bundle = load_dataset(config)
    split_hash = split_fingerprint(bundle.frame)
    if bundle.data_sha256 != manifest["data_sha256"] or split_hash != manifest["split_fingerprint"]:
        raise ValueError("Live data/splits differ from base checkpoint")
    if cascade_manifest["data_sha256"] != bundle.data_sha256 or cascade_manifest.get("seed") != manifest.get("seed"):
        raise ValueError("Cascade and base data/seed differ")
    components = cp_contract["component_contract"]
    base_manifest_path = checkpoint.parent / "run_manifest.json"
    if sha256(base_manifest_path) != components["qwen_run_manifest_sha256"]:
        raise ValueError("Wrong base run for this cascade")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if (base_manifest.get("details") or {}).get("experiment_contract_sha256") != contract["sha256"]:
        raise ValueError("Base evaluation/checkpoint contracts differ")
    if manifest.get("base_training_selection"):
        verify_base_training_selection(bundle, config, manifest["base_training_selection"])
    history_k, half_life = _history_settings(cp_contract, config)
    models = joblib.load(cascade / "checkpoint/cascade_models.joblib")
    if set(models) != {"memory", "vram"}:
        raise ValueError("Expected exactly two frozen resource models")
    original_policies = {r: m.policy() for r, m in models.items()}
    canonical_models = models
    if args.operation == "no_trust":
        models = {r: copy(m) for r, m in models.items()}
        for m in models.values():
            m.config = replace(m.config, trust_region_enabled=False)
    preprocessors = QwenPreprocessors.load(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True,
        trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)))
    ensure_tokenizer_padding(tokenizer)
    tokenizer.padding_side = "right"
    effective = manifest["effective_parameters"]
    device, dtype = _resolve_device(args.device), None
    dtype = _model_dtype(config, device)
    torch.manual_seed(int(manifest["seed"]))
    model = GatedQwenRegressor.from_checkpoint(checkpoint, torch_dtype=dtype,
        model_name_override=base_override, attention_implementation="sdpa" if device.type == "cuda" else "eager").to(device).eval()
    _validate_base_architecture(model, preprocessors, bundle, manifest, contract)
    collator = QwenBatchCollator(tokenizer, max_tokens=int(effective["max_tokens"]),
        head_fraction=float(effective.get("head_fraction", .6)), prompt_prefix=str(effective.get("prompt_prefix", "")),
        prompt_suffix=str(effective.get("prompt_suffix", "")), pad_to_multiple_of=8 if device.type == "cuda" else None)
    live = TimedBasePredictor(model, preprocessors, collator, device, dtype)
    reference_path = Path(args.parity_predictions).expanduser().resolve() if args.parity_predictions else cascade / "predictions.csv"
    if sha256(reference_path) != sha256(cascade / "predictions.csv"):
        raise ValueError("Parity input differs from the sealed cascade predictions")
    archive = pd.read_csv(reference_path, dtype={"sample_id": str})
    archived_base_path = checkpoint.parent / "predictions.csv"
    if sha256(archived_base_path) != components["qwen_predictions_sha256"]:
        raise ValueError("Archived base prediction file differs from cascade component contract")
    original_base = archived_base_jobs(pd.read_csv(archived_base_path, dtype={"sample_id": str}))
    reference_options = dict(history_k=history_k, half_life_hours=half_life)
    original_canonical = canonical_predictions(archive, original_base, bundle, canonical_models, **reference_options)
    # First reproduce the sealed results using the *original* numeric inputs.
    # This strict check rejects a changed history, gate, policy or blend formula.
    original_structure = structural_archive_check(original_canonical, archive, args.operation)
    archived_expected = frozen_reference(original_canonical, args.operation)
    checkpoint_hashes = {p.relative_to(checkpoint).as_posix(): sha256(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    namespace = canonical_sha256({"experiment_contract": contract["sha256"], "checkpoint_files": checkpoint_hashes})
    runtime_environment = _runtime_environment(device)
    benchmark_source = source_contract(Path(__file__).resolve().parents[1])
    all_eval = bundle.frame.loc[bundle.frame.split.isin(["validation", "test"])].sort_values(["submit_time", "sample_id"], kind="stable").reset_index(drop=True)
    reference_dir = Path(args.reference_dir).expanduser().resolve() if args.reference_dir else output / "batch_one_reference"
    reference_base, reference_metadata = load_or_build_base_reference(reference_dir,
        frame=all_eval, model=model, preprocessors=preprocessors, collator=collator, device=device, dtype=dtype,
        identity={"checkpoint_namespace_sha256": namespace, "data_sha256": bundle.data_sha256,
            "split_fingerprint": split_hash, "source_contract_sha256": benchmark_source["sha256"],
            "dtype": str(dtype), "num_threads": args.num_threads, "seed": int(manifest["seed"]),
            "attention_implementation": model.attention_implementation,
            "experiment_contract_sha256": contract["sha256"], **runtime_environment})
    canonical_batch_one = canonical_predictions(archive, reference_base, bundle, canonical_models, **reference_options)
    reference = frozen_reference(canonical_batch_one, args.operation)
    keep_splits = ["validation", "test"] if args.cache_start == "validation" else ["test"]
    frame = bundle.frame.loc[bundle.frame.split.isin(keep_splits)].sort_values(["submit_time", "sample_id"], kind="stable").reset_index(drop=True)
    expected_test_n = int(bundle.frame.split.eq("test").sum())
    if args.max_test_jobs is not None:
        ids = set(frame.loc[frame.split.eq("test"), "sample_id"].astype(str).iloc[:args.max_test_jobs])
        frame = frame.loc[~frame.split.eq("test") | frame.sample_id.astype(str).isin(ids)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No submissions to measure")
    for index in range(args.warmup_jobs):
        live.predict(frame.iloc[index % len(frame)], retain_embedding=False)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cache = ServingLRU(args.cache_capacity, namespace)
    t = time.perf_counter()
    history = OnlineHistory(bundle, history_k=history_k, half_life_hours=half_life)
    history.activate_before(int(pd.Timestamp(frame.iloc[0].submit_time).value))
    initialization_seconds = time.perf_counter() - t
    records, events, prefix_jobs, max_entries, max_pending_entries, max_cache_bytes = [], [], 0, 0, 0, 0
    for _, row in frame.iterrows():
        _synchronize(device)
        total = t = time.perf_counter()
        submit_ns = int(pd.Timestamp(row.submit_time).value)
        activated = history.activate_before(submit_ns)
        times = {name: 0. for name in STAGES}
        times["history_activation_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        features = history.query(row)  # includes exact script hashing
        identity, observed = str(features.frame.iloc[0].script_sha256), bool(features.frame.iloc[0].script_observed)
        times["history_features_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        routes = [route_frozen(models[r], features, i, args.operation) for i, r in enumerate(("memory", "vram"))]
        required = any(r["applicable"] and not r["bypass"] for r in routes)
        times["pre_gate_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        cache.advance(submit_ns)
        times["cache_commit_ms"] = (time.perf_counter() - t) * 1000
        cache_hit, encoder_forward = False, False
        qwen, gates, user_known = np.zeros(2), np.zeros(3), False
        if required:
            t = time.perf_counter()
            cached = cache.get(identity, observed) if "code" in model.enabled_modalities else None
            cache_hit = cached is not None
            times["cache_lookup_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            qwen, gates, user_known, created, encoder_forward, inner = live.predict(row, cached, retain_embedding=cache.capacity > 0 and observed)
            times["model_path_ms"] = (time.perf_counter() - t) * 1000
            times.update(inner)
            t = time.perf_counter()
            if created is not None:
                cache.put_after_group(identity, created, observed)
            times["cache_store_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        predictions = []
        for i, r in enumerate(("memory", "vram")):
            route = routes[i]
            if not route["applicable"]:
                predictions.append(float("nan"))
            elif route["bypass"]:
                predictions.append(float(np.clip(route["history_prediction"], 0., 1.)))
            else:
                predictions.append(fuse_frozen(models[r], features, i, qwen, gates, user_known, args.operation))
        times["post_fusion_ms"] = (time.perf_counter() - t) * 1000
        times["total_ms"] = (time.perf_counter() - total) * 1000
        # Accounting/report construction are excluded from per-request latency.
        max_entries, max_pending_entries = max(max_entries, len(cache.values)), max(max_pending_entries, len(cache.pending))
        max_cache_bytes = max(max_cache_bytes, sum(v.nelement() * v.element_size() for v in list(cache.values.values()) + list(cache.pending.values())))
        events.append({"sample_id": str(row.sample_id), "submit_time": row.submit_time,
            "identity": identity, "observed": observed, "model_path_required": required,
            "cache_hit": cache_hit, "encoder_forward": encoder_forward})
        if str(row.split) != "test":
            prefix_jobs += 1
            continue
        records.append({"sample_id": str(row.sample_id), "submit_time": row.submit_time, **times,
            "activated_outcomes": activated, "model_path_required": required, "cache_hit": cache_hit,
            "encoder_forward": encoder_forward, "all_applicable_resources_bypassed": not required,
            "qwen_memory": float(qwen[0]) if required else np.nan,
            "qwen_vram": float(qwen[1]) if required else np.nan,
            "gate_code": float(gates[0]) if required else np.nan,
            "gate_user": float(gates[1]) if required else np.nan,
            "gate_metadata": float(gates[2]) if required else np.nan,
            "user_known": user_known,
            **{f"prediction_{r}": predictions[i] for i, r in enumerate(("memory", "vram"))},
            **{f"bypass_{r}": routes[i]["bypass"] for i, r in enumerate(("memory", "vram"))},
            **{f"applicable_{r}": routes[i]["applicable"] for i, r in enumerate(("memory", "vram"))}})
    cache.flush()
    measured = pd.DataFrame(records)
    if measured.empty:
        raise ValueError("No measured test requests")
    # Persist measurements and outlier details BEFORE any check can reject them.
    measured.to_csv(output / "request_timings.csv", index=False)
    details = comparison_details(measured, reference, archived_expected)
    details.to_csv(output / "prediction_comparison.csv", index=False)
    checks = {"archived_structural_parity": original_structure}
    try:
        checks["base_batch_one_parity"] = base_parity(measured, reference_base)
        parity = parity_check(measured, reference, args.parity_atol)
        parity["reference_mode"] = "independent batch-one uncached model.forward plus canonical offline cascade"
        checks["offline_parity"] = parity
        same_inputs = inject_live_inputs(reference_base, measured)
        same_input_expected = frozen_reference(canonical_predictions(archive, same_inputs, bundle, canonical_models, **reference_options), args.operation)
        same_input_parity = parity_check(measured, same_input_expected, 1e-10)
        same_input_parity["reference_mode"] = "canonical offline history and cascade with recorded live Qwen inputs"
        checks["live_input_structural_parity"] = same_input_parity
        cache_parity = cache_parity_check(pd.DataFrame(events), reference, args.cache_capacity) if "code" in model.enabled_modalities else {"performed": False, "reason": "No-Code has no encoder cache"}
        checks["cache_parity"] = cache_parity
    except ValueError as error:
        (output / "parity_failure.json").write_text(json.dumps({"passed": False, "error": str(error), "completed_checks": checks,
            "diagnostics": ["request_timings.csv", "prediction_comparison.csv"],
            "archived_execution_comparison": drift_summary(details)}, indent=2, allow_nan=False), encoding="utf-8")
        raise
    settings = {"operation": args.operation, "seed": int(manifest["seed"]), "repeat": args.repeat,
        "cache_capacity_entries": args.cache_capacity, "cache_start": args.cache_start,
        "warmup_jobs": args.warmup_jobs, "num_threads": args.num_threads, "device": args.device,
        "parity_atol": args.parity_atol, "max_test_jobs": args.max_test_jobs}
    report = {"schema": "reliability-cascade-live-path/v3", "settings": settings, **settings,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": manifest.get("variant"), "data_sha256": bundle.data_sha256,
        "split_fingerprint": split_hash, "source_contract": benchmark_source,
        "checkpoint_namespace_sha256": namespace, "checkpoint_files_sha256": checkpoint_hashes,
        "base_run_manifest_sha256": sha256(base_manifest_path),
        "cascade_run_manifest_sha256": sha256(cascade / "run_manifest.json"),
        "cascade_models_sha256": sha256(cascade / "checkpoint/cascade_models.joblib"),
        "offline_reference_sha256": sha256(reference_path), "frozen_resource_policies": original_policies,
        "batch_one_reference_dir": str(reference_dir), "batch_one_reference": reference_metadata,
        "archived_execution_comparison": drift_summary(details),
        "prediction_comparison_sha256": sha256(output / "prediction_comparison.csv"),
        "history_settings": {"history_k": history_k, "decay_half_life_hours": half_life},
        "cache_same_timestamp_policy": "delayed insertion; last-encounter order; no same-group reuse",
        "cache_evictions_total_including_prefix": cache.evictions,
        "cache_final_entries": len(cache.values), "max_resident_cache_entries": max_entries,
        "max_pending_same_timestamp_entries": max_pending_entries,
        "max_cached_tensor_bytes_including_pending": max_cache_bytes,
        "cache_storage": "CPU projected code embeddings; user/request heads recomputed; capacity bounds committed entries, pending timestamp group reported separately",
        "history_initialization_seconds_excluded": initialization_seconds,
        "warmup_model_calls_excluded": args.warmup_jobs, "prefix_jobs_processed_not_measured": prefix_jobs,
        "complete_test_period": args.max_test_jobs is None and len(measured) == expected_test_n,
        "expected_test_jobs": expected_test_n, "cuda_synchronized": device.type == "cuda",
        "torch_num_threads": torch.get_num_threads(),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "measurement_scope": "serial in-process prediction from standardized/redacted job object; incremental label activation, script hashing, history, gates, bounded LRU, numeric/user preparation, tokenization, actual conditional encoder, heads and post-fusion",
        "excluded_scope": ["CSV import; raw Slurm parsing and script redaction", "model/tokenizer/process startup", "independent batch-one reference generation and offline replays", "initial historical-state preload", "report construction and parity checks", "RPC; concurrency and queueing; failure handling", "scheduler placement or utilization"],
        "stage_accounting": "numeric_user_ms, tokenization_ms, code_path_ms and fusion_heads_ms are nested in model_path_ms; do not add both levels. GPU stage boundaries synchronize and include synchronization overhead.",
        "interpretation": "Frozen controls isolate operations without refitting or test selection. Validation prefix uses the final frozen policy retrospectively. This is not a deployed scheduler experiment.",
        "all_requests": latency_summary(measured),
        "by_route": {name: latency_summary(measured.loc[mask]) for name, mask in {
            "history_bypass": ~measured.model_path_required,
            "exact_cache_hit": measured.cache_hit,
            "encoder_miss": measured.encoder_forward,
            "no_code_model_path": measured.model_path_required & ~measured.cache_hit & ~measured.encoder_forward,
        }.items()},
        "live_accuracy": _accuracy(measured, reference), "archived_accuracy": _accuracy(measured.assign(**{
            f"prediction_{r}": measured.sample_id.map(archived_expected.loc[archived_expected.resource.eq(r)].set_index("sample_id").prediction)
            for r in ("memory", "vram")}), archived_expected),
        **checks, **runtime_environment}
    report["request_timings_sha256"] = sha256(output / "request_timings.csv")
    (output / "online_benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"settings": settings, "all_requests": report["all_requests"], "offline_parity": parity}, indent=2))
    print(f"Completed live-path benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
