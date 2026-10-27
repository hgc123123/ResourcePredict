"""Dependency-light reporting and strict live/offline numerical checks."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


OPERATIONS = ("full", "no_bypass", "exact_bypass", "no_post", "no_trust")
STAGES = (
    "total_ms", "history_activation_ms", "history_features_ms", "pre_gate_ms",
    "cache_commit_ms", "cache_lookup_ms", "model_path_ms", "numeric_user_ms",
    "tokenization_ms", "code_path_ms", "fusion_heads_ms", "cache_store_ms",
    "post_fusion_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contract(root: Path) -> dict:
    paths = []
    for folder in ("common", "qwen", "reliability_cascade", "online_study"):
        paths.extend(sorted((root / folder).glob("*.py")))
    paths.extend(root / "scripts" / name for name in (
        "run_online_study.py", "summarize_online_study.py", "run_online_study.slurm",
    ))
    files = {path.relative_to(root).as_posix(): sha256(path) for path in paths if path.is_file()}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def strict_bool(series: pd.Series, name: str) -> np.ndarray:
    text = series.astype(str).str.lower()
    if series.isna().any() or not text.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"Invalid boolean field: {name}")
    return text.isin(("true", "1")).to_numpy()


def frozen_reference(frame: pd.DataFrame, operation: str) -> pd.DataFrame:
    """Reference values from sealed offline outputs, outside every timed region."""
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown frozen operation: {operation}")
    output = frame.copy()
    bypass = strict_bool(frame.history_bypass, "history_bypass").copy()
    history = strict_bool(frame.history_available, "history_available")
    exact = strict_bool(frame.exact_script_history_available, "exact_script_history_available")
    applicable = strict_bool(frame.resource_applicable, "resource_applicable")
    if operation == "no_bypass":
        bypass[:] = False
    elif operation == "exact_bypass":
        bypass = applicable & history & exact
    qwen = frame.qwen_prediction.to_numpy(float)
    h = np.where(history, frame.history_expert_prediction.to_numpy(float), qwen)
    post = frame.post_blend_prediction.to_numpy(float)
    if operation == "no_post":
        post = qwen
    elif operation == "no_trust":
        gamma = frame.post_history_weight.to_numpy(float)
        post = gamma * h + (1.0 - gamma) * qwen
    output["prediction"] = np.clip(np.where(bypass, h, post), 0.0, 1.0)
    output["history_bypass"] = bypass
    output["required_resource"] = applicable & ~bypass
    if not np.isfinite(output.loc[applicable, "prediction"]).all():
        raise ValueError("Non-finite applicable offline reference")
    return output


def parity_check(records: pd.DataFrame, reference: pd.DataFrame, atol: float) -> dict:
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("Parity tolerance must be finite and nonnegative")
    ref = reference.loc[reference.split.eq("test")].copy()
    keys = ["sample_id", "resource"]
    ref["sample_id"] = ref.sample_id.astype(str)
    if ref.duplicated(keys).any() or records.sample_id.astype(str).duplicated().any():
        raise ValueError("Duplicate parity keys")
    live = pd.concat([
        records[["sample_id", f"prediction_{r}", f"bypass_{r}", f"applicable_{r}"]].rename(
            columns={f"prediction_{r}": "live_prediction", f"bypass_{r}": "live_bypass",
                     f"applicable_{r}": "live_applicable"}).assign(resource=r)
        for r in ("memory", "vram")
    ], ignore_index=True)
    live["sample_id"] = live.sample_id.astype(str)
    # A debugging prefix can compare a subset; paper runs separately require
    # every test job, so missing measured keys are always fatal here.
    merged = live.merge(ref, on=keys, how="left", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Parity reference lacks measured jobs/resources")
    applicable = strict_bool(merged.resource_applicable, "resource_applicable")
    live_applicable = strict_bool(merged.live_applicable, "live_applicable")
    if not np.array_equal(applicable, live_applicable):
        raise ValueError("Live/offline applicability differs")
    live_bypass = strict_bool(merged.live_bypass, "live_bypass")
    ref_bypass = strict_bool(merged.history_bypass, "history_bypass")
    route_mismatch = int(np.sum(live_bypass != ref_bypass))
    prediction = merged.live_prediction.to_numpy(float)
    expected = merged.prediction.to_numpy(float)
    if not np.isfinite(prediction[applicable]).all() or not np.isfinite(expected[applicable]).all():
        raise ValueError("Non-finite applicable live/offline prediction")
    difference = np.abs(prediction[applicable] - expected[applicable])
    maximum = float(difference.max()) if len(difference) else 0.0
    valid = strict_bool(merged.valid, "valid") if "valid" in merged else applicable
    result = {"performed": True, "rows": len(merged), "applicable_rows": int(applicable.sum()),
              "valid_evaluation_rows": int((valid & applicable).sum()),
              "applicable_without_label_rows": int((~valid & applicable).sum()),
              "route_mismatch_rows": route_mismatch, "max_prediction_abs_difference": maximum,
              "mean_prediction_abs_difference": float(difference.mean()) if len(difference) else 0.,
              "atol": atol, "passed": route_mismatch == 0 and maximum <= atol}
    if not result["passed"]:
        raise ValueError(f"Live/offline numerical parity failed: {result}")
    return result


def latency_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    result = {"n": len(frame)}
    for name in STAGES:
        values = frame[name].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid measured times: {name}")
        result[name] = dict(zip(("mean", "p50", "p95", "p99"), map(float,
            (values.mean(), *np.percentile(values, [50, 95, 99])))))
    seconds = float(frame.total_ms.sum() / 1000)
    required = int(frame.model_path_required.sum())
    result.update({
        "serial_service_throughput_jobs_per_second": len(frame) / seconds if seconds else None,
        "encoder_forward_n": int(frame.encoder_forward.sum()),
        "encoder_forward_rate": float(frame.encoder_forward.mean()),
        "cache_hit_n": int(frame.cache_hit.sum()),
        "cache_hit_rate_all_jobs": float(frame.cache_hit.mean()),
        "cache_hit_rate_required_jobs": float(frame.cache_hit.sum() / required) if required else None,
        "history_bypass_job_rate": float(frame.all_applicable_resources_bypassed.mean()),
        "model_path_required_n": required,
    })
    return result


def cache_parity_check(events: pd.DataFrame, reference: pd.DataFrame, capacity: int) -> dict:
    """Independent key-only offline replay; no GPU calls and no timing credit."""
    ref = reference.copy()
    ref["sample_id"] = ref.sample_id.astype(str)
    required = pd.Series(strict_bool(ref.required_resource, "required_resource"),
                         index=ref.sample_id).groupby(level=0).any()
    if events.sample_id.duplicated().any() or not set(events.sample_id).issubset(required.index):
        raise ValueError("Cache parity has duplicate or missing jobs")
    values, expected = OrderedDict(), []
    for _, group in events.groupby("submit_time", sort=False):
        pending = OrderedDict()
        for row in group.itertuples(index=False):
            needs = bool(required.loc[row.sample_id])
            hit = needs and row.observed and capacity > 0 and row.identity in values
            if hit:
                values.move_to_end(row.identity)
            if needs and not hit and row.observed and capacity > 0:
                pending.pop(row.identity, None)
                pending[row.identity] = True
            expected.append((needs, hit, needs and not hit))
        for key in pending:
            values.pop(key, None)
            values[key] = True
            while len(values) > capacity:
                values.popitem(last=False)
    actual = events[["model_path_required", "cache_hit", "encoder_forward"]].to_numpy(bool)
    wanted = np.asarray(expected, dtype=bool)
    mismatch = np.any(actual != wanted, axis=1)
    result = {"performed": True, "events_including_prefix": len(events),
              "mismatch_jobs": int(mismatch.sum()), "passed": not mismatch.any()}
    if not result["passed"]:
        raise ValueError(f"Live/offline cache parity failed: {result}")
    return result
