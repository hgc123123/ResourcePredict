from __future__ import annotations

import hashlib
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from common.metrics import add_error_columns, compute_metrics
from reliability_cascade.history import history_candidate
from reliability_cascade.pipeline import PreparedCascade
from reliability_cascade.selection import SelectedCascade


def _first(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _simulate_embedding_cache(
    frame: pd.DataFrame,
    qwen_required_by_job: pd.Series,
) -> pd.DataFrame:
    memory = frame.loc[frame["resource"].astype(str).eq("memory")].copy()
    if memory["sample_id"].astype(str).duplicated().any():
        raise ValueError("Memory rows do not uniquely identify jobs for cache replay")
    hash_name = _first(memory, ("script_sha256_history", "script_sha256"))
    if hash_name is None:
        raise ValueError("Cache replay requires exact script SHA-256 identities")
    jobs = memory[
        ["sample_id", "split", "submit_time", "script_observed", hash_name]
    ].copy()
    jobs["sample_id"] = jobs["sample_id"].astype(str)
    jobs["submit_time"] = pd.to_datetime(jobs["submit_time"], errors="raise", utc=True)
    jobs["qwen_required_job"] = jobs["sample_id"].map(qwen_required_by_job).fillna(False)
    jobs = jobs.sort_values(["submit_time", "sample_id"], kind="mergesort")

    cache: set[str] = set()
    jobs["embedding_cache_hit_job"] = False
    jobs["encoder_forward_required_job"] = False
    # Jobs submitted at the same timestamp observe the same cache state. This
    # avoids crediting a concurrent job with an entry created by its peer.
    for _, group in jobs.groupby("submit_time", sort=False):
        pending: set[str] = set()
        for index, row in group.iterrows():
            required = bool(row["qwen_required_job"])
            observed = bool(row["script_observed"])
            identity = str(row[hash_name])
            hit = required and observed and identity in cache
            jobs.at[index, "embedding_cache_hit_job"] = hit
            jobs.at[index, "encoder_forward_required_job"] = required and not hit
            if required and observed and not hit:
                pending.add(identity)
        cache.update(pending)
    return jobs.sort_index(kind="mergesort").reset_index(drop=True)


def apply_selected_cascade(
    prepared: PreparedCascade,
    selected: SelectedCascade,
    *,
    model_name: str,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    frame = prepared.features.frame
    outputs = {
        resource: model.predict(prepared.features)
        for resource, model in selected.models.items()
    }
    size = len(frame)
    prediction = np.zeros(size, dtype=float)
    history_prediction = np.full(size, np.nan, dtype=float)
    probability = np.zeros(size, dtype=float)
    bypass = np.zeros(size, dtype=bool)
    required_resource = np.zeros(size, dtype=bool)
    gamma = np.zeros(size, dtype=float)
    post_prediction = np.zeros(size, dtype=float)
    trust_applied = np.zeros(size, dtype=bool)
    trust_radius = np.full(size, np.nan, dtype=float)
    selected_expert = np.full(size, "", dtype=object)
    selected_threshold = np.full(size, np.nan, dtype=float)
    for resource, output in outputs.items():
        mask = output["resource_mask"]
        prediction[mask] = output["prediction"][mask]
        history_prediction[mask] = output["history_prediction"][mask]
        probability[mask] = output["reliability_probability"][mask]
        bypass[mask] = output["history_bypass"][mask]
        required_resource[mask] = output["qwen_required_resource"][mask]
        gamma[mask] = output["post_gamma"][mask]
        post_prediction[mask] = output["post_prediction"][mask]
        trust_applied[mask] = output["trust_region_applied"][mask]
        trust_radius[mask] = output["trust_region_radius"][mask]
        selected_expert[mask] = selected.models[resource].candidate_name
        selected_threshold[mask] = selected.models[resource].threshold

    route = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "qwen_required_resource": required_resource,
        }
    )
    qwen_required_by_job = route.groupby("sample_id", sort=False)[
        "qwen_required_resource"
    ].any()
    qwen_required_job = (
        frame["sample_id"].astype(str).map(qwen_required_by_job).to_numpy(bool)
    )
    cache_jobs = _simulate_embedding_cache(frame, qwen_required_by_job)
    cache_hit_map = cache_jobs.set_index("sample_id")["embedding_cache_hit_job"]
    encoder_map = cache_jobs.set_index("sample_id")["encoder_forward_required_job"]
    cache_hit = frame["sample_id"].astype(str).map(cache_hit_map).to_numpy(bool)
    encoder_required = frame["sample_id"].astype(str).map(encoder_map).to_numpy(bool)

    history = frame["user_available"].astype(bool).to_numpy()
    exact = frame["exact_available"].astype(bool).to_numpy()
    applicable = frame["resource_applicable"].astype(bool).to_numpy()
    route_reason = np.full(size, "qwen_dynamic_blend", dtype=object)
    route_reason[~applicable] = "resource_not_applicable"
    route_reason[applicable & ~history] = "cold_start_qwen"
    route_reason[applicable & history & ~exact] = "no_exact_history_qwen_blend"
    route_reason[applicable & history & exact & ~bypass] = (
        "unreliable_exact_history_qwen_blend"
    )
    route_reason[bypass] = "reliable_history_bypass"

    qwen = frame["qwen_prediction"].to_numpy(float)
    hierarchical_last = history_candidate(frame, "hierarchical_last")
    hierarchical_reference = np.where(history, hierarchical_last, qwen)
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "job_id": frame["job_id"].astype(str).to_numpy(),
            "user_id": frame["user_id"].astype(str).to_numpy(),
            "split": frame["split"].astype(str).to_numpy(),
            "resource": frame["resource"].astype(str).to_numpy(),
            "target": frame["target"].to_numpy(float),
            "raw_prediction": prediction,
            "prediction": np.clip(prediction, 0.0, 1.0),
            "valid": frame["valid"].astype(bool).to_numpy(),
            "model": model_name,
            "seed": int(seed),
            "qwen_prediction": qwen,
            "last1_prediction": frame["last1_prediction"].to_numpy(float),
            "hierarchical_reference_prediction": np.clip(
                hierarchical_reference, 0.0, 1.0
            ),
            "history_expert_prediction": history_prediction,
            "history_available": history,
            "exact_script_history_available": exact,
            "same_last1": frame["same_last1"].astype(bool).to_numpy(),
            "selected_history_expert": selected_expert,
            "reliability_threshold": selected_threshold,
            "reliability_probability": probability,
            "history_bypass": bypass,
            "post_history_weight": gamma,
            "post_blend_prediction": post_prediction,
            "trust_region_applied": trust_applied,
            "trust_region_radius": trust_radius,
            "resource_applicable": applicable,
            "qwen_required_resource": required_resource,
            "qwen_required_job": qwen_required_job,
            "embedding_cache_hit_job": cache_hit,
            "encoder_forward_required_job": encoder_required,
            "route_reason": route_reason,
            "history_count": frame["user_count"].to_numpy(int),
            "history_age_hours": frame["user_age_hours"].to_numpy(float),
            "history_mad5": frame["user_mad5"].to_numpy(float),
            "exact_history_count": frame["exact_count"].to_numpy(int),
            "exact_history_age_hours": frame["exact_age_hours"].to_numpy(float),
            "exact_history_mad5": frame["exact_mad5"].to_numpy(float),
        }
    )
    for name in (
        "job_name",
        "partition",
        "submit_time",
        "start_time",
        "completion_time",
        "job_state",
        "exit_code",
        "gpu_count",
    ):
        column = _first(frame, (name, f"{name}_qwen", f"{name}_last1"))
        if column is not None:
            output[name] = frame[column].to_numpy()
    output = add_error_columns(output)
    metrics = compute_metrics(output)

    reference = output.copy()
    reference["raw_prediction"] = reference["hierarchical_reference_prediction"]
    reference["prediction"] = reference["hierarchical_reference_prediction"]
    reference["model"] = "hierarchical_same_script_last1_with_qwen_cold_start"
    reference = add_error_columns(reference)
    reference_metrics = compute_metrics(reference)

    routing = routing_metrics(output, cache_jobs)
    importance = pd.DataFrame(
        [
            row
            for resource in ("memory", "vram")
            for row in selected.models[resource].feature_importance()
        ]
    ).sort_values(
        ["resource", "stage", "importance", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    route_contract = output[
        [
            "sample_id",
            "resource",
            "history_bypass",
            "qwen_required_resource",
            "qwen_required_job",
            "encoder_forward_required_job",
        ]
    ].to_csv(index=False, lineterminator="\n")
    route_sha256 = hashlib.sha256(route_contract.encode("utf-8")).hexdigest()
    return output, metrics, reference_metrics, routing, importance, route_sha256


def routing_metrics(
    predictions: pd.DataFrame,
    cache_jobs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for split in ("validation", "test"):
        jobs = cache_jobs.loc[cache_jobs["split"].astype(str).eq(split)]
        qwen_n = int(jobs["qwen_required_job"].sum())
        cache_n = int(jobs["embedding_cache_hit_job"].sum())
        encoder_n = int(jobs["encoder_forward_required_job"].sum())
        rows.append(
            {
                "split": split,
                "scope": "job",
                "resource": "all",
                "n": int(len(jobs)),
                "history_bypass_n": int(len(jobs) - qwen_n),
                "history_bypass_rate": float(1.0 - qwen_n / len(jobs)),
                "qwen_required_n": qwen_n,
                "qwen_required_rate": float(qwen_n / len(jobs)),
                "embedding_cache_hit_n": cache_n,
                "embedding_cache_hit_rate_among_qwen": (
                    float(cache_n / qwen_n) if qwen_n else 0.0
                ),
                "encoder_forward_required_n": encoder_n,
                "encoder_forward_required_rate": float(encoder_n / len(jobs)),
            }
        )
        for resource in ("memory", "vram"):
            part = predictions.loc[
                predictions["split"].astype(str).eq(split)
                & predictions["resource"].astype(str).eq(resource)
                & predictions["resource_applicable"].astype(bool)
            ]
            bypass_n = int(part["history_bypass"].sum())
            required_n = int(part["qwen_required_resource"].sum())
            rows.append(
                {
                    "split": split,
                    "scope": "resource",
                    "resource": resource,
                    "n": int(len(part)),
                    "history_bypass_n": bypass_n,
                    "history_bypass_rate": float(bypass_n / len(part)),
                    "qwen_required_n": required_n,
                    "qwen_required_rate": float(required_n / len(part)),
                    "embedding_cache_hit_n": np.nan,
                    "embedding_cache_hit_rate_among_qwen": np.nan,
                    "encoder_forward_required_n": np.nan,
                    "encoder_forward_required_rate": np.nan,
                }
            )
    return pd.DataFrame(rows)


__all__ = ["apply_selected_cascade", "routing_metrics"]
