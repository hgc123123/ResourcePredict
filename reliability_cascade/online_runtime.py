"""Incremental, event-time serving primitives shared by the measured benchmark.

No saved Qwen predictions are read by this module. Outcome labels are exposed
to history only after their recorded completion strictly precedes submission.
The trace is an event source, not an online scheduler or a queueing simulator.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from common.data import DatasetBundle
from common.history import RESOURCE_SPECS, script_sha256, validate_causal_history_bundle
from reliability_cascade.features import BASE_PRE_FEATURES, CascadeFeatures, post_features
from reliability_cascade.history import (
    UserResourceState,
    _activate,
    _prefix,
    _script_observed,
    _window_summary,
    history_candidate,
)


class ExactLRU:
    """Bounded LRU with delayed insertion for same-submission-time requests.

    Values are script-branch tensors, never complete job predictions. A value is
    added only after a real encoder call. Capacity zero disables exact reuse.
    A namespace tied to the frozen encoder/tokenizer contract prevents reuse
    across versions; one benchmark process owns one namespace.
    """

    def __init__(self, capacity: int, namespace: str):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError("Cache capacity must be a non-negative integer")
        if not namespace:
            raise ValueError("Cache namespace cannot be empty")
        self.capacity = int(capacity)
        self.namespace = str(namespace)
        self.values: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.pending: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.timestamp: int | None = None
        self.evictions = 0

    def advance(self, submit_ns: int) -> None:
        if self.timestamp is not None and submit_ns < self.timestamp:
            raise ValueError("Cache requests must follow submission order")
        if self.timestamp is not None and submit_ns != self.timestamp:
            self.flush()
        self.timestamp = int(submit_ns)

    def get(self, identity: str, observed: bool = True) -> Any | None:
        key = (self.namespace, str(identity))
        if not observed or self.capacity == 0 or key not in self.values:
            return None
        value = self.values.pop(key)
        self.values[key] = value
        return value

    def put_after_group(self, identity: str, value: Any, observed: bool = True) -> None:
        if observed and self.capacity:
            self.pending[(self.namespace, str(identity))] = value

    def flush(self) -> None:
        for key, value in self.pending.items():
            self.values.pop(key, None)
            self.values[key] = value
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
                self.evictions += 1
        self.pending.clear()


class OnlineHistory:
    def __init__(
        self, bundle: DatasetBundle, *, history_k: int = 5, half_life_hours: float = 168.0
    ):
        validate_causal_history_bundle(bundle)
        if history_k < 5 or not np.isfinite(half_life_hours) or half_life_hours <= 0:
            raise ValueError("History requires k >= 5 and a positive finite half-life")
        self.numeric_features = list(bundle.numeric_features)
        self.history_k = int(history_k)
        self.half_life_hours = float(half_life_hours)
        source = bundle.retained_frame
        self.events = (bundle.frame if source is None else source).reset_index(drop=True)
        self.by_id = {
            str(row["sample_id"]): row for _, row in self.events.iterrows()
        }
        self.histories: dict[str, dict[str, UserResourceState]] = {
            resource: {} for resource in RESOURCE_SPECS
        }
        self.heap = [
            (int(pd.Timestamp(row["completion_time"]).value), int(row["source_row"]), i)
            for i, row in self.events.iterrows()
        ]
        heapq.heapify(self.heap)
        self.last_submit_ns: int | None = None
        self.activated_count = 0

    def activate_before(self, submit_ns: int) -> int:
        if self.last_submit_ns is not None and submit_ns < self.last_submit_ns:
            raise ValueError("History queries must follow submission order")
        self.last_submit_ns = int(submit_ns)
        count = 0
        while self.heap and self.heap[0][0] < submit_ns:
            completion_ns, _, index = heapq.heappop(self.heap)
            _activate(
                self.events.iloc[index], completion_ns, self.histories,
                history_k=self.history_k, code_column="code",
            )
            count += 1
        self.activated_count += count
        return count

    def query(self, row: pd.Series) -> CascadeFeatures:
        submit_ns = int(pd.Timestamp(row["submit_time"]).value)
        if self.last_submit_ns != submit_ns:
            raise RuntimeError("Call activate_before for the current submission first")
        user_id = str(row["user_id"])
        observed = _script_observed(row["code"])
        identity = script_sha256(row["code"])
        outputs = []
        pre_rows = []
        expected_names = None
        for resource in RESOURCE_SPECS:
            state = self.histories[resource].get(user_id)
            general = _window_summary(
                None if state is None else state.general, submit_ns,
                half_life_hours=self.half_life_hours,
            )
            exact = _window_summary(
                None if state is None or not observed else state.scripts.get(identity),
                submit_ns, half_life_hours=self.half_life_hours,
            )
            source_id = exact["source_sample_id"] if exact["available"] else general["source_sample_id"]
            source = self.by_id.get(str(source_id))
            same_last1 = bool(exact["available"] and exact["source_sample_id"] == general["source_sample_id"])
            output = {
                "sample_id": str(row["sample_id"]), "resource": resource,
                "user_id": user_id, "script_observed": observed,
                "script_sha256": identity,
                "resource_applicable": resource == "memory" or float(row["gpu_count"]) > 0,
                "qwen_prediction": 0.0, "same_last1": same_last1,
                **_prefix(general, "user"), **_prefix(exact, "exact"),
            }
            pre = {
                "history_available": float(general["available"]),
                "history_count_log1p": np.log1p(general["count"]),
                "history_recent_count": general["recent_count"],
                "history_age_log_hours": np.log1p(max(float(general["age_hours"]), 0)) if general["available"] else 0.0,
                "history_std5": general["std5"] if general["available"] else 0.0,
                "history_mad5": general["mad5"] if general["available"] else 0.0,
                "history_range5": general["range5"] if general["available"] else 0.0,
                "history_last_median_gap": abs(general["last_value"] - general["median3"]) if general["available"] else 0.0,
                "history_median_decay_gap": abs(general["median3"] - general["decayed_mean"]) if general["available"] else 0.0,
                "exact_available": float(exact["available"]),
                "exact_count_log1p": np.log1p(exact["count"]),
                "exact_recent_count": exact["recent_count"],
                "exact_age_log_hours": np.log1p(max(float(exact["age_hours"]), 0)) if exact["available"] else 0.0,
                "exact_std5": exact["std5"] if exact["available"] else 0.0,
                "exact_mad5": exact["mad5"] if exact["available"] else 0.0,
                "exact_range5": exact["range5"] if exact["available"] else 0.0,
                "exact_last_median_gap": abs(exact["last_value"] - exact["median3"]) if exact["available"] else 0.0,
                "exact_median_decay_gap": abs(exact["median3"] - exact["decayed_mean"]) if exact["available"] else 0.0,
                "same_last1": float(same_last1),
                "exact_user_gap": abs(exact["last_value"] - general["last_value"]) if exact["available"] and general["available"] else 0.0,
                "source_metadata_available": float(source is not None),
            }
            if tuple(pre) != tuple(BASE_PRE_FEATURES):
                raise AssertionError("Incremental pre-route feature order differs")
            for name in self.numeric_features:
                short = name.removeprefix("num__")
                current = float(row[name])
                previous = float(source[name]) if source is not None else np.nan
                pre[f"request_current__{short}"] = current
                pre[f"request_abs_delta__{short}"] = abs(current - previous) if np.isfinite(previous) else 0.0
            names = list(pre)
            if expected_names is not None and names != expected_names:
                raise AssertionError("Resource feature schemas disagree")
            expected_names = names
            outputs.append(output)
            pre_rows.append(list(pre.values()))
        return CascadeFeatures(
            frame=pd.DataFrame(outputs), pre_values=np.asarray(pre_rows, dtype=float),
            pre_names=expected_names or [], post_base_values=np.zeros((2, 5), dtype=float),
            post_base_names=["qwen_gate_code", "qwen_gate_user", "qwen_gate_metadata", "qwen_gate_entropy", "qwen_user_known"],
            history_contract={"strict_completion_relation": "completion_time < submit_time", "incremental": True},
        )


def pre_route(model: Any, features: CascadeFeatures, index: int) -> dict[str, Any]:
    if features.pre_names != model.pre_names:
        raise ValueError("Online pre-route schema differs from saved gate")
    row = features.frame.iloc[index]
    if str(row["resource"]) != model.resource:
        raise ValueError("Wrong resource gate selected")
    applicable = bool(row["resource_applicable"])
    eligible = applicable and bool(row["user_available"])
    if model.config.require_exact_script_for_bypass:
        eligible = eligible and bool(row["exact_available"])
    probability = 0.0
    if eligible:
        values = model.pre_imputer.transform(features.pre_values[index:index + 1])
        probability = float(model.pre_model.predict_proba(values)[0, 1])
    history = float(history_candidate(features.frame, model.candidate_name)[index])
    return {
        "applicable": applicable, "eligible": eligible,
        "probability": probability,
        "bypass": eligible and probability >= model.threshold,
        "history_prediction": history,
    }


def post_fuse(
    model: Any, features: CascadeFeatures, index: int, qwen_values: np.ndarray,
    gate_weights: np.ndarray, user_known: bool,
) -> float:
    """Exact saved post-gate/trust-region computation on a live model output."""
    qwen_values = np.asarray(qwen_values, dtype=float)
    gates = np.asarray(gate_weights, dtype=float)
    if qwen_values.shape != (2,) or gates.shape != (3,):
        raise ValueError("Expected two resource values and three fusion weights")
    if not np.isfinite(qwen_values).all() or not np.isfinite(gates).all():
        raise ValueError("Non-finite live code-model result")
    features.frame["qwen_prediction"] = qwen_values
    positive = np.clip(gates, 0.0, 1.0)
    entropy = -np.sum(np.where(positive > 0, positive * np.log(positive + 1e-12), 0))
    features.post_base_values[:] = [*gates, entropy, float(user_known)]
    values, names, history_values = post_features(features, model.candidate_name)
    if names != model.post_names:
        raise ValueError("Online post-route schema differs from saved gate")
    row = features.frame.iloc[index]
    qwen = float(qwen_values[index])
    if not bool(row["user_available"]):
        return float(np.clip(qwen, 0.0, 1.0))
    history = float(history_values[index])
    weight = float(np.clip(model.post_model.predict(model.post_imputer.transform(values[index:index + 1]))[0], 0.0, 1.0))
    result = weight * history + (1.0 - weight) * qwen
    if model.config.post_blend_enabled and model.config.trust_region_enabled and bool(row["exact_available"]):
        mad = float(row["exact_mad5"])
        radius = min(model.config.trust_region_max, model.config.trust_region_base + model.config.trust_region_mad_scale * max(mad if np.isfinite(mad) else 0.0, 0.0))
        result = float(np.clip(result, history - radius, history + radius))
    return float(np.clip(result, 0.0, 1.0))


__all__ = ["ExactLRU", "OnlineHistory", "pre_route", "post_fuse"]
