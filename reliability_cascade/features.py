from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from common.data import DatasetBundle
from common.metrics import coerce_valid_mask
from reliability_cascade.history import build_robust_history_long, history_candidate


FORBIDDEN_PRE_FEATURE_TOKENS = (
    "target",
    "label",
    "error",
    "qwen",
    "prediction",
    "completion",
    "elapsed",
    "observed",
    "peak",
)

BASE_PRE_FEATURES = (
    "history_available",
    "history_count_log1p",
    "history_recent_count",
    "history_age_log_hours",
    "history_std5",
    "history_mad5",
    "history_range5",
    "history_last_median_gap",
    "history_median_decay_gap",
    "exact_available",
    "exact_count_log1p",
    "exact_recent_count",
    "exact_age_log_hours",
    "exact_std5",
    "exact_mad5",
    "exact_range5",
    "exact_last_median_gap",
    "exact_median_decay_gap",
    "same_last1",
    "exact_user_gap",
    "source_metadata_available",
)

BASE_POST_FEATURES = (
    "qwen_value",
    "history_value",
    "expert_delta",
    "expert_abs_delta",
    "expert_mean",
    "qwen_gate_code",
    "qwen_gate_user",
    "qwen_gate_metadata",
    "qwen_gate_entropy",
    "qwen_user_known",
)


def _first(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _bool(frame: pd.DataFrame, names: Sequence[str], default: bool = False) -> np.ndarray:
    name = _first(frame, names)
    if name is None:
        return np.full(len(frame), default, dtype=bool)
    return coerce_valid_mask(frame[name], name)


def _numeric(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    default: float = np.nan,
) -> np.ndarray:
    name = _first(frame, names)
    if name is None:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _context(
    bundle: DatasetBundle,
    *,
    include_boundary_purged: bool = False,
) -> pd.DataFrame:
    retained = getattr(bundle, "retained_frame", None)
    source = retained if include_boundary_purged and retained is not None else bundle.frame
    frame = source.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    columns = [
        "sample_id",
        "job_id",
        "user_id",
        "submit_time",
        "completion_time",
        "gpu_count",
        *bundle.numeric_features,
    ]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Cascade context is missing dataset columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Cascade context contains duplicate sample IDs")
    return frame[columns]


@dataclass
class CascadeFeatures:
    frame: pd.DataFrame
    pre_values: np.ndarray
    pre_names: list[str]
    post_base_values: np.ndarray
    post_base_names: list[str]
    history_contract: Dict[str, object]

    def resource(self, resource: str) -> np.ndarray:
        return self.frame["resource"].astype(str).eq(resource).to_numpy()


def build_cascade_features(
    paired: pd.DataFrame,
    bundle: DatasetBundle,
    *,
    history_k: int = 5,
    half_life_hours: float = 168.0,
) -> CascadeFeatures:
    """Construct a target-free pre-route matrix and a separate post-Qwen matrix."""

    context = _context(bundle)
    source_context = _context(bundle, include_boundary_purged=True)
    history = build_robust_history_long(
        bundle,
        history_k=history_k,
        half_life_hours=half_life_hours,
    )
    history_contract = dict(history.attrs.get("causal_history_protocol") or {})
    frame = paired.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.merge(
        context,
        on="sample_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_dataset"),
    )
    frame = frame.merge(
        history,
        on=["sample_id", "resource"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_history"),
    )
    if frame["submit_time"].isna().any():
        raise ValueError("Component predictions contain samples absent from the dataset")

    last1_available = frame["history_available"].astype(bool).to_numpy()
    replay_available = frame["user_available"].astype(bool).to_numpy()
    if not np.array_equal(last1_available, replay_available):
        raise ValueError("Last-1 history availability disagrees with robust replay")
    replay_last = pd.to_numeric(frame["user_last_value"], errors="coerce").to_numpy(float)
    last1_value = pd.to_numeric(frame["last1_prediction"], errors="coerce").to_numpy(float)
    mismatch = last1_available & ~np.isclose(
        replay_last, last1_value, rtol=0.0, atol=1e-12
    )
    if mismatch.any():
        raise ValueError(
            "Last-1 predictions disagree with robust replay for "
            f"{int(mismatch.sum())} rows"
        )
    source_name = _first(
        frame,
        ("history_source_sample_id", "history_source_sample_id_last1"),
    )
    if source_name is not None:
        exported_source = frame[source_name].fillna("").astype(str).to_numpy()
        replay_source = frame["user_source_sample_id"].fillna("").astype(str).to_numpy()
        mismatch = last1_available & (exported_source != replay_source)
        if mismatch.any():
            raise ValueError(
                "Last-1 source IDs disagree with robust replay for "
                f"{int(mismatch.sum())} rows"
            )

    qwen_hash_name = _first(frame, ("script_sha256", "script_sha256_qwen"))
    if qwen_hash_name is not None:
        mismatch = (
            frame[qwen_hash_name].astype(str).str.lower().to_numpy()
            != frame["script_sha256_history"].astype(str).str.lower().to_numpy()
        )
        if mismatch.any():
            raise ValueError(
                "Qwen script identities disagree with causal replay for "
                f"{int(mismatch.sum())} rows"
            )

    source = source_context.add_prefix("source__").rename(
        columns={"source__sample_id": "preferred_source_sample_id"}
    )
    frame["preferred_source_sample_id"] = (
        frame["preferred_source_sample_id"].fillna("").astype(str)
    )
    frame = frame.merge(
        source,
        on="preferred_source_sample_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    gpu_count = pd.to_numeric(frame["gpu_count"], errors="coerce").to_numpy(float)
    if not np.isfinite(gpu_count).all() or np.any(gpu_count < 0.0):
        raise ValueError("gpu_count must be finite and non-negative for routing")
    resource = frame["resource"].astype(str).to_numpy()
    frame["resource_applicable"] = (resource == "memory") | (gpu_count > 0.0)
    valid = frame["valid"].astype(bool).to_numpy()
    if np.any(valid & (resource == "vram") & ~frame["resource_applicable"].to_numpy(bool)):
        raise ValueError("A valid VRAM label is attached to a CPU-only job")

    history_available = frame["user_available"].astype(bool).to_numpy()
    exact_available = frame["exact_available"].astype(bool).to_numpy()
    history_age = pd.to_numeric(frame["user_age_hours"], errors="coerce").to_numpy(float)
    exact_age = pd.to_numeric(frame["exact_age_hours"], errors="coerce").to_numpy(float)
    user_last = pd.to_numeric(frame["user_last_value"], errors="coerce").to_numpy(float)
    user_median = pd.to_numeric(frame["user_median3"], errors="coerce").to_numpy(float)
    user_decay = pd.to_numeric(frame["user_decayed_mean"], errors="coerce").to_numpy(float)
    exact_last = pd.to_numeric(frame["exact_last_value"], errors="coerce").to_numpy(float)
    exact_median = pd.to_numeric(frame["exact_median3"], errors="coerce").to_numpy(float)
    exact_decay = pd.to_numeric(frame["exact_decayed_mean"], errors="coerce").to_numpy(float)

    pre: Dict[str, np.ndarray] = {
        "history_available": history_available.astype(float),
        "history_count_log1p": np.log1p(
            pd.to_numeric(frame["user_count"], errors="coerce").fillna(0).to_numpy(float)
        ),
        "history_recent_count": pd.to_numeric(
            frame["user_recent_count"], errors="coerce"
        ).fillna(0).to_numpy(float),
        "history_age_log_hours": np.where(
            history_available, np.log1p(np.maximum(history_age, 0.0)), 0.0
        ),
        "history_std5": pd.to_numeric(
            frame["user_std5"], errors="coerce"
        ).fillna(0).to_numpy(float),
        "history_mad5": pd.to_numeric(
            frame["user_mad5"], errors="coerce"
        ).fillna(0).to_numpy(float),
        "history_range5": pd.to_numeric(
            frame["user_range5"], errors="coerce"
        ).fillna(0).to_numpy(float),
        "history_last_median_gap": np.where(
            history_available, np.abs(user_last - user_median), 0.0
        ),
        "history_median_decay_gap": np.where(
            history_available, np.abs(user_median - user_decay), 0.0
        ),
        "exact_available": exact_available.astype(float),
        "exact_count_log1p": np.log1p(
            pd.to_numeric(frame["exact_count"], errors="coerce").fillna(0).to_numpy(float)
        ),
        "exact_recent_count": pd.to_numeric(
            frame["exact_recent_count"], errors="coerce"
        ).fillna(0).to_numpy(float),
        "exact_age_log_hours": np.where(
            exact_available, np.log1p(np.maximum(exact_age, 0.0)), 0.0
        ),
        "exact_std5": pd.to_numeric(frame["exact_std5"], errors="coerce").fillna(0).to_numpy(float),
        "exact_mad5": pd.to_numeric(frame["exact_mad5"], errors="coerce").fillna(0).to_numpy(float),
        "exact_range5": pd.to_numeric(
            frame["exact_range5"], errors="coerce"
        ).fillna(0).to_numpy(float),
        "exact_last_median_gap": np.where(
            exact_available, np.abs(exact_last - exact_median), 0.0
        ),
        "exact_median_decay_gap": np.where(
            exact_available, np.abs(exact_median - exact_decay), 0.0
        ),
        "same_last1": frame["same_last1"].astype(bool).to_numpy(float),
        "exact_user_gap": np.where(
            exact_available & history_available, np.abs(exact_last - user_last), 0.0
        ),
        "source_metadata_available": frame["source__submit_time"].notna().to_numpy(float),
    }

    for feature_name in bundle.numeric_features:
        if not feature_name.startswith("num__"):
            raise ValueError(f"Unexpected numeric feature name: {feature_name}")
        short = feature_name.removeprefix("num__")
        current = pd.to_numeric(frame[feature_name], errors="coerce").to_numpy(float)
        previous = pd.to_numeric(
            frame[f"source__{feature_name}"], errors="coerce"
        ).to_numpy(float)
        available = np.isfinite(previous)
        pre[f"request_current__{short}"] = current
        pre[f"request_abs_delta__{short}"] = np.where(
            available, np.abs(current - previous), 0.0
        )

    pre_names = list(pre)
    if pre_names[: len(BASE_PRE_FEATURES)] != list(BASE_PRE_FEATURES):
        raise AssertionError("Base pre-route feature order changed")
    forbidden = [
        name
        for name in pre_names
        if any(token in name.lower() for token in FORBIDDEN_PRE_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Pre-route feature leakage detected: {forbidden}")
    pre_values = np.column_stack([np.asarray(pre[name], dtype=float) for name in pre_names])

    qwen = frame["qwen_prediction"].to_numpy(float)
    # Candidate-specific history values are appended by post_features().
    gates = np.column_stack(
        [
            _numeric(frame, ("gate_code", "gate_code_qwen"), default=0.0),
            _numeric(frame, ("gate_user", "gate_user_qwen"), default=0.0),
            _numeric(frame, ("gate_metadata", "gate_metadata_qwen"), default=0.0),
        ]
    )
    positive = np.clip(gates, 0.0, 1.0)
    entropy = -np.sum(
        np.where(positive > 0.0, positive * np.log(positive + 1e-12), 0.0), axis=1
    )
    post_base: Dict[str, np.ndarray] = {
        "qwen_gate_code": gates[:, 0],
        "qwen_gate_user": gates[:, 1],
        "qwen_gate_metadata": gates[:, 2],
        "qwen_gate_entropy": entropy,
        "qwen_user_known": _bool(
            frame, ("user_known", "user_known_qwen"), default=False
        ).astype(float),
    }
    post_names = list(post_base)
    post_values = np.column_stack(
        [np.asarray(post_base[name], dtype=float) for name in post_names]
    )
    if not np.isfinite(qwen).all():
        raise ValueError("Qwen predictions must be finite")
    return CascadeFeatures(
        frame=frame,
        pre_values=pre_values,
        pre_names=pre_names,
        post_base_values=post_values,
        post_base_names=post_names,
        history_contract=history_contract,
    )


def post_features(
    features: CascadeFeatures,
    candidate_name: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    history = history_candidate(features.frame, candidate_name)
    qwen = features.frame["qwen_prediction"].to_numpy(float)
    available = features.frame["user_available"].astype(bool).to_numpy()
    safe_history = np.where(available, history, qwen)
    candidate_columns = np.column_stack(
        [
            qwen,
            safe_history,
            safe_history - qwen,
            np.abs(safe_history - qwen),
            0.5 * (safe_history + qwen),
        ]
    )
    names = [
        "qwen_value",
        "history_value",
        "expert_delta",
        "expert_abs_delta",
        "expert_mean",
        *features.post_base_names,
    ]
    values = np.column_stack(
        [features.pre_values, candidate_columns, features.post_base_values]
    )
    full_names = [*features.pre_names, *names]
    return values, full_names, history


__all__ = [
    "BASE_PRE_FEATURES",
    "CascadeFeatures",
    "FORBIDDEN_PRE_FEATURE_TOKENS",
    "build_cascade_features",
    "post_features",
]
