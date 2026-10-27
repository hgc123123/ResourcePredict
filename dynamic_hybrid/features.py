from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from common.data import DatasetBundle
from common.history import build_causal_history_features
from qwen.dataset import script_sha256


# Features are constructed explicitly below. Never infer gate inputs from all
# numeric prediction columns because those artifacts also contain targets/errors.
BASE_FEATURE_WHITELIST = (
    "qwen_prediction",
    "last1_prediction",
    "prediction_delta",
    "prediction_abs_delta",
    "prediction_mean",
    "history_available",
    "history_age_log_hours",
    "history_age_missing",
    "history_source_features_available",
    "same_script_as_last1",
    "history_count_log1p",
    "history_last_k_count",
    "history_last_k_mean",
    "history_last_k_std",
    "history_last_k_median",
    "history_ema",
    "history_same_script_available",
    "history_same_script_last_value",
    "history_same_script_age_log_hours",
    "history_same_script_age_missing",
    "qwen_gate_code",
    "qwen_gate_user",
    "qwen_gate_metadata",
    "qwen_gate_entropy",
    "qwen_user_known",
)

FORBIDDEN_TOKENS = (
    "target",
    "error",
    "label",
    "completion",
    "elapsed",
    "observed",
    "capacity_max",
    "peak_absolute",
)


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _numeric_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    default: float,
) -> np.ndarray:
    name = _first_column(frame, candidates)
    if name is None:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(float)


def _bool_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    default: bool,
) -> np.ndarray:
    name = _first_column(frame, candidates)
    if name is None:
        return np.full(len(frame), default, dtype=bool)
    values = frame[name]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].unique().tolist())[:5]
        raise ValueError(f"Invalid boolean values in {name}: {examples}")
    return normalized.map(mapping).to_numpy(bool)


def _context_frame(bundle: DatasetBundle) -> pd.DataFrame:
    retained = getattr(bundle, "retained_frame", None)
    sources = [bundle.frame]
    if retained is not None and not retained.empty:
        sources.append(retained)
    context = pd.concat(sources, ignore_index=True)
    context["sample_id"] = context["sample_id"].astype(str)
    context = context.drop_duplicates("sample_id", keep="first").copy()
    context["derived_script_sha256"] = [script_sha256(value) for value in context["code"]]
    columns = [
        "sample_id",
        "submit_time",
        "derived_script_sha256",
        *bundle.numeric_features,
    ]
    return context[columns]


@dataclass
class GateFeatures:
    paired: pd.DataFrame
    values: np.ndarray
    names: List[str]

    def subset(self, mask: np.ndarray) -> "GateFeatures":
        return GateFeatures(
            paired=self.paired.loc[mask].reset_index(drop=True),
            values=self.values[mask],
            names=list(self.names),
        )


def build_gate_features(
    paired: pd.DataFrame,
    bundle: DatasetBundle,
    *,
    history_k: int = 5,
    history_ema_alpha: float = 0.5,
) -> GateFeatures:
    context = _context_frame(bundle)
    causal_history = build_causal_history_features(
        bundle, k=history_k, ema_alpha=history_ema_alpha
    )
    current = paired.merge(
        context,
        on="sample_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if current["submit_time"].isna().any():
        raise ValueError("Some paired samples are absent from the configured dataset")
    current = current.merge(
        causal_history,
        on="sample_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    qwen = current["qwen_prediction"].to_numpy(float)
    history = current["history_available"].to_numpy(bool)
    last1_raw = current["last1_prediction"].to_numpy(float)
    # A neutral finite value prevents missing-history artifacts from entering the
    # MLP; the model separately forces the effective Last-1 gate to zero.
    last1 = np.where(history, last1_raw, qwen)
    resource_names = current["resource"].astype(str).to_numpy()

    def history_values(suffix: str, *, default: float = np.nan) -> np.ndarray:
        output = np.full(len(current), default, dtype=np.float64)
        for resource in ("memory", "vram"):
            mask = resource_names == resource
            output[mask] = pd.to_numeric(
                current.loc[mask, f"{resource}_history_{suffix}"], errors="coerce"
            ).to_numpy(float)
        return output

    causal_available = history_values("available", default=0.0).astype(bool)
    if not np.array_equal(history, causal_available):
        raise ValueError(
            "Last-1 history_available disagrees with shared causal history replay"
        )
    causal_last = history_values("last_value")
    mismatch = history & ~np.isclose(last1_raw, causal_last, rtol=0.0, atol=1e-12)
    if mismatch.any():
        raise ValueError(
            "Last-1 values disagree with shared causal history replay for "
            f"{int(mismatch.sum())} rows"
        )

    source_id_column = _first_column(
        current,
        ("history_source_sample_id", "history_source_sample_id_last1"),
    )
    if source_id_column is None:
        source_ids = pd.Series("", index=current.index, dtype="string")
    else:
        source_ids = current[source_id_column].astype("string").fillna("")
    causal_source_ids = np.full(len(current), "", dtype=object)
    for resource in ("memory", "vram"):
        mask = resource_names == resource
        causal_source_ids[mask] = (
            current.loc[mask, f"{resource}_history_source_sample_id"]
            .astype(str)
            .to_numpy()
        )
    source_mismatch = history & (source_ids.astype(str).to_numpy() != causal_source_ids)
    if source_mismatch.any():
        raise ValueError(
            "Last-1 history_source_sample_id disagrees with shared causal history replay "
            f"for {int(source_mismatch.sum())} rows"
        )
    source_context = context.add_prefix("history__")
    source_context = source_context.rename(
        columns={"history__sample_id": "history_source_sample_id_join"}
    )
    current["history_source_sample_id_join"] = source_ids.astype(str)
    current = current.merge(
        source_context,
        on="history_source_sample_id_join",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    source_time_column = _first_column(
        current,
        ("history_source_time", "history_source_time_last1"),
    )
    if source_time_column is None:
        source_time = pd.Series(pd.NaT, index=current.index)
    else:
        source_time = pd.to_datetime(current[source_time_column], errors="coerce", utc=True)
    submit_time = pd.to_datetime(current["submit_time"], errors="coerce", utc=True)
    age_hours = (submit_time - source_time).dt.total_seconds().to_numpy(float) / 3600.0
    invalid_age = history & (~np.isfinite(age_hours) | (age_hours <= 0.0))
    if invalid_age.any():
        raise ValueError(
            f"Invalid causal history timestamps for {int(invalid_age.sum())} paired rows"
        )
    causal_age_seconds = history_values("age_seconds")
    age_mismatch = history & ~np.isclose(
        age_hours * 3600.0,
        causal_age_seconds,
        rtol=0.0,
        atol=1e-6,
    )
    if age_mismatch.any():
        raise ValueError(
            "Last-1 history_source_time disagrees with shared causal history replay "
            f"for {int(age_mismatch.sum())} rows"
        )
    age_missing = ~history | ~np.isfinite(age_hours)
    age_log = np.where(age_missing, 0.0, np.log1p(np.maximum(age_hours, 0.0)))

    history_count = history_values("count", default=0.0)
    history_last_k_count = history_values("last_k_count", default=0.0)
    history_last_k_mean = history_values("last_k_mean")
    history_last_k_std = history_values("last_k_std")
    history_last_k_median = history_values("last_k_median")
    history_ema = history_values("ema")
    history_same_script_available = history_values(
        "same_script_available", default=0.0
    ).astype(bool)
    history_same_script_last = history_values("same_script_last_value")
    history_same_script_age_seconds = history_values("same_script_age_seconds")
    history_same_script_age_missing = (
        ~history_same_script_available | ~np.isfinite(history_same_script_age_seconds)
    )
    history_same_script_age_log = np.where(
        history_same_script_age_missing,
        0.0,
        np.log1p(np.maximum(history_same_script_age_seconds, 0.0) / 3600.0),
    )

    source_available = (
        history
        & current["history__derived_script_sha256"].notna().to_numpy(bool)
    )
    if np.any(history & ~source_available):
        raise ValueError(
            "Available Last-1 history has an empty or unknown history_source_sample_id"
        )
    same_script = (
        source_available
        & (
            current["derived_script_sha256"].astype(str).to_numpy()
            == current["history__derived_script_sha256"].astype(str).to_numpy()
        )
    )

    qwen_hash_column = _first_column(current, ("script_sha256", "script_sha256_qwen"))
    if qwen_hash_column is not None:
        exported_hash = current[qwen_hash_column].astype(str).to_numpy()
        derived_hash = current["derived_script_sha256"].astype(str).to_numpy()
        mismatch = exported_hash != derived_hash
        if mismatch.any():
            raise ValueError(
                "Qwen script hashes disagree with the configured/redacted dataset for "
                f"{int(mismatch.sum())} rows"
            )

    gates = np.column_stack(
        [
            _numeric_column(current, ("gate_code", "gate_code_qwen"), default=0.0),
            _numeric_column(current, ("gate_user", "gate_user_qwen"), default=0.0),
            _numeric_column(
                current, ("gate_metadata", "gate_metadata_qwen"), default=0.0
            ),
        ]
    )
    positive_gates = np.clip(gates, 0.0, 1.0)
    gate_entropy = -np.sum(
        np.where(positive_gates > 0.0, positive_gates * np.log(positive_gates + 1e-12), 0.0),
        axis=1,
    )
    user_known = _bool_column(
        current, ("user_known", "user_known_qwen"), default=False
    ).astype(float)

    columns: Dict[str, np.ndarray] = {
        "qwen_prediction": qwen,
        "last1_prediction": last1,
        "prediction_delta": last1 - qwen,
        "prediction_abs_delta": np.abs(last1 - qwen),
        "prediction_mean": 0.5 * (last1 + qwen),
        "history_available": history.astype(float),
        "history_age_log_hours": age_log,
        "history_age_missing": age_missing.astype(float),
        "history_source_features_available": source_available.astype(float),
        "same_script_as_last1": same_script.astype(float),
        "history_count_log1p": np.log1p(np.maximum(history_count, 0.0)),
        "history_last_k_count": history_last_k_count,
        "history_last_k_mean": np.where(
            history, history_last_k_mean, qwen
        ),
        "history_last_k_std": np.where(
            history, history_last_k_std, 0.0
        ),
        "history_last_k_median": np.where(
            history, history_last_k_median, qwen
        ),
        "history_ema": np.where(history, history_ema, qwen),
        "history_same_script_available": history_same_script_available.astype(float),
        "history_same_script_last_value": np.where(
            history_same_script_available, history_same_script_last, qwen
        ),
        "history_same_script_age_log_hours": history_same_script_age_log,
        "history_same_script_age_missing": history_same_script_age_missing.astype(float),
        "qwen_gate_code": gates[:, 0],
        "qwen_gate_user": gates[:, 1],
        "qwen_gate_metadata": gates[:, 2],
        "qwen_gate_entropy": gate_entropy,
        "qwen_user_known": user_known,
    }

    for metadata_name in bundle.numeric_features:
        if not metadata_name.startswith("num__"):
            raise ValueError(f"Unexpected non-numeric metadata feature: {metadata_name}")
        current_values = pd.to_numeric(current[metadata_name], errors="coerce").to_numpy(float)
        source_values = pd.to_numeric(
            current[f"history__{metadata_name}"], errors="coerce"
        ).to_numpy(float)
        short_name = metadata_name.removeprefix("num__")
        columns[f"metadata_current__{short_name}"] = current_values
        columns[f"metadata_delta__{short_name}"] = np.where(
            source_available, current_values - source_values, 0.0
        )

    names = list(columns)
    if names[: len(BASE_FEATURE_WHITELIST)] != list(BASE_FEATURE_WHITELIST):
        raise AssertionError("Base gate feature whitelist order changed unexpectedly")
    forbidden = [name for name in names if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"Forbidden post-outcome gate features: {forbidden}")
    values = np.column_stack([np.asarray(columns[name], dtype=np.float64) for name in names])
    return GateFeatures(paired=current, values=values, names=names)


@dataclass
class RobustFeatureScaler:
    names: List[str]
    median: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, names: Sequence[str]) -> "RobustFeatureScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Gate feature matrix/schema mismatch")
        finite_values = np.where(np.isfinite(values), values, np.nan)
        median = np.nanmedian(finite_values, axis=0)
        if not np.isfinite(median).all():
            raise ValueError("At least one gate feature is entirely non-finite")
        filled = np.where(np.isfinite(values), values, median)
        q25 = np.quantile(filled, 0.25, axis=0)
        q75 = np.quantile(filled, 0.75, axis=0)
        scale = q75 - q25
        scale = np.where(scale > 1e-8, scale, 1.0)
        lower = np.quantile(filled, 0.01, axis=0)
        upper = np.quantile(filled, 0.99, axis=0)
        # Quantile clipping must not erase rare binary events such as cold starts
        # or same-script history when their positive rate is below one percent.
        binary = np.all(
            np.isclose(filled, 0.0, rtol=0.0, atol=1e-12)
            | np.isclose(filled, 1.0, rtol=0.0, atol=1e-12),
            axis=0,
        )
        lower = np.where(binary, 0.0, lower)
        upper = np.where(binary, 1.0, upper)
        scale = np.where(binary, 1.0, scale)
        return cls(list(names), median, scale, lower, upper)

    def transform(self, values: np.ndarray, names: Sequence[str]) -> np.ndarray:
        if list(names) != self.names:
            raise ValueError("Gate feature schema/order differs from the fitted scaler")
        values = np.asarray(values, dtype=np.float64)
        filled = np.where(np.isfinite(values), values, self.median)
        clipped = np.clip(filled, self.lower, self.upper)
        transformed = (clipped - self.median) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Scaled gate features contain non-finite values")
        return transformed.astype(np.float32)

    def save(self, path: Path) -> None:
        payload = {
            "names": self.names,
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, path: Path) -> "RobustFeatureScaler":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            names=[str(value) for value in payload["names"]],
            median=np.asarray(payload["median"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            lower=np.asarray(payload["lower"], dtype=np.float64),
            upper=np.asarray(payload["upper"], dtype=np.float64),
        )
