from __future__ import annotations

import hashlib
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .data import DatasetBundle


RESOURCE_SPECS: Mapping[str, Tuple[str, str]] = {
    "memory": ("target_memory", "valid_memory"),
    "vram": ("target_vram", "valid_vram"),
}
EMPTY_SCRIPT = "<EMPTY_SCRIPT>"
HISTORY_PROTOCOL_VERSION = "causal-history-features/v2"
SAME_SCRIPT_IDENTITY = "sha256(exact redacted script encoder input)"
EMPTY_SCRIPT_SAME_HISTORY_POLICY = "unavailable"


@dataclass(frozen=True)
class HistoryObservation:
    value: float
    completion_ns: int
    sample_id: str
    script_hash: str


@dataclass
class ResourceHistory:
    count: int = 0
    ema: float = float("nan")
    recent: Deque[HistoryObservation] = field(default_factory=deque)
    same_script: Dict[str, HistoryObservation] = field(default_factory=dict)


def _script_text(value: object) -> str:
    if value is None:
        return ""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    return str(value)


def script_sha256(value: object) -> str:
    """Hash the exact redacted script text used by the Qwen tokenizer."""
    # Keep hashing identical to qwen.dataset.script_encoder_input. Missing-value
    # handling affects match availability, not the recorded Qwen input hash.
    text = str(value)
    canonical = text if text.strip() else EMPTY_SCRIPT
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _has_observed_script(value: object) -> bool:
    """Treat missing/blank code as unavailable, not as a shared script identity."""
    return bool(_script_text(value).strip())


def causal_history_contract(k: int, ema_alpha: float) -> Dict[str, object]:
    """Return the checkpoint-relevant semantics of causal history features."""
    return {
        "protocol_version": HISTORY_PROTOCOL_VERSION,
        "strict_completion_relation": "completion_time < submit_time",
        "completion_tie_break": "higher source_row is activated later",
        "history_k": int(k),
        "history_ema_alpha": float(ema_alpha),
        "same_script_identity": SAME_SCRIPT_IDENTITY,
        "empty_script_same_history_policy": EMPTY_SCRIPT_SAME_HISTORY_POLICY,
    }


def _history_frame(bundle: DatasetBundle) -> pd.DataFrame:
    retained = getattr(bundle, "retained_frame", None)
    return bundle.frame if retained is None else retained


def validate_causal_history_bundle(
    bundle: DatasetBundle,
    *,
    code_column: str = "code",
) -> None:
    """Validate the same strict event-time contract used by causal Last-1."""
    if not isinstance(code_column, str) or not code_column:
        raise ValueError("code_column must be a non-empty string")
    effective = bundle.frame
    events = _history_frame(bundle)
    required = {
        "sample_id",
        "source_row",
        "user_id",
        "submit_time",
        "completion_time",
        code_column,
    }
    for target_column, valid_column in RESOURCE_SPECS.values():
        required.update((target_column, valid_column))

    if not bundle.protocol.get("verified_chronological", False):
        raise ValueError(
            "Causal history requires split.order_semantics=submission_time"
        )
    if not bundle.protocol.get("has_submission_time", False):
        raise ValueError("Causal history requires a complete submit_time column")
    if not bundle.protocol.get("has_completion_time", False):
        raise ValueError("Causal history requires a complete completion_time column")
    for name, frame in (("effective", effective), ("retained", events)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} causal history frame is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"The {name} causal history frame is empty")
        if frame["submit_time"].isna().any():
            raise ValueError(f"The {name} frame has missing submit_time values")
        if frame["completion_time"].isna().any():
            raise ValueError(f"The {name} frame has missing completion_time values")
        if not frame["submit_time"].is_monotonic_increasing:
            raise ValueError(f"The {name} frame is not ordered by submit_time")
        invalid = frame["completion_time"] < frame["submit_time"]
        if invalid.any():
            raise ValueError(
                f"Found {int(invalid.sum())} {name} rows whose completion precedes "
                "submission"
            )
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"The {name} frame contains duplicate sample_id values")
        if frame["source_row"].isna().any():
            raise ValueError(f"The {name} frame has missing source_row values")
        for resource, (target_column, valid_column) in RESOURCE_SPECS.items():
            if frame[valid_column].isna().any():
                raise ValueError(
                    f"The {name} frame has missing {valid_column} values"
                )
            valid = frame[valid_column].astype(bool).to_numpy()
            targets = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            invalid_targets = valid & ~np.isfinite(targets)
            if invalid_targets.any():
                raise ValueError(
                    f"The {name} frame marks {int(invalid_targets.sum())} {resource} "
                    "targets valid although they are non-finite"
                )

    effective_ids = set(effective["sample_id"].astype(str))
    event_ids = set(events["sample_id"].astype(str))
    missing_events = effective_ids.difference(event_ids)
    if missing_events:
        raise ValueError(
            "The retained causal history population is missing "
            f"{len(missing_events)} effective samples"
        )


def _empty_feature_arrays(size: int) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for resource in RESOURCE_SPECS:
        prefix = f"{resource}_history"
        arrays[f"{prefix}_available"] = np.zeros(size, dtype=bool)
        arrays[f"{prefix}_count"] = np.zeros(size, dtype=np.int64)
        arrays[f"{prefix}_last_value"] = np.full(size, np.nan, dtype=float)
        arrays[f"{prefix}_last_k_count"] = np.zeros(size, dtype=np.int64)
        arrays[f"{prefix}_last_k_mean"] = np.full(size, np.nan, dtype=float)
        arrays[f"{prefix}_last_k_std"] = np.full(size, np.nan, dtype=float)
        arrays[f"{prefix}_last_k_median"] = np.full(size, np.nan, dtype=float)
        arrays[f"{prefix}_ema"] = np.full(size, np.nan, dtype=float)
        arrays[f"{prefix}_age_seconds"] = np.full(size, np.nan, dtype=float)
        arrays[f"{prefix}_source_sample_id"] = np.full(size, "", dtype=object)
        arrays[f"{prefix}_same_script_available"] = np.zeros(size, dtype=bool)
        arrays[f"{prefix}_same_script_last_value"] = np.full(
            size, np.nan, dtype=float
        )
        arrays[f"{prefix}_same_script_age_seconds"] = np.full(
            size, np.nan, dtype=float
        )
        arrays[f"{prefix}_same_script_source_sample_id"] = np.full(
            size, "", dtype=object
        )
    return arrays


def _activate_event(
    row: pd.Series,
    completion_ns: int,
    histories: Dict[str, Dict[str, ResourceHistory]],
    *,
    k: int,
    ema_alpha: float,
    code_column: str,
) -> None:
    user = str(row["user_id"])
    script_observed = _has_observed_script(row[code_column])
    current_hash = script_sha256(row[code_column])
    for resource, (target_column, valid_column) in RESOURCE_SPECS.items():
        if not bool(row[valid_column]):
            continue
        value = float(row[target_column])
        if not np.isfinite(value):
            raise ValueError(
                f"Row {row['sample_id']} marks {resource} valid but has a non-finite target"
            )
        state = histories[resource].setdefault(user, ResourceHistory())
        observation = HistoryObservation(
            value=value,
            completion_ns=completion_ns,
            sample_id=str(row["sample_id"]),
            script_hash=current_hash,
        )
        state.count += 1
        state.ema = (
            value
            if state.count == 1
            else ema_alpha * value + (1.0 - ema_alpha) * state.ema
        )
        state.recent.append(observation)
        if len(state.recent) > k:
            state.recent.popleft()
        if script_observed:
            state.same_script[current_hash] = observation


def _record_features(
    row_index: int,
    row: pd.Series,
    submit_ns: int,
    histories: Dict[str, Dict[str, ResourceHistory]],
    arrays: Dict[str, np.ndarray],
    *,
    code_column: str,
) -> None:
    user = str(row["user_id"])
    script_observed = _has_observed_script(row[code_column])
    current_hash = script_sha256(row[code_column])
    for resource in RESOURCE_SPECS:
        state = histories[resource].get(user)
        if state is None or not state.recent:
            continue

        prefix = f"{resource}_history"
        last = state.recent[-1]
        values = np.fromiter(
            (observation.value for observation in state.recent), dtype=float
        )
        arrays[f"{prefix}_available"][row_index] = True
        arrays[f"{prefix}_count"][row_index] = state.count
        arrays[f"{prefix}_last_value"][row_index] = last.value
        arrays[f"{prefix}_last_k_count"][row_index] = len(values)
        arrays[f"{prefix}_last_k_mean"][row_index] = float(values.mean())
        # Population standard deviation is zero for one available observation.
        arrays[f"{prefix}_last_k_std"][row_index] = float(values.std(ddof=0))
        arrays[f"{prefix}_last_k_median"][row_index] = float(np.median(values))
        arrays[f"{prefix}_ema"][row_index] = state.ema
        arrays[f"{prefix}_age_seconds"][row_index] = (
            submit_ns - last.completion_ns
        ) / 1_000_000_000.0
        arrays[f"{prefix}_source_sample_id"][row_index] = last.sample_id

        if not script_observed:
            continue
        same_script = state.same_script.get(current_hash)
        if same_script is None:
            continue
        arrays[f"{prefix}_same_script_available"][row_index] = True
        arrays[f"{prefix}_same_script_last_value"][row_index] = same_script.value
        arrays[f"{prefix}_same_script_age_seconds"][row_index] = (
            submit_ns - same_script.completion_ns
        ) / 1_000_000_000.0
        arrays[f"{prefix}_same_script_source_sample_id"][row_index] = (
            same_script.sample_id
        )


def build_causal_history_features(
    bundle: DatasetBundle,
    *,
    k: int = 5,
    ema_alpha: float = 0.5,
    code_column: str = "code",
) -> pd.DataFrame:
    """Build per-user features from labels completed before each submission.

    The returned rows correspond only to ``bundle.frame``. When present,
    ``bundle.retained_frame`` supplies the event stream, so boundary-purged jobs
    become usable history after completion without becoming prediction rows.
    Last-k statistics use completion-event order, standard deviation uses
    ``ddof=0``, and EMA updates as ``alpha * new + (1-alpha) * previous``.
    Missing histories are represented by false availability, zero counts, NaN
    numeric values, and empty source IDs; no label-derived fallback is applied.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    try:
        alpha = float(ema_alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ema_alpha must be finite and in (0, 1]") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("ema_alpha must be finite and in (0, 1]")

    validate_causal_history_bundle(bundle, code_column=code_column)
    effective = bundle.frame.reset_index(drop=True)
    events = _history_frame(bundle).reset_index(drop=True)
    arrays = _empty_feature_arrays(len(effective))
    histories: Dict[str, Dict[str, ResourceHistory]] = {
        resource: {} for resource in RESOURCE_SPECS
    }

    # The tie break exactly matches Last-1: later source_row wins among jobs
    # sharing one completion timestamp because it is activated last.
    completion_heap: List[Tuple[int, int, int]] = []
    for event_index, row in events.iterrows():
        completion_ns = int(pd.Timestamp(row["completion_time"]).value)
        heapq.heappush(
            completion_heap,
            (completion_ns, int(row["source_row"]), event_index),
        )

    for row_index, row in effective.iterrows():
        submit_ns = int(pd.Timestamp(row["submit_time"]).value)
        while completion_heap and completion_heap[0][0] < submit_ns:
            completion_ns, _, event_index = heapq.heappop(completion_heap)
            _activate_event(
                events.iloc[event_index],
                completion_ns,
                histories,
                k=k,
                ema_alpha=alpha,
                code_column=code_column,
            )
        _record_features(
            row_index,
            row,
            submit_ns,
            histories,
            arrays,
            code_column=code_column,
        )

    result = effective[["sample_id"]].copy()
    for column, values in arrays.items():
        result[column] = values
    semantic_contract = causal_history_contract(k, alpha)
    result.attrs["causal_history_protocol"] = {
        **semantic_contract,
        "k": semantic_contract["history_k"],
        "ema_alpha": semantic_contract["history_ema_alpha"],
        "standard_deviation_ddof": 0,
        "prediction_population_rows": len(effective),
        "history_event_population_rows": len(events),
        "boundary_purged_rows_available_after_completion": len(events)
        - len(effective),
        "prediction_population_empty_script_rows": int(
            (~effective[code_column].map(_has_observed_script)).sum()
        ),
        "history_event_population_empty_script_rows": int(
            (~events[code_column].map(_has_observed_script)).sum()
        ),
    }
    return result


def history_features_to_long(features: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide feature frame to a sample/resource keyed table."""
    if "sample_id" not in features.columns:
        raise ValueError("History feature frame has no sample_id column")
    parts = []
    for resource in RESOURCE_SPECS:
        prefix = f"{resource}_history_"
        resource_columns = [
            column for column in features.columns if column.startswith(prefix)
        ]
        if not resource_columns:
            raise ValueError(f"History feature frame has no {resource} columns")
        part = features[["sample_id", *resource_columns]].copy()
        part = part.rename(
            columns={
                column: f"history_{column[len(prefix):]}"
                for column in resource_columns
            }
        )
        part.insert(1, "resource", resource)
        parts.append(part)

    result = pd.concat(parts, axis=0).sort_index(kind="stable").reset_index(drop=True)
    if result.duplicated(["sample_id", "resource"]).any():
        raise ValueError("Long history features do not have unique sample/resource keys")
    result.attrs.update(features.attrs)
    result.attrs["history_feature_layout"] = "long"
    return result


def build_causal_history_long(
    bundle: DatasetBundle,
    *,
    k: int = 5,
    ema_alpha: float = 0.5,
    code_column: str = "code",
) -> pd.DataFrame:
    """Build causal features keyed uniquely by ``sample_id`` and ``resource``."""
    features = build_causal_history_features(
        bundle,
        k=k,
        ema_alpha=ema_alpha,
        code_column=code_column,
    )
    return history_features_to_long(features)


__all__ = [
    "EMPTY_SCRIPT_SAME_HISTORY_POLICY",
    "EMPTY_SCRIPT",
    "HISTORY_PROTOCOL_VERSION",
    "RESOURCE_SPECS",
    "SAME_SCRIPT_IDENTITY",
    "build_causal_history_features",
    "build_causal_history_long",
    "causal_history_contract",
    "history_features_to_long",
    "script_sha256",
    "validate_causal_history_bundle",
]
