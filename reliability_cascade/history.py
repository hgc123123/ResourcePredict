from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable

import numpy as np
import pandas as pd

from common.data import DatasetBundle
from common.history import (
    EMPTY_SCRIPT,
    RESOURCE_SPECS,
    script_sha256,
    validate_causal_history_bundle,
)


HISTORY_CONTRACT_VERSION = "reliability-cascade-history/v1"


@dataclass(frozen=True)
class Observation:
    value: float
    completion_ns: int
    sample_id: str


@dataclass
class WindowState:
    maxlen: int
    count: int = 0
    recent: Deque[Observation] = field(default_factory=deque)

    def add(self, observation: Observation) -> None:
        self.count += 1
        self.recent.append(observation)
        while len(self.recent) > self.maxlen:
            self.recent.popleft()


@dataclass
class UserResourceState:
    maxlen: int
    general: WindowState = field(init=False)
    scripts: Dict[str, WindowState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.general = WindowState(self.maxlen)

    def script(self, identity: str) -> WindowState:
        return self.scripts.setdefault(identity, WindowState(self.maxlen))


def _script_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _script_observed(value: object) -> bool:
    return bool(_script_text(value).strip())


def _window_summary(
    state: WindowState | None,
    submit_ns: int,
    *,
    half_life_hours: float,
) -> Dict[str, object]:
    if state is None or not state.recent:
        return {
            "available": False,
            "count": 0,
            "recent_count": 0,
            "last_value": np.nan,
            "median3": np.nan,
            "median5": np.nan,
            "mean5": np.nan,
            "std5": np.nan,
            "mad5": np.nan,
            "range5": np.nan,
            "decayed_mean": np.nan,
            "age_hours": np.nan,
            "source_sample_id": "",
            "source_completion_time": pd.NaT,
        }

    observations = list(state.recent)
    values = np.asarray([item.value for item in observations], dtype=np.float64)
    last3 = values[-3:]
    last5 = values[-5:]
    ages = np.asarray(
        [(submit_ns - item.completion_ns) / 3_600_000_000_000.0 for item in observations],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(ages)) or np.any(ages <= 0.0):
        raise AssertionError("Causal history contains a non-positive age")

    # Normalize relative exponential weights to avoid underflow for old histories.
    relative_age = ages - float(np.min(ages))
    weights = np.exp(-math.log(2.0) * relative_age / half_life_hours)
    decayed_mean = float(np.sum(weights * values) / np.sum(weights))
    median5 = float(np.median(last5))
    last = observations[-1]
    return {
        "available": True,
        "count": int(state.count),
        "recent_count": int(len(values)),
        "last_value": float(values[-1]),
        "median3": float(np.median(last3)),
        "median5": median5,
        "mean5": float(np.mean(last5)),
        "std5": float(np.std(last5, ddof=0)),
        "mad5": float(np.median(np.abs(last5 - median5))),
        "range5": float(np.max(last5) - np.min(last5)),
        "decayed_mean": decayed_mean,
        "age_hours": float(ages[-1]),
        "source_sample_id": last.sample_id,
        "source_completion_time": pd.Timestamp(last.completion_ns, tz="UTC"),
    }


def _activate(
    row: pd.Series,
    completion_ns: int,
    histories: Dict[str, Dict[str, UserResourceState]],
    *,
    history_k: int,
    code_column: str,
) -> None:
    user_id = str(row["user_id"])
    observed_script = _script_observed(row[code_column])
    identity = script_sha256(row[code_column])
    for resource, (target_column, valid_column) in RESOURCE_SPECS.items():
        if not bool(row[valid_column]):
            continue
        value = float(row[target_column])
        if not np.isfinite(value):
            raise ValueError(
                f"Sample {row['sample_id']} marks {resource} valid with a non-finite target"
            )
        state = histories[resource].setdefault(
            user_id, UserResourceState(maxlen=history_k)
        )
        observation = Observation(
            value=value,
            completion_ns=completion_ns,
            sample_id=str(row["sample_id"]),
        )
        state.general.add(observation)
        if observed_script:
            state.script(identity).add(observation)


def _prefix(values: Dict[str, object], name: str) -> Dict[str, object]:
    return {f"{name}_{key}": value for key, value in values.items()}


def build_robust_history_long(
    bundle: DatasetBundle,
    *,
    history_k: int = 5,
    half_life_hours: float = 168.0,
    code_column: str = "code",
) -> pd.DataFrame:
    """Replay causal history and return robust per-resource submission-time features.

    Labels enter state only when ``completion_time < current submit_time``. The
    returned values may therefore be used by a pre-Qwen router without outcome
    leakage. Boundary-purged rows remain eligible event sources after completion.
    """

    if isinstance(history_k, bool) or not isinstance(history_k, int) or history_k < 5:
        raise ValueError("history_k must be an integer of at least 5")
    if not np.isfinite(half_life_hours) or half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be finite and positive")
    validate_causal_history_bundle(bundle, code_column=code_column)

    effective = bundle.frame.reset_index(drop=True)
    retained = getattr(bundle, "retained_frame", None)
    events = (effective if retained is None else retained).reset_index(drop=True)
    histories: Dict[str, Dict[str, UserResourceState]] = {
        resource: {} for resource in RESOURCE_SPECS
    }
    completion_heap: list[tuple[int, int, int]] = []
    for event_index, row in events.iterrows():
        heapq.heappush(
            completion_heap,
            (
                int(pd.Timestamp(row["completion_time"]).value),
                int(row["source_row"]),
                int(event_index),
            ),
        )

    rows: list[Dict[str, object]] = []
    for _, row in effective.iterrows():
        submit_ns = int(pd.Timestamp(row["submit_time"]).value)
        while completion_heap and completion_heap[0][0] < submit_ns:
            completion_ns, _, event_index = heapq.heappop(completion_heap)
            _activate(
                events.iloc[event_index],
                completion_ns,
                histories,
                history_k=history_k,
                code_column=code_column,
            )

        user_id = str(row["user_id"])
        observed_script = _script_observed(row[code_column])
        identity = script_sha256(row[code_column])
        for resource in RESOURCE_SPECS:
            state = histories[resource].get(user_id)
            general_state = None if state is None else state.general
            exact_state = (
                None
                if state is None or not observed_script
                else state.scripts.get(identity)
            )
            general = _window_summary(
                general_state, submit_ns, half_life_hours=half_life_hours
            )
            exact = _window_summary(
                exact_state, submit_ns, half_life_hours=half_life_hours
            )
            output: Dict[str, object] = {
                "sample_id": str(row["sample_id"]),
                "resource": resource,
                "script_observed": observed_script,
                "script_sha256": identity,
                **_prefix(general, "user"),
                **_prefix(exact, "exact"),
            }
            output["same_last1"] = bool(
                exact["available"]
                and exact["source_sample_id"] == general["source_sample_id"]
            )
            output["preferred_source_sample_id"] = (
                exact["source_sample_id"]
                if exact["available"]
                else general["source_sample_id"]
            )
            rows.append(output)

    result = pd.DataFrame(rows)
    if result.duplicated(["sample_id", "resource"]).any():
        raise AssertionError("Robust history replay produced duplicate keys")
    result.attrs["causal_history_protocol"] = {
        "version": HISTORY_CONTRACT_VERSION,
        "strict_completion_relation": "completion_time < submit_time",
        "completion_tie_break": "higher source_row activates later",
        "same_script_identity": "sha256(exact redacted script encoder input)",
        "blank_script_match": False,
        "history_k": history_k,
        "decay_half_life_hours": float(half_life_hours),
        "prediction_population_rows": int(len(effective)),
        "history_event_population_rows": int(len(events)),
        "boundary_purged_rows_available_after_completion": int(
            len(events) - len(effective)
        ),
        "empty_script_sentinel": EMPTY_SCRIPT,
    }
    return result


def history_candidate(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Return one hierarchical robust expert with exact-script preference."""

    candidates = {
        "hierarchical_last": ("exact_last_value", "user_last_value"),
        "hierarchical_median3": ("exact_median3", "user_median3"),
        "hierarchical_median5": ("exact_median5", "user_median5"),
        "hierarchical_decay": ("exact_decayed_mean", "user_decayed_mean"),
    }
    if name not in candidates:
        raise ValueError(f"Unknown history expert {name!r}; choose {sorted(candidates)}")
    exact_name, user_name = candidates[name]
    exact_available = frame["exact_available"].astype(bool).to_numpy()
    exact = pd.to_numeric(frame[exact_name], errors="coerce").to_numpy(float)
    user = pd.to_numeric(frame[user_name], errors="coerce").to_numpy(float)
    output = np.where(exact_available, exact, user)
    history = frame["user_available"].astype(bool).to_numpy()
    if np.any(history & ~np.isfinite(output)):
        raise ValueError(f"History expert {name} is non-finite despite available history")
    return output


def history_candidate_names() -> Iterable[str]:
    return (
        "hierarchical_last",
        "hierarchical_median3",
        "hierarchical_median5",
        "hierarchical_decay",
    )


__all__ = [
    "HISTORY_CONTRACT_VERSION",
    "build_robust_history_long",
    "history_candidate",
    "history_candidate_names",
]
