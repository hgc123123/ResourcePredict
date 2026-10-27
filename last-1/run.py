#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import (  # noqa: E402
    build_prediction_frame,
    output_directory,
    write_evaluation,
)
from common.config import get_value, load_config  # noqa: E402
from common.data import DatasetBundle, load_dataset  # noqa: E402


RESOURCE_SPECS = {
    "memory": ("target_memory", "valid_memory"),
    "vram": ("target_vram", "valid_vram"),
}
SUPPORTED_PROTOCOLS = {"completed_before_submission", "ordered_previous_rows"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the per-user Last-1 baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def training_medians(frame: pd.DataFrame) -> Dict[str, float]:
    train = frame.loc[frame["split"] == "train"]
    medians: Dict[str, float] = {}
    for resource, (target_column, valid_column) in RESOURCE_SPECS.items():
        values = train.loc[train[valid_column], target_column].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"Training split has no valid {resource} targets for fallback")
        medians[resource] = float(np.median(values))
    return medians


def empty_replay(frame: pd.DataFrame, fallback: Mapping[str, float]) -> Dict[str, Dict[str, np.ndarray]]:
    size = len(frame)
    replay: Dict[str, Dict[str, np.ndarray]] = {}
    for resource in RESOURCE_SPECS:
        replay[resource] = {
            "prediction": np.full(size, fallback[resource], dtype=float),
            "history_available": np.zeros(size, dtype=bool),
            "history_source_sample_id": np.full(size, "", dtype=object),
            "history_source_time": np.full(size, "", dtype=object),
        }
    return replay


def activate_history(
    frame: pd.DataFrame,
    row_index: int,
    histories: Dict[str, Dict[str, Tuple[float, str, str]]],
    source_time: str,
) -> None:
    row = frame.iloc[row_index]
    user = str(row["user_id"])
    for resource, (target_column, valid_column) in RESOURCE_SPECS.items():
        if bool(row[valid_column]):
            histories[resource][user] = (
                float(row[target_column]),
                str(row["sample_id"]),
                source_time,
            )


def make_prediction(
    frame: pd.DataFrame,
    row_index: int,
    histories: Dict[str, Dict[str, Tuple[float, str, str]]],
    replay: Dict[str, Dict[str, np.ndarray]],
) -> None:
    user = str(frame.iloc[row_index]["user_id"])
    for resource in RESOURCE_SPECS:
        history = histories[resource].get(user)
        if history is None:
            continue
        value, sample_id, source_time = history
        replay[resource]["prediction"][row_index] = value
        replay[resource]["history_available"][row_index] = True
        replay[resource]["history_source_sample_id"][row_index] = sample_id
        replay[resource]["history_source_time"][row_index] = source_time


def replay_ordered_previous_rows(
    frame: pd.DataFrame,
    fallback: Mapping[str, float],
) -> Dict[str, Dict[str, np.ndarray]]:
    replay = empty_replay(frame, fallback)
    histories: Dict[str, Dict[str, Tuple[float, str, str]]] = {
        resource: {} for resource in RESOURCE_SPECS
    }
    for row_index in range(len(frame)):
        make_prediction(frame, row_index, histories, replay)
        source_time = str(frame.iloc[row_index]["order_value"])
        activate_history(frame, row_index, histories, source_time)
    return replay


def validate_causal_protocol(bundle: DatasetBundle) -> None:
    frame = bundle.frame
    retained_frame = getattr(bundle, "retained_frame", None)
    history_frame = frame if retained_frame is None else retained_frame
    if not bundle.protocol.get("verified_chronological", False):
        raise ValueError(
            "completed_before_submission requires split.order_semantics=submission_time"
        )
    if not bundle.protocol.get("has_submission_time", False):
        raise ValueError("completed_before_submission requires a complete submit_time column")
    if not bundle.protocol.get("has_completion_time", False):
        raise ValueError("completed_before_submission requires a complete completion_time column")
    if not frame["submit_time"].is_monotonic_increasing:
        raise ValueError(
            "Rows/splits are not ordered by submit_time; verify data.columns.order"
        )
    if history_frame.empty:
        raise ValueError("The retained causal replay population is empty")
    if not history_frame["submit_time"].is_monotonic_increasing:
        raise ValueError(
            "Retained rows are not ordered by submit_time; verify boundary-purge handling"
        )
    effective_ids = set(frame["sample_id"].astype(str))
    retained_ids = set(history_frame["sample_id"].astype(str))
    missing = effective_ids.difference(retained_ids)
    if missing:
        raise ValueError(
            "The retained causal replay population is missing "
            f"{len(missing)} effective samples"
        )
    invalid = history_frame["completion_time"] < history_frame["submit_time"]
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} retained rows whose completion precedes submission"
        )


def replay_completed_before_submission(
    bundle: DatasetBundle,
    fallback: Mapping[str, float],
) -> Dict[str, Dict[str, np.ndarray]]:
    validate_causal_protocol(bundle)
    frame = bundle.frame
    retained_frame = getattr(bundle, "retained_frame", None)
    history_frame = (
        frame if retained_frame is None else retained_frame
    ).reset_index(drop=True)
    replay = empty_replay(frame, fallback)
    histories: Dict[str, Dict[str, Tuple[float, str, str]]] = {
        resource: {} for resource in RESOURCE_SPECS
    }

    # Completion events reveal labels. A strict '<' prevents a result completed
    # at the exact submission timestamp from being treated as already known.
    completion_heap: List[Tuple[int, int, int]] = []
    for history_index, row in history_frame.iterrows():
        completion_ns = int(row["completion_time"].value)
        heapq.heappush(
            completion_heap,
            (completion_ns, int(row["source_row"]), history_index),
        )

    for row_index in range(len(frame)):
        submit_ns = int(frame.iloc[row_index]["submit_time"].value)
        while completion_heap and completion_heap[0][0] < submit_ns:
            _, _, completed_index = heapq.heappop(completion_heap)
            completed_at = history_frame.iloc[completed_index][
                "completion_time"
            ].isoformat()
            activate_history(history_frame, completed_index, histories, completed_at)
        make_prediction(frame, row_index, histories, replay)
    return replay


def build_coverage(
    frame: pd.DataFrame,
    replay: Mapping[str, Mapping[str, np.ndarray]],
    protocol: str,
) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        split_mask = frame["split"].eq(split).to_numpy()
        for resource, (_, valid_column) in RESOURCE_SPECS.items():
            eligible = split_mask & frame[valid_column].to_numpy(bool)
            available = replay[resource]["history_available"].astype(bool) & eligible
            count = int(eligible.sum())
            history_count = int(available.sum())
            rows.append(
                {
                    "split": split,
                    "resource": resource,
                    "protocol": protocol,
                    "eligible_n": count,
                    "history_available_n": history_count,
                    "fallback_n": count - history_count,
                    "coverage": float(history_count / count) if count else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def attach_history_columns(
    predictions: pd.DataFrame,
    replay: Mapping[str, Mapping[str, np.ndarray]],
    protocol: str,
) -> pd.DataFrame:
    result = predictions.copy()
    result["history_available"] = False
    result["history_source_sample_id"] = ""
    result["history_source_time"] = ""
    result["protocol"] = protocol
    for resource in RESOURCE_SPECS:
        mask = result["resource"].eq(resource)
        resource_replay = replay[resource]
        result.loc[mask, "history_available"] = resource_replay["history_available"]
        result.loc[mask, "history_source_sample_id"] = resource_replay[
            "history_source_sample_id"
        ]
        result.loc[mask, "history_source_time"] = resource_replay["history_source_time"]
    result["fallback_used"] = ~result["history_available"].astype(bool)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    protocol = str(get_value(config, "last1.protocol", "completed_before_submission"))
    fallback_name = str(get_value(config, "last1.fallback", "train_resource_median"))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported last1.protocol: {protocol}")
    if fallback_name != "train_resource_median":
        raise ValueError("last1.fallback must be exactly train_resource_median")

    fallback = training_medians(bundle.frame)
    if protocol == "completed_before_submission":
        replay = replay_completed_before_submission(bundle, fallback)
        causal = True
    else:
        replay = replay_ordered_previous_rows(bundle.frame, fallback)
        causal = False

    seed = int(get_value(config, "seed", 42))
    predictions = build_prediction_frame(
        bundle.frame,
        {resource: replay[resource]["prediction"] for resource in RESOURCE_SPECS},
        model_name="last-1",
        seed=seed,
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)),
    )
    predictions = attach_history_columns(predictions, replay, protocol)
    coverage = build_coverage(bundle.frame, replay, protocol)
    output_dir = output_directory(config, "last-1", f"{protocol}/seed_{seed}")
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    retained_frame = getattr(bundle, "retained_frame", None)
    history_event_frame = (
        retained_frame
        if causal and retained_frame is not None
        else bundle.frame
    )

    details: Dict[str, Any] = {
        "protocol": protocol,
        "causal_online_protocol": causal,
        "fallback": fallback_name,
        "fallback_values": fallback,
        "resource_specific_history": True,
        "prediction_population_rows": int(len(bundle.frame)),
        "history_event_population_rows": int(len(history_event_frame)),
        "boundary_purged_rows_available_after_completion": int(
            len(history_event_frame) - len(bundle.frame)
        ),
        "strict_completion_relation": "completion_time < submit_time" if causal else None,
        "warning": (
            None
            if causal
            else "Smoke test only: previous ordered rows may not have completed at prediction time."
        ),
        "coverage": json.loads(coverage.to_json(orient="records")),
    }
    metrics = write_evaluation(
        config,
        bundle,
        model_name="last-1",
        predictions=predictions,
        output_dir=output_dir,
        details=details,
    )
    print(metrics.to_string(index=False))
    print(coverage.to_string(index=False))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
