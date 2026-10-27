"""Completion-safe base-model selection without changing the outer split.

This module deliberately imports neither torch nor transformers, so the split and
provenance checks can also run before submitting a GPU job.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping

import pandas as pd

from common.config import get_value
from common.data import DatasetBundle, split_fingerprint


BASE_SELECTION_PROTOCOL = "train-tail-base-selection/v1"
LEGACY_MODE = "legacy_validation"
MODES = (LEGACY_MODE, "train_tail", "shared_validation_control")


def base_selection_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = get_value(config, "qwen.base_selection", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("qwen.base_selection must be a mapping")
    unknown = set(raw) - {"mode", "fit_fraction"}
    if unknown:
        raise ValueError(f"Unknown qwen.base_selection settings: {sorted(unknown)}")
    mode = str(raw.get("mode", LEGACY_MODE)).strip().lower()
    if mode not in MODES:
        raise ValueError(f"qwen.base_selection.mode must be one of {MODES}")
    fraction = float(raw.get("fit_fraction", 0.90))
    if not math.isfinite(fraction) or not 0.5 <= fraction < 1.0:
        raise ValueError("qwen.base_selection.fit_fraction must be in [0.5, 1)")
    return {"mode": mode, "fit_fraction": fraction}


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timestamps(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"sample_id", "submit_time", "completion_time", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing selection columns: {missing}")
    output = frame.copy()
    if output["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate sample IDs")
    for column in ("submit_time", "completion_time"):
        output[column] = pd.to_datetime(output[column], errors="raise", utc=True)
        if output[column].isna().any():
            raise ValueError(f"{name} contains missing {column}")
    if (output["completion_time"] < output["submit_time"]).any():
        raise ValueError(f"{name} completion precedes submission")
    # Stable original row order breaks ties; tied submissions remain one group.
    return output.sort_values("submit_time", kind="mergesort").reset_index(drop=True)


def _valid_counts(frame: pd.DataFrame, name: str) -> Dict[str, int]:
    counts = {}
    for resource in ("memory", "vram"):
        column = f"valid_{resource}"
        if column not in frame:
            raise ValueError(f"{name} has no {column}")
        counts[resource] = int(frame[column].astype(bool).sum())
        if counts[resource] == 0:
            raise ValueError(f"{name} has no valid {resource} labels")
    return counts


@dataclass
class BaseTrainingSelection:
    optimization: pd.DataFrame
    early_stopping: pd.DataFrame
    manifest: Dict[str, Any] | None


def select_base_training_frames(
    bundle: DatasetBundle, config: Dict[str, Any]
) -> BaseTrainingSelection:
    settings = base_selection_settings(config)
    if settings["mode"] == LEGACY_MODE:
        return BaseTrainingSelection(bundle.split("train"),
                                     bundle.split("validation"), None)
    if not bool(bundle.protocol.get("verified_chronological", False)):
        raise ValueError("Train-tail base selection requires verified submission chronology")
    train = _timestamps(bundle.split("train"), "Outer training split")
    validation = _timestamps(bundle.split("validation"), "Outer validation split")
    boundary_index = int(len(train) * settings["fit_fraction"])
    if not 0 < boundary_index < len(train):
        raise ValueError("Train-tail base selection creates an empty block")
    cutoff = train.iloc[boundary_index]["submit_time"]
    leading = train["submit_time"] < cutoff
    tail_mask = ~leading
    fit_mask = leading & (train["completion_time"] < cutoff)
    purged_mask = leading & ~fit_mask
    optimization = train.loc[fit_mask].reset_index(drop=True)
    tail = train.loc[tail_mask].reset_index(drop=True)
    if optimization.empty or tail.empty:
        raise ValueError("Completion-safe train-tail selection creates an empty block")
    validation_start = validation["submit_time"].min()
    if not (train["completion_time"] < validation_start).all():
        raise ValueError("Outer training labels must finish before outer validation begins")
    is_control = settings["mode"] == "shared_validation_control"
    early_stopping = validation.copy() if is_control else tail.copy()
    if is_control:
        test = _timestamps(bundle.split("test"), "Outer test split")
        if not (early_stopping["completion_time"] < test["submit_time"].min()).all():
            raise ValueError("Control validation labels must finish before test begins")
    fit_counts = _valid_counts(optimization, "Base optimization block")
    stop_counts = _valid_counts(early_stopping, "Base early-stopping block")
    optimization_ids = optimization["sample_id"].astype(str).tolist()
    stop_ids = early_stopping["sample_id"].astype(str).tolist()
    tail_ids = tail["sample_id"].astype(str).tolist()
    purged_ids = train.loc[purged_mask, "sample_id"].astype(str).tolist()
    roles = []
    for ids, role in ((optimization_ids, "optimization"),
                      (tail_ids, "reserved_train_tail" if is_control else "early_stopping"),
                      (purged_ids, "purged_optimization")):
        roles.extend({"sample_id": sample_id, "role": role} for sample_id in ids)
    if is_control:
        roles.extend({"sample_id": sample_id, "role": "early_stopping"}
                     for sample_id in stop_ids)
    if set(optimization_ids) & set(stop_ids):
        raise AssertionError("Base optimization and early-stopping jobs overlap")
    record = {
        "protocol": BASE_SELECTION_PROTOCOL,
        **settings,
        "source_data_sha256": bundle.data_sha256,
        "source_split_fingerprint": split_fingerprint(bundle.frame),
        "optimization_source": "outer_train_leading_completion_safe_block",
        "early_stopping_source": "outer_validation" if is_control else "outer_train_tail",
        "preprocessing_fit_source": "optimization_only",
        "cutoff_submit_time": cutoff.isoformat(),
        "outer_validation_start_submit_time": validation_start.isoformat(),
        "optimization_max_completion_time": optimization["completion_time"].max().isoformat(),
        "early_stopping_max_completion_time": early_stopping["completion_time"].max().isoformat(),
        "optimization_sample_ids": optimization_ids,
        "early_stopping_sample_ids": stop_ids,
        "reserved_train_tail_sample_ids": tail_ids,
        "purged_optimization_sample_ids": purged_ids,
        "optimization_count": len(optimization),
        "early_stopping_count": len(early_stopping),
        "reserved_train_tail_count": len(tail),
        "purged_optimization_count": len(purged_ids),
        "optimization_valid_by_resource": fit_counts,
        "early_stopping_valid_by_resource": stop_counts,
        "sample_roles": roles,
        "outer_split_unchanged": True,
        "equal_submit_times_split_across_blocks": False,
        "optimization_completion_relation": "completion_time < train_tail_start_submit_time",
        "base_checkpoint_validation_overlap": is_control,
        "validation_used_for_early_stopping": is_control,
        "test_used_for_fit_or_selection": False,
        "refit_after_checkpoint_selection": False,
    }
    record["sample_roles_sha256"] = _record_hash({"roles": roles})
    record["sha256"] = _record_hash(record)
    return BaseTrainingSelection(optimization, early_stopping, record)


def base_preprocessing_bundle(
    bundle: DatasetBundle, selection: BaseTrainingSelection
) -> DatasetBundle:
    """Present only the internal gradient/early-stop rows to the existing fitter."""
    train = selection.optimization.copy()
    validation = selection.early_stopping.copy()
    train["split"] = "train"
    validation["split"] = "validation"
    return replace(bundle, frame=pd.concat([train, validation], ignore_index=True))


def verify_base_training_selection(
    bundle: DatasetBundle, config: Dict[str, Any], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Rebuild roles from the unchanged source and reject tampered provenance."""
    if not isinstance(record, Mapping) or record.get("sha256") != _record_hash(record):
        raise ValueError("Base training selection fingerprint is invalid")
    expected = select_base_training_frames(bundle, config).manifest
    if expected is None or dict(record) != expected:
        raise ValueError("Base training selection does not match source data/configuration")
    return copy.deepcopy(expected)
