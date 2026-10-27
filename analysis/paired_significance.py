#!/usr/bin/env python3
"""Paired, time-blocked significance analysis for deployable predictions."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (  # noqa: E402
    get_value,
    load_config,
    resolve_path,
    write_resolved_config,
)
from common.data import load_dataset, sha256_file  # noqa: E402
from common.metrics import coerce_valid_mask, validate_prediction_frame  # noqa: E402


PAIR_KEYS = ["sample_id", "split", "resource"]
EVALUATION_SPLITS = ("validation", "test")
RESOURCES = ("memory", "vram")
FORBIDDEN_FLAGS = (
    "uses_targets",
    "diagnostic_only",
    "not_a_deployable_model",
    "test_labels_used_for_selection",
    "uses_test_labels",
    "test_labels_used",
)
FORBIDDEN_IDENTITY_TOKENS = ("oracle", "diagnostic", "post_hoc", "post-hoc")
METRIC_DIRECTIONS = {
    "mae": "lower_is_better",
    "mse": "lower_is_better",
    "rmse": "lower_is_better",
    "acc_at_10": "higher_is_better",
}
SAFE_PREDICTION_COLUMNS = (
    "prediction",
    "qwen_prediction",
    "last1_prediction",
    "history_expert_prediction",
    "ordinary_safe_reference_prediction",
    "safe_reference_prediction",
    "hierarchical_reference_prediction",
)
OUTPUT_ARTIFACT_NAMES = (
    "paired_metrics.csv",
    "paired_differences.csv",
    "mcnemar.csv",
    "block_summary.csv",
    "significance_summary.json",
    "run_manifest.json",
    "config.resolved.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deployable proposed/reference predictions with paired "
            "submission-time-block bootstrap intervals and McNemar tests"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--proposed-predictions", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument(
        "--reference-column",
        choices=SAFE_PREDICTION_COLUMNS,
        default="prediction",
        help=(
            "Target-independent prediction column to compare; use "
            "hierarchical_reference_prediction for cascade artifacts, or "
            "safe_reference_prediction for legacy safe-hybrid artifacts"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing significance artifact set in --output-dir",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=EVALUATION_SPLITS,
        help="Evaluation split to analyze; repeat to include both (default: test)",
    )
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument(
        "--block-frequency",
        help="Fixed pandas time frequency used to floor UTC submit times (default: 1D)",
    )
    parser.add_argument("--confidence-level", type=float)
    parser.add_argument(
        "--mcnemar-method",
        choices=("auto", "exact", "asymptotic"),
        default="auto",
    )
    parser.add_argument("--mcnemar-exact-threshold", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _flag_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _contains_forbidden_flag(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_FLAGS and _flag_is_true(child):
                return key
            found = _contains_forbidden_flag(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_forbidden_flag(child)
            if found:
                return found
    return None


def _identity_is_forbidden(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return any(token in normalized for token in FORBIDDEN_IDENTITY_TOKENS)


def _contains_forbidden_identity(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"model", "analysis", "artifact_type", "diagnostic_type"}:
                if _identity_is_forbidden(child):
                    return key
            found = _contains_forbidden_identity(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_forbidden_identity(child)
            if found:
                return found
    return None


def _split_fingerprint(manifest: dict[str, Any], role: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{role} run manifest has no split_fingerprint")
    return str(value)


def load_input_manifest(prediction_path: Path, role: str) -> tuple[dict[str, Any], Path]:
    manifest_path = prediction_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{role} predictions require their run manifest: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{role} run manifest must be a JSON object")
    for name in ("model", "seed", "data_sha256"):
        if manifest.get(name) is None:
            raise ValueError(f"{role} run manifest has no {name}")
    _split_fingerprint(manifest, role)
    forbidden = _contains_forbidden_flag(manifest)
    if forbidden:
        raise ValueError(
            f"{role} is a diagnostic/target-using artifact ({forbidden}=true)"
        )
    forbidden_identity = _contains_forbidden_identity(manifest)
    if forbidden_identity:
        raise ValueError(
            f"{role} manifest identifies a diagnostic/oracle artifact in "
            f"{forbidden_identity}"
        )
    return manifest, manifest_path


def validate_artifact_identity(
    predictions: pd.DataFrame,
    manifest: dict[str, Any],
    role: str,
) -> None:
    validate_prediction_frame(predictions)
    models = predictions["model"].astype(str).unique().tolist()
    if len(models) != 1 or models[0] != str(manifest["model"]):
        raise ValueError(
            f"{role} prediction model disagrees with its manifest: {models}"
        )
    if _identity_is_forbidden(models[0]):
        raise ValueError(f"{role} predictions identify a diagnostic/oracle model")

    seeds = pd.to_numeric(predictions["seed"], errors="coerce")
    try:
        manifest_seed = int(manifest["seed"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{role} manifest seed is not an integer") from exc
    if seeds.isna().any() or not (seeds == manifest_seed).all():
        raise ValueError(f"{role} prediction seed disagrees with its manifest")

    for flag in FORBIDDEN_FLAGS:
        if flag not in predictions:
            continue
        try:
            marked = coerce_valid_mask(predictions[flag], f"{role} {flag}")
        except ValueError as exc:
            raise ValueError(f"Invalid diagnostic flag in {role} predictions: {exc}") from exc
        if marked.any():
            raise ValueError(f"{role} predictions are marked {flag}=true")


def validate_shared_provenance(
    proposed_manifest: dict[str, Any],
    reference_manifest: dict[str, Any],
) -> tuple[str, str]:
    proposed_hash = str(proposed_manifest["data_sha256"])
    reference_hash = str(reference_manifest["data_sha256"])
    if proposed_hash != reference_hash:
        raise ValueError("Proposed and reference artifacts use different datasets")
    proposed_split = _split_fingerprint(proposed_manifest, "Proposed")
    reference_split = _split_fingerprint(reference_manifest, "Reference")
    if proposed_split != reference_split:
        raise ValueError("Proposed and reference artifacts use different splits")
    return proposed_hash, proposed_split


def validate_reference_column_contract(
    manifest: dict[str, Any],
    column: str,
) -> None:
    """Require validation-frozen provenance for nonstandard hybrid columns."""
    if column == "prediction":
        return
    model = str(manifest.get("model", ""))
    if column == "hierarchical_reference_prediction":
        details = manifest.get("details") or {}
        if not model.startswith("reliability_cascade_"):
            raise ValueError(
                "hierarchical_reference_prediction is accepted only from a "
                "reliability-cascade artifact"
            )
        if str(details.get("protocol", "")) != (
            "history-first-reliability-cascade/v1"
        ):
            raise ValueError(
                "Cascade hierarchical reference lacks the versioned protocol contract"
            )
        if details.get("test_labels_used_for_fit_or_selection") is not False:
            raise ValueError(
                "Cascade hierarchical reference does not declare test-label exclusion"
            )
        if details.get("route_decision_uses_qwen_output") is not False:
            raise ValueError(
                "Cascade manifest does not declare a pre-Qwen routing contract"
            )
        return
    if "hybrid" not in model:
        raise ValueError(
            f"Reference column {column!r} is accepted only from a formal hybrid artifact"
        )
    required = {
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
    }
    for name, expected in required.items():
        observed = manifest.get(name)
        matches = observed is expected if isinstance(expected, bool) else observed == expected
        if not matches:
            raise ValueError(
                f"Reference column {column!r} lacks frozen validation provenance: "
                f"{name}={observed!r}"
            )
    if column in {
        "safe_reference_prediction",
        "ordinary_safe_reference_prediction",
        "history_expert_prediction",
    }:
        if manifest.get("selection_protocol_version") != (
            "pareto_safe_script_history_blend_v1"
        ):
            raise ValueError(
                f"Reference column {column!r} requires a same-script safe-hybrid protocol"
            )
        policies = manifest.get("safe_reference_policies")
        if column == "safe_reference_prediction":
            if not isinstance(policies, dict) or not all(
                isinstance(policies.get(resource), dict) for resource in RESOURCES
            ):
                raise ValueError(
                    "safe_reference_prediction requires memory/vram "
                    "safe_reference_policies in its manifest"
                )
            for resource in RESOURCES:
                policy = policies[resource]
                try:
                    alpha = float(policy.get("alpha"))
                    beta = float(policy.get("same_script_beta"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "safe_reference_prediction has a malformed hierarchical "
                        f"policy for {resource}"
                    ) from exc
                if not np.isclose(alpha, 1.0) or not np.isclose(beta, 1.0):
                    raise ValueError(
                        "safe_reference_prediction is not a strict hierarchical "
                        f"same-script reference for {resource}: alpha={alpha}, beta={beta}"
                    )
        if column == "history_expert_prediction":
            selected = manifest.get("script_history_policies")
            if not isinstance(selected, dict) or not all(
                isinstance(selected.get(resource), dict) for resource in RESOURCES
            ):
                raise ValueError(
                    "history_expert_prediction requires memory/vram "
                    "script_history_policies in its manifest"
                )


def _scoped_predictions(
    frame: pd.DataFrame,
    role: str,
    splits: Sequence[str],
) -> pd.DataFrame:
    validate_prediction_frame(frame)
    observed_splits = set(frame["split"].astype(str))
    observed_resources = set(frame["resource"].astype(str))
    unknown_splits = observed_splits - {"train", *EVALUATION_SPLITS}
    unknown_resources = observed_resources - set(RESOURCES)
    if unknown_splits:
        raise ValueError(f"{role} has unknown splits: {sorted(unknown_splits)}")
    if unknown_resources:
        raise ValueError(f"{role} has unknown resources: {sorted(unknown_resources)}")

    scoped = frame.loc[
        frame["split"].astype(str).isin(splits)
        & frame["resource"].astype(str).isin(RESOURCES)
    ].copy()
    scoped["sample_id"] = scoped["sample_id"].astype(str)
    scoped["job_id"] = scoped["job_id"].astype(str)
    scoped["split"] = scoped["split"].astype(str)
    scoped["resource"] = scoped["resource"].astype(str)
    if scoped.duplicated(PAIR_KEYS).any():
        raise ValueError(f"{role} has duplicate sample/split/resource keys")
    for split in splits:
        for resource in RESOURCES:
            if scoped.loc[
                (scoped["split"] == split) & (scoped["resource"] == resource)
            ].empty:
                raise ValueError(
                    f"{role} has no rows for split={split}, resource={resource}"
                )
    return scoped


def _targets_match(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    same_nonfinite = (
        (np.isnan(left) & np.isnan(right))
        | (np.isposinf(left) & np.isposinf(right))
        | (np.isneginf(left) & np.isneginf(right))
    )
    return (left_finite & right_finite & np.isclose(left, right, rtol=0.0, atol=1e-12)) | (
        ~left_finite & ~right_finite & same_nonfinite
    )


def pair_prediction_frames(
    proposed: pd.DataFrame,
    reference: pd.DataFrame,
    splits: Sequence[str] = ("test",),
) -> pd.DataFrame:
    """Strictly align two prediction artifacts without dropping unmatched rows."""
    normalized_splits = tuple(str(value) for value in splits)
    if (
        not normalized_splits
        or len(set(normalized_splits)) != len(normalized_splits)
        or set(normalized_splits) - set(EVALUATION_SPLITS)
    ):
        raise ValueError("splits must be a non-empty unique subset of validation/test")

    proposed_scoped = _scoped_predictions(proposed, "Proposed", normalized_splits)
    reference_scoped = _scoped_predictions(reference, "Reference", normalized_splits)
    proposed_keys = pd.MultiIndex.from_frame(proposed_scoped[PAIR_KEYS])
    reference_keys = pd.MultiIndex.from_frame(reference_scoped[PAIR_KEYS])
    missing_reference = proposed_keys.difference(reference_keys)
    missing_proposed = reference_keys.difference(proposed_keys)
    if len(missing_reference) or len(missing_proposed):
        raise ValueError(
            "Proposed/reference prediction keys differ: "
            f"missing_from_reference={len(missing_reference)}, "
            f"missing_from_proposed={len(missing_proposed)}"
        )

    optional = [name for name in ("user_id", "submit_time") if name in proposed_scoped]
    proposed_view = proposed_scoped[
        PAIR_KEYS
        + ["job_id", "target", "prediction", "valid", "model", "seed"]
        + optional
    ].rename(
        columns={
            name: f"proposed_{name}"
            for name in [
                "job_id",
                "target",
                "prediction",
                "valid",
                "model",
                "seed",
                *optional,
            ]
        }
    )
    optional = [name for name in ("user_id", "submit_time") if name in reference_scoped]
    reference_view = reference_scoped[
        PAIR_KEYS
        + ["job_id", "target", "prediction", "valid", "model", "seed"]
        + optional
    ].rename(
        columns={
            name: f"reference_{name}"
            for name in [
                "job_id",
                "target",
                "prediction",
                "valid",
                "model",
                "seed",
                *optional,
            ]
        }
    )
    paired = proposed_view.merge(
        reference_view,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    job_match = paired["proposed_job_id"].astype(str) == paired[
        "reference_job_id"
    ].astype(str)
    if not job_match.all():
        raise ValueError(
            f"Proposed/reference job_id differs for {int((~job_match).sum())} rows"
        )
    if "proposed_user_id" in paired and "reference_user_id" in paired:
        user_match = paired["proposed_user_id"].astype(str) == paired[
            "reference_user_id"
        ].astype(str)
        if not user_match.all():
            raise ValueError(
                f"Proposed/reference user_id differs for {int((~user_match).sum())} rows"
            )

    proposed_valid = coerce_valid_mask(paired["proposed_valid"], "proposed valid")
    reference_valid = coerce_valid_mask(paired["reference_valid"], "reference valid")
    if not np.array_equal(proposed_valid, reference_valid):
        raise ValueError("Proposed/reference validity masks differ")

    proposed_target = pd.to_numeric(paired["proposed_target"], errors="coerce").to_numpy(
        float
    )
    reference_target = pd.to_numeric(
        paired["reference_target"], errors="coerce"
    ).to_numpy(float)
    matches = _targets_match(proposed_target, reference_target)
    if not matches.all():
        raise ValueError(
            f"Proposed/reference targets differ for {int((~matches).sum())} rows"
        )

    paired["job_id"] = paired["proposed_job_id"].astype(str)
    paired["target"] = proposed_target
    paired["valid"] = proposed_valid
    paired["proposed_prediction"] = pd.to_numeric(
        paired["proposed_prediction"], errors="coerce"
    )
    paired["reference_prediction"] = pd.to_numeric(
        paired["reference_prediction"], errors="coerce"
    )
    return paired


def attach_submission_blocks(
    paired: pd.DataFrame,
    dataset_frame: pd.DataFrame,
    block_frequency: str = "1D",
) -> pd.DataFrame:
    """Attach UTC submit-time blocks and recheck predictions against the dataset."""
    required = {
        "sample_id",
        "job_id",
        "split",
        "submit_time",
        "target_memory",
        "valid_memory",
        "target_vram",
        "valid_vram",
    }
    missing = required - set(dataset_frame.columns)
    if missing:
        raise ValueError(
            f"Configured dataset is missing fields needed for pairing: {sorted(missing)}"
        )
    context = dataset_frame.loc[
        dataset_frame["split"].astype(str).isin(set(paired["split"].astype(str)))
    ].copy()
    context["sample_id"] = context["sample_id"].astype(str)
    context["job_id"] = context["job_id"].astype(str)
    context["split"] = context["split"].astype(str)
    if context.duplicated(["sample_id", "split"]).any():
        raise ValueError("Configured dataset has duplicate sample_id/split rows")

    expected_parts = []
    for resource in RESOURCES:
        expected_parts.append(
            pd.DataFrame(
                {
                    "sample_id": context["sample_id"],
                    "split": context["split"],
                    "resource": resource,
                    "dataset_job_id": context["job_id"],
                    "dataset_target": context[f"target_{resource}"],
                    "dataset_valid": context[f"valid_{resource}"],
                    "dataset_submit_time": context["submit_time"],
                }
            )
        )
    expected = pd.concat(expected_parts, ignore_index=True)
    output = paired.merge(
        expected,
        on=PAIR_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not output["_merge"].eq("both").all():
        raise ValueError(
            f"Configured dataset is missing {int(output['_merge'].ne('both').sum())} paired rows"
        )
    output = output.drop(columns="_merge")

    job_match = output["job_id"].astype(str) == output["dataset_job_id"].astype(str)
    if not job_match.all():
        raise ValueError(
            f"Prediction job_id differs from the dataset for {int((~job_match).sum())} rows"
        )
    dataset_valid = coerce_valid_mask(output["dataset_valid"], "dataset valid")
    if not np.array_equal(output["valid"].to_numpy(bool), dataset_valid):
        raise ValueError("Prediction validity masks differ from the configured dataset")
    dataset_target = pd.to_numeric(output["dataset_target"], errors="coerce").to_numpy(
        float
    )
    target_match = _targets_match(output["target"].to_numpy(float), dataset_target)
    if not target_match.all():
        raise ValueError(
            f"Prediction targets differ from the dataset for {int((~target_match).sum())} rows"
        )

    submit_time = pd.to_datetime(output["dataset_submit_time"], errors="coerce", utc=True)
    if submit_time.isna().any():
        raise ValueError("Configured evaluation rows have missing/unparseable submit_time")
    for role in ("proposed", "reference"):
        name = f"{role}_submit_time"
        if name not in output:
            continue
        artifact_time = pd.to_datetime(output[name], errors="coerce", utc=True)
        if artifact_time.isna().any() or not (artifact_time == submit_time).all():
            raise ValueError(f"{role.capitalize()} submit_time differs from the dataset")
    try:
        time_block = submit_time.dt.floor(block_frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "block_frequency must be a fixed pandas frequency such as 12h, 1D, or 7D"
        ) from exc
    if time_block.isna().any():
        raise ValueError("Time-block construction produced missing values")
    output["submit_time"] = submit_time
    output["time_block"] = time_block
    return output


def _point_metrics(
    part: pd.DataFrame,
    tolerance: float,
) -> tuple[dict[str, float], dict[str, int]]:
    target = part["target"].to_numpy(float)
    proposed = part["proposed_prediction"].to_numpy(float)
    reference = part["reference_prediction"].to_numpy(float)
    proposed_error = proposed - target
    reference_error = reference - target
    proposed_hit = np.abs(proposed_error) <= tolerance
    reference_hit = np.abs(reference_error) <= tolerance
    values = {
        "proposed_mae": float(np.mean(np.abs(proposed_error))),
        "reference_mae": float(np.mean(np.abs(reference_error))),
        "proposed_mse": float(np.mean(np.square(proposed_error))),
        "reference_mse": float(np.mean(np.square(reference_error))),
        "proposed_acc_at_10": float(np.mean(proposed_hit)),
        "reference_acc_at_10": float(np.mean(reference_hit)),
    }
    values["proposed_rmse"] = float(np.sqrt(values["proposed_mse"]))
    values["reference_rmse"] = float(np.sqrt(values["reference_mse"]))
    counts = {
        "both_hit": int(np.sum(proposed_hit & reference_hit)),
        "proposed_only_hit": int(np.sum(proposed_hit & ~reference_hit)),
        "reference_only_hit": int(np.sum(~proposed_hit & reference_hit)),
        "neither_hit": int(np.sum(~proposed_hit & ~reference_hit)),
    }
    return values, counts


def _bootstrap_group(
    part: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
    tolerance: float,
) -> dict[str, np.ndarray]:
    working = part.copy()
    target = working["target"].to_numpy(float)
    proposed_error = working["proposed_prediction"].to_numpy(float) - target
    reference_error = working["reference_prediction"].to_numpy(float) - target
    working["proposed_absolute"] = np.abs(proposed_error)
    working["reference_absolute"] = np.abs(reference_error)
    working["proposed_squared"] = np.square(proposed_error)
    working["reference_squared"] = np.square(reference_error)
    working["proposed_hit"] = (np.abs(proposed_error) <= tolerance).astype(int)
    working["reference_hit"] = (np.abs(reference_error) <= tolerance).astype(int)
    block = (
        working.groupby("time_block", sort=True, observed=True)
        .agg(
            n=("sample_id", "size"),
            proposed_absolute=("proposed_absolute", "sum"),
            reference_absolute=("reference_absolute", "sum"),
            proposed_squared=("proposed_squared", "sum"),
            reference_squared=("reference_squared", "sum"),
            proposed_hit=("proposed_hit", "sum"),
            reference_hit=("reference_hit", "sum"),
        )
        .reset_index()
    )
    block_count = len(block)
    if block_count < 2:
        raise ValueError("Time-block bootstrap requires at least two non-empty blocks")
    counts = block["n"].to_numpy(float)
    sums = {
        name: block[name].to_numpy(float)
        for name in (
            "proposed_absolute",
            "reference_absolute",
            "proposed_squared",
            "reference_squared",
            "proposed_hit",
            "reference_hit",
        )
    }
    output = {name: np.empty(replicates, dtype=float) for name in METRIC_DIRECTIONS}
    chunk_size = min(1000, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        draws = rng.integers(0, block_count, size=(stop - start, block_count))
        sampled_n = counts[draws].sum(axis=1)
        proposed_mae = sums["proposed_absolute"][draws].sum(axis=1) / sampled_n
        reference_mae = sums["reference_absolute"][draws].sum(axis=1) / sampled_n
        proposed_mse = sums["proposed_squared"][draws].sum(axis=1) / sampled_n
        reference_mse = sums["reference_squared"][draws].sum(axis=1) / sampled_n
        output["mae"][start:stop] = proposed_mae - reference_mae
        output["mse"][start:stop] = proposed_mse - reference_mse
        output["rmse"][start:stop] = np.sqrt(proposed_mse) - np.sqrt(reference_mse)
        output["acc_at_10"][start:stop] = (
            sums["proposed_hit"][draws].sum(axis=1) / sampled_n
            - sums["reference_hit"][draws].sum(axis=1) / sampled_n
        )
    return output


def paired_block_bootstrap(
    paired: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 2026,
    confidence_level: float = 0.95,
    tolerance: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return point metrics, paired metric differences, and block audit rows."""
    if replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0,1)")
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("Paired paper evaluation fixes Acc@10% tolerance at 0.10")
    if "time_block" not in paired:
        raise ValueError("Paired rows have no time_block")

    groups = [
        (str(split), resource)
        for split in sorted(paired["split"].astype(str).unique())
        for resource in RESOURCES
    ]
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(groups))
    metric_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    tail = (1.0 - confidence_level) / 2.0

    for stream_index, ((split, resource), child) in enumerate(
        zip(groups, child_sequences)
    ):
        part = paired.loc[
            (paired["split"].astype(str) == split)
            & (paired["resource"].astype(str) == resource)
            & paired["valid"].astype(bool)
        ].copy()
        if part.empty:
            raise ValueError(f"No valid paired rows for split={split}, resource={resource}")
        numeric = part[
            ["target", "proposed_prediction", "reference_prediction"]
        ].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(
                f"Non-finite valid values for split={split}, resource={resource}"
            )
        part[["target", "proposed_prediction", "reference_prediction"]] = numeric
        point, hit_counts = _point_metrics(part, tolerance)
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        draws = _bootstrap_group(
            part, replicates, np.random.default_rng(child), tolerance
        )
        for role in ("proposed", "reference"):
            metric_rows.append(
                {
                    "split": split,
                    "resource": resource,
                    "role": role,
                    "model": str(part[f"{role}_model"].iloc[0]),
                    "seed": int(part[f"{role}_seed"].iloc[0]),
                    "n": int(len(part)),
                    "mae": point[f"{role}_mae"],
                    "mse": point[f"{role}_mse"],
                    "rmse": point[f"{role}_rmse"],
                    "acc_tolerance": tolerance,
                    "hits_at_tolerance": int(
                        hit_counts["both_hit"]
                        + hit_counts[f"{role}_only_hit"]
                    ),
                    "acc_at_10": point[f"{role}_acc_at_10"],
                }
            )
        for metric, direction in METRIC_DIRECTIONS.items():
            proposed_value = point[f"proposed_{metric}"]
            reference_value = point[f"reference_{metric}"]
            delta = proposed_value - reference_value
            lower, upper = np.quantile(draws[metric], [tail, 1.0 - tail])
            difference_rows.append(
                {
                    "split": split,
                    "resource": resource,
                    "metric": metric,
                    "n": int(len(part)),
                    "proposed_value": proposed_value,
                    "reference_value": reference_value,
                    "difference_proposed_minus_reference": delta,
                    "relative_error_reduction": (
                        float((reference_value - proposed_value) / reference_value)
                        if metric != "acc_at_10" and reference_value > 0.0
                        else float("nan")
                    ),
                    "confidence_level": confidence_level,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                    "favorable_direction": direction,
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_seed": int(seed),
                    "bootstrap_stream_index": int(stream_index),
                    "bootstrap_child_seed": child_seed,
                }
            )
        sizes = part.groupby("time_block", observed=True).size()
        block_rows.append(
            {
                "split": split,
                "resource": resource,
                "n": int(len(part)),
                "time_block_n": int(len(sizes)),
                "min_rows_per_block": int(sizes.min()),
                "median_rows_per_block": float(sizes.median()),
                "max_rows_per_block": int(sizes.max()),
                "first_time_block": pd.Timestamp(sizes.index.min()).isoformat(),
                "last_time_block": pd.Timestamp(sizes.index.max()).isoformat(),
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(difference_rows),
        pd.DataFrame(block_rows),
    )


def mcnemar_result(
    part: pd.DataFrame,
    *,
    tolerance: float = 0.10,
    method: str = "auto",
    exact_threshold: int = 25,
) -> dict[str, Any]:
    if method not in {"auto", "exact", "asymptotic"}:
        raise ValueError("McNemar method must be auto, exact, or asymptotic")
    if exact_threshold < 0:
        raise ValueError("mcnemar_exact_threshold must be non-negative")
    point, counts = _point_metrics(part, tolerance)
    proposed_only = counts["proposed_only_hit"]
    reference_only = counts["reference_only_hit"]
    discordant = proposed_only + reference_only
    selected_method = (
        "exact"
        if method == "exact" or (method == "auto" and discordant <= exact_threshold)
        else "asymptotic_continuity_corrected"
    )
    if selected_method == "exact":
        statistic = float(min(proposed_only, reference_only))
        if discordant == 0:
            p_value = 1.0
        else:
            tail = min(proposed_only, reference_only)
            log_probabilities = np.asarray(
                [
                    math.lgamma(discordant + 1)
                    - math.lgamma(index + 1)
                    - math.lgamma(discordant - index + 1)
                    - discordant * math.log(2.0)
                    for index in range(tail + 1)
                ],
                dtype=float,
            )
            maximum = float(log_probabilities.max())
            lower_tail = math.exp(maximum) * float(
                np.exp(log_probabilities - maximum).sum()
            )
            p_value = min(1.0, 2.0 * lower_tail)
    else:
        statistic = (
            0.0
            if discordant == 0
            else float((max(abs(proposed_only - reference_only) - 1, 0) ** 2) / discordant)
        )
        # A chi-square(1) survival probability is erfc(sqrt(x/2)).
        p_value = (
            1.0
            if discordant == 0
            else float(math.erfc(math.sqrt(statistic / 2.0)))
        )
    return {
        "n": int(len(part)),
        **counts,
        "discordant_n": int(discordant),
        "proposed_acc_at_10": point["proposed_acc_at_10"],
        "reference_acc_at_10": point["reference_acc_at_10"],
        "acc_difference_proposed_minus_reference": float(
            point["proposed_acc_at_10"] - point["reference_acc_at_10"]
        ),
        "acc_tolerance": float(tolerance),
        "method": selected_method,
        "exact_threshold": int(exact_threshold),
        "statistic": statistic,
        "p_value_two_sided": p_value,
    }


def mcnemar_table(
    paired: pd.DataFrame,
    *,
    tolerance: float = 0.10,
    method: str = "auto",
    exact_threshold: int = 25,
) -> pd.DataFrame:
    rows = []
    for split in sorted(paired["split"].astype(str).unique()):
        for resource in RESOURCES:
            part = paired.loc[
                (paired["split"].astype(str) == split)
                & (paired["resource"].astype(str) == resource)
                & paired["valid"].astype(bool)
            ].copy()
            if part.empty:
                raise ValueError(
                    f"No valid paired rows for McNemar: split={split}, resource={resource}"
                )
            rows.append(
                {
                    "split": split,
                    "resource": resource,
                    **mcnemar_result(
                        part,
                        tolerance=tolerance,
                        method=method,
                        exact_threshold=exact_threshold,
                    ),
                }
            )
    result = pd.DataFrame(rows)
    raw = result["p_value_two_sided"].to_numpy(float)
    order = np.argsort(raw, kind="mergesort")
    adjusted = np.empty_like(raw)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * raw[index]))
        adjusted[index] = running
    result["holm_family_size"] = total
    result["p_value_holm"] = adjusted
    result["reject_at_0_05_holm"] = result["p_value_holm"] < 0.05
    return result


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _positive_integer(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def prepare_output_directory(output_dir: Path, overwrite: bool = False) -> None:
    """Create the output directory and protect this analysis's artifact files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [
        name for name in OUTPUT_ARTIFACT_NAMES if (output_dir / name).exists()
    ]
    if conflicts and not overwrite:
        formatted = ", ".join(conflicts)
        raise FileExistsError(
            "Paired-significance output files already exist: "
            f"{formatted}. Choose another --output-dir or pass --overwrite."
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    splits = tuple(args.splits or ["test"])
    if len(set(splits)) != len(splits):
        raise ValueError("--split values must not be repeated")
    replicates = _positive_integer(
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else get_value(config, "analysis.paired_significance.bootstrap_replicates", 10_000),
        "bootstrap_replicates",
    )
    bootstrap_seed = int(
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else get_value(config, "analysis.paired_significance.bootstrap_seed", 2026)
    )
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    confidence_level = float(
        args.confidence_level
        if args.confidence_level is not None
        else get_value(config, "analysis.paired_significance.confidence_level", 0.95)
    )
    block_frequency = str(
        args.block_frequency
        or get_value(config, "analysis.paired_significance.block_frequency", "1D")
    )
    exact_threshold = int(
        args.mcnemar_exact_threshold
        if args.mcnemar_exact_threshold is not None
        else get_value(config, "analysis.paired_significance.mcnemar_exact_threshold", 25)
    )
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("Paired paper evaluation requires evaluation.acc_tolerance=0.10")

    proposed_path = resolve_path(config, args.proposed_predictions)
    reference_path = resolve_path(config, args.reference_predictions)
    for path, role in ((proposed_path, "Proposed"), (reference_path, "Reference")):
        if not path.is_file():
            raise FileNotFoundError(f"{role} prediction file not found: {path}")
    proposed = pd.read_csv(proposed_path, dtype={"sample_id": str, "job_id": str})
    reference = pd.read_csv(reference_path, dtype={"sample_id": str, "job_id": str})
    proposed_manifest, proposed_manifest_path = load_input_manifest(
        proposed_path, "Proposed"
    )
    reference_manifest, reference_manifest_path = load_input_manifest(
        reference_path, "Reference"
    )
    validate_artifact_identity(proposed, proposed_manifest, "Proposed")
    validate_artifact_identity(reference, reference_manifest, "Reference")
    data_sha256, split_fingerprint = validate_shared_provenance(
        proposed_manifest, reference_manifest
    )
    reference_column = str(args.reference_column)
    if reference_column not in reference.columns:
        raise ValueError(
            f"Reference predictions do not contain {reference_column!r}"
        )
    validate_reference_column_contract(reference_manifest, reference_column)
    if reference_column != "prediction":
        reference = reference.copy()
        reference["prediction"] = pd.to_numeric(
            reference[reference_column], errors="coerce"
        )
        reference["model"] = (
            reference["model"].astype(str) + ":" + reference_column
        )

    bundle = load_dataset(config)
    if str(bundle.data_sha256) != data_sha256:
        raise ValueError("Configured dataset differs from the prediction artifacts")
    configured_split = str(bundle.protocol.get("split_fingerprint", ""))
    if configured_split != split_fingerprint:
        raise ValueError("Configured data split differs from the prediction artifacts")

    paired = pair_prediction_frames(proposed, reference, splits=splits)
    paired = attach_submission_blocks(paired, bundle.frame, block_frequency)
    point_metrics, differences, block_summary = paired_block_bootstrap(
        paired,
        replicates=replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
        tolerance=tolerance,
    )
    mcnemar = mcnemar_table(
        paired,
        tolerance=tolerance,
        method=args.mcnemar_method,
        exact_threshold=exact_threshold,
    )

    output_dir = resolve_path(
        config,
        args.output_dir
        or get_value(
            config,
            "analysis.paired_significance.output_dir",
            "outputs/analysis/paired_significance",
        ),
    )
    prepare_output_directory(output_dir, overwrite=bool(args.overwrite))
    point_metrics.to_csv(output_dir / "paired_metrics.csv", index=False)
    differences.to_csv(output_dir / "paired_differences.csv", index=False)
    mcnemar.to_csv(output_dir / "mcnemar.csv", index=False)
    block_summary.to_csv(output_dir / "block_summary.csv", index=False)
    write_resolved_config(config, output_dir / "config.resolved.yaml")

    summary = {
        "analysis": "paired_time_block_significance",
        "difference_definition": "proposed minus reference",
        "loss_interpretation": "negative MAE/MSE/RMSE differences favor proposed",
        "acc_interpretation": "positive Acc@10 difference favors proposed",
        "selected_splits": list(splits),
        "bootstrap": {
            "method": "submission-time-block cluster bootstrap with replacement",
            "metrics": ["mae", "mse", "rmse", "acc_at_10"],
            "time_zone": "UTC",
            "block_frequency": block_frequency,
            "replicates": replicates,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "interval": "percentile",
        },
        "mcnemar": {
            "inference_role": "sensitivity_analysis",
            "requested_method": args.mcnemar_method,
            "exact_threshold": exact_threshold,
            "alternative": "two-sided",
            "asymptotic_correction": "continuity corrected",
            "multiple_comparison_correction": (
                "Holm across selected split/resource cells"
            ),
        },
        "paired_metrics": _json_records(point_metrics),
        "paired_differences": _json_records(differences),
        "mcnemar_results": _json_records(mcnemar),
        "block_summary": _json_records(block_summary),
    }
    with (output_dir / "significance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "paired_time_block_significance",
        "python": platform.python_version(),
        "dependencies": _package_versions(),
        "paired_keys": PAIR_KEYS,
        "selected_splits": list(splits),
        "resources": list(RESOURCES),
        "model_selection_performed": False,
        "overwrite_requested": bool(args.overwrite),
        "difference_definition": "proposed minus reference",
        "proposed": {
            "model": str(proposed_manifest["model"]),
            "seed": int(proposed_manifest["seed"]),
            "predictions": str(proposed_path),
            "predictions_sha256": sha256_file(proposed_path),
            "run_manifest": str(proposed_manifest_path),
            "run_manifest_sha256": sha256_file(proposed_manifest_path),
        },
        "reference": {
            "model": str(reference["model"].iloc[0]),
            "source_model": str(reference_manifest["model"]),
            "comparison_model": str(reference["model"].iloc[0]),
            "prediction_column": reference_column,
            "embedded_column_contract_validated": reference_column != "prediction",
            "seed": int(reference_manifest["seed"]),
            "predictions": str(reference_path),
            "predictions_sha256": sha256_file(reference_path),
            "run_manifest": str(reference_manifest_path),
            "run_manifest_sha256": sha256_file(reference_manifest_path),
        },
        "data_path": str(bundle.data_path),
        "data_sha256": data_sha256,
        "split_fingerprint": split_fingerprint,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "block_frequency": block_frequency,
        "block_time_zone": "UTC",
        "acc_at_10_primary_inference": "submission-time-block bootstrap interval",
        "mcnemar_requested_method": args.mcnemar_method,
        "mcnemar_inference_role": "sensitivity_analysis",
        "mcnemar_exact_threshold": exact_threshold,
        "mcnemar_multiple_comparison_correction": (
            "Holm across selected split/resource cells"
        ),
        "artifacts": {
            "paired_metrics": "paired_metrics.csv",
            "paired_differences": "paired_differences.csv",
            "mcnemar": "mcnemar.csv",
            "block_summary": "block_summary.csv",
            "summary": "significance_summary.json",
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print("Paired metric differences (proposed minus reference):")
    print(differences.to_string(index=False))
    print("McNemar tests:")
    print(mcnemar.to_string(index=False))
    print(f"Significance artifacts: {output_dir}")


if __name__ == "__main__":
    main()
