#!/usr/bin/env python3
"""Target-independent subgroup evaluation for the three-expert hybrid."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
from common.metrics import (  # noqa: E402
    coerce_valid_mask,
    compute_metrics,
    validate_prediction_frame,
)
from hybrid.fixed_blend import (  # noqa: E402
    EVALUATION_SPLITS,
    RESOURCES,
    apply_script_history_policies,
    pair_predictions,
)
from hybrid.script_context import attach_script_history_relation  # noqa: E402
from qwen.contract import contract_sha256  # noqa: E402


SUBGROUPS = (
    "same_last1",
    "older_same_script",
    "no_same_script_history",
    "cold_start",
)
MODELS = (
    "ordinary_last1_with_qwen_cold_start",
    "hierarchical_same_script_with_qwen_cold_start",
    "qwen_only",
    "proposed",
)
OUTPUT_ARTIFACTS = (
    "subgroup_predictions.csv",
    "subgroup_metrics.csv",
    "config.resolved.yaml",
    "run_manifest.json",
)
PAIR_KEYS = ["sample_id", "split", "resource"]
REFERENCE_COLUMNS = {
    "ordinary_last1_with_qwen_cold_start": "ordinary_safe_reference_prediction",
    "hierarchical_same_script_with_qwen_cold_start": "safe_reference_prediction",
}
PROVENANCE_FIELDS = (
    "code_redaction",
    "split_manifest_sha256",
    "order_fingerprint",
    "retained_row_count",
    "event_time_validation",
    "completed_labels_at_boundaries",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen, causal Last-1, hierarchical same-script Last-1, "
            "and a frozen three-expert hybrid in target-independent history groups"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--hybrid-predictions")
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--reference-metrics")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing subgroup-analysis artifact set",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _strict_bool(values: pd.Series, name: str) -> np.ndarray:
    try:
        return coerce_valid_mask(values, name=name)
    except ValueError as exc:
        raise ValueError(f"Invalid subgroup contract: {exc}") from exc


def assign_history_subgroups(frame: pd.DataFrame) -> pd.Series:
    """Assign exhaustive groups without consulting targets or prediction errors."""
    required = {
        "history_available",
        "same_script_history_available",
        "same_script_as_last1",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Subgroup assignment lacks columns: {missing}")

    history = _strict_bool(frame["history_available"], "history_available")
    same_available = _strict_bool(
        frame["same_script_history_available"],
        "same_script_history_available",
    )
    same_last1 = _strict_bool(frame["same_script_as_last1"], "same_script_as_last1")
    if np.any(same_available & ~history):
        raise ValueError("Same-script history cannot exist without causal user history")
    if np.any(same_last1 & ~same_available):
        raise ValueError("same_script_as_last1 requires same-script history")

    masks = {
        "same_last1": history & same_available & same_last1,
        "older_same_script": history & same_available & ~same_last1,
        "no_same_script_history": history & ~same_available,
        "cold_start": ~history,
    }
    membership = np.column_stack([masks[name] for name in SUBGROUPS]).sum(axis=1)
    if not np.all(membership == 1):
        raise AssertionError("History subgroups must be mutually exclusive and exhaustive")
    values = np.select(
        [masks[name] for name in SUBGROUPS],
        list(SUBGROUPS),
        default="",
    )
    return pd.Series(values, index=frame.index, name="subgroup", dtype="string")


def build_comparison_predictions(
    frame: pd.DataFrame,
    *,
    clip_predictions: bool,
) -> pd.DataFrame:
    """Build four fixed model outputs; no target is used in any prediction rule."""
    required = {
        "sample_id",
        "split",
        "resource",
        "target",
        "valid",
        "qwen_prediction",
        "last1_prediction",
        "same_script_history_prediction",
        "history_available",
        "same_script_history_available",
        "same_script_as_last1",
        "prediction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Comparison frame lacks columns: {missing}")

    output = frame.copy()
    output["subgroup"] = assign_history_subgroups(output)
    history = _strict_bool(output["history_available"], "history_available")
    same_available = _strict_bool(
        output["same_script_history_available"],
        "same_script_history_available",
    )
    qwen = pd.to_numeric(output["qwen_prediction"], errors="coerce").to_numpy(float)
    last1 = pd.to_numeric(output["last1_prediction"], errors="coerce").to_numpy(float)
    same_script = pd.to_numeric(
        output["same_script_history_prediction"], errors="coerce"
    ).to_numpy(float)
    proposed = pd.to_numeric(output["prediction"], errors="coerce").to_numpy(float)
    if not np.isfinite(qwen).all():
        raise ValueError("Qwen predictions contain non-finite values")
    if np.any(history & ~np.isfinite(last1)):
        raise ValueError("Causal Last-1 is non-finite where history is available")
    if np.any(same_available & ~np.isfinite(same_script)):
        raise ValueError("Same-script prediction is non-finite where history is available")

    qwen_effective = np.clip(qwen, 0.0, 1.0) if clip_predictions else qwen
    ordinary = np.where(history, last1, qwen_effective)
    hierarchical = ordinary.copy()
    hierarchical[same_available] = same_script[same_available]
    if clip_predictions:
        ordinary = np.clip(ordinary, 0.0, 1.0)
        hierarchical = np.clip(hierarchical, 0.0, 1.0)

    predictions = {
        "ordinary_last1_with_qwen_cold_start": ordinary,
        "hierarchical_same_script_with_qwen_cold_start": hierarchical,
        "qwen_only": qwen_effective,
        "proposed": proposed,
    }
    identity_columns = [
        name
        for name in (
            "sample_id",
            "job_id",
            "user_id",
            "split",
            "resource",
            "target",
            "valid",
            "subgroup",
            "history_available",
            "same_script_history_available",
            "same_script_as_last1",
            "history_script_relation",
            "qwen_prediction",
            "last1_prediction",
            "same_script_history_prediction",
            "same_script_history_source_sample_id",
        )
        if name in output.columns
    ]
    frames = []
    for model in MODELS:
        part = output[identity_columns].copy()
        part["model"] = model
        part["prediction"] = predictions[model]
        part["subgroup_assignment_uses_targets"] = False
        part["model_selection_performed"] = False
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    cold = result["subgroup"].eq("cold_start")
    cold_values = result.loc[cold].pivot(
        index=[name for name in PAIR_KEYS if name in result.columns],
        columns="model",
        values="prediction",
    )
    if not cold_values.empty:
        reference = cold_values["qwen_only"].to_numpy(float)
        for model in MODELS:
            if not np.allclose(
                cold_values[model].to_numpy(float),
                reference,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"Cold-start policy for {model} is not Qwen-only")
    return result


def compute_subgroup_metrics(
    comparisons: pd.DataFrame,
    *,
    tolerance: float = 0.10,
    splits: tuple[str, ...] = EVALUATION_SPLITS,
) -> pd.DataFrame:
    """Compute fixed subgroup metrics, retaining explicit zero-sized cells."""
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("Subgroup analysis fixes Acc@10% tolerance at 0.10")
    required = {
        "split",
        "resource",
        "subgroup",
        "model",
        "target",
        "prediction",
        "valid",
    }
    missing = sorted(required.difference(comparisons.columns))
    if missing:
        raise ValueError(f"Subgroup metrics lack columns: {missing}")
    unknown_groups = set(comparisons["subgroup"].astype(str)) - set(SUBGROUPS)
    unknown_models = set(comparisons["model"].astype(str)) - set(MODELS)
    if unknown_groups:
        raise ValueError(f"Unknown subgroups: {sorted(unknown_groups)}")
    if unknown_models:
        raise ValueError(f"Unknown comparison models: {sorted(unknown_models)}")

    rows: list[dict[str, Any]] = []
    for split in splits:
        for resource in RESOURCES:
            for subgroup in SUBGROUPS:
                counts: list[int] = []
                for model in MODELS:
                    part = comparisons.loc[
                        comparisons["split"].astype(str).eq(split)
                        & comparisons["resource"].astype(str).eq(resource)
                        & comparisons["subgroup"].astype(str).eq(subgroup)
                        & comparisons["model"].astype(str).eq(model)
                    ]
                    valid = _strict_bool(part["valid"], "valid")
                    part = part.loc[valid]
                    target = pd.to_numeric(part["target"], errors="coerce").to_numpy(float)
                    prediction = pd.to_numeric(
                        part["prediction"], errors="coerce"
                    ).to_numpy(float)
                    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
                        raise ValueError(
                            "Valid subgroup rows contain non-finite target/prediction"
                        )
                    n = int(len(part))
                    counts.append(n)
                    if n:
                        error = prediction - target
                        absolute = np.abs(error)
                        squared = np.square(error)
                        hits = int(np.sum(absolute <= tolerance))
                        mae = float(np.mean(absolute))
                        mse = float(np.mean(squared))
                        rmse = float(np.sqrt(mse))
                        accuracy = float(hits / n)
                    else:
                        hits = 0
                        mae = mse = rmse = accuracy = float("nan")
                    rows.append(
                        {
                            "split": split,
                            "resource": resource,
                            "subgroup": subgroup,
                            "model": model,
                            "n": n,
                            "mae": mae,
                            "mse": mse,
                            "rmse": rmse,
                            "acc_tolerance": float(tolerance),
                            "hits_at_tolerance": hits,
                            "acc_at_10": accuracy,
                            "subgroup_assignment_uses_targets": False,
                            "model_selection_performed": False,
                        }
                    )
                if len(set(counts)) != 1:
                    raise AssertionError("All models must evaluate the same subgroup rows")
    return pd.DataFrame(rows)


def _load_json(path: Path, component: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{component} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{component} must be a JSON object: {path}")
    return value


def load_qwen_training_manifest(
    config: dict[str, Any],
    qwen_path: Path,
    qwen_manifest: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Locate and validate training provenance for legacy Qwen evaluations."""
    candidates = [qwen_path.parent / "checkpoint" / "training_manifest.json"]
    details = qwen_manifest.get("details") or {}
    declared_checkpoint = details.get("checkpoint") if isinstance(details, Mapping) else None
    if declared_checkpoint:
        checkpoint = resolve_path(config, str(declared_checkpoint))
        candidates.append(
            checkpoint
            if checkpoint.name == "training_manifest.json"
            else checkpoint / "training_manifest.json"
        )

    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    if not existing:
        return None, None
    loaded_candidates = [
        (path, _load_json(path, "Qwen training manifest")) for path in existing
    ]
    manifest = loaded_candidates[0][1]
    if any(candidate != manifest for _, candidate in loaded_candidates[1:]):
        raise ValueError(
            "Qwen run directory and declared checkpoint contain different "
            "training manifests"
        )

    path = existing[0]
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError("Qwen training manifest seed differs from the evaluated run")
    if str(manifest.get("variant", "")) != variant:
        raise ValueError("Qwen training manifest variant differs from the evaluated run")
    for field in ("data_sha256", "split_fingerprint"):
        evaluated = qwen_manifest.get(field)
        trained = manifest.get(field)
        if not evaluated or not trained or str(evaluated) != str(trained):
            raise ValueError(
                f"Qwen training and evaluation manifests disagree on {field}"
            )
    contract = manifest.get("experiment_contract")
    if isinstance(contract, Mapping) and isinstance(contract.get("payload"), Mapping):
        payload = contract["payload"]
        if int(contract.get("version", -1)) != int(
            payload.get("contract_version", -2)
        ):
            raise ValueError("Qwen training contract version fields disagree")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError("Qwen training contract seed differs from the evaluated run")
        semantics = payload.get("qwen_semantics") or {}
        if not isinstance(semantics, Mapping) or semantics.get("variant") != variant:
            raise ValueError("Qwen training contract variant differs from the evaluated run")
        for field in ("enabled_modalities", "fusion_mode"):
            trained = manifest.get(field)
            contracted = semantics.get(field)
            evaluated = details.get(field) if isinstance(details, Mapping) else None
            if trained != contracted:
                raise ValueError(
                    f"Qwen training manifest and contract disagree on {field}"
                )
            if evaluated is not None and evaluated != trained:
                raise ValueError(
                    f"Qwen evaluation and training manifests disagree on {field}"
                )
    return manifest, path


def _split_fingerprint(manifest: Mapping[str, Any], component: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{component} manifest has no split fingerprint")
    return str(value)


def _validate_identity(
    predictions: pd.DataFrame,
    manifest: Mapping[str, Any],
    component: str,
) -> None:
    validate_prediction_frame(predictions)
    models = predictions["model"].astype(str).unique().tolist()
    seeds = pd.to_numeric(predictions["seed"], errors="coerce")
    if models != [str(manifest.get("model"))]:
        raise ValueError(f"{component} model identity disagrees with its manifest")
    if seeds.isna().any() or len(seeds.unique()) != 1:
        raise ValueError(f"{component} predictions must contain one finite seed")
    if int(seeds.iloc[0]) != int(manifest.get("seed", -1)):
        raise ValueError(f"{component} seed disagrees with its manifest")


def _require_hierarchical_reference(
    manifest: Mapping[str, Any],
    policy_record: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Require the saved reference to be the true alpha=1, beta=1 hierarchy."""
    policies = manifest.get("safe_reference_policies")
    record_policies = policy_record.get("safe_reference_policies")
    if policies != record_policies:
        raise ValueError(
            "Hybrid manifest and policy record disagree on safe-reference policies"
        )
    if not isinstance(policies, dict) or set(policies) != set(RESOURCES):
        raise ValueError("Safe-reference policies must cover memory and vram")

    normalized: dict[str, dict[str, float]] = {}
    for resource in RESOURCES:
        policy = policies[resource]
        if not isinstance(policy, Mapping):
            raise ValueError(f"Invalid safe-reference policy for {resource}")
        alpha = float(policy.get("alpha", float("nan")))
        beta = float(policy.get("same_script_beta", float("nan")))
        if not np.isclose(alpha, 1.0, rtol=0.0, atol=1e-12) or not np.isclose(
            beta, 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "The saved safe reference is not hierarchical Same-Script-Last-1 "
                f"for {resource}: alpha={alpha!r}, same_script_beta={beta!r}. "
                "Do not label a beta=0 sparse-support fallback as hierarchical."
            )
        normalized[resource] = {"alpha": alpha, "same_script_beta": beta}
    return normalized


def validate_selection_contract(
    manifest: Mapping[str, Any],
    policy_record: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Reject target-selected, test-selected, or non-three-expert artifacts."""
    expected_model = f"safe_script_history_hybrid_{variant}"
    required_manifest = {
        "model": expected_model,
        "seed": seed,
        "selection_protocol_version": "pareto_safe_script_history_blend_v1",
        "selection_objective": "script_history_constrained_mae",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "last1_protocol": "completed_before_submission",
        "causal_last1_required": True,
    }
    for name, expected in required_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"Hybrid manifest violates {name}: expected {expected!r}, "
                f"found {manifest.get(name)!r}"
            )
    for flag in ("uses_targets", "diagnostic_only", "not_a_deployable_model"):
        if bool(manifest.get(flag, False)):
            raise ValueError(f"Target-using/diagnostic hybrid artifact is forbidden: {flag}")

    required_record = {
        "protocol_version": "pareto_safe_script_history_blend_v1",
        "selection_objective": "script_history_constrained_mae",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "no_history_policy": "Qwen only",
    }
    for name, expected in required_record.items():
        if policy_record.get(name) != expected:
            raise ValueError(
                f"Hybrid policy record violates {name}: expected {expected!r}, "
                f"found {policy_record.get(name)!r}"
            )
    for flag in ("uses_targets", "diagnostic_only", "not_a_deployable_model"):
        if bool(policy_record.get(flag, False)):
            raise ValueError(f"Target-using/diagnostic policy is forbidden: {flag}")
    if not isinstance(policy_record.get("clip_predictions"), bool):
        raise ValueError("Hybrid policy record must declare clip_predictions")
    if not np.isclose(
        float(policy_record.get("acc_tolerance", float("nan"))),
        0.10,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Hybrid policy record is not defined for Acc@10%")
    policies = manifest.get("script_history_policies")
    if policies != policy_record.get("script_history_policies"):
        raise ValueError("Hybrid manifest and policy record disagree on frozen policies")
    if not isinstance(policies, dict) or set(policies) != set(RESOURCES):
        raise ValueError("Frozen script-history policies must cover memory and vram")
    normalized: dict[str, dict[str, float]] = {}
    for resource in RESOURCES:
        policy = policies[resource]
        if not isinstance(policy, Mapping):
            raise ValueError(f"Invalid frozen policy for {resource}")
        alpha = float(policy.get("alpha", float("nan")))
        beta = float(policy.get("same_script_beta", float("nan")))
        if not np.isfinite([alpha, beta]).all() or not (
            0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0
        ):
            raise ValueError(f"Invalid alpha/beta for {resource}")
        normalized[resource] = {"alpha": alpha, "same_script_beta": beta}
    _require_hierarchical_reference(manifest, policy_record)
    return normalized


def qwen_provenance(
    qwen_manifest: Mapping[str, Any],
    hybrid_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract Qwen provenance, recovering legacy evaluation fields from training."""
    details = qwen_manifest.get("details") or {}
    if not isinstance(details, Mapping):
        raise ValueError("Qwen manifest details must be a mapping")
    training = training_manifest or {}
    if not isinstance(training, Mapping):
        raise ValueError("Qwen training manifest must be a mapping")

    effective_candidates = [
        ("Qwen evaluation manifest", details.get("effective_parameters")),
        ("Qwen training manifest", training.get("effective_parameters")),
    ]
    effective_sources = [
        (name, dict(value))
        for name, value in effective_candidates
        if isinstance(value, Mapping)
    ]
    if not effective_sources:
        raise ValueError(
            "Qwen manifests lack effective training parameters; expected either "
            "run_manifest.json details.effective_parameters or "
            "checkpoint/training_manifest.json effective_parameters"
        )
    effective_source, effective = effective_sources[0]
    for name, candidate in effective_sources[1:]:
        if candidate != effective:
            raise ValueError(
                f"{name} disagrees with {effective_source} on effective parameters"
            )
    try:
        max_tokens = int(effective.get("max_tokens"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Qwen effective max_tokens must be a positive integer") from exc
    if isinstance(effective.get("max_tokens"), bool) or max_tokens <= 0:
        raise ValueError("Qwen effective max_tokens must be a positive integer")
    if float(effective["max_tokens"]) != float(max_tokens):
        raise ValueError("Qwen effective max_tokens must be an integer")

    saved_contract = training.get("experiment_contract")
    training_contract_sha: str | None = None
    if saved_contract is not None:
        if not isinstance(saved_contract, Mapping) or not isinstance(
            saved_contract.get("payload"), Mapping
        ):
            raise ValueError("Qwen training manifest has a malformed experiment contract")
        training_contract_sha = str(saved_contract.get("sha256", ""))
        expected_contract_sha = contract_sha256(saved_contract["payload"])
        if training_contract_sha != expected_contract_sha:
            raise ValueError("Qwen training experiment-contract fingerprint is corrupt")
        contract_effective = saved_contract["payload"].get(
            "effective_qwen_parameters"
        )
        if not isinstance(contract_effective, Mapping):
            raise ValueError(
                "Qwen training experiment contract lacks effective_qwen_parameters"
            )
        if dict(contract_effective) != effective:
            raise ValueError(
                "Qwen effective parameters disagree with the training experiment contract"
            )

    detail_contract_sha = details.get("experiment_contract_sha256")
    if detail_contract_sha is not None and training_contract_sha is not None and str(
        detail_contract_sha
    ) != training_contract_sha:
        raise ValueError(
            "Qwen evaluation and training manifests use different experiment contracts"
        )
    contract_sha = (
        str(detail_contract_sha)
        if detail_contract_sha is not None
        else training_contract_sha
    )
    if not isinstance(contract_sha, str) or len(contract_sha) != 64 or any(
        character not in "0123456789abcdef" for character in contract_sha.lower()
    ):
        raise ValueError(
            "Qwen evaluation/training manifests lack a valid experiment-contract SHA-256"
        )

    runtime_candidates = [
        ("Qwen evaluation manifest", details.get("training_runtime")),
        ("Qwen training manifest", training.get("training_runtime")),
    ]
    runtime_sources = [
        (name, dict(value))
        for name, value in runtime_candidates
        if isinstance(value, Mapping)
    ]
    runtime_source: str | None = None
    runtime: dict[str, Any] | None = None
    if runtime_sources:
        runtime_source, runtime = runtime_sources[0]
        for name, candidate in runtime_sources[1:]:
            if candidate != runtime:
                raise ValueError(
                    f"{name} disagrees with {runtime_source} on training runtime"
                )

    required_bindings = (
        ("qwen_effective_parameters", effective),
        ("qwen_experiment_contract_sha256", contract_sha),
    )
    for name, expected in required_bindings:
        recorded = hybrid_manifest.get(name)
        if recorded is not None and recorded != expected:
            raise ValueError(f"Hybrid and Qwen manifests disagree on {name}")
    recorded_runtime = hybrid_manifest.get("qwen_training_runtime")
    if recorded_runtime is not None:
        if runtime is None:
            if not isinstance(recorded_runtime, Mapping):
                raise ValueError("Hybrid qwen_training_runtime must be a mapping")
            runtime = dict(recorded_runtime)
            runtime_source = "Hybrid manifest"
        elif recorded_runtime != runtime:
            raise ValueError("Hybrid and Qwen manifests disagree on qwen_training_runtime")
    return {
        "max_tokens": max_tokens,
        "experiment_contract_sha256": contract_sha,
        "effective_parameters": effective,
        "training_runtime": runtime,
        "effective_parameters_source": effective_source,
        "training_runtime_source": runtime_source,
        "training_runtime_available": runtime is not None,
    }


def qwen_numeric_feature_provenance(
    training_manifest: Mapping[str, Any] | None,
    current_features: list[str],
) -> dict[str, Any]:
    """Verify saved metadata-column order without inventing legacy provenance."""
    if training_manifest is None or "numeric_features" not in training_manifest:
        return {
            "numeric_features": None,
            "numeric_features_source": None,
            "numeric_features_verified": False,
        }
    saved = training_manifest["numeric_features"]
    if not isinstance(saved, list) or any(
        not isinstance(name, str) or not name for name in saved
    ):
        raise ValueError(
            "Qwen training manifest numeric_features must be a list of non-empty strings"
        )
    if len(saved) != len(set(saved)):
        raise ValueError("Qwen training manifest numeric_features contains duplicates")
    if saved != list(current_features):
        raise ValueError(
            "Qwen training manifest numeric_features differ from the configured dataset"
        )
    return {
        "numeric_features": list(saved),
        "numeric_features_source": "Qwen training manifest",
        "numeric_features_verified": True,
    }


def _assert_numeric_match(
    actual: pd.Series | np.ndarray,
    expected: pd.Series | np.ndarray,
    name: str,
) -> None:
    left = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(float)
    right = pd.to_numeric(pd.Series(expected), errors="coerce").to_numpy(float)
    if left.shape != right.shape or not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise ValueError(f"{name} finite-value pattern differs")
    finite = np.isfinite(left)
    if not np.allclose(left[finite], right[finite], rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} values differ")


def _aligned(frame: pd.DataFrame, keys: pd.DataFrame, component: str) -> pd.DataFrame:
    scoped = frame.copy()
    for name in PAIR_KEYS:
        scoped[name] = scoped[name].astype(str)
    if scoped.duplicated(PAIR_KEYS).any():
        raise ValueError(f"{component} contains duplicate prediction keys")
    wanted = pd.MultiIndex.from_frame(keys[PAIR_KEYS].astype(str))
    observed = pd.MultiIndex.from_frame(scoped[PAIR_KEYS])
    if len(wanted.difference(observed)) or len(observed.difference(wanted)):
        raise ValueError(f"{component} prediction keys differ from component artifacts")
    return scoped.set_index(PAIR_KEYS).loc[wanted].reset_index()


def _global_reference_metrics(
    hybrid: pd.DataFrame,
    prediction_column: str,
    reference: str,
    tolerance: float,
) -> pd.DataFrame:
    frame = hybrid.copy()
    frame["prediction"] = pd.to_numeric(frame[prediction_column], errors="coerce")
    metrics = compute_metrics(frame, tolerance=tolerance, splits=EVALUATION_SPLITS)
    metrics.insert(0, "reference", reference)
    return metrics


def audit_reference_metrics(
    hybrid: pd.DataFrame,
    artifact: pd.DataFrame,
    *,
    tolerance: float,
) -> None:
    expected = pd.concat(
        [
            _global_reference_metrics(hybrid, column, reference, tolerance)
            for reference, column in REFERENCE_COLUMNS.items()
        ],
        ignore_index=True,
    )
    keys = ["reference", "split", "resource"]
    if artifact.duplicated(keys).any():
        raise ValueError("reference_metrics.csv contains duplicate rows")
    left = artifact.copy().sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    if not left[keys].astype(str).equals(right[keys].astype(str)):
        raise ValueError("reference_metrics.csv has unexpected reference/split/resource rows")
    for name in (
        "n",
        "mae",
        "mse",
        "rmse",
        "acc_tolerance",
        "hits_at_tolerance",
        "acc_at_10",
    ):
        if name not in left:
            raise ValueError(f"reference_metrics.csv lacks {name}")
        _assert_numeric_match(left[name], right[name], f"reference_metrics.{name}")


def _resolve_existing(config: dict, value: str | Path, component: str) -> Path:
    path = resolve_path(config, value)
    if not path.is_file():
        raise FileNotFoundError(f"{component} not found: {path}")
    return path


def _source_path(
    config: dict,
    explicit: str | None,
    manifest: Mapping[str, Any],
    key: str,
    component: str,
) -> Path:
    value = explicit or manifest.get(key)
    if not value:
        raise ValueError(
            f"Hybrid manifest has no {key}; provide --{key.replace('_', '-')}"
        )
    return _resolve_existing(config, str(value), component)


def default_analysis_output(
    config: dict[str, Any],
    *,
    protocol: str,
    variant: str,
    seed: int,
    max_tokens: int,
) -> Path:
    root = Path(str(get_value(config, "outputs.root", "outputs")))
    return (
        root
        / "analysis"
        / "hybrid_subgroups"
        / protocol
        / f"tokens_{max_tokens}"
        / variant
        / f"seed_{seed}"
    )


def prepare_output_directory(
    output_dir: Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Prevent analysis reruns from replacing inputs or prior results silently."""
    output_dir = output_dir.resolve()
    output_paths = {
        (output_dir / filename).resolve() for filename in OUTPUT_ARTIFACTS
    }
    protected = {path.resolve() for path in protected_paths}
    collisions = sorted(str(path) for path in output_paths.intersection(protected))
    if collisions:
        raise ValueError(
            "Subgroup output would overwrite an input artifact: "
            + ", ".join(collisions)
        )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(sorted(str(path) for path in existing))
        raise FileExistsError(
            "Subgroup artifacts already exist; use --overwrite only after "
            f"verifying the run identity: {rendered}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed = int(get_value(config, "seed", 42))
    variant = str(get_value(config, "hybrid.qwen_variant", "full"))
    protocol = str(get_value(config, "last1.protocol", "completed_before_submission"))
    configured_max_tokens = int(get_value(config, "qwen.max_tokens", 2048))
    output_root = Path(str(get_value(config, "outputs.root", "outputs")))
    default_hybrid_dir = get_value(
        config,
        "hybrid.script_history_output_dir",
        str(
            output_root
            / "safe_script_history_hybrid"
            / protocol
            / f"tokens_{configured_max_tokens}"
            / variant
            / f"seed_{seed}"
        ),
    )
    hybrid_path = _resolve_existing(
        config,
        args.hybrid_predictions or str(Path(str(default_hybrid_dir)) / "predictions.csv"),
        "Three-expert hybrid predictions",
    )
    hybrid_dir = hybrid_path.parent
    hybrid_manifest_path = hybrid_dir / "run_manifest.json"
    policy_path = hybrid_dir / "script_history_policy.json"
    hybrid_manifest = _load_json(hybrid_manifest_path, "Hybrid run manifest")
    policy_record = _load_json(policy_path, "Frozen script-history policy")
    policies = validate_selection_contract(
        hybrid_manifest,
        policy_record,
        variant=variant,
        seed=seed,
    )
    clip_predictions = bool(policy_record.get("clip_predictions"))
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("Subgroup analysis requires evaluation.acc_tolerance=0.10")

    qwen_path = _source_path(
        config,
        args.qwen_predictions,
        hybrid_manifest,
        "qwen_predictions",
        "Qwen predictions",
    )
    last1_path = _source_path(
        config,
        args.last1_predictions,
        hybrid_manifest,
        "last1_predictions",
        "Last-1 predictions",
    )
    reference_path = _resolve_existing(
        config,
        args.reference_metrics or str(hybrid_dir / "reference_metrics.csv"),
        "Hybrid reference metrics",
    )
    if sha256_file(qwen_path) != hybrid_manifest.get("qwen_predictions_sha256"):
        raise ValueError("Qwen prediction hash differs from the hybrid manifest")
    if sha256_file(last1_path) != hybrid_manifest.get("last1_predictions_sha256"):
        raise ValueError("Last-1 prediction hash differs from the hybrid manifest")

    hybrid = pd.read_csv(hybrid_path, dtype={"sample_id": str, "job_id": str})
    qwen = pd.read_csv(qwen_path, dtype={"sample_id": str, "job_id": str})
    last1 = pd.read_csv(last1_path, dtype={"sample_id": str, "job_id": str})
    qwen_manifest_path = qwen_path.parent / "run_manifest.json"
    last1_manifest_path = last1_path.parent / "run_manifest.json"
    qwen_manifest = _load_json(qwen_manifest_path, "Qwen manifest")
    last1_manifest = _load_json(
        last1_manifest_path, "Last-1 manifest"
    )
    _validate_identity(hybrid, hybrid_manifest, "Hybrid")
    _validate_identity(qwen, qwen_manifest, "Qwen")
    _validate_identity(last1, last1_manifest, "Last-1")
    if int(qwen_manifest.get("seed", -1)) != seed or int(
        last1_manifest.get("seed", -1)
    ) != seed:
        raise ValueError("All component seeds must match the configured seed")
    if qwen_manifest.get("model") != f"qwen_{variant}" or (
        qwen_manifest.get("details") or {}
    ).get("variant") != variant:
        raise ValueError("Qwen component does not match the configured variant")
    qwen_training_manifest, qwen_training_manifest_path = load_qwen_training_manifest(
        config,
        qwen_path,
        qwen_manifest,
        variant=variant,
        seed=seed,
    )
    qwen_run = qwen_provenance(
        qwen_manifest,
        hybrid_manifest,
        qwen_training_manifest,
    )
    last1_details = last1_manifest.get("details") or {}
    if not bool(last1_details.get("causal_online_protocol", False)) or str(
        last1_details.get("protocol", "")
    ) != "completed_before_submission":
        raise ValueError("Last-1 component is not strict completed-before-submission replay")

    data_hashes = {
        str(hybrid_manifest.get("data_sha256", "")),
        str(qwen_manifest.get("data_sha256", "")),
        str(last1_manifest.get("data_sha256", "")),
    }
    split_hashes = {
        _split_fingerprint(hybrid_manifest, "Hybrid"),
        _split_fingerprint(qwen_manifest, "Qwen"),
        _split_fingerprint(last1_manifest, "Last-1"),
    }
    if len(data_hashes) != 1 or "" in data_hashes:
        raise ValueError("Hybrid/Qwen/Last-1 artifacts reference different datasets")
    if len(split_hashes) != 1:
        raise ValueError("Hybrid/Qwen/Last-1 artifacts reference different splits")

    bundle = load_dataset(config)
    if bundle.data_sha256 not in data_hashes:
        raise ValueError("Configured dataset hash differs from component artifacts")
    if str(bundle.protocol.get("split_fingerprint", "")) not in split_hashes:
        raise ValueError("Configured split differs from component artifacts")
    qwen_numeric_features = qwen_numeric_feature_provenance(
        qwen_training_manifest,
        list(bundle.numeric_features),
    )
    qwen_protocol = qwen_manifest.get("protocol") or {}
    last1_protocol = last1_manifest.get("protocol") or {}
    for field in PROVENANCE_FIELDS:
        values = (qwen_protocol.get(field), last1_protocol.get(field), bundle.protocol.get(field))
        if any(value is None for value in values) or not (values[0] == values[1] == values[2]):
            raise ValueError(f"Script replay protocol mismatch for {field}")

    replay = attach_script_history_relation(pair_predictions(qwen, last1), bundle)
    hybrid = _aligned(hybrid, replay[PAIR_KEYS], "Hybrid")
    if not np.array_equal(
        _strict_bool(hybrid["valid"], "valid"),
        _strict_bool(replay["valid"], "valid"),
    ):
        raise ValueError("Hybrid validity mask differs from component artifacts")
    for hybrid_name, replay_name in (
        ("job_id", "job_id_qwen"),
        ("user_id", "user_id_qwen"),
    ):
        if hybrid_name in hybrid and replay_name in replay:
            if not np.array_equal(
                hybrid[hybrid_name].astype(str).to_numpy(),
                replay[replay_name].astype(str).to_numpy(),
            ):
                raise ValueError(
                    f"Hybrid {hybrid_name} differs from component artifacts"
                )
    for name, expected in (
        ("target", replay["target"]),
        ("qwen_prediction", replay["qwen_prediction"]),
        ("last1_prediction", replay["last1_prediction"]),
        ("same_script_history_prediction", replay["same_script_history_prediction"]),
    ):
        if name not in hybrid:
            raise ValueError(f"Hybrid predictions lack audited column {name}")
        _assert_numeric_match(hybrid[name], expected, name)
    for name in (
        "history_available",
        "same_script_history_available",
        "same_script_as_last1",
    ):
        if name not in hybrid:
            raise ValueError(f"Hybrid predictions lack audited column {name}")
        if not np.array_equal(_strict_bool(hybrid[name], name), _strict_bool(replay[name], name)):
            raise ValueError(f"Hybrid {name} differs from independent causal replay")

    reconstructed = apply_script_history_policies(
        replay,
        policies,
        clip_predictions=clip_predictions,
    )
    _assert_numeric_match(hybrid["prediction"], reconstructed["prediction"], "Proposed")
    ordinary_policies = {
        resource: {"alpha": 1.0, "same_script_beta": 0.0}
        for resource in RESOURCES
    }
    ordinary = apply_script_history_policies(
        replay, ordinary_policies, clip_predictions=clip_predictions
    )["prediction"]
    for column in REFERENCE_COLUMNS.values():
        if column not in hybrid:
            raise ValueError(f"Hybrid predictions lack audited column {column}")
    _assert_numeric_match(
        hybrid["ordinary_safe_reference_prediction"], ordinary, "ordinary reference"
    )
    safe_reference_policies = hybrid_manifest.get("safe_reference_policies")
    if not isinstance(safe_reference_policies, dict):
        raise ValueError("Hybrid manifest lacks frozen safe-reference policies")
    artifact_reference = apply_script_history_policies(
        replay,
        safe_reference_policies,
        clip_predictions=clip_predictions,
    )["prediction"]
    _assert_numeric_match(
        hybrid["safe_reference_prediction"], artifact_reference, "safe reference"
    )
    audit_reference_metrics(
        hybrid,
        pd.read_csv(reference_path),
        tolerance=tolerance,
    )

    comparisons = build_comparison_predictions(
        hybrid,
        clip_predictions=clip_predictions,
    )
    metrics = compute_subgroup_metrics(comparisons, tolerance=tolerance)
    output_dir = resolve_path(
        config,
        args.output_dir
        or get_value(
            config,
            "analysis.hybrid_subgroups_output_dir",
            str(
                default_analysis_output(
                    config,
                    protocol=protocol,
                    variant=variant,
                    seed=seed,
                    max_tokens=int(qwen_run["max_tokens"]),
                )
            ),
        ),
    )
    prepare_output_directory(
        output_dir,
        overwrite=bool(args.overwrite),
        protected_paths=tuple(
            path
            for path in (
                Path(str(config["_config_path"])),
                hybrid_path,
                hybrid_manifest_path,
                policy_path,
                qwen_path,
                qwen_manifest_path,
                qwen_training_manifest_path,
                last1_path,
                last1_manifest_path,
                reference_path,
            )
            if path is not None
        ),
    )
    comparisons.to_csv(output_dir / "subgroup_predictions.csv", index=False)
    metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    write_resolved_config(config, output_dir / "config.resolved.yaml")
    output_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "target_independent_causal_history_subgroups",
        "model": str(hybrid_manifest["model"]),
        "seed": seed,
        "qwen_variant": variant,
        "qwen_max_tokens": int(qwen_run["max_tokens"]),
        "qwen_experiment_contract_sha256": qwen_run[
            "experiment_contract_sha256"
        ],
        "qwen_effective_parameters": qwen_run["effective_parameters"],
        "qwen_training_runtime": qwen_run["training_runtime"],
        "qwen_effective_parameters_source": qwen_run[
            "effective_parameters_source"
        ],
        "qwen_training_runtime_source": qwen_run["training_runtime_source"],
        "qwen_training_runtime_available": qwen_run[
            "training_runtime_available"
        ],
        "qwen_numeric_features": qwen_numeric_features["numeric_features"],
        "qwen_numeric_features_source": qwen_numeric_features[
            "numeric_features_source"
        ],
        "qwen_numeric_features_verified": qwen_numeric_features[
            "numeric_features_verified"
        ],
        "python": platform.python_version(),
        "data_sha256": next(iter(data_hashes)),
        "split_fingerprint": next(iter(split_hashes)),
        "hybrid_predictions": str(hybrid_path),
        "hybrid_predictions_sha256": sha256_file(hybrid_path),
        "hybrid_manifest": str(hybrid_manifest_path),
        "hybrid_manifest_sha256": sha256_file(hybrid_manifest_path),
        "policy_record": str(policy_path),
        "policy_record_sha256": sha256_file(policy_path),
        "qwen_predictions": str(qwen_path),
        "qwen_predictions_sha256": sha256_file(qwen_path),
        "qwen_manifest": str(qwen_manifest_path),
        "qwen_manifest_sha256": sha256_file(qwen_manifest_path),
        "qwen_training_manifest": (
            str(qwen_training_manifest_path)
            if qwen_training_manifest_path is not None
            else None
        ),
        "qwen_training_manifest_sha256": (
            sha256_file(qwen_training_manifest_path)
            if qwen_training_manifest_path is not None
            else None
        ),
        "last1_predictions": str(last1_path),
        "last1_predictions_sha256": sha256_file(last1_path),
        "last1_manifest": str(last1_manifest_path),
        "last1_manifest_sha256": sha256_file(last1_manifest_path),
        "reference_metrics": str(reference_path),
        "reference_metrics_sha256": sha256_file(reference_path),
        "hierarchical_reference_policies": safe_reference_policies,
        "selection_contract_verified": True,
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "subgroup_assignment_uses_targets": False,
        "targets_used_only_for_metrics": True,
        "model_selection_performed": False,
        "overwrite_authorized": bool(args.overwrite),
        "causal_last1_required": True,
        "last1_protocol": "completed_before_submission",
        "subgroups": list(SUBGROUPS),
        "models": list(MODELS),
        "acc_tolerance": tolerance,
        "zero_sized_subgroups_retained": True,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2, ensure_ascii=True)
    print(metrics.to_string(index=False))
    print(f"Subgroup artifacts: {output_dir}")


if __name__ == "__main__":
    main()
