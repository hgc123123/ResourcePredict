#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from common.metrics import add_error_columns, compute_metrics  # noqa: E402
from hybrid.fixed_blend import (  # noqa: E402
    EVALUATION_SPLITS,
    RESOURCES,
    alpha_grid,
    apply_guarded_policies,
    apply_script_history_policies,
    apply_script_guarded_policies,
    apply_alphas,
    guard_budget_grid,
    pair_predictions,
    select_guarded_resource_policies,
    select_safe_script_history_policies,
    select_script_guarded_resource_policies,
    select_safe_resource_alphas,
    select_resource_alphas,
)
from hybrid.script_context import attach_script_history_relation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a per-resource Qwen/Last-1 blend on validation and evaluate test"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--safe-alpha-record")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _prediction_path(config: dict, cli_value: str | None, key: str, default: str) -> Path:
    value = cli_value or get_value(config, key, default)
    path = resolve_path(config, value)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return path


def _input_manifest(prediction_path: Path, component: str) -> dict:
    path = prediction_path.parent / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{component} predictions require their run manifest: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Input run manifest must be a JSON object: {path}")
    if not value.get("data_sha256"):
        raise ValueError(f"{component} run manifest has no data_sha256: {path}")
    return value


def _split_fingerprint(manifest: dict, component: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{component} run manifest has no split fingerprint")
    return str(value)


def _load_frozen_safe_alphas(
    path: Path,
    *,
    qwen_path: Path,
    last1_path: Path,
    qwen_variant: str,
    seed: int,
    data_sha256: str,
    split_fingerprint: str,
    constraints: dict,
    tolerance: float,
    clip_predictions: bool,
) -> tuple[dict[str, float], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Safe-hybrid alpha record not found: {path}")
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Safe-hybrid alpha record requires its run manifest: {manifest_path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(record, dict) or not isinstance(manifest, dict):
        raise ValueError("Safe-hybrid alpha artifact must contain JSON objects")
    if record.get("protocol_version") != "pareto_safe_fixed_blend_v1":
        raise ValueError("Guarded hybrid requires a Pareto-safe fixed-blend artifact")
    if manifest.get("selection_protocol_version") != record.get("protocol_version"):
        raise ValueError("Safe-hybrid alpha record and manifest protocols differ")
    if manifest.get("model") != f"safe_hybrid_{qwen_variant}":
        raise ValueError("Safe-hybrid artifact model/variant differs from this run")
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError("Safe-hybrid artifact seed differs from this run")
    if manifest.get("data_sha256") != data_sha256:
        raise ValueError("Safe-hybrid artifact dataset differs from this run")
    if manifest.get("split_fingerprint") != split_fingerprint:
        raise ValueError("Safe-hybrid artifact split differs from this run")
    if manifest.get("qwen_predictions_sha256") != sha256_file(qwen_path):
        raise ValueError("Safe-hybrid artifact used different Qwen predictions")
    if manifest.get("last1_predictions_sha256") != sha256_file(last1_path):
        raise ValueError("Safe-hybrid artifact used different Last-1 predictions")
    if manifest.get("selection_objective") != "constrained_mae":
        raise ValueError("Safe-hybrid artifact was not selected with constrained MAE")
    if record.get("selection_objective") != "constrained_mae":
        raise ValueError("Safe-hybrid alpha record has the wrong selection objective")
    if manifest.get("selection_constraints") != constraints:
        raise ValueError("Guarded and safe-hybrid constraint definitions differ")
    if record.get("constraints") != constraints:
        raise ValueError("Safe-hybrid alpha record constraints differ from this run")
    if bool(record.get("test_labels_used_for_selection", True)):
        raise ValueError("Safe-hybrid artifact does not exclude test labels")
    if bool(manifest.get("test_labels_used_for_selection", True)):
        raise ValueError("Safe-hybrid manifest does not exclude test labels")
    expected_reference = "Last-1 when causal history is available; Qwen otherwise"
    if record.get("safe_reference_policy") != expected_reference:
        raise ValueError("Safe-hybrid alpha record has the wrong reference policy")
    if manifest.get("safe_reference_policy") != expected_reference:
        raise ValueError("Safe-hybrid manifest has the wrong reference policy")
    if record.get("no_history_policy") != "Qwen only":
        raise ValueError("Safe-hybrid artifact has the wrong cold-start policy")
    if manifest.get("last1_protocol") != "completed_before_submission" or not bool(
        manifest.get("causal_last1_required", False)
    ):
        raise ValueError("Safe-hybrid artifact does not require causal Last-1 history")
    if float(record.get("acc_tolerance", float("nan"))) != tolerance:
        raise ValueError("Safe-hybrid artifact Acc tolerance differs from this run")
    if bool(record.get("clip_predictions")) != clip_predictions:
        raise ValueError("Safe-hybrid artifact clipping policy differs from this run")
    alphas = record.get("alphas")
    if not isinstance(alphas, dict) or set(alphas) != set(RESOURCES):
        raise ValueError("Safe-hybrid artifact must contain memory and vram alphas")
    frozen = {resource: float(alphas[resource]) for resource in RESOURCES}
    if any(not 0.0 <= value <= 1.0 for value in frozen.values()):
        raise ValueError("Safe-hybrid artifact contains an invalid alpha")
    if manifest.get("alphas") != alphas:
        raise ValueError("Safe-hybrid alpha record and manifest values differ")
    return frozen, {
        "alpha_record": str(path),
        "alpha_record_sha256": sha256_file(path),
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "model": manifest["model"],
        "alphas": frozen,
    }


def _validate_component_identity(
    predictions: pd.DataFrame,
    manifest: dict,
    component: str,
) -> None:
    if "model" not in predictions or "seed" not in predictions:
        raise ValueError(f"{component} predictions have no model/seed identity columns")
    if predictions["model"].isna().any():
        raise ValueError(f"{component} predictions contain missing model identities")
    models = predictions["model"].astype(str).unique().tolist()
    seeds = pd.to_numeric(predictions["seed"], errors="coerce")
    if len(models) != 1 or models[0] != str(manifest.get("model")):
        raise ValueError(
            f"{component} prediction model disagrees with its manifest: {models}"
        )
    if seeds.isna().any() or len(seeds.unique()) != 1:
        raise ValueError(f"{component} predictions must contain one finite seed")
    if int(seeds.iloc[0]) != int(manifest.get("seed")) or not (
        seeds == int(seeds.iloc[0])
    ).all():
        raise ValueError(f"{component} prediction seed disagrees with its manifest")


def _output_frame(blended: pd.DataFrame, seed: int, model_name: str) -> pd.DataFrame:
    job_column = "job_id_qwen" if "job_id_qwen" in blended else "job_id_last1"
    result = pd.DataFrame(
        {
            "sample_id": blended["sample_id"].astype(str),
            "job_id": blended[job_column].astype(str),
            "split": blended["split"].astype(str),
            "resource": blended["resource"].astype(str),
            "target": blended["target"].to_numpy(float),
            "raw_prediction": blended["raw_prediction"].to_numpy(float),
            "prediction": blended["prediction"].to_numpy(float),
            "valid": blended["valid"].to_numpy(bool),
            "model": model_name,
            "seed": int(seed),
            "qwen_prediction": blended["qwen_prediction"].to_numpy(float),
            "last1_prediction": blended["last1_prediction"].to_numpy(float),
            "history_available": blended["history_available"].to_numpy(bool),
            "used_qwen_only": blended["used_qwen_only"].to_numpy(bool),
            "selected_alpha": blended["selected_alpha"].to_numpy(float),
            "alpha": blended["alpha"].to_numpy(float),
            "effective_alpha": blended["alpha"].to_numpy(float),
        }
    )
    if "safe_reference_prediction" in blended:
        result["safe_reference_prediction"] = blended[
            "safe_reference_prediction"
        ].to_numpy(float)
    if "ordinary_safe_reference_prediction" in blended:
        result["ordinary_safe_reference_prediction"] = blended[
            "ordinary_safe_reference_prediction"
        ].to_numpy(float)
    if "selection_status" in blended:
        result["selection_status"] = blended["selection_status"].astype(str)
    for name in (
        "guard_budget",
        "guard_budget_fraction",
        "same_script_guard_budget",
        "changed_script_guard_budget",
        "same_script_history_prediction",
        "same_script_history_age_seconds",
        "selected_same_script_beta",
        "same_script_beta",
        "history_expert_prediction",
        "same_script_history_adjustment",
        "uncapped_blend_prediction",
        "candidate_delta",
        "guarded_adjustment",
    ):
        if name in blended:
            result[name] = blended[name].to_numpy(float)
    if "guard_was_active" in blended:
        result["guard_was_active"] = blended["guard_was_active"].to_numpy(bool)
    for name in ("same_script_history_available", "same_script_as_last1"):
        if name in blended:
            result[name] = blended[name].to_numpy(bool)
    if "history_script_relation" in blended:
        result["history_script_relation"] = blended[
            "history_script_relation"
        ].astype(str)
    if "same_script_history_source_sample_id" in blended:
        result["same_script_history_source_sample_id"] = blended[
            "same_script_history_source_sample_id"
        ].astype(str)
    if "user_id_qwen" in blended:
        result.insert(2, "user_id", blended["user_id_qwen"].astype(str))
    return add_error_columns(result)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed = int(get_value(config, "seed", 42))
    qwen_variant = str(get_value(config, "hybrid.qwen_variant", "full"))
    selection_objective = str(
        get_value(config, "hybrid.selection_objective", "mae")
    ).strip().lower()
    if selection_objective not in {
        "mae",
        "constrained_mae",
        "guarded_constrained_acc",
        "script_guarded_constrained_acc",
        "script_history_constrained_mae",
    }:
        raise ValueError(
            "hybrid.selection_objective must be mae, constrained_mae, or "
            "guarded_constrained_acc/script_guarded_constrained_acc/"
            "script_history_constrained_mae"
        )
    safe_selection = selection_objective in {
        "constrained_mae",
        "guarded_constrained_acc",
        "script_guarded_constrained_acc",
        "script_history_constrained_mae",
    }
    guarded_selection = selection_objective in {
        "guarded_constrained_acc",
        "script_guarded_constrained_acc",
    }
    script_guarded_selection = (
        selection_objective == "script_guarded_constrained_acc"
    )
    script_history_selection = (
        selection_objective == "script_history_constrained_mae"
    )
    last1_protocol = str(
        get_value(config, "last1.protocol", "completed_before_submission")
    )
    qwen_path = _prediction_path(
        config,
        args.qwen_predictions,
        "hybrid.qwen_predictions",
        f"outputs/qwen/{qwen_variant}/seed_{seed}/predictions.csv",
    )
    last1_path = _prediction_path(
        config,
        args.last1_predictions,
        "hybrid.last1_predictions",
        f"outputs/last-1/{last1_protocol}/seed_{seed}/predictions.csv",
    )
    output_dir = resolve_path(
        config,
        args.output_dir
        or get_value(
            config,
            (
                "hybrid.script_history_output_dir"
                if script_history_selection
                else "hybrid.script_guarded_output_dir"
                if script_guarded_selection
                else "hybrid.guarded_output_dir"
                if guarded_selection
                else "hybrid.safe_output_dir"
                if safe_selection
                else "hybrid.output_dir"
            ),
            (
                f"outputs/safe_script_history_hybrid/{last1_protocol}/{qwen_variant}/seed_{seed}"
                if script_history_selection
                else f"outputs/script_guarded_hybrid/{last1_protocol}/{qwen_variant}/seed_{seed}"
                if script_guarded_selection
                else f"outputs/guarded_hybrid/{last1_protocol}/{qwen_variant}/seed_{seed}"
                if guarded_selection
                else f"outputs/safe_hybrid/{last1_protocol}/{qwen_variant}/seed_{seed}"
                if safe_selection
                else f"outputs/hybrid/{last1_protocol}/{qwen_variant}/seed_{seed}"
            ),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen = pd.read_csv(qwen_path, dtype={"sample_id": str, "job_id": str})
    last1 = pd.read_csv(last1_path, dtype={"sample_id": str, "job_id": str})
    qwen_manifest = _input_manifest(qwen_path, "Qwen")
    last1_manifest = _input_manifest(last1_path, "Last-1")
    _validate_component_identity(qwen, qwen_manifest, "Qwen")
    _validate_component_identity(last1, last1_manifest, "Last-1")
    if int(qwen_manifest["seed"]) != seed or int(last1_manifest["seed"]) != seed:
        raise ValueError(
            "Qwen and Last-1 manifest seeds must match the configured hybrid seed"
        )
    qwen_details = qwen_manifest.get("details") or {}
    if (
        str(qwen_manifest.get("model")) != f"qwen_{qwen_variant}"
        or str(qwen_details.get("variant", "")) != qwen_variant
    ):
        raise ValueError(
            "Qwen predictions do not match hybrid.qwen_variant="
            f"{qwen_variant!r}"
        )
    qwen_hash = str(qwen_manifest["data_sha256"])
    last1_hash = str(last1_manifest["data_sha256"])
    if qwen_hash != last1_hash:
        raise ValueError("Qwen and Last-1 run manifests reference different datasets")
    qwen_split_fingerprint = _split_fingerprint(qwen_manifest, "Qwen")
    last1_split_fingerprint = _split_fingerprint(last1_manifest, "Last-1")
    if qwen_split_fingerprint != last1_split_fingerprint:
        raise ValueError("Qwen and Last-1 run manifests reference different splits")
    last1_details = last1_manifest.get("details") or {}
    if not bool(last1_details.get("causal_online_protocol", False)):
        raise ValueError(
            "The deployable hybrid requires causal completed-before-submission "
            "Last-1 predictions; ordered-row smoke-test predictions are rejected"
        )
    manifest_protocol = str(last1_details.get("protocol", ""))
    if manifest_protocol != "completed_before_submission":
        raise ValueError(
            "The Last-1 manifest does not identify completed_before_submission "
            f"history: {manifest_protocol!r}"
        )
    paired = pair_predictions(qwen, last1)
    script_context_record = None
    if script_guarded_selection or script_history_selection:
        bundle = load_dataset(config)
        if bundle.data_sha256 != qwen_hash:
            raise ValueError(
                "Configured data hash differs from script-aware component artifacts"
            )
        bundle_split_fingerprint = str(
            bundle.protocol.get("split_fingerprint", "")
        )
        if bundle_split_fingerprint != qwen_split_fingerprint:
            raise ValueError(
                "Configured split differs from script-aware component artifacts"
            )
        qwen_artifact_protocol = qwen_manifest.get("protocol") or {}
        last1_artifact_protocol = last1_manifest.get("protocol") or {}
        provenance_fields = (
            "code_redaction",
            "split_manifest_sha256",
            "order_fingerprint",
            "retained_row_count",
            "event_time_validation",
            "completed_labels_at_boundaries",
        )
        for field in provenance_fields:
            qwen_value = qwen_artifact_protocol.get(field)
            last1_value = last1_artifact_protocol.get(field)
            configured_value = bundle.protocol.get(field)
            if qwen_value is None or last1_value is None or configured_value is None:
                raise ValueError(
                    f"Script-aware replay requires {field} in both component "
                    "artifacts and the configured dataset protocol"
                )
            if not (qwen_value == last1_value == configured_value):
                raise ValueError(
                    f"Script-aware replay protocol mismatch for {field}"
                )
        bundle_code_redaction = bundle.protocol["code_redaction"]
        paired = attach_script_history_relation(paired, bundle)
        script_context_record = {
            "identity": "sha256(exact nonblank redacted script encoder input)",
            "empty_script_policy": "same-script history unavailable",
            "relation": (
                "same_script when the causal Last-1 source has the current script hash; "
                "changed_script otherwise"
            ),
            "strict_completion_relation": "completion_time < submit_time",
            "history_scope": "per user and resource",
            "event_order": "completion time, then source_row for equal timestamps",
            "code_redaction": bundle_code_redaction,
            "history_replay": paired.attrs.get(
                "causal_script_history_protocol", {}
            ),
            "artifact_protocol_binding": list(provenance_fields),
        }
        if script_guarded_selection:
            script_context_record.update(
                {
                    "min_stratum_n": int(
                        get_value(config, "hybrid.script_guard_min_stratum_n", 100)
                    ),
                    "require_changed_budget_ge_same": bool(
                        get_value(
                            config,
                            "hybrid.script_guard_require_changed_budget_ge_same",
                            True,
                        )
                    ),
                }
            )
        if script_history_selection:
            script_context_record.update(
                {
                    "same_script_expert": (
                        "most recently completed job from the same user, resource, "
                        "and exact nonblank redacted script"
                    ),
                    "min_effective_same_script_n": int(
                        get_value(
                            config,
                            "hybrid.same_script_min_effective_n",
                            100,
                        )
                    ),
                }
            )

    # The selector receives validation rows only. Test targets are used later solely
    # by compute_metrics after the selected alphas have been frozen.
    validation = paired.loc[paired["split"] == "validation"].copy()
    grid = alpha_grid(float(get_value(config, "hybrid.alpha_grid_step", 0.01)))
    clip_predictions = bool(get_value(config, "evaluation.clip_predictions", True))
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    alpha_sweep = None
    constraint_record = None
    guarded_policies = None
    script_history_policies = None
    same_script_beta_values = None
    guard_budgets = None
    safe_candidate_artifact = None
    if safe_selection:
        constraint_record = {
            "max_acc_drop_vs_reference": float(
                get_value(config, "hybrid.max_acc_drop_vs_reference", 0.0)
            ),
            "max_rmse_increase_vs_reference": float(
                get_value(config, "hybrid.max_rmse_increase_vs_reference", 0.0)
            ),
            "max_mae_increase_vs_reference": float(
                get_value(config, "hybrid.max_mae_increase_vs_reference", 0.0)
            ),
        }
        if guarded_selection:
            default_safe_dir = get_value(
                config,
                "hybrid.safe_output_dir",
                f"outputs/safe_hybrid/{last1_protocol}/{qwen_variant}/seed_{seed}",
            )
            safe_alpha_path = resolve_path(
                config,
                args.safe_alpha_record
                or get_value(
                    config,
                    "hybrid.safe_alpha_record",
                    str(Path(str(default_safe_dir)) / "alphas.json"),
                ),
            )
            frozen_safe_alphas, safe_candidate_artifact = _load_frozen_safe_alphas(
                safe_alpha_path,
                qwen_path=qwen_path,
                last1_path=last1_path,
                qwen_variant=qwen_variant,
                seed=seed,
                data_sha256=qwen_hash,
                split_fingerprint=qwen_split_fingerprint,
                constraints=constraint_record,
                tolerance=tolerance,
                clip_predictions=clip_predictions,
            )
            guard_fractions = get_value(
                config,
                "hybrid.guard_budget_fractions",
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )
            if not isinstance(guard_fractions, (list, tuple)):
                raise ValueError("hybrid.guard_budget_fractions must be a list")
            guard_budgets = guard_budget_grid(tolerance, guard_fractions)
            if script_guarded_selection:
                guarded_policies, diagnostics, alpha_sweep = (
                    select_script_guarded_resource_policies(
                        validation,
                        frozen_safe_alphas,
                        guard_budgets,
                        clip_predictions=clip_predictions,
                        tolerance=tolerance,
                        min_stratum_n=get_value(
                            config,
                            "hybrid.script_guard_min_stratum_n",
                            100,
                        ),
                        require_changed_budget_ge_same=bool(
                            get_value(
                                config,
                                "hybrid.script_guard_require_changed_budget_ge_same",
                                True,
                            )
                        ),
                        **constraint_record,
                    )
                )
            else:
                guarded_policies, diagnostics, alpha_sweep = (
                    select_guarded_resource_policies(
                        validation,
                        frozen_safe_alphas,
                        guard_budgets,
                        clip_predictions=clip_predictions,
                        tolerance=tolerance,
                        **constraint_record,
                    )
                )
            alphas = frozen_safe_alphas
        elif script_history_selection:
            configured_betas = get_value(
                config,
                "hybrid.same_script_beta_values",
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )
            if not isinstance(configured_betas, (list, tuple)):
                raise ValueError("hybrid.same_script_beta_values must be a list")
            same_script_beta_values = [float(value) for value in configured_betas]
            script_history_policies, diagnostics, alpha_sweep = (
                select_safe_script_history_policies(
                    validation,
                    grid,
                    same_script_beta_values,
                    clip_predictions=clip_predictions,
                    tolerance=tolerance,
                    min_effective_same_script_n=get_value(
                        config,
                        "hybrid.same_script_min_effective_n",
                        100,
                    ),
                    **constraint_record,
                )
            )
            alphas = {
                resource: float(policy["alpha"])
                for resource, policy in script_history_policies.items()
            }
        else:
            alphas, diagnostics, alpha_sweep = select_safe_resource_alphas(
                validation,
                grid,
                clip_predictions=clip_predictions,
                tolerance=tolerance,
                **constraint_record,
            )
    else:
        alphas, diagnostics = select_resource_alphas(
            validation,
            grid,
            clip_predictions=clip_predictions,
        )
    if script_history_selection:
        blended = apply_script_history_policies(
            paired,
            script_history_policies,
            clip_predictions=clip_predictions,
        )
    elif script_guarded_selection:
        blended = apply_script_guarded_policies(
            paired, guarded_policies, clip_predictions=clip_predictions
        )
        blended["guard_budget_fraction"] = (
            blended["guard_budget"].to_numpy(float) / tolerance
        )
    elif guarded_selection:
        blended = apply_guarded_policies(
            paired, guarded_policies, clip_predictions=clip_predictions
        )
        blended["guard_budget_fraction"] = (
            blended["guard_budget"].to_numpy(float) / tolerance
        )
    else:
        blended = apply_alphas(
            paired,
            alphas,
            clip_predictions=clip_predictions,
        )
    if safe_selection:
        ordinary_safe_reference = apply_alphas(
            paired,
            {resource: 1.0 for resource in RESOURCES},
            clip_predictions=clip_predictions,
        )
        if script_history_selection:
            reference_policies = {
                resource: dict(details["reference_policy"])
                for resource, details in diagnostics.items()
            }
            safe_reference = apply_script_history_policies(
                paired,
                reference_policies,
                clip_predictions=clip_predictions,
            )
            blended["ordinary_safe_reference_prediction"] = (
                ordinary_safe_reference["prediction"].to_numpy(float)
            )
        else:
            safe_reference = ordinary_safe_reference
        blended["safe_reference_prediction"] = safe_reference["prediction"].to_numpy(
            float
        )
        blended["selection_status"] = blended["resource"].map(
            {
                resource: str(details["selection_status"])
                for resource, details in diagnostics.items()
            }
        )
        if blended["selection_status"].isna().any():
            raise AssertionError("Safe hybrid selection status is missing")

    model_name = (
        f"safe_script_history_hybrid_{qwen_variant}"
        if script_history_selection
        else f"script_guarded_hybrid_{qwen_variant}"
        if script_guarded_selection
        else f"guarded_hybrid_{qwen_variant}"
        if guarded_selection
        else f"safe_hybrid_{qwen_variant}"
        if safe_selection
        else f"hybrid_{qwen_variant}"
    )
    predictions = _output_frame(blended, seed, model_name)
    metrics = compute_metrics(predictions, tolerance=tolerance, splits=EVALUATION_SPLITS)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    reference_metrics = None
    if script_history_selection:
        reference_rows = []
        for reference_name, prediction_column in (
            (
                "ordinary_last1_with_qwen_cold_start",
                "ordinary_safe_reference_prediction",
            ),
            (
                "hierarchical_same_script_with_qwen_cold_start",
                "safe_reference_prediction",
            ),
        ):
            reference_frame = predictions.copy()
            reference_frame["prediction"] = reference_frame[prediction_column]
            resource_metrics = compute_metrics(
                reference_frame,
                tolerance=tolerance,
                splits=EVALUATION_SPLITS,
            )
            resource_metrics.insert(0, "reference", reference_name)
            reference_rows.append(resource_metrics)
        reference_metrics = pd.concat(reference_rows, ignore_index=True)
        reference_metrics.to_csv(output_dir / "reference_metrics.csv", index=False)
    if alpha_sweep is not None:
        sweep_name = (
            "script_history_sweep.csv"
            if script_history_selection
            else "guard_sweep.csv"
            if guarded_selection
            else "alpha_sweep.csv"
        )
        alpha_sweep.to_csv(
            output_dir / sweep_name,
            index=False,
        )
    write_resolved_config(config, output_dir / "config.resolved.yaml")

    if script_history_selection:
        protocol_version = "pareto_safe_script_history_blend_v1"
        formula = (
            "H=(1-beta)*Last-1+beta*Same-Script-Last-1 when same-script "
            "history exists, otherwise H=Last-1; prediction=alpha*H+"
            "(1-alpha)*Qwen with Qwen-only cold starts"
        )
        tie_break = (
            "lowest MAE, highest Acc@tolerance, lowest RMSE, smallest beta, "
            "largest alpha"
        )
        selection_record_name = "script_history_policy.json"
    elif script_guarded_selection:
        protocol_version = "pareto_script_guarded_fixed_blend_v1"
        formula = (
            "reference + clip(alpha*Last-1 + (1-alpha)*Qwen - reference, "
            "-budget(script_relation), budget(script_relation))"
        )
        tie_break = (
            "highest Acc@tolerance, lowest MAE, lowest RMSE, smallest mean "
            "assigned history budget, smallest maximum/summed budgets"
        )
        selection_record_name = "script_guard.json"
    elif guarded_selection:
        protocol_version = "pareto_guarded_fixed_blend_v1"
        formula = (
            "reference + clip(alpha*Last-1 + (1-alpha)*Qwen - reference, "
            "-guard_budget, guard_budget)"
        )
        tie_break = (
            "highest Acc@tolerance, lowest MAE, lowest RMSE, smallest guard budget"
        )
        selection_record_name = "guard.json"
    elif safe_selection:
        protocol_version = "pareto_safe_fixed_blend_v1"
        formula = "prediction = alpha * Last-1 + (1 - alpha) * Qwen"
        tie_break = "lowest MAE, highest Acc@tolerance, lowest RMSE, largest alpha"
        selection_record_name = "alphas.json"
    else:
        protocol_version = "mae_fixed_blend_v1"
        formula = "prediction = alpha * Last-1 + (1 - alpha) * Qwen"
        tie_break = "smallest alpha (grid is ascending)"
        selection_record_name = "alphas.json"

    alpha_record = {
        "protocol_version": protocol_version,
        "formula": formula,
        "selection_split": "validation",
        "selection_objective": selection_objective,
        "selection_metric": (
            "maximum Acc@tolerance subject to safe-reference Acc/MAE/RMSE constraints"
            if guarded_selection
            else "minimum MAE subject to safe-reference Acc/MAE/RMSE constraints"
            if safe_selection
            else "mae"
        ),
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "acc_tolerance": tolerance,
        "clip_predictions": clip_predictions,
        "no_history_policy": "Qwen only",
        "safe_reference_policy": (
            (
                "Same-Script-Last-1 when available, otherwise Last-1; Qwen for "
                "cold starts (beta=0 fallback when validation support is sparse)"
                if script_history_selection
                else "Last-1 when causal history is available; Qwen otherwise"
            )
            if safe_selection
            else None
        ),
        "safe_reference_policies": (
            {
                resource: dict(details["reference_policy"])
                for resource, details in diagnostics.items()
            }
            if script_history_selection
            else None
        ),
        "constraints": constraint_record,
        "tie_break": tie_break,
        "grid": None if guarded_selection else [float(value) for value in grid],
        "frozen_candidate_alphas": alphas if guarded_selection else None,
        "guard_budget_grid": (
            [float(value) for value in guard_budgets]
            if guard_budgets is not None
            else None
        ),
        "guard_budget_fraction_grid": (
            [float(value / tolerance) for value in guard_budgets]
            if guard_budgets is not None
            else None
        ),
        "trust_region_guarantee": (
            "abs(prediction - safe_reference_prediction) <= guard_budget"
            if guarded_selection
            else None
        ),
        "script_stratification": script_context_record,
        "same_script_beta_grid": same_script_beta_values,
        "alphas": alphas,
        "script_history_policies": script_history_policies,
        "guarded_policies": guarded_policies,
        "safe_candidate_artifact": safe_candidate_artifact,
        "validation_diagnostics": diagnostics,
        "reference_metrics_artifact": (
            "reference_metrics.csv" if reference_metrics is not None else None
        ),
    }
    selection_record_path = output_dir / selection_record_name
    with selection_record_path.open("w", encoding="utf-8") as handle:
        json.dump(alpha_record, handle, indent=2, ensure_ascii=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "seed": seed,
        "python": platform.python_version(),
        "qwen_predictions": str(qwen_path),
        "qwen_predictions_sha256": sha256_file(qwen_path),
        "qwen_variant": qwen_variant,
        "qwen_experiment_contract_sha256": qwen_details.get(
            "experiment_contract_sha256"
        ),
        "qwen_effective_parameters": qwen_details.get("effective_parameters"),
        "qwen_training_runtime": qwen_details.get("training_runtime"),
        "last1_predictions": str(last1_path),
        "last1_predictions_sha256": sha256_file(last1_path),
        "data_sha256": qwen_hash,
        "split_fingerprint": qwen_split_fingerprint,
        "last1_protocol": manifest_protocol,
        "causal_last1_required": True,
        "selection_split": "validation",
        "selection_protocol_version": alpha_record["protocol_version"],
        "selection_objective": selection_objective,
        "safe_reference_policy": alpha_record["safe_reference_policy"],
        "safe_reference_policies": alpha_record["safe_reference_policies"],
        "selection_constraints": constraint_record,
        "selection_tie_break": alpha_record["tie_break"],
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "alphas": alphas,
        "same_script_beta_grid": same_script_beta_values,
        "script_history_policies": script_history_policies,
        "guarded_policies": guarded_policies,
        "safe_candidate_artifact": safe_candidate_artifact,
        "script_stratification": script_context_record,
        "validation_diagnostics": diagnostics,
        "reference_metrics_artifact": alpha_record["reference_metrics_artifact"],
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print(json.dumps(alpha_record, indent=2))
    print(metrics.to_string(index=False))
    if reference_metrics is not None:
        print("Reference metrics:")
        print(reference_metrics.to_string(index=False))
    print(f"Predictions: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
