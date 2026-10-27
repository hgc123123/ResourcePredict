#!/usr/bin/env python3
"""Target-using upper-bound diagnostics for Qwen/Last-1 complementarity."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path, write_resolved_config  # noqa: E402
from common.data import sha256_file  # noqa: E402
from common.metrics import add_error_columns, compute_metrics  # noqa: E402
from hybrid.fixed_blend import EVALUATION_SPLITS, pair_predictions  # noqa: E402


DIAGNOSTIC_FLAGS = {
    "uses_targets": True,
    "diagnostic_only": True,
    "not_a_deployable_model": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute target-using choice and line-segment oracle diagnostics for "
            "paired Qwen/causal-Last-1 predictions"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _prediction_path(config: dict, cli_value: str | None, key: str, default: str) -> Path:
    path = resolve_path(config, cli_value or get_value(config, key, default))
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
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not manifest.get("data_sha256"):
        raise ValueError(f"Invalid {component} run manifest: {path}")
    return manifest


def _split_fingerprint(manifest: dict, component: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{component} run manifest has no split fingerprint")
    return str(value)


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
    manifest_seed = int(manifest.get("seed"))
    if int(seeds.iloc[0]) != manifest_seed or not (seeds == manifest_seed).all():
        raise ValueError(f"{component} prediction seed disagrees with its manifest")


def _base_oracle_frame(
    paired: pd.DataFrame,
    prediction: np.ndarray,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    job_column = "job_id_qwen" if "job_id_qwen" in paired else "job_id_last1"
    output = pd.DataFrame(
        {
            "sample_id": paired["sample_id"].astype(str),
            "job_id": paired[job_column].astype(str),
            "split": paired["split"].astype(str),
            "resource": paired["resource"].astype(str),
            "target": paired["target"].to_numpy(float),
            "raw_prediction": prediction,
            "prediction": prediction,
            "valid": paired["valid"].to_numpy(bool),
            "model": model_name,
            "seed": int(seed),
            "qwen_prediction": paired["qwen_prediction"].to_numpy(float),
            "last1_prediction": paired["last1_prediction"].to_numpy(float),
            "history_available": paired["history_available"].to_numpy(bool),
            "uses_targets": True,
            "diagnostic_only": True,
            "not_a_deployable_model": True,
        }
    )
    if "user_id_qwen" in paired:
        output.insert(2, "user_id", paired["user_id_qwen"].astype(str))
    return add_error_columns(output)


def choice_oracle_predictions(
    paired: pd.DataFrame,
    seed: int,
    variant: str,
) -> pd.DataFrame:
    """Choose the lower-error endpoint per valid row using its true target."""
    target = paired["target"].to_numpy(float)
    qwen = paired["qwen_prediction"].to_numpy(float)
    last1 = paired["last1_prediction"].to_numpy(float)
    valid = paired["valid"].to_numpy(bool) & np.isfinite(target)
    history = paired["history_available"].to_numpy(bool)

    # Ties and rows without causal history deterministically use Qwen.
    choose_last1 = valid & history & (np.abs(last1 - target) < np.abs(qwen - target))
    prediction = np.where(choose_last1, last1, qwen)
    output = _base_oracle_frame(
        paired,
        prediction,
        model_name=f"diagnostic_choice_oracle_{variant}",
        seed=seed,
    )
    output["oracle_choice"] = np.where(choose_last1, "last1", "qwen")
    return output


def line_segment_oracle_predictions(
    paired: pd.DataFrame,
    seed: int,
    variant: str,
) -> pd.DataFrame:
    """Project each target onto the numeric segment between the two predictions."""
    target = paired["target"].to_numpy(float)
    qwen = paired["qwen_prediction"].to_numpy(float)
    last1 = paired["last1_prediction"].to_numpy(float)
    valid = paired["valid"].to_numpy(bool) & np.isfinite(target)
    history = paired["history_available"].to_numpy(bool)
    active = valid & history

    lower = np.minimum(qwen, last1)
    upper = np.maximum(qwen, last1)
    prediction = qwen.copy()
    prediction[active] = np.clip(target[active], lower[active], upper[active])

    alpha = np.zeros(len(paired), dtype=float)
    denominator = last1 - qwen
    interpolated = active & ~np.isclose(denominator, 0.0, rtol=0.0, atol=1e-15)
    alpha[interpolated] = (
        (prediction[interpolated] - qwen[interpolated]) / denominator[interpolated]
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    source = np.full(len(paired), "not_evaluated", dtype=object)
    source[valid & ~history] = "qwen_only_no_history"
    qwen_endpoint = active & np.isclose(prediction, qwen, rtol=0.0, atol=1e-12)
    last1_endpoint = (
        active
        & ~qwen_endpoint
        & np.isclose(prediction, last1, rtol=0.0, atol=1e-12)
    )
    interior = active & ~qwen_endpoint & ~last1_endpoint
    source[qwen_endpoint] = "qwen_endpoint"
    source[last1_endpoint] = "last1_endpoint"
    source[interior] = "line_segment_interior"

    output = _base_oracle_frame(
        paired,
        prediction,
        model_name=f"diagnostic_line_segment_oracle_{variant}",
        seed=seed,
    )
    output["oracle_alpha"] = alpha
    output["oracle_projection"] = source
    return output


def hit_overlap_counts(paired: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Count endpoint Acc@10 overlap, using Qwen as both endpoints without history."""
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("Hybrid-oracle hit overlap is fixed to Acc@10% tolerance 0.10")
    rows = []
    for split in EVALUATION_SPLITS:
        for resource in ("memory", "vram"):
            part = paired.loc[
                (paired["split"] == split)
                & (paired["resource"] == resource)
                & paired["valid"].astype(bool)
            ].copy()
            if part.empty:
                raise ValueError(
                    f"No valid rows for hit overlap: split={split}, resource={resource}"
                )
            target = part["target"].to_numpy(float)
            qwen = part["qwen_prediction"].to_numpy(float)
            last1 = part["last1_prediction"].to_numpy(float)
            history = part["history_available"].to_numpy(bool)
            effective_last1 = np.where(history, last1, qwen)
            qwen_hit = np.abs(qwen - target) <= tolerance
            last1_hit = np.abs(effective_last1 - target) <= tolerance
            counts = {
                "qwen_only_hit": int(np.sum(qwen_hit & ~last1_hit)),
                "last1_only_hit": int(np.sum(~qwen_hit & last1_hit)),
                "both_hit": int(np.sum(qwen_hit & last1_hit)),
                "neither_hit": int(np.sum(~qwen_hit & ~last1_hit)),
            }
            rows.append(
                {
                    "split": split,
                    "resource": resource,
                    "n": int(len(part)),
                    "history_available_n": int(history.sum()),
                    "acc_tolerance": float(tolerance),
                    **counts,
                }
            )
    return pd.DataFrame(rows)


def _mark_diagnostic_table(frame: pd.DataFrame, diagnostic: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "diagnostic", diagnostic)
    for name, value in DIAGNOSTIC_FLAGS.items():
        output[name] = value
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed = int(get_value(config, "seed", 42))
    variant = str(get_value(config, "hybrid.qwen_variant", "full"))
    protocol = str(get_value(config, "last1.protocol", "completed_before_submission"))
    qwen_path = _prediction_path(
        config,
        args.qwen_predictions,
        "analysis.qwen_predictions",
        f"outputs/qwen/{variant}/seed_{seed}/predictions.csv",
    )
    last1_path = _prediction_path(
        config,
        args.last1_predictions,
        "analysis.last1_predictions",
        f"outputs/last-1/{protocol}/seed_{seed}/predictions.csv",
    )
    output_dir = resolve_path(
        config,
        args.output_dir
        or get_value(
            config,
            "analysis.hybrid_oracle_output_dir",
            f"outputs/analysis/hybrid_oracle/{protocol}/{variant}/seed_{seed}",
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
            "Qwen and Last-1 manifest seeds must match the configured diagnostic seed"
        )
    qwen_details = qwen_manifest.get("details") or {}
    if (
        str(qwen_manifest.get("model")) != f"qwen_{variant}"
        or str(qwen_details.get("variant", "")) != variant
    ):
        raise ValueError(
            f"Qwen predictions do not match hybrid.qwen_variant={variant!r}"
        )
    if str(qwen_manifest["data_sha256"]) != str(last1_manifest["data_sha256"]):
        raise ValueError("Qwen and Last-1 oracle diagnostics reference different datasets")
    qwen_split = _split_fingerprint(qwen_manifest, "Qwen")
    last1_split = _split_fingerprint(last1_manifest, "Last-1")
    if qwen_split != last1_split:
        raise ValueError("Qwen and Last-1 oracle diagnostics reference different splits")
    last1_details = last1_manifest.get("details") or {}
    if (
        not bool(last1_details.get("causal_online_protocol", False))
        or str(last1_details.get("protocol", "")) != "completed_before_submission"
    ):
        raise ValueError(
            "Hybrid-oracle diagnostics require causal completed-before-submission "
            "Last-1 predictions"
        )

    paired = pair_predictions(qwen, last1)
    choice = choice_oracle_predictions(paired, seed, variant)
    line_segment = line_segment_oracle_predictions(paired, seed, variant)
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    choice_metrics = _mark_diagnostic_table(
        compute_metrics(choice, tolerance=tolerance, splits=EVALUATION_SPLITS),
        "target_using_choice_oracle",
    )
    line_segment_metrics = _mark_diagnostic_table(
        compute_metrics(line_segment, tolerance=tolerance, splits=EVALUATION_SPLITS),
        "target_using_line_segment_oracle",
    )
    overlap = _mark_diagnostic_table(
        hit_overlap_counts(paired, tolerance),
        "endpoint_hit_overlap",
    )

    choice.to_csv(output_dir / "choice_oracle_predictions.csv", index=False)
    line_segment.to_csv(
        output_dir / "line_segment_oracle_predictions.csv", index=False
    )
    choice_metrics.to_csv(output_dir / "choice_oracle_metrics.csv", index=False)
    line_segment_metrics.to_csv(
        output_dir / "line_segment_oracle_metrics.csv", index=False
    )
    overlap.to_csv(output_dir / "hit_overlap_counts.csv", index=False)
    write_resolved_config(config, output_dir / "config.resolved.yaml")

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "hybrid_oracle_upper_bounds",
        "model": f"diagnostic_hybrid_oracle_{variant}",
        "seed": seed,
        "python": platform.python_version(),
        **DIAGNOSTIC_FLAGS,
        "unbiased_evaluation": False,
        "allowed_uses": ["post_hoc_complementarity_diagnosis", "upper_bound_analysis"],
        "forbidden_uses": [
            "training",
            "model_selection",
            "routing_policy_selection",
            "formal_model_ranking",
            "deployment",
        ],
        "evaluated_splits": list(EVALUATION_SPLITS),
        "choice_oracle_definition": (
            "Per valid row, use its true target to choose the lower-absolute-error "
            "Qwen or Last-1 endpoint; ties and unavailable history use Qwen."
        ),
        "line_segment_oracle_definition": (
            "Per valid row with history, project the true target onto the closed "
            "numeric interval between Qwen and Last-1; unavailable history uses Qwen."
        ),
        "hit_overlap_definition": (
            "Acc@10 endpoint overlap after replacing unavailable Last-1 history "
            "with Qwen, matching the hybrid no-history policy."
        ),
        "qwen_model": str(qwen_manifest["model"]),
        "qwen_variant": variant,
        "qwen_predictions": str(qwen_path),
        "qwen_predictions_sha256": sha256_file(qwen_path),
        "last1_predictions": str(last1_path),
        "last1_predictions_sha256": sha256_file(last1_path),
        "data_sha256": str(qwen_manifest["data_sha256"]),
        "split_fingerprint": qwen_split,
        "last1_protocol": "completed_before_submission",
        "causal_last1_required": True,
        "metrics_files": [
            "choice_oracle_metrics.csv",
            "line_segment_oracle_metrics.csv",
        ],
        "formal_metrics_filename_intentionally_absent": "metrics.csv",
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print("DIAGNOSTIC ONLY: these target-using oracle results are not deployable.")
    print("Choice oracle:\n" + choice_metrics.to_string(index=False))
    print("Line-segment oracle:\n" + line_segment_metrics.to_string(index=False))
    print("Hit overlap:\n" + overlap.to_string(index=False))
    print(f"Diagnostic artifacts: {output_dir}")


if __name__ == "__main__":
    main()
