#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.data import sha256_file  # noqa: E402
from hybrid.fixed_blend import pair_predictions  # noqa: E402


THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post hoc diagnostic for test cases with Last-1 absolute error above 0.10"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def checked_path(config, cli_value, key, default) -> Path:
    path = resolve_path(config, cli_value or get_value(config, key, default))
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return path


def load_manifest(prediction_path: Path, component: str) -> dict:
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


def split_fingerprint(manifest: dict, component: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{component} run manifest has no split fingerprint")
    return str(value)


def validate_component_identity(
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed = int(get_value(config, "seed", 42))
    variant = str(get_value(config, "hybrid.qwen_variant", "full"))
    protocol = str(get_value(config, "last1.protocol", "completed_before_submission"))
    qwen_path = checked_path(
        config,
        args.qwen_predictions,
        "analysis.qwen_predictions",
        f"outputs/qwen/{variant}/seed_{seed}/predictions.csv",
    )
    last1_path = checked_path(
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
            "analysis.history_stress_output_dir",
            f"outputs/analysis/history_stress/{protocol}/{variant}/seed_{seed}",
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen = pd.read_csv(qwen_path, dtype={"sample_id": str, "job_id": str})
    last1 = pd.read_csv(last1_path, dtype={"sample_id": str, "job_id": str})
    qwen_manifest = load_manifest(qwen_path, "Qwen")
    last1_manifest = load_manifest(last1_path, "Last-1")
    validate_component_identity(qwen, qwen_manifest, "Qwen")
    validate_component_identity(last1, last1_manifest, "Last-1")
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
            "Qwen predictions do not match hybrid.qwen_variant=" f"{variant!r}"
        )
    if str(qwen_manifest["data_sha256"]) != str(last1_manifest["data_sha256"]):
        raise ValueError("Qwen and Last-1 diagnostics reference different datasets")
    qwen_split = split_fingerprint(qwen_manifest, "Qwen")
    last1_split = split_fingerprint(last1_manifest, "Last-1")
    if qwen_split != last1_split:
        raise ValueError("Qwen and Last-1 diagnostics reference different splits")
    last1_details = last1_manifest.get("details") or {}
    if (
        not bool(last1_details.get("causal_online_protocol", False))
        or str(last1_details.get("protocol", "")) != "completed_before_submission"
    ):
        raise ValueError(
            "History-stress diagnostics require causal completed-before-submission "
            "Last-1 predictions; ordered-row smoke-test output is rejected"
        )
    paired = pair_predictions(qwen, last1)
    test = paired.loc[
        (paired["split"] == "test")
        & paired["valid"].astype(bool)
        & paired["history_available"].astype(bool)
    ].copy()
    test["last1_absolute_error"] = np.abs(test["last1_prediction"] - test["target"])
    test["qwen_absolute_error"] = np.abs(test["qwen_prediction"] - test["target"])
    diagnostic = test.loc[test["last1_absolute_error"] > THRESHOLD].copy()
    diagnostic["selection_rule"] = "Last-1 absolute error > 0.10"

    rows = []
    for resource in ("memory", "vram"):
        part = diagnostic.loc[diagnostic["resource"] == resource]
        if part.empty:
            raise ValueError(f"No history-stress test rows were selected for {resource}")
        last1_error = part["last1_prediction"].to_numpy(float) - part["target"].to_numpy(float)
        qwen_error = part["qwen_prediction"].to_numpy(float) - part["target"].to_numpy(float)
        last1_mae = float(np.mean(np.abs(last1_error)))
        qwen_mae = float(np.mean(np.abs(qwen_error)))
        rows.append(
            {
                "subset": "history_stress_diagnostic",
                "resource": resource,
                "n": int(len(part)),
                "selection_threshold": THRESHOLD,
                "last1_mae": last1_mae,
                "qwen_mae": qwen_mae,
                "conditional_mae_reduction": (
                    float((last1_mae - qwen_mae) / last1_mae)
                    if last1_mae > 0
                    else float("nan")
                ),
                "last1_mse": float(np.mean(np.square(last1_error))),
                "qwen_mse": float(np.mean(np.square(qwen_error))),
                "last1_acc_at_10": float(np.mean(np.abs(last1_error) <= THRESHOLD)),
                "qwen_acc_at_10": float(np.mean(np.abs(qwen_error) <= THRESHOLD)),
            }
        )

    metrics = pd.DataFrame(rows)
    diagnostic.to_csv(output_dir / "selected_predictions.csv", index=False)
    metrics.to_csv(output_dir / "diagnostic_metrics.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "post_hoc_history_stress_diagnostic",
        "seed": seed,
        "qwen_model": str(qwen_manifest["model"]),
        "qwen_variant": variant,
        "selection_split": "test",
        "selection_reference": "Last-1",
        "selection_rule": "absolute error > 0.10",
        "used_for_training_or_model_selection": False,
        "unbiased_test_set": False,
        "qwen_predictions": str(qwen_path),
        "qwen_predictions_sha256": sha256_file(qwen_path),
        "last1_predictions": str(last1_path),
        "last1_predictions_sha256": sha256_file(last1_path),
        "data_sha256": str(qwen_manifest["data_sha256"]),
        "split_fingerprint": qwen_split,
        "last1_protocol": "completed_before_submission",
        "causal_last1_required": True,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(metrics.to_string(index=False))
    print(f"Diagnostic artifacts: {output_dir}")


if __name__ == "__main__":
    main()
