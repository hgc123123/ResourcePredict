#!/usr/bin/env python3
"""Target-independent subgroup and routing analysis for the reliability cascade."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.data import sha256_file
from common.metrics import coerce_valid_mask, validate_prediction_frame
from reliability_cascade.integrity import read_json, verify_run_directory


PREDICTORS = {
    "reliability_cascade": "prediction",
    "qwen_1p5b": "qwen_prediction",
    "hierarchical_same_script_last1": "hierarchical_reference_prediction",
}
ANALYSIS_SCHEMA = "reliability-cascade-subgroups/v1"
SUPPORTED_SPLITS = {"validation", "test"}
SUPPORTED_RESOURCES = {"memory", "vram"}
REQUIRED_SUBGROUP_COLUMNS = {
    *PREDICTORS.values(),
    "history_available",
    "exact_script_history_available",
    "same_last1",
    "resource_applicable",
    "history_bypass",
    "qwen_required_resource",
    "post_history_weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cascade accuracy and routing by pre-outcome history subgroup"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _history_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "history_available",
        "exact_script_history_available",
        "same_last1",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cascade predictions lack subgroup fields: {sorted(missing)}")
    history = coerce_valid_mask(frame["history_available"], "history_available")
    exact = coerce_valid_mask(
        frame["exact_script_history_available"],
        "exact_script_history_available",
    )
    same = coerce_valid_mask(frame["same_last1"], "same_last1")
    if np.any(exact & ~history):
        raise ValueError("Exact-script history cannot exist without same-user history")
    if np.any(same & ~exact):
        raise ValueError("same_last1 requires eligible exact-script history")
    return history, exact, same


def assign_subgroup(frame: pd.DataFrame) -> pd.Series:
    history, exact, same = _history_masks(frame)
    values = np.full(len(frame), "no_exact_script_history", dtype=object)
    values[~history] = "cold_start"
    values[exact & ~same] = "older_exact_script_history"
    values[exact & same] = "same_last1_exact_script"
    return pd.Series(values, index=frame.index, dtype="string")


def validate_subgroup_frame(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    checkpoint_contract: dict[str, Any] | None = None,
) -> None:
    """Validate analysis fields and their cascade invariants before grouping."""
    validate_prediction_frame(frame)
    missing = REQUIRED_SUBGROUP_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Cascade predictions lack subgroup fields: {sorted(missing)}")

    splits = set(frame["split"].astype(str))
    unknown_splits = splits.difference(SUPPORTED_SPLITS)
    if unknown_splits:
        raise ValueError(f"Cascade predictions contain unknown splits: {sorted(unknown_splits)}")
    resources = set(frame["resource"].astype(str))
    unknown_resources = resources.difference(SUPPORTED_RESOURCES)
    if unknown_resources:
        raise ValueError(
            f"Cascade predictions contain unknown resources: {sorted(unknown_resources)}"
        )

    model = str(manifest.get("model", ""))
    observed_model = frame["model"].astype(str).unique().tolist()
    if observed_model != [model]:
        raise ValueError("Cascade prediction model disagrees with its run manifest")
    try:
        manifest_seed = int(manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cascade run manifest requires an integer seed") from exc
    observed_seed = pd.to_numeric(frame["seed"], errors="coerce").to_numpy(float)
    if not np.all(observed_seed == manifest_seed):
        raise ValueError("Cascade prediction seed disagrees with its run manifest")

    history, exact, _ = _history_masks(frame)
    applicable = coerce_valid_mask(
        frame["resource_applicable"], "resource_applicable"
    )
    valid = coerce_valid_mask(frame["valid"], "valid")
    bypass = coerce_valid_mask(frame["history_bypass"], "history_bypass")
    required = coerce_valid_mask(
        frame["qwen_required_resource"], "qwen_required_resource"
    )
    memory = frame["resource"].astype(str).eq("memory").to_numpy()
    if np.any(memory & ~applicable):
        raise ValueError("Every memory row must be resource-applicable")
    if np.any(valid & ~applicable):
        raise ValueError("A valid target cannot belong to a non-applicable resource row")
    if np.any(bypass & (~applicable | ~history)):
        raise ValueError("History bypass requires applicable same-user history")
    expected_required = applicable & ~bypass
    if not np.array_equal(required, expected_required):
        raise ValueError(
            "qwen_required_resource must equal resource_applicable AND NOT history_bypass"
        )

    details = manifest.get("details") or {}
    if not isinstance(details, dict):
        raise ValueError("Cascade run manifest details must be an object")
    model_config = (checkpoint_contract or {}).get("cascade_model_config") or {}
    if not isinstance(model_config, dict):
        raise ValueError("Cascade checkpoint model configuration must be an object")
    require_exact = model_config.get("require_exact_script_for_bypass")
    if require_exact is True and np.any(bypass & ~exact):
        raise ValueError("This run contract permits bypass only with exact-script history")

    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(float)
    if not np.isfinite(target[valid]).all():
        raise ValueError("Valid subgroup rows contain non-finite targets")
    for column in PREDICTORS.values():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values[valid]).all():
            raise ValueError(f"Valid subgroup rows contain non-finite {column}")
    weights = pd.to_numeric(
        frame["post_history_weight"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(weights[applicable]).all():
        raise ValueError("Applicable rows contain non-finite post_history_weight")
    if np.any((weights[applicable] < 0.0) | (weights[applicable] > 1.0)):
        raise ValueError("post_history_weight must lie within [0,1]")


def load_verified_predictions(
    source: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the exact prediction bytes bound by a sealed cascade run."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cascade predictions not found: {source}")
    expected_source = (source.parent / "predictions.csv").resolve()
    if source != expected_source:
        raise ValueError(
            "Subgroup analysis requires the sealed cascade artifact named predictions.csv"
        )

    manifest = verify_run_directory(source.parent)
    integrity = read_json(source.parent / "artifact_integrity.json")
    checkpoint_contract = read_json(
        source.parent / "checkpoint" / "checkpoint_contract.json"
    )
    hashes = integrity.get("files") or {}
    expected_sha256 = hashes.get("predictions.csv")
    if not isinstance(expected_sha256, str):
        raise ValueError("Cascade artifact integrity does not bind predictions.csv")
    payload = source.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Cascade predictions changed after run verification")
    frame = pd.read_csv(io.BytesIO(payload), dtype={"sample_id": str, "job_id": str})
    validate_subgroup_frame(frame, manifest, checkpoint_contract)

    details = manifest.get("details") or {}
    model_config = checkpoint_contract["cascade_model_config"]
    provenance = {
        "run_directory": str(source.parent),
        "predictions": str(source),
        "predictions_sha256": observed_sha256,
        "model": str(manifest["model"]),
        "seed": int(manifest["seed"]),
        "variant": details.get("variant"),
        "max_tokens": (details.get("qwen_effective_parameters") or {}).get(
            "max_tokens"
        ),
        "run_request_sha256": details.get("run_request_sha256"),
        "artifact_integrity_sha256": details.get("artifact_integrity_sha256"),
        "require_exact_script_for_bypass": model_config.get(
            "require_exact_script_for_bypass"
        ),
    }
    return frame, provenance


def subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, resource, subgroup), part in frame.groupby(
        ["split", "resource", "subgroup"], sort=True
    ):
        valid = coerce_valid_mask(part["valid"])
        part = part.loc[valid]
        if part.empty:
            continue
        target = pd.to_numeric(part["target"], errors="raise").to_numpy(float)
        if not np.isfinite(target).all():
            raise ValueError("Valid subgroup rows contain non-finite targets")
        for predictor, column in PREDICTORS.items():
            prediction = pd.to_numeric(part[column], errors="raise").to_numpy(float)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Valid subgroup rows contain non-finite {column}")
            absolute = np.abs(prediction - target)
            squared = np.square(prediction - target)
            rows.append(
                {
                    "split": split,
                    "resource": resource,
                    "subgroup": subgroup,
                    "predictor": predictor,
                    "n": int(len(part)),
                    "mae": float(np.mean(absolute)),
                    "mse": float(np.mean(squared)),
                    "rmse": float(np.sqrt(np.mean(squared))),
                    "hits_at_tolerance": int(np.sum(absolute <= 0.10)),
                    "acc_at_10": float(np.mean(absolute <= 0.10)),
                }
            )
    return pd.DataFrame(rows)


def subgroup_routing(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, resource, subgroup), part in frame.groupby(
        ["split", "resource", "subgroup"], sort=True
    ):
        applicable = coerce_valid_mask(part["resource_applicable"], "resource_applicable")
        part = part.loc[applicable]
        if part.empty:
            continue
        bypass = coerce_valid_mask(part["history_bypass"], "history_bypass")
        required = coerce_valid_mask(
            part["qwen_required_resource"], "qwen_required_resource"
        )
        weights = pd.to_numeric(
            part["post_history_weight"], errors="raise"
        ).to_numpy(float)
        if not np.isfinite(weights).all():
            raise ValueError("Applicable rows contain non-finite post_history_weight")
        rows.append(
            {
                "split": split,
                "resource": resource,
                "subgroup": subgroup,
                "n": int(len(part)),
                "history_bypass_n": int(bypass.sum()),
                "history_bypass_rate": float(bypass.mean()),
                "qwen_required_n": int(required.sum()),
                "qwen_required_rate": float(required.mean()),
                "mean_post_history_weight": float(np.mean(weights)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source = Path(args.predictions).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    required_outputs = ("subgroup_metrics.csv", "subgroup_routing.csv", "summary.json")
    existing = [name for name in required_outputs if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Subgroup outputs already exist ({', '.join(existing)}); pass --overwrite"
        )
    frame, provenance = load_verified_predictions(source)
    frame["subgroup"] = assign_subgroup(frame)
    metrics = subgroup_metrics(frame)
    routing = subgroup_routing(frame)
    if metrics.empty or routing.empty:
        raise ValueError("Subgroup analysis produced no reportable rows")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "subgroup_metrics.csv"
    routing_path = output / "subgroup_routing.csv"
    metrics.to_csv(metrics_path, index=False)
    routing.to_csv(routing_path, index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": ANALYSIS_SCHEMA,
                "analysis_implementation": {
                    "path": "analysis/cascade_subgroups.py",
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "predictions": str(source),
                "source_provenance": provenance,
                "output_sha256": {
                    "subgroup_metrics.csv": sha256_file(metrics_path),
                    "subgroup_routing.csv": sha256_file(routing_path),
                },
                "predictors": PREDICTORS,
                "acc_tolerance": 0.10,
                "input_rows": int(len(frame)),
                "subgroups_are_target_independent": True,
                "subgroup_definition": [
                    "same_last1_exact_script",
                    "older_exact_script_history",
                    "no_exact_script_history",
                    "cold_start",
                ],
                "targets_used_for_group_assignment": False,
                "test_targets_used_for_group_assignment": False,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(metrics.to_string(index=False))
    print(f"Subgroup metrics: {output / 'subgroup_metrics.csv'}")


if __name__ == "__main__":
    main()
