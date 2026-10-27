#!/usr/bin/env python3
"""Run fictional CPU plumbing checks; never a trained-Qwen experiment.

Minimum dependencies: numpy pandas PyYAML. --fit-gates additionally needs
scikit-learn and runs the real cascade fit/select/apply code on explicitly
synthetic, deterministic base estimates. No torch or GPU is needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config, set_value
from common.data import load_dataset
from common.history import script_sha256
from common.preprocessing import NumericScaler, UserVocabulary
from qwen.base_selection import base_preprocessing_bundle, select_base_training_frames
from reliability_cascade.features import build_cascade_features
from reliability_cascade.pipeline import chronological_calibration_masks
from scripts.make_synthetic_trace import generate_synthetic_trace


def _last1_module():
    spec = importlib.util.spec_from_file_location("synthetic_smoke_last1", PROJECT_ROOT / "last-1" / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import real Last-1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_last1(bundle, replay, fallback) -> dict[str, int]:
    events = bundle.retained_frame.sort_values(["completion_time", "source_row"], kind="mergesort")
    effective = bundle.frame.reset_index(drop=True)
    effective_ids = set(effective["sample_id"].astype(str))
    purged_sources = 0
    available_count = 0
    for index, row in effective.iterrows():
        for resource in ("memory", "vram"):
            eligible = events.loc[(events["user_id"].astype(str) == str(row["user_id"]))
                                  & (events["completion_time"] < row["submit_time"])
                                  & events[f"valid_{resource}"].astype(bool)]
            source = eligible.iloc[-1] if len(eligible) else None
            actual = replay[resource]
            assert bool(actual["history_available"][index]) == (source is not None)
            if source is None:
                assert np.isclose(actual["prediction"][index], fallback[resource])
            else:
                available_count += 1
                assert str(actual["history_source_sample_id"][index]) == str(source["sample_id"])
                assert np.isclose(actual["prediction"][index], source[f"target_{resource}"])
                assert source["completion_time"] < row["submit_time"]
                if str(source["sample_id"]) not in effective_ids:
                    purged_sources += 1
    assert purged_sources > 0, "Fixture failed to exercise post-completion use of purged history"
    return {"checked_job_resource_predictions": int(2 * len(effective)),
            "history_available_predictions": available_count,
            "predictions_using_completed_boundary_purged_history": purged_sources}


def _synthetic_base_pairs(bundle, replay, optimization) -> pd.DataFrame:
    """Deterministic feature-only estimates; no observed targets enter this rule."""
    known_users = set(optimization["user_id"].astype(str))
    rows = []
    for index, row in bundle.frame.reset_index(drop=True).iterrows():
        identity = script_sha256(row["code"])
        # A fictional score from script identity and submission-visible requests.
        code_term = (int(identity[:8], 16) % 100) / 100.0
        for resource, offset in (("memory", 0.0), ("vram", 0.03)):
            estimate = float(np.clip(0.22 + 0.30 * code_term + offset, 0, 1))
            actual = replay[resource]
            rows.append({
                "sample_id": str(row["sample_id"]), "split": str(row["split"]),
                "resource": resource, "target": float(row[f"target_{resource}"]),
                "valid": bool(row[f"valid_{resource}"]), "qwen_prediction": estimate,
                "last1_prediction": float(actual["prediction"][index]),
                "history_available": bool(actual["history_available"][index]),
                "history_source_sample_id": str(actual["history_source_sample_id"][index]),
                "history_source_time": str(actual["history_source_time"][index]),
                "gate_code": 0.4, "gate_user": 0.3, "gate_metadata": 0.3,
                "user_known": str(row["user_id"]) in known_users,
                "script_sha256": identity,
                "synthetic_plumbing_only": True,
            })
    return pd.DataFrame(rows)


def run_smoke(output_dir: Path, *, template_config: Path | None = None,
              count: int = 320, seed: int = 2026, fit_gates: bool = False) -> dict[str, Any]:
    if fit_gates and importlib.util.find_spec("sklearn") is None:
        raise RuntimeError("--fit-gates needs scikit-learn; install it or omit this flag for role/replay checks")
    generated = generate_synthetic_trace(output_dir / "synthetic_input",
                                          template_config=template_config, count=count, seed=seed)
    config = load_config(generated["config"])
    assert config["synthetic_experiment"]["enabled"] is True
    assert config["synthetic_experiment"]["purpose"] == "protocol_smoke_only_not_paper_accuracy_evidence"
    bundle = load_dataset(config)
    strict = select_base_training_frames(bundle, config)
    record = strict.manifest
    assert record is not None and record["mode"] == "train_tail"
    assert record["purged_optimization_count"] > 0
    assert record["base_checkpoint_validation_overlap"] is False
    scoped = base_preprocessing_bundle(bundle, strict)
    fit = scoped.split("train")
    scaler = NumericScaler.fit(fit, bundle.numeric_features)
    vocabulary = UserVocabulary.fit(fit["user_id"].astype(str))
    assert np.isfinite(scaler.transform(bundle.frame)).all()
    assert set(vocabulary.user_to_id) - {UserVocabulary.UNK} == set(fit["user_id"].astype(str))
    assert len(bundle.retained_frame) > len(bundle.frame)
    last1 = _last1_module()
    fallback = last1.training_medians(bundle.frame)
    replay = last1.replay_completed_before_submission(bundle, fallback)
    last1_checks = _check_last1(bundle, replay, fallback)
    paired = _synthetic_base_pairs(bundle, replay, strict.optimization)
    features = build_cascade_features(paired, bundle, history_k=5, half_life_hours=168.0)
    # This is synthetic provenance used only to exercise role verification. No
    # checkpoint or Qwen run_manifest is created or presented as a trained run.
    prepared = SimpleNamespace(
        bundle=bundle, features=features,
        components=SimpleNamespace(qwen_manifest={"details": {
            "synthetic_plumbing_only": True,
            "base_training_selection": record,
            "effective_parameters": {"base_training_selection_sha256": record["sha256"]},
        }}),
    )
    gate_fit, gate_select, gate_protocol = chronological_calibration_masks(prepared, config)
    assert not np.any(gate_fit & gate_select)
    assert not np.any((gate_fit | gate_select) & features.frame["split"].eq("test").to_numpy())
    assert gate_protocol["base_checkpoint_validation_overlap"] is False
    assert gate_protocol["inner_boundary_purged_fit_jobs"] > 0
    summary = {
        "synthetic_plumbing_only": True,
        "warning": "FICTIONAL DATA AND BASE ESTIMATES; NOT PAPER ACCURACY OR A TRAINED QWEN RUN",
        "real_trace_used": False, "qwen_training_run": False, "gpu_required": False,
        "effective_rows": len(bundle.frame), "retained_rows": len(bundle.retained_frame),
        "outer_split_counts": bundle.protocol["split_counts"],
        "outer_boundary_purged_rows": len(bundle.retained_frame) - len(bundle.frame),
        "base_optimization_rows": len(strict.optimization),
        "base_early_stopping_rows": len(strict.early_stopping),
        "base_internal_purged_rows": record["purged_optimization_count"],
        "scaler_and_vocabulary_fit_on_optimization_only": True,
        "last1_replay_checks": last1_checks,
        "gate_fit_unique_jobs": gate_protocol["fit_unique_jobs"],
        "gate_select_unique_jobs": gate_protocol["selection_unique_jobs"],
        "gate_inner_boundary_purged_rows": gate_protocol["inner_boundary_purged_fit_jobs"],
        "base_checkpoint_validation_overlap": gate_protocol["base_checkpoint_validation_overlap"],
        "synthetic_gate_models_fit": False,
        "gate_fit_skip_reason": "Not requested; add --fit-gates with scikit-learn installed",
    }
    if fit_gates:
        from reliability_cascade.pipeline import cascade_model_config
        from reliability_cascade.selection import select_cascade
        from reliability_cascade.evaluation import apply_selected_cascade
        # Intentionally small synthetic plumbing models, not paper settings.
        set_value(config, "reliability_cascade.pre_gate.n_estimators", 5)
        set_value(config, "reliability_cascade.post_gate.n_estimators", 5)
        set_value(config, "reliability_cascade.pre_gate.thresholds", [0.0, 0.5, 1.01])
        selected = select_cascade(prepared, config, gate_fit, gate_select,
                                  cascade_model_config(config, seed=42))
        predictions, _, _, _, _, _ = apply_selected_cascade(
            prepared, selected, model_name="SYNTHETIC_PLUMBING_CASCADE_NOT_QWEN", seed=42)
        assert np.isfinite(predictions["prediction"].to_numpy(float)).all()
        summary["synthetic_gate_models_fit"] = True
        summary["gate_fit_skip_reason"] = None
        summary["synthetic_applied_job_resource_rows"] = len(predictions)
        summary["synthetic_policy_selected"] = True
        # Deliberately do not publish these arbitrary accuracy values as results.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "SYNTHETIC_SMOKE_REPORT.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-config", type=Path)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fit-gates", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir, template_config=args.template_config,
                               count=args.rows, seed=args.seed, fit_gates=args.fit_gates), indent=2))


if __name__ == "__main__":
    main()
