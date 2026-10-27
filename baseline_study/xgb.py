"""Unchanged XGBoost algorithm with optimization-only fitting and tail stopping."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def original_module():
    # The source folder is named xgboost, which also names the installed package.
    # Load the source explicitly instead of importing xgboost.run.
    path = Path(__file__).resolve().parents[1] / "xgboost/run.py"
    spec = importlib.util.spec_from_file_location("_matched_xgboost_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(config, bundle, selection, output_dir):
    import joblib
    import numpy as np

    from baseline_study.protocol import fitting_bundle
    from common.artifacts import build_prediction_frame, write_evaluation
    from common.config import get_value
    from qwen.base_selection import verify_base_training_selection

    verify_base_training_selection(bundle, config, selection.manifest)
    legacy = original_module()
    scoped = fitting_bundle(bundle, selection)
    numeric, categorical = legacy.feature_columns(bundle, config)
    fit_metadata = legacy.metadata_frame(scoped.frame, numeric, categorical)
    metadata = legacy.metadata_frame(bundle.frame, numeric, categorical)
    preprocessor = legacy.build_preprocessor(numeric, categorical, config)
    preprocessor.fit(fit_metadata.loc[scoped.frame["split"].eq("train")])
    fitting = preprocessor.transform(fit_metadata)
    transformed = preprocessor.transform(metadata)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8")
    seed = int(get_value(config, "seed", 42))
    predictions, resources = {}, {}
    for resource in legacy.RESOURCE_SPECS:
        model, _, details = legacy.fit_resource_model(
            resource, scoped.frame, fitting, config, seed)
        joblib.dump(model, output_dir / f"model_{resource}.joblib")
        predictions[resource] = np.asarray(model.predict(transformed), dtype=float)
        details["optimization_valid_n"] = details.pop("train_n")
        details["early_stopping_valid_n"] = details.pop("validation_n")
        resources[resource] = details
    # Keep outer validation and test names; the early-stop tail is never exported
    # under the outer-validation label.
    evaluation = bundle.frame["split"].isin(["validation", "test"]).to_numpy()
    frame = bundle.frame.loc[evaluation].reset_index(drop=True)
    prediction_frame = build_prediction_frame(
        frame, {r: values[evaluation] for r, values in predictions.items()},
        model_name="xgboost", seed=seed,
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)))
    write_evaluation(config, bundle, "xgboost", prediction_frame, output_dir, details={
        "metadata_only": True,
        "base_training_selection": selection.manifest,
        "preprocessing_fit": "optimization_only",
        "checkpoint_selection": "outer_train_tail",
        "outer_validation_used_for_selection": False,
        "test_used_for_fit_or_selection": False,
        "refit_after_selection": False,
        "numeric_features": numeric, "categorical_features": categorical,
        "transformed_feature_count": len(feature_names), "resources": resources,
    })
