#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate metadata-only XGBoost")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def feature_columns(
    bundle: DatasetBundle,
    config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    numeric = list(dict.fromkeys(bundle.numeric_features))
    categorical = list(dict.fromkeys(bundle.categorical_features))
    include_user = bool(get_value(config, "xgboost.include_user", True))
    if include_user:
        source_user = get_value(config, "data.columns.user")
        standardized_duplicate = f"cat__{source_user}" if source_user else None
        categorical = [name for name in categorical if name != standardized_duplicate]
        categorical.append("user_id")
    categorical = list(dict.fromkeys(categorical))
    if not numeric and not categorical:
        raise ValueError("XGBoost requires at least one configured metadata feature")
    return numeric, categorical


def build_preprocessor(
    numeric: List[str],
    categorical: List[str],
    config: Dict[str, Any],
) -> ColumnTransformer:
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                )
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        minimum_frequency = get_value(config, "xgboost.one_hot_min_frequency", None)
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=minimum_frequency,
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def metadata_frame(
    frame: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
) -> pd.DataFrame:
    columns = numeric + categorical
    result = frame.loc[:, columns].copy()
    for column in categorical:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def model_parameters(config: Dict[str, Any], seed: int, base_score: float) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": str(get_value(config, "xgboost.eval_metric", "rmse")),
        "n_estimators": int(get_value(config, "xgboost.n_estimators", 500)),
        "learning_rate": float(get_value(config, "xgboost.learning_rate", 0.02)),
        "max_depth": int(get_value(config, "xgboost.max_depth", 2)),
        "min_child_weight": float(get_value(config, "xgboost.min_child_weight", 50.0)),
        "gamma": float(get_value(config, "xgboost.gamma", 1.0)),
        "reg_alpha": float(get_value(config, "xgboost.reg_alpha", 10.0)),
        "reg_lambda": float(get_value(config, "xgboost.reg_lambda", 10.0)),
        "subsample": float(get_value(config, "xgboost.subsample", 1.0)),
        "colsample_bytree": float(get_value(config, "xgboost.colsample_bytree", 1.0)),
        "tree_method": str(get_value(config, "xgboost.tree_method", "hist")),
        "n_jobs": int(get_value(config, "xgboost.n_jobs", 8)),
        "random_state": seed,
        "base_score": base_score,
        "verbosity": 0,
    }
    early_stopping = int(get_value(config, "xgboost.early_stopping_rounds", 30))
    if early_stopping > 0:
        parameters["early_stopping_rounds"] = early_stopping
    device = get_value(config, "xgboost.device", None)
    if device:
        parameters["device"] = str(device)
    return parameters


def fit_resource_model(
    resource: str,
    frame: pd.DataFrame,
    transformed: Any,
    config: Dict[str, Any],
    seed: int,
) -> Tuple[XGBRegressor, np.ndarray, Dict[str, Any]]:
    target_column, valid_column = RESOURCE_SPECS[resource]
    train_mask = frame["split"].eq("train").to_numpy() & frame[valid_column].to_numpy(bool)
    validation_mask = frame["split"].eq("validation").to_numpy() & frame[
        valid_column
    ].to_numpy(bool)
    if not train_mask.any():
        raise ValueError(f"Training split has no valid {resource} targets")
    if not validation_mask.any():
        raise ValueError(f"Validation split has no valid {resource} targets")

    target = frame[target_column].to_numpy(float)
    base_score = float(np.mean(target[train_mask]))
    parameters = model_parameters(config, seed, base_score)
    model = XGBRegressor(**parameters)
    model.fit(
        transformed[train_mask],
        target[train_mask],
        eval_set=[(transformed[validation_mask], target[validation_mask])],
        verbose=False,
    )
    prediction = np.asarray(model.predict(transformed), dtype=float)
    details = {
        "train_n": int(train_mask.sum()),
        "validation_n": int(validation_mask.sum()),
        "base_score": base_score,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "best_score": float(model.best_score) if hasattr(model, "best_score") else None,
        "parameters": parameters,
    }
    return model, prediction, details


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    frame = bundle.frame
    seed = int(get_value(config, "seed", 42))
    numeric, categorical = feature_columns(bundle, config)
    metadata = metadata_frame(frame, numeric, categorical)

    train_mask = frame["split"].eq("train").to_numpy()
    preprocessor = build_preprocessor(numeric, categorical, config)
    preprocessor.fit(metadata.loc[train_mask])
    transformed = preprocessor.transform(metadata)

    output_dir = output_directory(config, "xgboost", f"seed_{seed}")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2, ensure_ascii=True)

    predictions: Dict[str, np.ndarray] = {}
    resources: Dict[str, Any] = {}
    for resource in RESOURCE_SPECS:
        model, prediction, model_details = fit_resource_model(
            resource, frame, transformed, config, seed
        )
        joblib.dump(model, output_dir / f"model_{resource}.joblib")
        predictions[resource] = prediction
        resources[resource] = model_details

    prediction_frame = build_prediction_frame(
        frame,
        predictions,
        model_name="xgboost",
        seed=seed,
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)),
        extra_columns={"metadata_only": np.ones(len(frame), dtype=bool)},
    )
    details: Dict[str, Any] = {
        "metadata_only": True,
        "preprocessor_fit_split": "train",
        "numeric_features": numeric,
        "categorical_features": categorical,
        "user_encoding": "one_hot_train_only" if "user_id" in categorical else "excluded",
        "transformed_feature_count": int(len(feature_names)),
        "resources": resources,
    }
    metrics = write_evaluation(
        config,
        bundle,
        model_name="xgboost",
        predictions=prediction_frame,
        output_dir=output_dir,
        details=details,
    )
    print(metrics.to_string(index=False))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
