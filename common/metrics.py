from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "sample_id",
    "job_id",
    "split",
    "resource",
    "target",
    "prediction",
    "valid",
    "model",
    "seed",
}


def coerce_valid_mask(values: pd.Series, name: str = "valid") -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"{name} contains missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{name} must contain only boolean or 0/1 values")
        return numeric.to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        invalid = sorted(normalized.loc[~normalized.isin(mapping)].unique().tolist())[:5]
        raise ValueError(f"{name} contains invalid boolean values: {invalid}")
    return normalized.map(mapping).to_numpy(bool)


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    duplicates = predictions.duplicated(["sample_id", "split", "resource"])
    if duplicates.any():
        raise ValueError(f"Prediction frame has {int(duplicates.sum())} duplicate sample/resource rows")
    for column in ("sample_id", "job_id", "split", "resource", "model", "seed"):
        if predictions[column].isna().any():
            raise ValueError(f"Prediction frame contains missing {column} values")
    models = predictions["model"].astype(str).unique()
    if len(models) != 1:
        raise ValueError("Prediction frame must contain exactly one model identity")
    seeds = pd.to_numeric(predictions["seed"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(seeds).all()
        or not np.allclose(seeds, np.round(seeds), rtol=0.0, atol=0.0)
        or len(np.unique(seeds)) != 1
    ):
        raise ValueError("Prediction frame must contain exactly one integer seed")
    valid = coerce_valid_mask(predictions["valid"])
    target = pd.to_numeric(predictions["target"], errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(predictions["prediction"], errors="coerce").to_numpy(float)
    if not np.isfinite(target[valid]).all():
        raise ValueError("Valid prediction rows contain non-finite targets")
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("Valid prediction rows contain non-finite predictions")


def compute_metrics(
    predictions: pd.DataFrame,
    tolerance: float = 0.10,
    splits: Iterable[str] = ("validation", "test"),
) -> pd.DataFrame:
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("This experiment contract fixes Acc@10% tolerance at 0.10")
    validate_prediction_frame(predictions)
    expected_splits = tuple(str(split) for split in splits)
    if not expected_splits or len(set(expected_splits)) != len(expected_splits):
        raise ValueError("Expected metric splits must be a non-empty unique sequence")
    rows = []
    for split in expected_splits:
        for resource in ("memory", "vram"):
            part = predictions.loc[
                (predictions["split"] == split)
                & (predictions["resource"] == resource)
            ].copy()
            if part.empty:
                raise ValueError(f"No prediction rows for split={split}, resource={resource}")
            valid = coerce_valid_mask(part["valid"])
            if not valid.any():
                raise ValueError(f"No valid labels for split={split}, resource={resource}")
            valid_part = part.loc[valid].copy()
            target = pd.to_numeric(valid_part["target"], errors="coerce").to_numpy(float)
            prediction = pd.to_numeric(
                valid_part["prediction"], errors="coerce"
            ).to_numpy(float)
            if not np.isfinite(target).all():
                raise ValueError(
                    f"Valid rows contain non-finite targets for split={split}, resource={resource}"
                )
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Valid rows contain non-finite predictions for split={split}, resource={resource}"
                )
            error = prediction - target
            absolute = np.abs(error)
            squared = np.square(error)
            hits = int(np.sum(absolute <= tolerance))
            rows.append(
                {
                    "split": split,
                    "resource": resource,
                    "n": int(len(valid_part)),
                    "mae": float(np.mean(absolute)),
                    "mse": float(np.mean(squared)),
                    "rmse": float(np.sqrt(np.mean(squared))),
                    "acc_tolerance": float(tolerance),
                    "hits_at_tolerance": hits,
                    "acc_at_10": float(hits / len(valid_part)),
                }
            )
    return pd.DataFrame(rows)


def add_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    valid = coerce_valid_mask(output["valid"])
    output["absolute_error"] = np.where(
        valid,
        np.abs(output["prediction"] - output["target"]),
        np.nan,
    )
    output["squared_error"] = np.where(
        valid,
        np.square(output["prediction"] - output["target"]),
        np.nan,
    )
    return output
