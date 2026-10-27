import math

import pandas as pd
import pytest

from common.metrics import compute_metrics


def prediction_frame(rows):
    frame = pd.DataFrame(
        rows,
        columns=["sample_id", "job_id", "split", "resource", "target", "prediction", "valid"],
    )
    frame["model"] = "fixture"
    frame["seed"] = 42
    return frame


def test_metrics_and_inclusive_acc_tolerance():
    predictions = prediction_frame(
        [
            ("a", "1", "test", "memory", 0.0, 0.1, True),
            ("b", "2", "test", "memory", 0.5, 0.7, True),
            ("c", "3", "test", "memory", 1.0, 0.7, True),
            ("a", "1", "test", "vram", 0.0, 0.0, True),
            ("b", "2", "test", "vram", 0.5, 0.5, True),
            ("c", "3", "test", "vram", 1.0, 1.0, True),
        ]
    )
    metrics = compute_metrics(predictions, tolerance=0.10, splits=("test",))
    result = metrics.loc[metrics["resource"] == "memory"].iloc[0]
    assert result["n"] == 3
    assert result["hits_at_tolerance"] == 1
    assert math.isclose(result["mae"], 0.2)
    assert math.isclose(result["mse"], (0.01 + 0.04 + 0.09) / 3)
    assert math.isclose(result["rmse"], math.sqrt(result["mse"]))
    assert math.isclose(result["acc_at_10"], 1 / 3)


def test_invalid_label_is_excluded():
    predictions = prediction_frame(
        [
            ("a", "1", "test", "memory", 0.2, 0.2, True),
            ("b", "2", "test", "memory", 0.3, 0.3, True),
            ("a", "1", "test", "vram", 0.5, 0.6, True),
            ("b", "2", "test", "vram", float("nan"), 0.0, False),
        ]
    )
    metrics = compute_metrics(predictions, splits=("test",))
    result = metrics.loc[metrics["resource"] == "vram"].iloc[0]
    assert result["n"] == 1


@pytest.mark.parametrize("column", ["target", "prediction"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_valid_nonfinite_value_fails_closed(column, nonfinite):
    predictions = prediction_frame(
        [
            ("a", "1", "test", "memory", 0.2, 0.2, True),
            ("a", "1", "test", "vram", 0.3, 0.3, True),
        ]
    )
    predictions.loc[predictions["resource"] == "memory", column] = nonfinite
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(predictions, splits=("test",))


def test_missing_expected_resource_fails_closed():
    predictions = prediction_frame(
        [("a", "1", "test", "memory", 0.2, 0.2, True)]
    )
    with pytest.raises(ValueError, match="resource=vram"):
        compute_metrics(predictions, splits=("test",))


def test_expected_resource_requires_a_valid_label():
    predictions = prediction_frame(
        [
            ("a", "1", "test", "memory", 0.2, 0.2, True),
            ("a", "1", "test", "vram", float("nan"), float("nan"), False),
        ]
    )
    with pytest.raises(ValueError, match="No valid labels.*resource=vram"):
        compute_metrics(predictions, splits=("test",))


@pytest.mark.parametrize("column", ["model", "seed"])
def test_missing_prediction_identity_fails_closed(column):
    predictions = prediction_frame(
        [
            ("a", "1", "test", "memory", 0.2, 0.2, True),
            ("a", "1", "test", "vram", 0.3, 0.3, True),
        ]
    )
    predictions.loc[0, column] = None
    with pytest.raises(ValueError, match=f"missing {column}"):
        compute_metrics(predictions, splits=("test",))
