import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.artifacts import build_prediction_frame, write_evaluation
from common.data import DatasetBundle


def make_bundle(tmp_path: Path) -> DatasetBundle:
    frame = pd.DataFrame(
        {
            "sample_id": ["train::0", "validation::1", "test::2"],
            "job_id": ["0", "1", "2"],
            "user_id": ["u0", "u1", "u2"],
            "split": ["train", "validation", "test"],
            "target_memory": [0.1, 0.2, 0.3],
            "target_vram": [0.4, 0.5, 0.6],
            "valid_memory": [True, True, True],
            "valid_vram": [True, True, True],
        }
    )
    return DatasetBundle(
        frame=frame,
        data_path=tmp_path / "fixture.csv",
        data_sha256="a" * 64,
        manifest_path=None,
        numeric_features=[],
        categorical_features=[],
        protocol={
            "split_fingerprint": "b" * 64,
            "order_fingerprint": "c" * 64,
            "split_manifest_sha256": None,
        },
    )


def make_predictions(bundle: DatasetBundle) -> pd.DataFrame:
    return build_prediction_frame(
        bundle.frame,
        {
            "memory": np.array([0.1, 0.2, 0.3]),
            "vram": np.array([0.4, 0.5, 0.6]),
        },
        model_name="fixture",
        seed=42,
        clip_predictions=True,
    )


def test_write_evaluation_accepts_complete_validation_and_test_coverage(tmp_path):
    bundle = make_bundle(tmp_path)
    predictions = make_predictions(bundle)
    output = tmp_path / "output"
    output.mkdir()
    metrics = write_evaluation(
        {"evaluation": {"acc_tolerance": 0.10}, "seed": 42},
        bundle,
        "fixture",
        predictions,
        output,
    )
    assert len(metrics) == 4
    assert (output / "predictions.csv").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "run_manifest.json").is_file()


def test_write_evaluation_can_record_a_model_specific_seed(tmp_path):
    bundle = make_bundle(tmp_path)
    predictions = make_predictions(bundle)
    predictions["seed"] = 7
    output = tmp_path / "output"
    output.mkdir()

    write_evaluation(
        {"evaluation": {"acc_tolerance": 0.10}, "seed": 42},
        bundle,
        "fixture",
        predictions,
        output,
        run_seed=7,
    )

    with (output / "run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["seed"] == 7


def test_prediction_context_excludes_workdir_and_node_list(tmp_path):
    bundle = make_bundle(tmp_path)
    bundle.frame["job_name"] = ["a", "b", "c"]
    bundle.frame["partition"] = "gpu"
    bundle.frame["submit_time"] = pd.date_range(
        "2026-01-01", periods=3, tz="UTC"
    )
    bundle.frame["gpu_count"] = [0, 1, 1]
    bundle.frame["node_list"] = ["node1", "node2", "node3"]
    bundle.frame["workdir"] = ["/private/a", "/private/b", "/private/c"]

    predictions = make_predictions(bundle)

    assert {"job_name", "partition", "submit_time", "gpu_count"}.issubset(
        predictions.columns
    )
    assert "node_list" not in predictions.columns
    assert "workdir" not in predictions.columns


def test_write_evaluation_rejects_missing_sample_resource(tmp_path):
    bundle = make_bundle(tmp_path)
    predictions = make_predictions(bundle)
    missing = predictions.loc[
        ~(
            (predictions["sample_id"] == "test::2")
            & (predictions["resource"] == "vram")
        )
    ].copy()
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="coverage mismatch"):
        write_evaluation({}, bundle, "fixture", missing, output)


@pytest.mark.parametrize("field", ["target", "valid"])
def test_write_evaluation_rejects_target_or_validity_mismatch(tmp_path, field):
    bundle = make_bundle(tmp_path)
    predictions = make_predictions(bundle)
    row = (predictions["sample_id"] == "validation::1") & (
        predictions["resource"] == "memory"
    )
    if field == "target":
        predictions.loc[row, field] += 0.01
        message = "targets differ"
    else:
        predictions.loc[row, field] = False
        message = "validity mask"
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match=message):
        write_evaluation({}, bundle, "fixture", predictions, output)


def test_write_evaluation_distinguishes_nan_from_infinity(tmp_path):
    bundle = make_bundle(tmp_path)
    row = bundle.frame["sample_id"] == "validation::1"
    bundle.frame.loc[row, "target_vram"] = np.nan
    bundle.frame.loc[row, "valid_vram"] = False
    predictions = make_predictions(bundle)
    prediction_row = (predictions["sample_id"] == "validation::1") & (
        predictions["resource"] == "vram"
    )
    predictions.loc[prediction_row, "target"] = np.inf
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="targets differ"):
        write_evaluation({}, bundle, "fixture", predictions, output)
