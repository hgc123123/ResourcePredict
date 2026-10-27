from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from common.metrics import compute_metrics
from scripts.verify_baseline_run import (
    COMPLETION_FILE,
    COMPLETION_SCHEMA,
    seal_baseline_run,
    verify_baseline_run,
)


DATA_SHA256 = "a" * 64
SPLIT_FINGERPRINT = "b" * 64
NEURAL_SPLIT_FINGERPRINT = "c" * 64


def _config(seed: int = 42) -> dict:
    return {
        "seed": seed,
        "data": {
            "expected_sha256": DATA_SHA256,
            "expected_valid_by_split": {
                "memory": {"validation": 2, "test": 2},
                "vram": {"validation": 1, "test": 2},
            },
        },
        "split": {
            "expected_fingerprint": SPLIT_FINGERPRINT,
            "expected_counts": {"train": 4, "validation": 2, "test": 2},
        },
        "evaluation": {"acc_tolerance": 0.10},
    }


def _predictions(model: str, seed: int) -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        for resource in ("memory", "vram"):
            for index in range(2):
                valid = not (
                    split == "validation" and resource == "vram" and index == 1
                )
                target = 0.2 + 0.1 * index
                prediction = target + 0.02
                rows.append(
                    {
                        "sample_id": f"{split}-{index}",
                        "job_id": f"job-{split}-{index}",
                        "split": split,
                        "resource": resource,
                        "target": target,
                        "prediction": prediction,
                        "valid": valid,
                        "model": model,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(tmp_path: Path, model: str = "xgboost", seed: int = 42) -> Path:
    run_dir = tmp_path / model / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    predictions = _predictions(model, seed)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    compute_metrics(predictions).to_csv(run_dir / "metrics.csv", index=False)

    numeric_features = ["cpu_request"]
    manifest = {
        "model": model,
        "seed": seed,
        "data_sha256": DATA_SHA256,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "protocol": {
            "data_sha256": DATA_SHA256,
            "split_fingerprint": SPLIT_FINGERPRINT,
        },
        "details": {
            "checkpoint_identity": {
                "data_sha256": DATA_SHA256,
                # Neural checkpoint identity uses a distinct serialization
                # from the dataset protocol fingerprint.
                "split_fingerprint": NEURAL_SPLIT_FINGERPRINT,
                "seed": seed,
                "numeric_features": numeric_features,
            }
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(_config(seed)), encoding="utf-8"
    )

    if model == "xgboost":
        _write_json(run_dir / "feature_names.json", ["cpu_request"])
        for name in ("preprocessor.joblib", "model_memory.joblib", "model_vram.joblib"):
            (run_dir / name).write_bytes(b"model")
    else:
        (run_dir / "model.pt").write_bytes(b"model")
        pd.DataFrame(
            [{"epoch": 1, "train_loss": 0.1, "validation_mean_mae": 0.2}]
        ).to_csv(run_dir / "training_history.csv", index=False)
        _write_json(run_dir / "architecture.json", {"hidden_dim": 8})
        _write_json(run_dir / "token_vocabulary.json", {"<PAD>": 0, "<UNK>": 1})
        _write_json(run_dir / "user_vocabulary.json", {"<UNK>": 0})
        _write_json(
            run_dir / "numeric_scaler.json",
            {
                "columns": numeric_features,
                "medians": [1.0],
                "means": [1.0],
                "scales": [1.0],
                "standardize": True,
            },
        )
    seal_baseline_run(
        run_dir,
        config=_config(seed),
        expected_model=model,
        expected_seed=seed,
    )
    return run_dir


@pytest.mark.parametrize("model", ["xgboost", "textcnn", "lstm"])
def test_verify_baseline_run_accepts_complete_consistent_run(
    tmp_path: Path, model: str
) -> None:
    run_dir = _make_run(tmp_path, model=model)

    result = verify_baseline_run(
        run_dir,
        config=_config(),
        expected_model=model,
        expected_seed=42,
    )

    assert result["model"] == model
    assert result["seed"] == 42
    assert result["data_sha256"] == DATA_SHA256
    assert result["split_fingerprint"] == SPLIT_FINGERPRINT


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "textcnn", "Manifest model mismatch"),
        ("seed", 41, "Manifest seed mismatch"),
        ("data_sha256", "c" * 64, "data SHA-256"),
        ("split_fingerprint", "d" * 64, "split fingerprint"),
    ],
)
def test_verify_baseline_run_rejects_manifest_identity_or_provenance_mismatch(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run_dir = _make_run(tmp_path)
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(path, manifest)

    with pytest.raises(ValueError, match=message):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


def test_verify_baseline_run_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    (run_dir / "run_manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON artifact"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


@pytest.mark.parametrize(("column", "value"), [("model", "textcnn"), ("seed", 41)])
def test_verify_baseline_run_rejects_prediction_identity_mismatch(
    tmp_path: Path, column: str, value: object
) -> None:
    run_dir = _make_run(tmp_path)
    predictions_path = run_dir / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions[column] = value
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match=f"Prediction {column} disagrees"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


def test_verify_baseline_run_rejects_metrics_not_matching_predictions(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(tmp_path)
    metrics_path = run_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "mae"] += 0.01
    metrics.to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="differs from predictions.csv recomputation"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


def test_verify_baseline_run_rejects_incomplete_prediction_coverage(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(tmp_path)
    predictions_path = run_dir / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions = predictions.loc[
        ~(
            predictions["split"].eq("validation")
            & predictions["resource"].eq("vram")
            & ~predictions["valid"]
        )
    ].copy()
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="Prediction coverage"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


def test_verify_baseline_run_rejects_missing_model_artifact(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, model="textcnn")
    (run_dir / "model.pt").unlink()

    with pytest.raises(FileNotFoundError, match="model.pt"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="textcnn",
            expected_seed=42,
        )


def test_completion_seal_binds_model_artifacts_and_normalized_config(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(tmp_path, model="xgboost")
    seal = json.loads((run_dir / COMPLETION_FILE).read_text(encoding="utf-8"))

    assert seal["schema"] == COMPLETION_SCHEMA
    assert seal["model"] == "xgboost"
    assert seal["seed"] == 42
    assert set(seal["files_sha256"]) == {
        "predictions.csv",
        "metrics.csv",
        "run_manifest.json",
        "config.resolved.yaml",
        "preprocessor.joblib",
        "feature_names.json",
        "model_memory.joblib",
        "model_vram.joblib",
    }
    assert len(seal["normalized_config_sha256"]) == 64
    assert len(seal["artifact_set_sha256"]) == 64


def test_verify_baseline_run_rejects_missing_completion_seal(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    (run_dir / COMPLETION_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="completion seal"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


@pytest.mark.parametrize(
    ("model", "artifact"),
    [
        ("xgboost", "model_memory.joblib"),
        ("textcnn", "model.pt"),
        ("lstm", "model.pt"),
    ],
)
def test_verify_baseline_run_rejects_corrupted_model_artifact(
    tmp_path: Path, model: str, artifact: str
) -> None:
    run_dir = _make_run(tmp_path, model=model)
    with (run_dir / artifact).open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ValueError, match=f"artifact hash mismatch: {artifact}"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model=model,
            expected_seed=42,
        )


def test_verify_baseline_run_rejects_resolved_config_file_tampering(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(tmp_path)
    config_path = run_dir / "config.resolved.yaml"
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("# byte-level tampering\n")

    with pytest.raises(ValueError, match="artifact hash mismatch: config.resolved.yaml"):
        verify_baseline_run(
            run_dir,
            config=_config(),
            expected_model="xgboost",
            expected_seed=42,
        )


def test_verify_baseline_run_rejects_requested_config_drift(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    changed = _config()
    changed["evaluation"]["clip_predictions"] = False

    with pytest.raises(ValueError, match="normalized requested experiment"):
        verify_baseline_run(
            run_dir,
            config=changed,
            expected_model="xgboost",
            expected_seed=42,
        )
