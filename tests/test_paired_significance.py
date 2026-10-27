import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import analysis.paired_significance as significance
from analysis.paired_significance import (
    attach_submission_blocks,
    load_input_manifest,
    mcnemar_result,
    pair_prediction_frames,
    paired_block_bootstrap,
    prepare_output_directory,
    validate_artifact_identity,
    validate_reference_column_contract,
    validate_shared_provenance,
)


DATA_HASH = "a" * 64
SPLIT_FINGERPRINT = "b" * 64


def _dataset(days: int = 4) -> pd.DataFrame:
    submit_time = pd.date_range("2026-01-01T12:00:00Z", periods=days, freq="1D")
    return pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(days)],
            "job_id": [f"job-{index}" for index in range(days)],
            "user_id": ["user"] * days,
            "split": ["test"] * days,
            "submit_time": submit_time,
            "target_memory": [0.5] * days,
            "valid_memory": [True] * days,
            "target_vram": [0.5] * days,
            "valid_vram": [True] * days,
        }
    )


def _predictions(model: str, error: float, days: int = 4) -> pd.DataFrame:
    rows = []
    for index in range(days):
        for resource in ("memory", "vram"):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "job_id": f"job-{index}",
                    "user_id": "user",
                    "split": "test",
                    "resource": resource,
                    "target": 0.5,
                    "prediction": 0.5 + error,
                    "valid": True,
                    "model": model,
                    "seed": 42,
                }
            )
    return pd.DataFrame(rows)


def _manifest(model: str, **extra) -> dict:
    return {
        "model": model,
        "seed": 42,
        "data_sha256": DATA_HASH,
        "split_fingerprint": SPLIT_FINGERPRINT,
        **extra,
    }


def _safe_hybrid_manifest() -> dict:
    return _manifest(
        "safe_script_history_hybrid_full",
        selection_split="validation",
        test_labels_used_for_selection=False,
        candidate_policy_frozen_before_test_evaluation=True,
        selection_protocol_version="pareto_safe_script_history_blend_v1",
        safe_reference_policies={
            "memory": {"alpha": 1.0, "same_script_beta": 1.0},
            "vram": {"alpha": 1.0, "same_script_beta": 1.0},
        },
    )


def test_strict_pairing_and_dataset_time_blocks():
    proposed = _predictions("proposed", 0.05)
    reference = _predictions("reference", 0.15)
    paired = pair_prediction_frames(proposed, reference)
    blocked = attach_submission_blocks(paired, _dataset(), "1D")

    assert len(blocked) == 8
    assert blocked["time_block"].nunique() == 4
    assert blocked["valid"].all()
    assert np.allclose(blocked["proposed_prediction"], 0.55)
    assert np.allclose(blocked["reference_prediction"], 0.65)


@pytest.mark.parametrize("failure", ["key", "job", "valid", "target"])
def test_pairing_rejects_misaligned_inputs(failure: str):
    proposed = _predictions("proposed", 0.05)
    reference = _predictions("reference", 0.15)
    if failure == "key":
        reference.loc[0, "sample_id"] = "different"
    elif failure == "job":
        reference.loc[0, "job_id"] = "different"
    elif failure == "valid":
        reference.loc[0, "valid"] = False
    else:
        reference.loc[0, "target"] = 0.6

    with pytest.raises(ValueError):
        pair_prediction_frames(proposed, reference)


def test_dataset_recheck_rejects_target_change():
    paired = pair_prediction_frames(
        _predictions("proposed", 0.05), _predictions("reference", 0.15)
    )
    dataset = _dataset()
    dataset.loc[0, "target_memory"] = 0.6
    with pytest.raises(ValueError, match="targets differ from the dataset"):
        attach_submission_blocks(paired, dataset)


def test_time_block_bootstrap_is_deterministic_and_paired():
    paired = pair_prediction_frames(
        _predictions("proposed", 0.05), _predictions("reference", 0.15)
    )
    blocked = attach_submission_blocks(paired, _dataset())
    metrics_a, differences_a, blocks_a = paired_block_bootstrap(
        blocked, replicates=200, seed=17
    )
    metrics_b, differences_b, blocks_b = paired_block_bootstrap(
        blocked, replicates=200, seed=17
    )

    pd.testing.assert_frame_equal(metrics_a, metrics_b)
    pd.testing.assert_frame_equal(differences_a, differences_b)
    pd.testing.assert_frame_equal(blocks_a, blocks_b)
    expected = {
        "mae": -0.10,
        "mse": -0.02,
        "rmse": -0.10,
        "acc_at_10": 1.0,
    }
    for metric, delta in expected.items():
        rows = differences_a.loc[differences_a["metric"] == metric]
        assert np.allclose(rows["difference_proposed_minus_reference"], delta)
        assert np.allclose(rows["ci_lower"], delta)
        assert np.allclose(rows["ci_upper"], delta)
        assert rows["ci_excludes_zero"].all()
    assert blocks_a["time_block_n"].eq(4).all()


def test_exact_and_asymptotic_mcnemar_definitions():
    target = np.zeros(6)
    exact = pd.DataFrame(
        {
            "target": target,
            "proposed_prediction": [0.0] * 5 + [0.5],
            "reference_prediction": [0.5] * 5 + [0.0],
        }
    )
    result = mcnemar_result(exact, method="exact")
    assert result["proposed_only_hit"] == 5
    assert result["reference_only_hit"] == 1
    assert result["method"] == "exact"
    assert result["p_value_two_sided"] == pytest.approx(0.21875)

    asymptotic = pd.DataFrame(
        {
            "target": [0.0] * 40,
            "proposed_prediction": [0.0] * 30 + [0.5] * 10,
            "reference_prediction": [0.5] * 30 + [0.0] * 10,
        }
    )
    result = mcnemar_result(asymptotic, method="auto", exact_threshold=25)
    expected_statistic = 19**2 / 40
    assert result["method"] == "asymptotic_continuity_corrected"
    assert result["statistic"] == pytest.approx(expected_statistic)
    assert result["p_value_two_sided"] == pytest.approx(
        math.erfc(math.sqrt(expected_statistic / 2.0))
    )


def test_mcnemar_table_adds_holm_correction():
    paired = pair_prediction_frames(
        _predictions("proposed", 0.05), _predictions("reference", 0.15)
    )
    result = significance.mcnemar_table(paired, method="exact")
    assert result["holm_family_size"].eq(2).all()
    assert (result["p_value_holm"] >= result["p_value_two_sided"]).all()


def test_diagnostic_and_oracle_artifacts_are_rejected(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prediction_path = run_dir / "predictions.csv"
    _predictions("oracle_router", 0.05).to_csv(prediction_path, index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("oracle_router")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="diagnostic/oracle"):
        load_input_manifest(prediction_path, "Proposed")

    (run_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("proposed", details={"diagnostic_only": True})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="diagnostic_only=true"):
        load_input_manifest(prediction_path, "Proposed")

    predictions = _predictions("proposed", 0.05)
    predictions["diagnostic_only"] = True
    with pytest.raises(ValueError, match="diagnostic_only=true"):
        validate_artifact_identity(predictions, _manifest("proposed"), "Proposed")


def test_shared_provenance_rejects_data_and_split_mismatch():
    with pytest.raises(ValueError, match="different datasets"):
        validate_shared_provenance(
            _manifest("proposed"),
            {**_manifest("reference"), "data_sha256": "c" * 64},
        )
    with pytest.raises(ValueError, match="different splits"):
        validate_shared_provenance(
            _manifest("proposed"),
            {**_manifest("reference"), "split_fingerprint": "d" * 64},
        )


def test_safe_reference_contract_requires_frozen_validation_provenance():
    with pytest.raises(ValueError, match="frozen validation provenance"):
        validate_reference_column_contract(
            _manifest("safe_script_history_hybrid_full"),
            "safe_reference_prediction",
        )
    validate_reference_column_contract(
        _safe_hybrid_manifest(), "safe_reference_prediction"
    )


def test_safe_reference_contract_rejects_sparse_nonhierarchical_fallback():
    manifest = _safe_hybrid_manifest()
    manifest["safe_reference_policies"]["vram"]["same_script_beta"] = 0.0

    with pytest.raises(ValueError, match="not a strict hierarchical"):
        validate_reference_column_contract(manifest, "safe_reference_prediction")


def test_output_file_protection_requires_explicit_overwrite(tmp_path: Path):
    output_dir = tmp_path / "paired"
    prepare_output_directory(output_dir)
    (output_dir / "paired_metrics.csv").write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_output_directory(output_dir)
    prepare_output_directory(output_dir, overwrite=True)


def test_main_writes_reproducible_csv_json_and_manifest(tmp_path: Path, monkeypatch):
    proposed_dir = tmp_path / "proposed"
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "output"
    proposed_dir.mkdir()
    reference_dir.mkdir()
    proposed_path = proposed_dir / "predictions.csv"
    reference_path = reference_dir / "predictions.csv"
    _predictions("proposed", 0.05).to_csv(proposed_path, index=False)
    _predictions("reference", 0.15).to_csv(reference_path, index=False)
    (proposed_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("proposed")), encoding="utf-8"
    )
    (reference_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("reference")), encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation:\n  acc_tolerance: 0.10\n", encoding="utf-8")
    fake_bundle = SimpleNamespace(
        frame=_dataset(),
        data_path=tmp_path / "dataset.csv",
        data_sha256=DATA_HASH,
        protocol={"split_fingerprint": SPLIT_FINGERPRINT},
    )
    monkeypatch.setattr(significance, "load_dataset", lambda config: fake_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paired_significance.py",
            "--config",
            str(config_path),
            "--proposed-predictions",
            str(proposed_path),
            "--reference-predictions",
            str(reference_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-replicates",
            "100",
            "--bootstrap-seed",
            "17",
        ],
    )
    significance.main()

    for name in (
        "paired_metrics.csv",
        "paired_differences.csv",
        "mcnemar.csv",
        "block_summary.csv",
        "significance_summary.json",
        "run_manifest.json",
        "config.resolved.yaml",
    ):
        assert (output_dir / name).is_file()
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["analysis"] == "paired_time_block_significance"
    assert manifest["model_selection_performed"] is False
    assert manifest["selected_splits"] == ["test"]
    assert manifest["bootstrap_replicates"] == 100
    assert manifest["bootstrap_seed"] == 17
    assert manifest["data_sha256"] == DATA_HASH
    differences = pd.read_csv(output_dir / "paired_differences.csv")
    assert set(differences["metric"]) == {"mae", "mse", "rmse", "acc_at_10"}
    assert len(differences) == 8

    rerun_args = list(sys.argv)
    with pytest.raises(FileExistsError, match="--overwrite"):
        significance.main()
    monkeypatch.setattr(sys, "argv", [*rerun_args, "--overwrite"])
    significance.main()
    overwritten = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert overwritten["overwrite_requested"] is True


def test_main_can_compare_a_declared_safe_reference_column(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "hybrid"
    output_dir = tmp_path / "output"
    run_dir.mkdir()
    predictions = _predictions("safe_script_history_hybrid_full", 0.05)
    predictions["safe_reference_prediction"] = 0.65
    path = run_dir / "predictions.csv"
    predictions.to_csv(path, index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("safe_script_history_hybrid_full")), encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation:\n  acc_tolerance: 0.10\n", encoding="utf-8")
    fake_bundle = SimpleNamespace(
        frame=_dataset(),
        data_path=tmp_path / "dataset.csv",
        data_sha256=DATA_HASH,
        protocol={"split_fingerprint": SPLIT_FINGERPRINT},
    )
    monkeypatch.setattr(significance, "load_dataset", lambda config: fake_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paired_significance.py",
            "--config",
            str(config_path),
            "--proposed-predictions",
            str(path),
            "--reference-predictions",
            str(path),
            "--reference-column",
            "safe_reference_prediction",
            "--output-dir",
            str(output_dir),
            "--bootstrap-replicates",
            "100",
        ],
    )
    with pytest.raises(ValueError, match="frozen validation provenance"):
        significance.main()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(_safe_hybrid_manifest()), encoding="utf-8"
    )
    significance.main()
    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["reference"]["prediction_column"] == "safe_reference_prediction"
    metrics = pd.read_csv(output_dir / "paired_metrics.csv")
    assert metrics.loc[metrics["role"] == "reference", "model"].str.contains(
        "safe_reference_prediction", regex=False
    ).all()
