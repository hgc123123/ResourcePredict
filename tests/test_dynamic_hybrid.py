from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from common.data import DatasetBundle
from dynamic_hybrid.contracts import (
    ComponentInputs,
    pair_component_predictions,
    read_and_pair,
    validate_paired_against_bundle,
)
from dynamic_hybrid.features import RobustFeatureScaler, build_gate_features
from dynamic_hybrid.model import DynamicHybridModel, ResourceGate, dynamic_hybrid_loss
from dynamic_hybrid.pipeline import gate_fit_and_selection_masks
from qwen.dataset import script_sha256


def test_no_history_is_exact_qwen_even_with_residual_enabled():
    gate = ResourceGate(3, hidden_dims=(4,), dropout=0.0, residual_max=0.2)
    with torch.no_grad():
        gate.gate_head.weight.fill_(1.0)
        gate.gate_head.bias.fill_(3.0)
        gate.residual_head.weight.fill_(1.0)
        gate.residual_head.bias.fill_(3.0)
    qwen = torch.tensor([0.2, 0.8])
    output = gate(
        torch.ones(2, 3),
        qwen,
        torch.tensor([0.9, 0.1]),
        torch.tensor([False, False]),
    )
    assert torch.equal(output["prediction"], qwen)
    assert torch.equal(output["raw_prediction"], qwen)
    assert torch.equal(output["gate"], torch.zeros(2))
    assert torch.equal(output["residual"], torch.zeros(2))


def test_routing_supervision_prefers_the_line_segment_projection():
    target = torch.tensor([0.5])
    qwen = torch.tensor([0.2])
    last1 = torch.tensor([0.8])
    available = torch.tensor([True])

    def routing_term(gate_value: float) -> float:
        output = {
            "prediction": target.clone(),
            "learned_gate": torch.tensor([gate_value]),
            "residual": torch.zeros(1),
        }
        _, terms = dynamic_hybrid_loss(
            output,
            target,
            qwen,
            last1,
            available,
            mse_weight=0.0,
            tolerance_weight=0.0,
            underprediction_weight=0.0,
            routing_weight=1.0,
            residual_weight=0.0,
        )
        return float(terms["routing"])

    assert routing_term(0.5) == pytest.approx(0.0)
    assert routing_term(0.0) > routing_term(0.5)


def test_rare_binary_scaler_feature_is_not_erased_by_quantile_clipping():
    values = np.zeros((200, 2), dtype=float)
    values[:, 1] = np.arange(200, dtype=float)
    values[-1, 0] = 1.0
    scaler = RobustFeatureScaler.fit(values, ["rare_binary", "continuous"])
    transformed = scaler.transform(values, ["rare_binary", "continuous"])
    assert transformed[-1, 0] == pytest.approx(1.0)
    assert transformed[:-1, 0].sum() == pytest.approx(0.0)


def test_input_layer_normalization_can_be_disabled():
    gate = ResourceGate(
        3,
        hidden_dims=(4,),
        dropout=0.0,
        input_normalization="none",
    )
    assert isinstance(gate.trunk[0], torch.nn.Identity)


def _component_frame(model: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["v", "t"],
            "job_id": ["1", "2"],
            "user_id": ["u", "u"],
            "split": ["validation", "test"],
            "resource": ["memory", "memory"],
            "target": [0.2, 0.3],
            "raw_prediction": [0.2, 0.3],
            "prediction": [0.2, 0.3],
            "valid": [True, True],
            "model": [model, model],
            "seed": [42, 42],
        }
    )


def test_pairing_rejects_target_disagreement():
    qwen = _component_frame("qwen_full")
    last1 = _component_frame("last-1")
    last1["history_available"] = True
    last1.loc[last1["split"].eq("test"), "target"] = 0.31
    with pytest.raises(ValueError, match="targets differ"):
        pair_component_predictions(qwen, last1)


def test_paired_rows_are_checked_against_dataset_before_training():
    frame = pd.DataFrame(
        {
            "sample_id": ["v", "t"],
            "job_id": ["1", "2"],
            "user_id": ["u", "u"],
            "split": ["validation", "test"],
            "target_memory": [0.2, 0.3],
            "valid_memory": [True, True],
            "target_vram": [0.4, 0.5],
            "valid_vram": [True, True],
        }
    )
    bundle = DatasetBundle(
        frame=frame,
        data_path=Path("unused.csv"),
        data_sha256="x",
        manifest_path=None,
        numeric_features=[],
        categorical_features=[],
        protocol={},
    )
    qwen = pd.concat(
        [
            _component_frame("qwen_full"),
            _component_frame("qwen_full").assign(resource="vram", target=[0.4, 0.5]),
        ],
        ignore_index=True,
    )
    last1 = qwen.assign(model="last-1", history_available=True)
    paired = pair_component_predictions(qwen, last1)
    validate_paired_against_bundle(paired, bundle)

    paired.loc[
        paired["resource"].eq("memory") & paired["split"].eq("test"), "target"
    ] = 0.31
    with pytest.raises(ValueError, match="targets differ from the configured dataset"):
        validate_paired_against_bundle(paired, bundle)


def test_equal_history_source_time_is_rejected():
    times = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "source_row": [0, 1],
            "job_id": ["1", "2"],
            "user_id": ["u", "u"],
            "split": ["train", "validation"],
            "submit_time": times,
            "completion_time": pd.to_datetime(
                ["2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"]
            ),
            "code": ["echo a", "echo b"],
            "target_memory": [0.2, 0.4],
            "valid_memory": [True, True],
            "target_vram": [np.nan, np.nan],
            "valid_vram": [False, False],
            "num__cpu": [1.0, 2.0],
        }
    )
    bundle = DatasetBundle(
        frame=frame,
        data_path=Path("unused.csv"),
        data_sha256="x",
        manifest_path=None,
        numeric_features=["num__cpu"],
        categorical_features=[],
        protocol={
            "verified_chronological": True,
            "has_submission_time": True,
            "has_completion_time": True,
        },
        retained_frame=frame,
    )
    paired = pd.DataFrame(
        {
            "sample_id": ["b"],
            "split": ["validation"],
            "resource": ["memory"],
            "qwen_prediction": [0.3],
            "last1_prediction": [0.2],
            "history_available": [True],
            "history_source_sample_id": ["a"],
            # Equal to b's submit time, which violates strict '<'.
            "history_source_time": ["2026-01-01T02:00:00+00:00"],
            "gate_code": [0.4],
            "gate_user": [0.3],
            "gate_metadata": [0.3],
            "user_known": [True],
            "script_sha256": [script_sha256("echo b")],
            "valid": [True],
            "target": [0.4],
        }
    )
    with pytest.raises(ValueError, match="Invalid causal history timestamps"):
        build_gate_features(paired, bundle)


def test_history_source_time_must_match_shared_replay():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "source_row": [0, 1],
            "job_id": ["1", "2"],
            "user_id": ["u", "u"],
            "split": ["train", "validation"],
            "submit_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"]
            ),
            "completion_time": pd.to_datetime(
                ["2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"]
            ),
            "code": ["echo a", "echo b"],
            "target_memory": [0.2, 0.4],
            "valid_memory": [True, True],
            "target_vram": [np.nan, np.nan],
            "valid_vram": [False, False],
            "num__cpu": [1.0, 2.0],
        }
    )
    bundle = DatasetBundle(
        frame=frame,
        retained_frame=frame,
        data_path=Path("unused.csv"),
        data_sha256="x",
        manifest_path=None,
        numeric_features=["num__cpu"],
        categorical_features=[],
        protocol={
            "verified_chronological": True,
            "has_submission_time": True,
            "has_completion_time": True,
        },
    )
    paired = pd.DataFrame(
        {
            "sample_id": ["b"],
            "split": ["validation"],
            "resource": ["memory"],
            "qwen_prediction": [0.3],
            "last1_prediction": [0.2],
            "history_available": [True],
            "history_source_sample_id": ["a"],
            # Causal, but not a's real 01:00 completion time.
            "history_source_time": ["2026-01-01T00:30:00+00:00"],
            "gate_code": [0.4],
            "gate_user": [0.3],
            "gate_metadata": [0.3],
            "user_known": [True],
            "script_sha256": [script_sha256("echo b")],
            "valid": [True],
            "target": [0.4],
        }
    )
    with pytest.raises(ValueError, match="history_source_time disagrees"):
        build_gate_features(paired, bundle)


def test_validation_calibration_masks_are_chronological_and_job_aligned():
    rows = []
    for index in range(6):
        split = "validation" if index < 4 else "test"
        for resource in ("memory", "vram"):
            rows.append(
                {
                    "sample_id": f"job-{index}",
                    "split": split,
                    "resource": resource,
                    "submit_time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                }
            )
    frame = pd.DataFrame(rows)
    prepared = SimpleNamespace(
        features=SimpleNamespace(paired=frame),
        components=SimpleNamespace(qwen_manifest={"details": {}}),
    )
    fit, selection, protocol = gate_fit_and_selection_masks(
        prepared, {"dynamic_hybrid": {"validation_fit_fraction": 0.5}}
    )
    assert not np.any((frame["split"] == "test").to_numpy() & (fit | selection))
    assert protocol["test_used_for_fit_or_selection"] is False
    for _, indices in frame.groupby("sample_id").groups.items():
        indices = list(indices)
        assert len(set(fit[indices])) == 1
        assert len(set(selection[indices])) == 1


def test_oof_gate_fit_requires_strict_row_level_provenance():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "v", "t"],
            "split": ["train", "train", "validation", "test"],
            "resource": ["memory"] * 4,
            "submit_time": pd.to_datetime(
                [
                    "2026-01-02T00:00Z",
                    "2026-01-03T00:00Z",
                    "2026-01-04T00:00Z",
                    "2026-01-05T00:00Z",
                ]
            ),
            "oof_fold": [0, 1, np.nan, np.nan],
            "oof_train_cutoff": [
                "2026-01-01T00:00Z",
                "2026-01-02T00:00Z",
                None,
                None,
            ],
        }
    )
    details = {
        "train_predictions_out_of_fold": True,
        "oof_protocol": {
            "type": "expanding_window",
            "strict_cutoff_relation": "training_data_end < prediction_submit_time",
            "fold_count": 2,
        },
    }
    prepared = SimpleNamespace(
        features=SimpleNamespace(paired=frame),
        components=SimpleNamespace(qwen_manifest={"details": details}),
    )
    fit, selection, protocol = gate_fit_and_selection_masks(prepared, {})
    assert fit.tolist() == [True, True, False, False]
    assert selection.tolist() == [False, False, True, False]
    assert protocol["oof_observed_fold_count"] == 2

    prepared.components.qwen_manifest["details"][
        "train_predictions_out_of_fold"
    ] = False
    with pytest.raises(ValueError, match="out_of_fold=true"):
        gate_fit_and_selection_masks(prepared, {})


def test_full_qwen_gate_contract_rejects_non_normalized_weights(tmp_path):
    qwen = _component_frame("qwen_full")
    qwen["gate_code"] = 0.4
    qwen["gate_user"] = 0.3
    qwen["gate_metadata"] = 0.2
    qwen["user_known"] = True
    qwen["script_sha256"] = "a" * 64
    last1 = _component_frame("last-1")
    last1["history_available"] = False
    last1["history_source_sample_id"] = ""
    last1["history_source_time"] = ""
    qwen_path = tmp_path / "qwen.csv"
    last1_path = tmp_path / "last1.csv"
    qwen.to_csv(qwen_path, index=False)
    last1.to_csv(last1_path, index=False)
    inputs = ComponentInputs(
        qwen_path=qwen_path,
        last1_path=last1_path,
        qwen_manifest={"model": "qwen_full", "seed": 42, "details": {"variant": "full"}},
        last1_manifest={"model": "last-1", "seed": 42},
        data_sha256="x",
        split_fingerprint="y",
        seed=42,
    )
    with pytest.raises(ValueError, match="sum to one"):
        read_and_pair(inputs)


def test_resource_gates_are_independent():
    model = DynamicHybridModel(
        2,
        hidden_dims=(3,),
        dropout=0.0,
        residual_max=0.0,
        initial_alphas={"memory": 0.8, "vram": 0.2},
    )
    features = torch.zeros(1, 2)
    qwen = torch.tensor([0.1])
    last1 = torch.tensor([0.9])
    history = torch.tensor([True])
    memory = model("memory", features, qwen, last1, history)
    vram = model("vram", features, qwen, last1, history)
    assert memory["gate"].item() == pytest.approx(0.8)
    assert vram["gate"].item() == pytest.approx(0.2)
