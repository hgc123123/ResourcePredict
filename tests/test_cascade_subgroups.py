import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import analysis.cascade_subgroups as cascade_subgroups
from analysis.cascade_subgroups import (
    PREDICTORS,
    assign_subgroup,
    load_verified_predictions,
    subgroup_metrics,
    subgroup_routing,
    validate_subgroup_frame,
)
from common.data import sha256_file


def _frame() -> pd.DataFrame:
    definitions = (
        ("cold_start", False, False, False, False, True, 0.0),
        ("no_exact_script_history", True, False, False, False, True, 0.25),
        ("older_exact_script_history", True, True, False, True, False, 0.75),
        ("same_last1_exact_script", True, True, True, False, True, 0.50),
    )
    rows = []
    for index, (group, history, exact, same, bypass, required, weight) in enumerate(
        definitions
    ):
        target = 0.50
        rows.append(
            {
                "sample_id": f"s{index}",
                "job_id": str(index),
                "split": "test",
                "resource": "memory",
                "target": target,
                "prediction": target + 0.01 * (index + 1),
                "valid": True,
                "model": "reliability_cascade_full",
                "seed": 42,
                "qwen_prediction": target + 0.10,
                "hierarchical_reference_prediction": target - 0.05,
                "history_available": history,
                "exact_script_history_available": exact,
                "same_last1": same,
                "resource_applicable": True,
                "history_bypass": bypass,
                "qwen_required_resource": required,
                "post_history_weight": weight,
                "expected_subgroup": group,
            }
        )
    return pd.DataFrame(rows)


def _manifest() -> dict:
    return {
        "model": "reliability_cascade_full",
        "seed": 42,
        "details": {
            "variant": "full",
            "require_exact_script_for_bypass": True,
            "qwen_effective_parameters": {"max_tokens": 4096},
            "run_request_sha256": "1" * 64,
            "artifact_integrity_sha256": "2" * 64,
        },
    }


def _checkpoint_contract() -> dict:
    return {
        "cascade_model_config": {"require_exact_script_for_bypass": True}
    }


def test_subgroups_are_exhaustive_target_independent_and_causally_valid():
    frame = _frame()
    assigned = assign_subgroup(frame)
    assert assigned.tolist() == frame["expected_subgroup"].tolist()

    relabeled = frame.copy()
    relabeled["target"] = np.linspace(-100.0, 100.0, len(frame))
    relabeled["prediction"] = np.linspace(100.0, -100.0, len(frame))
    assert assign_subgroup(relabeled).equals(assigned)

    impossible = frame.copy()
    impossible.loc[0, "exact_script_history_available"] = True
    with pytest.raises(ValueError, match="without same-user history"):
        assign_subgroup(impossible)
    impossible = frame.copy()
    impossible.loc[1, "same_last1"] = True
    with pytest.raises(ValueError, match="requires eligible exact-script history"):
        assign_subgroup(impossible)


def test_subgroup_metrics_and_routing_match_manual_calculations():
    frame = _frame()
    frame["subgroup"] = assign_subgroup(frame)
    metrics = subgroup_metrics(frame)
    routing = subgroup_routing(frame)

    assert len(metrics) == len(frame) * len(PREDICTORS)
    assert metrics["n"].eq(1).all()
    cascade = metrics.loc[metrics["predictor"].eq("reliability_cascade")]
    expected_mae = {
        row.expected_subgroup: 0.01 * (index + 1)
        for index, row in enumerate(frame.itertuples(index=False))
    }
    for row in cascade.itertuples(index=False):
        assert row.mae == pytest.approx(expected_mae[row.subgroup])
        assert row.rmse == pytest.approx(expected_mae[row.subgroup])
        assert row.acc_at_10 == pytest.approx(1.0)

    routed = routing.set_index("subgroup")
    assert routed.loc["older_exact_script_history", "history_bypass_rate"] == 1.0
    assert routed.loc["older_exact_script_history", "qwen_required_rate"] == 0.0
    assert routed.loc["no_exact_script_history", "mean_post_history_weight"] == 0.25


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("model", "another_model", "model disagrees"),
        ("seed", 41, "seed disagrees"),
        ("qwen_required_resource", False, "qwen_required_resource"),
        ("post_history_weight", 1.1, "within \\[0,1\\]"),
    ],
)
def test_subgroup_validation_fails_closed_on_identity_and_route_errors(
    column, value, message
):
    frame = _frame()
    if column in {"model", "seed"}:
        frame[column] = value
    else:
        frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_subgroup_frame(frame, _manifest(), _checkpoint_contract())


def _write_bound_predictions(run_dir: Path, frame: pd.DataFrame) -> Path:
    run_dir.mkdir(parents=True)
    source = run_dir / "predictions.csv"
    frame.to_csv(source, index=False)
    (run_dir / "artifact_integrity.json").write_text(
        json.dumps({"files": {"predictions.csv": sha256_file(source)}}),
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoint" / "checkpoint_contract.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(json.dumps(_checkpoint_contract()), encoding="utf-8")
    return source


def test_verified_loader_uses_sealed_bytes_and_records_provenance(tmp_path, monkeypatch):
    source = _write_bound_predictions(tmp_path / "run", _frame())
    calls = []

    def verify(run_dir):
        calls.append(run_dir)
        return copy.deepcopy(_manifest())

    monkeypatch.setattr(cascade_subgroups, "verify_run_directory", verify)
    loaded, provenance = load_verified_predictions(source)

    assert calls == [source.parent.resolve()]
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        _frame().reset_index(drop=True),
        check_dtype=False,
    )
    assert provenance["predictions_sha256"] == sha256_file(source)
    assert provenance["model"] == "reliability_cascade_full"
    assert provenance["seed"] == 42
    assert provenance["max_tokens"] == 4096


def test_verified_loader_propagates_integrity_failure(tmp_path, monkeypatch):
    source = _write_bound_predictions(tmp_path / "run", _frame())

    def reject(_run_dir):
        raise ValueError("Cascade artifact hash mismatch: predictions.csv")

    monkeypatch.setattr(cascade_subgroups, "verify_run_directory", reject)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_verified_predictions(source)


def test_verified_loader_detects_change_after_run_verification(tmp_path, monkeypatch):
    source = _write_bound_predictions(tmp_path / "run", _frame())

    def tamper_after_verification(_run_dir):
        source.write_text("tampered\n", encoding="utf-8")
        return copy.deepcopy(_manifest())

    monkeypatch.setattr(
        cascade_subgroups, "verify_run_directory", tamper_after_verification
    )
    with pytest.raises(ValueError, match="changed after run verification"):
        load_verified_predictions(source)


def test_main_writes_hashed_reproducibility_summary(tmp_path, monkeypatch):
    source = _write_bound_predictions(tmp_path / "run", _frame())
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        cascade_subgroups,
        "verify_run_directory",
        lambda _run_dir: copy.deepcopy(_manifest()),
    )
    monkeypatch.setattr(
        cascade_subgroups,
        "parse_args",
        lambda: SimpleNamespace(
            predictions=str(source), output_dir=str(output), overwrite=False
        ),
    )

    cascade_subgroups.main()

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "reliability-cascade-subgroups/v1"
    assert summary["targets_used_for_group_assignment"] is False
    assert summary["test_targets_used_for_group_assignment"] is False
    assert summary["output_sha256"] == {
        "subgroup_metrics.csv": sha256_file(output / "subgroup_metrics.csv"),
        "subgroup_routing.csv": sha256_file(output / "subgroup_routing.csv"),
    }
    assert summary["analysis_implementation"]["sha256"] == sha256_file(
        Path(cascade_subgroups.__file__).resolve()
    )
