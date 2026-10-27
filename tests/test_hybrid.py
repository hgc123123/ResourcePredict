import json

import numpy as np
import pandas as pd
import pytest

from common.data import sha256_file
from hybrid.fixed_blend import (
    alpha_grid,
    apply_alphas,
    apply_guarded_policies,
    apply_script_guarded_policies,
    apply_script_history_policies,
    guard_budget_grid,
    select_guarded_resource_policies,
    select_script_guarded_resource_policies,
    select_safe_script_history_policies,
    select_resource_alphas,
    select_safe_resource_alphas,
)
from hybrid.run import _load_frozen_safe_alphas


def validation_rows():
    return pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "resource": ["memory", "memory", "vram", "vram"],
            "valid": [True] * 4,
            "target": [0.8, 0.7, 0.6, 0.5],
            "qwen_prediction": [0.1, 0.2, 0.1, 0.2],
            "last1_prediction": [0.8, 0.7, 0.6, 0.5],
            "history_available": [True] * 4,
        }
    )


def test_alpha_is_selected_from_validation_mae():
    alphas, diagnostics = select_resource_alphas(validation_rows(), alpha_grid(0.1))
    assert alphas == {"memory": 1.0, "vram": 1.0}
    assert diagnostics["memory"]["validation_mae"] == pytest.approx(0.0)


def test_alpha_selector_rejects_test_rows():
    rows = validation_rows()
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_resource_alphas(rows, alpha_grid(0.1))


def test_safe_alpha_selects_pareto_improving_blend():
    rows = validation_rows()
    rows["target"] = [0.5, 0.5, 0.5, 0.5]
    rows["qwen_prediction"] = [0.4, 0.4, 0.4, 0.4]
    rows["last1_prediction"] = [0.6, 0.6, 0.6, 0.6]

    alphas, diagnostics, sweep = select_safe_resource_alphas(
        rows,
        alpha_grid(0.1),
    )

    assert alphas == {"memory": 0.5, "vram": 0.5}
    assert diagnostics["memory"]["selection_status"] == "safe_blend_selected"
    assert sweep.loc[sweep["selected"], "feasible"].all()


def test_safe_alpha_uses_qwen_for_cold_start_reference():
    rows = validation_rows()
    rows["history_available"] = False
    rows["last1_prediction"] = 0.99
    rows["qwen_prediction"] = rows["target"]

    alphas, diagnostics, _ = select_safe_resource_alphas(
        rows,
        alpha_grid(0.1),
    )

    assert alphas == {"memory": 1.0, "vram": 1.0}
    assert diagnostics["vram"]["reference_metrics"]["mae"] == pytest.approx(0.0)


def test_safe_alpha_selector_rejects_test_rows():
    rows = validation_rows()
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_safe_resource_alphas(rows, alpha_grid(0.1))


def test_guard_budget_grid_is_scaled_by_acc_tolerance():
    assert guard_budget_grid(0.10).tolist() == pytest.approx(
        [0.0, 0.025, 0.05, 0.075, 0.10]
    )


def test_guard_budget_grid_rejects_fraction_above_tolerance():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        guard_budget_grid(0.10, [0.0, 1.1])


def test_guarded_policy_selects_a_bounded_pareto_improvement():
    rows = validation_rows()
    rows["target"] = [0.5, 0.5, 0.5, 0.5]
    rows["qwen_prediction"] = [0.4, 0.4, 0.4, 0.4]
    rows["last1_prediction"] = [0.6, 0.6, 0.6, 0.6]

    policies, diagnostics, sweep = select_guarded_resource_policies(
        rows,
        {"memory": 0.5, "vram": 0.5},
        guard_budget_grid(0.10),
    )

    assert policies["memory"]["alpha"] == pytest.approx(0.5)
    assert policies["memory"]["guard_budget"] == pytest.approx(0.10)
    assert policies["memory"]["guard_budget_fraction"] == pytest.approx(1.0)
    assert diagnostics["vram"]["selection_status"] == "guarded_blend_selected"
    assert sweep.loc[sweep["selected"], "feasible"].all()

    applied = apply_guarded_policies(rows, policies)
    adjustment = np.abs(
        applied["prediction"].to_numpy(float)
        - applied["safe_reference_prediction"].to_numpy(float)
    )
    assert np.all(adjustment <= applied["guard_budget"].to_numpy(float) + 1e-12)


def test_guarded_policy_keeps_cold_start_exactly_on_qwen():
    rows = validation_rows()
    rows["history_available"] = False
    rows["last1_prediction"] = 0.99
    policies = {
        "memory": {"alpha": 0.5, "guard_budget": 0.05},
        "vram": {"alpha": 0.5, "guard_budget": 0.05},
    }

    applied = apply_guarded_policies(rows, policies)

    assert np.array_equal(
        applied["prediction"].to_numpy(float),
        rows["qwen_prediction"].to_numpy(float),
    )
    assert not applied["guard_was_active"].any()
    assert (applied["alpha"] == 0.0).all()


def test_guarded_policy_rejects_invalid_application_parameters():
    rows = validation_rows()
    policies = {
        "memory": {"alpha": 0.5, "guard_budget": -0.01},
        "vram": {"alpha": 0.5, "guard_budget": 0.05},
    }
    with pytest.raises(ValueError, match="budget"):
        apply_guarded_policies(rows, policies)


def test_guarded_selector_rejects_test_rows():
    rows = validation_rows()
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_guarded_resource_policies(
            rows,
            {"memory": 0.5, "vram": 0.5},
            [0.0, 0.05],
        )


def test_zero_guard_budget_exactly_reproduces_reference():
    rows = validation_rows()
    policies, _, _ = select_guarded_resource_policies(
        rows,
        {"memory": 0.2, "vram": 0.8},
        [0.0],
    )
    applied = apply_guarded_policies(rows, policies)
    assert np.array_equal(
        applied["prediction"].to_numpy(float),
        rows["last1_prediction"].to_numpy(float),
    )


def script_guard_rows():
    rows = []
    for resource in ("memory", "vram"):
        rows.extend(
            [
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.5,
                    "qwen_prediction": 0.3,
                    "last1_prediction": 0.5,
                    "history_available": True,
                    "same_script_as_last1": True,
                },
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.5,
                    "qwen_prediction": 0.3,
                    "last1_prediction": 0.5,
                    "history_available": True,
                    "same_script_as_last1": True,
                },
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.5,
                    "qwen_prediction": 0.3,
                    "last1_prediction": 0.7,
                    "history_available": True,
                    "same_script_as_last1": False,
                },
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.5,
                    "qwen_prediction": 0.3,
                    "last1_prediction": 0.7,
                    "history_available": True,
                    "same_script_as_last1": False,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_script_guard_selects_more_budget_after_a_script_change():
    rows = script_guard_rows()
    policies, diagnostics, sweep = select_script_guarded_resource_policies(
        rows,
        {"memory": 0.5, "vram": 0.5},
        [0.0, 0.1],
        min_stratum_n=1,
    )

    for resource in ("memory", "vram"):
        assert policies[resource]["same_script_guard_budget"] == pytest.approx(0.0)
        assert policies[resource]["changed_script_guard_budget"] == pytest.approx(0.1)
        assert diagnostics[resource]["stratified_selection_active"]
    assert sweep.loc[sweep["selected"], "feasible"].all()

    applied = apply_script_guarded_policies(rows, policies)
    same = applied["same_script_as_last1"].to_numpy(bool)
    assert np.allclose(applied.loc[same, "prediction"], 0.5)
    assert np.allclose(applied.loc[~same, "prediction"], 0.6)


def test_script_guard_nests_the_global_guard_exactly():
    rows = script_guard_rows()
    global_policies = {
        "memory": {"alpha": 0.5, "guard_budget": 0.05},
        "vram": {"alpha": 0.5, "guard_budget": 0.05},
    }
    script_policies = {
        "memory": {
            "alpha": 0.5,
            "same_script_guard_budget": 0.05,
            "changed_script_guard_budget": 0.05,
        },
        "vram": {
            "alpha": 0.5,
            "same_script_guard_budget": 0.05,
            "changed_script_guard_budget": 0.05,
        },
    }

    global_result = apply_guarded_policies(rows, global_policies)
    script_result = apply_script_guarded_policies(rows, script_policies)
    assert np.array_equal(
        global_result["prediction"].to_numpy(float),
        script_result["prediction"].to_numpy(float),
    )


def test_script_guard_sparse_strata_back_off_to_one_budget():
    rows = script_guard_rows()
    policies, diagnostics, sweep = select_script_guarded_resource_policies(
        rows,
        {"memory": 0.5, "vram": 0.5},
        [0.0, 0.05, 0.1],
        min_stratum_n=3,
    )

    for resource in ("memory", "vram"):
        assert not diagnostics[resource]["stratified_selection_active"]
        assert policies[resource]["same_script_guard_budget"] == pytest.approx(
            policies[resource]["changed_script_guard_budget"]
        )
    candidates = sweep.loc[sweep["resource"] == "memory"]
    assert np.allclose(
        candidates["same_script_guard_budget"],
        candidates["changed_script_guard_budget"],
    )


def test_script_guard_rejects_test_selection_and_invalid_cold_start_relation():
    rows = script_guard_rows()
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_script_guarded_resource_policies(
            rows,
            {"memory": 0.5, "vram": 0.5},
            [0.0, 0.1],
            min_stratum_n=1,
        )

    rows.loc[0, "split"] = "validation"
    rows.loc[0, "history_available"] = False
    with pytest.raises(ValueError, match="without causal history"):
        apply_script_guarded_policies(
            rows,
            {
                "memory": {
                    "alpha": 0.5,
                    "same_script_guard_budget": 0.0,
                    "changed_script_guard_budget": 0.1,
                },
                "vram": {
                    "alpha": 0.5,
                    "same_script_guard_budget": 0.0,
                    "changed_script_guard_budget": 0.1,
                },
            },
        )


def script_history_rows():
    rows = []
    for resource in ("memory", "vram"):
        rows.extend(
            [
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.8,
                    "qwen_prediction": 0.1,
                    "last1_prediction": 0.2,
                    "history_available": True,
                    "same_script_history_available": True,
                    "same_script_history_prediction": 0.8,
                    "same_script_as_last1": False,
                },
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.7,
                    "qwen_prediction": 0.2,
                    "last1_prediction": 0.3,
                    "history_available": True,
                    "same_script_history_available": True,
                    "same_script_history_prediction": 0.7,
                    "same_script_as_last1": False,
                },
                {
                    "split": "validation",
                    "resource": resource,
                    "valid": True,
                    "target": 0.4,
                    "qwen_prediction": 0.4,
                    "last1_prediction": 0.9,
                    "history_available": False,
                    "same_script_history_available": False,
                    "same_script_history_prediction": np.nan,
                    "same_script_as_last1": False,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_script_history_selector_finds_the_same_script_expert():
    rows = script_history_rows()
    policies, diagnostics, sweep = select_safe_script_history_policies(
        rows,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        min_effective_same_script_n=1,
    )

    for resource in ("memory", "vram"):
        assert policies[resource] == {"alpha": 1.0, "same_script_beta": 1.0}
        assert diagnostics[resource]["beta_selection_active"]
    assert sweep.loc[sweep["selected"], "feasible"].all()


def test_script_history_beta_zero_exactly_nests_safe_hybrid():
    rows = script_history_rows()
    alphas = {"memory": 0.25, "vram": 0.75}
    policies = {
        resource: {"alpha": alpha, "same_script_beta": 0.0}
        for resource, alpha in alphas.items()
    }

    ordinary = apply_alphas(rows, alphas, clip_predictions=True)
    three_expert = apply_script_history_policies(rows, policies)

    assert np.array_equal(
        ordinary["prediction"].to_numpy(float),
        three_expert["prediction"].to_numpy(float),
    )


def test_script_history_reference_and_cold_start_are_exact():
    rows = script_history_rows().drop(columns="target")
    policies = {
        resource: {"alpha": 1.0, "same_script_beta": 0.0}
        for resource in ("memory", "vram")
    }

    applied = apply_script_history_policies(rows, policies)
    history = applied["history_available"].to_numpy(bool)
    assert np.array_equal(
        applied.loc[history, "prediction"].to_numpy(float),
        rows.loc[history, "last1_prediction"].to_numpy(float),
    )
    assert np.array_equal(
        applied.loc[~history, "prediction"].to_numpy(float),
        rows.loc[~history, "qwen_prediction"].to_numpy(float),
    )


def test_script_history_sparse_support_forces_beta_zero():
    rows = script_history_rows()
    policies, diagnostics, sweep = select_safe_script_history_policies(
        rows,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        min_effective_same_script_n=3,
    )

    for resource in ("memory", "vram"):
        assert policies[resource]["same_script_beta"] == pytest.approx(0.0)
        assert not diagnostics[resource]["beta_selection_active"]
    assert (sweep["same_script_beta"] == 0.0).all()


@pytest.mark.parametrize("invalid_support", [True, 1.5, -1])
def test_script_history_support_threshold_requires_a_nonnegative_integer(
    invalid_support,
):
    with pytest.raises(ValueError, match="non-negative integer"):
        select_safe_script_history_policies(
            script_history_rows(),
            [0.0, 1.0],
            [0.0, 1.0],
            min_effective_same_script_n=invalid_support,
        )


def test_script_history_reference_is_selected_per_resource_support():
    rows = script_history_rows()
    vram_same = (rows["resource"] == "vram") & rows[
        "same_script_history_available"
    ]
    rows.loc[vram_same, "same_script_as_last1"] = True
    rows.loc[vram_same, "same_script_history_prediction"] = rows.loc[
        vram_same, "last1_prediction"
    ]

    _, diagnostics, _ = select_safe_script_history_policies(
        rows,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        min_effective_same_script_n=1,
    )

    assert diagnostics["memory"]["reference_policy"] == {
        "alpha": 1.0,
        "same_script_beta": 1.0,
    }
    assert diagnostics["vram"]["reference_policy"] == {
        "alpha": 1.0,
        "same_script_beta": 0.0,
    }


def test_script_history_rejects_test_rows_and_nonfinite_available_history():
    rows = script_history_rows()
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_safe_script_history_policies(
            rows,
            [0.0, 1.0],
            [0.0, 1.0],
            min_effective_same_script_n=1,
        )

    rows.loc[0, "split"] = "validation"
    rows.loc[0, "same_script_history_prediction"] = np.nan
    with pytest.raises(ValueError, match="same-script.*non-finite"):
        select_safe_script_history_policies(
            rows,
            [0.0, 1.0],
            [0.0, 1.0],
            min_effective_same_script_n=1,
        )


def test_guarded_policy_verifies_the_frozen_safe_alpha_artifact(tmp_path):
    qwen_path = tmp_path / "qwen.csv"
    last1_path = tmp_path / "last1.csv"
    qwen_path.write_text("qwen\n", encoding="utf-8")
    last1_path.write_text("last1\n", encoding="utf-8")
    artifact_dir = tmp_path / "safe"
    artifact_dir.mkdir()
    constraints = {
        "max_acc_drop_vs_reference": 0.0,
        "max_rmse_increase_vs_reference": 0.0,
        "max_mae_increase_vs_reference": 0.0,
    }
    alphas = {"memory": 0.42, "vram": 0.75}
    record = {
        "protocol_version": "pareto_safe_fixed_blend_v1",
        "selection_objective": "constrained_mae",
        "constraints": constraints,
        "test_labels_used_for_selection": False,
        "acc_tolerance": 0.10,
        "clip_predictions": True,
        "safe_reference_policy": (
            "Last-1 when causal history is available; Qwen otherwise"
        ),
        "no_history_policy": "Qwen only",
        "alphas": alphas,
    }
    manifest = {
        "selection_protocol_version": "pareto_safe_fixed_blend_v1",
        "model": "safe_hybrid_full",
        "seed": 42,
        "data_sha256": "d" * 64,
        "split_fingerprint": "s" * 64,
        "qwen_predictions_sha256": sha256_file(qwen_path),
        "last1_predictions_sha256": sha256_file(last1_path),
        "selection_objective": "constrained_mae",
        "selection_constraints": constraints,
        "test_labels_used_for_selection": False,
        "safe_reference_policy": (
            "Last-1 when causal history is available; Qwen otherwise"
        ),
        "last1_protocol": "completed_before_submission",
        "causal_last1_required": True,
        "alphas": alphas,
    }
    alpha_path = artifact_dir / "alphas.json"
    alpha_path.write_text(json.dumps(record), encoding="utf-8")
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    frozen, provenance = _load_frozen_safe_alphas(
        alpha_path,
        qwen_path=qwen_path,
        last1_path=last1_path,
        qwen_variant="full",
        seed=42,
        data_sha256="d" * 64,
        split_fingerprint="s" * 64,
        constraints=constraints,
        tolerance=0.10,
        clip_predictions=True,
    )

    assert frozen == alphas
    assert provenance["alpha_record_sha256"] == sha256_file(alpha_path)
