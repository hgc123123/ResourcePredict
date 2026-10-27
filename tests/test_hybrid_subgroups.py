import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.hybrid_subgroups import (
    MODELS,
    SUBGROUPS,
    assign_history_subgroups,
    audit_reference_metrics,
    build_comparison_predictions,
    compute_subgroup_metrics,
    default_analysis_output,
    load_qwen_training_manifest,
    prepare_output_directory,
    qwen_numeric_feature_provenance,
    qwen_provenance,
    validate_selection_contract,
)
from common.metrics import compute_metrics
from qwen.contract import contract_sha256


def _rows() -> pd.DataFrame:
    definitions = (
        # group, ordinary history, same-script history, same source
        ("same_last1", True, True, True),
        ("older_same_script", True, True, False),
        ("no_same_script_history", True, False, False),
        ("cold_start", False, False, False),
    )
    rows = []
    sample = 0
    for split in ("validation", "test"):
        for resource in ("memory", "vram"):
            for expected, history, same_available, same_last1 in definitions:
                rows.append(
                    {
                        "sample_id": f"s{sample}",
                        "job_id": str(sample),
                        "user_id": f"u{sample % 3}",
                        "split": split,
                        "resource": resource,
                        "target": 0.50,
                        "valid": True,
                        "qwen_prediction": 0.55,
                        "last1_prediction": 0.65 if history else np.nan,
                        "same_script_history_prediction": (
                            0.65
                            if same_last1
                            else 0.45
                            if same_available
                            else np.nan
                        ),
                        "history_available": history,
                        "same_script_history_available": same_available,
                        "same_script_as_last1": same_last1,
                        "prediction": 0.52 if history else 0.55,
                        "expected_subgroup": expected,
                    }
                )
                sample += 1
    return pd.DataFrame(rows)


def test_history_groups_are_exhaustive_and_do_not_use_targets():
    rows = _rows()
    assigned = assign_history_subgroups(rows)
    assert assigned.tolist() == rows["expected_subgroup"].tolist()
    assert set(assigned) == set(SUBGROUPS)

    changed_targets = rows.copy()
    changed_targets["target"] = np.linspace(-100.0, 100.0, len(rows))
    changed_targets["prediction"] = np.linspace(100.0, -100.0, len(rows))
    assert assign_history_subgroups(changed_targets).equals(assigned)


@pytest.mark.parametrize(
    "column,index",
    [
        ("same_script_history_available", 3),
        ("same_script_as_last1", 2),
    ],
)
def test_history_groups_reject_impossible_causal_relations(column, index):
    rows = _rows()
    rows.loc[index, column] = True
    with pytest.raises(ValueError):
        assign_history_subgroups(rows)


def test_comparison_predictions_use_fixed_cold_start_and_hierarchy_rules():
    rows = _rows().loc[lambda frame: frame["split"].eq("validation")].copy()
    comparisons = build_comparison_predictions(rows, clip_predictions=True)

    relabeled = rows.copy()
    relabeled["target"] = np.linspace(-10.0, 10.0, len(relabeled))
    relabeled_comparisons = build_comparison_predictions(
        relabeled, clip_predictions=True
    )
    assert np.allclose(
        comparisons["prediction"],
        relabeled_comparisons["prediction"],
        rtol=0.0,
        atol=0.0,
    )

    cold = comparisons.loc[comparisons["subgroup"].eq("cold_start")]
    assert cold.groupby(["sample_id", "resource"])["prediction"].nunique().eq(1).all()
    assert cold["prediction"].eq(0.55).all()

    older = comparisons.loc[comparisons["subgroup"].eq("older_same_script")]
    ordinary = older.loc[
        older["model"].eq("ordinary_last1_with_qwen_cold_start"), "prediction"
    ]
    hierarchical = older.loc[
        older["model"].eq("hierarchical_same_script_with_qwen_cold_start"),
        "prediction",
    ]
    assert ordinary.eq(0.65).all()
    assert hierarchical.eq(0.45).all()
    assert comparisons["subgroup_assignment_uses_targets"].eq(False).all()
    assert comparisons["model_selection_performed"].eq(False).all()


def test_subgroup_metrics_cover_all_cells_and_match_manual_values():
    comparisons = build_comparison_predictions(_rows(), clip_predictions=True)
    metrics = compute_subgroup_metrics(comparisons)
    assert len(metrics) == 2 * 2 * len(SUBGROUPS) * len(MODELS)
    assert metrics["n"].eq(1).all()
    assert metrics["subgroup_assignment_uses_targets"].eq(False).all()
    assert metrics["model_selection_performed"].eq(False).all()

    selected = metrics.loc[
        metrics["split"].eq("test")
        & metrics["resource"].eq("memory")
        & metrics["subgroup"].eq("older_same_script")
        & metrics["model"].eq(
            "hierarchical_same_script_with_qwen_cold_start"
        )
    ].iloc[0]
    assert selected["mae"] == pytest.approx(0.05)
    assert selected["mse"] == pytest.approx(0.0025)
    assert selected["rmse"] == pytest.approx(0.05)
    assert selected["acc_at_10"] == pytest.approx(1.0)


def _selection_artifacts():
    policies = {
        "memory": {"alpha": 0.16, "same_script_beta": 1.0},
        "vram": {"alpha": 0.87, "same_script_beta": 1.0},
    }
    safe_references = {
        "memory": {"alpha": 1.0, "same_script_beta": 1.0},
        "vram": {"alpha": 1.0, "same_script_beta": 1.0},
    }
    manifest = {
        "model": "safe_script_history_hybrid_full",
        "seed": 42,
        "selection_protocol_version": "pareto_safe_script_history_blend_v1",
        "selection_objective": "script_history_constrained_mae",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "last1_protocol": "completed_before_submission",
        "causal_last1_required": True,
        "script_history_policies": policies,
        "safe_reference_policies": copy.deepcopy(safe_references),
    }
    record = {
        "protocol_version": "pareto_safe_script_history_blend_v1",
        "selection_objective": "script_history_constrained_mae",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidate_policy_frozen_before_test_evaluation": True,
        "no_history_policy": "Qwen only",
        "acc_tolerance": 0.10,
        "clip_predictions": True,
        "script_history_policies": policies,
        "safe_reference_policies": copy.deepcopy(safe_references),
    }
    return manifest, record


def test_selection_contract_accepts_only_frozen_validation_selected_policy():
    manifest, record = _selection_artifacts()
    assert validate_selection_contract(
        manifest, record, variant="full", seed=42
    ) == manifest["script_history_policies"]

    for field, bad_value in (
        ("selection_split", "test"),
        ("test_labels_used_for_selection", True),
        ("candidate_policy_frozen_before_test_evaluation", False),
    ):
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest[field] = bad_value
        with pytest.raises(ValueError):
            validate_selection_contract(
                bad_manifest, record, variant="full", seed=42
            )
    diagnostic = copy.deepcopy(manifest)
    diagnostic["uses_targets"] = True
    with pytest.raises(ValueError):
        validate_selection_contract(diagnostic, record, variant="full", seed=42)


def test_selection_contract_rejects_sparse_fallback_as_hierarchical():
    manifest, record = _selection_artifacts()
    for artifact in (manifest, record):
        artifact["safe_reference_policies"]["memory"]["same_script_beta"] = 0.0
    with pytest.raises(ValueError, match="not hierarchical"):
        validate_selection_contract(manifest, record, variant="full", seed=42)


def test_qwen_provenance_binds_context_contract_and_runtime():
    effective = {
        "max_tokens": 4096,
        "effective_global_batch_size": 32,
    }
    runtime = {"elapsed_seconds": 123.0, "world_size": 8}
    contract_sha = "a" * 64
    qwen_manifest = {
        "details": {
            "effective_parameters": effective,
            "experiment_contract_sha256": contract_sha,
            "training_runtime": runtime,
        }
    }
    hybrid_manifest = {
        "qwen_effective_parameters": copy.deepcopy(effective),
        "qwen_experiment_contract_sha256": contract_sha,
        "qwen_training_runtime": copy.deepcopy(runtime),
    }
    provenance = qwen_provenance(qwen_manifest, hybrid_manifest)
    assert provenance == {
        "max_tokens": 4096,
        "experiment_contract_sha256": contract_sha,
        "effective_parameters": effective,
        "training_runtime": runtime,
        "effective_parameters_source": "Qwen evaluation manifest",
        "training_runtime_source": "Qwen evaluation manifest",
        "training_runtime_available": True,
    }

    tampered = copy.deepcopy(hybrid_manifest)
    tampered["qwen_effective_parameters"]["max_tokens"] = 2048
    with pytest.raises(ValueError, match="qwen_effective_parameters"):
        qwen_provenance(qwen_manifest, tampered)


def test_qwen_provenance_recovers_legacy_fields_from_training_manifest():
    effective = {"max_tokens": 4096, "effective_global_batch_size": 32}
    payload = {
        "seed": 42,
        "qwen_semantics": {"variant": "full"},
        "effective_qwen_parameters": copy.deepcopy(effective),
    }
    contract_sha = contract_sha256(payload)
    training_manifest = {
        "effective_parameters": copy.deepcopy(effective),
        "experiment_contract": {
            "sha256": contract_sha,
            "payload": payload,
        },
    }

    provenance = qwen_provenance(
        {"details": {"variant": "full"}},
        {},
        training_manifest,
    )
    assert provenance["effective_parameters"] == effective
    assert provenance["effective_parameters_source"] == "Qwen training manifest"
    assert provenance["experiment_contract_sha256"] == contract_sha
    assert provenance["training_runtime"] is None
    assert provenance["training_runtime_available"] is False

    missing_effective_contract = copy.deepcopy(training_manifest)
    del missing_effective_contract["experiment_contract"]["payload"][
        "effective_qwen_parameters"
    ]
    missing_effective_contract["experiment_contract"]["sha256"] = contract_sha256(
        missing_effective_contract["experiment_contract"]["payload"]
    )
    with pytest.raises(ValueError, match="lacks effective_qwen_parameters"):
        qwen_provenance(
            {"details": {"variant": "full"}},
            {},
            missing_effective_contract,
        )


def test_qwen_training_manifest_loader_checks_legacy_checkpoint_identity(
    tmp_path: Path,
):
    run_dir = tmp_path / "qwen"
    qwen_path = run_dir / "predictions.csv"
    qwen_path.parent.mkdir(parents=True)
    qwen_path.write_text("sample_id\n", encoding="utf-8")
    training_path = run_dir / "checkpoint" / "training_manifest.json"
    training_path.parent.mkdir(parents=True)
    training = {
        "seed": 42,
        "variant": "full",
        "enabled_modalities": ["code", "user", "metadata"],
        "fusion_mode": "gated",
        "data_sha256": "a" * 64,
        "split_fingerprint": "b" * 64,
        "experiment_contract": {
            "version": 4,
            "payload": {
                "contract_version": 4,
                "seed": 42,
                "qwen_semantics": {
                    "variant": "full",
                    "enabled_modalities": ["code", "user", "metadata"],
                    "fusion_mode": "gated",
                },
            }
        },
    }
    training_path.write_text(json.dumps(training), encoding="utf-8")
    evaluated = {
        "data_sha256": "a" * 64,
        "split_fingerprint": "b" * 64,
        "details": {},
    }

    loaded, loaded_path = load_qwen_training_manifest(
        {}, qwen_path, evaluated, variant="full", seed=42
    )
    assert loaded == training
    assert loaded_path == training_path.resolve()

    evaluated["data_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="data_sha256"):
        load_qwen_training_manifest(
            {}, qwen_path, evaluated, variant="full", seed=42
        )


def test_qwen_numeric_feature_provenance_checks_order_and_legacy_absence():
    features = ["num__cpu", "num__memory"]
    verified = qwen_numeric_feature_provenance(
        {"numeric_features": features.copy()},
        features,
    )
    assert verified == {
        "numeric_features": features,
        "numeric_features_source": "Qwen training manifest",
        "numeric_features_verified": True,
    }

    legacy = qwen_numeric_feature_provenance({}, features)
    assert legacy == {
        "numeric_features": None,
        "numeric_features_source": None,
        "numeric_features_verified": False,
    }

    for malformed in (
        {"numeric_features": list(reversed(features))},
        {"numeric_features": ["num__cpu"]},
        {"numeric_features": "num__cpu"},
        {"numeric_features": ["num__cpu", "num__cpu"]},
    ):
        with pytest.raises(ValueError, match="numeric_features"):
            qwen_numeric_feature_provenance(malformed, features)


def test_token_scoped_default_output_and_overwrite_guard(tmp_path: Path):
    output = default_analysis_output(
        {"outputs": {"root": str(tmp_path / "paper")}},
        protocol="completed_before_submission",
        variant="full",
        seed=42,
        max_tokens=4096,
    )
    assert output == (
        tmp_path
        / "paper"
        / "analysis"
        / "hybrid_subgroups"
        / "completed_before_submission"
        / "tokens_4096"
        / "full"
        / "seed_42"
    )

    prepare_output_directory(output, overwrite=False)
    (output / "subgroup_metrics.csv").write_text("n\n1\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_output_directory(output, overwrite=False)
    prepare_output_directory(output, overwrite=True)

    protected = output / "run_manifest.json"
    with pytest.raises(ValueError, match="overwrite an input artifact"):
        prepare_output_directory(
            output,
            overwrite=True,
            protected_paths=(protected,),
        )


def _hybrid_reference_frame() -> pd.DataFrame:
    frame = _rows().copy()
    frame["model"] = "safe_script_history_hybrid_full"
    frame["seed"] = 42
    frame["ordinary_safe_reference_prediction"] = np.where(
        frame["history_available"],
        frame["last1_prediction"],
        frame["qwen_prediction"],
    )
    frame["safe_reference_prediction"] = frame[
        "ordinary_safe_reference_prediction"
    ].copy()
    same = frame["same_script_history_available"]
    frame.loc[same, "safe_reference_prediction"] = frame.loc[
        same, "same_script_history_prediction"
    ]
    return frame


def test_reference_metric_audit_detects_tampering():
    hybrid = _hybrid_reference_frame()
    expected = []
    for reference, column in (
        (
            "ordinary_last1_with_qwen_cold_start",
            "ordinary_safe_reference_prediction",
        ),
        (
            "hierarchical_same_script_with_qwen_cold_start",
            "safe_reference_prediction",
        ),
    ):
        frame = hybrid.copy()
        frame["prediction"] = frame[column]
        table = compute_metrics(frame)
        table.insert(0, "reference", reference)
        expected.append(table)
    artifact = pd.concat(expected, ignore_index=True)
    audit_reference_metrics(hybrid, artifact, tolerance=0.10)

    tampered = artifact.copy()
    tampered.loc[0, "mae"] += 0.01
    with pytest.raises(ValueError):
        audit_reference_metrics(hybrid, tampered, tolerance=0.10)
