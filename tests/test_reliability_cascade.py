from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common.data import DatasetBundle
from common.history import script_sha256
from reliability_cascade.evaluation import _simulate_embedding_cache
from reliability_cascade.evaluation import apply_selected_cascade
from reliability_cascade.features import CascadeFeatures, build_cascade_features
from reliability_cascade.history import build_robust_history_long
from reliability_cascade.integrity import _route_mask_sha256
from reliability_cascade.model import CascadeConfig, ResourceCascade
from reliability_cascade.pipeline import _qwen_max_tokens
from reliability_cascade.pipeline import _validate_completed_label_boundaries
from reliability_cascade.pipeline import _validate_protocol_identity
from reliability_cascade.pipeline import chronological_calibration_masks
from reliability_cascade.pipeline import resolve_component_paths
from reliability_cascade.selection import ResourceOption, _job_call_rate
from reliability_cascade.selection import select_cascade


def _bundle() -> DatasetBundle:
    submit = pd.to_datetime(
        [
            "2026-01-01T00:00Z",
            "2026-01-01T02:00Z",
            "2026-01-01T04:00Z",
            "2026-01-01T06:00Z",
            "2026-01-01T08:00Z",
            "2026-01-01T10:00Z",
            "2026-01-01T12:00Z",
        ]
    )
    completion = pd.to_datetime(
        [
            # This is equal to row b's submit time and is not visible to b.
            "2026-01-01T02:00Z",
            "2026-01-01T03:00Z",
            "2026-01-01T05:00Z",
            "2026-01-01T07:00Z",
            "2026-01-01T09:00Z",
            "2026-01-01T11:00Z",
            "2026-01-01T13:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "sample_id": list("abcdefg"),
            "source_row": np.arange(7),
            "job_id": [str(value) for value in range(7)],
            "user_id": ["u"] * 7,
            "split": ["train", "train", "train", "validation", "validation", "test", "test"],
            "submit_time": submit,
            "completion_time": completion,
            "code": ["echo x", "echo x", "echo x", "echo x", "echo x", "", ""],
            "target_memory": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
            "valid_memory": [True] * 7,
            "target_vram": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
            "valid_vram": [True] * 7,
            "gpu_count": [1] * 7,
            "num__cpu": np.arange(1, 8, dtype=float),
        }
    )
    return DatasetBundle(
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


def test_robust_history_is_strict_and_blank_scripts_never_match():
    history = build_robust_history_long(_bundle(), history_k=5, half_life_hours=24.0)
    memory = history.loc[history["resource"].eq("memory")].set_index("sample_id")
    assert not bool(memory.loc["b", "user_available"])
    assert bool(memory.loc["c", "user_available"])
    assert memory.loc["c", "user_last_value"] == pytest.approx(0.20)
    assert memory.loc["e", "exact_count"] == 4
    assert memory.loc["e", "exact_median3"] == pytest.approx(0.30)
    assert memory.loc["e", "exact_mad5"] == pytest.approx(0.10)
    assert not bool(memory.loc["g", "exact_available"])
    assert memory.loc["g", "script_observed"] == np.bool_(False)


def _paired_from_bundle(bundle: DatasetBundle) -> pd.DataFrame:
    history = build_robust_history_long(bundle, history_k=5, half_life_hours=24.0)
    context = bundle.frame.set_index("sample_id")
    rows = []
    for row in history.itertuples(index=False):
        current = context.loc[row.sample_id]
        resource = row.resource
        target = current[f"target_{resource}"]
        available = bool(row.user_available)
        last1 = row.user_last_value if available else 0.25
        rows.append(
            {
                "sample_id": row.sample_id,
                "split": current["split"],
                "resource": resource,
                "target": target,
                "valid": True,
                "qwen_prediction": 0.5,
                "last1_prediction": last1,
                "history_available": available,
                "history_source_sample_id": row.user_source_sample_id,
                "history_source_time": row.user_source_completion_time,
                "gate_code": 0.4,
                "gate_user": 0.3,
                "gate_metadata": 0.3,
                "user_known": True,
                "script_sha256": script_sha256(current["code"]),
            }
        )
    return pd.DataFrame(rows)


def test_pre_route_features_contain_no_qwen_or_outcome_fields():
    bundle = _bundle()
    paired = _paired_from_bundle(bundle)
    features = build_cascade_features(
        paired, bundle, history_k=5, half_life_hours=24.0
    )
    assert not any("qwen" in name.lower() for name in features.pre_names)
    assert not any("target" in name.lower() for name in features.pre_names)
    original = features.pre_values.copy()
    paired["qwen_prediction"] = np.linspace(0.0, 1.0, len(paired))
    changed = build_cascade_features(
        paired, bundle, history_k=5, half_life_hours=24.0
    )
    np.testing.assert_array_equal(original, changed.pre_values)


def test_boundary_purged_history_source_retains_submission_metadata():
    original = _bundle()
    retained = original.frame.copy()
    effective = retained.loc[retained["sample_id"].ne("d")].copy()
    bundle = DatasetBundle(
        frame=effective,
        retained_frame=retained,
        data_path=original.data_path,
        data_sha256=original.data_sha256,
        manifest_path=original.manifest_path,
        numeric_features=original.numeric_features,
        categorical_features=original.categorical_features,
        protocol=original.protocol,
    )
    features = build_cascade_features(
        _paired_from_bundle(bundle),
        bundle,
        history_k=5,
        half_life_hours=24.0,
    )
    row = features.frame["sample_id"].eq("e") & features.frame["resource"].eq(
        "memory"
    )
    index = int(np.flatnonzero(row.to_numpy())[0])
    available = features.pre_names.index("source_metadata_available")
    cpu_delta = features.pre_names.index("request_abs_delta__cpu")
    assert features.pre_values[index, available] == pytest.approx(1.0)
    assert features.pre_values[index, cpu_delta] == pytest.approx(1.0)


def test_qwen_context_contract_is_explicit_and_drives_default_path():
    manifest = {"details": {"effective_parameters": {"max_tokens": 4096}}}
    assert _qwen_max_tokens(manifest) == 4096
    with pytest.raises(ValueError, match="max_tokens"):
        _qwen_max_tokens({"details": {"effective_parameters": {}}})
    qwen, _ = resolve_component_paths(
        {
            "seed": 7,
            "qwen": {"max_tokens": 2048},
            "last1": {"protocol": "completed_before_submission"},
            "reliability_cascade": {"qwen_variant": "full"},
        },
        None,
        None,
    )
    assert "tokens_2048" in qwen.as_posix()


def test_cascade_accepts_completed_zero_crossing_boundary_audit():
    bundle = SimpleNamespace(
        protocol={
            "completed_labels_at_boundaries": {
                "required": True,
                "checked": True,
                "remaining_train_cross_boundary_count": 0,
                "remaining_validation_cross_boundary_count": 0,
            }
        }
    )
    _validate_completed_label_boundaries(bundle)


@pytest.mark.parametrize(
    "boundary,match",
    [
        (None, "protocol metadata"),
        (
            {
                "required": False,
                "checked": True,
                "remaining_train_cross_boundary_count": 0,
                "remaining_validation_cross_boundary_count": 0,
            },
            "required=true",
        ),
        (
            {
                "required": True,
                "checked": False,
                "remaining_train_cross_boundary_count": 0,
                "remaining_validation_cross_boundary_count": 0,
            },
            "checked=true",
        ),
        (
            {
                "required": True,
                "checked": True,
                "remaining_train_cross_boundary_count": 1,
                "remaining_validation_cross_boundary_count": 0,
            },
            "remaining_train_cross_boundary_count=0",
        ),
        (
            {
                "required": True,
                "checked": True,
                "remaining_train_cross_boundary_count": 0,
                "remaining_validation_cross_boundary_count": 1,
            },
            "remaining_validation_cross_boundary_count=0",
        ),
    ],
)
def test_cascade_rejects_unsafe_or_incomplete_boundary_audit(boundary, match):
    bundle = SimpleNamespace(
        protocol=(
            {}
            if boundary is None
            else {"completed_labels_at_boundaries": boundary}
        )
    )
    with pytest.raises(ValueError, match=match):
        _validate_completed_label_boundaries(bundle)


def test_component_manifests_cannot_agree_on_an_unsafe_boundary_protocol():
    protocol = {
        "order_fingerprint": "order",
        "split_manifest_sha256": "a" * 64,
        "code_redaction": {"enabled": True},
        "event_time_validation": {"checked": True},
        "completed_labels_at_boundaries": {
            "required": False,
            "checked": False,
            "remaining_train_cross_boundary_count": 0,
            "remaining_validation_cross_boundary_count": 0,
        },
    }
    inputs = SimpleNamespace(
        qwen_manifest={"protocol": protocol},
        last1_manifest={"protocol": protocol},
    )
    with pytest.raises(ValueError, match="Qwen artifact.*required=true"):
        _validate_protocol_identity(inputs)


def _manual_features() -> CascadeFeatures:
    n = 16
    history = np.ones(n, dtype=bool)
    history[-2:] = False
    exact = history.copy()
    qwen = np.linspace(0.15, 0.75, n)
    historical = np.linspace(0.20, 0.70, n)
    target = historical.copy()
    target[1::2] = qwen[1::2]
    frame = pd.DataFrame(
        {
            "sample_id": [f"j{i}" for i in range(n)],
            "job_id": [str(i) for i in range(n)],
            "user_id": ["u"] * n,
            "split": ["validation"] * 12 + ["test"] * 4,
            "resource": ["memory"] * n,
            "target": target,
            "valid": True,
            "qwen_prediction": qwen,
            "last1_prediction": np.where(history, historical, 0.1),
            "user_available": history,
            "exact_available": exact,
            "resource_applicable": True,
            "exact_mad5": 0.01,
            "exact_last_value": historical,
            "user_last_value": historical,
            "exact_median3": historical,
            "user_median3": historical,
            "exact_median5": historical,
            "user_median5": historical,
            "exact_decayed_mean": historical,
            "user_decayed_mean": historical,
        }
    )
    pre = np.column_stack(
        [history.astype(float), exact.astype(float), np.linspace(0.0, 1.0, n)]
    )
    return CascadeFeatures(
        frame=frame,
        pre_values=pre,
        pre_names=["history_available", "exact_available", "history_age"],
        post_base_values=np.zeros((n, 5), dtype=float),
        post_base_names=[
            "qwen_gate_code",
            "qwen_gate_user",
            "qwen_gate_metadata",
            "qwen_gate_entropy",
            "qwen_user_known",
        ],
        history_contract={},
    )


def test_route_is_pre_qwen_and_cold_start_is_exact_qwen():
    features = _manual_features()
    fit = np.arange(len(features.frame)) < 12
    config = CascadeConfig(
        seed=7,
        pre_n_estimators=5,
        post_n_estimators=5,
        trust_region_enabled=False,
    )
    model = ResourceCascade.fit(
        features,
        resource="memory",
        candidate_name="hierarchical_last",
        fit_mask=fit,
        config=config,
        threshold=0.0,
    )
    before = model.predict(features)
    assert before["history_bypass"][:-2].all()
    np.testing.assert_allclose(
        before["prediction"][-2:],
        features.frame["qwen_prediction"].to_numpy(float)[-2:],
    )

    mutated = _manual_features()
    mutated.frame["qwen_prediction"] = 1.0 - mutated.frame["qwen_prediction"]
    after = model.predict(mutated)
    np.testing.assert_array_equal(
        before["history_bypass"], after["history_bypass"]
    )
    np.testing.assert_array_equal(
        before["qwen_required_resource"], after["qwen_required_resource"]
    )


def test_disabling_post_blend_makes_the_qwen_path_exact_qwen():
    features = _manual_features()
    fit = np.arange(len(features.frame)) < 12
    model = ResourceCascade.fit(
        features,
        resource="memory",
        candidate_name="hierarchical_last",
        fit_mask=fit,
        config=CascadeConfig(
            seed=7,
            pre_n_estimators=5,
            post_n_estimators=5,
            post_blend_enabled=False,
            trust_region_enabled=True,
        ),
        threshold=1.01,
    )
    output = model.predict(features)
    np.testing.assert_allclose(
        output["prediction"],
        features.frame["qwen_prediction"].to_numpy(float),
    )
    assert not output["trust_region_applied"].any()


def test_job_call_rate_uses_or_across_resource_heads():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "a", "b"],
            "resource": ["memory", "memory", "vram", "vram"],
            "resource_applicable": True,
        }
    )
    prepared = SimpleNamespace(features=SimpleNamespace(frame=frame))
    memory_required = np.array([True, False, False, False])
    vram_required = np.array([False, False, False, True])
    base = {
        "resource_mask": np.array([True, True, False, False]),
        "qwen_required_resource": memory_required,
    }
    memory = ResourceOption("m", "memory", None, base, {}, 0.5)
    vram = ResourceOption(
        "v",
        "vram",
        None,
        {
            "resource_mask": np.array([False, False, True, True]),
            "qwen_required_resource": vram_required,
        },
        {},
        0.5,
    )
    n, calls, rate = _job_call_rate(
        prepared, memory, vram, np.ones(4, dtype=bool)
    )
    assert (n, calls, rate) == (2, 2, 1.0)


def test_bypassed_jobs_do_not_populate_embedding_cache():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "resource": ["memory"] * 3,
            "split": ["validation", "validation", "test"],
            "submit_time": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T02:00Z"]
            ),
            "script_observed": True,
            "script_sha256_history": ["x"] * 3,
        }
    )
    required = pd.Series({"a": False, "b": True, "c": True})
    replay = _simulate_embedding_cache(frame, required).set_index("sample_id")
    assert not bool(replay.loc["a", "embedding_cache_hit_job"])
    assert not bool(replay.loc["b", "embedding_cache_hit_job"])
    assert bool(replay.loc["c", "embedding_cache_hit_job"])


def test_equal_submit_time_jobs_do_not_populate_each_others_cache():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "resource": ["memory"] * 3,
            "split": ["validation", "validation", "test"],
            "submit_time": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T00:00Z", "2026-01-01T01:00Z"]
            ),
            "script_observed": True,
            "script_sha256_history": ["x"] * 3,
        }
    )
    required = pd.Series({"a": True, "b": True, "c": True})
    replay = _simulate_embedding_cache(frame, required).set_index("sample_id")
    assert not bool(replay.loc["a", "embedding_cache_hit_job"])
    assert not bool(replay.loc["b", "embedding_cache_hit_job"])
    assert bool(replay.loc["c", "embedding_cache_hit_job"])


def test_validation_masks_are_chronological_and_exclude_test():
    rows = []
    for index in range(8):
        split = "validation" if index < 6 else "test"
        for resource in ("memory", "vram"):
            rows.append(
                {
                    "sample_id": f"j{index}",
                    "resource": resource,
                    "split": split,
                    "submit_time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "completion_time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=index, minutes=30),
                }
            )
    prepared = SimpleNamespace(features=SimpleNamespace(frame=pd.DataFrame(rows)))
    fit, selection, protocol = chronological_calibration_masks(
        prepared,
        {"reliability_cascade": {"validation_fit_fraction": 0.5}},
    )
    frame = prepared.features.frame
    assert not np.any(frame["split"].eq("test").to_numpy() & (fit | selection))
    assert protocol["test_used_for_fit_or_selection"] is False
    assert set(frame.loc[fit, "sample_id"]) == {"j0", "j1", "j2"}
    assert set(frame.loc[selection, "sample_id"]) == {"j3", "j4", "j5"}


def test_validation_masks_keep_equal_submit_times_together_and_purge_crossers():
    jobs = ["a", "b", "c", "d", "e", "f", "t"]
    submit = pd.to_datetime(
        [
            "2026-01-01T00:00Z",
            "2026-01-01T01:00Z",
            "2026-01-01T02:00Z",
            "2026-01-01T02:00Z",
            "2026-01-01T02:00Z",
            "2026-01-01T03:00Z",
            "2026-01-01T04:00Z",
        ]
    )
    completion = pd.to_datetime(
        [
            "2026-01-01T00:30Z",
            "2026-01-01T02:30Z",  # Crosses the inner selection boundary.
            "2026-01-01T02:30Z",
            "2026-01-01T02:30Z",
            "2026-01-01T02:30Z",
            "2026-01-01T03:30Z",
            "2026-01-01T04:30Z",
        ]
    )
    rows = []
    for index, sample_id in enumerate(jobs):
        for resource in ("memory", "vram"):
            rows.append(
                {
                    "sample_id": sample_id,
                    "resource": resource,
                    "split": "validation" if sample_id != "t" else "test",
                    "submit_time": submit[index],
                    "completion_time": completion[index],
                }
            )
    prepared = SimpleNamespace(features=SimpleNamespace(frame=pd.DataFrame(rows)))
    fit, selection, protocol = chronological_calibration_masks(
        prepared,
        {"reliability_cascade": {"validation_fit_fraction": 0.5}},
    )
    frame = prepared.features.frame
    assert set(frame.loc[fit, "sample_id"]) == {"a"}
    assert set(frame.loc[selection, "sample_id"]) == {"c", "d", "e", "f"}
    assert protocol["inner_boundary_purged_fit_jobs"] == 1
    assert protocol["equal_submit_times_split_across_blocks"] is False


def _dual_resource_features() -> CascadeFeatures:
    jobs = 20
    rows = []
    pre = []
    post = []
    for resource_index, resource in enumerate(("memory", "vram")):
        for index in range(jobs):
            available = index >= 2
            historical = 0.15 + 0.02 * index + 0.03 * resource_index
            qwen = 0.20 + 0.015 * index + 0.02 * resource_index
            target = historical if index % 2 == 0 else qwen
            rows.append(
                {
                    "sample_id": f"j{index}",
                    "job_id": str(index),
                    "user_id": "u",
                    "split": "validation" if index < 16 else "test",
                    "resource": resource,
                    "target": target,
                    "valid": True,
                    "qwen_prediction": qwen,
                    "last1_prediction": historical if available else 0.1,
                    "user_available": available,
                    "exact_available": available,
                    "resource_applicable": True,
                    "exact_mad5": 0.01,
                    "exact_last_value": historical,
                    "user_last_value": historical,
                    "exact_median3": historical,
                    "user_median3": historical,
                    "exact_median5": historical,
                    "user_median5": historical,
                    "exact_decayed_mean": historical,
                    "user_decayed_mean": historical,
                    "same_last1": True,
                    "user_count": index,
                    "user_age_hours": 1.0,
                    "user_mad5": 0.01,
                    "exact_count": index,
                    "exact_age_hours": 1.0,
                    "script_observed": True,
                    "script_sha256_history": f"script-{index // 2}",
                    "submit_time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "gpu_count": 1,
                }
            )
            pre.append([float(available), float(available), index / jobs])
            post.append([0.4, 0.3, 0.3, 1.0, 1.0])
    return CascadeFeatures(
        frame=pd.DataFrame(rows),
        pre_values=np.asarray(pre, dtype=float),
        pre_names=["history_available", "exact_available", "history_age"],
        post_base_values=np.asarray(post, dtype=float),
        post_base_names=[
            "qwen_gate_code",
            "qwen_gate_user",
            "qwen_gate_metadata",
            "qwen_gate_entropy",
            "qwen_user_known",
        ],
        history_contract={"version": "test"},
    )


def test_non_applicable_vram_row_never_bypasses_or_requires_qwen():
    features = _dual_resource_features()
    frame = features.frame
    fit = frame["split"].eq("validation").to_numpy()
    model = ResourceCascade.fit(
        features,
        resource="vram",
        candidate_name="hierarchical_last",
        fit_mask=fit,
        config=CascadeConfig(
            seed=3,
            pre_n_estimators=5,
            post_n_estimators=5,
            trust_region_enabled=False,
        ),
        threshold=0.0,
    )
    row = frame["resource"].eq("vram") & frame["sample_id"].eq("j19")
    frame.loc[row, "resource_applicable"] = False
    output = model.predict(features)
    index = int(np.flatnonzero(row.to_numpy())[0])
    assert not bool(output["history_bypass"][index])
    assert not bool(output["qwen_required_resource"][index])


def test_joint_selection_and_evaluation_emit_auditable_job_routes(tmp_path):
    features = _dual_resource_features()
    frame = features.frame
    fit = frame["split"].eq("validation").to_numpy() & (
        frame["sample_id"].str.removeprefix("j").astype(int).to_numpy() < 10
    )
    selection = frame["split"].eq("validation").to_numpy() & ~fit
    prepared = SimpleNamespace(features=features)
    config = {
        "evaluation": {"acc_tolerance": 0.10},
        "reliability_cascade": {
            "history_experts": ["hierarchical_last"],
            "refit_on_full_validation": False,
            "pre_gate": {"thresholds": [0.0, 1.01]},
            "selection": {
                "objective": "constrained_efficiency",
                "max_mae_increase_vs_accuracy_reference": 1.0,
                "max_rmse_increase_vs_accuracy_reference": 1.0,
                "max_acc_drop_vs_accuracy_reference": 1.0,
            },
        },
    }
    selected = select_cascade(
        prepared,
        config,
        fit,
        selection,
        CascadeConfig(
            seed=3,
            pre_n_estimators=5,
            post_n_estimators=5,
            trust_region_enabled=False,
        ),
    )
    assert int(selected.sweep["selected"].sum()) == 1
    predictions, metrics, references, routing, importance, route_hash = (
        apply_selected_cascade(
            prepared,
            selected,
            model_name="reliability_cascade_full",
            seed=3,
        )
    )
    assert len(metrics) == 4
    assert len(references) == 4
    assert len(route_hash) == 64
    prediction_path = tmp_path / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    assert _route_mask_sha256(prediction_path) == route_hash
    assert not routing.empty
    assert not importance.empty
    per_job = predictions.groupby("sample_id")["qwen_required_job"].nunique()
    assert (per_job == 1).all()
    assert set(predictions["route_reason"]).issubset(
        {
            "reliable_history_bypass",
            "cold_start_qwen",
            "unreliable_exact_history_qwen_blend",
        }
    )
