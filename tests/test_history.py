from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.data import DatasetBundle
from common.history import (
    build_causal_history_features,
    build_causal_history_long,
    causal_history_contract,
    script_sha256,
)
from hybrid.script_context import attach_script_history_relation


def make_bundle(
    frame: pd.DataFrame,
    tmp_path: Path,
    *,
    retained: pd.DataFrame | None = None,
) -> DatasetBundle:
    return DatasetBundle(
        frame=frame.reset_index(drop=True),
        retained_frame=None if retained is None else retained.reset_index(drop=True),
        data_path=tmp_path / "fixture.csv",
        data_sha256="fixture",
        manifest_path=None,
        numeric_features=[],
        categorical_features=[],
        protocol={
            "verified_chronological": True,
            "has_submission_time": True,
            "has_completion_time": True,
        },
    )


def load_last1_module():
    path = Path(__file__).resolve().parents[1] / "last-1" / "run.py"
    spec = importlib.util.spec_from_file_location("history_test_last1_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_relation_matches_the_causal_last1_source(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["source", "same", "changed"],
            "source_row": [0, 1, 2],
            "user_id": ["u", "u", "u"],
            "code": ["run-a", "run-a", "run-b"],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:02Z",
                    "2026-01-01T00:03Z",
                ],
                utc=True,
            ),
            "completion_time": pd.to_datetime(
                [
                    "2026-01-01T00:01Z",
                    "2026-01-01T00:10Z",
                    "2026-01-01T00:11Z",
                ],
                utc=True,
            ),
            "target_memory": [0.2, 0.4, 0.6],
            "valid_memory": [True, True, True],
            "target_vram": [0.3, 0.5, 0.7],
            "valid_vram": [True, True, True],
        }
    )
    paired = pd.DataFrame(
        {
            "sample_id": ["same", "changed", "same", "changed"],
            "resource": ["memory", "memory", "vram", "vram"],
            "history_available": [True, True, True, True],
            "last1_prediction": [0.2, 0.2, 0.3, 0.3],
            "history_source_sample_id": ["source"] * 4,
            "script_sha256": [
                script_sha256("run-a"),
                script_sha256("run-b"),
                script_sha256("run-a"),
                script_sha256("run-b"),
            ],
        }
    )

    mismatched = paired.copy()
    mismatched.loc[0, "script_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Qwen script hashes"):
        attach_script_history_relation(
            mismatched,
            make_bundle(frame, tmp_path),
        )

    result = attach_script_history_relation(
        paired,
        make_bundle(frame, tmp_path),
    )

    assert result["same_script_as_last1"].tolist() == [True, False, True, False]
    assert result["history_script_relation"].tolist() == [
        "same_script",
        "changed_script",
        "same_script",
        "changed_script",
    ]


def test_strict_completion_last_k_ema_and_resource_mask(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "equal", "d", "later"],
            "source_row": [0, 1, 2, 3, 4, 5],
            "user_id": ["u"] * 6,
            "code": [
                "script-a",
                "script-b",
                "script-b",
                "script-a",
                "script-a",
                "script-new",
            ],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:05Z",
                    "2026-01-01T00:07Z",
                    "2026-01-01T00:10Z",
                    "2026-01-01T00:11Z",
                    "2026-01-01T00:21Z",
                ],
                utc=True,
            ),
            "completion_time": pd.to_datetime(
                [
                    "2026-01-01T00:10Z",
                    "2026-01-01T00:06Z",
                    "2026-01-01T00:20Z",
                    "2026-01-01T00:40Z",
                    "2026-01-01T00:30Z",
                    "2026-01-01T00:50Z",
                ],
                utc=True,
            ),
            "target_memory": [0.2, 0.8, 0.4, 0.6, 0.5, 0.7],
            "valid_memory": [True] * 6,
            # b is deliberately invalid for VRAM and must not update its history.
            "target_vram": [0.3, 0.8, 0.4, 0.6, 0.5, 0.7],
            "valid_vram": [True, False, True, True, True, True],
        }
    )
    features = build_causal_history_features(
        make_bundle(frame, tmp_path), k=2, ema_alpha=0.5
    ).set_index("sample_id")

    # At 00:07 only b has completed.
    row = features.loc["c"]
    assert row["memory_history_available"]
    assert row["memory_history_count"] == 1
    assert row["memory_history_last_value"] == pytest.approx(0.8)
    assert row["memory_history_age_seconds"] == pytest.approx(60.0)
    assert row["memory_history_same_script_last_value"] == pytest.approx(0.8)
    assert not row["vram_history_available"]

    # a completes exactly when this row is submitted, so strict '<' excludes a.
    row = features.loc["equal"]
    assert row["memory_history_source_sample_id"] == "b"
    assert row["memory_history_count"] == 1

    # At 00:11 the completion order is b then a, despite submission order a then b.
    row = features.loc["d"]
    assert row["memory_history_count"] == 2
    assert row["memory_history_last_k_count"] == 2
    assert row["memory_history_last_value"] == pytest.approx(0.2)
    assert row["memory_history_last_k_mean"] == pytest.approx(0.5)
    assert row["memory_history_last_k_std"] == pytest.approx(0.3)
    assert row["memory_history_last_k_median"] == pytest.approx(0.5)
    assert row["memory_history_ema"] == pytest.approx(0.5)
    assert row["memory_history_age_seconds"] == pytest.approx(60.0)
    assert row["memory_history_same_script_available"]
    assert row["memory_history_same_script_source_sample_id"] == "a"
    assert row["memory_history_same_script_last_value"] == pytest.approx(0.2)
    assert row["memory_history_same_script_age_seconds"] == pytest.approx(60.0)

    # Invalid b VRAM does not increment, aggregate, or overwrite a's value.
    assert row["vram_history_count"] == 1
    assert row["vram_history_last_value"] == pytest.approx(0.3)
    assert row["vram_history_source_sample_id"] == "a"

    # Total count and EMA use all history; Last-k statistics retain only k=2.
    row = features.loc["later"]
    assert row["memory_history_count"] == 3
    assert row["memory_history_last_k_count"] == 2
    assert row["memory_history_last_value"] == pytest.approx(0.4)
    assert row["memory_history_last_k_mean"] == pytest.approx(0.3)
    assert row["memory_history_last_k_std"] == pytest.approx(0.1)
    assert row["memory_history_ema"] == pytest.approx(0.45)


def test_history_is_per_user_and_same_script_is_not_global(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["u1-old", "u2-query", "u1-query"],
            "source_row": [0, 1, 2],
            "user_id": ["u1", "u2", "u1"],
            "code": ["same-script", "same-script", "different-script"],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:02Z",
                    "2026-01-01T00:03Z",
                ],
                utc=True,
            ),
            "completion_time": pd.to_datetime(
                [
                    "2026-01-01T00:01Z",
                    "2026-01-01T00:10Z",
                    "2026-01-01T00:11Z",
                ],
                utc=True,
            ),
            "target_memory": [0.7, 0.2, 0.4],
            "valid_memory": [True] * 3,
            "target_vram": [0.6, 0.3, 0.5],
            "valid_vram": [True] * 3,
        }
    )
    features = build_causal_history_features(make_bundle(frame, tmp_path)).set_index(
        "sample_id"
    )

    assert not features.loc["u2-query", "memory_history_available"]
    assert features.loc["u1-query", "memory_history_available"]
    assert not features.loc[
        "u1-query", "memory_history_same_script_available"
    ]


def test_blank_scripts_never_form_same_script_history(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["blank-source", "blank-query"],
            "source_row": [0, 1],
            "user_id": ["u", "u"],
            "code": ["", "   "],
            "submit_time": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T00:02Z"], utc=True
            ),
            "completion_time": pd.to_datetime(
                ["2026-01-01T00:01Z", "2026-01-01T00:03Z"], utc=True
            ),
            "target_memory": [0.7, 0.5],
            "valid_memory": [True, True],
            "target_vram": [0.6, 0.4],
            "valid_vram": [True, True],
        }
    )

    features = build_causal_history_features(make_bundle(frame, tmp_path))
    row = features.set_index("sample_id").loc["blank-query"]

    assert row["memory_history_available"]
    assert not row["memory_history_same_script_available"]
    protocol = features.attrs["causal_history_protocol"]
    assert protocol["empty_script_same_history_policy"] == "unavailable"
    assert protocol["prediction_population_empty_script_rows"] == 2


def test_causal_history_contract_versions_same_script_semantics():
    contract = causal_history_contract(5, 0.5)

    assert contract["protocol_version"] == "causal-history-features/v2"
    assert contract["history_k"] == 5
    assert contract["history_ema_alpha"] == pytest.approx(0.5)
    assert contract["empty_script_same_history_policy"] == "unavailable"


def test_equal_completion_times_use_last1_source_row_tie_break(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["higher-source", "lower-source", "query"],
            "source_row": [20, 10, 30],
            "user_id": ["u"] * 3,
            "code": ["same", "same", "same"],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:03Z",
                ],
                utc=True,
            ),
            "completion_time": pd.to_datetime(
                [
                    "2026-01-01T00:02Z",
                    "2026-01-01T00:02Z",
                    "2026-01-01T00:10Z",
                ],
                utc=True,
            ),
            "target_memory": [0.2, 0.8, 0.4],
            "valid_memory": [True] * 3,
            "target_vram": [0.3, 0.9, 0.5],
            "valid_vram": [True] * 3,
        }
    )
    row = build_causal_history_features(
        make_bundle(frame, tmp_path), k=2
    ).set_index("sample_id").loc["query"]

    # source_row=20 activates after source_row=10 and is the Last-1 value.
    assert row["memory_history_source_sample_id"] == "higher-source"
    assert row["memory_history_last_value"] == pytest.approx(0.2)
    assert row["memory_history_last_k_mean"] == pytest.approx(0.5)
    assert row["memory_history_same_script_source_sample_id"] == "higher-source"


def test_boundary_purged_rows_enter_history_only_after_completion(tmp_path):
    retained = pd.DataFrame(
        {
            "sample_id": ["train-kept", "train-purged", "val-first", "val-later"],
            "source_row": [0, 1, 2, 3],
            "user_id": ["u"] * 4,
            "code": ["kept", "script-p", "other", "script-p"],
            "split": ["train", "train", "validation", "validation"],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:05Z",
                    "2026-01-01T00:10Z",
                    "2026-01-01T00:20Z",
                ],
                utc=True,
            ),
            "completion_time": pd.to_datetime(
                [
                    "2026-01-01T00:30Z",
                    "2026-01-01T00:15Z",
                    "2026-01-01T00:12Z",
                    "2026-01-01T00:25Z",
                ],
                utc=True,
            ),
            "target_memory": [0.2, 0.9, 0.4, 0.6],
            "valid_memory": [True] * 4,
            "target_vram": [0.3, 0.8, 0.5, 0.7],
            "valid_vram": [True] * 4,
        }
    )
    effective = retained.loc[
        retained["sample_id"].ne("train-purged")
    ].reset_index(drop=True)
    wide = build_causal_history_features(
        make_bundle(effective, tmp_path, retained=retained), k=3
    )
    features = wide.set_index("sample_id")

    assert len(features) == len(effective)
    assert not features.loc["val-first", "memory_history_available"]
    row = features.loc["val-later"]
    assert row["memory_history_count"] == 2
    assert row["memory_history_last_value"] == pytest.approx(0.9)
    assert row["memory_history_source_sample_id"] == "train-purged"
    assert row["memory_history_same_script_source_sample_id"] == "train-purged"
    assert row["memory_history_same_script_age_seconds"] == pytest.approx(300.0)
    protocol = wide.attrs["causal_history_protocol"]
    assert protocol["history_event_population_rows"] == 4
    assert protocol["prediction_population_rows"] == 3
    assert protocol["boundary_purged_rows_available_after_completion"] == 1

    long = build_causal_history_long(
        make_bundle(effective, tmp_path, retained=retained), k=3
    )
    assert len(long) == 2 * len(effective)
    assert not long.duplicated(["sample_id", "resource"]).any()
    long_row = long.loc[
        long["sample_id"].eq("val-later") & long["resource"].eq("memory")
    ].iloc[0]
    assert long_row["history_last_value"] == pytest.approx(0.9)
    assert long_row["history_same_script_source_sample_id"] == "train-purged"
    assert long.attrs["history_feature_layout"] == "long"


def test_last_value_and_availability_match_existing_last1_replay(tmp_path):
    last1 = load_last1_module()
    retained = pd.DataFrame(
        {
            "sample_id": ["a", "purged", "b", "c"],
            "source_row": [0, 1, 2, 3],
            "user_id": ["u", "u", "u", "u"],
            "code": ["a", "p", "b", "c"],
            "split": ["train", "train", "validation", "test"],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:01Z",
                    "2026-01-01T00:02Z",
                    "2026-01-01T00:05Z",
                ],
                utc=True,
            ),
            "completion_time": pd.to_datetime(
                [
                    "2026-01-01T00:04Z",
                    "2026-01-01T00:03Z",
                    "2026-01-01T00:05Z",
                    "2026-01-01T00:10Z",
                ],
                utc=True,
            ),
            "target_memory": [0.2, 0.9, 0.4, 0.6],
            "valid_memory": [True] * 4,
            "target_vram": [0.3, 0.8, 0.5, 0.7],
            "valid_vram": [True, False, True, True],
        }
    )
    effective = retained.loc[retained["sample_id"].ne("purged")].reset_index(drop=True)
    bundle = make_bundle(effective, tmp_path, retained=retained)
    features = build_causal_history_features(bundle, k=5)
    replay = last1.replay_completed_before_submission(
        bundle, {"memory": 0.5, "vram": 0.5}
    )

    for resource in ("memory", "vram"):
        available = features[f"{resource}_history_available"].to_numpy(bool)
        assert available.tolist() == replay[resource]["history_available"].tolist()
        assert np.allclose(
            features.loc[available, f"{resource}_history_last_value"].to_numpy(float),
            replay[resource]["prediction"][available],
        )
        assert (
            features.loc[available, f"{resource}_history_source_sample_id"].tolist()
            == replay[resource]["history_source_sample_id"][available].tolist()
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must be a positive integer"),
        ({"ema_alpha": 0.0}, "ema_alpha must be finite"),
        ({"ema_alpha": 1.1}, "ema_alpha must be finite"),
    ],
)
def test_invalid_history_hyperparameters_are_rejected(tmp_path, kwargs, message):
    frame = pd.DataFrame(
        {
            "sample_id": ["a"],
            "source_row": [0],
            "user_id": ["u"],
            "code": ["run"],
            "submit_time": pd.to_datetime(["2026-01-01T00:00Z"], utc=True),
            "completion_time": pd.to_datetime(["2026-01-01T00:01Z"], utc=True),
            "target_memory": [0.2],
            "valid_memory": [True],
            "target_vram": [np.nan],
            "valid_vram": [False],
        }
    )
    with pytest.raises(ValueError, match=message):
        build_causal_history_features(make_bundle(frame, tmp_path), **kwargs)
