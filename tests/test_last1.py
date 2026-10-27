import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import DatasetBundle


def load_last1_module():
    path = Path(__file__).resolve().parents[1] / "last-1" / "run.py"
    spec = importlib.util.spec_from_file_location("last1_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_causal_replay_uses_only_completed_history(tmp_path):
    module = load_last1_module()
    submitted = pd.to_datetime(
        ["2026-01-01T00:00Z", "2026-01-01T00:05Z", "2026-01-01T00:07Z", "2026-01-01T00:11Z"],
        utc=True,
    )
    completed = pd.to_datetime(
        ["2026-01-01T00:10Z", "2026-01-01T00:06Z", "2026-01-01T00:20Z", "2026-01-01T00:30Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "source_row": [0, 1, 2, 3],
            "user_id": ["u"] * 4,
            "submit_time": submitted,
            "completion_time": completed,
            "target_memory": [0.2, 0.8, 0.4, 0.6],
            "valid_memory": [True] * 4,
            "target_vram": [np.nan] * 4,
            "valid_vram": [False] * 4,
        }
    )
    bundle = DatasetBundle(
        frame=frame,
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
    replay = module.replay_completed_before_submission(
        bundle, {"memory": 0.5, "vram": 0.5}
    )
    assert np.allclose(replay["memory"]["prediction"], [0.5, 0.5, 0.8, 0.2])
    assert replay["memory"]["history_available"].tolist() == [False, False, True, True]


def test_boundary_purged_job_is_history_but_not_prediction_row(tmp_path):
    module = load_last1_module()
    retained = pd.DataFrame(
        {
            "sample_id": ["train-kept", "train-purged", "val-first", "val-later"],
            "source_row": [0, 1, 2, 3],
            "user_id": ["u"] * 4,
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
    bundle = DatasetBundle(
        frame=effective,
        retained_frame=retained,
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

    fallback = module.training_medians(bundle.frame)
    replay = module.replay_completed_before_submission(bundle, fallback)

    # The purged training label cannot change the fallback fitted on effective train.
    assert fallback == {"memory": 0.2, "vram": 0.3}
    assert len(replay["memory"]["prediction"]) == len(effective)
    assert np.allclose(replay["memory"]["prediction"], [0.2, 0.2, 0.9])
    assert replay["memory"]["history_source_sample_id"].tolist() == [
        "",
        "",
        "train-purged",
    ]
