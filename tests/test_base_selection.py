from copy import deepcopy
from pathlib import Path
import unittest

import pandas as pd

from common.data import DatasetBundle, split_fingerprint
from common.preprocessing import NumericScaler, UserVocabulary
from qwen.base_selection import (
    base_preprocessing_bundle,
    base_selection_settings,
    select_base_training_frames,
    verify_base_training_selection,
)
from qwen.contract import experiment_contract, validate_and_restore_contract


def bundle_fixture():
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame({
        "sample_id": [f"job_{i}" for i in range(14)],
        "split": ["train"] * 10 + ["validation"] * 2 + ["test"] * 2,
        "submit_time": [origin + pd.Timedelta(hours=i) for i in list(range(10)) + [20, 21, 30, 31]],
        "user_id": ["fit_user"] * 9 + ["tail_user", "val_user", "val_user", "test_user", "test_user"],
        "cpu_request": list(range(10)) + [1_000_000] * 4,
        "valid_memory": [True] * 14,
        "valid_vram": [True] * 14,
        "target_memory": [0.2] * 14,
        "target_vram": [0.3] * 14,
    })
    frame["completion_time"] = frame["submit_time"] + pd.Timedelta(minutes=30)
    # Equal completion/submission must be purged under the strict relation.
    frame.loc[8, "completion_time"] = origin + pd.Timedelta(hours=9)
    return DatasetBundle(frame, Path("private.csv"), "a" * 64, None,
                         ["cpu_request"], [], {"verified_chronological": True},
                         retained_frame=frame.copy())


def config(mode="train_tail", fraction=0.9):
    return {"seed": 42, "qwen": {"base_selection": {
        "mode": mode, "fit_fraction": fraction,
    }}}


def test_train_tail_keeps_outer_split_and_purges_equal_completion():
    bundle = bundle_fixture()
    original = bundle.frame.copy(deep=True)
    selection = select_base_training_frames(bundle, config())
    assert selection.optimization["sample_id"].tolist() == [f"job_{i}" for i in range(8)]
    assert selection.early_stopping["sample_id"].tolist() == ["job_9"]
    assert selection.manifest["purged_optimization_sample_ids"] == ["job_8"]
    assert selection.manifest["base_checkpoint_validation_overlap"] is False
    assert selection.manifest["source_split_fingerprint"] == split_fingerprint(original)
    pd.testing.assert_frame_equal(bundle.frame, original)
    pd.testing.assert_frame_equal(bundle.retained_frame, original)
    assert verify_base_training_selection(bundle, config(), selection.manifest) == selection.manifest


def test_tied_submissions_are_never_split():
    bundle = bundle_fixture()
    bundle.frame.loc[8, "submit_time"] = bundle.frame.loc[9, "submit_time"]
    selection = select_base_training_frames(bundle, config())
    assert selection.early_stopping["sample_id"].tolist() == ["job_8", "job_9"]
    assert not (set(selection.optimization.submit_time) & set(selection.early_stopping.submit_time))


def test_matched_control_has_identical_optimization_and_preprocessing():
    bundle = bundle_fixture()
    strict = select_base_training_frames(bundle, config())
    control = select_base_training_frames(bundle, config("shared_validation_control"))
    pd.testing.assert_frame_equal(strict.optimization, control.optimization)
    assert control.early_stopping["sample_id"].tolist() == ["job_10", "job_11"]
    assert control.manifest["base_checkpoint_validation_overlap"] is True
    assert control.manifest["reserved_train_tail_sample_ids"] == ["job_9"]
    assert set(control.manifest["reserved_train_tail_sample_ids"]).isdisjoint(
        control.manifest["optimization_sample_ids"]
    )
    for selected in (strict, control):
        scoped = base_preprocessing_bundle(bundle, selected)
        train = scoped.split("train")
        scaler = NumericScaler.fit(train, bundle.numeric_features)
        vocabulary = UserVocabulary.fit(train["user_id"])
        assert scaler.means == [3.5]
        assert "tail_user" not in vocabulary.user_to_id
        assert "val_user" not in vocabulary.user_to_id
        assert "test_user" not in vocabulary.user_to_id


def test_original_validation_test_labels_do_not_change_strict_selection():
    bundle = bundle_fixture()
    original = select_base_training_frames(bundle, config())
    for column in ("target_memory", "target_vram"):
        bundle.frame.loc[bundle.frame["split"] != "train", column] = 0.999
    for column in ("valid_memory", "valid_vram"):
        bundle.frame.loc[bundle.frame["split"] != "train", column] = False
    changed = select_base_training_frames(bundle, config())
    assert changed.manifest == original.manifest
    pd.testing.assert_frame_equal(changed.optimization, original.optimization)
    pd.testing.assert_frame_equal(changed.early_stopping, original.early_stopping)


def test_missing_vram_in_internal_stop_fails_closed():
    bundle = bundle_fixture()
    bundle.frame.loc[9, "valid_vram"] = False
    with unittest.TestCase().assertRaisesRegex(ValueError, "early-stopping.*vram"):
        select_base_training_frames(bundle, config())


def test_missing_times_and_outer_completion_crossing_fail_closed():
    bundle = bundle_fixture()
    bundle.frame.loc[0, "submit_time"] = pd.NaT
    with unittest.TestCase().assertRaisesRegex(ValueError, "missing submit_time"):
        select_base_training_frames(bundle, config())
    bundle = bundle_fixture()
    bundle.frame.loc[9, "completion_time"] = bundle.frame.loc[10, "submit_time"]
    with unittest.TestCase().assertRaisesRegex(ValueError, "finish before outer validation"):
        select_base_training_frames(bundle, config())


def test_manifest_tampering_and_changed_selection_are_rejected():
    bundle = bundle_fixture()
    selected = select_base_training_frames(bundle, config())
    record = deepcopy(selected.manifest)
    record["optimization_sample_ids"].append("job_13")
    with unittest.TestCase().assertRaisesRegex(ValueError, "fingerprint"):
        verify_base_training_selection(bundle, config(), record)
    with unittest.TestCase().assertRaisesRegex(ValueError, "does not match"):
        verify_base_training_selection(bundle, config(fraction=0.8), selected.manifest)


def test_legacy_contract_and_split_behavior_are_unchanged():
    bundle = bundle_fixture()
    selected = select_base_training_frames(bundle, {})
    pd.testing.assert_frame_equal(selected.optimization, bundle.split("train"))
    pd.testing.assert_frame_equal(selected.early_stopping, bundle.split("validation"))
    assert selected.manifest is None
    implicit = experiment_contract({}, {})
    explicit = experiment_contract(config("legacy_validation"), {})
    assert implicit == explicit
    assert "base_selection" not in implicit["payload"]["qwen_semantics"]


def test_new_selection_configuration_is_contract_protected():
    settings = config()
    contract = experiment_contract(settings, {})
    manifest = {
        "seed": 42, "variant": "full", "enabled_modalities": ["code", "user", "metadata"],
        "fusion_mode": "gated", "effective_parameters": {}, "experiment_contract": contract,
    }
    assert validate_and_restore_contract(deepcopy(settings), manifest) == contract
    for changed in (config(fraction=0.8), config("shared_validation_control"), {}):
        with unittest.TestCase().assertRaisesRegex(ValueError, "does not match"):
            validate_and_restore_contract(changed, manifest)


def test_invalid_configuration_fails_closed():
    for settings in [
        {"mode": "unknown"}, {"mode": "train_tail", "fit_fraction": 1.0},
        {"mode": "train_tail", "fit_fraction": float("nan")},
        {"mode": "train_tail", "typo": 0.9},
    ]:
        with unittest.TestCase().assertRaises(ValueError):
            base_selection_settings({"qwen": {"base_selection": settings}})


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
