from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from common.data import DatasetBundle
from common.neural import (
    checkpoint_identity,
    split_fingerprint,
    validate_checkpoint_identity,
    validate_training_label_coverage,
)


def make_bundle(frame: pd.DataFrame, data_sha256: str = "a" * 64) -> DatasetBundle:
    return DatasetBundle(
        frame=frame,
        data_path=Path("trace.csv"),
        data_sha256=data_sha256,
        manifest_path=Path("split.csv"),
        numeric_features=["num__cpu"],
        categorical_features=[],
        protocol={},
    )


def complete_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "validation", "validation"],
            "valid_memory": [True, True, True, True],
            "valid_vram": [True, True, True, True],
        }
    )


def test_split_fingerprint_changes_with_assignment():
    original = complete_frame()
    changed = original.copy()
    changed.loc[1, "split"] = "validation"

    assert split_fingerprint(original) != split_fingerprint(changed)


def test_checkpoint_identity_accepts_exact_match():
    expected = checkpoint_identity(
        make_bundle(complete_frame()), seed=42, max_tokens=512
    )
    validate_checkpoint_identity(deepcopy(expected), expected)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("data_sha256", "b" * 64),
        ("standardized_input_fingerprint", "d" * 64),
        ("split_fingerprint", "c" * 64),
        ("seed", 7),
        ("max_tokens", 1024),
        ("numeric_features", ["num__memory"]),
    ],
)
def test_checkpoint_identity_rejects_protocol_mismatch(field, replacement):
    expected = checkpoint_identity(
        make_bundle(complete_frame()), seed=42, max_tokens=512
    )
    stored = deepcopy(expected)
    stored[field] = replacement

    with pytest.raises(ValueError, match=field):
        validate_checkpoint_identity(stored, expected)


def test_checkpoint_identity_rejects_legacy_checkpoint():
    expected = checkpoint_identity(
        make_bundle(complete_frame()), seed=42, max_tokens=512
    )

    with pytest.raises(ValueError, match="no provenance identity"):
        validate_checkpoint_identity(None, expected)


def test_label_coverage_accepts_both_resources_in_train_and_validation():
    counts = validate_training_label_coverage(make_bundle(complete_frame()))

    assert counts == {
        "train_memory": 2,
        "train_vram": 2,
        "validation_memory": 2,
        "validation_vram": 2,
    }


@pytest.mark.parametrize(
    ("split", "valid_column", "resource"),
    [
        ("train", "valid_memory", "memory"),
        ("train", "valid_vram", "vram"),
        ("validation", "valid_memory", "memory"),
        ("validation", "valid_vram", "vram"),
    ],
)
def test_label_coverage_rejects_missing_resource(split, valid_column, resource):
    frame = complete_frame()
    frame.loc[frame["split"].eq(split), valid_column] = False

    with pytest.raises(
        ValueError,
        match=rf"{split.capitalize()} split has no valid {resource} targets",
    ):
        validate_training_label_coverage(make_bundle(frame))
