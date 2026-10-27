import csv
import math
from pathlib import Path

import pandas as pd
import pytest

from common.data import load_dataset, parse_slurm_minutes, sha256_file


def make_config(csv_path: Path, manifest_path: Path):
    return {
        "data": {
            "path": str(csv_path),
            "label_scale": "ratio",
            "columns": {
                "job_id": "jobid",
                "user": "user",
                "code": "script",
                "gpu_count": "gpu",
                "memory_peak": "mem_peak",
                "vram_peak": "vram_peak",
                "order": "submitted",
                "submit_time": "submitted",
                "completion_time": "completed",
                "time_limit": "walltime",
            },
            "metadata_numeric": ["cpu", "gpu", "memory_request"],
            "metadata_categorical": [],
        },
        "split": {
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
            "order_type": "datetime",
            "order_semantics": "submission_time",
            "manifest": str(manifest_path),
            "create_if_missing": True,
        },
    }


def write_fixture(path: Path, count: int = 10):
    rows = []
    for index in range(count):
        rows.append(
            {
                "jobid": index,
                "user": f"u{index % 2}",
                "script": "python train.py",
                "cpu": 4,
                "gpu": 1 if index % 2 else 0,
                "memory_request": 16,
                "allocated_vram": 80 if index % 2 else 0,
                "walltime": "01:00:00",
                "mem_peak": index / max(count, 10),
                "vram_peak": index / max(count, 10) if index % 2 else None,
                "submitted": f"2026-01-{index + 1:02d}T00:00:00Z",
                "completed": f"2026-01-{index + 1:02d}T01:00:00Z",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_split_manifest_and_vram_mask(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    bundle = load_dataset(make_config(csv_path, manifest_path))
    assert bundle.protocol["split_counts"] == {"train": 8, "validation": 1, "test": 1}
    assert bundle.protocol["verified_chronological"] is True
    assert bundle.frame["valid_memory"].sum() == 10
    assert bundle.frame["valid_vram"].sum() == 5
    assert manifest_path.exists()
    second = load_dataset(make_config(csv_path, manifest_path))
    assert second.frame["split"].tolist() == bundle.frame["split"].tolist()


def test_wrong_explicit_label_scale_fails(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[0, "mem_peak"] = 50.0
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="outside"):
        load_dataset(make_config(csv_path, manifest_path))


def test_out_of_range_mask_is_explicit_and_audited(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[0, "mem_peak"] = 1.1
    frame.to_csv(csv_path, index=False)
    config = make_config(csv_path, manifest_path)
    config["data"]["label_out_of_range"] = "mask"
    bundle = load_dataset(config)
    assert bundle.protocol["out_of_range_memory_masked"] == 1
    assert bundle.protocol["valid_memory"] == 9


def test_paper_metadata_feature_formulas(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["data"].pop("metadata_numeric")
    config["data"]["metadata_standardization"] = "none"
    config["data"]["metadata_features"] = [
        {"name": "cpu", "column": "cpu", "transform": "log_max", "maximum": 256},
        {
            "name": "walltime",
            "column": "walltime",
            "parser": "slurm_minutes",
            "transform": "log_max",
            "maximum": 21600,
        },
        {
            "name": "memory_per_core",
            "numerator": "memory_request",
            "denominator": "cpu",
            "transform": "ratio",
            "denominator_floor": 1,
        },
        {
            "name": "vram_per_gpu",
            "numerator": "allocated_vram",
            "denominator": "gpu",
            "transform": "ratio",
            "denominator_floor": 1,
            "zero_if_denominator_nonpositive": True,
        },
    ]
    bundle = load_dataset(config)
    gpu_row = bundle.frame.loc[bundle.frame["job_id"] == "1"].iloc[0]
    cpu_row = bundle.frame.loc[bundle.frame["job_id"] == "0"].iloc[0]
    assert gpu_row["num__cpu"] == pytest.approx(math.log1p(4) / math.log1p(256))
    assert gpu_row["num__walltime"] == pytest.approx(math.log1p(60) / math.log1p(21600))
    assert gpu_row["num__memory_per_core"] == pytest.approx(4.0)
    assert gpu_row["num__vram_per_gpu"] == pytest.approx(80.0)
    assert cpu_row["num__vram_per_gpu"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("raw", "minutes"),
    [
        ("30", 30.0),
        ("30:00", 30.0),
        ("01:30:00", 90.0),
        ("2-03", 3060.0),
        ("2-03:30", 3090.0),
        ("2-03:30:15", 3090.25),
        ("41-15:00:+", 59940.0),
        ("10-00:00:+", 14400.0),
        ("416-15:00+", 599940.0),
        ("20-00:00:+", 28800.0),
    ],
)
def test_slurm_duration_formats(raw, minutes):
    assert parse_slurm_minutes(raw) == pytest.approx(minutes)


def test_slurm_unlimited_requires_an_explicit_finite_replacement():
    assert math.isnan(parse_slurm_minutes("UNLIMITED"))
    assert parse_slurm_minutes(
        "UNLIMITED", unlimited_minutes=599940
    ) == pytest.approx(599940.0)
    with pytest.raises(ValueError, match="unlimited_minutes"):
        parse_slurm_minutes("UNLIMITED", unlimited_minutes=float("inf"))


def test_slurm_truncation_can_be_rejected_explicitly():
    assert math.isnan(parse_slurm_minutes("41-15:00:+", allow_truncated=False))


def test_manifest_rejects_interleaved_assignment(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    load_dataset(make_config(csv_path, manifest_path))
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "split"] = "test"
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="contiguous"):
        load_dataset(make_config(csv_path, manifest_path))


def test_manifest_rejects_assignment_fingerprint_tampering(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    load_dataset(make_config(csv_path, manifest_path))
    manifest = pd.read_csv(manifest_path)
    manifest["split_fingerprint"] = "0" * 64
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="assignment fingerprint"):
        load_dataset(make_config(csv_path, manifest_path))


@pytest.mark.parametrize("column", ["dataset_sha256", "order_fingerprint", "train_ratio"])
def test_manifest_rejects_missing_provenance_cells(tmp_path, column):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    load_dataset(make_config(csv_path, manifest_path))
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, column] = None
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match=rf"column {column} must be non-null"):
        load_dataset(make_config(csv_path, manifest_path))


@pytest.mark.parametrize(
    ("config_path", "replacement", "message"),
    [
        ("order_semantics", "completion_time", "order_semantics mismatch"),
        ("train_ratio", 0.7, "train_ratio mismatch"),
    ],
)
def test_manifest_rejects_split_setting_changes(
    tmp_path, config_path, replacement, message
):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    load_dataset(make_config(csv_path, manifest_path))
    changed = make_config(csv_path, manifest_path)
    changed["split"][config_path] = replacement
    with pytest.raises(ValueError, match=message):
        load_dataset(changed)


def test_protocol_records_split_provenance(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    bundle = load_dataset(make_config(csv_path, manifest_path))
    assert len(bundle.protocol["order_fingerprint"]) == 64
    assert len(bundle.protocol["split_fingerprint"]) == 64
    assert len(bundle.protocol["split_manifest_sha256"]) == 64
    assert {
        "order_column": "submitted",
        "order_type": "datetime",
        "order_semantics": "submission_time",
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
    }.items() <= bundle.protocol["split_settings"].items()


def test_expected_counts_pass_and_fail_closed(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["data"].update(
        {
            "expected_raw_row_count": 10,
            "expected_retained_row_count": 10,
            "expected_row_count": 10,
            "expected_valid_memory": 10,
            "expected_valid_vram": 5,
        }
    )
    config["split"]["expected_counts"] = {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    config["data"]["expected_valid_by_split"] = {
        "memory": {"train": 8, "validation": 1, "test": 1},
        "vram": {"train": 4, "validation": 0, "test": 1},
    }
    load_dataset(config)

    config["data"]["expected_valid_vram"] = 6
    with pytest.raises(ValueError, match="expected_valid_vram mismatch"):
        load_dataset(config)


def test_expected_data_sha256_passes_and_fails_closed(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["data"]["expected_sha256"] = sha256_file(csv_path)
    load_dataset(config)

    config["data"]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data.expected_sha256 mismatch"):
        load_dataset(config)


def test_expected_valid_by_split_mismatch_fails_closed(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["data"]["expected_valid_by_split"] = {
        "memory": {"train": 8, "validation": 1, "test": 1},
        "vram": {"train": 4, "validation": 0, "test": 0},
    }
    with pytest.raises(
        ValueError,
        match=r"data.expected_valid_by_split.vram.test mismatch",
    ):
        load_dataset(config)


def test_expected_split_fingerprint_passes_and_fails_closed(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    first = load_dataset(config)

    config["split"]["expected_fingerprint"] = first.protocol["split_fingerprint"]
    load_dataset(config)

    config["split"]["expected_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="split.expected_fingerprint mismatch"):
        load_dataset(config)


@pytest.mark.parametrize(
    ("config_key", "protocol_key"),
    [
        ("expected_order_fingerprint", "order_fingerprint"),
        ("expected_manifest_sha256", "split_manifest_sha256"),
    ],
)
def test_expected_split_identity_hashes_pass_and_fail_closed(
    tmp_path, config_key, protocol_key
):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    first = load_dataset(config)

    config["split"][config_key] = first.protocol[protocol_key]
    load_dataset(config)

    config["split"][config_key] = "0" * 64
    with pytest.raises(ValueError, match=rf"split\.{config_key} mismatch"):
        load_dataset(config)


def test_expected_split_count_mismatch_fails(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["split"]["expected_counts"] = {
        "train": 7,
        "validation": 1,
        "test": 2,
    }
    with pytest.raises(ValueError, match=r"expected_counts\.train mismatch"):
        load_dataset(config)


def test_completion_before_submission_fails(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[0, "completed"] = "2025-12-31T23:59:59Z"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="completion_time must be greater"):
        load_dataset(make_config(csv_path, manifest_path))


def test_completed_label_boundaries_are_checked_and_recorded(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["split"]["require_completed_labels_at_boundaries"] = True
    bundle = load_dataset(config)
    boundary = bundle.protocol["completed_labels_at_boundaries"]
    assert boundary["required"] is True
    assert boundary["policy"] == "error"
    assert boundary["protocol_version"] == "completion-before-next-submit/v1"
    assert boundary["checked"] is True
    assert boundary["train_cross_boundary_count"] == 0
    assert boundary["validation_cross_boundary_count"] == 0
    assert boundary["pre_purge_row_count"] == 10
    assert boundary["post_purge_row_count"] == 10
    assert boundary["purged_row_count"] == 0
    assert boundary["purged_sample_ids"] == []
    assert boundary["remaining_train_cross_boundary_count"] == 0
    assert boundary["remaining_validation_cross_boundary_count"] == 0


def test_completed_label_boundary_crossing_fails(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[7, "completed"] = "2026-01-09T00:00:00Z"
    frame.to_csv(csv_path, index=False)
    config = make_config(csv_path, manifest_path)
    config["split"]["require_completed_labels_at_boundaries"] = True
    with pytest.raises(
        ValueError, match="train_jobs_not_completed_before_validation=1"
    ):
        load_dataset(config)


def test_boundary_purge_requires_completed_label_check(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["split"]["boundary_completion_policy"] = "purge"

    with pytest.raises(ValueError, match="require_completed_labels_at_boundaries"):
        load_dataset(config)


def test_invalid_boundary_completion_policy_fails_closed(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["split"]["require_completed_labels_at_boundaries"] = True
    config["split"]["boundary_completion_policy"] = "drop-and-resplit"

    with pytest.raises(ValueError, match="boundary_completion_policy"):
        load_dataset(config)


def make_boundary_purge_config(csv_path: Path, manifest_path: Path):
    config = make_config(csv_path, manifest_path)
    config["split"].update(
        {
            "require_completed_labels_at_boundaries": True,
            "boundary_completion_policy": "purge",
        }
    )
    return config


def write_boundary_crossing_fixture(path: Path):
    write_fixture(path, count=20)
    frame = pd.read_csv(path)
    # Base split: train=0..15, validation=16..17, test=18..19.
    frame.loc[15, "completed"] = "2026-01-17T00:00:00Z"
    frame.loc[17, "completed"] = "2026-01-19T00:00:00Z"
    frame.to_csv(path, index=False)


def test_boundary_purge_uses_fixed_base_split_without_redistribution(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "purged-split.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, manifest_path)

    bundle = load_dataset(config)
    boundary = bundle.protocol["completed_labels_at_boundaries"]

    assert bundle.retained_frame is not None
    assert len(bundle.retained_frame) == 20
    assert bundle.retained_frame["sample_id"].tolist() == [
        f"{index}::{index}" for index in range(20)
    ]
    assert bundle.protocol["retained_trace_row_count"] == 20
    assert bundle.protocol["split_counts"] == {
        "train": 15,
        "validation": 1,
        "test": 2,
    }
    assert bundle.frame.loc[bundle.frame["split"] == "train", "job_id"].tolist() == [
        str(index) for index in range(15)
    ]
    assert bundle.frame.loc[
        bundle.frame["split"] == "validation", "job_id"
    ].tolist() == ["16"]
    assert bundle.frame.loc[bundle.frame["split"] == "test", "job_id"].tolist() == [
        "18",
        "19",
    ]
    assert set(bundle.frame["sample_id"]) == {
        f"{index}::{index}" for index in range(20) if index not in {15, 17}
    }

    assert boundary["required"] is True
    assert boundary["policy"] == "purge"
    assert boundary["protocol_version"] == "completion-before-next-submit/v1"
    assert boundary["checked"] is True
    assert boundary["train_cross_boundary_count"] == 1
    assert boundary["validation_cross_boundary_count"] == 1
    assert pd.Timestamp(boundary["first_validation_submit_time"]) == pd.Timestamp(
        "2026-01-17T00:00:00Z"
    )
    assert pd.Timestamp(boundary["first_test_submit_time"]) == pd.Timestamp(
        "2026-01-19T00:00:00Z"
    )
    assert boundary["pre_purge_row_count"] == 20
    assert boundary["post_purge_row_count"] == 18
    assert boundary["purged_row_count"] == 2
    assert boundary["purged_train_count"] == 1
    assert boundary["purged_validation_count"] == 1
    assert boundary["purged_sample_ids"] == ["15::15", "17::17"]
    assert boundary["remaining_train_cross_boundary_count"] == 0
    assert boundary["remaining_validation_cross_boundary_count"] == 0
    for key in (
        "base_order_fingerprint",
        "base_split_fingerprint",
        "effective_order_fingerprint",
        "effective_split_fingerprint",
        "boundary_purge_fingerprint",
    ):
        assert len(boundary[key]) == 64
    assert boundary["base_order_fingerprint"] != boundary["effective_order_fingerprint"]
    assert boundary["base_split_fingerprint"] != boundary["effective_split_fingerprint"]
    assert (
        boundary["effective_order_fingerprint"]
        == bundle.protocol["order_fingerprint"]
    )
    assert (
        boundary["effective_split_fingerprint"]
        == bundle.protocol["split_fingerprint"]
    )

    manifest = pd.read_csv(
        manifest_path, dtype={"sample_id": str, "job_id": str}
    )
    assert len(manifest) == 20
    assert (
        manifest.loc[manifest["sample_id"] == "15::15", "boundary_action"].item()
        == "purge_train_completion"
    )
    assert (
        manifest.loc[manifest["sample_id"] == "17::17", "boundary_action"].item()
        == "purge_validation_completion"
    )
    assert set(manifest["boundary_action"]) == {
        "keep",
        "purge_train_completion",
        "purge_validation_completion",
    }
    assert manifest.loc[manifest["sample_id"] == "15::15", "split"].item() == "train"
    assert (
        manifest.loc[manifest["sample_id"] == "17::17", "split"].item()
        == "validation"
    )
    assert manifest["boundary_completion_policy"].unique().tolist() == ["purge"]
    assert manifest["boundary_protocol_version"].unique().tolist() == [
        "completion-before-next-submit/v1"
    ]
    assert manifest["boundary_purge_fingerprint"].unique().tolist() == [
        boundary["boundary_purge_fingerprint"]
    ]

    reloaded = load_dataset(config)
    assert reloaded.frame["sample_id"].tolist() == bundle.frame["sample_id"].tolist()
    assert reloaded.frame["split"].tolist() == bundle.frame["split"].tolist()
    assert (
        reloaded.protocol["completed_labels_at_boundaries"]
        == bundle.protocol["completed_labels_at_boundaries"]
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        "boundary_action",
        "boundary_completion_policy",
        "boundary_protocol_version",
        "boundary_purge_fingerprint",
    ],
)
def test_purge_manifest_missing_boundary_provenance_fails_closed(
    tmp_path, missing_column
):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "purged-split.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, manifest_path)
    load_dataset(config)

    manifest = pd.read_csv(manifest_path)
    manifest.drop(columns=[missing_column]).to_csv(manifest_path, index=False)

    with pytest.raises(
        ValueError, match="Purge split manifest is missing boundary provenance columns"
    ):
        load_dataset(config)


def test_boundary_purge_requires_an_auditable_manifest(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, tmp_path / "unused.csv")
    config["split"]["manifest"] = None

    with pytest.raises(ValueError, match="requires split.manifest"):
        load_dataset(config)


def test_boundary_purge_requires_manifest_creation_to_be_enabled(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "missing-purged-split.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, manifest_path)
    config["split"]["create_if_missing"] = False

    with pytest.raises(ValueError, match="create_if_missing is false"):
        load_dataset(config)


@pytest.mark.parametrize("column", ["job_id", "source_row", "order_value"])
def test_purge_manifest_rejects_tampered_audit_columns(tmp_path, column):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "purged-split.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, manifest_path)
    load_dataset(config)

    manifest = pd.read_csv(manifest_path)
    # This fixture intentionally corrupts numeric audit columns with text.
    # Explicit object dtype keeps that setup valid under pandas 3 as well.
    manifest[column] = manifest[column].astype(object)
    manifest.loc[0, column] = "tampered"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match=rf"{column} audit column is invalid"):
        load_dataset(config)


def test_purge_manifest_rejects_tampered_boundary_action(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "purged-split.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, manifest_path)
    load_dataset(config)

    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "boundary_action"] = "purge_train_completion"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="boundary_action values do not match"):
        load_dataset(config)


def test_purge_manifest_rejects_tampered_boundary_fingerprint(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "purged-split.csv"
    write_boundary_crossing_fixture(csv_path)
    config = make_boundary_purge_config(csv_path, manifest_path)
    load_dataset(config)

    manifest = pd.read_csv(manifest_path)
    manifest["boundary_purge_fingerprint"] = "0" * 64
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="boundary purge fingerprint is invalid"):
        load_dataset(config)


@pytest.mark.parametrize("leaked_source", ["mem_peak", "vram_peak", "completed"])
def test_direct_metadata_label_leakage_is_rejected(tmp_path, leaked_source):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["data"]["metadata_numeric"].append(leaked_source)
    with pytest.raises(ValueError, match="cannot reference outcome"):
        load_dataset(config)


def test_ratio_metadata_label_leakage_is_rejected(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    config = make_config(csv_path, manifest_path)
    config["data"].pop("metadata_numeric")
    config["data"]["metadata_features"] = [
        {
            "name": "leaked_ratio",
            "transform": "ratio",
            "numerator": "mem_peak",
            "denominator": "cpu",
        }
    ]
    with pytest.raises(ValueError, match="cannot reference outcome"):
        load_dataset(config)


def test_explicit_validity_flags_are_combined_with_eligibility(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame["mem_valid"] = True
    frame.loc[1, "mem_valid"] = False
    frame["gpu_valid"] = 1
    frame.loc[1, "gpu_valid"] = 0
    frame.to_csv(csv_path, index=False)
    config = make_config(csv_path, manifest_path)
    config["data"]["columns"]["memory_valid"] = "mem_valid"
    config["data"]["columns"]["vram_valid"] = "gpu_valid"
    bundle = load_dataset(config)
    assert bundle.protocol["valid_memory"] == 9
    assert bundle.protocol["valid_vram"] == 4
    assert bundle.protocol["explicit_validity_flags"] == {
        "memory": True,
        "vram": True,
        "memory_column": "mem_valid",
        "vram_column": "gpu_valid",
    }


def test_invalid_explicit_validity_value_is_rejected(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame["mem_valid"] = "true"
    frame.loc[0, "mem_valid"] = "unknown"
    frame.to_csv(csv_path, index=False)
    config = make_config(csv_path, manifest_path)
    config["data"]["columns"]["memory_valid"] = "mem_valid"
    with pytest.raises(ValueError, match="Validity column mem_valid"):
        load_dataset(config)


def test_finite_zero_labels_remain_valid_without_explicit_flags(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    manifest_path = tmp_path / "split.csv"
    write_fixture(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[1, "vram_peak"] = 0.0
    frame.to_csv(csv_path, index=False)
    bundle = load_dataset(make_config(csv_path, manifest_path))
    zero_memory = bundle.frame.loc[bundle.frame["job_id"] == "0"].iloc[0]
    zero_vram = bundle.frame.loc[bundle.frame["job_id"] == "1"].iloc[0]
    assert bool(zero_memory["valid_memory"]) is True
    assert bool(zero_vram["valid_vram"]) is True


TRACE_COLUMNS = [
    "jobid",
    "jobname",
    "user",
    "partition",
    "submit",
    "start1",
    "end1",
    "state1",
    "elapsed",
    "exitcode",
    "cpu",
    "gpu",
    "mem",
    "time",
    "nodelist",
    "cpu_eff",
    "mem_eff",
    "mem_max",
    "mem_p95",
    "mem_capacity_max",
    "gpu_util",
    "vram_eff",
    "vram_max",
    "vram_p95",
    "vram_capacity_max",
    "script_content",
    "workdir",
]

TRACE_FORBIDDEN_METADATA = [
    "start1",
    "end1",
    "state1",
    "elapsed",
    "exitcode",
    "nodelist",
    "cpu_eff",
    "mem_eff",
    "mem_max",
    "mem_p95",
    "mem_capacity_max",
    "gpu_util",
    "vram_eff",
    "vram_max",
    "vram_p95",
    "vram_capacity_max",
]


def make_trace_row(index, **overrides):
    submit = pd.Timestamp("2026-02-01T00:00:00Z") + pd.Timedelta(days=index)
    start = submit + pd.Timedelta(minutes=10)
    completion = start + pd.Timedelta(minutes=30)
    row = {
        "jobid": str(1000 + index),
        "jobname": f"job-{index}",
        "user": f"user-{index % 3}",
        "partition": "gpu",
        "submit": submit.isoformat(),
        "start1": start.isoformat(),
        "end1": completion.isoformat(),
        "state1": "COMPLETED",
        "elapsed": "00:30:00",
        "exitcode": "0:0",
        "cpu": 8,
        "gpu": 1,
        "mem": 64,
        "time": "02:00:00",
        "nodelist": f"node-{index % 2}",
        "cpu_eff": 0.50,
        "mem_eff": 0.40,
        "mem_max": 0.50 + index / 100.0,
        "mem_p95": 0.45,
        "mem_capacity_max": 32.0 + index,
        "gpu_util": 0.60,
        "vram_eff": 0.55,
        "vram_max": 0.60 + index / 100.0,
        "vram_p95": 0.58,
        "vram_capacity_max": 24.0,
        "script_content": "#!/bin/bash\npython train.py",
        "workdir": f"/gpfsnyu/home/user-{index % 3}/project",
    }
    row.update(overrides)
    return row


def write_trace_fixture(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_trace_config(csv_path: Path):
    return {
        "data": {
            "path": str(csv_path),
            "label_scale": "ratio",
            "label_out_of_range": "error",
            "timestamp_timezone": "UTC",
            "columns": {
                "job_id": "jobid",
                "job_name": "jobname",
                "user": "user",
                "partition": "partition",
                "code": "script_content",
                "gpu_count": "gpu",
                "memory_peak": "mem_max",
                "vram_peak": "vram_max",
                "memory_peak_absolute": "mem_capacity_max",
                "vram_peak_absolute": "vram_capacity_max",
                "order": "submit",
                "submit_time": "submit",
                "start_time": "start1",
                "completion_time": "end1",
                "state": "state1",
                "exit_code": "exitcode",
                "elapsed": "elapsed",
                "node_list": "nodelist",
                "time_limit": "time",
                "workdir": "workdir",
            },
            "filters": {
                "states": ["COMPLETED"],
                "exit_codes": ["0:0"],
            },
            "time_validation": {
                "check_elapsed": True,
                "tolerance_seconds": 1,
            },
            "forbidden_metadata_columns": TRACE_FORBIDDEN_METADATA.copy(),
            "code_redaction": {
                "enabled": False,
                "path_prefixes": ["/gpfsnyu/home", "/gpfsnyu/scratch"],
            },
            "metadata_numeric": ["cpu", "gpu", "mem"],
            "metadata_categorical": [],
            "require_complete_metadata": True,
        },
        "split": {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "order_type": "datetime",
            "order_semantics": "submission_time",
            "require_completed_labels_at_boundaries": False,
        },
    }


def test_complete_27_column_trace_contract_loads(tmp_path):
    csv_path = tmp_path / "trace.csv"
    write_trace_fixture(csv_path, [make_trace_row(index) for index in range(10)])
    config = make_trace_config(csv_path)
    config["data"]["expected_columns"] = TRACE_COLUMNS.copy()
    bundle = load_dataset(config)
    assert len(bundle.frame) == 10
    assert bundle.protocol["raw_row_count"] == 10
    assert bundle.protocol["retained_row_count"] == 10
    assert bundle.protocol["dropped_row_count"] == 0
    assert bundle.frame["source_row"].tolist() == list(range(10))
    assert {
        "job_name",
        "partition",
        "start_time",
        "job_state",
        "exit_code",
        "elapsed_minutes",
        "node_list",
        "observed_memory_peak_absolute",
        "observed_vram_peak_absolute",
        "workdir",
    }.issubset(bundle.frame.columns)


def test_configured_unlimited_and_truncated_walltimes_are_audited(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    rows[0]["time"] = "UNLIMITED"
    rows[1]["time"] = "41-15:00:+"
    rows[2]["time"] = "416-15:00+"
    write_trace_fixture(csv_path, rows)
    config = make_trace_config(csv_path)
    config["data"].pop("metadata_numeric")
    config["data"]["metadata_features"] = [
        {
            "name": "walltime_request",
            "column": "time",
            "parser": "slurm_minutes",
            "transform": "identity",
            "allow_truncated": True,
            "unlimited_as_max": True,
            "unlimited_minutes": 599940,
        }
    ]

    bundle = load_dataset(config)
    by_job = bundle.frame.set_index("job_id")
    assert by_job.loc["1000", "num__walltime_request"] == pytest.approx(599940.0)
    assert by_job.loc["1001", "num__walltime_request"] == pytest.approx(59940.0)
    assert by_job.loc["1002", "num__walltime_request"] == pytest.approx(599940.0)
    audit = bundle.protocol["metadata_parsing"]["walltime_request"]
    assert audit["unlimited_count"] == 1
    assert audit["truncated_count"] == 2
    assert audit["invalid_count_after_parsing"] == 0
    assert audit["unlimited_minutes"] == pytest.approx(599940.0)


def test_expected_header_order_is_enforced(tmp_path):
    csv_path = tmp_path / "trace.csv"
    write_trace_fixture(csv_path, [make_trace_row(index) for index in range(10)])
    config = make_trace_config(csv_path)
    config["data"]["expected_columns"] = TRACE_COLUMNS.copy()
    config["data"]["expected_columns"][0:2] = ["jobname", "jobid"]
    with pytest.raises(ValueError, match="exactly match"):
        load_dataset(config)


def test_filters_run_before_split_and_preserve_original_source_rows(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(8)]
    rows[1].update(
        {
            "state1": "FAILED",
            "start1": "",
            "end1": "",
            "elapsed": "",
            "mem_max": "",
        }
    )
    rows[3].update(
        {
            "exitcode": "1:0",
            "start1": "",
            "end1": "",
            "elapsed": "",
            "vram_max": "",
        }
    )
    write_trace_fixture(csv_path, rows)
    config = make_trace_config(csv_path)
    config["split"]["train_ratio"] = 0.5
    config["split"]["validation_ratio"] = 0.25
    bundle = load_dataset(config)

    assert bundle.protocol["raw_row_count"] == 8
    assert bundle.protocol["retained_row_count"] == 6
    assert bundle.protocol["dropped_row_count"] == 2
    dropped = bundle.protocol["dropped_by_filter"]
    assert dropped.get("state", dropped.get("states")) == 1
    assert dropped.get("exit_code", dropped.get("exit_codes")) == 1
    assert bundle.frame["source_row"].tolist() == [0, 2, 4, 5, 6, 7]
    assert bundle.protocol["split_counts"] == {
        "train": 3,
        "validation": 1,
        "test": 2,
    }


@pytest.mark.parametrize(
    ("feature_kind", "source"),
    [
        ("direct", "cpu_eff"),
        ("ratio", "mem_p95"),
        ("categorical", "state1"),
    ],
)
def test_configured_forbidden_metadata_is_rejected(
    tmp_path, feature_kind, source
):
    csv_path = tmp_path / "trace.csv"
    write_trace_fixture(csv_path, [make_trace_row(index) for index in range(10)])
    config = make_trace_config(csv_path)
    if feature_kind == "direct":
        config["data"]["metadata_numeric"].append(source)
    elif feature_kind == "categorical":
        config["data"]["metadata_categorical"].append(source)
    else:
        config["data"]["metadata_numeric"] = []
        config["data"]["metadata_features"] = [
            {
                "name": "forbidden_ratio",
                "transform": "ratio",
                "numerator": source,
                "denominator": "cpu",
            }
        ]
    with pytest.raises(ValueError, match="forbidden|cannot reference"):
        load_dataset(config)


def test_code_redaction_replaces_only_user_path_segment(tmp_path):
    csv_path = tmp_path / "trace.csv"
    script = (
        "#!/bin/bash\n"
        "cd /gpfsnyu/home/alice/project\n"
        'python "/gpfsnyu/scratch/bob/run,1/train.py"\n'
        "echo /opt/shared/model"
    )
    rows = [make_trace_row(index) for index in range(10)]
    rows[0]["script_content"] = script
    write_trace_fixture(csv_path, rows)
    config = make_trace_config(csv_path)
    config["data"]["code_redaction"]["enabled"] = True
    bundle = load_dataset(config)
    redacted = bundle.frame.loc[bundle.frame["job_id"] == "1000", "code"].iloc[0]
    assert "/gpfsnyu/home/<USER>/project" in redacted
    assert "/gpfsnyu/scratch/<USER>/run,1/train.py" in redacted
    assert "alice" not in redacted
    assert "bob" not in redacted
    assert "/opt/shared/model" in redacted


@pytest.mark.parametrize(
    "bad_times",
    [
        {
            "submit": "2026-02-01T00:20:00Z",
            "start1": "2026-02-01T00:10:00Z",
            "end1": "2026-02-01T00:40:00Z",
        },
        {
            "submit": "2026-02-01T00:00:00Z",
            "start1": "2026-02-01T00:40:00Z",
            "end1": "2026-02-01T00:30:00Z",
        },
    ],
)
def test_submit_start_completion_order_is_enforced(tmp_path, bad_times):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    rows[0].update(bad_times)
    write_trace_fixture(csv_path, rows)
    with pytest.raises(ValueError, match="submit|start|completion"):
        load_dataset(make_trace_config(csv_path))


def test_elapsed_must_match_start_to_completion_within_tolerance(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    rows[0]["elapsed"] = "00:29:58"
    write_trace_fixture(csv_path, rows)
    with pytest.raises(ValueError, match="[Ee]lapsed"):
        load_dataset(make_trace_config(csv_path))


def test_elapsed_difference_equal_to_tolerance_is_accepted(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    rows[0]["end1"] = "2026-02-01T00:40:01Z"
    write_trace_fixture(csv_path, rows)
    bundle = load_dataset(make_trace_config(csv_path))
    validation = bundle.protocol["event_time_validation"]
    assert validation["elapsed_checked"] is True
    assert validation["elapsed_tolerance_seconds"] == pytest.approx(1.0)
    assert validation["max_elapsed_delta_seconds"] == pytest.approx(1.0)


def test_naive_timestamps_use_configured_timezone(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    for index, row in enumerate(rows):
        base = pd.Timestamp("2026-02-01T00:00:00") + pd.Timedelta(days=index)
        row["submit"] = base.isoformat()
        row["start1"] = (base + pd.Timedelta(minutes=10)).isoformat()
        row["end1"] = (base + pd.Timedelta(minutes=40)).isoformat()
    write_trace_fixture(csv_path, rows)
    config = make_trace_config(csv_path)
    config["data"]["timestamp_timezone"] = "Asia/Shanghai"
    bundle = load_dataset(config)
    expected = pd.Timestamp("2026-02-01T00:00:00", tz="Asia/Shanghai").tz_convert("UTC")
    assert bundle.frame.iloc[0]["submit_time"] == expected


def test_strict_csv_preserves_multiline_comma_and_quote_in_script(tmp_path):
    csv_path = tmp_path / "trace.csv"
    script = '#!/bin/bash\necho "alpha,beta"\npython train.py --name "quoted value"'
    rows = [make_trace_row(index) for index in range(10)]
    rows[0]["script_content"] = script
    write_trace_fixture(csv_path, rows)
    bundle = load_dataset(make_trace_config(csv_path))
    assert bundle.frame.iloc[0]["code"] == script
    assert bundle.frame["source_row"].tolist() == list(range(10))


def test_strict_csv_rejects_wrong_logical_record_width(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TRACE_COLUMNS)
        for index, row in enumerate(rows):
            values = [row[column] for column in TRACE_COLUMNS]
            writer.writerow(values[:-1] if index == 4 else values)
    with pytest.raises(ValueError, match="field|width|column"):
        load_dataset(make_trace_config(csv_path))


def test_strict_csv_rejects_duplicate_header_names(tmp_path):
    csv_path = tmp_path / "trace.csv"
    rows = [make_trace_row(index) for index in range(10)]
    duplicate_header = TRACE_COLUMNS + ["user"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(duplicate_header)
        for row in rows:
            writer.writerow([row[column] for column in TRACE_COLUMNS] + ["duplicate"])
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        load_dataset(make_trace_config(csv_path))
