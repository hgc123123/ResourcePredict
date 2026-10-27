from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from common.data import DatasetBundle, sha256_file
from common.metrics import coerce_valid_mask, validate_prediction_frame


KEYS = ["sample_id", "split", "resource"]
REQUIRED_SPLITS = {"validation", "test"}
RESOURCES = {"memory", "vram"}


@dataclass(frozen=True)
class ComponentInputs:
    qwen_path: Path
    last1_path: Path
    qwen_manifest: Dict[str, Any]
    last1_manifest: Dict[str, Any]
    data_sha256: str
    split_fingerprint: str
    seed: int

    def checkpoint_contract(self) -> Dict[str, Any]:
        return {
            "data_sha256": self.data_sha256,
            "split_fingerprint": self.split_fingerprint,
            "seed": self.seed,
            "qwen_predictions_sha256": sha256_file(self.qwen_path),
            "last1_predictions_sha256": sha256_file(self.last1_path),
            "qwen_run_manifest_sha256": sha256_file(
                self.qwen_path.parent / "run_manifest.json"
            ),
            "last1_run_manifest_sha256": sha256_file(
                self.last1_path.parent / "run_manifest.json"
            ),
            "last1_protocol": "completed_before_submission",
        }


def _read_manifest(prediction_path: Path, component: str) -> Dict[str, Any]:
    path = prediction_path.parent / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{component} run manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{component} run manifest must be a JSON object")
    if not value.get("data_sha256"):
        raise ValueError(f"{component} run manifest has no data_sha256")
    return value


def _split_fingerprint(manifest: Dict[str, Any], component: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{component} run manifest has no split fingerprint")
    return str(value)


def load_component_inputs(qwen_path: Path, last1_path: Path) -> ComponentInputs:
    if not qwen_path.is_file():
        raise FileNotFoundError(f"Qwen predictions not found: {qwen_path}")
    if not last1_path.is_file():
        raise FileNotFoundError(f"Last-1 predictions not found: {last1_path}")
    qwen_manifest = _read_manifest(qwen_path, "Qwen")
    last1_manifest = _read_manifest(last1_path, "Last-1")

    qwen_hash = str(qwen_manifest["data_sha256"])
    last1_hash = str(last1_manifest["data_sha256"])
    if qwen_hash != last1_hash:
        raise ValueError("Qwen and Last-1 manifests reference different datasets")
    qwen_split = _split_fingerprint(qwen_manifest, "Qwen")
    last1_split = _split_fingerprint(last1_manifest, "Last-1")
    if qwen_split != last1_split:
        raise ValueError("Qwen and Last-1 manifests reference different splits")

    last1_details = last1_manifest.get("details") or {}
    if not bool(last1_details.get("causal_online_protocol", False)):
        raise ValueError("Dynamic hybrid requires a causal online Last-1 run")
    if str(last1_details.get("protocol", "")) != "completed_before_submission":
        raise ValueError("Last-1 must use completed_before_submission history")
    strict_relation = last1_details.get("strict_completion_relation")
    if strict_relation is not None and str(strict_relation) != "completion_time < submit_time":
        raise ValueError("Last-1 manifest does not use strict completion-before-submit history")

    try:
        qwen_seed = int(qwen_manifest["seed"])
        last1_seed = int(last1_manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Both component manifests require an integer seed") from exc
    if qwen_seed != last1_seed:
        raise ValueError("Qwen and Last-1 manifest seeds differ")

    return ComponentInputs(
        qwen_path=qwen_path,
        last1_path=last1_path,
        qwen_manifest=qwen_manifest,
        last1_manifest=last1_manifest,
        data_sha256=qwen_hash,
        split_fingerprint=qwen_split,
        seed=qwen_seed,
    )


def _boolean(values: pd.Series, name: str) -> pd.Series:
    try:
        return pd.Series(coerce_valid_mask(values, name), index=values.index, dtype=bool)
    except ValueError as exc:
        raise ValueError(f"Invalid boolean column {name}") from exc


def _prepare(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    validate_prediction_frame(frame)
    output = frame.copy()
    output["sample_id"] = output["sample_id"].astype(str)
    output["split"] = output["split"].astype(str)
    output["resource"] = output["resource"].astype(str)
    observed_splits = set(output["split"])
    if not REQUIRED_SPLITS.issubset(observed_splits):
        raise ValueError(
            f"{component} predictions require validation and test; observed "
            f"{sorted(observed_splits)}"
        )
    unknown_resources = set(output["resource"]) - RESOURCES
    if unknown_resources:
        raise ValueError(f"{component} contains unknown resources: {sorted(unknown_resources)}")
    if output.duplicated(KEYS).any():
        raise ValueError(f"{component} contains duplicate sample/split/resource keys")
    return output


def validate_component_identity(
    frame: pd.DataFrame,
    manifest: Dict[str, Any],
    component: str,
) -> None:
    models = frame["model"].astype(str).unique().tolist()
    seeds = pd.to_numeric(frame["seed"], errors="coerce")
    if len(models) != 1 or models[0] != str(manifest.get("model")):
        raise ValueError(f"{component} prediction model disagrees with its manifest")
    if seeds.isna().any() or len(seeds.unique()) != 1:
        raise ValueError(f"{component} predictions require one finite seed")
    if int(seeds.iloc[0]) != int(manifest.get("seed")):
        raise ValueError(f"{component} prediction seed disagrees with its manifest")


def read_and_pair(inputs: ComponentInputs) -> pd.DataFrame:
    qwen = pd.read_csv(inputs.qwen_path, dtype={"sample_id": str, "job_id": str})
    last1 = pd.read_csv(inputs.last1_path, dtype={"sample_id": str, "job_id": str})
    validate_component_identity(qwen, inputs.qwen_manifest, "Qwen")
    validate_component_identity(last1, inputs.last1_manifest, "Last-1")
    qwen_details = inputs.qwen_manifest.get("details") or {}
    variant = str(qwen_details.get("variant", ""))
    if not variant or str(inputs.qwen_manifest.get("model")) != f"qwen_{variant}":
        raise ValueError("Qwen manifest has an inconsistent model/variant identity")
    if variant == "full":
        required = {
            "gate_code",
            "gate_user",
            "gate_metadata",
            "user_known",
            "script_sha256",
        }
        missing = required.difference(qwen.columns)
        if missing:
            raise ValueError(
                "Full Qwen predictions are missing required dynamic-gate fields: "
                f"{sorted(missing)}"
            )
        gates = qwen[["gate_code", "gate_user", "gate_metadata"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(gates).all():
            raise ValueError("Full Qwen gate values must all be finite")
        if np.any((gates < 0.0) | (gates > 1.0)):
            raise ValueError("Full Qwen gate values must be within [0,1]")
        if not np.allclose(gates.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("Full Qwen gate values must sum to one per row")
        _boolean(qwen["user_known"], "user_known")
        hashes = qwen["script_sha256"].astype(str).str.strip()
        if hashes.eq("").any() or not hashes.str.fullmatch(r"[0-9a-fA-F]{64}").all():
            raise ValueError("Full Qwen script_sha256 values must be non-empty SHA-256 hex")
    required_last1 = {
        "history_available",
        "history_source_sample_id",
        "history_source_time",
    }
    missing_last1 = required_last1.difference(last1.columns)
    if missing_last1:
        raise ValueError(
            "Causal Last-1 predictions are missing dynamic-history fields: "
            f"{sorted(missing_last1)}"
        )
    return pair_component_predictions(qwen, last1)


def pair_component_predictions(qwen: pd.DataFrame, last1: pd.DataFrame) -> pd.DataFrame:
    """Align components without allowing Last-1-only rows to enter gate fitting."""
    qwen = _prepare(qwen, "Qwen")
    last1 = _prepare(last1, "Last-1")
    observed_splits = set(qwen["split"])
    last1 = last1.loc[last1["split"].isin(observed_splits)].copy()

    qwen_keys = pd.MultiIndex.from_frame(qwen[KEYS])
    last1_keys = pd.MultiIndex.from_frame(last1[KEYS])
    missing = qwen_keys.difference(last1_keys)
    extra = last1_keys.difference(qwen_keys)
    if len(missing):
        raise ValueError(
            "Qwen and Last-1 prediction keys differ in shared splits: "
            f"missing_from_last1={len(missing)}"
        )
    if len(extra):
        extra_frame = extra.to_frame(index=False)
        non_train_extra = extra_frame.loc[extra_frame["split"].astype(str) != "train"]
        if not non_train_extra.empty:
            raise ValueError(
                "Last-1 has validation/test rows absent from Qwen predictions: "
                f"count={len(non_train_extra)}"
            )
        # Expanding-window OOF prediction files legitimately omit the initial
        # train prefix used to fit the first fold.
        qwen_key_frame = qwen[KEYS].copy()
        last1 = last1.merge(qwen_key_frame, on=KEYS, how="inner", validate="one_to_one")

    paired = qwen.merge(
        last1,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_qwen", "_last1"),
        sort=False,
    )
    for identity in ("job_id", "user_id"):
        left, right = f"{identity}_qwen", f"{identity}_last1"
        if left in paired and right in paired:
            mismatch = paired[left].astype(str) != paired[right].astype(str)
            if mismatch.any():
                raise ValueError(
                    f"Qwen and Last-1 disagree on {identity} for {int(mismatch.sum())} rows"
                )

    qwen_valid = _boolean(paired["valid_qwen"], "valid_qwen")
    last1_valid = _boolean(paired["valid_last1"], "valid_last1")
    if not qwen_valid.equals(last1_valid):
        raise ValueError("Qwen and Last-1 validity masks differ")

    qwen_target = pd.to_numeric(paired["target_qwen"], errors="coerce").to_numpy(float)
    last1_target = pd.to_numeric(paired["target_last1"], errors="coerce").to_numpy(float)
    qwen_finite = np.isfinite(qwen_target)
    last1_finite = np.isfinite(last1_target)
    if not np.array_equal(qwen_finite, last1_finite):
        raise ValueError("Qwen and Last-1 target missingness differs")
    mismatch = qwen_finite & ~np.isclose(qwen_target, last1_target, rtol=0.0, atol=1e-12)
    if mismatch.any():
        raise ValueError(f"Qwen and Last-1 targets differ for {int(mismatch.sum())} rows")

    history_name = next(
        (name for name in ("history_available", "has_history", "last1_available") if name in paired),
        None,
    )
    if history_name is None:
        # An overlapping history_available column receives the Last-1 suffix.
        history_name = next(
            (
                name
                for name in (
                    "history_available_last1",
                    "has_history_last1",
                    "last1_available_last1",
                )
                if name in paired
            ),
            None,
        )
    if history_name is None:
        raise ValueError("Last-1 predictions require an explicit history_available column")

    paired["valid"] = qwen_valid
    paired["target"] = qwen_target
    paired["qwen_prediction"] = pd.to_numeric(
        paired["prediction_qwen"], errors="coerce"
    )
    paired["last1_prediction"] = pd.to_numeric(
        paired["prediction_last1"], errors="coerce"
    )
    paired["history_available"] = _boolean(paired[history_name], history_name)
    if not np.isfinite(paired["qwen_prediction"].to_numpy(float)).all():
        raise ValueError("Qwen predictions contain non-finite values")
    history = paired["history_available"].to_numpy(bool)
    last1_finite = np.isfinite(paired["last1_prediction"].to_numpy(float))
    if np.any(history & ~last1_finite):
        raise ValueError("Last-1 prediction is non-finite despite available history")
    return paired


def validate_paired_against_bundle(
    paired: pd.DataFrame,
    bundle: DatasetBundle,
) -> None:
    """Verify component rows against the configured dataset before gate fitting."""
    expected_parts = []
    for resource, target_column, valid_column in (
        ("memory", "target_memory", "valid_memory"),
        ("vram", "target_vram", "valid_vram"),
    ):
        expected_parts.append(
            pd.DataFrame(
                {
                    "sample_id": bundle.frame["sample_id"].astype(str).to_numpy(),
                    "job_id_expected": bundle.frame["job_id"].astype(str).to_numpy(),
                    "user_id_expected": bundle.frame["user_id"].astype(str).to_numpy(),
                    "split": bundle.frame["split"].astype(str).to_numpy(),
                    "resource": resource,
                    "target_expected": bundle.frame[target_column].to_numpy(float),
                    "valid_expected": bundle.frame[valid_column].to_numpy(bool),
                }
            )
        )
    expected = pd.concat(expected_parts, ignore_index=True)

    actual = paired.copy()
    for name in KEYS:
        actual[name] = actual[name].astype(str)
    keys = list(KEYS)
    expected_keys = pd.MultiIndex.from_frame(expected[keys])
    actual_keys = pd.MultiIndex.from_frame(actual[keys])
    extra = actual_keys.difference(expected_keys)
    if len(extra):
        raise ValueError(
            "Component predictions contain rows outside the configured dataset/split: "
            f"count={len(extra)}"
        )

    expected_evaluation = expected.loc[
        expected["split"].isin(REQUIRED_SPLITS), keys
    ]
    actual_evaluation = actual.loc[actual["split"].isin(REQUIRED_SPLITS), keys]
    expected_evaluation_keys = pd.MultiIndex.from_frame(expected_evaluation)
    actual_evaluation_keys = pd.MultiIndex.from_frame(actual_evaluation)
    missing_evaluation = expected_evaluation_keys.difference(actual_evaluation_keys)
    extra_evaluation = actual_evaluation_keys.difference(expected_evaluation_keys)
    if len(missing_evaluation) or len(extra_evaluation):
        raise ValueError(
            "Component evaluation coverage differs from the configured dataset: "
            f"missing={len(missing_evaluation)}, extra={len(extra_evaluation)}"
        )

    checked = actual.merge(expected, on=keys, how="inner", validate="one_to_one")
    if len(checked) != len(actual):
        raise AssertionError("Dataset-contract merge dropped component prediction rows")

    job_column = "job_id_qwen" if "job_id_qwen" in checked else "job_id_last1"
    if not (
        checked[job_column].astype(str) == checked["job_id_expected"].astype(str)
    ).all():
        raise ValueError("Component prediction job IDs differ from the configured dataset")
    user_column = "user_id_qwen" if "user_id_qwen" in checked else None
    if user_column is not None and not (
        checked[user_column].astype(str) == checked["user_id_expected"].astype(str)
    ).all():
        raise ValueError("Component prediction user IDs differ from the configured dataset")

    actual_valid = _boolean(checked["valid"], "paired valid").to_numpy(bool)
    expected_valid = checked["valid_expected"].to_numpy(bool)
    if not np.array_equal(actual_valid, expected_valid):
        raise ValueError("Component validity masks differ from the configured dataset")

    actual_target = pd.to_numeric(checked["target"], errors="coerce").to_numpy(float)
    expected_target = checked["target_expected"].to_numpy(float)
    actual_finite = np.isfinite(actual_target)
    expected_finite = np.isfinite(expected_target)
    if not np.array_equal(actual_finite, expected_finite):
        raise ValueError("Component target missingness differs from the configured dataset")
    mismatch = expected_finite & ~np.isclose(
        actual_target,
        expected_target,
        rtol=0.0,
        atol=1e-12,
    )
    if mismatch.any():
        raise ValueError(
            "Component targets differ from the configured dataset for "
            f"{int(mismatch.sum())} rows"
        )
