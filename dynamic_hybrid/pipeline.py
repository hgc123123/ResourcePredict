from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from common.config import get_value, resolve_path
from common.data import DatasetBundle, load_dataset
from dynamic_hybrid.contracts import (
    ComponentInputs,
    load_component_inputs,
    read_and_pair,
    validate_paired_against_bundle,
)
from dynamic_hybrid.features import GateFeatures, build_gate_features


@dataclass
class PreparedInputs:
    components: ComponentInputs
    bundle: DatasetBundle
    features: GateFeatures


def resolve_component_paths(
    config: Dict[str, Any],
    qwen_value: str | None,
    last1_value: str | None,
) -> tuple[Path, Path]:
    seed = int(get_value(config, "seed", 42))
    variant = str(
        get_value(
            config,
            "dynamic_hybrid.qwen_variant",
            get_value(config, "hybrid.qwen_variant", "full"),
        )
    )
    protocol = str(get_value(config, "last1.protocol", "completed_before_submission"))
    qwen_path = resolve_path(
        config,
        qwen_value
        or get_value(
            config,
            "dynamic_hybrid.qwen_predictions",
            f"outputs/qwen/{variant}/seed_{seed}/predictions.csv",
        ),
    )
    last1_path = resolve_path(
        config,
        last1_value
        or get_value(
            config,
            "dynamic_hybrid.last1_predictions",
            f"outputs/last-1/{protocol}/seed_{seed}/predictions.csv",
        ),
    )
    return qwen_path, last1_path


def prepare_inputs(
    config: Dict[str, Any],
    qwen_value: str | None,
    last1_value: str | None,
) -> PreparedInputs:
    qwen_path, last1_path = resolve_component_paths(config, qwen_value, last1_value)
    components = load_component_inputs(qwen_path, last1_path)
    configured_seed = int(get_value(config, "seed", 42))
    if configured_seed != components.seed:
        raise ValueError(
            f"Configured seed {configured_seed} differs from component seed {components.seed}"
        )
    expected_variant = str(
        get_value(
            config,
            "dynamic_hybrid.qwen_variant",
            get_value(config, "hybrid.qwen_variant", "full"),
        )
    )
    actual_variant = str((components.qwen_manifest.get("details") or {}).get("variant", ""))
    if actual_variant != expected_variant:
        raise ValueError(
            f"Configured Qwen variant {expected_variant!r} differs from artifact {actual_variant!r}"
        )

    bundle = load_dataset(config)
    if bundle.data_sha256 != components.data_sha256:
        raise ValueError("Configured dataset hash differs from component prediction manifests")
    bundle_split = str(bundle.protocol.get("split_fingerprint", ""))
    if bundle_split != components.split_fingerprint:
        raise ValueError("Configured split fingerprint differs from component manifests")

    paired = read_and_pair(components)
    validate_paired_against_bundle(paired, bundle)
    history_k = int(get_value(config, "dynamic_hybrid.history_k", 5))
    history_ema_alpha = float(
        get_value(config, "dynamic_hybrid.history_ema_alpha", 0.5)
    )
    features = build_gate_features(
        paired,
        bundle,
        history_k=history_k,
        history_ema_alpha=history_ema_alpha,
    )
    return PreparedInputs(components, bundle, features)


def gate_fit_and_selection_masks(
    prepared: PreparedInputs,
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    frame = prepared.features.paired
    split = frame["split"].astype(str).to_numpy()
    has_train = bool(np.any(split == "train"))
    if has_train:
        details = prepared.components.qwen_manifest.get("details") or {}
        if not bool(details.get("train_predictions_out_of_fold", False)):
            raise ValueError(
                "Qwen train predictions may fit the gate only when the manifest declares "
                "train_predictions_out_of_fold=true"
            )
        oof_protocol = details.get("oof_protocol")
        if not isinstance(oof_protocol, dict):
            raise ValueError("OOF train predictions require a structured details.oof_protocol")
        if str(oof_protocol.get("type", "")) != "expanding_window":
            raise ValueError("Dynamic hybrid currently accepts expanding-window OOF only")
        if str(oof_protocol.get("strict_cutoff_relation", "")) != (
            "training_data_end < prediction_submit_time"
        ):
            raise ValueError("OOF protocol does not declare the strict temporal cutoff relation")
        required_oof_columns = {"oof_fold", "oof_train_cutoff"}
        missing_oof = required_oof_columns.difference(frame.columns)
        if missing_oof:
            raise ValueError(
                "OOF train predictions are missing row-level provenance: "
                f"{sorted(missing_oof)}"
            )
        train_rows = frame.loc[frame["split"].eq("train")].copy()
        if train_rows[["oof_fold", "oof_train_cutoff"]].isna().any().any():
            raise ValueError("OOF train rows contain missing fold/cutoff provenance")
        cutoff = pd.to_datetime(train_rows["oof_train_cutoff"], errors="raise", utc=True)
        submit = pd.to_datetime(train_rows["submit_time"], errors="raise", utc=True)
        if not (cutoff < submit).all():
            raise ValueError("OOF train cutoff is not strictly before each predicted row")
        cutoff_counts = train_rows.assign(_cutoff=cutoff).groupby("oof_fold")["_cutoff"].nunique()
        if not (cutoff_counts == 1).all():
            raise ValueError("Each OOF fold must declare exactly one training cutoff")
        observed_folds = int(train_rows["oof_fold"].nunique())
        declared_folds = int(oof_protocol.get("fold_count", 0))
        if declared_folds != observed_folds or observed_folds < 2:
            raise ValueError("OOF fold count/provenance is inconsistent")
        fit = split == "train"
        selection = split == "validation"
        protocol = {
            "fit_source": "rolling/out-of-fold train component predictions",
            "selection_source": "validation",
            "test_used_for_fit_or_selection": False,
            "oof_protocol": oof_protocol,
            "oof_observed_fold_count": observed_folds,
        }
        return fit, selection, protocol

    validation = frame.loc[frame["split"].eq("validation"), ["sample_id", "submit_time"]]
    validation = validation.drop_duplicates("sample_id").copy()
    validation["submit_time"] = pd.to_datetime(
        validation["submit_time"], errors="raise", utc=True
    )
    validation = validation.sort_values(["submit_time", "sample_id"], kind="mergesort")
    fraction = float(get_value(config, "dynamic_hybrid.validation_fit_fraction", 0.80))
    if not 0.5 <= fraction < 1.0:
        raise ValueError("dynamic_hybrid.validation_fit_fraction must be in [0.5,1)")
    boundary = int(len(validation) * fraction)
    if boundary <= 0 or boundary >= len(validation):
        raise ValueError("Validation calibration split is empty")
    fit_ids = set(validation.iloc[:boundary]["sample_id"].astype(str))
    selection_ids = set(validation.iloc[boundary:]["sample_id"].astype(str))
    sample_ids = frame["sample_id"].astype(str)
    fit = frame["split"].eq("validation").to_numpy() & sample_ids.isin(fit_ids).to_numpy()
    selection = (
        frame["split"].eq("validation").to_numpy()
        & sample_ids.isin(selection_ids).to_numpy()
    )
    if np.any(fit & selection) or np.any((split == "test") & (fit | selection)):
        raise AssertionError("Test rows entered gate fit/selection masks")
    protocol = {
        "fit_source": "chronological leading validation calibration block",
        "selection_source": "chronological trailing validation calibration block",
        "validation_fit_fraction": fraction,
        "fit_unique_jobs": len(fit_ids),
        "selection_unique_jobs": len(selection_ids),
        "test_used_for_fit_or_selection": False,
        "warning": (
            "The standard Qwen artifact has no train OOF predictions. Validation is split "
            "chronologically for gate fitting and checkpoint selection; full-validation "
            "metrics are descriptive, not an independent validation estimate."
        ),
    }
    return fit, selection, protocol
