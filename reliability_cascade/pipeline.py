from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

from common.config import get_value, resolve_path
from common.data import DatasetBundle, load_dataset
from qwen.base_selection import (
    LEGACY_MODE,
    base_selection_settings,
    verify_base_training_selection,
)
from reliability_cascade.contracts import (
    ComponentInputs,
    load_component_inputs,
    read_and_pair,
    validate_paired_against_bundle,
)
from reliability_cascade.features import CascadeFeatures, build_cascade_features
from reliability_cascade.integrity import validate_variant

if TYPE_CHECKING:
    from reliability_cascade.model import CascadeConfig


PROTOCOL_VERSION = "history-first-reliability-cascade/v1"


@dataclass
class PreparedCascade:
    components: ComponentInputs
    bundle: DatasetBundle
    features: CascadeFeatures
    qwen_backbone: str


def _validate_completed_label_boundary_payload(
    boundary: object,
    *,
    source: str,
) -> None:
    """Require a completed, zero-crossing outer split-boundary audit."""

    if not isinstance(boundary, dict):
        raise ValueError(
            f"Cascade requires {source} completed_labels_at_boundaries protocol metadata"
        )
    for flag in ("required", "checked"):
        if boundary.get(flag) is not True:
            raise ValueError(
                f"Cascade requires {source} completed_labels_at_boundaries."
                f"{flag}=true"
            )
    for count in (
        "remaining_train_cross_boundary_count",
        "remaining_validation_cross_boundary_count",
    ):
        value = boundary.get(count)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"Cascade requires an integer {source} "
                f"completed_labels_at_boundaries.{count}"
            )
        if int(value) != 0:
            raise ValueError(
                f"Cascade requires {source} "
                f"completed_labels_at_boundaries.{count}=0; "
                f"observed {value}"
            )


def _validate_completed_label_boundaries(bundle: DatasetBundle) -> None:
    _validate_completed_label_boundary_payload(
        bundle.protocol.get("completed_labels_at_boundaries"),
        source="dataset",
    )


def resolve_component_paths(
    config: Dict[str, Any],
    qwen_value: str | None,
    last1_value: str | None,
) -> tuple[Path, Path]:
    seed = int(get_value(config, "seed", 42))
    variant = validate_variant(
        get_value(config, "reliability_cascade.qwen_variant", "full")
    )
    max_tokens = int(get_value(config, "qwen.max_tokens", 4096))
    if max_tokens <= 0:
        raise ValueError("qwen.max_tokens must be positive")
    protocol = str(get_value(config, "last1.protocol", "completed_before_submission"))
    qwen_path = resolve_path(
        config,
        qwen_value
        or get_value(
            config,
            "reliability_cascade.qwen_predictions",
            f"outputs/qwen/tokens_{max_tokens}/{variant}/seed_{seed}/predictions.csv",
        ),
    )
    last1_path = resolve_path(
        config,
        last1_value
        or get_value(
            config,
            "reliability_cascade.last1_predictions",
            f"outputs/last-1/{protocol}/seed_{seed}/predictions.csv",
        ),
    )
    return qwen_path, last1_path


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_protocol_identity(inputs: ComponentInputs) -> None:
    qwen_protocol = inputs.qwen_manifest.get("protocol") or {}
    last1_protocol = inputs.last1_manifest.get("protocol") or {}
    fields = (
        "order_fingerprint",
        "split_manifest_sha256",
        "code_redaction",
        "event_time_validation",
        "completed_labels_at_boundaries",
    )
    for field in fields:
        left = inputs.qwen_manifest.get(field, qwen_protocol.get(field))
        right = inputs.last1_manifest.get(field, last1_protocol.get(field))
        if left is None or right is None:
            raise ValueError(f"Component manifests do not both record {field}")
        if _canonical(left) != _canonical(right):
            raise ValueError(f"Qwen and Last-1 manifests disagree on {field}")
        if field == "completed_labels_at_boundaries":
            _validate_completed_label_boundary_payload(left, source="Qwen artifact")
            _validate_completed_label_boundary_payload(right, source="Last-1 artifact")


def _qwen_backbone(manifest: Dict[str, Any]) -> str:
    details = manifest.get("details") or {}
    candidates = (
        details.get("base_model_override"),
        details.get("model_name_or_path"),
        (details.get("effective_parameters") or {}).get("model_name_or_path"),
    )
    value = next((str(item).strip() for item in candidates if item), "")
    if not value:
        raise ValueError(
            "Qwen manifest lacks a backbone identifier; model=qwen_full is not sufficient"
        )
    return value


def _qwen_max_tokens(manifest: Dict[str, Any]) -> int:
    details = manifest.get("details") or {}
    effective = details.get("effective_parameters") or {}
    value = effective.get("max_tokens")
    if isinstance(value, bool) or value is None:
        raise ValueError("Qwen manifest lacks an effective max_tokens value")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Qwen effective max_tokens must be an integer") from exc
    if result <= 0 or float(value) != float(result):
        raise ValueError("Qwen effective max_tokens must be a positive integer")
    return result


def _base_checkpoint_selection_protocol(
    bundle: DatasetBundle | None,
    config: Dict[str, Any],
    qwen_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive checkpoint overlap from source-rebuilt, hash-bound provenance.

    A configuration flag can require independence, but cannot attest it. New
    selection modes must carry the record produced by Qwen training and the
    same record hash in the effective checkpoint parameters. Old artifacts
    remain explicitly unverified and retain the original overlap warning.
    """
    required = get_value(
        config, "reliability_cascade.require_independent_base_selection", False
    )
    if not isinstance(required, bool):
        raise ValueError(
            "reliability_cascade.require_independent_base_selection must be true or false"
        )
    settings = base_selection_settings(config)
    details = qwen_manifest.get("details") or {}
    if not isinstance(details, dict):
        raise ValueError("Qwen manifest details must be an object")
    effective = details.get("effective_parameters") or {}
    if not isinstance(effective, dict):
        raise ValueError("Qwen effective_parameters must be an object")
    record = details.get("base_training_selection")
    bound_hash = effective.get("base_training_selection_sha256")

    if settings["mode"] == LEGACY_MODE:
        if record is not None or bound_hash is not None:
            raise ValueError(
                "Qwen base-selection provenance is incompatible with legacy_validation configuration"
            )
        if required:
            raise ValueError(
                "Independent base checkpoint selection requires verified train_tail provenance; "
                "the legacy artifact has no such record"
            )
        return {
            "base_checkpoint_validation_overlap": True,
            "base_checkpoint_selection_provenance": "legacy_unverified",
            "base_selection_mode": LEGACY_MODE,
            "base_training_selection_sha256": None,
            "warning": (
                "The frozen Qwen checkpoint was selected on the full validation period. "
                "Use expanding-window OOF base predictions and a future untouched holdout "
                "for confirmatory claims."
            ),
        }

    if record is None:
        raise ValueError(
            "Qwen artifact lacks base_training_selection provenance for the configured selection mode"
        )
    if bundle is None:
        raise ValueError("Verified base-selection provenance requires the source dataset bundle")
    verified = verify_base_training_selection(bundle, config, record)
    if not isinstance(bound_hash, str) or bound_hash != verified["sha256"]:
        raise ValueError(
            "Qwen effective_parameters.base_training_selection_sha256 does not match "
            "the verified base-selection record"
        )
    overlap = verified["base_checkpoint_validation_overlap"]
    if not isinstance(overlap, bool):
        raise ValueError("Verified base checkpoint overlap must be a boolean")
    if required and (settings["mode"] != "train_tail" or overlap):
        raise ValueError(
            "Independent base checkpoint selection requires train_tail provenance; "
            "shared_validation_control deliberately reuses the gate validation period"
        )
    warning = (
        "The base checkpoint uses only a completion-safe outer-training tail for selection, "
        "disjoint from gate fitting and policy selection. This removes checkpoint-selection "
        "overlap; it does not make the previously inspected test period an untouched holdout."
        if not overlap else
        "This matched control deliberately selects the base checkpoint on the full validation "
        "period, which also supplies gate fitting and policy selection. It does not provide "
        "independent base checkpoint selection."
    )
    return {
        "base_checkpoint_validation_overlap": overlap,
        "base_checkpoint_selection_provenance": f"verified_{settings['mode']}",
        "base_selection_mode": settings["mode"],
        "base_training_selection_sha256": verified["sha256"],
        "warning": warning,
    }


def prepare_cascade(
    config: Dict[str, Any],
    qwen_value: str | None = None,
    last1_value: str | None = None,
) -> PreparedCascade:
    qwen_path, last1_path = resolve_component_paths(config, qwen_value, last1_value)
    components = load_component_inputs(qwen_path, last1_path)
    _validate_protocol_identity(components)
    seed = int(get_value(config, "seed", 42))
    if seed != components.seed:
        raise ValueError(
            f"Configured seed {seed} differs from component seed {components.seed}"
        )
    variant = validate_variant(
        get_value(config, "reliability_cascade.qwen_variant", "full")
    )
    actual_variant = str(
        (components.qwen_manifest.get("details") or {}).get("variant", "")
    )
    if actual_variant != variant:
        raise ValueError(
            f"Configured Qwen variant {variant!r} differs from artifact {actual_variant!r}"
        )
    backbone = _qwen_backbone(components.qwen_manifest)
    artifact_max_tokens = _qwen_max_tokens(components.qwen_manifest)
    configured_max_tokens = int(get_value(config, "qwen.max_tokens", 4096))
    if artifact_max_tokens != configured_max_tokens:
        raise ValueError(
            "Configured qwen.max_tokens differs from the Qwen artifact: "
            f"configured={configured_max_tokens}, artifact={artifact_max_tokens}"
        )
    expected = str(
        get_value(
            config,
            "reliability_cascade.expected_backbone_contains",
            "Qwen2.5-Coder-1.5B-Instruct",
        )
        or ""
    )
    if expected and expected.lower() not in backbone.lower():
        raise ValueError(
            f"Qwen backbone {backbone!r} does not contain expected identity {expected!r}"
        )

    bundle = load_dataset(config)
    _validate_completed_label_boundaries(bundle)
    if bundle.data_sha256 != components.data_sha256:
        raise ValueError("Configured dataset hash differs from component manifests")
    if str(bundle.protocol.get("split_fingerprint", "")) != components.split_fingerprint:
        raise ValueError("Configured split fingerprint differs from component manifests")
    _base_checkpoint_selection_protocol(bundle, config, components.qwen_manifest)
    paired = read_and_pair(components)
    validate_paired_against_bundle(paired, bundle)
    features = build_cascade_features(
        paired,
        bundle,
        history_k=int(get_value(config, "reliability_cascade.history_k", 5)),
        half_life_hours=float(
            get_value(config, "reliability_cascade.history_half_life_hours", 168.0)
        ),
    )
    return PreparedCascade(components, bundle, features, backbone)


def chronological_calibration_masks(
    prepared: PreparedCascade,
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    frame = prepared.features.frame
    validation = frame.loc[
        frame["split"].astype(str).eq("validation"),
        ["sample_id", "submit_time", "completion_time"],
    ].drop_duplicates("sample_id")
    validation["submit_time"] = pd.to_datetime(
        validation["submit_time"], errors="raise", utc=True
    )
    validation["completion_time"] = pd.to_datetime(
        validation["completion_time"], errors="raise", utc=True
    )
    validation = validation.sort_values(
        ["submit_time", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    fraction = float(
        get_value(config, "reliability_cascade.validation_fit_fraction", 0.70)
    )
    if not 0.5 <= fraction < 1.0:
        raise ValueError("validation_fit_fraction must be in [0.5, 1)")
    desired_boundary = int(len(validation) * fraction)
    if desired_boundary <= 0 or desired_boundary >= len(validation):
        raise ValueError("Chronological validation calibration creates an empty block")
    selection_start = validation.iloc[desired_boundary]["submit_time"]
    leading = validation["submit_time"] < selection_start
    selection_rows = validation["submit_time"] >= selection_start
    # Every fitted label must be observable before the first policy-selection
    # event. Rows crossing this inner boundary are excluded from gate fitting.
    fit_eligible = leading & (validation["completion_time"] < selection_start)
    fit_purged = leading & ~fit_eligible
    fit_ids = set(validation.loc[fit_eligible, "sample_id"].astype(str))
    selection_ids = set(validation.loc[selection_rows, "sample_id"].astype(str))
    if not fit_ids or not selection_ids:
        raise ValueError("Completion-safe validation calibration creates an empty block")
    sample_id = frame["sample_id"].astype(str)
    fit = frame["split"].astype(str).eq("validation").to_numpy() & sample_id.isin(
        fit_ids
    ).to_numpy()
    selection = frame["split"].astype(str).eq(
        "validation"
    ).to_numpy() & sample_id.isin(selection_ids).to_numpy()
    test = frame["split"].astype(str).eq("test").to_numpy()
    if np.any(fit & selection) or np.any(test & (fit | selection)):
        raise AssertionError("Test rows entered cascade fitting or selection")
    base_selection_protocol = _base_checkpoint_selection_protocol(
        getattr(prepared, "bundle", None), config,
        getattr(getattr(prepared, "components", None), "qwen_manifest", {}),
    )
    return fit, selection, {
        "protocol": PROTOCOL_VERSION,
        "fit_source": "chronological leading validation block",
        "selection_source": "chronological trailing validation block",
        "validation_fit_fraction": fraction,
        "desired_fit_unique_jobs_before_inner_purge": int(leading.sum()),
        "fit_unique_jobs": len(fit_ids),
        "selection_unique_jobs": len(selection_ids),
        "inner_boundary_purged_fit_jobs": int(fit_purged.sum()),
        "selection_start_submit_time": selection_start.isoformat(),
        "fit_completion_boundary_relation": (
            "fit completion_time < first selection submit_time"
        ),
        "equal_submit_times_split_across_blocks": False,
        "test_used_for_fit_or_selection": False,
        **base_selection_protocol,
    }


def cascade_model_config(config: Dict[str, Any], *, seed: int) -> CascadeConfig:
    from reliability_cascade.model import CascadeConfig

    prefix = "reliability_cascade"
    return CascadeConfig(
        seed=seed,
        require_exact_script_for_bypass=bool(
            get_value(config, f"{prefix}.require_exact_script_for_bypass", True)
        ),
        reliability_error_cap=float(
            get_value(config, f"{prefix}.reliability_error_cap", 0.10)
        ),
        pre_n_estimators=int(
            get_value(config, f"{prefix}.pre_gate.n_estimators", 100)
        ),
        pre_learning_rate=float(
            get_value(config, f"{prefix}.pre_gate.learning_rate", 0.03)
        ),
        pre_max_depth=int(get_value(config, f"{prefix}.pre_gate.max_depth", 2)),
        post_n_estimators=int(
            get_value(config, f"{prefix}.post_gate.n_estimators", 120)
        ),
        post_learning_rate=float(
            get_value(config, f"{prefix}.post_gate.learning_rate", 0.03)
        ),
        post_max_depth=int(get_value(config, f"{prefix}.post_gate.max_depth", 2)),
        post_blend_enabled=bool(
            get_value(config, f"{prefix}.post_gate.enabled", True)
        ),
        trust_region_enabled=bool(
            get_value(config, f"{prefix}.trust_region.enabled", True)
        ),
        trust_region_base=float(
            get_value(config, f"{prefix}.trust_region.base_radius", 0.025)
        ),
        trust_region_mad_scale=float(
            get_value(config, f"{prefix}.trust_region.mad_scale", 2.0)
        ),
        trust_region_max=float(
            get_value(config, f"{prefix}.trust_region.max_radius", 0.10)
        ),
    )


__all__ = [
    "PROTOCOL_VERSION",
    "PreparedCascade",
    "_validate_completed_label_boundaries",
    "_validate_completed_label_boundary_payload",
    "_validate_protocol_identity",
    "_base_checkpoint_selection_protocol",
    "cascade_model_config",
    "chronological_calibration_masks",
    "prepare_cascade",
    "resolve_component_paths",
]
