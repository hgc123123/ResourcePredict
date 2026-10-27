from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from common.config import get_value
from reliability_cascade.history import history_candidate_names
from reliability_cascade.model import CascadeConfig, ResourceCascade
from reliability_cascade.pipeline import PreparedCascade


@dataclass
class ResourceOption:
    option_id: str
    resource: str
    model: ResourceCascade
    output: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    qwen_required_rate: float


@dataclass
class SelectedCascade:
    models: Dict[str, ResourceCascade]
    sweep: pd.DataFrame
    policy: Dict[str, Any]
    selection_outputs: Dict[str, Dict[str, np.ndarray]]


def _metric(
    prepared: PreparedCascade,
    resource: str,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    tolerance: float,
) -> Dict[str, float]:
    frame = prepared.features.frame
    use = (
        np.asarray(mask, dtype=bool)
        & prepared.features.resource(resource)
        & frame["valid"].astype(bool).to_numpy()
    )
    if not use.any():
        raise ValueError(f"No valid {resource} rows in the selection block")
    target = frame["target"].to_numpy(float)[use]
    values = np.asarray(prediction, dtype=float)[use]
    error = values - target
    absolute = np.abs(error)
    squared = np.square(error)
    return {
        "n": int(len(target)),
        "mae": float(np.mean(absolute)),
        "mse": float(np.mean(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "acc_at_10": float(np.mean(absolute <= tolerance)),
    }


def _resource_call_rate(
    prepared: PreparedCascade,
    output: Dict[str, np.ndarray],
    selection_mask: np.ndarray,
) -> float:
    frame = prepared.features.frame
    applicable = (
        np.asarray(selection_mask, dtype=bool)
        & output["resource_mask"]
        & frame["resource_applicable"].astype(bool).to_numpy()
    )
    if not applicable.any():
        return 0.0
    return float(np.mean(output["qwen_required_resource"][applicable]))


def _job_call_rate(
    prepared: PreparedCascade,
    memory: ResourceOption,
    vram: ResourceOption,
    selection_mask: np.ndarray,
) -> tuple[int, int, float]:
    frame = prepared.features.frame
    selected = frame.loc[
        np.asarray(selection_mask, dtype=bool), ["sample_id"]
    ].copy()
    selected["qwen_required"] = (
        memory.output["qwen_required_resource"][selection_mask]
        | vram.output["qwen_required_resource"][selection_mask]
    )
    jobs = selected.groupby("sample_id", sort=False)["qwen_required"].any()
    return int(len(jobs)), int(jobs.sum()), float(jobs.mean())


def _lowest_mae_reference(options: Iterable[ResourceOption]) -> ResourceOption:
    return min(
        options,
        key=lambda option: (
            option.metrics["mae"],
            option.metrics["rmse"],
            -option.metrics["acc_at_10"],
            option.qwen_required_rate,
            option.option_id,
        ),
    )


def select_cascade(
    prepared: PreparedCascade,
    config: Dict[str, Any],
    fit_mask: np.ndarray,
    selection_mask: np.ndarray,
    model_config: CascadeConfig,
) -> SelectedCascade:
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    if not np.isclose(tolerance, 0.10, rtol=0.0, atol=1e-12):
        raise ValueError("Cascade experiments fix Acc@10% tolerance at 0.10")
    configured_candidates = get_value(
        config,
        "reliability_cascade.history_experts",
        list(history_candidate_names()),
    )
    candidates = tuple(str(value) for value in configured_candidates)
    known = set(history_candidate_names())
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("history_experts must be a non-empty unique sequence")
    unknown = set(candidates) - known
    if unknown:
        raise ValueError(f"Unknown history experts: {sorted(unknown)}")
    raw_thresholds = get_value(
        config,
        "reliability_cascade.pre_gate.thresholds",
        [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01],
    )
    thresholds = tuple(float(value) for value in raw_thresholds)
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(not np.isfinite(value) or value < 0.0 for value in thresholds)
    ):
        raise ValueError("pre_gate.thresholds must be unique finite non-negative values")

    by_resource: Dict[str, list[ResourceOption]] = {"memory": [], "vram": []}
    for resource in ("memory", "vram"):
        for candidate in candidates:
            fitted = ResourceCascade.fit(
                prepared.features,
                resource=resource,
                candidate_name=candidate,
                fit_mask=fit_mask,
                config=model_config,
            )
            for threshold in thresholds:
                model = fitted.with_threshold(threshold)
                output = model.predict(prepared.features)
                metrics = _metric(
                    prepared,
                    resource,
                    output["prediction"],
                    selection_mask,
                    tolerance=tolerance,
                )
                option_id = f"{resource}:{candidate}:threshold={threshold:.6g}"
                by_resource[resource].append(
                    ResourceOption(
                        option_id=option_id,
                        resource=resource,
                        model=model,
                        output=output,
                        metrics=metrics,
                        qwen_required_rate=_resource_call_rate(
                            prepared, output, selection_mask
                        ),
                    )
                )

    references = {
        resource: _lowest_mae_reference(options)
        for resource, options in by_resource.items()
    }
    max_mae = float(
        get_value(
            config,
            "reliability_cascade.selection.max_mae_increase_vs_accuracy_reference",
            0.001,
        )
    )
    max_rmse = float(
        get_value(
            config,
            "reliability_cascade.selection.max_rmse_increase_vs_accuracy_reference",
            0.002,
        )
    )
    max_acc_drop = float(
        get_value(
            config,
            "reliability_cascade.selection.max_acc_drop_vs_accuracy_reference",
            0.005,
        )
    )
    if any(value < 0.0 or not np.isfinite(value) for value in (max_mae, max_rmse, max_acc_drop)):
        raise ValueError("Selection constraint budgets must be finite and non-negative")
    objective = str(
        get_value(
            config,
            "reliability_cascade.selection.objective",
            "constrained_efficiency",
        )
    )
    if objective not in {"constrained_efficiency", "accuracy_first"}:
        raise ValueError("selection.objective must be constrained_efficiency or accuracy_first")

    rows: list[Dict[str, Any]] = []
    pairs: list[tuple[ResourceOption, ResourceOption, bool, tuple[object, ...]]] = []
    for memory, vram in product(by_resource["memory"], by_resource["vram"]):
        feasible = True
        for option in (memory, vram):
            reference = references[option.resource]
            feasible &= option.metrics["mae"] <= reference.metrics["mae"] + max_mae + 1e-15
            feasible &= option.metrics["rmse"] <= reference.metrics["rmse"] + max_rmse + 1e-15
            feasible &= (
                option.metrics["acc_at_10"] + max_acc_drop + 1e-15
                >= reference.metrics["acc_at_10"]
            )
        job_n, qwen_n, qwen_rate = _job_call_rate(
            prepared, memory, vram, selection_mask
        )
        mean_mae = 0.5 * (memory.metrics["mae"] + vram.metrics["mae"])
        mean_rmse = 0.5 * (memory.metrics["rmse"] + vram.metrics["rmse"])
        mean_acc = 0.5 * (
            memory.metrics["acc_at_10"] + vram.metrics["acc_at_10"]
        )
        if objective == "constrained_efficiency":
            rank = (qwen_rate, mean_mae, mean_rmse, -mean_acc, memory.option_id, vram.option_id)
        else:
            rank = (mean_mae, mean_rmse, -mean_acc, qwen_rate, memory.option_id, vram.option_id)
        pairs.append((memory, vram, bool(feasible), rank))
        rows.append(
            {
                "memory_option": memory.option_id,
                "vram_option": vram.option_id,
                "memory_mae": memory.metrics["mae"],
                "memory_rmse": memory.metrics["rmse"],
                "memory_acc_at_10": memory.metrics["acc_at_10"],
                "vram_mae": vram.metrics["mae"],
                "vram_rmse": vram.metrics["rmse"],
                "vram_acc_at_10": vram.metrics["acc_at_10"],
                "selection_job_n": job_n,
                "qwen_required_job_n": qwen_n,
                "qwen_required_job_rate": qwen_rate,
                "feasible": bool(feasible),
                "selected": False,
            }
        )
    feasible_pairs = [item for item in pairs if item[2]]
    if not feasible_pairs:
        raise AssertionError("Accuracy-reference pair should always satisfy its own budgets")
    selected_memory, selected_vram, _, _ = min(feasible_pairs, key=lambda item: item[3])
    sweep = pd.DataFrame(rows)
    selected_mask = sweep["memory_option"].eq(
        selected_memory.option_id
    ) & sweep["vram_option"].eq(selected_vram.option_id)
    if int(selected_mask.sum()) != 1:
        raise AssertionError("Joint cascade selection is not unique")
    sweep.loc[selected_mask, "selected"] = True

    refit = bool(
        get_value(config, "reliability_cascade.refit_on_full_validation", False)
    )
    if refit:
        raise ValueError(
            "Protocol v1 requires refit_on_full_validation=false so the selected "
            "threshold retains its calibrated probability scale"
        )
    selected_options = {"memory": selected_memory, "vram": selected_vram}
    models: Dict[str, ResourceCascade] = {}
    for resource, option in selected_options.items():
        models[resource] = option.model

    policy = {
        "protocol": "validation-frozen joint resource selection",
        "objective": objective,
        "refit_on_full_validation": refit,
        "test_labels_used": False,
        "candidate_history_experts": list(candidates),
        "candidate_reliability_thresholds": list(thresholds),
        "accuracy_reference_definition": (
            "lowest selection-block MAE per resource; ties use RMSE, "
            "Acc@10%, Qwen-call rate, and option ID"
        ),
        "constraints": {
            "max_mae_increase_vs_accuracy_reference": max_mae,
            "max_rmse_increase_vs_accuracy_reference": max_rmse,
            "max_acc_drop_vs_accuracy_reference": max_acc_drop,
        },
        "selection_reference": {
            resource: {
                "option": reference.option_id,
                **reference.metrics,
            }
            for resource, reference in references.items()
        },
        "selected": {
            "memory": models["memory"].policy(),
            "vram": models["vram"].policy(),
        },
        "joint_selection": sweep.loc[selected_mask].iloc[0].to_dict(),
    }
    return SelectedCascade(
        models=models,
        sweep=sweep,
        policy=policy,
        selection_outputs={
            "memory": selected_memory.output,
            "vram": selected_vram.output,
        },
    )


__all__ = ["SelectedCascade", "select_cascade"]
