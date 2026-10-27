from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from common.metrics import validate_prediction_frame


PAIR_KEYS = ["sample_id", "split", "resource"]
EVALUATION_SPLITS = ("validation", "test")
RESOURCES = ("memory", "vram")
HISTORY_COLUMNS = ("history_available", "has_history", "last1_available")


def _boolean_series(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} contains missing boolean values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{name} must contain only boolean or 0/1 values")
        return numeric.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = ~normalized.isin(mapping)
    if unknown.any():
        examples = sorted(normalized.loc[unknown].unique().tolist())[:5]
        raise ValueError(f"{name} contains invalid boolean values: {examples}")
    return normalized.map(mapping).astype(bool)


def _history_column(frame: pd.DataFrame) -> str:
    found = [name for name in HISTORY_COLUMNS if name in frame.columns]
    if not found:
        raise ValueError(
            "Last-1 predictions must include history_available (or the supported "
            "aliases has_history/last1_available); fallback predictions cannot be "
            "used to infer whether causal history exists."
        )
    if len(found) > 1:
        reference = _boolean_series(frame[found[0]], found[0])
        for name in found[1:]:
            candidate = _boolean_series(frame[name], name)
            if not candidate.equals(reference):
                raise ValueError(f"Conflicting Last-1 history columns: {found}")
    return found[0]


def _scoped_predictions(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    validate_prediction_frame(frame)
    scoped = frame.loc[
        frame["split"].astype(str).isin(EVALUATION_SPLITS)
        & frame["resource"].astype(str).isin(RESOURCES)
    ].copy()
    observed_splits = set(scoped["split"].astype(str))
    if observed_splits != set(EVALUATION_SPLITS):
        raise ValueError(
            f"{source} must contain validation and test predictions; observed "
            f"{sorted(observed_splits)}"
        )
    scoped["sample_id"] = scoped["sample_id"].astype(str)
    scoped["split"] = scoped["split"].astype(str)
    scoped["resource"] = scoped["resource"].astype(str)
    if scoped.duplicated(PAIR_KEYS).any():
        raise ValueError(f"{source} contains duplicate sample/split/resource keys")
    return scoped


def _assert_same_keys(qwen: pd.DataFrame, last1: pd.DataFrame) -> None:
    qwen_keys = pd.MultiIndex.from_frame(qwen[PAIR_KEYS])
    last1_keys = pd.MultiIndex.from_frame(last1[PAIR_KEYS])
    missing_last1 = qwen_keys.difference(last1_keys)
    missing_qwen = last1_keys.difference(qwen_keys)
    if len(missing_last1) or len(missing_qwen):
        raise ValueError(
            "Qwen and Last-1 prediction keys differ: "
            f"missing_from_last1={len(missing_last1)}, "
            f"missing_from_qwen={len(missing_qwen)}"
        )


def _assert_matching_identity_and_labels(paired: pd.DataFrame) -> None:
    for name in ("job_id", "user_id"):
        left = f"{name}_qwen"
        right = f"{name}_last1"
        if left in paired.columns and right in paired.columns:
            mismatch = paired[left].astype(str) != paired[right].astype(str)
            if mismatch.any():
                raise ValueError(f"Qwen and Last-1 disagree on {name} for {int(mismatch.sum())} rows")

    valid_qwen = _boolean_series(paired["valid_qwen"], "valid_qwen")
    valid_last1 = _boolean_series(paired["valid_last1"], "valid_last1")
    if not valid_qwen.equals(valid_last1):
        raise ValueError("Qwen and Last-1 validity masks differ")

    target_qwen = pd.to_numeric(paired["target_qwen"], errors="coerce").to_numpy(float)
    target_last1 = pd.to_numeric(paired["target_last1"], errors="coerce").to_numpy(float)
    finite_qwen = np.isfinite(target_qwen)
    finite_last1 = np.isfinite(target_last1)
    if not np.array_equal(finite_qwen, finite_last1):
        raise ValueError("Qwen and Last-1 target missingness differs")
    mismatch = finite_qwen & ~np.isclose(target_qwen, target_last1, rtol=0.0, atol=1e-12)
    if mismatch.any():
        raise ValueError(f"Qwen and Last-1 targets differ for {int(mismatch.sum())} rows")


def pair_predictions(qwen: pd.DataFrame, last1: pd.DataFrame) -> pd.DataFrame:
    """Strictly align independent predictions without using labels for matching."""
    qwen_scoped = _scoped_predictions(qwen, "Qwen")
    last1_scoped = _scoped_predictions(last1, "Last-1")
    history_name = _history_column(last1_scoped)
    last1_scoped = last1_scoped.copy()
    last1_scoped["causal_history_available"] = _boolean_series(
        last1_scoped[history_name], history_name
    )
    _assert_same_keys(qwen_scoped, last1_scoped)

    paired = qwen_scoped.merge(
        last1_scoped,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_qwen", "_last1"),
        sort=False,
    )
    _assert_matching_identity_and_labels(paired)
    paired["valid"] = _boolean_series(paired["valid_qwen"], "valid_qwen")
    paired["target"] = pd.to_numeric(paired["target_qwen"], errors="coerce")
    paired["qwen_prediction"] = pd.to_numeric(
        paired["prediction_qwen"], errors="coerce"
    )
    paired["last1_prediction"] = pd.to_numeric(
        paired["prediction_last1"], errors="coerce"
    )
    paired["history_available"] = _boolean_series(
        paired["causal_history_available"], "causal_history_available"
    )

    if not np.isfinite(paired["qwen_prediction"].to_numpy(float)).all():
        raise ValueError("Qwen predictions contain non-finite values")
    history = paired["history_available"].to_numpy(bool)
    last1_finite = np.isfinite(paired["last1_prediction"].to_numpy(float))
    if np.any(history & ~last1_finite):
        raise ValueError("Last-1 predictions are non-finite despite available history")
    return paired


def alpha_grid(step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("hybrid.alpha_grid_step must be in (0, 1]")
    values = np.arange(0.0, 1.0 + step * 0.5, step, dtype=float)
    values = np.clip(values, 0.0, 1.0)
    values = np.unique(np.append(values, 1.0))
    return values


def select_resource_alphas(
    validation: pd.DataFrame,
    grid: Iterable[float],
    clip_predictions: bool = True,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float | int]]]:
    """Select each alpha from validation labels only; test rows are rejected."""
    if set(validation["split"].astype(str)) - {"validation"}:
        raise ValueError("Alpha selection accepts validation rows only")
    grid_values = np.asarray(list(grid), dtype=float)
    if grid_values.size == 0 or np.any(~np.isfinite(grid_values)):
        raise ValueError("Alpha grid must contain finite values")
    if np.any((grid_values < 0.0) | (grid_values > 1.0)):
        raise ValueError("Alpha values must be within [0,1]")
    grid_values = np.unique(np.sort(grid_values))

    alphas: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, float | int]] = {}
    for resource in RESOURCES:
        part = validation.loc[
            (validation["resource"] == resource) & validation["valid"]
        ].copy()
        target = pd.to_numeric(part["target"], errors="coerce").to_numpy(float)
        qwen = pd.to_numeric(part["qwen_prediction"], errors="coerce").to_numpy(float)
        last1 = pd.to_numeric(part["last1_prediction"], errors="coerce").to_numpy(float)
        history = part["history_available"].to_numpy(bool)
        usable = np.isfinite(target) & np.isfinite(qwen) & (~history | np.isfinite(last1))
        target, qwen, last1, history = (
            target[usable],
            qwen[usable],
            last1[usable],
            history[usable],
        )
        if target.size == 0:
            raise ValueError(f"No valid validation labels are available for {resource}")

        losses = []
        for alpha in grid_values:
            mixed = np.where(history, alpha * last1 + (1.0 - alpha) * qwen, qwen)
            if clip_predictions:
                mixed = np.clip(mixed, 0.0, 1.0)
            losses.append(float(np.mean(np.abs(mixed - target))))
        best_index = int(np.argmin(np.asarray(losses)))
        best_alpha = float(grid_values[best_index])
        alphas[resource] = best_alpha
        diagnostics[resource] = {
            "validation_n": int(target.size),
            "validation_history_n": int(history.sum()),
            "validation_mae": float(losses[best_index]),
        }
    return alphas, diagnostics


def _metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> Dict[str, float | int]:
    error = prediction - target
    absolute = np.abs(error)
    mse = float(np.mean(np.square(error)))
    hits = int(np.sum(absolute <= tolerance))
    return {
        "n": int(target.size),
        "hits_at_tolerance": hits,
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "acc_at_tolerance": float(hits / target.size),
    }


def _blend_values(
    qwen: np.ndarray,
    last1: np.ndarray,
    history: np.ndarray,
    alpha: float,
    clip_predictions: bool,
) -> np.ndarray:
    mixed = qwen.copy()
    mixed[history] = (
        alpha * last1[history] + (1.0 - alpha) * qwen[history]
    )
    return np.clip(mixed, 0.0, 1.0) if clip_predictions else mixed


def select_safe_resource_alphas(
    validation: pd.DataFrame,
    grid: Iterable[float],
    *,
    clip_predictions: bool = True,
    tolerance: float = 0.10,
    max_acc_drop_vs_reference: float = 0.0,
    max_rmse_increase_vs_reference: float = 0.0,
    max_mae_increase_vs_reference: float = 0.0,
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Select the lowest-MAE alpha that is Pareto-safe on validation."""
    if set(validation["split"].astype(str)) - {"validation"}:
        raise ValueError("Safe alpha selection accepts validation rows only")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Safe alpha tolerance must be finite and positive")
    margins = {
        "max_acc_drop_vs_reference": max_acc_drop_vs_reference,
        "max_rmse_increase_vs_reference": max_rmse_increase_vs_reference,
        "max_mae_increase_vs_reference": max_mae_increase_vs_reference,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in margins.values()):
        raise ValueError("Safe alpha constraint margins must be finite and non-negative")

    grid_values = np.asarray(list(grid), dtype=float)
    if grid_values.size == 0 or np.any(~np.isfinite(grid_values)):
        raise ValueError("Safe alpha grid must contain finite values")
    if np.any((grid_values < 0.0) | (grid_values > 1.0)):
        raise ValueError("Safe alpha values must be within [0,1]")
    grid_values = np.unique(np.sort(grid_values))
    if not np.isclose(grid_values, 1.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Safe alpha grid must contain alpha=1 reference policy")

    alphas: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    sweep_rows: list[Dict[str, Any]] = []
    numerical_slack = 1e-12
    for resource in RESOURCES:
        part = validation.loc[
            (validation["resource"] == resource) & validation["valid"]
        ].copy()
        target = pd.to_numeric(part["target"], errors="coerce").to_numpy(float)
        qwen = pd.to_numeric(part["qwen_prediction"], errors="coerce").to_numpy(float)
        last1 = pd.to_numeric(part["last1_prediction"], errors="coerce").to_numpy(float)
        history = part["history_available"].to_numpy(bool)
        usable = np.isfinite(target) & np.isfinite(qwen) & (~history | np.isfinite(last1))
        target, qwen, last1, history = (
            target[usable],
            qwen[usable],
            last1[usable],
            history[usable],
        )
        if target.size == 0:
            raise ValueError(f"No valid validation labels are available for {resource}")

        reference_prediction = _blend_values(
            qwen, last1, history, 1.0, clip_predictions
        )
        reference = _metric_values(reference_prediction, target, tolerance)
        resource_rows: list[Dict[str, Any]] = []
        for alpha in grid_values:
            prediction = _blend_values(
                qwen, last1, history, float(alpha), clip_predictions
            )
            metrics = _metric_values(prediction, target, tolerance)
            acc_ok = bool(
                metrics["acc_at_tolerance"]
                >= reference["acc_at_tolerance"]
                - max_acc_drop_vs_reference
                - numerical_slack
            )
            rmse_ok = bool(
                metrics["rmse"]
                <= reference["rmse"]
                + max_rmse_increase_vs_reference
                + numerical_slack
            )
            mae_ok = bool(
                metrics["mae"]
                <= reference["mae"]
                + max_mae_increase_vs_reference
                + numerical_slack
            )
            row: Dict[str, Any] = {
                "resource": resource,
                "alpha": float(alpha),
                "history_n": int(history.sum()),
                **metrics,
                "mae_delta_vs_reference": float(metrics["mae"] - reference["mae"]),
                "rmse_delta_vs_reference": float(
                    metrics["rmse"] - reference["rmse"]
                ),
                "acc_delta_vs_reference": float(
                    metrics["acc_at_tolerance"]
                    - reference["acc_at_tolerance"]
                ),
                "acc_constraint_ok": acc_ok,
                "rmse_constraint_ok": rmse_ok,
                "mae_constraint_ok": mae_ok,
                "feasible": bool(acc_ok and rmse_ok and mae_ok),
                "selected": False,
            }
            resource_rows.append(row)

        feasible = [row for row in resource_rows if row["feasible"]]
        if not feasible:
            reference_rows = [
                row
                for row in resource_rows
                if np.isclose(row["alpha"], 1.0, rtol=0.0, atol=1e-12)
            ]
            if len(reference_rows) != 1:
                raise RuntimeError(f"No auditable alpha=1 row exists for {resource}")
            raise RuntimeError(
                f"Safe alpha constraints unexpectedly reject their own {resource} reference"
            )
        selected = min(
            feasible,
            key=lambda row: (
                row["mae"],
                -row["acc_at_tolerance"],
                row["rmse"],
                -row["alpha"],
            ),
        )
        selected["selected"] = True
        selected_alpha = float(selected["alpha"])
        alphas[resource] = selected_alpha
        diagnostics[resource] = {
            "validation_n": int(target.size),
            "validation_history_n": int(history.sum()),
            "reference_alpha": 1.0,
            "reference_metrics": reference,
            "selected_alpha": selected_alpha,
            "selected_metrics": {
                name: selected[name]
                for name in (
                    "n",
                    "hits_at_tolerance",
                    "mae",
                    "mse",
                    "rmse",
                    "acc_at_tolerance",
                )
            },
            "feasible_alpha_count": int(len(feasible)),
            "grid_alpha_count": int(len(resource_rows)),
            "selection_status": (
                "safe_reference_selected"
                if np.isclose(selected_alpha, 1.0, rtol=0.0, atol=1e-12)
                else "safe_blend_selected"
            ),
        }
        sweep_rows.extend(resource_rows)
    return alphas, diagnostics, pd.DataFrame(sweep_rows)


def _script_history_blend_values(
    qwen: np.ndarray,
    last1: np.ndarray,
    same_script: np.ndarray,
    history: np.ndarray,
    same_script_available: np.ndarray,
    *,
    alpha: float,
    beta: float,
    clip_predictions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    history_expert = last1.copy()
    history_expert[same_script_available] = (
        (1.0 - beta) * last1[same_script_available]
        + beta * same_script[same_script_available]
    )
    prediction = qwen.copy()
    prediction[history] = (
        alpha * history_expert[history]
        + (1.0 - alpha) * qwen[history]
    )
    if clip_predictions:
        prediction = np.clip(prediction, 0.0, 1.0)
        history_expert = np.clip(history_expert, 0.0, 1.0)
    return prediction, history_expert


def select_safe_script_history_policies(
    validation: pd.DataFrame,
    alpha_values: Iterable[float],
    beta_values: Iterable[float],
    *,
    clip_predictions: bool = True,
    tolerance: float = 0.10,
    max_acc_drop_vs_reference: float = 0.0,
    max_rmse_increase_vs_reference: float = 0.0,
    max_mae_increase_vs_reference: float = 0.0,
    min_effective_same_script_n: int = 100,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Select a safe Qwen/general-history/same-script-history blend."""
    if set(validation["split"].astype(str)) - {"validation"}:
        raise ValueError("Script-history policy selection accepts validation rows only")
    required = {
        "same_script_history_available",
        "same_script_history_prediction",
        "same_script_as_last1",
    }
    missing = sorted(required.difference(validation.columns))
    if missing:
        raise ValueError(f"Script-history selection is missing columns: {missing}")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Script-history tolerance must be finite and positive")
    if (
        isinstance(min_effective_same_script_n, bool)
        or int(min_effective_same_script_n) != min_effective_same_script_n
        or int(min_effective_same_script_n) < 0
    ):
        raise ValueError("min_effective_same_script_n must be a non-negative integer")
    min_effective_same_script_n = int(min_effective_same_script_n)

    margins = {
        "max_acc_drop_vs_reference": max_acc_drop_vs_reference,
        "max_rmse_increase_vs_reference": max_rmse_increase_vs_reference,
        "max_mae_increase_vs_reference": max_mae_increase_vs_reference,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in margins.values()):
        raise ValueError("Script-history margins must be finite and non-negative")

    alphas = np.asarray(list(alpha_values), dtype=float)
    betas = np.asarray(list(beta_values), dtype=float)
    if alphas.size == 0 or np.any(~np.isfinite(alphas)):
        raise ValueError("Script-history alpha grid must contain finite values")
    if betas.size == 0 or np.any(~np.isfinite(betas)):
        raise ValueError("Script-history beta grid must contain finite values")
    if np.any((alphas < 0.0) | (alphas > 1.0)):
        raise ValueError("Script-history alphas must be within [0,1]")
    if np.any((betas < 0.0) | (betas > 1.0)):
        raise ValueError("Script-history betas must be within [0,1]")
    alphas = np.unique(np.sort(alphas))
    betas = np.unique(np.sort(betas))
    if not np.isclose(alphas, 1.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Script-history alpha grid must contain alpha=1")
    if not np.isclose(betas, 0.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Script-history beta grid must contain beta=0")
    if not np.isclose(betas, 1.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Script-history beta grid must contain beta=1")

    policies: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    sweep_rows: list[Dict[str, Any]] = []
    numerical_slack = 1e-12
    for resource in RESOURCES:
        part = validation.loc[
            (validation["resource"] == resource) & validation["valid"]
        ].copy()
        target = pd.to_numeric(part["target"], errors="coerce").to_numpy(float)
        qwen = pd.to_numeric(part["qwen_prediction"], errors="coerce").to_numpy(float)
        last1 = pd.to_numeric(part["last1_prediction"], errors="coerce").to_numpy(float)
        same_script = pd.to_numeric(
            part["same_script_history_prediction"], errors="coerce"
        ).to_numpy(float)
        history = _boolean_series(
            part["history_available"], "history_available"
        ).to_numpy(bool)
        same_available = _boolean_series(
            part["same_script_history_available"],
            "same_script_history_available",
        ).to_numpy(bool)
        same_as_last1 = _boolean_series(
            part["same_script_as_last1"], "same_script_as_last1"
        ).to_numpy(bool)
        if np.any(same_available & ~history):
            raise ValueError("Same-script history cannot exist without general history")
        if np.any(same_as_last1 & ~same_available):
            raise ValueError("same_script_as_last1 requires same-script history")
        if target.size == 0:
            raise ValueError(f"No valid validation labels are available for {resource}")
        if not np.isfinite(target).all():
            raise ValueError(f"Valid {resource} validation targets are non-finite")
        if not np.isfinite(qwen).all():
            raise ValueError(f"{resource} Qwen validation predictions are non-finite")
        if np.any(history & ~np.isfinite(last1)):
            raise ValueError(
                f"{resource} Last-1 validation predictions are non-finite when available"
            )
        if np.any(same_available & ~np.isfinite(same_script)):
            raise ValueError(
                f"{resource} same-script validation predictions are non-finite when available"
            )

        effective_same = same_available & ~same_as_last1
        effective_same_n = int(effective_same.sum())
        beta_selection_active = effective_same_n >= min_effective_same_script_n
        resource_betas = betas if beta_selection_active else np.asarray([0.0])
        ordinary_reference_prediction, _ = _script_history_blend_values(
            qwen,
            last1,
            same_script,
            history,
            same_available,
            alpha=1.0,
            beta=0.0,
            clip_predictions=clip_predictions,
        )
        ordinary_reference = _metric_values(
            ordinary_reference_prediction, target, tolerance
        )
        reference_beta = 1.0 if beta_selection_active else 0.0
        reference_prediction, _ = _script_history_blend_values(
            qwen,
            last1,
            same_script,
            history,
            same_available,
            alpha=1.0,
            beta=reference_beta,
            clip_predictions=clip_predictions,
        )
        reference = _metric_values(reference_prediction, target, tolerance)
        resource_rows: list[Dict[str, Any]] = []
        for alpha in alphas:
            for beta in resource_betas:
                prediction, history_expert = _script_history_blend_values(
                    qwen,
                    last1,
                    same_script,
                    history,
                    same_available,
                    alpha=float(alpha),
                    beta=float(beta),
                    clip_predictions=clip_predictions,
                )
                metrics = _metric_values(prediction, target, tolerance)
                acc_ok = bool(
                    metrics["acc_at_tolerance"]
                    >= reference["acc_at_tolerance"]
                    - max_acc_drop_vs_reference
                    - numerical_slack
                )
                rmse_ok = bool(
                    metrics["rmse"]
                    <= reference["rmse"]
                    + max_rmse_increase_vs_reference
                    + numerical_slack
                )
                mae_ok = bool(
                    metrics["mae"]
                    <= reference["mae"]
                    + max_mae_increase_vs_reference
                    + numerical_slack
                )
                same_adjustment = np.abs(history_expert - last1)
                row: Dict[str, Any] = {
                    "resource": resource,
                    "alpha": float(alpha),
                    "same_script_beta": float(beta),
                    "history_n": int(history.sum()),
                    "same_script_history_n": int(same_available.sum()),
                    "effective_same_script_history_n": effective_same_n,
                    "beta_selection_active": bool(beta_selection_active),
                    "reference_same_script_beta": reference_beta,
                    **metrics,
                    "same_script_adjusted_n": int(
                        np.sum(same_adjustment > 1e-12)
                    ),
                    "mean_abs_same_script_adjustment": float(
                        np.mean(same_adjustment[same_available])
                        if same_available.any()
                        else 0.0
                    ),
                    "mae_delta_vs_reference": float(metrics["mae"] - reference["mae"]),
                    "rmse_delta_vs_reference": float(
                        metrics["rmse"] - reference["rmse"]
                    ),
                    "acc_delta_vs_reference": float(
                        metrics["acc_at_tolerance"]
                        - reference["acc_at_tolerance"]
                    ),
                    "mae_delta_vs_ordinary_reference": float(
                        metrics["mae"] - ordinary_reference["mae"]
                    ),
                    "rmse_delta_vs_ordinary_reference": float(
                        metrics["rmse"] - ordinary_reference["rmse"]
                    ),
                    "acc_delta_vs_ordinary_reference": float(
                        metrics["acc_at_tolerance"]
                        - ordinary_reference["acc_at_tolerance"]
                    ),
                    "acc_constraint_ok": acc_ok,
                    "rmse_constraint_ok": rmse_ok,
                    "mae_constraint_ok": mae_ok,
                    "feasible": bool(acc_ok and rmse_ok and mae_ok),
                    "selected": False,
                }
                resource_rows.append(row)

        feasible = [row for row in resource_rows if row["feasible"]]
        if not feasible:
            raise RuntimeError(
                f"Script-history constraints unexpectedly reject their own {resource} reference"
            )
        selected = min(
            feasible,
            key=lambda row: (
                row["mae"],
                -row["acc_at_tolerance"],
                row["rmse"],
                row["same_script_beta"],
                -row["alpha"],
            ),
        )
        selected["selected"] = True
        policy = {
            "alpha": float(selected["alpha"]),
            "same_script_beta": float(selected["same_script_beta"]),
        }
        policies[resource] = policy
        diagnostics[resource] = {
            "validation_n": int(target.size),
            "validation_history_n": int(history.sum()),
            "validation_same_script_history_n": int(same_available.sum()),
            "validation_effective_same_script_history_n": effective_same_n,
            "min_effective_same_script_n": min_effective_same_script_n,
            "beta_selection_active": bool(beta_selection_active),
            "ordinary_reference_policy": {
                "alpha": 1.0,
                "same_script_beta": 0.0,
            },
            "ordinary_reference_metrics": ordinary_reference,
            "reference_policy": {
                "alpha": 1.0,
                "same_script_beta": reference_beta,
            },
            "reference_metrics": reference,
            "selected_policy": policy,
            "selected_metrics": {
                name: selected[name]
                for name in (
                    "n",
                    "hits_at_tolerance",
                    "mae",
                    "mse",
                    "rmse",
                    "acc_at_tolerance",
                    "same_script_adjusted_n",
                    "mean_abs_same_script_adjustment",
                )
            },
            "feasible_policy_count": int(len(feasible)),
            "candidate_policy_count": int(len(resource_rows)),
            "selection_status": (
                "safe_reference_selected"
                if np.isclose(policy["alpha"], 1.0, rtol=0.0, atol=1e-12)
                and np.isclose(
                    policy["same_script_beta"],
                    reference_beta,
                    rtol=0.0,
                    atol=1e-12,
                )
                else "safe_script_history_blend_selected"
            ),
        }
        sweep_rows.extend(resource_rows)
    return policies, diagnostics, pd.DataFrame(sweep_rows)


def apply_script_history_policies(
    paired: pd.DataFrame,
    policies: Mapping[str, Mapping[str, float]],
    *,
    clip_predictions: bool = True,
) -> pd.DataFrame:
    """Apply frozen three-expert policies without reading target columns."""
    required = {
        "same_script_history_available",
        "same_script_history_prediction",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"Script-history application is missing columns: {missing}")
    if set(policies) != set(RESOURCES):
        raise ValueError("Script-history policies must contain memory and vram")
    for resource, policy in policies.items():
        alpha = float(policy["alpha"])
        beta = float(policy["same_script_beta"])
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Script-history {resource} alpha must be within [0,1]")
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError(f"Script-history {resource} beta must be within [0,1]")

    output = paired.copy()
    alpha_values = output["resource"].map(
        {resource: float(policy["alpha"]) for resource, policy in policies.items()}
    )
    beta_values = output["resource"].map(
        {
            resource: float(policy["same_script_beta"])
            for resource, policy in policies.items()
        }
    )
    if alpha_values.isna().any() or beta_values.isna().any():
        raise ValueError("No frozen script-history policy is available for every resource")

    qwen = output["qwen_prediction"].to_numpy(float)
    last1 = output["last1_prediction"].to_numpy(float)
    same_script = pd.to_numeric(
        output["same_script_history_prediction"], errors="coerce"
    ).to_numpy(float)
    history = _boolean_series(
        output["history_available"], "history_available"
    ).to_numpy(bool)
    same_available = _boolean_series(
        output["same_script_history_available"],
        "same_script_history_available",
    ).to_numpy(bool)
    if np.any(same_available & ~history):
        raise ValueError("Same-script history cannot exist without general history")
    if np.any(history & ~np.isfinite(last1)):
        raise ValueError("General history prediction is non-finite when available")
    if np.any(same_available & ~np.isfinite(same_script)):
        raise ValueError("Same-script history prediction is non-finite when available")

    alpha = alpha_values.to_numpy(float)
    beta = beta_values.to_numpy(float)
    history_expert = last1.copy()
    history_expert[same_available] = (
        (1.0 - beta[same_available]) * last1[same_available]
        + beta[same_available] * same_script[same_available]
    )
    raw = qwen.copy()
    raw[history] = (
        alpha[history] * history_expert[history]
        + (1.0 - alpha[history]) * qwen[history]
    )
    prediction = np.clip(raw, 0.0, 1.0) if clip_predictions else raw

    output["selected_alpha"] = alpha
    output["alpha"] = np.where(history, alpha, 0.0)
    output["selected_same_script_beta"] = beta
    output["same_script_beta"] = np.where(same_available, beta, 0.0)
    output["history_expert_prediction"] = history_expert
    output["same_script_history_adjustment"] = np.where(
        same_available, history_expert - last1, 0.0
    )
    output["raw_prediction"] = raw
    output["prediction"] = prediction
    output["used_qwen_only"] = ~history
    if not np.allclose(
        prediction[~history],
        np.clip(qwen[~history], 0.0, 1.0) if clip_predictions else qwen[~history],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Script-history cold-start rows did not use Qwen")
    return output


def guard_budget_grid(
    tolerance: float,
    fractions: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> np.ndarray:
    """Build a deterministic adjustment-budget grid scaled by Acc tolerance."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Guard tolerance must be finite and positive")
    values = np.asarray(list(fractions), dtype=float)
    if (
        values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("Guard budget fractions must be finite and within [0,1]")
    budgets = np.unique(np.clip(values * tolerance, 0.0, 1.0))
    if not np.isclose(budgets, 0.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Guard budget grid must include a zero-adjustment policy")
    return budgets


def _guarded_values(
    reference: np.ndarray,
    candidate: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = candidate - reference
    guarded_delta = np.clip(delta, -budget, budget)
    return reference + guarded_delta, np.abs(delta) > budget + 1e-12


def select_guarded_resource_policies(
    validation: pd.DataFrame,
    alpha_values: Iterable[float] | Mapping[str, float],
    guard_budgets: Iterable[float],
    *,
    clip_predictions: bool = True,
    tolerance: float = 0.10,
    max_acc_drop_vs_reference: float = 0.0,
    max_rmse_increase_vs_reference: float = 0.0,
    max_mae_increase_vs_reference: float = 0.0,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Select a validation-safe blend whose per-row adjustment is bounded."""
    if not clip_predictions:
        raise ValueError("Guarded hybrid requires evaluation.clip_predictions=true")
    if set(validation["split"].astype(str)) - {"validation"}:
        raise ValueError("Guarded policy selection accepts validation rows only")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Guarded policy tolerance must be finite and positive")
    margins = {
        "max_acc_drop_vs_reference": max_acc_drop_vs_reference,
        "max_rmse_increase_vs_reference": max_rmse_increase_vs_reference,
        "max_mae_increase_vs_reference": max_mae_increase_vs_reference,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in margins.values()):
        raise ValueError("Guarded policy margins must be finite and non-negative")

    if isinstance(alpha_values, Mapping):
        if set(alpha_values) != set(RESOURCES):
            raise ValueError("Guarded policies require fixed memory and vram alphas")
        resource_alphas = {
            resource: np.asarray([float(alpha_values[resource])], dtype=float)
            for resource in RESOURCES
        }
    else:
        shared_alphas = np.asarray(list(alpha_values), dtype=float)
        resource_alphas = {resource: shared_alphas for resource in RESOURCES}
    budgets = np.asarray(list(guard_budgets), dtype=float)
    for values in resource_alphas.values():
        if values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError("Guarded policy alpha grid must contain finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Guarded policy alphas must be within [0,1]")
    if budgets.size == 0 or np.any(~np.isfinite(budgets)) or np.any(budgets < 0.0):
        raise ValueError("Guard budgets must be finite and non-negative")
    resource_alphas = {
        resource: np.unique(np.sort(values))
        for resource, values in resource_alphas.items()
    }
    budgets = np.unique(np.sort(budgets))
    if not np.isclose(budgets, 0.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Guarded policy budget grid must contain zero")

    policies: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    sweep_rows: list[Dict[str, Any]] = []
    numerical_slack = 1e-12
    for resource in RESOURCES:
        part = validation.loc[
            (validation["resource"] == resource) & validation["valid"]
        ].copy()
        target = pd.to_numeric(part["target"], errors="coerce").to_numpy(float)
        qwen = pd.to_numeric(part["qwen_prediction"], errors="coerce").to_numpy(float)
        last1 = pd.to_numeric(part["last1_prediction"], errors="coerce").to_numpy(float)
        history = part["history_available"].to_numpy(bool)
        usable = np.isfinite(target) & np.isfinite(qwen) & (~history | np.isfinite(last1))
        target, qwen, last1, history = (
            target[usable],
            qwen[usable],
            last1[usable],
            history[usable],
        )
        if target.size == 0:
            raise ValueError(f"No valid validation labels are available for {resource}")

        reference_prediction = _blend_values(
            qwen, last1, history, 1.0, clip_predictions=True
        )
        reference = _metric_values(reference_prediction, target, tolerance)
        resource_rows: list[Dict[str, Any]] = []
        for alpha in resource_alphas[resource]:
            candidate = _blend_values(
                qwen, last1, history, float(alpha), clip_predictions=True
            )
            for budget in budgets:
                prediction, guard_active = _guarded_values(
                    reference_prediction, candidate, float(budget)
                )
                metrics = _metric_values(prediction, target, tolerance)
                adjustment = np.abs(prediction - reference_prediction)
                acc_ok = bool(
                    metrics["acc_at_tolerance"]
                    >= reference["acc_at_tolerance"]
                    - max_acc_drop_vs_reference
                    - numerical_slack
                )
                rmse_ok = bool(
                    metrics["rmse"]
                    <= reference["rmse"]
                    + max_rmse_increase_vs_reference
                    + numerical_slack
                )
                mae_ok = bool(
                    metrics["mae"]
                    <= reference["mae"]
                    + max_mae_increase_vs_reference
                    + numerical_slack
                )
                row: Dict[str, Any] = {
                    "resource": resource,
                    "alpha": float(alpha),
                    "guard_budget": float(budget),
                    "guard_budget_fraction": float(budget / tolerance),
                    "history_n": int(history.sum()),
                    **metrics,
                    "guard_active_n": int(guard_active.sum()),
                    "changed_n": int(np.sum(adjustment > 1e-12)),
                    "mean_abs_adjustment": float(np.mean(adjustment)),
                    "max_abs_adjustment": float(np.max(adjustment)),
                    "mae_delta_vs_reference": float(
                        metrics["mae"] - reference["mae"]
                    ),
                    "rmse_delta_vs_reference": float(
                        metrics["rmse"] - reference["rmse"]
                    ),
                    "acc_delta_vs_reference": float(
                        metrics["acc_at_tolerance"]
                        - reference["acc_at_tolerance"]
                    ),
                    "acc_constraint_ok": acc_ok,
                    "rmse_constraint_ok": rmse_ok,
                    "mae_constraint_ok": mae_ok,
                    "feasible": bool(acc_ok and rmse_ok and mae_ok),
                    "selected": False,
                }
                resource_rows.append(row)

        feasible = [row for row in resource_rows if row["feasible"]]
        if not feasible:
            raise RuntimeError(
                f"Guarded constraints unexpectedly reject their own {resource} reference"
            )
        selected = min(
            feasible,
            key=lambda row: (
                -row["acc_at_tolerance"],
                row["mae"],
                row["rmse"],
                row["guard_budget"],
                -row["alpha"],
            ),
        )
        selected["selected"] = True
        policy = {
            "alpha": float(selected["alpha"]),
            "guard_budget": float(selected["guard_budget"]),
            "guard_budget_fraction": float(selected["guard_budget_fraction"]),
        }
        policies[resource] = policy
        diagnostics[resource] = {
            "validation_n": int(target.size),
            "validation_history_n": int(history.sum()),
            "reference_metrics": reference,
            "selected_policy": policy,
            "selected_metrics": {
                name: selected[name]
                for name in (
                    "n",
                    "hits_at_tolerance",
                    "mae",
                    "mse",
                    "rmse",
                    "acc_at_tolerance",
                    "guard_active_n",
                    "changed_n",
                    "mean_abs_adjustment",
                    "max_abs_adjustment",
                )
            },
            "feasible_policy_count": int(len(feasible)),
            "candidate_policy_count": int(len(resource_rows)),
            "selection_status": (
                "safe_reference_selected"
                if selected["changed_n"] == 0
                else "guarded_blend_selected"
            ),
        }
        sweep_rows.extend(resource_rows)
    return policies, diagnostics, pd.DataFrame(sweep_rows)


def select_script_guarded_resource_policies(
    validation: pd.DataFrame,
    alpha_values: Mapping[str, float],
    guard_budgets: Iterable[float],
    *,
    clip_predictions: bool = True,
    tolerance: float = 0.10,
    max_acc_drop_vs_reference: float = 0.0,
    max_rmse_increase_vs_reference: float = 0.0,
    max_mae_increase_vs_reference: float = 0.0,
    min_stratum_n: int = 100,
    require_changed_budget_ge_same: bool = True,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Select separate trust budgets for unchanged and changed-script history."""
    if not clip_predictions:
        raise ValueError("Script-guarded hybrid requires evaluation.clip_predictions=true")
    if set(validation["split"].astype(str)) - {"validation"}:
        raise ValueError("Script-guarded policy selection accepts validation rows only")
    if "same_script_as_last1" not in validation:
        raise ValueError("Script-guarded selection requires same_script_as_last1")
    if set(alpha_values) != set(RESOURCES):
        raise ValueError("Script-guarded policies require fixed memory and vram alphas")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Script-guarded tolerance must be finite and positive")
    if isinstance(min_stratum_n, bool) or int(min_stratum_n) != min_stratum_n:
        raise ValueError("min_stratum_n must be a non-negative integer")
    min_stratum_n = int(min_stratum_n)
    if min_stratum_n < 0:
        raise ValueError("min_stratum_n must be a non-negative integer")

    margins = {
        "max_acc_drop_vs_reference": max_acc_drop_vs_reference,
        "max_rmse_increase_vs_reference": max_rmse_increase_vs_reference,
        "max_mae_increase_vs_reference": max_mae_increase_vs_reference,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in margins.values()):
        raise ValueError("Script-guarded margins must be finite and non-negative")

    frozen_alphas = {resource: float(alpha_values[resource]) for resource in RESOURCES}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in frozen_alphas.values()):
        raise ValueError("Script-guarded alphas must be finite and within [0,1]")
    budgets = np.asarray(list(guard_budgets), dtype=float)
    if budgets.size == 0 or np.any(~np.isfinite(budgets)) or np.any(budgets < 0.0):
        raise ValueError("Script-guarded budgets must be finite and non-negative")
    budgets = np.unique(np.sort(budgets))
    if not np.isclose(budgets, 0.0, rtol=0.0, atol=1e-12).any():
        raise ValueError("Script-guarded budget grid must contain zero")

    policies: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    sweep_rows: list[Dict[str, Any]] = []
    numerical_slack = 1e-12
    for resource in RESOURCES:
        part = validation.loc[
            (validation["resource"] == resource) & validation["valid"]
        ].copy()
        target = pd.to_numeric(part["target"], errors="coerce").to_numpy(float)
        qwen = pd.to_numeric(part["qwen_prediction"], errors="coerce").to_numpy(float)
        last1 = pd.to_numeric(part["last1_prediction"], errors="coerce").to_numpy(float)
        history = _boolean_series(
            part["history_available"], "history_available"
        ).to_numpy(bool)
        same_script = _boolean_series(
            part["same_script_as_last1"], "same_script_as_last1"
        ).to_numpy(bool)
        if np.any(same_script & ~history):
            raise ValueError("same_script_as_last1 cannot be true without causal history")
        usable = np.isfinite(target) & np.isfinite(qwen) & (~history | np.isfinite(last1))
        target, qwen, last1, history, same_script = (
            target[usable],
            qwen[usable],
            last1[usable],
            history[usable],
            same_script[usable],
        )
        if target.size == 0:
            raise ValueError(f"No valid validation labels are available for {resource}")

        changed_script = history & ~same_script
        same_n = int(same_script.sum())
        changed_n = int(changed_script.sum())
        stratified = same_n >= min_stratum_n and changed_n >= min_stratum_n
        alpha = frozen_alphas[resource]
        reference_prediction = _blend_values(
            qwen, last1, history, 1.0, clip_predictions=True
        )
        candidate = _blend_values(
            qwen, last1, history, alpha, clip_predictions=True
        )
        reference = _metric_values(reference_prediction, target, tolerance)
        resource_rows: list[Dict[str, Any]] = []
        for same_budget in budgets:
            for changed_budget in budgets:
                if not stratified and not np.isclose(
                    same_budget, changed_budget, rtol=0.0, atol=1e-12
                ):
                    continue
                if (
                    require_changed_budget_ge_same
                    and changed_budget + numerical_slack < same_budget
                ):
                    continue
                assigned_budget = np.zeros(target.size, dtype=float)
                assigned_budget[same_script] = float(same_budget)
                assigned_budget[changed_script] = float(changed_budget)
                prediction, guard_active = _guarded_values(
                    reference_prediction, candidate, assigned_budget
                )
                prediction = np.clip(prediction, 0.0, 1.0)
                metrics = _metric_values(prediction, target, tolerance)
                adjustment = np.abs(prediction - reference_prediction)
                acc_ok = bool(
                    metrics["acc_at_tolerance"]
                    >= reference["acc_at_tolerance"]
                    - max_acc_drop_vs_reference
                    - numerical_slack
                )
                rmse_ok = bool(
                    metrics["rmse"]
                    <= reference["rmse"]
                    + max_rmse_increase_vs_reference
                    + numerical_slack
                )
                mae_ok = bool(
                    metrics["mae"]
                    <= reference["mae"]
                    + max_mae_increase_vs_reference
                    + numerical_slack
                )
                history_budget = assigned_budget[history]
                row: Dict[str, Any] = {
                    "resource": resource,
                    "alpha": alpha,
                    "same_script_guard_budget": float(same_budget),
                    "changed_script_guard_budget": float(changed_budget),
                    "same_script_guard_budget_fraction": float(same_budget / tolerance),
                    "changed_script_guard_budget_fraction": float(
                        changed_budget / tolerance
                    ),
                    "history_n": int(history.sum()),
                    "same_script_history_n": same_n,
                    "changed_script_history_n": changed_n,
                    "stratified_selection_active": bool(stratified),
                    **metrics,
                    "guard_active_n": int(guard_active.sum()),
                    "changed_n": int(np.sum(adjustment > 1e-12)),
                    "mean_abs_adjustment": float(np.mean(adjustment)),
                    "max_abs_adjustment": float(np.max(adjustment)),
                    "mean_assigned_history_budget": float(
                        np.mean(history_budget) if history_budget.size else 0.0
                    ),
                    "mae_delta_vs_reference": float(metrics["mae"] - reference["mae"]),
                    "rmse_delta_vs_reference": float(
                        metrics["rmse"] - reference["rmse"]
                    ),
                    "acc_delta_vs_reference": float(
                        metrics["acc_at_tolerance"]
                        - reference["acc_at_tolerance"]
                    ),
                    "acc_constraint_ok": acc_ok,
                    "rmse_constraint_ok": rmse_ok,
                    "mae_constraint_ok": mae_ok,
                    "feasible": bool(acc_ok and rmse_ok and mae_ok),
                    "selected": False,
                }
                resource_rows.append(row)

        feasible = [row for row in resource_rows if row["feasible"]]
        if not feasible:
            raise RuntimeError(
                f"Script-guarded constraints unexpectedly reject their own {resource} reference"
            )
        selected = min(
            feasible,
            key=lambda row: (
                -row["acc_at_tolerance"],
                row["mae"],
                row["rmse"],
                row["mean_assigned_history_budget"],
                max(
                    row["same_script_guard_budget"],
                    row["changed_script_guard_budget"],
                ),
                row["same_script_guard_budget"]
                + row["changed_script_guard_budget"],
                -row["alpha"],
            ),
        )
        selected["selected"] = True
        policy = {
            "alpha": alpha,
            "same_script_guard_budget": float(
                selected["same_script_guard_budget"]
            ),
            "changed_script_guard_budget": float(
                selected["changed_script_guard_budget"]
            ),
            "same_script_guard_budget_fraction": float(
                selected["same_script_guard_budget_fraction"]
            ),
            "changed_script_guard_budget_fraction": float(
                selected["changed_script_guard_budget_fraction"]
            ),
        }
        policies[resource] = policy
        diagnostics[resource] = {
            "validation_n": int(target.size),
            "validation_history_n": int(history.sum()),
            "validation_same_script_history_n": same_n,
            "validation_changed_script_history_n": changed_n,
            "min_stratum_n": min_stratum_n,
            "stratified_selection_active": bool(stratified),
            "require_changed_budget_ge_same": bool(
                require_changed_budget_ge_same
            ),
            "reference_metrics": reference,
            "selected_policy": policy,
            "selected_metrics": {
                name: selected[name]
                for name in (
                    "n",
                    "hits_at_tolerance",
                    "mae",
                    "mse",
                    "rmse",
                    "acc_at_tolerance",
                    "guard_active_n",
                    "changed_n",
                    "mean_abs_adjustment",
                    "max_abs_adjustment",
                    "mean_assigned_history_budget",
                )
            },
            "feasible_policy_count": int(len(feasible)),
            "candidate_policy_count": int(len(resource_rows)),
            "selection_status": (
                "safe_reference_selected"
                if selected["changed_n"] == 0
                else "script_stratified_guard_selected"
                if not np.isclose(
                    selected["same_script_guard_budget"],
                    selected["changed_script_guard_budget"],
                    rtol=0.0,
                    atol=1e-12,
                )
                else "global_guard_selected"
            ),
        }
        sweep_rows.extend(resource_rows)
    return policies, diagnostics, pd.DataFrame(sweep_rows)


def apply_guarded_policies(
    paired: pd.DataFrame,
    policies: Mapping[str, Mapping[str, float]],
    *,
    clip_predictions: bool = True,
) -> pd.DataFrame:
    """Apply frozen per-resource alpha and adjustment budgets without labels."""
    if not clip_predictions:
        raise ValueError("Guarded hybrid requires evaluation.clip_predictions=true")
    if set(policies) != set(RESOURCES):
        raise ValueError("Guarded policies must contain memory and vram")
    for resource, policy in policies.items():
        alpha = float(policy["alpha"])
        budget = float(policy["guard_budget"])
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Guarded {resource} alpha must be within [0,1]")
        if not np.isfinite(budget) or budget < 0.0:
            raise ValueError(f"Guarded {resource} budget must be finite and non-negative")
    output = paired.copy()
    selected_alpha = output["resource"].map(
        {resource: float(policy["alpha"]) for resource, policy in policies.items()}
    )
    guard_budget = output["resource"].map(
        {
            resource: float(policy["guard_budget"])
            for resource, policy in policies.items()
        }
    )
    if selected_alpha.isna().any() or guard_budget.isna().any():
        raise ValueError("No frozen guarded policy is available for every resource")

    qwen = output["qwen_prediction"].to_numpy(float)
    last1 = output["last1_prediction"].to_numpy(float)
    history = output["history_available"].to_numpy(bool)
    alpha_values = selected_alpha.to_numpy(float)
    budget_values = guard_budget.to_numpy(float)
    reference = qwen.copy()
    reference[history] = last1[history]
    reference = np.clip(reference, 0.0, 1.0)
    candidate = qwen.copy()
    candidate[history] = (
        alpha_values[history] * last1[history]
        + (1.0 - alpha_values[history]) * qwen[history]
    )
    candidate = np.clip(candidate, 0.0, 1.0)
    delta = candidate - reference
    guarded_delta = np.clip(delta, -budget_values, budget_values)
    prediction = np.clip(reference + guarded_delta, 0.0, 1.0)
    scale = np.divide(
        guarded_delta,
        delta,
        out=np.zeros_like(guarded_delta),
        where=np.abs(delta) > 1e-12,
    )
    effective_alpha = 1.0 - scale * (1.0 - alpha_values)
    effective_alpha = np.where(history, effective_alpha, 0.0)

    output["selected_alpha"] = alpha_values
    output["alpha"] = effective_alpha
    output["guard_budget"] = budget_values
    output["safe_reference_prediction"] = reference
    output["uncapped_blend_prediction"] = candidate
    output["candidate_delta"] = candidate - reference
    output["guarded_adjustment"] = prediction - reference
    output["guard_was_active"] = history & (
        np.abs(delta) > budget_values + 1e-12
    )
    output["raw_prediction"] = prediction
    output["prediction"] = prediction
    output["used_qwen_only"] = ~history
    if np.any(np.abs(prediction - reference) > budget_values + 1e-12):
        raise AssertionError("Guarded prediction exceeded its trust-region budget")
    if not np.allclose(
        prediction[~history],
        np.clip(qwen[~history], 0.0, 1.0),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Guarded cold-start rows did not use Qwen")
    return output


def apply_script_guarded_policies(
    paired: pd.DataFrame,
    policies: Mapping[str, Mapping[str, float]],
    *,
    clip_predictions: bool = True,
) -> pd.DataFrame:
    """Apply frozen script-stratified trust budgets without reading labels."""
    if not clip_predictions:
        raise ValueError("Script-guarded hybrid requires evaluation.clip_predictions=true")
    if "same_script_as_last1" not in paired:
        raise ValueError("Script-guarded application requires same_script_as_last1")
    if set(policies) != set(RESOURCES):
        raise ValueError("Script-guarded policies must contain memory and vram")
    for resource, policy in policies.items():
        alpha = float(policy["alpha"])
        same_budget = float(policy["same_script_guard_budget"])
        changed_budget = float(policy["changed_script_guard_budget"])
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Script-guarded {resource} alpha must be within [0,1]")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (same_budget, changed_budget)
        ):
            raise ValueError(
                f"Script-guarded {resource} budgets must be finite and non-negative"
            )

    output = paired.copy()
    selected_alpha = output["resource"].map(
        {resource: float(policy["alpha"]) for resource, policy in policies.items()}
    )
    same_budget = output["resource"].map(
        {
            resource: float(policy["same_script_guard_budget"])
            for resource, policy in policies.items()
        }
    )
    changed_budget = output["resource"].map(
        {
            resource: float(policy["changed_script_guard_budget"])
            for resource, policy in policies.items()
        }
    )
    if selected_alpha.isna().any() or same_budget.isna().any() or changed_budget.isna().any():
        raise ValueError("No frozen script-guarded policy is available for every resource")

    qwen = output["qwen_prediction"].to_numpy(float)
    last1 = output["last1_prediction"].to_numpy(float)
    history = _boolean_series(
        output["history_available"], "history_available"
    ).to_numpy(bool)
    same_script = _boolean_series(
        output["same_script_as_last1"], "same_script_as_last1"
    ).to_numpy(bool)
    if np.any(same_script & ~history):
        raise ValueError("same_script_as_last1 cannot be true without causal history")
    alpha_values = selected_alpha.to_numpy(float)
    same_budget_values = same_budget.to_numpy(float)
    changed_budget_values = changed_budget.to_numpy(float)
    budget_values = np.where(same_script, same_budget_values, changed_budget_values)
    budget_values = np.where(history, budget_values, 0.0)

    reference = qwen.copy()
    reference[history] = last1[history]
    reference = np.clip(reference, 0.0, 1.0)
    candidate = qwen.copy()
    candidate[history] = (
        alpha_values[history] * last1[history]
        + (1.0 - alpha_values[history]) * qwen[history]
    )
    candidate = np.clip(candidate, 0.0, 1.0)
    delta = candidate - reference
    guarded_delta = np.clip(delta, -budget_values, budget_values)
    prediction = np.clip(reference + guarded_delta, 0.0, 1.0)
    scale = np.divide(
        guarded_delta,
        delta,
        out=np.zeros_like(guarded_delta),
        where=np.abs(delta) > 1e-12,
    )
    effective_alpha = np.where(history, 1.0 - scale * (1.0 - alpha_values), 0.0)

    output["selected_alpha"] = alpha_values
    output["alpha"] = effective_alpha
    output["same_script_guard_budget"] = same_budget_values
    output["changed_script_guard_budget"] = changed_budget_values
    output["guard_budget"] = budget_values
    output["safe_reference_prediction"] = reference
    output["uncapped_blend_prediction"] = candidate
    output["candidate_delta"] = delta
    output["guarded_adjustment"] = prediction - reference
    output["guard_was_active"] = history & (
        np.abs(delta) > budget_values + 1e-12
    )
    output["raw_prediction"] = prediction
    output["prediction"] = prediction
    output["used_qwen_only"] = ~history
    if np.any(np.abs(prediction - reference) > budget_values + 1e-12):
        raise AssertionError("Script-guarded prediction exceeded its row-level budget")
    if not np.allclose(
        prediction[~history],
        np.clip(qwen[~history], 0.0, 1.0),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Script-guarded cold-start rows did not use Qwen")
    return output


def apply_alphas(
    paired: pd.DataFrame,
    alphas: Mapping[str, float],
    clip_predictions: bool,
) -> pd.DataFrame:
    output = paired.copy()
    alpha_values = output["resource"].map(alphas)
    if alpha_values.isna().any():
        missing = sorted(output.loc[alpha_values.isna(), "resource"].unique().tolist())
        raise ValueError(f"No frozen alpha is available for resources: {missing}")
    qwen = output["qwen_prediction"].to_numpy(float)
    last1 = output["last1_prediction"].to_numpy(float)
    history = output["history_available"].to_numpy(bool)
    selected_alpha = alpha_values.to_numpy(float)
    effective_alpha = np.where(history, selected_alpha, 0.0)
    raw = np.where(
        history,
        selected_alpha * last1 + (1.0 - selected_alpha) * qwen,
        qwen,
    )
    output["selected_alpha"] = selected_alpha
    output["alpha"] = effective_alpha
    output["raw_prediction"] = raw
    output["prediction"] = np.clip(raw, 0.0, 1.0) if clip_predictions else raw
    output["used_qwen_only"] = ~history
    return output
