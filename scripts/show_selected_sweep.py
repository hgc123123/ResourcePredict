#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DISPLAY_COLUMNS = (
    "resource",
    "alpha",
    "same_script_beta",
    "reference_same_script_beta",
    "history_n",
    "same_script_history_n",
    "effective_same_script_history_n",
    "mae",
    "mse",
    "rmse",
    "hits_at_tolerance",
    "acc_at_tolerance",
    "mae_delta_vs_reference",
    "rmse_delta_vs_reference",
    "acc_delta_vs_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the validation-selected three-expert sweep rows"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Hybrid output directory containing script_history_sweep.csv",
    )
    return parser.parse_args()


def boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("The selected column contains invalid boolean values")
    return normalized.isin({"true", "1"})


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    sweep_path = run_dir / "script_history_sweep.csv"
    policy_path = run_dir / "script_history_policy.json"

    if not sweep_path.is_file():
        raise FileNotFoundError(f"Sweep file not found: {sweep_path}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    sweep = pd.read_csv(sweep_path)
    if "selected" not in sweep:
        raise ValueError(f"Sweep file has no selected column: {sweep_path}")
    selected = sweep.loc[boolean_mask(sweep["selected"])].copy()
    if set(selected.get("resource", [])) != {"memory", "vram"}:
        raise ValueError(
            "Expected exactly one selected Memory row and one selected VRAM row"
        )
    if selected["resource"].duplicated().any():
        raise ValueError("The sweep contains multiple selected rows for one resource")

    with policy_path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    columns = [name for name in DISPLAY_COLUMNS if name in selected.columns]
    print("Selected sweep rows:")
    print(selected[columns].sort_values("resource").to_string(index=False))

    print("\nFrozen policies:")
    print(json.dumps(policy.get("script_history_policies"), indent=2))

    print("\nSafe reference policies:")
    print(json.dumps(policy.get("safe_reference_policies"), indent=2))

    diagnostics = policy.get("validation_diagnostics") or {}
    print("\nValidation diagnostics:")
    for resource in ("memory", "vram"):
        details = diagnostics.get(resource) or {}
        print(
            f"{resource}: status={details.get('selection_status')}, "
            f"effective_same_script_n="
            f"{details.get('validation_effective_same_script_history_n')}, "
            f"beta_active={details.get('beta_selection_active')}"
        )


if __name__ == "__main__":
    main()
