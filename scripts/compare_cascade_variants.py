#!/usr/bin/env python3
"""Verify and compare five-seed Full and No-Code cascade artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_significance import (  # noqa: E402
    pair_prediction_frames,
    validate_artifact_identity,
)
from common.data import sha256_file  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
import reliability_cascade.integrity as cascade_integrity  # noqa: E402
from reliability_cascade.integrity import (  # noqa: E402
    canonical_sha256,
    implementation_contract,
)


FORMAL_SEEDS = (40, 41, 42, 43, 44)
VARIANTS = ("full", "no_code")
METRICS = ("mae", "mse", "rmse", "acc_at_10")
LOSS_METRICS = ("mae", "mse", "rmse")
OUTPUT_NAMES = (
    "cascade_variant_metrics.csv",
    "cascade_variant_summary.csv",
    "no_code_minus_full_by_seed.csv",
    "no_code_minus_full_summary.csv",
    "cascade_variant_routing.csv",
    "cascade_variant_routing_summary.csv",
    "comparison_manifest.json",
)
ORCHESTRATION_ONLY_FILES = {"scripts/run_reliability_cascade_matrix.py"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Full and No-Code reliability-cascade matrices and report "
            "paired seed-wise pipeline-level sensitivity results"
        )
    )
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--no-code-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=("validation", "test"),
        help="Repeat to compare multiple splits (default: test)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object: {path}")
    return value


def _split_fingerprint(manifest: dict[str, Any]) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError("Cascade manifest has no split_fingerprint")
    return str(value)


def _run_parent(root: Path, *, variant: str, max_tokens: int) -> Path:
    return root / "reliability_cascade" / f"tokens_{max_tokens}" / variant


def _run_dir(root: Path, *, variant: str, max_tokens: int, seed: int) -> Path:
    return _run_parent(root, variant=variant, max_tokens=max_tokens) / f"seed_{seed}"


def _validate_matrix_plan(
    root: Path,
    *,
    variant: str,
    seeds: Sequence[int],
    max_tokens: int,
) -> dict[str, Any]:
    plan_path = root / "cascade_experiment_plan.json"
    plan = _read_json(plan_path, f"{variant} cascade matrix plan")
    if plan.get("schema") != "reliability-cascade-matrix-plan/v1":
        raise ValueError(f"{variant} matrix has an unknown plan schema")
    if plan.get("variant") != variant:
        raise ValueError(f"{variant} matrix plan declares variant={plan.get('variant')!r}")
    if int(plan.get("max_tokens", -1)) != max_tokens:
        raise ValueError(f"{variant} matrix plan has a different context budget")
    plan_seeds = plan.get("seeds")
    if plan_seeds != list(seeds):
        raise ValueError(
            f"{variant} matrix plan seeds must be exactly {list(seeds)}, "
            f"observed={plan_seeds!r}"
        )
    recorded_implementation = plan.get("implementation")
    if not isinstance(recorded_implementation, dict):
        raise ValueError(f"{variant} matrix plan has no implementation contract")
    recorded_files = recorded_implementation.get("files")
    if not isinstance(recorded_files, dict) or not recorded_files:
        raise ValueError(f"{variant} matrix plan has no implementation file hashes")
    if recorded_implementation.get("sha256") != canonical_sha256(recorded_files):
        raise ValueError(f"{variant} matrix plan implementation hash is inconsistent")

    parent = _run_parent(root, variant=variant, max_tokens=max_tokens)
    if not parent.is_dir():
        raise FileNotFoundError(f"Missing {variant} cascade run directory: {parent}")
    observed = sorted(
        int(path.name.removeprefix("seed_"))
        for path in parent.glob("seed_*")
        if path.is_dir() and path.name.removeprefix("seed_").isdigit()
    )
    if observed != sorted(seeds):
        raise ValueError(
            f"{variant} matrix seed directories must be exactly {sorted(seeds)}, "
            f"observed={observed}"
        )
    return plan


def _implementation_differences(
    left: dict[str, Any], right: dict[str, Any]
) -> list[str]:
    left_files = left.get("files") or {}
    right_files = right.get("files") or {}
    if not isinstance(left_files, dict) or not isinstance(right_files, dict):
        raise ValueError("Matrix implementation contracts must contain file maps")
    names = sorted(set(left_files) | set(right_files))
    return [name for name in names if left_files.get(name) != right_files.get(name)]


def _verify_sealed_run(
    run_dir: Path,
    *,
    plan: dict[str, Any],
    seed: int,
    variant: str,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    """Verify all seals while honoring the implementation recorded at run time."""
    recorded = plan["implementation"]
    request_document = _read_json(run_dir / "run_request.json", "cascade run request")
    request = request_document.get("request") or {}
    if not isinstance(request, dict) or request.get("implementation") != recorded:
        raise ValueError(f"Run implementation does not match its matrix plan: {run_dir}")

    current = implementation_contract()
    mode = "current_source"
    if recorded != current:
        differences = _implementation_differences(recorded, current)
        unexpected = sorted(set(differences) - ORCHESTRATION_ONLY_FILES)
        if unexpected:
            raise ValueError(
                "Historical run differs from current predictive implementation files: "
                f"{unexpected}"
            )
        mode = "recorded_orchestration_contract"

    original = cascade_integrity.implementation_contract
    cascade_integrity.implementation_contract = lambda: recorded
    try:
        manifest = cascade_integrity.verify_run_directory(
            run_dir,
            expected_seed=seed,
            expected_model=f"reliability_cascade_{variant}",
            expected_variant=variant,
            expected_max_tokens=max_tokens,
        )
    finally:
        cascade_integrity.implementation_contract = original
    return manifest, mode


def _assert_saved_metrics_match(run_dir: Path, recomputed: pd.DataFrame) -> None:
    saved_path = run_dir / "metrics.csv"
    saved = pd.read_csv(saved_path)
    keys = ["split", "resource"]
    columns = ["n", *METRICS]
    left = recomputed[keys + columns].sort_values(keys).reset_index(drop=True)
    right = saved.loc[saved["split"].isin(left["split"].unique()), keys + columns]
    right = right.sort_values(keys).reset_index(drop=True)
    if left[keys].astype(str).to_dict("records") != right[keys].astype(str).to_dict("records"):
        raise ValueError(f"Saved metric cells differ from recomputation: {run_dir}")
    if not np.array_equal(left["n"].to_numpy(int), right["n"].to_numpy(int)):
        raise ValueError(f"Saved metric counts differ from recomputation: {run_dir}")
    if not np.allclose(
        left[list(METRICS)].to_numpy(float),
        right[list(METRICS)].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(f"Saved metrics differ from recomputation: {run_dir}")


def _variant_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, split, resource), part in metrics.groupby(
        ["variant", "split", "resource"], sort=True
    ):
        counts = part["n"].astype(int).unique()
        if len(counts) != 1:
            raise ValueError(f"Evaluation n changes across seeds for {variant}/{split}/{resource}")
        row: dict[str, Any] = {
            "variant": variant,
            "split": split,
            "resource": resource,
            "n": int(counts[0]),
            "seeds": int(len(part)),
        }
        for metric in METRICS:
            values = part[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["split", "resource", "variant"], kind="mergesort"
    ).reset_index(drop=True)


def _seed_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics.loc[metrics["variant"].eq("full")].drop(columns="variant")
    no_code = metrics.loc[metrics["variant"].eq("no_code")].drop(columns="variant")
    keys = ["seed", "split", "resource"]
    merged = full.merge(
        no_code,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_full", "_no_code"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Full and No-Code metric cells do not match")
    if not np.array_equal(merged["n_full"].to_numpy(int), merged["n_no_code"].to_numpy(int)):
        raise ValueError("Full and No-Code metric counts differ")

    output = merged[keys].copy()
    output["n"] = merged["n_full"].astype(int)
    for metric in METRICS:
        full_values = merged[f"{metric}_full"].astype(float)
        no_code_values = merged[f"{metric}_no_code"].astype(float)
        output[f"full_{metric}"] = full_values
        output[f"no_code_{metric}"] = no_code_values
        output[f"{metric}_delta_no_code_minus_full"] = no_code_values - full_values
        if metric in LOSS_METRICS:
            if (full_values <= 0.0).any():
                raise ValueError(f"Full {metric} must be positive for relative changes")
            output[f"{metric}_relative_change_percent"] = (
                (no_code_values - full_values) / full_values * 100.0
            )
    output["acc_at_10_delta_percentage_points"] = (
        output["acc_at_10_delta_no_code_minus_full"] * 100.0
    )
    return output.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, resource), part in deltas.groupby(["split", "resource"], sort=True):
        row: dict[str, Any] = {
            "split": split,
            "resource": resource,
            "seeds": int(len(part)),
            "n": int(part["n"].iloc[0]),
        }
        if part["n"].nunique() != 1:
            raise ValueError(f"Evaluation n changes across seeds for {split}/{resource}")
        for metric in METRICS:
            name = f"{metric}_delta_no_code_minus_full"
            values = part[name].astype(float)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_sample_sd"] = float(values.std(ddof=1))
            row[f"{name}_min"] = float(values.min())
            row[f"{name}_max"] = float(values.max())
            row[f"{metric}_no_code_worse_seed_count"] = int(
                (values > 0.0).sum() if metric in LOSS_METRICS else (values < 0.0).sum()
            )
        relative = part["mae_relative_change_percent"].astype(float)
        row["mae_relative_change_percent_mean"] = float(relative.mean())
        row["mae_relative_change_percent_sample_sd"] = float(relative.std(ddof=1))
        acc_pp = part["acc_at_10_delta_percentage_points"].astype(float)
        row["acc_at_10_delta_percentage_points_mean"] = float(acc_pp.mean())
        row["acc_at_10_delta_percentage_points_sample_sd"] = float(acc_pp.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "resource"]).reset_index(drop=True)


def _routing_summary(routing: pd.DataFrame) -> pd.DataFrame:
    numeric = (
        "history_bypass_rate",
        "qwen_required_rate",
        "embedding_cache_hit_rate_among_qwen",
        "encoder_forward_required_rate",
    )
    rows: list[dict[str, Any]] = []
    for keys, part in routing.groupby(
        ["variant", "split", "scope", "resource"], dropna=False, sort=True
    ):
        row: dict[str, Any] = dict(zip(("variant", "split", "scope", "resource"), keys))
        row["seeds"] = int(part["seed"].nunique())
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{column}_sample_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_outputs(output_dir: Path, frames: dict[str, pd.DataFrame], manifest: dict[str, Any], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Comparison outputs already exist: " + ", ".join(conflicts) + 
            "; use --overwrite to replace only this known artifact set"
        )
    staged: list[tuple[Path, Path]] = []
    try:
        for name, frame in frames.items():
            destination = output_dir / name
            temporary = output_dir / f".{name}.tmp-{uuid4().hex}"
            frame.to_csv(temporary, index=False)
            staged.append((temporary, destination))
        manifest_name = "comparison_manifest.json"
        temporary = output_dir / f".{manifest_name}.tmp-{uuid4().hex}"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True, allow_nan=False)
        staged.append((temporary, output_dir / manifest_name))
        for temporary, destination in staged:
            temporary.replace(destination)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if tuple(args.seeds) != FORMAL_SEEDS:
        raise ValueError("Formal comparison requires --seeds 40 41 42 43 44 in this order")
    splits = tuple(args.splits or ["test"])
    if len(set(splits)) != len(splits):
        raise ValueError("--split values must be unique")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    roots = {"full": _resolve(args.full_root), "no_code": _resolve(args.no_code_root)}
    plans = {
        variant: _validate_matrix_plan(
            roots[variant],
            variant=variant,
            seeds=args.seeds,
            max_tokens=args.max_tokens,
        )
        for variant in VARIANTS
    }
    if plans["full"].get("protocol") != plans["no_code"].get("protocol"):
        raise ValueError("Full and No-Code matrices use different cascade protocols")
    implementation_differences = _implementation_differences(
        plans["full"]["implementation"], plans["no_code"]["implementation"]
    )
    unexpected_implementation_differences = sorted(
        set(implementation_differences) - ORCHESTRATION_ONLY_FILES
    )
    if unexpected_implementation_differences:
        raise ValueError(
            "Full and No-Code matrices differ in predictive implementation files: "
            f"{unexpected_implementation_differences}"
        )

    metric_rows: list[pd.DataFrame] = []
    routing_rows: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    shared_identity: tuple[str, str] | None = None
    baseline_predictions: pd.DataFrame | None = None
    by_seed: dict[tuple[str, int], pd.DataFrame] = {}

    for variant in VARIANTS:
        for seed in args.seeds:
            run_dir = _run_dir(
                roots[variant],
                variant=variant,
                max_tokens=args.max_tokens,
                seed=seed,
            )
            manifest, verification_mode = _verify_sealed_run(
                run_dir,
                plan=plans[variant],
                seed=seed,
                variant=variant,
                max_tokens=args.max_tokens,
            )
            identity = (str(manifest["data_sha256"]), _split_fingerprint(manifest))
            if shared_identity is None:
                shared_identity = identity
            elif identity != shared_identity:
                raise ValueError(f"{variant} seed {seed} uses a different dataset or split")

            prediction_path = run_dir / "predictions.csv"
            predictions = pd.read_csv(
                prediction_path, dtype={"sample_id": str, "job_id": str}
            )
            validate_artifact_identity(predictions, manifest, f"{variant} seed {seed}")
            if baseline_predictions is None:
                baseline_predictions = predictions
            else:
                pair_prediction_frames(baseline_predictions, predictions, splits=splits)
            by_seed[(variant, seed)] = predictions

            metrics = compute_metrics(predictions, splits=splits)
            _assert_saved_metrics_match(run_dir, metrics)
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "variant", variant)
            metric_rows.append(metrics)

            routing = pd.read_csv(run_dir / "routing_metrics.csv")
            routing = routing.loc[routing["split"].astype(str).isin(splits)].copy()
            routing.insert(0, "seed", seed)
            routing.insert(0, "variant", variant)
            routing_rows.append(routing)
            input_records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "predictions_sha256": sha256_file(prediction_path),
                    "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                    "artifact_integrity_sha256": sha256_file(
                        run_dir / "artifact_integrity.json"
                    ),
                    "verification_mode": verification_mode,
                }
            )

    for seed in args.seeds:
        pair_prediction_frames(
            by_seed[("full", seed)], by_seed[("no_code", seed)], splits=splits
        )
    if shared_identity is None:
        raise AssertionError("No cascade runs were loaded")

    metrics = pd.concat(metric_rows, ignore_index=True).sort_values(
        ["split", "resource", "variant", "seed"], kind="mergesort"
    )
    expected_run_count = len(VARIANTS) * len(args.seeds) * len(splits) * 2
    if len(metrics) != expected_run_count:
        raise ValueError(
            f"Expected {expected_run_count} metric rows, observed {len(metrics)}"
        )
    summary = _variant_summary(metrics)
    deltas = _seed_deltas(metrics)
    delta_summary = _delta_summary(deltas)
    routing = pd.concat(routing_rows, ignore_index=True).sort_values(
        ["split", "scope", "resource", "variant", "seed"], kind="mergesort"
    )
    routing_summary = _routing_summary(routing)

    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else roots["no_code"] / "comparison_vs_full"
    )
    manifest = {
        "schema": "full-vs-no-code-cascade-comparison/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "five-seed pipeline-level input-modality sensitivity",
        "difference_definition": "No-Code minus Full",
        "interpretation": {
            "positive_loss_delta": "removing learned code representations worsens error",
            "negative_acc_delta": "removing learned code representations worsens Acc@10%",
            "scope": (
                "No-Code removes script-content encoding from the learned branch but "
                "retains exact-script identity for event-time history and routing."
            ),
            "routing_warning": (
                "No-Code has no code encoder; its encoder/cache route columns are "
                "simulation bookkeeping and must not be reported as measured encoder savings."
            ),
        },
        "seeds": list(args.seeds),
        "splits": list(splits),
        "max_tokens": args.max_tokens,
        "data_sha256": shared_identity[0],
        "split_fingerprint": shared_identity[1],
        "implementation": implementation_contract(),
        "implementation_comparison": {
            "full_recorded_sha256": plans["full"]["implementation"]["sha256"],
            "no_code_recorded_sha256": plans["no_code"]["implementation"]["sha256"],
            "differing_files": implementation_differences,
            "allowed_difference_scope": sorted(ORCHESTRATION_ONLY_FILES),
            "predictive_implementation_files_match": not bool(
                unexpected_implementation_differences
            ),
            "historical_verification_note": (
                "A historical run is accepted only when its internally sealed request "
                "matches its matrix plan and every differing source hash is confined "
                "to the matrix orchestration script. All artifact, checkpoint, route, "
                "request, and completion seals are still verified."
            ),
        },
        "matrix_plans": {
            variant: {
                "path": str(roots[variant] / "cascade_experiment_plan.json"),
                "sha256": sha256_file(roots[variant] / "cascade_experiment_plan.json"),
                "protocol": plans[variant].get("protocol"),
            }
            for variant in VARIANTS
        },
        "inputs": input_records,
        "outputs": list(OUTPUT_NAMES),
    }
    frames = {
        "cascade_variant_metrics.csv": metrics,
        "cascade_variant_summary.csv": summary,
        "no_code_minus_full_by_seed.csv": deltas,
        "no_code_minus_full_summary.csv": delta_summary,
        "cascade_variant_routing.csv": routing,
        "cascade_variant_routing_summary.csv": routing_summary,
    }
    _write_outputs(output_dir, frames, manifest, bool(args.overwrite))
    print("Full versus No-Code test summary:")
    print(summary.to_string(index=False))
    print("No-Code minus Full seed-paired differences:")
    print(delta_summary.to_string(index=False))
    print(f"Comparison artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
