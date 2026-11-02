#!/usr/bin/env python3
"""Export the primary exact-history rule from frozen Full cascade outputs.

This is an offline policy transformation, not a live latency benchmark. It
never fits a model, selects a policy on test labels, or changes No-Code. The
stored post-blend predictions and selected history expert remain frozen. Live
execution of the same rule is online_study's ``exact_bypass`` operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primary_predictions(reference, *, require_equivalence: bool = False):
    """Apply label-independent eligibility to sealed offline Full predictions.

    The exported file intentionally omits cache-hit/encoder-forward columns:
    these depend on a cache replay or measured execution and cannot be copied
    from a different routing policy. The target is used only downstream when
    calculating metrics, never to determine eligibility or a prediction.
    """
    import numpy as np
    from common.metrics import add_error_columns, coerce_valid_mask, validate_prediction_frame

    validate_prediction_frame(reference)
    required = {
        "resource_applicable", "history_available", "exact_script_history_available",
        "history_expert_prediction", "post_blend_prediction", "history_bypass",
        "reliability_threshold", "prediction",
    }
    if missing := required - set(reference.columns):
        raise ValueError(f"Missing frozen policy columns: {sorted(missing)}")
    if not set(reference["model"].astype(str)).issubset({"reliability_cascade_full"}):
        raise ValueError("Primary export accepts only the Full comparative engine, not No-Code")
    applicable = coerce_valid_mask(reference["resource_applicable"], "resource_applicable")
    history = coerce_valid_mask(reference["history_available"], "history_available")
    exact = coerce_valid_mask(reference["exact_script_history_available"], "exact_script_history_available")
    eligible = applicable & history & exact
    history_value = reference["history_expert_prediction"].to_numpy(float)
    post = reference["post_blend_prediction"].to_numpy(float)
    if not np.isfinite(history_value[eligible]).all() or not np.isfinite(post[~eligible]).all():
        raise ValueError("Frozen predictions required by the primary rule must be finite")
    prediction = np.clip(np.where(eligible, history_value, post), 0.0, 1.0)
    reference_prediction = reference["prediction"].to_numpy(float)
    reference_bypass = coerce_valid_mask(reference["history_bypass"], "history_bypass")
    thresholds = reference["reliability_threshold"].to_numpy(float)
    zero_thresholds = bool(np.isfinite(thresholds).all() and np.all(thresholds == 0.0))
    predictions_equal = bool(np.allclose(prediction, reference_prediction, atol=1e-12, rtol=0))
    routes_equal = bool(np.array_equal(eligible, reference_bypass))
    report = {
        "rows": len(reference),
        "all_reference_thresholds_zero": zero_thresholds,
        "prediction_equivalence_atol": 1e-12,
        "predictions_equivalent": predictions_equal,
        "routes_equivalent": routes_equal,
        "different_prediction_rows": int(np.sum(np.abs(prediction-reference_prediction) > 1e-12)),
        "different_route_rows": int(np.sum(eligible != reference_bypass)),
        "max_prediction_abs_difference": float(np.max(np.abs(prediction-reference_prediction))),
        "per_resource": {
            resource: {
                "eligible_rows": int(np.sum(eligible & reference["resource"].astype(str).eq(resource).to_numpy())),
                "thresholds": sorted(set(thresholds[reference["resource"].astype(str).eq(resource).to_numpy()].tolist())),
            }
            for resource in ("memory", "vram")
        },
    }
    if require_equivalence and not (zero_thresholds and predictions_equal and routes_equal):
        raise ValueError(
            "Paper-mode equivalence failed: the frozen Full reference does not reproduce "
            "the zero-threshold primary path. Preserve this run and report "
            f"its new policy separately. Audit: {report}"
        )

    output = reference.copy()
    for name in ("reliability_threshold", "reliability_probability", "history_bypass", "prediction"):
        if name in output:
            output[f"classifier_reference_{name}"] = output[name]
    output = output.drop(columns=[name for name in (
        "reliability_threshold", "reliability_probability", "embedding_cache_hit_job",
        "encoder_forward_required_job", "trust_region_radius",
    ) if name in output])
    output["model"] = "primary_exact_history_bypass_full"
    output["raw_prediction"] = prediction
    output["prediction"] = prediction
    output["history_bypass"] = eligible
    output["pre_classifier_evaluated"] = False
    output["qwen_required_resource"] = applicable & ~eligible
    job_need = output.groupby("sample_id", sort=False)["qwen_required_resource"].any()
    output["qwen_required_job"] = output["sample_id"].map(job_need).to_numpy(bool)
    # Projection exists in the frozen engine but is not executed in this rule:
    # every eligible exact-history case bypasses the Qwen/post-fusion path.
    output["trust_region_applied"] = False
    output["route_reason"] = np.select(
        [~applicable, eligible, ~history],
        ["resource_not_applicable", "exact_history_eligibility_bypass", "cold_start_qwen"],
        default="no_exact_history_qwen_blend",
    )
    return add_error_columns(output), report


def export_run(source: Path, destination: Path, *, seed: int, require_equivalence: bool, resume: bool = False):
    import pandas as pd
    from common.metrics import compute_metrics
    from reliability_cascade.integrity import verify_run_directory

    verify_run_directory(source, expected_seed=seed, expected_model="reliability_cascade_full",
                         expected_variant="full", expected_max_tokens=4096)
    identity = {"source_predictions_sha256": sha256(source / "predictions.csv"),
                "source_selected_policy_sha256": sha256(source / "selected_policy.json"),
                "source_integrity_sha256": sha256(source / "artifact_integrity.json"),
                "export_implementation_sha256": sha256(Path(__file__)),
                "require_equivalence": require_equivalence, "seed": seed}
    source_policy = json.loads((source / "selected_policy.json").read_text(encoding="utf-8"))
    for resource in ("memory", "vram"):
        if source_policy["selected"][resource]["configuration"].get("require_exact_script_for_bypass") is not True:
            raise ValueError("Primary export requires a frozen exact-script-eligible reference")
    manifest_path = destination / "export_manifest.json"
    if destination.exists() and any(destination.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError(f"Preserve existing exports; use --resume with identical inputs: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError(f"Export source or implementation changed: {destination}")
        for name, digest in manifest["files"].items():
            if sha256(destination / name) != digest:
                raise ValueError(f"Export artifact changed: {destination / name}")
        return manifest
    predictions = pd.read_csv(source / "predictions.csv", dtype={"sample_id": str, "job_id": str, "user_id": str})
    primary, report = primary_predictions(predictions, require_equivalence=require_equivalence)
    metrics = compute_metrics(primary)
    destination.mkdir(parents=True, exist_ok=True)
    primary.to_csv(destination / "predictions.csv", index=False)
    metrics.to_csv(destination / "metrics.csv", index=False)
    policy = {
        "schema": "primary-exact-history-bypass/v1", "fitted_pre_classifier_required": False,
        "rule": "resource_applicable and history_available and exact_script_history_available",
        "otherwise": "frozen selected post-Qwen blend", "refitted": False,
        "test_labels_used_for_selection": False,
        "selected_history_experts": {r: source_policy["selected"][r]["history_expert"] for r in ("memory", "vram")},
        "scope": "offline frozen-policy export; no cache or latency measurement; No-Code unchanged",
    }
    (destination / "primary_policy.json").write_text(json.dumps(policy, indent=2)+"\n", encoding="utf-8")
    manifest = {"schema": "primary-exact-history-export/v1", "identity": identity, "equivalence": report,
                "files": {name: sha256(destination / name) for name in ("predictions.csv", "metrics.csv", "primary_policy.json")}}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", default="outputs/study")
    parser.add_argument("--output-root", default="outputs/primary_exact_bypass")
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--paper-mode", action="store_true",
                        help="Require all zero thresholds and identical reference predictions/routes; fail instead of silently relabelling")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("study_root", "output_root"):
        value = Path(getattr(args, name))
        setattr(args, name, (ROOT / value).resolve() if not value.is_absolute() else value.resolve())
    if (args.output_root == ROOT or args.output_root in ROOT.parents or args.output_root == args.study_root
            or args.study_root in args.output_root.parents or args.output_root in args.study_root.parents):
        parser.error("output-root must be separate from the project root and source study")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be unique nonnegative integers")
    return args


def main(argv=None):
    args = parse_args(argv)
    for seed in args.seeds:
        source = args.study_root / "train_tail/cascade/reliability_cascade/tokens_4096/full" / f"seed_{seed}"
        destination = args.output_root / f"seed_{seed}"
        if args.dry_run:
            print(f"[PLAN] Frozen primary export {source} -> {destination}; paper_mode={args.paper_mode}")
            continue
        manifest = export_run(source, destination, seed=seed, require_equivalence=args.paper_mode, resume=args.resume)
        print(f"[DONE] seed={seed} {manifest['equivalence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
