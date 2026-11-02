"""Collect aggregate evidence; never replace absent results with paper values."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SCOPE = (
    "Full: primary exact-history bypass; its archived classifier-equipped reference "
    "is prediction-equivalent only after validation-selected zero-threshold verification. "
    "No-Code retains its independently selected adaptive routes."
)
CAUTIONS = [
    PRIMARY_SCOPE,
    "Training seeds share one trace and test period; sample SD is not a confidence interval.",
    "Pointwise paired intervals are not pooled across seeds or multiplicity-adjusted.",
    "Temporal windows retain independently selected adaptive-routing policies at seed 42.",
    "Residual calibration uses a separate 70/15/15 validation partition, not the primary 70/30 partition.",
    "Online accuracy is measured batch-one accuracy; P95/P99 average within-run percentiles.",
    "Sampled-ratio shortfalls are not OOM probabilities, allocation guarantees, or physical savings.",
    "Generated figures preserve numerical inputs and statistical definitions.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, default=ROOT / "outputs/study")
    p.add_argument("--primary-root", type=Path, default=ROOT / "outputs/primary_exact_bypass")
    p.add_argument("--baselines-root", type=Path, default=ROOT / "outputs/baselines")
    p.add_argument("--followup-root", type=Path, default=ROOT / "outputs/followup")
    p.add_argument("--hardware-root", type=Path, default=ROOT / "outputs/hardware")
    p.add_argument("--windows-root", type=Path, default=ROOT / "outputs/temporal")
    p.add_argument("--online-root", type=Path, default=ROOT / "outputs/online")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/publication")
    p.add_argument("--no-figures", action="store_true", help="Only export aggregate tables and their evidence index")
    p.add_argument("--dry-run", action="store_true", help="List expected inputs without importing numerical packages or writing files")
    p.add_argument("--require-complete", action="store_true", help="Exit nonzero if any expected evidence source or figure input is missing")
    p.add_argument("--overwrite", action="store_true", help="Replace report files in a previously generated report directory")
    return p.parse_args(argv)


def input_plan(a):
    analysis = a.study_root / "train_tail/analysis"
    entries = [
        ("Table 1 / population eligibility", analysis / "population_shift_summary.csv"),
        ("Table 2 / resolved protocol configuration", a.study_root / "protocol/train_tail.yaml"),
        ("Table 3 / primary and history models", analysis / "test_metrics_summary.csv"),
        ("Table 3 / matched conventional baselines", a.baselines_root / "metrics_summary.csv"),
        ("Tables 4-5 / paired intervals", analysis / "paired_bootstrap_per_seed.csv"),
        ("Table 7 / signed and tail errors", analysis / "risk_metrics_summary.csv"),
        ("Table 8 / calibration joint shortfall", a.followup_root / "allocation/joint_bounds_metrics_summary.csv"),
        ("Table 8 / calibration resource bounds", a.followup_root / "allocation/resource_bounds_metrics_summary.csv"),
        ("Classifier follow-up / budget metrics", a.followup_root / "gate/gate_metrics_summary.csv"),
        ("Fig. 2 / paired daily differences", analysis / "daily_comparisons_per_seed.csv"),
        ("Fig. 3 / target-independent subgroups", analysis / "stratified_metrics_summary.csv"),
        ("Fig. 4 / tokenizer-only coverage", analysis / "token_coverage.csv"),
        ("Fig. 5 / fixed-policy accuracy", analysis / "frozen_accuracy_summary.csv"),
        ("Fig. 5 / bounded cache replay", analysis / "cache_routing_summary.csv"),
        ("Fig. 7 / frozen-prediction hardware slices", a.hardware_root / "capacity_metrics_summary.csv"),
        ("Fig. 8 / fixed additive margins", analysis / "margin_diagnostics_summary.csv"),
        ("Fig. 9 / measured online summary", a.online_root / "summary/summary.csv"),
    ]
    for fraction in (600, 800, 1000):
        entries.append((f"Fig. 6 / adaptive prefix {fraction/1000:g}", a.windows_root / f"prefix_{fraction:04d}/study/train_tail/analysis/test_metrics_summary.csv"))
    for seed in range(40, 45):
        entries.append((f"Primary equivalence / seed {seed}", a.primary_root / f"seed_{seed}/export_manifest.json"))
    return [{"purpose": purpose, "path": str(path.resolve()), "exists": path.is_file()} for purpose, path in entries]


def verify_primary_exports(a, sources, missing):
    """Never relabel new classifier outputs as the primary method without proof."""
    complete = True
    for seed in range(40, 45):
        folder = a.primary_root / f"seed_{seed}"
        path = folder / "export_manifest.json"
        if not path.is_file():
            missing.append({"purpose": f"Primary zero-threshold equivalence seed {seed}", "path": str(path.resolve())})
            complete = False
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity, audit = manifest.get("identity", {}), manifest.get("equivalence", {})
        if (manifest.get("schema") != "primary-exact-history-export/v1" or identity.get("seed") != seed
                or identity.get("require_equivalence") is not True
                or not all(audit.get(k) is True for k in
                           ("all_reference_thresholds_zero", "predictions_equivalent", "routes_equivalent"))):
            raise ValueError(f"Unverified primary equivalence: {path}")
        reference = a.study_root / f"train_tail/cascade/reliability_cascade/tokens_4096/full/seed_{seed}"
        for filename, key in (("predictions.csv", "source_predictions_sha256"),
                              ("selected_policy.json", "source_selected_policy_sha256"),
                              ("artifact_integrity.json", "source_integrity_sha256")):
            original = reference / filename
            if not original.is_file() or sha256(original) != identity.get(key):
                raise ValueError(f"Primary export belongs to different frozen inputs: {original}")
        for filename in ("predictions.csv", "metrics.csv", "primary_policy.json"):
            artifact = folder / filename
            if not artifact.is_file() or sha256(artifact) != manifest.get("files", {}).get(filename):
                raise ValueError(f"Primary export checksum failed: {artifact}")
        sources[str(path.resolve())] = {"sha256": sha256(path), "purpose": "Primary route/prediction equivalence"}
    return complete


def run(a):
    import numpy as np
    import pandas as pd

    out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"Report directory is nonempty: {out}; use a fresh directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    analysis = a.study_root / "train_tail/analysis"
    sources, outputs, missing = {}, {}, []
    primary_verified = verify_primary_exports(a, sources, missing)
    primary_scope = PRIMARY_SCOPE if primary_verified else (
        "Classifier-equipped Full reference only: primary exact-bypass equivalence has not been verified. "
        "No-Code retains its independently selected adaptive routes."
    )
    if not primary_verified and not a.no_figures:
        raise ValueError("Primary figure labels require verified primary-export artifacts; run primary-export first or use --no-figures")

    def read(path, role, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                missing.append({"purpose": role, "path": str(path.resolve())})
            return None
        frame = pd.read_csv(path)
        sources[str(path.resolve())] = {"sha256": sha256(path), "rows": len(frame), "purpose": role}
        return frame

    def save(frame, name, scope):
        if frame is None:
            return
        path = tables / (name + ".csv")
        frame.to_csv(path, index=False)
        outputs[str(path.relative_to(out))] = {"sha256": sha256(path), "rows": len(frame), "scope": scope}

    # Only aggregate/seed-aggregate inputs are copied. Never ingest raw predictions,
    # executable scripts, node lists, per-job bounds, or unknown-node diagnostics.
    main = read(analysis / "test_metrics_summary.csv", "Table 3 / primary/history models")
    baselines = read(a.baselines_root / "metrics_summary.csv", "Table 3 / matched conventional baselines")
    main_parts = []
    if main is not None:
        if primary_verified:
            primary_rows = []
            for seed in range(40, 45):
                frame = read(a.primary_root / f"seed_{seed}/metrics.csv", f"Primary exported metrics seed {seed}")
                primary_rows.append(frame.loc[frame.split.eq("test")].assign(seed=seed))
            frame = pd.concat(primary_rows, ignore_index=True)
            cols = [c for c in ("n", "mae", "mse", "rmse", "acc_at_10") if c in frame]
            grouped = frame.groupby("resource")[cols].agg(["mean", "std"])
            grouped.columns = [f"{m}_{'sd' if s == 'std' else s}" for m, s in grouped.columns]
            grouped = grouped.reset_index().assign(method="Cascade full", seed_n=5)
            main = pd.concat([main.loc[~main.method.eq("Cascade full")], grouped], ignore_index=True)
        main_parts.append(main.loc[~main.method.eq("Standalone no_code")].assign(experiment_scope=primary_scope))
    if baselines is not None:
        block = baselines.loc[baselines["split"].eq("test")].copy()
        block["method"] = block["model"].map({"lstm": "BiLSTM", "textcnn": "TextCNN", "xgboost": "XGBoost"}).fillna(block["model"])
        main_parts.append(block.assign(experiment_scope="matched optimization/early-stopping baseline study"))
    fixed_rows = []
    for seed in (40, 41, 42, 43, 44):
        path = a.study_root / f"train_tail/base/safe_script_history_hybrid/completed_before_submission/tokens_4096/full/seed_{seed}/metrics.csv"
        frame = read(path, f"Table 3 / Fixed Hybrid seed {seed}")
        if frame is not None:
            fixed_rows.append(frame.loc[frame.split.eq("test")].assign(seed=seed, method="Fixed Hybrid"))
    if fixed_rows:
        fixed = pd.concat(fixed_rows, ignore_index=True)
        grouped = fixed.groupby(["resource", "method"])[["n", "mae", "mse", "rmse", "acc_at_10"]].agg(["mean", "std"])
        grouped.columns = [f"{m}_{'sd' if s == 'std' else s}" for m, s in grouped.columns]
        grouped = grouped.reset_index().merge(fixed.groupby(["resource", "method"]).seed.nunique().rename("seed_n").reset_index(), on=["resource", "method"])
        main_parts.append(grouped.assign(experiment_scope="fixed hybrid with validation-selected mixture"))
    if main_parts:
        save(pd.concat(main_parts, ignore_index=True), "table3_main_comparison", primary_scope)

    config_path = a.study_root / "protocol/train_tail.yaml"
    if config_path.is_file():
        settings = []
        # Preserve the resolved YAML text instead of reinterpreting scalar types
        # or environment placeholders. No YAML dependency is needed for reporting.
        sections = {"qwen", "training", "textcnn", "lstm", "xgboost", "reliability_cascade", "last1", "hybrid"}
        current = None
        for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line[0].isspace():
                current = line.split(":", 1)[0]
            key = line.strip().split(":", 1)[0]
            if current in sections and "path" not in key.lower():
                settings.append({"section": current, "source_line": line_number, "resolved_yaml": line})
        sources[str(config_path.resolve())] = {"sha256": sha256(config_path), "purpose": "Table 2 / resolved training and model configuration"}
        save(pd.DataFrame(settings), "table2_resolved_training_settings", "Resolved config settings; path fields omitted; runtime effective parameters remain in run manifests")
    else:
        missing.append({"purpose": "Table 2 / resolved training configuration", "path": str(config_path.resolve())})

    population = read(analysis / "population_shift_summary.csv", "Table 1 / resource eligibility")
    save(population, "table1_population_eligibility", "Effective and retained populations remain explicitly labelled")
    paired = read(analysis / "paired_bootstrap_per_seed.csv", "Tables 4-5 / paired uncertainty")
    if paired is not None:
        save(paired.loc[paired.reference.eq("Hierarchical Same-Script Last-1") & paired.subgroup.eq("all_valid") & paired.cluster.eq("day")],
             "table4_paired_hierarchical_day", "Proposed minus Hierarchical; seed-specific pointwise intervals")
        save(paired.loc[paired.reference.eq("Cascade no_code") & ~paired.subgroup.eq("all_valid") & paired.cluster.eq("day") & paired.metric.eq("mae")],
             "table5_no_code_subgroup_mae", "All available seed/resource/subgroup MAE differences, never pooled across seeds")
        save(paired, "paired_cluster_sensitivities", "Separate day/3-day/7-day/user/script sensitivities")

    # Pipeline comparisons below repeat selection; they are not the frozen controls.
    component_rows = []
    for case, method in (("exact_history_bypass", "Exact Bypass pipeline"), ("no_history_bypass", "No History Bypass pipeline"),
                         ("pre_route_only", "Pre-Route Only pipeline"), ("no_trust_region", "No Trust Region pipeline")):
        for seed in (40, 41, 42, 43, 44):
            path = a.study_root / f"train_tail/components/{case}/reliability_cascade/tokens_4096/full/seed_{seed}/metrics.csv"
            frame = read(path, f"Table 6 / {case} seed {seed}")
            if frame is not None:
                component_rows.append(frame.loc[frame.split.eq("test")].assign(seed=seed, method=method))
    if component_rows:
        frame = pd.concat(component_rows, ignore_index=True)
        save(frame, "table6_pipeline_components_per_seed", "Variant-specific fitting and policy selection")
        metric_cols = [c for c in ("n", "mae", "mse", "rmse", "acc_at_10") if c in frame]
        grouped = frame.groupby(["method", "resource"])[metric_cols].agg(["mean", "std"])
        grouped.columns = [f"{m}_{'sd' if s == 'std' else s}" for m, s in grouped.columns]
        grouped = grouped.reset_index()
        counts = frame.groupby(["method", "resource"]).seed.nunique().rename("seed_n").reset_index()
        save(grouped.merge(counts, on=["method", "resource"]), "table6_pipeline_components_summary", "Partial matrices retain observed seed counts")
    if main is not None:
        base_cascade = main.loc[main.method.str.startswith(("Cascade", "Standalone"))].copy()
        if primary_verified:
            reference_rows = main.loc[main.method.eq("Cascade full")].assign(method="Classifier-equipped Full reference")
            base_cascade = pd.concat([base_cascade, reference_rows], ignore_index=True)
        save(base_cascade, "table6_base_and_cascade", primary_scope)
    save(read(analysis / "risk_metrics_summary.csv", "Table 7 / risk diagnostics"), "table7_signed_tail_high_utilization", "Normalized sampled peaks, no allocation guarantee")
    for kind in ("joint", "resource"):
        save(read(a.followup_root / f"allocation/{kind}_bounds_metrics_summary.csv", f"Table 8 / {kind} calibrated bounds"),
             f"table8_calibration_{kind}", "Separate adaptive-routing 70/15/15 validation/calibration study")

    auxiliary = [
        (analysis, "gate_fit_counts_per_seed.csv", "classifier_fitting_counts"),
        (analysis, "gate_diagnostics_summary.csv", "classifier_diagnostics"),
        (analysis, "bypass_diagnostics_summary.csv", "bypass_regret"),
        (analysis, "frozen_accuracy_summary.csv", "frozen_component_accuracy"),
        (analysis, "cache_routing_summary.csv", "offline_bounded_cache"),
        (analysis, "stratified_metrics_summary.csv", "descriptive_slices"),
        (analysis, "margin_diagnostics_summary.csv", "fixed_margin_shortfall"),
        (analysis, "mcnemar_per_seed.csv", "mcnemar_sensitivity"),
        (a.followup_root / "gate", "gate_metrics_summary.csv", "budget_routing_metrics"),
        (a.followup_root / "gate", "gate_job_costs_summary.csv", "budget_routing_costs"),
        (a.followup_root / "gate", "gate_paired_intervals_per_seed.csv", "budget_routing_paired"),
        (a.hardware_root, "capacity_metrics_summary.csv", "hardware_strata"),
        (a.hardware_root, "capacity_paired_intervals_per_seed.csv", "hardware_paired_intervals"),
        (a.hardware_root, "hardware_monthly_composition.csv", "hardware_composition"),
        (a.online_root / "summary", "summary.csv", "measured_online_summary"),
    ]
    for folder, name, label in auxiliary:
        save(read(folder / name, label), label, "Aggregate evidence; original analysis scope is retained")

    figures = []
    if not a.no_figures:
        from publication_reporting.figures import make_figures
        figures = make_figures(a, read, out / "figures", missing)
        for item in figures:
            for filename in item["files"]:
                path = out / "figures" / filename
                outputs[str(path.relative_to(out))] = {"sha256": sha256(path), "scope": item["scope"]}

    # Deduplicate unavailable inputs even where used by both a table and a figure.
    missing = list({(v["path"], v["purpose"]): v for v in missing}.values())
    manifest = {"schema": "publication-evidence-report/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources": sources, "outputs": outputs, "figures": figures, "missing_inputs": missing,
                "complete": not missing, "cautions": CAUTIONS,
                "primary_equivalence_verified": primary_verified,
                "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Evaluation results index", "", *["- " + value for value in CAUTIONS], "",
             "The index records only files found at the requested roots. Missing experiments are listed explicitly; no paper values are substituted.", "",
             "| Generated artifact | Rows | Scope |", "|---|---:|---|"]
    for name, item in outputs.items():
        lines.append(f"| [{name}]({name}) | {item.get('rows', '')} | {item['scope']} |")
    if missing:
        lines.extend(["", "## Missing inputs", ""])
        lines.extend(f"- {item['purpose']}: `{item['path']}`" for item in missing)
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "generated_files": len(outputs), "missing_inputs": len(missing)}, indent=2))
    return 1 if missing and a.require_complete else 0


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        print(json.dumps({"input_plan": input_plan(a), "output_dir": str(a.output_dir.resolve()),
                          "additional_inputs": "Table 6 per-seed component metrics; classifier/follow-up aggregate diagnostics",
                          "cautions": CAUTIONS}, indent=2, ensure_ascii=False))
        return 0
    return run(a)
