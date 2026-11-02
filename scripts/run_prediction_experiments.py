#!/usr/bin/env python3
"""Run an independent prediction study with new checkpoints and predictions.

Planning uses the standard library only. Data preparation and actual execution
require the dependencies in requirements*.txt and the private source trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_paper_experiments import (  # noqa: E402
    ExperimentSpec, build_plan, completion_status,
)

ARMS = ("train_tail", "shared_validation_control")
STAGES = ("prepare", "base", "cascade", "components", "diagnostics", "coverage", "all")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory() -> dict:
    paths = [ROOT / "requirements.txt", ROOT / "requirements-qwen.txt"]
    for name in ("common", "qwen", "last-1", "hybrid", "reliability_cascade", "analysis", "scripts"):
        paths.extend((ROOT / name).rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def locked_json(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"Study protocol/source changed: {path}. Preserve this root and use a new study root.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/paper.local.yaml"))
    p.add_argument("--output-root", default="outputs/prediction_study")
    p.add_argument("--data-csv", default=os.environ.get("JOB_EFFICIENCY_CSV"))
    p.add_argument("--model-path", default=os.environ.get("MODEL_DIR"))
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=["train_tail"])
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    p.add_argument("--variants", nargs="+", choices=("full", "no_code", "code_only", "no_user", "no_metadata", "concat_fusion"), default=["full", "no_code"])
    p.add_argument("--num-processes", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--fit-fraction", type=float, default=0.90)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands only; no writes or data/model imports")
    a = p.parse_args(argv)
    for key in ("num_processes", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "bootstrap_replicates"):
        if getattr(a, key) <= 0:
            p.error(f"{key} must be positive")
    if not 0.5 <= a.fit_fraction < 1:
        p.error("fit-fraction must be in [0.5, 1)")
    if min(a.seeds) < 0 or len(set(a.seeds)) != len(a.seeds):
        p.error("seeds must be nonnegative and unique")
    if len(set(a.arms)) != len(a.arms) or len(set(a.variants)) != len(a.variants):
        p.error("arms and variants must be unique")
    a.config = Path(a.config).resolve()
    a.output_root = Path(a.output_root)
    if not a.output_root.is_absolute():
        a.output_root = ROOT / a.output_root
    a.output_root = a.output_root.resolve()
    if a.output_root == ROOT or ROOT in a.output_root.parents and len(a.output_root.relative_to(ROOT).parts) < 2:
        p.error("output-root must be a specific experiment directory, not the project or its immediate child")
    if not a.config.is_file():
        p.error(f"Configuration not found: {a.config}")
    return a


def arm_paths(a: argparse.Namespace, arm: str) -> tuple[Path, Path, Path]:
    return (a.output_root / "protocol" / f"{arm}.yaml",
            a.output_root / arm / "base", a.output_root / arm / "cascade")


def base_steps(a: argparse.Namespace, arm: str):
    cfg, base, _ = arm_paths(a, arm)
    return build_plan(ExperimentSpec(cfg, base, tuple(a.seeds), tuple(a.variants),
                                    a.num_processes, a.model_path, (4096,),
                                    a.batch_size, a.eval_batch_size, a.gradient_accumulation_steps))


def request(a: argparse.Namespace) -> dict:
    return {
        "schema": "prediction-study/v1", "config_sha256": digest(a.config),
        "source_sha256": inventory(), "source_config": str(a.config),
        "output_root": str(a.output_root), "data_csv": a.data_csv,
        "model_path": a.model_path, "arms": a.arms, "seeds": a.seeds,
        "base_model_inventory_sha256": os.environ.get("BASE_MODEL_INVENTORY_SHA256"),
        "variants": a.variants, "num_processes": a.num_processes,
        "batch_size": a.batch_size, "eval_batch_size": a.eval_batch_size,
        "accumulation": a.gradient_accumulation_steps,
        "effective_global_batch": a.num_processes * a.batch_size * a.gradient_accumulation_steps,
        "fit_fraction": a.fit_fraction, "max_tokens": 4096,
        "bootstrap_replicates": a.bootstrap_replicates, "bootstrap_seed": a.bootstrap_seed,
        "original_outer_split_unchanged": True,
        "test_previously_inspected": True, "new_training_from_scratch": True,
        "new_test_predictions_are_report_only": True,
    }


def run(command: Sequence[object], dry: bool = False) -> None:
    values = [str(x) for x in command]
    print(("[PLAN] " if dry else "[RUN] ") + shlex.join(values), flush=True)
    if not dry:
        subprocess.run(values, cwd=ROOT, check=True)


def prepare(a: argparse.Namespace) -> None:
    if a.dry_run:
        print("[PLAN] Prepare a new outer split, verify source identity, and write one immutable config/selection manifest per arm.")
        return
    from common.config import load_config, set_value, resolved_serializable_config
    from common.data import load_dataset, split_fingerprint
    from qwen.base_selection import select_base_training_frames
    import yaml
    if not a.data_csv or not Path(a.data_csv).is_file():
        raise FileNotFoundError("Set JOB_EFFICIENCY_CSV or --data-csv to the private raw CSV")
    if not a.model_path or not Path(a.model_path).is_dir():
        raise FileNotFoundError("Set MODEL_DIR or --model-path to the local Qwen2.5-Coder-1.5B-Instruct directory")
    config = load_config(a.config)
    for key, value in {
        "data.path": str(Path(a.data_csv).resolve()),
        "qwen.model_name_or_path": str(Path(a.model_path).resolve()),
        "split.manifest": str(a.output_root / "protocol" / "outer_split.csv"),
        "split.create_if_missing": True,
    }.items():
        set_value(config, key, value)
    bundle = load_dataset(config)
    locked_json(a.output_root / "protocol" / "data_identity.json", {
        "data_sha256": bundle.data_sha256, "split_fingerprint": split_fingerprint(bundle.frame),
        "split_manifest_sha256": digest(bundle.manifest_path),
        "effective_counts": bundle.frame.groupby("split").size().to_dict(),
        "protocol": bundle.protocol,
    })
    # Metadata only: no expensive/in-sample Qwen predictions are needed for
    # training-user/script novelty or the full-stream coverage audit.
    retained = bundle.retained_frame.copy()
    retained["effective_analysis_row"] = retained.sample_id.isin(set(bundle.frame.sample_id))
    retained["script_sha256"] = retained["code"].map(
        lambda x: hashlib.sha256((str(x) if str(x).strip() else "<EMPTY_SCRIPT>").encode("utf-8")).hexdigest())
    columns = ["sample_id", "user_id", "script_sha256", "split", "gpu_count",
               "submit_time", "completion_time", "valid_memory", "valid_vram", "effective_analysis_row"]
    index_path = a.output_root / "protocol" / "population_index.csv"
    index_text = retained[columns].to_csv(index=False, lineterminator="\n")
    if index_path.exists() and index_path.read_text(encoding="utf-8") != index_text:
        raise ValueError("Population index changed within the frozen study")
    if not index_path.exists():
        index_path.write_text(index_text, encoding="utf-8", newline="")
    locked_json(index_path.with_suffix(".json"), {
        "data_sha256": bundle.data_sha256, "split_fingerprint": split_fingerprint(bundle.frame),
        "population_index_sha256": digest(index_path), "rows": len(retained),
        "contains_private_identifiers": True,
    })
    set_value(config, "split.create_if_missing", False)
    for arm in a.arms:
        cfg, base, _ = arm_paths(a, arm)
        for key, value in {
            "qwen.base_selection.mode": arm, "qwen.base_selection.fit_fraction": a.fit_fraction,
            "qwen.batch_size": a.batch_size, "qwen.eval_batch_size": a.eval_batch_size,
            "qwen.gradient_accumulation_steps": a.gradient_accumulation_steps,
            "qwen.max_tokens": 4096,
            "reliability_cascade.require_independent_base_selection": arm == "train_tail",
            "outputs.root": str(base),
        }.items():
            set_value(config, key, value)
        selected = select_base_training_frames(bundle, config)
        locked_json(a.output_root / "protocol" / f"{arm}_base_selection.json", selected.manifest)
        import pandas as pd
        roles = a.output_root / "protocol" / f"{arm}_base_roles.csv"
        if not roles.exists():
            pd.DataFrame(selected.manifest["sample_roles"]).to_csv(roles, index=False)
        payload = resolved_serializable_config(config)
        if cfg.exists():
            if yaml.safe_load(cfg.read_text(encoding="utf-8")) != payload:
                raise FileExistsError(f"Prepared configuration changed: {cfg}")
        else:
            cfg.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        print(f"[AUDIT] {arm}: gradient={len(selected.optimization)}, checkpoint={len(selected.early_stopping)}, overlap={selected.manifest['base_checkpoint_validation_overlap']}")


def archive_interrupted(a: argparse.Namespace, step) -> None:
    """Keep interrupted artifacts recoverable instead of deleting directories."""
    out = step.output_dir.resolve()
    if a.output_root not in out.parents or step.output_dir.is_symlink():
        raise ValueError(f"Refuse an unmanaged output: {out}")
    if not out.exists():
        return
    if step.stage == "qwen_evaluate":
        files = [out / name for name in ("predictions.csv", "metrics.csv", "run_manifest.json") if (out / name).exists()]
        # A freshly completed training directory is the normal input to the
        # first evaluation. It is not an interrupted evaluation output.
        if not files:
            return
    if not a.resume:
        raise FileExistsError(f"Output already exists: {out}; use --resume with the identical study arguments")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = a.output_root / "interrupted" / stamp / step.step_id
    if step.stage == "qwen_evaluate":
        if files:
            dst.mkdir(parents=True)
            for source in files:
                shutil.move(str(source), str(dst / source.name))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dst))
    print(f"[RECOVERY] Preserved interrupted artifacts under {dst}")


def train_base(a: argparse.Namespace, arm: str) -> None:
    executed = set()
    for step in base_steps(a, arm):
        if a.dry_run:
            run(step.command, True)
            continue
        state = completion_status(step)
        upstream_reexecuted = any(dependency in executed for dependency in step.depends_on)
        if state.complete and not upstream_reexecuted:
            if not a.resume:
                raise FileExistsError(f"Complete run exists; use --resume: {step.output_dir}")
            print(f"[SKIP] {step.step_id}: {state.reason}")
            continue
        if upstream_reexecuted:
            print(f"[RECOVERY] Re-run {step.step_id} because an upstream artifact was regenerated")
        archive_interrupted(a, step)
        run(step.command)
        executed.add(step.step_id)
        state = completion_status(step)
        if not state.complete:
            raise RuntimeError(f"Output verification failed for {step.step_id}: {state.reason}")


def cascade(a: argparse.Namespace, arm: str, components: bool = False) -> None:
    cfg, base, output = arm_paths(a, arm)
    cases = {"selected": []} if not components else {
        "exact_history_bypass": ["reliability_cascade.pre_gate.thresholds=[0.0]"],
        "no_history_bypass": ["reliability_cascade.pre_gate.thresholds=[1.01]"],
        "pre_route_only": ["reliability_cascade.post_gate.enabled=false"],
        "no_trust_region": ["reliability_cascade.trust_region.enabled=false"],
    }
    variants = [v for v in a.variants if not components or v == "full"]
    for case, overrides in cases.items():
        dest = output if case == "selected" else a.output_root / arm / "components" / case
        for variant in variants:
            # Individual sealed runs avoid a Full/No-Code matrix-plan collision
            # while retaining one canonical prediction tree for paired analysis.
            for seed in a.seeds:
                run_dir = dest / "reliability_cascade" / "tokens_4096" / variant / f"seed_{seed}"
                qwen = base / "qwen" / "tokens_4096" / variant / f"seed_{seed}" / "predictions.csv"
                last1 = base / "last-1" / "completed_before_submission" / f"seed_{seed}" / "predictions.csv"
                individual = [sys.executable, "-m", "reliability_cascade.run", "--config", cfg,
                              "--qwen-predictions", qwen, "--last1-predictions", last1,
                              "--output-dir", run_dir, "--set", f"seed={seed}",
                              "--set", f"reliability_cascade.qwen_variant={variant}"]
                for value in overrides:
                    individual.extend(["--set", value])
                if a.dry_run:
                    run(individual, True)
                    continue
                if run_dir.exists():
                    if not a.resume:
                        raise FileExistsError(f"Use --resume for existing cascade: {run_dir}")
                    from common.config import load_config
                    from reliability_cascade.integrity import build_run_request, verify_run_directory
                    config = load_config(cfg, [f"seed={seed}", f"reliability_cascade.qwen_variant={variant}", *overrides])
                    expected = build_run_request(config, qwen, last1, protocol="history-first-reliability-cascade/v1")
                    try:
                        verify_run_directory(run_dir, expected_seed=seed,
                                   expected_model=f"reliability_cascade_{variant}",
                                   expected_variant=variant, expected_max_tokens=4096,
                                   expected_run_request_sha256=expected["run_request_sha256"])
                    except (OSError, ValueError, RuntimeError) as exc:
                        from types import SimpleNamespace
                        print(f"[RECOVERY] Incomplete or stale cascade: {exc}")
                        archive_interrupted(a, SimpleNamespace(output_dir=run_dir, stage="cascade",
                                            step_id=f"{arm}.{case}.{variant}.seed_{seed}"))
                    else:
                        print(f"[SKIP] verified cascade {arm}/{case}/{variant}/seed_{seed}")
                        continue
                run(individual)


def analyze(a: argparse.Namespace, arm: str) -> None:
    cfg, base, casc = arm_paths(a, arm)
    if not {"full", "no_code"}.issubset(a.variants):
        raise ValueError("Paired diagnostics require full and no_code; include both variants in the study")
    cmd = [sys.executable, "analysis/prediction_diagnostics.py", "--config", cfg, "--base-root", base,
         "--cascade-root", casc, "--output-dir", a.output_root / arm / "analysis",
         "--population-index", a.output_root / "protocol" / "population_index.csv",
         "--seeds", *a.seeds, "--variants", "full", "no_code", "--max-tokens", 4096,
         "--bootstrap-replicates", a.bootstrap_replicates, "--bootstrap-seed", a.bootstrap_seed]
    if a.resume:
        cmd.append("--overwrite")
    run(cmd, a.dry_run)
    run([sys.executable, "analysis/plot_prediction_diagnostics.py", "--analysis-dir", a.output_root / arm / "analysis"], a.dry_run)


def coverage(a: argparse.Namespace, arm: str) -> None:
    _, base, _ = arm_paths(a, arm)
    seed = 42 if 42 in a.seeds else a.seeds[0]
    saved = base / "qwen" / "tokens_4096" / "full" / f"seed_{seed}"
    run([sys.executable, "analysis/token_coverage.py", "--config", saved / "config.resolved.yaml",
         "--checkpoint", saved / "checkpoint", "--budgets", 2048, 4096, 8192,
         "--output", a.output_root / arm / "analysis" / "token_coverage.csv"], a.dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not a.dry_run:
        locked_json(a.output_root / "prediction_study_plan.json", request(a))
    if a.stage in ("prepare", "all"):
        prepare(a)
    for arm in a.arms:
        cfg, _, _ = arm_paths(a, arm)
        if not a.dry_run and not cfg.exists():
            raise FileNotFoundError(f"Run --stage prepare first with identical arguments: {cfg}")
        for stage, action in (("base", train_base), ("cascade", cascade),
                              ("components", lambda x, y: cascade(x, y, True)),
                              ("diagnostics", analyze), ("coverage", coverage)):
            if a.stage in (stage, "all"):
                action(a, arm)
    print("[DONE] Commands planned." if a.dry_run else "[DONE] Requested stage complete. New outcomes are not expected to equal archived paper numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
