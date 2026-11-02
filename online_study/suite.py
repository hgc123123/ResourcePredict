"""Run a serial, balanced online-path study without retraining any model.

Only the standard library is imported here. Every child performs genuine
conditional inference; prediction CSV files are numerical references, not a
replacement for inference. Completed runs are reusable only after verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "reliability-cascade-online-suite/v1"
REPORT_SCHEMA = "reliability-cascade-live-path/v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, help="Completed prediction study root containing train_tail/")
    parser.add_argument("--output-root", required=True, help="A new directory reserved for this timing study")
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--repeats", type=int, default=3, help="Technical repeats per seed and case; not additional training seeds")
    parser.add_argument("--cache-capacities", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--operations", nargs="+", choices=("full", "no_bypass", "no_post", "no_trust", "exact_bypass"),
                        default=["full", "no_bypass", "no_post", "no_trust", "exact_bypass"],
                        help="Frozen operations; full is required as the reference")
    parser.add_argument("--control-cache-capacity", type=int, default=1024)
    parser.add_argument("--cache-start", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-csv")
    parser.add_argument("--base-model")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--warmup-jobs", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--parity-atol", type=float, default=0.005)
    parser.add_argument("--order-seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan without creating files")
    parser.add_argument("--resume", action="store_true", help="Reuse only verified, fully completed runs of the identical plan")
    return parser.parse_args(argv)


def _frozen_inputs(base: Path, cascade: Path) -> dict[str, str]:
    """Small sealed manifests plus the selected policies and serialized gates.

    The child additionally verifies the full checkpoint closure. Hashing frozen
    inputs here stops a resume from silently switching checkpoints or policies.
    """
    paths = [base / "config.resolved.yaml", base / "run_manifest.json", base / "predictions.csv",
             base / "checkpoint" / "training_manifest.json",
             base / "checkpoint" / "model_config.json",
             cascade / "run_manifest.json", cascade / "selected_policy.json",
             cascade / "artifact_integrity.json", cascade / "predictions.csv",
             cascade / "checkpoint" / "cascade_models.joblib"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Completed experiment inputs are missing:\n" + "\n".join(missing))
    # Include adapter/head/tokenizer files, not only their manifest, so a
    # resume cannot silently retain results after a checkpoint file changes.
    paths = sorted(set(paths) | {path for path in (base / "checkpoint").rglob("*") if path.is_file()})
    return {str(path): sha256_file(path) for path in paths}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("--seeds must be nonempty and contain no duplicates")
    if "full" not in args.operations or len(set(args.operations)) != len(args.operations):
        raise ValueError("--operations must contain full and have no duplicates")
    if args.repeats < 1 or args.num_threads < 1 or args.warmup_jobs < 0:
        raise ValueError("Repeats/threads must be positive and warmup jobs nonnegative")
    if args.parity_atol < 0 or args.control_cache_capacity < 1 or any(x < 0 for x in args.cache_capacities):
        raise ValueError("Cache capacities and parity tolerance are invalid; control capacity must be positive")
    study = Path(args.study_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    if output == study or study in output.parents and output.name == "train_tail":
        raise ValueError("Use a separate output directory, not the completed study root or train_tail/")
    capacities = sorted(set([0, args.control_cache_capacity, *args.cache_capacities]))
    cases = [{"operation": "full", "cache_capacity_entries": capacity,
              "case": f"full_lru_{capacity}"} for capacity in capacities]
    cases.extend({"operation": operation, "cache_capacity_entries": args.control_cache_capacity,
                  "case": f"{operation}_lru_{args.control_cache_capacity}"}
                 for operation in args.operations if operation != "full")
    inputs: dict[str, str] = {}
    seed_dirs: dict[int, tuple[Path, Path]] = {}
    for seed in args.seeds:
        base = study / "train_tail" / "base" / "qwen" / "tokens_4096" / "full" / f"seed_{seed}"
        cascade = study / "train_tail" / "cascade" / "reliability_cascade" / "tokens_4096" / "full" / f"seed_{seed}"
        inputs.update(_frozen_inputs(base, cascade))
        seed_dirs[seed] = base, cascade
    data = Path(args.data_csv).expanduser().resolve() if args.data_csv else None
    if data is not None:
        if not data.is_file():
            raise FileNotFoundError(data)
        inputs[str(data)] = sha256_file(data)
    required_sources = [Path(__file__), Path(__file__).with_name("summarize.py"), Path(__file__).with_name("benchmark.py")]
    missing_sources = [str(path) for path in required_sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError("Timing package files are missing: " + ", ".join(missing_sources))
    source_files = [path for folder in ("common", "qwen", "reliability_cascade", "online_study")
                    for path in sorted((PROJECT_ROOT / folder).glob("*.py"))]
    source_files.extend(PROJECT_ROOT / "scripts" / name for name in (
        "run_online_study.py", "summarize_online_study.py", "run_online_study.slurm"))
    sources = {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in source_files if path.is_file()}
    source_hash = canonical_hash(sources)
    rng = random.Random(args.order_seed)
    runs: list[dict[str, Any]] = []
    # Each repeat uses a shuffled case order, rotated across seeds so that
    # cases do not always occupy the same relative position. Seed order is
    # shuffled independently. All children still run strictly serially.
    for repeat in range(1, args.repeats + 1):
        case_order = list(cases)
        seed_order = list(args.seeds)
        rng.shuffle(case_order)
        rng.shuffle(seed_order)
        for seed_position, seed in enumerate(seed_order):
            offset = (seed_position + repeat - 1) % len(case_order)
            ordered = case_order[offset:] + case_order[:offset]
            base, cascade = seed_dirs[seed]
            for case in ordered:
                destination = output / "runs" / case["case"] / f"seed_{seed}" / f"repeat_{repeat}"
                command = [sys.executable, "-m", "online_study.benchmark", "--config", str(base / "config.resolved.yaml"),
                           "--checkpoint", str(base / "checkpoint"), "--cascade-dir", str(cascade),
                           "--output-dir", str(destination), "--operation", case["operation"],
                           "--cache-capacity", str(case["cache_capacity_entries"]), "--cache-start", args.cache_start,
                           "--device", args.device, "--warmup-jobs", str(args.warmup_jobs),
                           "--num-threads", str(args.num_threads), "--repeat", str(repeat),
                           "--parity-predictions", str(cascade / "predictions.csv"), "--parity-atol", str(args.parity_atol),
                           "--reference-dir", str(output / "batch_one_references" / f"seed_{seed}")]
                if data:
                    command.extend(["--data-csv", str(data)])
                if args.base_model:
                    command.extend(["--base-model", str(Path(args.base_model).expanduser().resolve())])
                run = {**case, "seed": seed, "repeat": repeat, "cache_start": args.cache_start,
                       "output_dir": str(destination), "command": command, "source_contract_sha256": source_hash}
                run["command_sha256"] = canonical_hash(command)
                runs.append(run)
    identity = {"schema": PLAN_SCHEMA, "study_root": str(study), "output_root": str(output),
                "settings": {"seeds": args.seeds, "technical_repeats": args.repeats, "cache_start": args.cache_start,
                             "order_seed": args.order_seed, "device": args.device, "num_threads": args.num_threads,
                             "operations": args.operations,
                             "warmup_jobs": args.warmup_jobs, "parity_atol": args.parity_atol},
                "frozen_inputs_sha256": inputs, "suite_sources_sha256": sources,
                "source_contract_sha256": source_hash, "cases": cases, "runs": runs}
    identity["plan_sha256"] = canonical_hash(identity)
    return identity


def report_setting(report: dict[str, Any], key: str) -> Any:
    return report.get("settings", {}).get(key, report.get(key))


def verify_report(run: dict[str, Any], *, require_receipt: bool = False) -> dict[str, Any]:
    destination = Path(run["output_dir"])
    report_path = destination / "online_benchmark.json"
    timings_path = destination / "request_timings.csv"
    if not report_path.is_file() or not timings_path.is_file():
        raise ValueError(f"Incomplete run at {destination}; inspect it and use a new output root rather than silently skipping it")
    report = read_json(report_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"Unsupported benchmark report schema: {report_path}")
    if report.get("complete_test_period") is not True or report.get("offline_parity", {}).get("passed") is not True:
        raise ValueError(f"Full-period prediction parity is required: {report_path}")
    for check in ("archived_structural_parity", "live_input_structural_parity", "base_batch_one_parity", "cache_parity"):
        if report.get(check, {}).get("passed") is not True:
            raise ValueError(f"Verified {check} is required: {report_path}")
    if report.get("source_contract", {}).get("sha256") != run["source_contract_sha256"]:
        raise ValueError(f"Benchmark source differs from the immutable plan: {report_path}")
    for key in ("seed", "repeat", "operation", "cache_capacity_entries", "cache_start"):
        if report_setting(report, key) != run[key]:
            raise ValueError(f"Report {key} differs from the immutable plan: {report_path}")
    if report.get("request_timings_sha256") != sha256_file(timings_path):
        raise ValueError(f"Timing CSV checksum mismatch: {timings_path}")
    if require_receipt:
        receipt_path = destination / "suite_completion.json"
        if not receipt_path.is_file():
            raise ValueError(f"No verified completion receipt: {destination}; the suite will not silently reuse this directory")
        receipt = read_json(receipt_path)
        expected = {"command_sha256": run["command_sha256"], "report_sha256": sha256_file(report_path),
                    "request_timings_sha256": report["request_timings_sha256"]}
        if receipt != expected:
            raise ValueError(f"Completion receipt or run identity changed: {destination}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    output = Path(plan["output_root"])
    plan_path = output / "online_suite_plan.json"
    if args.dry_run:
        preview = {"schema": plan["schema"], "plan_sha256": plan["plan_sha256"],
                   "settings": plan["settings"], "cases": plan["cases"],
                   "verified_frozen_input_files": len(plan["frozen_inputs_sha256"]),
                   "serial_run_count": len(plan["runs"]), "first_child_command": plan["runs"][0]["command"],
                   "run_order": [{key: run[key] for key in ("case", "seed", "repeat")} for run in plan["runs"]]}
        print(json.dumps(preview, indent=2, ensure_ascii=True))
        print(f"Validated {len(plan['runs'])} serial runs; no files were created.")
        return 0
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError(f"A suite plan already exists: {plan_path}; use identical arguments with --resume")
        if read_json(plan_path) != plan:
            raise ValueError("The existing timing plan differs from the requested immutable inputs/commands. Use a new output root.")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty and has no matching suite plan: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(plan_path, plan)
    for index, run in enumerate(plan["runs"], start=1):
        destination = Path(run["output_dir"])
        if destination.exists() and any(destination.iterdir()):
            if not args.resume:
                raise FileExistsError(f"Existing run requires --resume: {destination}")
            verify_report(run, require_receipt=True)
            print(f"[VERIFIED {index}/{len(plan['runs'])}] {destination}", flush=True)
            continue
        print(f"[RUN {index}/{len(plan['runs'])}] {subprocess.list2cmdline(run['command'])}", flush=True)
        subprocess.run(run["command"], cwd=PROJECT_ROOT, check=True)
        report = verify_report(run)
        atomic_json(destination / "suite_completion.json", {
            "command_sha256": run["command_sha256"],
            "report_sha256": sha256_file(destination / "online_benchmark.json"),
            "request_timings_sha256": report["request_timings_sha256"],
        })
    from online_study.summarize import summarize_plan
    summarize_plan(plan_path, output / "summary")
    print(f"Completed {len(plan['runs'])} serial runs. Summary: {output / 'summary'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
