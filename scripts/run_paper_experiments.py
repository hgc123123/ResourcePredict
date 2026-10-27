#!/usr/bin/env python3
"""Build and run an auditable Qwen paper-experiment matrix.

The runner deliberately does not read metrics or select a configuration. All
model and blend selection remains validation-only in the underlying trainers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "full",
    "code_only",
    "no_code",
    "no_user",
    "no_metadata",
    "concat_fusion",
)
CAUSAL_LAST1_PROTOCOL = "completed_before_submission"
SCRIPT_HISTORY_OBJECTIVE = "script_history_constrained_mae"
PLAN_SCHEMA_VERSION = 3
PLAN_SOURCE_DIRECTORIES = ("common", "hybrid")
PLAN_SOURCE_FILES = (
    "scripts/run_paper_experiments.py",
    "qwen/contract.py",
    "qwen/dataset.py",
    "qwen/evaluate.py",
    "qwen/model.py",
    "qwen/train.py",
    "last-1/run.py",
    "requirements.txt",
    "requirements-qwen.txt",
)


@dataclass(frozen=True)
class ExperimentSpec:
    config: Path
    output_root: Path
    seeds: tuple[int, ...]
    variants: tuple[str, ...]
    num_processes: int
    model_path: str | None
    max_tokens: tuple[int, ...]
    batch_size: int | None
    eval_batch_size: int | None
    gradient_accumulation_steps: int | None


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    stage: str
    command: tuple[str, ...]
    depends_on: tuple[str, ...]
    seed: int
    output_dir: Path
    variant: str | None = None
    max_tokens: int | None = None

    def as_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "stage": self.stage,
            "depends_on": list(self.depends_on),
            "seed": self.seed,
            "variant": self.variant,
            "max_tokens": self.max_tokens,
            "output_dir": str(self.output_dir),
            "command": list(self.command),
        }


@dataclass(frozen=True)
class CompletionStatus:
    complete: bool
    reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local-Qwen paper experiments without using test metrics for selection"
        )
    )
    parser.add_argument("--config", required=True, help="Shared paper YAML configuration")
    parser.add_argument(
        "--output-root",
        default="outputs/paper_experiments",
        help="Dedicated root for this immutable experiment matrix",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=("40", "41", "42", "43", "44"),
        metavar="SEED",
        help="Space- or comma-separated seeds (default: 40 41 42 43 44)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("full",),
        metavar="VARIANT",
        help="Space- or comma-separated Qwen modality variants",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=3,
        help="Processes/GPUs for single-node accelerate training (default: 3)",
    )
    parser.add_argument(
        "--model-path",
        help="Local Qwen model directory; when omitted, use qwen.model_name_or_path",
    )
    parser.add_argument(
        "--max-tokens",
        nargs="+",
        default=("4096",),
        metavar="TOKENS",
        help="Space- or comma-separated context lengths",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-process Qwen training batch size; otherwise use the YAML value",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Single-process Qwen evaluation batch size; otherwise use the YAML value",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Qwen gradient accumulation; otherwise use the YAML value",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete plan without writing files or launching commands",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps whose complete artifact set already exists under the same plan",
    )
    return parser.parse_args(argv)


def _split_cli_values(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for raw in values:
        for value in str(raw).split(","):
            stripped = value.strip()
            if stripped:
                result.append(stripped)
    return result


def _unique(values: Iterable[object]) -> tuple:
    return tuple(dict.fromkeys(values))


def _positive_int_values(values: Iterable[object], name: str, *, allow_zero: bool) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in _split_cli_values(values):
        try:
            integer = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must contain integers; got {value!r}") from exc
        minimum = 0 if allow_zero else 1
        if integer < minimum:
            comparison = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} values must be {comparison}; got {integer}")
        parsed.append(integer)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return _unique(parsed)


def _variant_values(values: Iterable[object]) -> tuple[str, ...]:
    variants = _unique(_split_cli_values(values))
    if not variants:
        raise ValueError("--variants must not be empty")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}; expected one or more of {list(VARIANTS)}"
        )
    return variants


def make_spec(args: argparse.Namespace) -> ExperimentSpec:
    config = Path(args.config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Configuration not found: {config}")

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()

    if args.num_processes <= 0:
        raise ValueError("--num-processes must be positive")
    for name in ("batch_size", "eval_batch_size", "gradient_accumulation_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    seeds = _positive_int_values(args.seeds, "--seeds", allow_zero=True)
    max_tokens = _positive_int_values(
        args.max_tokens, "--max-tokens", allow_zero=False
    )
    model_path = str(args.model_path).strip() if args.model_path else None
    if model_path == "":
        model_path = None

    return ExperimentSpec(
        config=config,
        output_root=output_root,
        seeds=seeds,
        variants=_variant_values(args.variants),
        num_processes=int(args.num_processes),
        model_path=model_path,
        max_tokens=max_tokens,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def _append_override(command: list[str], key: str, value: object) -> None:
    command.extend(("--set", f"{key}={value}"))


def _qwen_overrides(spec: ExperimentSpec, max_tokens: int) -> list[tuple[str, object]]:
    values: list[tuple[str, object]] = [
        ("qwen.max_tokens", max_tokens),
        ("outputs.root", str(spec.output_root)),
    ]
    if spec.model_path:
        values.append(("qwen.model_name_or_path", spec.model_path))
    if spec.batch_size is not None:
        values.append(("qwen.batch_size", spec.batch_size))
    if spec.eval_batch_size is not None:
        values.append(("qwen.eval_batch_size", spec.eval_batch_size))
    if spec.gradient_accumulation_steps is not None:
        values.append(
            ("qwen.gradient_accumulation_steps", spec.gradient_accumulation_steps)
        )
    return values


def _last1_output(spec: ExperimentSpec, seed: int) -> Path:
    return (
        spec.output_root
        / "last-1"
        / CAUSAL_LAST1_PROTOCOL
        / f"seed_{seed}"
    )


def _qwen_output(
    spec: ExperimentSpec, max_tokens: int, variant: str, seed: int
) -> Path:
    return (
        spec.output_root
        / "qwen"
        / f"tokens_{max_tokens}"
        / variant
        / f"seed_{seed}"
    )


def _hybrid_output(
    spec: ExperimentSpec, max_tokens: int, variant: str, seed: int
) -> Path:
    return (
        spec.output_root
        / "safe_script_history_hybrid"
        / CAUSAL_LAST1_PROTOCOL
        / f"tokens_{max_tokens}"
        / variant
        / f"seed_{seed}"
    )


def _last1_step(spec: ExperimentSpec, seed: int) -> PlanStep:
    output_dir = _last1_output(spec, seed)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "last-1" / "run.py"),
        "--config",
        str(spec.config),
    ]
    _append_override(command, "seed", seed)
    _append_override(command, "outputs.root", spec.output_root)
    _append_override(command, "last1.protocol", CAUSAL_LAST1_PROTOCOL)
    return PlanStep(
        step_id=f"last1.seed_{seed}",
        stage="last1",
        command=tuple(command),
        depends_on=(),
        seed=seed,
        output_dir=output_dir,
    )


def _train_step(
    spec: ExperimentSpec, max_tokens: int, variant: str, seed: int
) -> PlanStep:
    output_dir = _qwen_output(spec, max_tokens, variant, seed)
    command = [sys.executable, "-m", "accelerate.commands.launch"]
    if spec.num_processes > 1:
        command.append("--multi_gpu")
    command.extend(
        (
            "--num_machines",
            "1",
            "--num_processes",
            str(spec.num_processes),
            "--main_process_port",
            "0",
            "-m",
            "qwen.train",
            "--config",
            str(spec.config),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        )
    )
    for key, value in _qwen_overrides(spec, max_tokens):
        _append_override(command, key, value)
    return PlanStep(
        step_id=f"qwen.train.tokens_{max_tokens}.{variant}.seed_{seed}",
        stage="qwen_train",
        command=tuple(command),
        depends_on=(),
        seed=seed,
        variant=variant,
        max_tokens=max_tokens,
        output_dir=output_dir,
    )


def _evaluate_step(
    spec: ExperimentSpec,
    max_tokens: int,
    variant: str,
    seed: int,
    train_step: PlanStep,
) -> PlanStep:
    output_dir = _qwen_output(spec, max_tokens, variant, seed)
    command = [
        sys.executable,
        "-m",
        "qwen.evaluate",
        "--config",
        str(spec.config),
        "--checkpoint",
        str(output_dir / "checkpoint"),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
    ]
    if spec.model_path:
        command.extend(("--base-model", spec.model_path))
    for key, value in _qwen_overrides(spec, max_tokens):
        _append_override(command, key, value)
    return PlanStep(
        step_id=f"qwen.evaluate.tokens_{max_tokens}.{variant}.seed_{seed}",
        stage="qwen_evaluate",
        command=tuple(command),
        depends_on=(train_step.step_id,),
        seed=seed,
        variant=variant,
        max_tokens=max_tokens,
        output_dir=output_dir,
    )


def _hybrid_step(
    spec: ExperimentSpec,
    max_tokens: int,
    variant: str,
    seed: int,
    evaluate_step: PlanStep,
    last1_step: PlanStep,
) -> PlanStep:
    qwen_output = _qwen_output(spec, max_tokens, variant, seed)
    output_dir = _hybrid_output(spec, max_tokens, variant, seed)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "hybrid" / "run.py"),
        "--config",
        str(spec.config),
        "--qwen-predictions",
        str(qwen_output / "predictions.csv"),
        "--last1-predictions",
        str(last1_step.output_dir / "predictions.csv"),
        "--output-dir",
        str(output_dir),
    ]
    overrides = (
        ("seed", seed),
        ("outputs.root", str(spec.output_root)),
        ("last1.protocol", CAUSAL_LAST1_PROTOCOL),
        ("hybrid.qwen_variant", variant),
        ("hybrid.selection_objective", SCRIPT_HISTORY_OBJECTIVE),
        ("hybrid.alpha_grid_step", 0.01),
        ("hybrid.same_script_beta_values", "[0.0,0.25,0.5,0.75,1.0]"),
        ("hybrid.same_script_min_effective_n", 100),
        ("hybrid.max_acc_drop_vs_reference", 0.0),
        ("hybrid.max_rmse_increase_vs_reference", 0.0),
        ("hybrid.max_mae_increase_vs_reference", 0.0),
    )
    for key, value in overrides:
        _append_override(command, key, value)
    for key, value in _qwen_overrides(spec, max_tokens):
        _append_override(command, key, value)
    return PlanStep(
        step_id=f"hybrid.tokens_{max_tokens}.{variant}.seed_{seed}",
        stage="safe_script_history_hybrid",
        command=tuple(command),
        depends_on=(evaluate_step.step_id, last1_step.step_id),
        seed=seed,
        variant=variant,
        max_tokens=max_tokens,
        output_dir=output_dir,
    )


def build_plan(spec: ExperimentSpec) -> list[PlanStep]:
    steps: list[PlanStep] = []
    last1_steps: dict[int, PlanStep] = {}
    for seed in spec.seeds:
        step = _last1_step(spec, seed)
        last1_steps[seed] = step
        steps.append(step)

    for max_tokens in spec.max_tokens:
        for variant in spec.variants:
            for seed in spec.seeds:
                train = _train_step(spec, max_tokens, variant, seed)
                evaluate = _evaluate_step(
                    spec, max_tokens, variant, seed, train
                )
                hybrid = _hybrid_step(
                    spec,
                    max_tokens,
                    variant,
                    seed,
                    evaluate,
                    last1_steps[seed],
                )
                steps.extend((train, evaluate, hybrid))

    step_ids = [step.step_id for step in steps]
    if len(step_ids) != len(set(step_ids)):
        raise RuntimeError("Experiment plan contains duplicate step identifiers")
    qwen_and_hybrid_outputs = [
        str(step.output_dir)
        for step in steps
        if step.stage in {"qwen_train", "safe_script_history_hybrid"}
    ]
    if len(qwen_and_hybrid_outputs) != len(set(qwen_and_hybrid_outputs)):
        raise RuntimeError("Experiment plan contains colliding model output directories")
    return steps


def _source_fingerprints() -> dict[str, str]:
    paths = [PROJECT_ROOT / value for value in PLAN_SOURCE_FILES]
    for directory in PLAN_SOURCE_DIRECTORIES:
        paths.extend((PROJECT_ROOT / directory).rglob("*.py"))
    fingerprints: dict[str, str] = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            raise FileNotFoundError(f"Planned source file not found: {path}")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def plan_payload(spec: ExperimentSpec, steps: Sequence[PlanStep]) -> dict:
    model_inventory_sha256 = os.environ.get("BASE_MODEL_INVENTORY_SHA256")
    if model_inventory_sha256 is not None:
        model_inventory_sha256 = model_inventory_sha256.strip().lower()
        if len(model_inventory_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in model_inventory_sha256
        ):
            raise ValueError(
                "BASE_MODEL_INVENTORY_SHA256 must be a SHA-256 hex digest"
            )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "config": str(spec.config),
        "config_sha256": hashlib.sha256(spec.config.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprints(),
        "output_root": str(spec.output_root),
        "seeds": list(spec.seeds),
        "variants": list(spec.variants),
        "max_tokens": list(spec.max_tokens),
        "num_processes": spec.num_processes,
        "model_path": spec.model_path,
        "base_model_inventory_sha256": model_inventory_sha256,
        "batch_size_per_process_override": spec.batch_size,
        "eval_batch_size_override": spec.eval_batch_size,
        "gradient_accumulation_steps_override": spec.gradient_accumulation_steps,
        "declared_effective_global_batch_size": (
            spec.batch_size
            * spec.gradient_accumulation_steps
            * spec.num_processes
            if spec.batch_size is not None
            and spec.gradient_accumulation_steps is not None
            else None
        ),
        "last1_protocol": CAUSAL_LAST1_PROTOCOL,
        "selection_policy": {
            "selection_split": "validation",
            "primary_metric": "macro_average_validation_mae",
            "test_split_role": "report_only",
            "runner_reads_metric_files": False,
            "hybrid_objective": SCRIPT_HISTORY_OBJECTIVE,
            "note": (
                "Freeze the validation-selected configuration before inspecting or "
                "claiming performance on a new chronological holdout."
            ),
        },
        "steps": [step.as_dict() for step in steps],
    }


def _write_plan(spec: ExperimentSpec, payload: dict) -> Path:
    spec.output_root.mkdir(parents=True, exist_ok=True)
    path = spec.output_root / "paper_experiment_plan.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise FileExistsError(
                f"A different experiment plan already exists at {path}; use a new --output-root"
            )
        return path
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _display(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def _read_json_object(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty JSON artifact: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def _require_nonempty_files(directory: Path, names: Sequence[str]) -> None:
    missing = [
        name
        for name in names
        if not (directory / name).is_file() or (directory / name).stat().st_size <= 0
    ]
    if missing:
        raise ValueError(f"missing or empty artifacts: {', '.join(missing)}")


def _validate_qwen_training_completion(step: PlanStep) -> dict:
    checkpoint = step.output_dir / "checkpoint"
    _read_json_object(checkpoint / "model_config.json")
    manifest = _read_json_object(checkpoint / "training_manifest.json")
    completion = _read_json_object(step.output_dir / "training_complete.json")

    if completion.get("status") != "complete":
        raise ValueError("training_complete.json does not declare status=complete")
    if int(manifest.get("seed", -1)) != step.seed:
        raise ValueError("training manifest seed differs from the planned seed")
    if manifest.get("variant") != step.variant:
        raise ValueError("training manifest variant differs from the planned variant")
    effective = manifest.get("effective_parameters") or {}
    if int(effective.get("max_tokens", -1)) != step.max_tokens:
        raise ValueError("training manifest max_tokens differs from the planned context")
    contract = manifest.get("experiment_contract") or {}
    contract_sha = str(contract.get("sha256", ""))
    if not contract_sha or completion.get("experiment_contract_sha256") != contract_sha:
        raise ValueError("training completion marker disagrees with the experiment contract")
    best_epoch = int(manifest.get("best_epoch", -1))
    if int(completion.get("best_epoch", -2)) != best_epoch or best_epoch <= 0:
        raise ValueError("training completion marker disagrees with the best epoch")
    completed_epochs = int(completion.get("completed_epochs", -1))
    if completed_epochs < best_epoch:
        raise ValueError("training completion marker has an invalid completed epoch count")
    runtime = manifest.get("training_runtime")
    if not isinstance(runtime, dict) or int(runtime.get("completed_epochs", -1)) != completed_epochs:
        raise ValueError("training manifest has no matching final runtime record")
    return manifest


def completion_status(step: PlanStep) -> CompletionStatus:
    try:
        if step.stage == "last1":
            _require_nonempty_files(step.output_dir, ("predictions.csv", "metrics.csv"))
            manifest = _read_json_object(step.output_dir / "run_manifest.json")
            details = manifest.get("details") or {}
            if manifest.get("model") != "last-1" or int(manifest.get("seed", -1)) != step.seed:
                raise ValueError("Last-1 manifest identity differs from the plan")
            if (
                details.get("protocol") != CAUSAL_LAST1_PROTOCOL
                or details.get("causal_online_protocol") is not True
            ):
                raise ValueError("Last-1 manifest is not the required causal protocol")
        elif step.stage == "qwen_train":
            _validate_qwen_training_completion(step)
        elif step.stage == "qwen_evaluate":
            training_manifest = _validate_qwen_training_completion(step)
            _require_nonempty_files(step.output_dir, ("predictions.csv", "metrics.csv"))
            manifest = _read_json_object(step.output_dir / "run_manifest.json")
            details = manifest.get("details") or {}
            if (
                manifest.get("model") != f"qwen_{step.variant}"
                or int(manifest.get("seed", -1)) != step.seed
                or details.get("variant") != step.variant
            ):
                raise ValueError("Qwen evaluation manifest identity differs from the plan")
            contract_sha = (training_manifest.get("experiment_contract") or {}).get("sha256")
            if not contract_sha or details.get("experiment_contract_sha256") != contract_sha:
                raise ValueError("Qwen evaluation manifest uses a different training contract")
        elif step.stage == "safe_script_history_hybrid":
            _require_nonempty_files(
                step.output_dir,
                ("predictions.csv", "metrics.csv", "reference_metrics.csv"),
            )
            manifest = _read_json_object(step.output_dir / "run_manifest.json")
            policy = _read_json_object(step.output_dir / "script_history_policy.json")
            if (
                manifest.get("model") != f"safe_script_history_hybrid_{step.variant}"
                or int(manifest.get("seed", -1)) != step.seed
                or manifest.get("qwen_variant") != step.variant
            ):
                raise ValueError("hybrid manifest identity differs from the plan")
            if (
                manifest.get("selection_objective") != SCRIPT_HISTORY_OBJECTIVE
                or manifest.get("selection_split") != "validation"
                or manifest.get("test_labels_used_for_selection") is not False
            ):
                raise ValueError("hybrid manifest does not declare validation-only selection")
            if (
                policy.get("selection_objective") != SCRIPT_HISTORY_OBJECTIVE
                or policy.get("selection_split") != "validation"
                or policy.get("test_labels_used_for_selection") is not False
            ):
                raise ValueError("hybrid policy does not declare validation-only selection")
        else:
            raise ValueError(f"unknown plan stage: {step.stage}")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return CompletionStatus(False, str(exc))
    return CompletionStatus(True, "validated complete artifact set")


def _expected_output(spec: ExperimentSpec, step: PlanStep) -> Path:
    if step.stage == "last1":
        return _last1_output(spec, step.seed)
    if step.stage in {"qwen_train", "qwen_evaluate"}:
        if step.max_tokens is None or step.variant is None:
            raise RuntimeError("Qwen plan step has no context length or variant")
        return _qwen_output(spec, step.max_tokens, step.variant, step.seed)
    if step.stage == "safe_script_history_hybrid":
        if step.max_tokens is None or step.variant is None:
            raise RuntimeError("Hybrid plan step has no context length or variant")
        return _hybrid_output(spec, step.max_tokens, step.variant, step.seed)
    raise RuntimeError(f"Unknown plan stage: {step.stage}")


def _validated_managed_output(spec: ExperimentSpec, step: PlanStep) -> Path:
    root = spec.output_root.resolve()
    if step.output_dir.is_symlink():
        raise RuntimeError(f"Refuse to reset a symlinked run output: {step.output_dir}")
    expected = _expected_output(spec, step).resolve()
    actual = step.output_dir.resolve()
    if actual != expected:
        raise RuntimeError(
            f"Refuse to reset an output that differs from the planned path: {actual}"
        )
    try:
        relative = actual.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refuse to reset output outside {root}: {actual}") from exc
    if len(relative.parts) < 3:
        raise RuntimeError(f"Refuse to reset an overly broad output path: {actual}")
    return actual


def _reset_step_artifacts(
    spec: ExperimentSpec, step: PlanStep, *, dry_run: bool
) -> None:
    output = _validated_managed_output(spec, step)
    if step.stage == "qwen_evaluate":
        names = ("predictions.csv", "metrics.csv", "run_manifest.json")
        existing = [output / name for name in names if (output / name).exists()]
        if existing:
            print(
                f"[RESET] {step.step_id}: remove incomplete evaluation artifacts",
                flush=True,
            )
            if not dry_run:
                for path in existing:
                    path.unlink()
        return
    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"Planned run output is not a directory: {output}")
        print(
            f"[RESET] {step.step_id}: remove only this managed run directory: {output}",
            flush=True,
        )
        if not dry_run:
            shutil.rmtree(output)


def _dependent_steps(steps: Sequence[PlanStep], step_id: str) -> list[PlanStep]:
    dependency_ids = {step_id}
    result: list[PlanStep] = []
    for candidate in steps:
        if any(value in dependency_ids for value in candidate.depends_on):
            dependency_ids.add(candidate.step_id)
            result.append(candidate)
    return result


def execute_plan(
    spec: ExperimentSpec,
    steps: Sequence[PlanStep],
    *,
    dry_run: bool,
    resume: bool = False,
) -> None:
    prefix = "DRY-RUN" if dry_run else "RUN"
    environment = os.environ.copy()
    if spec.model_path:
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")

    executed_steps: set[str] = set()
    for step in steps:
        if resume:
            status = completion_status(step)
            upstream_executed = any(
                dependency in executed_steps for dependency in step.depends_on
            )
            if status.complete and not upstream_executed:
                print(
                    f"[SKIP] {step.step_id}: {status.reason}",
                    flush=True,
                )
                continue
            reason = (
                "an upstream dependency was rerun"
                if upstream_executed
                else status.reason
            )
            print(f"[RERUN] {step.step_id}: {reason}", flush=True)
            _reset_step_artifacts(spec, step, dry_run=dry_run)
            for dependent in _dependent_steps(steps, step.step_id):
                _reset_step_artifacts(spec, dependent, dry_run=dry_run)
        print(f"[{prefix}] {step.step_id}: {_display(step.command)}", flush=True)
        if not dry_run:
            subprocess.run(
                list(step.command),
                cwd=PROJECT_ROOT,
                check=True,
                env=environment,
            )
        executed_steps.add(step.step_id)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = make_spec(args)
    steps = build_plan(spec)
    payload = plan_payload(spec, steps)

    print(
        "[POLICY] This runner never reads metrics or selects from test results; "
        "configuration selection is validation-only.",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    else:
        plan_path = _write_plan(spec, payload)
        print(f"[PLAN] {plan_path}", flush=True)
    execute_plan(
        spec,
        steps,
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
