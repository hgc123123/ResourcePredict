#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible experiment pipeline in order")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--with-qwen", action="store_true")
    parser.add_argument("--with-hybrid", action="store_true")
    parser.add_argument(
        "--with-safe-hybrid",
        action="store_true",
        help="Evaluate the validation-constrained Pareto-safe fixed blend",
    )
    parser.add_argument(
        "--with-guarded-hybrid",
        action="store_true",
        help="Evaluate the trust-region guard over the frozen safe blend",
    )
    parser.add_argument(
        "--with-script-guarded-hybrid",
        action="store_true",
        help="Evaluate script-stratified trust budgets over the frozen safe blend",
    )
    parser.add_argument(
        "--with-script-history-hybrid",
        action="store_true",
        help="Evaluate the validation-safe Qwen/Last-1/same-script blend",
    )
    parser.add_argument(
        "--with-dynamic-hybrid",
        action="store_true",
        help="Fit/evaluate the resource-specific causal Qwen/Last-1 gate",
    )
    parser.add_argument(
        "--overwrite-qwen",
        action="store_true",
        help="Explicitly replace the selected Qwen run checkpoint",
    )
    parser.add_argument(
        "--overwrite-dynamic-hybrid",
        action="store_true",
        help="Explicitly replace the selected dynamic-hybrid checkpoint",
    )
    parser.add_argument(
        "--qwen-variant",
        choices=(
            "full",
            "code_only",
            "no_code",
            "no_user",
            "no_metadata",
            "concat_fusion",
        ),
        default="full",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _entrypoint(directory: str, dry_run: bool) -> Path:
    root = PROJECT_ROOT / directory
    candidates = [root / "run.py", root / "train.py"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if dry_run:
        return candidates[0]
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No entrypoint found for {directory}; expected one of: {expected}")


def _config_arguments(config: Path, overrides: Iterable[str]) -> list[str]:
    result = ["--config", str(config)]
    for value in overrides:
        result.extend(["--set", value])
    return result


def _display(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run(command: Sequence[str], dry_run: bool) -> None:
    print(f"[RUN] {_display(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = Path(args.config).expanduser().resolve()
    if not config.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Configuration not found: {config}")
    shared = _config_arguments(config, args.overrides)

    commands: list[list[str]] = []
    if not args.skip_prepare:
        commands.append([sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_data.py"), *shared])
    for model_dir in ("last-1", "xgboost", "textcnn", "lstm"):
        commands.append([sys.executable, str(_entrypoint(model_dir, args.dry_run)), *shared])
    if args.with_qwen:
        qwen_command = [
            sys.executable,
            str(_entrypoint("qwen", args.dry_run)),
            *shared,
            "--variant",
            args.qwen_variant,
        ]
        if args.overwrite_qwen:
            qwen_command.append("--overwrite")
        commands.append(qwen_command)
    if args.with_hybrid:
        commands.append(
            [
                sys.executable,
                str(_entrypoint("hybrid", args.dry_run)),
                *shared,
                "--set",
                f"hybrid.qwen_variant={args.qwen_variant}",
            ]
        )
    if args.with_safe_hybrid:
        commands.append(
            [
                sys.executable,
                str(_entrypoint("hybrid", args.dry_run)),
                *shared,
                "--set",
                f"hybrid.qwen_variant={args.qwen_variant}",
                "--set",
                "hybrid.selection_objective=constrained_mae",
            ]
        )
    if args.with_guarded_hybrid:
        commands.append(
            [
                sys.executable,
                str(_entrypoint("hybrid", args.dry_run)),
                *shared,
                "--set",
                f"hybrid.qwen_variant={args.qwen_variant}",
                "--set",
                "hybrid.selection_objective=guarded_constrained_acc",
            ]
        )
    if args.with_script_guarded_hybrid:
        commands.append(
            [
                sys.executable,
                str(_entrypoint("hybrid", args.dry_run)),
                *shared,
                "--set",
                f"hybrid.qwen_variant={args.qwen_variant}",
                "--set",
                "hybrid.selection_objective=script_guarded_constrained_acc",
            ]
        )
    if args.with_script_history_hybrid:
        commands.append(
            [
                sys.executable,
                str(_entrypoint("hybrid", args.dry_run)),
                *shared,
                "--set",
                f"hybrid.qwen_variant={args.qwen_variant}",
                "--set",
                "hybrid.selection_objective=script_history_constrained_mae",
            ]
        )
    if args.with_dynamic_hybrid:
        dynamic_command = [
            sys.executable,
            str(_entrypoint("dynamic_hybrid", args.dry_run)),
            *shared,
            "--set",
            f"dynamic_hybrid.qwen_variant={args.qwen_variant}",
        ]
        if args.overwrite_dynamic_hybrid:
            dynamic_command.append("--overwrite")
        commands.append(dynamic_command)
    if not args.skip_aggregate:
        aggregate = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "aggregate_results.py"),
            *shared,
        ]
        selected_models = ["last-1", "xgboost", "textcnn", "lstm"]
        if args.with_qwen:
            selected_models.append(f"qwen_{args.qwen_variant}")
        if args.with_hybrid:
            selected_models.append(f"hybrid_{args.qwen_variant}")
        if args.with_safe_hybrid:
            selected_models.append(f"safe_hybrid_{args.qwen_variant}")
        if args.with_guarded_hybrid:
            selected_models.append(f"guarded_hybrid_{args.qwen_variant}")
        if args.with_script_guarded_hybrid:
            selected_models.append(f"script_guarded_hybrid_{args.qwen_variant}")
        if args.with_script_history_hybrid:
            selected_models.append(f"safe_script_history_hybrid_{args.qwen_variant}")
        if args.with_dynamic_hybrid:
            selected_models.append(f"dynamic_hybrid_{args.qwen_variant}")
        for model in selected_models:
            aggregate.extend(["--model", model])
        commands.append(aggregate)

    for command in commands:
        _run(command, args.dry_run)


if __name__ == "__main__":
    main()
