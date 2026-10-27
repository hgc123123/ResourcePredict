#!/usr/bin/env python3
"""Run Qwen training followed by validation/test evaluation of the best checkpoint."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import output_directory  # noqa: E402
from common.config import get_value, load_config, resolve_path  # noqa: E402


VARIANTS = (
    "full",
    "code_only",
    "no_code",
    "no_user",
    "no_metadata",
    "concat_fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Qwen peak-utilization model, then evaluate the best checkpoint "
            "on validation and test"
        )
    )
    parser.add_argument("--config", required=True, help="Shared YAML experiment configuration")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument(
        "--output-dir",
        help="Run directory; defaults to outputs/qwen/<variant>/seed_<seed>",
    )
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Evaluate an already existing checkpoint in the resolved run directory",
    )
    parser.add_argument(
        "--eval-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the final single-process evaluation",
    )
    parser.add_argument("--base-model", help="Override the base model when loading a LoRA adapter")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; may be repeated",
    )
    return parser.parse_args()


def _append_overrides(command: list[str], overrides: Sequence[str]) -> None:
    for override in overrides:
        command.extend(("--set", override))


def _run(command: Sequence[str]) -> None:
    print(f"[RUN] {shlex.join(command)}", flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    config_path = Path(config["_config_path"])
    seed = int(args.seed if args.seed is not None else get_value(config, "seed", 42))

    if args.output_dir:
        run_dir = resolve_path(config, args.output_dir)
    else:
        run_dir = output_directory(config, "qwen", args.variant) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoint"

    if not args.skip_train:
        train_command = [
            sys.executable,
            "-m",
            "qwen.train",
            "--config",
            str(config_path),
            "--variant",
            args.variant,
            "--output-dir",
            str(run_dir),
        ]
        _append_overrides(train_command, args.overrides)
        if args.seed is not None:
            train_command.extend(("--seed", str(args.seed)))
        if args.no_lora:
            train_command.append("--no-lora")
        if args.overwrite:
            train_command.append("--overwrite")
        _run(train_command)

    if not (checkpoint_dir / "model_config.json").is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_dir}")

    evaluate_command = [
        sys.executable,
        "-m",
        "qwen.evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_dir),
        "--output-dir",
        str(run_dir),
        "--device",
        args.eval_device,
    ]
    _append_overrides(evaluate_command, args.overrides)
    if args.no_lora and not args.skip_train:
        evaluate_command.extend(("--set", "qwen.lora.enabled=false"))
    if args.base_model:
        evaluate_command.extend(("--base-model", args.base_model))
    _run(evaluate_command)

    print(f"Qwen run complete: {run_dir}")
    print(f"Predictions: {run_dir / 'predictions.csv'}")
    print(f"Metrics: {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
