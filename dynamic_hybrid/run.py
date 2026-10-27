#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the dynamic hybrid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _shared_args(args: argparse.Namespace) -> list[str]:
    values = ["--config", args.config]
    if args.qwen_predictions:
        values.extend(["--qwen-predictions", args.qwen_predictions])
    if args.last1_predictions:
        values.extend(["--last1-predictions", args.last1_predictions])
    for override in args.overrides:
        values.extend(["--set", override])
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    component_seed = int(get_value(config, "seed", 42))
    gate_seed = int(get_value(config, "dynamic_hybrid.seed", component_seed))
    variant = str(
        get_value(
            config,
            "dynamic_hybrid.qwen_variant",
            get_value(config, "hybrid.qwen_variant", "full"),
        )
    )
    run_dir = resolve_path(
        config,
        args.output_dir
        or get_value(
            config,
            "dynamic_hybrid.output_dir",
            "outputs/dynamic_hybrid/completed_before_submission/"
            f"{variant}/"
            + (
                f"seed_{component_seed}"
                if gate_seed == component_seed
                else f"seed_{component_seed}_gate_{gate_seed}"
            ),
        ),
    )
    shared = _shared_args(args)
    train = [
        sys.executable,
        "-m",
        "dynamic_hybrid.train",
        *shared,
        "--output-dir",
        str(run_dir),
    ]
    if args.overwrite:
        train.append("--overwrite")
    subprocess.run(train, cwd=PROJECT_ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamic_hybrid.evaluate",
            *shared,
            "--checkpoint",
            str(run_dir / "checkpoint"),
            "--output-dir",
            str(run_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
