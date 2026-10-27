#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_VARIANTS = (
    "full",
    "code_only",
    "no_code",
    "no_user",
    "no_metadata",
    "concat_fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen modality ablations with one shared protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="+", choices=ALLOWED_VARIANTS, default=list(ALLOWED_VARIANTS))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for variant in args.variants:
        command = [
            sys.executable,
            "-m",
            "qwen.run",
            "--config",
            args.config,
            "--variant",
            variant,
        ]
        for override in args.overrides:
            command.extend(["--set", override])
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
