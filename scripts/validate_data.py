#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config  # noqa: E402
from common.data import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trace schema and experiment protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when chronology or completion-time checks are unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    print(json.dumps(bundle.protocol, indent=2))
    if not bundle.protocol["verified_chronological"]:
        message = (
            "split.order_semantics is not submission_time; "
            "the run is not a verified chronological evaluation"
        )
        if args.strict:
            raise ValueError(message)
        print(f"WARNING: {message}.", file=sys.stderr)
    if not bundle.protocol["has_completion_time"]:
        message = "completion timestamps are absent; strict causal Last-1 is unavailable"
        if args.strict:
            raise ValueError(message)
        print(f"WARNING: {message}.", file=sys.stderr)


if __name__ == "__main__":
    main()

