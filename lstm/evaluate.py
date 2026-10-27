#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.neural import evaluate_script_checkpoint  # noqa: E402
from model import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved BiLSTM checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    output = evaluate_script_checkpoint(config, bundle, "lstm", build_model)
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()

