#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import output_directory  # noqa: E402
from common.config import get_value, load_config  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.neural import evaluate_script_checkpoint  # noqa: E402
from model import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved TextCNN checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    # The kernel maximum is read from the saved architecture to keep collation compatible.
    seed = int(get_value(config, "seed", 42))
    output_root = output_directory(config, "textcnn", f"seed_{seed}")
    architecture_path = output_root / "architecture.json"
    with architecture_path.open("r", encoding="utf-8") as handle:
        architecture = json.load(handle)
    output = evaluate_script_checkpoint(
        config,
        bundle,
        "textcnn",
        build_model,
        minimum_length=max(architecture["kernel_sizes"]),
    )
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
