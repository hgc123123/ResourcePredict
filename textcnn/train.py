#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.neural import train_script_model  # noqa: E402
from model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TextCNN peak-utilization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    kernels = list(get_value(config, "textcnn.kernel_sizes", [3, 4, 5]))
    architecture = {
        "embedding_dim": int(get_value(config, "text.embedding_dim", 128)),
        "kernel_sizes": kernels,
        "filters_per_kernel": int(get_value(config, "textcnn.filters_per_kernel", 64)),
        "user_embedding_dim": int(get_value(config, "textcnn.user_embedding_dim", 16)),
        "metadata_hidden_dim": int(get_value(config, "textcnn.metadata_hidden_dim", 32)),
        "hidden_dim": int(get_value(config, "textcnn.hidden_dim", 128)),
        "dropout": float(get_value(config, "textcnn.dropout", 0.5)),
    }
    output = train_script_model(
        config,
        bundle,
        model_name="textcnn",
        model_builder=build_model,
        architecture=architecture,
        minimum_length=max(kernels),
    )
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()

