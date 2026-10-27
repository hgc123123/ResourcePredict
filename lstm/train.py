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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BiLSTM peak-utilization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    architecture = {
        "embedding_dim": int(get_value(config, "text.embedding_dim", 128)),
        "hidden_dim": int(get_value(config, "lstm.hidden_dim", 128)),
        "layers": int(get_value(config, "lstm.layers", 1)),
        "bidirectional": bool(get_value(config, "lstm.bidirectional", True)),
        "user_embedding_dim": int(get_value(config, "lstm.user_embedding_dim", 16)),
        "metadata_hidden_dim": int(get_value(config, "lstm.metadata_hidden_dim", 32)),
        "head_hidden_dim": int(get_value(config, "lstm.head_hidden_dim", 128)),
        "dropout": float(get_value(config, "lstm.dropout", 0.3)),
    }
    output = train_script_model(
        config,
        bundle,
        model_name="lstm",
        model_builder=build_model,
        architecture=architecture,
    )
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()

