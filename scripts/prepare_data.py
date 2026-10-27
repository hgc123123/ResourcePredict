#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.data import load_dataset, write_data_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a split manifest and write a data audit report")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    bundle = load_dataset(config)
    output_root = resolve_path(config, get_value(config, "outputs.root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "data_report.json"
    write_data_report(bundle, report_path)
    print(json.dumps(bundle.protocol, indent=2))
    print(f"Split manifest: {bundle.manifest_path}")
    print(f"Data report: {report_path}")


if __name__ == "__main__":
    main()

