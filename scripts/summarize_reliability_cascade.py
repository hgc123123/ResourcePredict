#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reliability_cascade.integrity import verify_run_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen cascade policies and route rates across seeds"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root.resolve()
    )
    destinations = {
        "routing": output / "cascade_routing_runs.csv",
        "routing_summary": output / "cascade_routing_summary.csv",
        "policies": output / "cascade_policy_summary.csv",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Summary outputs already exist: " + ", ".join(str(path) for path in existing)
        )

    routing_rows = []
    policy_rows = []
    identities = set()
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not str(manifest.get("model", "")).startswith("reliability_cascade_"):
            continue
        run = manifest_path.parent
        verify_run_directory(run)
        routing_path = run / "routing_metrics.csv"
        policy_path = run / "selected_policy.json"
        if not routing_path.is_file() or not policy_path.is_file():
            raise FileNotFoundError(f"Incomplete cascade artifact: {run}")
        seed = int(manifest["seed"])
        split_fingerprint = str(
            manifest.get("split_fingerprint")
            or (manifest.get("protocol") or {}).get("split_fingerprint")
        )
        identities.add((str(manifest["data_sha256"]), split_fingerprint))
        routing = pd.read_csv(routing_path)
        routing.insert(0, "seed", seed)
        routing["run"] = str(run.relative_to(root)).replace("\\", "/")
        routing_rows.append(routing)
        with policy_path.open("r", encoding="utf-8") as handle:
            policy = json.load(handle)
        for resource in ("memory", "vram"):
            selected = policy["selected"][resource]
            policy_rows.append(
                {
                    "seed": seed,
                    "resource": resource,
                    "history_expert": selected["history_expert"],
                    "reliability_threshold": selected["reliability_threshold"],
                    "fit_rows": selected["fit_rows"],
                    "fit_reliable_rate": selected["fit_reliable_rate"],
                    "objective": policy["objective"],
                    "refit_on_full_validation": policy["refit_on_full_validation"],
                    "run": str(run.relative_to(root)).replace("\\", "/"),
                }
            )
    if not routing_rows:
        raise FileNotFoundError(f"No reliability-cascade manifests found under {root}")
    if len(identities) != 1:
        raise ValueError("Cascade runs reference multiple datasets or frozen splits")

    routing_runs = pd.concat(routing_rows, ignore_index=True).sort_values(
        ["split", "scope", "resource", "seed"], kind="mergesort"
    )
    numeric = [
        "history_bypass_rate",
        "qwen_required_rate",
        "embedding_cache_hit_rate_among_qwen",
        "encoder_forward_required_rate",
    ]
    routing_summary = (
        routing_runs.groupby(["split", "scope", "resource"], dropna=False)[numeric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    routing_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in routing_summary.columns
    ]
    policies = pd.DataFrame(policy_rows).sort_values(
        ["resource", "seed"], kind="mergesort"
    )
    output.mkdir(parents=True, exist_ok=True)
    routing_runs.to_csv(destinations["routing"], index=False)
    routing_summary.to_csv(destinations["routing_summary"], index=False)
    policies.to_csv(destinations["policies"], index=False)
    print(routing_summary.to_string(index=False))
    print(f"Policy summary: {destinations['policies']}")


if __name__ == "__main__":
    main()
