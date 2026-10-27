#!/usr/bin/env python3
"""Create a fictional, schema-compatible trace for protocol smoke tests only.

Requires numpy, pandas and PyYAML (the same loader dependencies as the paper
code). No torch, sklearn, private trace, checkpoint or GPU is required. Generated
targets are arbitrary simulated ratios, never evidence for paper accuracy.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.data import sha256_file


def synthetic_rows(count: int = 320, seed: int = 2026) -> list[dict[str, Any]]:
    if isinstance(count, bool) or count < 200 or count % 20:
        raise ValueError("Synthetic row count must be a multiple of 20 and at least 200")
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2026-01-01T08:00:00+08:00")
    train_end = int(count * 0.8)
    validation_end = train_end + int(count * 0.1)
    train_tail_cut = (int((train_end - 2) * 0.9) // 2) * 2
    validation_gate_cut = train_end + (int((validation_end - train_end - 2) * 0.7) // 2) * 2

    def submit_at(index: int) -> pd.Timestamp:
        # Every consecutive pair submits simultaneously.
        return origin + pd.Timedelta(minutes=10 * (index // 2))

    crossing = {
        train_end - 2: submit_at(train_end) + pd.Timedelta(minutes=15),
        train_end - 1: submit_at(train_end) + pd.Timedelta(minutes=15),
        validation_end - 2: submit_at(validation_end) + pd.Timedelta(minutes=15),
        validation_end - 1: submit_at(validation_end) + pd.Timedelta(minutes=15),
        train_tail_cut - 2: submit_at(train_tail_cut) + pd.Timedelta(minutes=5),
        train_tail_cut - 1: submit_at(train_tail_cut) + pd.Timedelta(minutes=5),
        validation_gate_cut - 2: submit_at(validation_gate_cut) + pd.Timedelta(minutes=5),
        validation_gate_cut - 1: submit_at(validation_gate_cut) + pd.Timedelta(minutes=5),
    }
    rows = []
    for index in range(count):
        submit = submit_at(index)
        start = submit + pd.Timedelta(seconds=30)
        completion = crossing.get(index, start + pd.Timedelta(minutes=2))
        user = f"synth_user_{(index // 2) % 6}"
        if index >= validation_end and index % 11 == 0:
            user = f"synth_new_test_user_{index}"
        if index in (train_end - 2, train_end - 1, train_end + 4):
            user = "synth_boundary_user"
        if index in (validation_end - 2, validation_end - 1, validation_end + 4):
            user = "synth_late_boundary_user"
        gpu = int((index // 2) % 3 != 0)
        if index in crossing or index in (train_end + 4, validation_end + 4):
            gpu = 1
        cpu = (4, 8, 16)[index % 3]
        memory_request = (32, 64, 128)[index % 3]
        memory = float(np.clip(0.25 + 0.1 * ((index // 2) % 4)
                               + rng.normal(0, 0.025), 0.02, 0.98))
        vram = float(np.clip(0.30 + 0.12 * ((index // 2) % 3)
                             + rng.normal(0, 0.025), 0.02, 0.98)) if gpu else ""
        elapsed_seconds = int((completion - start).total_seconds())
        hours, rest = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        script_type = f"new_{index}" if index % 13 == 0 else str((index // 2) % 4)
        script = ("#!/bin/bash\n# FICTIONAL SYNTHETIC SMOKE DATA\n"
                  f"python /gpfsnyu/home/{user}/demo/task_{script_type}.py --synthetic\n")
        rows.append({
            "jobid": str(9_000_000 + index), "jobname": f"synthetic-{index}",
            "user": user, "partition": "synthetic-gpu" if gpu else "synthetic-cpu",
            "submit": submit.isoformat(), "start1": start.isoformat(),
            "end1": completion.isoformat(), "state1": "COMPLETED",
            "elapsed": f"{hours:02}:{minutes:02}:{seconds:02}", "exitcode": "0:0",
            "cpu": cpu, "gpu": gpu, "mem": memory_request, "time": "02:00:00",
            "nodelist": f"synthetic_node_{index % 4}", "cpu_eff": 0.5,
            "mem_eff": memory * 0.8, "mem_max": memory, "mem_p95": memory * 0.9,
            "mem_capacity_max": memory * memory_request,
            "gpu_util": 0.6 if gpu else "", "vram_eff": float(vram) * 0.8 if gpu else "",
            "vram_max": vram, "vram_p95": float(vram) * 0.9 if gpu else "",
            "vram_capacity_max": float(vram) * 24 if gpu else "",
            "script_content": script, "workdir": f"/synthetic/{user}/work",
        })
    return rows


def generate_synthetic_trace(
    output_dir: Path, *, template_config: Path | None = None,
    count: int = 320, seed: int = 2026,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    template = (template_config or PROJECT_ROOT / "configs" / "paper.local.yaml").resolve()
    original = load_config(template)
    settings = copy.deepcopy(original)
    settings.pop("_config_path", None)
    data = settings["data"]
    split = settings["split"]
    if float(split.get("train_ratio", 0.8)) != 0.8 or float(split.get("validation_ratio", 0.1)) != 0.1:
        raise ValueError("Synthetic crossing fixtures require the paper's fixed 80/10/10 outer ratios")
    # Only this explicitly synthetic copy relaxes private-paper identity checks.
    removed = []
    for prefix, section in (("data", data), ("split", split)):
        for key in list(section):
            if key.startswith("expected_") and key != "expected_columns":
                removed.append(f"{prefix}.{key}")
                del section[key]
    rows = synthetic_rows(count, seed)
    fields = data.get("expected_columns")
    if not fields or set(fields) != set(rows[0]):
        raise ValueError("Template expected_columns does not match the documented synthetic schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_trace.csv"
    config_path = output_dir / "synthetic.config.json"
    manifest_path = output_dir / "synthetic_manifest.json"
    split_path = output_dir / "synthetic_split_manifest.csv"
    for target in (csv_path, config_path, manifest_path, split_path):
        if target.exists():
            raise FileExistsError(f"Synthetic destination already exists; choose a new directory: {target}")
    with csv_path.open("w", encoding=str(data.get("csv_encoding", "utf-8-sig")), newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    data["path"] = str(csv_path)
    split["manifest"] = str(split_path)
    split["create_if_missing"] = True
    split["require_completed_labels_at_boundaries"] = True
    split["boundary_completion_policy"] = "purge"
    settings.setdefault("qwen", {})["base_selection"] = {
        "mode": "train_tail", "fit_fraction": 0.90,
    }
    settings.setdefault("training", {})["device"] = "cpu"
    settings.setdefault("outputs", {})["root"] = str(output_dir / "synthetic_outputs")
    settings.setdefault("reliability_cascade", {})["require_independent_base_selection"] = True
    settings["synthetic_experiment"] = {
        "enabled": True, "seed": int(seed),
        "purpose": "protocol_smoke_only_not_paper_accuracy_evidence",
        "real_trace_used": False, "model_predictions_generated": False,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        # JSON is a YAML subset and is read by the unchanged common.config loader.
        json.dump(settings, handle, indent=2, ensure_ascii=False)
    manifest = {
        "synthetic": True,
        "purpose": "Fictional protocol test data; never paper-result evidence",
        "csv": str(csv_path), "config": str(config_path),
        "csv_sha256": sha256_file(csv_path), "template": str(template),
        "template_sha256": sha256_file(template), "rows": count, "seed": seed,
        "private_constraints_removed_in_synthetic_copy_only": sorted(removed),
        "scenarios": ["CPU/GPU mix", "repeated and novel users/scripts", "tied submissions",
                      "outer train/validation crossing completions", "inner checkpoint/gate crossing completions"],
        "no_model_training_or_predictions": True,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-config", type=Path)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(generate_synthetic_trace(args.output_dir, template_config=args.template_config,
                                             count=args.rows, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
