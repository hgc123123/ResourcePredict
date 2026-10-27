#!/usr/bin/env python3
"""Canonical entry point for the final manuscript's experiments (code-only release).

Planning needs only Python's standard library. Real runs require the external
trace/model; hardware stratification additionally requires external registries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (40, 41, 42, 43, 44)
CORE = ("prepare", "base", "cascade", "components", "diagnostics", "coverage")
TEMPORAL = ("temporal-0600", "temporal-0800", "temporal-1000")
ORDER = (*CORE, "primary-export", "baselines", "followup", "temporal-prepare",
         *TEMPORAL, "hardware", "online", "report")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=(*ORDER, "smoke", "all"), default="prepare")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--config", type=Path, default=ROOT / "configs/paper.local.yaml")
    p.add_argument("--data-csv", default=os.environ.get("JOB_EFFICIENCY_CSV"))
    p.add_argument("--model-path", default=os.environ.get("MODEL_DIR"))
    p.add_argument("--capacity-map", default=os.environ.get("GPU_CAPACITY_MAP"))
    p.add_argument("--model-map", default=os.environ.get("GPU_MODEL_MAP"))
    p.add_argument("--registry-provenance", default=os.environ.get("GPU_REGISTRY_PROVENANCE"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands only; no dependencies, data or writes")
    a = p.parse_args(argv)
    a.output_root = a.output_root.resolve()
    a.config = a.config.resolve()
    if a.output_root == ROOT or a.output_root in ROOT.parents:
        p.error("Choose a dedicated output directory, never the source root or its ancestors")
    if not a.config.is_file():
        p.error(f"Configuration not found: {a.config}")
    if not a.dry_run and a.stage in ("hardware", "all"):
        if not all((a.data_csv, a.capacity_map, a.model_map)):
            p.error("Hardware analysis requires --data-csv, --capacity-map and --model-map (or corresponding env vars)")
    return a


def command(*args):
    return [sys.executable, *map(str, args)]


def optional(cmd, option, value):
    if value:
        cmd.extend([option, str(value)])


def core_command(a, stage, *, config=None, study=None, data=None, seeds=SEEDS):
    cmd = command("scripts/run_reviewer_experiments.py", "--config", config or a.config,
                  "--output-root", study or a.output_root / "study", "--stage", stage,
                  "--arms", "train_tail", "--seeds", *seeds, "--variants", "full", "no_code",
                  "--num-processes", 3, "--batch-size", 2, "--eval-batch-size", 4,
                  "--gradient-accumulation-steps", 2, "--fit-fraction", .9,
                  "--bootstrap-replicates", 10000, "--bootstrap-seed", 2026)
    optional(cmd, "--data-csv", data or a.data_csv)
    optional(cmd, "--model-path", a.model_path)
    if a.resume:
        cmd.append("--resume")
    return cmd


def run(a, cmd):
    print(("[PLAN] " if a.dry_run else "[RUN] ") + shlex.join(list(map(str, cmd))), flush=True)
    if not a.dry_run:
        subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def hardware_complete(a, study):
    """Reuse a hardware analysis only if inputs, code and all outputs still match."""
    folder = a.output_root / "hardware"
    manifest = folder / "analysis_manifest.json"
    if not manifest.is_file():
        return False
    report = json.loads(manifest.read_text(encoding="utf-8"))
    identity = json.loads((study / "protocol/data_identity.json").read_text(encoding="utf-8"))
    expected = {"schema": "hardware-stratified-primary-study/v2", "seeds": list(SEEDS),
                "bootstrap_replicates": 10000, "bootstrap_seed": 2026,
                "data_sha256": file_digest(a.data_csv),
                "split_fingerprint": identity["split_fingerprint"],
                "capacity_map_sha256": file_digest(a.capacity_map), "model_map_sha256": file_digest(a.model_map),
                "external_registry_provenance_sha256": file_digest(a.registry_provenance) if a.registry_provenance else None,
                "analysis_source_sha256": file_digest(ROOT / "hardware_study/run_capacity_analysis.py"),
                "reused_diagnostics_source_sha256": file_digest(ROOT / "analysis/reviewer_diagnostics.py"),
                "historical_mapping_confirmed_by_author": False, "capacity_unit": "as registered"}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing hardware analysis has different inputs/code/settings; use a new output root")
    for record in report["inputs"]:
        directory = Path(record["directory"])
        if (study.resolve() not in directory.resolve().parents
                or file_digest(directory / "predictions.csv") != record["predictions_sha256"]
                or file_digest(directory / "run_manifest.json") != record["run_manifest_sha256"]):
            raise ValueError("Hardware analysis input predictions changed")
    for filename, digest in report["output_sha256"].items():
        if file_digest(folder / filename) != digest:
            raise ValueError(f"Hardware output changed: {filename}")
    return True


def temporal_inputs(a, label):
    folder = a.output_root / "temporal" / label
    if a.dry_run:
        return folder / "window.yaml", a.data_csv if label == "prefix_1000" else folder / "PRIVATE_prefix.csv"
    manifest = folder.parent / "windows.json"
    if not manifest.is_file():
        raise FileNotFoundError("Run --stage temporal-prepare first")
    item = next((r for r in json.loads(manifest.read_text(encoding="utf-8"))["windows"]
                 if r["window"] == label), None)
    if item is None:
        raise ValueError(f"Window absent from frozen manifest: {label}")
    data = Path(item["data_csv"])
    if file_digest(data) != item["window_data_sha256"]:
        raise ValueError(f"Temporal input changed: {data}")
    return Path(item["config"]), data


def execute_stage(a, stage):
    out, study = a.output_root, a.output_root / "study"
    if stage in CORE:
        run(a, core_command(a, stage))
    elif stage == "primary-export":
        cmd = command("scripts/export_primary_exact_bypass.py", "--study-root", study,
                      "--output-root", out / "primary_exact_bypass", "--seeds", *SEEDS,
                      "--paper-mode")
        if a.resume:
            cmd.append("--resume")
        run(a, cmd)
    elif stage == "baselines":
        cmd = command("scripts/run_matched_baselines.py", "--reference-study", study,
                      "--output-root", out / "baselines", "--stage", "all", "--seeds", *SEEDS)
        optional(cmd, "--data-csv", a.data_csv)
        if a.resume:
            cmd.append("--resume")
        run(a, cmd)
    elif stage == "followup":
        cmd = command("followup_study/run_followup.py", "--project-dir", ROOT,
                      "--study-root", study, "--output-root", out / "followup",
                      "--stage", "all", "--seeds", *SEEDS)
        optional(cmd, "--data-csv", a.data_csv)
        if a.resume:
            cmd.append("--resume")
        run(a, cmd)
    elif stage == "temporal-prepare":
        manifest = out / "temporal/windows.json"
        if a.resume and manifest.is_file() and not a.dry_run:
            rows = json.loads(manifest.read_text(encoding="utf-8"))["windows"]
            if [r["fraction"] for r in rows] != [.6, .8, 1.]:
                raise ValueError("Existing window fractions differ from the manuscript")
            identity = json.loads((study / "protocol/data_identity.json").read_text(encoding="utf-8"))
            if any(r["source_data_sha256"] != identity["data_sha256"] for r in rows):
                raise ValueError("Existing windows belong to a different source trace")
            for r in rows:
                temporal_inputs(a, r["window"])
            print("[SKIP] Existing window identities verified")
        else:
            run(a, command("scripts/prepare_reviewer_windows.py", "--config", study / "protocol/train_tail.yaml",
                           "--output-root", out / "temporal", "--fractions", .6, .8, 1.))
    elif stage in TEMPORAL:
        label = "prefix_" + stage.split("-")[1]
        cfg, data = temporal_inputs(a, label)
        run(a, core_command(a, "all", config=cfg, study=out / "temporal" / label / "study",
                            data=data, seeds=(42,)))
    elif stage == "hardware":
        if not a.dry_run:
            from types import SimpleNamespace
            from publication_reporting.report import verify_primary_exports
            if not verify_primary_exports(SimpleNamespace(study_root=study,
                    primary_root=out / "primary_exact_bypass"), {}, []):
                raise FileNotFoundError("Run primary-export before primary hardware stratification")
            if a.resume and hardware_complete(a, study):
                print("[SKIP] Hardware analysis inputs/code/output hashes verified")
                return
        cmd = command("hardware_study/run_capacity_analysis.py", "--project-root", ROOT,
                      "--study-root", study, "--output-dir", out / "hardware", "--seeds", *SEEDS,
                      "--data-csv", a.data_csv or "<EXTERNAL_CSV>",
                      "--capacity-map", a.capacity_map or "<EXTERNAL_CAPACITY_JSON>",
                      "--model-map", a.model_map or "<EXTERNAL_MODEL_JSON>",
                      "--bootstrap-replicates", 10000, "--bootstrap-seed", 2026)
        optional(cmd, "--registry-provenance", a.registry_provenance)
        run(a, cmd)
    elif stage == "online":
        cmd = command("-m", "online_study.suite", "--study-root", study,
                      "--output-root", out / "online", "--seeds", *SEEDS, "--repeats", 3,
                      "--cache-capacities", 128, 1024, "--control-cache-capacity", 1024,
                      "--operations", "full", "no_bypass", "no_post", "no_trust", "exact_bypass",
                      "--cache-start", "validation", "--num-threads", 4, "--warmup-jobs", 10,
                      "--device", "cuda", "--parity-atol", .005, "--order-seed", 2026)
        optional(cmd, "--data-csv", a.data_csv)
        optional(cmd, "--base-model", a.model_path)
        if a.resume:
            cmd.append("--resume")
        run(a, cmd)
    elif stage == "report":
        cmd = command("scripts/report_publication_results.py", "--study-root", study,
                      "--primary-root", out / "primary_exact_bypass",
                      "--baselines-root", out / "baselines", "--followup-root", out / "followup",
                      "--hardware-root", out / "hardware", "--windows-root", out / "temporal",
                      "--online-root", out / "online", "--output-dir", out / "publication",
                      "--require-complete")
        if a.resume:
            cmd.append("--overwrite")
        run(a, cmd)
    elif stage == "smoke":
        run(a, command("scripts/smoke_reviewer_pipeline.py", "--output-dir", out / "synthetic_smoke",
                       "--template-config", a.config, "--fit-gates"))
    else:
        raise ValueError(stage)


def main(argv=None):
    a = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for stage in ORDER if a.stage == "all" else (a.stage,):
        execute_stage(a, stage)
    print("[DONE] Commands planned; no files written." if a.dry_run else "[DONE] Requested stage completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
