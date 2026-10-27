#!/usr/bin/env python3
"""Read-only capacity-stratified evaluation of the sealed primary reviewer study.

No model fitting, policy selection, target normalization, or GPU use occurs here.
The existing reviewer_diagnostics.py supplies integrity checks and paired metrics.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BANDS = [
    (16384, "up_to_16384"), (32768, "16384_to_32768"),
    (65536, "32768_to_65536"), (86016, "65536_to_86016"),
    (math.inf, "over_86016"),
]
KEYS = ["sample_id", "resource"]
SCOPES = ("capacity_exact", "capacity_band", "gpu_model")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_diagnostics(project: Path):
    path = project / "analysis/reviewer_diagnostics.py"
    if not path.is_file():
        raise FileNotFoundError(f"Reviewer analysis source not found: {path}")
    spec = importlib.util.spec_from_file_location("capacity_reviewer_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def top_level_parts(text: str) -> list[str]:
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char == "[":
            depth += 1
            if depth != 1:
                raise ValueError("Nested node-list brackets")
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("Unmatched node-list bracket")
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    if depth:
        raise ValueError("Unclosed node-list bracket")
    parts.append(text[start:].strip())
    if any(not p for p in parts):
        raise ValueError("Empty node-list component")
    return parts


def expand_nodelist(expression: str, max_nodes: int = 10000) -> list[str]:
    """Expand Slurm lists/ranges and repeated bracket Cartesian products safely."""
    text = str(expression).strip()
    if not text or text.lower() in {"nan", "none", "n/a", "(null)", "null"}:
        return []
    if text.startswith(("['", '["', "('", '("')):
        values = ast.literal_eval(text)
        if not isinstance(values, (list, tuple)) or not all(isinstance(x, str) for x in values):
            raise ValueError("Expected a list of node strings")
        text = ",".join(values)
    nodes = []
    for part in top_level_parts(text):
        expanded = [""]
        position = 0
        for match in re.finditer(r"\[([^\[\]]+)\]", part):
            prefix = part[position:match.start()]
            if not re.fullmatch(r"[A-Za-z0-9_.-]*", prefix):
                raise ValueError("Invalid node-name characters")
            choices = []
            for token in match.group(1).split(","):
                token = token.strip()
                if token.isdigit():
                    choices.append(token)
                else:
                    interval = re.fullmatch(r"(\d+)-(\d+)", token)
                    if interval is None:
                        raise ValueError("Unsupported node range")
                    left, right = interval.groups()
                    lo, hi = int(left), int(right)
                    if hi < lo or hi - lo + 1 > max_nodes:
                        raise ValueError("Descending or excessive node range")
                    width = max(len(left), len(right)) if left.startswith("0") or right.startswith("0") else 0
                    choices.extend(str(i).zfill(width) for i in range(lo, hi + 1))
                if len(choices) > max_nodes:
                    raise ValueError("Excessive node-list expansion")
            if len(expanded) * len(choices) > max_nodes:
                raise ValueError("Excessive Cartesian node-list expansion")
            expanded = [base + prefix + choice for base in expanded for choice in choices]
            position = match.end()
        suffix = part[position:]
        if not re.fullmatch(r"[A-Za-z0-9_.-]*", suffix):
            raise ValueError("Unparsed node-list expression")
        nodes.extend(base + suffix for base in expanded)
        if len(nodes) > max_nodes:
            raise ValueError("Excessive node-list length")
    if any(not x for x in nodes) or len(nodes) != len(set(nodes)):
        raise ValueError("Empty or duplicate expanded node")
    return nodes


def load_inventory(path: Path) -> dict[str, int]:
    inventory = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Capacity registry must be a nonempty node-to-capacity JSON object")
    for node, value in inventory.items():
        if not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", node):
            raise ValueError("Invalid registry node name")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Every registered capacity must be a positive integer")
    return inventory


def load_model_inventory(path: Path) -> dict[str, dict]:
    inventory = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Model registry must be a nonempty node-to-record JSON object")
    for node, record in inventory.items():
        if not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", node):
            raise ValueError("Invalid model registry node name")
        if not isinstance(record, dict) or set(record) != {"gres_type", "installed_gpu_count"}:
            raise ValueError("Model record needs gres_type and installed_gpu_count")
        model, count = record["gres_type"], record["installed_gpu_count"]
        if model is not None and (not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", model)):
            raise ValueError("Invalid GRES model type")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Installed GPU count must be a positive integer")
    return inventory


def classify_model(nodelist: str, gpu_count: int, inventory: dict[str, dict]) -> dict:
    common = {"unknown_model_nodes": ()}
    if gpu_count <= 0:
        return {**common, "model_status": "cpu_only", "gpu_model": "cpu_only"}
    try:
        nodes = expand_nodelist(nodelist)
    except (ValueError, SyntaxError):
        status = "unparsed_nodelist"
    else:
        unknown = tuple(sorted(node for node in nodes if node not in inventory))
        common["unknown_model_nodes"] = unknown
        if not nodes:
            status = "missing_nodelist"
        elif unknown:
            status = "unmapped_model_node"
        elif any(inventory[node]["gres_type"] is None for node in nodes):
            status = "untyped_gres"
        else:
            models = {inventory[node]["gres_type"].lower() for node in nodes}
            if len(models) != 1:
                status = "mixed_gpu_models"
            else:
                return {**common, "model_status": "mapped_homogeneous",
                        "gpu_model": "gpu_type_" + next(iter(models))}
    return {**common, "model_status": status, "gpu_model": status}


def restore_string_job_ids(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Restore exact CSV text after the legacy reader's numeric job_id inference."""
    exact = pd.read_csv(path, usecols=KEYS + ["job_id"], dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    if exact.duplicated(KEYS).any() or exact[KEYS + ["job_id"]].apply(
            lambda s: s.str.strip().eq("").any()).any():
        raise ValueError("Missing/duplicate exact prediction identities")
    restored = frame.drop(columns="job_id").merge(exact, on=KEYS, how="outer",
                        validate="one_to_one", indicator="_identity_merge", sort=False)
    if not restored._identity_merge.eq("both").all():
        raise ValueError("Exact prediction identities differ from checked frame")
    restored = restored.drop(columns="_identity_merge")
    if not restored.groupby("sample_id").job_id.nunique().eq(1).all():
        raise ValueError("Memory/VRAM job IDs disagree")
    return restored


def classify_nodes(nodelist: str, gpu_count: int, inventory: dict[str, int]) -> dict:
    common = {"registered_capacity": np.nan, "expanded_node_count": 0, "unknown_nodes": ()}
    if gpu_count <= 0:
        return {**common, "capacity_status": "cpu_only", "capacity_exact": "cpu_only", "capacity_band": "cpu_only"}
    try:
        nodes = expand_nodelist(nodelist)
    except (ValueError, SyntaxError):
        status = "unparsed_nodelist"
        return {**common, "capacity_status": status, "capacity_exact": status, "capacity_band": status}
    common["expanded_node_count"] = len(nodes)
    unknown = tuple(sorted(node for node in nodes if node not in inventory))
    common["unknown_nodes"] = unknown
    if not nodes:
        status = "missing_nodelist"
    elif unknown:
        status = "unmapped_node"
    elif len({inventory[node] for node in nodes}) != 1:
        status = "mixed_registered_capacity"
    else:
        value = inventory[nodes[0]]
        return {**common, "capacity_status": "mapped_homogeneous", "registered_capacity": value,
                "capacity_exact": f"capacity_{value}",
                "capacity_band": next(label for ceiling, label in BANDS if value <= ceiling)}
    return {**common, "capacity_status": status, "capacity_exact": status, "capacity_band": status}


def numeric_gpu(values: pd.Series, name: str) -> pd.Series:
    gpu = pd.to_numeric(values, errors="raise")
    if not np.isfinite(gpu).all() or (gpu < 0).any() or (gpu != np.floor(gpu)).any():
        raise ValueError(f"Invalid nonnegative integer GPU counts: {name}")
    return gpu.astype(int)


def attach_metadata(raw: pd.DataFrame, population: pd.DataFrame, split: pd.DataFrame,
                    inventory: dict[str, int], diag) -> pd.DataFrame:
    """Join only audited identities; never use labels or capacity to infer identity."""
    for frame, keys, role in ((raw, ["job_id"], "raw CSV"),
                              (population, ["sample_id"], "population index"),
                              (split, ["sample_id"], "split manifest")):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"Missing/duplicate identity in {role}")
    if raw.job_id.str.strip().eq("").any():
        raise ValueError("Empty raw job ID")
    index = population.merge(split[["sample_id", "job_id", "source_row"]],
                             on="sample_id", how="left", validate="one_to_one")
    if index[["job_id", "source_row"]].isna().any().any():
        raise ValueError("Population absent from frozen split manifest")
    joined = index.merge(raw, on="job_id", how="left", validate="one_to_one", indicator=True,
                         suffixes=("", "_raw"))
    if not joined._merge.eq("both").all():
        raise ValueError("Population job IDs absent from raw CSV")
    if not np.array_equal(pd.to_numeric(joined.source_row), joined.raw_source_row):
        raise ValueError("Frozen split source rows disagree with raw CSV order")
    joined["gpu_count"] = numeric_gpu(joined.gpu_count, "population index")
    if not np.array_equal(joined.gpu_count, numeric_gpu(joined.raw_gpu_count, "raw CSV")):
        raise ValueError("Raw/index GPU requests differ")
    for name in ("valid_memory", "valid_vram", "effective_analysis_row"):
        joined[name] = diag.boolean(joined[name], name)
    joined["submit_time"] = pd.to_datetime(joined.submit_time, utc=True, errors="raise")
    joined["month"] = joined.submit_time.dt.strftime("%Y-%m")
    attributes = pd.DataFrame([classify_nodes(node, count, inventory)
                               for node, count in zip(joined.nodelist, joined.gpu_count)])
    return pd.concat([joined.reset_index(drop=True), attributes], axis=1).drop(columns="_merge")


def capacity_metrics(y, p, diag, tolerance: float) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    error = p - y
    shortfall = np.maximum(-error, 0)
    return {**diag.metrics(y, p, tolerance), "mean_signed_error": float(error.mean()),
            "underprediction_rate": float((error < -1e-12).mean()),
            "overprediction_rate": float((error > 1e-12).mean()),
            "underprediction_gt_010_rate": float((shortfall > .10).mean()),
            "p95_positive_shortfall": float(np.quantile(shortfall, .95)),
            "p99_positive_shortfall": float(np.quantile(shortfall, .99))}


def aggregate_seeds(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    output = []
    for identity, part in frame.groupby(keys, sort=True, dropna=False):
        if part.seed.duplicated().any():
            raise ValueError("Duplicate seed within summary group")
        if part.n.nunique() != 1 or part.day_blocks.nunique() != 1:
            raise ValueError("Capacity population differs across seeds")
        if not isinstance(identity, tuple):
            identity = (identity,)
        row = dict(zip(keys, identity))
        row.update(seed_count=int(part.seed.nunique()), n=int(part.n.iloc[0]),
                   day_blocks=int(part.day_blocks.iloc[0]))
        for value in values:
            row[value + "_mean"] = float(part[value].mean())
            row[value + "_sample_std"] = float(part[value].std(ddof=1)) if len(part) > 1 else np.nan
        output.append(row)
    return pd.DataFrame(output)


def make_plots(summary: pd.DataFrame, output: Path, unit: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    band_order = [label for _, label in BANDS] + ["unmapped_node", "missing_nodelist",
                  "unparsed_nodelist", "mixed_registered_capacity", "cpu_only"]
    selected = summary.loc[summary.scope.eq("capacity_band") & summary.method.isin(
        ["Full Cascade", "No-Code Cascade", "Hierarchical Same-Script Last-1"])]
    labels = {"up_to_16384": "<=16,384", "16384_to_32768": "(16,384, 32,768]",
              "32768_to_65536": "(32,768, 65,536]", "65536_to_86016": "(65,536, 86,016]",
              "over_86016": ">86,016", "cpu_only": "CPU only",
              "unmapped_node": "Unmapped", "mixed_registered_capacity": "Mixed capacity",
              "missing_nodelist": "Missing nodes", "unparsed_nodelist": "Unparsed nodes"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    styles = [("Full Cascade", "#21618c", "o"), ("No-Code Cascade", "#b74e2e", "s"),
              ("Hierarchical Same-Script Last-1", "#326c46", "^")]
    for metric, ylabel in (("mae", "MAE"), ("underprediction_rate", "Underprediction rate")):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), layout="constrained")
        for ax, resource, letter in zip(axes, ["memory", "vram"], ["a", "b"]):
            part = selected.loc[selected.resource.eq(resource)]
            groups = [key for key in band_order if key in set(part.group)]
            x = np.arange(len(groups))
            for i, (method, color, marker) in enumerate(styles):
                rows = part.loc[part.method.eq(method)].set_index("group").reindex(groups)
                ax.errorbar(x + (i - 1) * .17, rows[metric + "_mean"],
                            yerr=rows[metric + "_sample_std"].fillna(0), fmt=marker,
                            color=color, capsize=3, label=method, markersize=5, linestyle="none")
            count = part.loc[part.method.eq("Full Cascade")].set_index("group")
            ax.set_xticks(x, [labels[g] + f"\nn={int(count.loc[g, 'n']):,}" for g in groups],
                          rotation=28, ha="right")
            ax.set_title(f"({letter}) {'Memory' if resource == 'memory' else 'VRAM'}", fontsize=11)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f"Per-GPU registered capacity ({unit}); CPU jobs shown separately")
            ax.grid(axis="y", color="#dedede", linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside upper center", ncol=3, frameon=False)
        for extension in ("svg", "pdf", "png"):
            fig.savefig(output / f"capacity_{metric}.{extension}", dpi=300)
        plt.close(fig)
    selected = summary.loc[summary.scope.eq("gpu_model") & summary.method.isin([s[0] for s in styles])]
    model_groups = sorted(set(selected.group) - {"cpu_only"})
    model_groups += ["cpu_only"] if selected.group.eq("cpu_only").any() else []
    names = {"untyped_gres": "Unspecified GRES type", "unmapped_model_node": "Unmapped model",
             "mixed_gpu_models": "Mixed models", "missing_nodelist": "Missing nodes",
             "unparsed_nodelist": "Unparsed nodes", "cpu_only": "CPU only"}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, max(5.4, .42 * len(model_groups) + 1.8)),
                             layout="constrained")
    for ax, resource, letter in zip(axes, ["memory", "vram"], ["a", "b"]):
        part = selected.loc[selected.resource.eq(resource)]
        groups = [g for g in model_groups if g in set(part.group)]
        y = np.arange(len(groups))
        for i, (method, color, marker) in enumerate(styles):
            rows = part.loc[part.method.eq(method)].set_index("group").reindex(groups)
            ax.errorbar(rows.mae_mean, y + (i - 1) * .17,
                        xerr=rows.mae_sample_std.fillna(0), fmt=marker, color=color,
                        capsize=3, label=method, markersize=4.5, linestyle="none")
        count = part.loc[part.method.eq("Full Cascade")].set_index("group")
        ax.set_yticks(y, [names.get(g, g.removeprefix("gpu_type_").upper()) +
                          f" (n={int(count.loc[g, 'n']):,})" for g in groups])
        ax.invert_yaxis()
        ax.set_title(f"({letter}) {'Memory' if resource == 'memory' else 'VRAM'}", fontsize=11)
        ax.set_xlabel("MAE (utilization ratio)")
        ax.set_ylabel("GPU type from author-provided registry")
        ax.grid(axis="x", color="#dedede", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_names = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_names, loc="outside upper center", ncol=3, frameon=False)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output / f"gpu_model_mae.{extension}", dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=HERE.parent / "outputs/study")
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=HERE.parent)
    parser.add_argument("--capacity-map", type=Path, required=True,
                        help="External site-maintained per-GPU capacity registry; examples are synthetic")
    parser.add_argument("--model-map", type=Path, required=True,
                        help="External site-maintained node-to-GRES model registry")
    parser.add_argument("--registry-provenance", type=Path,
                        help="Optional external JSON describing registry source and effective dates")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "outputs/hardware")
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--capacity-unit", default="as registered", help="Display only; no unit conversion")
    parser.add_argument("--require-complete-gpu-map", action="store_true")
    parser.add_argument("--historical-mapping-confirmed", action="store_true",
                        help="Author attests that node model/capacity mappings apply throughout the trace")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or args.bootstrap_replicates < 1:
        raise ValueError("Seeds must be unique and bootstrap count positive")
    study, data_path, registry = args.study_root.resolve(), args.data_csv.resolve(), args.capacity_map.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; use a new --output-dir: {output}")
    diag, diag_path = load_diagnostics(args.project_root.resolve())
    inventory = load_inventory(registry)
    model_registry = args.model_map.resolve()
    models = load_model_inventory(model_registry)
    protocol = study / "protocol"
    identity = json.loads((protocol / "data_identity.json").read_text(encoding="utf-8-sig"))
    raw_sha = digest(data_path)
    if raw_sha != identity["data_sha256"]:
        raise ValueError("Raw CSV hash differs from frozen primary study; do not edit or substitute the CSV")
    sidecar = json.loads((protocol / "population_index.json").read_text(encoding="utf-8-sig"))
    diag.require_identity(identity, sidecar)
    if digest(protocol / "population_index.csv") != sidecar["population_index_sha256"]:
        raise ValueError("Population index checksum mismatch")
    population = pd.read_csv(protocol / "population_index.csv", dtype={"sample_id": str})
    split_path = protocol / "outer_split.csv"
    if digest(split_path) != identity["split_manifest_sha256"]:
        raise ValueError("Frozen split manifest checksum mismatch")
    split = pd.read_csv(split_path, dtype={"sample_id": str, "job_id": str})
    raw = pd.read_csv(data_path, usecols=["jobid", "nodelist", "gpu"], dtype=str,
                      keep_default_na=False, encoding="utf-8-sig").rename(
        columns={"jobid": "job_id", "gpu": "raw_gpu_count"})
    raw["raw_source_row"] = np.arange(len(raw))
    metadata = attach_metadata(raw, population, split, inventory, diag)
    model_metadata = pd.DataFrame([classify_model(node, count, models)
                                  for node, count in zip(metadata.nodelist, metadata.gpu_count)])
    metadata = pd.concat([metadata, model_metadata], axis=1)
    print(f"Audited raw CSV and {len(metadata):,} retained jobs; no models are being trained.", flush=True)

    coverage_rows = []
    for effective in (False, True):
        pop = metadata.loc[metadata.effective_analysis_row] if effective else metadata
        for scope in SCOPES + ("capacity_status", "model_status"):
            for (period, group), block in pop.groupby(["split", scope], sort=True):
                coverage_rows.append({"population": "effective" if effective else "retained",
                    "split": period, "scope": scope, "group": group, "n": len(block),
                    "gpu_job_n": int(block.gpu_count.gt(0).sum()),
                    "valid_memory_n": int(block.valid_memory.sum()), "valid_vram_n": int(block.valid_vram.sum()),
                    "fraction_of_split": len(block) / int(pop.split.eq(period).sum())})
    bad = metadata.loc[metadata.gpu_count.gt(0) & metadata.capacity_status.ne("mapped_homogeneous")]
    bad_model = metadata.loc[metadata.gpu_count.gt(0) & metadata.model_status.ne("mapped_homogeneous")]
    if args.require_complete_gpu_map and (len(bad) or len(bad_model)):
        raise ValueError(f"Unresolved hardware: {len(bad)} capacity and {len(bad_model)} model GPU jobs")

    metric_rows, paired_rows, input_records = [], [], []
    base_population = None
    for seed in args.seeds:
        root = study / "train_tail/cascade/reliability_cascade/tokens_4096"
        full, fm, fp = diag.read_run(root / "full" / f"seed_{seed}", seed, sealed=True)
        nc, nm, npv = diag.read_run(root / "no_code" / f"seed_{seed}", seed, sealed=True)
        full = restore_string_job_ids(full, root / "full" / f"seed_{seed}" / "predictions.csv")
        nc = restore_string_job_ids(nc, root / "no_code" / f"seed_{seed}" / "predictions.csv")
        diag.require_identity(identity, fm)
        diag.require_identity(identity, nm)
        pairs = diag.align_predictions(full, nc, "_no_code")
        for column in ("job_id", "gpu_count", "resource_applicable"):
            if not pairs[column].equals(pairs[column + "_no_code"]):
                raise ValueError(f"Full/No-Code paired {column} differs")
        expected = metadata.loc[metadata.effective_analysis_row & metadata.split.isin(["validation", "test"])]
        for resource in ("memory", "vram"):
            if set(full.loc[full.resource.eq(resource), "sample_id"]) != set(expected.sample_id):
                raise ValueError("Predictions do not cover the frozen validation/test population")
        test = pairs.loc[pairs.split.eq("test")].copy()
        columns = ["sample_id", "job_id", "gpu_count", "split", "submit_time", "valid_memory", "valid_vram",
                   "capacity_exact", "capacity_band", "capacity_status", "gpu_model", "model_status"]
        test = test.merge(metadata.loc[metadata.effective_analysis_row, columns], on="sample_id", how="left",
                          suffixes=("", "_metadata"), validate="many_to_one", indicator=True)
        if not test._merge.eq("both").all():
            raise ValueError("Test predictions lack metadata")
        for column in ("job_id", "split", "submit_time"):
            if not test[column].equals(test[column + "_metadata"]):
                raise ValueError(f"Prediction/metadata {column} mismatch")
        if not np.array_equal(numeric_gpu(test.gpu_count, "prediction"), test.gpu_count_metadata):
            raise ValueError("Prediction/metadata GPU request mismatch")
        expected_valid = np.where(test.resource.eq("memory"), test.valid_memory, test.valid_vram)
        expected_applicable = test.resource.eq("memory") | test.gpu_count.gt(0)
        if not np.array_equal(test.valid, expected_valid) or not np.array_equal(test.resource_applicable, expected_applicable):
            raise ValueError("Prediction label-valid/applicable masks disagree with population")
        signature = test[KEYS + ["target", "valid", "submit_time"]].sort_values(KEYS).reset_index(drop=True)
        if base_population is not None and not signature.equals(base_population):
            raise ValueError("Test population or labels change between seeds")
        base_population = signature
        methods = {"Full Cascade": "prediction", "No-Code Cascade": "prediction_no_code",
                   "Hierarchical Same-Script Last-1": "hierarchical_reference_prediction",
                   "Standalone Qwen": "qwen_prediction"}
        for resource, valid in test.loc[test.valid & test.resource_applicable].groupby("resource"):
            groups = [("all_valid", "all", valid)]
            for scope in SCOPES:
                groups.extend((scope, name, block) for name, block in valid.groupby(scope, sort=True))
            for scope, group, block in groups:
                days = diag.cluster_labels(block, "day", "UTC")
                info = {"seed": seed, "resource": resource, "scope": scope, "group": group,
                        "day_blocks": len(set(days))}
                for method, column in methods.items():
                    p = pd.to_numeric(block[column], errors="raise").to_numpy(float)
                    row = {**info, "method": method, **capacity_metrics(block.target, p, diag, .10)}
                    metric_rows.append(row)
                for reference in ("Hierarchical Same-Script Last-1", "No-Code Cascade"):
                    results = diag.paired_bootstrap(block.target, block.prediction, block[methods[reference]], days,
                        replicates=args.bootstrap_replicates,
                        rng=diag.named_rng(args.bootstrap_seed, "hardware", seed, resource, scope, group, reference))
                    for result in results:
                        paired_rows.append({**info, "reference": reference, **result,
                            "small_group": len(block) < 30 or len(set(days)) < 5})
        input_records.extend([fp, npv])
        print(f"Finished seed {seed}; paired day-block results computed.", flush=True)

    metrics = pd.DataFrame(metric_rows)
    paired = pd.DataFrame(paired_rows)
    values = ["mae", "mse", "rmse", "acc_at_10", "mean_signed_error", "underprediction_rate",
              "overprediction_rate", "underprediction_gt_010_rate", "p95_positive_shortfall", "p99_positive_shortfall"]
    summary = aggregate_seeds(metrics, ["resource", "method", "scope", "group"], values)
    ci_rows = []
    for key, block in paired.groupby(["resource", "scope", "group", "reference", "metric"], sort=True):
        ci_rows.append({**dict(zip(["resource", "scope", "group", "reference", "metric"], key)),
            "seed_count": block.seed.nunique(), "n": int(block.n.iloc[0]),
            "day_blocks": int(block.day_blocks.iloc[0]), "mean_delta": block.delta.mean(),
            "favorable_seed_count": int(block.favorable_interval.sum()),
            "intervals_available": int(block.status.eq("ok").sum()),
            "small_group": bool(block.small_group.any())})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(output / "capacity_population_coverage.csv", index=False)
    metrics.to_csv(output / "capacity_metrics_per_seed.csv", index=False)
    summary.to_csv(output / "capacity_metrics_summary.csv", index=False)
    paired.to_csv(output / "capacity_paired_intervals_per_seed.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output / "capacity_paired_support_summary.csv", index=False)
    unknown = Counter(node for nodes in bad.unknown_nodes for node in nodes)
    pd.DataFrame([{"node": node, "retained_gpu_job_count": count} for node, count in sorted(unknown.items())],
                 columns=["node", "retained_gpu_job_count"]).to_csv(output / "PRIVATE_unknown_nodes.csv", index=False)
    anomalies = bad.groupby(["split", "capacity_status"]).size().reset_index(name="n")
    anomalies.to_csv(output / "capacity_mapping_anomalies.csv", index=False)
    bad_model.groupby(["split", "model_status"]).size().reset_index(name="n").to_csv(
        output / "model_mapping_anomalies.csv", index=False)
    unknown_models = Counter(node for nodes in bad_model.unknown_model_nodes for node in nodes)
    pd.DataFrame([{"node": node, "retained_gpu_job_count": count} for node, count in sorted(unknown_models.items())],
                 columns=["node", "retained_gpu_job_count"]).to_csv(output / "PRIVATE_unknown_model_nodes.csv", index=False)
    monthly_rows = []
    gpu_population = metadata.loc[metadata.effective_analysis_row & metadata.gpu_count.gt(0)]
    for scope in SCOPES:
        for (period, month, group), block in gpu_population.groupby(["split", "month", scope], sort=True):
            denominator = int((gpu_population.split.eq(period) & gpu_population.month.eq(month)).sum())
            monthly_rows.append({"split": period, "month": month, "scope": scope, "group": group,
                                 "gpu_job_n": len(block), "valid_vram_n": int(block.valid_vram.sum()),
                                 "fraction_of_gpu_jobs_in_split_month": len(block) / denominator})
    pd.DataFrame(monthly_rows).to_csv(output / "hardware_monthly_composition.csv", index=False)
    plot_status = "disabled_by_flag"
    if not args.no_plots:
        try:
            make_plots(summary, output, args.capacity_unit)
            plot_status = "generated"
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            plot_status = "skipped_matplotlib_not_installed"
            print("Matplotlib is unavailable; aggregate CSVs and the analysis manifest are still produced.", flush=True)
    report = {
        "schema": "hardware-stratified-primary-study/v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": raw_sha, "split_fingerprint": identity["split_fingerprint"],
        "capacity_map_sha256": digest(registry), "capacity_map_nodes": len(inventory),
        "model_map_sha256": digest(model_registry), "model_map_nodes": len(models),
        "model_map_source": "External site-maintained registry supplied by the caller; type normalized to lowercase",
        "external_registry_provenance": (
            json.loads(args.registry_provenance.read_text(encoding="utf-8"))
            if args.registry_provenance else None),
        "external_registry_provenance_sha256": (
            digest(args.registry_provenance) if args.registry_provenance else None),
        "primary_prediction_scope": "Unchanged frozen Full and adaptive No-Code predictions; Full is equivalent to exact bypass only for verified zero-threshold policies",
        "historical_mapping_confirmed_by_author": args.historical_mapping_confirmed,
        "capacity_unit": args.capacity_unit, "unit_conversion_performed": False,
        "analysis_source_sha256": digest(Path(__file__)), "reused_diagnostics_source_sha256": digest(diag_path),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "plot_status": plot_status,
        "seeds": args.seeds, "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed, "bootstrap_blocks": "nonempty UTC submission days within each stratum",
        "confidence_level": .95, "differences": "Full minus reference; MAE/MSE/RMSE negative favors Full; Acc positive favors Full",
        "capacity_bands": [{"upper_inclusive": hi if math.isfinite(hi) else None, "label": label} for hi, label in BANDS],
        "retained_jobs": len(metadata), "effective_jobs": int(metadata.effective_analysis_row.sum()),
        "gpu_jobs_without_single_known_capacity": len(bad),
        "gpu_jobs_without_single_known_model": len(bad_model),
        "test_valid_counts": metrics.loc[metrics.scope.eq("all_valid") & metrics.method.eq("Full Cascade")][
            ["seed", "resource", "n", "day_blocks"]].to_dict("records"),
        "inputs": input_records,
        "caveats": [
            "Registries are author-supplied; current sinfo does not independently establish historical effective dates.",
            "GPU models are GRES labels, not inferred from VRAM capacity. Untyped GRES remains explicit.",
            "Installed GPU count is registry metadata, not the number requested/used by a job.",
            "Capacity is per GPU; it is not multiplied by requested GPU count and targets are not renormalized.",
            "Exact registered capacities are reported separately; bands are fixed before inspecting outcomes.",
            "CPU-only, unmapped, unparsed, and mixed-capacity jobs remain explicit; all-valid totals include them.",
            "Means/SD are across training seeds on one shared test period; n is not multiplied by seed count.",
            "Paired intervals are pointwise descriptive analyses, not multiplicity-adjusted confirmation.",
            "Single-day strata have no bootstrap interval; small groups require caution.",
            "These are within-stratum performance descriptions, not causal GPU model effects or cross-hardware generalization.",
            "This does not establish telemetry coverage, business causes, OOM risk, or scheduler savings.",
        ],
    }
    report["output_sha256"] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "analysis_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Capacity analysis complete: {output}", flush=True)
    if len(bad):
        print(f"Mapping audit: {len(bad)} retained GPU jobs have unresolved capacities; see anomaly and PRIVATE unknown-node files.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
