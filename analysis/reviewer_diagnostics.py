#!/usr/bin/env python3
"""Portable fixed-prediction reviewer diagnostics (numpy/pandas only).

This program never fits a model or selects a policy. It deliberately refuses
misaligned populations, different dataset/split contracts, and inconsistent
saved routing/cache flags. Outputs are aggregates, never job/user/script IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = ("mae", "mse", "rmse", "acc_at_10")
EMPTY_KEY = hashlib.sha256(b"<EMPTY_SCRIPT>").hexdigest()
BOOL_COLUMNS = (
    "valid", "history_available", "exact_script_history_available", "same_last1",
    "resource_applicable", "history_bypass", "qwen_required_resource",
    "qwen_required_job", "embedding_cache_hit_job", "encoder_forward_required_job",
    "trust_region_applied", "user_known",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def boolean(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"Missing boolean values in {name}")
    text = values.astype(str).str.strip().str.lower()
    if not text.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"Invalid boolean values in {name}")
    return text.isin(("true", "1"))


def read_predictions(path: Path, seed: int) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"sample_id": str, "resource": str, "user_id": str})
    required = {"sample_id", "resource", "split", "valid", "target", "prediction", "submit_time", "user_id"}
    if missing := required - set(data):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if data.duplicated(["sample_id", "resource"]).any():
        raise ValueError(f"{path}: duplicate sample/resource keys")
    if data[["sample_id", "resource", "user_id", "submit_time"]].isna().any().any():
        raise ValueError(f"{path}: missing pairing/cluster identifiers")
    if "seed" in data and set(pd.to_numeric(data.seed)) != {seed}:
        raise ValueError(f"{path}: seed differs from requested seed {seed}")
    for col in BOOL_COLUMNS:
        if col in data:
            data[col] = boolean(data[col], col)
    for col in ("submit_time", "completion_time"):
        if col in data:
            data[col] = pd.to_datetime(data[col], utc=True, errors="raise")
            if data[col].isna().any():
                raise ValueError(f"{path}: missing {col}")
    good = data.valid
    y = pd.to_numeric(data.loc[good, "target"], errors="raise").to_numpy(float)
    p = pd.to_numeric(data.loc[good, "prediction"], errors="raise").to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError(f"{path}: nonfinite valid targets/predictions")
    if np.any((y < 0) | (y > 1)) or np.any((p < 0) | (p > 1)):
        raise ValueError(f"{path}: normalized values outside [0,1]")
    return data.sort_values(["submit_time", "sample_id", "resource"], kind="stable").reset_index(drop=True)


def read_run(directory: Path, seed: int, *, sealed: bool) -> tuple[pd.DataFrame, dict, dict]:
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("data_sha256", "split_fingerprint"):
        if not manifest.get(key):
            raise ValueError(f"{manifest_path}: absent {key}")
    seal_path, completion_path = directory / "artifact_integrity.json", directory / "completion_record.json"
    if sealed:
        if not seal_path.is_file() or not completion_path.is_file():
            raise ValueError(f"Cascade is not sealed/completed: {directory}")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        for name, expected in seal["files"].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or sha256(target) != expected:
                raise ValueError(f"Cascade artifact integrity failure: {name}")
        for field, filename in (("run_manifest_sha256", "run_manifest.json"),
                                ("artifact_integrity_sha256", "artifact_integrity.json"),
                                ("checkpoint_contract_sha256", "checkpoint/checkpoint_contract.json")):
            if completion[field] != sha256(directory / filename):
                raise ValueError(f"Completion record integrity failure: {filename}")
        if completion["run_request_sha256"] != seal["run_request_sha256"]:
            raise ValueError("Completion/seal request mismatch")
    frame = read_predictions(directory / "predictions.csv", seed)
    provenance = {"directory": str(directory.resolve()), "seed": seed,
                  "predictions_sha256": sha256(directory / "predictions.csv"),
                  "run_manifest_sha256": sha256(manifest_path), "sealed": sealed,
                  **{k: manifest[k] for k in ("data_sha256", "split_fingerprint")}}
    return frame, manifest, provenance


def require_identity(first: dict, second: dict) -> None:
    for name in ("data_sha256", "split_fingerprint"):
        if first[name] != second[name]:
            raise ValueError(f"Cannot compare different {name} contracts")


def align_predictions(left: pd.DataFrame, right: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Exact population/label/mask/time alignment before any valid-row filter."""
    keys = ["sample_id", "resource"]
    aligned = left.merge(right, on=keys, how="outer", suffixes=("", suffix), validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError("Prediction populations differ")
    for column in ("valid", "split", "submit_time", "completion_time", "user_id",
                   "history_available", "exact_script_history_available", "same_last1", "script_sha256"):
        other = column + suffix
        if column in left and column in right and not aligned[column].equals(aligned[other]):
            raise ValueError(f"Paired {column} differs")
    if not np.allclose(aligned.target, aligned["target" + suffix], rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("Paired target values differ")
    return aligned.drop(columns="_merge")


def metrics(y: Any, p: Any, tolerance: float = .10) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) != len(p) or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require nonempty aligned finite arrays")
    error = p - y
    return {"n": len(y), "mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "acc_at_10": float((np.abs(error) <= tolerance).mean())}


def cluster_labels(part: pd.DataFrame, name: str, zone: str = "UTC") -> np.ndarray:
    if name in ("user", "script"):
        column = "user_id" if name == "user" else "script_sha256"
        if column not in part or part[column].isna().any():
            raise ValueError(f"Cannot build {name} clusters without complete identities")
        # Empty scripts are not semantically identical; keep them as singleton blocks.
        if name == "script":
            return np.where(part[column].eq(EMPTY_KEY), "empty:" + part.sample_id, part[column]).astype(str)
        return part[column].astype(str).to_numpy()
    day = part.submit_time.dt.tz_convert(zone).dt.floor("1D")
    if name == "day":
        return day.astype(str).to_numpy()
    if name in ("3day", "7day"):
        width = int(name[0])
        return ((day - day.min()).dt.days // width).to_numpy()
    raise ValueError(f"Unknown cluster type {name}")


def named_rng(seed: int, *labels: Any) -> np.random.Generator:
    text = json.dumps([seed, *labels], separators=(",", ":"))
    number = int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "big")
    return np.random.default_rng(number)


def paired_bootstrap(y: Any, proposed: Any, reference: Any, clusters: Any, *,
                     replicates: int, rng: np.random.Generator, confidence: float = .95,
                     tolerance: float = .10) -> list[dict]:
    """Resample clusters uniformly and retain all paired jobs in each draw."""
    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    y, p, r, labels = np.asarray(y, float), np.asarray(proposed, float), np.asarray(reference, float), np.asarray(clusters)
    pm, rm = metrics(y, p, tolerance), metrics(y, r, tolerance)
    if len(labels) != len(y) or pd.isna(labels).any():
        raise ValueError("Clusters must align with all paired observations")
    pe, re = p-y, r-y
    blocks = pd.DataFrame({"cluster": labels, "n": 1., "pa": abs(pe), "ra": abs(re),
                           "ps": pe**2, "rs": re**2, "ph": (abs(pe)<=tolerance).astype(float),
                           "rh": (abs(re)<=tolerance).astype(float)})
    grouped = blocks.groupby("cluster", sort=True).sum()
    array = grouped.to_numpy(float)
    k = len(array)
    if k < 2:
        return [{"metric": m, "n": len(y), "clusters": k, "status": "insufficient_clusters",
                 "proposed_value": pm[m], "reference_value": rm[m], "delta": pm[m]-rm[m],
                 "ci_lower": np.nan, "ci_upper": np.nan, "favorable_interval": False} for m in METRICS]
    deltas = np.empty((replicates, len(METRICS)))
    # Multinomial counts avoid a replicates x clusters x metrics temporary.
    chunk = max(1, min(500, 1_000_000 // k))
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        counts = rng.multinomial(k, np.full(k, 1/k), size=stop-start)
        sums = counts @ array
        n = sums[:, 0]
        deltas[start:stop] = np.column_stack(((sums[:,1]-sums[:,2])/n, (sums[:,3]-sums[:,4])/n,
            np.sqrt(sums[:,3]/n)-np.sqrt(sums[:,4]/n), (sums[:,5]-sums[:,6])/n))
    alpha = (1-confidence)/2
    interval = np.quantile(deltas, [alpha, 1-alpha], axis=0)
    return [{"metric": m, "n": len(y), "clusters": k, "status": "ok",
             "min_cluster_n": int(grouped.n.min()), "max_cluster_n": int(grouped.n.max()),
             "proposed_value": pm[m], "reference_value": rm[m], "delta": pm[m]-rm[m],
             "ci_lower": float(interval[0,i]), "ci_upper": float(interval[1,i]),
             "favorable_interval": bool(interval[0,i]>0 if m == "acc_at_10" else interval[1,i]<0)}
            for i,m in enumerate(METRICS)]


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm correction requires finite probabilities")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted.tolist()


def mcnemar(y: Any, p: Any, r: Any, tolerance: float) -> dict:
    y, p, r = np.asarray(y,float), np.asarray(p,float), np.asarray(r,float)
    ph, rh = abs(p-y)<=tolerance, abs(r-y)<=tolerance
    b, c = int((ph & ~rh).sum()), int((~ph & rh).sum())
    statistic = max(abs(b-c)-1, 0)**2/(b+c) if b+c else 0.
    return {"proposed_only_hit_n": b, "reference_only_hit_n": c,
            "chi_square_continuity_corrected": statistic,
            "p_value": math.erfc(math.sqrt(statistic/2)), "n": len(y)}


def risk_metrics(y: Any, p: Any) -> dict:
    y, p = np.asarray(y,float), np.asarray(p,float)
    error = p-y
    shortfall = np.maximum(-error, 0.)
    under, over = error < -1e-12, error > 1e-12
    return {"mean_signed_error": float(error.mean()), "underprediction_rate": float(under.mean()),
            "overprediction_rate": float(over.mean()), "mean_positive_shortfall_ratio": float(shortfall.mean()),
            "p95_positive_shortfall_ratio": float(np.quantile(shortfall,.95)),
            "p99_positive_shortfall_ratio": float(np.quantile(shortfall,.99)),
            "p95_shortfall_given_underprediction": float(np.quantile(shortfall[under],.95)) if under.any() else np.nan,
            **metrics(y,p)}


def probability_diagnostics(label: Any, score: Any) -> tuple[dict,list[dict]]:
    label, score = np.asarray(label,bool), np.asarray(score,float)
    if len(label)!=len(score) or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError("Invalid probability diagnostics inputs")
    n, positives = len(label), int(label.sum())
    if not n:
        return {"n": 0, "status": "empty"}, []
    rank = pd.Series(score).rank(method="average").to_numpy()
    auc = (rank[label].sum()-positives*(positives+1)/2)/(positives*(n-positives)) if 0<positives<n else np.nan
    order = np.argsort(-score,kind="stable")
    yy, ss = label[order], score[order]
    ends = np.r_[np.where(np.diff(ss)!=0)[0], n-1]
    tp = np.cumsum(yy)[ends]
    recall = tp/positives if positives else np.zeros(len(tp))
    ap = float(np.sum(np.diff(np.r_[0,recall])*tp/(ends+1))) if positives else np.nan
    bins, ece = [], 0.
    bin_id = np.minimum((score*10).astype(int),9)
    for i in range(10):
        take = bin_id==i
        if take.any():
            predicted, observed = float(score[take].mean()), float(label[take].mean())
            ece += take.mean()*abs(predicted-observed)
            bins.append({"bin": i, "n": int(take.sum()), "mean_probability": predicted, "observed_positive_rate": observed})
    return {"n": n, "positive_n": positives, "positive_prevalence": positives/n,
            "auroc": float(auc), "average_precision": ap, "brier": float(np.mean((score-label)**2)),
            "ece_10_equal_width": float(ece), "status": "ok"}, bins


def reliable_labels(frame: pd.DataFrame, cap: float) -> np.ndarray:
    h = abs(frame.history_expert_prediction.to_numpy(float)-frame.target.to_numpy(float))
    q = abs(frame.qwen_prediction.to_numpy(float)-frame.target.to_numpy(float))
    return (h<=q+1e-12) & (h<=cap)


def frozen_operation(frame: pd.DataFrame, mode: str, threshold: float|None = None) -> tuple[np.ndarray,np.ndarray]:
    """Counterfactual operation with every learned component/policy held fixed."""
    bypass = frame.history_bypass.to_numpy(bool).copy()
    history = frame.history_available.to_numpy(bool)
    exact = frame.exact_script_history_available.to_numpy(bool)
    applicable = frame.resource_applicable.to_numpy(bool)
    if mode == "no_bypass":
        bypass[:] = False
    elif mode == "exact_bypass":
        bypass = history & exact & applicable
    elif mode == "threshold":
        if threshold is None or not math.isfinite(threshold):
            raise ValueError("Threshold is required")
        bypass = history & exact & applicable & (frame.reliability_probability.to_numpy(float)>=threshold)
    elif mode not in ("full", "no_post", "no_trust"):
        raise ValueError(f"Unknown frozen operation {mode}")
    q = frame.qwen_prediction.to_numpy(float)
    h = np.where(history,frame.history_expert_prediction.to_numpy(float),q)
    post = frame.post_blend_prediction.to_numpy(float)
    if mode == "no_post":
        post = q
    elif mode == "no_trust":
        gamma = frame.post_history_weight.to_numpy(float)
        post = gamma*h+(1-gamma)*q
    prediction = np.clip(np.where(bypass,h,post),0.,1.)
    if not np.isfinite(prediction).all():
        raise ValueError("Frozen operation produced nonfinite predictions")
    return prediction, applicable & ~bypass


def make_jobs(frame: pd.DataFrame, required_resource: Any) -> pd.DataFrame:
    required = pd.Series(np.asarray(required_resource,bool),index=frame.sample_id).groupby(level=0).any()
    memory = frame.loc[frame.resource.eq("memory"),["sample_id","split","submit_time","script_sha256"]].copy()
    if memory.sample_id.duplicated().any() or len(memory)!=frame.sample_id.nunique():
        raise ValueError("Cache replay requires exactly one memory row per job")
    memory["required"] = memory.sample_id.map(required)
    if memory[["required","script_sha256"]].isna().any().any():
        raise ValueError("Cache replay is missing required/script identities")
    return memory.sort_values(["submit_time","sample_id"],kind="stable").reset_index(drop=True)


def lru_replay(jobs: pd.DataFrame, capacity: int|None) -> tuple[np.ndarray,np.ndarray]:
    """Cold at first provided event; populate only required observed-script calls.

    Equal-time submissions observe a common pre-group cache. No intra-group
    coalescing is credited. Hits are touched before insertions; ties use sample
    order. Cached values are script embeddings, not user-dependent predictions.
    """
    if capacity is not None and capacity<0:
        raise ValueError("Capacity must be nonnegative or unbounded")
    cache: OrderedDict[str,None] = OrderedDict()
    hit = np.zeros(len(jobs),bool)
    for indices in jobs.groupby("submit_time",sort=False).indices.values():
        touches, pending = [], []
        for i in indices:
            row = jobs.iloc[i]
            key = str(row.script_sha256)
            if not bool(row.required) or key==EMPTY_KEY:
                continue
            if key in cache:
                hit[i] = True
                touches.append(key)
            else:
                pending.append(key)
        for key in touches+pending:
            cache[key] = None
            cache.move_to_end(key)
            if capacity is not None:
                while len(cache)>capacity:
                    cache.popitem(last=False)
    return hit, jobs.required.to_numpy(bool) & ~hit


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Sample SD across seeds; one row per seed/group is mandatory."""
    if frame.empty:
        return frame
    if frame.duplicated(["seed",*groups]).any():
        raise ValueError(f"Duplicate seed/group summary rows for {groups}")
    fields = [name for name in frame.select_dtypes(include="number").columns if name not in ["seed",*groups]]
    grouped = frame.groupby(groups,dropna=False)
    result = grouped[fields].agg(["mean","std"])
    result.columns = [f"{name}_{'sd' if stat=='std' else stat}" for name,stat in result.columns]
    result["seed_n"] = grouped.seed.nunique()
    return result.reset_index()


def read_population_index(path: Path) -> pd.DataFrame:
    """One row per retained event; raw targets/scripts are deliberately absent."""
    population = pd.read_csv(path,dtype={"sample_id":str,"user_id":str,"script_sha256":str})
    required = {"sample_id","user_id","script_sha256","split","submit_time","completion_time","valid_memory","valid_vram"}
    if missing := required-set(population):
        raise ValueError(f"Population index lacks {sorted(missing)}")
    if population.sample_id.duplicated().any() or population[["sample_id","user_id","script_sha256"]].isna().any().any():
        raise ValueError("Population index has missing/duplicate identities")
    if not population.script_sha256.str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("Population index has malformed script identities")
    population["script_sha256"] = population.script_sha256.str.lower()
    for name in ("submit_time","completion_time"):
        population[name] = pd.to_datetime(population[name],utc=True,errors="raise")
        if population[name].isna().any():
            raise ValueError(f"Population index has missing {name}")
    for name in ("valid_memory","valid_vram","effective_analysis_row"):
        if name in population:
            population[name] = boolean(population[name],name)
    return population


def validate_population_index(population: pd.DataFrame, base: pd.DataFrame, manifest: dict) -> None:
    """Check each predicted event against the source-population identity view."""
    key = population.set_index("sample_id")
    if not set(base.sample_id).issubset(set(key.index)):
        raise ValueError("Population index does not contain every base-prediction event")
    for name in ("user_id","script_sha256","split","submit_time","completion_time"):
        if name not in base:
            raise ValueError(f"Base predictions do not preserve {name}")
        expected = base.sample_id.map(key[name]).reset_index(drop=True)
        actual = base[name].reset_index(drop=True)
        if name=="script_sha256":
            actual = actual.str.lower()
        if not actual.equals(expected):
            raise ValueError(f"Population index/base {name} identity mismatch")
    for resource in sorted(base.resource.unique()):
        part = base.loc[base.resource.eq(resource)]
        name = "valid_"+resource
        if name not in key or not np.array_equal(part.valid.to_numpy(bool),part.sample_id.map(key[name]).to_numpy(bool)):
            raise ValueError("Population index/base resource validity mismatch")
    if "gpu_count" in key and "gpu_count" in base:
        left = pd.to_numeric(base.gpu_count,errors="raise").to_numpy(float)
        right = pd.to_numeric(base.sample_id.map(key.gpu_count),errors="raise").to_numpy(float)
        if not np.allclose(left,right,rtol=0,atol=0,equal_nan=True):
            raise ValueError("Population index/base GPU request mismatch")
    selection = (manifest.get("details") or {}).get("base_training_selection")
    if selection:
        if selection.get("source_data_sha256")!=manifest.get("data_sha256") or selection.get("source_split_fingerprint")!=manifest.get("split_fingerprint"):
            raise ValueError("Base selection/population source contract mismatch")
        for name in ("optimization_sample_ids","reserved_train_tail_sample_ids","purged_optimization_sample_ids"):
            ids = list(map(str,selection.get(name,[])))
            if ids and (not set(ids).issubset(set(key.index)) or not key.loc[ids,"split"].eq("train").all()):
                raise ValueError(f"Population index lacks training-selection identity roles: {name}")


def enrich(cascade: pd.DataFrame, base: pd.DataFrame, base_manifest: dict,
           population: pd.DataFrame|None = None) -> pd.DataFrame:
    needed = ["sample_id","resource","script_sha256"]
    if "user_known" in base:
        needed.append("user_known")
    if "script_sha256" not in base or not base.script_sha256.astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("Base predictions must preserve canonical exact-script SHA256")
    output = cascade.merge(base[needed],on=["sample_id","resource"],how="left",validate="one_to_one",suffixes=("","_base"))
    for name in ("script_sha256","user_known"):
        if name+"_base" in output:
            if not output[name].equals(output[name+"_base"]):
                raise ValueError(f"Cascade/base {name} disagrees")
            output = output.drop(columns=name+"_base")
    if output.script_sha256.isna().any():
        raise ValueError("Base predictions do not cover all cascade rows")
    output["script_sha256"] = output.script_sha256.str.lower()
    selection = (base_manifest.get("details") or {}).get("base_training_selection")
    source_frame = population if population is not None else base
    training = source_frame.loc[source_frame.split.eq("train")]
    if "effective_analysis_row" in training:
        training = training.loc[training.effective_analysis_row]
    if selection:
        # These roles partition the effective outer train; exclude outer-boundary
        # purges even when the population index retains them as history events.
        train_ids = set().union(*(set(map(str,selection.get(name,[]))) for name in (
            "optimization_sample_ids","reserved_train_tail_sample_ids","purged_optimization_sample_ids")))
        if train_ids:
            if not train_ids.issubset(set(source_frame.sample_id)):
                if population is not None:
                    raise ValueError("Population index does not cover base training roles")
                training = training.iloc[:0]
            else:
                training = training.loc[training.sample_id.isin(train_ids)]
    if not training.empty:
        output["user_outer_train_status"] = np.where(output.user_id.isin(set(training.user_id)),"seen_outer_train","new_since_outer_train")
        if selection and selection.get("optimization_sample_ids"):
            optimization_ids = set(map(str,selection["optimization_sample_ids"]))
            if not optimization_ids.issubset(set(training.sample_id)):
                raise ValueError("Training novelty source lacks optimization rows")
            optimization = training.loc[training.sample_id.isin(optimization_ids)]
            source = "optimization"
        else:
            optimization,source = training,"outer_train"
        output["script_training_status"] = np.where(output.script_sha256.isin(set(optimization.script_sha256.str.lower())),f"seen_{source}",f"unseen_{source}")
    else:
        output.attrs["novelty_skipped_reason"] = "No training population supplied: base predictions contain validation/test only. Supply --population-index to measure outer-training user and script novelty."
    if "user_known" in output:
        output["base_user_status"] = np.where(output.user_known,"known_to_base_vocabulary","unknown_to_base_vocabulary")
    output["history_subgroup"] = np.select([
        ~output.history_available,
        output.history_available & output.same_last1,
        output.history_available & output.exact_script_history_available & ~output.same_last1,
        output.history_available & ~output.exact_script_history_available,
    ],["cold_start","same_last1","older_same_script","no_same_script_history"],default="invalid")
    if output.history_subgroup.eq("invalid").any() or (output.same_last1 & ~output.exact_script_history_available).any():
        raise ValueError("Inconsistent history subgroup flags")
    output["month"] = output.submit_time.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m")
    if "gpu_count" in output:
        gpu = pd.to_numeric(output.gpu_count,errors="coerce")
        output["gpu_count_group"] = np.select([gpu.isna(),gpu.le(0),gpu.eq(1),gpu.eq(2),gpu.le(4)],
            ["missing","0","1","2","3-4"],default=">=5")
    return output


def run_diagnostics(args: argparse.Namespace) -> dict:
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Analysis output is nonempty: {out}; use another directory or --overwrite")
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.variants))!=len(args.variants):
        raise ValueError("Seeds/variants must be unique")
    if args.proposed_variant not in args.variants:
        raise ValueError("Proposed variant must be in --variants")
    frames: dict[str,list[dict]] = {name:[] for name in (
        "test_metrics", "stratified_metrics", "paired_bootstrap", "mcnemar", "risk_metrics",
        "margin_diagnostics", "daily_comparisons", "gate_diagnostics", "gate_calibration_bins",
        "gate_fit_counts", "bypass_diagnostics", "frozen_accuracy", "cache_routing", "population_shift")}
    provenance, skipped, identity = [], [], None
    population = read_population_index(args.population_index) if args.population_index else None
    population_source = None
    if args.population_index:
        population_source = {"path":str(args.population_index.resolve()),"sha256":sha256(args.population_index),"retained_event_n":len(population)}
        # The preparer may add a sidecar binding the index to the raw/split SHA.
        sidecar = args.population_index.with_suffix(".json")
        if sidecar.is_file():
            population_source["identity_manifest_sha256"] = sha256(sidecar)
            population_source["identity"] = json.loads(sidecar.read_text(encoding="utf-8"))
            expected = population_source["identity"].get("population_index_sha256")
            if expected is not None and expected!=population_source["sha256"]:
                raise ValueError("Population index sidecar SHA256 mismatch")
    for seed in args.seeds:
        variants, manifests = {}, {}
        for variant in args.variants:
            values = dict(seed=seed,variant=variant,max_tokens=args.max_tokens)
            directory = args.cascade_root/args.cascade_template.format(**values)
            data, manifest, source = read_run(directory,seed,sealed=not args.allow_unsealed)
            base_dir = args.base_root/args.base_template.format(**values)
            base, base_manifest, base_source = read_run(base_dir,seed,sealed=False)
            require_identity(manifest,base_manifest)
            if population is not None:
                validate_population_index(population,base,base_manifest)
                if population_source.get("identity"):
                    require_identity(manifest,population_source["identity"])
            if identity is not None:
                require_identity(identity,manifest)
            else:
                identity = manifest
            # This also verifies all cascade labels/times against the base output.
            common = base.loc[base["sample_id"].isin(set(data.sample_id))]
            align_predictions(data,common,"_base")
            data = enrich(data,base,base_manifest,population)
            if reason := data.attrs.get("novelty_skipped_reason"):
                skipped.append({"seed":seed,"variant":variant,"diagnostic":"training_user_and_script_novelty","reason":reason})
            variants[variant], manifests[variant] = data, manifest
            provenance.extend([{**source,"variant":variant,"role":"cascade"},{**base_source,"variant":variant,"role":"base"}])
            for scope, group in (("predicted_split",base.groupby("split")),("predicted_month",base.assign(month=base.submit_time.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m")).groupby("month"))):
                for key, block in group:
                    vram = block.loc[block.resource.eq("vram")]
                    if not len(vram):
                        continue
                    row = {"seed":seed,"variant":variant,"scope":scope,"group":key,"n":len(vram),
                           "valid_vram_n":int(vram.valid.sum()),"valid_vram_rate":float(vram.valid.mean())}
                    if "gpu_count" in vram:
                        gpu = pd.to_numeric(vram.gpu_count,errors="coerce")
                        row.update(positive_gpu_rate=float(gpu.gt(0).mean()),missing_gpu_count_n=int(gpu.isna().sum()))
                    frames["population_shift"].append(row)
            if population is not None:
                index = population.assign(month=population.submit_time.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m"))
                groups = [("retained_split",index.groupby("split")),("retained_month",index.groupby("month"))]
                if "effective_analysis_row" in index:
                    effective = index.loc[index.effective_analysis_row]
                    groups.extend([("effective_split",effective.groupby("split")),("effective_month",effective.groupby("month"))])
                for scope,group in groups:
                    for key,block in group:
                        row = {"seed":seed,"variant":variant,"scope":scope,"group":key,"n":len(block),
                            "valid_vram_n":int(block.valid_vram.sum()),"valid_vram_rate":float(block.valid_vram.mean())}
                        if "gpu_count" in block:
                            gpu = pd.to_numeric(block.gpu_count,errors="coerce")
                            row.update(positive_gpu_rate=float(gpu.gt(0).mean()),missing_gpu_count_n=int(gpu.isna().sum()))
                        frames["population_shift"].append(row)
        full = variants[args.proposed_variant]
        test = full.loc[full.split.eq(args.split)].copy()
        for variant,data in variants.items():
            paired = align_predictions(full,data,"_variant")
            variants[variant] = data
            if variant!=args.proposed_variant:
                columns = ["sample_id","resource","prediction_variant","qwen_prediction_variant"]
                test = test.merge(paired[columns].rename(columns={"prediction_variant":"variant_"+variant,
                    "qwen_prediction_variant":"base_variant_"+variant}),on=["sample_id","resource"],validate="one_to_one")
        methods = {"Cascade "+args.proposed_variant:"prediction", "Hierarchical Same-Script Last-1":"hierarchical_reference_prediction",
                   "Submission-Time Last-1":"last1_prediction", "Standalone "+args.proposed_variant:"qwen_prediction"}
        methods.update({"Cascade "+v:"variant_"+v for v in args.variants if v!=args.proposed_variant})
        methods.update({"Standalone "+v:"base_variant_"+v for v in args.variants if v!=args.proposed_variant})
        references = {"Hierarchical Same-Script Last-1":"hierarchical_reference_prediction",
                      **{"Cascade "+v:"variant_"+v for v in args.variants if v!=args.proposed_variant}}
        for resource,part in test.loc[test.valid].groupby("resource",sort=True):
            for method,column in methods.items():
                if column not in part:
                    raise ValueError(f"Missing required comparator column {column}")
                meta = {"seed":seed,"resource":resource,"method":method}
                frames["test_metrics"].append({**meta,**metrics(part.target,part[column],args.tolerance)})
                for subset,block in (("all_valid",part),(f"target_ge_{args.high_target:g}",part.loc[part.target.ge(args.high_target)])):
                    if block.empty:
                        continue
                    frames["risk_metrics"].append({**meta,"subset":subset,**risk_metrics(block.target,block[column])})
                    for margin in args.margins:
                        p = np.clip(block[column].to_numpy(float)+margin,0.,1.)
                        y = block.target.to_numpy(float)
                        frames["margin_diagnostics"].append({**meta,"subset":subset,"margin":margin,"n":len(block),
                            "below_sampled_peak_rate":float((p<y-1e-12).mean()),"mean_recommendation_ratio":float(p.mean()),
                            "mean_positive_shortfall_ratio":float(np.maximum(y-p,0).mean()),
                            "mean_positive_excess_ratio":float(np.maximum(p-y,0).mean())})
                for scope in ("history_subgroup","month","user_outer_train_status","base_user_status","script_training_status","gpu_count_group"):
                    if scope in part:
                        for key,block in part.groupby(scope):
                            frames["stratified_metrics"].append({**meta,"scope":scope,"group":key,**metrics(block.target,block[column],args.tolerance)})
            for reference,column in references.items():
                for cluster in args.clusters:
                    labels = cluster_labels(part,cluster,args.block_timezone)
                    rows = paired_bootstrap(part.target,part.prediction,part[column],labels,
                        replicates=args.bootstrap_replicates,rng=named_rng(args.bootstrap_seed,seed,resource,reference,cluster,"all_valid"),
                        confidence=args.confidence_level,tolerance=args.tolerance)
                    frames["paired_bootstrap"].extend({"seed":seed,"resource":resource,"reference":reference,"subgroup":"all_valid","cluster":cluster,**row} for row in rows)
                for subgroup,block in part.groupby("history_subgroup"):
                    rows = paired_bootstrap(block.target,block.prediction,block[column],cluster_labels(block,"day",args.block_timezone),
                        replicates=args.bootstrap_replicates,rng=named_rng(args.bootstrap_seed,seed,resource,reference,"day",subgroup),
                        confidence=args.confidence_level,tolerance=args.tolerance)
                    frames["paired_bootstrap"].extend({"seed":seed,"resource":resource,"reference":reference,"subgroup":subgroup,"cluster":"day",**row} for row in rows)
                frames["mcnemar"].append({"seed":seed,"resource":resource,"reference":reference,**mcnemar(part.target,part.prediction,part[column],args.tolerance)})
                days = cluster_labels(part,"day",args.block_timezone)
                for day,block in part.assign(day=days).groupby("day"):
                    proposed,ref = metrics(block.target,block.prediction,args.tolerance),metrics(block.target,block[column],args.tolerance)
                    frames["daily_comparisons"].append({"seed":seed,"resource":resource,"reference":reference,"day":day,"n":len(block),
                        **{"delta_"+m:proposed[m]-ref[m] for m in METRICS}})
        # Gate and component diagnostics are performed per variant, with policies frozen.
        for variant,data in variants.items():
            manifest = manifests[variant]
            contract_path = args.cascade_root/args.cascade_template.format(seed=seed,variant=variant,max_tokens=args.max_tokens)/"checkpoint/checkpoint_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
            model_cfg = contract.get("cascade_model_config",{})
            cap = float(model_cfg.get("reliability_error_cap",.10))
            exact_required = bool(model_cfg.get("require_exact_script_for_bypass",True))
            calibration = (manifest.get("details") or {}).get("calibration_protocol",{})
            cutoff_value = calibration.get("selection_start_submit_time")
            if cutoff_value is None:
                skipped.append({"seed":seed,"variant":variant,"diagnostic":"gate_fit_and_selection","reason":"missing archived inner cutoff"})
                periods = [(args.split,data.split.eq(args.split))]
            else:
                cutoff = pd.Timestamp(cutoff_value)
                if cutoff.tzinfo is None:
                    raise ValueError("Gate selection cutoff lacks timezone")
                fit = data.split.eq("validation") & data.submit_time.lt(cutoff) & data.completion_time.lt(cutoff)
                periods = [("gate_fit",fit),("validation_selection",data.split.eq("validation") & data.submit_time.ge(cutoff)),(args.split,data.split.eq(args.split))]
            for period,mask in periods:
                for resource in sorted(data.resource.unique()):
                    eligible = mask & data.resource.eq(resource) & data.valid & data.resource_applicable & data.history_available
                    if exact_required:
                        eligible &= data.exact_script_history_available
                    block = data.loc[eligible]
                    diagnostics,bins = probability_diagnostics(reliable_labels(block,cap),block.reliability_probability)
                    meta = {"seed":seed,"variant":variant,"resource":resource,"period":period}
                    frames["gate_diagnostics"].append({**meta,**diagnostics})
                    frames["gate_calibration_bins"].extend({**meta,**row} for row in bins)
                    if period=="gate_fit":
                        frames["gate_fit_counts"].append({**meta,"pre_fit_n":len(block),"pre_positive_n":int(reliable_labels(block,cap).sum()),
                            "post_fit_n":int((mask & data.resource.eq(resource) & data.valid & data.history_available).sum())})
                    bypass = data.loc[mask & data.resource.eq(resource) & data.valid & data.history_bypass]
                    if len(bypass):
                        h = abs(bypass.history_expert_prediction-bypass.target).to_numpy(float)
                        q = abs(bypass.qwen_prediction-bypass.target).to_numpy(float)
                        post = abs(bypass.post_blend_prediction-bypass.target).to_numpy(float)
                        frames["bypass_diagnostics"].append({**meta,"n":len(bypass),"reliable_precision":float(reliable_labels(bypass,cap).mean()),
                            "bypass_mae":float(h.mean()),"signed_regret_vs_qwen_mae":float((h-q).mean()),
                            "positive_regret_vs_qwen_mean":float(np.maximum(h-q,0).mean()),
                            "positive_regret_vs_qwen_rate":float((h>q+1e-12).mean()),"signed_regret_vs_post_mae":float((h-post).mean())})
            cases = [("full","full",None),("no_bypass_frozen","no_bypass",None),("no_post_frozen","no_post",None),
                     ("no_trust_frozen","no_trust",None),("unconditional_exact_bypass","exact_bypass",None)]
            cases.extend((f"threshold_{value:g}","threshold",value) for value in args.thresholds)
            for name,mode,threshold in cases:
                prediction,required = frozen_operation(data,mode,threshold)
                if name=="full" and not np.allclose(prediction,data.prediction,rtol=0,atol=1e-12):
                    raise ValueError("Saved Full predictions differ from frozen reconstruction")
                jobs = make_jobs(data,required)
                for capacity in (None,*args.cache_capacities):
                    hit,forward = lru_replay(jobs,capacity)
                    if name=="full" and capacity is None:
                        expected = data.loc[data.resource.eq("memory")].set_index("sample_id")
                        if not np.array_equal(hit,jobs.sample_id.map(expected.embedding_cache_hit_job).to_numpy(bool)):
                            raise ValueError("Saved unbounded cache hits do not reproduce")
                        if not np.array_equal(forward,jobs.sample_id.map(expected.encoder_forward_required_job).to_numpy(bool)):
                            raise ValueError("Saved encoder-forward flags do not reproduce")
                    for period,block in jobs.groupby("split"):
                        idx = block.index.to_numpy()
                        need = block.required.to_numpy(bool)
                        frames["cache_routing"].append({"seed":seed,"variant":variant,"operation":name,"capacity_entries":"unbounded" if capacity is None else str(capacity),
                            "period":period,"n":len(block),"qwen_required_n":int(need.sum()),"qwen_required_rate":float(need.mean()),
                            "cache_hit_n":int(hit[idx].sum()),"encoder_forward_n":int(forward[idx].sum()),"encoder_forward_rate":float(forward[idx].mean())})
                for period,mask in periods:
                    if period=="gate_fit":
                        continue
                    for resource in sorted(data.resource.unique()):
                        take = mask & data.resource.eq(resource) & data.valid
                        if take.any():
                            frames["frozen_accuracy"].append({"seed":seed,"variant":variant,"operation":name,"period":period,"resource":resource,
                                **metrics(data.loc[take,"target"],prediction[take.to_numpy()],args.tolerance)})
    mc = frames["mcnemar"]
    for seed in args.seeds:
        for reference in sorted({row["reference"] for row in mc}):
            group = [row for row in mc if row["seed"]==seed and row["reference"]==reference]
            for row,value in zip(group,holm_adjust([row["p_value"] for row in group])):
                row["p_holm_within_seed_reference"] = value
                row["holm_family_size"] = len(group)
    out.mkdir(parents=True,exist_ok=True)
    summaries = {"test_metrics":["resource","method"],"stratified_metrics":["resource","method","scope","group"],
        "risk_metrics":["resource","method","subset"],"margin_diagnostics":["resource","method","subset","margin"],
        "gate_diagnostics":["variant","resource","period"],"gate_fit_counts":["variant","resource","period"],
        "bypass_diagnostics":["variant","resource","period"],"frozen_accuracy":["variant","operation","period","resource"],
        "cache_routing":["variant","operation","capacity_entries","period"],"population_shift":["variant","scope","group"]}
    inventory = {}
    for name,rows in frames.items():
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["status"])
        path = out/(name+"_per_seed.csv")
        frame.to_csv(path,index=False)
        inventory[path.name] = {"rows":len(frame),"sha256":sha256(path)}
        if name in summaries and not frame.empty:
            result = summarize(frame,summaries[name])
            path = out/(name+"_summary.csv")
            result.to_csv(path,index=False)
            inventory[path.name] = {"rows":len(result),"sha256":sha256(path)}
    result = {"schema":"reviewer-fixed-prediction-diagnostics/v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "seeds":args.seeds,"variants":args.variants,"split":args.split,"bootstrap_replicates":args.bootstrap_replicates,
        "bootstrap_seed":args.bootstrap_seed,"confidence_level":args.confidence_level,"block_timezone":args.block_timezone,
        "clusters":args.clusters,"delta_definition":"proposed minus reference","summary_sd":"sample standard deviation, ddof=1",
        "training_or_selection_performed":False,"inputs":provenance,"outputs":inventory,"skipped":skipped,
        "analysis_source_sha256":sha256(Path(__file__)),"population_index":population_source,
        "caveats":["Test predictions are retrospective; this analysis does not create an untouched confirmation set.",
            "Seeds share one dataset and test period; per-seed intervals condition on fixed fitted models.",
            "Intervals are pointwise percentile intervals; no multiplicity correction across metrics/strata/clustering choices.",
            "Holm correction is only across resource-wise McNemar tests within each seed/reference family; these job-level tests are sensitivity analyses.",
            "3day/7day clusters are nonoverlapping calendar blocks anchored at the first eligible day; not moving-block bootstrap.",
            "User/script cluster bootstraps are separate sensitivities and do not jointly control time-user-script dependence.",
            "Threshold sweeps and unconditional exact bypass are frozen retrospective sensitivities, never test-selected deployed policies.",
            "Unconditional exact bypass retains the variant's selected robust-history expert; it does not silently replace that expert with ordinary Last-1.",
            "Counterfactuals reuse saved post-gate outputs even on bypassed jobs; they change an operation, not retraining or policy selection.",
            "Cache replay starts empty at first cascade prediction event (normally validation), inserts required observed-script calls only, and counts no intra-timestamp reuse.",
            "LRU replay estimates encoder call counts, not measured serving latency, queueing, byte capacity, or bounded-memory hardware performance.",
            "NoCode removes learned script encoding but retains script identity for causal history/routing/cache.",
            "GPU-count strata are scheduler request strata, not accelerator-type or hardware-capacity strata.",
            "High-target and fixed normalized margins describe sampled-ratio shortfall, not actual OOM probability or scheduler savings.",
            "Aggregate CSVs omit job/user/script identifiers; private local provenance paths may still require redaction before public release."]}
    if args.config:
        result["config_sha256"] = sha256(args.config)
    (out/"analysis_manifest.json").write_text(json.dumps(jsonable(result),indent=2,allow_nan=False),encoding="utf-8")
    return result


def parse_args(argv: list[str]|None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root",required=True,type=Path)
    parser.add_argument("--cascade-root",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--config",type=Path,help="Optional config provenance hash; no YAML parsing or configuration mutation")
    parser.add_argument("--population-index",type=Path,help="Prepared all-event population index; needed for novelty and full-trace coverage when base outputs contain validation/test only")
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(40,45)))
    parser.add_argument("--variants",nargs="+",default=["full","no_code"])
    parser.add_argument("--proposed-variant",default="full")
    parser.add_argument("--max-tokens",type=int,default=4096)
    parser.add_argument("--base-template",default="qwen/tokens_{max_tokens}/{variant}/seed_{seed}")
    parser.add_argument("--cascade-template",default="reliability_cascade/tokens_{max_tokens}/{variant}/seed_{seed}")
    parser.add_argument("--split",default="test",choices=["test"])
    parser.add_argument("--bootstrap-replicates",type=int,default=10000)
    parser.add_argument("--bootstrap-seed",type=int,default=2026)
    parser.add_argument("--confidence-level",type=float,default=.95)
    parser.add_argument("--clusters",nargs="+",choices=["day","3day","7day","user","script"],default=["day","3day","7day","user","script"])
    parser.add_argument("--block-timezone",default="UTC")
    parser.add_argument("--tolerance",type=float,default=.10)
    parser.add_argument("--high-target",type=float,default=.80)
    parser.add_argument("--margins",nargs="+",type=float,default=[0.,.05,.10,.20])
    parser.add_argument("--thresholds",nargs="*",type=float,default=[0.,.25,.50,.75,.90,.95,1.01])
    parser.add_argument("--cache-capacities",nargs="*",type=int,default=[128,256,512,1024,2048])
    parser.add_argument("--allow-unsealed",action="store_true",help="For legacy/synthetic diagnostics only; skip original cascade integrity seals")
    parser.add_argument("--overwrite",action="store_true",help="Replace analysis aggregates, never model/run inputs")
    args = parser.parse_args(argv)
    if args.bootstrap_replicates<1 or not 0<args.confidence_level<1 or not math.isclose(args.tolerance,.10,rel_tol=0,abs_tol=1e-12):
        parser.error("Invalid bootstrap/confidence/tolerance settings")
    if not 0<=args.high_target<=1 or any(x<0 or not math.isfinite(x) for x in args.margins):
        parser.error("Invalid normalized target threshold/margins")
    if any(x<0 for x in args.cache_capacities) or len(set(args.cache_capacities))!=len(args.cache_capacities):
        parser.error("Cache capacities must be unique and nonnegative")
    return args


def main(argv: list[str]|None = None) -> int:
    args = parse_args(argv)
    result = run_diagnostics(args)
    print(json.dumps({"output_dir":str(args.output_dir.resolve()),"seeds":result["seeds"],"aggregate_files":len(result["outputs"])},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
