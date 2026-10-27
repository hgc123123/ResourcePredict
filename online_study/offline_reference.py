"""Canonical offline cascade replay, independent of the streaming runtime.

The archived batched outputs and the new uncached batch-one outputs are kept
separate. Replaying the latter never replaces the paper's archived results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from online_study.measurement import frozen_reference, parity_check, strict_bool


BASE_COLUMNS = ["sample_id", "qwen_memory", "qwen_vram", "gate_code", "gate_user", "gate_metadata", "user_known"]


def archived_base_jobs(base: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "resource"]
    if base.duplicated(keys).any():
        raise ValueError("Duplicate archived base prediction keys")
    memory = base.loc[base.resource.eq("memory")].copy()
    vram = base.loc[base.resource.eq("vram"), ["sample_id", "prediction"]].rename(columns={"prediction": "qwen_vram"})
    output = memory.rename(columns={"prediction": "qwen_memory"}).merge(vram, on="sample_id", validate="one_to_one")
    if len(output) * 2 != len(base):
        raise ValueError("Archived base outputs lack a resource")
    output["user_known"] = strict_bool(output.user_known, "user_known")
    return output[BASE_COLUMNS].copy()


def canonical_predictions(archive, base_jobs, bundle, models, *, history_k, half_life_hours):
    from reliability_cascade.features import build_cascade_features
    from common.history import script_sha256
    required = ["sample_id", "resource", "split", "target", "valid", "last1_prediction", "history_available"]
    paired = archive[required].copy()
    for name in ("valid", "history_available"):
        paired[name] = strict_bool(paired[name], name)
    paired["sample_id"] = paired.sample_id.astype(str)
    context = bundle.frame.loc[bundle.frame.sample_id.astype(str).isin(set(paired.sample_id)), ["sample_id", "code"]]
    hashes = {str(row.sample_id): script_sha256(row.code) for row in context.itertuples(index=False)}
    paired["script_sha256"] = paired.sample_id.map(hashes)
    if paired.script_sha256.isna().any():
        raise ValueError("Offline reference job is absent from the effective trace")
    jobs = base_jobs[BASE_COLUMNS].copy()
    jobs["sample_id"] = jobs.sample_id.astype(str)
    if jobs.sample_id.duplicated().any():
        raise ValueError("Duplicate batch-one base job keys")
    paired = paired.merge(jobs, on="sample_id", how="left", validate="many_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Base reference does not cover the sealed prediction population")
    paired = paired.drop(columns="_merge")
    paired["qwen_prediction"] = np.where(paired.resource.eq("memory"), paired.qwen_memory, paired.qwen_vram)
    paired["user_known"] = strict_bool(paired.user_known, "user_known")
    if not np.isfinite(paired[["qwen_prediction", "gate_code", "gate_user", "gate_metadata"]].to_numpy(float)).all():
        raise ValueError("Non-finite independent base reference")
    features = build_cascade_features(paired, bundle, history_k=history_k, half_life_hours=half_life_hours)
    output = archive.copy()
    for name in ("history_expert_prediction", "post_history_weight", "post_blend_prediction", "prediction"):
        output[name] = np.nan
    output["resource_applicable"] = features.frame.resource_applicable.to_numpy(bool)
    output["history_bypass"] = False
    output["qwen_prediction"] = features.frame.qwen_prediction.to_numpy(float)
    for name in ("gate_code", "gate_user", "gate_metadata", "user_known"):
        output[name] = features.frame[name].to_numpy()
    for resource, model in models.items():
        values = model.predict(features)  # original vectorized offline implementation
        mask = values["resource_mask"]
        for target, source in (("prediction", "prediction"),
                               ("history_expert_prediction", "history_prediction"),
                               ("post_history_weight", "post_gamma"),
                               ("post_blend_prediction", "post_prediction"),
                               ("history_bypass", "history_bypass")):
            output.loc[mask, target] = values[source][mask]
    if not np.array_equal(output[["sample_id", "resource"]].to_numpy(), features.frame[["sample_id", "resource"]].to_numpy()):
        raise ValueError("Canonical offline replay changed row ordering")
    return output


def reference_records(long_frame):
    memory = long_frame.loc[long_frame.resource.eq("memory")].copy()
    vram = long_frame.loc[long_frame.resource.eq("vram")].copy()
    fields = ["sample_id", "prediction", "history_bypass", "resource_applicable"]
    memory = memory[fields].rename(columns={"prediction": "prediction_memory", "history_bypass": "bypass_memory", "resource_applicable": "applicable_memory"})
    vram = vram[fields].rename(columns={"prediction": "prediction_vram", "history_bypass": "bypass_vram", "resource_applicable": "applicable_vram"})
    return memory.merge(vram, on="sample_id", validate="one_to_one")


def structural_archive_check(canonical, archive, operation):
    expected, actual = frozen_reference(archive, operation), frozen_reference(canonical, operation)
    actual = actual.loc[actual.split.eq("test")]
    result = parity_check(reference_records(actual), expected, 1e-10)
    result["reference_mode"] = "archived batched base inputs; independent canonical offline cascade"
    return result


def inject_live_inputs(base_jobs, measured):
    output = base_jobs.copy()
    lookup = measured.set_index("sample_id")
    if not lookup.index.is_unique:
        raise ValueError("Duplicate live input job keys")
    used = output.sample_id.isin(lookup.index)
    for index in output.index[used]:
        sample_id = str(output.loc[index, "sample_id"])
        row = lookup.loc[sample_id]
        if not bool(row.model_path_required):
            continue
        for name in ("qwen_memory", "qwen_vram", "gate_code", "gate_user", "gate_metadata", "user_known"):
            output.loc[index, name] = row[name]
    return output


def base_parity(measured, independent, atol=1e-5):
    live = measured.loc[measured.model_path_required].copy()
    columns = ["qwen_memory", "qwen_vram", "gate_code", "gate_user", "gate_metadata"]
    merged = live[["sample_id", *columns, "user_known"]].merge(independent[BASE_COLUMNS], on="sample_id", how="left", suffixes=("_live", "_reference"), validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Independent batch-one reference lacks model-path jobs")
    deltas = {name: np.abs(merged[name+"_live"].to_numpy(float) - merged[name+"_reference"].to_numpy(float)) for name in columns}
    if any(not np.isfinite(values).all() for values in deltas.values()):
        raise ValueError("Non-finite batch-one model parity values")
    mismatched_user = int(np.sum(strict_bool(merged.user_known_live, "user_known_live") != strict_bool(merged.user_known_reference, "user_known_reference")))
    maximum = max((float(v.max()) for v in deltas.values() if len(v)), default=0.)
    result = {"performed": True, "reference_mode": "independent uncached model.forward; batch one",
              "jobs": len(live), "atol": atol, "max_abs_difference": maximum,
              "max_by_field": {name: float(v.max()) if len(v) else 0. for name, v in deltas.items()},
              "user_mismatch_jobs": mismatched_user, "passed": maximum <= atol and mismatched_user == 0}
    if not result["passed"]:
        raise ValueError(f"Batch-one base/embedding parity failed: {result}")
    return result


def comparison_details(measured, matched, archived):
    """One row per resource; saved even when a later parity check fails."""
    live = []
    for resource in ("memory", "vram"):
        part = measured[["sample_id", "model_path_required", "cache_hit", "encoder_forward", f"prediction_{resource}", f"qwen_{resource}", "gate_code", "gate_user", "gate_metadata"]].rename(columns={f"prediction_{resource}": "live_prediction", f"qwen_{resource}": "live_qwen_prediction", "gate_code": "live_gate_code", "gate_user": "live_gate_user", "gate_metadata": "live_gate_metadata"})
        part["resource"] = resource
        live.append(part)
    result = pd.concat(live, ignore_index=True)
    for name, reference in (("matched", matched), ("archived", archived)):
        fields = ["sample_id", "resource", "prediction", "resource_applicable", "history_bypass",
                  "qwen_prediction", "gate_code", "gate_user", "gate_metadata", "post_history_weight"]
        other = reference.loc[reference.split.eq("test"), fields].rename(columns={c: name+"_"+c for c in fields if c not in ("sample_id", "resource")})
        result = result.merge(other, on=["sample_id", "resource"], how="left", validate="one_to_one")
        result[name+"_abs_difference"] = abs(result.live_prediction - result[name+"_prediction"])
        for field in ("qwen_prediction", "gate_code", "gate_user", "gate_metadata"):
            result[name+"_"+field+"_abs_difference"] = abs(result["live_"+field] - result[name+"_"+field])
    return result


def drift_summary(details):
    output = {"role": "descriptive comparison with archived batched execution; never a replacement for matched-reference parity", "by_resource": {}}
    for resource, part in details.groupby("resource", sort=True):
        applicable = strict_bool(part.archived_resource_applicable, "archived_resource_applicable")
        values = part.loc[applicable, "archived_abs_difference"].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite archived execution drift")
        output["by_resource"][resource] = {"n": len(values), "mean_abs_difference": float(values.mean()) if len(values) else 0.,
            "max_abs_difference": float(values.max()) if len(values) else 0.,
            "p95_abs_difference": float(np.percentile(values, 95)) if len(values) else 0.,
            "rows_exceeding_original_0_005": int(np.sum(values > .005))}
    return output
