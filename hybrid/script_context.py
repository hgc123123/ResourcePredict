from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from common.data import DatasetBundle
from common.history import build_causal_history_long, script_sha256


PAIR_KEYS = ["sample_id", "resource"]


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def attach_script_history_relation(
    paired: pd.DataFrame,
    bundle: DatasetBundle,
) -> pd.DataFrame:
    """Attach a causal same-script flag and verify it against Last-1."""
    required = {"sample_id", "resource", "history_available", "last1_prediction"}
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"Paired predictions lack script-context columns: {missing}")

    history = build_causal_history_long(bundle, k=1, ema_alpha=1.0)
    history_protocol = dict(history.attrs.get("causal_history_protocol", {}))
    history = history[
        [
            "sample_id",
            "resource",
            "history_available",
            "history_last_value",
            "history_source_sample_id",
            "history_same_script_available",
            "history_same_script_last_value",
            "history_same_script_age_seconds",
            "history_same_script_source_sample_id",
        ]
    ].copy()
    history["sample_id"] = history["sample_id"].astype(str)
    history = history.rename(
        columns={
            name: f"replay_{name}"
            for name in history.columns
            if name not in PAIR_KEYS
        }
    )

    output = paired.copy()
    output["sample_id"] = output["sample_id"].astype(str)

    qwen_hash_column = _first_column(
        output, ("script_sha256", "script_sha256_qwen")
    )
    if qwen_hash_column is None:
        raise ValueError(
            "Qwen predictions do not contain script_sha256 for script replay binding"
        )
    current_scripts = bundle.frame[["sample_id", "code"]].copy()
    current_scripts["sample_id"] = current_scripts["sample_id"].astype(str)
    if current_scripts["sample_id"].duplicated().any():
        raise ValueError("Configured prediction population has duplicate sample IDs")
    current_scripts["replay_script_sha256"] = current_scripts["code"].map(
        script_sha256
    )
    replay_hash_by_id = current_scripts.set_index("sample_id")[
        "replay_script_sha256"
    ]
    replay_hashes = output["sample_id"].map(replay_hash_by_id)
    if replay_hashes.isna().any():
        raise ValueError("Some paired predictions have no configured script identity")
    artifact_hashes = output[qwen_hash_column].astype("string").fillna("").astype(str)
    script_hash_mismatch = artifact_hashes.to_numpy() != replay_hashes.to_numpy()
    if script_hash_mismatch.any():
        raise ValueError(
            "Configured redacted scripts disagree with Qwen script hashes for "
            f"{int(script_hash_mismatch.sum())} rows"
        )
    output["_verified_script_sha256"] = replay_hashes.to_numpy()

    output = output.merge(
        history,
        on=PAIR_KEYS,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if output["replay_history_available"].isna().any():
        raise ValueError("Some paired predictions are absent from causal history replay")

    available = output["history_available"].astype(bool).to_numpy()
    replay_available = output["replay_history_available"].astype(bool).to_numpy()
    if not np.array_equal(available, replay_available):
        raise ValueError("Last-1 availability disagrees with causal history replay")

    last1 = pd.to_numeric(output["last1_prediction"], errors="coerce").to_numpy(float)
    replay_last1 = pd.to_numeric(
        output["replay_history_last_value"], errors="coerce"
    ).to_numpy(float)
    mismatch = available & ~np.isclose(last1, replay_last1, rtol=0.0, atol=1e-12)
    if mismatch.any():
        raise ValueError(
            "Last-1 predictions disagree with causal history replay for "
            f"{int(mismatch.sum())} rows"
        )

    source_column = _first_column(
        output,
        ("history_source_sample_id", "history_source_sample_id_last1"),
    )
    if source_column is None:
        raise ValueError("Last-1 predictions do not identify their history source")
    source_ids = output[source_column].astype("string").fillna("").astype(str)
    replay_source_ids = (
        output["replay_history_source_sample_id"]
        .astype("string")
        .fillna("")
        .astype(str)
    )
    source_mismatch = available & (source_ids.to_numpy() != replay_source_ids.to_numpy())
    if source_mismatch.any():
        raise ValueError(
            "Last-1 source IDs disagree with causal history replay for "
            f"{int(source_mismatch.sum())} rows"
        )

    same_script_available = (
        output["replay_history_same_script_available"].astype(bool).to_numpy()
    )
    same_script_prediction = pd.to_numeric(
        output["replay_history_same_script_last_value"], errors="coerce"
    ).to_numpy(float)
    invalid_same_prediction = same_script_available & ~np.isfinite(
        same_script_prediction
    )
    if invalid_same_prediction.any():
        raise ValueError(
            "Causal same-script history has a non-finite value for "
            f"{int(invalid_same_prediction.sum())} rows"
        )
    same_script_source = (
        output["replay_history_same_script_source_sample_id"]
        .astype("string")
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    same_script_as_last1 = (
        available
        & same_script_available
        & (replay_source_ids.to_numpy() == same_script_source)
    )
    if np.any(same_script_as_last1 & ~available):
        raise AssertionError("A cold-start row cannot match a Last-1 script")
    value_mismatch = same_script_as_last1 & ~np.isclose(
        same_script_prediction, last1, rtol=0.0, atol=1e-12
    )
    if value_mismatch.any():
        raise ValueError(
            "Same-source script history disagrees with Last-1 for "
            f"{int(value_mismatch.sum())} rows"
        )

    output["same_script_history_available"] = same_script_available
    output["same_script_history_prediction"] = same_script_prediction
    output["same_script_history_age_seconds"] = pd.to_numeric(
        output["replay_history_same_script_age_seconds"], errors="coerce"
    ).to_numpy(float)
    output["same_script_history_source_sample_id"] = same_script_source
    output["same_script_as_last1"] = same_script_as_last1
    output["history_script_relation"] = np.select(
        [~available, same_script_as_last1],
        ["no_history", "same_script"],
        default="changed_script",
    )
    fingerprint_rows = sorted(
        zip(
            output["sample_id"].astype(str),
            output["_verified_script_sha256"].astype(str),
        ),
        key=lambda item: item[0],
    )
    fingerprint_payload = "\n".join(
        f"{sample_id}\0{script_hash}" for sample_id, script_hash in fingerprint_rows
    )
    history_protocol["prediction_script_mapping_sha256"] = hashlib.sha256(
        fingerprint_payload.encode("utf-8")
    ).hexdigest()
    history_protocol["qwen_script_hashes_verified"] = True
    history_protocol["qwen_script_hash_column"] = qwen_hash_column
    output = output.drop(columns="_verified_script_sha256")
    output.attrs["causal_script_history_protocol"] = history_protocol
    return output


__all__ = ["attach_script_history_relation"]
