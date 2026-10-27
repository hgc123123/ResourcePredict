"""Verified, label-independent batch-one base predictions for timing parity.

The reference executes the original model forward, without cache or routing.
Its generation is deliberately separate from the timed online replay.  A final
completion record seals both files; partial or stale references are never reused.
Torch and Qwen imports stay inside generation so reuse checks can run on a CPU
machine with only pandas and NumPy installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "online-study-batch-one-base-reference-v1"
CSV_NAME = "base_batch_one.csv"
METADATA_NAME = "base_batch_one.json"
COMPLETE_NAME = "COMPLETE.json"
COLUMNS = ("sample_id", "qwen_memory", "qwen_vram", "gate_code", "gate_user", "gate_metadata", "user_known")
NUMERIC_COLUMNS = COLUMNS[1:-1]
NUMERICAL_MODE = {
    "batch_size": 1,
    "no_cache": True,
    "all_jobs_forwarded": True,
    "original_model_forward": True,
    "label_dependent_filtering": False,
    "training": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_identity(identity: dict) -> dict:
    if not isinstance(identity, dict) or not identity:
        raise ValueError("A nonempty JSON-serializable reference identity is required")
    try:
        return json.loads(_canonical_bytes(identity))
    except (TypeError, ValueError) as error:
        raise ValueError("Reference identity must contain only finite JSON-serializable values") from error


def _population_ids(frame: pd.DataFrame) -> list[str]:
    if "sample_id" not in frame or frame.empty or frame.sample_id.isna().any():
        raise ValueError("Reference population requires nonempty, nonmissing sample_id values")
    ids = frame.sample_id.astype(str).tolist()
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Reference population sample_id values must be unique and nonempty")
    return ids


def _validate_frame(reference: pd.DataFrame, expected_ids: list[str]) -> pd.DataFrame:
    if list(reference.columns) != list(COLUMNS):
        raise ValueError("Base reference CSV columns do not match its schema")
    actual_ids = _population_ids(reference)
    if actual_ids != expected_ids:
        raise ValueError("Base reference population/order differs from the requested frame")
    try:
        values = reference[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Base reference contains nonnumeric model outputs") from error
    if not np.isfinite(values).all():
        raise ValueError("Base reference contains non-finite model outputs")
    # Do not accept arbitrary strings as truthy flags.
    known = reference.user_known.astype(str).str.lower()
    if not known.isin(("true", "false", "0", "1")).all():
        raise ValueError("Base reference user_known must be a boolean flag")
    reference = reference.copy()
    reference["user_known"] = known.isin(("true", "1"))
    return reference


def _expected_contract(identity: dict, ids: list[str]) -> dict:
    normalized = _json_identity(identity)
    return {
        "schema": SCHEMA,
        "identity": normalized,
        "identity_sha256": hashlib.sha256(_canonical_bytes(normalized)).hexdigest(),
        "population_sha256": hashlib.sha256(_canonical_bytes(ids)).hexdigest(),
        "reference_source_sha256": _sha256(Path(__file__)),
        "n": len(ids),
        "numerical_mode": NUMERICAL_MODE,
    }


def _load_reference(directory: Path, *, identity: dict, ids: list[str]) -> tuple[pd.DataFrame, dict]:
    expected = _expected_contract(identity, ids)
    paths = {name: directory / name for name in (CSV_NAME, METADATA_NAME, COMPLETE_NAME)}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            f"Incomplete batch-one reference at {directory}: missing {missing}. "
            "Preserve it for inspection and select a new reference directory; nothing was deleted."
        )
    try:
        complete = json.loads(paths[COMPLETE_NAME].read_text(encoding="utf-8"))
        metadata = json.loads(paths[METADATA_NAME].read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read sealed batch-one reference at {directory}") from error
    if complete.get("schema") != SCHEMA or complete.get("complete") is not True:
        raise ValueError("Batch-one reference lacks a valid completion record")
    hashes = complete.get("files", {})
    for name in (CSV_NAME, METADATA_NAME):
        if hashes.get(name) != _sha256(paths[name]):
            raise ValueError(f"Batch-one reference integrity failure: {name}")
    changed = [key for key, value in expected.items() if metadata.get(key) != value]
    if changed:
        raise ValueError(
            f"Batch-one reference identity/population/source changed: {changed}; "
            "use a new reference directory rather than replacing the previous reference."
        )
    if metadata.get("base_csv_sha256") != hashes[CSV_NAME]:
        raise ValueError("Batch-one reference metadata and CSV hash disagree")
    try:
        reference = pd.read_csv(paths[CSV_NAME], dtype={"sample_id": str}, keep_default_na=False)
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read batch-one reference CSV") from error
    reference = _validate_frame(reference, ids)
    return reference, {**metadata, "reused": True, "directory": str(directory)}


def _atomic_write(path: Path, write) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _seal_reference(directory: Path, reference: pd.DataFrame, metadata: dict) -> dict:
    """Write data first and a completion seal last; never alter an existing file."""
    for name in (CSV_NAME, METADATA_NAME, COMPLETE_NAME):
        if (directory / name).exists():
            raise FileExistsError(f"Refusing to replace existing reference file: {directory / name}")
    _atomic_write(directory / CSV_NAME, lambda handle: reference.to_csv(handle, index=False, float_format="%.17g"))
    metadata = {**metadata, "base_csv_sha256": _sha256(directory / CSV_NAME)}
    _atomic_write(directory / METADATA_NAME, lambda handle: json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    seal = {"schema": SCHEMA, "complete": True, "files": {
        name: _sha256(directory / name) for name in (CSV_NAME, METADATA_NAME)
    }}
    _atomic_write(directory / COMPLETE_NAME, lambda handle: json.dump(seal, handle, sort_keys=True, indent=2))
    return metadata


def load_or_build_base_reference(
    directory: Path, *, frame, model, preprocessors, collator, device, dtype, identity: dict,
) -> tuple[pd.DataFrame, dict]:
    """Reuse an exactly matching reference or generate every row with batch one.

    ``identity`` must bind the immutable checkpoint, source/data/split hashes,
    hardware, software, dtype and thread settings.  Generation time, model
    loading and reference validation are excluded from online latency results.
    References are independent of each cascade operation and timing repeat.
    """
    directory = Path(directory).expanduser().resolve()
    ids = _population_ids(frame)
    contract = _expected_contract(identity, ids)
    if directory.exists() and any(directory.iterdir()):
        return _load_reference(directory, identity=identity, ids=ids)
    directory.mkdir(parents=True, exist_ok=True)

    # Lazy imports keep integrity/reuse tests independent of Torch/Transformers.
    import torch
    from qwen.benchmark import _autocast, _synchronize
    from qwen.dataset import QwenJobDataset

    dataset = QwenJobDataset(frame.reset_index(drop=True), preprocessors, training=False, require_any_label=False)
    if len(dataset) != len(ids):
        raise ValueError("Original dataset unexpectedly filtered the batch-one reference population")
    was_training = bool(model.training)
    model.eval()
    _synchronize(device)
    start = time.perf_counter()
    rows = []
    try:
        with torch.inference_mode():
            for index, sample_id in enumerate(ids):
                example = dataset[index]
                batch = collator([example])
                if int(batch["input_ids"].shape[0]) != 1:
                    raise ValueError("Base reference collator must produce exactly one example")
                inputs = {name: batch[name].to(device) for name in ("input_ids", "attention_mask", "user_ids", "metadata")}
                # The original forward is independent of online code-embedding reuse.
                with _autocast(device, dtype):
                    result = model(**inputs)
                predictions = result["predictions"].detach().float().cpu().numpy()
                gates = result["gate_weights"].detach().float().cpu().numpy()
                if predictions.shape != (1, 2) or gates.shape != (1, 3):
                    raise ValueError("Unexpected original model output shape in batch-one reference")
                if not np.isfinite(predictions).all() or not np.isfinite(gates).all():
                    raise ValueError(f"Non-finite original model outputs for reference row {index}")
                rows.append({"sample_id": sample_id, "qwen_memory": float(predictions[0, 0]),
                    "qwen_vram": float(predictions[0, 1]), "gate_code": float(gates[0, 0]),
                    "gate_user": float(gates[0, 1]), "gate_metadata": float(gates[0, 2]),
                    "user_known": bool(example["user_id"] != 0)})
                if (index + 1) % 100 == 0 or index + 1 == len(ids):
                    print(f"[BASE-REFERENCE] {index + 1}/{len(ids)} batch-one no-cache forwards", flush=True)
        _synchronize(device)
    finally:
        model.train(was_training)
    reference = _validate_frame(pd.DataFrame(rows, columns=COLUMNS), ids)
    metadata = {**contract, "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_seconds_excluded_from_timing": time.perf_counter() - start,
        "encoder_forwards": len(ids) if "code" in model.enabled_modalities else 0,
        "base_forward_calls": len(ids)}
    metadata = _seal_reference(directory, reference, metadata)
    # Read back the sealed representation, exercising the same checks as reuse.
    loaded, validated = _load_reference(directory, identity=identity, ids=ids)
    return loaded, {**validated, "reused": False}
