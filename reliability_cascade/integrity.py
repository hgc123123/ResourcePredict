from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from common.config import get_value, serializable_config
from common.data import sha256_file


ROUTE_COLUMNS = (
    "sample_id",
    "resource",
    "history_bypass",
    "qwen_required_resource",
    "qwen_required_job",
    "encoder_forward_required_job",
)


INTEGRITY_SCHEMA = "reliability-cascade-artifacts/v1"
RUN_REQUEST_SCHEMA = "reliability-cascade-run-request/v1"
COMPLETION_SCHEMA = "reliability-cascade-completion/v1"
CHECKPOINT_SCHEMA = "reliability-cascade-checkpoint/v1"
ARTIFACT_FILES = (
    "predictions.csv",
    "metrics.csv",
    "reference_metrics.csv",
    "routing_metrics.csv",
    "selection_sweep.csv",
    "selected_policy.json",
    "feature_importance.csv",
    "config.resolved.yaml",
    "run_request.json",
    "checkpoint/cascade_models.joblib",
    "checkpoint/checkpoint_contract.json",
)
CHECKPOINT_BOUND_FILES = tuple(
    name for name in ARTIFACT_FILES if not name.startswith("checkpoint/")
)
REQUIRED_RUN_FILES = (
    *ARTIFACT_FILES,
    "artifact_integrity.json",
    "run_manifest.json",
    "completion_record.json",
)


_SAFE_VARIANT = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z"
)
_WINDOWS_RESERVED_COMPONENTS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_variant(value: object) -> str:
    """Return a variant that is safe to use as one portable path component."""
    if not isinstance(value, str):
        raise ValueError("Qwen variant must be a string")
    if value != value.strip():
        raise ValueError("Qwen variant must not contain leading or trailing whitespace")
    reserved_stem = value.split(".", 1)[0].upper()
    if (
        value in {"", ".", ".."}
        or _SAFE_VARIANT.fullmatch(value) is None
        or reserved_stem in _WINDOWS_RESERVED_COMPONENTS
    ):
        raise ValueError(
            "Qwen variant must be one non-empty path component containing only "
            "ASCII letters, digits, '.', '_' or '-'"
        )
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def implementation_contract() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((project_root / "common").glob("*.py")),
        *sorted((project_root / "reliability_cascade").glob("*.py")),
        project_root / "scripts" / "run_reliability_cascade_matrix.py",
    ]
    files = {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError("No reliability-cascade implementation files found")
    return {"sha256": canonical_sha256(files), "files": files}


def build_run_request(
    config: Dict[str, Any],
    qwen_predictions: Path,
    last1_predictions: Path,
    *,
    protocol: str,
) -> Dict[str, Any]:
    qwen_predictions = qwen_predictions.expanduser().resolve()
    last1_predictions = last1_predictions.expanduser().resolve()
    qwen_manifest = qwen_predictions.parent / "run_manifest.json"
    last1_manifest = last1_predictions.parent / "run_manifest.json"
    required = (qwen_predictions, last1_predictions, qwen_manifest, last1_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run-request inputs are missing: {missing}")
    payload = {
        "schema": RUN_REQUEST_SCHEMA,
        "protocol": protocol,
        "seed": int(get_value(config, "seed", 42)),
        "variant": str(get_value(config, "reliability_cascade.qwen_variant", "full")),
        "resolved_config": serializable_config(config),
        "implementation": implementation_contract(),
        "components": {
            "qwen_predictions": str(qwen_predictions),
            "qwen_predictions_sha256": sha256_file(qwen_predictions),
            "qwen_run_manifest": str(qwen_manifest.resolve()),
            "qwen_run_manifest_sha256": sha256_file(qwen_manifest),
            "last1_predictions": str(last1_predictions),
            "last1_predictions_sha256": sha256_file(last1_predictions),
            "last1_run_manifest": str(last1_manifest.resolve()),
            "last1_run_manifest_sha256": sha256_file(last1_manifest),
        },
    }
    return {"request": payload, "run_request_sha256": canonical_sha256(payload)}


def write_artifact_integrity(
    run_dir: Path,
    *,
    run_request_sha256: str,
) -> Dict[str, Any]:
    missing = [name for name in ARTIFACT_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot seal an incomplete cascade run: {missing}")
    payload = {
        "schema": INTEGRITY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_request_sha256": str(run_request_sha256),
        "files": {
            name: sha256_file(run_dir / name)
            for name in ARTIFACT_FILES
        },
    }
    write_json(run_dir / "artifact_integrity.json", payload)
    return payload


def write_completion_record(
    run_dir: Path,
    *,
    run_request_sha256: str,
) -> Dict[str, Any]:
    manifest = run_dir / "run_manifest.json"
    integrity = run_dir / "artifact_integrity.json"
    contract = run_dir / "checkpoint" / "checkpoint_contract.json"
    required = (manifest, integrity, contract)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot complete an unsealed cascade run: {missing}")
    payload = {
        "schema": COMPLETION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_request_sha256": str(run_request_sha256),
        "run_manifest_sha256": sha256_file(manifest),
        "artifact_integrity_sha256": sha256_file(integrity),
        "checkpoint_contract_sha256": sha256_file(contract),
    }
    write_json(run_dir / "completion_record.json", payload)
    return payload


def _route_mask_sha256(predictions_path: Path) -> str:
    import pandas as pd

    frame = pd.read_csv(predictions_path, dtype={"sample_id": str, "resource": str})
    missing = [name for name in ROUTE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Cascade predictions lack route columns: {missing}")
    payload = frame[list(ROUTE_COLUMNS)].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_run_directory(
    run_dir: Path,
    *,
    expected_run_request_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_model: str | None = None,
    expected_variant: str | None = None,
    expected_max_tokens: int | None = None,
) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cascade run is incomplete: {missing}")

    request_document = read_json(run_dir / "run_request.json")
    request = request_document.get("request")
    if not isinstance(request, dict):
        raise ValueError("run_request.json has no request object")
    request_sha256 = canonical_sha256(request)
    if request_document.get("run_request_sha256") != request_sha256:
        raise ValueError("run_request.json fingerprint does not match its payload")
    if expected_run_request_sha256 and request_sha256 != expected_run_request_sha256:
        raise ValueError("Cascade run request differs from the requested experiment")
    if request.get("implementation") != implementation_contract():
        raise ValueError("Cascade run was produced by a different implementation")

    integrity_path = run_dir / "artifact_integrity.json"
    integrity = read_json(integrity_path)
    if integrity.get("schema") != INTEGRITY_SCHEMA:
        raise ValueError("Unknown cascade artifact-integrity schema")
    if integrity.get("run_request_sha256") != request_sha256:
        raise ValueError("Artifact integrity belongs to a different run request")
    hashes = integrity.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_FILES):
        raise ValueError("Artifact integrity does not bind the complete artifact set")
    for name in ARTIFACT_FILES:
        if hashes[name] != sha256_file(run_dir / name):
            raise ValueError(f"Cascade artifact hash mismatch: {name}")

    manifest = read_json(run_dir / "run_manifest.json")
    details = manifest.get("details") or {}
    if not isinstance(details, dict):
        raise ValueError("Cascade run manifest details must be an object")
    if details.get("run_request_sha256") != request_sha256:
        raise ValueError("Run manifest belongs to a different run request")
    if details.get("artifact_integrity_sha256") != sha256_file(integrity_path):
        raise ValueError("Run manifest does not bind artifact_integrity.json")
    contract_path = run_dir / "checkpoint" / "checkpoint_contract.json"
    if details.get("checkpoint_contract_sha256") != sha256_file(contract_path):
        raise ValueError("Run manifest does not bind the checkpoint contract")
    request_seed = int(request.get("seed", -1))
    request_variant = str(request.get("variant", ""))
    if int(manifest.get("seed", -1)) != request_seed:
        raise ValueError("Run manifest seed differs from the run request")
    if str(manifest.get("model", "")) != f"reliability_cascade_{request_variant}":
        raise ValueError("Run manifest model differs from the run-request variant")
    if str(details.get("variant", "")) != request_variant:
        raise ValueError("Run manifest variant differs from the run request")
    resolved = request.get("resolved_config") or {}
    configured_qwen = resolved.get("qwen") or {}
    configured_max_tokens = int(configured_qwen.get("max_tokens", -1))
    effective = details.get("qwen_effective_parameters") or {}
    artifact_max_tokens = int(effective.get("max_tokens", -1))
    if configured_max_tokens <= 0 or artifact_max_tokens != configured_max_tokens:
        raise ValueError("Run manifest Qwen context differs from the run request")
    contract = read_json(contract_path)
    if contract.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unknown cascade checkpoint-contract schema")
    if contract.get("run_request_sha256") != request_sha256:
        raise ValueError("Checkpoint contract belongs to a different run request")
    if contract.get("protocol") != request.get("protocol"):
        raise ValueError("Checkpoint protocol differs from the run request")
    for name in (
        "component_contract",
        "calibration_protocol",
        "history_contract",
        "selection_contract",
        "cascade_model_config",
    ):
        if not isinstance(contract.get(name), dict):
            raise ValueError(f"Checkpoint contract lacks {name}")
    request_components = request.get("components") or {}
    component_contract = contract["component_contract"]
    for name in (
        "qwen_predictions_sha256",
        "qwen_run_manifest_sha256",
        "last1_predictions_sha256",
        "last1_run_manifest_sha256",
    ):
        if component_contract.get(name) != request_components.get(name):
            raise ValueError(f"Checkpoint component contract mismatch: {name}")
    if contract.get("route_decision_uses_qwen_output") is not False:
        raise ValueError("Checkpoint route contract is not pre-Qwen")
    if contract.get("route_mask_sha256") != _route_mask_sha256(
        run_dir / "predictions.csv"
    ):
        raise ValueError("Checkpoint route-mask hash differs from predictions.csv")
    model_path = run_dir / "checkpoint" / "cascade_models.joblib"
    if contract.get("models_sha256") != sha256_file(model_path):
        raise ValueError("Checkpoint contract does not match cascade_models.joblib")
    bound = contract.get("artifact_sha256")
    if not isinstance(bound, dict) or set(bound) != set(CHECKPOINT_BOUND_FILES):
        raise ValueError("Checkpoint contract does not bind the expected artifacts")
    for name in CHECKPOINT_BOUND_FILES:
        if sha256_file(run_dir / name) != bound[name]:
            raise ValueError(f"Checkpoint contract artifact mismatch: {name}")

    completion = read_json(run_dir / "completion_record.json")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValueError("Unknown cascade completion-record schema")
    expected_completion = {
        "run_request_sha256": request_sha256,
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "artifact_integrity_sha256": sha256_file(integrity_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
    }
    for key, expected in expected_completion.items():
        if completion.get(key) != expected:
            raise ValueError(f"Completion record mismatch: {key}")
    if expected_seed is not None and int(manifest.get("seed", -1)) != expected_seed:
        raise ValueError("Cascade run seed differs from the requested seed")
    if expected_model is not None and str(manifest.get("model", "")) != expected_model:
        raise ValueError("Cascade run model differs from the requested model")
    if expected_variant is not None and request_variant != expected_variant:
        raise ValueError("Cascade run variant differs from the requested variant")
    if (
        expected_max_tokens is not None
        and configured_max_tokens != expected_max_tokens
    ):
        raise ValueError("Cascade run context differs from the requested max_tokens")
    return manifest


__all__ = [
    "ARTIFACT_FILES",
    "CHECKPOINT_BOUND_FILES",
    "CHECKPOINT_SCHEMA",
    "REQUIRED_RUN_FILES",
    "build_run_request",
    "canonical_sha256",
    "implementation_contract",
    "validate_variant",
    "read_json",
    "verify_run_directory",
    "write_artifact_integrity",
    "write_completion_record",
    "write_json",
]
