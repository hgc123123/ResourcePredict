"""Read-only primary-study inputs and isolated baseline-run provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("xgboost", "textcnn", "lstm")
SCHEMA = "matched-training-tail-baselines/v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def locked_json(path: Path, payload: Any) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"Existing experiment contract differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        # Link a fully written file into place without replacing another
        # process's contract. Concurrent model jobs may prepare the same root.
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != payload:
                raise FileExistsError(f"Concurrent experiment contract differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def source_inventory() -> dict[str, str]:
    paths = []
    for folder in ("baseline_study", "common", "textcnn", "lstm", "xgboost"):
        paths.extend((ROOT / folder).glob("*.py"))
    paths.extend([ROOT / "qwen/base_selection.py", ROOT / "requirements.txt",
                  ROOT / "scripts/run_matched_baselines.py"])
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def load_primary(reference: Path, data_csv: Path, config_path: Path | None = None):
    """Rebuild, then verify exactly the already reported primary sample roles."""
    from common.config import load_config, set_value
    from common.data import load_dataset, split_fingerprint
    from qwen.base_selection import (select_base_training_frames,
                                     verify_base_training_selection)

    protocol_dir = reference / "protocol"
    config_path = config_path or protocol_dir / "train_tail.yaml"
    for path in (config_path, data_csv, protocol_dir / "outer_split.csv",
                 protocol_dir / "data_identity.json",
                 protocol_dir / "train_tail_base_selection.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required primary-study input is missing: {path}")
    config = load_config(config_path)
    set_value(config, "data.path", str(data_csv))
    set_value(config, "split.manifest", str(protocol_dir / "outer_split.csv"))
    set_value(config, "split.create_if_missing", False)
    set_value(config, "qwen.base_selection.mode", "train_tail")
    archived = read_json(protocol_dir / "train_tail_base_selection.json")
    if archived.get("mode") != "train_tail":
        raise ValueError("Reference must use independent train_tail checkpoint selection")
    set_value(config, "qwen.base_selection.fit_fraction", archived["fit_fraction"])
    bundle = load_dataset(config)
    identity = read_json(protocol_dir / "data_identity.json")
    if bundle.data_sha256 != identity["data_sha256"]:
        raise ValueError("Raw CSV differs from the primary study")
    if split_fingerprint(bundle.frame) != identity["split_fingerprint"]:
        raise ValueError("Outer split differs from the primary study")
    if digest(protocol_dir / "outer_split.csv") != identity["split_manifest_sha256"]:
        raise ValueError("Outer split manifest bytes differ from the primary study")
    counts = bundle.frame.groupby("split").size().to_dict()
    if counts != identity["effective_counts"]:
        raise ValueError("Outer split counts differ from the primary study")
    verify_base_training_selection(bundle, config, archived)
    selection = select_base_training_frames(bundle, config)
    return config, bundle, selection, {
        "schema": SCHEMA,
        "reference_study": str(reference),
        "reference_config_sha256": digest(config_path),
        "data_sha256": bundle.data_sha256,
        "outer_split_fingerprint": identity["split_fingerprint"],
        "outer_split_manifest_sha256": digest(protocol_dir / "outer_split.csv"),
        "base_selection_sha256": archived["sha256"],
        "optimization_count": archived["optimization_count"],
        "early_stopping_count": archived["early_stopping_count"],
        "purged_optimization_count": archived["purged_optimization_count"],
        "preprocessing_fit_source": "optimization_only",
        "outer_validation_used_for_selection": False,
        "test_used_for_fit_or_selection": False,
        "source_sha256": source_inventory(),
    }


def fitting_bundle(bundle, selection):
    """Expose only optimization and training-tail labels to a legacy fitter.

    The separate original bundle remains authoritative for outer evaluation.
    """
    from qwen.base_selection import base_preprocessing_bundle
    return base_preprocessing_bundle(bundle, selection)


def seal_run(output: Path, contract: dict) -> None:
    files = sorted(p for p in output.iterdir() if p.is_file()
                   and p.name not in {"completion.json", "request.json"})
    required = {"metrics.csv", "predictions.csv", "run_manifest.json",
                "config.resolved.yaml", "base_training_selection.json"}
    if not required <= {p.name for p in files}:
        raise ValueError(f"Incomplete baseline run: {output}")
    write_json(output / "completion.json", {
        "schema": SCHEMA, "contract": contract,
        "artifacts_sha256": {p.name: digest(p) for p in files},
    })


def verify_seal(output: Path, contract: dict | None = None) -> dict:
    record = read_json(output / "completion.json")
    if record.get("schema") != SCHEMA or not record.get("artifacts_sha256"):
        raise ValueError(f"Invalid completion record: {output}")
    if contract is not None and record["contract"] != contract:
        raise ValueError(f"Completed run has a different contract: {output}")
    if read_json(output / "request.json") != record["contract"]:
        raise ValueError(f"Run request differs from its completion record: {output}")
    required = {"metrics.csv", "predictions.csv", "run_manifest.json",
                "config.resolved.yaml", "base_training_selection.json"}
    if not required <= record["artifacts_sha256"].keys():
        raise ValueError(f"Completion record lacks required artifacts: {output}")
    for name, expected in record["artifacts_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Completion record contains an invalid artifact path")
        path = output / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Baseline artifact has changed or is missing: {path}")
    return record
