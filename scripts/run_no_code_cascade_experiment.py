#!/usr/bin/env python3
"""Preflight and run the formal five-seed No-Code reliability cascade."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_significance import (  # noqa: E402
    pair_prediction_frames,
    validate_artifact_identity,
)
from common.data import sha256_file  # noqa: E402


FORMAL_SEEDS = (40, 41, 42, 43, 44)
VARIANT = "no_code"
MODEL = "qwen_no_code"
LAST1_MODEL = "last-1"
PAIR_SPLITS = ("validation", "test")
REQUIRED_MODALITIES = {"user", "metadata"}
SCRIPT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the saved No-Code components, then fit and evaluate the "
            "five-seed reliability cascade"
        )
    )
    parser.add_argument("--config", default="configs/paper.local.yaml")
    parser.add_argument("--component-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--allow-seed-subset",
        action="store_true",
        help="Permit a nonformal subset for debugging; omit for the paper run",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _read_manifest(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role} manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{role} manifest must contain a JSON object: {path}")
    return manifest


def _split_fingerprint(manifest: dict[str, Any], role: str) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"{role} manifest has no split_fingerprint")
    return str(value)


def _manifest_identity(manifest: dict[str, Any], role: str) -> tuple[str, str, str, str]:
    required = ("data_sha256", "order_fingerprint", "split_manifest_sha256")
    missing = [name for name in required if not manifest.get(name)]
    if missing:
        raise ValueError(f"{role} manifest lacks provenance fields: {missing}")
    return (
        str(manifest["data_sha256"]),
        _split_fingerprint(manifest, role),
        str(manifest["order_fingerprint"]),
        str(manifest["split_manifest_sha256"]),
    )


def _read_predictions(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {role} predictions: {path}")
    frame = pd.read_csv(path, dtype={"sample_id": str, "job_id": str})
    if frame.empty:
        raise ValueError(f"{role} predictions contain no rows: {path}")
    return frame


def _validate_qwen_component(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    seed: int,
    max_tokens: int,
    expected_backbone: str,
) -> None:
    role = f"No-Code Qwen seed {seed}"
    if manifest.get("model") != MODEL:
        raise ValueError(f"{role} model must be {MODEL!r}")
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"{role} manifest seed does not match its directory")
    details = manifest.get("details") or {}
    if details.get("variant") != VARIANT:
        raise ValueError(f"{role} is not variant={VARIANT}")
    modalities = details.get("enabled_modalities")
    if not isinstance(modalities, list) or set(map(str, modalities)) != REQUIRED_MODALITIES:
        raise ValueError(
            f"{role} must enable exactly user and metadata, observed={modalities!r}"
        )
    effective = details.get("effective_parameters") or {}
    if int(effective.get("max_tokens", -1)) != max_tokens:
        raise ValueError(f"{role} does not use max_tokens={max_tokens}")
    backbone = str(details.get("base_model_override") or "")
    if expected_backbone not in backbone:
        raise ValueError(
            f"{role} backbone does not contain {expected_backbone!r}: {backbone!r}"
        )

    validate_artifact_identity(frame, manifest, role)
    required = {"gate_code", "gate_user", "gate_metadata", "script_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{role} predictions lack No-Code contract columns: {missing}")
    gates = frame[["gate_code", "gate_user", "gate_metadata"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(gates.to_numpy(float)).all():
        raise ValueError(f"{role} contains non-finite gate values")
    if not np.allclose(gates["gate_code"], 0.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{role} must have gate_code=0 on every row")
    user_metadata = gates[["gate_user", "gate_metadata"]].to_numpy(float)
    if (user_metadata < -1e-8).any() or (user_metadata > 1.0 + 1e-8).any():
        raise ValueError(f"{role} user/metadata gates must lie in [0,1]")
    if not np.allclose(user_metadata.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{role} gate_user + gate_metadata must equal one")
    hashes = frame["script_sha256"].astype(str).str.lower()
    if not hashes.map(lambda value: SCRIPT_HASH_PATTERN.fullmatch(value) is not None).all():
        raise ValueError(f"{role} contains an invalid script_sha256")


def _validate_last1_component(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    seed: int,
) -> None:
    role = f"Last-1 seed {seed}"
    if manifest.get("model") != LAST1_MODEL:
        raise ValueError(f"{role} model must be {LAST1_MODEL!r}")
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"{role} manifest seed does not match its directory")
    details = manifest.get("details") or {}
    if details.get("protocol") != "completed_before_submission":
        raise ValueError(f"{role} is not completed-before-submission Last-1")
    if details.get("causal_online_protocol") is not True:
        raise ValueError(f"{role} is not marked as a causal online protocol")
    relation = details.get("strict_completion_relation")
    if relation != "completion_time < submit_time":
        raise ValueError(f"{role} has an unexpected completion relation: {relation!r}")
    validate_artifact_identity(frame, manifest, role)


def preflight_components(
    component_root: Path,
    *,
    seeds: Sequence[int],
    max_tokens: int,
    expected_backbone: str = "Qwen2.5-Coder-1.5B-Instruct",
    expected_qwen_rows: int | None = 7726,
    expected_last1_rows: int | None = 38642,
) -> dict[str, Any]:
    """Validate every component before allowing the first cascade seed to start."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    component_root = component_root.expanduser().resolve()
    records: list[dict[str, Any]] = []
    shared_identity: tuple[str, str, str, str] | None = None
    baseline: pd.DataFrame | None = None

    for seed in seeds:
        qwen_dir = component_root / "qwen" / f"tokens_{max_tokens}" / VARIANT / f"seed_{seed}"
        last1_dir = component_root / "last-1" / "completed_before_submission" / f"seed_{seed}"
        qwen_path = qwen_dir / "predictions.csv"
        last1_path = last1_dir / "predictions.csv"
        qwen_manifest_path = qwen_dir / "run_manifest.json"
        last1_manifest_path = last1_dir / "run_manifest.json"
        qwen_manifest = _read_manifest(qwen_manifest_path, f"No-Code Qwen seed {seed}")
        last1_manifest = _read_manifest(last1_manifest_path, f"Last-1 seed {seed}")
        qwen = _read_predictions(qwen_path, f"No-Code Qwen seed {seed}")
        last1 = _read_predictions(last1_path, f"Last-1 seed {seed}")

        _validate_qwen_component(
            qwen,
            qwen_manifest,
            seed=seed,
            max_tokens=max_tokens,
            expected_backbone=expected_backbone,
        )
        _validate_last1_component(last1, last1_manifest, seed=seed)
        if expected_qwen_rows is not None and len(qwen) != expected_qwen_rows:
            raise ValueError(
                f"No-Code Qwen seed {seed} has {len(qwen)} rows; "
                f"expected {expected_qwen_rows}"
            )
        if expected_last1_rows is not None and len(last1) != expected_last1_rows:
            raise ValueError(
                f"Last-1 seed {seed} has {len(last1)} rows; "
                f"expected {expected_last1_rows}"
            )

        qwen_identity = _manifest_identity(qwen_manifest, f"No-Code Qwen seed {seed}")
        last1_identity = _manifest_identity(last1_manifest, f"Last-1 seed {seed}")
        if qwen_identity != last1_identity:
            raise ValueError(f"Seed {seed} Qwen and Last-1 provenance differs")
        if shared_identity is None:
            shared_identity = qwen_identity
        elif qwen_identity != shared_identity:
            raise ValueError(f"Seed {seed} uses a different dataset or frozen split")

        pair_prediction_frames(qwen, last1, splits=PAIR_SPLITS)
        if baseline is None:
            baseline = qwen
        else:
            pair_prediction_frames(baseline, qwen, splits=PAIR_SPLITS)

        records.append(
            {
                "seed": seed,
                "qwen_predictions": str(qwen_path),
                "qwen_predictions_sha256": sha256_file(qwen_path),
                "qwen_manifest": str(qwen_manifest_path),
                "qwen_manifest_sha256": sha256_file(qwen_manifest_path),
                "qwen_rows": int(len(qwen)),
                "last1_predictions": str(last1_path),
                "last1_predictions_sha256": sha256_file(last1_path),
                "last1_manifest": str(last1_manifest_path),
                "last1_manifest_sha256": sha256_file(last1_manifest_path),
                "last1_rows": int(len(last1)),
            }
        )

    if shared_identity is None:
        raise ValueError("At least one seed is required")
    return {
        "schema": "no-code-cascade-component-preflight/v1",
        "status": "ok",
        "variant": VARIANT,
        "seeds": list(seeds),
        "max_tokens": max_tokens,
        "expected_backbone": expected_backbone,
        "data_sha256": shared_identity[0],
        "split_fingerprint": shared_identity[1],
        "order_fingerprint": shared_identity[2],
        "split_manifest_sha256": shared_identity[3],
        "components": records,
    }


def build_matrix_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_reliability_cascade_matrix.py"),
        "--config",
        str(_resolve(args.config)),
        "--component-root",
        str(_resolve(args.component_root)),
        "--output-root",
        str(_resolve(args.output_root)),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--variant",
        VARIANT,
        "--experiment-label",
        "no_code_encoder",
        "--max-tokens",
        str(args.max_tokens),
    ]
    for override in args.overrides:
        command.extend(["--set", override])
    if args.dry_run:
        command.append("--dry-run")
    elif args.resume:
        command.append("--resume")
    elif args.overwrite:
        command.append("--overwrite")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    if not args.allow_seed_subset and tuple(args.seeds) != FORMAL_SEEDS:
        raise ValueError(
            "The formal experiment requires --seeds 40 41 42 43 44 in this order; "
            "use --allow-seed-subset only for debugging"
        )
    config_path = _resolve(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    report = preflight_components(
        _resolve(args.component_root),
        seeds=args.seeds,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if args.validate_only:
        return 0

    command = build_matrix_command(args)
    print("[DELEGATE] " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
