from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.compare_cascade_variants import (
    _delta_summary,
    _implementation_differences,
    _seed_deltas,
    _variant_summary,
)
from scripts.run_no_code_cascade_experiment import (
    build_matrix_command,
    preflight_components,
)


DATA_HASH = "a" * 64
SPLIT_HASH = "b" * 64
ORDER_HASH = "c" * 64
MANIFEST_HASH = "d" * 64
SCRIPT_HASH = "e" * 64


def _predictions(model: str, seed: int) -> pd.DataFrame:
    rows = []
    index = 0
    for split in ("validation", "test"):
        for resource in ("memory", "vram"):
            index += 1
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "job_id": f"job-{index}",
                    "split": split,
                    "resource": resource,
                    "target": 0.1 * index,
                    "prediction": 0.1 * index + 0.01,
                    "valid": True,
                    "model": model,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def _manifest(model: str, seed: int, *, qwen: bool) -> dict[str, object]:
    details: dict[str, object]
    if qwen:
        details = {
            "variant": "no_code",
            "enabled_modalities": ["user", "metadata"],
            "effective_parameters": {"max_tokens": 4096},
            "base_model_override": "/models/Qwen2.5-Coder-1.5B-Instruct",
        }
    else:
        details = {
            "protocol": "completed_before_submission",
            "causal_online_protocol": True,
            "strict_completion_relation": "completion_time < submit_time",
        }
    return {
        "model": model,
        "seed": seed,
        "data_sha256": DATA_HASH,
        "split_fingerprint": SPLIT_HASH,
        "order_fingerprint": ORDER_HASH,
        "split_manifest_sha256": MANIFEST_HASH,
        "details": details,
    }


def _write_components(root: Path, seed: int = 40) -> tuple[Path, Path]:
    qwen_dir = root / "qwen" / "tokens_4096" / "no_code" / f"seed_{seed}"
    last1_dir = root / "last-1" / "completed_before_submission" / f"seed_{seed}"
    qwen_dir.mkdir(parents=True)
    last1_dir.mkdir(parents=True)

    qwen = _predictions("qwen_no_code", seed)
    qwen["gate_code"] = 0.0
    qwen["gate_user"] = 0.25
    qwen["gate_metadata"] = 0.75
    qwen["script_sha256"] = SCRIPT_HASH
    qwen.to_csv(qwen_dir / "predictions.csv", index=False)
    _predictions("last-1", seed).to_csv(last1_dir / "predictions.csv", index=False)
    (qwen_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("qwen_no_code", seed, qwen=True)), encoding="utf-8"
    )
    (last1_dir / "run_manifest.json").write_text(
        json.dumps(_manifest("last-1", seed, qwen=False)), encoding="utf-8"
    )
    return qwen_dir, last1_dir


def test_preflight_accepts_aligned_no_code_components(tmp_path: Path) -> None:
    _write_components(tmp_path)
    report = preflight_components(
        tmp_path,
        seeds=[40],
        max_tokens=4096,
        expected_qwen_rows=None,
        expected_last1_rows=None,
    )
    assert report["status"] == "ok"
    assert report["seeds"] == [40]
    assert report["data_sha256"] == DATA_HASH
    assert report["components"][0]["qwen_rows"] == 4


def test_preflight_rejects_nonzero_code_gate(tmp_path: Path) -> None:
    qwen_dir, _ = _write_components(tmp_path)
    path = qwen_dir / "predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "gate_code"] = 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="gate_code=0"):
        preflight_components(
            tmp_path,
            seeds=[40],
            max_tokens=4096,
            expected_qwen_rows=None,
            expected_last1_rows=None,
        )


def test_preflight_rejects_target_mismatch(tmp_path: Path) -> None:
    _, last1_dir = _write_components(tmp_path)
    path = last1_dir / "predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "target"] = 0.9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="targets differ"):
        preflight_components(
            tmp_path,
            seeds=[40],
            max_tokens=4096,
            expected_qwen_rows=None,
            expected_last1_rows=None,
        )


def test_matrix_command_fixes_no_code_variant(tmp_path: Path) -> None:
    args = Namespace(
        config=str(tmp_path / "config.yaml"),
        component_root=str(tmp_path / "components"),
        output_root=str(tmp_path / "outputs"),
        seeds=[40, 41, 42, 43, 44],
        max_tokens=4096,
        overrides=["evaluation.clip_predictions=true"],
        dry_run=True,
        resume=False,
        overwrite=False,
    )
    command = build_matrix_command(args)
    assert command[command.index("--variant") + 1] == "no_code"
    assert command[command.index("--max-tokens") + 1] == "4096"
    assert "--dry-run" in command


def _metric_rows() -> pd.DataFrame:
    rows = []
    for variant in ("full", "no_code"):
        for seed in range(40, 45):
            offset = seed - 40
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "split": "test",
                    "resource": "memory",
                    "n": 100,
                    "mae": 0.08 + 0.001 * offset + (0.01 if variant == "no_code" else 0.0),
                    "mse": 0.03 + 0.001 * offset + (0.005 if variant == "no_code" else 0.0),
                    "rmse": 0.17 + 0.001 * offset + (0.01 if variant == "no_code" else 0.0),
                    "acc_at_10": 0.80 - 0.001 * offset - (0.02 if variant == "no_code" else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_comparison_deltas_have_documented_direction() -> None:
    metrics = _metric_rows()
    summary = _variant_summary(metrics)
    deltas = _seed_deltas(metrics)
    delta_summary = _delta_summary(deltas)
    assert len(summary) == 2
    assert (deltas["mae_delta_no_code_minus_full"] > 0.0).all()
    assert (deltas["acc_at_10_delta_no_code_minus_full"] < 0.0).all()
    assert delta_summary.loc[0, "mae_no_code_worse_seed_count"] == 5
    assert delta_summary.loc[0, "acc_at_10_no_code_worse_seed_count"] == 5


def test_implementation_difference_reports_exact_file() -> None:
    left = {"files": {"common/data.py": "same", "scripts/runner.py": "old"}}
    right = {"files": {"common/data.py": "same", "scripts/runner.py": "new"}}
    assert _implementation_differences(left, right) == ["scripts/runner.py"]
