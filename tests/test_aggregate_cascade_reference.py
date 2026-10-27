from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.data import sha256_file
from common.metrics import compute_metrics
import scripts.aggregate_cascade_reference as aggregate
from scripts.aggregate_cascade_reference import (
    EVALUATION_SPLITS,
    MANIFEST_NAME,
    METHOD,
    REFERENCE_COLUMN,
    RUNS_NAME,
    SUMMARY_NAME,
    _assert_saved_matches_recomputed,
    _plan_request_hashes,
    _reference_contract,
    _summarize,
    _validate_formal_seeds,
    _validate_metric_frame,
)


DATA_HASH = "a" * 64
SPLIT_HASH = "b" * 64
ORDER_HASH = "c" * 64
SPLIT_MANIFEST_HASH = "d" * 64


def _predictions(seed: int, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    index = 0
    for split in EVALUATION_SPLITS:
        for resource in ("memory", "vram"):
            for replicate in range(2):
                index += 1
                target = 0.2 + 0.1 * replicate
                rows.append(
                    {
                        "sample_id": f"sample-{index}",
                        "job_id": f"job-{index}",
                        "split": split,
                        "resource": resource,
                        "target": target,
                        "prediction": target + 0.02,
                        "valid": True,
                        "model": "reliability_cascade_full",
                        "seed": seed,
                        REFERENCE_COLUMN: target + 0.01 + offset,
                    }
                )
    return pd.DataFrame(rows)


def _reference_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = predictions.copy()
    reference["prediction"] = reference[REFERENCE_COLUMN]
    return compute_metrics(reference, splits=EVALUATION_SPLITS)


def _manifest(seed: int, *, split_hash: str = SPLIT_HASH) -> dict[str, object]:
    return {
        "model": "reliability_cascade_full",
        "seed": seed,
        "data_sha256": DATA_HASH,
        "split_fingerprint": split_hash,
        "order_fingerprint": ORDER_HASH,
        "split_manifest_sha256": SPLIT_MANIFEST_HASH,
        "details": {
            "variant": "full",
            "qwen_backbone": "Qwen2.5-Coder-1.5B-Instruct",
            "qwen_effective_parameters": {"max_tokens": 4096},
            "history_contract": {
                "version": "reliability-cascade-history/v1",
                "strict_completion_relation": "completion_time < submit_time",
            },
        },
    }


def _request(seed: int, *, learning_rate: float = 1e-4) -> dict[str, object]:
    return {
        "schema": "reliability-cascade-run-request/v1",
        "protocol": "history-first-reliability-cascade/v1",
        "seed": seed,
        "variant": "full",
        "resolved_config": {
            "seed": seed,
            "qwen": {"max_tokens": 4096, "learning_rate": learning_rate},
            "evaluation": {"acc_tolerance": 0.10},
            "reliability_cascade": {"history_k": 5},
            "outputs": {"root": f"outputs/seed-{seed}"},
        },
    }


def _write_fake_matrix(
    root: Path,
    *,
    changed_config_seed: int | None = None,
    changed_split_seed: int | None = None,
) -> dict[str, object]:
    plan_runs = []
    for seed in aggregate.FORMAL_SEEDS:
        run_dir = aggregate._run_dir(
            root, variant="full", max_tokens=4096, seed=seed
        )
        run_dir.mkdir(parents=True)
        predictions = _predictions(seed, offset=(seed - 40) * 0.001)
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        _reference_metrics(predictions).to_csv(
            run_dir / "reference_metrics.csv", index=False
        )
        manifest = _manifest(
            seed,
            split_hash="e" * 64 if seed == changed_split_seed else SPLIT_HASH,
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        request = _request(
            seed,
            learning_rate=2e-4 if seed == changed_config_seed else 1e-4,
        )
        request_sha256 = aggregate.canonical_sha256(request)
        (run_dir / "run_request.json").write_text(
            json.dumps(
                {"request": request, "run_request_sha256": request_sha256}
            ),
            encoding="utf-8",
        )
        (run_dir / "artifact_integrity.json").write_text("{}", encoding="utf-8")
        (run_dir / "completion_record.json").write_text("{}", encoding="utf-8")
        plan_runs.append({"seed": seed, "run_request_sha256": request_sha256})
    plan = {
        "schema": "reliability-cascade-matrix-plan/v1",
        "protocol": "history-first-reliability-cascade/v1",
        "variant": "full",
        "max_tokens": 4096,
        "seeds": list(aggregate.FORMAL_SEEDS),
        "runs": plan_runs,
    }
    (root / "cascade_experiment_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    return plan


def _install_fake_integrity_verifiers(
    monkeypatch: pytest.MonkeyPatch, plan: dict[str, object]
) -> list[int]:
    verified: list[int] = []
    monkeypatch.setattr(aggregate, "_validate_matrix_plan", lambda *args, **kwargs: plan)

    def verify(run_dir, *, seed, **kwargs):
        verified.append(seed)
        manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
        return manifest, "test_sealed_verifier"

    monkeypatch.setattr(aggregate, "_verify_sealed_run", verify)
    return verified


def test_duplicate_seed_arguments_and_plan_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate seeds"):
        _validate_formal_seeds([40, 41, 42, 43, 43])
    plan = {
        "runs": [
            {"seed": 40, "run_request_sha256": "1" * 64},
            {"seed": 40, "run_request_sha256": "2" * 64},
        ]
    }
    with pytest.raises(ValueError, match="duplicate seed 40"):
        _plan_request_hashes(plan, [40])


def test_saved_reference_metrics_are_recomputed_from_reference_column() -> None:
    predictions = _predictions(40)
    saved = _validate_metric_frame(
        _reference_metrics(predictions), Path("reference_metrics.csv")
    )
    recomputed = aggregate._recompute_reference_metrics(predictions, tolerance=0.10)
    _assert_saved_matches_recomputed(saved, recomputed, Path("reference_metrics.csv"))

    tampered = saved.copy()
    tampered.loc[0, "mae"] += 0.01
    with pytest.raises(ValueError, match="mae.*recomputation"):
        _assert_saved_matches_recomputed(
            tampered, recomputed, Path("reference_metrics.csv")
        )


def test_summary_uses_sample_standard_deviation() -> None:
    frames = []
    for seed in aggregate.FORMAL_SEEDS:
        metrics = _reference_metrics(_predictions(seed, offset=(seed - 40) * 0.001))
        metrics.insert(0, "seed", seed)
        metrics.insert(0, "method", METHOD)
        frames.append(metrics)
    runs = pd.concat(frames, ignore_index=True)
    summary = _summarize(runs, expected_seed_count=5)
    memory_test = summary.loc[
        summary["split"].eq("test") & summary["resource"].eq("memory")
    ].iloc[0]
    expected = np.std([0.010, 0.011, 0.012, 0.013, 0.014], ddof=1)
    assert memory_test["seeds"] == 5
    assert memory_test["mae_sample_sd"] == pytest.approx(expected)


def test_reference_contract_ignores_seed_and_output_but_rejects_config_change() -> None:
    first = _reference_contract(_request(40), _manifest(40))
    second = _reference_contract(_request(41), _manifest(41))
    changed = _reference_contract(_request(41, learning_rate=2e-4), _manifest(41))
    assert aggregate.canonical_sha256(first) == aggregate.canonical_sha256(second)
    assert aggregate.canonical_sha256(first) != aggregate.canonical_sha256(changed)


def test_main_verifies_all_seeds_and_writes_hash_bound_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "matrix"
    plan = _write_fake_matrix(root)
    verified = _install_fake_integrity_verifiers(monkeypatch, plan)
    output = tmp_path / "summary"

    assert aggregate.main(["--root", str(root), "--output-dir", str(output)]) == 0
    assert verified == list(aggregate.FORMAL_SEEDS)
    assert (output / RUNS_NAME).is_file()
    assert (output / SUMMARY_NAME).is_file()
    manifest = json.loads((output / MANIFEST_NAME).read_text("utf-8"))
    assert manifest["seeds"] == list(aggregate.FORMAL_SEEDS)
    assert manifest["data_sha256"] == DATA_HASH
    assert manifest["outputs"][RUNS_NAME] == sha256_file(output / RUNS_NAME)
    assert manifest["outputs"][SUMMARY_NAME] == sha256_file(output / SUMMARY_NAME)


@pytest.mark.parametrize(
    ("changed_config_seed", "changed_split_seed", "message"),
    [
        (44, None, "different resolved configuration"),
        (None, 44, "different dataset or frozen split"),
    ],
)
def test_main_rejects_mixed_config_or_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_config_seed: int | None,
    changed_split_seed: int | None,
    message: str,
) -> None:
    root = tmp_path / "matrix"
    plan = _write_fake_matrix(
        root,
        changed_config_seed=changed_config_seed,
        changed_split_seed=changed_split_seed,
    )
    _install_fake_integrity_verifiers(monkeypatch, plan)
    with pytest.raises(ValueError, match=message):
        aggregate.main(["--root", str(root), "--output-dir", str(tmp_path / "out")])
