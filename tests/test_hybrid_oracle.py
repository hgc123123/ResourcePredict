import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.hybrid_oracle import (
    choice_oracle_predictions,
    hit_overlap_counts,
    line_segment_oracle_predictions,
    main as oracle_main,
)
from scripts.aggregate_results import _is_forbidden_diagnostic


def _paired_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["inside", "outside", "no-history", "tie"],
            "job_id_qwen": ["1", "2", "3", "4"],
            "user_id_qwen": ["u"] * 4,
            "split": ["validation"] * 4,
            "resource": ["memory"] * 4,
            "target": [0.50, 0.00, 1.00, 0.50],
            "valid": [True] * 4,
            "qwen_prediction": [0.25, 0.25, 0.25, 0.25],
            "last1_prediction": [0.75, 0.75, 1.00, 0.75],
            "history_available": [True, True, False, True],
        }
    )


def test_choice_and_line_segment_oracle_definitions():
    paired = _paired_rows()
    choice = choice_oracle_predictions(paired, seed=42, variant="full")
    segment = line_segment_oracle_predictions(paired, seed=42, variant="full")

    # Choice ties use Qwen, and unavailable history can never select Last-1.
    assert choice["prediction"].tolist() == [0.25, 0.25, 0.25, 0.25]
    assert choice["oracle_choice"].tolist() == ["qwen"] * 4

    # Projection is exact inside the interval, uses the nearest endpoint outside,
    # and is forced to Qwen when causal history is unavailable.
    assert np.allclose(segment["prediction"], [0.5, 0.25, 0.25, 0.5])
    assert np.allclose(segment["oracle_alpha"], [0.5, 0.0, 0.0, 0.5])
    assert segment["oracle_projection"].tolist() == [
        "line_segment_interior",
        "qwen_endpoint",
        "qwen_only_no_history",
        "line_segment_interior",
    ]
    for frame in (choice, segment):
        assert frame["uses_targets"].eq(True).all()
        assert frame["diagnostic_only"].eq(True).all()
        assert frame["not_a_deployable_model"].eq(True).all()


def test_hit_overlap_counts_all_four_categories():
    rows = []
    for split in ("validation", "test"):
        for resource in ("memory", "vram"):
            for index, (qwen, last1) in enumerate(
                [(0.50, 0.80), (0.80, 0.50), (0.55, 0.54), (0.80, 0.70)]
            ):
                rows.append(
                    {
                        "sample_id": f"{split}-{resource}-{index}",
                        "split": split,
                        "resource": resource,
                        "target": 0.50,
                        "valid": True,
                        "qwen_prediction": qwen,
                        "last1_prediction": last1,
                        "history_available": True,
                    }
                )
    counts = hit_overlap_counts(pd.DataFrame(rows), tolerance=0.10)
    assert len(counts) == 4
    assert counts["n"].eq(4).all()
    assert counts["qwen_only_hit"].eq(1).all()
    assert counts["last1_only_hit"].eq(1).all()
    assert counts["both_hit"].eq(1).all()
    assert counts["neither_hit"].eq(1).all()
    assert (
        counts[
            ["qwen_only_hit", "last1_only_hit", "both_hit", "neither_hit"]
        ].sum(axis=1)
        == counts["n"]
    ).all()


def _component_predictions(model: str, last1: bool = False) -> pd.DataFrame:
    rows = []
    sample_index = 0
    for split in ("validation", "test"):
        for resource in ("memory", "vram"):
            for local_index in range(2):
                target = 0.25 + 0.25 * local_index
                prediction = target + (0.08 if model == "qwen_full" else -0.06)
                row = {
                    "sample_id": f"sample-{sample_index}",
                    "job_id": str(sample_index),
                    "user_id": f"u{local_index}",
                    "split": split,
                    "resource": resource,
                    "target": target,
                    "prediction": prediction,
                    "valid": True,
                    "model": model,
                    "seed": 42,
                }
                if last1:
                    row.update(
                        {
                            "history_available": local_index == 0,
                            "history_source_sample_id": (
                                "prior" if local_index == 0 else ""
                            ),
                            "history_source_time": (
                                "2026-01-01T00:00:00+00:00" if local_index == 0 else ""
                            ),
                            "protocol": "completed_before_submission",
                            "fallback_used": local_index != 0,
                        }
                    )
                rows.append(row)
                sample_index += 1
    return pd.DataFrame(rows)


def test_oracle_main_marks_artifacts_and_avoids_formal_metrics_filename(
    tmp_path: Path, monkeypatch
):
    qwen_dir = tmp_path / "qwen"
    last1_dir = tmp_path / "last1"
    output_dir = tmp_path / "oracle"
    qwen_dir.mkdir()
    last1_dir.mkdir()

    qwen = _component_predictions("qwen_full")
    last1 = _component_predictions("last-1", last1=True)
    qwen.to_csv(qwen_dir / "predictions.csv", index=False)
    last1.to_csv(last1_dir / "predictions.csv", index=False)
    shared = {
        "seed": 42,
        "data_sha256": "a" * 64,
        "split_fingerprint": "b" * 64,
    }
    (qwen_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **shared,
                "model": "qwen_full",
                "details": {"variant": "full"},
            }
        ),
        encoding="utf-8",
    )
    (last1_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **shared,
                "model": "last-1",
                "details": {
                    "causal_online_protocol": True,
                    "protocol": "completed_before_submission",
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 42",
                "evaluation:",
                "  acc_tolerance: 0.10",
                "hybrid:",
                "  qwen_variant: full",
                "last1:",
                "  protocol: completed_before_submission",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_oracle.py",
            "--config",
            str(config),
            "--qwen-predictions",
            str(qwen_dir / "predictions.csv"),
            "--last1-predictions",
            str(last1_dir / "predictions.csv"),
            "--output-dir",
            str(output_dir),
        ],
    )
    oracle_main()

    manifest = json.loads((output_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["uses_targets"] is True
    assert manifest["diagnostic_only"] is True
    assert manifest["not_a_deployable_model"] is True
    assert _is_forbidden_diagnostic(manifest)
    assert not (output_dir / "metrics.csv").exists()
    for name in (
        "choice_oracle_predictions.csv",
        "line_segment_oracle_predictions.csv",
        "choice_oracle_metrics.csv",
        "line_segment_oracle_metrics.csv",
        "hit_overlap_counts.csv",
    ):
        assert (output_dir / name).is_file()
        table = pd.read_csv(output_dir / name)
        assert table["uses_targets"].eq(True).all()
        assert table["diagnostic_only"].eq(True).all()
        assert table["not_a_deployable_model"].eq(True).all()


def test_aggregate_guard_recognizes_top_level_and_nested_flags():
    for flag in ("uses_targets", "diagnostic_only", "not_a_deployable_model"):
        assert _is_forbidden_diagnostic({flag: True})
        assert _is_forbidden_diagnostic({"details": {flag: True}})
    assert not _is_forbidden_diagnostic({"diagnostic_only": False})
