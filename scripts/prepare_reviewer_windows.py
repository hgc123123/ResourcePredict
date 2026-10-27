#!/usr/bin/env python3
"""Prepare preregistered expanding-prefix temporal sensitivity windows.

The private prefix CSVs are internal artifacts and must not be publicly released.
No targets or model errors determine the prefix cutoffs. The 100% window reuses
the source bytes but still requires independently trained models in its own root.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prefix_size(submit_times, fraction: float) -> int:
    """Keep equal submission times together by moving an interior cut left."""
    import pandas as pd
    times = pd.to_datetime(submit_times, utc=True, errors="raise")
    if not 0 < fraction <= 1 or len(times) < 10 or times.isna().any():
        raise ValueError("Need at least ten valid chronological submissions and fraction in (0,1]")
    if not times.is_monotonic_increasing:
        raise ValueError("Submission stream is not chronological")
    if fraction == 1:
        return len(times)
    n = int(len(times) * fraction)
    if n < 1:
        raise ValueError("Empty prefix")
    while n > 0 and times.iloc[n - 1] == times.iloc[n]:
        n -= 1
    if not n:
        raise ValueError("All candidate prefix submissions have identical timestamps")
    return n


def window_labels(fractions):
    labels = [f"prefix_{round(fraction * 1000):04d}" for fraction in fractions]
    if len(set(labels)) != len(labels):
        raise ValueError("Fractions produce colliding window names; use more widely spaced fractions")
    return labels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Prepared strict arm config from the main study")
    p.add_argument("--output-root", default="outputs/reviewer_rolling")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.60, 0.80, 1.0])
    a = p.parse_args(argv)
    from common.config import load_config, set_value, resolved_serializable_config, get_value
    from common.data import load_dataset, _read_csv_strict, _computed_splits, sha256_file
    import yaml
    config = load_config(a.config)
    bundle = load_dataset(config)
    retained = bundle.retained_frame.sort_values(["submit_time", "source_row"], kind="mergesort").reset_index(drop=True)
    raw, _ = _read_csv_strict(bundle.data_path, config)
    out = Path(a.output_root)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Preserve existing rolling-window artifacts; output already exists: {out}")
    if sorted(set(a.fractions)) != a.fractions:
        raise ValueError("fractions must be strictly increasing and unique")
    # Compute/validate all sizes before making any files.
    sizes = [prefix_size(retained.submit_time, f) for f in a.fractions]
    labels = window_labels(a.fractions)
    if len(set(sizes)) != len(sizes):
        raise ValueError("Distinct fractions produced duplicate windows")
    bounds = []
    for n in sizes:
        assignment = _computed_splits(n, float(get_value(config, "split.train_ratio", .8)),
                                     float(get_value(config, "split.validation_ratio", .1)))
        test_times = retained.iloc[:n].loc[assignment == "test", "submit_time"]
        bounds.append((test_times.min(), test_times.max()))
    if any(left[1] >= right[0] for left, right in zip(bounds, bounds[1:])):
        raise ValueError("Chosen prefixes yield overlapping test intervals; no window files were created")
    records = []
    for fraction, n, label in zip(a.fractions, sizes, labels):
        folder = out / label
        folder.mkdir(parents=True)
        if fraction == 1:
            data_path = bundle.data_path
        else:
            data_path = folder / "PRIVATE_prefix.csv"
            raw.iloc[retained.iloc[:n].source_row.to_numpy(int)].to_csv(data_path, index=False, encoding="utf-8-sig")
        cfg = copy.deepcopy(config)
        # Original full-trace expected counts/hashes must not be asserted for
        # independently purged prefixes. New identities are computed and locked.
        for section in ("data", "split"):
            for key in list(cfg[section]):
                if key.startswith("expected_") and key != "expected_columns":
                    del cfg[section][key]
        set_value(cfg, "data.path", str(data_path.resolve()))
        set_value(cfg, "data.expected_sha256", sha256_file(data_path))
        set_value(cfg, "split.manifest", str(folder / "source_split.csv"))
        set_value(cfg, "split.create_if_missing", True)
        set_value(cfg, "qwen.base_selection.mode", "train_tail")
        set_value(cfg, "reliability_cascade.require_independent_base_selection", True)
        window = load_dataset(cfg)
        # The main runner will independently create an identical split file
        # inside this window's study directory; it never borrows predictions.
        cfg_path = folder / "window.yaml"
        cfg_path.write_text(yaml.safe_dump(resolved_serializable_config(cfg), sort_keys=False), encoding="utf-8")
        rec = {"window": label, "fraction": fraction, "retained_rows": n,
               "source_data_sha256": bundle.data_sha256, "window_data_sha256": window.data_sha256,
               "config": str(cfg_path), "data_csv": str(data_path.resolve()),
               "train_count": len(window.split("train")),
               "validation_count": len(window.split("validation")), "test_count": len(window.split("test")),
               "test_start": window.split("test").submit_time.min().isoformat(),
               "test_end": window.split("test").submit_time.max().isoformat(),
               "uses_error_or_target_for_window_selection": False}
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=True))
        print(f"Run scripts/run_reviewer_experiments.py --config {cfg_path} --data-csv {data_path} --output-root {folder / 'study'} --seeds 42 --stage all")
    for left, right in zip(records, records[1:]):
        if left["test_end"] >= right["test_start"]:
            raise ValueError("Chosen prefixes yield overlapping test intervals; revise fractions before training")
    (out / "windows.json").write_text(json.dumps({"schema": "reviewer-prefix-windows/v1", "windows": records,
        "interpretation": "Expanding-prefix rolling-origin sensitivity with disjoint test periods; not independent clusters."},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
