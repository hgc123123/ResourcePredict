#!/usr/bin/env python3
"""Replay an exact script-embedding cache in chronological submission order."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import output_directory  # noqa: E402
from common.config import load_config, resolve_path  # noqa: E402
from common.data import load_dataset  # noqa: E402
from qwen.dataset import EMPTY_SCRIPT, script_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronologically replay an unbounded exact script-embedding cache"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Defaults to outputs/qwen/cache_replay.json")
    parser.add_argument(
        "--cache-hit-path-ms",
        type=float,
        help="Measured total latency of a cache-hit path; requires --cache-miss-path-ms",
    )
    parser.add_argument(
        "--cache-miss-path-ms",
        type=float,
        help="Measured total latency of a cache-miss/encoder path",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _counter() -> Dict[str, int]:
    return {"lookups": 0, "hits": 0, "misses": 0}


def replay_exact_cache(bundle: Any) -> Tuple[Dict[str, Dict[str, int]], set[str], int]:
    """Replay every observed script arrival, including boundary-purged jobs."""
    retained_frame = getattr(bundle, "retained_frame", None)
    replay_frame = bundle.frame if retained_frame is None else retained_frame
    if replay_frame.empty:
        raise ValueError("The exact-cache replay population is empty")
    if (
        "submit_time" in replay_frame
        and not replay_frame["submit_time"].is_monotonic_increasing
    ):
        raise ValueError("Retained cache-replay rows are not ordered by submit_time")

    counts: Dict[str, Dict[str, int]] = {
        "train": _counter(),
        "validation": _counter(),
        "test": _counter(),
        "overall": _counter(),
    }
    seen: set[str] = set()
    for row in replay_frame.itertuples(index=False):
        split = str(row.split)
        if split not in counts:
            raise ValueError(f"Unexpected split during cache replay: {split}")
        key = script_sha256(row.code)
        hit = key in seen
        for name in (split, "overall"):
            counts[name]["lookups"] += 1
            counts[name]["hits"] += int(hit)
            counts[name]["misses"] += int(not hit)
        seen.add(key)
    return counts, seen, int(len(replay_frame))


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    if (args.cache_hit_path_ms is None) != (args.cache_miss_path_ms is None):
        raise ValueError("Provide both cache path costs or neither")
    if args.cache_hit_path_ms is not None and (
        args.cache_hit_path_ms < 0 or args.cache_miss_path_ms < 0
    ):
        raise ValueError("Cache path costs must be non-negative")
    bundle = load_dataset(config)
    if not bundle.protocol.get("verified_chronological", False):
        raise ValueError(
            "Exact-cache replay requires split.order_semantics=submission_time"
        )

    counts, seen, replay_population_rows = replay_exact_cache(bundle)
    boundary_protocol = bundle.protocol.get("completed_labels_at_boundaries") or {}

    split_results: Dict[str, Dict[str, Any]] = {}
    for split, values in counts.items():
        lookups = values["lookups"]
        split_results[split] = {
            **values,
            "hit_rate": float(values["hits"] / lookups) if lookups else None,
        }

    result: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(bundle.data_path),
        "data_sha256": bundle.data_sha256,
        "split_manifest": str(bundle.manifest_path) if bundle.manifest_path else None,
        "split_fingerprint": bundle.protocol.get("split_fingerprint"),
        "effective_order_fingerprint": bundle.protocol.get("order_fingerprint"),
        "base_split_fingerprint": boundary_protocol.get("base_split_fingerprint"),
        "base_order_fingerprint": boundary_protocol.get("base_order_fingerprint"),
        "order_semantics": bundle.protocol["order_semantics"],
        "population_selection": bundle.protocol.get("filters"),
        "cache_policy": "unbounded_exact_match",
        "initial_cache_state": "empty",
        "eviction_policy": "none",
        "cache_key": "sha256(utf8(canonical_script_encoder_input))",
        "replay_population": (
            "retained_filtered_trace"
            if getattr(bundle, "retained_frame", None) is not None
            else "effective_frame_legacy_fallback"
        ),
        "replay_population_rows": replay_population_rows,
        "effective_model_population_rows": int(len(bundle.frame)),
        "boundary_purged_rows_replayed": int(replay_population_rows - len(bundle.frame)),
        "script_normalization": {
            "blank_script_sentinel": EMPTY_SCRIPT,
            "loader_code_redaction": bundle.protocol.get("code_redaction"),
        },
        "unique_cache_keys_after_replay": len(seen),
        "results": split_results,
        "note": (
            "All values are computed from this trace; no paper hit rate or latency is preset."
        ),
    }
    if args.cache_hit_path_ms is not None:
        test_hit_rate = float(split_results["test"]["hit_rate"])
        mean_path_ms = (
            test_hit_rate * args.cache_hit_path_ms
            + (1.0 - test_hit_rate) * args.cache_miss_path_ms
        )
        result["test_latency_estimate"] = {
            "formula": "hit_rate * hit_path_ms + (1-hit_rate) * miss_path_ms",
            "hit_path_ms": float(args.cache_hit_path_ms),
            "miss_path_ms": float(args.cache_miss_path_ms),
            "estimated_mean_path_ms": float(mean_path_ms),
            "costs_are_user_supplied": True,
        }

    output = (
        resolve_path(config, args.output)
        if args.output
        else output_directory(config, "qwen") / "cache_replay.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"Cache replay artifact: {output}")


if __name__ == "__main__":
    main()
