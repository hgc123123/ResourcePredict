#!/usr/bin/env python3
"""Measure script-token coverage without exposing private script contents."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.data import load_dataset  # noqa: E402
from qwen.contract import validate_and_restore_contract  # noqa: E402
from qwen.dataset import script_encoder_input  # noqa: E402


DEFAULT_BUDGETS = (2048, 4096, 8192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report exact Qwen script-token coverage for context-length ablations"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Qwen run checkpoint; uses its saved tokenizer when supplied",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer directory/model identifier; valid only without --checkpoint",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(DEFAULT_BUDGETS),
        metavar="TOKENS",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", help="CSV path; a provenance JSON is written beside it")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def summarize_token_coverage(
    token_lengths: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    budgets: Iterable[int],
    *,
    prompt_tokens: int,
    special_tokens: int,
) -> pd.DataFrame:
    """Return aggregate truncation statistics for each split and token budget."""
    lengths = np.asarray(token_lengths, dtype=np.int64)
    split_values = np.asarray(splits, dtype=str)
    if lengths.ndim != 1 or split_values.ndim != 1 or len(lengths) != len(split_values):
        raise ValueError("token_lengths and splits must be aligned one-dimensional arrays")
    if len(lengths) == 0 or np.any(lengths < 0):
        raise ValueError("token_lengths must be non-empty and non-negative")
    if prompt_tokens < 0 or special_tokens < 0:
        raise ValueError("prompt_tokens and special_tokens must be non-negative")

    normalized_budgets = sorted({int(value) for value in budgets})
    if not normalized_budgets or normalized_budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    overhead = int(prompt_tokens + special_tokens)
    if any(value <= overhead for value in normalized_budgets):
        raise ValueError("Every max-token budget must exceed prompt and special-token overhead")

    ordered_scopes = ["overall"] + [
        name for name in ("train", "validation", "test") if name in set(split_values)
    ]
    extra_scopes = sorted(set(split_values) - set(ordered_scopes))
    ordered_scopes.extend(extra_scopes)

    rows: list[dict[str, Any]] = []
    for scope in ordered_scopes:
        scoped = lengths if scope == "overall" else lengths[split_values == scope]
        if len(scoped) == 0:
            continue
        quantiles = np.quantile(scoped, [0.50, 0.90, 0.95, 0.99])
        for max_tokens in normalized_budgets:
            script_budget = max_tokens - overhead
            truncated = scoped > script_budget
            retained_fraction = np.minimum(1.0, script_budget / np.maximum(scoped, 1))
            rows.append(
                {
                    "scope": scope,
                    "n": int(len(scoped)),
                    "max_tokens": int(max_tokens),
                    "prompt_tokens": int(prompt_tokens),
                    "special_tokens": int(special_tokens),
                    "script_token_budget": int(script_budget),
                    "script_tokens_mean": float(np.mean(scoped)),
                    "script_tokens_p50": float(quantiles[0]),
                    "script_tokens_p90": float(quantiles[1]),
                    "script_tokens_p95": float(quantiles[2]),
                    "script_tokens_p99": float(quantiles[3]),
                    "script_tokens_max": int(np.max(scoped)),
                    "truncated_n": int(np.sum(truncated)),
                    "truncated_rate": float(np.mean(truncated)),
                    "mean_retained_token_fraction": float(np.mean(retained_fraction)),
                }
            )
    return pd.DataFrame(rows)


def _token_lengths(tokenizer: Any, scripts: Sequence[str], batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    lengths: list[int] = []
    for start in range(0, len(scripts), batch_size):
        encoded = tokenizer(
            list(scripts[start : start + batch_size]),
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def _tokenizer_source(args: argparse.Namespace, config: dict[str, Any]) -> Path | str:
    if args.checkpoint and args.tokenizer:
        raise ValueError(
            "--tokenizer cannot override a checkpoint tokenizer; omit --checkpoint "
            "for an independent tokenizer-only coverage analysis"
        )
    if args.tokenizer:
        candidate = resolve_path(config, args.tokenizer)
        return candidate if candidate.exists() else args.tokenizer
    if args.checkpoint:
        checkpoint = resolve_path(config, args.checkpoint)
        tokenizer_dir = checkpoint / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint tokenizer not found: {tokenizer_dir}")
        return tokenizer_dir
    configured = str(get_value(config, "qwen.model_name_or_path"))
    candidate = resolve_path(config, configured)
    return candidate if candidate.exists() else configured


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)

    checkpoint_manifest: dict[str, Any] | None = None
    saved_contract: dict[str, Any] | None = None
    if args.checkpoint:
        checkpoint = resolve_path(config, args.checkpoint)
        manifest_path = checkpoint / "training_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Checkpoint training manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            checkpoint_manifest = json.load(handle)
        if not isinstance(checkpoint_manifest, dict):
            raise ValueError(f"Checkpoint training manifest must be an object: {manifest_path}")
        saved_contract = validate_and_restore_contract(config, checkpoint_manifest)

    # Contract restoration must precede data loading: redaction, filtering, split,
    # prompt, and tokenizer semantics all affect the sequence being measured.
    bundle = load_dataset(config)
    if checkpoint_manifest is not None:
        if str(checkpoint_manifest.get("data_sha256")) != str(bundle.data_sha256):
            raise ValueError("Token-coverage data differs from the checkpoint training data")
        if str(checkpoint_manifest.get("split_fingerprint")) != str(
            bundle.protocol.get("split_fingerprint")
        ):
            raise ValueError("Token-coverage split differs from the checkpoint")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in the Qwen environment
        raise RuntimeError("Install requirements-qwen.txt to measure token coverage") from exc

    source = _tokenizer_source(args, config)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)),
        local_files_only=Path(str(source)).exists(),
    )
    effective = dict(
        (checkpoint_manifest or {}).get("effective_parameters") or {}
    )
    prefix = str(
        effective.get("prompt_prefix", get_value(config, "qwen.prompt_prefix", ""))
    )
    suffix = str(
        effective.get("prompt_suffix", get_value(config, "qwen.prompt_suffix", ""))
    )
    prompt_tokens = len(tokenizer.encode(prefix, add_special_tokens=False)) + len(
        tokenizer.encode(suffix, add_special_tokens=False)
    )
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))

    scripts = [script_encoder_input(value) for value in bundle.frame["code"].tolist()]
    lengths = _token_lengths(tokenizer, scripts, args.batch_size)
    summary = summarize_token_coverage(
        lengths,
        bundle.frame["split"].astype(str).to_numpy(),
        args.budgets,
        prompt_tokens=prompt_tokens,
        special_tokens=special_tokens,
    )

    output = (
        resolve_path(config, args.output)
        if args.output
        else resolve_path(config, get_value(config, "outputs.root", "outputs"))
        / "analysis"
        / "token_coverage.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(bundle.data_path),
        "data_sha256": bundle.data_sha256,
        "split_fingerprint": bundle.protocol.get("split_fingerprint"),
        "tokenizer_source": str(source),
        "checkpoint": str(resolve_path(config, args.checkpoint)) if args.checkpoint else None,
        "checkpoint_experiment_contract_sha256": (
            saved_contract.get("sha256") if saved_contract else None
        ),
        "checkpoint_contract_validated": saved_contract is not None,
        "checkpoint_effective_parameters": (
            checkpoint_manifest.get("effective_parameters")
            if checkpoint_manifest is not None
            else None
        ),
        "prompt_tokens": prompt_tokens,
        "special_tokens": special_tokens,
        "budgets": sorted({int(value) for value in args.budgets}),
        "population": "effective train/validation/test rows after configured filtering and boundary purge",
        "script_input": "canonical redacted encoder input; blank scripts use <EMPTY_SCRIPT>",
        "private_script_contents_written": False,
        "summary_csv": str(output),
    }
    json_path = output.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=True)
    print(summary.to_string(index=False))
    print(f"Token-coverage table: {output}")
    print(f"Token-coverage provenance: {json_path}")


if __name__ == "__main__":
    main()
