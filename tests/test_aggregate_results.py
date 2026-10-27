import hashlib
import json

import pytest

from scripts.aggregate_results import (
    _context_max_tokens,
    _manifest_experiment_semantics,
    _normalized_experiment_config,
)


def test_context_tokens_are_read_from_qwen_and_hybrid_manifests():
    assert (
        _context_max_tokens(
            {"details": {"effective_parameters": {"max_tokens": 4096}}}
        )
        == 4096
    )
    assert (
        _context_max_tokens(
            {"qwen_effective_parameters": {"max_tokens": 2048}}
        )
        == 2048
    )
    assert _context_max_tokens({"details": {}}) is None


def test_context_tokens_must_be_positive():
    with pytest.raises(ValueError, match="non-positive"):
        _context_max_tokens(
            {"details": {"effective_parameters": {"max_tokens": 0}}}
        )


def _fingerprint(value):
    canonical = json.dumps(
        _normalized_experiment_config(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_replicate_config_fingerprint_ignores_seed_and_output_location():
    first = {
        "seed": 40,
        "outputs": {"root": "outputs/a"},
        "qwen": {"max_tokens": 4096, "learning_rate": 1e-4},
    }
    second = {
        "seed": 44,
        "outputs": {"root": "outputs/b"},
        "qwen": {"max_tokens": 4096, "learning_rate": 1e-4},
    }
    assert _fingerprint(first) == _fingerprint(second)


def test_replicate_config_fingerprint_detects_training_changes():
    first = {"seed": 40, "qwen": {"max_tokens": 4096, "learning_rate": 1e-4}}
    second = {"seed": 41, "qwen": {"max_tokens": 4096, "learning_rate": 2e-4}}
    assert _fingerprint(first) != _fingerprint(second)


def test_qwen_manifest_semantics_include_effective_global_batch():
    first = {
        "model": "qwen_full",
        "details": {
            "variant": "full",
            "enabled_modalities": ["code", "user", "metadata"],
            "fusion_mode": "gated",
            "effective_parameters": {
                "max_tokens": 4096,
                "world_size": 8,
                "effective_global_batch_size": 32,
            },
        },
    }
    second = json.loads(json.dumps(first))
    second["details"]["effective_parameters"]["world_size"] = 4
    second["details"]["effective_parameters"]["effective_global_batch_size"] = 16

    assert _manifest_experiment_semantics(first) != _manifest_experiment_semantics(
        second
    )


def test_cascade_replicates_ignore_seed_specific_components_and_selected_thresholds():
    first = {
        "model": "reliability_cascade_full",
        "details": {
            "variant": "full",
            "qwen_backbone": "Qwen2.5-Coder-1.5B-Instruct",
            "qwen_effective_parameters": {"max_tokens": 4096},
            "component_contract": {"qwen_predictions_sha256": "a" * 64},
            "history_contract": {"version": "v1"},
            "calibration_protocol": {"protocol": "chronological"},
            "selection_policy": {
                "objective": "constrained_efficiency",
                "constraints": {"max_mae": 0.001},
                "selected": {"memory": {"threshold": 0.5}},
            },
            "route_feature_whitelist": ["history_count"],
            "route_decision_uses_qwen_output": False,
        },
    }
    second = json.loads(json.dumps(first))
    second["details"]["component_contract"]["qwen_predictions_sha256"] = "b" * 64
    second["details"]["selection_policy"]["selected"]["memory"]["threshold"] = 0.9
    assert _manifest_experiment_semantics(first) == _manifest_experiment_semantics(
        second
    )

    second["details"]["qwen_effective_parameters"]["max_tokens"] = 2048
    assert _manifest_experiment_semantics(first) != _manifest_experiment_semantics(
        second
    )
