from copy import deepcopy

import pytest

from qwen.contract import (
    CONTRACT_VERSION,
    experiment_contract,
    validate_base_model_override,
)


def _config():
    return {
        "seed": 42,
        "data": {
            "label_scale": "ratio",
            "label_out_of_range": "error",
            "columns": {
                "job_id": "jobid",
                "user": "user",
                "code": "script_content",
                "state": "state1",
                "exit_code": "exitcode",
            },
            "filters": {
                "states": ["COMPLETED"],
                "exit_codes": ["0:0"],
            },
            "code_redaction": {
                "enabled": True,
                "path_prefixes": ["/gpfsnyu/home", "/gpfsnyu/scratch"],
            },
            "metadata_numeric": ["cpu", "gpu", "mem"],
            "metadata_categorical": [],
        },
        "split": {
            "require_completed_labels_at_boundaries": True,
            "boundary_completion_policy": "error",
        },
    }


def _sha(config):
    return experiment_contract(config, effective_parameters={})["sha256"]


def test_qwen_contract_version_is_four():
    contract = experiment_contract(_config(), effective_parameters={})

    assert CONTRACT_VERSION == 4
    assert contract["version"] == 4
    assert contract["payload"]["contract_version"] == 4


def test_changing_data_filters_changes_contract_sha():
    original = _config()
    changed = deepcopy(original)
    changed["data"]["filters"]["states"] = ["COMPLETED", "TIMEOUT"]

    assert _sha(original) != _sha(changed)


def test_changing_code_redaction_changes_contract_sha():
    original = _config()
    changed = deepcopy(original)
    changed["data"]["code_redaction"]["path_prefixes"].append("/site/project")

    assert _sha(original) != _sha(changed)


def test_changing_boundary_completion_policy_changes_contract_sha():
    original = _config()
    changed = deepcopy(original)
    changed["split"]["boundary_completion_policy"] = "purge"

    assert _sha(original) != _sha(changed)


def test_boundary_completion_policy_is_canonicalized_in_contract():
    canonical = _config()
    canonical["split"]["boundary_completion_policy"] = "purge"
    mixed_case = deepcopy(canonical)
    mixed_case["split"]["boundary_completion_policy"] = "  PuRgE  "

    assert _sha(canonical) == _sha(mixed_case)


def test_base_model_override_must_match_checkpoint_contract():
    config = _config()
    config["qwen"] = {"model_name_or_path": "/models/qwen-trained"}
    contract = experiment_contract(config, effective_parameters={})

    assert (
        validate_base_model_override(contract, "/models/qwen-trained")
        == "/models/qwen-trained"
    )
    assert validate_base_model_override(contract, None) is None
    with pytest.raises(ValueError, match="differs"):
        validate_base_model_override(contract, "/models/qwen-different")
