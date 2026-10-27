import json
from pathlib import Path

import pytest

from common.data import sha256_file
from reliability_cascade.integrity import (
    ARTIFACT_FILES,
    CHECKPOINT_BOUND_FILES,
    CHECKPOINT_SCHEMA,
    _route_mask_sha256,
    canonical_sha256,
    implementation_contract,
    verify_run_directory,
    write_artifact_integrity,
    write_completion_record,
    write_json,
)
import scripts.run_reliability_cascade_matrix as matrix_runner
from scripts.run_reliability_cascade_matrix import (
    _acquire_matrix_root_lock,
    _complete,
    _ensure_plan,
    _preflight_output_directories,
    _release_matrix_root_lock,
    _run_output,
)
from reliability_cascade.run import _publish_directory, _recover_interrupted_publish


def _sealed_run(root: Path, *, seed: int = 42) -> str:
    component_hashes = {
        "qwen_predictions_sha256": "1" * 64,
        "qwen_run_manifest_sha256": "2" * 64,
        "last1_predictions_sha256": "3" * 64,
        "last1_run_manifest_sha256": "4" * 64,
    }
    request = {
        "schema": "test-request/v1",
        "protocol": "test",
        "seed": seed,
        "variant": "full",
        "resolved_config": {"qwen": {"max_tokens": 4096}},
        "implementation": implementation_contract(),
        "components": component_hashes,
    }
    request_sha256 = canonical_sha256(request)
    root.mkdir(parents=True)
    for name in ARTIFACT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{name}\n", encoding="utf-8")
    write_json(
        root / "run_request.json",
        {"request": request, "run_request_sha256": request_sha256},
    )
    (root / "predictions.csv").write_text(
        "sample_id,resource,history_bypass,qwen_required_resource,"
        "qwen_required_job,encoder_forward_required_job\n"
        "a,memory,False,True,True,True\n",
        encoding="utf-8",
    )
    model_path = root / "checkpoint" / "cascade_models.joblib"
    write_json(
        root / "checkpoint" / "checkpoint_contract.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "protocol": "test",
            "run_request_sha256": request_sha256,
            "component_contract": component_hashes,
            "calibration_protocol": {},
            "history_contract": {},
            "selection_contract": {},
            "cascade_model_config": {},
            "route_decision_uses_qwen_output": False,
            "route_mask_sha256": _route_mask_sha256(root / "predictions.csv"),
            "models_sha256": sha256_file(model_path),
            "artifact_sha256": {
                name: sha256_file(root / name) for name in CHECKPOINT_BOUND_FILES
            },
        },
    )
    write_artifact_integrity(root, run_request_sha256=request_sha256)
    contract = root / "checkpoint" / "checkpoint_contract.json"
    integrity = root / "artifact_integrity.json"
    write_json(
        root / "run_manifest.json",
        {
            "model": "reliability_cascade_full",
            "seed": seed,
            "details": {
                "variant": "full",
                "qwen_effective_parameters": {"max_tokens": 4096},
                "run_request_sha256": request_sha256,
                "checkpoint_contract_sha256": sha256_file(contract),
                "artifact_integrity_sha256": sha256_file(integrity),
            },
        },
    )
    write_completion_record(root, run_request_sha256=request_sha256)
    return request_sha256


def test_integrity_verifier_detects_artifact_tampering(tmp_path):
    run_dir = tmp_path / "run"
    request_sha256 = _sealed_run(run_dir)
    verify_run_directory(
        run_dir,
        expected_run_request_sha256=request_sha256,
        expected_seed=42,
        expected_model="reliability_cascade_full",
    )
    (run_dir / "metrics.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.csv"):
        verify_run_directory(run_dir)


def test_completion_record_detects_manifest_tampering(tmp_path):
    run_dir = tmp_path / "run"
    _sealed_run(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 99
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="seed|run_manifest_sha256"):
        verify_run_directory(run_dir)


def test_matrix_completion_requires_matching_request_fingerprint(tmp_path):
    run_dir = tmp_path / "run"
    request_sha256 = _sealed_run(run_dir)
    assert _complete(
        run_dir,
        seed=42,
        model="reliability_cascade_full",
        run_request_sha256=request_sha256,
    )
    assert not _complete(
        run_dir,
        seed=42,
        model="reliability_cascade_full",
        run_request_sha256="0" * 64,
    )


def test_matrix_output_separates_context_and_variant(tmp_path):
    path = _run_output(
        tmp_path,
        seed=40,
        variant="no_user",
        max_tokens=2048,
    )
    assert path == (
        tmp_path
        / "reliability_cascade"
        / "tokens_2048"
        / "no_user"
        / "seed_40"
    )


def test_matrix_preflight_rejects_existing_seed_before_launch(tmp_path):
    fresh = tmp_path / "seed_40"
    existing = tmp_path / "seed_42"
    existing.mkdir()

    with pytest.raises(FileExistsError, match=r"--resume.*incomplete seeds"):
        _preflight_output_directories(
            [fresh, existing],
            resume=False,
            overwrite=False,
        )


def test_matrix_preflight_allows_explicit_resume_or_overwrite(tmp_path):
    existing = tmp_path / "seed_42"
    existing.mkdir()
    _preflight_output_directories([existing], resume=True, overwrite=False)
    _preflight_output_directories([existing], resume=False, overwrite=True)


def test_matrix_root_lock_rejects_a_concurrent_writer(tmp_path):
    output_root = tmp_path / "matrix"
    first = _acquire_matrix_root_lock(output_root)
    try:
        with pytest.raises(RuntimeError, match="matrix process"):
            _acquire_matrix_root_lock(output_root)
    finally:
        _release_matrix_root_lock(first)

    second = _acquire_matrix_root_lock(output_root)
    _release_matrix_root_lock(second)


def test_matrix_plan_uses_unique_temporary_and_verifies_persisted_json(tmp_path):
    path = tmp_path / "matrix" / "cascade_experiment_plan.json"
    plan = {"schema": "test-plan/v1", "seeds": [40, 41, 42]}
    _ensure_plan(path, plan)

    assert json.loads(path.read_text(encoding="utf-8")) == plan
    assert not list(path.parent.glob(f".{path.name}.tmp-*"))
    _ensure_plan(path, plan)
    with pytest.raises(ValueError, match="different cascade experiment plan"):
        _ensure_plan(path, {**plan, "seeds": [42]})


def test_matrix_plan_post_write_verification_detects_corrupt_output(
    tmp_path, monkeypatch
):
    path = tmp_path / "matrix" / "cascade_experiment_plan.json"
    plan = {"schema": "test-plan/v1", "seeds": [42]}
    original = matrix_runner.write_json

    def write_corrupt_plan(target, payload):
        original(target, {**payload, "seeds": [99]})

    monkeypatch.setattr(matrix_runner, "write_json", write_corrupt_plan)
    with pytest.raises(ValueError, match="post-write verification"):
        _ensure_plan(path, plan)
    assert not list(path.parent.glob(f".{path.name}.tmp-*"))


def test_atomic_publish_replaces_only_after_staging_is_ready(tmp_path):
    destination = tmp_path / "seed_42"
    destination.mkdir()
    (destination / "value.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".seed_42.staging-test"
    staging.mkdir()
    (staging / "value.txt").write_text("new", encoding="utf-8")
    _publish_directory(staging, destination, overwrite=True)
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()
    assert not list(tmp_path.glob(".seed_42.backup-*"))


def test_interrupted_publish_backup_is_recovered(tmp_path):
    destination = tmp_path / "seed_42"
    backup = tmp_path / ".seed_42.backup-test"
    backup.mkdir()
    (backup / "value.txt").write_text("old", encoding="utf-8")
    _recover_interrupted_publish(destination)
    assert (destination / "value.txt").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
