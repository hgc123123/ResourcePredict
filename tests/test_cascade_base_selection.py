"""CPU-only provenance checks; no model fitting, prediction, or YAML I/O.

Run with ``python -m unittest tests.test_cascade_base_selection -v``.
When PyYAML is absent, an import-only substitute raises if any YAML parsing is
attempted. No estimator, torch operation, or training result is mocked.
Without scikit-learn, namespace-package loading skips only the package's eager
public estimator exports; the real pipeline and all provenance code execute.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys
import unittest

import numpy as np
import pandas as pd


try:
    import yaml  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "yaml":
        raise

    def _yaml_unavailable(*args, **kwargs):
        raise RuntimeError("PyYAML is unavailable; these tests must not perform YAML I/O")

    _yaml_import_only = ModuleType("yaml")
    _yaml_import_only.safe_load = _yaml_unavailable
    _yaml_import_only.safe_dump = _yaml_unavailable
    sys.modules["yaml"] = _yaml_import_only

if importlib.util.find_spec("sklearn") is None and "reliability_cascade" not in sys.modules:
    _cascade_namespace = ModuleType("reliability_cascade")
    _cascade_namespace.__path__ = [str(Path(__file__).resolve().parents[1] / "reliability_cascade")]
    sys.modules["reliability_cascade"] = _cascade_namespace

from common.data import DatasetBundle
from qwen.base_selection import _record_hash, select_base_training_frames
from reliability_cascade.pipeline import (
    PreparedCascade,
    _base_checkpoint_selection_protocol,
    chronological_calibration_masks,
)


def _bundle():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    hours = list(range(10)) + list(range(20, 30)) + [40, 41]
    frame = pd.DataFrame({
        "sample_id": [f"sample_{index}" for index in range(len(hours))],
        "split": ["train"] * 10 + ["validation"] * 10 + ["test"] * 2,
        "submit_time": [start + pd.Timedelta(hours=hour) for hour in hours],
        "valid_memory": True,
        "valid_vram": True,
        "target_memory": 0.2,
        "target_vram": 0.3,
    })
    frame["completion_time"] = frame["submit_time"] + pd.Timedelta(minutes=30)
    frame.loc[8, "completion_time"] = frame.loc[9, "submit_time"]
    return DatasetBundle(
        frame, Path("private-not-opened.csv"), "a" * 64, None, [], [],
        {"verified_chronological": True}, retained_frame=frame.copy(),
    )


def _config(mode="train_tail", require_independent=False):
    return {
        "qwen": {"base_selection": {"mode": mode, "fit_fraction": 0.9}},
        "reliability_cascade": {
            "validation_fit_fraction": 0.7,
            "require_independent_base_selection": require_independent,
        },
    }


def _manifest(bundle, config):
    record = select_base_training_frames(bundle, config).manifest
    details = {}
    if record is not None:
        details = {
            "base_training_selection": record,
            "effective_parameters": {"base_training_selection_sha256": record["sha256"]},
        }
    return {"details": details}


class CascadeBaseSelectionTests(unittest.TestCase):
    def test_verified_independent_record_derives_no_overlap(self):
        bundle = _bundle()
        config = _config(require_independent=True)
        manifest = _manifest(bundle, config)
        protocol = _base_checkpoint_selection_protocol(bundle, config, manifest)
        self.assertIs(protocol["base_checkpoint_validation_overlap"], False)
        self.assertEqual(protocol["base_checkpoint_selection_provenance"], "verified_train_tail")
        self.assertEqual(protocol["base_training_selection_sha256"],
                         manifest["details"]["base_training_selection"]["sha256"])
        self.assertIn("previously inspected", protocol["warning"])

    def test_verified_control_explicitly_reports_overlap(self):
        bundle = _bundle()
        config = _config("shared_validation_control")
        protocol = _base_checkpoint_selection_protocol(bundle, config, _manifest(bundle, config))
        self.assertIs(protocol["base_checkpoint_validation_overlap"], True)
        self.assertEqual(protocol["base_checkpoint_selection_provenance"],
                         "verified_shared_validation_control")
        self.assertIn("deliberately", protocol["warning"])

    def test_strict_requirement_rejects_control(self):
        bundle = _bundle()
        config = _config("shared_validation_control", require_independent=True)
        with self.assertRaisesRegex(ValueError, "requires train_tail provenance"):
            _base_checkpoint_selection_protocol(bundle, config, _manifest(bundle, config))

    def test_legacy_artifacts_keep_original_warning(self):
        protocol = _base_checkpoint_selection_protocol(_bundle(), {}, {})
        self.assertIs(protocol["base_checkpoint_validation_overlap"], True)
        self.assertEqual(protocol["base_checkpoint_selection_provenance"], "legacy_unverified")
        self.assertIn("full validation period", protocol["warning"])
        self.assertIsNone(protocol["base_training_selection_sha256"])

    def test_strict_flag_cannot_self_certify_missing_record(self):
        with self.assertRaisesRegex(ValueError, "legacy artifact has no such record"):
            _base_checkpoint_selection_protocol(
                _bundle(), {"reliability_cascade": {"require_independent_base_selection": True}}, {}
            )
        with self.assertRaisesRegex(ValueError, "lacks base_training_selection"):
            _base_checkpoint_selection_protocol(_bundle(), _config(require_independent=True), {})

    def test_new_mode_without_record_fails_even_without_strict_flag(self):
        with self.assertRaisesRegex(ValueError, "lacks base_training_selection"):
            _base_checkpoint_selection_protocol(_bundle(), _config(), {})

    def test_tampered_record_fails_fingerprint_check(self):
        bundle, config = _bundle(), _config()
        manifest = _manifest(bundle, config)
        manifest["details"]["base_training_selection"]["base_checkpoint_validation_overlap"] = True
        with self.assertRaisesRegex(ValueError, "fingerprint is invalid"):
            _base_checkpoint_selection_protocol(bundle, config, manifest)

    def test_rehashed_tampering_still_fails_source_reconstruction(self):
        bundle, config = _bundle(), _config()
        manifest = _manifest(bundle, config)
        record = manifest["details"]["base_training_selection"]
        record["early_stopping_sample_ids"] = ["sample_20"]
        record["sha256"] = _record_hash(record)
        manifest["details"]["effective_parameters"]["base_training_selection_sha256"] = record["sha256"]
        with self.assertRaisesRegex(ValueError, "does not match source data/configuration"):
            _base_checkpoint_selection_protocol(bundle, config, manifest)

    def test_effective_hash_must_be_present_and_match(self):
        bundle, config = _bundle(), _config()
        for value in (None, "b" * 64, True):
            manifest = _manifest(bundle, config)
            manifest["details"]["effective_parameters"]["base_training_selection_sha256"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "does not match"):
                _base_checkpoint_selection_protocol(bundle, config, manifest)

    def test_changed_source_or_fraction_is_rejected(self):
        bundle, config = _bundle(), _config()
        manifest = _manifest(bundle, config)
        altered = deepcopy(bundle)
        altered.frame.loc[7, "completion_time"] = altered.frame.loc[9, "submit_time"]
        with self.assertRaisesRegex(ValueError, "does not match source data/configuration"):
            _base_checkpoint_selection_protocol(altered, config, manifest)
        changed = deepcopy(config)
        changed["qwen"]["base_selection"]["fit_fraction"] = 0.8
        with self.assertRaisesRegex(ValueError, "does not match source data/configuration"):
            _base_checkpoint_selection_protocol(bundle, changed, manifest)

    def test_new_record_with_legacy_config_is_rejected(self):
        bundle = _bundle()
        with self.assertRaisesRegex(ValueError, "incompatible with legacy_validation"):
            _base_checkpoint_selection_protocol(bundle, {}, _manifest(bundle, _config()))

    def test_independence_requirement_must_be_boolean(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be true or false"):
                _base_checkpoint_selection_protocol(
                    _bundle(), {"reliability_cascade": {"require_independent_base_selection": value}}, {}
                )

    def test_calibration_masks_propagate_verified_overlap_and_preserve_test_exclusion(self):
        bundle, config = _bundle(), _config(require_independent=True)
        frame = pd.concat([
            bundle.frame.assign(resource="memory"), bundle.frame.assign(resource="vram")
        ], ignore_index=True)
        prepared = PreparedCascade(
            components=SimpleNamespace(qwen_manifest=_manifest(bundle, config)),
            bundle=bundle, features=SimpleNamespace(frame=frame), qwen_backbone="unused",
        )
        fit, selection, protocol = chronological_calibration_masks(prepared, config)
        self.assertIs(protocol["base_checkpoint_validation_overlap"], False)
        self.assertEqual(protocol["fit_unique_jobs"], 7)
        self.assertEqual(protocol["selection_unique_jobs"], 3)
        self.assertFalse(np.any(fit & selection))
        test = frame["split"].eq("test").to_numpy()
        self.assertFalse(np.any(test & (fit | selection)))
        self.assertEqual(int(fit.sum()), 14)
        self.assertEqual(int(selection.sum()), 6)

    def test_legacy_mask_only_fixture_preserves_original_behavior(self):
        bundle = _bundle()
        prepared = SimpleNamespace(features=SimpleNamespace(frame=bundle.frame.copy()))
        fit, selection, protocol = chronological_calibration_masks(prepared, {})
        self.assertEqual(int(fit.sum()), 7)
        self.assertEqual(int(selection.sum()), 3)
        self.assertIs(protocol["base_checkpoint_validation_overlap"], True)
        self.assertEqual(protocol["base_checkpoint_selection_provenance"], "legacy_unverified")


if __name__ == "__main__":
    unittest.main()
