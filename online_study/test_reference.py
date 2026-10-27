"""Reference reuse/integrity checks require neither Torch nor a checkpoint."""
from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from online_study.reference import (
    COMPLETE_NAME, CSV_NAME, METADATA_NAME, _expected_contract, _population_ids,
    _seal_reference, _sha256, _validate_frame, load_or_build_base_reference,
)


class BaseReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parent / (".reference_test_" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.frame = pd.DataFrame({"sample_id": ["001", "NA", "b"]})
        self.identity = {"checkpoint_sha256": "frozen", "source_sha256": "source-v1", "device": {"name": "test"}}
        self.reference = pd.DataFrame({
            "sample_id": ["001", "NA", "b"], "qwen_memory": [.1, .2, .3],
            "qwen_vram": [.4, .5, .6], "gate_code": [.2, .3, .4],
            "gate_user": [.3, .2, .1], "gate_metadata": [.5, .5, .5],
            "user_known": [True, False, True],
        })

    def tearDown(self):
        resolved = self.directory.resolve()
        if resolved.parent == Path(__file__).resolve().parent and resolved.name.startswith(".reference_test_"):
            shutil.rmtree(resolved)

    def seal(self):
        contract = _expected_contract(self.identity, _population_ids(self.frame))
        _seal_reference(self.directory, self.reference, contract)

    def load(self, **changes):
        args = dict(frame=self.frame, model=None, preprocessors=None, collator=None,
                    device=None, dtype=None, identity=self.identity)
        args.update(changes)
        return load_or_build_base_reference(self.directory, **args)

    def test_valid_reuse_requires_no_model_or_torch_and_preserves_ids(self):
        self.seal()
        frame, metadata = self.load()
        pd.testing.assert_frame_equal(frame, self.reference)
        self.assertTrue(metadata["reused"])
        self.assertEqual(metadata["numerical_mode"]["batch_size"], 1)
        self.assertFalse(metadata["numerical_mode"]["label_dependent_filtering"])

    def test_identity_source_and_order_changes_reject_reuse(self):
        self.seal()
        for key in ("checkpoint_sha256", "source_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "changed"):
                self.load(identity={**self.identity, key: "changed"})
        with self.assertRaisesRegex(ValueError, "changed"):
            self.load(frame=self.frame.iloc[::-1].reset_index(drop=True))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.load(frame=self.frame.iloc[:-1])

    def test_unsealed_and_corrupted_outputs_fail_without_removal(self):
        (self.directory / CSV_NAME).write_text("incomplete", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()
        self.assertTrue((self.directory / CSV_NAME).exists())

    def test_both_metadata_and_csv_hashes_are_verified(self):
        self.seal()
        for name in (CSV_NAME, METADATA_NAME):
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "integrity failure"):
                self.load()
            path.write_bytes(original)

    def test_valid_seal_cannot_hide_invalid_predictions_or_population(self):
        self.seal()
        corrupted = self.reference.copy()
        corrupted.loc[0, "qwen_memory"] = np.inf
        path = self.directory / CSV_NAME
        corrupted.to_csv(path, index=False)
        meta_path = self.directory / METADATA_NAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["base_csv_sha256"] = _sha256(path)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        seal_path = self.directory / COMPLETE_NAME
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["files"] = {name: _sha256(self.directory / name) for name in (CSV_NAME, METADATA_NAME)}
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.load()

    def test_reference_population_and_flags_are_strict(self):
        for ids in (["a", "a"], ["a", None], ["a", ""]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "sample_id"):
                _population_ids(pd.DataFrame({"sample_id": ids}))
        bad = self.reference.copy()
        bad["user_known"] = ["true", "maybe", "false"]
        with self.assertRaisesRegex(ValueError, "boolean"):
            _validate_frame(bad, self.frame.sample_id.tolist())
        with self.assertRaisesRegex(ValueError, "order"):
            _validate_frame(self.reference.iloc[::-1], self.frame.sample_id.tolist())


if __name__ == "__main__":
    unittest.main()
