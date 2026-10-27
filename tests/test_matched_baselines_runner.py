"""Standard-library checks for the new runner's execution and resume contract."""
from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from baseline_study.protocol import (locked_json, read_json, seal_run,
                                     source_inventory, verify_seal, write_json)

ROOT = Path(__file__).resolve().parents[1]


class BaselineRunnerTests(unittest.TestCase):
    def test_dry_run_needs_no_model_data_or_learning_dependencies(self):
        completed = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_matched_baselines.py"),
            "--dry-run", "--models", "xgboost", "textcnn", "lstm",
            "--seeds", "40", "41", "42", "43", "44",
        ], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.count("[PLAN] train "), 15)
        self.assertIn("primary optimization/early-stop", completed.stdout)

    def test_locked_contract_rejects_changed_parameters(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            locked_json(path, {"seed": 40, "mode": "train_tail"})
            locked_json(path, {"seed": 40, "mode": "train_tail"})
            with self.assertRaises(FileExistsError):
                locked_json(path, {"seed": 44, "mode": "train_tail"})
            self.assertEqual(read_json(path)["seed"], 40)

    def test_parallel_prepare_keeps_one_complete_contract(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            payload = {"source": "new-primary", "roles": list(range(1000))}
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: locked_json(path, payload), range(16)))
            self.assertEqual(len(results), 16)
            self.assertEqual(read_json(path), payload)

    def test_resume_seal_rejects_changed_or_missing_outputs(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            contract = {"model": "xgboost", "seed": 40}
            write_json(output / "request.json", contract)
            for name in ("metrics.csv", "predictions.csv", "run_manifest.json",
                         "config.resolved.yaml", "base_training_selection.json"):
                (output / name).write_text("original", encoding="utf-8")
            seal_run(output, contract)
            self.assertEqual(verify_seal(output, contract)["contract"], contract)
            with self.assertRaises(ValueError):
                verify_seal(output, {"model": "xgboost", "seed": 44})
            (output / "predictions.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                verify_seal(output, contract)

    def test_source_inventory_includes_unchanged_model_builders(self):
        files = source_inventory()
        for name in ("textcnn/model.py", "lstm/model.py", "xgboost/run.py",
                     "common/neural.py", "qwen/base_selection.py",
                     "baseline_study/neural.py", "scripts/run_matched_baselines.py"):
            self.assertIn(name, files)
            self.assertEqual(len(files[name]), 64)


if __name__ == "__main__":
    unittest.main()
