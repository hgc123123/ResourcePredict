from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import shutil
from uuid import uuid4
from pathlib import Path

import pandas as pd

from common.config import load_config
from common.data import load_dataset
from scripts.make_synthetic_trace import generate_synthetic_trace, synthetic_rows
from scripts.smoke_prediction_pipeline import run_smoke


class TestSyntheticTrace(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("_synthetic_test_" + uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.template = self.root / "template.config.json"
        # JSON is valid YAML. These documented export mappings are intentionally
        # the same physical fields used by configs/paper.local.yaml.
        config = {
            "seed": 42,
            "data": {
                "path": "private.csv", "expected_sha256": "a" * 64,
                "expected_raw_row_count": 19352, "expected_row_count": 19321,
                "expected_columns": list(synthetic_rows(200)[0]),
                "csv_encoding": "utf-8-sig", "timestamp_timezone": "Asia/Shanghai",
                "require_unique_job_ids": True, "label_scale": "ratio",
                "label_out_of_range": "error", "require_complete_metadata": True,
                "filters": {"states": ["COMPLETED"], "exit_codes": ["0:0"]},
                "time_validation": {"check_elapsed": True, "tolerance_seconds": 1},
                "code_redaction": {"enabled": True, "path_prefixes": ["/gpfsnyu/home", "/gpfsnyu/scratch"]},
                "columns": {
                    "job_id": "jobid", "job_name": "jobname", "user": "user",
                    "partition": "partition", "code": "script_content", "gpu_count": "gpu",
                    "memory_peak": "mem_max", "vram_peak": "vram_max",
                    "memory_valid": None, "vram_valid": None, "state": "state1",
                    "exit_code": "exitcode", "order": "submit", "submit_time": "submit",
                    "start_time": "start1", "completion_time": "end1", "elapsed": "elapsed",
                    "time_limit": "time", "node_list": "nodelist",
                    "memory_peak_absolute": "mem_capacity_max", "vram_peak_absolute": "vram_capacity_max",
                    "workdir": "workdir",
                },
                "metadata_features": [
                    {"name": "cpu_request", "column": "cpu", "transform": "identity"},
                    {"name": "memory_request", "column": "mem", "transform": "identity"},
                    {"name": "gpu_request", "column": "gpu", "transform": "identity"},
                    {"name": "walltime_request", "column": "time", "parser": "slurm_minutes", "transform": "identity"},
                    {"name": "memory_per_core", "numerator": "mem", "denominator": "cpu", "transform": "ratio", "denominator_floor": 1},
                ],
                "metadata_standardization": "zscore", "metadata_categorical": [],
            },
            "split": {"train_ratio": 0.8, "validation_ratio": 0.1,
                      "order_type": "datetime", "order_semantics": "submission_time",
                      "manifest": "private_split.csv", "create_if_missing": False,
                      "expected_counts": {"train": 15458, "validation": 1927, "test": 1936},
                      "expected_fingerprint": "b" * 64},
            "evaluation": {"clip_predictions": True, "acc_tolerance": 0.10},
            "last1": {"protocol": "completed_before_submission", "fallback": "train_resource_median"},
            "qwen": {}, "reliability_cascade": {"validation_fit_fraction": 0.70},
        }
        with self.template.open("w", encoding="utf-8") as handle:
            json.dump(config, handle)

    def test_generated_trace_uses_exact_schema_and_private_copy_is_unchanged(self):
        original = self.template.read_bytes()
        manifest = generate_synthetic_trace(self.root / "generated", template_config=self.template, count=200)
        settings = load_config(manifest["config"])
        self.assertEqual(self.template.read_bytes(), original)
        self.assertNotIn("expected_sha256", settings["data"])
        self.assertNotIn("expected_counts", settings["split"])
        self.assertIn("expected_columns", settings["data"])
        raw = pd.read_csv(manifest["csv"])
        self.assertEqual(list(raw), settings["data"]["expected_columns"])
        self.assertTrue(raw["submit"].duplicated().any())
        self.assertEqual(set(raw["gpu"].unique()), {0, 1})
        self.assertTrue(raw["user"].str.contains("new_test_user").any())
        bundle = load_dataset(settings)
        self.assertEqual(len(bundle.retained_frame), 200)
        self.assertEqual(len(bundle.frame), 196)
        self.assertEqual(bundle.protocol["split_counts"], {"train": 158, "test": 20, "validation": 18})

    def test_generator_is_deterministic_and_refuses_existing_artifacts(self):
        first = generate_synthetic_trace(self.root / "a", template_config=self.template, count=200, seed=7)
        second = generate_synthetic_trace(self.root / "b", template_config=self.template, count=200, seed=7)
        self.assertEqual(first["csv_sha256"], second["csv_sha256"])
        with self.assertRaises(FileExistsError):
            generate_synthetic_trace(self.root / "a", template_config=self.template, count=200)

    def test_real_loader_train_tail_history_and_gate_roles(self):
        result = run_smoke(self.root / "smoke", template_config=self.template, count=200)
        self.assertTrue(result["synthetic_plumbing_only"])
        self.assertFalse(result["qwen_training_run"])
        self.assertFalse(result["synthetic_gate_models_fit"])
        self.assertFalse(result["base_checkpoint_validation_overlap"])
        self.assertEqual(result["outer_boundary_purged_rows"], 4)
        self.assertEqual(result["base_internal_purged_rows"], 2)
        self.assertEqual(result["gate_inner_boundary_purged_rows"], 2)
        self.assertGreater(result["last1_replay_checks"]["predictions_using_completed_boundary_purged_history"], 0)
        self.assertTrue((self.root / "smoke" / "SYNTHETIC_SMOKE_REPORT.json").is_file())

    def test_actual_gate_fit_select_when_sklearn_is_available(self):
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn is unavailable; real gate fitting is not claimed")
        result = run_smoke(self.root / "fit_smoke", template_config=self.template, count=320, fit_gates=True)
        self.assertTrue(result["synthetic_gate_models_fit"])
        self.assertTrue(result["synthetic_policy_selected"])

    def test_invalid_trace_sizes_are_rejected(self):
        for count in (199, 210, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                synthetic_rows(count)

    def test_new_study_prepare_freezes_roles_and_population_without_training(self):
        from scripts.run_prediction_experiments import parse_args, prepare
        generated = generate_synthetic_trace(self.root / "study_source", template_config=self.template, count=200)
        model = self.root / "EMPTY_MODEL_PATH_ONLY_NOT_TRAINING"
        model.mkdir()
        out = self.root / "prepared_study"
        args = parse_args(["--config", generated["config"], "--data-csv", generated["csv"],
                           "--model-path", str(model), "--output-root", str(out), "--stage", "prepare"])
        prepare(args)
        prepare(args)  # same request is idempotent
        strict = load_config(out / "protocol" / "train_tail.yaml")
        self.assertFalse(strict["split"]["create_if_missing"])
        self.assertTrue(strict["reliability_cascade"]["require_independent_base_selection"])
        roles = json.loads((out / "protocol" / "train_tail_base_selection.json").read_text())
        self.assertFalse(roles["base_checkpoint_validation_overlap"])
        population = pd.read_csv(out / "protocol" / "population_index.csv")
        self.assertEqual(len(population), 200)
        self.assertEqual(int(population.effective_analysis_row.sum()), 196)
        self.assertTrue(population.script_sha256.str.fullmatch(r"[0-9a-f]{64}").all())
        self.assertFalse((out / "train_tail" / "base").exists())


if __name__ == "__main__":
    unittest.main()
