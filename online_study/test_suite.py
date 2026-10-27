"""CPU-only planning, integrity, and aggregation checks (no ML imports)."""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
import shutil
import unittest
import uuid

from online_study import suite
from online_study.summarize import summarize_plan


class SuiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / (".suite_test_" + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self) -> None:
        resolved = self.root.resolve()
        if resolved.parent == Path(__file__).resolve().parent and resolved.name.startswith(".suite_test_"):
            shutil.rmtree(resolved)

    def _completed_inputs(self, seed: int = 40) -> Path:
        study = self.root / "study"
        base = study / "train_tail" / "base" / "qwen" / "tokens_4096" / "full" / f"seed_{seed}"
        cascade = study / "train_tail" / "cascade" / "reliability_cascade" / "tokens_4096" / "full" / f"seed_{seed}"
        for directory, names in ((base, ["config.resolved.yaml", "run_manifest.json", "predictions.csv", "checkpoint/training_manifest.json", "checkpoint/model_config.json"]),
                                 (cascade, ["run_manifest.json", "selected_policy.json", "artifact_integrity.json", "predictions.csv", "checkpoint/cascade_models.joblib"])):
            for name in names:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
        return study

    def _plan_args(self) -> list[str]:
        study = self._completed_inputs(40)
        self._completed_inputs(41)
        return ["--study-root", str(study), "--output-root", str(self.root / "online"),
                "--seeds", "40", "41", "--repeats", "2"]

    def _report(self, run: dict, mean: float, gpu: str = "A800", data: str = "same-data") -> dict:
        destination = Path(run["output_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        timing = destination / "request_timings.csv"
        timing.write_text("sample_id,total_ms\na,1\nb,2\nc,3\n", encoding="utf-8")
        report = {"schema": suite.REPORT_SCHEMA, "settings": {key: run[key] for key in ("seed", "repeat", "operation", "cache_capacity_entries", "cache_start")},
                  "complete_test_period": True, "offline_parity": {"passed": True},
                  "request_timings_sha256": suite.sha256_file(timing),
                  "data_sha256": data, "split_fingerprint": "same-split", "source_contract": {"sha256": "same-source"},
                  "hardware": {"hostname": "one-host", "gpu_name": gpu}, "software": {"python": "3.12"},
                  "all_requests": {"n": 3, "total_ms": {"mean": mean, "p50": mean, "p95": mean + 1, "p99": mean + 2},
                                   "encoder_forward_rate": 1 / 3, "cache_hit_rate_all_jobs": 1 / 3, "history_bypass_job_rate": 1 / 3,
                                   "serial_service_throughput_jobs_per_second": 1000 / mean},
                  "by_route": {"history_bypass": {"n": 1, "total_ms": {"mean": 1.0}}, "exact_cache_hit": {"n": 0}, "encoder_miss": {"n": 2}}}
        report["live_accuracy"] = [{"resource": resource, "n": 3, "mae": mean / 100, "mse": .01, "rmse": .1, "acc_at_10": .8}
                                   for resource in ("memory", "vram")]
        for check in ("archived_structural_parity", "live_input_structural_parity", "base_batch_one_parity", "cache_parity"):
            report[check] = {"passed": True}
        suite.atomic_json(destination / "online_benchmark.json", report)
        suite.atomic_json(destination / "suite_completion.json", {"command_sha256": run["command_sha256"],
                           "report_sha256": suite.sha256_file(destination / "online_benchmark.json"),
                           "request_timings_sha256": report["request_timings_sha256"]})
        return report

    def _synthetic_plan(self) -> tuple[Path, dict]:
        output = self.root / "synthetic"
        case = {"case": "full_lru_1024", "operation": "full", "cache_capacity_entries": 1024}
        runs = []
        for seed, values in ((40, [2.0, 4.0]), (41, [10.0, 12.0])):
            for repeat, value in enumerate(values, start=1):
                run = {**case, "seed": seed, "repeat": repeat, "cache_start": "validation",
                       "output_dir": str(output / f"seed_{seed}" / f"repeat_{repeat}"), "command_sha256": f"cmd-{seed}-{repeat}",
                       "source_contract_sha256": "same-source"}
                runs.append(run)
                self._report(run, value)
        plan = {"schema": suite.PLAN_SCHEMA, "settings": {"seeds": [40, 41], "technical_repeats": 2}, "cases": [case], "runs": runs}
        plan["plan_sha256"] = suite.canonical_hash(plan)
        path = output / "online_suite_plan.json"
        suite.atomic_json(path, plan)
        return path, plan

    def test_case_matrix_is_complete_reproducible_and_reuses_frozen_inputs(self) -> None:
        args = suite.parse_args(self._plan_args())
        first = suite.build_plan(args)
        second = suite.build_plan(args)
        self.assertEqual(first, second)
        self.assertEqual(len(first["runs"]), 2 * 2 * 7)
        self.assertEqual({case["case"] for case in first["cases"]},
                         {"full_lru_0", "full_lru_128", "full_lru_1024", "no_bypass_lru_1024", "no_post_lru_1024", "no_trust_lru_1024", "exact_bypass_lru_1024"})
        self.assertTrue(first["frozen_inputs_sha256"])
        for run in first["runs"]:
            self.assertIn("--parity-predictions", run["command"])
            self.assertNotIn("--overwrite", run["command"])
        self.assertFalse((self.root / "online").exists())

    def test_dry_run_does_not_create_output(self) -> None:
        argv = self._plan_args() + ["--dry-run"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main(argv), 0)
        self.assertFalse((self.root / "online").exists())

    def test_changed_frozen_input_changes_plan_identity(self) -> None:
        args = suite.parse_args(self._plan_args())
        first = suite.build_plan(args)
        path = next(Path(name) for name in first["frozen_inputs_sha256"] if name.endswith("selected_policy.json"))
        path.write_text('{"changed": true}', encoding="utf-8")
        self.assertNotEqual(first["plan_sha256"], suite.build_plan(args)["plan_sha256"])

    def test_duplicate_seeds_and_invalid_capacity_are_rejected(self) -> None:
        argv = self._plan_args()
        with self.assertRaises(ValueError):
            suite.build_plan(suite.parse_args(argv + ["--seeds", "40", "40"]))
        with self.assertRaises(ValueError):
            suite.build_plan(suite.parse_args(argv + ["--cache-capacities", "-1"]))

    def test_verified_resume_rejects_modified_timing_and_missing_receipt(self) -> None:
        _, plan = self._synthetic_plan()
        run = plan["runs"][0]
        suite.verify_report(run, require_receipt=True)
        destination = Path(run["output_dir"])
        (destination / "request_timings.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            suite.verify_report(run, require_receipt=True)
        self._report(run, 2)
        (destination / "suite_completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "receipt"):
            suite.verify_report(run, require_receipt=True)

    def test_summary_averages_repeats_then_uses_training_seed_sample_sd(self) -> None:
        path, _ = self._synthetic_plan()
        result = summarize_plan(path, self.root / "summary")
        row = next(row for row in result["summary"] if row["metric"] == "all_requests.total_ms.mean")
        self.assertAlmostEqual(row["mean"], 7)
        self.assertAlmostEqual(row["sd"], math.sqrt(32))
        self.assertEqual(row["n_training_seeds"], 2)
        self.assertTrue(row["all_seeds_and_repeats_observed"])
        self.assertEqual({row["mean_over_technical_repeats"] for row in result["per_seed"] if row["metric"] == "all_requests.total_ms.mean"}, {3, 11})
        self.assertTrue((self.root / "summary" / "summary.csv").is_file())
        accuracy = next(row for row in result["summary"] if row["metric"] == "live_accuracy.memory.mae")
        self.assertAlmostEqual(accuracy["mean"], .07)

    def test_summary_rejects_mixed_hardware_or_dataset(self) -> None:
        path, plan = self._synthetic_plan()
        self._report(plan["runs"][-1], 12, gpu="4090D")
        with self.assertRaisesRegex(ValueError, "hardware"):
            summarize_plan(path, self.root / "summary")
        self._report(plan["runs"][-1], 12, data="different-data")
        with self.assertRaisesRegex(ValueError, "datasets"):
            summarize_plan(path, self.root / "summary")

    def test_summary_rejects_incomplete_matrix_even_with_valid_plan_hash(self) -> None:
        path, plan = self._synthetic_plan()
        plan["runs"].pop()
        plan["plan_sha256"] = suite.canonical_hash({key: value for key, value in plan.items() if key != "plan_sha256"})
        suite.atomic_json(path, plan)
        with self.assertRaisesRegex(ValueError, "complete"):
            summarize_plan(path, self.root / "summary")

    def test_failed_parity_cannot_be_reused(self) -> None:
        _, plan = self._synthetic_plan()
        run = plan["runs"][0]
        destination = Path(run["output_dir"])
        report = suite.read_json(destination / "online_benchmark.json")
        report["offline_parity"]["passed"] = False
        suite.atomic_json(destination / "online_benchmark.json", report)
        with self.assertRaisesRegex(ValueError, "parity"):
            suite.verify_report(run)


if __name__ == "__main__":
    unittest.main()
