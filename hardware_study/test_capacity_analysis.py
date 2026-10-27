"""Synthetic checks plus optional read-only sealed-output smoke; no private CSV."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("capacity_analysis_under_test", HERE / "run_capacity_analysis.py")
cap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cap)
DIAG, _ = cap.load_diagnostics(HERE.parent)


class SyntheticWorkspace(unittest.TestCase):
    """Use a unique normal directory; no dependence on historical private input."""

    def setUp(self):
        self.workspace = Path(tempfile.gettempdir()) / ("capacity-synthetic-" + uuid.uuid4().hex)
        self.workspace.mkdir()

    def tearDown(self):
        self.assertTrue(self.workspace.name.startswith("capacity-synthetic-"))
        shutil.rmtree(self.workspace)

    @staticmethod
    def save_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_sealed_run(self, directory, predictions, identity, seed):
        directory.mkdir(parents=True)
        predictions.to_csv(directory / "predictions.csv", index=False)
        self.save_json(directory / "run_manifest.json", {**identity, "seed": seed})
        self.save_json(directory / "checkpoint/checkpoint_contract.json", {"synthetic": True})
        request_hash = "a" * 64
        seal = {"schema": "reliability-cascade-artifacts/v1", "run_request_sha256": request_hash,
                "files": {name: cap.digest(directory / name) for name in
                    ("predictions.csv", "run_manifest.json", "checkpoint/checkpoint_contract.json")}}
        self.save_json(directory / "artifact_integrity.json", seal)
        self.save_json(directory / "completion_record.json", {
            "run_request_sha256": request_hash,
            "run_manifest_sha256": cap.digest(directory / "run_manifest.json"),
            "artifact_integrity_sha256": cap.digest(directory / "artifact_integrity.json"),
            "checkpoint_contract_sha256": cap.digest(directory / "checkpoint/checkpoint_contract.json")})


class NodeExpansionTests(unittest.TestCase):
    def test_compact_ranges_and_separate_nodes(self):
        self.assertEqual(cap.expand_nodelist("gpu[4,6-8],synthetic7"),
                         ["gpu4", "gpu6", "gpu7", "gpu8", "synthetic7"])

    def test_padding_and_cartesian_products(self):
        self.assertEqual(cap.expand_nodelist("rack[01-02]gpu[3,05]"),
                         ["rack01gpu3", "rack01gpu05", "rack02gpu3", "rack02gpu05"])

    def test_literal_list_is_not_executed(self):
        self.assertEqual(cap.expand_nodelist("['gpu4', 'gpu[6-7]']"),
                         ["gpu4", "gpu6", "gpu7"])
        with self.assertRaises(ValueError):
            cap.expand_nodelist("__import__('os').system('false')")

    def test_empty_markers(self):
        for value in ("", "(null)", "None", "n/a"):
            self.assertEqual(cap.expand_nodelist(value), [])

    def test_reject_malformed_duplicate_and_excessive_lists(self):
        for value in ("gpu[4-2]", "gpu[1-3", "gpu1]", "gpu[[1]]", "gpu[1:3]",
                      "gpu[1,,2]", "gpu1,", "gpu1,gpu1", "gpu[1-2],gpu2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cap.expand_nodelist(value)
        for value in ("gpu[1-11]", "rack[1-4]gpu[1-4]", "gpu[1-10],synthetic7"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cap.expand_nodelist(value, max_nodes=10)


class CapacityRegistryTests(unittest.TestCase):
    inventory = {"gpu1": 24564, "gpu2": 24564, "gpu3": 24576, "gpu4": 81920}

    def test_nearby_values_stay_distinct_without_unit_conversion(self):
        a = cap.classify_nodes("gpu1", 1, self.inventory)
        b = cap.classify_nodes("gpu3", 8, self.inventory)
        self.assertEqual((a["registered_capacity"], b["registered_capacity"]), (24564, 24576))
        self.assertNotEqual(a["capacity_exact"], b["capacity_exact"])
        self.assertEqual(a["capacity_band"], b["capacity_band"])

    def test_capacity_is_per_gpu_not_total_allocation(self):
        row = cap.classify_nodes("gpu[1-2]", 16, self.inventory)
        self.assertEqual(row["registered_capacity"], 24564)
        self.assertEqual(row["expanded_node_count"], 2)
        self.assertEqual(row["capacity_status"], "mapped_homogeneous")

    def test_unknown_mixed_and_unparsed_remain_explicit(self):
        for nodes, status in (("gpu[1,99]", "unmapped_node"),
                              ("gpu[1,3]", "mixed_registered_capacity"),
                              ("gpu[1-", "unparsed_nodelist"),
                              ("", "missing_nodelist")):
            with self.subTest(nodes=nodes):
                row = cap.classify_nodes(nodes, 1, self.inventory)
                self.assertEqual(row["capacity_status"], status)
                self.assertTrue(np.isnan(row["registered_capacity"]))
        self.assertEqual(cap.classify_nodes("gpu99", 1, self.inventory)["unknown_nodes"], ("gpu99",))

    def test_cpu_jobs_need_no_gpu_inventory(self):
        self.assertEqual(cap.classify_nodes("cpu[001-004]", 0, self.inventory)["capacity_status"], "cpu_only")

    def test_inventory_values_are_positive_integers(self):
        for invalid in ({}, {"gpu1": 0}, {"gpu1": -1}, {"gpu1": True},
                        {"gpu1": "24564"}, {"gpu1": 24.5}, {"gpu[1]": 24564}):
            with self.subTest(invalid=invalid), patch.object(Path, "read_text", return_value=json.dumps(invalid)):
                with self.assertRaises(ValueError):
                    cap.load_inventory(Path("unused.json"))


class ModelRegistryTests(unittest.TestCase):
    inventory = {"gpu1": {"gres_type": "P100", "installed_gpu_count": 2},
                 "gpu2": {"gres_type": "p100", "installed_gpu_count": 4},
                 "gpu3": {"gres_type": "3080Ti", "installed_gpu_count": 8},
                 "untypedgpu": {"gres_type": None, "installed_gpu_count": 10}}

    def test_case_normalization_without_model_guessing(self):
        row = cap.classify_model("gpu[1-2]", 6, self.inventory)
        self.assertEqual(row["model_status"], "mapped_homogeneous")
        self.assertEqual(row["gpu_model"], "gpu_type_p100")
        self.assertEqual(cap.classify_model("gpu3", 1, self.inventory)["gpu_model"], "gpu_type_3080ti")

    def test_mixed_and_untyped_nodes_stay_explicit(self):
        self.assertEqual(cap.classify_model("gpu[1,3]", 2, self.inventory)["model_status"], "mixed_gpu_models")
        row = cap.classify_model("untypedgpu", 1, self.inventory)
        self.assertEqual(row["model_status"], "untyped_gres")
        self.assertEqual(row["gpu_model"], "untyped_gres")

    def test_unknown_node_and_parse_failures_are_not_silently_dropped(self):
        row = cap.classify_model("gpu[1,99]", 2, self.inventory)
        self.assertEqual(row["model_status"], "unmapped_model_node")
        self.assertEqual(row["unknown_model_nodes"], ("gpu99",))
        self.assertEqual(cap.classify_model("gpu[1-", 1, self.inventory)["model_status"], "unparsed_nodelist")
        self.assertEqual(cap.classify_model("", 1, self.inventory)["model_status"], "missing_nodelist")
        self.assertEqual(cap.classify_model("cpu999", 0, self.inventory)["model_status"], "cpu_only")

    def test_capacity_and_model_are_independent_information(self):
        self.assertEqual(cap.classify_nodes("gpu1", 1, {})["capacity_status"], "unmapped_node")
        self.assertEqual(cap.classify_model("gpu1", 1, self.inventory)["gpu_model"], "gpu_type_p100")
        self.assertEqual(cap.classify_nodes("untypedgpu", 1, {"untypedgpu": 12288})["registered_capacity"], 12288)
        self.assertEqual(cap.classify_model("untypedgpu", 1, self.inventory)["model_status"], "untyped_gres")

    def test_invalid_model_registry_records_are_rejected(self):
        invalids = ({}, {"gpu1": "P100"}, {"gpu1": {"gres_type": "P100"}},
                    {"gpu1": {"gres_type": "P100", "installed_gpu_count": 0}},
                    {"gpu1": {"gres_type": "P100", "installed_gpu_count": True}},
                    {"gpu1": {"gres_type": "P100", "installed_gpu_count": 1.5}},
                    {"gpu1": {"gres_type": "", "installed_gpu_count": 1}})
        for invalid in invalids:
            with self.subTest(invalid=invalid), patch.object(Path, "read_text", return_value=json.dumps(invalid)):
                with self.assertRaises(ValueError):
                    cap.load_model_inventory(Path("unused.json"))

    def test_valid_untyped_registry_is_allowed(self):
        with patch.object(Path, "read_text", return_value=json.dumps(self.inventory)):
            self.assertEqual(cap.load_model_inventory(Path("unused.json")), self.inventory)


class MetadataIdentityTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({"job_id": ["0001", "2"], "nodelist": ["gpu1", "cpu1"],
                                 "raw_gpu_count": ["1", "0"], "raw_source_row": [0, 1]})
        self.population = pd.DataFrame({"sample_id": ["0001::0", "2::1"],
            "split": ["test", "test"], "gpu_count": [1, 0],
            "submit_time": ["2026-04-10T01:00:00Z", "2026-04-11T01:00:00Z"],
            "valid_memory": [True, True], "valid_vram": [True, False],
            "effective_analysis_row": [True, True]})
        self.split = pd.DataFrame({"sample_id": ["0001::0", "2::1"],
                                    "job_id": ["0001", "2"], "source_row": [0, 1]})

    def attach(self):
        return cap.attach_metadata(self.raw, self.population, self.split, {"gpu1": 24564}, DIAG)

    def test_exact_string_id_and_raw_order(self):
        joined = self.attach()
        self.assertEqual(joined.job_id.tolist(), ["0001", "2"])
        self.assertEqual(joined.capacity_status.tolist(), ["mapped_homogeneous", "cpu_only"])
        self.assertEqual(joined.month.tolist(), ["2026-04", "2026-04"])

    def test_no_numeric_coercion_for_identity(self):
        self.raw.loc[0, "job_id"] = "1"
        with self.assertRaisesRegex(ValueError, "absent from raw CSV"):
            self.attach()

    def test_changed_raw_row_order_fails(self):
        self.raw["raw_source_row"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "source rows"):
            self.attach()

    def test_changed_gpu_request_fails(self):
        self.raw.loc[0, "raw_gpu_count"] = "2"
        with self.assertRaisesRegex(ValueError, "GPU requests differ"):
            self.attach()

    def test_missing_and_duplicate_id_fail(self):
        self.raw.loc[1, "job_id"] = "0001"
        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            self.attach()
        self.raw.loc[1, "job_id"] = None
        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            self.attach()

    def test_gpu_counts_are_strict_nonnegative_integers(self):
        for values in ([1.5], [-1], [float("inf")], [float("nan")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cap.numeric_gpu(pd.Series(values), "synthetic")


class MetricMeaningTests(unittest.TestCase):
    def test_signed_error_and_shortfall_are_in_utilization_ratio_units(self):
        row = cap.capacity_metrics([0.8, 0.1], [0.5, 0.2], DIAG, .10)
        self.assertAlmostEqual(row["mean_signed_error"], -.1)
        self.assertEqual(row["underprediction_rate"], .5)
        self.assertEqual(row["underprediction_gt_010_rate"], .5)
        self.assertEqual(row["n"], 2)

    def test_seed_summary_does_not_multiply_population(self):
        frame = pd.DataFrame({"resource": ["vram", "vram"], "seed": [40, 41],
                              "n": [100, 100], "day_blocks": [5, 5], "mae": [.10, .12]})
        summary = cap.aggregate_seeds(frame, ["resource"], ["mae"])
        self.assertEqual(summary.loc[0, "n"], 100)
        self.assertEqual(summary.loc[0, "seed_count"], 2)
        self.assertAlmostEqual(summary.loc[0, "mae_mean"], .11)
        self.assertAlmostEqual(summary.loc[0, "mae_sample_std"], np.sqrt(.0002))

    def test_seed_population_changes_are_rejected(self):
        frame = pd.DataFrame({"resource": ["vram", "vram"], "seed": [40, 41],
                              "n": [100, 99], "day_blocks": [5, 5], "mae": [.10, .12]})
        with self.assertRaisesRegex(ValueError, "population differs"):
            cap.aggregate_seeds(frame, ["resource"], ["mae"])


class SealedIntegrationTests(SyntheticWorkspace):
    def build_study(self):
        study = self.workspace / "study"
        protocol = study / "protocol"
        protocol.mkdir(parents=True)
        nodes = ["cpu1", "syntheticgpu1", "syntheticgpu1", "syntheticgpu2", "cpu1",
                 "syntheticgpu1", "syntheticgpu2", "syntheticgpu3",
                 "syntheticgpu[1-2]", "syntheticgpu99"]
        ids = [str(i).zfill(6) for i in range(len(nodes))]
        gpu = [0 if node.startswith("cpu") else 1 for node in nodes]
        split = ["train"] * 2 + ["validation"] * 2 + ["test"] * 6
        times = pd.date_range("2026-01-01", periods=len(nodes), freq="D", tz="UTC")
        sample_ids = [f"{job_id}::{i}" for i, job_id in enumerate(ids)]
        raw = pd.DataFrame({"jobid": ids, "nodelist": nodes, "gpu": gpu})
        raw_path = self.workspace / "synthetic.csv"
        raw.to_csv(raw_path, index=False)
        manifest = pd.DataFrame({"sample_id": sample_ids, "job_id": ids,
                                 "source_row": np.arange(len(nodes)), "split": split})
        manifest.to_csv(protocol / "outer_split.csv", index=False)
        population = pd.DataFrame({"sample_id": sample_ids, "split": split, "gpu_count": gpu,
            "submit_time": times.astype(str), "completion_time": (times + pd.Timedelta(hours=1)).astype(str),
            "valid_memory": True, "valid_vram": np.asarray(gpu) > 0, "effective_analysis_row": True})
        population.to_csv(protocol / "population_index.csv", index=False)
        identity = {"data_sha256": cap.digest(raw_path), "split_fingerprint": "b" * 64,
                    "split_manifest_sha256": cap.digest(protocol / "outer_split.csv")}
        self.save_json(protocol / "data_identity.json", identity)
        self.save_json(protocol / "population_index.json", {**identity,
            "population_index_sha256": cap.digest(protocol / "population_index.csv")})
        registry = self.workspace / "capacities.json"
        self.save_json(registry, {"syntheticgpu1": 24564, "syntheticgpu2": 24564, "syntheticgpu3": 81920})
        model_registry = self.workspace / "models.json"
        self.save_json(model_registry, {"syntheticgpu1": {"gres_type": "typeA", "installed_gpu_count": 8},
            "syntheticgpu2": {"gres_type": "typeB", "installed_gpu_count": 8},
            "syntheticgpu3": {"gres_type": None, "installed_gpu_count": 8}})
        for seed in range(40, 45):
            for variant in ("full", "no_code"):
                rows = []
                for i in range(2, len(nodes)):
                    for resource in ("memory", "vram"):
                        applicable = resource == "memory" or gpu[i] > 0
                        target = .20 + .02 * i if applicable else np.nan
                        rows.append({"sample_id": sample_ids[i], "job_id": ids[i], "resource": resource,
                            "seed": seed, "user_id": "synthetic_user", "gpu_count": gpu[i],
                            "split": split[i], "submit_time": str(times[i]),
                            "completion_time": str(times[i] + pd.Timedelta(hours=1)),
                            "valid": applicable, "resource_applicable": applicable, "target": target,
                            "prediction": target + (.02 if variant == "full" else .04) + .001 * (seed - 40),
                            "hierarchical_reference_prediction": target + .06,
                            "qwen_prediction": target + .08})
                directory = study / "train_tail/cascade/reliability_cascade/tokens_4096" / variant / f"seed_{seed}"
                self.make_sealed_run(directory, pd.DataFrame(rows), identity, seed)
        return study, raw_path, registry, model_registry

    def test_full_five_seed_analysis_uses_real_seal_reader_and_keeps_unknowns(self):
        study, raw_path, registry, model_registry = self.build_study()
        output = self.workspace / "results"
        rc = cap.main(["--study-root", str(study), "--data-csv", str(raw_path),
                       "--capacity-map", str(registry), "--model-map", str(model_registry), "--output-dir", str(output),
                       "--bootstrap-replicates", "20", "--no-plots"])
        self.assertEqual(rc, 0)
        summary = pd.read_csv(output / "capacity_metrics_summary.csv")
        totals = summary.loc[summary.scope.eq("all_valid") & summary.method.eq("Full Cascade")].set_index("resource")
        self.assertEqual(int(totals.loc["memory", "n"]), 6)
        self.assertEqual(int(totals.loc["vram", "n"]), 5)
        self.assertTrue(totals.seed_count.eq(5).all())
        self.assertAlmostEqual(totals.loc["memory", "mae_mean"], .022)
        coverage = pd.read_csv(output / "capacity_population_coverage.csv")
        self.assertIn("unmapped_node", set(coverage.group))
        self.assertIn("unmapped_model_node", set(coverage.group))
        self.assertIn("untyped_gres", set(coverage.group))
        self.assertIn("mixed_gpu_models", set(coverage.group))
        model_metrics = summary.loc[summary.scope.eq("gpu_model") & summary.resource.eq("vram") &
                                    summary.method.eq("Full Cascade")]
        self.assertEqual(int(model_metrics.n.sum()), 5)
        report = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(report["unit_conversion_performed"])
        self.assertEqual(report["capacity_unit"], "as registered")


class ExactPredictionIdentityTests(SyntheticWorkspace):
    def setUp(self):
        super().setUp()
        self.path = self.workspace / "predictions.csv"
        self.rows = pd.DataFrame({"sample_id": ["000001::0", "000002::1"],
                                  "resource": ["memory", "vram"], "job_id": ["000001", "000002"]})
        self.rows.to_csv(self.path, index=False)
        self.frame = self.rows.iloc[::-1].reset_index(drop=True).copy()
        self.frame["job_id"] = [2, 1]

    def test_recovers_original_lexical_ids_by_keys_not_row_order(self):
        result = cap.restore_string_job_ids(self.frame, self.path)
        self.assertEqual(result.set_index("sample_id").job_id.to_dict(),
                         {"000001::0": "000001", "000002::1": "000002"})

    def test_huge_ids_are_restored_without_float_rounding(self):
        self.rows.loc[0, "job_id"] = "123456789012345678901234567890"
        self.rows.to_csv(self.path, index=False)
        self.frame["job_id"] = [2.0, 1.2345678901234568e29]
        result = cap.restore_string_job_ids(self.frame, self.path)
        self.assertEqual(result.set_index("sample_id").loc["000001::0", "job_id"], "123456789012345678901234567890")

    def test_mismatched_prediction_population_is_rejected(self):
        self.rows.loc[0, "sample_id"] = "different::0"
        self.rows.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            cap.restore_string_job_ids(self.frame, self.path)

    def test_duplicate_keys_are_rejected(self):
        pd.concat([self.rows, self.rows.iloc[[0]]], ignore_index=True).to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            cap.restore_string_job_ids(self.frame, self.path)

    def test_missing_job_id_is_rejected(self):
        self.rows.loc[0, "job_id"] = ""
        self.rows.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            cap.restore_string_job_ids(self.frame, self.path)

    def test_resources_cannot_disagree_on_job_identity(self):
        self.rows.loc[1, "sample_id"] = self.rows.loc[0, "sample_id"]
        self.rows.to_csv(self.path, index=False)
        self.frame.loc[0, "sample_id"] = self.rows.loc[0, "sample_id"]
        with self.assertRaisesRegex(ValueError, "Memory/VRAM job IDs disagree"):
            cap.restore_string_job_ids(self.frame, self.path)


@unittest.skipUnless(os.environ.get("HARDWARE_REAL_SMOKE") == "1", "Optional local sealed-result smoke")
class RealSealedPredictionsSmokeTests(unittest.TestCase):
    def test_five_seed_sealed_runs_and_exact_ids_without_loading_raw_csv(self):
        study = Path(os.environ.get("STUDY_ROOT", str(HERE.parent / "outputs/study")))
        identity = json.loads((study / "protocol/data_identity.json").read_text(encoding="utf-8"))
        metric_rows = []
        for seed in range(40, 45):
            frames = []
            for variant in ("full", "no_code"):
                directory = study / "train_tail/cascade/reliability_cascade/tokens_4096" / variant / f"seed_{seed}"
                frame, manifest, _ = DIAG.read_run(directory, seed, sealed=True)
                DIAG.require_identity(identity, manifest)
                frame = cap.restore_string_job_ids(frame, directory / "predictions.csv")
                self.assertTrue(frame.job_id.eq(frame.sample_id.str.rsplit("::", n=1).str[0]).all())
                frames.append(frame)
            pair = DIAG.align_predictions(frames[0], frames[1], "_no_code")
            self.assertTrue(pair.job_id.equals(pair.job_id_no_code))
            for resource, expected_count in (("memory", 1936), ("vram", 1926)):
                block = pair.loc[pair.split.eq("test") & pair.resource.eq(resource) & pair.valid & pair.resource_applicable]
                metric = cap.capacity_metrics(block.target, block.prediction, DIAG, .10)
                self.assertEqual(metric["n"], expected_count)
                metric_rows.append({"seed": seed, "resource": resource, "day_blocks": 67, **metric})
        summary = cap.aggregate_seeds(pd.DataFrame(metric_rows), ["resource"], ["mae", "rmse"])
        self.assertTrue(summary.seed_count.eq(5).all())
        self.assertEqual(summary.set_index("resource").n.to_dict(), {"memory": 1936, "vram": 1926})


if __name__ == "__main__":
    unittest.main(verbosity=2)
