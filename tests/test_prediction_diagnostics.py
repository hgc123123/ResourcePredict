from __future__ import annotations

import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from analysis.prediction_diagnostics import (
    EMPTY_KEY, align_predictions, boolean, cluster_labels, enrich, frozen_operation,
    holm_adjust, lru_replay, make_jobs, metrics, named_rng, paired_bootstrap,
    parse_args, probability_diagnostics, read_predictions, read_run, risk_metrics, run_diagnostics,
    summarize, validate_population_index,
)


@contextmanager
def temporary_directory():
    # Some Windows file permissions prevent reopening tempfile's mode-0700 dirs.
    # A unique directory with inherited workspace ACLs remains portable.
    root = Path(__file__).resolve().parent/('_prediction_test_'+uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


class PairedDiagnosticsTests(unittest.TestCase):
    def test_metrics_and_signed_tail_are_in_correct_direction(self):
        result = metrics([.2,.8],[.3,.4])
        self.assertAlmostEqual(result["mae"],.25)
        self.assertAlmostEqual(result["mse"],.085)
        self.assertAlmostEqual(risk_metrics([.2,.8],[.3,.4])["mean_signed_error"],-.15)
        self.assertAlmostEqual(risk_metrics([.2,.8],[.3,.4])["mean_positive_shortfall_ratio"],.2)

    def test_pairing_rejects_changed_invalid_masks_targets_or_jobs(self):
        left = pd.DataFrame({"sample_id":["a","b"],"resource":["memory"]*2,
                             "valid":[True,False],"target":[.2,np.nan],"prediction":[.3,.5]})
        for column,value in (("valid",True),("target",.9),("sample_id","z")):
            right = left.copy()
            right.loc[1,column] = value
            with self.subTest(column=column),self.assertRaises(ValueError):
                align_predictions(left,right,"_r")
        aligned = align_predictions(left,left.iloc[::-1],"_r")
        self.assertEqual(len(aligned),2)

    def test_identical_paired_predictions_have_exact_zero_intervals(self):
        result = paired_bootstrap([.1,.8,.3],[.2,.5,.4],[.2,.5,.4],["a","a","b"],
            replicates=101,rng=named_rng(2026,"same"))
        for row in result:
            self.assertEqual(row["delta"],0)
            self.assertEqual(row["ci_lower"],0)
            self.assertEqual(row["ci_upper"],0)

    def test_cluster_bootstrap_weights_jobs_not_cluster_means(self):
        # Unequal blocks: full-sample difference is (1+1+1+0)/4, not .5.
        result = paired_bootstrap([0,0,0,0],[1,1,1,0],[0,0,0,0],["a","a","a","b"],
            replicates=301,rng=named_rng(1,"weighted"))
        self.assertAlmostEqual(result[0]["delta"],.75)
        self.assertEqual(result[0]["min_cluster_n"],1)
        self.assertEqual(result[0]["max_cluster_n"],3)
        self.assertEqual(result[0]["ci_lower"],0)
        self.assertEqual(result[0]["ci_upper"],1)

    def test_single_cluster_is_not_given_a_degenerate_significance_claim(self):
        result = paired_bootstrap([.5],[.5],[.9],["a"],replicates=10,rng=named_rng(1))
        self.assertEqual(result[0]["status"],"insufficient_clusters")
        self.assertFalse(result[0]["favorable_interval"])

    def test_named_rng_is_stable_to_unrelated_analysis_order(self):
        first = named_rng(2026,42,"vram","day").integers(100,size=10)
        named_rng(2026,"unrelated").integers(100,size=200)
        np.testing.assert_array_equal(first,named_rng(2026,42,"vram","day").integers(100,size=10))

    def test_holm_preserves_input_order_and_monotonicity(self):
        np.testing.assert_allclose(holm_adjust([.04,.001,.03]),[.06,.003,.06])

    def test_probability_ties_and_one_class(self):
        result,bins = probability_diagnostics([False,True],[.5,.5])
        self.assertEqual(result["auroc"],.5)
        self.assertEqual(result["average_precision"],.5)
        self.assertEqual(result["brier"],.25)
        self.assertEqual(sum(row["n"] for row in bins),2)
        result,_ = probability_diagnostics([True,True],[.8,.9])
        self.assertTrue(np.isnan(result["auroc"]))
        self.assertEqual(result["average_precision"],1)

    def test_script_cluster_keeps_missing_scripts_as_singletons(self):
        frame = pd.DataFrame({"sample_id":["a","b","c"],"script_sha256":[EMPTY_KEY,EMPTY_KEY,"x"]})
        self.assertEqual(len(set(cluster_labels(frame,"script"))),3)

    def test_summary_is_sample_sd_and_actual_seed_count(self):
        result = summarize(pd.DataFrame({"seed":[40,41],"group":["a"]*2,"mae":[1.,3.]}),["group"])
        self.assertEqual(result.mae_mean.iloc[0],2)
        self.assertAlmostEqual(result.mae_sd.iloc[0],np.sqrt(2))
        self.assertEqual(result.seed_n.iloc[0],2)

    def test_bool_parser_does_not_treat_false_strings_as_truthy(self):
        np.testing.assert_array_equal(boolean(pd.Series(["False","True","0","1"]),"x"),[False,True,False,True])
        with self.assertRaises(ValueError):
            boolean(pd.Series(["unknown"]),"x")


class CacheCounterfactualTests(unittest.TestCase):
    def jobs(self):
        return pd.DataFrame({"sample_id":["a","b","c","d","e","f","g"],
            "submit_time":pd.to_datetime(["2026-01-01","2026-01-01","2026-01-02","2026-01-03","2026-01-04","2026-01-05","2026-01-06"],utc=True),
            "script_sha256":["x","x","x","y","x",EMPTY_KEY,EMPTY_KEY],
            "required":[True,True,True,False,True,True,True],"split":["test"]*7})

    def test_equal_timestamp_jobs_neither_share_nor_cache_empty_scripts(self):
        hits,forward = lru_replay(self.jobs(),None)
        np.testing.assert_array_equal(hits,[False,False,True,False,True,False,False])
        np.testing.assert_array_equal(forward,[True,True,False,False,False,True,True])

    def test_bypassed_jobs_do_not_populate_or_evict_cache(self):
        hits,_ = lru_replay(self.jobs(),1)
        self.assertTrue(hits[4])
        zero,forward = lru_replay(self.jobs(),0)
        self.assertFalse(zero.any())
        np.testing.assert_array_equal(forward,self.jobs().required)

    def test_lru_eviction_is_actual_recency(self):
        jobs = self.jobs().iloc[:5].copy()
        jobs["submit_time"] = pd.date_range("2026-01-01",periods=5,tz="UTC")
        jobs["script_sha256"] = ["a","b","a","c","b"]
        jobs["required"] = True
        hits,_ = lru_replay(jobs,2)
        np.testing.assert_array_equal(hits,[False,False,True,False,False])

    def test_frozen_operations_change_only_requested_operation(self):
        frame = pd.DataFrame({"history_bypass":[False,True],"history_available":[True,True],
            "exact_script_history_available":[True,True],"resource_applicable":[True,True],
            "reliability_probability":[.1,.9],"qwen_prediction":[.2,.2],
            "history_expert_prediction":[.8,.8],"post_blend_prediction":[.7,.7],"post_history_weight":[.5,.5]})
        np.testing.assert_allclose(frozen_operation(frame,"full")[0],[.7,.8])
        np.testing.assert_allclose(frozen_operation(frame,"no_post")[0],[.2,.8])
        np.testing.assert_allclose(frozen_operation(frame,"no_trust")[0],[.5,.8])
        p,required = frozen_operation(frame,"exact_bypass")
        np.testing.assert_allclose(p,[.8,.8])
        self.assertFalse(required.any())


def synthetic_run(root: Path,variant: str,seed: int) -> None:
    """Small diagnostic fixture, not a training/model implementation."""
    rows = []
    for i in range(11):
        split = "train" if i<3 else "validation" if i<7 else "test"
        for resource in ("memory","vram"):
            rows.append({"sample_id":f"{i:03d}","resource":resource,"split":split,"seed":seed,
                "user_id":"u" if i%3 else "v","target":.55,"valid":True,"prediction":.4,
                "submit_time":pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(days=i),
                "completion_time":pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(days=i,hours=1),
                "gpu_count":1,"script_sha256":("a" if i%2 else "b")*64,"user_known":True})
    base = pd.DataFrame(rows)
    manifest = {"data_sha256":"d"*64,"split_fingerprint":"s"*64,"details":{}}
    directory = root/f"qwen/tokens_4096/{variant}/seed_{seed}"
    directory.mkdir(parents=True)
    base.to_csv(directory/"predictions.csv",index=False)
    (directory/"run_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    cascade = base.loc[base.split.ne("train")].drop(columns=["script_sha256","user_known"]).copy().reset_index(drop=True)
    for name,value in {"history_available":True,"exact_script_history_available":True,"same_last1":True,
            "resource_applicable":True,"history_bypass":False,"qwen_prediction":.4,"last1_prediction":.6,
            "hierarchical_reference_prediction":.6,"history_expert_prediction":.6,
            "reliability_probability":.4,"post_history_weight":.5,"post_blend_prediction":.5 if variant=="full" else .48}.items():
        cascade[name] = value
    cascade["prediction"] = cascade.post_blend_prediction
    enriched = cascade.merge(base[["sample_id","resource","script_sha256"]],on=["sample_id","resource"])
    jobs = make_jobs(enriched,np.ones(len(enriched),bool))
    hit,forward = lru_replay(jobs,None)
    for name,values in (("embedding_cache_hit_job",hit),("encoder_forward_required_job",forward)):
        cascade[name] = cascade.sample_id.map(dict(zip(jobs.sample_id,values)))
    cascade["qwen_required_job"] = True
    cascade["qwen_required_resource"] = True
    directory = root/f"reliability_cascade/tokens_4096/{variant}/seed_{seed}"
    (directory/"checkpoint").mkdir(parents=True)
    cascade.to_csv(directory/"predictions.csv",index=False)
    manifest["details"] = {"calibration_protocol":{"selection_start_submit_time":"2026-01-06T00:00:00+00:00"}}
    (directory/"run_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    (directory/"checkpoint/checkpoint_contract.json").write_text(json.dumps({"cascade_model_config":{"reliability_error_cap":.1,"require_exact_script_for_bypass":True}}),encoding="utf-8")


def index_from_base(base: pd.DataFrame) -> pd.DataFrame:
    population = base.loc[base.resource.eq("memory"),["sample_id","user_id","script_sha256","split","gpu_count","submit_time","completion_time"]].copy()
    for resource in ("memory","vram"):
        part = base.loc[base.resource.eq(resource)].set_index("sample_id")
        population["valid_"+resource] = population.sample_id.map(part.valid)
    population["effective_analysis_row"] = True
    return population


class EndToEndDiagnosticsTests(unittest.TestCase):
    def test_two_variant_two_seed_diagnostics_do_not_export_identifiers(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            for seed in (40,41):
                for variant in ("full","no_code"):
                    synthetic_run(root,variant,seed)
            # Actual qwen.evaluate output contains validation/test rows only.
            first = read_predictions(root/"qwen/tokens_4096/full/seed_40/predictions.csv",40)
            population = index_from_base(first)
            index_path = root/"population_index.csv"
            population.to_csv(index_path,index=False)
            for path in root.glob("qwen/tokens_4096/*/seed_*/predictions.csv"):
                data = pd.read_csv(path,dtype={"sample_id":str})
                data.loc[data.split.ne("train")].to_csv(path,index=False)
            out = root/"diagnostics"
            args = parse_args(["--base-root",str(root),"--cascade-root",str(root),"--output-dir",str(out),
                "--population-index",str(index_path),"--seeds","40","41","--bootstrap-replicates","20","--thresholds","--cache-capacities","1","--allow-unsealed"])
            manifest = run_diagnostics(args)
            self.assertEqual(manifest["seeds"],[40,41])
            summary = pd.read_csv(out/"test_metrics_summary.csv")
            self.assertTrue(summary.seed_n.eq(2).all())
            shifts = pd.read_csv(out/"population_shift_per_seed.csv")
            self.assertTrue(((shifts.scope=="retained_split") & (shifts.group=="train")).any())
            self.assertEqual(manifest["population_index"]["retained_event_n"],11)
            for path in out.glob("*.csv"):
                columns = pd.read_csv(path).columns
                self.assertFalse({"sample_id","user_id","script_sha256","job_id"}&set(columns),path.name)
            with self.assertRaises(FileExistsError):
                run_diagnostics(args)

    def test_new_cascade_requires_integrity_seal(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            synthetic_run(root,"full",40)
            with self.assertRaises(ValueError):
                read_run(root/"reliability_cascade/tokens_4096/full/seed_40",40,sealed=True)

    def test_val_test_only_base_omits_unavailable_novelty(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            synthetic_run(root,"full",40)
            base = read_predictions(root/"qwen/tokens_4096/full/seed_40/predictions.csv",40)
            cascade = read_predictions(root/"reliability_cascade/tokens_4096/full/seed_40/predictions.csv",40)
            result = enrich(cascade,base.loc[base.split.ne("train")],{})
            self.assertNotIn("user_outer_train_status",result)
            self.assertNotIn("script_training_status",result)
            self.assertIn("novelty_skipped_reason",result.attrs)

    def test_index_restores_training_novelty_and_checks_identity(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            synthetic_run(root,"full",40)
            all_base = read_predictions(root/"qwen/tokens_4096/full/seed_40/predictions.csv",40)
            base = all_base.loc[all_base.split.ne("train")].copy()
            cascade = read_predictions(root/"reliability_cascade/tokens_4096/full/seed_40/predictions.csv",40)
            population = index_from_base(all_base)
            last = base.sample_id.max()
            base.loc[base.sample_id.eq(last),"user_id"] = "new_user"
            cascade.loc[cascade.sample_id.eq(last),"user_id"] = "new_user"
            population.loc[population.sample_id.eq(last),"user_id"] = "new_user"
            manifest = {"data_sha256":"d"*64,"split_fingerprint":"s"*64,"details":{"base_training_selection":{
                "source_data_sha256":"d"*64,"source_split_fingerprint":"s"*64,
                "optimization_sample_ids":["000"],"reserved_train_tail_sample_ids":["001","002"],"purged_optimization_sample_ids":[]}}}
            validate_population_index(population,base,manifest)
            result = enrich(cascade,base,manifest,population)
            self.assertTrue(result.loc[result.sample_id.ne(last),"user_outer_train_status"].eq("seen_outer_train").all())
            self.assertTrue(result.loc[result.sample_id.eq(last),"user_outer_train_status"].eq("new_since_outer_train").all())
            self.assertTrue(result.loc[result.script_sha256.eq("b"*64),"script_training_status"].eq("seen_optimization").all())
            self.assertTrue(result.loc[result.script_sha256.eq("a"*64),"script_training_status"].eq("unseen_optimization").all())
            for name,value in (("user_id","wrong"),("script_sha256","f"*64),("valid_memory",False)):
                wrong = population.copy()
                wrong.loc[wrong.sample_id.eq(last),name] = value
                with self.subTest(field=name),self.assertRaises(ValueError):
                    validate_population_index(wrong,base,manifest)
            wrong_manifest = {**manifest,"data_sha256":"wrong"}
            with self.assertRaises(ValueError):
                validate_population_index(population,base,wrong_manifest)


if __name__=="__main__":
    unittest.main()
