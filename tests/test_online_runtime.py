from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from common.data import DatasetBundle
from reliability_cascade.features import build_cascade_features
from reliability_cascade.history import build_robust_history_long
from reliability_cascade.online_runtime import ExactLRU, OnlineHistory, post_fuse, pre_route


def fixture():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame({
        "sample_id": list("abcdefgh"), "source_row": list(range(8)),
        "job_id": list("abcdefgh"), "user_id": ["u"] * 7 + ["new"],
        "split": ["train"] * 3 + ["validation"] * 2 + ["test"] * 3,
        "submit_time": [start + pd.Timedelta(hours=i) for i in [0, 2, 4, 6, 8, 10, 10, 12]],
        "completion_time": [start + pd.Timedelta(hours=i) for i in [2, 3, 5, 7, 9, 11, 11, 13]],
        "code": ["echo a", "echo a", "echo b", "echo a", "echo a", "", "", "echo a"],
        "target_memory": [0.1, 0.3, 0.7, 0.4, 0.5, 0.6, 0.8, 0.9],
        "target_vram": [0.2, 0.4, 0.6, 0.5, 0.6, 0.7, np.nan, 0.8],
        "valid_memory": [True] * 8, "valid_vram": [True] * 6 + [False, True],
        "gpu_count": [1] * 6 + [0, 1],
        "num__cpu": [1, 2, 3, 4, 5, 6, 6, 8], "num__gpu": [1] * 6 + [0, 1],
    })
    effective = frame.loc[frame.sample_id.ne("d")].copy()  # boundary-purged event
    return DatasetBundle(effective, Path("unused.csv"), "synthetic", None,
                         ["num__cpu", "num__gpu"], [],
                         {"verified_chronological": True, "has_submission_time": True,
                          "has_completion_time": True}, frame)


def paired(bundle):
    history = build_robust_history_long(bundle)
    rows = []
    for item in history.itertuples(index=False):
        rows.append({"sample_id": item.sample_id, "resource": item.resource,
                     "valid": True, "qwen_prediction": 0.5,
                     "last1_prediction": item.user_last_value if item.user_available else 0.2,
                     "history_available": item.user_available,
                     "history_source_sample_id": item.user_source_sample_id,
                     "gate_code": 0.4, "gate_user": 0.3, "gate_metadata": 0.3,
                     "user_known": True, "script_sha256": item.script_sha256})
    output = pd.DataFrame(rows)
    # CPU-only VRAM labels are invalid, as in the production protocol.
    output.loc[output.sample_id.eq("g") & output.resource.eq("vram"), "valid"] = False
    return output


class IdentityImputer:
    def transform(self, values):
        return np.asarray(values)


class Probability:
    def predict_proba(self, values):
        return np.tile([0.2, 0.8], (len(values), 1))


class Weight:
    def predict(self, values):
        return np.full(len(values), 0.5)


class OnlineRuntimeTests(unittest.TestCase):
    def test_online_features_equal_whole_trace_replay_including_purged_source(self):
        bundle = fixture()
        offline = build_cascade_features(paired(bundle), bundle)
        online = OnlineHistory(bundle)
        for _, row in bundle.frame.iterrows():
            online.activate_before(pd.Timestamp(row.submit_time).value)
            current = online.query(row)
            self.assertEqual(current.pre_names, offline.pre_names)
            selected = offline.frame.sample_id.eq(row.sample_id)
            np.testing.assert_allclose(current.pre_values, offline.pre_values[selected], rtol=0, atol=1e-12)
        # Equal completion time is never visible to the next request.
        online = OnlineHistory(bundle)
        row_b = bundle.frame.loc[bundle.frame.sample_id.eq("b")].iloc[0]
        self.assertEqual(online.activate_before(pd.Timestamp(row_b.submit_time).value), 0)
        self.assertFalse(online.query(row_b).frame.iloc[0].user_available)

    def test_query_requires_current_activation_and_disallows_time_reversal(self):
        bundle = fixture()
        online = OnlineHistory(bundle)
        row = bundle.frame.iloc[2]
        with self.assertRaises(RuntimeError):
            online.query(row)
        online.activate_before(pd.Timestamp(row.submit_time).value)
        with self.assertRaises(ValueError):
            online.activate_before(pd.Timestamp(bundle.frame.iloc[0].submit_time).value)

    def test_future_labels_do_not_change_pre_route_features(self):
        bundle = fixture()
        row = bundle.frame.loc[bundle.frame.sample_id.eq("e")].iloc[0]
        first = OnlineHistory(bundle)
        first.activate_before(pd.Timestamp(row.submit_time).value)
        expected = first.query(row).pre_values
        changed = fixture()
        future = changed.retained_frame.completion_time.ge(row.submit_time)
        changed.retained_frame.loc[future, ["target_memory", "target_vram"]] = 0.99
        second = OnlineHistory(changed)
        second.activate_before(pd.Timestamp(row.submit_time).value)
        np.testing.assert_array_equal(second.query(row).pre_values, expected)

    def test_lru_delays_insert_and_evicts_the_least_recent_entry(self):
        cache = ExactLRU(2, "checkpoint-v1")
        cache.advance(1)
        cache.put_after_group("a", 10)
        self.assertIsNone(cache.get("a"))
        cache.advance(2)
        self.assertEqual(cache.get("a"), 10)
        cache.put_after_group("b", 20)
        cache.advance(3)
        self.assertEqual(cache.get("a"), 10)  # now b is oldest
        cache.put_after_group("c", 30)
        cache.advance(4)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 10)
        self.assertEqual(cache.get("c"), 30)
        self.assertEqual(cache.evictions, 1)
        cache.put_after_group("blank", 99, observed=False)
        cache.advance(5)
        self.assertIsNone(cache.get("blank"))

    def test_zero_capacity_never_reuses_and_nonchronological_cache_fails(self):
        cache = ExactLRU(0, "namespace")
        cache.advance(5)
        cache.put_after_group("a", 10)
        cache.advance(6)
        self.assertIsNone(cache.get("a"))
        with self.assertRaises(ValueError):
            cache.advance(4)
        with self.assertRaises(ValueError):
            ExactLRU(-1, "namespace")

    def test_live_pre_route_and_post_trust_have_expected_values(self):
        bundle = fixture()
        offline = build_cascade_features(paired(bundle), bundle)
        online = OnlineHistory(bundle)
        row = bundle.frame.loc[bundle.frame.sample_id.eq("e")].iloc[0]
        online.activate_before(pd.Timestamp(row.submit_time).value)
        features = online.query(row)
        from reliability_cascade.features import post_features
        _, post_names, _ = post_features(offline, "hierarchical_last")
        model = SimpleNamespace(
            resource="memory", candidate_name="hierarchical_last", threshold=0.75,
            pre_names=features.pre_names, post_names=post_names,
            config=SimpleNamespace(require_exact_script_for_bypass=True,
                                   post_blend_enabled=True, trust_region_enabled=True,
                                   trust_region_base=0.025, trust_region_mad_scale=2.0,
                                   trust_region_max=0.1),
            pre_imputer=IdentityImputer(), post_imputer=IdentityImputer(),
            pre_model=Probability(), post_model=Weight(),
        )
        route = pre_route(model, features, 0)
        self.assertTrue(route["bypass"])
        self.assertAlmostEqual(route["probability"], 0.8)
        self.assertAlmostEqual(route["history_prediction"], 0.4)  # purged source d
        # mean(.9,.4)=.65; exact-history trust radius clips to .5.
        result = post_fuse(model, features, 0, np.array([0.9, 0.8]), np.array([0.4, 0.3, 0.3]), True)
        self.assertAlmostEqual(result, 0.5)
        model.config.trust_region_enabled = False
        self.assertAlmostEqual(post_fuse(model, features, 0, np.array([0.9, 0.8]), np.array([0.4, 0.3, 0.3]), True), 0.65)


if __name__ == "__main__":
    unittest.main()
