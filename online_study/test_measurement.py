from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from online_study.measurement import cache_parity_check, frozen_reference, parity_check


def reference():
    return pd.DataFrame({
        "sample_id": ["a", "a", "b", "b"], "resource": ["memory", "vram"] * 2,
        "split": ["test"] * 4, "valid": [True, True, True, False],
        "resource_applicable": [True, True, True, False],
        "history_available": [True, True, False, False],
        "exact_script_history_available": [True, True, False, False],
        "history_bypass": [True, False, False, False],
        "qwen_prediction": [.9, .8, .3, .4], "history_expert_prediction": [.2, .1, np.nan, np.nan],
        "post_history_weight": [.5, .5, 0., 0.], "post_blend_prediction": [.3, .2, .3, .4],
    })


def live():
    return pd.DataFrame({"sample_id": ["a", "b"], "prediction_memory": [.2, .3],
        "prediction_vram": [.2, np.nan], "bypass_memory": [True, False],
        "bypass_vram": [False, False], "applicable_memory": [True, True],
        "applicable_vram": [True, False]})


class MeasurementTests(unittest.TestCase):
    def test_frozen_controls_preserve_exact_intended_operations(self):
        expected = {"full": [.2, .2, .3, .4], "no_bypass": [.3, .2, .3, .4],
                    "exact_bypass": [.2, .1, .3, .4], "no_post": [.2, .8, .3, .4],
                    "no_trust": [.2, .45, .3, .4]}
        for operation, values in expected.items():
            np.testing.assert_allclose(frozen_reference(reference(), operation).prediction, values)

    def test_parity_checks_applicability_independently_of_finite_prediction(self):
        ref = frozen_reference(reference(), "full")
        self.assertTrue(parity_check(live(), ref, 1e-12)["passed"])
        broken = live()
        broken.loc[0, "prediction_vram"] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            parity_check(broken, ref, .005)
        # Missing labels on a GPU job never exempt the predicted value.
        ref.loc[1, "valid"] = False
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            parity_check(broken, ref, .005)

    def test_route_missing_duplicate_and_numeric_failures_are_not_silently_accepted(self):
        ref = frozen_reference(reference(), "full")
        broken = live()
        broken.loc[0, "bypass_memory"] = False
        with self.assertRaisesRegex(ValueError, "parity failed"):
            parity_check(broken, ref, .005)
        with self.assertRaisesRegex(ValueError, "lacks measured"):
            parity_check(live(), ref.iloc[1:], .005)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parity_check(pd.concat([live(), live()]), ref, .005)
        with self.assertRaisesRegex(ValueError, "parity failed"):
            parity_check(live().assign(prediction_memory=[.25, .3]), ref, .005)

    def test_same_timestamp_a_b_a_uses_last_encounter_order(self):
        events = pd.DataFrame({"sample_id": list("abcdef"), "submit_time": [1,1,1,2,3,4],
            "identity": ["A", "B", "A", "C", "A", "B"], "observed": [True] * 6,
            "model_path_required": [True] * 6, "cache_hit": [False,False,False,False,True,False],
            "encoder_forward": [True,True,True,True,False,True]})
        ref = pd.DataFrame({"sample_id": list("abcdef"), "required_resource": [True] * 6})
        self.assertTrue(cache_parity_check(events, ref, 2)["passed"])
        broken = events.copy()
        broken.loc[4, ["cache_hit", "encoder_forward"]] = [False, True]
        with self.assertRaisesRegex(ValueError, "cache parity failed"):
            cache_parity_check(broken, ref, 2)


if __name__ == "__main__":
    unittest.main()
