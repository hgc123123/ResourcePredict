from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from online_study.measurement import parity_check
from online_study.offline_reference import archived_base_jobs, base_parity, inject_live_inputs


def base():
    return pd.DataFrame({"sample_id": ["a", "b"], "qwen_memory": [.49, .2], "qwen_vram": [.3, .4],
        "gate_code": [.8, .8], "gate_user": [.1, .1], "gate_metadata": [.1, .1], "user_known": [True, False]})


class OfflineReferenceTests(unittest.TestCase):
    def test_base_parity_catches_corrupt_embeddings_even_if_final_prediction_is_similar(self):
        measured = base().assign(model_path_required=True)
        self.assertTrue(base_parity(measured, base())["passed"])
        measured.loc[0, "qwen_memory"] += .0001
        with self.assertRaisesRegex(ValueError, "base/embedding parity failed"):
            base_parity(measured, base())

    def test_inject_live_inputs_never_uses_missing_outputs_from_bypassed_job(self):
        measured = base().assign(model_path_required=[True, False])
        measured.loc[0, "qwen_memory"] = .4901
        measured.loc[1, "qwen_memory"] = np.nan
        actual = inject_live_inputs(base(), measured)
        np.testing.assert_allclose(actual.qwen_memory, [.4901, .2])

    def test_archived_base_resource_alignment_is_explicit(self):
        rows = []
        for item in base().itertuples(index=False):
            for resource in ("memory", "vram"):
                rows.append({"sample_id": item.sample_id, "resource": resource,
                    "prediction": getattr(item, "qwen_"+resource), "gate_code": item.gate_code,
                    "gate_user": item.gate_user, "gate_metadata": item.gate_metadata, "user_known": item.user_known})
        pd.testing.assert_frame_equal(archived_base_jobs(pd.DataFrame(rows)), base())
        with self.assertRaisesRegex(ValueError, "lack a resource"):
            archived_base_jobs(pd.DataFrame(rows).iloc[:-1])

    def test_tree_threshold_amplification_requires_a_matched_numerical_reference(self):
        # A representative tree split: a tiny Q change can switch the mixture.
        history = .9
        q_archived, q_live = .4999, .5001
        blend = lambda q: history if q >= .5 else q
        archived = pd.DataFrame({"sample_id": ["a", "a"], "resource": ["memory", "vram"],
            "split": ["test", "test"], "valid": [True, True], "resource_applicable": [True, True],
            "history_bypass": [False, False], "prediction": [blend(q_archived)] * 2})
        measured = pd.DataFrame({"sample_id": ["a"], "prediction_memory": [blend(q_live)],
            "prediction_vram": [blend(q_live)], "bypass_memory": [False], "bypass_vram": [False],
            "applicable_memory": [True], "applicable_vram": [True]})
        with self.assertRaisesRegex(ValueError, "parity failed"):
            parity_check(measured, archived, .005)
        matched = archived.assign(prediction=blend(q_live))
        self.assertTrue(parity_check(measured, matched, 1e-10)["passed"])
        # The matched reference still rejects an actual wrong final blend.
        measured.loc[0, "prediction_memory"] -= .01
        with self.assertRaisesRegex(ValueError, "parity failed"):
            parity_check(measured, matched, .005)


if __name__ == "__main__":
    unittest.main()
