"""The primary policy must not depend on labels or classifier probabilities."""
import unittest

import numpy as np
import pandas as pd

from scripts.export_primary_exact_bypass import primary_predictions


def frozen_predictions():
    rows = []
    for split in ("validation", "test"):
        for resource in ("memory", "vram"):
            for job, history, exact, applicable in (("exact", True, True, True),
                                                     ("user", True, False, True),
                                                     ("cold", False, False, True),
                                                     ("na", True, True, False)):
                eligible = applicable and history and exact
                rows.append({"sample_id": f"{split}-{job}", "job_id": f"{split}-{job}", "split": split,
                             "resource": resource, "target": .4, "valid": applicable,
                             "prediction": .3 if eligible else .6, "model": "reliability_cascade_full", "seed": 42,
                             "resource_applicable": applicable, "history_available": history,
                             "exact_script_history_available": exact, "history_expert_prediction": .3,
                             "post_blend_prediction": .6, "history_bypass": eligible,
                             "reliability_threshold": 0., "reliability_probability": .1,
                             "embedding_cache_hit_job": True, "encoder_forward_required_job": False})
    return pd.DataFrame(rows)


def check_zero_threshold_full_is_equivalent_and_has_no_cache_claim():
    reference = frozen_predictions()
    untouched = reference.copy(deep=True)
    output, report = primary_predictions(reference, require_equivalence=True)
    pd.testing.assert_frame_equal(reference, untouched)
    assert report["all_reference_thresholds_zero"]
    assert report["predictions_equivalent"] and report["routes_equivalent"]
    np.testing.assert_allclose(output.prediction, reference.prediction)
    assert not output.pre_classifier_evaluated.any()
    assert not output.trust_region_applied.any()
    assert "encoder_forward_required_job" not in output
    assert "embedding_cache_hit_job" not in output


def check_policy_and_job_demand_do_not_read_targets_valid_masks_or_classifier_scores():
    reference = frozen_predictions()
    first, _ = primary_predictions(reference)
    changed = reference.copy()
    changed["target"] = .99
    changed["valid"] = ~changed.valid
    changed["reliability_probability"] = .99
    second, _ = primary_predictions(changed)
    for name in ("prediction", "history_bypass", "qwen_required_job", "qwen_required_resource"):
        pd.testing.assert_series_equal(first[name], second[name])
    # One inapplicable resource does not remove a job's other required resource.
    reference.loc[(reference.resource == "vram") & reference.sample_id.str.endswith("user"), "resource_applicable"] = False
    output, _ = primary_predictions(reference)
    assert output.loc[output.sample_id.str.endswith("user"), "qwen_required_job"].all()


def check_nonzero_reference_is_not_silently_relabelled_as_paper_equivalence():
    reference = frozen_predictions()
    mask = reference.sample_id.str.endswith("exact")
    reference.loc[mask, "reliability_threshold"] = .9
    reference.loc[mask, "history_bypass"] = False
    reference.loc[mask, "prediction"] = .6
    with unittest.TestCase().assertRaisesRegex(ValueError, "Paper-mode equivalence failed"):
        primary_predictions(reference, require_equivalence=True)
    output, audit = primary_predictions(reference, require_equivalence=False)
    assert not audit["predictions_equivalent"]
    assert not audit["routes_equivalent"]
    assert output.loc[mask, "prediction"].eq(.3).all()


def check_no_code_comparator_cannot_be_overwritten_by_primary_export():
    reference = frozen_predictions()
    reference["model"] = "reliability_cascade_no_code"
    with unittest.TestCase().assertRaisesRegex(ValueError, "not No-Code"):
        primary_predictions(reference)


def check_boolean_csv_strings_are_read_strictly():
    reference = frozen_predictions()
    reference["resource_applicable"] = reference.resource_applicable.map({True: "true", False: "false"})
    output, report = primary_predictions(reference, require_equivalence=True)
    assert report["routes_equivalent"]
    reference.loc[0, "resource_applicable"] = "possibly"
    with unittest.TestCase().assertRaisesRegex(ValueError, "invalid boolean values"):
        primary_predictions(reference)


class PrimaryExactBypassTests(unittest.TestCase):
    test_zero_threshold_equivalence = staticmethod(check_zero_threshold_full_is_equivalent_and_has_no_cache_claim)
    test_no_label_or_score_dependence = staticmethod(check_policy_and_job_demand_do_not_read_targets_valid_masks_or_classifier_scores)
    test_nonzero_reference_audited = staticmethod(check_nonzero_reference_is_not_silently_relabelled_as_paper_equivalence)
    test_no_code_unchanged = staticmethod(check_no_code_comparator_cannot_be_overwritten_by_primary_export)
    test_strict_booleans = staticmethod(check_boolean_csv_strings_are_read_strictly)


if __name__ == "__main__":
    unittest.main()
