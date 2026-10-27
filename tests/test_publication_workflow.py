"""The public workflow must keep experiment scopes and frozen evidence separate."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.run_publication_experiments import main
from publication_reporting.report import parse_args, verify_primary_exports


class PublicationWorkflowTests(unittest.TestCase):
    def test_plan_contains_all_experiment_families_without_writing(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "absent"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--stage", "all", "--dry-run", "--output-root", str(target)]), 0)
            text = output.getvalue()
            for token in ("run_matched_baselines.py", "run_followup.py", "run_capacity_analysis.py",
                          "online_study.suite", "report_publication_results.py", "--paper-mode",
                          "--num-processes 3", "--num-threads 4", "--primary-root"):
                self.assertIn(token, text)
            self.assertFalse(target.exists())
            for label in ("0600", "0800", "1000"):
                self.assertIn("prefix_" + label, text)

    def test_report_cannot_treat_missing_exports_as_primary(self):
        with TemporaryDirectory() as folder:
            args = parse_args(["--primary-root", folder])
            missing = []
            self.assertFalse(verify_primary_exports(args, {}, missing))
            self.assertEqual(len(missing), 5)

    def test_report_rejects_a_claim_of_nonzero_threshold_equivalence(self):
        with TemporaryDirectory() as folder:
            seed = Path(folder) / "seed_40"
            seed.mkdir()
            payload = {"schema": "primary-exact-history-export/v1",
                       "identity": {"seed": 40, "require_equivalence": True},
                       "equivalence": {"all_reference_thresholds_zero": False,
                                       "predictions_equivalent": True, "routes_equivalent": True}}
            (seed / "export_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unverified primary"):
                verify_primary_exports(parse_args(["--primary-root", folder]), {}, [])


if __name__ == "__main__":
    unittest.main()
