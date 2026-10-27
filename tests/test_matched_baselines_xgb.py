"""Small actual-XGBoost smoke test, when the cluster dependencies are installed."""
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

HAS_DEPS = all(importlib.util.find_spec(name) for name in
               ("numpy", "pandas", "yaml", "sklearn", "xgboost"))
try:
    # The local source directory named xgboost is a namespace, not the external
    # installed distribution; find_spec alone can falsely report it present.
    importlib.metadata.version("xgboost")
except importlib.metadata.PackageNotFoundError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "XGBoost smoke requires the baseline dependencies")
class MatchedXGBoostTests(unittest.TestCase):
    def test_fits_only_optimization_stops_only_on_tail_and_exports_outer_splits(self):
        import pandas as pd
        from test_matched_baselines_neural import fixture
        from baseline_study.xgb import train
        from qwen.base_selection import select_base_training_frames

        bundle, config = fixture()
        config["xgboost"] = {"n_estimators": 3, "early_stopping_rounds": 1,
                             "n_jobs": 1, "max_depth": 1}
        selection = select_base_training_frames(bundle, config)
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            train(config, bundle, selection, output)
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(set(predictions.sample_id), {"s10", "s11", "s12", "s13"})
            manifest = json.loads((output / "run_manifest.json").read_text())
            for resource in ("memory", "vram"):
                details = manifest["details"]["resources"][resource]
                self.assertEqual(details["optimization_valid_n"], 8)
                self.assertEqual(details["early_stopping_valid_n"], 1)
            features = (output / "feature_names.json").read_text()
            self.assertIn("fit_user", features)
            for excluded in ("purge_user", "tail_user", "outer_user", "test_user"):
                self.assertNotIn(excluded, features)
            self.assertFalse(manifest["details"]["outer_validation_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
