from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HAS_DATA_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "yaml"))
if HAS_DATA_DEPS:
    import pandas as pd
    from baseline_study.neural import fit_preprocessors
    from common.data import DatasetBundle
    from qwen.base_selection import select_base_training_frames


def fixture():
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(14)],
        "job_id": [f"j{i}" for i in range(14)],
        "split": ["train"] * 10 + ["validation"] * 2 + ["test"] * 2,
        "submit_time": [origin + pd.Timedelta(hours=i) for i in list(range(10)) + [20, 21, 30, 31]],
        "code": ["fitword shared"] * 8 + ["purgeword", "tailword", "outerword", "outerword", "testword", "testword"],
        "user_id": ["fit_user"] * 8 + ["purge_user", "tail_user", "outer_user", "outer_user", "test_user", "test_user"],
        "cpu_request": list(range(8)) + [1_000_000] * 6,
        "valid_memory": [True] * 14, "valid_vram": [True] * 14,
        "target_memory": [0.2] * 14, "target_vram": [0.3] * 14,
    })
    frame["completion_time"] = frame.submit_time + pd.Timedelta(minutes=30)
    frame.loc[8, "completion_time"] = origin + pd.Timedelta(hours=9)
    bundle = DatasetBundle(frame, Path("private.csv"), "a" * 64, None,
                           ["cpu_request"], [], {"verified_chronological": True}, frame.copy())
    config = {"seed": 40, "qwen": {"base_selection": {"mode": "train_tail", "fit_fraction": .9}},
              "text": {"min_frequency": 1, "max_vocabulary": 100, "max_tokens": 16, "embedding_dim": 4},
              "training": {"device": "cpu", "batch_size": 4, "epochs": 1, "patience": 1, "num_workers": 0},
              "textcnn": {"kernel_sizes": [2, 3], "filters_per_kernel": 2, "user_embedding_dim": 2,
                          "metadata_hidden_dim": 2, "hidden_dim": 4, "dropout": 0},
              "lstm": {"hidden_dim": 2, "user_embedding_dim": 2, "metadata_hidden_dim": 2,
                       "head_hidden_dim": 4, "dropout": 0},
              "evaluation": {"acc_tolerance": .1, "clip_predictions": True}}
    return bundle, config


@unittest.skipUnless(HAS_DATA_DEPS, "Data isolation tests require numpy, pandas and PyYAML")
class MatchedNeuralTests(unittest.TestCase):
    def test_preprocessing_excludes_tail_purge_outer_validation_and_test(self):
        bundle, config = fixture()
        selection = select_base_training_frames(bundle, config)
        vocabulary, users, scaler = fit_preprocessors(config, selection.optimization, bundle.numeric_features)
        self.assertEqual(set(vocabulary.token_to_id), {"<PAD>", "<UNK>", "fitword", "shared"})
        self.assertEqual(set(users.user_to_id), {"<UNK>", "fit_user"})
        self.assertEqual(scaler.means, [3.5])
        self.assertEqual(selection.early_stopping.sample_id.tolist(), ["s9"])
        outer_validation = set(bundle.split("validation").sample_id)
        self.assertTrue(outer_validation.isdisjoint(selection.optimization.sample_id))
        self.assertTrue(outer_validation.isdisjoint(selection.early_stopping.sample_id))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU training smoke requires torch")
    def test_tiny_cpu_training_exports_original_outer_validation_and_test(self):
        import common.neural as utilities
        from baseline_study.neural import train

        for name in ("textcnn", "lstm"):
            with self.subTest(model=name), TemporaryDirectory() as temporary:
                tmp_path = Path(temporary)
                bundle, config = fixture()
                original = deepcopy(bundle.frame)
                config["baseline_study"] = {"model": name}
                selection = select_base_training_frames(bundle, config)
                stopping_sizes = []
                original_metric = utilities.validation_mean_mae

                def metric(predicted, targets, masks):
                    stopping_sizes.append(len(targets))
                    return original_metric(predicted, targets, masks)

                with patch.object(utilities, "validation_mean_mae", metric):
                    train(config, bundle, selection, tmp_path)
                self.assertEqual(stopping_sizes, [1])
                prediction = pd.read_csv(tmp_path / "predictions.csv")
                self.assertEqual(set(prediction.sample_id), {"s10", "s11", "s12", "s13"})
                self.assertEqual(len(prediction), 8)
                self.assertEqual(prediction.groupby("split").size().to_dict(), {"test": 4, "validation": 4})
                manifest = json.loads((tmp_path / "run_manifest.json").read_text())
                self.assertEqual(manifest["details"]["base_training_selection"], selection.manifest)
                self.assertFalse(manifest["details"]["outer_validation_used_for_selection"])
                pd.testing.assert_frame_equal(bundle.frame, original)


if __name__ == "__main__":
    unittest.main()
