from pathlib import Path
import unittest
import pandas as pd
from scripts.export_anonymized_predictions import deidentify


class ReleaseTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({"sample_id":["private-job"]*2,"user_id":["private-user"]*2,
            "script_sha256":["a"*64]*2,"resource":["memory","vram"],"prediction":[.2,.4],
            "submit_time":["2026-01-02T08:00:00Z"]*2,"completion_time":["2026-01-02T10:00:00Z"]*2,
            "workdir":["/private/path"]*2,"code":["secret source"]*2})

    def test_ids_stable_and_raw_identifiers_paths_scripts_absent(self):
        result=deidentify(self.fixture(),b"x"*32,pd.Timestamp("2026-01-01",tz="UTC"))
        self.assertEqual(result.sample_id.nunique(),1)
        self.assertNotIn("private-job",result.to_csv(index=False))
        self.assertNotIn("workdir",result)
        self.assertNotIn("code",result)
        self.assertNotIn("submit_time",result)
        self.assertEqual(result.submit_time_relative_seconds.iloc[0],115200)

    def test_key_change_changes_id_but_not_predictions_or_time_intervals(self):
        anchor=pd.Timestamp("2026-01-01",tz="UTC")
        a=deidentify(self.fixture(),b"a"*32,anchor)
        b=deidentify(self.fixture(),b"b"*32,anchor)
        self.assertNotEqual(a.sample_id.iloc[0],b.sample_id.iloc[0])
        self.assertTrue(a.prediction.equals(b.prediction))
        self.assertTrue((a.completion_time_relative_seconds-a.submit_time_relative_seconds).eq(7200).all())

    def test_short_key_is_rejected(self):
        with self.assertRaises(ValueError):
            deidentify(self.fixture(),b"weak",pd.Timestamp("2026-01-01",tz="UTC"))
