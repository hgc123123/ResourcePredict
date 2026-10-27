from types import SimpleNamespace

import pandas as pd

from qwen.cache_replay import replay_exact_cache


def test_cache_replay_includes_boundary_purged_script_arrival():
    retained = pd.DataFrame(
        {
            "sample_id": ["train-kept", "train-purged", "test-kept"],
            "split": ["train", "train", "test"],
            "submit_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:05Z",
                    "2026-01-01T00:10Z",
                ],
                utc=True,
            ),
            "code": ["first-script", "reused-script", "reused-script"],
        }
    )
    effective = retained.loc[
        retained["sample_id"].ne("train-purged")
    ].reset_index(drop=True)
    bundle = SimpleNamespace(frame=effective, retained_frame=retained)

    counts, seen, replay_rows = replay_exact_cache(bundle)

    assert len(effective) == 2
    assert replay_rows == 3
    assert counts["overall"] == {"lookups": 3, "hits": 1, "misses": 2}
    assert counts["test"] == {"lookups": 1, "hits": 1, "misses": 0}
    assert len(seen) == 2
