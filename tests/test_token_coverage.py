import argparse

import numpy as np
import pytest

from analysis.token_coverage import _tokenizer_source, summarize_token_coverage


def test_token_coverage_uses_script_budget_after_overhead():
    result = summarize_token_coverage(
        [5, 8, 9, 20],
        ["train", "train", "test", "test"],
        [10, 20],
        prompt_tokens=1,
        special_tokens=1,
    )

    overall_10 = result.loc[
        (result["scope"] == "overall") & (result["max_tokens"] == 10)
    ].iloc[0]
    assert overall_10["script_token_budget"] == 8
    assert overall_10["truncated_n"] == 2
    assert overall_10["truncated_rate"] == pytest.approx(0.5)

    test_20 = result.loc[
        (result["scope"] == "test") & (result["max_tokens"] == 20)
    ].iloc[0]
    assert test_20["n"] == 2
    assert test_20["truncated_n"] == 1


def test_token_coverage_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="aligned"):
        summarize_token_coverage([1, 2], ["train"], [10], prompt_tokens=0, special_tokens=1)
    with pytest.raises(ValueError, match="exceed"):
        summarize_token_coverage([1], ["train"], [2], prompt_tokens=1, special_tokens=1)
    with pytest.raises(ValueError, match="non-negative"):
        summarize_token_coverage(np.array([-1]), ["train"], [10], prompt_tokens=0, special_tokens=1)


def test_checkpoint_tokenizer_cannot_be_overridden():
    args = argparse.Namespace(checkpoint="checkpoint", tokenizer="different-tokenizer")
    with pytest.raises(ValueError, match="cannot override"):
        _tokenizer_source(args, {})
