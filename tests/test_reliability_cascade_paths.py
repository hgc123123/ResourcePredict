from pathlib import Path

import pytest

from reliability_cascade.integrity import validate_variant
from reliability_cascade.run import _output_dir
from scripts.run_reliability_cascade_matrix import main as matrix_main


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        " full",
        "full ",
        "full.",
        "a/b",
        r"a\b",
        "name:part",
        "CON",
        "r\u00e9sum\u00e9",
    ],
)
def test_variant_rejects_unsafe_path_components(value):
    with pytest.raises(ValueError, match="variant"):
        validate_variant(value)


@pytest.mark.parametrize("value", ["full", "no_code", "concat-fusion", "v1.5"])
def test_variant_accepts_portable_path_components(value):
    assert validate_variant(value) == value


def test_direct_run_output_rejects_parent_variant(tmp_path):
    config = {
        "_config_dir": str(tmp_path),
        "seed": 42,
        "qwen": {"max_tokens": 4096},
        "reliability_cascade": {"qwen_variant": ".."},
    }
    with pytest.raises(ValueError, match="variant"):
        _output_dir(config, None)


def test_matrix_dry_run_rejects_parent_variant(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        matrix_main(
            [
                "--config",
                str(config),
                "--component-root",
                str(tmp_path / "components"),
                "--output-root",
                str(tmp_path / "outputs"),
                "--seeds",
                "42",
                "--variant",
                "..",
                "--dry-run",
            ]
        )
