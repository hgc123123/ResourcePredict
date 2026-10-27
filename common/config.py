from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_value(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def require_value(config: Dict[str, Any], dotted_key: str) -> Any:
    value = get_value(config, dotted_key)
    if value is None:
        raise ValueError(f"Missing required configuration value: {dotted_key}")
    return value


def set_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot override non-mapping key: {dotted_key}")
    target[keys[-1]] = value


def parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def load_config(path: Union[str, Path], overrides: Iterable[str] = ()) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a mapping")
    config = copy.deepcopy(loaded)
    for key, value in parse_overrides(overrides).items():
        set_value(config, key, value)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: Dict[str, Any], value: Union[str, Path]) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_root() / expanded).resolve()


def serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    return result


def resolved_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize known filesystem fields for a portable run-local archive."""
    result = serializable_config(config)
    for dotted_key in (
        "data.path",
        "split.manifest",
        "qwen.model_name_or_path",
        "outputs.root",
    ):
        value = get_value(result, dotted_key)
        if value is not None and str(value).strip():
            set_value(result, dotted_key, str(resolve_path(config, value)))
    return result


def write_resolved_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_serializable_config(config), handle, sort_keys=False)
