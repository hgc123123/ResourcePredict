from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import get_value, require_value, resolve_path
from .protocols import (
    BOUNDARY_COMPLETION_PROTOCOL_VERSION,
    normalize_boundary_completion_policy,
)


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    data_path: Path
    data_sha256: str
    manifest_path: Optional[Path]
    numeric_features: List[str]
    categorical_features: List[str]
    protocol: Dict[str, Any]
    # Full event stream for causal history/cache replay, never supervised fitting.
    retained_frame: Optional[pd.DataFrame] = None

    def split(self, name: str) -> pd.DataFrame:
        part = self.frame.loc[self.frame["split"] == name].copy()
        if part.empty:
            raise ValueError(f"Split is empty: {name}")
        return part.reset_index(drop=True)


def _read_csv_strict(
    path: Path,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate logical CSV records before pandas parses multiline scripts."""
    encoding = str(get_value(config, "data.csv_encoding", "utf-8-sig"))
    raw_field_limit = get_value(
        config, "data.csv_field_size_limit_bytes", 64 * 1024 * 1024
    )
    try:
        field_size_limit = int(raw_field_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("data.csv_field_size_limit_bytes must be a positive integer") from exc
    if field_size_limit <= 0:
        raise ValueError("data.csv_field_size_limit_bytes must be a positive integer")
    expected_columns = get_value(config, "data.expected_columns")
    if expected_columns is not None:
        if not isinstance(expected_columns, list) or not all(
            isinstance(column, str) and column for column in expected_columns
        ):
            raise ValueError("data.expected_columns must be a list of non-empty names")

    logical_rows = 0
    previous_field_limit = csv.field_size_limit()
    csv.field_size_limit(field_size_limit)
    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("Dataset CSV has no header") from exc
            if not header or any(not name for name in header):
                raise ValueError("Dataset CSV header contains an empty column name")
            duplicates = sorted({name for name in header if header.count(name) > 1})
            if duplicates:
                raise ValueError(f"Dataset CSV header contains duplicate columns: {duplicates}")
            if expected_columns is not None and header != expected_columns:
                raise ValueError(
                    "Dataset CSV header does not exactly match data.expected_columns; "
                    f"expected={expected_columns}, observed={header}"
                )
            width = len(header)
            for logical_rows, row in enumerate(reader, start=1):
                if len(row) != width:
                    raise ValueError(
                        "Malformed CSV logical record: "
                        f"record={logical_rows}, expected_fields={width}, "
                        f"observed_fields={len(row)}"
                    )
                if row == header:
                    raise ValueError(
                        f"Dataset CSV contains a repeated header at logical record {logical_rows}"
                    )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Cannot parse dataset CSV as strict RFC 4180 records: {exc}") from exc
    finally:
        csv.field_size_limit(previous_field_limit)

    raw = pd.read_csv(
        path,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error",
    )
    if len(raw) != logical_rows:
        raise ValueError(
            "CSV parser record-count disagreement: "
            f"strict_reader={logical_rows}, pandas={len(raw)}"
        )
    return raw, {
        "encoding": encoding,
        "field_size_limit_bytes": field_size_limit,
        "column_count": int(len(raw.columns)),
        "logical_record_count": int(logical_rows),
        "expected_columns_checked": expected_columns is not None,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_slurm_minutes(
    value: Any,
    unlimited_minutes: Optional[float] = None,
    allow_truncated: bool = True,
) -> float:
    """Parse Slurm durations, including sacct values truncated with a trailing '+'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    if text.upper() == "UNLIMITED":
        if unlimited_minutes is None:
            return float("nan")
        try:
            replacement = float(unlimited_minutes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("unlimited_minutes must be a finite non-negative value") from exc
        if not math.isfinite(replacement) or replacement < 0:
            raise ValueError("unlimited_minutes must be a finite non-negative value")
        return replacement

    if "+" in text:
        if not allow_truncated or not text.endswith("+") or text.count("+") != 1:
            return float("nan")
        text = text[:-1].rstrip()
        if text.endswith(":"):
            text += "0"
        else:
            duration = text.split("-", 1)[-1]
            if duration.count(":") == 1:
                text += ":00"
    try:
        days = 0
        has_day_prefix = False
        if "-" in text:
            day_text, text = text.split("-", 1)
            days = int(day_text)
            has_day_prefix = True
        fields = text.split(":")
        if len(fields) == 3:
            hours, minutes, seconds = (int(x) for x in fields)
        elif len(fields) == 2:
            first, second = (int(x) for x in fields)
            if has_day_prefix:
                hours, minutes, seconds = first, second, 0
            else:
                hours, minutes, seconds = 0, first, second
        elif len(fields) == 1:
            amount = int(fields[0])
            if days < 0 or amount < 0:
                return float("nan")
            if has_day_prefix:
                return float(days * 1440 + amount * 60)
            return float(amount)
        else:
            return float("nan")
        if (
            days < 0
            or hours < 0
            or minutes < 0
            or seconds < 0
            or minutes >= 60
            or seconds >= 60
            or (has_day_prefix and hours >= 24)
        ):
            return float("nan")
        return float(days * 1440 + hours * 60 + minutes + seconds / 60.0)
    except (TypeError, ValueError):
        return float("nan")


def _nonempty_string_list(value: Any, key: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{key} cannot contain empty values")
    return normalized


def _value_counts(series: pd.Series) -> Dict[str, int]:
    values = series.astype("string").fillna("<MISSING>")
    return {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).sort_index().items()
    }


def _apply_row_filters(
    raw: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    filters = get_value(config, "data.filters", {}) or {}
    if not isinstance(filters, dict):
        raise ValueError("data.filters must be a mapping")
    unknown = sorted(set(filters) - {"states", "exit_codes"})
    if unknown:
        raise ValueError(f"Unsupported data.filters keys: {unknown}")

    columns = require_value(config, "data.columns")
    keep = pd.Series(True, index=raw.index, dtype=bool)
    dropped = {"state": 0, "exit_code": 0}
    settings: Dict[str, Any] = {"states": None, "exit_codes": None}
    observed_states: Dict[str, int] = {}
    observed_exit_codes: Dict[str, int] = {}

    if filters.get("states") is not None:
        allowed_states = [
            item.upper() for item in _nonempty_string_list(filters["states"], "data.filters.states")
        ]
        source = columns.get("state")
        if not source:
            raise ValueError("data.filters.states requires data.columns.state")
        if source not in raw.columns:
            raise ValueError(f"Missing state filter column: {source}")
        state = raw[source].astype("string").fillna("").str.strip().str.upper()
        observed_states = _value_counts(state)
        rejected = keep & ~state.isin(allowed_states)
        dropped["state"] = int(rejected.sum())
        keep &= ~rejected
        settings["states"] = allowed_states

    if filters.get("exit_codes") is not None:
        allowed_exit_codes = _nonempty_string_list(
            filters["exit_codes"], "data.filters.exit_codes"
        )
        source = columns.get("exit_code")
        if not source:
            raise ValueError("data.filters.exit_codes requires data.columns.exit_code")
        if source not in raw.columns:
            raise ValueError(f"Missing exit-code filter column: {source}")
        exit_code = raw[source].astype("string").fillna("").str.strip()
        observed_exit_codes = _value_counts(exit_code.loc[keep])
        malformed = keep & ~exit_code.str.fullmatch(r"\d+:\d+", na=False)
        if malformed.any():
            examples = exit_code.loc[malformed].head(10).tolist()
            raise ValueError(
                "Exit-code filter column must use Slurm status:signal syntax; "
                f"invalid_count={int(malformed.sum())}, examples={examples}"
            )
        rejected = keep & ~exit_code.isin(allowed_exit_codes)
        dropped["exit_code"] = int(rejected.sum())
        keep &= ~rejected
        settings["exit_codes"] = allowed_exit_codes

    retained = raw.loc[keep].copy()
    if retained.empty:
        raise ValueError("All dataset rows were removed by data.filters")
    raw_count = int(len(raw))
    retained_count = int(len(retained))
    return retained, {
        "version": 1,
        "enabled": any(value is not None for value in settings.values()),
        "settings": settings,
        "raw_row_count": raw_count,
        "retained_row_count": retained_count,
        "dropped_row_count": raw_count - retained_count,
        "dropped_by_filter": dropped,
        "observed_state_counts": observed_states,
        "observed_exit_code_counts_after_state_filter": observed_exit_codes,
    }


def _parse_datetime_series(
    values: pd.Series,
    name: str,
    assumed_timezone: str,
) -> pd.Series:
    parsed_values: List[pd.Timestamp] = []
    invalid: List[str] = []
    for value in values.tolist():
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError("missing timestamp")
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    assumed_timezone,
                    ambiguous="raise",
                    nonexistent="raise",
                )
            timestamp = timestamp.tz_convert("UTC")
        except (TypeError, ValueError, OverflowError, KeyError) as exc:
            invalid.append(f"{value!r} ({exc})")
            continue
        parsed_values.append(timestamp)
    if invalid:
        raise ValueError(
            f"Column {name} contains invalid timestamps; examples={invalid[:5]}"
        )
    return pd.Series(pd.DatetimeIndex(parsed_values), index=values.index)


def _redact_code_paths(
    series: pd.Series,
    config: Dict[str, Any],
) -> tuple[pd.Series, Dict[str, Any]]:
    settings = get_value(config, "data.code_redaction", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("data.code_redaction must be a mapping")
    unknown = sorted(set(settings) - {"enabled", "path_prefixes"})
    if unknown:
        raise ValueError(f"Unsupported data.code_redaction keys: {unknown}")
    enabled = bool(settings.get("enabled", False))
    prefixes_value = settings.get("path_prefixes", []) or []
    if not isinstance(prefixes_value, list):
        raise ValueError("data.code_redaction.path_prefixes must be a list")
    prefixes = [str(prefix).rstrip("/") for prefix in prefixes_value]
    if any(not prefix or prefix == "/" for prefix in prefixes):
        raise ValueError("Code-redaction path prefixes must be specific non-root paths")
    if enabled and not prefixes:
        raise ValueError("Enabled code redaction requires at least one path prefix")

    redacted = series.astype("string").fillna("").copy()
    original = redacted.copy()
    if enabled:
        for prefix in prefixes:
            pattern = re.compile(
                re.escape(prefix) + r"/[^/\s\"'`;,|&<>()]+"
            )
            redacted = redacted.map(
                lambda value, regex=pattern, base=prefix: regex.sub(
                    base + "/<USER>", str(value)
                )
            ).astype("string")
    return redacted, {
        "version": 1,
        "enabled": enabled,
        "path_prefixes": prefixes,
        "replacement": "<USER>",
        "rows_changed": int((redacted != original).sum()),
    }


def _normalize_label(
    series: pd.Series,
    scale: str,
    name: str,
    out_of_range: str,
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if scale == "percent":
        values = values / 100.0
    elif scale != "ratio":
        raise ValueError("data.label_scale must be exactly 'ratio' or 'percent'")
    invalid = np.isfinite(values) & ((values < 0.0) | (values > 1.0))
    finite = values[np.isfinite(values)]
    if invalid.any() and out_of_range == "error":
        minimum = float(finite.min())
        maximum = float(finite.max())
        raise ValueError(
            f"{name} is outside [0,1] after configured scaling: "
            f"min={minimum:.6g}, max={maximum:.6g}"
        )
    if out_of_range == "mask":
        values = values.mask(invalid)
    elif out_of_range != "error":
        raise ValueError("data.label_out_of_range must be exactly 'error' or 'mask'")
    return values, pd.Series(invalid, index=series.index, dtype=bool)


def _required_source_columns(config: Dict[str, Any]) -> List[str]:
    columns = require_value(config, "data.columns")
    keys = ["job_id", "user", "code", "gpu_count", "memory_peak", "vram_peak", "order"]
    missing_mappings = [key for key in keys if not columns.get(key)]
    if missing_mappings:
        raise ValueError(f"Missing required data.columns mappings: {missing_mappings}")
    _validate_metadata_sources(config, columns)
    required = [columns[key] for key in keys]
    for key in (
        "memory_valid",
        "vram_valid",
        "job_name",
        "partition",
        "submit_time",
        "start_time",
        "completion_time",
        "state",
        "exit_code",
        "elapsed",
        "time_limit",
        "node_list",
        "memory_peak_absolute",
        "vram_peak_absolute",
        "workdir",
    ):
        if columns.get(key):
            required.append(columns[key])
    feature_specs = get_value(config, "data.metadata_features", []) or []
    if feature_specs:
        if not isinstance(feature_specs, list):
            raise ValueError("data.metadata_features must be a list")
        for spec in feature_specs:
            if not isinstance(spec, dict):
                raise ValueError("Each data.metadata_features entry must be a mapping")
            if spec.get("transform") == "ratio":
                required.extend([spec.get("numerator"), spec.get("denominator")])
            else:
                required.append(spec.get("column"))
        if any(not value for value in required):
            raise ValueError("Every metadata feature must name its source column(s)")
    else:
        required.extend(get_value(config, "data.metadata_numeric", []) or [])
    required.extend(get_value(config, "data.metadata_categorical", []) or [])
    return list(dict.fromkeys(required))


def _validate_metadata_sources(config: Dict[str, Any], columns: Dict[str, Any]) -> None:
    forbidden = {
        str(columns[key])
        for key in (
            "memory_peak",
            "vram_peak",
            "memory_valid",
            "vram_valid",
            "start_time",
            "completion_time",
            "state",
            "exit_code",
            "elapsed",
            "node_list",
            "memory_peak_absolute",
            "vram_peak_absolute",
            "workdir",
        )
        if columns.get(key)
    }
    configured_forbidden = get_value(config, "data.forbidden_metadata_columns", []) or []
    if not isinstance(configured_forbidden, list) or not all(
        isinstance(source, str) and source for source in configured_forbidden
    ):
        raise ValueError("data.forbidden_metadata_columns must be a list of non-empty names")
    forbidden.update(configured_forbidden)
    referenced: List[str] = []
    for source in get_value(config, "data.metadata_numeric", []) or []:
        referenced.append(str(source))
    for source in get_value(config, "data.metadata_categorical", []) or []:
        referenced.append(str(source))
    for spec in get_value(config, "data.metadata_features", []) or []:
        if not isinstance(spec, dict):
            continue
        if spec.get("transform") == "ratio":
            for key in ("numerator", "denominator"):
                if spec.get(key):
                    referenced.append(str(spec[key]))
        elif spec.get("column"):
            referenced.append(str(spec["column"]))
    leaked = sorted(set(referenced) & forbidden)
    if leaked:
        raise ValueError(
            "Metadata features cannot reference outcome or post-submission columns: "
            f"{leaked}. Use only information available at the prediction timestamp."
        )


def _strict_boolean_source(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Validity column {name} contains missing values")

    parsed: List[bool] = []
    invalid: List[str] = []
    for value in series.tolist():
        converted: Optional[bool] = None
        if isinstance(value, (bool, np.bool_)):
            converted = bool(value)
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if int(value) in (0, 1):
                converted = bool(value)
        elif isinstance(value, (float, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric) and numeric in (0.0, 1.0):
                converted = bool(numeric)
        elif isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                converted = True
            elif normalized in {"false", "0"}:
                converted = False
        if converted is None:
            invalid.append(repr(value))
        else:
            parsed.append(converted)

    if invalid:
        raise ValueError(
            f"Validity column {name} must contain only booleans, 0/1, or "
            f"true/false strings; invalid examples={invalid[:5]}"
        )
    return pd.Series(parsed, index=series.index, dtype=bool)


def _validity_source(
    raw: pd.DataFrame,
    columns: Dict[str, Any],
    key: str,
) -> pd.Series:
    source = columns.get(key)
    if not source:
        return pd.Series(True, index=raw.index, dtype=bool)
    return _strict_boolean_source(raw[source], str(source))


def _numeric_source(raw: pd.DataFrame, spec: Dict[str, Any], column: str) -> pd.Series:
    parser = str(spec.get("parser", "numeric"))
    if parser == "numeric":
        return pd.to_numeric(raw[column], errors="coerce").astype(float)
    if parser == "slurm_minutes":
        raw_unlimited = spec.get("unlimited_as_max", False)
        raw_allow_truncated = spec.get("allow_truncated", True)
        if not isinstance(raw_unlimited, (bool, np.bool_)):
            raise ValueError(f"Metadata feature {column} unlimited_as_max must be true or false")
        if not isinstance(raw_allow_truncated, (bool, np.bool_)):
            raise ValueError(f"Metadata feature {column} allow_truncated must be true or false")

        unlimited_minutes: Optional[float] = None
        if bool(raw_unlimited):
            configured = spec.get("unlimited_minutes", spec.get("maximum"))
            if configured is None:
                raise ValueError(
                    f"Metadata feature {column} enables unlimited_as_max but provides "
                    "neither unlimited_minutes nor maximum"
                )
            unlimited_minutes = parse_slurm_minutes(
                "UNLIMITED", unlimited_minutes=configured
            )
        return raw[column].map(
            lambda value: parse_slurm_minutes(
                value,
                unlimited_minutes=unlimited_minutes,
                allow_truncated=bool(raw_allow_truncated),
            )
        ).astype(float)
    raise ValueError(f"Unsupported metadata parser: {parser}")


def _slurm_parser_protocol(
    raw: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    audits: Dict[str, Dict[str, Any]] = {}
    specs = get_value(config, "data.metadata_features", []) or []
    candidates: List[tuple[str, str, Dict[str, Any]]] = []
    for spec in specs:
        if isinstance(spec, dict) and str(spec.get("parser", "numeric")) == "slurm_minutes":
            candidates.append((str(spec.get("name", "")), str(spec.get("column", "")), spec))
    if not specs:
        columns = require_value(config, "data.columns")
        source = columns.get("time_limit")
        if source:
            candidates.append(("time_limit_minutes", str(source), {"parser": "slurm_minutes"}))

    for name, source, spec in candidates:
        text_values = raw[source].astype("string").fillna("").str.strip()
        parsed = _numeric_source(raw, spec, source)
        unlimited_as_max = bool(spec.get("unlimited_as_max", False))
        configured_replacement = (
            spec.get("unlimited_minutes", spec.get("maximum"))
            if unlimited_as_max
            else None
        )
        audits[name] = {
            "source_column": source,
            "row_count": int(len(raw)),
            "unlimited_count": int(text_values.str.upper().eq("UNLIMITED").sum()),
            "truncated_count": int(text_values.str.endswith("+").sum()),
            "invalid_count_after_parsing": int((~np.isfinite(parsed)).sum()),
            "unlimited_as_max": unlimited_as_max,
            "unlimited_minutes": (
                float(configured_replacement)
                if configured_replacement is not None
                else None
            ),
            "allow_truncated": bool(spec.get("allow_truncated", True)),
            "truncated_policy": "parse_visible_fields_and_fill_missing_seconds_with_zero",
        }
    return audits


def _validate_nonnegative(values: pd.Series, name: str) -> None:
    finite = values[np.isfinite(values)]
    if not finite.empty and (finite < 0).any():
        raise ValueError(f"Metadata feature {name} contains negative request values")


def _engineered_metadata(
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    specs: List[Dict[str, Any]],
) -> List[str]:
    features: List[str] = []
    seen_names = set()
    for spec in specs:
        name = str(spec.get("name", "")).strip()
        if not name or name in seen_names:
            raise ValueError(f"Metadata feature names must be non-empty and unique: {name!r}")
        seen_names.add(name)
        transform = str(spec.get("transform", "identity"))

        if transform == "ratio":
            numerator_column = str(spec["numerator"])
            denominator_column = str(spec["denominator"])
            numerator = pd.to_numeric(raw[numerator_column], errors="coerce").astype(float)
            denominator = pd.to_numeric(raw[denominator_column], errors="coerce").astype(float)
            _validate_nonnegative(numerator, name)
            _validate_nonnegative(denominator, name)
            floor = float(spec.get("denominator_floor", 1.0))
            if floor <= 0:
                raise ValueError(f"Metadata ratio {name} requires denominator_floor > 0")
            values = numerator / np.maximum(denominator, floor)
            if bool(spec.get("zero_if_denominator_nonpositive", False)):
                values = values.where(denominator > 0, 0.0)
        else:
            source_column = str(spec["column"])
            values = _numeric_source(raw, spec, source_column)
            _validate_nonnegative(values, name)
            if transform == "log_max":
                try:
                    maximum = float(spec.get("maximum", 0.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Metadata log_max feature {name} requires a numeric maximum; "
                        f"replace the configuration placeholder"
                    ) from exc
                if maximum <= 0:
                    raise ValueError(f"Metadata log_max feature {name} requires maximum > 0")
                finite = values[np.isfinite(values)]
                over = finite > maximum
                if over.any() and not bool(spec.get("clip_to_max", False)):
                    raise ValueError(
                        f"Metadata feature {name} exceeds configured maximum {maximum}; "
                        "fix the scheduler bound or explicitly set clip_to_max=true"
                    )
                bounded = values.clip(lower=0.0, upper=maximum)
                values = np.log1p(bounded) / np.log1p(maximum)
            elif transform != "identity":
                raise ValueError(f"Unsupported metadata transform: {transform}")

        target = f"num__{name}"
        frame[target] = pd.Series(values, index=raw.index, dtype=float)
        features.append(target)
    return features


def _parse_order(
    values: pd.Series,
    order_type: str,
    assumed_timezone: str = "UTC",
    name: str = "order",
) -> pd.Series:
    if order_type == "numeric":
        parsed = pd.to_numeric(values, errors="coerce")
    elif order_type == "datetime":
        return _parse_datetime_series(values, name, assumed_timezone)
    elif order_type == "string":
        parsed = values.astype("string")
    elif order_type == "row":
        parsed = pd.Series(np.arange(len(values)), index=values.index)
    else:
        raise ValueError("split.order_type must be numeric, datetime, string, or row")
    if parsed.isna().any():
        count = int(parsed.isna().sum())
        raise ValueError(f"The split order column has {count} unparseable/missing values")
    return parsed


def _standardize_frame(
    raw: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
    columns = require_value(config, "data.columns")
    missing = [name for name in _required_source_columns(config) if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    frame = pd.DataFrame(index=raw.index)
    try:
        frame["source_row"] = raw.index.to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Internal source-row identifiers must be integer CSV record offsets") from exc
    frame["job_id"] = raw[columns["job_id"]].astype("string").fillna("")
    if frame["job_id"].str.strip().eq("").any():
        raise ValueError("The configured job-id column contains empty values")
    duplicate_job_ids = frame["job_id"].duplicated(keep=False)
    if duplicate_job_ids.any() and bool(
        get_value(config, "data.require_unique_job_ids", False)
    ):
        examples = frame.loc[duplicate_job_ids, "job_id"].astype(str).head(10).tolist()
        raise ValueError(
            "The configured job-id column is not unique after filtering; "
            f"duplicate_rows={int(duplicate_job_ids.sum())}, examples={examples}"
        )
    frame["user_id"] = raw[columns["user"]].astype("string").fillna("")
    frame.loc[frame["user_id"].str.strip().eq(""), "user_id"] = "__MISSING_USER__"
    frame["code"], redaction_protocol = _redact_code_paths(raw[columns["code"]], config)
    frame["gpu_count"] = pd.to_numeric(raw[columns["gpu_count"]], errors="coerce")

    string_context = {
        "job_name": "job_name",
        "partition": "partition",
        "state": "job_state",
        "exit_code": "exit_code",
        "node_list": "node_list",
        "workdir": "workdir",
    }
    for config_key, target in string_context.items():
        source = columns.get(config_key)
        if source:
            frame[target] = raw[source].astype("string").fillna("")

    numeric_context = {
        "memory_peak_absolute": "observed_memory_peak_absolute",
        "vram_peak_absolute": "observed_vram_peak_absolute",
    }
    for config_key, target in numeric_context.items():
        source = columns.get(config_key)
        if source:
            frame[target] = pd.to_numeric(raw[source], errors="coerce").astype(float)

    scale = str(get_value(config, "data.label_scale", "ratio"))
    out_of_range = str(get_value(config, "data.label_out_of_range", "error"))
    frame["target_memory"], frame["out_of_range_memory"] = _normalize_label(
        raw[columns["memory_peak"]], scale, "memory_peak", out_of_range
    )
    frame["target_vram"], frame["out_of_range_vram"] = _normalize_label(
        raw[columns["vram_peak"]], scale, "vram_peak", out_of_range
    )
    memory_valid_source = _validity_source(raw, columns, "memory_valid")
    vram_valid_source = _validity_source(raw, columns, "vram_valid")
    frame["valid_memory"] = (
        np.isfinite(frame["target_memory"]) & memory_valid_source
    )
    frame["valid_vram"] = (
        np.isfinite(frame["target_vram"])
        & np.isfinite(frame["gpu_count"])
        & (frame["gpu_count"] > 0)
        & vram_valid_source
    )

    feature_specs = get_value(config, "data.metadata_features", []) or []
    if feature_specs:
        numeric_features = _engineered_metadata(raw, frame, feature_specs)
    else:
        numeric_features = []
        for source in get_value(config, "data.metadata_numeric", []) or []:
            target = f"num__{source}"
            frame[target] = pd.to_numeric(raw[source], errors="coerce")
            numeric_features.append(target)

        time_limit = columns.get("time_limit")
        if time_limit:
            frame["num__time_limit_minutes"] = raw[time_limit].map(parse_slurm_minutes)
            numeric_features.append("num__time_limit_minutes")

    if bool(get_value(config, "data.require_complete_metadata", False)):
        missing_counts = {
            column: int(frame[column].isna().sum())
            for column in numeric_features
            if frame[column].isna().any()
        }
        if missing_counts:
            raise ValueError(
                "Configured metadata features contain missing/unparseable values: "
                f"{missing_counts}"
            )

    categorical_features: List[str] = []
    for source in get_value(config, "data.metadata_categorical", []) or []:
        target = f"cat__{source}"
        frame[target] = raw[source].astype("string").fillna("__MISSING__")
        categorical_features.append(target)

    order_source = columns.get("order")
    if not order_source:
        raise ValueError("data.columns.order must name an explicit ordering column")
    assumed_timezone = str(get_value(config, "data.timestamp_timezone", "UTC"))
    frame["order_value"] = _parse_order(
        raw[order_source],
        str(get_value(config, "split.order_type", "datetime")),
        assumed_timezone=assumed_timezone,
        name=str(order_source),
    )

    for config_key, target in (
        ("submit_time", "submit_time"),
        ("start_time", "start_time"),
        ("completion_time", "completion_time"),
    ):
        source = columns.get(config_key)
        if source:
            frame[target] = _parse_datetime_series(
                raw[source], str(source), assumed_timezone
            )
        else:
            frame[target] = pd.NaT

    elapsed_source = columns.get("elapsed")
    if elapsed_source:
        frame["elapsed_minutes"] = raw[elapsed_source].map(parse_slurm_minutes).astype(float)
    else:
        frame["elapsed_minutes"] = np.nan

    frame = frame.sort_values(["order_value", "source_row"], kind="mergesort").reset_index(drop=True)
    frame["sample_id"] = frame["job_id"].astype(str) + "::" + frame["source_row"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise AssertionError("Internal sample_id generation produced duplicates")
    standardization_protocol = {
        "duplicate_job_id_rows": int(duplicate_job_ids.sum()),
        "unique_job_ids_required": bool(
            get_value(config, "data.require_unique_job_ids", False)
        ),
        "timestamp_timezone_for_naive_values": assumed_timezone,
        "code_redaction": redaction_protocol,
        "metadata_parsing": _slurm_parser_protocol(raw, config),
    }
    return frame, numeric_features, categorical_features, standardization_protocol


def _computed_splits(n: int, train_ratio: float, validation_ratio: float) -> np.ndarray:
    if n < 3:
        raise ValueError("At least three rows are required for train/validation/test")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must be positive and sum to less than one")
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    if train_end == 0 or validation_end <= train_end or validation_end >= n:
        raise ValueError("The configured ratios produce an empty split")
    result = np.full(n, "test", dtype=object)
    result[:train_end] = "train"
    result[train_end:validation_end] = "validation"
    return result


def _fingerprint_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def order_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for sample_id, order_value in frame[["sample_id", "order_value"]].itertuples(
        index=False, name=None
    ):
        digest.update(
            f"{sample_id}\t{_fingerprint_value(order_value)}\n".encode("utf-8")
        )
    return digest.hexdigest()


def split_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for sample_id, split in frame[["sample_id", "split"]].itertuples(
        index=False, name=None
    ):
        digest.update(f"{sample_id}\t{split}\n".encode("utf-8"))
    return digest.hexdigest()


def _boundary_completion_settings(config: Dict[str, Any]) -> tuple[bool, str]:
    raw_required = get_value(
        config, "split.require_completed_labels_at_boundaries", False
    )
    if not isinstance(raw_required, (bool, np.bool_)):
        raise ValueError(
            "split.require_completed_labels_at_boundaries must be true or false"
        )
    required = bool(raw_required)

    policy = normalize_boundary_completion_policy(
        get_value(config, "split.boundary_completion_policy", "error")
    )
    if not required and policy != "error":
        raise ValueError(
            "split.boundary_completion_policy=purge requires "
            "split.require_completed_labels_at_boundaries=true"
        )
    return required, policy


def _split_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    columns = require_value(config, "data.columns")
    boundary_required, boundary_policy = _boundary_completion_settings(config)
    return {
        "order_column": str(columns["order"]),
        "order_type": str(get_value(config, "split.order_type", "datetime")),
        "order_semantics": str(get_value(config, "split.order_semantics", "unspecified")),
        "train_ratio": float(get_value(config, "split.train_ratio", 0.8)),
        "validation_ratio": float(get_value(config, "split.validation_ratio", 0.1)),
        "require_completed_labels_at_boundaries": boundary_required,
        "boundary_completion_policy": boundary_policy,
        "boundary_protocol_version": BOUNDARY_COMPLETION_PROTOCOL_VERSION,
    }


def _validate_contiguous_splits(frame: pd.DataFrame, settings: Dict[str, Any]) -> None:
    expected = _computed_splits(
        len(frame), settings["train_ratio"], settings["validation_ratio"]
    )
    actual = frame["split"].astype(str).to_numpy()
    if not np.array_equal(actual, expected):
        mismatch = np.flatnonzero(actual != expected)
        preview = mismatch[:10].tolist()
        raise ValueError(
            "Split assignments must be contiguous train -> validation -> test in "
            f"the configured sample order; mismatched_positions={preview}, "
            f"total_mismatches={len(mismatch)}"
        )


def _validate_ordered_effective_splits(frame: pd.DataFrame) -> None:
    allowed = {"train", "validation", "test"}
    actual = frame["split"].astype(str)
    observed = set(actual)
    if observed != allowed:
        missing = sorted(allowed - observed)
        raise ValueError(
            "Boundary purge must leave every split non-empty; "
            f"empty_splits={missing}"
        )
    rank = actual.map({"train": 0, "validation": 1, "test": 2}).to_numpy()
    if np.any(np.diff(rank) < 0):
        raise ValueError(
            "Effective split assignments must remain ordered "
            "train -> validation -> test after boundary purge"
        )


def _manifest_scalar(manifest: pd.DataFrame, column: str) -> Any:
    if manifest[column].isna().any():
        raise ValueError(
            f"Split manifest column {column} must be non-null on every row"
        )
    values = manifest[column].unique().tolist()
    if len(values) != 1:
        raise ValueError(
            f"Split manifest column {column} must contain exactly one value"
        )
    return values[0]


def _manifest_bool(manifest: pd.DataFrame, column: str) -> bool:
    value = _manifest_scalar(manifest, column)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Split manifest column {column} must contain true or false")


def _boundary_action_fingerprint(
    frame: pd.DataFrame,
    actions: pd.Series,
) -> str:
    if len(frame) != len(actions):
        raise ValueError("Boundary action count does not match the base split")
    digest = hashlib.sha256()
    rows = frame[["sample_id", "split"]].itertuples(index=False, name=None)
    for (sample_id, split), action in zip(rows, actions.astype(str).tolist()):
        digest.update(f"{sample_id}\t{split}\t{action}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_manifest_boundary_settings(
    manifest: pd.DataFrame,
    settings: Dict[str, Any],
) -> bool:
    boundary_columns = {
        "boundary_action",
        "require_completed_labels_at_boundaries",
        "boundary_completion_policy",
        "boundary_protocol_version",
        "boundary_purge_fingerprint",
    }
    audit_columns = {"job_id", "source_row", "order_value"}
    provenance_columns = boundary_columns | audit_columns
    has_any_boundary = bool(boundary_columns.intersection(manifest.columns))
    missing = provenance_columns - set(manifest.columns)
    if settings["boundary_completion_policy"] == "purge" and missing:
        raise ValueError(
            "Purge split manifest is missing boundary provenance columns: "
            f"{sorted(missing)}. Configure a new split.manifest path so the "
            "base split and purged samples can be audited."
        )
    if has_any_boundary and missing:
        raise ValueError(
            "Split manifest has incomplete boundary provenance columns: "
            f"{sorted(missing)}"
        )
    if not has_any_boundary:
        return False

    stored_required = _manifest_bool(
        manifest, "require_completed_labels_at_boundaries"
    )
    configured_required = bool(settings["require_completed_labels_at_boundaries"])
    if stored_required != configured_required:
        raise ValueError(
            "Split manifest require_completed_labels_at_boundaries mismatch: "
            f"stored={stored_required}, configured={configured_required}"
        )
    for key in ("boundary_completion_policy", "boundary_protocol_version"):
        stored = str(_manifest_scalar(manifest, key))
        if stored != str(settings[key]):
            raise ValueError(
                f"Split manifest {key} mismatch: stored={stored!r}, "
                f"configured={settings[key]!r}"
            )
    return True


def _validate_manifest_audit_columns(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    settings: Dict[str, Any],
) -> None:
    aligned = manifest.set_index("sample_id").loc[frame["sample_id"].astype(str)]

    stored_job_ids = aligned["job_id"].astype(str).to_numpy()
    current_job_ids = frame["job_id"].astype(str).to_numpy()
    if not np.array_equal(stored_job_ids, current_job_ids):
        raise ValueError("Split manifest job_id audit column is invalid")

    stored_source_rows = pd.to_numeric(aligned["source_row"], errors="coerce")
    current_source_rows = frame["source_row"].to_numpy(dtype=np.int64)
    if (
        stored_source_rows.isna().any()
        or not np.equal(stored_source_rows.to_numpy(), current_source_rows).all()
    ):
        raise ValueError("Split manifest source_row audit column is invalid")

    order_type = str(settings["order_type"])
    if order_type == "datetime":
        stored_order = pd.DatetimeIndex(
            pd.to_datetime(aligned["order_value"], errors="coerce", utc=True)
        )
        current_order = pd.DatetimeIndex(frame["order_value"])
        order_matches = not stored_order.isna().any() and np.array_equal(
            stored_order.asi8, current_order.asi8
        )
    elif order_type in {"numeric", "row"}:
        stored_order = pd.to_numeric(aligned["order_value"], errors="coerce")
        current_order = pd.to_numeric(frame["order_value"], errors="coerce")
        order_matches = (
            not stored_order.isna().any()
            and not current_order.isna().any()
            and np.allclose(
                stored_order.to_numpy(float),
                current_order.to_numpy(float),
                rtol=0.0,
                atol=1e-12,
            )
        )
    else:
        order_matches = np.array_equal(
            aligned["order_value"].astype(str).to_numpy(),
            frame["order_value"].astype(str).to_numpy(),
        )
    if not order_matches:
        raise ValueError("Split manifest order_value audit column is invalid")


def _apply_manifest(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    data_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[Path], Dict[str, Any]]:
    settings = _split_settings(config)
    current_order_fingerprint = order_fingerprint(frame)
    manifest_value = get_value(config, "split.manifest")
    manifest_path = resolve_path(config, manifest_value) if manifest_value else None
    create_if_missing = bool(get_value(config, "split.create_if_missing", False))
    if settings["boundary_completion_policy"] == "purge":
        if manifest_path is None:
            raise ValueError(
                "split.boundary_completion_policy=purge requires split.manifest "
                "for an auditable base split"
            )
        if not manifest_path.exists() and not create_if_missing:
            raise ValueError(
                "Purge split manifest does not exist and "
                "split.create_if_missing is false"
            )
    manifest: Optional[pd.DataFrame] = None
    manifest_has_boundary_provenance = False
    if manifest_path and manifest_path.exists():
        manifest = pd.read_csv(
            manifest_path,
            dtype={"sample_id": str, "job_id": str, "order_value": str},
        )
        required = {
            "sample_id",
            "split",
            "dataset_sha256",
            "order_fingerprint",
            "split_fingerprint",
            "order_column",
            "order_type",
            "order_semantics",
            "train_ratio",
            "validation_ratio",
        }
        if not required.issubset(manifest.columns):
            raise ValueError(f"Split manifest is missing columns: {sorted(required - set(manifest.columns))}")
        stored_data_sha256 = str(_manifest_scalar(manifest, "dataset_sha256"))
        if stored_data_sha256 != data_sha256:
            raise ValueError("Split manifest data hash does not match the configured CSV")
        if manifest["sample_id"].duplicated().any():
            raise ValueError("Split manifest contains duplicate sample_id values")
        stored_order_fingerprint = str(_manifest_scalar(manifest, "order_fingerprint"))
        if stored_order_fingerprint != current_order_fingerprint:
            raise ValueError(
                "Split manifest order fingerprint does not match the configured sample order"
            )
        for key in ("order_column", "order_type", "order_semantics"):
            stored = str(_manifest_scalar(manifest, key))
            if stored != str(settings[key]):
                raise ValueError(
                    f"Split manifest {key} mismatch: stored={stored!r}, "
                    f"configured={settings[key]!r}"
                )
        for key in ("train_ratio", "validation_ratio"):
            stored = float(_manifest_scalar(manifest, key))
            if not np.isclose(stored, settings[key], rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Split manifest {key} mismatch: stored={stored}, "
                    f"configured={settings[key]}"
                )
        manifest_has_boundary_provenance = _validate_manifest_boundary_settings(
            manifest, settings
        )
        mapping = manifest.set_index("sample_id")["split"]
        missing = frame.loc[~frame["sample_id"].isin(mapping.index), "sample_id"]
        extra = mapping.index.difference(frame["sample_id"])
        if not missing.empty or len(extra):
            raise ValueError(
                f"Split manifest mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        frame["split"] = frame["sample_id"].map(mapping).astype(str)
        _validate_contiguous_splits(frame, settings)
        stored_split_fingerprint = str(_manifest_scalar(manifest, "split_fingerprint"))
        current_split_fingerprint = split_fingerprint(frame)
        if stored_split_fingerprint != current_split_fingerprint:
            raise ValueError("Split manifest assignment fingerprint is invalid")
        if manifest_has_boundary_provenance:
            _validate_manifest_audit_columns(frame, manifest, settings)
    else:
        frame["split"] = _computed_splits(
            len(frame),
            settings["train_ratio"],
            settings["validation_ratio"],
        )
        _validate_contiguous_splits(frame, settings)

    allowed = {"train", "validation", "test"}
    observed = set(frame["split"].dropna().astype(str))
    if observed != allowed:
        raise ValueError(f"Expected splits {sorted(allowed)}, observed {sorted(observed)}")

    effective_frame, boundary_protocol, boundary_actions = (
        _boundary_completion_protocol(frame, config)
    )
    current_boundary_fingerprint = _boundary_action_fingerprint(
        frame, boundary_actions
    )
    if manifest is not None and manifest_has_boundary_provenance:
        action_mapping = manifest.set_index("sample_id")["boundary_action"]
        stored_actions = frame["sample_id"].map(action_mapping)
        if stored_actions.isna().any():
            raise ValueError("Split manifest boundary_action mapping is incomplete")
        if not np.array_equal(
            stored_actions.astype(str).to_numpy(),
            boundary_actions.astype(str).to_numpy(),
        ):
            raise ValueError(
                "Split manifest boundary_action values do not match the "
                "recomputed boundary policy"
            )
        stored_boundary_fingerprint = str(
            _manifest_scalar(manifest, "boundary_purge_fingerprint")
        )
        if stored_boundary_fingerprint != current_boundary_fingerprint:
            raise ValueError("Split manifest boundary purge fingerprint is invalid")

    if (
        manifest is None
        and manifest_path
        and create_if_missing
    ):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = frame[
            ["sample_id", "job_id", "source_row", "order_value", "split"]
        ].copy()
        manifest["dataset_sha256"] = data_sha256
        manifest["order_fingerprint"] = current_order_fingerprint
        manifest["split_fingerprint"] = split_fingerprint(frame)
        for key, value in settings.items():
            manifest[key] = value
        manifest["boundary_action"] = boundary_actions.astype(str).to_numpy()
        manifest["boundary_purge_fingerprint"] = current_boundary_fingerprint
        manifest.to_csv(manifest_path, index=False)

    return (
        effective_frame,
        frame.copy().reset_index(drop=True),
        manifest_path,
        boundary_protocol,
    )


def _validate_event_times(
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    has_submission = bool(frame["submit_time"].notna().all())
    has_start = bool(frame["start_time"].notna().all())
    has_completion = bool(frame["completion_time"].notna().all())
    result: Dict[str, Any] = {
        "submission_completion_checked": False,
        "submission_start_completion_checked": False,
        "elapsed_checked": False,
        "elapsed_tolerance_seconds": None,
        "max_elapsed_delta_seconds": None,
    }

    if has_submission and has_completion:
        invalid = frame["completion_time"] < frame["submit_time"]
        if invalid.any():
            examples = frame.loc[invalid, "sample_id"].astype(str).head(10).tolist()
            raise ValueError(
                "completion_time must be greater than or equal to submit_time; "
                f"invalid_count={int(invalid.sum())}, sample_ids={examples}"
            )
        result["submission_completion_checked"] = True

    if has_start:
        if not (has_submission and has_completion):
            raise ValueError(
                "start_time validation requires complete submission and completion timestamps"
            )
        before_submit = frame["start_time"] < frame["submit_time"]
        after_completion = frame["start_time"] > frame["completion_time"]
        if before_submit.any() or after_completion.any():
            invalid = before_submit | after_completion
            examples = frame.loc[invalid, "sample_id"].astype(str).head(10).tolist()
            raise ValueError(
                "Event times must satisfy submit_time <= start_time <= completion_time; "
                f"invalid_count={int(invalid.sum())}, sample_ids={examples}"
            )
        result["submission_start_completion_checked"] = True

    settings = get_value(config, "data.time_validation", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("data.time_validation must be a mapping")
    unknown = sorted(set(settings) - {"check_elapsed", "tolerance_seconds"})
    if unknown:
        raise ValueError(f"Unsupported data.time_validation keys: {unknown}")
    check_elapsed = bool(settings.get("check_elapsed", False))
    try:
        tolerance_seconds = float(settings.get("tolerance_seconds", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("data.time_validation.tolerance_seconds must be numeric") from exc
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("data.time_validation.tolerance_seconds must be non-negative")
    if check_elapsed:
        if not has_start or not has_completion:
            raise ValueError("Elapsed-time validation requires start and completion timestamps")
        if frame["elapsed_minutes"].isna().any():
            count = int(frame["elapsed_minutes"].isna().sum())
            raise ValueError(f"Configured elapsed column has {count} invalid values")
        observed_seconds = frame["elapsed_minutes"].to_numpy(float) * 60.0
        timestamp_seconds = (
            frame["completion_time"] - frame["start_time"]
        ).dt.total_seconds().to_numpy(float)
        delta = np.abs(observed_seconds - timestamp_seconds)
        invalid = delta > tolerance_seconds
        if invalid.any():
            examples = frame.loc[invalid, "sample_id"].astype(str).head(10).tolist()
            raise ValueError(
                "Elapsed duration does not match completion_time - start_time; "
                f"invalid_count={int(invalid.sum())}, "
                f"max_delta_seconds={float(delta.max()):.6g}, sample_ids={examples}"
            )
        result.update(
            {
                "elapsed_checked": True,
                "elapsed_tolerance_seconds": tolerance_seconds,
                "max_elapsed_delta_seconds": float(delta.max()) if len(delta) else 0.0,
            }
        )
    return result


def _boundary_completion_protocol(
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any], pd.Series]:
    required, policy = _boundary_completion_settings(config)
    actions = pd.Series("keep", index=frame.index, dtype=object)
    base_order_hash = order_fingerprint(frame)
    base_split_hash = split_fingerprint(frame)
    result: Dict[str, Any] = {
        "required": required,
        "policy": policy,
        "protocol_version": BOUNDARY_COMPLETION_PROTOCOL_VERSION,
        "checked": False,
        "train_cross_boundary_count": None,
        "validation_cross_boundary_count": None,
        "first_validation_submit_time": None,
        "first_test_submit_time": None,
        "pre_purge_row_count": int(len(frame)),
        "post_purge_row_count": int(len(frame)),
        "purged_row_count": 0,
        "purged_train_count": 0,
        "purged_validation_count": 0,
        "purged_sample_ids": [],
        "base_split_counts": {
            name: int((frame["split"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "effective_split_counts": {
            name: int((frame["split"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "base_order_fingerprint": base_order_hash,
        "base_split_fingerprint": base_split_hash,
        "effective_order_fingerprint": base_order_hash,
        "effective_split_fingerprint": base_split_hash,
        "boundary_purge_fingerprint": None,
        "remaining_train_cross_boundary_count": None,
        "remaining_validation_cross_boundary_count": None,
    }
    if not required:
        result["boundary_purge_fingerprint"] = _boundary_action_fingerprint(
            frame, actions
        )
        return frame.copy().reset_index(drop=True), result, actions
    columns = require_value(config, "data.columns")
    if (
        str(get_value(config, "split.order_semantics", "unspecified"))
        != "submission_time"
        or columns.get("order") != columns.get("submit_time")
    ):
        raise ValueError(
            "Boundary label-availability checks require a submission-time ordered split"
        )
    if not frame["submit_time"].notna().all() or not frame["completion_time"].notna().all():
        raise ValueError(
            "Boundary label-availability checks require complete submission and completion timestamps"
        )

    validation = frame.loc[frame["split"] == "validation"]
    test = frame.loc[frame["split"] == "test"]
    if validation.empty or test.empty:
        raise ValueError(
            "Boundary label-availability checks require non-empty validation and test splits"
        )
    first_validation_submit = validation["submit_time"].iloc[0]
    first_test_submit = test["submit_time"].iloc[0]
    train_crossing = (
        (frame["split"] == "train")
        & (frame["completion_time"] >= first_validation_submit)
    )
    validation_crossing = validation["completion_time"] >= first_test_submit
    train_count = int(train_crossing.sum())
    validation_count = int(validation_crossing.sum())
    result.update(
        {
            "checked": True,
            "train_cross_boundary_count": train_count,
            "validation_cross_boundary_count": validation_count,
            "first_validation_submit_time": first_validation_submit.isoformat(),
            "first_test_submit_time": first_test_submit.isoformat(),
        }
    )
    if policy == "error" and (train_count or validation_count):
        raise ValueError(
            "Completed-label boundary requirement failed: "
            f"train_jobs_not_completed_before_validation={train_count}, "
            f"validation_jobs_not_completed_before_test={validation_count}"
        )

    if policy == "purge":
        actions.loc[train_crossing] = "purge_train_completion"
        validation_crossing_index = validation.index[validation_crossing]
        actions.loc[validation_crossing_index] = "purge_validation_completion"

    keep = actions == "keep"
    effective = frame.loc[keep].copy().reset_index(drop=True)
    _validate_ordered_effective_splits(effective)

    remaining_train = int(
        (
            (effective["split"] == "train")
            & (effective["completion_time"] >= first_validation_submit)
        ).sum()
    )
    remaining_validation = int(
        (
            (effective["split"] == "validation")
            & (effective["completion_time"] >= first_test_submit)
        ).sum()
    )
    if remaining_train or remaining_validation:
        raise RuntimeError(
            "Boundary purge did not remove every cross-boundary completion"
        )

    purged_ids = frame.loc[~keep, "sample_id"].astype(str).tolist()
    result.update(
        {
            "post_purge_row_count": int(len(effective)),
            "purged_row_count": int((~keep).sum()),
            "purged_train_count": train_count if policy == "purge" else 0,
            "purged_validation_count": (
                validation_count if policy == "purge" else 0
            ),
            "purged_sample_ids": purged_ids,
            "effective_split_counts": {
                name: int((effective["split"] == name).sum())
                for name in ("train", "validation", "test")
            },
            "effective_order_fingerprint": order_fingerprint(effective),
            "effective_split_fingerprint": split_fingerprint(effective),
            "boundary_purge_fingerprint": _boundary_action_fingerprint(
                frame, actions
            ),
            "remaining_train_cross_boundary_count": remaining_train,
            "remaining_validation_cross_boundary_count": remaining_validation,
        }
    )
    return effective, result, actions


def _expected_nonnegative_int(config: Dict[str, Any], key: str) -> Optional[int]:
    value = get_value(config, key)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a non-negative integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return int(numeric)


def _validate_expected_counts(protocol: Dict[str, Any], config: Dict[str, Any]) -> None:
    checks = {
        "data.expected_raw_row_count": protocol["raw_row_count"],
        "data.expected_retained_row_count": protocol["retained_row_count"],
        "data.expected_row_count": protocol["row_count"],
        "data.expected_valid_memory": protocol["valid_memory"],
        "data.expected_valid_vram": protocol["valid_vram"],
    }
    for key, actual in checks.items():
        expected = _expected_nonnegative_int(config, key)
        if expected is not None and actual != expected:
            raise ValueError(f"{key} mismatch: expected={expected}, actual={actual}")

    expected_splits = get_value(config, "split.expected_counts")
    if expected_splits is not None:
        if not isinstance(expected_splits, dict):
            raise ValueError(
                "split.expected_counts must map train/validation/test to counts"
            )
        required = {"train", "validation", "test"}
        if set(expected_splits) != required:
            raise ValueError(
                "split.expected_counts must contain exactly train, validation, and test"
            )
        for split in ("train", "validation", "test"):
            expected = _expected_nonnegative_int(
                {"expected": {split: expected_splits[split]}}, f"expected.{split}"
            )
            if expected is None:
                raise ValueError(f"split.expected_counts.{split} cannot be null")
            actual = int(protocol["split_counts"][split])
            if actual != expected:
                raise ValueError(
                    f"split.expected_counts.{split} mismatch: "
                    f"expected={expected}, actual={actual}"
                )

    expected_valid_by_split = get_value(config, "data.expected_valid_by_split")
    if expected_valid_by_split is not None:
        if not isinstance(expected_valid_by_split, dict) or set(
            expected_valid_by_split
        ) != {"memory", "vram"}:
            raise ValueError(
                "data.expected_valid_by_split must map exactly memory and vram"
            )
        for resource in ("memory", "vram"):
            expected_resource = expected_valid_by_split[resource]
            if not isinstance(expected_resource, dict) or set(expected_resource) != {
                "train",
                "validation",
                "test",
            }:
                raise ValueError(
                    "data.expected_valid_by_split."
                    f"{resource} must map exactly train/validation/test"
                )
            for split in ("train", "validation", "test"):
                expected = _expected_nonnegative_int(
                    {"expected": {split: expected_resource[split]}},
                    f"expected.{split}",
                )
                if expected is None:
                    raise ValueError(
                        f"data.expected_valid_by_split.{resource}.{split} cannot be null"
                    )
                actual = int(protocol["valid_by_split"][resource][split])
                if actual != expected:
                    raise ValueError(
                        "data.expected_valid_by_split."
                        f"{resource}.{split} mismatch: expected={expected}, actual={actual}"
                    )

    expected_split_fingerprint = get_value(config, "split.expected_fingerprint")
    if expected_split_fingerprint is not None:
        expected_split_fingerprint = str(expected_split_fingerprint).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_split_fingerprint):
            raise ValueError("split.expected_fingerprint must be a SHA-256 hex digest")
        actual_split_fingerprint = str(protocol["split_fingerprint"]).lower()
        if actual_split_fingerprint != expected_split_fingerprint:
            raise ValueError(
                "split.expected_fingerprint mismatch: "
                f"expected={expected_split_fingerprint}, actual={actual_split_fingerprint}"
            )

    for key, protocol_key in (
        ("split.expected_order_fingerprint", "order_fingerprint"),
        ("split.expected_manifest_sha256", "split_manifest_sha256"),
    ):
        expected_hash = get_value(config, key)
        if expected_hash is None:
            continue
        expected_hash = str(expected_hash).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"{key} must be a SHA-256 hex digest")
        actual_hash = str(protocol.get(protocol_key) or "").lower()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{key} mismatch: expected={expected_hash}, actual={actual_hash}"
            )


def load_dataset(config: Dict[str, Any]) -> DatasetBundle:
    data_path = resolve_path(config, require_value(config, "data.path"))
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data_sha256 = sha256_file(data_path)
    expected_data_sha256 = get_value(config, "data.expected_sha256")
    if expected_data_sha256 is not None:
        expected_data_sha256 = str(expected_data_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_data_sha256):
            raise ValueError("data.expected_sha256 must be a SHA-256 hex digest")
        if data_sha256.lower() != expected_data_sha256:
            raise ValueError(
                "data.expected_sha256 mismatch: "
                f"expected={expected_data_sha256}, actual={data_sha256.lower()}"
            )
    raw, csv_protocol = _read_csv_strict(data_path, config)
    if raw.empty:
        raise ValueError("Dataset is empty")
    raw, filter_protocol = _apply_row_filters(raw, config)
    frame, numeric_features, categorical_features, standardization_protocol = (
        _standardize_frame(raw, config)
    )
    event_time_protocol = _validate_event_times(frame, config)
    frame, retained_frame, manifest_path, boundary_protocol = _apply_manifest(
        frame, config, data_sha256
    )

    order_semantics = str(get_value(config, "split.order_semantics", "unspecified"))
    configured_columns = require_value(config, "data.columns")
    has_submission_time = bool(frame["submit_time"].notna().all())
    order_matches_submission = bool(
        configured_columns.get("order")
        and configured_columns.get("order") == configured_columns.get("submit_time")
    )
    protocol = {
        "raw_row_count": filter_protocol["raw_row_count"],
        "retained_row_count": filter_protocol["retained_row_count"],
        "dropped_row_count": filter_protocol["dropped_row_count"],
        "dropped_by_filter": filter_protocol["dropped_by_filter"],
        "filters": filter_protocol,
        "strict_csv": csv_protocol,
        "row_count": int(len(frame)),
        "retained_trace_row_count": int(len(retained_frame)),
        "data_sha256": data_sha256,
        "order_semantics": order_semantics,
        "order_column_matches_submission_time": order_matches_submission,
        "verified_chronological": (
            order_semantics == "submission_time"
            and has_submission_time
            and order_matches_submission
        ),
        "has_submission_time": has_submission_time,
        "has_completion_time": bool(frame["completion_time"].notna().all()),
        "split_counts": {key: int(value) for key, value in frame["split"].value_counts().items()},
        "valid_memory": int(frame["valid_memory"].sum()),
        "valid_vram": int(frame["valid_vram"].sum()),
        "valid_by_split": {
            resource: {
                split: int(
                    frame.loc[frame["split"].eq(split), f"valid_{resource}"].sum()
                )
                for split in ("train", "validation", "test")
            }
            for resource in ("memory", "vram")
        },
        "explicit_validity_flags": {
            "memory": bool(configured_columns.get("memory_valid")),
            "vram": bool(configured_columns.get("vram_valid")),
            "memory_column": (
                str(configured_columns["memory_valid"])
                if configured_columns.get("memory_valid")
                else None
            ),
            "vram_column": (
                str(configured_columns["vram_valid"])
                if configured_columns.get("vram_valid")
                else None
            ),
        },
        "out_of_range_memory_masked": int(frame["out_of_range_memory"].sum()),
        "out_of_range_vram_masked": int(frame["out_of_range_vram"].sum()),
        "order_fingerprint": order_fingerprint(frame),
        "split_fingerprint": split_fingerprint(frame),
        "split_manifest_sha256": (
            sha256_file(manifest_path)
            if manifest_path is not None and manifest_path.is_file()
            else None
        ),
        "split_settings": _split_settings(config),
        "completion_after_submission_checked": event_time_protocol[
            "submission_completion_checked"
        ],
        "event_time_validation": event_time_protocol,
        "code_redaction": standardization_protocol["code_redaction"],
        "metadata_parsing": standardization_protocol["metadata_parsing"],
        "duplicate_job_id_rows": standardization_protocol["duplicate_job_id_rows"],
        "unique_job_ids_required": standardization_protocol[
            "unique_job_ids_required"
        ],
        "timestamp_timezone_for_naive_values": standardization_protocol[
            "timestamp_timezone_for_naive_values"
        ],
        "completed_labels_at_boundaries": boundary_protocol,
    }
    _validate_expected_counts(protocol, config)
    return DatasetBundle(
        frame=frame,
        data_path=data_path,
        data_sha256=data_sha256,
        manifest_path=manifest_path,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        protocol=protocol,
        retained_frame=retained_frame,
    )


def write_data_report(bundle: DatasetBundle, path: Union[str, Path]) -> None:
    report = dict(bundle.protocol)
    report["data_path"] = str(bundle.data_path)
    report["manifest_path"] = str(bundle.manifest_path) if bundle.manifest_path else None
    report["numeric_features"] = bundle.numeric_features
    report["categorical_features"] = bundle.categorical_features
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
