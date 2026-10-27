from __future__ import annotations

from typing import Any


BOUNDARY_COMPLETION_PROTOCOL_VERSION = "completion-before-next-submit/v1"


def normalize_boundary_completion_policy(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("split.boundary_completion_policy must be error or purge")
    policy = value.strip().lower()
    if policy not in {"error", "purge"}:
        raise ValueError("split.boundary_completion_policy must be error or purge")
    return policy
