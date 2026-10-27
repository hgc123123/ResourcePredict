"""History-first reliability-gated cascade for frozen Qwen predictions."""

from .history import HISTORY_CONTRACT_VERSION, build_robust_history_long

def __getattr__(name):
    # Causal history/protocol utilities do not require sklearn. Preserve the
    # public estimator exports while importing them only when actually used.
    if name in {"CascadeConfig", "ResourceCascade"}:
        from .model import CascadeConfig, ResourceCascade
        return {"CascadeConfig": CascadeConfig, "ResourceCascade": ResourceCascade}[name]
    raise AttributeError(name)

__all__ = [
    "CascadeConfig",
    "HISTORY_CONTRACT_VERSION",
    "ResourceCascade",
    "build_robust_history_long",
]

