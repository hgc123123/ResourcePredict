"""Causal, resource-specific dynamic fusion of Qwen and Last-1 predictions."""

from .model import DynamicHybridModel, ResourceGate

__all__ = ["DynamicHybridModel", "ResourceGate"]
