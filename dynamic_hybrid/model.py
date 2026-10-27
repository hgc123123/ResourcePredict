from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


RESOURCES = ("memory", "vram")


class ResourceGate(nn.Module):
    """Small reliability gate for one resource; base predictors stay frozen."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.10,
        residual_max: float = 0.05,
        initial_alpha: float = 0.5,
        input_normalization: str = "layernorm",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not math.isfinite(residual_max) or residual_max < 0.0:
            raise ValueError("residual_max must be finite and non-negative")
        if not 0.0 < initial_alpha < 1.0:
            raise ValueError("initial_alpha must be strictly within (0,1)")
        if input_normalization not in {"layernorm", "none"}:
            raise ValueError("input_normalization must be layernorm or none")

        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim)
            if input_normalization == "layernorm"
            else nn.Identity()
        ]
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                [nn.Linear(previous, int(width)), nn.GELU(), nn.Dropout(dropout)]
            )
            previous = int(width)
        self.trunk = nn.Sequential(*layers)
        self.gate_head = nn.Linear(previous, 1)
        self.residual_head = nn.Linear(previous, 1)
        self.residual_max = float(residual_max)

        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(
            self.gate_head.bias,
            math.log(initial_alpha / (1.0 - initial_alpha)),
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        qwen_prediction: torch.Tensor,
        last1_prediction: torch.Tensor,
        history_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        learned_gate = torch.sigmoid(self.gate_head(hidden).squeeze(-1))
        history = history_available.to(dtype=torch.bool)
        gate = torch.where(history, learned_gate, torch.zeros_like(learned_gate))
        if self.residual_max > 0.0:
            learned_residual = self.residual_max * torch.tanh(
                self.residual_head(hidden).squeeze(-1)
            )
        else:
            learned_residual = torch.zeros_like(gate)
        # Cold-start behavior is an exact, auditable Qwen fallback.
        residual = torch.where(history, learned_residual, torch.zeros_like(learned_residual))
        safe_last1 = torch.where(history, last1_prediction, qwen_prediction)
        raw_prediction = gate * safe_last1 + (1.0 - gate) * qwen_prediction + residual
        prediction = torch.clamp(raw_prediction, 0.0, 1.0)
        return {
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "gate": gate,
            "learned_gate": learned_gate,
            "residual": residual,
        }


class DynamicHybridModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.10,
        residual_max: float = 0.05,
        initial_alphas: Mapping[str, float] | None = None,
        input_normalization: str = "layernorm",
    ) -> None:
        super().__init__()
        alphas = dict(initial_alphas or {resource: 0.5 for resource in RESOURCES})
        if set(alphas) != set(RESOURCES):
            raise ValueError("initial_alphas must contain memory and vram")
        self.resource_gates = nn.ModuleDict(
            {
                resource: ResourceGate(
                    input_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    residual_max=residual_max,
                    initial_alpha=float(alphas[resource]),
                    input_normalization=input_normalization,
                )
                for resource in RESOURCES
            }
        )

    def forward(
        self,
        resource: str,
        features: torch.Tensor,
        qwen_prediction: torch.Tensor,
        last1_prediction: torch.Tensor,
        history_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if resource not in self.resource_gates:
            raise ValueError(f"Unknown resource: {resource}")
        return self.resource_gates[resource](
            features, qwen_prediction, last1_prediction, history_available
        )


def dynamic_hybrid_loss(
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    qwen_prediction: torch.Tensor,
    last1_prediction: torch.Tensor,
    history_available: torch.Tensor,
    *,
    smooth_l1_beta: float = 0.05,
    mse_weight: float = 0.25,
    tolerance_weight: float = 0.50,
    tolerance: float = 0.10,
    tolerance_temperature: float = 0.02,
    underprediction_weight: float = 0.05,
    routing_weight: float = 0.05,
    residual_weight: float = 0.01,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    prediction = output["prediction"]
    error = prediction - target
    huber = F.smooth_l1_loss(
        prediction, target, beta=smooth_l1_beta, reduction="mean"
    )
    mse = torch.mean(error.square())
    abs_error = torch.abs(error)
    tolerance_loss = tolerance_temperature * F.softplus(
        (abs_error - tolerance) / tolerance_temperature
    ).mean()
    underprediction = torch.relu(target - prediction).square().mean()

    history = history_available.to(dtype=torch.bool)
    if bool(history.any()):
        qwen_history = qwen_prediction[history]
        last1_history = last1_prediction[history]
        target_history = target[history]
        denominator = last1_history - qwen_history
        stable = torch.abs(denominator) > 1e-6
        route_target = torch.zeros_like(denominator)
        route_target[stable] = torch.clamp(
            (target_history[stable] - qwen_history[stable]) / denominator[stable],
            0.0,
            1.0,
        )
        route_target[~stable] = 0.5
        qwen_error = torch.abs(qwen_history - target_history)
        last1_error = torch.abs(last1_history - target_history)
        route_weight = torch.clamp(
            torch.maximum(
                torch.abs(qwen_error - last1_error),
                torch.abs(denominator),
            )
            / max(tolerance, 1e-8),
            max=1.0,
        )
        route_error = F.smooth_l1_loss(
            output["learned_gate"][history],
            route_target,
            beta=0.10,
            reduction="none",
        )
        routing = torch.sum(route_error * route_weight) / route_weight.sum().clamp_min(1.0)
    else:
        routing = prediction.new_zeros(())
    residual_penalty = torch.mean(output["residual"].square())

    total = (
        huber
        + mse_weight * mse
        + tolerance_weight * tolerance_loss
        + underprediction_weight * underprediction
        + routing_weight * routing
        + residual_weight * residual_penalty
    )
    terms = {
        "loss": total.detach(),
        "huber": huber.detach(),
        "mse": mse.detach(),
        "tolerance": tolerance_loss.detach(),
        "underprediction": underprediction.detach(),
        "routing": routing.detach(),
        "residual": residual_penalty.detach(),
    }
    return total, terms
