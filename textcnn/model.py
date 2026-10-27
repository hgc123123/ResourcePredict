from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn


class TextCNNRegressor(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        user_count: int,
        numeric_dim: int,
        embedding_dim: int,
        kernel_sizes: Sequence[int],
        filters_per_kernel: int,
        user_embedding_dim: int,
        metadata_hidden_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, filters_per_kernel, kernel_size=kernel, bias=False)
                for kernel in self.kernel_sizes
            ]
        )
        self.user_embedding = nn.Embedding(user_count, user_embedding_dim)
        self.metadata = nn.Sequential(
            nn.Linear(numeric_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(metadata_hidden_dim),
        )
        fused_dim = filters_per_kernel * len(self.kernel_sizes) + user_embedding_dim + metadata_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        user_ids: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for kernel, convolution in zip(self.kernel_sizes, self.convolutions):
            features = torch.relu(convolution(embedded))
            valid_windows = (lengths - kernel + 1).clamp_min(1)
            positions = torch.arange(features.shape[-1], device=features.device).view(1, 1, -1)
            valid = positions < valid_windows.view(-1, 1, 1)
            features = features.masked_fill(~valid, torch.finfo(features.dtype).min)
            pooled.append(features.max(dim=-1).values)
        code_features = torch.cat(pooled, dim=-1)
        user_features = self.user_embedding(user_ids)
        metadata_features = self.metadata(numeric)
        return self.head(torch.cat([code_features, user_features, metadata_features], dim=-1))


def build_model(arguments: Dict[str, Any]) -> TextCNNRegressor:
    return TextCNNRegressor(**arguments)

