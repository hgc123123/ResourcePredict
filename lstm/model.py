from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        user_count: int,
        numeric_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        bidirectional: bool,
        user_embedding_dim: int,
        metadata_hidden_dim: int,
        head_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        code_dim = hidden_dim * (2 if bidirectional else 1)
        self.user_embedding = nn.Embedding(user_count, user_embedding_dim)
        self.metadata = nn.Sequential(
            nn.Linear(numeric_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(metadata_hidden_dim),
        )
        fused_dim = code_dim + user_embedding_dim + metadata_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 2),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        user_ids: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.lstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        positions = torch.arange(encoded.shape[1], device=encoded.device).view(1, -1)
        valid = positions < lengths.view(-1, 1)
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min)
        code_features = encoded.max(dim=1).values
        user_features = self.user_embedding(user_ids)
        metadata_features = self.metadata(numeric)
        return self.head(torch.cat([code_features, user_features, metadata_features], dim=-1))


def build_model(arguments: Dict[str, Any]) -> BiLSTMRegressor:
    return BiLSTMRegressor(**arguments)

