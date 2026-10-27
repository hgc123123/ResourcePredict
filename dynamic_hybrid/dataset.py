from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ResourceArrays:
    features: np.ndarray
    qwen_prediction: np.ndarray
    last1_prediction: np.ndarray
    history_available: np.ndarray
    target: np.ndarray
    row_index: np.ndarray


class ResourceGateDataset(Dataset):
    def __init__(self, arrays: ResourceArrays) -> None:
        lengths = {
            len(arrays.features),
            len(arrays.qwen_prediction),
            len(arrays.last1_prediction),
            len(arrays.history_available),
            len(arrays.target),
            len(arrays.row_index),
        }
        if len(lengths) != 1:
            raise ValueError("Resource gate array lengths differ")
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.as_tensor(self.arrays.features[index], dtype=torch.float32),
            "qwen_prediction": torch.tensor(
                self.arrays.qwen_prediction[index], dtype=torch.float32
            ),
            "last1_prediction": torch.tensor(
                self.arrays.last1_prediction[index], dtype=torch.float32
            ),
            "history_available": torch.tensor(
                self.arrays.history_available[index], dtype=torch.bool
            ),
            "target": torch.tensor(self.arrays.target[index], dtype=torch.float32),
            "row_index": torch.tensor(self.arrays.row_index[index], dtype=torch.long),
        }

