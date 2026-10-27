from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+|[^\s]")


def tokenize_code(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text))


@dataclass
class NumericScaler:
    columns: List[str]
    medians: List[float]
    means: List[float]
    scales: List[float]
    standardize: bool = True

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        columns: Sequence[str],
        standardize: bool = True,
    ) -> "NumericScaler":
        values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(values), values, medians)
        if standardize:
            means = np.mean(filled, axis=0)
            scales = np.std(filled, axis=0)
            scales = np.where(scales > 1e-12, scales, 1.0)
        else:
            means = np.zeros(filled.shape[1], dtype=float)
            scales = np.ones(filled.shape[1], dtype=float)
        return cls(
            list(columns),
            medians.tolist(),
            means.tolist(),
            scales.tolist(),
            bool(standardize),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        medians = np.asarray(self.medians)
        values = np.where(np.isfinite(values), values, medians)
        return ((values - np.asarray(self.means)) / np.asarray(self.scales)).astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NumericScaler":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


class TokenVocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        max_size: int,
        min_frequency: int,
    ) -> "TokenVocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize_code(str(text)))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        kept = [token for token, count in ordered if count >= min_frequency]
        kept = kept[: max(0, max_size - 2)]
        mapping = {cls.PAD: 0, cls.UNK: 1}
        mapping.update({token: index + 2 for index, token in enumerate(kept)})
        return cls(mapping)

    def encode(self, text: str, max_tokens: int) -> List[int]:
        ids = [self.token_to_id.get(token, 1) for token in tokenize_code(str(text))[:max_tokens]]
        return ids or [1]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.token_to_id, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TokenVocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))


class UserVocabulary:
    UNK = "<UNK>"

    def __init__(self, user_to_id: Dict[str, int]):
        self._validate_mapping(user_to_id)
        self.user_to_id = dict(user_to_id)

    @classmethod
    def _validate_mapping(cls, mapping: Dict[str, int]) -> None:
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("User vocabulary must be a non-empty mapping")
        if mapping.get(cls.UNK) != 0:
            raise ValueError(f"Reserved user token {cls.UNK!r} must have index 0")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in mapping.values()
        ):
            raise ValueError("User vocabulary indices must be integers")
        expected = list(range(len(mapping)))
        if sorted(mapping.values()) != expected:
            raise ValueError(
                "User vocabulary indices must be unique and contiguous from 0"
            )

    @classmethod
    def fit(cls, users: Iterable[str]) -> "UserVocabulary":
        unique = sorted({str(user) for user in users if str(user) != cls.UNK})
        mapping = {cls.UNK: 0}
        mapping.update({user: index + 1 for index, user in enumerate(unique)})
        return cls(mapping)

    def encode(self, users: Iterable[str]) -> np.ndarray:
        return np.asarray(
            [self.user_to_id.get(str(user), 0) for user in users], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.user_to_id)

    def save(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.user_to_id, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "UserVocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))
