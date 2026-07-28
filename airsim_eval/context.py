"""Shared evaluation state passed to optional analysis modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import AirSimRecord


@dataclass
class EvaluationContext:
    distance_matrix: np.ndarray
    query_features: torch.Tensor
    gallery_features: torch.Tensor
    query_records: list[AirSimRecord]
    gallery_records: list[AirSimRecord]
    reranked: bool = False
    distance_name: str = "euclidean"

    def __post_init__(self) -> None:
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=np.float64)
        expected = (len(self.query_records), len(self.gallery_records))
        if self.distance_matrix.shape != expected:
            raise ValueError(
                f"distance matrix shape {self.distance_matrix.shape} != {expected}"
            )
        if not np.isfinite(self.distance_matrix).all():
            raise ValueError("distance matrix contains NaN or infinite values")

    @property
    def q_pids(self) -> np.ndarray:
        return np.asarray([record.pid for record in self.query_records])

    @property
    def g_pids(self) -> np.ndarray:
        return np.asarray([record.pid for record in self.gallery_records])

    @property
    def q_camids(self) -> np.ndarray:
        return np.asarray([record.camid for record in self.query_records])

    @property
    def g_camids(self) -> np.ndarray:
        return np.asarray([record.camid for record in self.gallery_records])

    @property
    def q_azimuths(self) -> np.ndarray:
        return np.asarray([record.azimuth for record in self.query_records])

    @property
    def g_azimuths(self) -> np.ndarray:
        return np.asarray([record.azimuth for record in self.gallery_records])

    def summary(self) -> dict[str, Any]:
        return {
            "queries": len(self.query_records),
            "gallery": len(self.gallery_records),
            "vehicles": len(set(self.q_pids) | set(self.g_pids)),
            "cameras": len(set(self.q_camids) | set(self.g_camids)),
            "reranked": self.reranked,
            "distance": self.distance_name,
        }

    def ensure_output_dir(self, path: str | Path) -> Path:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        return output
