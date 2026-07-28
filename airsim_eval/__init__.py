"""Modular AirSim vehicle-ReID evaluation helpers."""

from .context import EvaluationContext
from .dataset import AirSimDataset, AirSimRecord, build_airsim_datasets

__all__ = [
    "AirSimDataset",
    "AirSimRecord",
    "EvaluationContext",
    "build_airsim_datasets",
]
