"""Core DiDi-Merging algorithm."""

from .core import (
    DiDiModel,
    FactorizedMatrix,
    allocate_ranks,
    build_task_vectors,
    fit_didi,
)

__all__ = [
    "DiDiModel",
    "FactorizedMatrix",
    "allocate_ranks",
    "build_task_vectors",
    "fit_didi",
]
