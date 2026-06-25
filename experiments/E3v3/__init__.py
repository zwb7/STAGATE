"""Isolated E3v3 experiment package."""

from .boundary import (
    BoundaryMiningResult,
    boundary_margin_loss,
    compute_boundary_scores,
    compute_domain_prototypes,
    gate_budget_loss,
    preserve_loss,
)
from .model import E3v3STAGATE
from .train_e3v3 import E3v3Config, train_e3v3

__all__ = [
    "BoundaryMiningResult",
    "E3v3Config",
    "E3v3STAGATE",
    "boundary_margin_loss",
    "compute_boundary_scores",
    "compute_domain_prototypes",
    "gate_budget_loss",
    "preserve_loss",
    "train_e3v3",
]
