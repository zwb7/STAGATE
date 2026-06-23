from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class EdgeReliabilityMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class ScorerTrainingResult:
    model: EdgeReliabilityMLP
    scores: np.ndarray
    history: list[dict[str, float | int]]
    train_size: int
    validation_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_edge_scorer(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    validation_ratio: float,
    seed: int,
    device: torch.device,
) -> ScorerTrainingResult:
    if features.ndim != 2:
        raise ValueError(f"Features must be two-dimensional: {features.shape}")
    if targets.shape != (features.shape[0],):
        raise ValueError(
            f"Targets must have shape ({features.shape[0]},), got {targets.shape}"
        )
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")

    set_seed(seed)
    permutation = np.random.default_rng(seed).permutation(features.shape[0])
    validation_size = max(1, int(round(features.shape[0] * validation_ratio)))
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    if train_indices.size == 0:
        raise ValueError("Not enough edges for the requested validation split")

    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    train_index_tensor = torch.as_tensor(
        train_indices,
        dtype=torch.long,
        device=device,
    )
    validation_index_tensor = torch.as_tensor(
        validation_indices,
        dtype=torch.long,
        device=device,
    )

    model = EdgeReliabilityMLP(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(feature_tensor[train_index_tensor])
        train_loss = criterion(
            train_logits,
            target_tensor[train_index_tensor],
        )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(feature_tensor[validation_index_tensor])
            validation_loss = criterion(
                validation_logits,
                target_tensor[validation_index_tensor],
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.detach().cpu()),
                "validation_loss": float(validation_loss.detach().cpu()),
            }
        )

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(feature_tensor)).detach().cpu().numpy()
    return ScorerTrainingResult(
        model=model,
        scores=scores,
        history=history,
        train_size=int(train_indices.size),
        validation_size=int(validation_indices.size),
    )

