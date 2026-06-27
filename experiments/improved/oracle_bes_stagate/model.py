"""Model wrappers for isolated Oracle-BES-STAGATE experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from STAGATE_pyG.STAGATE import STAGATE


class ResidualShapingHead(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, embedding: torch.Tensor, gamma: float) -> torch.Tensor:
        return embedding + gamma * self.net(embedding)


class FrozenEmbeddingShaper(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.shaping_head = ResidualShapingHead(embedding_dim, dropout=dropout)

    def forward(self, embedding: torch.Tensor, gamma: float) -> torch.Tensor:
        return self.shaping_head(embedding, gamma=gamma)


class OracleBESSTAGATE(nn.Module):
    """STAGATE with a residual shaping head between encoder and decoder.

    This class duplicates the baseline forward pass in an experiment-local
    wrapper. It does not modify the official STAGATE_pyG implementation.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        gamma: float = 0.05,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = STAGATE(hidden_dims=hidden_dims)
        self.shaping_head = ResidualShapingHead(hidden_dims[-1], dropout=dropout)
        self.gamma = gamma

    def encode(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = F.elu(self.backbone.conv1(features, edge_index))
        return self.backbone.conv2(h1, edge_index, attention=False)

    def decode(
        self,
        refined_embedding: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        self.backbone.conv3.lin_src.data = self.backbone.conv2.lin_src.transpose(0, 1)
        self.backbone.conv3.lin_dst.data = self.backbone.conv2.lin_dst.transpose(0, 1)
        self.backbone.conv4.lin_src.data = self.backbone.conv1.lin_src.transpose(0, 1)
        self.backbone.conv4.lin_dst.data = self.backbone.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(
            self.backbone.conv3(
                refined_embedding,
                edge_index,
                attention=True,
                tied_attention=self.backbone.conv1.attentions,
            )
        )
        return self.backbone.conv4(h3, edge_index, attention=False)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        apply_shaping: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(features, edge_index)
        refined = (
            self.shaping_head(embedding, gamma=self.gamma)
            if apply_shaping
            else embedding
        )
        reconstruction = self.decode(refined, edge_index)
        return embedding, refined, reconstruction

    def freeze_except_last_encoder_and_shaping(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.conv2.parameters():
            parameter.requires_grad = True
        for parameter in self.shaping_head.parameters():
            parameter.requires_grad = True
