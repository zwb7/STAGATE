import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gated_gat_conv import GatedGATConv


class BoundaryAwareEdgeGate(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        gate_dim: Optional[int] = None,
        gamma: float = 1.0,
        beta: float = 2.0,
        eps: float = 1e-8,
        learn_gamma: bool = False,
    ):
        super().__init__()
        gate_dim = gate_dim or embedding_dim
        self.query = nn.Linear(embedding_dim, gate_dim, bias=False)
        self.key = nn.Linear(embedding_dim, gate_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.eps = eps
        self.scale = math.sqrt(gate_dim)

    def forward(
        self,
        embedding: torch.Tensor,
        edge_index: torch.Tensor,
        assignment: torch.Tensor,
        boundary_score: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        q_dst = self.query(embedding[dst])
        k_src = self.key(embedding[src])
        compatibility = (q_dst * k_src).sum(dim=1) / self.scale

        boundary_strength = torch.maximum(boundary_score[src], boundary_score[dst])
        consistency = (assignment[src] * assignment[dst]).sum(dim=1).clamp_min(self.eps)
        logits = compatibility + self.gamma * boundary_strength * torch.log(consistency) + self.beta
        return torch.sigmoid(logits)


class E3v3STAGATE(torch.nn.Module):
    def __init__(
        self,
        hidden_dims,
        gate_dim: Optional[int] = None,
        gate_gamma: float = 1.0,
        gate_beta: float = 2.0,
        learn_gate_gamma: bool = False,
    ):
        super().__init__()

        in_dim, num_hidden, out_dim = hidden_dims
        self.conv1 = GatedGATConv(
            in_dim,
            num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv2 = GatedGATConv(
            num_hidden,
            out_dim,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv3 = GatedGATConv(
            out_dim,
            num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv4 = GatedGATConv(
            num_hidden,
            in_dim,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.edge_gate = BoundaryAwareEdgeGate(
            out_dim,
            gate_dim=gate_dim,
            gamma=gate_gamma,
            beta=gate_beta,
            learn_gamma=learn_gate_gamma,
        )

    def compute_edge_gate(
        self,
        embedding: torch.Tensor,
        edge_index: torch.Tensor,
        assignment: torch.Tensor,
        boundary_score: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_gate(embedding, edge_index, assignment, boundary_score)

    def forward(self, features, edge_index, edge_gate: Optional[torch.Tensor] = None):
        h1 = F.elu(self.conv1(features, edge_index, edge_gate=edge_gate))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True, tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)
        return h2, h4
