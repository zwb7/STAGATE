from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class BoundaryMiningResult:
    scores: torch.Tensor
    mask: torch.Tensor
    margin: torch.Tensor
    local_shift: torch.Tensor
    threshold: torch.Tensor


def _validate_assignment(assignment: torch.Tensor) -> torch.Tensor:
    if assignment.dim() != 2:
        raise ValueError("assignment must be a 2D tensor with shape [num_nodes, num_domains].")
    if assignment.size(1) < 2:
        raise ValueError("assignment must contain at least two domains.")
    return assignment


def compute_assignment_margin(assignment: torch.Tensor) -> torch.Tensor:
    assignment = _validate_assignment(assignment)
    top2 = torch.topk(assignment, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def compute_local_domain_shift(
    assignment: torch.Tensor,
    edge_index: torch.Tensor,
    ignore_self_loops: bool = True,
) -> torch.Tensor:
    assignment = _validate_assignment(assignment)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    src = edge_index[0]
    dst = edge_index[1]
    if ignore_self_loops:
        keep = src != dst
        src = src[keep]
        dst = dst[keep]

    num_nodes = assignment.size(0)
    if src.numel() == 0:
        return assignment.new_zeros(num_nodes)

    consistency = (assignment[src] * assignment[dst]).sum(dim=1)
    edge_shift = 1.0 - consistency

    shift_sum = assignment.new_zeros(num_nodes)
    degree = assignment.new_zeros(num_nodes)
    shift_sum.index_add_(0, dst, edge_shift)
    degree.index_add_(0, dst, torch.ones_like(edge_shift))
    return shift_sum / degree.clamp_min(1.0)


def compute_boundary_scores(
    assignment: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 0.5,
    top_q: float = 0.10,
    ignore_self_loops: bool = True,
) -> BoundaryMiningResult:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if not 0.0 < top_q < 1.0:
        raise ValueError("top_q must be in (0, 1).")

    margin = compute_assignment_margin(assignment)
    local_shift = compute_local_domain_shift(
        assignment,
        edge_index,
        ignore_self_loops=ignore_self_loops,
    )
    scores = alpha * (1.0 - margin) + (1.0 - alpha) * local_shift

    num_nodes = scores.numel()
    top_k = max(1, int(round(num_nodes * top_q)))
    top_values = torch.topk(scores, k=top_k, largest=True).values
    threshold = top_values.min()
    mask = scores >= threshold

    return BoundaryMiningResult(
        scores=scores,
        mask=mask,
        margin=margin,
        local_shift=local_shift,
        threshold=threshold,
    )


def compute_domain_prototypes(
    embedding: torch.Tensor,
    assignment: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    assignment = _validate_assignment(assignment)
    weights = assignment / assignment.sum(dim=0, keepdim=True).clamp_min(eps)
    return weights.transpose(0, 1).matmul(embedding)


def boundary_margin_loss(
    embedding: torch.Tensor,
    assignment: torch.Tensor,
    boundary_mask: torch.Tensor,
    margin: float = 1.0,
    detach_prototypes: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    assignment = _validate_assignment(assignment)
    if boundary_mask.dtype != torch.bool:
        boundary_mask = boundary_mask.bool()
    if boundary_mask.sum() == 0:
        return embedding.sum() * 0.0

    prototypes = compute_domain_prototypes(embedding, assignment, eps=eps)
    if detach_prototypes:
        prototypes = prototypes.detach()

    positive_domain = assignment.argmax(dim=1)
    masked_assignment = assignment.clone()
    masked_assignment[torch.arange(assignment.size(0), device=assignment.device), positive_domain] = -1.0
    negative_domain = masked_assignment.argmax(dim=1)

    boundary_embedding = embedding[boundary_mask]
    pos_proto = prototypes[positive_domain[boundary_mask]]
    neg_proto = prototypes[negative_domain[boundary_mask]]

    pos_dist = (boundary_embedding - pos_proto).pow(2).sum(dim=1)
    neg_dist = (boundary_embedding - neg_proto).pow(2).sum(dim=1)
    return F.relu(margin + pos_dist - neg_dist).mean()


def gate_budget_loss(edge_gate: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    return ((1.0 - edge_gate).mean() - rho).pow(2)


def preserve_loss(
    edge_gate: torch.Tensor,
    assignment: torch.Tensor,
    edge_index: torch.Tensor,
    boundary_score: torch.Tensor,
    consistency_threshold: float = 0.90,
    boundary_threshold: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assignment = _validate_assignment(assignment)
    src = edge_index[0]
    dst = edge_index[1]

    consistency = (assignment[src] * assignment[dst]).sum(dim=1)
    if boundary_threshold is None:
        boundary_threshold = torch.quantile(boundary_score.detach(), 0.90)

    preserve_mask = (
        (consistency > consistency_threshold)
        & (boundary_score[src] < boundary_threshold)
        & (boundary_score[dst] < boundary_threshold)
    )
    if preserve_mask.sum() == 0:
        return edge_gate.sum() * 0.0
    return (1.0 - edge_gate[preserve_mask]).pow(2).mean()
