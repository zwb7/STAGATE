"""Loss functions for Oracle-BES-STAGATE."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_prototypes(
    embedding: torch.Tensor,
    labels: np.ndarray,
    core_masks: dict[int, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    prototype_labels = sorted(core_masks)
    prototypes = []
    for label in prototype_labels:
        indices_np = np.flatnonzero(core_masks[label])
        if indices_np.size == 0:
            raise ValueError(f"No core spots available for label {label}")
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        prototypes.append(embedding.detach().index_select(0, indices).mean(dim=0))
    prototype_tensor = torch.stack(prototypes, dim=0).detach()
    label_to_proto = {label: index for index, label in enumerate(prototype_labels)}
    return prototype_tensor, prototype_labels, label_to_proto


def target_indices_for_spots(
    spot_indices: np.ndarray,
    labels: np.ndarray,
    label_to_proto: dict[int, int],
) -> np.ndarray:
    return np.asarray(
        [label_to_proto[int(labels[index])] for index in spot_indices],
        dtype=np.int64,
    )


def all_domain_prototype_loss(
    refined_embedding: torch.Tensor,
    prototypes: torch.Tensor,
    train_indices: torch.Tensor,
    target_proto_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    z_train = F.normalize(refined_embedding.index_select(0, train_indices), dim=1)
    prototype_norm = F.normalize(prototypes, dim=1)
    logits = z_train @ prototype_norm.T / temperature
    return F.cross_entropy(logits, target_proto_indices)


def adjacent_domain_prototype_loss(
    refined_embedding: torch.Tensor,
    prototypes: torch.Tensor,
    train_indices_np: np.ndarray,
    train_indices: torch.Tensor,
    target_proto_indices: torch.Tensor,
    adjacent_negative_labels: list[list[int]],
    label_to_proto: dict[int, int],
    temperature: float,
) -> torch.Tensor:
    z_train = F.normalize(refined_embedding.index_select(0, train_indices), dim=1)
    prototype_norm = F.normalize(prototypes, dim=1)
    losses = []
    for row_index, spot_index in enumerate(train_indices_np):
        positive_proto = int(target_proto_indices[row_index].item())
        negative_proto_indices = [
            label_to_proto[label]
            for label in adjacent_negative_labels[int(spot_index)]
            if label in label_to_proto and label_to_proto[label] != positive_proto
        ]
        if not negative_proto_indices:
            continue
        candidates = torch.as_tensor(
            [positive_proto] + negative_proto_indices,
            dtype=torch.long,
            device=refined_embedding.device,
        )
        logits = (
            z_train[row_index : row_index + 1]
            @ prototype_norm.index_select(0, candidates).T
            / temperature
        )
        target = torch.zeros(1, dtype=torch.long, device=refined_embedding.device)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return refined_embedding.sum() * 0.0
    return torch.stack(losses).mean()


def interior_preservation_loss(
    original_embedding: torch.Tensor,
    refined_embedding: torch.Tensor,
    interior_indices: torch.Tensor,
) -> torch.Tensor:
    if interior_indices.numel() == 0:
        return refined_embedding.sum() * 0.0
    original = original_embedding.detach().index_select(0, interior_indices)
    refined = refined_embedding.index_select(0, interior_indices)
    return torch.mean(torch.sum((refined - original) ** 2, dim=1))
