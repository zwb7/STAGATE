from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from experiments.soft_gate.model import AdaptiveEdgeGate, GatedSTAGATE

Variant = Literal[
    "baseline",
    "extra_training",
    "current_gate_only",
    "stabilized_unnormalized",
    "stabilized_renormalized",
    "uniform_gate",
    "shuffled_gate",
    "boundary_focused",
    # Backward-compatible aliases from E3-v1. They are mapped in run.py.
    "gate_only",
    "gate_distribution",
    "full",
]


@dataclass
class SoftGateTrainingData:
    data: Data
    edge_pair_id: Tensor
    edge_is_self_loop: Tensor
    pair_node_a: Tensor
    pair_node_b: Tensor
    pair_boundary_candidate: Tensor


@dataclass
class SoftGateTrainingResult:
    adata: sc.AnnData
    warmup_embedding: np.ndarray
    pair_gates: np.ndarray
    effective_degree: np.ndarray
    history: list[dict[str, float | int]]
    final_losses: dict[str, float]
    learned_bias: float | None
    renormalize_gate: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dense_expression(adata: sc.AnnData) -> np.ndarray:
    if "highly_variable" in adata.var.columns:
        adata_vars = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_vars = adata.copy()
    matrix = adata_vars.X
    if sp.issparse(matrix):
        matrix = matrix.todense()
    return np.asarray(matrix, dtype=np.float32)


def build_training_data(
    adata: sc.AnnData,
    edge_priors: pd.DataFrame,
    device: torch.device,
) -> SoftGateTrainingData:
    graph = adata.uns["Spatial_Net"].copy()
    node_to_index = {str(node): index for index, node in enumerate(adata.obs_names)}
    source = graph["Cell1"].astype(str).map(node_to_index)
    target = graph["Cell2"].astype(str).map(node_to_index)
    if source.isna().any() or target.isna().any():
        raise ValueError("Spatial_Net contains spot IDs absent from adata.obs_names")

    adjacency = sp.coo_matrix(
        (
            np.ones(graph.shape[0], dtype=np.float32),
            (source.astype(int), target.astype(int)),
        ),
        shape=(adata.n_obs, adata.n_obs),
    )
    adjacency = adjacency + sp.eye(adata.n_obs, dtype=np.float32)
    row, col = np.nonzero(adjacency)

    pair_lookup: dict[tuple[int, int], int] = {}
    for record in edge_priors.itertuples(index=False):
        pair_lookup[(int(record.node_a_index), int(record.node_b_index))] = int(
            record.pair_id
        )

    edge_pair_id = np.full(row.shape[0], -1, dtype=np.int64)
    edge_is_self_loop = row == col
    for edge_position, (left, right) in enumerate(zip(row, col, strict=True)):
        if left == right:
            continue
        key = (min(int(left), int(right)), max(int(left), int(right)))
        if key not in pair_lookup:
            raise ValueError(f"Spatial edge is absent from edge priors: {key}")
        edge_pair_id[edge_position] = pair_lookup[key]

    expression = _dense_expression(adata)
    data = Data(
        edge_index=torch.as_tensor(np.vstack([row, col]), dtype=torch.long),
        x=torch.as_tensor(expression, dtype=torch.float32),
    ).to(device)

    if "boundary_candidate" in edge_priors:
        boundary_candidate = edge_priors["boundary_candidate"].to_numpy(dtype=bool)
    else:
        boundary_candidate = np.ones(edge_priors.shape[0], dtype=bool)

    return SoftGateTrainingData(
        data=data,
        edge_pair_id=torch.as_tensor(edge_pair_id, dtype=torch.long, device=device),
        edge_is_self_loop=torch.as_tensor(
            edge_is_self_loop,
            dtype=torch.bool,
            device=device,
        ),
        pair_node_a=torch.as_tensor(
            edge_priors["node_a_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        pair_node_b=torch.as_tensor(
            edge_priors["node_b_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        pair_boundary_candidate=torch.as_tensor(
            boundary_candidate,
            dtype=torch.bool,
            device=device,
        ),
    )


def effective_degree_from_pair_gates(
    pair_gates: Tensor,
    pair_node_a: Tensor,
    pair_node_b: Tensor,
    n_nodes: int,
) -> Tensor:
    degree = torch.zeros(n_nodes, dtype=pair_gates.dtype, device=pair_gates.device)
    degree.index_add_(0, pair_node_a, pair_gates)
    degree.index_add_(0, pair_node_b, pair_gates)
    return degree


def pair_gates_from_edge_gates(
    edge_gate: Tensor,
    edge_pair_id: Tensor,
    edge_is_self_loop: Tensor,
    n_pairs: int,
) -> Tensor:
    non_self = ~edge_is_self_loop
    pair_id = edge_pair_id[non_self]
    values = edge_gate[non_self]
    sums = torch.zeros(n_pairs, dtype=values.dtype, device=values.device)
    counts = torch.zeros(n_pairs, dtype=values.dtype, device=values.device)
    sums.index_add_(0, pair_id, values)
    counts.index_add_(0, pair_id, torch.ones_like(values))
    return sums / counts.clamp_min(1.0)


def canonical_variant(variant: Variant) -> Variant:
    # Old E3-v1 names are preserved as aliases so older commands fail less often,
    # but E3-v2 reports should use the explicit ASG variants.
    if variant == "gate_only":
        return "current_gate_only"
    if variant in {"gate_distribution", "full"}:
        return "stabilized_renormalized"
    return variant


def variant_renormalizes(variant: Variant) -> bool:
    return variant in {
        "stabilized_renormalized",
        "uniform_gate",
        "shuffled_gate",
        "boundary_focused",
    }


def build_gate_module(
    variant: Variant,
    embedding_dim: int,
    gate_dim: int,
    g_min: float,
    initial_mean_gate: float,
    temperature: float,
    logit_clip: float,
    device: torch.device,
) -> AdaptiveEdgeGate | None:
    if variant in {"baseline", "extra_training", "uniform_gate"}:
        return None
    centered = variant in {
        "stabilized_unnormalized",
        "stabilized_renormalized",
        "shuffled_gate",
        "boundary_focused",
    }
    bounded = centered
    boundary_focused = variant == "boundary_focused"
    gate = AdaptiveEdgeGate(
        embedding_dim,
        gate_dim,
        centered=centered,
        bounded=bounded,
        g_min=g_min,
        initial_mean_gate=initial_mean_gate,
        temperature=temperature,
        logit_clip=logit_clip,
        boundary_focused=boundary_focused,
    ).to(device)
    if variant == "current_gate_only":
        with torch.no_grad():
            gate.bias.zero_()
    return gate


def train_soft_gate_stagate(
    adata: sc.AnnData,
    edge_priors: pd.DataFrame,
    warmup_embedding: np.ndarray | None,
    *,
    variant: Variant,
    hidden_dims: list[int],
    warmup_epochs: int,
    gate_epochs: int,
    lr: float,
    weight_decay: float,
    gradient_clipping: float,
    gate_dim: int,
    rho: float,
    lambda_budget: float,
    key_added: str,
    random_seed: int,
    save_loss: bool,
    save_reconstruction: bool,
    device: torch.device,
    g_min: float = 0.80,
    initial_mean_gate: float = 0.95,
    temperature: float = 2.0,
    logit_clip: float = 5.0,
) -> SoftGateTrainingResult:
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if gate_epochs < 0:
        raise ValueError("gate_epochs must be non-negative")
    if len(hidden_dims) != 2:
        raise ValueError("hidden_dims must contain [hidden_dim, latent_dim]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")

    variant = canonical_variant(variant)
    no_gate_mode = variant in {"baseline", "extra_training"} or gate_epochs == 0
    renormalize_gate = False if no_gate_mode else variant_renormalizes(variant)
    set_seed(random_seed)
    adata.X = sp.csr_matrix(adata.X)
    training_data = build_training_data(adata, edge_priors, device)
    model = GatedSTAGATE(
        hidden_dims=[training_data.data.x.shape[1]] + hidden_dims
    ).to(device)

    parameters: list[torch.nn.Parameter] = list(model.parameters())
    warmup_history: list[dict[str, float | int | str]] = []
    if warmup_epochs > 0:
        warmup_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        for epoch in tqdm(range(1, warmup_epochs + 1), desc="warm-up"):
            model.train()
            warmup_optimizer.zero_grad()
            latent, reconstructed = model(
                training_data.data.x,
                training_data.data.edge_index,
                edge_gate=None,
                renormalize_gate=False,
            )
            del latent
            reconstruction_loss = F.mse_loss(training_data.data.x, reconstructed)
            reconstruction_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clipping)
            warmup_optimizer.step()
            warmup_history.append(
                {
                    "stage": "warmup",
                    "epoch": epoch,
                    "total_loss": float(reconstruction_loss.detach().cpu()),
                    "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                    "budget_loss": 0.0,
                    "mean_gate": 1.0,
                    "std_gate": 0.0,
                }
            )
        model.eval()
        with torch.no_grad():
            warmup_latent, _ = model(
                training_data.data.x,
                training_data.data.edge_index,
                edge_gate=None,
                renormalize_gate=False,
            )
        warmup_embedding = warmup_latent.detach().cpu().numpy()

    if warmup_embedding is None:
        raise ValueError(
            "warmup_embedding is required when warmup_epochs is 0. "
            "Use --warmup-epochs > 0 to compute it inside this run."
        )
    warmup_tensor = torch.as_tensor(warmup_embedding, dtype=torch.float32, device=device)
    gate_module = None if no_gate_mode else build_gate_module(
        variant,
        warmup_tensor.shape[1],
        gate_dim,
        g_min,
        initial_mean_gate,
        temperature,
        logit_clip,
        device,
    )
    if gate_module is not None:
        parameters.extend(gate_module.parameters())

    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float | int | str]] = list(warmup_history)
    iterator = tqdm(range(1, gate_epochs + 1), desc="gate")
    n_pairs = edge_priors.shape[0]
    non_self = ~training_data.edge_is_self_loop
    shuffle_permutation = torch.randperm(int(non_self.sum()), device=device)

    if warmup_history:
        last_warmup = warmup_history[-1]
        last_losses: dict[str, Tensor] = {
            "total_loss": torch.tensor(float(last_warmup["total_loss"]), device=device),
            "reconstruction_loss": torch.tensor(
                float(last_warmup["reconstruction_loss"]),
                device=device,
            ),
            "budget_loss": torch.tensor(0.0, device=device),
        }
    else:
        last_losses = {}
    final_edge_gate = None
    for epoch in iterator:
        model.train()
        if gate_module is not None:
            gate_module.train()
        optimizer.zero_grad()

        if no_gate_mode:
            edge_gate = torch.ones(
                training_data.data.edge_index.shape[1],
                dtype=torch.float32,
                device=device,
            )
        elif variant == "uniform_gate":
            edge_gate = torch.ones(
                training_data.data.edge_index.shape[1],
                dtype=torch.float32,
                device=device,
            )
            edge_gate[non_self] = initial_mean_gate
        else:
            if gate_module is None:
                raise RuntimeError("Gate module was not initialized")
            edge_gate = gate_module.edge_gates(
                warmup_tensor,
                training_data.data.edge_index,
                training_data.edge_pair_id,
                training_data.edge_is_self_loop,
                training_data.pair_boundary_candidate,
            )
            if variant == "shuffled_gate":
                shuffled = edge_gate.clone()
                non_self_values = edge_gate[non_self]
                shuffled[non_self] = non_self_values[shuffle_permutation]
                edge_gate = shuffled

        pair_gate = pair_gates_from_edge_gates(
            edge_gate,
            training_data.edge_pair_id,
            training_data.edge_is_self_loop,
            n_pairs,
        )
        latent, reconstructed = model(
            training_data.data.x,
            training_data.data.edge_index,
            edge_gate=edge_gate,
            renormalize_gate=renormalize_gate,
        )
        del latent
        reconstruction_loss = F.mse_loss(training_data.data.x, reconstructed)
        if no_gate_mode:
            budget_loss = torch.zeros((), dtype=torch.float32, device=device)
        else:
            budget_loss = (torch.mean(1.0 - pair_gate) - rho).pow(2)
        total_loss = reconstruction_loss + lambda_budget * budget_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clipping)
        optimizer.step()

        last_losses = {
            "total_loss": total_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "budget_loss": budget_loss.detach(),
        }
        history.append(
            {
                "stage": "gate",
                "epoch": epoch,
                "total_loss": float(last_losses["total_loss"].cpu()),
                "reconstruction_loss": float(
                    last_losses["reconstruction_loss"].cpu()
                ),
                "budget_loss": float(last_losses["budget_loss"].cpu()),
                "mean_gate": float(pair_gate.detach().mean().cpu()),
                "std_gate": float(pair_gate.detach().std().cpu()),
            }
        )
        final_edge_gate = edge_gate.detach()

    model.eval()
    if gate_module is not None:
        gate_module.eval()
    with torch.no_grad():
        if no_gate_mode:
            final_edge_gate = torch.ones(
                training_data.data.edge_index.shape[1],
                dtype=torch.float32,
                device=device,
            )
        elif variant == "uniform_gate":
            final_edge_gate = torch.ones(
                training_data.data.edge_index.shape[1],
                dtype=torch.float32,
                device=device,
            )
            final_edge_gate[non_self] = initial_mean_gate
        else:
            if gate_module is None:
                raise RuntimeError("Gate module was not initialized")
            final_edge_gate = gate_module.edge_gates(
                warmup_tensor,
                training_data.data.edge_index,
                training_data.edge_pair_id,
                training_data.edge_is_self_loop,
                training_data.pair_boundary_candidate,
            )
            if variant == "shuffled_gate":
                shuffled = final_edge_gate.clone()
                non_self_values = final_edge_gate[non_self]
                shuffled[non_self] = non_self_values[shuffle_permutation]
                final_edge_gate = shuffled

        final_pair_gate = pair_gates_from_edge_gates(
            final_edge_gate,
            training_data.edge_pair_id,
            training_data.edge_is_self_loop,
            n_pairs,
        )
        latent, reconstructed = model(
            training_data.data.x,
            training_data.data.edge_index,
            edge_gate=final_edge_gate,
            renormalize_gate=renormalize_gate,
        )
        del reconstructed
        effective_degree = effective_degree_from_pair_gates(
            final_pair_gate,
            training_data.pair_node_a,
            training_data.pair_node_b,
            adata.n_obs,
        )

    adata.obsm[key_added] = latent.detach().cpu().numpy()
    final_pair_gate_np = final_pair_gate.detach().cpu().numpy()
    effective_degree_np = effective_degree.detach().cpu().numpy()
    if save_loss:
        adata.uns["STAGATE_loss"] = float(
            last_losses.get("reconstruction_loss", torch.tensor(float("nan"))).cpu()
        )
    if save_reconstruction:
        with torch.no_grad():
            _, final_reconstruction = model(
                training_data.data.x,
                training_data.data.edge_index,
                edge_gate=final_edge_gate,
                renormalize_gate=renormalize_gate,
            )
        reconstruction = final_reconstruction.detach().cpu().numpy()
        reconstruction[reconstruction < 0] = 0
        adata.layers["STAGATE_ReX"] = reconstruction

    final_losses = {key: float(value.cpu()) for key, value in last_losses.items()}
    return SoftGateTrainingResult(
        adata=adata,
        warmup_embedding=warmup_embedding,
        pair_gates=final_pair_gate_np,
        effective_degree=effective_degree_np,
        history=history,
        final_losses=final_losses,
        learned_bias=(
            float(gate_module.bias.detach().cpu()) if gate_module is not None else None
        ),
        renormalize_gate=renormalize_gate,
    )