import json
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from STAGATE_pyG.utils import Transfer_pytorch_Data

from .boundary import (
    boundary_margin_loss,
    compute_boundary_scores,
    gate_budget_loss,
    preserve_loss,
)
from .model import E3v3STAGATE


@dataclass
class E3v3Config:
    hidden_dims: List[int] = None
    warmup_epochs: int = 500
    stage2_epochs: int = 500
    lr: float = 0.001
    weight_decay: float = 0.0001
    gradient_clipping: float = 5.0
    boundary_alpha: float = 0.5
    boundary_top_q: float = 0.10
    prototype_margin: float = 1.0
    lambda_boundary: float = 0.1
    lambda_gate: float = 0.01
    lambda_preserve: float = 0.1
    gate_rho: float = 0.05
    gate_gamma: float = 1.0
    gate_beta: float = 2.0
    gate_dim: Optional[int] = None
    learn_gate_gamma: bool = False
    preserve_consistency_threshold: float = 0.90
    assignment_method: str = "gmm"
    update_gate_embedding: bool = True
    detach_prototypes: bool = False
    random_seed: int = 0
    verbose: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 30]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _softmax_numpy(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)


def soft_assignment_from_embedding(
    embedding: np.ndarray,
    n_clusters: int,
    method: str = "gmm",
    random_seed: int = 0,
) -> np.ndarray:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if method not in {"gmm", "kmeans"}:
        raise ValueError("assignment_method must be 'gmm' or 'kmeans'.")

    if method == "gmm":
        try:
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                random_state=random_seed,
            )
            model.fit(embedding)
            return model.predict_proba(embedding)
        except Exception:
            if method == "gmm":
                method = "kmeans"

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20)
    kmeans.fit(embedding)
    distances = pairwise_distances(embedding, kmeans.cluster_centers_, metric="sqeuclidean")
    return _softmax_numpy(-distances, axis=1)


def _prepare_data(adata):
    adata.X = sp.csr_matrix(adata.X)
    if "highly_variable" in adata.var.columns:
        adata_vars = adata[:, adata.var["highly_variable"]]
    else:
        adata_vars = adata
    if "Spatial_Net" not in adata.uns:
        raise ValueError("Spatial_Net is not existed. Run Cal_Spatial_Net first.")
    return Transfer_pytorch_Data(adata_vars)


def _loss_dict_to_float(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}


def train_e3v3(
    adata,
    n_clusters: int,
    key_added: str = "E3v3",
    config: Optional[E3v3Config] = None,
    save_loss: bool = True,
    save_reconstruction: bool = False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """Train isolated E3v3 without modifying the official STAGATE baseline."""

    config = config or E3v3Config()
    seed_everything(config.random_seed)

    data = _prepare_data(adata).to(device)
    if config.verbose:
        print("Size of Input: ", tuple(data.x.shape))

    model = E3v3STAGATE(
        hidden_dims=[data.x.shape[1]] + config.hidden_dims,
        gate_dim=config.gate_dim,
        gate_gamma=config.gate_gamma,
        gate_beta=config.gate_beta,
        learn_gate_gamma=config.learn_gate_gamma,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    warmup_losses = []
    iterator = range(1, config.warmup_epochs + 1)
    if config.verbose:
        iterator = tqdm(iterator, desc="E3v3 warm-up")
    for _ in iterator:
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index, edge_gate=None)
        loss = F.mse_loss(data.x, out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        optimizer.step()
        warmup_losses.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        z0, _ = model(data.x, data.edge_index, edge_gate=None)

    assignment_np = soft_assignment_from_embedding(
        z0.detach().cpu().numpy(),
        n_clusters=n_clusters,
        method=config.assignment_method,
        random_seed=config.random_seed,
    )
    assignment = torch.as_tensor(assignment_np, dtype=data.x.dtype, device=device)
    boundary = compute_boundary_scores(
        assignment,
        data.edge_index,
        alpha=config.boundary_alpha,
        top_q=config.boundary_top_q,
    )
    gate_embedding = z0.detach()

    stage2_losses = []
    last_edge_gate = None
    iterator = range(1, config.stage2_epochs + 1)
    if config.verbose:
        iterator = tqdm(iterator, desc="E3v3 stage-2")
    for _ in iterator:
        model.train()
        optimizer.zero_grad()

        edge_gate = model.compute_edge_gate(
            gate_embedding,
            data.edge_index,
            assignment,
            boundary.scores,
        )
        z, out = model(data.x, data.edge_index, edge_gate=edge_gate)

        losses = {
            "rec": F.mse_loss(data.x, out),
            "boundary": boundary_margin_loss(
                z,
                assignment,
                boundary.mask,
                margin=config.prototype_margin,
                detach_prototypes=config.detach_prototypes,
            ),
            "gate": gate_budget_loss(edge_gate, rho=config.gate_rho),
            "preserve": preserve_loss(
                edge_gate,
                assignment,
                data.edge_index,
                boundary.scores,
                consistency_threshold=config.preserve_consistency_threshold,
                boundary_threshold=boundary.threshold,
            ),
        }
        loss = (
            losses["rec"]
            + config.lambda_boundary * losses["boundary"]
            + config.lambda_gate * losses["gate"]
            + config.lambda_preserve * losses["preserve"]
        )
        losses["total"] = loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        optimizer.step()

        if config.update_gate_embedding:
            gate_embedding = z.detach()
        last_edge_gate = edge_gate.detach()
        stage2_losses.append(_loss_dict_to_float(losses))

    model.eval()
    with torch.no_grad():
        if last_edge_gate is None:
            final_gate = model.compute_edge_gate(gate_embedding, data.edge_index, assignment, boundary.scores)
        else:
            final_gate = last_edge_gate
        z, out = model(data.x, data.edge_index, edge_gate=final_gate)

    adata.obsm[key_added] = z.detach().cpu().numpy()
    adata.obsm[key_added + "_warmup"] = z0.detach().cpu().numpy()
    adata.obsm[key_added + "_soft_assignment"] = assignment.detach().cpu().numpy()
    adata.obs[key_added + "_boundary_score"] = boundary.scores.detach().cpu().numpy()
    adata.obs[key_added + "_is_boundary"] = boundary.mask.detach().cpu().numpy()

    adata.uns[key_added + "_config"] = json.loads(json.dumps(asdict(config)))
    adata.uns[key_added + "_diagnostics"] = {
        "boundary_threshold": float(boundary.threshold.detach().cpu().item()),
        "boundary_fraction": float(boundary.mask.float().mean().detach().cpu().item()),
        "gate_mean": float(final_gate.mean().detach().cpu().item()),
        "gate_min": float(final_gate.min().detach().cpu().item()),
        "gate_max": float(final_gate.max().detach().cpu().item()),
    }
    if save_loss:
        adata.uns[key_added + "_warmup_loss"] = np.asarray(warmup_losses, dtype=np.float32)
        if stage2_losses:
            loss_names = list(stage2_losses[0].keys())
            loss_values = [
                [loss_row[name] for name in loss_names]
                for loss_row in stage2_losses
            ]
            adata.uns[key_added + "_stage2_loss_names"] = np.asarray(loss_names, dtype=str)
            adata.uns[key_added + "_stage2_loss"] = np.asarray(loss_values, dtype=np.float32)
        else:
            adata.uns[key_added + "_stage2_loss_names"] = np.asarray([], dtype=str)
            adata.uns[key_added + "_stage2_loss"] = np.empty((0, 0), dtype=np.float32)
    if save_reconstruction:
        reconstructed = out.detach().cpu().numpy()
        reconstructed[reconstructed < 0] = 0
        adata.layers[key_added + "_ReX"] = reconstructed

    return adata
