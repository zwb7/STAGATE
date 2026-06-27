"""mclust pseudo-label and posterior helpers for BA-STAGATE diagnostics."""

from __future__ import annotations

import numpy as np


def mclust_with_posterior(
    embedding: np.ndarray,
    num_cluster: int,
    model_names: str = "EEE",
    random_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run R mclust and return labels, posterior probabilities, and confidence.

    This intentionally lives outside ``STAGATE_pyG.utils.mclust_R`` so Phase 1
    diagnostics do not alter the official baseline behavior.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or infinite values")

    np.random.seed(random_seed)

    import rpy2.robjects as robjects
    from rpy2.robjects.vectors import FloatVector, IntVector, StrVector

    robjects.r.library("mclust")
    robjects.r["set.seed"](random_seed)

    r_embedding = robjects.r["matrix"](
        FloatVector(embedding.ravel(order="C")),
        nrow=embedding.shape[0],
        ncol=embedding.shape[1],
        byrow=True,
    )
    r_embedding = robjects.r["colnames<-"](
        r_embedding,
        StrVector(
            [f"STAGATE_{index + 1}" for index in range(embedding.shape[1])]
        ),
    )
    result = robjects.r["Mclust"](
        r_embedding,
        G=IntVector([num_cluster]),
        modelNames=StrVector([model_names]),
    )

    labels = np.asarray(list(result.rx2("classification")), dtype=int)
    posterior = np.asarray(result.rx2("z"), dtype=np.float64)
    if posterior.shape[0] != embedding.shape[0]:
        raise RuntimeError(
            "mclust posterior row count does not match embedding row count: "
            f"{posterior.shape[0]} != {embedding.shape[0]}"
        )
    confidence = posterior.max(axis=1)
    return {
        "labels": labels,
        "posterior": posterior,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
    }
