"""Run vanilla STAGATE and export baseline artifacts.

This script is intentionally a thin wrapper around the official STAGATE_pyG
implementation. It does not change preprocessing, model structure, loss, or
training behavior. Use it on the remote server to produce reusable baseline
outputs for later boundary diagnosis and BAGR graph-refinement experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import STAGATE_pyG as STAGATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vanilla STAGATE and save standardized baseline outputs."
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=None,
        help=(
            "Input AnnData file. It should already contain the intended "
            "preprocessing and, preferably, adata.uns['Spatial_Net']. Use "
            "--dataset dlpfc to load a raw DLPFC Visium slice instead."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["h5ad", "dlpfc"],
        default="h5ad",
        help="Input source. Use dlpfc for dataset/DLPFC/<sample-id> raw Visium data.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/DLPFC"),
        help="Dataset root used by --dataset dlpfc.",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument(
        "--truth-file",
        type=Path,
        default=None,
        help="Ground-truth annotation file. Defaults to the slice truth file for DLPFC.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["log-normalize", "none"],
        default="log-normalize",
        help="Expression preprocessing for raw dataset inputs.",
    )
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "results/stagate_baseline/<sample-id>/seed_<seed>."
        ),
    )
    parser.add_argument(
        "--ground-truth-key",
        default="Ground Truth",
        help="Observation column containing ground-truth labels when available.",
    )
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--mclust-model", default="EEE")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clipping", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example cpu, cuda:0, or auto.",
    )
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument(
        "--build-spatial-net",
        action="store_true",
        help=(
            "Build Spatial_Net if it is absent. This should only be used when "
            "the chosen baseline protocol explicitly builds the graph here."
        ),
    )
    parser.add_argument(
        "--spatial-net-model",
        choices=["Radius", "KNN"],
        default="Radius",
    )
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--k-cutoff", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return Path("results") / "stagate_baseline" / args.sample_id / f"seed_{args.seed}"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def preprocess_expression(adata: sc.AnnData, args: argparse.Namespace) -> None:
    if args.preprocess_mode == "none":
        adata.uns["preprocessing"] = {"mode": "none"}
        return
    if args.n_top_genes <= 0:
        raise ValueError("--n-top-genes must be positive for log-normalize preprocessing.")
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=args.n_top_genes,
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns["preprocessing"] = {
        "mode": "log-normalize",
        "n_top_genes": int(args.n_top_genes),
        "normalization": "scanpy.normalize_total(target_sum=1e4)+log1p",
        "hvg_method": "scanpy.highly_variable_genes(flavor='seurat_v3')",
    }


def load_dlpfc_slice(args: argparse.Namespace) -> sc.AnnData:
    sample_dir = args.data_root / args.sample_id
    if not sample_dir.exists():
        raise FileNotFoundError(f"DLPFC sample directory not found: {sample_dir}")
    count_path = sample_dir / args.count_file
    if not count_path.exists():
        raise FileNotFoundError(f"DLPFC count file not found: {count_path}")

    adata = sc.read_visium(path=sample_dir, count_file=args.count_file)
    adata.var_names_make_unique()

    truth_file = args.truth_file or sample_dir / f"{args.sample_id}_truth.txt"
    if truth_file.exists():
        truth = pd.read_csv(
            truth_file,
            sep="	",
            header=None,
            names=["spot_id", args.ground_truth_key],
            dtype=str,
        )
        if truth["spot_id"].duplicated().any():
            raise ValueError(f"Ground-truth spot IDs contain duplicates: {truth_file}")
        truth = truth.set_index("spot_id")
        adata.obs[args.ground_truth_key] = truth.reindex(adata.obs_names)[
            args.ground_truth_key
        ]
        adata.obs[args.ground_truth_key] = adata.obs[args.ground_truth_key].astype(
            "category"
        )
    else:
        print(f"Ground-truth file not found and will be skipped: {truth_file}")

    preprocess_expression(adata, args)
    if args.radius is None and args.spatial_net_model == "Radius":
        args.radius = 150.0
    args.build_spatial_net = True
    adata.uns["dataset_source"] = {
        "dataset": "dlpfc",
        "sample_id": args.sample_id,
        "sample_dir": str(sample_dir),
        "count_file": args.count_file,
        "truth_file": str(truth_file),
    }
    return adata


def load_input_adata(args: argparse.Namespace) -> sc.AnnData:
    if args.dataset == "dlpfc":
        print(f"Loading raw DLPFC slice {args.sample_id} from {args.data_root}")
        return load_dlpfc_slice(args)
    if args.input_h5ad is None:
        raise ValueError("--input-h5ad is required when --dataset h5ad")
    print(f"Loading AnnData from {args.input_h5ad}")
    return sc.read_h5ad(args.input_h5ad)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def ensure_spatial_net(adata: sc.AnnData, args: argparse.Namespace) -> None:
    if "Spatial_Net" in adata.uns:
        return
    if not args.build_spatial_net:
        raise KeyError(
            "adata.uns['Spatial_Net'] is missing. Provide an input h5ad with the "
            "baseline spatial graph, or pass --build-spatial-net with the exact "
            "official graph-construction parameters."
        )
    if args.spatial_net_model == "Radius":
        if args.radius is None:
            raise ValueError("--radius is required when --spatial-net-model Radius")
        STAGATE.Cal_Spatial_Net(
            adata,
            rad_cutoff=args.radius,
            model="Radius",
            verbose=True,
        )
    else:
        if args.k_cutoff is None:
            raise ValueError("--k-cutoff is required when --spatial-net-model KNN")
        STAGATE.Cal_Spatial_Net(
            adata,
            k_cutoff=args.k_cutoff,
            model="KNN",
            verbose=True,
        )


def validate_spatial_net(adata: sc.AnnData) -> pd.DataFrame:
    graph = adata.uns["Spatial_Net"]
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("adata.uns['Spatial_Net'] must be a pandas DataFrame")
    required = {"Cell1", "Cell2", "Distance"}
    missing = sorted(required.difference(graph.columns))
    if missing:
        raise ValueError(f"Spatial_Net is missing columns: {missing}")
    return graph.copy()


def run_mclust_with_posterior(
    embedding: np.ndarray,
    clusters: int,
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run R mclust and return classification plus posterior probabilities."""

    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be two-dimensional, got {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values.")

    import rpy2.robjects as robjects
    from rpy2.robjects.vectors import FloatVector, IntVector, StrVector

    robjects.r.library("mclust")
    robjects.r["set.seed"](seed)
    r_embedding = robjects.r["matrix"](
        FloatVector(np.asarray(embedding, dtype=np.float64).ravel(order="C")),
        nrow=embedding.shape[0],
        ncol=embedding.shape[1],
        byrow=True,
    )
    r_embedding = robjects.r["colnames<-"](
        r_embedding,
        StrVector([f"STAGATE_{index + 1}" for index in range(embedding.shape[1])]),
    )
    result = robjects.r["Mclust"](
        r_embedding,
        G=IntVector([clusters]),
        modelNames=StrVector([model_name]),
    )
    labels = np.asarray(list(result.rx2("classification")), dtype=int)
    posterior = np.asarray(result.rx2("z"), dtype=float)
    if posterior.shape[0] != embedding.shape[0]:
        posterior = posterior.T
    if posterior.shape[0] != embedding.shape[0]:
        raise ValueError(
            "Unexpected mclust posterior shape: "
            f"{posterior.shape}, expected first dimension {embedding.shape[0]}"
        )
    return labels, posterior


def best_label_match_correctness(
    truth: pd.Series,
    predicted: pd.Series,
) -> tuple[pd.Series, dict[str, str]]:
    valid = truth.notna() & predicted.notna()
    correctness = pd.Series(False, index=truth.index, dtype=bool)
    if not valid.any():
        return correctness, {}

    truth_valid = truth.loc[valid].astype(str)
    pred_valid = predicted.loc[valid].astype(str)
    truth_values = sorted(truth_valid.unique())
    pred_values = sorted(pred_valid.unique())
    truth_to_index = {value: index for index, value in enumerate(truth_values)}
    pred_to_index = {value: index for index, value in enumerate(pred_values)}
    contingency = np.zeros((len(pred_values), len(truth_values)), dtype=int)
    for pred_label, truth_label in zip(pred_valid, truth_valid):
        contingency[pred_to_index[pred_label], truth_to_index[truth_label]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {
        pred_values[row]: truth_values[col]
        for row, col in zip(row_ind, col_ind)
        if contingency[row, col] > 0
    }
    matched_pred = pred_valid.map(mapping)
    correctness.loc[valid] = matched_pred == truth_valid
    return correctness, mapping


def save_labels(
    output_dir: Path,
    obs_names: pd.Index,
    labels: np.ndarray,
    ground_truth: pd.Series | None,
) -> pd.Series:
    pred = pd.Series(labels.astype(str), index=obs_names, name="pred_label")
    pd.DataFrame({"spot_id": obs_names, "pred_label": pred.values}).to_csv(
        output_dir / "pred_labels.csv",
        index=False,
    )
    if ground_truth is not None:
        pd.DataFrame(
            {
                "spot_id": obs_names,
                "ground_truth": ground_truth.astype("object").values,
            }
        ).to_csv(output_dir / "labels_gt.csv", index=False)
    return pred


def save_spot_correctness(
    output_dir: Path,
    pred: pd.Series,
    ground_truth: pd.Series | None,
) -> dict[str, Any]:
    if ground_truth is None:
        pd.DataFrame(
            {
                "spot_id": pred.index,
                "pred_label": pred.values,
                "ground_truth": pd.NA,
                "is_evaluated": False,
                "is_correct": pd.NA,
            }
        ).to_csv(output_dir / "spot_correctness.csv", index=False)
        return {"label_mapping": {}, "n_evaluated_spots": 0}

    correctness, mapping = best_label_match_correctness(ground_truth, pred)
    evaluated = ground_truth.notna() & pred.notna()
    pd.DataFrame(
        {
            "spot_id": pred.index,
            "pred_label": pred.values,
            "ground_truth": ground_truth.astype("object").values,
            "is_evaluated": evaluated.values,
            "is_correct": correctness.where(evaluated, other=pd.NA).values,
        }
    ).to_csv(output_dir / "spot_correctness.csv", index=False)
    return {
        "label_mapping": mapping,
        "n_evaluated_spots": int(evaluated.sum()),
    }


def compute_metrics(
    ground_truth: pd.Series | None,
    pred: pd.Series,
) -> dict[str, float | int | None]:
    if ground_truth is None:
        return {
            "ari": None,
            "nmi": None,
            "n_evaluated_spots": 0,
        }
    evaluation = pd.DataFrame(
        {"ground_truth": ground_truth, "pred_label": pred}
    ).dropna()
    if evaluation.empty:
        return {
            "ari": None,
            "nmi": None,
            "n_evaluated_spots": 0,
        }
    return {
        "ari": float(
            adjusted_rand_score(
                evaluation["ground_truth"].astype(str),
                evaluation["pred_label"].astype(str),
            )
        ),
        "nmi": float(
            normalized_mutual_info_score(
                evaluation["ground_truth"].astype(str),
                evaluation["pred_label"].astype(str),
            )
        ),
        "n_evaluated_spots": int(evaluation.shape[0]),
    }


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scanpy": sc.__version__,
        "torch": torch.__version__,
    }
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return versions


def train_and_export(args: argparse.Namespace) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user

    output_dir = resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to replace its files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_input_adata(args)
    ensure_spatial_net(adata, args)
    spatial_net = validate_spatial_net(adata)
    spatial_net.to_csv(output_dir / "spatial_edges.csv", index=False)

    ground_truth = (
        adata.obs[args.ground_truth_key].copy()
        if args.ground_truth_key in adata.obs
        else None
    )
    if ground_truth is None:
        print(
            f"Ground-truth key '{args.ground_truth_key}' was not found; "
            "global ARI/NMI and correctness will be omitted."
        )

    device = resolve_device(args.device)
    print(f"Training vanilla STAGATE on {device}")
    adata = STAGATE.train_STAGATE(
        adata,
        hidden_dims=[args.hidden_dim, args.latent_dim],
        n_epochs=args.epochs,
        lr=args.learning_rate,
        gradient_clipping=args.gradient_clipping,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
        save_loss=True,
        save_reconstrction=False,
        device=device,
    )

    embedding = np.asarray(adata.obsm["STAGATE"])
    np.save(output_dir / "embedding.npy", embedding)

    labels, posterior = run_mclust_with_posterior(
        embedding=embedding,
        clusters=args.clusters,
        model_name=args.mclust_model,
        seed=args.seed,
    )
    np.save(output_dir / "posterior.npy", posterior)
    adata.obs["mclust"] = pd.Categorical(labels.astype(str))

    pred = save_labels(output_dir, adata.obs_names, labels, ground_truth)
    correctness_info = save_spot_correctness(output_dir, pred, ground_truth)
    metrics = compute_metrics(ground_truth, pred)

    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss

    result: dict[str, Any] = {
        "sample_id": args.sample_id,
        "method": "vanilla_stagate",
        "dataset": args.dataset,
        "input_h5ad": str(args.input_h5ad) if args.input_h5ad is not None else None,
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "ground_truth_key": args.ground_truth_key,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_clusters": int(args.clusters),
        "mclust_model": args.mclust_model,
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "n_evaluated_spots": metrics["n_evaluated_spots"],
        "final_reconstruction_loss": final_loss,
        "seed": int(args.seed),
        "device": str(device),
        "hidden_dims": [int(args.hidden_dim), int(args.latent_dim)],
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "gradient_clipping": float(args.gradient_clipping),
        "spatial_edge_count": int(spatial_net.shape[0]),
        "git_commit": get_git_commit(),
        "package_versions": package_versions(),
        "label_mapping_for_correctness": correctness_info["label_mapping"],
    }
    write_json(output_dir / "metrics_global.json", result)
    write_json(
        output_dir / "run_config.json",
        {
            "argv": sys.argv,
            "args": json_safe(vars(args)),
            "git_commit": result["git_commit"],
            "package_versions": result["package_versions"],
        },
    )

    adata.write_h5ad(output_dir / f"{args.sample_id}_stagate_baseline.h5ad")
    print(f"Results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_and_export(parse_args())


if __name__ == "__main__":
    main()
