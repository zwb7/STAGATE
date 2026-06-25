import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.E3v3.train_e3v3 import E3v3Config, train_e3v3
from STAGATE_pyG.utils import Cal_Spatial_Net, mclust_R


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated E3v3 on a prepared DLPFC AnnData file.")
    parser.add_argument("--input-h5ad", required=True, help="Input AnnData file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics and result h5ad.")
    parser.add_argument("--n-clusters", type=int, required=True, help="Expected number of spatial domains.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=500)
    parser.add_argument("--stage2-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--rad-cutoff", type=float, default=None, help="Build Radius graph if Spatial_Net is absent.")
    parser.add_argument("--k-cutoff", type=int, default=None, help="Build KNN graph if Spatial_Net is absent.")
    parser.add_argument("--truth-key", default=None, help="Optional obs key for ARI/NMI reporting.")
    parser.add_argument("--run-mclust", action="store_true", help="Run official mclust_R clustering on E3v3 embedding.")
    parser.add_argument("--key-added", default="E3v3")
    parser.add_argument("--boundary-top-q", type=float, default=0.10)
    parser.add_argument("--lambda-boundary", type=float, default=0.1)
    parser.add_argument("--lambda-gate", type=float, default=0.01)
    parser.add_argument("--lambda-preserve", type=float, default=0.1)
    parser.add_argument("--gate-beta", type=float, default=2.0)
    parser.add_argument("--gate-gamma", type=float, default=1.0)
    parser.add_argument("--gate-rho", type=float, default=0.05)
    parser.add_argument("--preserve-consistency-threshold", type=float, default=0.90)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _build_graph_if_needed(adata, args):
    if "Spatial_Net" in adata.uns:
        return
    if args.rad_cutoff is not None:
        Cal_Spatial_Net(adata, rad_cutoff=args.rad_cutoff, model="Radius")
        return
    if args.k_cutoff is not None:
        Cal_Spatial_Net(adata, k_cutoff=args.k_cutoff, model="KNN")
        return
    raise ValueError("Spatial_Net is absent. Provide --rad-cutoff or --k-cutoff.")


def _add_metrics(adata, args):
    metrics = dict(adata.uns.get(args.key_added + "_diagnostics", {}))
    if args.truth_key is None or args.truth_key not in adata.obs:
        return metrics
    if args.key_added + "_mclust" not in adata.obs:
        return metrics

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    keep = (~adata.obs[args.truth_key].isna()) & (~adata.obs[args.key_added + "_mclust"].isna())
    truth = adata.obs.loc[keep, args.truth_key].astype(str).to_numpy()
    pred = adata.obs.loc[keep, args.key_added + "_mclust"].astype(str).to_numpy()
    metrics["ARI"] = float(adjusted_rand_score(truth, pred))
    metrics["NMI"] = float(normalized_mutual_info_score(truth, pred))
    metrics["n_eval_spots"] = int(keep.sum())
    return metrics


def main():
    args = parse_args()

    import scanpy as sc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    _build_graph_if_needed(adata, args)

    config = E3v3Config(
        warmup_epochs=args.warmup_epochs,
        stage2_epochs=args.stage2_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        boundary_top_q=args.boundary_top_q,
        lambda_boundary=args.lambda_boundary,
        lambda_gate=args.lambda_gate,
        lambda_preserve=args.lambda_preserve,
        gate_beta=args.gate_beta,
        gate_gamma=args.gate_gamma,
        gate_rho=args.gate_rho,
        preserve_consistency_threshold=args.preserve_consistency_threshold,
        random_seed=args.seed,
        verbose=True,
    )
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adata = train_e3v3(
        adata,
        n_clusters=args.n_clusters,
        key_added=args.key_added,
        config=config,
        save_loss=True,
        save_reconstruction=False,
        device=device,
    )

    if args.run_mclust:
        adata = mclust_R(adata, num_cluster=args.n_clusters, used_obsm=args.key_added, random_seed=args.seed)
        adata.obs[args.key_added + "_mclust"] = adata.obs["mclust"].copy()

    metrics = _add_metrics(adata, args)
    metrics["seed"] = args.seed
    metrics["n_clusters"] = args.n_clusters
    metrics["input_h5ad"] = args.input_h5ad

    result_path = output_dir / (args.key_added + "_result.h5ad")
    metrics_path = output_dir / "metrics.json"
    adata.write_h5ad(result_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
