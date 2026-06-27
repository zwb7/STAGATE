"""Shared preprocessing helpers for STAGATE example experiments."""

from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp


LOG_NORMALIZE = "log-normalize"
SCTRANSFORM = "sctransform"
PREPROCESS_MODES = (LOG_NORMALIZE, SCTRANSFORM)


def preprocess_expression(
    adata: sc.AnnData,
    n_top_genes: int,
    mode: str = LOG_NORMALIZE,
) -> None:
    """Preprocess expression values in-place and mark highly variable genes."""
    if mode == LOG_NORMALIZE:
        _preprocess_log_normalize(adata, n_top_genes)
        return
    if mode == SCTRANSFORM:
        _preprocess_sctransform(adata, n_top_genes)
        return
    raise ValueError(
        "Unsupported preprocessing mode: %s. Expected one of %s."
        % (mode, ", ".join(PREPROCESS_MODES))
    )


def _preprocess_log_normalize(adata: sc.AnnData, n_top_genes: int) -> None:
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns["preprocessing"] = {
        "mode": LOG_NORMALIZE,
        "n_top_genes": int(n_top_genes),
        "normalization": "scanpy.normalize_total(target_sum=1e4)+log1p",
        "hvg_method": "scanpy.highly_variable_genes(flavor='seurat_v3')",
    }


def _preprocess_sctransform(adata: sc.AnnData, n_top_genes: int) -> None:
    if n_top_genes <= 0:
        raise ValueError("--n-top-genes must be positive for SCTransform.")

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import FloatVector, IntVector, StrVector
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "SCTransform preprocessing requires rpy2 and an R installation with "
            "the Seurat and Matrix packages."
        ) from exc

    try:
        importr("Seurat")
        importr("Matrix")
        importr("methods")
    except Exception as exc:
        raise RuntimeError(
            "SCTransform preprocessing requires the R packages Seurat, Matrix, "
            "and methods. Install them on the server before using "
            "--preprocess-mode sctransform."
        ) from exc

    counts = _adata_counts_as_gene_by_cell_csc(adata)
    ro.globalenv["sct_i"] = IntVector((counts.indices + 1).tolist())
    ro.globalenv["sct_p"] = IntVector(counts.indptr.tolist())
    ro.globalenv["sct_x"] = FloatVector(counts.data.astype(float).tolist())
    ro.globalenv["sct_dims"] = IntVector([counts.shape[0], counts.shape[1]])
    ro.globalenv["sct_genes"] = StrVector([str(name) for name in adata.var_names])
    ro.globalenv["sct_cells"] = StrVector([str(name) for name in adata.obs_names])
    ro.globalenv["sct_n_top_genes"] = IntVector([int(n_top_genes)])

    ro.r(
        """
        sct_counts <- Matrix::sparseMatrix(
            i = sct_i,
            p = sct_p,
            x = sct_x,
            dims = sct_dims
        )
        rownames(sct_counts) <- sct_genes
        colnames(sct_counts) <- sct_cells

        sct_obj <- Seurat::CreateSeuratObject(
            counts = sct_counts,
            assay = "Spatial"
        )
        sct_obj <- Seurat::SCTransform(
            sct_obj,
            assay = "Spatial",
            new.assay.name = "SCT",
            variable.features.n = as.integer(sct_n_top_genes[[1]]),
            return.only.var.genes = FALSE,
            verbose = FALSE
        )

        get_sct_assay_data <- function(object, assay, layer_name) {
            tryCatch(
                Seurat::GetAssayData(
                    object = object,
                    assay = assay,
                    layer = layer_name
                ),
                error = function(e) {
                    Seurat::GetAssayData(
                        object = object,
                        assay = assay,
                        slot = layer_name
                    )
                }
            )
        }

        sct_data <- get_sct_assay_data(sct_obj, "SCT", "data")
        sct_data <- methods::as(sct_data, "dgCMatrix")
        sct_data_genes <- rownames(sct_data)
        sct_data_cells <- colnames(sct_data)
        sct_variable_features <- Seurat::VariableFeatures(sct_obj)
        """
    )

    sct_genes = [str(name) for name in ro.r["sct_data_genes"]]
    sct_cells = [str(name) for name in ro.r["sct_data_cells"]]
    adata.X = _align_sct_matrix_to_adata(
        _r_dgcmatrix_to_cell_by_gene_csr(ro.r["sct_data"]),
        sct_genes,
        sct_cells,
        adata,
    )
    variable_features = [str(name) for name in ro.r["sct_variable_features"]]
    adata.var["highly_variable"] = adata.var_names.isin(variable_features)
    adata.uns["preprocessing"] = {
        "mode": SCTRANSFORM,
        "n_top_genes": int(n_top_genes),
        "normalization": "Seurat::SCTransform assay='Spatial'",
        "expression_source": "SCT assay data layer",
        "hvg_method": "Seurat::VariableFeatures after SCTransform",
    }


def _adata_counts_as_gene_by_cell_csc(adata: sc.AnnData) -> sp.csc_matrix:
    if sp.issparse(adata.X):
        counts = adata.X.T.tocsc()
    else:
        counts = sp.csc_matrix(np.asarray(adata.X).T)
    counts.sort_indices()
    return counts


def _r_dgcmatrix_to_cell_by_gene_csr(r_matrix) -> sp.csr_matrix:
    rows = np.asarray(r_matrix.do_slot("i"), dtype=np.int32)
    indptr = np.asarray(r_matrix.do_slot("p"), dtype=np.int32)
    data = np.asarray(r_matrix.do_slot("x"), dtype=np.float32)
    shape = tuple(np.asarray(r_matrix.do_slot("Dim"), dtype=np.int32))
    gene_by_cell = sp.csc_matrix((data, rows, indptr), shape=shape)
    return gene_by_cell.T.tocsr()


def _align_sct_matrix_to_adata(
    sct_x: sp.csr_matrix,
    sct_genes: list[str],
    sct_cells: list[str],
    adata: sc.AnnData,
) -> sp.csr_matrix:
    if sct_x.shape != (len(sct_cells), len(sct_genes)):
        raise ValueError(
            "SCT matrix shape %s does not match returned cell/gene names "
            "(%d, %d)."
            % (sct_x.shape, len(sct_cells), len(sct_genes))
        )

    obs_names = [str(name) for name in adata.obs_names]
    var_names = [str(name) for name in adata.var_names]
    cell_positions = _positions_in_order(sct_cells, obs_names, "SCT cells")
    gene_positions = _positions_in_order(sct_genes, var_names, "SCT genes")

    placed = _place_sct_genes(sct_x, gene_positions, adata.n_vars).tocoo()
    return sp.coo_matrix(
        (placed.data, (cell_positions[placed.row], placed.col)),
        shape=adata.shape,
    ).tocsr()


def _place_sct_genes(
    sct_x: sp.csr_matrix,
    gene_positions: np.ndarray,
    n_vars: int,
) -> sp.csr_matrix:
    gene_index = sp.coo_matrix(
        (
            np.ones(len(gene_positions), dtype=np.float32),
            (np.arange(len(gene_positions)), gene_positions),
        ),
        shape=(len(gene_positions), n_vars),
    ).tocsr()
    return sct_x @ gene_index


def _positions_in_order(
    returned_names: list[str],
    expected_names: list[str],
    label: str,
) -> np.ndarray:
    expected_positions = {name: index for index, name in enumerate(expected_names)}
    positions = []
    missing = []
    for name in returned_names:
        position = expected_positions.get(name)
        if position is None:
            missing.append(name)
        else:
            positions.append(position)
    if missing:
        raise ValueError(
            "%s returned names absent from AnnData, for example: %s"
            % (label, ", ".join(missing[:5]))
        )
    return np.asarray(positions, dtype=np.int64)
