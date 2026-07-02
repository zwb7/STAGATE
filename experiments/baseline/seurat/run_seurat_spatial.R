suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(mclust)
  library(optparse)
  library(jsonlite)
})

option_list <- list(
  make_option("--data_dir", type = "character",
              help = "10x Visium sample directory."),
  make_option("--sample_id", type = "character",
              help = "Sample ID, e.g. 151507 or HBC."),
  make_option("--out_dir", type = "character", default = "results/seurat",
              help = "Output root directory [default: %default]."),
  make_option("--counts_file", type = "character",
              default = "filtered_feature_bc_matrix.h5",
              help = "Count matrix file inside data_dir [default: %default]."),
  make_option("--resolution", type = "double", default = 0.5,
              help = "Seurat clustering resolution [default: %default]."),
  make_option("--dims", type = "character", default = "1:30",
              help = "PCA dimensions, e.g. 1:30 [default: %default]."),
  make_option("--seed", type = "integer", default = 1234,
              help = "Random seed [default: %default]."),
  make_option("--min_features", type = "integer", default = 200,
              help = "Minimum detected genes per spot [default: %default]."),
  make_option("--min_counts", type = "integer", default = 500,
              help = "Minimum UMIs per spot [default: %default]."),
  make_option("--max_percent_mt", type = "double", default = 20,
              help = "Maximum mitochondrial percentage [default: %default]."),
  make_option("--nfeatures", type = "integer", default = 3000,
              help = "Number of variable features [default: %default]."),
  make_option("--ground_truth", type = "character", default = NULL,
              help = "Optional CSV/TSV/TXT metadata file with spot labels."),
  make_option("--ground_truth_no_header", action = "store_true", default = FALSE,
              help = "Set when ground-truth file has no header [default: %default]."),
  make_option("--barcode_col", type = "character", default = "barcode",
              help = "Barcode column in ground-truth file [default: %default]."),
  make_option("--label_col", type = "character", default = "ground_truth",
              help = "Label column in ground-truth file [default: %default].")
)

opt <- parse_args(OptionParser(option_list = option_list))

required_args <- c("data_dir", "sample_id")
missing_args <- required_args[vapply(required_args, function(x) is.null(opt[[x]]), logical(1))]
if (length(missing_args) > 0) {
  stop("Missing required arguments: ", paste(missing_args, collapse = ", "))
}

parse_dims <- function(dims_text) {
  if (grepl(":", dims_text, fixed = TRUE)) {
    bounds <- strsplit(dims_text, ":", fixed = TRUE)[[1]]
    return(seq.int(as.integer(bounds[1]), as.integer(bounds[2])))
  }
  as.integer(strsplit(dims_text, ",", fixed = TRUE)[[1]])
}

read_ground_truth <- function(path, has_header, barcode_col, label_col, spot_barcodes) {
  sep <- if (grepl("\\.csv$", path, ignore.case = TRUE)) "," else ""
  labels <- read.table(
    path,
    header = has_header,
    sep = sep,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    comment.char = "",
    quote = ""
  )

  if (has_header) {
    if (!all(c(barcode_col, label_col) %in% colnames(labels))) {
      stop("Ground-truth file must contain columns: ", barcode_col, ", ", label_col)
    }
    return(labels[, c(barcode_col, label_col), drop = FALSE])
  }

  if (ncol(labels) >= 2) {
    labels <- labels[, 1:2, drop = FALSE]
    colnames(labels) <- c(barcode_col, label_col)
    return(labels)
  }

  if (ncol(labels) == 1 && nrow(labels) == length(spot_barcodes)) {
    labels <- data.frame(
      barcode = spot_barcodes,
      ground_truth = labels[[1]],
      stringsAsFactors = FALSE
    )
    colnames(labels) <- c(barcode_col, label_col)
    return(labels)
  }

  stop(
    "Headerless ground-truth file must have at least two columns ",
    "(barcode and label), or one label column with exactly one row per retained spot."
  )
}

set.seed(opt$seed)
dims <- parse_dims(opt$dims)

sample_out <- file.path(opt$out_dir, opt$sample_id)
dir.create(sample_out, recursive = TRUE, showWarnings = FALSE)

params <- list(
  data_dir = opt$data_dir,
  sample_id = opt$sample_id,
  out_dir = opt$out_dir,
  counts_file = opt$counts_file,
  resolution = opt$resolution,
  dims = opt$dims,
  seed = opt$seed,
  min_features = opt$min_features,
  min_counts = opt$min_counts,
  max_percent_mt = opt$max_percent_mt,
  nfeatures = opt$nfeatures,
  ground_truth = opt$ground_truth,
  ground_truth_no_header = opt$ground_truth_no_header,
  barcode_col = opt$barcode_col,
  label_col = opt$label_col
)
write_json(params, file.path(sample_out, "params.json"), pretty = TRUE, auto_unbox = TRUE)

obj <- Load10X_Spatial(
  data.dir = opt$data_dir,
  filename = opt$counts_file,
  assay = "Spatial",
  slice = opt$sample_id
)

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

obj <- subset(
  obj,
  subset = nFeature_Spatial > opt$min_features &
    nCount_Spatial > opt$min_counts &
    percent.mt < opt$max_percent_mt
)

obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = opt$nfeatures)
obj <- ScaleData(obj, features = VariableFeatures(obj))
obj <- RunPCA(obj, features = VariableFeatures(obj), verbose = FALSE, seed.use = opt$seed)

obj <- FindNeighbors(obj, dims = dims)
obj <- FindClusters(obj, resolution = opt$resolution, random.seed = opt$seed)
obj <- RunUMAP(obj, dims = dims, seed.use = opt$seed)

if (!is.null(opt$ground_truth)) {
  labels <- read_ground_truth(
    opt$ground_truth,
    has_header = !opt$ground_truth_no_header,
    barcode_col = opt$barcode_col,
    label_col = opt$label_col,
    spot_barcodes = colnames(obj)
  )
  rownames(labels) <- labels[[opt$barcode_col]]
  common <- intersect(colnames(obj), rownames(labels))
  if (length(common) == 0) {
    stop("No shared barcodes between Seurat object and ground-truth file.")
  }

  obj$ground_truth <- NA_character_
  obj$ground_truth[common] <- as.character(labels[common, opt$label_col])

  valid <- !is.na(obj$ground_truth)
  if (sum(valid) == 0) {
    stop("No labeled spots available for ARI calculation.")
  }
  ari <- adjustedRandIndex(obj$seurat_clusters[valid], obj$ground_truth[valid])

  write.csv(
    data.frame(
      sample_id = opt$sample_id,
      resolution = opt$resolution,
      dims = opt$dims,
      n_spots = ncol(obj),
      n_labeled_spots = sum(valid),
      ari = ari
    ),
    file.path(sample_out, "metrics.csv"),
    row.names = FALSE
  )
}

spatial_clusters <- SpatialDimPlot(
  obj,
  group.by = "seurat_clusters",
  label = TRUE,
  label.size = 3
) + ggtitle(paste(opt$sample_id, "Seurat clusters"))

ggsave(
  file.path(sample_out, "spatial_clusters.pdf"),
  spatial_clusters,
  width = 7,
  height = 7
)

umap_clusters <- DimPlot(
  obj,
  reduction = "umap",
  group.by = "seurat_clusters",
  label = TRUE
) + ggtitle(paste(opt$sample_id, "UMAP"))

ggsave(
  file.path(sample_out, "umap_clusters.pdf"),
  umap_clusters,
  width = 7,
  height = 6
)

if ("ground_truth" %in% colnames(obj@meta.data)) {
  spatial_ground_truth <- SpatialDimPlot(
    obj,
    group.by = "ground_truth",
    label = TRUE,
    label.size = 3
  ) + ggtitle(paste(opt$sample_id, "ground truth"))

  ggsave(
    file.path(sample_out, "spatial_ground_truth.pdf"),
    spatial_ground_truth,
    width = 7,
    height = 7
  )
}

write.csv(obj@meta.data, file.path(sample_out, "metadata.csv"))
saveRDS(obj, file.path(sample_out, "seurat_object.rds"))

sink(file.path(sample_out, "sessionInfo.txt"))
sessionInfo()
sink()
