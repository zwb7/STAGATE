if (!requireNamespace("Seurat", quietly = TRUE)) {
  stop(
    "R package Seurat is required but not installed. Install it on the server first, for example:\n",
    "  conda install -c conda-forge r-seurat r-seuratobject r-hdf5r\n",
    "or inside R:\n",
    "  install.packages('Seurat', repos = 'https://cloud.r-project.org')",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(Seurat)
})

parse_cli_args <- function(argv) {
  defaults <- list(
    data_dir = NULL,
    sample_id = NULL,
    out_dir = "results/seurat",
    counts_file = "filtered_feature_bc_matrix.h5",
    resolution = 0.5,
    dims = "1:30",
    seed = 1234,
    min_features = 200,
    min_counts = 500,
    max_percent_mt = 20,
    nfeatures = 3000,
    ground_truth = NULL,
    ground_truth_no_header = FALSE,
    barcode_col = "barcode",
    label_col = "ground_truth"
  )

  numeric_args <- c(
    "resolution", "seed", "min_features", "min_counts", "max_percent_mt", "nfeatures"
  )
  flag_args <- c("ground_truth_no_header")
  args <- defaults
  i <- 1

  while (i <= length(argv)) {
    token <- argv[[i]]
    if (!startsWith(token, "--")) {
      stop("Unexpected argument: ", token)
    }

    key <- sub("^--", "", token)
    if (!key %in% names(args)) {
      stop("Unknown argument: ", token)
    }

    if (key %in% flag_args) {
      args[[key]] <- TRUE
      i <- i + 1
      next
    }

    if (i == length(argv) || startsWith(argv[[i + 1]], "--")) {
      stop("Missing value for argument: ", token)
    }

    value <- argv[[i + 1]]
    if (key %in% numeric_args) {
      value <- as.numeric(value)
      if (is.na(value)) {
        stop("Argument ", token, " must be numeric.")
      }
    }

    args[[key]] <- value
    i <- i + 2
  }

  required_args <- c("data_dir", "sample_id")
  missing_args <- required_args[vapply(required_args, function(x) is.null(args[[x]]), logical(1))]
  if (length(missing_args) > 0) {
    stop("Missing required arguments: ", paste(missing_args, collapse = ", "))
  }

  args$seed <- as.integer(args$seed)
  args$min_features <- as.integer(args$min_features)
  args$min_counts <- as.integer(args$min_counts)
  args$nfeatures <- as.integer(args$nfeatures)
  args
}

parse_dims <- function(dims_text) {
  if (grepl(":", dims_text, fixed = TRUE)) {
    bounds <- strsplit(dims_text, ":", fixed = TRUE)[[1]]
    return(seq.int(as.integer(bounds[1]), as.integer(bounds[2])))
  }
  as.integer(strsplit(dims_text, ",", fixed = TRUE)[[1]])
}

json_escape <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  gsub('"', '\\\\"', x)
}

json_value <- function(x) {
  if (is.null(x)) {
    return("null")
  }
  if (is.logical(x)) {
    return(ifelse(x, "true", "false"))
  }
  if (is.numeric(x)) {
    return(as.character(x))
  }
  paste0('"', json_escape(as.character(x)), '"')
}

write_params_json <- function(params, path) {
  lines <- vapply(
    names(params),
    function(name) paste0('  "', name, '": ', json_value(params[[name]])),
    character(1)
  )
  lines <- paste0(lines, ifelse(seq_along(lines) < length(lines), ",", ""))
  writeLines(c("{", lines, "}"), path, useBytes = TRUE)
}

adjusted_rand_index <- function(labels_a, labels_b) {
  labels_a <- as.factor(labels_a)
  labels_b <- as.factor(labels_b)
  tab <- table(labels_a, labels_b)
  n <- sum(tab)
  if (n < 2) {
    return(NA_real_)
  }

  choose2 <- function(x) x * (x - 1) / 2
  sum_comb <- sum(choose2(tab))
  row_comb <- sum(choose2(rowSums(tab)))
  col_comb <- sum(choose2(colSums(tab)))
  total_comb <- choose2(n)
  expected <- row_comb * col_comb / total_comb
  max_index <- (row_comb + col_comb) / 2

  if (max_index == expected) {
    return(0)
  }
  (sum_comb - expected) / (max_index - expected)
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

opt <- parse_cli_args(commandArgs(trailingOnly = TRUE))
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
write_params_json(params, file.path(sample_out, "params.json"))

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
  ari <- adjusted_rand_index(obj$seurat_clusters[valid], obj$ground_truth[valid])
  cat(sprintf("ARI: %.6f\n", ari))

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
} else {
  cat("ARI: NA (ground truth not provided)\n")
}

write.csv(obj@meta.data, file.path(sample_out, "metadata.csv"))
saveRDS(obj, file.path(sample_out, "seurat_object.rds"))

sink(file.path(sample_out, "sessionInfo.txt"))
sessionInfo()
sink()