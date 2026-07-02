# Seurat Spatial Clustering Baseline

This folder contains a Seurat-based clustering workflow plus Scanpy spatial
visualization for 10x Visium-style spatial transcriptomics data such as DLPFC
slices and HBC.

The script is intended as a comparison baseline. It does not modify the official
STAGATE implementation in `STAGATE_pyG/`.

## Server Environment

The R clustering script requires Seurat. It uses base R for argument parsing,
parameter logging, and ARI calculation, so `optparse`, `jsonlite`, and `mclust`
are not required.

Do not install Seurat into the existing Python/GPU environment if conda reports
large dependency conflicts. Use a separate R environment for the Seurat baseline:

```bash
mamba create -n stagate-seurat -c conda-forge --strict-channel-priority \
  r-base=4.3 r-seurat r-seuratobject r-hdf5r
conda activate stagate-seurat
```

If `mamba` is unavailable, use conda:

```bash
conda create -n stagate-seurat -c conda-forge --strict-channel-priority \
  r-base=4.3 r-seurat r-seuratobject r-hdf5r
conda activate stagate-seurat
```

Installing these packages directly into a mixed existing environment can force
conda to reconcile unrelated packages such as `hdf5`, `curl`, compiler runtimes,
and Python stack dependencies. A clean R environment is more reproducible and
keeps comparison-method dependencies isolated.

Alternatively, install from CRAN inside an R environment:

```r
install.packages("Seurat", repos = "https://cloud.r-project.org")
install.packages("hdf5r", repos = "https://cloud.r-project.org")
```

The PNG visualization script requires Python packages `scanpy`, `pandas`, and
`matplotlib`. If the current Python environment does not already have them,
install them in a Python visualization environment:

```bash
mamba create -n stagate-scanpy -c conda-forge --strict-channel-priority \
  python=3.10 scanpy pandas matplotlib h5py
conda activate stagate-scanpy
```

Or install them into an existing compatible Python environment:

```bash
mamba install -c conda-forge scanpy pandas matplotlib h5py
```

If `mamba` is unavailable, replace `mamba` with `conda`.

## Inputs

Expected sample directory layout:

```text
sample_dir/
  filtered_feature_bc_matrix.h5
  spatial/
    tissue_positions*.csv
    scalefactors_json.json
    tissue_*image*.png
```

Optional ground-truth metadata can be provided as CSV, TSV, or TXT.

Headered files should include barcode and label columns, for example:

```text
barcode,ground_truth
AAACAAGTATCTCCCA-1,Layer1
...
```

Headerless TXT files are also supported with `--ground_truth_no_header`. The
preferred headerless format is two whitespace-separated columns:

```text
AAACAAGTATCTCCCA-1 Layer1
AAACACCAATAACTGC-1 Layer2
```

A one-column headerless label file is accepted only when it has exactly one row
per retained spot after QC filtering; in that case labels are assigned in the
current Seurat object barcode order. Prefer the two-column format to avoid
ambiguous alignment.

## DLPFC example

Run one slice at a time. First run Seurat clustering and ARI calculation:

```bash
Rscript experiments/baseline/seurat/run_seurat_spatial.R \
  --data_dir /data1/zhangwenbo/Code/Dataset/LIBD/151674 \
  --sample_id 151674 \
  --out_dir results/seurat \
  --resolution 0.5 \
  --dims 1:30 \
  --ground_truth /data1/zhangwenbo/Code/Dataset/LIBD/151674/151674_truth.txt \
  --ground_truth_no_header
```

The R script prints ARI to the terminal when ground truth is provided, for
example:

```text
ARI: 0.512345
```

Then render PNG spatial visualizations with Scanpy `sc.pl.spatial`:

```bash
python experiments/baseline/seurat/plot_scanpy_spatial.py \
  --data_dir /path/to/DLPFC/151507 \
  --sample_id 151507 \
  --metadata results/seurat/dlpfc/151507/metadata.csv \
  --out_dir results/seurat/dlpfc
```

If the TXT file has no header but uses non-default column meanings, the first
column is treated as barcode and the second as label. You can still rename the
internal column names recorded in outputs:

```bash
Rscript experiments/baseline/seurat/run_seurat_spatial.R \
  --data_dir /path/to/DLPFC/151507 \
  --sample_id 151507 \
  --out_dir results/seurat/dlpfc \
  --resolution 0.5 \
  --dims 1:30 \
  --ground_truth /path/to/DLPFC/151507/metadata.txt \
  --ground_truth_no_header \
  --barcode_col barcode \
  --label_col layer_guess
```

## HBC example

```bash
Rscript experiments/baseline/seurat/run_seurat_spatial.R \
  --data_dir /path/to/HBC \
  --sample_id HBC \
  --out_dir results/seurat/hbc \
  --resolution 0.5 \
  --dims 1:30

python experiments/baseline/seurat/plot_scanpy_spatial.py \
  --data_dir /path/to/HBC \
  --sample_id HBC \
  --metadata results/seurat/hbc/HBC/metadata.csv \
  --out_dir results/seurat/hbc
```

## Outputs

For each sample, the R script writes:

- `metadata.csv`
- `seurat_object.rds`
- `params.json`
- `sessionInfo.txt`
- `metrics.csv`, only when ground truth is provided

The Scanpy visualization script writes PNG files:

- `spatial_clusters.png`
- `spatial_ground_truth.png`, only when ground truth is available

Generated results should remain under `results/` and should not be committed.
