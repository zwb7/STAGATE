# Seurat Spatial Clustering Baseline

This folder contains a Seurat-based clustering workflow plus Scanpy spatial
visualization for 10x Visium-style spatial transcriptomics data such as DLPFC
slices and HBC.

The script is intended as a comparison baseline. It does not modify the official
STAGATE implementation in `STAGATE_pyG/`.

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
