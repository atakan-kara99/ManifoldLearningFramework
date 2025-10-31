# Manifold Learning Framework

This repository contains a small Python toolkit used during a Bachelor's thesis for experimenting with dimensionality reduction (MDS, LLE, t-SNE), clustering (KMeans, DBSCAN) and several evaluation criteria (Frobenius norm of pairwise distances, trustworthiness / k-NN, silhouette coefficient, ARI / AMI). It includes data handling utilities, caching of intermediate results (low-dimensional embeddings, clustering outputs, evaluations) and plotting helpers for visual analysis.

**NOTE: THIS CODE FRAMEWORK IS RELATED TO MY [BACHELOR THESIS](https://github.com/atakan-kara99/BachelorThesis).**

## Features
- Compute low-dimensional embeddings with MDS, LLE and t-SNE (scikit-learn).
- Save and load computed embeddings (`data_low`) for later analysis.
- Evaluate embeddings with distance-based Frobenius norm (`DIST`), trustworthiness (`kNN`), and a combined ATA evaluation.
- Apply clustering (KMeans, DBSCAN) to high- or low-dimensional data and evaluate using Silhouette Coefficient (SC), ARI and AMI.
- Plotting functions and heatmaps to inspect pairwise distance differences and evaluation curves.

## Repository layout

Top-level files
- `main.py` — example workflows and common plotting/evaluation calls used interactively.
- `CluAlg.py` — clustering helper class (KMeans, DBSCAN), save/load cached clusterings and SC evaluation.
- `DataSet.py` — dataset loader/plotting utilities. Supports `COIL20`, `Mammoth`, `trans_Mammoth` and expects preprocessed files under `datasets/`.
- `DimRed.py` — dimensionality reduction wrapper for MDS, LLE and TSNE. Handles parameter ranges and saving/loading `data_low` results.
- `EvalCrit.py` — evaluation criteria implementation (DIST, kNN trustworthiness, ATA) and plotting utilities.
- `utils.py` — plotting heatmaps, matrix helpers and other small utilities.

Data & results (directories)
- `datasets/` — expected input datasets and preprocessed numpy files (COIL20 encodings, Mammoth JSON/NPY).
- `cluster/` — generated clustering result files (`.npy`).
- `dist_eval/` — saved distance-based evaluation result arrays.
- `knn_eval/` — saved k-NN / trustworthiness evaluation arrays.

> Note: Many scripts rely on cached .npy files produced by the library (see `DimRed.generate_filename` / `EvalCrit.generate_filename` / `CluAlg.generate_filename`). If the expected files are missing the code will perform the computation and save results.

## Requirements
- Python 3.8+
- numpy
- matplotlib
- scikit-learn
- pillow (PIL)
- pandas

Install dependencies (recommended in a virtualenv):

```bash
python -m venv venv
source venv/Scripts/activate    # on Windows Git Bash / bash.exe
pip install --upgrade pip
pip install numpy matplotlib scikit-learn pillow pandas
```

## Quick start / Typical workflows

1) Run the example script

```bash
python main.py
```

`main.py` contains many example calls (mostly commented) showing how to:
- compute embeddings: `DimRed.apply(dataset)`
- evaluate embeddings: `EvalCrit('DIST'|'1NN'|...)`
- compute and save clusterings: `CluAlg.kMeans(...)` / `CluAlg.dbSCAN(...)` and `.apply()`
- produce heatmaps and plots for inspection

2) Compute an embedding and evaluate

Python (interactive) example:

```python
from DataSet import DataSet
from DimRed import DimRed
from EvalCrit import EvalCrit

ds = DataSet('Mammoth')
dr = DimRed('LLE', parameter=30)
dr.apply(ds)                       # compute and cache low-dim results
eval = EvalCrit('DIST', ds, dr)
eval.eval()                        # compute and save evaluation
eval.plot_eval()
```

3) Run clustering on low-dimensional results

```python
from CluAlg import CluAlg
ca = CluAlg.kMeans(ds, dimred=dr, k=5)
clusters = ca.apply()              # save/load clustering
ca.ss_eval()                       # silhouette evaluation
ca.plot_eval()
```

4) DBSCAN parameter sweep example

```python
# create a DBSCAN configuration (low-dimensional sweep)
db = CluAlg.dbSCAN(ds, dimred=dr, eps=(0.1,2.0,0.1), minpts=(2,100,5))
db.apply()                          # will compute clustering grid and save
db.ss_eval()                        # compute silhouette for grid
```

## How results are stored
- Low-dimensional embeddings: saved under `data_low` via `DimRed.generate_filename` as `.npy` arrays.
- Evaluation results: saved to `dist_eval/` or `knn_eval/` depending on the criterion.
- Clusterings: saved to `cluster/` as `.npy` files. Filenames include algorithm, dataset, and parameter ranges for traceability.

This caching mechanism avoids recomputing expensive embeddings and evaluations. If you want to force recomputation, either delete the corresponding `.npy` file or run the functions with `saveNload=False` (where available).

## Notes, tips and known conventions
- `DimRed` expects either a single `parameter`, a `parameter_range` tuple (min, max, step), or `parameters` list depending on the method. MDS ignores neighbor/perplexity parameter.
- `EvalCrit` uses two storage folders: `dist_eval/` for distance-based evaluations and `knn_eval/` for trustworthiness results.
- `DataSet` currently supports `COIL20` (preprocessed `.npy` containing encoded images and labels) and `Mammoth`/`trans_Mammoth` (3D point clouds). Make sure those files exist in `datasets/`.
- The plotting and `main.py` are meant for interactive analysis — edit the calls in `main.py` for your experiments.
