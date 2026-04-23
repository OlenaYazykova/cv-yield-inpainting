# CV Yield Inpainting

Deep learning pipeline for spatial inpainting of corn (*Zea mays*) yield maps
using UNet architecture.

## Problem

Yield maps obtained from combine harvesters contain missing data regions (holes)
caused by GPS signal loss, sensor malfunctions, or field boundary effects.
This project reconstructs missing corn yield values within field contours using
surrounding observed data and multi-channel spatial predictors (terrain, soil,
vegetation).

## Project Structure

```
cv-yield-inpainting/
├── configs/
│   ├── baseline.yaml              # Shallow UNet config
│   ├── deep.yaml                  # Deep UNet config
│   └── optuna.yaml                # Optuna hyperparameter search config
├── scripts/                       # Data preparation and analysis utilities
│   ├── compute_stripe_angles.py   # Combine harvester stripe angle detection (required before training)
│   ├── correct_dem.py             # DEM artifact correction (forest canopy returns)
│   ├── feature_importance.py      # Occlusion sensitivity analysis
│   ├── get_ndvi_sentinelhub.py    # NDVI raster download from Sentinel Hub
│   ├── get_soils_sentinelhub.py   # Soil index raster download from Sentinel Hub
│   ├── raster_warp.py             # Raster reprojection and alignment to DEM grid
│   └── reclassify_aspect.py       # Aspect reclassification to 8 cardinal directions
├── src/
│   ├── data/
│   │   ├── dataset.py             # YieldDataset, patch sampling, hole simulation
│   │   ├── compute_stats.py       # Normalization statistics (computed on train fields only)
│   │   └── split.py               # Train/val/test split at field level (not patch level)
│   ├── models/
│   │   └── unet.py                # ShallowUNet, DeepUNet, build_model factory
│   ├── training/
│   │   ├── train.py               # Training loop with MLflow + TensorBoard logging
│   │   ├── losses.py              # Masked loss (MSE+MAE on holes), smoothness penalty
│   │   └── optuna_search.py       # Bayesian hyperparameter search + final training
│   ├── evaluation/
│   │   └── test.py                # Test inference on held-out fields, metrics, GeoTIFF export
│   ├── inference/
│   │   └── infer.py               # Production inference on real NaN holes, GeoTIFF export
│   └── utils/
├── notebooks/
│   └── test_dataset.ipynb         # Dataset validation and channel inspection notebook
├── artifacts/                     # Training outputs (auto-created, not tracked by git)
│   ├── baseline/
│   │   ├── baseline/
│   │   │   ├── best_model.pt
│   │   │   └── stats.json
│   │   └── test/
│   ├── deep/
│   │   ├── deep/
│   │   │   ├── best_model.pt
│   │   │   └── stats.json
│   │   └── test/
│   └── deep_optuna/
│       ├── best_model.pt
│       ├── stats.json
│       ├── optuna_best.json       # Best hyperparameters, auto-loaded by test/infer/feature_importance
│       ├── feature_importance.png
│       ├── test/
│       └── infer/
├── data/                          # Field raster data (not tracked by git)
│   ├── field_01/
│   │   ├── yield.tif
│   │   ├── dem.tif
│   │   ├── hand.tif
│   │   ├── ndvi.tif
│   │   ├── slope.tif
│   │   ├── twi.tif
│   │   ├── soil.tif
│   │   ├── rtp_local.tif
│   │   ├── rtp_regional.tif
│   │   ├── aspect.tif
│   │   ├── aspect_categ.tif
│   │   ├── geomorphons.tif
│   │   ├── relief_class.tif
│   │   └── field.gpkg             # Field boundary polygon
│   ├── ...
│   ├── fields.gpkg                # All field boundary polygons
│   └── stripe_angles.json         # Pre-computed combine harvester stripe angles
├── docs/
│   └── images/                    # Report figures
├── main.py
├── REPORT.md                      # Academic report
├── pyproject.toml
└── README.md
```

## Input Features (39 channels)

| Group | Features | Encoding | Channels |
|---|---|---|---|
| Continuous | DEM, HAND, NDVI, RTP local/regional, Slope, Soil, TWI | Z-score | 8 |
| Aspect | Aspect angle | sin + cos | 2 |
| Geomorphons | Landform type | One-hot (1–10) | 10 |
| Relief class | Relief complexity | One-hot (1–8) | 8 |
| Aspect category | Cardinal direction | One-hot (0–7) | 8 |
| Yield context | Masked yield + hole mask + field mask | — | 3 |

## Models

| Model | Depth | Base channels | Parameters |
|---|---|---|---|
| Baseline (ShallowUNet) | 3 + bottleneck | 16 | ~488K |
| Deep (DeepUNet) | 4 + bottleneck | 32 | ~7.8M |
| Deep + Optuna | 4 + bottleneck | 16 (found by search) | ~1.9M |

## Data Split

Fields are split at the **field level** (not patch level) to prevent data leakage
between train, validation, and test sets. The same `seed=42` is used across all
scripts to guarantee identical splits.

| Split | Fields | Purpose |
|---|---|---|
| Train | 20 (67%) | Model training |
| Validation | 6 (20%) | Early stopping, hyperparameter selection |
| Test | 4 (13%) | Final evaluation on held-out fields (full field inference)|

Normalization statistics (`stats.json`) are computed **on train fields only**
and reused during validation, test, and inference.

## Installation

```bash
git clone <repo>
cd cv-yield-inpainting
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Start monitoring

```bash
# Terminal 1 — MLflow
cd cv-yield-inpainting
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --allowed-hosts '*' \
  --cors-allowed-origins 'https://your-server-url'

# Terminal 2 — TensorBoard
tensorboard --logdir artifacts/ --host 0.0.0.0 --port 6006
```

### Validate dataset

```bash
jupyter notebook notebooks/test_dataset.ipynb
```

### Data preparation

> **Note:** The scripts below are part of a separate data preparation pipeline
> executed outside the scope of this project. They are provided for reference
> and reproducibility. Input rasters (DEM derivatives, NDVI, Soil) are assumed
> to be already prepared and aligned before training.

```bash
# 1. Correct DEM artifacts caused by forest canopy radar returns
python scripts/correct_dem.py \
  --data_root data/ \
  --gpkg_path data/fields.gpkg

# 2. Download NDVI rasters from Sentinel Hub (August–September, corn season)
python scripts/get_ndvi_sentinelhub.py

# 3. Download Soil index rasters from Sentinel Hub (March–April, bare soil)
python scripts/get_soils_sentinelhub.py

# 4. Reproject and align yield, NDVI, Soil rasters to DEM grid (5 m, UTM)
python scripts/raster_warp.py \
  --dem data/field_01/dem.tif \
  --gpkg data/field_01/field.gpkg \
  --rasters data/field_01/ndvi.tif \
            data/field_01/soil.tif \
            data/field_01/yield.tif \
  --output_dir data/field_01/

# 5. Reclassify continuous aspect raster to 8 cardinal directions
python scripts/reclassify_aspect.py \
  --input data/field_01/aspect.tif \
  --output data/field_01/aspect_categ.tif
```

### Step 1 — Pre-compute combine harvester stripe angles

**Required before training.** Real GPS dropout gaps in yield maps follow the
combine harvester traversal direction. This script detects the dominant stripe
angle for each field using FFT high-frequency residuals and Canny edge detection
with gradient direction voting.

This is a **one-time operation** — angles are a property of the data, not the
model. Results are saved to `data/stripe_angles.json` and automatically loaded
by `YieldDataset` during training to orient synthetic holes realistically.
If the file is not found, holes are generated axis-aligned (fallback).

```bash
python scripts/compute_stripe_angles.py \
  --data_root data/ \
  --gpkg_path data/fields.gpkg \
  --output data/stripe_angles.json
```

### Step 2 — Train models

```bash
# Baseline (ShallowUNet, 3 encoder levels, ~488K params, 50 epochs)
python -m src.training.train \
  --config configs/baseline.yaml \
  --data_root data/ \
  --gpkg_path data/fields.gpkg \
  --run_dir artifacts/baseline

# Deep UNet (4 encoder levels, ~7.8M params, 80 epochs)
python -m src.training.train \
  --config configs/deep.yaml \
  --data_root data/ \
  --gpkg_path data/fields.gpkg \
  --run_dir artifacts/deep

# Deep UNet + Optuna (50 trials x 15 epochs search, then 100 epochs final training)
# Best hyperparameters saved to artifacts/deep_optuna/optuna_best.json
python -m src.training.optuna_search \
  --config configs/optuna.yaml \
  --data_root data/ \
  --gpkg_path data/fields.gpkg \
  --run_dir artifacts/deep_optuna
```

### Step 3 — Evaluate on test fields

`--max_holes` sets the maximum number of synthetic holes per field during test
evaluation, scaled proportionally to field area.

For the Optuna model, `base_channels` is loaded automatically from
`optuna_best.json` — no manual override needed.

```bash
# Baseline
python -m src.evaluation.test \
  --config configs/baseline.yaml \
  --data_root data/ \
  --model_path artifacts/baseline/baseline/best_model.pt \
  --gpkg_path data/fields.gpkg \
  --run_dir artifacts/baseline/test \
  --max_holes 13

# Deep UNet
python -m src.evaluation.test \
  --config configs/deep.yaml \
  --data_root data/ \
  --model_path artifacts/deep/deep/best_model.pt \
  --gpkg_path data/fields.gpkg \
  --run_dir artifacts/deep/test \
  --max_holes 13

# Deep UNet + Optuna
python -m src.evaluation.test \
  --config configs/optuna.yaml \
  --data_root data/ \
  --model_path artifacts/deep_optuna/best_model.pt \
  --gpkg_path data/fields.gpkg \
  --run_dir artifacts/deep_optuna/test \
  --max_holes 13
```

**Output per test field** (saved to `--run_dir`):

| File | Description |
|---|---|
| `{field}_holes.png` | Yield map with synthetic holes visualized |
| `{field}_filled.png` | Yield map with holes filled by the model |
| `{field}_gt.png` | Full ground truth yield map |
| `{field}_error_map.png` | Prediction error map (RdBu_r colormap, t/ha) |
| `{field}_holes.tif` | GeoTIFF — yield map with holes (original CRS) |
| `{field}_filled.tif` | GeoTIFF — filled yield map (original CRS) |
| `{field}_gt.tif` | GeoTIFF — full ground truth yield map |
| `hist.png` | Error distribution histogram across all test fields |
| `per_field_metrics.json` | Per-field MAE, RMSE, Bias, MAPE, R² |

### Step 4 — Feature importance analysis

Occlusion sensitivity analysis on the best model. Each input channel group is
zeroed out independently and the resulting MAE increase on hole pixels across
all test fields is measured. A larger MAE increase indicates a more important
channel.

```bash
python scripts/feature_importance.py \
  --config configs/optuna.yaml \
  --model_path artifacts/deep_optuna/best_model.pt \
  --data_root data/ \
  --gpkg_path data/fields.gpkg \
  --output artifacts/deep_optuna/feature_importance.png
```

### Step 5 — Production inference on new fields

At inference time, real NaN regions from the yield raster are used directly
as the hole mask — no synthetic hole generation is performed.
`stripe_angles.json` is **not required** for inference.

Prepare a data directory with the same per-field structure as training data:

```
data_production/
├── field_new_01/
│   ├── yield.tif        # yield map with real NaN holes
│   ├── dem.tif
│   ├── hand.tif
│   ├── ndvi.tif
│   ├── slope.tif
│   ├── twi.tif
│   ├── soil.tif
│   ├── rtp_local.tif
│   ├── rtp_regional.tif
│   ├── aspect.tif
│   ├── aspect_categ.tif
│   ├── geomorphons.tif
│   ├── relief_class.tif
│   └── field.gpkg
└── field_new_02/
    └── ...
```

```bash
python -m src.inference.infer \
  --config configs/optuna.yaml \
  --data_root /path/to/data_production \
  --model_path artifacts/deep_optuna/best_model.pt \
  --run_dir artifacts/deep_optuna/infer
```

**Output per field** (saved to `--run_dir`):

| File | Description |
|---|---|
| `{field}_input.png` | Original yield map with NaN holes |
| `{field}_filled.png` | Yield map with holes filled by the model |
| `{field}_pred.png` | Full model prediction raster |
| `{field}_input.tif` | GeoTIFF — original yield map (original CRS) |
| `{field}_filled.tif` | GeoTIFF — filled yield map ready for GIS import |
| `{field}_pred.tif` | GeoTIFF — full model prediction raster |

All GeoTIFF outputs preserve the original coordinate reference system and can be
imported directly into QGIS or ArcGIS for agronomic analysis and management
zone delineation.

## Outputs Summary

| Script | Key outputs |
|---|---|
| `train.py` | `best_model.pt`, `stats.json`, TensorBoard scalars + patch images, MLflow metrics |
| `optuna_search.py` | `best_model.pt`, `stats.json`, `optuna_best.json`, TensorBoard + MLflow |
| `test.py` | Per-field PNG + GeoTIFF (`_holes`, `_filled`, `_gt`), error maps, histogram, MLflow |
| `infer.py` | Per-field PNG + GeoTIFF (`_input`, `_filled`, `_pred`), TensorBoard images, MLflow |
| `feature_importance.py` | Feature importance bar chart PNG, MLflow artifact |

## Metrics

All metrics are computed **exclusively on hole pixels** in real units (t/ha):

| Metric | Description |
|---|---|
| **MAE** | Mean absolute error |
| **RMSE** | Root mean squared error |
| **Bias** | Systematic over/underestimation (mean signed error) |
| **MAPE** | Mean absolute percentage error |
| **R²** | Coefficient of determination |

## Experiment Tracking

- **MLflow** — all training, test, and inference runs logged to `yield_comparison` experiment
- **TensorBoard** — training loss curves, validation patch reconstructions, error maps, histograms
