#!/usr/bin/env python3
"""
Feature importance analysis via occlusion sensitivity.
For each input channel, zeroes it out and measures the increase in MAE on hole pixels.
A larger MAE increase means the channel is more important for hole reconstruction.

Usage:
    python scripts/feature_importance.py \
        --config configs/optuna.yaml \
        --model_path artifacts/deep_optuna/best_model.pt \
        --data_root /srv/datasaver/raw/cv_project/data \
        --gpkg_path /srv/datasaver/raw/cv_project/data/fields.gpkg \
        --output artifacts/deep_optuna/feature_importance.png
"""

import argparse
import json
import random
from pathlib import Path
import mlflow

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import rasterio
from rasterio.features import rasterize
import geopandas as gpd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    read_raster, normalize, aspect_to_sin_cos,
    one_hot_encode, generate_holes,
)
from src.data.split import split_fields
from src.models.unet import build_model


# =========================
# geo utils
# =========================

def rasterize_field(gpkg_path, reference_raster):
    gdf = gpd.read_file(gpkg_path)
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        shape = (src.height, src.width)
        raster_crs = src.crs
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    shapes = [(geom, 1) for geom in gdf.geometry
              if geom is not None and not geom.is_empty]
    mask = rasterize(shapes=shapes, out_shape=shape,
                     transform=transform, fill=0, dtype=np.uint8)
    return mask.astype(np.float32)


def resolve_gpkg(field_dir, global_gpkg_path):
    local = field_dir / "field.gpkg"
    if local.exists():
        return local
    if global_gpkg_path is not None:
        return Path(global_gpkg_path)
    raise FileNotFoundError(f"No field.gpkg for {field_dir.name}")


def make_grid(size, patch, stride):
    if size < patch:
        return [0]
    coords = list(range(0, size - patch + 1, stride))
    if not coords:
        coords = [0]
    if coords[-1] != size - patch:
        coords.append(size - patch)
    return coords


# =========================
# inference on one field
# =========================

def predict_field(model, device, channels_list, field_mask,
                  eval_mask, patch_size, stride):
    """Run sliding-window inference. channels_list: list of (H,W) arrays."""
    H, W = field_mask.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)

    for y in make_grid(H, patch_size, stride):
        for x in make_grid(W, patch_size, stride):
            y1, x1 = y + patch_size, x + patch_size
            f_patch = field_mask[y:y1, x:x1]
            if f_patch.mean() < 0.05:
                continue

            patch_channels = [c[y:y1, x:x1] for c in channels_list]
            x_tensor = torch.tensor(
                np.stack(patch_channels), dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_patch = model(x_tensor)[0, 0].cpu().numpy()

            pred_sum[y:y1, x:x1] += pred_patch * f_patch
            count[y:y1, x:x1]    += f_patch

    return pred_sum / np.maximum(count, 1.0)


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Feature importance via occlusion sensitivity"
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--gpkg_path",  default=None)
    parser.add_argument("--output",     required=True,
                        help="Path to save feature importance PNG")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride",     type=int, default=32)
    parser.add_argument("--max_holes",  type=int, default=13)
    args = parser.parse_args()

    # load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg  = cfg.get("model", {})
    data_cfg   = cfg.get("data", {})
    holes_cfg  = cfg.get("holes", {})

    patch_size = data_cfg.get("patch_size", args.patch_size)
    stride     = data_cfg.get("stride",     args.stride)
    depth      = model_cfg.get("depth", 4)
    model_name = "baseline" if depth == 3 else "deep"
    base_ch = model_cfg.get("base_channels", 32)
    optuna_best_path = Path(args.model_path).parent / "optuna_best.json"
    if optuna_best_path.exists():
        with open(optuna_best_path, "r") as f:
            optuna_best = json.load(f)
        base_ch = int(optuna_best.get("base_channels", base_ch))
        print(f"Loaded base_channels={base_ch} from optuna_best.json")
    min_hole   = holes_cfg.get("min_hole_size", 10)
    max_hole   = holes_cfg.get("max_hole_size", 35)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load stats
    stats_path = Path(args.model_path).parent / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    cont_mean  = stats["continuous_mean"]
    cont_std   = stats["continuous_std"]
    cat_classes = stats["categorical_classes"]
    y_mean = cont_mean["yield"]
    y_std  = cont_std["yield"]

    # load stripe angles
    stripe_angles_path = Path(args.data_root) / "stripe_angles.json"
    if stripe_angles_path.exists():
        with open(stripe_angles_path, "r", encoding="utf-8") as f:
            stripe_angles = json.load(f)
    else:
        stripe_angles = {}

    # feature groups
    cont_feats = ["dem", "hand", "ndvi", "rtp_local",
                  "rtp_regional", "slope", "soil", "twi"]
    cat_feats  = ["geomorphons", "relief_class", "aspect_categ"]

    # build channel index map: channel_name -> list of indices
    channel_names  = []
    channel_groups = {}  # group_name -> [idx, ...]

    idx = 0
    for feat in cont_feats:
        channel_groups[feat] = [idx]
        channel_names.append(feat)
        idx += 1

    # aspect sin + cos — treat as one group
    channel_groups["aspect"] = [idx, idx + 1]
    channel_names += ["aspect_sin", "aspect_cos"]
    idx += 2

    for feat in cat_feats:
        n = len(cat_classes[feat])
        channel_groups[feat] = list(range(idx, idx + n))
        channel_names += [f"{feat}_{i}" for i in range(n)]
        idx += n

    # yield context channels — not occluded (they define the task)
    n_total = idx + 3  # masked_yield + hole_mask + field_mask

    # build model
    model = build_model(model_name, in_channels=n_total,
                        base_channels=base_ch).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # test fields
    _, _, test_fields = split_fields(
        args.data_root, seed=args.seed,
        val_ratio=0.2, test_ratio=0.2,
    )

    data_root = Path(args.data_root)

    # accumulate importance per group across all test fields
    group_names = list(cont_feats) + ["aspect"] + list(cat_feats)
    importance  = {g: [] for g in group_names}

    print(f"{'Field':<15} {'Baseline MAE':>14}", end="")
    for g in group_names:
        print(f"  {g:>14}", end="")
    print()
    print("-" * (15 + 16 + 16 * len(group_names)))

    for field_idx, field in enumerate(test_fields):
        field_dir  = data_root / field
        gpkg_path  = resolve_gpkg(field_dir, args.gpkg_path)

        # load rasters
        rasters = {}
        for feat in cont_feats:
            rasters[feat] = read_raster(field_dir / f"{feat}.tif")
        rasters["aspect"] = read_raster(field_dir / "aspect.tif")
        for feat in cat_feats:
            rasters[feat] = read_raster(field_dir / f"{feat}.tif")
        yield_r    = read_raster(field_dir / "yield.tif")
        field_mask = rasterize_field(gpkg_path, field_dir / "yield.tif")

        valid_mask = np.isfinite(yield_r).astype(np.float32) * field_mask

        rng = random.Random(args.seed + field_idx)
        eval_mask = generate_holes(
            valid_mask=valid_mask, rng=rng,
            max_holes=args.max_holes,
            min_size=min_hole, max_size=max_hole,
            min_valid_fraction=0.3,
            max_attempts_per_hole=50,
            stripe_angle=stripe_angles.get(field, None),
        ) * valid_mask

        missing = (eval_mask == 0) & (valid_mask == 1)
        if missing.sum() == 0:
            continue

        # build full channel list
        def build_channels(occlude_group=None):
            chs = []
            for feat in cont_feats:
                p = normalize(rasters[feat].copy(),
                              cont_mean[feat], cont_std[feat])
                if occlude_group == feat:
                    p = np.zeros_like(p)
                chs.append(p)

            a = rasters["aspect"].copy()
            nan_mask = np.isnan(a)
            a = np.nan_to_num(a, nan=0.0)
            s, c = aspect_to_sin_cos(a)
            s[nan_mask] = 0.0
            c[nan_mask] = 0.0
            if occlude_group == "aspect":
                s = np.zeros_like(s)
                c = np.zeros_like(c)
            chs += [s, c]

            for feat in cat_feats:
                p = rasters[feat].copy()
                p = np.nan_to_num(p, nan=-9999).astype(np.int32)
                oh = one_hot_encode(p, cat_classes[feat])
                if occlude_group == feat:
                    oh = np.zeros_like(oh)
                chs += list(oh)

            # yield context
            y_norm   = normalize(yield_r.copy(), y_mean, y_std)
            hole_ch  = eval_mask.copy()
            masked_y = y_norm * hole_ch
            chs += [masked_y, hole_ch, field_mask.copy()]

            return chs

        # baseline prediction
        base_chs  = build_channels(occlude_group=None)
        base_pred = predict_field(model, device, base_chs,
                                  field_mask, eval_mask,
                                  patch_size, stride)
        base_pred_real = base_pred * y_std + y_mean
        base_mae = float(np.mean(np.abs(
            base_pred_real[missing] - yield_r[missing]
        )))

        print(f"{field:<15} {base_mae:>14.4f}", end="")

        # occlusion per group
        for group in group_names:
            occ_chs  = build_channels(occlude_group=group)
            occ_pred = predict_field(model, device, occ_chs,
                                     field_mask, eval_mask,
                                     patch_size, stride)
            occ_pred_real = occ_pred * y_std + y_mean
            occ_mae = float(np.mean(np.abs(
                occ_pred_real[missing] - yield_r[missing]
            )))
            delta = occ_mae - base_mae
            importance[group].append(delta)
            print(f"  {delta:>+14.4f}", end="")

        print()

    # aggregate: mean importance across fields
    mean_imp = {g: float(np.mean(v)) if v else 0.0
                for g, v in importance.items()}

    print("\nMean importance (MAE increase when occluded):")
    for g, v in sorted(mean_imp.items(), key=lambda x: -x[1]):
        print(f"  {g:<20} {v:>+.4f} t/ha")

    # plot
    groups_sorted = sorted(mean_imp.keys(), key=lambda x: -mean_imp[x])
    values = [mean_imp[g] for g in groups_sorted]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(groups_sorted, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MAE increase when channel occluded (t/ha)", fontsize=12)
    ax.set_title("Feature Importance — Occlusion Sensitivity", fontsize=14)
    ax.invert_yaxis()

    red_patch  = mpatches.Patch(color="#d73027", label="Positive — channel helps")
    blue_patch = mpatches.Patch(color="#4575b4", label="Negative — channel hurts")
    ax.legend(handles=[red_patch, blue_patch], fontsize=10)

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    mlflow.set_experiment("yield_comparison")
    with mlflow.start_run(run_name="feature_importance"):
        mlflow.log_artifact(str(output_path))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
