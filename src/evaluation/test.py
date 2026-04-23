import argparse
import json
from pathlib import Path
import random
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from torch.utils.tensorboard import SummaryWriter
import mlflow
from src.data.split import split_fields
from src.models.unet import build_model
from src.data.dataset import (
    read_raster,
    normalize,
    aspect_to_sin_cos,
    one_hot_encode,
    generate_holes,
)


# =========================
# utils
# =========================

def denormalize(arr, mean, std):
    return arr * std + mean


def norm(x):
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    fill_value = float(np.nanmean(x[valid]))
    x = np.nan_to_num(x, nan=fill_value)

    x_min = float(x.min())
    x_max = float(x.max())

    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - x_min) / (x_max - x_min + 1e-6)).astype(np.float32)


def save_map(arr, path, title=""):
    arr = np.where(np.isfinite(arr), arr, np.nan)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_geotiff(ref_path, arr, out_path):
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999,
        compress="lzw",
    )

    arr = arr.astype(np.float32)
    arr[~np.isfinite(arr)] = -9999

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def plot_hist(errors, path):
    if len(errors) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=100)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def flatten_config(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def apply_colormap(arr: np.ndarray, field_mask: np.ndarray,
                   vmin: float = None, vmax: float = None) -> torch.Tensor:
    valid = (field_mask == 1) & np.isfinite(arr)
    out = np.zeros((*arr.shape, 3), dtype=np.float32)
    if valid.sum() == 0:
        return torch.zeros(3, *arr.shape, dtype=torch.float32)
    
    if vmin is None:
        vmin = float(arr[valid].min())
    if vmax is None:
        vmax = float(arr[valid].max())
    
    t = np.zeros_like(arr, dtype=np.float32)
    if abs(vmax - vmin) > 1e-6:
        t[valid] = np.clip((arr[valid] - vmin) / (vmax - vmin), 0, 1)
    out[..., 0] = np.where(valid, np.clip(1.5 - abs(t * 4 - 1.0), 0, 1), 0)
    out[..., 1] = np.where(valid, np.clip(1.5 - abs(t * 4 - 2.0), 0, 1), 0)
    out[..., 2] = np.where(valid, np.clip(1.5 - abs(t * 4 - 3.0), 0, 1), 0)
    return torch.tensor(out, dtype=torch.float32).permute(2, 0, 1)


# =========================
# geo
# =========================

def rasterize_field(gpkg_path, reference_raster):
    gdf = gpd.read_file(gpkg_path)

    with rasterio.open(reference_raster) as src:
        transform = src.transform
        shape = (src.height, src.width)
        raster_crs = src.crs

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

    mask = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask.astype(np.float32)


def resolve_gpkg(field_dir: Path, global_gpkg_path: str | None):
    local_gpkg = field_dir / "field.gpkg"
    if local_gpkg.exists():
        return local_gpkg

    if global_gpkg_path is not None:
        return Path(global_gpkg_path)

    raise FileNotFoundError(f"Не найден field.gpkg для поля: {field_dir.name}")


def make_grid(size, patch, stride):
    if size < patch:
        return [0]

    coords = list(range(0, size - patch + 1, stride))

    if len(coords) == 0:
        coords = [0]

    if coords[-1] != size - patch:
        coords.append(size - patch)

    return coords


# =========================
# config
# =========================

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config_to_args(args, cfg):
    experiment_name = cfg.get("experiment_name", None)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    holes_cfg = cfg.get("holes", {})
    runtime_cfg = cfg.get("runtime", {})

    if experiment_name is not None and getattr(args, "run_name", None) is None:
        args.run_name = experiment_name

    if "base_channels" in model_cfg:
        args.base_channels = model_cfg["base_channels"]

    depth = model_cfg.get("depth", None)

    if args.mode in ["baseline", "shallow", "deep"]:
        args.model_name = args.mode
    else:
        if depth == 3:
            args.model_name = "baseline"
        elif depth == 4:
            args.model_name = "deep"

    if "patch_size" in data_cfg:
        args.patch_size = data_cfg["patch_size"]
    if "stride" in data_cfg:
        args.stride = data_cfg["stride"]

    if args.max_holes is None:
        args.max_holes = holes_cfg.get("max_holes", 7)
    args.min_hole_size = holes_cfg.get("min_hole_size", args.min_hole_size)
    args.max_hole_size = holes_cfg.get("max_hole_size", args.max_hole_size)

    args.seed = runtime_cfg.get("seed", args.seed)
    args.val_ratio = runtime_cfg.get("val_ratio", args.val_ratio)
    args.test_ratio = runtime_cfg.get("test_ratio", args.test_ratio)

    return args


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpkg_path", default=None)

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "baseline", "shallow", "deep"],
    )

    parser.add_argument("--model_name", default="deep")
    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)

    parser.add_argument("--max_holes", type=int, default=None)
    parser.add_argument("--min_hole_size", type=int, default=5)
    parser.add_argument("--max_hole_size", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--run_dir", default="artifacts/test")
    parser.add_argument("--run_name", default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    args = apply_config_to_args(args, cfg)

    if args.run_name is None:
        args.run_name = cfg.get("experiment_name", "test_run")

    actual_mode = args.mode

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir / "tensorboard")

    #mlflow.set_experiment("yield_test")
    mlflow.set_experiment("yield_comparison")

    # =========================
    # split
    # =========================

    train_fields, val_fields, test_fields = split_fields(
        args.data_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # =========================
    # stats
    # =========================

    stats_path = Path(args.model_path).parent / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    cont_mean = stats["continuous_mean"]
    cont_std = stats["continuous_std"]
    cat_classes = stats["categorical_classes"]

    y_mean = cont_mean["yield"]
    y_std = cont_std["yield"]

    cont_feats = [
        "dem", "hand", "ndvi", "rtp_local",
        "rtp_regional", "slope", "soil", "twi"
    ]

    cat_feats = [
        "geomorphons", "relief_class", "aspect_categ"
    ]

    # =========================
    # model
    # =========================

    in_channels = len(cont_feats) + 2
    for feat in cat_feats:
        in_channels += len(cat_classes[feat])
    in_channels += 3

    # load best base_channels from optuna if available
    optuna_best_path = Path(args.model_path).parent / "optuna_best.json"
    if optuna_best_path.exists():
        with open(optuna_best_path, "r") as f:
            optuna_best = json.load(f)
        args.base_channels = int(optuna_best.get("base_channels", args.base_channels))
        print(f"Loaded base_channels={args.base_channels} from optuna_best.json")
        
    model = build_model(
        args.model_name,
        in_channels=in_channels,
        base_channels=args.base_channels,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # =========================
    # data
    # =========================

    data_root = Path(args.data_root)
    all_errors = []
    per_field_metrics = []
    all_gt = []

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_artifact(args.config)
        flat_cfg = flatten_config(cfg)
        mlflow.log_params({f"config.{k}": v for k, v in flat_cfg.items()})

        mlflow.log_params({
            "runtime.mode": actual_mode,
            "runtime.data_root": args.data_root,
            "runtime.gpkg_path": args.gpkg_path if args.gpkg_path is not None else "field.gpkg per field",
            "runtime.model_path": args.model_path,
            "runtime.run_dir": str(run_dir),
            "runtime.run_name": args.run_name,
            "effective.model_name": args.model_name,
            "effective.base_channels": args.base_channels,
            "effective.patch_size": args.patch_size,
            "effective.stride": args.stride,
            "effective.max_holes": args.max_holes,
            "effective.min_hole_size": args.min_hole_size,
            "effective.max_hole_size": args.max_hole_size,
            "effective.seed": args.seed,
            "effective.val_ratio": args.val_ratio,
            "effective.test_ratio": args.test_ratio,
            "effective.in_channels": in_channels,
            "effective.test_fields": len(test_fields),
        })

        mlflow.log_artifact(str(stats_path))

        # load pre-computed stripe angles if available
        stripe_angles_path = data_root / "stripe_angles.json"
        if stripe_angles_path.exists():
            with open(stripe_angles_path, "r", encoding="utf-8") as f:
                stripe_angles = json.load(f)
        else:
            stripe_angles = {}

        for field_idx, field in enumerate(test_fields):
            print("Processing:", field)

            field_dir = data_root / field
            gpkg_path = resolve_gpkg(field_dir, args.gpkg_path)

            rasters = {}
            for feat in cont_feats:
                rasters[feat] = read_raster(field_dir / f"{feat}.tif")

            rasters["aspect"] = read_raster(field_dir / "aspect.tif")

            for feat in cat_feats:
                rasters[feat] = read_raster(field_dir / f"{feat}.tif")

            yield_r = read_raster(field_dir / "yield.tif")

            field_mask = rasterize_field(gpkg_path, field_dir / "yield.tif")

            valid_mask = np.isfinite(yield_r).astype(np.float32)
            valid_mask = valid_mask * field_mask

            rng = random.Random(args.seed + field_idx)

            field_area = valid_mask.sum()
            hole_area = args.min_hole_size * args.max_hole_size
            area_based_holes = int(field_area * 0.15 / hole_area)
            dynamic_max_holes = min(args.max_holes, area_based_holes)
            dynamic_max_holes = max(dynamic_max_holes, 3)  # минимум 3

            eval_mask = generate_holes(
                valid_mask=valid_mask,
                rng=rng,
                max_holes=dynamic_max_holes,  # ← исправить
                min_size=args.min_hole_size,
                max_size=args.max_hole_size,
                min_valid_fraction=0.5,
                max_attempts_per_hole=20,
                stripe_angle=stripe_angles.get(field, None),
            )
            eval_mask = eval_mask * valid_mask

            H, W = yield_r.shape

            pred_sum = np.zeros((H, W), dtype=np.float32)
            count = np.zeros((H, W), dtype=np.float32)

            ys = make_grid(H, args.patch_size, args.stride)
            xs = make_grid(W, args.patch_size, args.stride)

            for y in ys:
                for x in xs:
                    y1 = y + args.patch_size
                    x1 = x + args.patch_size

                    f_patch = field_mask[y:y1, x:x1]

                    if f_patch.mean() < 0.05:
                        continue

                    channels = []

                    for feat in cont_feats:
                        p = rasters[feat][y:y1, x:x1]
                        p = normalize(p, cont_mean[feat], cont_std[feat])
                        channels.append(p)

                    a = rasters["aspect"][y:y1, x:x1]
                    nan_mask = np.isnan(a)
                    a = np.nan_to_num(a, nan=0.0)

                    s, c = aspect_to_sin_cos(a)
                    s[nan_mask] = 0.0
                    c[nan_mask] = 0.0
                    channels += [s, c]

                    for feat in cat_feats:
                        p = rasters[feat][y:y1, x:x1]
                        p = np.nan_to_num(p, nan=-9999).astype(np.int32)
                        oh = one_hot_encode(p, cat_classes[feat])
                        channels += list(oh)

                    y_patch = yield_r[y:y1, x:x1]
                    y_norm = normalize(y_patch, y_mean, y_std)

                    h_patch = eval_mask[y:y1, x:x1]
                    masked_y = y_norm * h_patch

                    channels += [masked_y, h_patch, f_patch]

                    x_tensor = torch.tensor(
                        np.stack(channels),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred_patch = model(x_tensor)[0, 0].cpu().numpy()

                    pred_sum[y:y1, x:x1] += pred_patch * f_patch
                    count[y:y1, x:x1] += f_patch

            pred = pred_sum / np.maximum(count, 1.0)
            pred = denormalize(pred, y_mean, y_std)

            missing = (eval_mask == 0) & (valid_mask == 1)

            gt = yield_r[missing]
            pr = pred[missing]

            errors = pr - gt
            all_errors.append(errors)
            all_gt.append(gt)

            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            bias = float(np.mean(errors))

            # MAPE
            mask_nz = np.abs(gt) > 0.1
            mape = float(np.mean(np.abs(errors[mask_nz] / gt[mask_nz])) * 100) \
                if mask_nz.sum() > 0 else float("nan")

            # R²
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((gt - np.mean(gt)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-8))

            per_field_metrics.append({
                "field": field,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "mape": mape,   # ← добавить
                "r2":   r2,     # ← добавить
            })

            print(field, mae, rmse, bias)

            mlflow.log_metric(f"{field}_mae", mae)
            mlflow.log_metric(f"{field}_rmse", rmse)
            mlflow.log_metric(f"{field}_bias", bias)
            mlflow.log_metric(f"{field}_mape", mape)
            mlflow.log_metric(f"{field}_r2",   r2)

            filled = yield_r.copy()
            filled[missing] = pred[missing]

            pred[field_mask == 0] = np.nan
            filled[field_mask == 0] = np.nan
            yield_vis = yield_r.copy()
            yield_vis[field_mask == 0] = np.nan

            gt_with_holes = yield_r.copy()
            gt_with_holes[missing] = np.nan
            gt_with_holes[field_mask == 0] = np.nan

            save_map(gt_with_holes, run_dir / f"{field}_holes.png",  "Field with holes")
            save_map(filled,        run_dir / f"{field}_filled.png", "Filled")
            save_map(yield_vis,     run_dir / f"{field}_gt.png",     "GT full")

            save_geotiff(field_dir / "yield.tif", gt_with_holes, run_dir / f"{field}_holes.tif")
            save_geotiff(field_dir / "yield.tif", filled,        run_dir / f"{field}_filled.tif")
            save_geotiff(field_dir / "yield.tif", yield_vis,     run_dir / f"{field}_gt.tif")

            # единая шкала по GT полного поля
            valid_gt = yield_vis[np.isfinite(yield_vis)]
            if valid_gt.size > 0:
                vmin = float(np.percentile(valid_gt, 0.05))
                vmax = float(np.percentile(valid_gt, 99.95))
            else:
                vmin, vmax = 0.0, 1.0

            # вспомогательная функция — серый фон вне поля
            def to_rgb_gray_bg(arr, field_mask, vmin, vmax):
                rgb = np.full((*arr.shape, 3), 0.35, dtype=np.float32)
                valid = (field_mask == 1) & np.isfinite(arr)
                if valid.sum() == 0:
                    return torch.tensor(rgb).permute(2, 0, 1)
                t = np.zeros_like(arr, dtype=np.float32)
                t[valid] = np.clip((arr[valid] - vmin) / (vmax - vmin + 1e-6), 0, 1)
                rgb[..., 0] = np.where(valid, np.clip(1.5 - abs(t * 4 - 1.0), 0, 1), rgb[..., 0])
                rgb[..., 1] = np.where(valid, np.clip(1.5 - abs(t * 4 - 2.0), 0, 1), rgb[..., 1])
                rgb[..., 2] = np.where(valid, np.clip(1.5 - abs(t * 4 - 3.0), 0, 1), rgb[..., 2])
                return torch.tensor(rgb).permute(2, 0, 1)

            # =========================
            # reconstruction
            # =========================

            # 1_input_masked
            input_rgb = to_rgb_gray_bg(gt_with_holes, field_mask, vmin, vmax).numpy()
            input_rgb[0, missing & (field_mask == 1)] = 1.0
            input_rgb[1, missing & (field_mask == 1)] = 1.0
            input_rgb[2, missing & (field_mask == 1)] = 1.0

            # 2_output_filled
            filled_rgb = to_rgb_gray_bg(filled, field_mask, vmin, vmax).numpy()
            hole_f = missing.astype(np.float32)
            border = np.zeros_like(hole_f, dtype=bool)
            border[1:-1, 1:-1] = (
                (hole_f[1:-1,1:-1] == 0) &
                (field_mask[1:-1,1:-1] == 1) & (
                    (hole_f[:-2,1:-1] == 1) & (field_mask[:-2,1:-1] == 1) |
                    (hole_f[2:,1:-1] == 1) & (field_mask[2:,1:-1] == 1) |
                    (hole_f[1:-1,:-2] == 1) & (field_mask[1:-1,:-2] == 1) |
                    (hole_f[1:-1,2:] == 1) & (field_mask[1:-1,2:] == 1)
                )
            )

            filled_rgb[0, border] = 0.0
            filled_rgb[1, border] = 0.0
            filled_rgb[2, border] = 0.0

            writer.add_image(f"{field}/1_input_masked",
                torch.tensor(input_rgb), 0)
            writer.add_image(f"{field}/2_output_filled",
                torch.tensor(filled_rgb), 0)
            writer.add_image(f"{field}/3_output",
                to_rgb_gray_bg(filled, field_mask, vmin, vmax), 0)
            writer.add_image(f"{field}/4_ground_truth",
                to_rgb_gray_bg(yield_vis, field_mask, vmin, vmax), 0)

            # =========================
            # masks
            # =========================

            # field_mask
            field_rgb = np.full((*field_mask.shape, 3), 0.35, dtype=np.float32)
            field_rgb[field_mask == 1] = 1.0
            writer.add_image(f"{field}/mask_field",
                torch.tensor(field_rgb).permute(2, 0, 1), 0)

            # hole_mask
            hole_rgb = np.full((*field_mask.shape, 3), 0.35, dtype=np.float32)
            hole_rgb[field_mask == 1] = 1.0                      
            hole_rgb[missing & (field_mask == 1)] = 0.0          
            writer.add_image(f"{field}/mask_holes",
                torch.tensor(hole_rgb).permute(2, 0, 1), 0)

            # residuals
            err_rgb = np.full((*yield_r.shape, 3), 0.35, dtype=np.float32)
            err_rgb[field_mask == 1] = 1.0
            if missing.any():
                err_vals = pred[missing] - yield_r[missing]
                max_abs_err = float(np.abs(err_vals).max())
                if max_abs_err > 1e-6:
                    t_err = np.clip(err_vals / max_abs_err, -1, 1) * 0.5 + 0.5
                    err_rgb[missing, 0] = np.clip(2 * t_err - 1, 0, 1)
                    err_rgb[missing, 1] = 1 - 2 * np.abs(t_err - 0.5)
                    err_rgb[missing, 2] = np.clip(1 - 2 * t_err, 0, 1)
            writer.add_image(f"{field}/mask_residuals",
                torch.tensor(err_rgb).permute(2, 0, 1), 0)

            error_map = np.full_like(yield_r, np.nan)
            error_map[missing] = pred[missing] - yield_r[missing]
            error_map[field_mask == 0] = np.nan
            max_abs = float(np.nanmax(np.abs(error_map[np.isfinite(error_map)])))
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(error_map, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
            plt.colorbar(im, ax=ax, label="Error (t/ha)")
            ax.set_title(f"{field} — prediction error (holes only)")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(run_dir / f"{field}_error_map.png", dpi=150)
            plt.close()

            mlflow.log_artifact(str(run_dir / f"{field}_holes.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_filled.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_gt.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_error_map.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_holes.tif"))
            mlflow.log_artifact(str(run_dir / f"{field}_filled.tif"))
            mlflow.log_artifact(str(run_dir / f"{field}_gt.tif"))

        all_errors = np.concatenate(all_errors) if len(all_errors) > 0 else np.array([], dtype=np.float32)
        all_gt = np.concatenate(all_gt) if len(all_gt) > 0 else np.array([], dtype=np.float32)

        test_mae = float(np.mean(np.abs(all_errors))) if all_errors.size > 0 else float("nan")
        test_rmse = float(np.sqrt(np.mean(all_errors ** 2))) if all_errors.size > 0 else float("nan")
        test_bias = float(np.mean(all_errors)) if all_errors.size > 0 else float("nan")

        mask_nz = np.abs(all_gt) > 0.1
        test_mape = float(np.mean(np.abs(all_errors[mask_nz] / all_gt[mask_nz])) * 100) \
                    if mask_nz.sum() > 0 else float("nan")

        ss_res = np.sum(all_errors ** 2)
        ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
        test_r2 = float(1 - ss_res / (ss_tot + 1e-8)) if all_gt.size > 0 else float("nan")

        print("GLOBAL:", test_mae, test_rmse, test_bias, test_mape, test_r2)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_bias", test_bias)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("test_r2",   test_r2)

        plot_hist(all_errors, run_dir / "hist.png")
        mlflow.log_artifact(str(run_dir / "hist.png"))

        with open(run_dir / "per_field_metrics.json", "w", encoding="utf-8") as f:
            json.dump(per_field_metrics, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(run_dir / "per_field_metrics.json"))

    writer.close()


if __name__ == "__main__":
    main()
