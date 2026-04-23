import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from torch.utils.tensorboard import SummaryWriter
import mlflow
from src.models.unet import build_model
from src.data.dataset import (
    read_raster,
    normalize,
    aspect_to_sin_cos,
    one_hot_encode,
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

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config(args, cfg):
    experiment_name = cfg.get("experiment_name", None)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    runtime_cfg = cfg.get("runtime", {})

    if experiment_name is not None and getattr(args, "run_name", None) is None:
        args.run_name = experiment_name

    if "base_channels" in model_cfg:
        args.base_channels = model_cfg["base_channels"]

    depth = model_cfg.get("depth", None)
    if depth == 3:
        args.model_name = "baseline"
    elif depth == 4:
        args.model_name = "deep"

    args.patch_size = data_cfg.get("patch_size", args.patch_size)
    args.stride = data_cfg.get("stride", args.stride)

    args.seed = runtime_cfg.get("seed", args.seed)

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

    parser.add_argument("--model_name", default="deep")
    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_dir", default="artifacts/infer")
    parser.add_argument("--run_name", default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    args = apply_config(args, cfg)

    if args.run_name is None:
        args.run_name = cfg.get("experiment_name", "infer")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir / "tensorboard")

    mlflow.set_experiment("yield_infer")

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

    # load base_channels from optuna if available
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
    fields = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_artifact(args.config)
        mlflow.log_params({f"config.{k}": v for k, v in flatten_config(cfg).items()})

        mlflow.log_params({
            "runtime.data_root": args.data_root,
            "runtime.model_path": args.model_path,
            "runtime.gpkg_path": args.gpkg_path if args.gpkg_path is not None else "field.gpkg per field",
            "runtime.run_dir": str(run_dir),
            "runtime.run_name": args.run_name,
            "effective.model_name": args.model_name,
            "effective.base_channels": args.base_channels,
            "effective.patch_size": args.patch_size,
            "effective.stride": args.stride,
            "effective.seed": args.seed,
            "effective.in_channels": in_channels,
        })

        mlflow.log_artifact(str(stats_path))

        for field in fields:
            print("Infer:", field)

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

            # real NaN holes for inpainting
            hole_mask_full = valid_mask.copy()
            missing_mask = (field_mask == 1) & (valid_mask == 0)

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

                    h_patch = hole_mask_full[y:y1, x:x1]
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

            filled = yield_r.copy()
            filled[missing_mask] = pred[missing_mask]

            pred[field_mask == 0] = np.nan
            filled[field_mask == 0] = np.nan

            input_vis = yield_r.copy()
            input_vis[field_mask == 0] = np.nan
            save_map(input_vis, run_dir / f"{field}_input.png", "Input with holes")

            save_map(pred, run_dir / f"{field}_pred.png", "Prediction")
            save_map(filled, run_dir / f"{field}_filled.png", "Filled")

            save_geotiff(field_dir / "yield.tif", pred, run_dir / f"{field}_pred.tif")
            save_geotiff(field_dir / "yield.tif", filled, run_dir / f"{field}_filled.tif")
            save_geotiff(field_dir / "yield.tif", yield_r, run_dir / f"{field}_input.tif")

            valid_input = input_vis[np.isfinite(input_vis)]
            if valid_input.size > 0:
                vmin = float(np.percentile(valid_input, 0.1))
                vmax = float(np.percentile(valid_input, 99.9))
            else:
                vmin, vmax = 0.0, 1.0

            def to_rgb_gray_bg(arr, field_mask, vmin, vmax):
                rgb = np.full((*arr.shape, 3), 0.35, dtype=np.float32)
                valid = (field_mask == 1) & np.isfinite(arr)
                if valid.sum() == 0:
                    return torch.tensor(rgb).permute(2, 0, 1)
                t = np.zeros_like(arr, dtype=np.float32)
                t[valid] = np.clip((arr[valid] - vmin) / (vmax - vmin + 1e-6), 0, 1)
                rgb[..., 0] = np.where(valid, np.clip(1.5 - abs(t*4-1.0), 0, 1), rgb[..., 0])
                rgb[..., 1] = np.where(valid, np.clip(1.5 - abs(t*4-2.0), 0, 1), rgb[..., 1])
                rgb[..., 2] = np.where(valid, np.clip(1.5 - abs(t*4-3.0), 0, 1), rgb[..., 2])
                return torch.tensor(rgb).permute(2, 0, 1)

           # 1. field contour
            field_contour = np.full((*field_mask.shape, 3), 0.35, dtype=np.float32)
            field_contour[field_mask == 1, 0] = 1.0 
            field_contour[field_mask == 1, 1] = 0.5 
            field_contour[field_mask == 1, 2] = 0.0 
            writer.add_image(f"{field}/1_field_contour",
                torch.tensor(field_contour).permute(2, 0, 1), 0)

            # 2. yield original
            writer.add_image(f"{field}/2_input",
                to_rgb_gray_bg(input_vis, field_mask, vmin, vmax), 0)

            # 3. yield original + mask
            input_holes_rgb = to_rgb_gray_bg(input_vis, field_mask, vmin, vmax).numpy()
            input_holes_rgb[0, missing_mask & (field_mask == 1)] = 0.0
            input_holes_rgb[1, missing_mask & (field_mask == 1)] = 0.0
            input_holes_rgb[2, missing_mask & (field_mask == 1)] = 0.0
            writer.add_image(f"{field}/3_input_with_holes",
                torch.tensor(input_holes_rgb), 0)

            # 4. reconstraction
            writer.add_image(f"{field}/4_output",
                to_rgb_gray_bg(filled, field_mask, vmin, vmax), 0)

            mlflow.log_artifact(str(run_dir / f"{field}_input.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_pred.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_filled.png"))
            mlflow.log_artifact(str(run_dir / f"{field}_input.tif"))
            mlflow.log_artifact(str(run_dir / f"{field}_pred.tif"))
            mlflow.log_artifact(str(run_dir / f"{field}_filled.tif"))

    writer.close()


if __name__ == "__main__":
    main()
