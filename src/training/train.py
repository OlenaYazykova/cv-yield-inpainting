import argparse
import json
from pathlib import Path
import random

import yaml
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mlflow

from src.data.dataset import YieldDataset
from src.data.compute_stats import compute_stats
from src.models.unet import build_model
from src.training.losses import masked_loss, masked_mae, masked_rmse

from src.data.split import split_fields

# =========================
# utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
                   vmin: float, vmax: float) -> torch.Tensor:
    valid = (field_mask == 1) & np.isfinite(arr)
    out = np.zeros((*arr.shape, 3), dtype=np.float32)
    if valid.sum() == 0:
        return torch.zeros(3, *arr.shape, dtype=torch.float32)
    t = np.zeros_like(arr, dtype=np.float32)
    t[valid] = np.clip((arr[valid] - vmin) / (vmax - vmin + 1e-6), 0, 1)
    out[..., 0] = np.where(valid, np.clip(1.5 - abs(t * 4 - 1.0), 0, 1), 0)
    out[..., 1] = np.where(valid, np.clip(1.5 - abs(t * 4 - 2.0), 0, 1), 0)
    out[..., 2] = np.where(valid, np.clip(1.5 - abs(t * 4 - 3.0), 0, 1), 0)
    return torch.tensor(out, dtype=torch.float32).permute(2, 0, 1)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# =========================
# train / val
# =========================

def train_one_epoch(model, loader, optimizer, scaler, device, known_w, smooth_w):
    model.train()

    total_loss = 0

    for x, y, hole, field in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        hole = hole.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            pred = model(x)

            loss = masked_loss(
                pred, y, hole, field,
                known_weight=known_w,
                smooth_weight=smooth_w
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, known_w, smooth_w, y_mean, y_std):
    model.eval()

    total_loss = 0
    all_pred = []
    all_gt   = []

    for x, y, hole, field in loader:
        x     = x.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)
        hole  = hole.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)

        pred = model(x)

        loss = masked_loss(pred, y, hole, field,
                           known_weight=known_w, smooth_weight=smooth_w)
        total_loss += loss.item()

        # denormalize and collect hole pixels only
        pred_real = (pred * y_std + y_mean)
        y_real    = (y    * y_std + y_mean)

        missing = ((1.0 - hole) * field).bool()

        all_pred.append(pred_real[missing].cpu().numpy())
        all_gt.append(y_real[missing].cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)
    errors   = all_pred - all_gt

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # MAPE
    mask_nz = np.abs(all_gt) > 0.1
    mape = float(np.mean(np.abs(errors[mask_nz] / all_gt[mask_nz])) * 100) \
           if mask_nz.sum() > 0 else float("nan")

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    n = max(len(loader), 1)
    return total_loss / n, mae, rmse, bias, mape, r2, errors


# =========================
# config
# =========================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config(args, cfg):
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    runtime_cfg = cfg.get("runtime", {})

    args.vis_field = runtime_cfg.get("vis_field", getattr(args, "vis_field", None))

    args.base_channels = model_cfg.get("base_channels", args.base_channels)

    depth = model_cfg.get("depth", 4)
    args.model_name = "baseline" if depth == 3 else "deep"

    args.patch_size = data_cfg.get("patch_size", args.patch_size)
    args.stride = data_cfg.get("stride", args.stride)

    args.batch_size = train_cfg.get("batch_size", args.batch_size)
    args.lr = train_cfg.get("lr", args.lr)
    args.epochs = train_cfg.get("epochs", args.epochs)

    args.known_weight = loss_cfg.get("known_weight", args.known_weight)
    args.smooth_weight = loss_cfg.get("smooth_weight", args.smooth_weight)

    args.seed = runtime_cfg.get("seed", args.seed)

    args.val_ratio = runtime_cfg.get("val_ratio", getattr(args, "val_ratio", 0.2))

    args.test_ratio = float(runtime_cfg.get("test_ratio", getattr(args, "test_ratio", 0.2)))

    return args


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--gpkg_path", required=True)

    parser.add_argument("--run_dir", default="artifacts/train")
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)

    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--known_weight", type=float, default=0.1)
    parser.add_argument("--smooth_weight", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--vis_field", type=str, default=None,
                    help="Field name to visualize during training")

    args = parser.parse_args()

    cfg = load_config(args.config)
    args = apply_config(args, cfg)

    if args.run_name is None:
        args.run_name = cfg.get("experiment_name", "train_run")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir / "tensorboard")

    mlflow.set_experiment("yield_comparison")

    # =========================
    # split
    # =========================

    train_fields, val_fields, _ = split_fields(
        args.data_root,
        seed=args.seed,
        val_ratio=args.val_ratio if hasattr(args, "val_ratio") else 0.2,
        test_ratio=args.test_ratio,
    )

    # =========================
    # stats
    # =========================

    stats_path = run_dir / "stats.json"

    if not stats_path.exists():
        compute_stats(args.data_root, train_fields, str(stats_path))

    with open(stats_path, "r") as f:
        stats = json.load(f)
    y_mean = float(stats["continuous_mean"]["yield"])
    y_std  = float(stats["continuous_std"]["yield"])

    # =========================
    # datasets
    # =========================

    train_ds = YieldDataset(
        root_dir=args.data_root,
        field_list=train_fields,
        stats_path=str(stats_path),
        gpkg_path=args.gpkg_path,
        patch_size=args.patch_size,
        stride=args.stride,
        mode="train",
        seed=None,
    )

    val_ds = YieldDataset(
        root_dir=args.data_root,
        field_list=val_fields,
        stats_path=str(stats_path),
        gpkg_path=args.gpkg_path,
        patch_size=args.patch_size,
        stride=args.stride,
        mode="val",
        seed=args.seed,
        max_holes=cfg.get("holes", {}).get("max_holes", 4),
        min_hole_size=cfg.get("holes", {}).get("min_hole_size", 5),
        max_hole_size=cfg.get("holes", {}).get("max_hole_size", 20),
    )

    train_loader = make_loader(train_ds, args.batch_size, True, 0)
    val_loader = make_loader(val_ds, args.batch_size, False, 0)

    # =========================
    # model
    # =========================

    sample_x, _, _, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    model = build_model(
        args.model_name,
        in_channels=in_channels,
        base_channels=args.base_channels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    best_val = float("inf")

    # =========================
    # training loop
    # =========================

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_artifact(args.config)
        mlflow.log_params(flatten_config(cfg))

        for epoch in range(args.epochs):

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler,
                device, args.known_weight, args.smooth_weight
            )

            val_loss, val_mae, val_rmse, val_bias, val_mape, val_r2, val_errors = validate(
                model, val_loader, device,
                args.known_weight, args.smooth_weight,
                y_mean, y_std
            )

            print(f"[{epoch}] train={train_loss:.4f} val={val_loss:.4f}")

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val",   val_loss,   epoch)
            writer.add_scalar("val/mae",    val_mae,    epoch)
            writer.add_scalar("val/rmse",   val_rmse,   epoch)
            writer.add_scalar("val/bias",   val_bias,   epoch)
            writer.add_scalar("val/mape",   val_mape,   epoch)
            writer.add_scalar("val/r2",     val_r2,     epoch)
            
            writer.add_histogram("val/error_distribution", val_errors, epoch)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            mlflow.log_metric("val_mae",    val_mae,    step=epoch)
            mlflow.log_metric("val_rmse",   val_rmse,   step=epoch)
            mlflow.log_metric("val_bias",   val_bias,   step=epoch)
            mlflow.log_metric("val_mape",   val_mape,   step=epoch)
            mlflow.log_metric("val_r2",     val_r2,     step=epoch)

            # =========================
            # validation patch visualization
            # =========================
            model.eval()
            with torch.no_grad():
                if args.vis_field is not None:
                    vis_idx = next(
                        (i for i, (f, y, x) in enumerate(val_ds.samples) if f == args.vis_field),
                        0  # fallback to first patch if field not found
                    )
                else:
                    vis_idx = 0
                vis_x, vis_y, vis_hole, vis_field_tensor = val_ds[vis_idx]
                vis_pred = model(
                    vis_x.unsqueeze(0).to(device)
                )[0].cpu()

            hole_np   = vis_hole[0].numpy()
            field_np  = vis_field_tensor[0].numpy()
            gt_norm   = vis_y[0].numpy()
            pred_norm = vis_pred[0].numpy()

            gt_real   = gt_norm   * y_std + y_mean
            pred_real = pred_norm * y_std + y_mean

            # full ground truth and ground truth with holes
            gt_full     = np.where((field_np == 1) & np.isfinite(gt_real), gt_real, np.nan)
            gt_masked   = np.where((field_np == 1) & (hole_np == 1) & np.isfinite(gt_real), gt_real, np.nan)
            pred_masked = np.where((field_np == 1) & np.isfinite(pred_real), pred_real, np.nan)

            # filled: observed pixels from GT + hole pixels from prediction
            filled = gt_full.copy()
            filled[hole_np == 0] = pred_masked[hole_np == 0]

            # unified color scale based on full ground truth
            valid_gt = gt_full[np.isfinite(gt_full)]
            if valid_gt.size > 0:
                vmin = float(np.percentile(valid_gt, 0.05))
                vmax = float(np.percentile(valid_gt, 99.95))
            else:
                vmin, vmax = 0.0, 1.0

            # helper function — grey background outside field boundary
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
            input_rgb = to_rgb_gray_bg(gt_masked, field_np, vmin, vmax).numpy()
            hole_inside = (hole_np == 0) & (field_np == 1)
            input_rgb[0, hole_inside] = 0.0
            input_rgb[1, hole_inside] = 0.0
            input_rgb[2, hole_inside] = 0.0

            # 2_output_filled
            filled_rgb = to_rgb_gray_bg(filled, field_np, vmin, vmax).numpy()
            
            # border pixels inside field boundary only
            hole_f = hole_np.astype(np.float32)
            field_f = field_np.astype(np.float32)
            hole_inside_f = ((hole_np == 0) & (field_np == 1)).astype(np.float32)
            border = np.zeros_like(hole_f, dtype=bool)
            border[1:-1, 1:-1] = (
                (hole_f[1:-1,1:-1] == 0) &
                (field_np[1:-1,1:-1] == 1) & (
                    (hole_f[:-2,1:-1] == 1) & (field_np[:-2,1:-1] == 1) |
                    (hole_f[2:,1:-1] == 1) & (field_np[2:,1:-1] == 1) |
                    (hole_f[1:-1,:-2] == 1) & (field_np[1:-1,:-2] == 1) |
                    (hole_f[1:-1,2:] == 1) & (field_np[1:-1,2:] == 1)
                )
            )
            filled_rgb[0, border] = 0.0
            filled_rgb[1, border] = 0.0
            filled_rgb[2, border] = 0.0

            writer.add_image("reconstruction/1_input_masked",  torch.tensor(input_rgb),                       epoch)
            writer.add_image("reconstruction/2_output_filled", torch.tensor(filled_rgb),                      epoch)
            writer.add_image("reconstruction/3_output",        to_rgb_gray_bg(filled,  field_np, vmin, vmax), epoch)
            writer.add_image("reconstruction/4_ground_truth",  to_rgb_gray_bg(gt_full, field_np, vmin, vmax), epoch)

            # =========================
            # masks
            # =========================

            # field_mask
            field_rgb = np.full((*field_np.shape, 3), 0.35, dtype=np.float32)
            field_rgb[field_np == 1] = 1.0
            writer.add_image("masks/field_mask",
                torch.tensor(field_rgb).permute(2, 0, 1), epoch)

            #   hole_mask
            hole_rgb = np.full((*field_np.shape, 3), 0.35, dtype=np.float32) 
            hole_rgb[field_np == 1] = 1.0                         
            hole_rgb[(field_np == 1) & (hole_np == 0)] = 0.0                 
            writer.add_image("masks/hole_mask",
                torch.tensor(hole_rgb).permute(2, 0, 1), epoch)

            # residuals
            err_rgb = np.full((*gt_real.shape, 3), 0.35, dtype=np.float32) 
            err_rgb[field_np == 1] = 1.0

            hole_pixels = (hole_np == 0) & (field_np == 1) & np.isfinite(gt_real) & np.isfinite(pred_real)
            if hole_pixels.any():
                errors_holes = pred_real[hole_pixels] - gt_real[hole_pixels]
                max_abs = float(np.abs(errors_holes).max())
                if max_abs > 1e-6:
                    t_err = np.clip(errors_holes / max_abs, -1, 1) * 0.5 + 0.5
                    err_rgb[hole_pixels, 0] = np.clip(2 * t_err - 1, 0, 1)
                    err_rgb[hole_pixels, 1] = 1 - 2 * np.abs(t_err - 0.5)
                    err_rgb[hole_pixels, 2] = np.clip(1 - 2 * t_err, 0, 1)

            writer.add_image("masks/residuals",
                torch.tensor(err_rgb).permute(2, 0, 1), epoch)

            # =========================
            # save error map PNG with colorbar every 5 epochs
            # =========================
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                import matplotlib.pyplot as plt
                error_map = np.full_like(gt_real, np.nan)
                if hole_pixels.any():
                    error_map[hole_pixels] = pred_real[hole_pixels] - gt_real[hole_pixels]
                error_map[field_np == 0] = np.nan
                max_abs_plot = float(np.nanmax(np.abs(error_map[np.isfinite(error_map)]))) \
                               if np.isfinite(error_map).any() else 1.0
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(error_map, cmap="RdBu_r",
                               vmin=-max_abs_plot, vmax=max_abs_plot)
                plt.colorbar(im, ax=ax, label="Error (t/ha)")
                ax.set_title(f"Epoch {epoch} — hole error")
                ax.axis("off")
                plt.tight_layout()
                err_path = run_dir / f"error_epoch_{epoch:03d}.png"
                plt.savefig(err_path, dpi=150)
                plt.close()
                mlflow.log_artifact(str(err_path))

            # save best
            if val_loss < best_val:
                best_val = val_loss

                torch.save(
                    model.state_dict(),
                    run_dir / "best_model.pt"
                )

    writer.close()


if __name__ == "__main__":
    main()
