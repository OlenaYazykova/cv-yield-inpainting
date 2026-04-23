import optuna
import torch
import argparse
from pathlib import Path
import json
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow

from src.data.dataset import YieldDataset
from src.models.unet import build_model
from src.training.losses import masked_loss
from src.data.compute_stats import compute_stats
from src.data.split import split_fields as split_fields_full

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


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def suggest_from_space(trial, name, space):
    if space["type"] == "float":
        return trial.suggest_float(name, space["low"], space["high"],
                                   log=space.get("log", False))
    elif space["type"] == "categorical":
        return trial.suggest_categorical(name, space["choices"])
    else:
        raise ValueError(f"Unknown type: {space['type']}")


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


# =========================
# train / val
# =========================

def train_one_epoch(model, loader, optimizer, scaler, device, known_w, smooth_w):
    model.train()
    total_loss = 0.0
    for x, y, hole, field in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        hole = hole.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            pred = model(x)
            loss = masked_loss(pred, y, hole, field,
                               known_weight=known_w, smooth_weight=smooth_w)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_full(model, loader, device, known_w, smooth_w, y_mean, y_std):
    """Full validation with all metrics."""
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []

    for x, y, hole, field in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        hole = hole.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)

        pred = model(x)
        loss = masked_loss(pred, y, hole, field,
                           known_weight=known_w, smooth_weight=smooth_w)
        total_loss += loss.item()

        pred_real = pred * y_std + y_mean
        y_real    = y    * y_std + y_mean
        missing   = ((1.0 - hole) * field).bool()

        all_pred.append(pred_real[missing].cpu().numpy())
        all_gt.append(y_real[missing].cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)
    errors   = all_pred - all_gt

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    mask_nz = np.abs(all_gt) > 0.1
    mape = float(np.mean(np.abs(errors[mask_nz] / all_gt[mask_nz])) * 100) \
           if mask_nz.sum() > 0 else float("nan")

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return total_loss / max(len(loader), 1), mae, rmse, bias, mape, r2, errors


@torch.no_grad()
def validate_loss_only(model, loader, device, known_w, smooth_w):
    """Fast validation — loss only for Optuna trials."""
    model.eval()
    total = 0.0
    for x, y, hole, field in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        hole = hole.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)
        pred = model(x)
        loss = masked_loss(pred, y, hole, field,
                           known_weight=known_w, smooth_weight=smooth_w)
        total += loss.item()
    return total / max(len(loader), 1)


# =========================
# objective
# =========================

def objective(trial, args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optuna_cfg   = cfg.get("optuna", {})
    data_cfg     = cfg.get("data", {})
    holes_cfg    = cfg.get("holes", {})
    train_cfg    = cfg.get("training", {})
    loss_cfg     = cfg.get("loss", {})
    runtime_cfg  = cfg.get("runtime", {})
    model_cfg    = cfg.get("model", {})
    search_space = cfg.get("search_space", {})

    seed      = optuna_cfg.get("seed", 42)
    val_ratio = runtime_cfg.get("val_ratio", 0.2)
    test_ratio = runtime_cfg.get("test_ratio", 0.2)
    set_seed(seed)

    params = {}
    for k, v in search_space.items():
        params[k] = suggest_from_space(trial, k, v)

    lr            = params.get("lr",           train_cfg.get("lr", 1e-3))
    batch_size    = int(params.get("batch_size",   train_cfg.get("batch_size", 8)))
    base_channels = int(params.get("base_channels", model_cfg.get("base_channels", 32)))
    patch_size    = int(params.get("patch_size",   data_cfg.get("patch_size", 128)))
    smooth_w      = params.get("smooth_weight", loss_cfg.get("smooth_weight", 0.02))
    known_w       = params.get("known_weight",  loss_cfg.get("known_weight", 0.1))
    weight_decay  = params.get("weight_decay",  train_cfg.get("weight_decay", 1e-4))

    stride          = data_cfg.get("stride", 64)
    min_valid_ratio = data_cfg.get("min_valid_ratio", 0.5)
    max_holes       = holes_cfg.get("max_holes", 3)
    min_hole_size   = holes_cfg.get("min_hole_size", 5)
    max_hole_size   = holes_cfg.get("max_hole_size", 20)
    epochs          = train_cfg.get("epochs", 10)
    num_workers     = runtime_cfg.get("num_workers", 0)

    train_fields, val_fields, _ = split_fields_full(args.data_root, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    stats_path = Path(args.run_dir) / "stats.json"
    if not stats_path.exists():
        compute_stats(args.data_root, train_fields, str(stats_path))

    train_ds = YieldDataset(
        root_dir=args.data_root, field_list=train_fields,
        stats_path=str(stats_path), gpkg_path=args.gpkg_path,
        patch_size=patch_size, stride=stride, min_valid_ratio=min_valid_ratio,
        max_holes=max_holes, min_hole_size=min_hole_size, max_hole_size=max_hole_size,
        mode="train", seed=None,
    )
    val_ds = YieldDataset(
        root_dir=args.data_root, field_list=val_fields,
        stats_path=str(stats_path), gpkg_path=args.gpkg_path,
        patch_size=patch_size, stride=stride, min_valid_ratio=min_valid_ratio,
        max_holes=max_holes, min_hole_size=min_hole_size, max_hole_size=max_hole_size,
        mode="val", seed=seed,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise optuna.exceptions.TrialPruned()

    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader   = make_loader(val_ds,   batch_size, False, num_workers)

    sample_x, _, _, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    depth      = model_cfg.get("depth", 4)
    model_name = "baseline" if depth == 3 else "deep"

    model = build_model(model_name, in_channels=in_channels,
                        base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    best_val = float("inf")

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, scaler, device, known_w, smooth_w)
        val_loss = validate_loss_only(model, val_loader, device, known_w, smooth_w)
        best_val = min(best_val, val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--gpkg_path", required=True)
    parser.add_argument("--run_dir",   required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Optuna search
    # =========================
    study = optuna.create_study(
        direction=cfg.get("optuna", {}).get("direction", "minimize"),
        sampler=optuna.samplers.TPESampler(seed=cfg.get("optuna", {}).get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda t: objective(t, args, cfg),
        n_trials=cfg.get("optuna", {}).get("n_trials", 30),
        timeout=cfg.get("optuna", {}).get("timeout", None),
    )

    print("BEST PARAMS:")
    print(study.best_params)

    with open(run_dir / "optuna_best.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    with open(run_dir / "optuna_best_value.json", "w") as f:
        json.dump({"best_value": study.best_value,
                   "best_trial": study.best_trial.number}, f, indent=2)

    # =========================
    # Final training with best params
    # =========================
    best        = study.best_params
    cfg_data    = cfg.get("data", {})
    cfg_holes   = cfg.get("holes", {})
    cfg_train   = cfg.get("training", {})
    cfg_loss    = cfg.get("loss", {})
    cfg_runtime = cfg.get("runtime", {})
    cfg_model   = cfg.get("model", {})

    lr            = best.get("lr",           cfg_train.get("lr", 1e-3))
    batch_size    = int(best.get("batch_size",   cfg_train.get("batch_size", 8)))
    base_channels = int(best.get("base_channels", cfg_model.get("base_channels", 32)))
    patch_size    = int(best.get("patch_size",   cfg_data.get("patch_size", 128)))
    smooth_w      = best.get("smooth_weight", cfg_loss.get("smooth_weight", 0.02))
    known_w       = best.get("known_weight",  cfg_loss.get("known_weight", 0.1))
    weight_decay  = best.get("weight_decay",  cfg_train.get("weight_decay", 1e-4))

    epochs          = cfg_train.get("epochs_final", cfg_train.get("epochs", 30))
    num_workers     = cfg_runtime.get("num_workers", 0)
    seed            = cfg.get("optuna", {}).get("seed", 42)
    val_ratio       = cfg_runtime.get("val_ratio", 0.2)
    test_ratio      = cfg_runtime.get("test_ratio", 0.2)
    stride          = cfg_data.get("stride", 64)
    min_valid_ratio = cfg_data.get("min_valid_ratio", 0.5)
    max_holes       = cfg_holes.get("max_holes", 3)
    min_hole_size   = cfg_holes.get("min_hole_size", 5)
    max_hole_size   = cfg_holes.get("max_hole_size", 20)
    run_name        = cfg.get("experiment_name", "deep_optuna")
    depth           = cfg_model.get("depth", 4)
    model_name      = "baseline" if depth == 3 else "deep"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    train_fields, val_fields, _ = split_fields_full(args.data_root, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    stats_path = run_dir / "stats.json"
    if not stats_path.exists():
        compute_stats(args.data_root, train_fields, str(stats_path))

    with open(stats_path, "r") as f:
        stats = json.load(f)
    y_mean = float(stats["continuous_mean"]["yield"])
    y_std  = float(stats["continuous_std"]["yield"])

    train_ds = YieldDataset(
        root_dir=args.data_root, field_list=train_fields,
        stats_path=str(stats_path), gpkg_path=args.gpkg_path,
        patch_size=patch_size, stride=stride, min_valid_ratio=min_valid_ratio,
        max_holes=max_holes, min_hole_size=min_hole_size, max_hole_size=max_hole_size,
        mode="train", seed=None,
    )
    val_ds = YieldDataset(
        root_dir=args.data_root, field_list=val_fields,
        stats_path=str(stats_path), gpkg_path=args.gpkg_path,
        patch_size=patch_size, stride=stride, min_valid_ratio=min_valid_ratio,
        max_holes=max_holes, min_hole_size=min_hole_size, max_hole_size=max_hole_size,
        mode="val", seed=seed,
    )

    train_loader = make_loader(train_ds, batch_size, True,  num_workers)
    val_loader   = make_loader(val_ds,   batch_size, False, num_workers)

    sample_x, _, _, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    model = build_model(model_name, in_channels=in_channels,
                        base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    writer = SummaryWriter(run_dir / "tensorboard")
    mlflow.set_experiment("yield_comparison")

    best_val = float("inf")

    print(f"\n=== FINAL TRAINING: {epochs} epochs | {model_name} | base_ch={base_channels} | patch={patch_size} ===")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(args.config)
        mlflow.log_artifact(str(run_dir / "optuna_best.json"))
        mlflow.log_params({
            "lr": lr, "batch_size": batch_size,
            "base_channels": base_channels, "patch_size": patch_size,
            "smooth_weight": smooth_w, "known_weight": known_w,
            "weight_decay": weight_decay, "epochs_final": epochs,
        })

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device, known_w, smooth_w)

            val_loss, val_mae, val_rmse, val_bias, val_mape, val_r2, val_errors = \
                validate_full(model, val_loader, device, known_w, smooth_w, y_mean, y_std)

            print(f"[{epoch}] train={train_loss:.4f} val={val_loss:.4f} "
                  f"mae={val_mae:.3f} r2={val_r2:.3f}")

            # TensorBoard scalars
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val",   val_loss,   epoch)
            writer.add_scalar("val/mae",    val_mae,    epoch)
            writer.add_scalar("val/rmse",   val_rmse,   epoch)
            writer.add_scalar("val/bias",   val_bias,   epoch)
            writer.add_scalar("val/mape",   val_mape,   epoch)
            writer.add_scalar("val/r2",     val_r2,     epoch)
            writer.add_histogram("val/error_distribution", val_errors, epoch)

            # MLflow metrics
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
                vis_x, vis_y, vis_hole, vis_field = val_ds[0]
                vis_pred = model(vis_x.unsqueeze(0).to(device))[0].cpu()

            hole_np   = vis_hole[0].numpy()
            field_np  = vis_field[0].numpy()
            gt_norm   = vis_y[0].numpy()
            pred_norm = vis_pred[0].numpy()

            gt_real   = gt_norm   * y_std + y_mean
            pred_real = pred_norm * y_std + y_mean

            gt_full     = np.where((field_np == 1) & np.isfinite(gt_real), gt_real, np.nan)
            gt_masked   = np.where((field_np == 1) & (hole_np == 1) & np.isfinite(gt_real), gt_real, np.nan)
            pred_masked = np.where((field_np == 1) & np.isfinite(pred_real), pred_real, np.nan)

            filled = gt_full.copy()
            filled[hole_np == 0] = pred_masked[hole_np == 0]

            valid_gt = gt_full[np.isfinite(gt_full)]
            if valid_gt.size > 0:
                vmin = float(np.percentile(valid_gt, 0.05))
                vmax = float(np.percentile(valid_gt, 99.95))
            else:
                vmin, vmax = 0.0, 1.0

            def to_rgb_gray_bg(arr, fmask, vmin, vmax):
                rgb = np.full((*arr.shape, 3), 0.35, dtype=np.float32)
                valid = (fmask == 1) & np.isfinite(arr)
                if valid.sum() == 0:
                    return torch.tensor(rgb).permute(2, 0, 1)
                t = np.zeros_like(arr, dtype=np.float32)
                t[valid] = np.clip((arr[valid] - vmin) / (vmax - vmin + 1e-6), 0, 1)
                rgb[..., 0] = np.where(valid, np.clip(1.5 - abs(t*4-1.0), 0, 1), rgb[..., 0])
                rgb[..., 1] = np.where(valid, np.clip(1.5 - abs(t*4-2.0), 0, 1), rgb[..., 1])
                rgb[..., 2] = np.where(valid, np.clip(1.5 - abs(t*4-3.0), 0, 1), rgb[..., 2])
                return torch.tensor(rgb).permute(2, 0, 1)

            # input_masked
            input_rgb = to_rgb_gray_bg(gt_masked, field_np, vmin, vmax).numpy()
            hole_inside = (hole_np == 0) & (field_np == 1)
            input_rgb[0, hole_inside] = 0.0
            input_rgb[1, hole_inside] = 0.0
            input_rgb[2, hole_inside] = 0.0

            # output_filled с контуром
            filled_rgb = to_rgb_gray_bg(filled, field_np, vmin, vmax).numpy()
            hole_f  = hole_np.astype(np.float32)
            hole_if = ((hole_np == 0) & (field_np == 1)).astype(np.float32)
            border  = np.zeros_like(hole_f, dtype=bool)
            border[1:-1, 1:-1] = (
                (hole_f[1:-1,1:-1] == 0) & (field_np[1:-1,1:-1] == 1) & (
                    (hole_if[:-2,1:-1] == 1) | (hole_if[2:,1:-1] == 1) |
                    (hole_if[1:-1,:-2] == 1) | (hole_if[1:-1,2:] == 1)
                )
            )
            filled_rgb[0, border] = 0.0
            filled_rgb[1, border] = 0.0
            filled_rgb[2, border] = 0.0

            writer.add_image("reconstruction/1_input_masked",
                torch.tensor(input_rgb), epoch)
            writer.add_image("reconstruction/2_output_filled",
                torch.tensor(filled_rgb), epoch)
            writer.add_image("reconstruction/3_output",
                to_rgb_gray_bg(filled, field_np, vmin, vmax), epoch)
            writer.add_image("reconstruction/4_ground_truth",
                to_rgb_gray_bg(gt_full, field_np, vmin, vmax), epoch)

            # masks
            field_rgb = np.full((*field_np.shape, 3), 0.35, dtype=np.float32)
            field_rgb[field_np == 1] = 1.0
            writer.add_image("masks/field_mask",
                torch.tensor(field_rgb).permute(2, 0, 1), epoch)

            hole_rgb = np.full((*field_np.shape, 3), 0.35, dtype=np.float32)
            hole_rgb[field_np == 1] = 1.0
            hole_rgb[(field_np == 1) & (hole_np == 0)] = 0.0
            writer.add_image("masks/hole_mask",
                torch.tensor(hole_rgb).permute(2, 0, 1), epoch)

            # residuals
            err_rgb = np.full((*gt_real.shape, 3), 0.35, dtype=np.float32)
            err_rgb[field_np == 1] = 1.0
            hole_pixels = (hole_np == 0) & (field_np == 1) & \
                          np.isfinite(gt_real) & np.isfinite(pred_real)
            if hole_pixels.any():
                errors_h = pred_real[hole_pixels] - gt_real[hole_pixels]
                max_abs  = float(np.abs(errors_h).max())
                if max_abs > 1e-6:
                    t_err = np.clip(errors_h / max_abs, -1, 1) * 0.5 + 0.5
                    err_rgb[hole_pixels, 0] = np.clip(2*t_err - 1, 0, 1)
                    err_rgb[hole_pixels, 1] = 1 - 2*np.abs(t_err - 0.5)
                    err_rgb[hole_pixels, 2] = np.clip(1 - 2*t_err, 0, 1)
            writer.add_image("masks/residuals",
                torch.tensor(err_rgb).permute(2, 0, 1), epoch)

            # PNG error map every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
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

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        mlflow.log_metric("best_val_loss", best_val)
        mlflow.log_artifact(str(run_dir / "best_model.pt"))

    writer.close()

    print(f"\nFinal model saved → {run_dir / 'best_model.pt'}")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
