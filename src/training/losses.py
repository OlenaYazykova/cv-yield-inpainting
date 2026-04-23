import torch
import torch.nn.functional as F


# =========================
# utils
# =========================

def _match_shapes(pred, target, hole_mask, field_mask):
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if hole_mask.ndim == 3:
        hole_mask = hole_mask.unsqueeze(1)
    if field_mask.ndim == 3:
        field_mask = field_mask.unsqueeze(1)

    _, _, hp, wp = pred.shape
    _, _, ht, wt = target.shape

    h = min(hp, ht)
    w = min(wp, wt)

    pred = pred[:, :, :h, :w]
    target = target[:, :, :h, :w]
    hole_mask = hole_mask[:, :, :h, :w]
    field_mask = field_mask[:, :, :h, :w]

    return pred, target, hole_mask, field_mask


def _safe_mean(x, mask):
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


# =========================
# main loss
# =========================

def masked_loss(
    pred,
    target,
    hole_mask,
    field_mask,
    known_weight=0.1,
    smooth_weight=0.02,
):
    pred, target, hole_mask, field_mask = _match_shapes(
        pred, target, hole_mask, field_mask
    )

    # -------- masks --------
    missing_mask = (1.0 - hole_mask) * field_mask
    known_mask = hole_mask * field_mask

    # -------- base --------
    diff = pred - target
    mse = diff ** 2
    mae = diff.abs()

    # -------- missing (главное) --------
    missing_mse = _safe_mean(mse, missing_mask)
    missing_mae = _safe_mean(mae, missing_mask)
    missing_loss = 0.5 * missing_mse + 0.5 * missing_mae

    # -------- known --------
    known_loss = _safe_mean(mae, known_mask)

    # -------- smooth --------
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    mask_x = missing_mask[:, :, :, 1:] * missing_mask[:, :, :, :-1]
    mask_y = missing_mask[:, :, 1:, :] * missing_mask[:, :, :-1, :]

    smooth_x = _safe_mean(dx.abs(), mask_x)
    smooth_y = _safe_mean(dy.abs(), mask_y)

    smooth_loss = smooth_x + smooth_y

    # -------- total --------
    total_loss = (
        missing_loss
        + known_weight * known_loss
        + smooth_weight * smooth_loss
    )

    return total_loss


# =========================
# metrics
# =========================

def masked_mae(pred, target, hole_mask, field_mask):
    pred, target, hole_mask, field_mask = _match_shapes(
        pred, target, hole_mask, field_mask
    )

    missing_mask = (1.0 - hole_mask) * field_mask
    diff = (pred - target).abs()

    return _safe_mean(diff, missing_mask)


def masked_rmse(pred, target, hole_mask, field_mask):
    pred, target, hole_mask, field_mask = _match_shapes(
        pred, target, hole_mask, field_mask
    )

    missing_mask = (1.0 - hole_mask) * field_mask
    diff2 = (pred - target) ** 2

    return torch.sqrt(_safe_mean(diff2, missing_mask))


def masked_bias(pred, target, hole_mask, field_mask):
    pred, target, hole_mask, field_mask = _match_shapes(
        pred, target, hole_mask, field_mask
    )

    missing_mask = (1.0 - hole_mask) * field_mask
    diff = pred - target

    return _safe_mean(diff, missing_mask)
