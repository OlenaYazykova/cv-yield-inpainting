#!/usr/bin/env python3
"""
Pre-computes combine harvester stripe angles for each field
and saves results to stripe_angles.json.

Usage:
    python scripts/compute_stripe_angles.py \
        --data_root /srv/datasaver/raw/cv_project/data \
        --gpkg_path /srv/datasaver/raw/cv_project/data/fields.gpkg \
        --output /srv/datasaver/raw/cv_project/data/stripe_angles.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import read_raster
from src.evaluation.test import rasterize_field, resolve_gpkg


def detect_stripe_angle(yield_r: np.ndarray, field_mask: np.ndarray) -> float | None:
    """
    Detects the dominant combine harvester stripe angle using:
    1. FFT high-frequency residual (removes terrain trends and large-scale patterns)
    2. Canny edge detection on the residual
    3. Gradient direction voting across all edge pixels

    Returns angle in degrees [0, 180), or None if angle cannot be determined.
    """
    # erode field mask to avoid boundary effects
    field_inner = binary_erosion(field_mask == 1, iterations=10)
    if field_inner.sum() < 200:
        field_inner = binary_erosion(field_mask == 1, iterations=5)
    if field_inner.sum() < 200:
        field_inner = (field_mask == 1)

    # normalize yield values inside the field
    arr = yield_r.copy()
    valid = np.isfinite(arr) & field_inner
    if valid.sum() < 100:
        return None

    arr_norm = np.zeros_like(arr)
    arr_norm[valid] = (arr[valid] - arr[valid].mean()) / (arr[valid].std() + 1e-6)
    arr_norm[~field_inner] = 0

    # compute FFT high-frequency residual:
    # subtract gaussian low-frequency trend to isolate periodic stripe pattern
    low_freq = gaussian_filter(arr_norm, sigma=15)
    high_freq = arr_norm - low_freq
    high_freq[~field_inner] = 0

    # normalize residual to uint8 for OpenCV
    hf_min, hf_max = high_freq.min(), high_freq.max()
    if hf_max - hf_min < 1e-6:
        return None
    hf_uint8 = ((high_freq - hf_min) / (hf_max - hf_min) * 255).astype(np.uint8)

    # detect edges using Canny on the high-frequency residual
    edges = cv2.Canny(hf_uint8, threshold1=30, threshold2=80)
    # mask edges outside the eroded field boundary
    edges[~field_inner] = 0

    # compute Sobel gradients for angle voting
    sobelx = cv2.Sobel(hf_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(hf_uint8, cv2.CV_64F, 0, 1, ksize=3)

    # select only edge pixels for voting
    edge_pixels = edges > 0
    if edge_pixels.sum() < 10:
        return None

    gx = sobelx[edge_pixels]
    gy = sobely[edge_pixels]

    # gradient direction is perpendicular to the edge
    # stripe angle = gradient angle + 90 degrees
    edge_angles = (np.rad2deg(np.arctan2(gy, gx)) + 90) % 180

    # build histogram and smooth to find dominant angle
    hist, bins = np.histogram(edge_angles, bins=180, range=(0, 180))
    hist_smooth = gaussian_filter(hist.astype(float), sigma=2)
    best_angle = float(bins[np.argmax(hist_smooth)])

    return best_angle


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute combine harvester stripe angles for all fields"
    )
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing field subdirectories")
    parser.add_argument("--gpkg_path", default=None,
                        help="Path to global fields.gpkg (used if field.gpkg not found per field)")
    parser.add_argument("--output", required=True,
                        help="Output path for stripe_angles.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    fields = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    print(f"Found {len(fields)} fields in {data_root}")
    print()
    print(f"{'Field':<15} {'Angle (°)':>10} {'Status':>12}")
    print("-" * 42)

    results = {}

    for field in fields:
        field_dir = data_root / field

        try:
            gpkg_path  = resolve_gpkg(field_dir, args.gpkg_path)
            yield_r    = read_raster(field_dir / "yield.tif")
            field_mask = rasterize_field(gpkg_path, field_dir / "yield.tif")

            angle = detect_stripe_angle(yield_r, field_mask)

            if angle is None:
                print(f"{field:<15} {'—':>10} {'not detected':>12}")
                results[field] = None
            else:
                print(f"{field:<15} {angle:>10.1f} {'OK':>12}")
                results[field] = angle

        except Exception as e:
            print(f"{field:<15} {'—':>10} {str(e)[:20]:>12}")
            results[field] = None

    # save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok  = sum(1 for v in results.values() if v is not None)
    n_fail = len(fields) - n_ok
    print()
    print(f"Done: {n_ok}/{len(fields)} fields successfully processed")
    if n_fail > 0:
        print(f"Warning: {n_fail} fields could not be processed — angle set to None")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
