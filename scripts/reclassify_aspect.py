#!/usr/bin/env python3
"""
Reclassify aspect raster (degrees 0–360) to 8 cardinal direction categories.

Categories:
  0 = N, 1 = NE, 2 = E, 3 = SE, 4 = S, 5 = SW, 6 = W, 7 = NW
  -9999 = nodata

Usage:
  python reclassify_aspect.py \
    --input dem/aspect.tif \
    --output dem/aspect_8dir.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio


LABELS = {
    0: "N",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SW",
    6: "W",
    7: "NW",
}


def aspect_to_8dir(aspect: np.ndarray, nodata_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Convert aspect in degrees [0, 360] to 8 cardinal direction categories.
    Pixels outside valid area are set to -9999 (nodata).
    """
    out = np.full(aspect.shape, -9999, dtype=np.int16)

    # Normalize to [0, 360)
    a = np.mod(aspect, 360.0)

    # Shift by 22.5° so intervals are centered on cardinal directions, then bin by 45°
    valid = np.isfinite(a)
    if nodata_mask is not None:
        valid &= ~nodata_mask

    out[valid] = (
        np.floor((a[valid] + 22.5) / 45.0) % 8
    ).astype(np.int16)

    return out


def reclassify_aspect_raster(input_path: str, output_path: str) -> None:
    """Read aspect raster, reclassify to 8 directions, save result."""

    with rasterio.open(input_path) as src:
        aspect = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    # Build nodata mask and replace nodata with NaN
    nodata_mask = None
    if nodata is not None:
        nodata_mask = aspect == nodata
        aspect = aspect.copy()
        aspect[nodata_mask] = np.nan

    out = aspect_to_8dir(aspect, nodata_mask=nodata_mask)

    profile.update(
        dtype="int16",
        count=1,
        nodata=-9999,
        compress="lzw"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reclassify aspect raster to 8 cardinal directions"
    )
    parser.add_argument("--input",  required=True, help="Path to input aspect raster (degrees 0–360)")
    parser.add_argument("--output", required=True, help="Path to output classified raster")
    args = parser.parse_args()

    reclassify_aspect_raster(
        input_path=args.input,
        output_path=args.output,
    )

    print(f"✅ Saved: {args.output}")
    print(f"   Categories: { {v: k for k, v in LABELS.items()} }")
