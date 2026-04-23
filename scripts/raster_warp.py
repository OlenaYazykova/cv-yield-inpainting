#!/usr/bin/env python3
"""
Raster preprocessing pipeline.

Goal: reproject and align any set of rasters to the reference DEM grid
(5 m/pixel, metric CRS).

Usage:
  # Single raster
  python preprocess_rasters.py \
    --dem path/to/dem.tif \
    --gpkg path/to/field.gpkg \
    --rasters path/to/ndvi.tiff \
    --output_dir ./preprocessed

  # Multiple rasters at once
  python preprocess_rasters.py \
    --dem path/to/dem.tif \
    --gpkg path/to/field.gpkg \
    --rasters path/to/ndvi.tiff path/to/soil.tiff path/to/yield.tif \
    --output_dir ./preprocessed
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ─── I/O ─────────────────────────────────────────────────────────────────────

def read_raster_with_meta(path: str):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        shape = data.shape
        nodata = src.nodata
        profile = src.profile.copy()
    return data, transform, crs, shape, nodata, profile


def save_like_reference(path: str, array: np.ndarray, profile: dict, nodata_value: float = -9999.0):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata_value, compress="lzw")

    out = array.copy()
    out[np.isnan(out)] = nodata_value

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out.astype(np.float32), 1)


# ─── Geometry ────────────────────────────────────────────────────────────────

def load_field_geometry_in_crs(gpkg_path: str, target_crs):
    gdf = gpd.read_file(gpkg_path)

    if gdf.empty:
        raise ValueError(f"Empty gpkg: {gpkg_path}")
    if gdf.crs is None:
        raise ValueError("gpkg has no CRS defined")

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    return [geom.__geo_interface__]


# ─── Raster ops ──────────────────────────────────────────────────────────────

def replace_nodata_with_nan(arr: np.ndarray, nodata):
    out = arr.copy().astype(np.float32)
    if nodata is not None:
        out[out == nodata] = np.nan
    return out


def mask_by_geometry(array: np.ndarray, transform, geom_list):
    inside = geometry_mask(
        geom_list,
        transform=transform,
        invert=True,
        out_shape=array.shape
    )
    out = array.copy()
    out[~inside] = np.nan
    return out, inside.astype(np.float32)


def fill_nan_inside_field(array: np.ndarray, field_mask: np.ndarray):
    out = array.copy()

    missing = np.isnan(out) & (field_mask == 1)
    known = (~np.isnan(out)) & (field_mask == 1)

    if not np.any(missing):
        return out
    if known.sum() < 4:
        out[missing] = 0.0
        return out

    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w]

    points = np.column_stack((xx[known], yy[known]))
    values = out[known]
    query = np.column_stack((xx[missing], yy[missing]))

    filled = griddata(points, values, query, method="linear")

    bad = np.isnan(filled)
    if np.any(bad):
        filled[bad] = griddata(points, values, query[bad], method="nearest")

    out[missing] = filled
    return out


def reproject_to_dem_grid(
    src_array, src_transform, src_crs,
    dst_shape, dst_transform, dst_crs,
    resampling=Resampling.bilinear
):
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def smooth_inside_mask(array: np.ndarray, field_mask: np.ndarray, sigma: float = 1.0):
    arr = array.copy()
    arr[np.isnan(arr)] = 0.0

    num = gaussian_filter(arr * field_mask, sigma=sigma)
    den = gaussian_filter(field_mask.astype(np.float32), sigma=sigma)

    out = num / (den + 1e-8)
    out[field_mask == 0] = np.nan
    return out.astype(np.float32)


# ─── Main pipeline ───────────────────────────────────────────────────────────

def prepare_raster(
    src_path: str,
    dem_path: str,
    gpkg_path: str,
    smooth_sigma: float = 1.0,
    resampling=Resampling.bilinear,
    fill_nan: bool = True, 
):
    # 1. Read source raster
    src, src_transform, src_crs, _, src_nodata, _ = read_raster_with_meta(src_path)
    src = replace_nodata_with_nan(src, src_nodata)

    # 2. Clip to field boundary in source CRS
    field_geom_src = load_field_geometry_in_crs(gpkg_path, src_crs)
    clipped_src, field_mask_src = mask_by_geometry(src, src_transform, field_geom_src)

    # 3. Fill NaN inside field (optional)
    filled_src = fill_nan_inside_field(clipped_src, field_mask_src) if fill_nan else clipped_src

    # 4. Read DEM as reference grid (5 m/pixel, metric CRS)
    _, dem_transform, dem_crs, dem_shape, _, dem_profile = read_raster_with_meta(dem_path)

    # 5. Reproject to DEM grid
    reproj = reproject_to_dem_grid(
        src_array=filled_src,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_shape=dem_shape,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=resampling,
    )

    # 6. Clip to field boundary in DEM CRS
    field_geom_dem = load_field_geometry_in_crs(gpkg_path, dem_crs)
    clipped_dem, field_mask_dem = mask_by_geometry(reproj, dem_transform, field_geom_dem)

    # 7. Smooth inside field mask (only if fill was applied)
    smoothed = smooth_inside_mask(clipped_dem, field_mask_dem, sigma=smooth_sigma) if fill_nan else clipped_dem

    return smoothed, field_mask_dem, dem_profile


def process_rasters(
    dem_path: str,
    gpkg_path: str,
    raster_paths: list,
    output_dir: str,
    smooth_sigma: float = 1.0,
    fill_nan: bool = True, 
):
    output_dir = Path(output_dir)

    for src_path in raster_paths:
        src_path = Path(src_path)
        print(f"\n🔄 Processing: {src_path.name}")

        raster, _, dem_profile = prepare_raster(
            src_path=str(src_path),
            dem_path=dem_path,
            gpkg_path=gpkg_path,
            smooth_sigma=smooth_sigma,
            fill_nan=fill_nan, 
        )

        dst_path = output_dir / src_path.name
        save_like_reference(str(dst_path), raster, dem_profile)
        print(f"   ✅ Saved: {dst_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproject and align rasters to DEM reference grid (5 m/pixel, metric CRS)"
    )
    parser.add_argument("--dem",        required=True, help="Path to reference DEM raster")
    parser.add_argument("--gpkg",       required=True, help="Path to field boundary (.gpkg)")
    parser.add_argument("--rasters",    required=True, nargs="+",
                                        help="One or more rasters to preprocess (NDVI, SOIL, Yield, ...)")
    parser.add_argument("--output_dir", required=True, help="Output directory for preprocessed rasters")
    parser.add_argument("--sigma",      type=float, default=1.0, help="Gaussian smoothing sigma")
    parser.add_argument("--no-fill", action="store_true",
                        help="Skip NaN filling and smoothing — reproject only")
    args = parser.parse_args()

    process_rasters(
        dem_path=args.dem,
        gpkg_path=args.gpkg,
        raster_paths=args.rasters,
        output_dir=args.output_dir,
        smooth_sigma=args.sigma,
        fill_nan=not args.no_fill,
    )


if __name__ == "__main__":
    main()
