import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box 

def fill_anomalies_with_adaptive_radius_meters(
    dem_holes: np.ndarray,
    dem: np.ndarray,
    anomaly_mask: np.ndarray,
    donor_coords: np.ndarray,
    donor_vals: np.ndarray,
    pixel_size: float,
    base_radius_m: float,
    max_radius_factor: int,
    min_points: int = 3,
) -> np.ndarray:
    donor_tree = cKDTree(donor_coords)
    dem_filled = dem_holes.copy()
    ai, aj = np.where(anomaly_mask)

    base_radius_px = base_radius_m / pixel_size
    max_radius_px = base_radius_px * max_radius_factor

    for i, j in zip(ai, aj):
        used_idxs = None

        for r in np.arange(base_radius_px, max_radius_px + 0.1, step=0.5):
            idxs = donor_tree.query_ball_point([i, j], r)
            if len(idxs) >= min_points:
                used_idxs = idxs
                break

        if used_idxs is not None:
            vals = donor_vals[used_idxs]
            mean_val = np.nanmean(vals)
            if not np.isnan(mean_val):
                dem_filled[i, j] = mean_val
        else:
            dem_filled[i, j] = dem[i, j]

    return dem_filled

def correct_dem_around_fields(
    dem_path: str,
    forest_mask_path: str,
    fields_path: str,
    out_dir: str = "dem_folder_corrected",
    suspect_out_m: float = 80.0,
    suspect_in_m: float = 15.0,
    forest_buffer_m: float = 15.0,
    diff_threshold: float = 2.5,
    ref_base_radius_m: float = 60.0,
    ref_max_radius_factor: int = 3,
    rebuild_every: int = 1
):
    """
    v5: DEM correction around fields using dynamically expanding reference zone.

    Logic:
      1) Geometric suspect zone:
         - ring around fields (suspect_out_m outside + suspect_in_m inside),
         - intersected with expanded forest area (forest_buffer_m outward).
      2) Initial reference zone:
         - pixels outside forest,
         - outside suspect zone,
         - with valid elevation.
      3) For EACH pixel in suspect zone (ordered from closest to farthest from reference):
         - find nearest reference pixels using adaptive radius (from ref_base_radius_m
           to ref_max_radius_factor * ref_base_radius_m),
         - compute median of reference,
         - diff = DEM - median_ref:
             * if diff > diff_threshold and positive → mark as anomalous;
             * otherwise, treat pixel as valid and ADD it to reference (dynamic growing).
      4) Donors for filling:
         - pixels inside fields,
         - outside suspect zone,
         - not NaN.
      5) Extract only anomalous pixels and fill using mean of nearby donor values.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        dem_profile = src.profile
        dem_transform = src.transform
        dem_crs = src.crs
        nodata = dem_profile.get("nodata", -9999.0)

    dem[dem == nodata] = np.nan
    px = abs(dem_transform.a)  # pixel size in meters

    # Read and reproject forest mask to DEM
    with rasterio.open(forest_mask_path) as src:
        forest_raw = src.read(1)
        forest_tr = src.transform
        forest_crs = src.crs

    forest_bin_10m = (forest_raw > 0).astype("float32")
    forest = np.zeros_like(dem, dtype="float32")

    reproject(
        source=forest_bin_10m,
        destination=forest,
        src_transform=forest_tr,
        src_crs=forest_crs,
        dst_transform=dem_profile["transform"],
        dst_crs=dem_crs,
        resampling = Resampling.average
    )

    forest_bin = (forest > 0).astype(bool)

    # Expand forest outward
    forest_buf_px = max(1, int(round(forest_buffer_m / px)))
    struct = np.ones((forest_buf_px * 2 + 1, forest_buf_px * 2 + 1), dtype=bool)
    forest_expanded = binary_dilation(forest_bin, structure=struct)

    all_fields_gdf = gpd.read_file(fields_path)

    if all_fields_gdf.crs != dem_crs:
        all_fields_gdf = all_fields_gdf.to_crs(dem_crs)

    dem_bounds = rasterio.open(dem_path).bounds
    dem_bbox_geom = gpd.GeoSeries([box(*dem_bounds)], crs=dem_crs)
    fields_gdf = all_fields_gdf[all_fields_gdf.geometry.intersects(dem_bbox_geom.iloc[0])]

    fields_geom = unary_union(fields_gdf.geometry)

    field_mask = rasterize(
        [(fields_geom, 1)],
        out_shape=dem.shape,
        transform=dem_transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    # Build suspect zone
    outer = fields_geom.buffer(suspect_out_m)
    inner = fields_geom.buffer(-suspect_in_m) if suspect_in_m > 0 else fields_geom
    ring_geom = outer.difference(inner)

    suspect_ring = rasterize(
        [(ring_geom, 1)],
        out_shape=dem.shape,
        transform=dem_transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    # Suspect zone: ring ∩ expanded forest
    suspect_mask_geom = suspect_ring & forest_expanded
    forest_and_field_mask = forest_bin & field_mask
    suspect_mask_geom |= forest_and_field_mask

    # Save suspect mask
    suspect_profile = dem_profile.copy()
    suspect_profile.update(dtype="uint8", nodata=0)

    # Initial reference zone: outside forest and suspect, not NaN
    reference_mask = (~forest_bin) & (~suspect_mask_geom) & (~np.isnan(dem))
    ref_coords = np.column_stack(np.where(reference_mask))
    ref_vals = dem[reference_mask]

    if len(ref_coords) == 0:
        return dem_path

    ref_coords_list = [tuple(c) for c in ref_coords]
    ref_vals_list = list(ref_vals)

    ref_coords_arr = np.array(ref_coords_list)
    ref_vals_arr = np.array(ref_vals_list)
    ref_tree = cKDTree(ref_coords_arr)

    # Donors: field pixels outside suspect, not NaN
    donors_mask = field_mask & (~suspect_mask_geom) & (~np.isnan(dem))
    donor_coords = np.column_stack(np.where(donors_mask))
    donor_vals = dem[donors_mask]

    if len(donor_coords) == 0:
        return dem_path

    donor_tree = cKDTree(donor_coords)

    # Order of suspect pixels: closest to reference first
    dist_to_ref = distance_transform_edt(~reference_mask) * px
    si, sj = np.where(suspect_mask_geom)
    if len(si) == 0:
        return dem_path

    dists = dist_to_ref[si, sj]
    order = np.argsort(dists)

    base_radius_px = max(1, int(round(ref_base_radius_m / px)))
    max_radius_px = base_radius_px * ref_max_radius_factor

    anomaly_mask = np.zeros_like(suspect_mask_geom, dtype=bool)
    added_since_rebuild = 0

    print(f"[INFO] Checking {len(si)} suspect pixels (dynamic growing)...")

    # Main loop
    for idx in order:
        i, j = si[idx], sj[idx]

        median_ref = None
        r = base_radius_px

        ref_coords_arr = np.array(ref_coords_list)
        ref_vals_arr = np.array(ref_vals_list)

        while r <= max_radius_px:
            ref_idxs = ref_tree.query_ball_point([i, j], r)
            if len(ref_idxs) >= 3:
                local_vals = ref_vals_arr[ref_idxs]
                m = np.nanmedian(local_vals)
                if not np.isnan(m):
                    median_ref = m
                    break
            r += base_radius_px

        if median_ref is None:
            continue

        diff = dem[i, j] - median_ref

        if diff > diff_threshold:
            anomaly_mask[i, j] = True
        else:
            if not reference_mask[i, j] and not np.isnan(dem[i, j]):
                reference_mask[i, j] = True
                ref_coords_list.append((i, j))
                ref_vals_list.append(dem[i, j])

                added_since_rebuild += 1

                if added_since_rebuild >= rebuild_every:
                    ref_coords_arr = np.array(ref_coords_list)
                    ref_vals_arr = np.array(ref_vals_list)
                    ref_tree = cKDTree(ref_coords_arr)
                    added_since_rebuild = 0

    ref_coords_arr = np.array(ref_coords_list)
    ref_vals_arr = np.array(ref_vals_list)
    ref_tree = cKDTree(ref_coords_arr)

    donor_mask = reference_mask & (~np.isnan(dem))
    donor_coords = np.column_stack(np.where(donor_mask))
    donor_vals = dem[donor_mask]

    # Create DEM with holes (NaNs only where anomalies)
    dem_holes = dem.copy()
    dem_holes[anomaly_mask] = np.nan

    dem_filled = fill_anomalies_with_adaptive_radius_meters(
        dem_holes,
        dem,
        anomaly_mask,
        donor_coords,
        donor_vals,
        pixel_size=px,
        base_radius_m=ref_base_radius_m,
        max_radius_factor=ref_max_radius_factor,
        min_points=3
    )


    dem_basename = os.path.basename(dem_path)
    group_name = dem_basename.replace("DEM_", "").replace(".tiff", "").replace(".tif", "")
    filled_path = os.path.join(out_dir, f"dem_corrected_{group_name}.tif")

    with rasterio.open(filled_path, "w", **dem_profile) as dst:
        dst.write(np.where(np.isnan(dem_filled), nodata, dem_filled), 1)

    print(f"[INFO] Corrected DEM saved: {filled_path}")

    with rasterio.open(dem_path, "w", **dem_profile) as dst:
        dst.write(np.where(np.isnan(dem_filled), nodata, dem_filled), 1)

    return dem_path
