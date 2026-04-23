#!/usr/bin/env python3
"""
Full standalone script for downloading and creating the SOIL layer.

Full pipeline:
  1. Connect to Sentinel Hub (Sentinel-2 L2A)
  2. Fetch NDVI + cloud cover statistics for the entire available period
  3. Filter: only cloud-free dates in March–April (bare soil)
  4. Download 12-band TIFF rasters for these dates
  5. Compute soil index: mean(B02, B03, B04) / max for each date
  6. Aggregate across dates → average → save SOIL_{field_name}.tiff

"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import geopandas as gpd
from pathlib import Path
from datetime import date

from sentinelhub import (
    SHConfig,
    Geometry,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubStatistical,
    SentinelHubRequest,
    bbox_to_dimensions,
    parse_time,
    BBox,
)

# ─── Credential Sentinel Hub ─────────────────────────────────────────────────
load_dotenv()

SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "")


# ─── Evalscripts ─────────────────────────────────────────────────────────────

# NDVI + cloud cover statistics (for identifying cloud-free dates)
EVALSCRIPT_STATS = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "data", sampleType: "FLOAT32", bands: 2 },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(sample) {
  let ndvi = index(sample.B08, sample.B04);
  let cloud = 0;
  let scl = sample.SCL;
  if (scl == 3 || scl == 7 || scl == 8 || scl == 9 || scl == 10) {
    cloud = 1;
  }
  return { data: [ndvi, cloud], dataMask: [sample.dataMask] };
}
"""

# 12-band Sentinel-2 L2A (for TIFF download)
EVALSCRIPT_ALL_BANDS = """
//VERSION=3
function evaluatePixel(samples) {
  return [
    samples.B01, samples.B02, samples.B03, samples.B04,
    samples.B05, samples.B06, samples.B07, samples.B08,
    samples.B8A, samples.B09, samples.B11, samples.B12
  ];
}
function setup() {
  return {
    input: [{
      bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
    }],
    output: { bands: 12, sampleType: SampleType.FLOAT32 }
  }
}
"""


def get_config() -> SHConfig:
    """Створює конфігурацію Sentinel Hub."""
    cfg = SHConfig()
    cfg.sh_client_id = SH_CLIENT_ID
    cfg.sh_client_secret = SH_CLIENT_SECRET
    cfg.download_timeout_seconds = 120
    cfg.max_download_attempts = 3
    return cfg


# ─── # Step 1: NDVI + cloud cover statistics ─────────────────────────────────────

def fetch_ndvi_stats(config: SHConfig, geometry_sh: Geometry,
                     start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily NDVI + cloud cover (%) statistics via the SentinelHubStatistical API.
    Returns a DataFrame with columns: interval_from, ndvi_mean, cloud_mean.
    """
    print("📡 Етап 1: Запит статистики NDVI + хмарності до Sentinel Hub...")

    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=EVALSCRIPT_STATS,
            time_interval=(start_date, end_date),
            aggregation_interval="P1D",
            resolution=(0.00009, 0.00009),  # ~10м
        ),
        input_data=[
            SentinelHubStatistical.input_data(DataCollection.SENTINEL2_L2A, maxcc=0.8),
        ],
        geometry=geometry_sh,
        config=config,
    )

    response = request.get_data()

    rows = []
    for single_data in response[0]["data"]:
        entry = {}
        is_valid = True
        entry["interval_from"] = parse_time(single_data["interval"]["from"]).date()

        for output_name, output_data in single_data["outputs"].items():
            for band_name, band_values in output_data["bands"].items():
                stats = band_values["stats"]
                if stats["sampleCount"] == stats["noDataCount"]:
                    is_valid = False
                    break
                for stat_name, value in stats.items():
                    if stat_name == "percentiles":
                        continue
                    entry[f"{output_name}_{band_name}_{stat_name}"] = value

        if is_valid:
            rows.append(entry)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["data_B0_mean"] = pd.to_numeric(df["data_B0_mean"], errors="coerce")
    df = df.loc[df["data_B0_mean"] <= 1]
    df["interval_from"] = pd.to_datetime(df["interval_from"])
    print(f"   ✅ Отримано {len(df)} дат зі статистикою")
    return df


# ─── # Step 2: Filtering cloud-free dates for March–April ───────────────────

def filter_cloud_free_spring_dates(df: pd.DataFrame) -> List[str]:
    """
    Filters only cloud-free dates (cloud == 0) for March and April.
    """
    print("Етап 2: Фільтрація безхмарних дат за березень-квітень...")

    # cloud == 0
    filtered = df[df["data_B1_mean"] == 0]

    dates = []
    for _, row in filtered.iterrows():
        date_str = row["interval_from"].strftime("%Y-%m-%d")
        month = date_str[5:7]
        if month in {"03", "04"}:
            dates.append(date_str)

    dates = sorted(set(dates))
    print(f"   ✅ Знайдено {len(dates)} безхмарних дат за березень-квітень")
    if dates:
        print(f"Приклади: {dates[:5]}{'...' if len(dates) > 5 else ''}")
    return dates


# ─── # Step 3: TIFF download ───────────────────────────────────────────────

def download_tiff(config: SHConfig, bbox: BBox, field_name: str,
                  date_str: str, tiff_dir: Path) -> Path:
    """
    Downloads a 12-band TIFF for a single date.
    """

    filename = f"{field_name}_{date_str}.tiff"
    file_path = tiff_dir / filename

    if file_path.exists():
        return file_path

    geom_size = bbox_to_dimensions(bbox, resolution=10)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_ALL_BANDS,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_str, date_str),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=geom_size,
        config=config,
        data_folder=str(tiff_dir),
    )
    request.download_list[0].filename = filename
    request.save_data()
    request.get_data(decode_data=False)

    return file_path


def download_all_tiffs(config: SHConfig, bbox: BBox, field_name: str,
                       dates: List[str], tiff_dir: Path) -> List[Path]:
    """
    Downloads TIFF for all dates.
    """

    print(f"Етап 3: Завантаження {len(dates)} TIFF растрів з Sentinel-2 L2A...")
    tiff_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for date_str in tqdm(dates, desc="TIFF Downloading"):
        path = download_tiff(config, bbox, field_name, date_str, tiff_dir)
        if path.exists():
            paths.append(path)
        else:
            print(f"Не вдалося завантажити {date_str}")

    print(f"✅ Завантажено {len(paths)} з {len(dates)} TIFF файлів")
    return paths


# ─── # Step 4: SOIL index computation ─────────────────────────────────────────

def compute_soil_from_tiff(tiff_path: Path) -> Optional[np.ma.MaskedArray]:
    """
    Computes the soil index from a single 12-band TIFF:
    soil = mean(B04, B03, B02) / max

    TIFF band order: B01(0), B02(1), B03(2), B04(3), B05(4), ...
    """
    with rasterio.open(tiff_path) as src:
        data = src.read(masked=True)

        if data.shape[0] < 4:
            return None

        # Mean of visible bands: B04(idx=3), B03(idx=2), B02(idx=1)
        soil = np.ma.mean(np.ma.stack((data[3], data[2], data[1])), axis=0)

        max_val = soil.max()
        if max_val and max_val > 0:
            soil = soil / max_val

        return soil


def compute_and_save_soil(tiff_paths: List[Path], field_name: str,
                          output_dir: Path) -> Path:
    """
    Computes SOIL for each TIFF, aggregates across dates, saves the result.
    """
    print(f"Етап 4: Обчислення soil index з {len(tiff_paths)} TIFF файлів...")
    output_dir.mkdir(parents=True, exist_ok=True)

    soil_layers: List[np.ma.MaskedArray] = []
    used_dates: List[str] = []
    profile = None

    for tiff_path in tqdm(tiff_paths, desc="Processing SOIL"):
        soil = compute_soil_from_tiff(tiff_path)
        if soil is None:
            print(f"Пропущено {tiff_path.name} (< 4 канали)")
            continue

        soil_layers.append(soil)

        if profile is None:
            with rasterio.open(tiff_path) as src:
                profile = src.profile
                profile.update(
                    count=1,
                    dtype="float32",
                    compress="lzw",
                    nodata=0.0,
                )

        stem = tiff_path.stem
        if "_" in stem:
            used_dates.append(stem.split("_")[-1])
        else:
            used_dates.append(stem)

    if not soil_layers or profile is None:
        print("   ❌ Не вдалося сформувати шар ґрунту: жоден TIFF не має потрібних каналів")
        sys.exit(1)

    print(f"📊 Етап 5: Агрегація {len(soil_layers)} шарів...")
    stacked = np.ma.stack(soil_layers, axis=0)
    soil_avg = np.ma.mean(stacked, axis=0)

    output_path = output_dir / f"SOIL_{field_name}.tiff"

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(soil_avg.filled(0).astype("float32"), 1)
        mask_array = np.ma.getmaskarray(soil_avg)
        mask = np.where(mask_array, 0, 255).astype("uint8")
        dst.write_mask(mask)

    print(f"Збережено: {output_path}")
    print(f"Використані дати ({len(used_dates)}): {used_dates}")
    return output_path


# ─── # Utility: GeoJSON loading ───────────────────────────────────────────

def load_geometry_from_geojson(path: str) -> dict:
    """Loads geometry from a GeoJSON file (Feature or FeatureCollection)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        return data["features"][0]["geometry"]
    elif data.get("type") == "Feature":
        return data["geometry"]
    elif data.get("type") in ("Polygon", "MultiPolygon"):
        return data
    else:
        raise ValueError(f"Невідомий формат GeoJSON: {data.get('type')}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Завантаження та формування шару SOIL з Sentinel-2 L2A"
    )
    parser.add_argument("--field_name", required=True, help="Назва поля (напр. P№1)")
    parser.add_argument("--geometry", required=True, help="Шлях до GeoJSON файлу з контуром поля")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="Конкретні дати (YYYY-MM-DD). Якщо не вказано — автопошук безхмарних дат за березень-квітень")
    parser.add_argument("--start_date", default="2017-01-01T00:00:00Z",
                        help="Початок періоду для статистики (за замовчуванням 2017-01-01)")
    parser.add_argument("--output_dir", default="./SOIL_output",
                        help="Папка для збереження результату (за замовчуванням ./SOIL_output)")
    parser.add_argument("--tiff_dir", default=None,
                        help="Папка для TIFF файлів (за замовчуванням {output_dir}/TIFF)")
    args = parser.parse_args()

    field_name = args.field_name
    output_dir = Path(args.output_dir)
    tiff_dir = Path(args.tiff_dir) if args.tiff_dir else output_dir / "TIFF"

    print("=" * 70)
    print(f"SOIL Pipeline для поля: {field_name}")
    print("=" * 70)

    geom = load_geometry_from_geojson(args.geometry)
    print(f"Геометрія завантажена з {args.geometry}")

    config = get_config()
    geometry_sh = Geometry(geometry=geom, crs=CRS.WGS84)
    bbox = BBox(geometry_sh.bbox, CRS.WGS84)

    if args.dates:
        target_dates = sorted(set(args.dates))
        print(f" Вказані дати вручну: {target_dates}")
    else:
        end_date = f"{date.today().isoformat()}T23:59:59Z"
        df = fetch_ndvi_stats(config, geometry_sh, args.start_date, end_date)
        if df.empty:
            print("❌ Не отримано жодної дати зі статистикою. Перевірте геометрію та креденшали.")
            sys.exit(1)
        target_dates = filter_cloud_free_spring_dates(df)
        if not target_dates:
            print("❌ Не знайдено безхмарних дат за березень-квітень.")
            sys.exit(1)

    tiff_paths = download_all_tiffs(config, bbox, field_name, target_dates, tiff_dir)
    if not tiff_paths:
        print("❌ Не вдалося завантажити жодного TIFF.")
        sys.exit(1)

    # SOIL layer generation
    soil_path = compute_and_save_soil(tiff_paths, field_name, output_dir)

    print()
    print("=" * 70)
    print(f"✅ Готово! SOIL шар збережено: {soil_path}")
    print("=" * 70)

def load_geometry_from_gpkg(path: str):
    gdf = gpd.read_file(path)

    geom = gdf.geometry.iloc[0]

    return geom.__geo_interface__  
    
if __name__ == "__main__":
    data_dir = Path("/srv/datasaver/raw/cv_project/vector")
    base_output = Path("/srv/datasaver/raw/cv_project/raw_data_inference/soil")
    

    config = get_config()

    for file in data_dir.glob("*"):
        if file.suffix not in [".geojson", ".gpkg"]:
            continue

        field_name = file.stem
        print(f"\n Обробка поля: {field_name}")

        if file.suffix == ".geojson":
            geom = load_geometry_from_geojson(str(file))
        else:
            geom = load_geometry_from_gpkg(str(file))

        geometry_sh = Geometry(geometry=geom, crs=CRS.WGS84)
        bbox = BBox(geometry_sh.bbox, CRS.WGS84)

        end_date = f"{date.today().isoformat()}T23:59:59Z"
        df = fetch_ndvi_stats(config, geometry_sh, "2017-01-01T00:00:00Z", end_date)
        dates = filter_cloud_free_spring_dates(df)

        output_dir = base_output / field_name
        tiff_dir = output_dir / "TIFF"

        tiffs = download_all_tiffs(config, bbox, field_name, dates, tiff_dir)

        compute_and_save_soil(tiffs, field_name, output_dir)
