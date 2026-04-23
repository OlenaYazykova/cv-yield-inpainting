from pathlib import Path
import json
import numpy as np

from src.data.dataset import read_raster


def compute_stats(root_dir: str, field_list: list[str], save_path: str):
    root = Path(root_dir)

    continuous_features = [
        "dem",
        "hand",
        "ndvi",
        "rtp_local",
        "rtp_regional",
        "slope",
        "soil",
        "twi",
        "yield",
    ]

    categorical_features = [
        "geomorphons",
        "relief_class",
        "aspect_categ",
    ]

    continuous_values = {k: [] for k in continuous_features}

    print("=== COMPUTE STATS ===")

    # =========================
    # LOOP on fields
    # =========================
    for field_name in field_list:
        print(f"Processing field: {field_name}")
        field_dir = root / field_name

        # ---------- continuous ----------
        for feat in continuous_features:
            path = field_dir / f"{feat}.tif"
            if not path.exists():
                print(f"WARNING: missing {path}")
                continue
            arr = read_raster(path)
            vals = arr[np.isfinite(arr)]
            if vals.size > 0:
                continuous_values[feat].append(vals)

        # ---------- categorical  ----------
        for feat in categorical_features:
            path = field_dir / f"{feat}.tif"
            if not path.exists():
                print(f"WARNING: missing {path}")

    # =========================
    # BUILD STATS
    # =========================

    stats = {
        "continuous_mean": {},
        "continuous_std":  {},
        # категориальные классы фиксированы — не зависят от данных
        "categorical_classes": {
            "geomorphons":  list(range(1, 11)),  # 1-10
            "relief_class": list(range(1, 9)),   # 1-8
            "aspect_categ": list(range(0, 8)),   # 0-7
        },
    }

    # ---------- continuous ----------
    for feat, chunks in continuous_values.items():
        if len(chunks) == 0:
            raise ValueError(f"No valid data for feature: {feat}")

        vals = np.concatenate(chunks, axis=0)
        mean = float(np.mean(vals))
        std  = float(np.std(vals))

        if std < 1e-6:
            print(f"WARNING: std≈0 for {feat}, forcing std=1")
            std = 1.0

        stats["continuous_mean"][feat] = mean
        stats["continuous_std"][feat]  = std

    # =========================
    # SAVE
    # =========================

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Stats saved to: {save_path}")
