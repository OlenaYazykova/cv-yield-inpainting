from pathlib import Path
import json
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import rasterio
from rasterio.features import rasterize
import geopandas as gpd

# =========================
# IO
# =========================

def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

    arr[arr == -9999] = np.nan
    return arr

# =========================
# geo
# =========================

def rasterize_field(gpkg_path: Path, reference_raster: Path) -> np.ndarray:
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


# =========================
# features
# =========================

def normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    arr = (arr - float(mean)) / std
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def aspect_to_sin_cos(aspect: np.ndarray):
    rad = np.deg2rad(aspect)
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def one_hot_encode(arr: np.ndarray, classes: list[int]) -> np.ndarray:
    classes_arr = np.asarray(classes, dtype=np.int32)
    out = (arr[None, :, :] == classes_arr[:, None, None]).astype(np.float32)
    return out


# =========================
# holes
# =========================

def generate_holes(
    valid_mask: np.ndarray,
    rng,
    max_holes: int = 3,
    min_size: int = 5,
    max_size: int = 20,
    min_valid_fraction: float = 0.3,
    max_attempts_per_hole: int = 50,
    stripe_angle: float = None,
) -> np.ndarray:
    from scipy.ndimage import rotate as ndimage_rotate

    h, w = valid_mask.shape
    hole_mask = valid_mask.copy().astype(np.float32)

    if h == 0 or w == 0:
        return hole_mask

    n_holes = rng.randint(3, max_holes)

    for _ in range(n_holes):
        short = rng.randint(min_size, min_size + 10)
        long  = rng.randint(max_size, int(max_size * 2.5))
        size_h, size_w = short, long

        if h - size_h <= 0 or w - size_w <= 0:
            continue

        placed = False

        for _attempt in range(max_attempts_per_hole):
            y = rng.randint(0, max(1, h))
            x = rng.randint(0, max(1, w))

            patch = valid_mask[y:y + size_h, x:x + size_w]
            if patch.size == 0:
                continue
            if patch.mean() < min_valid_fraction:
                continue

            if stripe_angle is not None:
                # create small rectangle and rotate it
                rect = np.ones((size_h, size_w), dtype=np.float32)
                rotated = ndimage_rotate(rect, -stripe_angle, reshape=True, order=1)
                rotated = (rotated > 0.5).astype(np.float32)
                rh, rw = rotated.shape

                # center of rectangle in valid_mask coordinates
                cy = y + size_h // 2
                cx = x + size_w // 2

                # top-left corner of rotated rectangle in valid_mask
                y0 = cy - rh // 2
                x0 = cx - rw // 2
                y1 = y0 + rh
                x1 = x0 + rw

                # clip to valid_mask bounds
                ry0 = max(0, -y0)
                rx0 = max(0, -x0)
                ry1 = rh - max(0, y1 - h)
                rx1 = rw - max(0, x1 - w)

                vy0 = max(0, y0)
                vx0 = max(0, x0)
                vy1 = min(h, y1)
                vx1 = min(w, x1)

                if ry0 >= ry1 or rx0 >= rx1:
                    continue

                # crop rotated mask to valid bounds
                rotated_crop = rotated[ry0:ry1, rx0:rx1]

                if rotated_crop.sum() < 3:
                    continue

                # # no overlap with existing holes
                if (rotated_crop * (1 - hole_mask[vy0:vy1, vx0:vx1])).sum() > 0:
                    continue
                
                # place hole — clipped to field boundary
                hole_mask[vy0:vy1, vx0:vx1][rotated_crop > 0] = 0.0

            else:
                # no rotation — axis-aligned rectangle
                hole_patch = hole_mask[y:y + size_h, x:x + size_w]
                if hole_patch.min() == 0:
                    continue
                hole_mask[y:y + size_h, x:x + size_w] = 0.0

            placed = True
            break

        if not placed:
            continue

    hole_mask = hole_mask * valid_mask.astype(np.float32)
    return hole_mask.astype(np.float32)

def generate_stripes(
    valid_mask: np.ndarray,
    rng,
    stripe_angle: float,
    stripe_width_range=(3, 8),
    stripe_spacing_range=(20, 60),
    coverage=0.2,
):
    h, w = valid_mask.shape

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    theta = np.deg2rad(stripe_angle)

    proj = xx * np.cos(theta) + yy * np.sin(theta)

    stripe_width = rng.uniform(*stripe_width_range)
    stripe_spacing = rng.uniform(*stripe_spacing_range)

    period = stripe_width + stripe_spacing

    stripes = (proj % period) < stripe_width

    rs = np.random.RandomState(rng.randint(0, 10_000_000))
    noise = rs.normal(0, 1, size=(h, w))

    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(noise, sigma=10)

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    local_width = stripe_width * (0.5 + noise)

    stripes_var = (proj % period) < local_width

    if stripes_var.mean() > coverage:
        threshold = np.quantile(local_width, 1 - coverage)
        stripes_var = (proj % period) < threshold

    hole_mask = valid_mask.copy().astype(np.float32)
    hole_mask[stripes_var & (valid_mask == 1)] = 0.0

    return hole_mask

# =========================
# dataset
# =========================

class YieldDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        field_list: list[str],
        stats_path: str,
        gpkg_path: str | None = None,
        patch_size: int = 64,
        stride: int = 32,
        min_valid_ratio: float = 0.5,
        max_holes: int = 3,
        min_hole_size: int = 5,
        max_hole_size: int = 20,
        mode: str = "train",
        seed: int | None = 42,
        cache_size: int = 8,
    ):
        super().__init__()

        self.root = Path(root_dir)
        self.fields = field_list
        self.patch_size = patch_size
        self.stride = stride
        self.min_valid_ratio = min_valid_ratio

        self.max_holes = max_holes
        self.min_hole_size = min_hole_size
        self.max_hole_size = max_hole_size

        self.mode = mode
        self.seed = seed
        self.gpkg_path = gpkg_path
        self.cache_size = cache_size

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        
        # load pre-computed stripe angles if available
        stripe_angles_path = Path(root_dir) / "stripe_angles.json"
        if stripe_angles_path.exists():
            with open(stripe_angles_path, "r", encoding="utf-8") as f:
                self.stripe_angles = json.load(f)
        else:
            self.stripe_angles = {}

        self.cont_mean = stats["continuous_mean"]
        self.cont_std = stats["continuous_std"]
        self.cat_classes = stats["categorical_classes"]

        self.cont_feats = [
            "dem",
            "hand",
            "ndvi",
            "rtp_local",
            "rtp_regional",
            "slope",
            "soil",
            "twi",
        ]

        self.cat_feats = [
            "geomorphons",
            "relief_class",
            "aspect_categ",
        ]

        self.cache = OrderedDict()
        self.samples = []
        self._prepare_index()

    # =========================

    def _get_field_gpkg_path(self, field: str) -> Path:
        field_dir = self.root / field

        local_gpkg = field_dir / "field.gpkg"
        if local_gpkg.exists():
            return local_gpkg

        if self.gpkg_path is not None:
            return Path(self.gpkg_path)

        raise FileNotFoundError(
            f"Не найден field.gpkg для поля {field}: {local_gpkg}"
        )

    # =========================

    def _put_cache(self, field: str, value):
        self.cache[field] = value
        self.cache.move_to_end(field)

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    # =========================

    def _load_field(self, field: str):
        if field in self.cache:
            self.cache.move_to_end(field)
            return self.cache[field]

        field_dir = self.root / field

        rasters = {}

        for feat in self.cont_feats:
            rasters[feat] = read_raster(field_dir / f"{feat}.tif")

        rasters["aspect"] = read_raster(field_dir / "aspect.tif")

        for feat in self.cat_feats:
            rasters[feat] = read_raster(field_dir / f"{feat}.tif")

        yield_r = read_raster(field_dir / "yield.tif")

        gpkg_path = self._get_field_gpkg_path(field)
        field_mask = rasterize_field(
            gpkg_path,
            field_dir / "yield.tif"
        )

        value = (rasters, yield_r, field_mask)
        self._put_cache(field, value)
        return value

    # =========================

    def _prepare_index(self):
        for field in self.fields:
            _, yield_r, field_mask = self._load_field(field)

            H, W = yield_r.shape

            if H < self.patch_size or W < self.patch_size:
                continue

            ys = list(range(0, H - self.patch_size + 1, self.stride))
            xs = list(range(0, W - self.patch_size + 1, self.stride))

            if len(ys) == 0:
                ys = [0]
            if len(xs) == 0:
                xs = [0]

            if ys[-1] != H - self.patch_size:
                ys.append(H - self.patch_size)
            if xs[-1] != W - self.patch_size:
                xs.append(W - self.patch_size)

            for y in ys:
                for x in xs:
                    y1 = y + self.patch_size
                    x1 = x + self.patch_size

                    field_patch = field_mask[y:y1, x:x1]

                    if field_patch.mean() < self.min_valid_ratio:
                        continue

                    self.samples.append((field, y, x))

    # =========================

    def _get_rng(self, idx: int):
        if self.mode == "train":
            if self.seed is None:
                return random.Random()
            return random.Random((self.seed + 1) * 10_000_003 + idx * 97)
        base_seed = 42 if self.seed is None else self.seed
        return random.Random(base_seed + idx)

    # =========================

    def __len__(self):
        return len(self.samples)

    # =========================

    def __getitem__(self, idx):
        field, y, x = self.samples[idx]

        rasters, yield_r, field_mask = self._load_field(field)

        y1 = y + self.patch_size
        x1 = x + self.patch_size

        field_patch = field_mask[y:y1, x:x1].astype(np.float32)
        yield_patch = yield_r[y:y1, x:x1].astype(np.float32)

        # base valid mask: inside field boundary and where real yield values exist
        valid_mask = np.isfinite(yield_patch).astype(np.float32)
        valid_mask = valid_mask * field_patch

        # hole_mask: 1 = observed, 0 = hole
        hole_mask = valid_mask.copy()

        if self.mode in ["train", "val", "test"]:
            rng = self._get_rng(idx)

            stripe_angle = self.stripe_angles.get(field, 0.0)

            hole_mask = generate_holes(
                valid_mask=valid_mask,
                rng=rng,
                max_holes=self.max_holes,
                min_size=self.min_hole_size,
                max_size=self.max_hole_size,
                min_valid_fraction=0.3,
                max_attempts_per_hole=20,
                stripe_angle=stripe_angle,
            )

            hole_mask = hole_mask * valid_mask

        # =========================
        # channels
        # =========================

        channels = []

        # continuous
        for feat in self.cont_feats:
            p = rasters[feat][y:y1, x:x1].astype(np.float32)
            p = normalize(p, self.cont_mean[feat], self.cont_std[feat])
            channels.append(p)

        # aspect -> sin/cos
        a = rasters["aspect"][y:y1, x:x1].astype(np.float32)
        nan_mask = np.isnan(a)
        a = np.nan_to_num(a, nan=0.0)

        s, c = aspect_to_sin_cos(a)
        s[nan_mask] = 0.0
        c[nan_mask] = 0.0
        channels += [s.astype(np.float32), c.astype(np.float32)]

        # categorical
        for feat in self.cat_feats:
            p = rasters[feat][y:y1, x:x1]
            p = np.nan_to_num(p, nan=-9999).astype(np.int32)
            oh = one_hot_encode(p, self.cat_classes[feat])
            channels += list(oh)

        # yield context
        y_norm = normalize(
            yield_patch,
            self.cont_mean["yield"],
            self.cont_std["yield"],
        )

        masked_y = y_norm * hole_mask

        channels += [
            masked_y.astype(np.float32),
            hole_mask.astype(np.float32),
            field_patch.astype(np.float32),
        ]

        x_tensor = torch.tensor(np.stack(channels), dtype=torch.float32)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)

        hole_tensor = torch.tensor(hole_mask, dtype=torch.float32).unsqueeze(0)
        field_tensor = torch.tensor(field_patch, dtype=torch.float32).unsqueeze(0)

        return x_tensor, y_tensor, hole_tensor, field_tensor
