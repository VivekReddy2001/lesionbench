"""Dataset preparation and splitting for HAM10000 (ISIC 2018 Task 3).

The raw challenge release is ~2.7 GB of 600x450 JPEGs. We center-crop each
image to a square, resize it, and store everything in a single compressed
``.npz`` so that experiments load in well under a second.

HAM10000 contains several photographs of the same physical lesion (7,470
lesions for 10,015 images). Splitting by *image* therefore lets near-duplicate
views of one lesion land in both train and test. ``make_split`` supports both
the leak-prone image-level split and the correct lesion-level split so the
difference can be measured rather than argued about.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Column order of the ISIC 2018 ground-truth CSV.
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAMES = {
    "MEL": "melanoma",
    "NV": "melanocytic nevus",
    "BCC": "basal cell carcinoma",
    "AKIEC": "actinic keratosis / Bowen's",
    "BKL": "benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
}

ISIC_BASE = "https://isic-challenge-data.s3.amazonaws.com/2018"
ISIC_FILES = {
    "images": "ISIC2018_Task3_Training_Input.zip",
    "labels": "ISIC2018_Task3_Training_GroundTruth.zip",
    "groups": "ISIC2018_Task3_Training_LesionGroupings.csv",
}


@dataclass
class Dataset:
    images: np.ndarray  # (N, H, W, 3) uint8
    labels: np.ndarray  # (N,) int64
    lesion_ids: np.ndarray  # (N,) str
    image_ids: np.ndarray  # (N,) str

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx: np.ndarray) -> Dataset:
        return Dataset(self.images[idx], self.labels[idx], self.lesion_ids[idx], self.image_ids[idx])


def _square_resize(img: Image.Image, size: int) -> np.ndarray:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _read_labels(labels_zip: Path) -> dict[str, int]:
    with zipfile.ZipFile(labels_zip) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.DictReader(io.TextIOWrapper(zf.open(name), encoding="utf-8")))
    out = {}
    for r in rows:
        onehot = [float(r[c]) for c in CLASSES]
        out[r["image"]] = int(np.argmax(onehot))
    return out


def _read_groups(groups_csv: Path) -> dict[str, str]:
    with open(groups_csv, newline="", encoding="utf-8") as f:
        return {r["image"]: r["lesion_id"] for r in csv.DictReader(f)}


def prepare(raw_dir: Path, out_path: Path, size: int = 64) -> Dataset:
    """Build ``out_path`` (.npz) from the three ISIC 2018 Task 3 files in ``raw_dir``."""
    raw_dir = Path(raw_dir)
    labels = _read_labels(raw_dir / ISIC_FILES["labels"])
    groups = _read_groups(raw_dir / ISIC_FILES["groups"])

    images, ys, lesions, ids = [], [], [], []
    with zipfile.ZipFile(raw_dir / ISIC_FILES["images"]) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".jpg"))
        for n in names:
            image_id = Path(n).stem
            if image_id not in labels:
                continue
            with zf.open(n) as fh:
                images.append(_square_resize(Image.open(fh), size))
            ys.append(labels[image_id])
            lesions.append(groups[image_id])
            ids.append(image_id)

    ds = Dataset(
        images=np.stack(images),
        labels=np.asarray(ys, dtype=np.int64),
        lesion_ids=np.asarray(lesions),
        image_ids=np.asarray(ids),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=ds.images, labels=ds.labels, lesion_ids=ds.lesion_ids, image_ids=ds.image_ids)
    return ds


def load(path: Path) -> Dataset:
    z = np.load(path, allow_pickle=False)
    return Dataset(z["images"], z["labels"], z["lesion_ids"], z["image_ids"])


def make_split(ds: Dataset, level: str, seed: int, n_folds: int = 5) -> dict[str, np.ndarray]:
    """Stratified train / val / test indices (~70 / 10 / 20).

    ``level="lesion"`` keeps every image of a lesion in the same partition;
    ``level="image"`` ignores lesion identity (the common, leaky protocol).
    """
    if level not in {"lesion", "image"}:
        raise ValueError(f"unknown split level: {level}")
    y = ds.labels
    idx = np.arange(len(y))

    def splitter(k: int, s: int):
        if level == "lesion":
            return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=s)

    groups = ds.lesion_ids if level == "lesion" else None
    # Test = one of five folds (20%).
    trval, test = next(splitter(n_folds, seed).split(idx, y, groups))
    # Val = one of eight folds of the remainder (~10% of the total).
    sub_groups = groups[trval] if groups is not None else None
    sub_tr, sub_val = next(splitter(8, seed + 1).split(trval, y[trval], sub_groups))
    return {"train": trval[sub_tr], "val": trval[sub_val], "test": test}


def lesion_overlap(ds: Dataset, a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of images in ``b`` whose lesion also appears in ``a``."""
    seen = set(ds.lesion_ids[a].tolist())
    return float(np.mean([lid in seen for lid in ds.lesion_ids[b]])) if len(b) else 0.0
