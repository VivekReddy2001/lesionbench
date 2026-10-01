"""Fast tests on tiny synthetic data; none of them needs the real dataset."""

from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from lesionbench import data
from lesionbench.gan import GanConfig, nearest_neighbour_distance, sample, train_gan
from lesionbench.metrics import evaluate, expected_calibration_error
from lesionbench.train import TrainConfig, augment, build_pool, to_tensor, train_classifier

K = len(data.CLASSES)


def fake_dataset(n_lesions: int = 300, seed: int = 0) -> data.Dataset:
    """Imbalanced toy data where some lesions have several photos, like HAM10000."""
    rng = np.random.default_rng(seed)
    probs = np.array([0.11, 0.67, 0.05, 0.03, 0.11, 0.015, 0.015])
    ys, lesions = [], []
    for i in range(n_lesions):
        y = rng.choice(K, p=probs / probs.sum())
        for _ in range(rng.choice([1, 1, 2, 3])):
            ys.append(y)
            lesions.append(f"L{i:04d}")
    ys = np.asarray(ys)
    images = (rng.random((len(ys), 16, 16, 3)) * 50 + ys[:, None, None, None] * 25).astype(np.uint8)
    return data.Dataset(images, ys, np.asarray(lesions), np.asarray([f"I{i}" for i in range(len(ys))]))


# ---------------------------------------------------------------- data
def test_prepare_reads_isic_layout(tmp_path):
    rng = np.random.default_rng(0)
    ids = [f"ISIC_{i:07d}" for i in range(6)]
    with zipfile.ZipFile(tmp_path / data.ISIC_FILES["images"], "w") as zf:
        for i in ids:
            buf = io.BytesIO()
            Image.fromarray((rng.random((45, 60, 3)) * 255).astype(np.uint8)).save(buf, "JPEG")
            zf.writestr(f"ISIC2018_Task3_Training_Input/{i}.jpg", buf.getvalue())
    labels = io.StringIO()
    w = csv.writer(labels)
    w.writerow(["image", *data.CLASSES])
    for k, i in enumerate(ids):
        w.writerow([i, *[1.0 if c == k % K else 0.0 for c in range(K)]])
    with zipfile.ZipFile(tmp_path / data.ISIC_FILES["labels"], "w") as zf:
        zf.writestr("ISIC2018_Task3_Training_GroundTruth/gt.csv", labels.getvalue())
    with open(tmp_path / data.ISIC_FILES["groups"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "lesion_id", "diagnosis_confirm_type"])
        for k, i in enumerate(ids):
            w.writerow([i, f"HAM_{k // 2}", "histo"])

    ds = data.prepare(tmp_path, tmp_path / "out.npz", size=8)
    assert ds.images.shape == (6, 8, 8, 3) and ds.images.dtype == np.uint8
    assert ds.labels.tolist() == [k % K for k in range(6)]
    assert ds.lesion_ids.tolist() == ["HAM_0", "HAM_0", "HAM_1", "HAM_1", "HAM_2", "HAM_2"]
    back = data.load(tmp_path / "out.npz")
    assert np.array_equal(back.images, ds.images)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lesion_split_never_leaks(seed):
    ds = fake_dataset(seed=seed)
    s = data.make_split(ds, "lesion", seed)
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        assert data.lesion_overlap(ds, s[a], s[b]) == 0.0
    assert sorted(np.concatenate(list(s.values()))) == list(range(len(ds)))
    assert 0.15 < len(s["test"]) / len(ds) < 0.25


def test_image_split_does_leak():
    ds = fake_dataset()
    s = data.make_split(ds, "image", 0)
    assert data.lesion_overlap(ds, s["train"], s["test"]) > 0.05


def test_splits_are_stratified():
    ds = fake_dataset(n_lesions=600)
    s = data.make_split(ds, "lesion", 0)
    full = np.bincount(ds.labels, minlength=K) / len(ds)
    test = np.bincount(ds.labels[s["test"]], minlength=K) / len(s["test"])
    assert np.abs(full - test).max() < 0.05


# ------------------------------------------------------------- metrics
def test_ece_zero_when_calibrated():
    # 80% confident and right 80% of the time
    probs = np.tile([0.8, 0.2], (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    probs = np.tile([0.99, 0.01], (10, 1))
    labels = np.array([0] * 5 + [1] * 5)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.49)


def test_balanced_accuracy_punishes_majority_guessing():
    labels = np.array([0] * 90 + [1] * 10)
    probs = np.tile([0.9, 0.1], (100, 1))
    m = evaluate(probs, labels, 2)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["per_class_recall"] == [1.0, 0.0]


# ----------------------------------------------------------- sampling
def test_oversampling_balances_classes():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    _, _, p = build_pool("oversample", np.zeros((100, 2, 2, 3), np.uint8), y, None, 3)
    mass = [p[y == c].sum() for c in range(3)]
    assert mass == pytest.approx([1 / 3] * 3)


def test_gan_top_up_fills_every_class_to_the_majority():
    y = np.array([0] * 50 + [1] * 10 + [2] * 5)
    xs = np.ones((200, 2, 2, 3), np.uint8)
    ys = np.repeat([1, 2], 100)
    _, yp, p = build_pool("gan", np.zeros((65, 2, 2, 3), np.uint8), y, (xs, ys), 3)
    assert np.bincount(yp).tolist() == [50, 50, 50]
    assert p is None


def test_gan_mix_puts_half_the_minority_mass_on_real_images():
    y = np.array([0] * 50 + [1] * 10)
    xr = np.zeros((60, 2, 2, 3), np.uint8)
    xs, ys = np.ones((100, 2, 2, 3), np.uint8), np.ones(100, np.int64)
    xp, yp, p = build_pool("gan_mix", xr, y, (xs, ys), 2)
    minority = yp == 1
    real = np.zeros(len(yp), bool)
    real[:60] = True
    assert p[minority & real].sum() == pytest.approx(p[minority & ~real].sum())
    assert p[yp == 0].sum() == pytest.approx(p[yp == 1].sum())


def test_augment_keeps_shape_and_range():
    x = to_tensor(np.random.default_rng(0).integers(0, 256, (8, 16, 16, 3), dtype=np.uint8))
    out = augment(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert out.min() >= -1 and out.max() <= 1


# ------------------------------------------------------------ training
@pytest.mark.parametrize("method", ["erm", "weighted", "focal", "oversample"])
def test_classifier_smoke(method):
    ds = fake_dataset()
    s = data.make_split(ds, "lesion", 0)
    parts = {k: (ds.images[v], ds.labels[v]) for k, v in s.items()}
    cfg = TrainConfig(method=method, steps=6, batch_size=16, eval_every=3, threads=1)
    res = train_classifier(cfg, parts["train"], parts["val"], parts["test"], K, log=lambda *_: None)
    assert 0.0 <= res["test"]["balanced_accuracy"] <= 1.0
    assert len(res["test"]["per_class_recall"]) == K
    assert res["test_probs"].shape == (len(s["test"]), K)


def test_gan_smoke_and_sampling():
    ds = fake_dataset(n_lesions=40)
    imgs = np.stack([np.asarray(Image.fromarray(i).resize((64, 64))) for i in ds.images])
    G = train_gan(
        GanConfig(iters=2, n_critic=1, batch_size=8, width=8, threads=1), imgs, ds.labels, K, log=lambda *_: None
    )
    xs, ys = sample(G, {1: 3, 5: 2})
    assert xs.shape == (5, 64, 64, 3) and xs.dtype == np.uint8
    assert ys.tolist() == [1, 1, 1, 5, 5]


def test_nearest_neighbour_distance():
    ref = np.random.default_rng(0).integers(0, 256, (5, 4, 4, 3), dtype=np.uint8)
    assert np.allclose(nearest_neighbour_distance(ref, ref), 0)
    assert (nearest_neighbour_distance(255 - ref, ref) > 0).all()
