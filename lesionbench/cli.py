"""Command-line entry point: ``python -m lesionbench <command>``.

prepare    download ISIC 2018 Task 3 (HAM10000) and build the .npz
gan        train the conditional WGAN-GP for one seed and sample from it
classify   train and evaluate one (method, split, seed) configuration
report     aggregate every finished run into tables and figures
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch

from . import data
from .gan import GanConfig, nearest_neighbour_distance, sample, train_gan
from .train import METHODS, TrainConfig, train_classifier

N_CLASSES = len(data.CLASSES)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists: {dest}")
        return
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def cmd_prepare(args) -> None:
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name in data.ISIC_FILES.values():
        _download(f"{data.ISIC_BASE}/{name}", raw / name)
    ds = data.prepare(raw, Path(args.data), size=args.size)
    print(
        f"wrote {args.data}: {ds.images.shape}, class counts {np.bincount(ds.labels).tolist()}, "
        f"{len(set(ds.lesion_ids.tolist()))} lesions"
    )


def _splits(args):
    ds = data.load(Path(args.data))
    split = data.make_split(ds, args.split, args.seed)
    parts = {k: (ds.images[v], ds.labels[v]) for k, v in split.items()}
    return ds, split, parts


def cmd_gan(args) -> None:
    args.split = "lesion"
    ds, split, parts = _splits(args)
    out = Path(args.out) / "gan" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr = parts["train"]
    cfg = GanConfig(iters=args.iters, seed=args.seed, threads=args.threads)
    G = train_gan(cfg, x_tr, y_tr, N_CLASSES)
    torch.save(G.state_dict(), out / "generator.pt")

    counts = np.bincount(y_tr, minlength=N_CLASSES)
    need = {c: int(counts.max() - counts[c]) for c in range(N_CLASSES) if counts[c] < counts.max()}
    xs, ys = sample(G, need, seed=args.seed)
    np.savez_compressed(out / "synthetic.npz", images=xs, labels=ys)

    # Memorisation check, per class: distance of synthetic images to the
    # training set vs. distance of held-out real test images to it.
    x_te, y_te = parts["test"]
    report = {}
    for c in need:
        tr_c = x_tr[y_tr == c]
        syn = nearest_neighbour_distance(xs[ys == c][:500], tr_c)
        real = nearest_neighbour_distance(x_te[y_te == c], tr_c)
        report[data.CLASSES[c]] = {
            "train_images": int(len(tr_c)),
            "synthetic_generated": int(need[c]),
            "synthetic_nn_median": float(np.median(syn)),
            "test_real_nn_median": float(np.median(real)),
            "synthetic_closer_than_5pct_real": float(np.mean(syn < np.percentile(real, 5))),
        }
    (out / "memorisation.json").write_text(json.dumps(report, indent=2))
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    print(json.dumps(report, indent=2))


def cmd_classify(args) -> None:
    ds, split, parts = _splits(args)
    synth = None
    if args.method in {"gan", "gan_mix", "synthetic_only"}:
        if args.split != "lesion":
            raise SystemExit("GAN methods are only run on the lesion-level split")
        z = np.load(Path(args.out) / "gan" / f"seed{args.seed}" / "synthetic.npz")
        synth = (z["images"], z["labels"])
    cfg = TrainConfig(method=args.method, steps=args.steps, seed=args.seed, threads=args.threads)
    res = train_classifier(cfg, parts["train"], parts["val"], parts["test"], N_CLASSES, synth=synth)
    res.pop("model")
    probs = res.pop("test_probs")
    res["split"] = args.split
    res["test_lesion_overlap"] = data.lesion_overlap(ds, split["train"], split["test"])
    out = Path(args.out) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.method}_s{args.seed}"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    np.save(out / f"{stem}_test_probs.npy", probs.astype(np.float32))
    t = res["test"]
    print(
        f"{stem}: bal-acc {t['balanced_accuracy']:.3f} macro-F1 {t['macro_f1']:.3f} "
        f"acc {t['accuracy']:.3f} ECE {t['ece']:.3f}"
    )


def cmd_report(args) -> None:
    from .report import build_report

    build_report(Path(args.out), Path(args.data))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="lesionbench")
    p.add_argument("--data", default="data/ham10000_64.npz")
    p.add_argument("--out", default="results")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare")
    s.add_argument("--raw-dir", default="data/raw")
    s.add_argument("--size", type=int, default=64)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("gan")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--iters", type=int, default=3000)
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(fn=cmd_gan)

    s = sub.add_parser("classify")
    s.add_argument("--method", choices=METHODS, required=True)
    s.add_argument("--split", choices=["lesion", "image"], default="lesion")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(fn=cmd_classify)

    s = sub.add_parser("report")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
