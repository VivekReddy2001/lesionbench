"""Aggregate finished runs into RESULTS.md, results/summary.json and figures."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data

METHOD_LABELS = {
    "erm": "Baseline (no augmentation)",
    "aug": "Classic augmentation",
    "weighted": "+ class-weighted loss",
    "focal": "+ focal loss",
    "oversample": "+ class-balanced oversampling",
    "gan": "+ GAN top-up (real + synthetic)",
    "gan_mix": "+ GAN mix (balanced, 50% synthetic per minority class)",
    "synthetic_only": "Synthetic only (1,000 GAN images per class; test on real)",
}
ORDER = list(METHOD_LABELS)
HEADLINE = ["balanced_accuracy", "macro_f1", "accuracy", "macro_auroc", "ece"]
NAMES = {
    "balanced_accuracy": "Bal. acc",
    "macro_f1": "Macro-F1",
    "accuracy": "Accuracy",
    "macro_auroc": "Macro AUROC",
    "ece": "ECE ↓",
}

# fixed categorical order for figures
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _load_runs(out: Path) -> dict[tuple[str, str], list[dict]]:
    runs = defaultdict(list)
    for p in sorted((out / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        runs[(r["split"], r["config"]["method"])].append(r)
    return runs


def _ms(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def _plot_recall(runs, path: Path, methods: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    for a in (fig, ax):
        a.set_facecolor("#fcfcfb")
    k = len(data.CLASSES)
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        rec = np.mean([r["test"]["per_class_recall"] for r in runs[("lesion", m)]], axis=0)
        xs = np.arange(k) + (i - (len(methods) - 1) / 2) * width
        ax.bar(xs, rec, width=width * 0.9, color=COLORS[ORDER.index(m)], label=m, zorder=2)
    ax.set_xticks(range(k), [f"{c}\n(n={n})" for c, n in zip(data.CLASSES, _class_counts(), strict=True)])
    ax.set_ylabel("Test recall (mean over seeds)", color="#52514e")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e4e3dd", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


_COUNTS: list[int] = []


def _class_counts() -> list[int]:
    return _COUNTS


def _sample_grid(out: Path, data_path: Path, path: Path, per_class: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(out / "gan" / "seed0" / "synthetic.npz")
    ds = data.load(data_path)
    rng = np.random.default_rng(0)
    rows = [c for c in range(len(data.CLASSES)) if (z["labels"] == c).any()]
    fig, axes = plt.subplots(
        len(rows),
        2 * per_class + 1,
        figsize=(2 * per_class * 0.62 + 0.6, len(rows) * 0.66),
        dpi=150,
        gridspec_kw={"wspace": 0.04, "hspace": 0.06},
    )
    fig.patch.set_facecolor("#fcfcfb")
    for ri, c in enumerate(rows):
        real = ds.images[rng.choice(np.flatnonzero(ds.labels == c), per_class, replace=False)]
        syn = z["images"][np.flatnonzero(z["labels"] == c)[:per_class]]
        for j in range(2 * per_class + 1):
            ax = axes[ri, j]
            ax.axis("off")
            if j < per_class:
                ax.imshow(real[j])
            elif j > per_class:
                ax.imshow(syn[j - per_class - 1])
        axes[ri, 0].text(-8, 32, data.CLASSES[c], ha="right", va="center", fontsize=7, color="#52514e")
    axes[0, per_class // 2].set_title("real", fontsize=8, color="#52514e")
    axes[0, per_class + 1 + per_class // 2].set_title("synthetic (WGAN-GP)", fontsize=8, color="#52514e")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def build_report(out: Path, data_path: Path) -> dict:
    runs = _load_runs(out)
    ds = data.load(data_path)
    _COUNTS[:] = np.bincount(ds.labels, minlength=len(data.CLASSES)).tolist()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    summary: dict = {"lesion": {}, "image": {}, "gan": {}}
    md = []

    # --- main table
    md += [
        "## Main results: lesion-level split",
        "",
        "Test set, mean ± std over seeds. Checkpoint chosen by validation balanced accuracy.",
        "",
        "| Method | " + " | ".join(NAMES[m] for m in HEADLINE) + " | Seeds |",
        "|---|" + "---:|" * (len(HEADLINE) + 1),
    ]
    for m in ORDER:
        rs = runs.get(("lesion", m))
        if not rs:
            continue
        cells = [_ms([r["test"][k] for r in rs]) for k in HEADLINE]
        md.append(f"| {METHOD_LABELS[m]} | " + " | ".join(cells) + f" | {len(rs)} |")
        summary["lesion"][m] = {k: float(np.mean([r["test"][k] for r in rs])) for k in HEADLINE}
        summary["lesion"][m]["per_class_recall"] = np.mean([r["test"]["per_class_recall"] for r in rs], axis=0).tolist()

    # --- per-class recall
    md += [
        "",
        "## Per-class recall (lesion-level split, mean over seeds)",
        "",
        "| Method | " + " | ".join(f"{c} (n={n})" for c, n in zip(data.CLASSES, _COUNTS, strict=True)) + " |",
        "|---|" + "---:|" * len(data.CLASSES),
    ]
    for m in ORDER:
        if m in summary["lesion"]:
            rec = summary["lesion"][m]["per_class_recall"]
            md.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in rec) + " |")

    # --- split comparison
    md += [
        "",
        "## Split protocol: image-level vs lesion-level",
        "",
        "| Method | Split | Test images whose lesion is also in train | Bal. acc | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in ORDER:
        for split in ("image", "lesion"):
            rs = runs.get((split, m))
            if not rs or not runs.get(("image", m)):
                continue
            ov = np.mean([r["test_lesion_overlap"] for r in rs])
            md.append(
                f"| {m} | {split} | {ov:.1%} | {_ms([r['test']['balanced_accuracy'] for r in rs])} | "
                f"{_ms([r['test']['accuracy'] for r in rs])} | {_ms([r['test']['macro_f1'] for r in rs])} |"
            )
            summary[split][m] = {k: float(np.mean([r["test"][k] for r in rs])) for k in HEADLINE}
            summary[split][m]["lesion_overlap"] = float(ov)

    # --- GAN memorisation
    mem = {}
    for p in sorted((out / "gan").glob("seed*/memorisation.json")):
        for c, v in json.loads(p.read_text()).items():
            mem.setdefault(c, []).append(v)
    if mem:
        md += [
            "",
            "## Does the GAN copy its training images?",
            "",
            "Nearest-neighbour L2 distance (pixels in [0,1], 64×64×3) to the class's *training* images. "
            "If the generator memorised, synthetic images would sit much closer to training images than "
            "held-out real test images do.",
            "",
            "| Class | Train images | Synthetic generated | Synthetic → train (median) | "
            "Real test → train (median) | Synthetic closer than 5th pct of real |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for c, vs in mem.items():
            md.append(
                f"| {c} | {np.mean([v['train_images'] for v in vs]):.0f} | "
                f"{np.mean([v['synthetic_generated'] for v in vs]):.0f} | "
                f"{np.mean([v['synthetic_nn_median'] for v in vs]):.2f} | "
                f"{np.mean([v['test_real_nn_median'] for v in vs]):.2f} | "
                f"{np.mean([v['synthetic_closer_than_5pct_real'] for v in vs]):.1%} |"
            )
            summary["gan"][c] = {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}

    # --- figures
    available = [m for m in ["aug", "oversample", "gan", "gan_mix"] if ("lesion", m) in runs]
    if available:
        _plot_recall(runs, figs / "per_class_recall.png", available)
    if (out / "gan" / "seed0" / "synthetic.npz").exists():
        _sample_grid(out, data_path, figs / "samples.png")

    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(md))
    return summary
