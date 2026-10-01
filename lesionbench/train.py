"""Classifier training for every imbalance strategy in the benchmark.

All methods share the same network, optimiser, schedule and *number of
gradient steps*. Fixing steps (not epochs) matters: an oversampled or
GAN-augmented training set is several times larger than the real one, and
counting epochs would silently give those methods more compute.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import evaluate
from .models import SmallResNet

METHODS = ["erm", "aug", "weighted", "focal", "oversample", "gan", "gan_mix", "synthetic_only"]


@dataclass
class TrainConfig:
    method: str = "aug"
    steps: int = 2000
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 5e-4
    eval_every: int = 100
    focal_gamma: float = 2.0
    seed: int = 0
    threads: int = 4


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> float NCHW in [-1, 1] (the GAN's output range)."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Batched dermoscopy augmentation: flips, 90-degree rotations (lesions
    have no canonical orientation), small translations and colour jitter."""
    n = x.size(0)
    flip_h = torch.rand(n, generator=gen) < 0.5
    flip_v = torch.rand(n, generator=gen) < 0.5
    x = torch.where(flip_h[:, None, None, None], x.flip(3), x)
    x = torch.where(flip_v[:, None, None, None], x.flip(2), x)
    k = torch.randint(0, 4, (n,), generator=gen)
    x = torch.stack([torch.rot90(img, int(r), (1, 2)) for img, r in zip(x, k, strict=True)])
    # translate by up to 4 px with reflection padding
    pad = 4
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dx = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    dy = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    h, w = x.shape[2:]
    x = torch.stack([xp[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w] for i in range(n)])
    # brightness / contrast jitter (+-15%)
    b = (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.3
    c = 1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.3
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean + b).clamp_(-1, 1)


def focal_loss(logits, y, gamma: float):
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, y[:, None]).squeeze(1)
    return (-((1 - logp_t.exp()) ** gamma) * logp_t).mean()


class _Sampler:
    """Draws training batches of indices into a pool of (image, label) pairs."""

    def __init__(self, labels: np.ndarray, probs: np.ndarray | None, rng: np.random.Generator):
        self.n = len(labels)
        self.probs = probs
        self.rng = rng

    def __call__(self, k: int) -> np.ndarray:
        return self.rng.choice(self.n, size=k, replace=True, p=self.probs)


def build_pool(method: str, x_real, y_real, synth: tuple[np.ndarray, np.ndarray] | None, n_classes: int):
    """Return (images, labels, sampling_probs) for a method.

    * ``oversample`` - every class equally likely, minority images repeat.
    * ``gan``        - each class topped up with synthetic images to the size
                       of the largest class, then sampled uniformly.
    * ``gan_mix``    - class-balanced sampling where a minority class draws
                       half its samples from real and half from synthetic.
    * ``synthetic_only`` - synthetic images only (train-on-synthetic,
                       test-on-real; a direct test of synthetic data quality).
    """
    counts = np.bincount(y_real, minlength=n_classes)
    if method in {"erm", "aug", "weighted", "focal"}:
        return x_real, y_real, None
    if method == "oversample":
        p = 1.0 / counts[y_real]
        return x_real, y_real, p / p.sum()
    if synth is None:
        raise ValueError(f"method {method!r} needs synthetic images")
    xs, ys = synth
    if method == "synthetic_only":
        p = 1.0 / np.bincount(ys, minlength=n_classes)[ys]
        return xs, ys, p / p.sum()
    target = counts.max()
    keep_x, keep_y = [x_real], [y_real]
    for c in range(n_classes):
        need = target - counts[c]
        if need > 0:
            idx = np.flatnonzero(ys == c)[:need]
            keep_x.append(xs[idx])
            keep_y.append(ys[idx])
    xp, yp = np.concatenate(keep_x), np.concatenate(keep_y)
    is_real = np.zeros(len(yp), dtype=bool)
    is_real[: len(y_real)] = True
    if method == "gan":
        return xp, yp, None
    if method == "gan_mix":
        # per class: 50% of probability mass on real, 50% on synthetic
        # (majority class has no synthetic images, so all mass goes to real)
        p = np.zeros(len(yp))
        for c in range(n_classes):
            real_c = (yp == c) & is_real
            syn_c = (yp == c) & ~is_real
            if syn_c.any():
                p[real_c] = 0.5 / real_c.sum()
                p[syn_c] = 0.5 / syn_c.sum()
            else:
                p[real_c] = 1.0 / real_c.sum()
        return xp, yp, p / p.sum()
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def predict(model, x: torch.Tensor, batch: int = 512) -> np.ndarray:
    model.eval()
    out = [F.softmax(model(x[i : i + batch]), dim=1) for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def train_classifier(cfg: TrainConfig, train, val, test, n_classes: int, synth=None, log=print) -> dict:
    """Train one model; select the checkpoint with the best validation
    balanced accuracy; report validation and test metrics for it."""
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method: {cfg.method}")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    x_tr, y_tr = train
    xp, yp, probs = build_pool(cfg.method, x_tr, y_tr, synth, n_classes)
    sampler = _Sampler(yp, probs, rng)
    x_pool, y_pool = to_tensor(xp), torch.from_numpy(yp)
    x_val, y_val = to_tensor(val[0]), val[1]
    x_te, y_te = to_tensor(test[0]), test[1]

    counts = np.bincount(y_tr, minlength=n_classes)
    class_w = torch.tensor(len(y_tr) / (n_classes * counts), dtype=torch.float32)

    model = SmallResNet(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.15)

    best, best_state, history = -1.0, None, []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.from_numpy(sampler(cfg.batch_size))
        xb, yb = x_pool[idx], y_pool[idx]
        if cfg.method != "erm":
            xb = augment(xb, gen)
        logits = model(xb)
        if cfg.method == "weighted":
            loss = F.cross_entropy(logits, yb, weight=class_w)
        elif cfg.method == "focal":
            loss = focal_loss(logits, yb, cfg.focal_gamma)
        else:
            loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(predict(model, x_val), y_val, n_classes)
            history.append({"step": step, "loss": loss.item(), "val_bal_acc": m["balanced_accuracy"]})
            if m["balanced_accuracy"] > best:
                best, best_state = m["balanced_accuracy"], copy.deepcopy(model.state_dict())
            log(
                f"[{cfg.method} s{cfg.seed}] step {step}/{cfg.steps} loss {loss.item():.3f} "
                f"val bal-acc {m['balanced_accuracy']:.3f} ({time.time() - t0:.0f}s)"
            )

    model.load_state_dict(best_state)
    val_probs, test_probs = predict(model, x_val), predict(model, x_te)
    return {
        "config": asdict(cfg),
        "val": evaluate(val_probs, y_val, n_classes),
        "test": evaluate(test_probs, y_te, n_classes),
        "history": history,
        "train_seconds": time.time() - t0,
        "pool_size": int(len(yp)),
        "model": model,
        "test_probs": test_probs,
    }
