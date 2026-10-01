"""Train a class-conditional WGAN-GP on the *training split only* and
sample synthetic images from it.

Training the generator on the training split of each seed is essential:
a GAN fitted on all of HAM10000 has seen the test images, and any accuracy
it "adds" is leakage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .models import Critic, Generator, gradient_penalty
from .train import to_tensor


@dataclass
class GanConfig:
    iters: int = 3000  # generator updates
    n_critic: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    gp_weight: float = 10.0
    z_dim: int = 128
    width: int = 32
    seed: int = 0
    threads: int = 4


def _flip_rot(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Symmetries that dermoscopy images genuinely have. Unlike colour or
    crop augmentation, these do not leak into the generated distribution."""
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x = torch.rot90(x, k, (2, 3))
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(3)
    return x


def train_gan(cfg: GanConfig, images: np.ndarray, labels: np.ndarray, n_classes: int, log=print):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    x = to_tensor(images)
    y = torch.from_numpy(labels)
    # Square-root class balancing: rare classes are seen more often than their
    # frequency, but not so often that the critic memorises them outright.
    counts = np.bincount(labels, minlength=n_classes)
    p = 1.0 / np.sqrt(counts[labels])
    p /= p.sum()

    G = Generator(n_classes, cfg.z_dim, cfg.width)
    D = Critic(n_classes, cfg.width)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.0, 0.9))

    t0 = time.time()
    for it in range(1, cfg.iters + 1):
        for _ in range(cfg.n_critic):
            idx = torch.from_numpy(rng.choice(len(y), cfg.batch_size, p=p))
            real, yb = _flip_rot(x[idx], gen), y[idx]
            fake = G(torch.randn(cfg.batch_size, cfg.z_dim), yb).detach()
            w_dist = D(real, yb).mean() - D(fake, yb).mean()
            loss_d = -w_dist + cfg.gp_weight * gradient_penalty(D, real, fake, yb)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
        yb = y[torch.from_numpy(rng.choice(len(y), cfg.batch_size, p=p))]
        loss_g = -D(G(torch.randn(cfg.batch_size, cfg.z_dim), yb), yb).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        if it % 250 == 0 or it == cfg.iters:
            log(
                f"[gan s{cfg.seed}] iter {it}/{cfg.iters} W-dist {w_dist.item():.3f} "
                f"loss_g {loss_g.item():.3f} ({time.time() - t0:.0f}s)"
            )
    return G


@torch.no_grad()
def sample(G: Generator, per_class: dict[int, int], seed: int = 0, batch: int = 256):
    """Return (uint8 NHWC images, labels) with ``per_class[c]`` images of class c."""
    G.eval()
    torch.manual_seed(seed)
    xs, ys = [], []
    for c, n in per_class.items():
        for start in range(0, n, batch):
            k = min(batch, n - start)
            out = G(torch.randn(k, G.z_dim), torch.full((k,), c, dtype=torch.long))
            xs.append(((out.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy())
            ys.append(np.full(k, c, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def nearest_neighbour_distance(query: np.ndarray, ref: np.ndarray, batch: int = 256) -> np.ndarray:
    """L2 distance (pixel space, [0,1] scale) from each query image to its
    nearest image in ``ref``. Used to check whether the GAN copies its
    training set: synthetic images should not sit closer to the training data
    than genuinely unseen real images do."""
    r = torch.from_numpy(ref.reshape(len(ref), -1)).float() / 255.0
    out = []
    for i in range(0, len(query), batch):
        q = torch.from_numpy(query[i : i + batch].reshape(-1, r.shape[1])).float() / 255.0
        out.append(torch.cdist(q, r).min(dim=1).values)
    return torch.cat(out).numpy()
