"""Networks: a compact ResNet classifier and a class-conditional WGAN-GP.

Everything is sized to train on a laptop CPU at 64x64 so that the full
benchmark is reproducible without a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + (x if self.skip is None else self.skip(x)))


class SmallResNet(nn.Module):
    """ResNet-10-style network (4 stages, one block each), ~1.2M parameters."""

    def __init__(self, n_classes: int = 7, width: int = 32, dropout: float = 0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.layers = nn.Sequential(
            _Block(w, w, 1), _Block(w, 2 * w, 2), _Block(2 * w, 4 * w, 2), _Block(4 * w, 8 * w, 2)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(8 * w, n_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


# --------------------------------------------------------------------------
# Class-conditional WGAN-GP (64x64)
# --------------------------------------------------------------------------
class Generator(nn.Module):
    """DCGAN-style generator; the class enters through a learned embedding
    concatenated with the noise vector."""

    def __init__(self, n_classes: int = 7, z_dim: int = 128, width: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.embed = nn.Embedding(n_classes, z_dim)
        w = width
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * z_dim, 8 * w, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8 * w),
            nn.ReLU(True),  # 4
            nn.ConvTranspose2d(8 * w, 4 * w, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * w),
            nn.ReLU(True),  # 8
            nn.ConvTranspose2d(4 * w, 2 * w, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(True),  # 16
            nn.ConvTranspose2d(2 * w, w, 4, 2, 1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(True),  # 32
            nn.ConvTranspose2d(w, 3, 4, 2, 1),
            nn.Tanh(),  # 64
        )

    def forward(self, z, y):
        h = torch.cat([z, self.embed(y)], dim=1)[:, :, None, None]
        return self.net(h)


class Critic(nn.Module):
    """Projection critic (Miyato & Koyama, 2018). No batch norm, because the
    gradient penalty is defined per sample; layer norm is used instead."""

    def __init__(self, n_classes: int = 7, width: int = 64):
        super().__init__()
        w = width

        def down(cin, cout, hw):
            return [nn.Conv2d(cin, cout, 4, 2, 1), nn.LayerNorm([cout, hw, hw]), nn.LeakyReLU(0.2, True)]

        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 2, 1),
            nn.LeakyReLU(0.2, True),  # 32
            *down(w, 2 * w, 16),
            *down(2 * w, 4 * w, 8),
            *down(4 * w, 8 * w, 4),
        )
        self.out = nn.Linear(8 * w * 16, 1)
        self.embed = nn.Embedding(n_classes, 8 * w * 16)

    def forward(self, x, y):
        h = self.net(x).flatten(1)
        return self.out(h).squeeze(1) + (self.embed(y) * h).sum(1)


def gradient_penalty(critic: Critic, real, fake, y) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(mix, y)
    (grad,) = torch.autograd.grad(score.sum(), mix, create_graph=True)
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
