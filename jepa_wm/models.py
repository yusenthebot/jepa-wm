"""Encoder, latent predictor, and a detached decoder (viz only).

Design notes:
- The predictor is built on GRUCell with a manual rollout loop. nn.GRU has had
  flaky MPS support; GRUCell + an explicit loop is reliable and is exactly what
  CEM-MPC needs to batch-rollout candidate action sequences anyway.
- The decoder is trained on stop-gradient(latent). It can never move the encoder
  or predictor. It exists only to turn imagined latents into pictures.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Small CNN: image -> latent vector."""

    def __init__(self, img_size: int, latent_dim: int, channels=(32, 64, 64, 128),
                 normalize: bool = False):
        super().__init__()
        layers = []
        c_in = 3
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=c_out),
                nn.SiLU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        feat = img_size // (2 ** len(channels))
        self.flat_dim = c_in * feat * feat
        self.head = nn.Linear(self.flat_dim, latent_dim)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1]
        h = self.conv(x)
        h = h.flatten(1)
        z = self.head(h)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class Predictor(nn.Module):
    """Latent dynamics: z_{t+1} ~ f(z_t, a_t) via a GRUCell core."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(latent_dim + action_dim, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, latent_dim)
        self.hidden = hidden

    def init_hidden(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device, dtype=dtype)

    def step(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor):
        """One latent step. Returns (z_next, h_next).

        We predict a residual delta on z so the dynamics start near identity,
        which is a sane prior for small physics steps and helps early training.
        """
        x = self.in_proj(torch.cat([z, a], dim=-1))
        h = self.cell(x, h)
        dz = self.out(h)
        return z + dz, h

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor):
        """Open-loop rollout. z0: (B, D), actions: (B, T, action_dim).

        Returns preds: (B, T, D) — predicted latents for steps 1..T.
        """
        b, t, _ = actions.shape
        h = self.init_hidden(b, z0.device, z0.dtype)
        z = z0
        preds = []
        for k in range(t):
            z, h = self.step(z, actions[:, k], h)
            preds.append(z)
        return torch.stack(preds, dim=1)


class Decoder(nn.Module):
    """Deconv decoder: latent -> image. Trained on stop-grad(latent) for viz."""

    def __init__(self, img_size: int, latent_dim: int, channels=(128, 64, 64, 32)):
        super().__init__()
        self.start = img_size // (2 ** len(channels))
        self.c0 = channels[0]
        self.fc = nn.Linear(latent_dim, self.c0 * self.start * self.start)
        layers = []
        c_in = channels[0]
        for c_out in channels[1:]:
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=c_out),
                nn.SiLU(),
            ]
            c_in = c_out
        layers += [nn.ConvTranspose2d(c_in, 3, kernel_size=4, stride=2, padding=1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.c0, self.start, self.start)
        return torch.sigmoid(self.deconv(h))


class EMA:
    """Exponential moving average target encoder. Stop-grad by construction."""

    def __init__(self, model: nn.Module, momentum: float):
        self.momentum = momentum
        self.target = copy.deepcopy(model)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module):
        m = self.momentum
        for tp, op in zip(self.target.parameters(), online.parameters()):
            tp.mul_(m).add_(op.detach(), alpha=1 - m)
        for tb, ob in zip(self.target.buffers(), online.buffers()):
            tb.copy_(ob)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)


def info_nce(z_a: torch.Tensor, z_p: torch.Tensor, temp: float = 0.1):
    """Symmetric InfoNCE over temporal positive pairs (z_a=o_t, z_p=o_{t+1}).

    In-batch negatives: anchor i's negatives are all other frames' positives,
    which sit at DIFFERENT positions, so they get pushed apart. This is what
    makes latent distance track state/spatial difference (unlike pure temporal
    invariance, which merges neighbors and erases position). Inputs are assumed
    L2-normalized, so cosine similarity == dot product.
    """
    logits = (z_a @ z_p.T) / temp           # (B, B)
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _var_cov(z: torch.Tensor, gamma: float):
    b, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    cov = (zc.T @ zc) / (b - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d
    return var_loss, cov_loss


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, gamma: float = 1.0):
    """Full VICReg over a positive pair (z1, z2).

    invariance: MSE(z1, z2)  -> here z1,z2 are temporally-near frames, so this
                pulls spatially-near observations together (latent metric ~ space).
    variance:   hinge to keep each dim's std >= gamma (anti-collapse).
    covariance: decorrelate dimensions.
    Returns (inv, var, cov).
    """
    inv = F.mse_loss(z1, z2)
    v1, c1 = _var_cov(z1, gamma)
    v2, c2 = _var_cov(z2, gamma)
    return inv, 0.5 * (v1 + v2), 0.5 * (c1 + c2)


def vicreg_var_cov(z: torch.Tensor, gamma: float = 1.0):
    """VICReg variance + covariance regularizers (no invariance term here — the
    prediction loss plays that role). Actively fights representation collapse.

    Returns (var_loss, cov_loss).
    """
    b, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    cov = (z.T @ z) / (b - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss, cov_loss
