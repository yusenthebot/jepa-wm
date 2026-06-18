"""Representation-health metrics. The point of these is to catch collapse that a
falling latent-MSE would otherwise hide."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def rankme(z: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023).

    RankMe(Z) = exp(-sum_i p_i log p_i), p_i = sigma_i / (||sigma||_1 + eps),
    where sigma_i are the singular values of the embedding matrix Z (N x D).
    A collapsed representation has effective rank ~1; a healthy one approaches D.
    """
    z = z.detach().float().cpu()
    # center is not part of the original RankMe; it uses raw singular values.
    sigma = torch.linalg.svdvals(z)
    p = sigma / (sigma.sum() + eps)
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def linear_probe_r2(z: np.ndarray, y: np.ndarray) -> float:
    """R^2 of a least-squares linear map z -> y (here y is true xy).

    Diagnostic only: the planner never sees xy. High R^2 means the latent
    actually encodes position, so latent distance is a sane planning cost.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = z.shape[0]
    split = max(1, int(0.8 * n))
    zt, yt = z[:split], y[:split]
    zv, yv = z[split:], y[split:]
    if zv.shape[0] == 0:
        zv, yv = zt, yt
    a = np.concatenate([zt, np.ones((zt.shape[0], 1))], axis=1)
    w, *_ = np.linalg.lstsq(a, yt, rcond=None)
    av = np.concatenate([zv, np.ones((zv.shape[0], 1))], axis=1)
    pred = av @ w
    ss_res = ((yv - pred) ** 2).sum()
    ss_tot = ((yv - yv.mean(axis=0)) ** 2).sum() + 1e-9
    return float(1.0 - ss_res / ss_tot)
