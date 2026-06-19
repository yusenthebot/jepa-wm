"""Frontier spike: can we encode position from the HARD small-ball observation
(the distractor case the floor crutched around)?

Tests distractor-robust mechanisms and reports corr(latent,spatial)+probe R2:
  - multi-step inverse dynamics (ACRO-style): predict a_t from (z_t, z_{t+k}).
    Larger k -> bigger displacement -> stronger signal that ignores static
    background (only the CONTROLLABLE state predicts actions).
  - same with a spatial-softmax encoder (keypoint inductive bias).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_wm.config import Config
from jepa_wm.data import collect_random, save_episodes, load_episodes, sample_frames_xy
from jepa_wm.metrics import linear_probe_r2, rankme
from jepa_wm.models import Encoder

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "runs"
dev = "mps" if torch.backends.mps.is_available() else "cpu"


class SSMEncoder(nn.Module):
    """Spatial-softmax: outputs expected (x,y) of K feature-map channels = keypoints."""
    def __init__(self, latent_dim, img_size=64):
        super().__init__()
        ch = [32, 64, 64]; layers = []; ci = 3
        for co in ch:
            layers += [nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(8, co), nn.SiLU()]; ci = co
        self.conv = nn.Sequential(*layers)
        self.K = latent_dim // 2
        self.red = nn.Conv2d(ci, self.K, 1)
        g = torch.linspace(-1, 1, img_size // 8)
        self.register_buffer("gx", g.view(1, 1, 1, -1))
        self.register_buffer("gy", g.view(1, 1, -1, 1))

    def forward(self, x):
        h = self.red(self.conv(x)); B = h.shape[0]; S = h.shape[-1]
        a = F.softmax(h.view(B, self.K, -1) / 0.5, -1).view(B, self.K, S, S)
        ex = (a * self.gx).sum((2, 3)); ey = (a * self.gy).sum((2, 3))
        return torch.cat([ex, ey], -1)


def get_data():
    p = RUNS / "data_smallball_80.npz"
    if p.exists():
        return load_episodes(str(p))
    cfg = Config()
    eps = collect_random(cfg.env_id, 80, cfg.img_size, cfg.max_episode_steps,
                         seed=0, enlarge_agent=False)
    save_episodes(eps, str(p))
    return eps


def evaluate(enc, eps, rng, tag):
    enc.eval()
    imgs, xy = sample_frames_xy(eps, 2000, rng)
    x = torch.from_numpy(imgs).float().to(dev) / 255.0
    with torch.no_grad():
        z = torch.cat([enc(x[i:i+256]) for i in range(0, x.shape[0], 256)], 0)
    zc = z.cpu().numpy(); n = zc.shape[0]; a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    ld = np.linalg.norm(zc[a]-zc[b], axis=1); sd = np.linalg.norm(xy[a]-xy[b], axis=1)
    print(f"{tag}: probeR2={linear_probe_r2(zc, xy):.3f} corr={np.corrcoef(ld, sd)[0,1]:.3f} RankMe={rankme(z):.2f}")


def train_multistep_inverse(eps, k, arch, dim=16, steps=1500):
    rng = np.random.default_rng(0)
    imgs = [torch.from_numpy(e.images).float() for e in eps]
    acts = [torch.from_numpy(e.actions).float() for e in eps]
    idx = np.array([(ei, t) for ei, e in enumerate(eps)
                    for t in range(e.actions.shape[0] - k)])
    enc = (SSMEncoder(dim) if arch == "ssm" else Encoder(64, dim, Config().enc_channels)).to(dev)
    inv = nn.Sequential(nn.Linear(2*dim, 256), nn.SiLU(), nn.Linear(256, 2)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(inv.parameters()), lr=3e-4)
    for _ in range(steps):
        pick = rng.integers(0, len(idx), 256)
        o1 = torch.stack([imgs[ei][t] for ei, t in idx[pick]]).to(dev) / 255.0
        o2 = torch.stack([imgs[ei][t+k] for ei, t in idx[pick]]).to(dev) / 255.0
        a = torch.stack([acts[ei][t] for ei, t in idx[pick]]).to(dev)
        z1, z2 = enc(o1), enc(o2)
        pa = inv(torch.cat([z1, z2], -1))
        std = torch.sqrt(z1.var(0) + 1e-4); var = F.relu(1.0 - std).mean()
        loss = F.mse_loss(pa, a) + 1.0 * var
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, float(F.mse_loss(pa, a))


def main():
    eps = get_data()
    allimg = np.concatenate([e.images for e in eps], 0).astype(np.float32)
    print(f"small-ball dataset: {len(eps)} eps, cross-frame std {allimg.std(0).mean():.2f}")
    rng = np.random.default_rng(1)
    for arch in ["cnn", "ssm"]:
        for k in [1, 4, 8]:
            enc, amse = train_multistep_inverse(eps, k, arch)
            evaluate(enc, eps, rng, f"inverse arch={arch} k={k} (act_mse={amse:.3f})")


if __name__ == "__main__":
    main()
