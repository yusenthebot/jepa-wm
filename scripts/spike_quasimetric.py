"""Round 4 spike: a learned latent TEMPORAL-DISTANCE cost for obstacle planning.

Greedy Euclidean-latent CEM caps UMaze because states across the wall are CLOSE in
Euclidean latent distance but FAR in steps (you must detour). We learn d(z_a, z_b) ~
steps-to-go from the random data (regress the temporal gap), then use it as the CEM
terminal cost. Compare planning success: Euclidean cost vs temporal-distance cost.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_wm.config import Config
from jepa_wm.data import load_episodes
from jepa_wm.eval import closed_loop_eval, random_baseline
from jepa_wm.models import Encoder, Predictor
from jepa_wm.train import cache_latents, pick_device, train_encoder, train_predictor

RUNS = Path(__file__).resolve().parents[1] / "runs"


class TemporalDistance(nn.Module):
    """d(z_a, z_b) >= 0, trained to regress steps-to-go. Asymmetric MLP is fine here."""
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, za, zb):
        return self.net(torch.cat([za, zb], dim=-1)).squeeze(-1)


def train_temporal_distance(latents, episodes, cfg, device, kmax=40, steps=4000):
    rng = np.random.default_rng(0)
    dnet = TemporalDistance(cfg.latent_dim).to(device)
    opt = torch.optim.AdamW(dnet.parameters(), lr=3e-4)
    lat = [z.to(device) for z in latents]
    idx = np.asarray([(ei, t) for ei, e in enumerate(episodes)
                      for t in range(e.actions.shape[0])])
    for s in range(steps):
        pick = rng.integers(0, len(idx), 256)
        k = rng.integers(1, kmax + 1, 256)
        za, zb, tgt = [], [], []
        for (ei, t), kk in zip(idx[pick], k):
            T = lat[ei].shape[0]
            tt = min(t, T - 1); uu = min(tt + int(kk), T - 1)
            za.append(lat[ei][tt]); zb.append(lat[ei][uu]); tgt.append(float(uu - tt))
        za = torch.stack(za); zb = torch.stack(zb)
        tgt = torch.tensor(tgt, device=device)
        pred = dnet(za, zb)
        loss = F.smooth_l1_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    dnet.eval()
    return dnet, float(loss)


def make_cost(dnet):
    @torch.no_grad()
    def cost(preds, z_goal):           # preds (N,H,D), z_goal (D,)
        n, h, d = preds.shape
        zg = z_goal.view(1, d).expand(n * h, d)
        dd = dnet(preds.reshape(n * h, d), zg).view(n, h)
        return dd[:, -1] + 0.25 * dd.mean(dim=1)
    return cost


def main():
    device = pick_device("mps")
    cfg = Config(env_id="PointMaze_UMaze-v3", cam_distance=4.0, max_episode_steps=200,
                 n_train_episodes=150, latent_dim=8, enc_epochs=12, pred_epochs=12)
    eps = load_episodes(str(RUNS / "data_umaze_150.npz"))
    enc = train_encoder(eps, cfg, device, log=lambda *_: None)
    lat = cache_latents(enc, eps, device)
    pred = train_predictor(lat, eps, cfg, device, log=lambda *_: None)
    dnet, dloss = train_temporal_distance(lat, eps, cfg, device)
    print(f"[dist] temporal-distance trained (huber={dloss:.3f})")

    base = replace(cfg, n_eval_episodes=20, cem_samples=200, cem_iters=3)
    rand = random_baseline(base, n_episodes=base.n_eval_episodes)
    print(f"[eval] random={rand:.2f}")
    cost = make_cost(dnet)
    for H in [12, 40]:
        c = replace(base, plan_horizon=H)
        eu, _ = closed_loop_eval(enc, pred, c, device, n_episodes=c.n_eval_episodes,
                                 capture_traj=False, log=lambda *_: None)
        qm, _ = closed_loop_eval(enc, pred, c, device, n_episodes=c.n_eval_episodes,
                                 capture_traj=False, log=lambda *_: None, cost_fn=cost)
        print(f"[eval] H={H:>2}  euclidean={eu['success_rate']:.2f}  "
              f"temporal-distance={qm['success_rate']:.2f}")


if __name__ == "__main__":
    main()
