"""Diagnose why planning fails: does latent distance track spatial distance?

If CEM-MPC minimizes latent distance to a goal latent, planning can only work if
latent L2 distance correlates with true spatial (xy) distance. We test that
directly on the trained encoder, plus the probe R2, to localize the failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_wm.config import Config
from jepa_wm.data import load_episodes, sample_frames_xy
from jepa_wm.metrics import linear_probe_r2, rankme
from jepa_wm.models import Encoder

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else str(RUNS / "round1_r1-floor" / "ckpt.pt")
    cfg = Config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = Encoder(cfg.img_size, cfg.latent_dim, cfg.enc_channels).to(device)
    enc.load_state_dict(ck["encoder"]); enc.eval()

    data_path = RUNS / f"data_{cfg.env_id}_{cfg.n_train_episodes}_{cfg.img_size}.npz"
    eps = load_episodes(str(data_path))
    rng = np.random.default_rng(0)
    imgs, xy = sample_frames_xy(eps, n=2000, rng=rng)
    x = torch.from_numpy(imgs).float().to(device) / 255.0
    with torch.no_grad():
        z = torch.cat([enc(x[i:i+256]) for i in range(0, x.shape[0], 256)], 0)
    z_np = z.cpu().numpy()

    print(f"[ckpt] {ckpt_path}")
    print(f"RankMe={rankme(z):.2f}/{cfg.latent_dim}  probe_R2(xy)={linear_probe_r2(z_np, xy):.3f}")

    # latent-distance vs spatial-distance correlation over random pairs
    n = z_np.shape[0]
    a = rng.integers(0, n, 4000); b = rng.integers(0, n, 4000)
    ld = np.linalg.norm(z_np[a] - z_np[b], axis=1)
    sd = np.linalg.norm(xy[a] - xy[b], axis=1)
    r = np.corrcoef(ld, sd)[0, 1]
    print(f"corr(latent_dist, spatial_dist) = {r:.3f}   <-- planning needs this HIGH (>0.7)")

    # how much latent variance is explained by xy (linear) vs total
    # (already have probe R2); also report per-dim std spread
    stds = z_np.std(0)
    print(f"latent per-dim std: min={stds.min():.3f} max={stds.max():.3f} mean={stds.mean():.3f}")
    print(f"frac dims with std>0.5: {(stds>0.5).mean():.2f}")


if __name__ == "__main__":
    main()
