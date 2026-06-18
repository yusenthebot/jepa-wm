"""Fast sweep: which latent_dim makes latent distance track spatial distance?

Reuses the already-collected dataset; trains ONLY the temporal-VICReg encoder
(no planning) and reports corr(latent_dist, spatial_dist) + probe R2 per dim.
Hypothesis: corr ~ sqrt(intrinsic_dim / latent_dim), so small dim wins.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_wm.config import Config
from jepa_wm.data import load_episodes, sample_frames_xy
from jepa_wm.metrics import linear_probe_r2, rankme
from jepa_wm.train import pick_device, train_encoder

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def corr_latent_spatial(z_np, xy, rng):
    n = z_np.shape[0]
    a, b = rng.integers(0, n, 5000), rng.integers(0, n, 5000)
    ld = np.linalg.norm(z_np[a] - z_np[b], axis=1)
    sd = np.linalg.norm(xy[a] - xy[b], axis=1)
    return float(np.corrcoef(ld, sd)[0, 1])


def main():
    device = pick_device("mps")
    base = Config()
    data_path = RUNS / f"data_{base.env_id}_{base.n_train_episodes}_{base.img_size}.npz"
    eps = load_episodes(str(data_path))
    rng = np.random.default_rng(0)
    imgs_np, xy = sample_frames_xy(eps, n=2500, rng=rng)
    frames = torch.from_numpy(imgs_np).float().to(device) / 255.0

    dims = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2", "4", "8", "16", "32"])]
    print(f"{'dim':>4} {'corr':>7} {'probeR2':>8} {'RankMe':>7}")
    for d in dims:
        cfg = replace(base, latent_dim=d, enc_epochs=8)
        enc = train_encoder(eps, cfg, device, log=lambda *_: None)
        with torch.no_grad():
            z = torch.cat([enc(frames[i:i + 256]) for i in range(0, frames.shape[0], 256)], 0)
        z_np = z.cpu().numpy()
        c = corr_latent_spatial(z_np, xy, rng)
        r2 = linear_probe_r2(z_np, xy)
        print(f"{d:>4} {c:>7.3f} {r2:>8.3f} {rankme(z):>7.2f}")


if __name__ == "__main__":
    main()
