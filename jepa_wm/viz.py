"""Human-facing artifacts: imagined-rollout GIF and performance curves.

The decoder used here is DETACHED — it consumes stop-grad(latent) and never
contributes gradient to the encoder or predictor. It is a window into the
model's imagination, not part of the world model.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


@torch.no_grad()
def imagined_rollout_gif(
    encoder,
    predictor,
    decoder,
    obs_seq: torch.Tensor,     # (T+1, 3, H, W) real frames, [0,1]
    actions: torch.Tensor,     # (T, action_dim)
    out_path: str,
    device: str = "cpu",
    fps: int = 8,
):
    """Render: top row = ground truth frames, bottom row = decoded imagined
    latents rolled open-loop from the FIRST frame only. Side-by-side so a human
    can see drift honestly."""
    encoder.eval(); predictor.eval(); decoder.eval()
    obs_seq = obs_seq.to(device)
    actions = actions.to(device)
    t = actions.shape[0]

    z0 = encoder(obs_seq[:1])                       # (1, D)
    preds = predictor.rollout(z0, actions.unsqueeze(0))[0]   # (T, D)
    latents = torch.cat([z0, preds], dim=0)         # (T+1, D)
    imagined = decoder(latents).cpu().numpy()       # (T+1, 3, H, W)
    truth = obs_seq.cpu().numpy()

    frames = []
    for k in range(t + 1):
        gt = np.transpose(truth[k], (1, 2, 0))
        im = np.transpose(imagined[k], (1, 2, 0))
        sep = np.ones((gt.shape[0], 2, 3))
        row = np.concatenate([gt, sep, im], axis=1)
        frames.append(_to_uint8(row))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    return out_path


def append_metrics(metrics_path: str, record: dict):
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_metrics(metrics_path: str) -> list[dict]:
    p = Path(metrics_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def performance_curves(metrics_path: str, out_path: str):
    """Plot success rate / rollout error / RankMe over rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_metrics(metrics_path)
    if not rows:
        return None
    rounds = [r["round"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(rounds, [r.get("success_rate") for r in rows], "o-", color="#1b7837", label="planner")
    if any("random_success_rate" in r for r in rows):
        axes[0].plot(rounds, [r.get("random_success_rate") for r in rows], "s--",
                     color="#999999", label="random")
    axes[0].set_title("planning success rate"); axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("round"); axes[0].legend(fontsize=8)

    axes[1].plot(rounds, [r.get("rollout_mse") for r in rows], "o-", color="#b2182b")
    axes[1].set_title("multi-step latent rollout MSE"); axes[1].set_xlabel("round")

    axes[2].plot(rounds, [r.get("rankme") for r in rows], "o-", color="#2166ac", label="RankMe")
    if any("probe_r2" in r for r in rows):
        ax2 = axes[2].twinx()
        ax2.plot(rounds, [r.get("probe_r2") for r in rows], "^--", color="#762a83",
                 label="xy probe R²")
        ax2.set_ylabel("xy probe R²", color="#762a83"); ax2.set_ylim(-0.05, 1.05)
    axes[2].set_title("RankMe effective rank"); axes[2].set_xlabel("round")
    axes[2].set_ylabel("RankMe", color="#2166ac")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
