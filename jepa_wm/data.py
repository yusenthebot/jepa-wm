"""Offline data collection with a random policy, plus a windowed sequence
sampler for multi-step JEPA training. Images stored as uint8 to keep it cheap."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .envs import PointMazePixels


@dataclass
class Episode:
    images: np.ndarray   # (T+1, 3, H, W) uint8
    actions: np.ndarray  # (T, action_dim) float32
    xy: np.ndarray       # (T+1, 2) float32  (eval/diagnostic only)


def collect_random(env_id: str, n_episodes: int, img_size: int,
                   max_steps: int, seed: int = 0) -> list[Episode]:
    env = PointMazePixels(env_id, img_size=img_size, max_episode_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    for ep in range(n_episodes):
        img, xy, _, _ = env.reset(seed=seed + ep)
        imgs = [(img * 255).astype(np.uint8)]
        acts, xys = [], [xy]
        # Smooth-ish random walk: low-pass the noise so the ball actually moves.
        a = np.zeros(env.action_dim, dtype=np.float32)
        for _ in range(max_steps):
            a = 0.7 * a + 0.3 * rng.uniform(-1, 1, size=env.action_dim).astype(np.float32)
            nimg, nxy, success, term, trunc, _ = env.step(a)
            imgs.append((nimg * 255).astype(np.uint8))
            acts.append(a.copy())
            xys.append(nxy)
            if term or trunc:
                break
        episodes.append(Episode(
            images=np.stack(imgs), actions=np.stack(acts), xy=np.stack(xys)))
    env.close()
    return episodes


def save_episodes(episodes: list[Episode], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n=len(episodes),
        **{f"img_{i}": e.images for i, e in enumerate(episodes)},
        **{f"act_{i}": e.actions for i, e in enumerate(episodes)},
        **{f"xy_{i}": e.xy for i, e in enumerate(episodes)},
    )


def load_episodes(path: str) -> list[Episode]:
    d = np.load(path)
    n = int(d["n"])
    return [Episode(d[f"img_{i}"], d[f"act_{i}"], d[f"xy_{i}"]) for i in range(n)]


class WindowSampler:
    """Samples (B, K+1) image windows + (B, K) actions for rollout training."""

    def __init__(self, episodes: list[Episode], k: int, device: str):
        self.k = k
        self.device = device
        self.images = [torch.from_numpy(e.images) for e in episodes]      # uint8
        self.actions = [torch.from_numpy(e.actions).float() for e in episodes]
        self.index = []  # (ep, start) valid windows of exactly k+1 frames
        for ei, e in enumerate(episodes):
            t = e.actions.shape[0]  # frames = t+1
            for s in range(0, t - k + 1):  # empty when t < k -> episode skipped
                self.index.append((ei, s))
        if not self.index:
            raise ValueError(f"no windows of length k+1={k+1}; episodes too short")
        self.index = np.asarray(self.index)

    def __len__(self):
        return len(self.index)

    def sample(self, batch: int, rng: np.random.Generator):
        pick = rng.integers(0, len(self.index), size=batch)
        img_w, act_w = [], []
        for idx in pick:
            ei, s = self.index[idx]
            img_w.append(self.images[ei][s:s + self.k + 1])
            act_w.append(self.actions[ei][s:s + self.k])
        imgs = torch.stack(img_w).to(self.device).float() / 255.0  # (B, K+1, 3, H, W)
        acts = torch.stack(act_w).to(self.device)                  # (B, K, A)
        return imgs, acts


def sample_frames_xy(episodes: list[Episode], n: int, rng: np.random.Generator):
    """Flat (n,3,H,W) frames + (n,2) xy for RankMe / linear probe diagnostics."""
    all_imgs, all_xy = [], []
    for e in episodes:
        all_imgs.append(e.images)
        all_xy.append(e.xy)
    imgs = np.concatenate(all_imgs, axis=0)
    xy = np.concatenate(all_xy, axis=0)
    if imgs.shape[0] > n:
        pick = rng.choice(imgs.shape[0], size=n, replace=False)
        imgs, xy = imgs[pick], xy[pick]
    return imgs, xy
