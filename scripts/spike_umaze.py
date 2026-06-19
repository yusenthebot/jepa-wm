"""Frontier spike: obstacle-aware planning on UMaze.

Greedy "reduce latent distance to goal" walks into the U-wall. The question: does
the learned latent world model (predictor) capture the wall, so CEM with a long
enough horizon finds the DETOUR purely by imagining rollouts? We sweep the plan
horizon and compare to random. If long horizon >> short horizon >> ... that's the
world model enabling planning around the obstacle.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_wm.config import Config
from jepa_wm.data import collect_random, load_episodes, save_episodes
from jepa_wm.eval import closed_loop_eval, random_baseline
from jepa_wm.train import pick_device, train_encoder, cache_latents, train_predictor

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main():
    device = pick_device("mps")
    cfg = Config(env_id="PointMaze_UMaze-v3", cam_distance=4.0, max_episode_steps=300,
                 enlarge_agent=True, latent_dim=8, n_train_episodes=150,
                 enc_epochs=12, pred_epochs=12, n_eval_episodes=30)
    dp = RUNS / "data_umaze_150.npz"
    if dp.exists():
        eps = load_episodes(str(dp))
    else:
        print("[data] collecting UMaze (big ball)...")
        eps = collect_random(cfg.env_id, cfg.n_train_episodes, cfg.img_size,
                             cfg.max_episode_steps, seed=0, enlarge_agent=True,
                             cam_distance=cfg.cam_distance)
        save_episodes(eps, str(dp))
    print(f"[data] {len(eps)} eps, {sum(e.actions.shape[0] for e in eps)} transitions")

    enc = train_encoder(eps, cfg, device)
    lat = cache_latents(enc, eps, device)
    pred = train_predictor(lat, eps, cfg, device)

    evalc = replace(cfg, max_episode_steps=150, n_eval_episodes=20)
    rand = random_baseline(evalc, n_episodes=evalc.n_eval_episodes)
    print(f"[eval] random baseline success={rand:.2f}")
    for H in [15, 40]:
        c = replace(evalc, plan_horizon=H, cem_samples=200, cem_iters=3)
        res, _ = closed_loop_eval(enc, pred, c, device, n_episodes=c.n_eval_episodes,
                                  capture_traj=False, log=lambda *_: None)
        print(f"[eval] horizon={H:>2}  planner success={res['success_rate']:.2f}  "
              f"(random {rand:.2f})")


if __name__ == "__main__":
    main()
