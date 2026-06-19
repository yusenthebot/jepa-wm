"""Closed-loop evaluation: the only acceptance that counts. Run CEM-MPC in the
real PointMaze and measure how often the agent actually reaches the goal."""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .envs import PointMazePixels
from .planner import CEMPlanner


def _to_z(encoder, img_chw: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(img_chw).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(x)[0]


@torch.no_grad()
def closed_loop_eval(encoder, predictor, cfg: Config, device: str,
                     n_episodes: int, seed_offset: int = 10_000,
                     capture_traj: bool = True, log=print, cost_fn=None):
    encoder.eval(); predictor.eval()
    env = PointMazePixels(cfg.env_id, img_size=cfg.img_size,
                          max_episode_steps=cfg.max_episode_steps,
                          cam_distance=cfg.cam_distance,
                          enlarge_agent=cfg.enlarge_agent, agent_size=cfg.agent_size)
    planner = CEMPlanner(
        predictor, horizon=cfg.plan_horizon, iters=cfg.cem_iters,
        samples=cfg.cem_samples, elite_frac=cfg.cem_elite_frac,
        init_std=cfg.cem_init_std, action_dim=cfg.action_dim, device=device,
        cost_fn=cost_fn)

    successes, steps_to_goal = [], []
    sample_traj = None
    for ep in range(n_episodes):
        img, xy, gxy, _ = env.reset(seed=seed_offset + ep)
        goal_img = env.render_goal_image(gxy)
        z_goal = _to_z(encoder, goal_img, device)
        traj_imgs, traj_acts = [img], []
        success = False
        for t in range(cfg.max_episode_steps):
            z0 = _to_z(encoder, img, device)
            a = planner.plan(z0, z_goal).cpu().numpy()
            img, xy, success, term, trunc, _ = env.step(a)
            traj_imgs.append(img); traj_acts.append(a)
            if success:
                steps_to_goal.append(t + 1)
                break
            if term or trunc:
                break
        successes.append(1.0 if success else 0.0)
        if capture_traj and ep == 0:
            sample_traj = (np.stack(traj_imgs), np.stack(traj_acts) if traj_acts else None)
        log(f"[eval] ep {ep+1}/{n_episodes} success={success} "
            f"final_dist={np.linalg.norm(xy-gxy):.3f}")
    env.close()
    sr = float(np.mean(successes))
    return {"success_rate": sr,
            "mean_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else None,
            "n_episodes": n_episodes}, sample_traj


@torch.no_grad()
def random_baseline(cfg: Config, n_episodes: int, seed_offset: int = 20_000):
    env = PointMazePixels(cfg.env_id, img_size=cfg.img_size,
                          max_episode_steps=cfg.max_episode_steps,
                          cam_distance=cfg.cam_distance,
                          enlarge_agent=cfg.enlarge_agent, agent_size=cfg.agent_size)
    rng = np.random.default_rng(123)
    succ = []
    for ep in range(n_episodes):
        env.reset(seed=seed_offset + ep)
        a = np.zeros(cfg.action_dim, dtype=np.float32)
        success = False
        for _ in range(cfg.max_episode_steps):
            a = 0.7 * a + 0.3 * rng.uniform(-1, 1, size=cfg.action_dim).astype(np.float32)
            _, _, success, term, trunc, _ = env.step(a)
            if success or term or trunc:
                break
        succ.append(1.0 if success else 0.0)
    env.close()
    return float(np.mean(succ))
