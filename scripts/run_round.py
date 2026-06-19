"""Run one round of the JEPA world-model loop end to end:

  collect (random policy) -> train JEPA -> train detached decoder
  -> closed-loop CEM-MPC eval (+ random baseline) -> rollout GIF + curves.

The acceptance signal is the real planning success rate, not the loss.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_wm.config import Config
from jepa_wm.data import collect_random, load_episodes, save_episodes
from jepa_wm.eval import closed_loop_eval, random_baseline
from jepa_wm.train import pick_device, train_decoder, train_jepa
from jepa_wm.viz import append_metrics, imagined_rollout_gif, performance_curves

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--tag", type=str, default="r1-floor")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test config")
    ap.add_argument("--reuse-data", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--preset", choices=["floor", "distractor"], default="floor",
                    help="floor = big ball + aug-VICReg; distractor = small ball + "
                         "spatial-softmax + multi-step inverse dynamics")
    args = ap.parse_args()

    if args.preset == "distractor":
        # hard small-ball observation: the controllable agent is a tiny distractor.
        cfg = Config(enlarge_agent=False, encoder_type="ssm",
                     encoder_objective="inverse", inverse_k=24, latent_dim=8)
    else:
        cfg = Config()
    if args.quick:
        cfg = Config(**{**cfg.to_dict(), "n_train_episodes": 12, "enc_epochs": 2,
                        "pred_epochs": 2, "dec_epochs": 2, "n_eval_episodes": 5,
                        "cem_samples": 64, "cem_iters": 2, "plan_horizon": 8})
    if args.eval_episodes is not None:
        cfg = Config(**{**cfg.to_dict(), "n_eval_episodes": args.eval_episodes})

    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    RUNS.mkdir(exist_ok=True)
    rdir = RUNS / f"round{args.round}_{args.tag}"
    rdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. data  (ball tag so big-ball and small-ball datasets don't collide)
    ball = "big" if cfg.enlarge_agent else "small"
    data_path = RUNS / f"data_{cfg.env_id}_{cfg.n_train_episodes}_{cfg.img_size}_{ball}.npz"
    if args.reuse_data and data_path.exists():
        print(f"[data] reuse {data_path}")
        episodes = load_episodes(str(data_path))
    else:
        print(f"[data] collecting {cfg.n_train_episodes} random episodes ({ball} ball)...")
        episodes = collect_random(cfg.env_id, cfg.n_train_episodes, cfg.img_size,
                                  cfg.max_episode_steps, seed=cfg.seed,
                                  enlarge_agent=cfg.enlarge_agent, agent_size=cfg.agent_size)
        save_episodes(episodes, str(data_path))
    n_trans = sum(e.actions.shape[0] for e in episodes)
    print(f"[data] {len(episodes)} episodes, {n_trans} transitions")

    # 2. train JEPA world model
    encoder, predictor, ema, diag = train_jepa(episodes, cfg, device)
    print(f"[diag] RankMe={diag['rankme']:.2f}/{cfg.latent_dim} "
          f"probe_R2={diag['probe_r2']:.3f} latent_spatial_corr={diag['latent_spatial_corr']:.3f} "
          f"rollout_mse={diag['rollout_mse']:.5f}")

    # 3. detached decoder (viz only)
    decoder = train_decoder(encoder, episodes, cfg, device)

    # 4. closed-loop planning eval (the real acceptance) + random baseline
    eval_res, sample_traj = closed_loop_eval(
        encoder, predictor, cfg, device, n_episodes=cfg.n_eval_episodes)
    rand_sr = random_baseline(cfg, n_episodes=cfg.n_eval_episodes)
    print(f"[eval] planner success={eval_res['success_rate']:.2f} "
          f"random success={rand_sr:.2f}")

    # 5a. imagined rollout GIF (detached decoder)
    gif_path = rdir / "imagined_rollout.gif"
    if sample_traj is not None and sample_traj[1] is not None:
        imgs, acts = sample_traj
        h = min(cfg.plan_horizon, acts.shape[0])
        obs_seq = torch.from_numpy(imgs[:h + 1]).float()
        act_seq = torch.from_numpy(acts[:h]).float()
        imagined_rollout_gif(encoder, predictor, decoder, obs_seq, act_seq,
                             str(gif_path), device=device)
        print(f"[viz] wrote {gif_path}")

    # 5b. metrics + curves
    record = {
        "round": args.round, "tag": args.tag,
        "success_rate": eval_res["success_rate"],
        "random_success_rate": rand_sr,
        "mean_steps_to_goal": eval_res["mean_steps_to_goal"],
        "rollout_mse": diag["rollout_mse"],
        "rankme": diag["rankme"], "latent_dim": diag["latent_dim"],
        "probe_r2": diag["probe_r2"],
        "latent_spatial_corr": diag["latent_spatial_corr"],
        "n_train_episodes": cfg.n_train_episodes, "n_transitions": n_trans,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    append_metrics(str(RUNS / "metrics.jsonl"), record)
    curve_path = performance_curves(str(RUNS / "metrics.jsonl"), str(RUNS / "performance.png"))
    print(f"[viz] wrote {curve_path}")

    # 6. checkpoint + round summary
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                "decoder": decoder.state_dict(), "cfg": cfg.to_dict()},
               rdir / "ckpt.pt")
    (rdir / "summary.json").write_text(json.dumps(record, indent=2))
    print(f"[done] round {args.round} in {record['minutes']} min")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
