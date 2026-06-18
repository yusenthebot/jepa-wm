# STATUS — jepa-wm

Round: 1 (FLOOR) — VERIFIED ✓  | Mode: evolving / frontier
Updated: 2026-06-19

## Goal (floor, not ceiling)
JEPA-style low-compute world model on PointMaze: CNN encoder + GRU latent predictor,
predict next latent (no pixel reconstruction in the loss), CEM-MPC plans in latent
space. Close the loop: encode -> latent rollout -> plan -> real env success rate.
Ship every round: (1) imagined-rollout GIF (detached decoder), (2) performance curve.

## Round 1 result (verified by REAL planning success, not loss)
planner success 0.62 vs random 0.12 | latent-spatial corr 0.42 | probe R² 0.77
| RankMe 8.0/8 (no collapse) | rollout MSE 0.007 | ~2.3 min on MPS.

## Final architecture (what actually works)
- Encoder: augmentation-VICReg (photometric views), latent_dim=8, FROZEN after training.
- Predictor: GRU on cached frozen latents (forward dynamics).
- Planner: CEM-MPC, goal latent = encode(agent placed at goal).
- Detached decoder: viz only.
- Env: top-down camera + enlarged agent ball (dynamics unchanged) so the ball is a
  signal objectives can't ignore.

## Key findings (round 1)
- Loss is NOT acceptance: an EMA-target joint design had rollout MSE 0.002 / RankMe 57
  but planner==random because corr(latent dist, spatial dist)=0.10. Decoupling
  (freeze encoder) + small latent_dim fixed it.
- Observation trap: default agent ball ~3px -> every SSL objective ignored it
  (supervised probe R²=0.999 though). Fixed by enlarging the ball.

## Next (frontier round candidates)
1. Shrink the ball back -> distractor-robust representation (the hard open part).
2. Push corr higher (better encoder / whitened planning metric) -> success > 0.8.
3. Stochastic latent / SSM predictor / rollout curriculum / harder mazes.

## Resume (cold start)
Read progress.md + this file + git log. Run:
  uv run python scripts/run_round.py --round 1 --tag r1-floor   (reproduce floor)
  uv run python scripts/sweep_latent_dim.py 4,8,16,32           (latent-dim study)
