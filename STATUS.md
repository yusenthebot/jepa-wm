# STATUS — jepa-wm

Round: 2 (FRONTIER, distractor-robust) — VERIFIED ✓  | Mode: evolving / frontier
Updated: 2026-06-19  | public: github.com/yusenthebot/jepa-wm

## Goal (floor, not ceiling)
JEPA-style low-compute world model on PointMaze: CNN/keypoint encoder + GRU latent
predictor, predict next latent (no pixel reconstruction in the loss), CEM-MPC plans in
latent space. Close the loop: encode -> latent rollout -> plan -> real env success.
Ship every round: (1) imagined-rollout GIF (detached decoder), (2) performance curve.

## Results (verified by REAL planning success, not loss)
- Round 1 FLOOR (big ball): planner 0.62 vs random 0.12 | corr 0.42 | probe R² 0.77 | RankMe 8/8.
- Round 2 FRONTIER (HARD small distractor ball): planner 0.50 vs random 0.12 | corr 0.69
  | probe R² 0.89 | RankMe 6.75/8. Mechanism = spatial-softmax keypoints + multi-step
  inverse dynamics (ACRO, k=24). Corr/probe went UP on a HARDER obs = real progress.
  Presets: --preset floor (R1) / --preset distractor (R2).

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
1. Push small-ball success past the floor (longer inverse horizon, forward+inverse combo,
   or whitened planning metric) -> success > 0.62 on the hard obs.
2. Stochastic/variational latent dynamics (multimodal futures).
3. SSM/Mamba predictor; rollout-horizon curriculum; harder mazes (UMaze/Medium) needing
   obstacle-aware planning (greedy latent-distance CEM will fail there -> learned value).

## Resume (cold start)
Read progress.md + this file + git log. Run with .venv/bin/python (NOT uv run on the
flaky net). Reproduce:
  scripts/run_round.py --round 1 --tag r1-floor                      (big-ball floor)
  scripts/run_round.py --round 2 --tag r2-distractor --preset distractor  (small-ball)
