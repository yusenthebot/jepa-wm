# STATUS — jepa-wm

Round: 3 (FRONTIER, obstacle-aware planning) — VERIFIED ✓  | Mode: evolving / frontier
Updated: 2026-06-19  | public: github.com/yusenthebot/jepa-wm

## Goal (floor, not ceiling)
JEPA-style low-compute world model on PointMaze: CNN/keypoint encoder + GRU latent
predictor, predict next latent (no pixel reconstruction in the loss), CEM-MPC plans in
latent space. Close the loop: encode -> latent rollout -> plan -> real env success.
Ship every round: (1) imagined-rollout GIF (detached decoder), (2) performance curve.

## Results (verified by REAL planning success, not loss) — capability ladder
- R1 FLOOR (open, big ball): planner 0.62 vs random 0.12 | corr 0.42 | probe 0.77.
- R2 (open, HARD small distractor ball): planner 0.50 vs random 0.12 | corr 0.69 | probe 0.89.
  Mechanism = spatial-softmax + multi-step inverse dynamics (ACRO, k=24).
- R3 (UMaze obstacle): planner 0.36 vs random 0.20 | corr 0.51 | probe 0.86. Mechanism =
  LONG-HORIZON CEM over learned dynamics finds the detour (horizon ablation 0.15→0.30→0.40).
  Tasks DIFFER -> success not comparable across rounds; signal = planner >> random per round.
  Presets: --preset floor (R1) / distractor (R2) / umaze (R3).

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

## Next (Round 4 — the clear lever)
Greedy Euclidean-latent cost caps UMaze (0.36). Learn a latent TEMPORAL-DISTANCE
(quasimetric / goal-conditioned steps-to-go) and use it as the CEM terminal cost ->
non-greedy obstacle planning -> push UMaze >0.5 and scale to Medium/Large mazes.
Then: stochastic latent / SSM predictor / curiosity data.

## Resume (cold start)
Read progress.md + this file + git log. Run with .venv/bin/python (NOT uv run on the
flaky net). Reproduce:
  scripts/run_round.py --round 1 --tag r1-floor                           (R1 big-ball floor)
  scripts/run_round.py --round 2 --tag r2-distractor --preset distractor  (R2 small-ball)
  scripts/run_round.py --round 3 --tag r3-umaze --preset umaze            (R3 UMaze)
