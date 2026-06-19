# progress — jepa-wm

## Current state (round 2, FRONTIER — VERIFIED 2026-06-19)
Public repo: github.com/yusenthebot/jepa-wm.
- Round 1 FLOOR (big ball): planner 0.62 vs random 0.12, corr 0.42, probe R² 0.77.
- Round 2 FRONTIER (HARD small distractor ball, the part R1 crutched around):
  **planner 0.50 vs random 0.12, corr 0.69, probe R² 0.89, RankMe 6.75/8**, ~3.2 min MPS.
  Mechanism: spatial-softmax keypoint encoder + multi-step inverse dynamics (ACRO,
  predict a_t from z_t,z_{t+k}, k=24) -> recovers the controllable state, ignores
  background. Corr/probe went UP on a HARDER obs = real frontier progress, not regression.
  Config presets: --preset floor (R1) / --preset distractor (R2).

### Round-2 spike findings (scripts/spike_distractor.py)
- Multi-step inverse is the ONLY thing that cracked the small ball (everything in R1 got
  corr~0.05). Larger k = bigger displacement = stronger signal: k=8 corr~0.2, k=24 corr~0.69.
- Spatial-softmax (K=4 keypoints, dim=8) >> plain CNN for the keypoint/metric quality.
- Predict a_t ONLY (not all-K actions) — all-K + VICReg was noisier and hurt.

### Final architecture
- Encoder: augmentation-VICReg (two photometric views of the same frame, invariance +
  variance/covariance), latent_dim=8. FROZEN after training.
- Latent dynamics: GRU predictor on cached frozen latents (z_{t+1}=z_t+g(z_t,a_t)).
- Planner: CEM-MPC in latent; goal latent = encode(agent placed at goal).
- Detached decoder: stop-grad, viz only (imagined-rollout GIF).
- Env: top-down camera + enlarged bright agent ball (dynamics unchanged).

## What worked
- Decoupling encoder (frozen) from predictor — stops the encoder scrambling its metric.
- Small latent_dim (8) — keeps latent distance aligned with spatial distance.
- Enlarging the agent ball — makes position a signal SSL objectives can't skip.
- aug-VICReg encoder: probe R² 0.58→0.93 across dims; corr peaks ~0.5 at dim 8.

## What did NOT work (and why — keep these, they're the lessons)
- Joint encoder+predictor + EMA target + VICReg: loss great (rollout 0.002, RankMe 57)
  but corr(latent,spatial)=0.10 -> planner==random. Expressive GRU lets the encoder
  pick any geometry. ROOT CAUSE of round-1 near-failure.
- Temporal-VICReg (pull t,t+Δ together): erases position (merges different states).
- InfoNCE (temporal positives, in-batch negatives): instance discrimination only,
  doesn't preserve the metric (corr~0.05).
- Inverse dynamics (predict a_t from z_t,z_{t+1}): 1-step displacement too small to
  infer the action -> no position pressure.
- Plain AE / foreground-AE / spatial-softmax+temporal: all IGNORED the tiny ball
  (it's ~2% of pixels) and modeled the static background. Supervised probe R²=0.999,
  so position IS learnable — the objectives just don't target it. Fixed by enlarging
  the ball (observation fix), not by a cleverer loss.
- uv install kept failing on the flaky travel network (atomic rollback). Fixed with:
  Tsinghua mirror + curl the big wheels + pip (non-atomic) retry loop.

## Next-round seed (frontier)
The floor deliberately sidesteps the hard part (small distractor ball). Best next
frontiers, ranked ambition×feasibility:
1. Shrink the agent ball back down -> distractor-robust representation (spatial-softmax
   keypoints, slot attention, motion/flow cues). This is the open research part.
2. Lift corr toward 1 (whitened/learned planning metric, or contrastive-on-displacement)
   -> success > 0.8 on the easy obs.
3. Stochastic/variational latent dynamics; SSM/Mamba predictor; rollout-horizon
   curriculum; harder mazes (UMaze/Medium) needing obstacle-aware planning;
   curiosity-driven data.

## Frontier
- Current ceiling: PointMaze-Open SMALL distractor ball, success 0.50, corr 0.69
  (R2 cleared the "small ball" bar that R1 sidestepped).
- Next frontier (bar to clear next): small-ball success > 0.62 (beat the easy-obs floor)
  via longer inverse horizon / forward+inverse / whitened planning metric; THEN harder
  mazes (UMaze) needing obstacle-aware planning (greedy latent-distance CEM fails there).
- Radical ideas weighed: forward+inverse joint (full controllable dynamics), learned
  terminal value for CEM (for non-greedy mazes), stochastic latent for multimodal futures.
- Fidelity/stack ladder: Open(big ball) -> Open(small ball) ✓ -> UMaze -> Medium/Large
  -> AntMaze -> higher-res / multi-view pixels.
