# progress — jepa-wm

## Current state (round 3, FRONTIER — VERIFIED 2026-06-19)
Public repo: github.com/yusenthebot/jepa-wm. Capability ladder, 3 verified rounds:
- R1 FLOOR (open, big ball): planner 0.62 vs random 0.12, corr 0.42, probe 0.77.
- R2 (open, HARD small distractor ball): planner 0.50 vs random 0.12, corr 0.69, probe 0.89.
  Mechanism: spatial-softmax + multi-step inverse dynamics (ACRO, k=24).
- R3 (UMaze obstacle, big ball): planner 0.36 vs random 0.20, corr 0.51, probe 0.86, ~4 min.
  Mechanism: LONG-HORIZON CEM over learned dynamics finds the detour around the wall.
  Horizon ablation (the world-model proof): random 0.15 -> H=15 0.30 -> H=40 0.40.
  Presets: --preset floor / distractor / umaze. NOTE: tasks differ -> success not
  comparable across rounds; the signal is planner >> random on each new capability.

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
- Current ceiling: UMaze obstacle, long-horizon CEM, success 0.36 vs random 0.20 (R3).
  World model DOES plan around the wall (horizon ablation proves it) but greedy
  Euclidean-latent cost caps it.
- Next frontier (Round 4 — the clear lever): learn a latent TEMPORAL-DISTANCE / quasimetric
  (steps-to-go between latents) and use it as the CEM terminal cost. Non-greedy -> handles
  obstacles properly -> push UMaze >0.5, scale to Medium/Large. Train it from the random
  data (k-step temporal regression, or contrastive on temporal gap) on frozen latents.
- Radical ideas weighed: goal-conditioned latent value/Q; forward+inverse joint encoder;
  stochastic latent for multimodal futures; hierarchy (subgoal latents).
- Fidelity/stack ladder: Open(big) ✓ -> Open(small) ✓ -> UMaze ✓ -> UMaze+temporal-dist
  -> Medium/Large -> AntMaze -> higher-res / multi-view pixels.
