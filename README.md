# jepa-wm

**A low-compute, JEPA-style world model that plans in latent space — trained on a laptop.**

![python](https://img.shields.io/badge/python-3.12-blue)
![pytorch](https://img.shields.io/badge/PyTorch-MPS-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/round_1-floor_verified-success)

Most pixel-based world models spend their compute on a **decoder** that reconstructs
every future pixel. JEPA-style models drop that: they learn dynamics by predicting the
**next latent** in representation space and never reconstruct pixels for control. No
decoder in the control loop means a small model — trainable on an Apple-Silicon laptop
with PyTorch-MPS in a couple of minutes.

This repo builds the smallest honest version and **closes the loop**:

> encode pixels → roll the latent forward → plan with CEM-MPC in latent space →
> measure the **real** success rate in the simulator.

A *detached* decoder exists only so humans can **see** what the model imagines — it is
stop-gradient and never trains the world model.

**Round 1 (this release) — verified floor on PointMaze-Open:**

| metric | value | meaning |
|---|---:|---|
| **planning success rate** | **0.62** | reaches the goal in the real sim |
| random baseline | 0.12 | same env, random policy (5× lower) |
| latent↔spatial corr | 0.42 | latent distance tracks real distance (what makes CEM work) |
| xy linear-probe R² | 0.77 | latent encodes position (diagnostic; planner never sees xy) |
| RankMe (eff. rank / 8) | 8.0 | **no representation collapse** |
| multi-step rollout MSE | 0.007 | latent dynamics quality |
| train time (M-series, MPS) | ~2.3 min | collect + train + plan, one round |

![imagined rollout](docs/imagined_rollout.gif)

*Left: ground truth. Right: frames decoded from the imagined latent rollout (open-loop
from the first frame). The decoder is detached — this is the model's imagination, not a
training signal.*

---

## Block diagram

```mermaid
flowchart TB
    subgraph DATA["1 - Offline data (random policy)"]
        ENV0["PointMaze-Open (top-down, prominent agent ball)"] -->|"render 64x64 RGB"| FR["frames o_t"]
        ENV0 -->|"random actions a_t"| ACT["actions a_t"]
    end

    subgraph ENC["2 - Encoder: augmentation-VICReg (no pixel reconstruction)"]
        FR --> AUG["2 photometric views of the SAME frame"]
        AUG --> E1["CNN encoder f_theta -> z (dim 8)"]
        E1 --> INV["invariance: pull the two views together"]
        E1 --> VC["variance + covariance: spread different frames, no collapse"]
        INV --> EOPT["AdamW"]
        VC --> EOPT
        EOPT -.->|"update"| E1
    end

    subgraph DYN["3 - Latent dynamics: predictor on FROZEN features"]
        E1 ==>|"FREEZE + cache latents"| ZC["z_t for every frame"]
        ZC --> PRED["GRU predictor g: z_hat_(t+1) = z_t + g(z_t, a_t)"]
        ACT --> PRED
        PRED --> PL["rollout MSE vs cached z_(t+1)"]
        PL --> POPT["AdamW"]
        POPT -.->|"update (encoder frozen)"| PRED
    end

    subgraph PLAN["4 - CEM-MPC planning in latent space (closed loop)"]
        OBS["current obs"] --> ENC2["frozen encoder"]
        GIMG["goal image (agent placed at goal)"] --> ENC2
        ENC2 -->|"z_0"| CEM["CEM optimizer"]
        ENC2 -.->|"z_goal"| CEM
        CEM -->|"sample action seqs"| ROLL["latent rollout (predictor g)"]
        ROLL -->|"cost = distance to z_goal"| CEM
        CEM -->|"first action"| ENV1["PointMaze"] --> OBS
    end

    subgraph OUT["5 - Mandatory per-round artifacts"]
        E1 --> DEC["DETACHED decoder (stop-grad, viz only)"]
        DEC --> GIF["imagined-rollout GIF"]
        PLAN --> SR["success rate vs random"]
        ENC --> RANK["RankMe / probe R² / latent-spatial corr"]
        SR --> CURVE["performance curve over rounds"]
        RANK --> CURVE
    end
```

Four ideas: (1) collect pixel transitions with a random policy; (2) learn a **position-
bearing** representation with augmentation-VICReg — *no pixels reconstructed*; (3)
**freeze** it, cache the latents, and train a GRU latent-dynamics predictor on those
frozen features; (4) plan in latent space with CEM-MPC against a goal latent and act in
the real env. The detached decoder in block 5 has **no arrow back** into the model.

---

## The honest part: loss is not the acceptance signal

The loop's rule is that **only the real planning success rate counts** — and round 1 is a
case study in why. An earlier design (joint encoder+predictor, EMA target, VICReg) looked
*great* by every loss: rollout MSE 0.002, RankMe 57/64 (no collapse). **Its planner tied
the random baseline (0.12 vs 0.12).** The tell was a single number the loss never shows:

```
corr(latent distance, true spatial distance) = 0.10   # ~zero
```

CEM minimizes latent distance to a goal latent, so if that distance is uncorrelated with
actually getting closer, planning is random — no matter how low the loss. Two fixes got
it to 0.62:

1. **Decouple representation from dynamics.** A GRU predictor is expressive enough to map
   `z_t → z_{t+1}` for *any* encoder geometry, so the prediction loss never forces a
   spatially-meaningful metric. Fix: learn the encoder *first* (augmentation-VICReg),
   **freeze** it, then train the predictor on cached latents. The encoder can't drift.
2. **Small latent_dim.** Position lives in ~2–3 dims; at dim 64 the latent metric is
   dominated by 60+ other unit-variance dims, so `corr ≈ √(2/64) ≈ 0.1`. Shrinking to
   dim 8 lifts corr to ~0.5 while keeping probe R² high.

![latent dim sweep](docs/latent_dim_sweep.png)

There was also an **observation** trap: the default agent is a ~3-pixel dot in a sea of
static background (cross-frame pixel std ≈ 1.8/255). *Every* unsupervised objective —
autoencoder, VICReg, InfoNCE, inverse dynamics — ignored it and modeled the background,
even though a **supervised** probe hits R²=0.999. The floor uses a top-down camera and a
prominent agent ball (dynamics unchanged) so position is a signal objectives can't skip.
Shrinking that ball back down is a great frontier task (distractor-robust representation).

We watch, every round:

| signal | catches | where |
|---|---|---|
| **planning success rate** (vs random) | does it actually work? | `eval.py` |
| **latent↔spatial corr** | the failure a low loss hides | `train.py` |
| **RankMe** effective rank | representation collapse | `metrics.py` |
| **xy linear-probe R²** | is position encoded at all? (diagnostic) | `metrics.py` |

---

## Method

- **Encoder — augmentation-VICReg.** Two photometric views (brightness/contrast/noise) of
  the *same* frame are pulled together (invariance); variance + covariance spread
  different frames and prevent collapse. Output is a small (dim 8) latent. No pixels
  reconstructed.
- **Latent dynamics — frozen-feature predictor.** Encoder is frozen; its latents are
  cached; a `GRUCell` predictor learns `z_{t+1} = z_t + g(z_t, a_t)` (residual / identity
  prior). Manual rollout loop — reliable on MPS and exactly what CEM batch-rolls.
- **Planner — CEM-MPC in latent space.** Sample action sequences, imagine their latent
  rollouts, score by distance to the goal latent, refit the elite Gaussian, execute the
  first action, replan.
- **Goal latent.** Render the agent *placed at the goal* and encode it; within an episode
  the goal marker is fixed, so latent distance isolates agent position.
- **Detached decoder.** Deconv net trained on `stop-grad(latent)` for visualization only.

---

## Install & run

Requires [`uv`](https://docs.astral.sh/uv/) and a Mac (MPS) or any CPU/CUDA box.

```bash
uv sync                                                              # install deps

uv run python scripts/probe_env.py                                  # sanity: render + goal-image trick
uv run python scripts/run_round.py --round 1 --tag r1-floor         # full round (~2-3 min on MPS)
uv run python scripts/run_round.py --round 1 --tag smoke --quick    # fast smoke test
uv run python scripts/sweep_latent_dim.py 4,8,16,32                 # reproduce the latent-dim study
```

Outputs land in `runs/` (heavy data + checkpoints are git-ignored); showcase artifacts
are copied to `docs/`.

---

## Project layout

```
jepa_wm/
  config.py     hyperparameters (small by design)
  envs.py       PointMaze -> top-down RGB, prominent agent ball, goal image
  data.py       random-policy offline collection + windowed sampler
  models.py     Encoder, Predictor (GRUCell rollout), Decoder (detached), VICReg/InfoNCE
  train.py      aug-VICReg encoder + frozen-feature predictor + detached decoder + diagnostics
  planner.py    CEM-MPC in latent space
  eval.py       closed-loop planning eval + random baseline
  metrics.py    RankMe, linear-probe R²
  viz.py        imagined-rollout GIF, performance curves
scripts/
  probe_env.py        PointMaze API / render sanity check
  run_round.py        one full round, end to end
  sweep_latent_dim.py latent-dim vs corr / probe-R² study
  diagnose.py         latent-distance vs spatial-distance diagnostic
```

---

## Roadmap (evolving)

Round 1 is the **floor**, not the goal. Later rounds diverge on **mechanism, not model
size**:

- shrink the agent ball back down → **distractor-robust** representation (the hard,
  open part this floor sidesteps)
- stochastic / variational latent dynamics (multimodal futures, uncertainty)
- SSM / Mamba or tiny-transformer predictor (long-range latent dynamics)
- rollout-horizon **curriculum** (k: 4 → 8 → 16) for long-horizon planning
- **curiosity**-driven self-collected data; harder mazes (UMaze → Medium) needing
  obstacle-aware planning

See `progress.md` for the running log and frontier ladder.

---

## References

- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — JEPA.
- Bardes et al., *VICReg* (2022); Grill et al., *BYOL* (2020) — non-contrastive SSL.
- Garrido et al., *RankMe* (2023) — effective-rank representation-quality metric.
- Finn et al., *Deep Spatial Autoencoders for Visuomotor Learning* (2016) — pixels→state.
- Chua et al., *PETS* (2018); Hafner et al., *PlaNet/Dreamer* — latent dynamics + MPC.

## License

MIT — see [LICENSE](LICENSE).
