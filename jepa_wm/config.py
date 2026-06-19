"""Central config. Small by design — this is a low-compute route on a Mac."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Config:
    # env
    env_id: str = "PointMaze_Open-v3"
    img_size: int = 64
    max_episode_steps: int = 200
    cam_distance: float = 6.5     # top-down camera zoom (ball must be prominent)
    enlarge_agent: bool = True    # floor crutch: big bright ball. False = hard small-ball
    agent_size: float = 0.5       # agent site radius when enlarge_agent

    # data
    n_train_episodes: int = 200
    n_eval_episodes: int = 40
    seed: int = 0

    # model
    latent_dim: int = 8           # small: position lives in ~2-3 dims; large dim
                                  # dilutes the latent metric and breaks CEM planning
    enc_channels: tuple[int, ...] = (32, 64, 64, 128)
    pred_hidden: int = 256
    action_dim: int = 2

    # --- Phase 1: encoder via temporal VICReg (slow-feature representation) ---
    # Positive pairs = frames delta steps apart. Invariance pulls temporally-near
    # frames together so latent distance tracks SPATIAL distance (what planning needs);
    # variance/covariance prevent collapse. This is the fix for the metric-scrambling
    # failure of joint encoder+predictor training (corr(latent,xy)~0 -> planner=random).
    enc_epochs: int = 12
    enc_lr: float = 3e-4
    # encoder architecture + objective (floor defaults; "distractor" preset overrides)
    encoder_type: str = "cnn"          # "cnn" | "ssm" (spatial-softmax keypoints)
    encoder_objective: str = "vicreg"  # "vicreg" | "inverse" (multi-step inverse dyn)
    inverse_k: int = 24                # ACRO horizon: predict a_t from (z_t, z_{t+k})
    inverse_hidden: int = 256
    # Positive pairs = two PHOTOMETRIC augmentations of the SAME frame. Invariance
    # keeps each position's identity; variance-across-batch separates different
    # positions; small latent_dim keeps the metric un-diluted. Background is static
    # and rendering deterministic, so ball position is the only signal to encode.
    vic_inv_coef: float = 25.0
    vic_var_coef: float = 25.0
    vic_cov_coef: float = 1.0
    var_gamma: float = 1.0
    aug_brightness: float = 0.3   # x in [1-b, 1+b]
    aug_contrast: float = 0.3
    aug_noise_std: float = 0.05
    normalize_latent: bool = False

    # --- Phase 2: predictor on FROZEN cached latents ---
    rollout_k: int = 4            # multi-step open-loop horizon supervised in training
    batch_size: int = 256
    pred_epochs: int = 12
    pred_lr: float = 3e-4

    # detached decoder (viz only)
    dec_epochs: int = 8
    dec_lr: float = 3e-4

    # CEM-MPC planner
    plan_horizon: int = 12
    cem_iters: int = 4
    cem_samples: int = 256
    cem_elite_frac: float = 0.1
    cem_init_std: float = 0.5
    success_threshold: float = 0.45  # PointMaze default goal radius

    # bookkeeping
    device: str = "mps"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
