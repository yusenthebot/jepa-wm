"""Decoupled JEPA training (the fix for metric-scrambling):

  Phase 1  encoder via TEMPORAL VICReg  -> latent distance tracks spatial distance
  Phase 2  freeze encoder, cache latents, train predictor on frozen features

No pixels are reconstructed for learning. A detached decoder (train_decoder) is
trained separately on stop-grad latents for visualization only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import Episode, sample_frames_xy
from .metrics import linear_probe_r2, rankme
import torch.nn as nn

from .models import Decoder, Encoder, Predictor, SpatialSoftmaxEncoder, vicreg_loss


def make_encoder(cfg: Config, device: str):
    if cfg.encoder_type == "ssm":
        return SpatialSoftmaxEncoder(cfg.img_size, cfg.latent_dim,
                                     normalize=cfg.normalize_latent).to(device)
    return Encoder(cfg.img_size, cfg.latent_dim, cfg.enc_channels,
                   normalize=cfg.normalize_latent).to(device)


def _photometric_aug(x: torch.Tensor, cfg: Config, rng_t: torch.Generator) -> torch.Tensor:
    """Position-preserving augmentation: brightness, contrast, gaussian noise.
    x: (B,3,H,W) in [0,1]. Returns an augmented copy in [0,1]."""
    b = x.shape[0]
    dev = x.device
    bri = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=rng_t) * 2 - 1) * cfg.aug_brightness
    con = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=rng_t) * 2 - 1) * cfg.aug_contrast
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    out = (x - mean) * con + mean * bri
    out = out + torch.randn(x.shape, device=dev, generator=rng_t) * cfg.aug_noise_std
    return out.clamp(0.0, 1.0)


def pick_device(prefer: str) -> str:
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Phase 1: encoder (temporal VICReg)
# --------------------------------------------------------------------------- #
def train_encoder(episodes: list[Episode], cfg: Config, device: str, log=print):
    """Dispatch to the configured encoder objective."""
    if cfg.encoder_objective == "inverse":
        return train_encoder_inverse(episodes, cfg, device, log)
    return train_encoder_vicreg(episodes, cfg, device, log)


def train_encoder_inverse(episodes: list[Episode], cfg: Config, device: str, log=print):
    """Multi-step inverse dynamics (ACRO-style): predict a_t from (z_t, z_{t+k}).
    To infer the action, the encoder must encode the CONTROLLABLE state (the agent)
    and can ignore the static background — the fix for the small distractor ball.
    Large k gives a bigger displacement = stronger signal. Pairs with spatial-softmax
    keypoints (encoder_type='ssm') for an object-coordinate inductive bias."""
    rng = np.random.default_rng(cfg.seed)
    images = [torch.from_numpy(e.images) for e in episodes]
    acts = [torch.from_numpy(e.actions).float() for e in episodes]
    k = cfg.inverse_k
    idx = np.asarray([(ei, t) for ei, e in enumerate(episodes)
                      for t in range(e.actions.shape[0] - k)])
    encoder = make_encoder(cfg, device)
    inv = nn.Sequential(nn.Linear(2 * cfg.latent_dim, cfg.inverse_hidden), nn.SiLU(),
                        nn.Linear(cfg.inverse_hidden, cfg.action_dim)).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(inv.parameters()), lr=cfg.enc_lr)
    steps = max(1, len(idx) // cfg.batch_size)
    log(f"[enc] multi-step inverse (k={k}, {cfg.encoder_type}) pairs={len(idx)} steps/epoch={steps}")
    for epoch in range(cfg.enc_epochs):
        tot = 0.0
        for _ in range(steps):
            pick = rng.integers(0, len(idx), size=cfg.batch_size)
            o1 = torch.stack([images[ei][t] for ei, t in idx[pick]]).to(device).float() / 255.0
            o2 = torch.stack([images[ei][t + k] for ei, t in idx[pick]]).to(device).float() / 255.0
            a = torch.stack([acts[ei][t] for ei, t in idx[pick]]).to(device)
            pa = inv(torch.cat([encoder(o1), encoder(o2)], dim=-1))
            loss = F.mse_loss(pa, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        log(f"[enc] epoch {epoch+1}/{cfg.enc_epochs} inverse_act_mse={tot/steps:.4f}")
    encoder.eval()
    return encoder


def train_encoder_vicreg(episodes: list[Episode], cfg: Config, device: str, log=print):
    """Augmentation-VICReg: two photometric views of the SAME frame are pulled
    together (invariance), variance/covariance spread DIFFERENT frames apart. The
    only thing varying across frames is ball position, so the encoder encodes it;
    at small latent_dim the latent metric tracks spatial distance."""
    rng = np.random.default_rng(cfg.seed)
    rng_t = torch.Generator(device=device); rng_t.manual_seed(cfg.seed)
    images = [torch.from_numpy(e.images) for e in episodes]  # uint8 (T,3,H,W)
    flat = np.asarray([(ei, t) for ei, e in enumerate(episodes)
                       for t in range(e.images.shape[0])])
    encoder = make_encoder(cfg, device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.enc_lr)
    steps = max(1, len(flat) // cfg.batch_size)
    log(f"[enc] aug-VICReg frames={len(flat)} steps/epoch={steps} device={device}")
    for epoch in range(cfg.enc_epochs):
        agg = {"inv": 0.0, "var": 0.0, "cov": 0.0}
        for _ in range(steps):
            pick = rng.integers(0, len(flat), size=cfg.batch_size)
            x = torch.stack([images[ei][t] for ei, t in flat[pick]]).to(device).float() / 255.0
            z1 = encoder(_photometric_aug(x, cfg, rng_t))
            z2 = encoder(_photometric_aug(x, cfg, rng_t))
            inv, var, cov = vicreg_loss(z1, z2, cfg.var_gamma)
            loss = cfg.vic_inv_coef * inv + cfg.vic_var_coef * var + cfg.vic_cov_coef * cov
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg["inv"] += float(inv); agg["var"] += float(var); agg["cov"] += float(cov)
        log(f"[enc] epoch {epoch+1}/{cfg.enc_epochs} "
            f"inv={agg['inv']/steps:.4f} var={agg['var']/steps:.4f} cov={agg['cov']/steps:.4f}")
    encoder.eval()
    return encoder


@torch.no_grad()
def cache_latents(encoder: Encoder, episodes: list[Episode], device: str) -> list[torch.Tensor]:
    """Encode every frame once with the FROZEN encoder. Phase 2 trains on these."""
    encoder.eval()
    out = []
    for e in episodes:
        imgs = torch.from_numpy(e.images).float().to(device) / 255.0
        z = torch.cat([encoder(imgs[i:i + 256]) for i in range(0, imgs.shape[0], 256)], 0)
        out.append(z.cpu())  # (T, D)
    return out


# --------------------------------------------------------------------------- #
# Phase 2: predictor on frozen latents
# --------------------------------------------------------------------------- #
def train_predictor(latents: list[torch.Tensor], episodes: list[Episode],
                    cfg: Config, device: str, log=print) -> Predictor:
    rng = np.random.default_rng(cfg.seed + 7)
    k = cfg.rollout_k
    actions = [torch.from_numpy(e.actions).float() for e in episodes]
    index = []
    for ei in range(len(episodes)):
        t = actions[ei].shape[0]
        for s in range(0, t - k + 1):
            index.append((ei, s))
    index = np.asarray(index)
    predictor = Predictor(cfg.latent_dim, cfg.action_dim, cfg.pred_hidden).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=cfg.pred_lr)
    steps = max(1, len(index) // cfg.batch_size)
    log(f"[pred] frozen-latent windows={len(index)} steps/epoch={steps}")
    for epoch in range(cfg.pred_epochs):
        tot = 0.0
        for _ in range(steps):
            pick = rng.integers(0, len(index), size=cfg.batch_size)
            zw = torch.stack([latents[ei][s:s + k + 1] for ei, s in index[pick]]).to(device)
            aw = torch.stack([actions[ei][s:s + k] for ei, s in index[pick]]).to(device)
            preds = predictor.rollout(zw[:, 0], aw)
            loss = F.mse_loss(preds, zw[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        log(f"[pred] epoch {epoch+1}/{cfg.pred_epochs} rollout_mse={tot/steps:.5f}")
    predictor.eval()
    return predictor


# --------------------------------------------------------------------------- #
# Orchestrator + diagnostics
# --------------------------------------------------------------------------- #
def train_jepa(episodes: list[Episode], cfg: Config, device: str, log=print):
    encoder = train_encoder(episodes, cfg, device, log)
    latents = cache_latents(encoder, episodes, device)
    predictor = train_predictor(latents, episodes, cfg, device, log)
    diag = _diagnostics(encoder, predictor, latents, episodes, cfg, device)
    return encoder, predictor, None, diag


@torch.no_grad()
def _diagnostics(encoder, predictor, latents, episodes, cfg, device):
    rng = np.random.default_rng(cfg.seed + 1)
    imgs_np, xy_np = sample_frames_xy(episodes, n=2000, rng=rng)
    frames = torch.from_numpy(imgs_np).float().to(device) / 255.0
    z = torch.cat([encoder(frames[i:i + 256]) for i in range(0, frames.shape[0], 256)], 0)
    z_np = z.cpu().numpy()
    rk = rankme(z)
    r2 = linear_probe_r2(z_np, xy_np)
    # the metric that actually predicts planning: latent dist vs spatial dist
    n = z_np.shape[0]
    a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    ld = np.linalg.norm(z_np[a] - z_np[b], axis=1)
    sd = np.linalg.norm(xy_np[a] - xy_np[b], axis=1)
    corr = float(np.corrcoef(ld, sd)[0, 1])

    # multi-step rollout MSE on frozen latents
    k = cfg.rollout_k
    idx = []
    acts = [torch.from_numpy(e.actions).float() for e in episodes]
    for ei in range(len(episodes)):
        for s in range(0, acts[ei].shape[0] - k + 1):
            idx.append((ei, s))
    pick = rng.integers(0, len(idx), size=min(512, len(idx)))
    zw = torch.stack([latents[ei][s:s + k + 1] for ei, s in [idx[i] for i in pick]]).to(device)
    aw = torch.stack([acts[ei][s:s + k] for ei, s in [idx[i] for i in pick]]).to(device)
    preds = predictor.rollout(zw[:, 0], aw)
    rollout_mse = float(F.mse_loss(preds, zw[:, 1:]))
    return {"rankme": rk, "probe_r2": r2, "latent_spatial_corr": corr,
            "rollout_mse": rollout_mse, "latent_dim": cfg.latent_dim}


def train_decoder(encoder: Encoder, episodes: list[Episode], cfg: Config, device: str,
                  log=print) -> Decoder:
    """Detached decoder for visualization. Trained on stop-grad(latent) ONLY."""
    rng = np.random.default_rng(cfg.seed + 3)
    images = [torch.from_numpy(e.images) for e in episodes]
    flat = [(ei, t) for ei, e in enumerate(episodes) for t in range(e.images.shape[0])]
    flat = np.asarray(flat)
    decoder = Decoder(cfg.img_size, cfg.latent_dim).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.dec_lr)
    encoder.eval()
    steps = max(1, len(flat) // cfg.batch_size)
    for epoch in range(cfg.dec_epochs):
        tot = 0.0
        for _ in range(steps):
            pick = rng.integers(0, len(flat), size=cfg.batch_size)
            x = torch.stack([images[ei][t] for ei, t in flat[pick]]).to(device).float() / 255.0
            with torch.no_grad():
                zc = encoder(x).detach()  # stop-grad
            recon = decoder(zc)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        log(f"[decoder] epoch {epoch+1}/{cfg.dec_epochs} mse={tot/steps:.4f}")
    decoder.eval()
    return decoder
