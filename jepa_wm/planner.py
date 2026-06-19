"""CEM-MPC planning in latent space.

Given the current latent z0 and a goal latent z_goal, search action sequences
whose imagined latent rollout ends nearest the goal. Execute the first action,
then replan (receding horizon).
"""
from __future__ import annotations

import torch

from .models import Predictor


class CEMPlanner:
    def __init__(
        self,
        predictor: Predictor,
        horizon: int = 12,
        iters: int = 4,
        samples: int = 256,
        elite_frac: float = 0.1,
        init_std: float = 0.5,
        action_dim: int = 2,
        device: str = "cpu",
        cost_fn=None,
    ):
        self.pred = predictor
        self.h = horizon
        self.iters = iters
        self.samples = samples
        self.n_elite = max(1, int(elite_frac * samples))
        self.init_std = init_std
        self.action_dim = action_dim
        self.device = device
        # cost_fn(preds (N,H,D), z_goal (D,)) -> (N,). Default: Euclidean latent distance.
        # Round 4 passes a learned temporal-distance (quasimetric) here for obstacle mazes.
        self.cost_fn = cost_fn

    @torch.no_grad()
    def _cost(self, z0: torch.Tensor, actions: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """actions: (N, H, A). Cost = terminal dist + small mean-progress term."""
        n = actions.shape[0]
        z0n = z0.unsqueeze(0).expand(n, -1)
        preds = self.pred.rollout(z0n, actions)          # (N, H, D)
        if self.cost_fn is not None:
            return self.cost_fn(preds, z_goal)
        d = torch.linalg.norm(preds - z_goal.unsqueeze(0).unsqueeze(0), dim=-1)  # (N, H)
        terminal = d[:, -1]
        mean_progress = d.mean(dim=1)
        return terminal + 0.25 * mean_progress

    @torch.no_grad()
    def plan(self, z0: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Return the first action of the optimized sequence. Shape (action_dim,)."""
        dev = self.device
        mu = torch.zeros(self.h, self.action_dim, device=dev)
        std = torch.full((self.h, self.action_dim), self.init_std, device=dev)
        for _ in range(self.iters):
            eps = torch.randn(self.samples, self.h, self.action_dim, device=dev)
            actions = (mu.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(-1.0, 1.0)
            costs = self._cost(z0, actions, z_goal)
            elite_idx = torch.topk(-costs, self.n_elite).indices
            elites = actions[elite_idx]
            mu = elites.mean(dim=0)
            std = elites.std(dim=0) + 1e-4
        return mu[0].clamp(-1.0, 1.0)
