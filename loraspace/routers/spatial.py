"""Spatial router: per-token top-k routing by normalised activation energy (SpectR-style)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Router


class SpatialRouter(Router):
    """Score s_i = ||y_i||^2 / ||x_proj||^2 per token; softmax over the top-k adapters.

    With ``step_schedule`` the temperature is T * (2 - step_fraction): a broad mix
    early in denoising and sharper routing late.
    """

    def __init__(self, num_adapters: int, top_k: int = 1, temperature: float = 1.0, step_schedule: bool = True):
        super().__init__(num_adapters)
        self.top_k = min(top_k, num_adapters)
        self.temperature = temperature
        self.step_schedule = step_schedule
        self._total_steps: Optional[int] = None
        self._step_frac = 0.0

    def set_total_steps(self, total: int) -> None:
        self._total_steps = total

    def step_update(self, step_idx: int, latents: Optional[Tensor] = None) -> None:
        if self._total_steps:
            self._step_frac = min(1.0, step_idx / max(self._total_steps, 1))

    def weight_outputs(self, x_proj: Tensor, y: Tensor, **context) -> Tensor:
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        eps = 1e-4 if x_proj.dtype == torch.float16 else 1e-8
        y_sq = (y ** 2).sum(dim=-1)  # (B, N, S)
        x_sq = (x_proj ** 2).sum(dim=-1).clamp(min=eps)  # (B, S)
        scores = torch.nan_to_num(y_sq / x_sq.unsqueeze(1), nan=0.0, posinf=0.0, neginf=0.0)

        t_eff = self.temperature * (2.0 - self._step_frac) if self.step_schedule else self.temperature
        topk_scores, topk_idx = scores.topk(self.top_k, dim=1)
        topk_routes = F.softmax(topk_scores / t_eff, dim=1)
        topk_routes = torch.nan_to_num(topk_routes, nan=1.0 / self.top_k, posinf=1.0 / self.top_k, neginf=0.0)

        routes = torch.zeros_like(scores)
        routes.scatter_(1, topk_idx, topk_routes)
        return torch.einsum("bns,bnsd->bsd", routes, y)
