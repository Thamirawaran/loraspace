"""Semantic router: per-region routing from per-adapter spatial masks.

Masks are supplied by the user, one (H, W) map per adapter, for example the
cross-attention concept masks of FreeFuse (Liu et al., 2025) or any
segmentation. On image-token layers each position mixes adapters by its mask
values. On text-token layers (cross-attention K/V), where masks are undefined,
it falls back to softmax over normalised activation energy.
"""

from __future__ import annotations

from math import isqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Router


def _is_spatial_token_axis(S: int) -> bool:
    """Square image-token grids have integer sqrt; CLIP text (S=77) does not."""
    if S < 64:
        return False
    root = isqrt(S)
    return root * root == S


class SemanticRouter(Router):
    def __init__(self, num_adapters: int, masks: Optional[Tensor] = None):
        super().__init__(num_adapters)
        self._masks: Optional[Tensor] = None
        if masks is not None:
            self.set_masks(masks)

    def set_masks(self, masks: Tensor) -> None:
        """(N, H, W) masks in [0, 1], ordered like the composed adapters."""
        if masks.ndim != 3 or masks.shape[0] != self.N:
            raise ValueError(f"expected masks of shape ({self.N}, H, W), got {tuple(masks.shape)}")
        self._masks = masks
        self._mask_hw = masks.shape[-2:]

    def _mask_routes(self, S: int, device, dtype) -> Tensor:
        masks = self._masks.to(device=device, dtype=dtype)  # (N, H, W)
        flat = masks.reshape(self.N, -1)
        if flat.shape[1] != S:
            root = isqrt(S)
            flat = F.interpolate(masks.unsqueeze(0), size=(root, root), mode="bilinear",
                                 align_corners=False).reshape(self.N, -1)
        # Positions no adapter claims get uniform routes (avoids fp16 denormals).
        degenerate = flat.sum(dim=0, keepdim=True) < 1e-4
        flat = torch.where(degenerate.expand_as(flat), torch.full_like(flat, 1.0 / self.N), flat)
        routes = flat / flat.sum(dim=0, keepdim=True).clamp(min=1e-6)
        return torch.nan_to_num(routes, nan=1.0 / self.N, posinf=1.0 / self.N, neginf=0.0)

    def _energy_routes(self, x_proj: Tensor, y: Tensor) -> Tensor:
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        eps = 1e-4 if x_proj.dtype == torch.float16 else 1e-8
        scores = (y ** 2).sum(dim=-1) / (x_proj ** 2).sum(dim=-1).clamp(min=eps).unsqueeze(1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        routes = F.softmax(scores, dim=1)
        return torch.nan_to_num(routes, nan=1.0 / self.N, posinf=1.0 / self.N, neginf=0.0)

    def weight_outputs(self, x_proj: Tensor, y: Tensor, **context) -> Tensor:
        if self._masks is None:
            raise RuntimeError("SemanticRouter needs set_masks() before inference")
        B, N, S, D = y.shape
        if _is_spatial_token_axis(S):
            routes = self._mask_routes(S, y.device, y.dtype).unsqueeze(0).expand(B, -1, -1)
        else:
            routes = self._energy_routes(x_proj, y)
        return torch.einsum("bns,bnsd->bsd", routes, y)
