"""Temporal router: per-step dominant-adapter schedule (CMLoRA-style) driven by a subspace signal."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from .base import Router


def _interval_seq(total: int) -> List[int]:
    """Steps at which the dominant adapter advances (CMLoRA caching schedule)."""
    b1, b2 = int(total * 0.375), int(total * 0.875)
    seq = []
    for s in range(total):
        if s < b1 and s % 2 == 0:
            seq.append(s)
        elif b1 <= s < b2 and s % 3 == 0:
            seq.append(s)
        elif s >= b2 and s % 2 == 0:
            seq.append(s)
    return seq or [0]


class TemporalRouter(Router):
    """One dominant adapter per step, cycling through the combo over denoising.

    The frequency signal m is the late-band energy ratio of ``x_proj`` in the
    shared subspace. While m >= g (high-frequency phase) dominance cycles
    through foreground adapters; below g it cycles through background adapters
    (names containing "background").
    """

    def __init__(self, adapter_names: List[str], total_steps: Optional[int] = None, g: float = 0.40):
        super().__init__(len(adapter_names))
        self.adapter_names = list(adapter_names)
        self.g = g
        self.real_list = [n for n in self.adapter_names if "background" not in n.lower()]
        self.abs_list = [n for n in self.adapter_names if "background" in n.lower()]
        self.name_to_idx = {n: i for i, n in enumerate(self.real_list + self.abs_list)}

        init_coeff = self.N - 0.5
        self._nondom_weight = self.N / (init_coeff + self.N - 1)
        self._dom_weight = self._nondom_weight * init_coeff

        self.interval_seq = _interval_seq(total_steps) if total_steps else [0]
        self.reset()

    def set_total_steps(self, total: int) -> None:
        self.interval_seq = _interval_seq(total)

    def reset(self) -> None:
        self.indicator = 0
        self.m = 1.0
        self._weights = torch.ones(self.N) / self.N
        self._needs_recompute = True

    def _weights_from_m(self, m: float) -> Tensor:
        n_real, n_abs = len(self.real_list), len(self.abs_list)
        tick = max(self.indicator, 1) - 1
        if n_real == 0 and n_abs == 0:
            dominant_idx = 0
        elif not self.abs_list or m >= self.g:
            dominant_idx = tick % max(n_real, 1)
        else:
            dominant_idx = n_real + (tick % max(n_abs, 1))

        ordered = torch.full((self.N,), self._nondom_weight)
        ordered[dominant_idx] = self._dom_weight
        w = torch.tensor([ordered[self.name_to_idx[n]] for n in self.adapter_names])
        return w / w.sum().clamp(min=1e-8)

    def step_update(self, step_idx: int, latents: Optional[Tensor] = None) -> None:
        if step_idx in self.interval_seq:
            self.indicator += 1
        self._needs_recompute = True

    def weight_outputs(self, x_proj: Tensor, y: Tensor, **context) -> Tensor:
        if self._needs_recompute:
            D = x_proj.shape[-1]
            split = D // 2 if D >= 2 else 0
            energy = x_proj.detach().float().pow(2).sum(dim=(0, 1))  # (D,)
            self.m = 1.0 if split == 0 else (energy[split:].sum() / energy.sum().clamp(min=1e-12)).item()
            self._weights = self._weights_from_m(self.m)
            self._needs_recompute = False
        w = self._weights.to(device=y.device, dtype=y.dtype)
        return torch.einsum("n,bnsd->bsd", w, y)
