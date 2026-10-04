"""Uniform router: w_i = 1/N everywhere (the paper's default)."""

from __future__ import annotations

from torch import Tensor

from .base import Router


class UniformRouter(Router):
    def weight_outputs(self, x_proj: Tensor, y: Tensor, **context) -> Tensor:
        return y.mean(dim=1)
