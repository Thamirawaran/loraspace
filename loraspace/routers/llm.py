"""LLM router: per-token top-k routing with per-(layer, adapter) magnitude normalisation."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Router


class LLMRouter(Router):
    """Score s_i = ||y_i|| / E_x[||y_i(x)||] at each layer; softmax over the top-k adapters.

    Args:
        adapter_names: adapters in cache order.
        adapter_norms: ``{layer_key: Tensor[N]}`` mean output norms
            (``loraspace.llm.compute_adapter_norms``). Layers without norms are scored unnormalised.
        top_k: adapters mixed per token.
        temperature: softmax temperature.
    """

    def __init__(self, adapter_names: List[str], adapter_norms: Optional[Dict[str, Tensor]] = None,
                 top_k: int = 3, temperature: float = 1.0):
        super().__init__(len(adapter_names))
        self.adapter_names = list(adapter_names)
        self.adapter_norms = dict(adapter_norms) if adapter_norms else {}
        self.top_k = min(top_k, self.N)
        self.temperature = float(temperature)

    def weight_outputs(self, x_proj: Tensor, y: Tensor, **context) -> Tensor:
        y_norm = torch.linalg.vector_norm(torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        norms = self.adapter_norms.get(context.get("layer_key", ""))
        if norms is not None:
            norms = norms.to(device=y_norm.device, dtype=y_norm.dtype)
            scores = y_norm / norms.view(1, -1, 1).clamp(min=1e-6)
        else:
            scores = y_norm
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        topk_scores, topk_idx = scores.topk(self.top_k, dim=1)
        topk_routes = F.softmax(topk_scores / max(self.temperature, 1e-6), dim=1)
        topk_routes = torch.nan_to_num(topk_routes, nan=1.0 / self.top_k, posinf=1.0 / self.top_k, neginf=0.0)
        routes = torch.zeros_like(scores)
        routes.scatter_(1, topk_idx, topk_routes)
        return torch.einsum("bns,bnsd->bsd", routes, y)
