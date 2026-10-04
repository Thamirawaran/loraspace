"""The LoRASpace composition operator (paper Eq. 5) and its two deployment modes.

    delta(x) = U ( sum_i w_i B_hat_i A_hat_i ) V^T x

Dynamic mode wraps each target layer and evaluates the operator in the
D-dimensional subspace at every forward pass, with weights from a router.
Static mode folds the inner sum once, M = sum_i w_i B_hat_i A_hat_i, and adds
the single ambient delta U M V^T to the base weights (merge cost at inference).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .cache import LayerCache, PoolCache
from .routers.base import Router

logger = logging.getLogger(__name__)

KeyFn = Callable[[str], str]


def subspace_delta(x: Tensor, layer_cache: LayerCache, router: Router, **context: Any) -> Tensor:
    """LoRASpace delta at one layer.

    Args:
        x: layer input, (B, S, d_in) for linear or (B, d_in, H, W) for conv1x1.
        layer_cache: U, V and projected factors of the active adapters.
        router: combines the per-adapter subspace outputs.
    """
    is_spatial = x.ndim == 4
    if is_spatial:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
    else:
        x_flat = x

    V = layer_cache.bases.V
    U = layer_cache.bases.U
    x_proj = x_flat.to(V.dtype) @ V  # (B, S, D), shared by all adapters
    z = torch.einsum("nrd,bsd->bnsr", layer_cache.A_hat_all, x_proj)  # (B, N, S, r)
    y = torch.einsum("ndr,bnsr->bnsd", layer_cache.B_hat_all, z)  # (B, N, S, D)
    y_mix = router.weight_outputs(x_proj, y, **context)  # (B, S, D)
    delta = y_mix @ U.T  # (B, S, d_out), shared back-projection

    if is_spatial:
        delta = delta.reshape(B, H, W, delta.shape[-1]).permute(0, 3, 1, 2)
    return delta.to(x.dtype)


class LoRASpaceLayer(nn.Module):
    """Wraps a target Linear / Conv2d and adds the scaled dynamic-mode delta."""

    def __init__(self, wrapped: nn.Module, layer_cache: LayerCache, router: Router,
                 scale: float = 1.0, layer_key: str = "") -> None:
        super().__init__()
        self.add_module("wrapped", wrapped)
        # Plain attributes, so the cache tensors are not registered as parameters.
        object.__setattr__(self, "_ls_layer_cache", layer_cache)
        object.__setattr__(self, "_ls_router", router)
        object.__setattr__(self, "_ls_scale", float(scale))
        object.__setattr__(self, "_ls_layer_key", str(layer_key))

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        out = self.wrapped(x, *args, **kwargs)
        delta = subspace_delta(x, self._ls_layer_cache, self._ls_router, layer_key=self._ls_layer_key)
        return out + delta * self._ls_scale

    @property
    def weight(self):
        return getattr(self.wrapped, "weight", None)

    @property
    def bias(self):
        return getattr(self.wrapped, "bias", None)

    def extra_repr(self) -> str:
        return f"scale={self._ls_scale}, router={type(self._ls_router).__name__}"


# ---------------------------------------------------------------------------
# Finding target layers
# ---------------------------------------------------------------------------


def key_sd15(name: str) -> str:
    """SD 1.5 cache keys are kohya-style: no ``unet.`` prefix, dots -> underscores."""
    return (name[5:] if name.startswith("unet.") else name).replace(".", "_")


def key_sdxl(name: str) -> str:
    """SDXL cache keys are diffusers module names without the ``unet.`` prefix."""
    return name[5:] if name.startswith("unet.") else name


def key_llm(name: str) -> str:
    """LLM cache keys are module names without a PEFT ``base_model.model.`` prefix."""
    for p in ("base_model.model.", "base_model."):
        if name.startswith(p):
            return name[len(p):]
    return name


def find_target_modules(root: nn.Module, cache: PoolCache, key_fn: KeyFn) -> Dict[str, Tuple[nn.Module, str, nn.Module]]:
    """Return ``{layer_key: (parent, attr, module)}`` for every cached Linear/Conv2d in ``root``."""
    cache_keys = set(cache.layer_caches)
    parents: Dict[str, Tuple[nn.Module, str]] = {}
    for parent_name, parent in root.named_modules():
        for child_name, _ in parent.named_children():
            parents[f"{parent_name}.{child_name}" if parent_name else child_name] = (parent, child_name)

    targets: Dict[str, Tuple[nn.Module, str, nn.Module]] = {}
    for name, module in root.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        key = key_fn(name)
        if key in cache_keys and name in parents:
            parent, attr = parents[name]
            targets[key] = (parent, attr, module)
    return targets


def _move_layer_cache(lc: LayerCache, device: torch.device) -> None:
    lc.bases.U = lc.bases.U.to(device)
    lc.bases.V = lc.bases.V.to(device)
    lc.A_hat_all = lc.A_hat_all.to(device)
    lc.B_hat_all = lc.B_hat_all.to(device)


# ---------------------------------------------------------------------------
# Dynamic mode
# ---------------------------------------------------------------------------

InjectionRecord = List[Tuple[nn.Module, str, nn.Module]]


def inject_dynamic(root: nn.Module, cache: PoolCache, router: Router, key_fn: KeyFn,
                   scale: float = 1.0) -> InjectionRecord:
    """Replace each cached layer of ``root`` with a ``LoRASpaceLayer``. Undo with ``restore``."""
    targets = find_target_modules(root, cache, key_fn)
    if not targets:
        raise RuntimeError(f"0 modules matched {len(cache.layer_caches)} cache keys; wrong key_fn?")
    device = next(root.parameters()).device
    record: InjectionRecord = []
    for key, (parent, attr, original) in targets.items():
        lc = cache.layer_caches[key]
        _move_layer_cache(lc, device)
        setattr(parent, attr, LoRASpaceLayer(original, lc, router, scale=scale, layer_key=key))
        record.append((parent, attr, original))
    logger.debug(f"Injected {len(record)} LoRASpace layers (router={type(router).__name__})")
    return record


def restore(record: InjectionRecord) -> None:
    for parent, attr, original in record:
        setattr(parent, attr, original)


# ---------------------------------------------------------------------------
# Static mode
# ---------------------------------------------------------------------------

FuseRecord = List[Tuple[nn.Module, Tensor]]


def static_delta(lc: LayerCache, weights: Sequence[float], device=None) -> Tensor:
    """Fold the inner sum: U (sum_i w_i B_hat_i A_hat_i) V^T, shape (d_out, d_in), fp32."""
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    B_hat = lc.B_hat_all.to(device=device, dtype=torch.float32)  # (N, D, r)
    A_hat = lc.A_hat_all.to(device=device, dtype=torch.float32)  # (N, r, D)
    M = torch.einsum("n,ndr,nrk->dk", w, B_hat, A_hat)  # (D, D)
    U = lc.bases.U.to(device=device, dtype=torch.float32)
    V = lc.bases.V.to(device=device, dtype=torch.float32)
    return U @ M @ V.T


@torch.no_grad()
def fuse_static(root: nn.Module, cache: PoolCache, key_fn: KeyFn, weights: Optional[Sequence[float]] = None,
                scale: float = 1.0) -> FuseRecord:
    """Add ``scale * U M V^T`` into the weights of every cached layer of ``root``.

    ``weights`` default to uniform 1/N. Original weights are kept on CPU so
    ``unfuse_static`` restores them exactly.
    """
    n = len(cache.adapter_names)
    weights = list(weights) if weights is not None else [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    targets = find_target_modules(root, cache, key_fn)
    if not targets:
        raise RuntimeError(f"0 modules matched {len(cache.layer_caches)} cache keys; wrong key_fn?")
    record: FuseRecord = []
    for key, (_parent, _attr, module) in targets.items():
        W = module.weight
        delta = static_delta(cache.layer_caches[key], weights, device=W.device) * scale
        if isinstance(module, nn.Conv2d):
            delta = delta.unsqueeze(-1).unsqueeze(-1)
        record.append((module, W.detach().to("cpu", copy=True)))
        W.copy_((W.float() + delta).to(W.dtype))
    logger.debug(f"Fused static LoRASpace delta into {len(record)} layers")
    return record


@torch.no_grad()
def unfuse_static(record: FuseRecord) -> None:
    for module, original in record:
        module.weight.copy_(original.to(module.weight.device))
