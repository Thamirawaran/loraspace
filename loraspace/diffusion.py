"""LoRASpace for diffusers text-to-image pipelines (SD 1.5 and SDXL)."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .cache import PoolCache
from .operator import FuseRecord, InjectionRecord, fuse_static, inject_dynamic, key_sd15, key_sdxl, restore, unfuse_static
from .routers import Router

logger = logging.getLogger(__name__)


class LoRASpacePipeline:
    """Compose adapters from a precomputed pool onto a diffusers pipeline.

    Example::

        pipe = StableDiffusionXLPipeline.from_pretrained(...).to("cuda")
        ls = LoRASpacePipeline(pipe, PoolCache.load("pool_c.pt"))
        ls.compose(["lighting_3", "material_2"])                      # static mode
        ls.compose(["lighting_3", "material_2"], router=SpatialRouter(2, top_k=1))  # dynamic mode
        image = ls("a rusted robot in a dim workshop", num_inference_steps=50).images[0]

    The pipeline must not have other LoRAs loaded. ``reset()`` restores the
    original weights and modules.
    """

    def __init__(self, pipeline, cache: PoolCache, scale: float = 0.8):
        self.pipeline = pipeline
        self.cache = cache
        self.scale = scale
        arch = cache.metadata.get("arch", "sd15")
        self._key_fn = key_sdxl if arch == "sdxl" else key_sd15
        self.combo: Optional[List[str]] = None
        self.router: Optional[Router] = None
        self._fused: Optional[FuseRecord] = None
        self._injected: Optional[InjectionRecord] = None

    def compose(self, adapters: Sequence[str], weights: Optional[Sequence[float]] = None,
                router: Optional[Router] = None) -> None:
        """Activate ``adapters``.

        Without a router this is static mode: ``U (sum_i w_i B_hat_i A_hat_i) V^T`` is
        folded into the UNet weights once (``weights`` default to 1/N), so
        inference costs the same as the base model. With a router this is
        dynamic mode: the operator runs in the subspace at every UNet pass with
        router-chosen weights.
        """
        self.reset()
        self.combo = list(adapters)
        subset = self.cache.subset(self.combo)
        if router is None:
            self._fused = fuse_static(self.pipeline.unet, subset, self._key_fn, weights=weights, scale=self.scale)
        else:
            if weights is not None:
                raise ValueError("weights apply to static mode; a router sets weights in dynamic mode")
            if router.N != len(self.combo):
                raise ValueError(f"router built for {router.N} adapters, composing {len(self.combo)}")
            self.router = router
            self._injected = inject_dynamic(self.pipeline.unet, subset, router, self._key_fn, scale=self.scale)

    def reset(self) -> None:
        if self._fused is not None:
            unfuse_static(self._fused)
            self._fused = None
        if self._injected is not None:
            restore(self._injected)
            self._injected = None
        self.router = None
        self.combo = None

    def _on_step_end(self, pipeline, step_index: int, timestep, callback_kwargs: dict):
        self.router.step_update(step_index, callback_kwargs.get("latents"))
        return callback_kwargs

    def __call__(self, *args, **kwargs):
        if self.combo is None:
            raise RuntimeError("call compose() first")
        if self.router is not None:
            self.router.reset()
            self.router.set_total_steps(kwargs.get("num_inference_steps", 50))
            if "callback_on_step_end" in kwargs:
                raise ValueError("callback_on_step_end is used by the router in dynamic mode")
            kwargs["callback_on_step_end"] = self._on_step_end
        return self.pipeline(*args, **kwargs)
