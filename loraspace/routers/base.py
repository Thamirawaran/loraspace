"""Router interface: how per-adapter subspace outputs are combined at each layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


class Router(ABC):
    """Base class for LoRASpace routers.

    Lifecycle per generation:
        router.reset()
        for each denoising step:
            router.step_update(step, latents)
            at each layer: router.weight_outputs(x_proj, y) -> (B, S, D)
    """

    def __init__(self, num_adapters: int):
        self.N = num_adapters

    def reset(self) -> None:
        """Reset per-generation state."""

    def set_total_steps(self, total: int) -> None:
        """Called before generation with the number of denoising steps."""

    def step_update(self, step_idx: int, latents: Optional[Tensor] = None) -> None:
        """Called before each denoising step."""

    @abstractmethod
    def weight_outputs(self, x_proj: Tensor, y: Tensor, **context) -> Tensor:
        """Combine per-adapter outputs.

        Args:
            x_proj: (B, S, D) input projected into the shared subspace.
            y: (B, N, S, D) per-adapter subspace outputs.
            context: extras such as ``layer_key``.

        Returns:
            (B, S, D) weighted combination.
        """
