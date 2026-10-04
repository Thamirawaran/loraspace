"""Pool-wide shared subspace (the LoRASpace substrate).

For a pool of K LoRAs with factors (B_i, A_i) and scaling s_i = alpha_i / r_i,
each layer gets shared bases U, V from paired marginal SVDs of the
Frobenius-normalised factor stacks, truncated at energy fraction tau, and
every LoRA is re-parameterised inside that subspace (paper Sec. 3.2-3.3):

    B_hat_i = sqrt(s_i) * U^T B_i      (D, r_i)
    A_hat_i = sqrt(s_i) * A_i V        (r_i, D)

The cache is computed once per pool and reused for any combination of
adapters drawn from it, in static or dynamic mode.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

Triple = Tuple[Tensor, Tensor, float]  # (B, A, scaling)


@dataclass
class SubspaceBases:
    """Shared (U, V) bases for one layer."""

    U: Tensor  # (d_out, D)
    V: Tensor  # (d_in, D)
    D: int
    energy: float


@dataclass
class LayerCache:
    """Shared bases plus the projected factors of every pool adapter at one layer."""

    bases: SubspaceBases
    A_hat_all: Tensor  # (K, r_max, D), zero-padded where r_i < r_max
    B_hat_all: Tensor  # (K, D, r_max)
    layer_key: str
    is_conv: bool = False


@dataclass
class PoolCache:
    """All layer caches of a pool, with adapter metadata."""

    adapter_names: List[str]
    adapter_to_idx: Dict[str, int]
    layer_caches: Dict[str, LayerCache] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def subset(self, names: Iterable[str]) -> "PoolCache":
        """Slice the cache to the adapters in ``names`` (bases are shared, not rebuilt)."""
        names = list(names)
        missing = [n for n in names if n not in self.adapter_to_idx]
        if missing:
            raise KeyError(f"adapters not in pool: {missing}")
        idx = [self.adapter_to_idx[n] for n in names]
        layer_caches: Dict[str, LayerCache] = {}
        for key, lc in self.layer_caches.items():
            A_sub = lc.A_hat_all[idx]
            B_sub = lc.B_hat_all[idx]
            # Drop rank padding that no adapter in the subset uses.
            r_actual = A_sub.shape[1]
            for r in range(A_sub.shape[1] - 1, 0, -1):
                if A_sub[:, r, :].abs().max() > 1e-7:
                    r_actual = r + 1
                    break
            layer_caches[key] = LayerCache(
                bases=lc.bases,
                A_hat_all=A_sub[:, :r_actual, :].contiguous(),
                B_hat_all=B_sub[:, :, :r_actual].contiguous(),
                layer_key=key,
                is_conv=lc.is_conv,
            )
        return PoolCache(
            adapter_names=names,
            adapter_to_idx={n: i for i, n in enumerate(names)},
            layer_caches=layer_caches,
            metadata=self.metadata,
        )

    def summary(self) -> Dict:
        """Subspace size and rank fill: rho = mean D / sum_i r_i (paper Sec. 4.2)."""
        dims = [lc.bases.D for lc in self.layer_caches.values()]
        ranks = self.metadata.get("adapter_ranks", {})
        sum_r = sum(ranks.get(n, 0) for n in self.adapter_names)
        d_mean = sum(dims) / len(dims) if dims else 0.0
        return {
            "K": len(self.adapter_names),
            "layers": len(dims),
            "tau": self.metadata.get("energy_threshold"),
            "D_mean": d_mean,
            "D_min": min(dims) if dims else 0,
            "D_max": max(dims) if dims else 0,
            "sum_r": sum_r,
            "rho": d_mean / sum_r if sum_r else float("nan"),
        }

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "adapter_names": self.adapter_names,
                "adapter_to_idx": self.adapter_to_idx,
                "metadata": self.metadata,
                "layers": {
                    key: {
                        "U": lc.bases.U,
                        "V": lc.bases.V,
                        "D": lc.bases.D,
                        "energy": lc.bases.energy,
                        "A_hat_all": lc.A_hat_all,
                        "B_hat_all": lc.B_hat_all,
                        "is_conv": lc.is_conv,
                    }
                    for key, lc in self.layer_caches.items()
                },
            },
            str(path),
        )
        logger.info(f"Cache saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PoolCache":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        layer_caches = {
            key: LayerCache(
                bases=SubspaceBases(U=ld["U"], V=ld["V"], D=ld["D"], energy=ld["energy"]),
                A_hat_all=ld["A_hat_all"],
                B_hat_all=ld["B_hat_all"],
                layer_key=key,
                is_conv=ld.get("is_conv", False),
            )
            for key, ld in data["layers"].items()
        }
        return cls(
            adapter_names=data["adapter_names"],
            adapter_to_idx=data["adapter_to_idx"],
            layer_caches=layer_caches,
            metadata=data["metadata"],
        )


# ---------------------------------------------------------------------------
# Per-layer subspace construction (shared by diffusion and LLM pools)
# ---------------------------------------------------------------------------


def _subspace_dim(singular_values: Tensor, threshold: float) -> int:
    """Smallest D whose cumulative singular-value energy reaches ``threshold``."""
    if singular_values.numel() == 0:
        return 1
    energy = singular_values ** 2
    total = energy.sum()
    if total < 1e-12 or not torch.isfinite(total):
        return 1
    cum = torch.cumsum(energy, dim=0) / total
    idx = (cum >= threshold).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return int(singular_values.numel())
    return int(idx[0].item()) + 1


def build_layer_cache(
    layer_key: str,
    present: Dict[str, Triple],
    adapter_names: List[str],
    energy_threshold: float = 0.99,
    cache_dtype: torch.dtype = torch.float16,
) -> Optional[LayerCache]:
    """Build one layer's cache from the adapters that train this layer.

    ``present`` maps adapter name -> (B, A, scaling) for adapters that have this
    layer. Adapters without it get a zero placeholder, so row i of the cache
    always corresponds to pool index i.
    """
    if not present:
        return None
    # Some community LoRAs leave whole blocks untrained: no signal to decompose.
    if all(A.abs().sum() == 0 or B.abs().sum() == 0 for (B, A, _s) in present.values()):
        return None

    sample_B, sample_A, _ = next(iter(present.values()))
    d_out, d_in = sample_B.shape[0], sample_A.shape[1]

    Bs: List[Tensor] = []
    As: List[Tensor] = []
    scalings: List[float] = []
    for name in adapter_names:
        if name in present:
            B, A, s = present[name]
        else:
            B = torch.zeros(d_out, 1, dtype=sample_B.dtype)
            A = torch.zeros(1, d_in, dtype=sample_A.dtype)
            s = 0.0
        Bs.append(B)
        As.append(A)
        scalings.append(s)

    ranks = [A.shape[0] for A in As]
    r_max = max(ranks)

    # Frobenius normalisation so no high-norm LoRA dominates the bases (Eq. 2).
    Bs_n = [B_i / torch.norm(B_i, "fro").clamp(min=1e-8) for B_i in Bs]
    As_n = [A_i / torch.norm(A_i, "fro").clamp(min=1e-8) for A_i in As]

    # Paired marginal SVDs with energy truncation (Eq. 3).
    U_full, s_B, _ = torch.linalg.svd(torch.cat(Bs_n, dim=1), full_matrices=False)
    D_B = _subspace_dim(s_B, energy_threshold)
    _, s_A, Vh_full = torch.linalg.svd(torch.cat(As_n, dim=0), full_matrices=False)
    D_A = _subspace_dim(s_A, energy_threshold)

    D = min(max(min(D_B, U_full.shape[1]), min(D_A, Vh_full.shape[0])), sum(ranks), d_in, d_out)

    if U_full.shape[1] < D:
        U = torch.cat([U_full, torch.zeros(d_out, D - U_full.shape[1], dtype=U_full.dtype)], dim=1)
    else:
        U = U_full[:, :D]
    if Vh_full.shape[0] < D:
        V_part = Vh_full.T
        V = torch.cat([V_part, torch.zeros(d_in, D - Vh_full.shape[0], dtype=V_part.dtype)], dim=1)
    else:
        V = Vh_full[:D, :].T

    eB = (s_B[: min(D_B, len(s_B))] ** 2).sum() / (s_B ** 2).sum().clamp(min=1e-12)
    eA = (s_A[: min(D_A, len(s_A))] ** 2).sum() / (s_A ** 2).sum().clamp(min=1e-12)
    energy = float(min(eB.item(), eA.item()))

    # Projected factors with the scaling split symmetrically (Eq. 4).
    A_hat_list, B_hat_list = [], []
    for B_i, A_i, s_i, r_i in zip(Bs, As, scalings, ranks):
        sqrt_s = s_i ** 0.5
        B_hat_i = sqrt_s * (U.T @ B_i)  # (D, r_i)
        A_hat_i = sqrt_s * (A_i @ V)  # (r_i, D)
        if r_i < r_max:
            B_hat_p = torch.zeros(D, r_max, dtype=B_hat_i.dtype)
            B_hat_p[:, :r_i] = B_hat_i
            A_hat_p = torch.zeros(r_max, D, dtype=A_hat_i.dtype)
            A_hat_p[:r_i, :] = A_hat_i
        else:
            B_hat_p, A_hat_p = B_hat_i, A_hat_i
        B_hat_list.append(B_hat_p)
        A_hat_list.append(A_hat_p)

    return LayerCache(
        bases=SubspaceBases(U=U.to(cache_dtype), V=V.to(cache_dtype), D=D, energy=energy),
        A_hat_all=torch.stack(A_hat_list).to(cache_dtype),
        B_hat_all=torch.stack(B_hat_list).to(cache_dtype),
        layer_key=layer_key,
        is_conv="proj_in" in layer_key or "proj_out" in layer_key,
    )


# ---------------------------------------------------------------------------
# Diffusion pools (kohya-format .safetensors LoRAs)
# ---------------------------------------------------------------------------

_PREFIX = "lora_unet_"


def sdxl_key_mapping() -> Dict[str, str]:
    """kohya -> diffusers layer names for the SDXL UNet."""
    text = resources.files("loraspace").joinpath("data/sdxl_key_mapping.json").read_text()
    return json.loads(text)


def _detect_arch(paths: Iterable[str]) -> str:
    from safetensors import safe_open

    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(_PREFIX + "input_blocks") or k.startswith(_PREFIX + "output_blocks"):
                    return "sdxl"
                if k.startswith(_PREFIX + "down_blocks") or k.startswith(_PREFIX + "up_blocks"):
                    return "sd15"
    raise ValueError("could not detect UNet architecture from LoRA keys")


def _scan_layer_index(
    adapter_paths: Dict[str, str], key_mapping: Optional[Dict[str, str]]
) -> Tuple[List[str], Dict[str, Dict[str, Tuple[str, str, str]]]]:
    """Map each adapter's layers to (down, up, alpha) keys without loading tensors."""
    from safetensors import safe_open

    layer_keys = set()
    index: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    for name, path in adapter_paths.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
        per: Dict[str, Tuple[str, str, str]] = {}
        for dk in sorted(keys):
            if not (dk.startswith(_PREFIX) and ".lora_down." in dk):
                continue
            key = dk.split(".lora_down.")[0].replace(_PREFIX, "")
            uk = dk.replace(".lora_down.", ".lora_up.")
            if uk not in keys:
                continue
            mapped = key_mapping.get(key, key) if key_mapping else key
            per[mapped] = (dk, uk, _PREFIX + key + ".alpha")
            layer_keys.add(mapped)
        index[name] = per
    return sorted(layer_keys), index


def _fetch_layer(path: str, dk: str, uk: str, ak: str) -> Optional[Triple]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        A = f.get_tensor(dk).float()
        B = f.get_tensor(uk).float()
        if A.ndim > 2:  # conv1x1 -> 2D
            A = A.squeeze(-1).squeeze(-1)
            B = B.squeeze(-1).squeeze(-1)
        if A.ndim > 2 or B.ndim > 2:  # conv3x3 cannot be absorbed by the 2D subspace
            return None
        r = A.shape[0]
        alpha = f.get_tensor(ak).float().item() if ak in keys else float(r)
        return B, A, alpha / r


def precompute_pool(
    adapters: Dict[str, Union[str, Path]],
    energy_threshold: float = 0.99,
    cache_dtype: torch.dtype = torch.float16,
) -> PoolCache:
    """Build the LoRASpace cache for a pool of kohya-format diffusion LoRAs.

    Args:
        adapters: ordered mapping adapter name -> path of its .safetensors file.
            SD 1.5 and SDXL LoRAs are supported (detected from the keys).
        energy_threshold: tau, the singular-value energy kept per layer (paper: 0.99).
        cache_dtype: storage dtype of U, V and the projected factors.

    Layers are processed one at a time with lazy reads, so peak memory is one
    layer of the pool, not the whole pool.
    """
    t0 = time.time()
    names = list(adapters)
    paths = {n: str(adapters[n]) for n in names}
    arch = _detect_arch(paths.values())
    key_mapping = sdxl_key_mapping() if arch == "sdxl" else None

    layer_keys, index = _scan_layer_index(paths, key_mapping)
    logger.info(f"Precompute: {len(names)} {arch} adapters, {len(layer_keys)} layer keys")

    layer_caches: Dict[str, LayerCache] = {}
    adapter_ranks: Dict[str, int] = {n: 0 for n in names}
    for i, lk in enumerate(layer_keys):
        present: Dict[str, Triple] = {}
        for name in names:
            keys = index[name].get(lk)
            if keys is None:
                continue
            triple = _fetch_layer(paths[name], *keys)
            if triple is not None:
                present[name] = triple
                adapter_ranks[name] = max(adapter_ranks[name], triple[1].shape[0])
        lc = build_layer_cache(lk, present, names, energy_threshold, cache_dtype)
        if lc is not None:
            layer_caches[lk] = lc
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(layer_keys)} layers")

    cache = PoolCache(
        adapter_names=names,
        adapter_to_idx={n: i for i, n in enumerate(names)},
        layer_caches=layer_caches,
        metadata={
            "arch": arch,
            "energy_threshold": energy_threshold,
            "cache_dtype": str(cache_dtype),
            "adapter_ranks": adapter_ranks,
            "precompute_seconds": time.time() - t0,
        },
    )
    s = cache.summary()
    logger.info(
        f"Precompute done in {time.time() - t0:.0f}s: {s['layers']} layers, "
        f"D mean {s['D_mean']:.0f} (min {s['D_min']}, max {s['D_max']}), rho {s['rho']:.2f}"
    )
    return cache
