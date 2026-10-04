"""LoRASpace for causal LMs with PEFT LoRA adapters.

The hybrid policy (paper Sec. 3.6 and App. B) pairs an external retriever with
LoRASpace: the retriever proposes the top-K candidate adapters for a prompt,
and ``LLMRouter`` mixes the top-k of those per token inside the shared subspace.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .cache import LayerCache, PoolCache, Triple, build_layer_cache
from .operator import find_target_modules, inject_dynamic, key_llm, restore
from .routers import LLMRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


def _resolve(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, allow_patterns=["adapter_model.*", "adapter_config.json"]))


def _load_peft(path: Path) -> Tuple[Dict[str, Tensor], dict]:
    from safetensors.torch import load_file

    cfg = json.loads((path / "adapter_config.json").read_text())
    if (path / "adapter_model.safetensors").exists():
        return load_file(str(path / "adapter_model.safetensors"), device="cpu"), cfg
    return torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True), cfg


def precompute_llm_pool(
    adapters: Dict[str, str],
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    num_layers: int = 32,
    energy_threshold: float = 0.99,
    cache_dtype: torch.dtype = torch.bfloat16,
) -> PoolCache:
    """Build the LoRASpace cache for a pool of PEFT LoRAs on a LLaMA-style model.

    Args:
        adapters: ordered mapping name -> local adapter directory or HF repo id.
        target_modules: attention projections the adapters train.
        num_layers: number of transformer blocks.
    """
    names = list(adapters)
    per_adapter: Dict[str, Dict[str, Triple]] = {}
    adapter_ranks: Dict[str, int] = {}
    for name in names:
        state, cfg = _load_peft(_resolve(adapters[name]))
        r, alpha = cfg["r"], cfg["lora_alpha"]
        adapter_ranks[name] = r
        layers: Dict[str, Triple] = {}
        for li in range(num_layers):
            for mod in target_modules:
                key = f"model.layers.{li}.self_attn.{mod}"
                B = state[f"base_model.model.{key}.lora_B.weight"].float()
                A = state[f"base_model.model.{key}.lora_A.weight"].float()
                layers[key] = (B, A, alpha / r)
        per_adapter[name] = layers

    layer_caches: Dict[str, LayerCache] = {}
    for key in sorted(next(iter(per_adapter.values()))):
        present = {n: per_adapter[n][key] for n in names if key in per_adapter[n]}
        lc = build_layer_cache(key, present, names, energy_threshold, cache_dtype)
        if lc is not None:
            layer_caches[key] = lc

    cache = PoolCache(
        adapter_names=names,
        adapter_to_idx={n: i for i, n in enumerate(names)},
        layer_caches=layer_caches,
        metadata={
            "arch": "llm",
            "energy_threshold": energy_threshold,
            "cache_dtype": str(cache_dtype),
            "adapter_ranks": adapter_ranks,
            "target_modules": list(target_modules),
        },
    )
    s = cache.summary()
    logger.info(f"LLM precompute: {s['layers']} layers, D mean {s['D_mean']:.0f}, rho {s['rho']:.2f}")
    return cache


# ---------------------------------------------------------------------------
# Router calibration: per-(layer, adapter) mean output norms
# ---------------------------------------------------------------------------


class _NormAccumulator:
    def __init__(self, lc: LayerCache, device):
        self.V = lc.bases.V.to(device)
        self.A_hat = lc.A_hat_all.to(device)
        self.B_hat = lc.B_hat_all.to(device)
        self.sum = torch.zeros(lc.A_hat_all.shape[0], dtype=torch.float32, device=device)
        self.count = 0

    def __call__(self, module, inp, out):
        x_proj = inp[0].to(self.V.dtype) @ self.V
        z = torch.einsum("nrd,bsd->bnsr", self.A_hat, x_proj)
        y = torch.einsum("ndr,bnsr->bnsd", self.B_hat, z)
        y_norm = torch.linalg.vector_norm(y, dim=-1)  # (B, N, S)
        self.sum += y_norm.float().sum(dim=(0, 2))
        self.count += y_norm.shape[0] * y_norm.shape[2]


@torch.no_grad()
def compute_adapter_norms(model: nn.Module, tokenizer, cache: PoolCache, prompts: Sequence[str],
                          max_length: int = 512) -> Dict[str, Tensor]:
    """Mean subspace output norm E_x[||y_i(x)||] per layer and adapter, over ``prompts``.

    Use prompts that represent each adapter's task (never the test set).
    """
    device = next(model.parameters()).device
    accs, handles = {}, []
    for key, (_p, _a, module) in find_target_modules(model, cache, key_llm).items():
        accs[key] = _NormAccumulator(cache.layer_caches[key], device)
        handles.append(module.register_forward_hook(accs[key]))
    model.eval()
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return {k: (a.sum / a.count).cpu() for k, a in accs.items() if a.count}


# ---------------------------------------------------------------------------
# Retrieval (LoraRetriever's INSTRUCTOR embedder)
# ---------------------------------------------------------------------------

_INSTRUCTION = "Represent the sentence for similar task retrieval: "


class InstructorRetriever:
    """Top-K adapter retrieval by INSTRUCTOR mean embeddings (Zhao et al., 2024).

    ``pool_samples`` maps adapter name -> a few example inputs of its task.
    Needs ``pip install loraspace[llm]``.
    """

    def __init__(self, pool_samples: Dict[str, Sequence[str]], embedder: str = "Styxxxx/lora_retriever"):
        import faiss
        import numpy as np
        from InstructorEmbedding import INSTRUCTOR

        self._np = np
        self.names = list(pool_samples)
        self.embedder = INSTRUCTOR(embedder)
        means = [np.mean(self.embedder.encode([[_INSTRUCTION, s] for s in pool_samples[n]]), axis=0)
                 for n in self.names]
        mat = np.vstack(means).astype(np.float32)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)

    def __call__(self, query: str, k: int = 5) -> List[str]:
        emb = self.embedder.encode([[_INSTRUCTION, query]])
        _, idx = self.index.search(self._np.array(emb), k)
        return [self.names[i] for i in idx[0]]


# ---------------------------------------------------------------------------
# Hybrid composition
# ---------------------------------------------------------------------------


class LoRASpaceLLM:
    """Retrieve candidate adapters per prompt, then route per token inside the shared subspace.

    Defaults are the paper's LoRA-Retriever configuration: 5 retrieved
    candidates, top-3 per token, softmax temperature 5.0, scale 1.0.
    """

    def __init__(self, model: nn.Module, tokenizer, cache: PoolCache,
                 retriever: Optional[Callable[[str, int], List[str]]] = None,
                 adapter_norms: Optional[Dict[str, Tensor]] = None,
                 num_candidates: int = 5, top_k: int = 3, temperature: float = 5.0, scale: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.retriever = retriever
        self.adapter_norms = adapter_norms
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.temperature = temperature
        self.scale = scale

    def _subset_norms(self, combo: List[str]) -> Optional[Dict[str, Tensor]]:
        if self.adapter_norms is None:
            return None
        idx = torch.tensor([self.cache.adapter_to_idx[n] for n in combo], dtype=torch.long)
        return {k: v[idx].clone() for k, v in self.adapter_norms.items()}

    @torch.no_grad()
    def generate(self, prompt: str, adapters: Optional[Sequence[str]] = None, query: Optional[str] = None,
                 **gen_kwargs) -> str:
        """Generate for ``prompt``.

        ``adapters`` fixes the candidate set. Otherwise the retriever proposes
        ``num_candidates`` adapters for ``query`` (defaults to ``prompt``).
        """
        if adapters is None:
            if self.retriever is None:
                raise ValueError("pass adapters= or construct with a retriever")
            adapters = self.retriever(query or prompt, self.num_candidates)
        combo = list(adapters)
        router = LLMRouter(combo, self._subset_norms(combo), top_k=min(self.top_k, len(combo)),
                           temperature=self.temperature)
        record = inject_dynamic(self.model, self.cache.subset(combo), router, key_llm, scale=self.scale)
        try:
            gen_kwargs.setdefault("max_new_tokens", 256)
            gen_kwargs.setdefault("do_sample", False)
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**enc, **gen_kwargs)
        finally:
            restore(record)
        return self.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
