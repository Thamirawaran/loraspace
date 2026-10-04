"""LoRASpace hybrid routing over the 48 LoRA-Retriever LLaMA-2-7B task LoRAs.

INSTRUCTOR retrieval proposes 5 candidate adapters per query, then LoRASpace
mixes the top 3 per token inside the shared subspace. Needs
``pip install -e .[llm]`` and access to meta-llama/Llama-2-7b-hf.

    python examples/llm_hybrid.py --query "Is this review positive? The film was a delight."
"""

import argparse
import json
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from loraspace.llm import InstructorRetriever, LoRASpaceLLM, compute_adapter_norms, precompute_llm_pool

# Pool definition and example inputs from LoRA-Retriever (Zhao et al., 2024).
CONFIG = "https://raw.githubusercontent.com/StyxXuan/LoraRetriever/main/config/config2.json"

ap = argparse.ArgumentParser()
ap.add_argument("--query", default="Is this review positive or negative? The film was a delight from start to end.")
ap.add_argument("--base", default="meta-llama/Llama-2-7b-hf")
args = ap.parse_args()

pool = json.load(urllib.request.urlopen(CONFIG))
names = [p["model_name"] for p in pool]
samples = {p["model_name"]: [s["inputs"] for s in p["sample"]] for p in pool}

cache = precompute_llm_pool({n: f"Styxxxx/llama2_7b_lora-{n}" for n in names})
print("pool summary:", cache.summary())

tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to("cuda").eval()

# Router calibration on the pool's example inputs (not on test data).
norms = compute_adapter_norms(model, tok, cache, [s for n in names for s in samples[n][:20]])

ls = LoRASpaceLLM(model, tok, cache, retriever=InstructorRetriever(samples), adapter_norms=norms)
print(ls.generate(args.query, max_new_tokens=32))
