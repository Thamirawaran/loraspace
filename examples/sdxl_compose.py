"""Compose Pool C SDXL LoRAs with LoRASpace in static and dynamic mode.

    python examples/sdxl_compose.py --cache pool_c.pt
"""

import argparse

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

from loraspace import LoRASpacePipeline, PoolCache, SpatialRouter

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="pool_c.pt")
ap.add_argument("--adapters", nargs="+", default=["character_1", "material_2", "lighting_3"])
ap.add_argument("--prompt", default="CamelliaXL, a young woman, detailed face, sitting at an old wooden table, "
                                    "wooden planks, chiaroscuro style, dim light")
args = ap.parse_args()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)

ls = LoRASpacePipeline(pipe, PoolCache.load(args.cache), scale=0.8)
gen = dict(prompt=args.prompt, num_inference_steps=50, guidance_scale=5.0, height=1024, width=1024)

# Static mode: one folded delta per layer, base-model inference cost.
ls.compose(args.adapters)
ls(**gen, generator=torch.manual_seed(111)).images[0].save("static.png")

# Dynamic mode: per-token routing inside the shared subspace at every UNet pass.
ls.compose(args.adapters, router=SpatialRouter(len(args.adapters), top_k=1))
ls(**gen, generator=torch.manual_seed(111)).images[0].save("dynamic_spatial.png")

ls.reset()
print("saved static.png and dynamic_spatial.png")
