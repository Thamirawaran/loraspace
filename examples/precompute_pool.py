"""Build the LoRASpace cache for a pool and print its subspace size and rank fill.

    python pools/download_adapters.py --pool pool_c
    python examples/precompute_pool.py --pool pool_c --out pool_c.pt
"""

import argparse
import json
import logging
from pathlib import Path

from loraspace import precompute_pool

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="pool_c", choices=["pool_a", "pool_b", "pool_c"])
ap.add_argument("--adapters", default=None, help="adapter directory (default: adapters/<pool>)")
ap.add_argument("--tau", type=float, default=0.99, help="energy threshold")
ap.add_argument("--out", default=None, help="where to save the cache (.pt)")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")

manifest = json.loads((ROOT / "pools" / f"{args.pool}.json").read_text())
adapter_dir = Path(args.adapters or f"adapters/{args.pool}")
adapters = {}
for entries in manifest.values():
    for e in entries:
        path = adapter_dir / f"{e['id']}.safetensors"
        if path.exists():
            adapters[e["id"]] = path
        else:
            print(f"missing {e['id']} ({e['distribution']}), building the pool without it")

cache = precompute_pool(adapters, energy_threshold=args.tau)
s = cache.summary()
print(f"\nK={s['K']}  layers={s['layers']}  D mean={s['D_mean']:.0f} (min {s['D_min']}, max {s['D_max']})")
print(f"sum of ranks={s['sum_r']}  rank fill rho={s['rho']:.2f}  (rho=1 means no overlap)")
if args.out:
    cache.save(args.out)
