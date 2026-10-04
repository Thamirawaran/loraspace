"""Download the adapters of a LoRASpace pool and verify their sha256.

Hosted adapters come from the Hugging Face dataset Thamirawaran/loraspace-pools.
Link-only adapters come from their Civitai pages (set CIVITAI_TOKEN if Civitai
asks for a login). Unavailable adapters are skipped.

    python pools/download_adapters.py --pool pool_c --out adapters/pool_c
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hosted(entry: dict, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    src = Path(hf_hub_download(entry["hf_repo"], entry["hf_path"], repo_type="dataset")).resolve()
    dest.unlink(missing_ok=True)
    try:
        dest.symlink_to(src)  # no second copy of the file
    except OSError:
        shutil.copyfile(src, dest)


def fetch_civitai(entry: dict, dest: Path) -> None:
    url = f"https://civitai.com/api/download/models/{entry['civitai_version_id']}"
    token = os.environ.get("CIVITAI_TOKEN")
    if token:
        url += f"?token={token}"
    req = urllib.request.Request(url, headers={"User-Agent": "loraspace-downloader"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, choices=["pool_a", "pool_b", "pool_c"])
    ap.add_argument("--out", default=None, help="output directory (default: adapters/<pool>)")
    ap.add_argument("--hosted-only", action="store_true", help="skip link-only adapters")
    args = ap.parse_args()

    pool = json.loads((HERE / f"{args.pool}.json").read_text())
    out = Path(args.out or f"adapters/{args.pool}")
    out.mkdir(parents=True, exist_ok=True)

    failed = []
    for entries in pool.values():
        for e in entries:
            dest = out / f"{e['id']}.safetensors"
            if e["distribution"] == "unavailable":
                print(f"skip     {e['id']:<14} unavailable (removed from source, not redistributed)")
                continue
            if e["distribution"] == "link-only" and args.hosted_only:
                print(f"skip     {e['id']:<14} link-only")
                continue
            if dest.exists() and sha256(dest) == e["sha256"]:
                print(f"ok       {e['id']:<14} cached")
                continue
            try:
                (fetch_hosted if e["distribution"] == "hosted" else fetch_civitai)(e, dest)
            except Exception as exc:  # network or auth failure: report and continue
                print(f"FAILED   {e['id']:<14} {exc}")
                failed.append(e["id"])
                continue
            if sha256(dest) != e["sha256"]:
                dest.unlink()
                print(f"FAILED   {e['id']:<14} sha256 mismatch")
                failed.append(e["id"])
            else:
                print(f"ok       {e['id']:<14} {e['distribution']}")
    if failed:
        print(f"\n{len(failed)} adapters failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
