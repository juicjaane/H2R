"""
Downloads the Depth Anything V2 Metric (Hypersim indoor, vits) checkpoint.
Run once before any pipeline usage:
    python scripts/download_metric_checkpoint.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DEPTH_METRIC_CKPT

URL = (
    "https://huggingface.co/depth-anything/"
    "Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/"
    "depth_anything_v2_metric_hypersim_vits.pth?download=true"
)


def download():
    if DEPTH_METRIC_CKPT.exists():
        size_mb = DEPTH_METRIC_CKPT.stat().st_size / 1e6
        print(f"Checkpoint already exists ({size_mb:.1f} MB): {DEPTH_METRIC_CKPT}")
        return

    print(f"Downloading metric checkpoint to {DEPTH_METRIC_CKPT} ...")
    DEPTH_METRIC_CKPT.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        resp = requests.get(URL, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(DEPTH_METRIC_CKPT, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="", flush=True)
        print(f"\nDone. Saved to {DEPTH_METRIC_CKPT}")
    except ImportError:
        # Fall back to urllib
        import urllib.request

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r  {pct:.1f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(URL, DEPTH_METRIC_CKPT, reporthook=progress)
        print(f"\nDone. Saved to {DEPTH_METRIC_CKPT}")


if __name__ == "__main__":
    download()
