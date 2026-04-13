#!/usr/bin/env python3
"""
Download LivePortrait ONNX weights (live_portrait + insightface) from Hugging Face.
Skips any file that already exists. Does not download TensorRT engines or the grid plugin
— build the plugin locally and run trtexec_all.py (see prepare_webcam_pipeline.py).
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF_ONNX_BASE = "https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main"

LIVE_PORTRAIT_ONNX = [
    "appearance_feature_extractor.onnx",
    "motion_extractor.onnx",
    "generator_fix_grid.onnx",
    "stitching.onnx",
    "stitching_eye.onnx",
    "stitching_lip.onnx",
]

INSIGHTFACE_ONNX = [
    "w600k_r50.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "landmark.onnx",
]


def _hf_url(filename: str) -> str:
    return f"{HF_ONNX_BASE}/{filename}?download=true"


from tqdm import tqdm

def _download_one(url: str, dest: Path) -> bool:
    """
    Download url to dest with a tqdm progress bar.
    Verified against Content-Length to handle truncated files.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    req = urllib.request.Request(url, method='HEAD', headers={"User-Agent": "Efficient-Live-Portrait/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            expected_size = int(resp.getheader('Content-Length', 0))
    except Exception:
        expected_size = 0

    if dest.exists():
        actual_size = dest.stat().st_size
        # The patched generator_fix_grid.onnx is slightly different in size (+8 bytes)
        # than the Huggingface original. We allow a small tolerance to prevent 
        # redownloading the patched file, while still catching truncated downloads.
        if expected_size > 0 and abs(actual_size - expected_size) <= 128:
            print(f"  ✓ {dest.name} (already exists, size matches)")
            return False
        elif expected_size > 0:
            print(f"  ↺ {dest.name} (size mismatch: local {actual_size} vs remote {expected_size}, redownloading)")
        else:
            print(f"  ✓ {dest.name} (already exists)")
            return False

    print(f"  ↓ {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Efficient-Live-Portrait/1.0"})
    
    with urllib.request.urlopen(req, timeout=600) as resp:
        total_size = int(resp.getheader('Content-Length', 0))
        block_size = 1024 * 64
        
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"    {dest.name}", leave=False) as bar:
            with open(dest, 'wb') as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    return True


def download_onnx_weights(weights_root: Path | str | None = None) -> None:
    """
    Download ONNX models into:
      {weights_root}/live_portrait/*.onnx
      {weights_root}/insightface/*.onnx
    """
    root = Path(weights_root) if weights_root is not None else Path.cwd() / "live_portrait_weights"
    root = root.resolve()

    print(f"ONNX download root: {root}")

    for name in LIVE_PORTRAIT_ONNX:
        dest = root / "live_portrait" / name
        try:
            _download_one(_hf_url(name), dest)
        except (urllib.error.URLError, OSError) as e:
            print(f"ERROR downloading {name}: {e}", file=sys.stderr)
            raise

    for name in INSIGHTFACE_ONNX:
        dest = root / "insightface" / name
        try:
            _download_one(_hf_url(name), dest)
        except (urllib.error.URLError, OSError) as e:
            print(f"ERROR downloading {name}: {e}", file=sys.stderr)
            raise

    print("ONNX download step finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-root",
        type=str,
        default=None,
        help="Directory containing live_portrait/ and insightface/ (default: ./live_portrait_weights from cwd)",
    )
    args = parser.parse_args()
    wr = Path(args.weights_root).resolve() if args.weights_root else None
    download_onnx_weights(wr)


if __name__ == "__main__":
    main()
