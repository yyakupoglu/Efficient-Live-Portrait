#!/usr/bin/env python3
"""
Convert every *.onnx in input_dir to TensorRT engines (FP16 and FP32), same naming as
run_webcam_stream.py expects under output_dir (e.g. *_fp16.engine / *_fp32.engine).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run_trtexec(onnx_path: Path, engine_out: Path, extra_flags: list[str] | None = None) -> bool:
    """Run trtexec and return True on success. Prints last 30 lines of output on failure."""
    if engine_out.exists():
        print(f"  skip (exists): {engine_out.name}", flush=True)
        return True

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
        "--skipInference",
    ] + (extra_flags or [])

    print(f"  building {engine_out.name}...", end="", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(" OK", flush=True)
        return True

    # Print only the useful tail of trtexec output (skip the massive help text)
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    lines = combined.strip().splitlines()
    tail = "\n".join(lines[-30:])
    print(f" FAILED (exit {result.returncode})", flush=True)
    print(f"  ── trtexec output (last 30 lines) ──\n{tail}\n  ──────────────────────────────────", flush=True)
    return False


def build_engines_from_onnx(input_dir: Path | str, output_dir: Path | str) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".onnx")
    if not onnx_files:
        print(f"No .onnx files in {input_dir}", file=sys.stderr)
        raise FileNotFoundError(input_dir)

    failed = []
    for onnx_path in onnx_files:
        stem = onnx_path.stem
        fp16_out = output_dir / f"{stem}_fp16.engine"
        fp32_out = output_dir / f"{stem}_fp32.engine"
        print(f"Processing {stem}...", flush=True)

        if not _run_trtexec(onnx_path, fp16_out, ["--fp16"]):
            failed.append(f"{stem} (FP16)")
        if not _run_trtexec(onnx_path, fp32_out):
            failed.append(f"{stem} (FP32)")

    if failed:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"FAILED to build {len(failed)} engine(s):", file=sys.stderr)
        for f in failed:
            print(f"  ✗ {f}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent
    default_in = repo / "live_portrait_weights" / "live_portrait"
    default_out = repo / "live_portrait_weights" / "live_portrait" / "10.14"
    parser.add_argument("--input_dir", type=str, default=str(default_in))
    parser.add_argument("--output_dir", type=str, default=str(default_out))
    args = parser.parse_args()
    build_engines_from_onnx(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
