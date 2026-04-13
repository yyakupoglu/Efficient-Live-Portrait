#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Idempotent setup + launch for Efficient-Live-Portrait
#
# Runs INSIDE the container with the project bind-mounted.
# Python deps and the GridSample3D plugin are already baked into the image.
# This script handles everything that depends on the mounted volume / GPU.
#
# Usage (from compose):  command: ["./run.sh"]
# Or interactively:      bash run.sh [--setup-only]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${SCRIPT_DIR}"
WEIGHTS="${REPO}/live_portrait_weights"
LP_WEIGHTS="${WEIGHTS}/live_portrait"
IF_WEIGHTS="${WEIGHTS}/insightface"
ENGINE_DIR="${LP_WEIGHTS}/10.14"
HF="https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main"

# Plugin .so is baked into the image by the Dockerfile
PLUGIN="${GRID_SAMPLE_3D_PLUGIN:-/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so}"

echo "══════════════════════════════════════════════════════════════"
echo "  Efficient-Live-Portrait — Idempotent Setup"
echo "  REPO: ${REPO}"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Download / verify all ONNX weights ──────────────────────────────────
# Always run the download script — it skips files that already exist and match
# the expected size, and re-downloads any truncated files automatically.
mkdir -p "${LP_WEIGHTS}" "${IF_WEIGHTS}" "${ENGINE_DIR}"

echo "▸ [1/4] Checking ONNX weights..."
python download_live_portrait_weights.py --weights-root "${WEIGHTS}"

# ── 2. Patch generator ONNX: GridSample → GridSample3D ─────────────────────
# Uses a sentinel file to track completion (the _3d.onnx output is moved
# over the original, so we can't use its existence as a check).
if [ ! -f "${LP_WEIGHTS}/.generator_patched" ]; then
    echo "▸ [2/4] Patching generator ONNX (GridSample → GridSample3D)..."
    cd "${REPO}"
    python patch_onnx_gridsample.py
    mv "${LP_WEIGHTS}/generator_fix_grid_3d.onnx" \
       "${LP_WEIGHTS}/generator_fix_grid.onnx"
    touch "${LP_WEIGHTS}/.generator_patched"
else
    echo "✓ [2/4] Generator ONNX already patched"
fi

# ── 4. Build TensorRT engines ──────────────────────────────────────────────
# trtexec_all.py writes *_fp16.engine and *_fp32.engine into ENGINE_DIR.
# Skip if at least one .engine file exists (engines are GPU-specific).
ENGINE_COUNT=$(find "${ENGINE_DIR}" -maxdepth 1 -name '*.engine' 2>/dev/null | wc -l)
if [ "${ENGINE_COUNT}" -eq 0 ]; then
    echo "▸ [3/4] Building TensorRT engines (this takes a while)..."
    cd "${REPO}"
    LD_PRELOAD="${PLUGIN}" python trtexec_all.py
else
    echo "✓ [3/4] TensorRT engines present (${ENGINE_COUNT} files)"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════════════════════════"

# ── Launch or stop ──────────────────────────────────────────────────────────
if [ "${1:-}" = "--setup-only" ]; then
    echo "  --setup-only passed, exiting."
    exit 0
fi

echo ""
echo "▸ [4/4] Launching run_webcam_stream.py --mixed on port 8890..."
cd "${REPO}"
export LD_PRELOAD="${PLUGIN}"
exec python run_webcam_stream.py --mixed "$@"
