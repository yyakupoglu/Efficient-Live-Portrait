"""
Patch generator_fix_grid.onnx: rename 5D GridSample ops to GridSample3D
so TRT's ONNX parser delegates to the custom plugin instead of
trying (and failing) to handle them natively.
"""
import onnx
import sys

from pathlib import Path

_REPO = Path(__file__).resolve().parent
INPUT  = str(_REPO / "live_portrait_weights" / "live_portrait" / "generator_fix_grid.onnx")
OUTPUT = str(_REPO / "live_portrait_weights" / "live_portrait" / "generator_fix_grid_3d.onnx")

model = onnx.load(INPUT)

count = 0
for node in model.graph.node:
    if node.op_type == "GridSample":
        node.op_type = "GridSample3D"
        # Set domain to empty so TRT looks up the plugin by name
        node.domain = ""
        count += 1
        print(f"  Patched node: {node.name}")

print(f"\nPatched {count} GridSample -> GridSample3D nodes")
onnx.save(model, OUTPUT)
print(f"Saved to: {OUTPUT}")
