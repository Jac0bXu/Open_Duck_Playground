import onnxruntime
paths = [
    "BEST_WALK_ONNX_2.onnx",
    "/workspace/checkpoints/00_baseline/2026_05_21_152518_151388160.onnx",
]
for path in paths:
    sess = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        print(f"{path}: input {inp.name} shape={inp.shape}")
    for out in sess.get_outputs():
        print(f"  output {out.name} shape={out.shape}")
