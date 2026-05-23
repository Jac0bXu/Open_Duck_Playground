"""Debug script to compare eval obs with training obs."""
import numpy as np
import mujoco
from playground.open_duck_mini_v2.eval_headless import HeadlessEval

onnx_path = "/workspace/checkpoints/00_baseline/2026_05_21_152518_151388160.onnx"
evaluator = HeadlessEval(
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    onnx_path,
)
evaluator.reset()

# Print initial state
print("gravity:", evaluator.get_gravity(evaluator.data))
print("qpos[:7]:", evaluator.data.qpos[:7])
print("default_actuator:", evaluator.default_actuator)
print("num_dofs:", evaluator.num_dofs)

# Get initial obs
obs = evaluator.get_obs(evaluator.data, np.zeros(7))
print("obs shape:", obs.shape)
print("obs:", obs)
print("obs range:", obs.min(), obs.max())

# Get action
action = evaluator.policy.infer(obs)
print("action:", action)
print("action range:", action.min(), action.max())

# Step 10 physics steps and check
for i in range(10):
    mujoco.mj_step(evaluator.model, evaluator.data)
obs2 = evaluator.get_obs(evaluator.data, np.zeros(7))
print("obs after 10 steps:", obs2)
action2 = evaluator.policy.infer(obs2)
print("action after 10 steps:", action2)
print("gravity after 10 steps:", evaluator.get_gravity(evaluator.data))

# Check ONNX model input/output shapes
import onnxruntime
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
for inp in session.get_inputs():
    print(f"ONNX input: {inp.name}, shape: {inp.shape}, type: {inp.type}")
for out in session.get_outputs():
    print(f"ONNX output: {out.name}, shape: {out.shape}, type: {out.type}")
