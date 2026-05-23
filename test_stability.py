"""Test: Can the robot stand still with default actuators (no policy)?"""
import numpy as np
import mujoco
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

model_path = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
base = MJInferBase(model_path)

# Just set ctrl to default and step
base.data.qpos[:] = base.model.keyframe("home").qpos
base.data.ctrl[:] = base.default_actuator
mujoco.mj_forward(base.model, base.data)

# Check solver settings
print(f"Timestep: {base.model.opt.timestep}")
print(f"Solver iterations: {base.model.opt.iterations}")
print(f"LS iterations: {base.model.opt.ls_iterations}")
print(f"Disable flags: {base.model.opt.disableflags}")

# Step for 5000 physics steps (10 seconds at 0.002s)
for step in range(5000):
    mujoco.mj_step(base.model, base.data)

    # Check gravity every 500 steps
    if step % 500 == 0:
        gravity = base.get_gravity(base.data)
        z = base.data.qpos[2]
        print(f"Step {step}: gravity_z={gravity[2]:.4f}, z_height={z:.4f}")
        if gravity[2] < 0:
            print("  -> FALLEN!")
            break

print(f"\nFinal z_height: {base.data.qpos[2]:.4f}")
print(f"Final gravity: {base.get_gravity(base.data)}")
