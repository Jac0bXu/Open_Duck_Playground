"""Minimal headless test using mujoco_infer.py's exact logic."""
import numpy as np
import mujoco
from playground.open_duck_mini_v2.mujoco_infer import MjInfer

onnx_path = "/workspace/checkpoints/00_baseline/2026_05_21_152518_151388160.onnx"
mjinfer = MjInfer(
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    onnx_path,
    standing=False,
)

# Run the exact same loop as mujoco_infer.py but headless
mjinfer.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # stand
counter = 0
survival = 0
max_policy_steps = 50

for _ in range(max_policy_steps * 10):  # 10x for decimation
    mujoco.mj_step(mjinfer.model, mjinfer.data)
    counter += 1

    if counter % mjinfer.decimation == 0:
        survival += 1

        # Update imitation phase (same as mujoco_infer.py)
        mjinfer.imitation_i += 1.0
        mjinfer.imitation_i = mjinfer.imitation_i % mjinfer.PRM.nb_steps_in_period
        mjinfer.imitation_phase = np.array([
            np.cos(mjinfer.imitation_i / mjinfer.PRM.nb_steps_in_period * 2 * np.pi),
            np.sin(mjinfer.imitation_i / mjinfer.PRM.nb_steps_in_period * 2 * np.pi),
        ])

        obs = mjinfer.get_obs(mjinfer.data, mjinfer.commands)
        action = mjinfer.policy.infer(obs)

        mjinfer.last_last_last_action = mjinfer.last_last_action.copy()
        mjinfer.last_last_action = mjinfer.last_action.copy()
        mjinfer.last_action = action.copy()

        mjinfer.motor_targets = mjinfer.default_actuator + action * mjinfer.action_scale

        if True:  # USE_MOTOR_SPEED_LIMITS
            mjinfer.motor_targets = np.clip(
                mjinfer.motor_targets,
                mjinfer.prev_motor_targets - mjinfer.max_motor_velocity * (mjinfer.sim_dt * mjinfer.decimation),
                mjinfer.prev_motor_targets + mjinfer.max_motor_velocity * (mjinfer.sim_dt * mjinfer.decimation),
            )
            mjinfer.prev_motor_targets = mjinfer.motor_targets.copy()

        mjinfer.data.ctrl = mjinfer.motor_targets.copy()

        # Check fallen
        gravity = mjinfer.get_gravity(mjinfer.data)
        if gravity[2] < 0.0:
            print(f"FALLEN at policy step {survival}, gravity={gravity}")
            break

        if survival % 10 == 0:
            print(f"Step {survival}: gravity={gravity}, action_range=[{action.min():.3f}, {action.max():.3f}]")

else:
    print(f"Survived all {survival} steps!")
    gravity = mjinfer.get_gravity(mjinfer.data)
    print(f"Final gravity: {gravity}")
