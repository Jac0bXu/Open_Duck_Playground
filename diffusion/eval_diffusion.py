"""Headless evaluation of a trained diffusion policy for Open Duck Mini V2.

Runs episodes without a viewer and reports the same metrics as eval_headless.py
so the two policies can be compared directly.

Usage:
    # Evaluate diffusion policy
    .venv/bin/python -m diffusion.eval_diffusion \\
        --checkpoint checkpoints/diffusion/best.pt \\
        --num_episodes 5 --max_steps 1000

    # Compare against PPO baseline (run eval_headless.py separately, then compare)
    .venv/bin/python playground/open_duck_mini_v2/eval_headless.py \\
        -o /path/to/BEST_WALK_ONNX_2.onnx --num_episodes 5 --max_steps 1000
"""

import argparse
import collections
import numpy as np
import torch
import mujoco
from etils import epath

from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.open_duck_mini_v2 import base
from diffusion.model import DiffusionPolicy, OBS_DIM, ACTION_DIM, OBS_HORIZON, PRED_HORIZON

USE_MOTOR_SPEED_LIMITS = True

# Command presets must match eval_headless.py exactly for valid comparison
COMMAND_PRESETS = {
    "stand":        np.array([0.0,   0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    "forward":      np.array([0.15,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    "forward_fast": np.array([0.15,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    "backward":     np.array([-0.15, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    "left":         np.array([0.0,   0.2,  0.0, 0.0, 0.0, 0.0, 0.0]),
    "right":        np.array([0.0,  -0.2,  0.0, 0.0, 0.0, 0.0, 0.0]),
    "turn_left":    np.array([0.0,   0.0,  1.0, 0.0, 0.0, 0.0, 0.0]),
    "turn_right":   np.array([0.0,   0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
}


def load_diffusion_policy(checkpoint_path: str, device: torch.device) -> DiffusionPolicy:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DiffusionPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    action_min = torch.from_numpy(np.array(ckpt["action_min"])).float()
    action_max = torch.from_numpy(np.array(ckpt["action_max"])).float()
    model.action_min = action_min.to(device)
    model.action_max = action_max.to(device)
    model.eval()
    return model


class DiffusionEval(MJInferBase):
    def __init__(
        self,
        model_path: str,
        reference_data: str,
        policy: DiffusionPolicy,
        device: torch.device,
    ):
        super().__init__(model_path)

        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.max_motor_velocity = 5.24
        self.PRM = PolyReferenceMotion(reference_data)
        self.policy = policy
        self.device = device

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.imitation_i = 0
        self.imitation_phase = np.array([1.0, 0.0])

        # Rolling obs buffer for diffusion conditioning (obs_horizon=2)
        self.obs_buf = collections.deque(maxlen=OBS_HORIZON)

    def _reset(self):
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.default_actuator
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.imitation_i = 0
        self.imitation_phase = np.array([1.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self, command: np.ndarray) -> np.ndarray:
        gyro = self.get_gyro(self.data)
        accel = self.get_accelerometer(self.data)
        accel[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(self.data.qpos)
        joint_vel = self.get_actuator_joints_qvel(self.data.qvel)
        contacts = self.get_feet_contacts(self.data)
        contact_arr = np.array([float(contacts[0]), float(contacts[1])])

        return np.concatenate([
            gyro,                                  # 3
            accel,                                 # 3
            command,                               # 7
            joint_angles - self.default_actuator,  # 14
            joint_vel * self.dof_vel_scale,        # 14
            self.last_action,                      # 14
            self.last_last_action,                 # 14
            self.last_last_last_action,            # 14
            self.motor_targets,                    # 14
            contact_arr,                           # 2
            self.imitation_phase,                  # 2
        ])  # total: 101

    def _predict_chunk(self) -> np.ndarray:
        """Stack obs buffer → tensor → DDIM sample → (PRED_HORIZON, ACTION_DIM) numpy."""
        obs_stack = np.stack(list(self.obs_buf), axis=0)      # (OBS_HORIZON, OBS_DIM)
        obs_flat = obs_stack.flatten()                         # (OBS_HORIZON * OBS_DIM,)
        obs_tensor = torch.from_numpy(obs_flat).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            chunk = self.policy.predict_action(obs_tensor)    # (1, PRED_HORIZON, ACTION_DIM)
        return chunk[0].cpu().numpy()                         # (PRED_HORIZON, ACTION_DIM)

    def is_fallen(self) -> bool:
        return self.get_gravity(self.data)[2] < 0.0

    def run_episode(self, command: np.ndarray, max_steps: int = 1000) -> dict:
        self._reset()

        # Pad obs buffer with the first observation (before any physics steps)
        first_obs = self._get_obs(command)
        for _ in range(OBS_HORIZON):
            self.obs_buf.append(first_obs.copy())

        metrics = {
            "survival_steps": 0,
            "tracking_lin_vel_sum": 0.0,
            "tracking_ang_vel_sum": 0.0,
            "action_rate_sum": 0.0,
            "fallen": False,
        }

        action_chunk = None
        ctrl_step = 0  # counts control steps (after decimation)

        for step in range(max_steps):
            mujoco.mj_step(self.model, self.data)

            if step % self.decimation != 0:
                continue

            # Update imitation phase
            self.imitation_i += 1.0
            self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
            self.imitation_phase = np.array([
                np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
            ])

            obs = self._get_obs(command)
            self.obs_buf.append(obs)

            # Fixed-horizon chunking: replan every PRED_HORIZON control steps
            chunk_idx = ctrl_step % PRED_HORIZON
            if chunk_idx == 0:
                action_chunk = self._predict_chunk()

            action = action_chunk[chunk_idx]
            ctrl_step += 1

            # Tracking metrics
            local_linvel = self.get_linvel(self.data)
            lin_vel_error = np.sum(np.square(
                np.array([command[0], command[1]]) - local_linvel[:2]
            ))
            tracking_lin = np.exp(-lin_vel_error / 0.01)

            gyro = self.get_gyro(self.data)
            ang_vel_error = np.square(command[2] - gyro[2])
            tracking_ang = np.exp(-ang_vel_error / 0.01)

            action_rate = np.sum(np.square(action - self.last_action))

            metrics["tracking_lin_vel_sum"] += tracking_lin
            metrics["tracking_ang_vel_sum"] += tracking_ang
            metrics["action_rate_sum"] += action_rate
            metrics["survival_steps"] += 1

            # Apply action
            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()

            self.motor_targets = self.default_actuator + action * self.action_scale

            if USE_MOTOR_SPEED_LIMITS:
                dt = self.data.model.opt.timestep * self.decimation
                self.motor_targets = np.clip(
                    self.motor_targets,
                    self.prev_motor_targets - self.max_motor_velocity * dt,
                    self.prev_motor_targets + self.max_motor_velocity * dt,
                )
                self.prev_motor_targets = self.motor_targets.copy()

            self.data.ctrl = self.motor_targets.copy()

            if self.is_fallen():
                metrics["fallen"] = True
                break

        steps = metrics["survival_steps"]
        metrics["avg_tracking_lin"] = metrics["tracking_lin_vel_sum"] / steps if steps > 0 else 0.0
        metrics["avg_tracking_ang"] = metrics["tracking_ang_vel_sum"] / steps if steps > 0 else 0.0
        metrics["avg_action_rate"] = metrics["action_rate_sum"] / steps if steps > 0 else 0.0

        return metrics


def evaluate_diffusion(checkpoint_path, model_path, reference_data, device_str,
                       num_episodes=5, max_steps=1000):
    device = torch.device(device_str)
    policy = load_diffusion_policy(checkpoint_path, device)

    evaluator = DiffusionEval(model_path, reference_data, policy, device)

    results = {}
    for preset_name, command in COMMAND_PRESETS.items():
        episodes = []
        for _ in range(num_episodes):
            m = evaluator.run_episode(command, max_steps=max_steps)
            episodes.append(m)
        results[preset_name] = episodes

    return results


def print_results(results, label="diffusion"):
    rows = []
    for preset_name, episodes in results.items():
        avg_steps = np.mean([e["survival_steps"] for e in episodes])
        fall_rate = np.mean([e["fallen"] for e in episodes])
        avg_track_lin = np.mean([e["avg_tracking_lin"] for e in episodes])
        avg_track_ang = np.mean([e["avg_tracking_ang"] for e in episodes])
        avg_action_rate = np.mean([e["avg_action_rate"] for e in episodes])
        rows.append([
            preset_name,
            f"{avg_steps:.0f}/1000",
            f"{fall_rate*100:.0f}%",
            f"{avg_track_lin:.3f}",
            f"{avg_track_ang:.3f}",
            f"{avg_action_rate:.4f}",
        ])

    print(f"\n{'='*80}")
    print(f"Policy: {label}")
    print(f"{'='*80}")
    header = (f"{'Command':<15} {'Avg Steps':>10} {'Fall Rate':>10} "
              f"{'Track Lin':>10} {'Track Ang':>10} {'Act Rate':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row[0]:<15} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")

    overall_steps = np.mean([e["survival_steps"] for eps in results.values() for e in eps])
    overall_fall = np.mean([e["fallen"] for eps in results.values() for e in eps])
    print(f"\nOverall: avg_steps={overall_steps:.0f}, fall_rate={overall_fall*100:.0f}%")
    print()
    return overall_steps, overall_fall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless evaluation of diffusion policy")
    parser.add_argument("--checkpoint", required=True, help="Path to diffusion checkpoint .pt")
    parser.add_argument("--model_path", type=str,
                        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml")
    parser.add_argument("--reference_data", type=str,
                        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for diffusion inference: cpu, mps, cuda")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    results = evaluate_diffusion(
        args.checkpoint, args.model_path, args.reference_data,
        args.device, args.num_episodes, args.max_steps,
    )
    label = args.checkpoint.split("/")[-1]
    print_results(results, label=label)
