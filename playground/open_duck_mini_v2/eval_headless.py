"""Headless evaluation of trained ONNX policies for Open Duck Mini V2.

Runs episodes without a viewer and reports metrics for comparing checkpoints.
"""

import argparse
import numpy as np
import mujoco
from etils import epath

from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.open_duck_mini_v2 import base

USE_MOTOR_SPEED_LIMITS = True


class HeadlessEval(MJInferBase):
    def __init__(self, model_path: str, reference_data: str, onnx_model_path: str):
        super().__init__(model_path)

        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.max_motor_velocity = 5.24
        self.PRM = PolyReferenceMotion(reference_data)
        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.imitation_i = 0
        self.imitation_phase = np.array([1.0, 0.0])

    def reset(self):
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

    def get_obs(self, data, command):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)
        contact_arr = np.array([float(contacts[0]), float(contacts[1])])

        obs = np.concatenate([
            gyro,
            accelerometer,
            command,
            joint_angles - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            self.motor_targets,
            contact_arr,
            self.imitation_phase,
        ])
        return obs

    def is_fallen(self, data):
        gravity = self.get_gravity(data)
        return gravity[2] < 0.0

    def run_episode(self, command, max_steps=1000):
        self.reset()
        total_reward_components = {
            "survival_steps": 0,
            "tracking_lin_vel_sum": 0.0,
            "tracking_ang_vel_sum": 0.0,
            "action_rate_sum": 0.0,
            "fallen": False,
        }

        for step in range(max_steps):
            mujoco.mj_step(self.model, self.data)

            if step % self.decimation != 0:
                continue

            self.imitation_i += 1.0
            self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
            self.imitation_phase = np.array([
                np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
            ])

            obs = self.get_obs(self.data, command)
            action = self.policy.infer(obs)

            # Compute tracking metrics
            local_linvel = self.get_linvel(self.data)
            lin_vel_error = np.sum(np.square(
                np.array([command[0], command[1]]) - local_linvel[:2]
            ))
            tracking_lin = np.exp(-lin_vel_error / 0.01)

            gyro = self.get_gyro(self.data)
            ang_vel_error = np.square(command[2] - gyro[2])
            tracking_ang = np.exp(-ang_vel_error / 0.01)

            action_rate = np.sum(np.square(action - self.last_action))

            total_reward_components["tracking_lin_vel_sum"] += tracking_lin
            total_reward_components["tracking_ang_vel_sum"] += tracking_ang
            total_reward_components["action_rate_sum"] += action_rate
            total_reward_components["survival_steps"] += 1

            # Apply action
            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()

            self.motor_targets = self.default_actuator + action * self.action_scale

            if USE_MOTOR_SPEED_LIMITS:
                self.motor_targets = np.clip(
                    self.motor_targets,
                    self.prev_motor_targets
                    - self.max_motor_velocity * (self.sim_dt * self.decimation),
                    self.prev_motor_targets
                    + self.max_motor_velocity * (self.sim_dt * self.decimation),
                )
                self.prev_motor_targets = self.motor_targets.copy()

            self.data.ctrl = self.motor_targets.copy()

            if self.is_fallen(self.data):
                total_reward_components["fallen"] = True
                break

        steps = total_reward_components["survival_steps"]
        if steps > 0:
            total_reward_components["avg_tracking_lin"] = (
                total_reward_components["tracking_lin_vel_sum"] / steps
            )
            total_reward_components["avg_tracking_ang"] = (
                total_reward_components["tracking_ang_vel_sum"] / steps
            )
            total_reward_components["avg_action_rate"] = (
                total_reward_components["action_rate_sum"] / steps
            )
        else:
            total_reward_components["avg_tracking_lin"] = 0.0
            total_reward_components["avg_tracking_ang"] = 0.0
            total_reward_components["avg_action_rate"] = 0.0

        return total_reward_components


# Command presets: [lin_vel_x, lin_vel_y, ang_vel_yaw, neck_pitch, head_pitch, head_yaw, head_roll]
COMMAND_PRESETS = {
    "stand": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "forward": np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "forward_fast": np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "backward": np.array([-0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "left": np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "right": np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "turn_left": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "turn_right": np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
}


def evaluate_policy(onnx_path, model_path, reference_data, num_episodes=5, max_steps=1000):
    evaluator = HeadlessEval(model_path, reference_data, onnx_path)

    results = {}
    for preset_name, command in COMMAND_PRESETS.items():
        episodes = []
        for ep in range(num_episodes):
            metrics = evaluator.run_episode(command, max_steps=max_steps)
            episodes.append(metrics)
        results[preset_name] = episodes

    return results


def print_results(results, onnx_label=""):
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
    print(f"Policy: {onnx_label}")
    print(f"{'='*80}")
    header = f"{'Command':<15} {'Avg Steps':>10} {'Fall Rate':>10} {'Track Lin':>10} {'Track Ang':>10} {'Act Rate':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row[0]:<15} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")

    overall_steps = np.mean([
        e["survival_steps"]
        for eps in results.values()
        for e in eps
    ])
    overall_fall = np.mean([
        e["fallen"]
        for eps in results.values()
        for e in eps
    ])
    print(f"\nOverall: avg_steps={overall_steps:.0f}, fall_rate={overall_fall*100:.0f}%")
    print()
    return overall_steps, overall_fall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless evaluation of trained policies")
    parser.add_argument("-o", "--onnx_paths", nargs="+", required=True,
                        help="One or more ONNX model paths to evaluate")
    parser.add_argument("--model_path", type=str,
                        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml")
    parser.add_argument("--reference_data", type=str,
                        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes per command preset")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max steps per episode (1000 = 20 seconds at 50Hz)")
    args = parser.parse_args()

    summary = []
    for onnx_path in args.onnx_paths:
        results = evaluate_policy(
            onnx_path, args.model_path, args.reference_data,
            num_episodes=args.num_episodes, max_steps=args.max_steps,
        )
        label = onnx_path.split("/")[-1]
        avg_steps, fall_rate = print_results(results, onnx_label=label)
        summary.append([label, f"{avg_steps:.0f}", f"{fall_rate*100:.0f}%"])

    if len(summary) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        header = f"{'Policy':<50} {'Avg Steps':>10} {'Fall Rate':>10}"
        print(header)
        print("-" * len(header))
        for row in summary:
            print(f"{row[0]:<50} {row[1]:>10} {row[2]:>10}")
