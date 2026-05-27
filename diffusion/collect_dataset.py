"""Collect PPO rollout dataset for diffusion policy training.

Rolls the PPO ONNX policy in MuJoCo CPU sim with randomized commands,
friction, initial poses, and mid-episode perturbations. Records (obs, action)
pairs at every control step (50 Hz) and saves to HDF5.

Usage:
    .venv/bin/python -m diffusion.collect_dataset \\
        --onnx /path/to/BEST_WALK_ONNX_2.onnx \\
        --n_traj 100 \\
        --output data/ppo_rollouts_100.hdf5
"""

import argparse
import os
import numpy as np
import mujoco
import h5py
from tqdm import tqdm

from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True

COMMANDS_RANGE = {
    "lin_vel_x":  (-0.15, 0.15),
    "lin_vel_y":  (-0.20, 0.20),
    "ang_vel_yaw": (-1.0,  1.0),
    "neck_pitch": (-0.34,  1.1),
    "head_pitch": (-0.78,  0.78),
    "head_yaw":   (-1.5,   1.5),
    "head_roll":  (-0.5,   0.5),
}


class DatasetCollector(MJInferBase):
    def __init__(self, model_path: str, reference_data: str, onnx_path: str):
        super().__init__(model_path)

        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.max_motor_velocity = 5.24  # rad/s

        self.PRM = PolyReferenceMotion(reference_data)
        self.policy = OnnxInfer(onnx_path, awd=True)

        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._base_floor_friction = self.model.geom_friction[self.floor_geom_id].copy()

        self._reset_state()

    # ── internal helpers ────────────────────────────────────────────────────

    def _reset_state(self):
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.imitation_i = 0.0
        self.imitation_phase = np.array([1.0, 0.0])

    def _reset_sim(self):
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_actuator.copy()

        # Randomise initial joint positions (× U[0.5, 1.5]) — mirrors training env
        actuator_addr = self.get_actuator_joints_addr()
        noise = np.random.uniform(0.5, 1.5, size=len(actuator_addr))
        self.data.qpos[actuator_addr] *= noise

        # Randomise initial yaw (freejoint quaternion, MuJoCo w-x-y-z order)
        yaw = np.random.uniform(-np.pi, np.pi)
        qbase = self._floating_base_qpos_addr
        self.data.qpos[qbase + 3] = np.cos(yaw / 2)
        self.data.qpos[qbase + 4] = 0.0
        self.data.qpos[qbase + 5] = 0.0
        self.data.qpos[qbase + 6] = np.sin(yaw / 2)

        # Small random xy offset
        dxy = np.random.uniform(-0.05, 0.05, size=2)
        self.data.qpos[qbase:qbase + 2] += dxy

        mujoco.mj_forward(self.model, self.data)
        self._reset_state()

    def _randomise_friction(self):
        factor = np.random.uniform(0.5, 1.5)
        self.model.geom_friction[self.floor_geom_id] = (
            self._base_floor_friction * factor
        )

    def _restore_friction(self):
        self.model.geom_friction[self.floor_geom_id] = self._base_floor_friction.copy()

    def _sample_command(self) -> np.ndarray:
        if np.random.random() < 0.1:
            return np.zeros(7)
        return np.array([
            np.random.uniform(*COMMANDS_RANGE["lin_vel_x"]),
            np.random.uniform(*COMMANDS_RANGE["lin_vel_y"]),
            np.random.uniform(*COMMANDS_RANGE["ang_vel_yaw"]),
            np.random.uniform(*COMMANDS_RANGE["neck_pitch"]),
            np.random.uniform(*COMMANDS_RANGE["head_pitch"]),
            np.random.uniform(*COMMANDS_RANGE["head_yaw"]),
            np.random.uniform(*COMMANDS_RANGE["head_roll"]),
        ])

    def _get_obs(self, command: np.ndarray) -> np.ndarray:
        gyro = self.get_gyro(self.data)
        accel = self.get_accelerometer(self.data)
        accel[0] += 1.3  # mirror training env bias correction

        joint_angles = self.get_actuator_joints_qpos(self.data.qpos)
        joint_vel = self.get_actuator_joints_qvel(self.data.qvel)
        contacts = self.get_feet_contacts(self.data)
        contact_arr = np.array([float(contacts[0]), float(contacts[1])])

        return np.concatenate([
            gyro,                                        # 3
            accel,                                       # 3
            command,                                     # 7
            joint_angles - self.default_actuator,        # 14
            joint_vel * self.dof_vel_scale,              # 14
            self.last_action,                            # 14
            self.last_last_action,                       # 14
            self.last_last_last_action,                  # 14
            self.motor_targets,                          # 14
            contact_arr,                                 # 2
            self.imitation_phase,                        # 2
        ])  # total: 101

    def _is_fallen(self) -> bool:
        return self.get_gravity(self.data)[2] < 0.0

    def _apply_action(self, action: np.ndarray) -> None:
        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        targets = self.default_actuator + action * self.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            dt = self.sim_dt * self.decimation
            targets = np.clip(
                targets,
                self.prev_motor_targets - self.max_motor_velocity * dt,
                self.prev_motor_targets + self.max_motor_velocity * dt,
            )
        self.motor_targets = targets
        self.prev_motor_targets = targets.copy()
        self.data.ctrl[:] = targets

    # ── public API ──────────────────────────────────────────────────────────

    def collect_trajectory(
        self,
        max_ctrl_steps: int = 300,
        push_interval: tuple[int, int] = (80, 160),
        cmd_change_interval: tuple[int, int] = (100, 200),
        min_steps: int = 30,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run one episode; return (obs_array, action_array) or (None, None) on early fall."""
        self._reset_sim()
        self._randomise_friction()

        command = self._sample_command()
        next_push = np.random.randint(*push_interval)
        next_cmd_change = np.random.randint(*cmd_change_interval)

        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []

        for ctrl_step in range(max_ctrl_steps):
            # ── update imitation phase ───────────────────────────────────
            self.imitation_i = (self.imitation_i + 1.0) % self.PRM.nb_steps_in_period
            phase = self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
            self.imitation_phase = np.array([np.cos(phase), np.sin(phase)])

            # ── random perturbation ──────────────────────────────────────
            if ctrl_step == next_push:
                angle = np.random.uniform(0, 2 * np.pi)
                mag = np.random.uniform(0.3, 1.5)
                push = np.array([np.cos(angle), np.sin(angle)]) * mag
                qv = self._floating_base_qvel_addr
                self.data.qvel[qv:qv + 2] += push
                next_push = ctrl_step + np.random.randint(*push_interval)

            # ── command re-sample ────────────────────────────────────────
            if ctrl_step == next_cmd_change:
                command = self._sample_command()
                next_cmd_change = ctrl_step + np.random.randint(*cmd_change_interval)

            # ── observe → act ────────────────────────────────────────────
            obs = self._get_obs(command)
            action = self.policy.infer(obs)

            obs_list.append(obs)
            action_list.append(action)

            self._apply_action(action)

            # ── simulate decimation substeps ─────────────────────────────
            for _ in range(self.decimation):
                mujoco.mj_step(self.model, self.data)

            if self._is_fallen():
                break

        self._restore_friction()

        if len(obs_list) < min_steps:
            return None, None

        return np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.float32)


def collect_dataset(
    onnx_path: str,
    output_path: str,
    n_traj: int,
    model_path: str,
    reference_data: str,
    max_ctrl_steps: int,
    seed: int,
) -> None:
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    collector = DatasetCollector(model_path, reference_data, onnx_path)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_ends: list[int] = []
    step_cursor = 0
    skipped = 0

    for i in tqdm(range(n_traj), desc="Collecting trajectories"):
        obs, actions = collector.collect_trajectory(max_ctrl_steps=max_ctrl_steps)
        if obs is None:
            skipped += 1
            continue
        all_obs.append(obs)
        all_actions.append(actions)
        step_cursor += len(obs)
        episode_ends.append(step_cursor)

    print(f"\nCollected {len(episode_ends)} trajectories ({skipped} skipped / fell immediately)")
    print(f"Total steps: {step_cursor}")

    obs_arr = np.concatenate(all_obs, axis=0)      # (total_steps, 101)
    act_arr = np.concatenate(all_actions, axis=0)   # (total_steps, 14)
    ep_ends = np.array(episode_ends, dtype=np.int64)

    # Per-dim action stats (for normalization at training time)
    act_min = act_arr.min(axis=0)
    act_max = act_arr.max(axis=0)
    act_mean = act_arr.mean(axis=0)
    act_std = act_arr.std(axis=0)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("obs", data=obs_arr, compression="gzip")
        f.create_dataset("actions", data=act_arr, compression="gzip")
        f.create_dataset("episode_ends", data=ep_ends)
        stats = f.create_group("stats")
        stats.create_dataset("action_min", data=act_min)
        stats.create_dataset("action_max", data=act_max)
        stats.create_dataset("action_mean", data=act_mean)
        stats.create_dataset("action_std", data=act_std)
        f.attrs["n_trajectories"] = len(episode_ends)
        f.attrs["total_steps"] = step_cursor
        f.attrs["obs_dim"] = obs_arr.shape[1]
        f.attrs["action_dim"] = act_arr.shape[1]
        f.attrs["onnx_path"] = onnx_path
        f.attrs["model_path"] = model_path

    print(f"Saved to {output_path}")
    print(f"  obs:     {obs_arr.shape}  {obs_arr.dtype}")
    print(f"  actions: {act_arr.shape}  {act_arr.dtype}")
    print(f"  action range: [{act_arr.min():.3f}, {act_arr.max():.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect PPO rollout dataset")
    parser.add_argument("--onnx", required=True, help="Path to PPO ONNX model")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--n_traj", type=int, default=5000)
    parser.add_argument("--max_ctrl_steps", type=int, default=300,
                        help="Max control steps per trajectory (300 = 6s at 50Hz)")
    parser.add_argument("--model_path", type=str,
                        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
    parser.add_argument("--reference_data", type=str,
                        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect_dataset(
        onnx_path=args.onnx,
        output_path=args.output,
        n_traj=args.n_traj,
        model_path=args.model_path,
        reference_data=args.reference_data,
        max_ctrl_steps=args.max_ctrl_steps,
        seed=args.seed,
    )
