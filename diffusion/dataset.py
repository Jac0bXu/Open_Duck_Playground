"""HDF5 dataset for diffusion policy training.

Loads the (obs, actions, episode_ends) arrays written by collect_dataset.py.
Samples windows of (obs_horizon past obs, pred_horizon future actions).
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from diffusion.model import OBS_HORIZON, PRED_HORIZON


class DiffusionDataset(Dataset):
    """Sliding-window dataset for BC diffusion policy.

    Each sample is:
        obs_cond  : (obs_horizon, obs_dim)   — last obs_horizon observations
        actions   : (pred_horizon, action_dim) — next pred_horizon actions

    A valid sample index t must satisfy:
        t >= episode_start + obs_horizon - 1
        t + pred_horizon <= episode_end
    """

    def __init__(
        self,
        hdf5_path: str,
        obs_horizon: int = OBS_HORIZON,
        pred_horizon: int = PRED_HORIZON,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        with h5py.File(hdf5_path, "r") as f:
            self.obs = f["obs"][:]            # (total_steps, obs_dim)
            self.actions = f["actions"][:]    # (total_steps, action_dim)
            self.episode_ends = f["episode_ends"][:]
            self.action_min = f["stats/action_min"][:]
            self.action_max = f["stats/action_max"][:]

        self._build_valid_indices()

    def _build_valid_indices(self) -> None:
        """Pre-compute all valid (step_index,) pairs across episodes."""
        oh = self.obs_horizon
        ph = self.pred_horizon

        episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        indices = []
        for start, end in zip(episode_starts, self.episode_ends):
            # t is the LAST obs step; we need [t-oh+1..t] obs and [t..t+ph) actions
            valid_start = start + oh - 1
            valid_end = end - ph       # inclusive
            if valid_end >= valid_start:
                indices.extend(range(int(valid_start), int(valid_end) + 1))

        self.indices = np.array(indices, dtype=np.int64)
        print(f"Dataset: {len(self.indices)} valid samples "
              f"from {len(self.episode_ends)} episodes "
              f"({self.obs.shape[0]} total steps)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        t = self.indices[idx]
        oh = self.obs_horizon
        ph = self.pred_horizon

        obs_seq = self.obs[t - oh + 1 : t + 1]      # (obs_horizon, obs_dim)
        act_seq = self.actions[t : t + ph]            # (pred_horizon, action_dim)

        return {
            "obs_cond": torch.from_numpy(obs_seq).float(),
            "actions": torch.from_numpy(act_seq).float(),
        }
