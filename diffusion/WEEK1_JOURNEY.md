# Week 1 Journey — Diffusion Policy on Open Duck Mini V2

**Goal (from project.md):** Generate PPO distillation dataset in MJX, port diffusion_policy to Open Duck obs/action shapes, validate one training run end-to-end.

---

## What was built

### Files created

| File | Purpose |
|------|---------|
| `diffusion/collect_dataset.py` | Rolls PPO ONNX policy in MuJoCo CPU sim, saves (obs, action) HDF5 dataset |
| `diffusion/model.py` | ConditionalUnet1D backbone + DDIM noise scheduler |
| `diffusion/dataset.py` | Sliding-window HDF5 dataset class for training |
| `diffusion/train.py` | AdamW + cosine LR training loop with checkpointing |
| `diffusion/udp_inference_server.py` | Updated: `make_diffusion_policy` stub now implemented |
| `data/.gitignore` | Excludes HDF5 files from git |

### Dataset (`data/ppo_rollouts_5k.hdf5`, 316MB, gitignored)
- **Policy**: `BEST_WALK_ONNX_2.onnx` (original repo policy — honest baseline)
- **Scene**: `scene_flat_terrain.xml` (no backlash, matches policy's training env)
- **Trajectories**: 4755 collected, 865,780 total steps at 50Hz (20ms control period)
- **Randomization per episode**: velocity commands (full range), floor friction (±50%), initial joint positions (50–150% of home), initial yaw (uniform), mid-episode impulse pushes
- **HDF5 structure**:
  ```
  obs          (865780, 101)  float32   — stacked observation per control step
  actions      (865780, 14)   float32   — raw PPO policy output (pre-scaling)
  episode_ends (4755,)        int64     — cumulative step counts at episode boundaries
  stats/
    action_min  (14,)  — per-dim minimum (for normalization)
    action_max  (14,)  — per-dim maximum
    action_mean (14,)
    action_std  (14,)
  ```

### Model architecture (`diffusion/model.py`)

**ConditionalUnet1D** — 1D temporal conv U-Net with FiLM conditioning, 16M parameters.

```
Input: noisy_actions (B, 16, 14)  ← action chunk, noisy
       timestep      (B,)
       global_cond   (B, 202)      ← 2 obs stacked, 2×101

Global conditioning:
  time_emb:  SinusoidalPosEmb(128) → Linear(128→512) → Mish → Linear(512→512)
  obs_emb:   Linear(202→512) → Mish → Linear(512→512)
  cond:      cat(time_emb, obs_emb) = (B, 1024)

U-Net (channels-first internally):
  init_conv:  Conv1d(14→128, k=1)                           T=16
  down1:      2× ResBlock(128→128, cond=1024)   [skip h1]   T=16
              Downsample: Conv1d(128→256, stride=2)         T=8
  down2:      2× ResBlock(256→256, cond=1024)   [skip h2]   T=8
              Downsample: Conv1d(256→512, stride=2)         T=4
  mid:        2× ResBlock(512→512, cond=1024)               T=4
  up1:        Upsample: ConvTranspose1d(512→256, stride=2)  T=8
              cat(h2) → (B, 512, 8)
              2× ResBlock(512→256, cond=1024)               T=8
  up2:        Upsample: ConvTranspose1d(256→128, stride=2)  T=16
              cat(h1) → (B, 256, 16)
              2× ResBlock(256→128, cond=1024)               T=16
  final_conv: Conv1d(128→14, k=1)                           T=16

Output: predicted noise (B, 16, 14)
```

Each `ResBlock` uses GroupNorm + Mish + FiLM conditioning (scale+shift from `Linear(1024, out_ch*2)`).

**NoiseScheduler** (DDPM forward + DDIM reverse):
- Cosine beta schedule, T_train=100 denoising steps
- DDIM inference: T_infer=16 steps, deterministic (η=0)
- Forward: `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1−ᾱ_t)·ε`
- Loss: MSE between predicted and true noise (epsilon prediction)
- Actions normalized to [−1, 1] per dimension before diffusion

### Committed design decisions (do not sweep)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `obs_dim` | 101 | Matches ONNX input shape, confirmed with onnxruntime |
| `action_dim` | 14 | 10 leg + 4 head joints, confirmed with ONNX output |
| `obs_horizon` | 2 | Standard for diffusion policy |
| `pred_horizon` | 16 | Action chunk; matches UDP server `CHUNK_SIZE` |
| `T_train` | 100 | Standard DDPM |
| `T_infer` | 16 | DDIM; fast enough for real-time inference |
| `base_dim` | 128 | 16M params; fits M2 RAM, trains in hours on MPS |

---

## Environment setup

**Python environment**: `.venv/bin/python` (Python 3.13, uv-managed venv).

Extra packages not in the original `pyproject.toml` were installed via:
```bash
uv pip install --python .venv/bin/python mujoco h5py torch tqdm onnxruntime playground==0.0.5
```

The venv's `pyvenv.cfg` uses `home = /Users/zhenghao/miniconda3/bin` (base conda Python 3.13).  
The `pyproject.toml` still has `jax[cuda12]` which can't be installed on Mac — this is fine, JAX is only needed for PPO training on the GPU cluster. All diffusion code uses PyTorch + MuJoCo CPU only.

---

## Key observations / gotchas

1. **Obs construction is identical to `mujoco_infer.py`**: `gyro(3) + accel(3) + cmd(7) + joint_angles−default(14) + joint_vel·scale(14) + last_action(14) + last_last_action(14) + last_last_last_action(14) + motor_targets(14) + contacts(2) + imitation_phase(2) = 101`. The accelerometer bias `+1.3` on index 0 is applied in both the PPO inference and the collector.

2. **Action space**: The diffusion policy learns to output `action` (raw network output), not `motor_targets`. Motor targets are computed downstream as `default_actuator + action × 0.25`. This makes the diffusion policy a drop-in replacement for the PPO ONNX policy.

3. **Collection speed**: ~30 traj/sec on M2 Mac. 5k trajectories in ~2.5 minutes.

4. **Training speed on M2**: ~4 batch/s at batch_size=16 on CPU. Use `--device mps` for ~10× speedup (MPS). No cloud GPU needed for 100 epochs.

5. **One bug fixed**: `NoiseScheduler._get` initially returned a 4D tensor (wrong `view(-1, *([1]*3))`) causing a permute error in the U-Net forward pass. Fixed to `view(-1, 1, 1)` for correct (B, 1, 1) broadcasting over (B, T, C) action tensors.

---

## Validation runs

```
# Model smoke test (forward pass + DDIM sample)
Parameters: 16,367,758
Train loss (random inputs): ~1.52
Predicted action shape: (4, 16, 14)  ✓

# Collection smoke test (10 trajectories)
Collected 9 trajectories (1 skipped / fell immediately)
Total steps: 1522
action range: [-0.990, 0.994]

# Training smoke test (5 epochs, batch=16, tiny dataset)
Epoch 1/5  train=0.5745  val=0.3228
Epoch 3/5  train=0.2136  val=0.2031
Epoch 5/5  train=0.1734  val=0.1722  ✓
```

---

## Week 2 instructions for the next agent

Week 2 goal: **Train + sim eval**. Full training run on the 5k dataset, build comparison table vs PPO baseline.

### Step 1 — Full training

```bash
.venv/bin/python -m diffusion.train \
  --dataset data/ppo_rollouts_5k.hdf5 \
  --output checkpoints/diffusion/ \
  --epochs 100 \
  --batch_size 256 \
  --device mps \
  --seed 42
```

Expected: ~2–3 hours on M2 with MPS. Best checkpoint saved to `checkpoints/diffusion/best.pt`.

**GPU note**: MPS (`--device mps`) is sufficient — no need to rent a cloud GPU for one training run. If doing multiple hyperparameter runs, rent the same RTX 4060 Ti used for PPO training (~$5/run).

### Step 2 — Build diffusion eval script

Extend `playground/open_duck_mini_v2/eval_headless.py` (which already evaluates the PPO policy) to also evaluate the diffusion policy. The diffusion policy needs a rolling obs buffer (obs_horizon=2).

Create `diffusion/eval_diffusion.py` that:
1. Loads a diffusion checkpoint (`checkpoints/diffusion/best.pt`)
2. Reuses `MJInferBase` + the same `get_obs()` construction as `collect_dataset.py`
3. Maintains a rolling obs deque of length `obs_horizon=2`
4. At each control step: stack obs → flatten → `model.predict_action(obs_cond)` → consume chunk[0] → advance
5. Action chunking: call `predict_action` every `pred_horizon` steps (or receding horizon — every step, always take action[0])

**Receding horizon inference** (simpler and usually better): call `predict_action` at EVERY control step, always execute only `action_chunk[0]`. This avoids stale-chunk drift at the cost of ~16 DDIM forward passes per second. At 16 steps × ~5ms/step on M2 CPU = 80ms per control step, which is slower than the 20ms control period. So use **fixed-horizon chunking**: re-plan every `pred_horizon=16` steps, execute all 16 actions in sequence.

6. Run the same command presets as `eval_headless.py` (`COMMAND_PRESETS`) for matched conditions
7. Output the same metrics: survival_steps, fall_rate, avg_tracking_lin, avg_tracking_ang, avg_action_rate

### Step 3 — Generate comparison table

Run both policies under matched conditions:
```bash
# PPO baseline (existing script)
.venv/bin/python playground/open_duck_mini_v2/eval_headless.py \
  -o /path/to/BEST_WALK_ONNX_2.onnx \
  --num_episodes 10 --max_steps 1000

# Diffusion policy
.venv/bin/python -m diffusion.eval_diffusion \
  --checkpoint checkpoints/diffusion/best.pt \
  --num_episodes 10 --max_steps 1000
```

Target metrics for the comparison table:
- Survival rate (steps survived / max_steps)
- Fall rate
- Velocity tracking (linear + angular)  
- Action smoothness (action_rate = mean ‖aₜ − aₜ₋₁‖²)
- Perturbation recovery (same COMMAND_PRESETS with perturbations injected)

### Step 4 — If results look bad

If the diffusion policy falls immediately:
1. Check action normalization — verify `action_min/max` from the dataset are being applied correctly at inference
2. Check obs construction — must exactly match `collect_dataset.py:_get_obs()`, especially the `imitation_phase` and `motor_targets` fields
3. Check the obs_horizon buffer initialization — pad with the first observation (like the UDP server does)
4. Try receding-horizon inference (replan every step, take action[0]) to rule out stale-chunk issues
5. Collect more data (10k trajectories) or train longer (200 epochs)

### Reference: key file paths

```
data/ppo_rollouts_5k.hdf5          ← 5k-traj dataset, 865K steps (gitignored)
checkpoints/diffusion/best.pt      ← best training checkpoint (to be created in Week 2)
diffusion/collect_dataset.py       ← dataset collector
diffusion/model.py                 ← ConditionalUnet1D + NoiseScheduler
diffusion/dataset.py               ← DiffusionDataset (HDF5 loader)
diffusion/train.py                 ← training script
diffusion/udp_inference_server.py  ← UDP server (make_diffusion_policy now implemented)
playground/open_duck_mini_v2/eval_headless.py  ← PPO eval (baseline reference)
```

### Useful constants (must match across train/eval)

```python
OBS_DIM = 101
ACTION_DIM = 14
OBS_HORIZON = 2
PRED_HORIZON = 16   # action chunk size
T_TRAIN = 100
T_INFER = 16
```

All defined in `diffusion/model.py` and importable from there.
