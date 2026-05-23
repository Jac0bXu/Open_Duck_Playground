# Training Your Own Walking Policy — Step-by-Step Guide

This guide walks you through training a custom locomotion policy for the Open Duck Mini v2 robot, from a completely fresh machine to deploying on hardware. All commands and file paths have been verified against the actual [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) repository.

---

## Table of Contents

1. [Overview of the Training Pipeline](#1-overview-of-the-training-pipeline)
2. [Hardware & OS Requirements](#2-hardware--os-requirements)
3. [Set Up a Vast.ai Cloud GPU Instance](#3-set-up-a-vastai-cloud-gpu-instance)
4. [Install CUDA Toolkit](#4-install-cuda-toolkit)
5. [Clone and Set Up the Playground Repo](#5-clone-and-set-up-the-playground-repo)
6. [Verify JAX GPU Support](#6-verify-jax-gpu-support)
7. [Reference: Key Dependencies](#7-reference-key-dependencies)
8. [Understand the Codebase](#8-understand-the-codebase)
9. [Configure the Environment and Rewards](#9-configure-the-environment-and-rewards)
10. [Run Training](#10-run-training)
11. [Monitor Training with TensorBoard](#11-monitor-training-with-tensorboard)
12. [Evaluate in Simulation](#12-evaluate-in-simulation)
13. [Find Your Exported ONNX Models](#13-find-your-exported-onnx-models)
14. [Deploy on the Real Robot](#14-deploy-on-the-real-robot)
15. [Debug Sim-to-Real](#15-debug-sim-to-real)
16. [Fallback: CPU-Only Training (Old Pipeline)](#16-fallback-cpu-only-training-old-pipeline)

---

## 1. Overview of the Training Pipeline

There are two training pipelines in the Open Duck Mini ecosystem:

| Pipeline | Location | Algorithm | Sim Engine | GPU | Status |
|---|---|---|---|---|---|
| **Playground (recommended)** | [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) | PPO via Brax | MuJoCo MJX (GPU) | Required (NVIDIA) | Current, maintained |
| Old RL experiments | `Open_Duck_Mini/experiments/RL/` | SAC/TQC via SB3 | MuJoCo (CPU) | Optional | Broken paths, unsupported |

The Playground pipeline is what produced the `BEST_WALK_ONNX_2.onnx` policy you are currently using. This guide focuses on that pipeline.

**The flow:**

```
Open_Duck_Playground (train in MuJoCo MJX on GPU)
    -> ONNX exported automatically at every checkpoint
        -> Copy .onnx to robot
            -> Run with Open_Duck_Mini_Runtime on Raspberry Pi
                -> Observe behavior, tune rewards, retrain
```

---

## 2. Hardware & OS Requirements

### GPU

You need an **NVIDIA GPU** with CUDA support. The Playground uses MuJoCo MJX, which runs physics on GPU via JAX.

| GPU | Usable? | Notes |
|---|---|---|
| NVIDIA RTX 3060+ | Yes | Comfortable |
| NVIDIA RTX 2060 | Yes | Slower but works |
| NVIDIA GTX 1060 | Barely | Very slow, may run out of memory |
| Apple M1/M2/M3 GPU | No | JAX/MJX is CUDA-only |
| No GPU | No for Playground | Use the old pipeline instead (Section 16) |

### Operating System

**Linux** (Ubuntu 20.04 or 22.04) is strongly recommended.

- macOS **will not work** for the Playground pipeline (no CUDA).
- Windows is possible via WSL2 but not covered here.

If you are on macOS, your options are:
1. Use a Linux desktop/laptop with an NVIDIA GPU
2. Rent a cloud GPU instance (see below)

### Cloud GPU Options (if you don't have a local GPU)

| Provider | GPU | Cost | Notes |
|---|---|---|---|
| **[Vast.ai](https://vast.ai/)** | Various | ~$0.10-0.80/hr | **Recommended** — cheapest, full desktop access, see Section 3 |
| [Google Colab](https://colab.research.google.com/) | T4 (free) or A100 (Pro) | Free / $10/mo | Easiest to start, may need to work around session limits |
| [Lambda Labs](https://lambdalabs.com/) | A6000, A100 | ~$0.50-1.10/hr | Good for long training runs |
| [RunPod](https://www.runpod.io/) | Various | ~$0.20-1.50/hr | Cheap, flexible |

### Disk Space

Allow **20-50 GB** free (MuJoCo, JAX, CUDA, model checkpoints, JAX compilation cache).

---

## 3. Set Up a Vast.ai Cloud GPU Instance

This section covers setting up a Vast.ai instance from scratch using the **Linux Desktop** template. This is the recommended approach if you don't have a local NVIDIA GPU.

### Why Vast.ai?

- Cheapest GPU rental (community-hosted machines)
- Full desktop environment with browser-based access (no local software needed)
- CUDA and NVIDIA drivers pre-installed on the Docker image
- Root access — install anything you need
- Persistent storage via the `/workspace` directory

### Choose a GPU

When searching on Vast.ai, filter for:

| Criteria | Recommendation |
|---|---|
| **GPU** | RTX 3090 or RTX 4090 (best value for training). RTX 3080/4080 also works. Avoid GTX 10xx series — too slow and may OOM. |
| **VRAM** | 16 GB+ recommended. 24 GB is comfortable. 10 GB may work but is tight. |
| **Disk space** | Request at least **40 GB** (set in the rental configuration). You need room for JAX compilation cache, model checkpoints, and dependencies. |
| **Image** | Use the **Linux Desktop** template (Docker image with Ubuntu, CUDA, and a browser-based desktop). |
| **Cost** | Expect ~$0.20-0.50/hr for an RTX 3090. A full 150M-step training run costs roughly **$3-8**. |

### Rent the Instance

1. Go to [vast.ai](https://vast.ai/) and create an account
2. Search for instances and filter by GPU model
3. Select the **Linux Desktop** template
4. Set disk size to **40 GB** or more
5. Click **Rent**

### Connect to Your Instance

After the instance starts (takes 1-2 minutes), you have several access methods:

| Method | Port | Best For |
|---|---|---|
| **Selkies WebRTC** | 6100 | Interactive desktop work (recommended) |
| **Guacamole VNC** | 6200 | Browser-based VNC |
| **SSH** | Default SSH port | Terminal-only work |
| **Jupyter** | 8080 | Browser-based notebooks and terminals |

Click the **Open** button for the easiest access (auto-authenticated Selkies desktop).

### First-Time Setup on the Instance

Open a terminal inside the desktop (or connect via SSH). The instance runs Ubuntu in a Docker container with CUDA already installed.

**Verify CUDA and GPU:**

```bash
nvidia-smi
```

You should see your GPU listed with driver and CUDA version. If this works, you can **skip Section 4 (Install CUDA Toolkit)** — it's already done.

**Set up the workspace directory:**

The `/workspace` directory persists across instance restarts. Do all your work there:

```bash
cd /workspace
```

**Install uv (Python package manager):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

**Now proceed to Section 5** (clone and set up the Playground repo) — you don't need to install CUDA manually.

### Set Up SSH Keys

Generate an SSH key on the instance for GitHub access and file transfers:

```bash
# Generate a new key (press Enter through all prompts for defaults)
ssh-keygen -t ed25519 -C "vast-ai-training" -f ~/.ssh/id_ed25519 -N ""

# Print the public key — copy this to GitHub (Settings > SSH and GPG keys > New SSH key)
cat ~/.ssh/id_ed25519.pub
```

**Reusing keys across instances:**

Vast.ai instances are Docker containers. Files outside `/workspace` are lost when the instance is destroyed. To reuse the same SSH key across multiple instances:

```bash
# 1. Generate the key once (locally or on the first instance)
ssh-keygen -t ed25519 -C "vast-ai-training" -f ~/.ssh/vast_ai_key -N ""

# 2. On each new instance, copy your key pair into /workspace first, then symlink:
#    From your local machine:
scp -P <ssh_port> ~/.ssh/vast_ai_key root@<instance_ip>:/workspace/.ssh/id_ed25519
scp -P <ssh_port> ~/.ssh/vast_ai_key.pub root@<instance_ip>:/workspace/.ssh/id_ed25519.pub

#    Then on the instance:
mkdir -p /workspace/.ssh
# (files already copied there)
chmod 600 /workspace/.ssh/id_ed25519
ln -sf /workspace/.ssh/id_ed25519 ~/.ssh/id_ed25519
ln -sf /workspace/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub
```

This way you add one key to GitHub and it works on every new instance.

### Important: Keeping Your Training Running

Vast.ai instances stay running as long as you keep paying. However:

- **Use `tmux` or `screen`** for long training runs so they survive if your browser disconnects:
  ```bash
  # Install tmux (may already be installed)
  apt update && apt install -y tmux

  # Start a new session
  tmux new -s training

  # Run your training inside tmux
  cd /workspace/Open_Duck_Playground
  uv run playground/open_duck_mini_v2/runner.py --num_timesteps 150000000

  # Detach: press Ctrl+B then D
  # Reattach later: tmux attach -t training
  ```

- **Cost monitoring:** Check the Vast.ai dashboard to monitor your balance and spending rate. A 150M-step run on an RTX 3090 (~8-16 hours at ~$0.30/hr) will cost roughly **$3-5**.

- **Data persistence:** Only files in `/workspace` survive if the instance is stopped. Training checkpoints default to `checkpoints/` — make sure they're under `/workspace`:
  ```bash
  cd /workspace/Open_Duck_Playground
  uv run playground/open_duck_mini_v2/runner.py --output_dir /workspace/checkpoints --num_timesteps 150000000
  ```

### Accessing TensorBoard on Vast.ai

Vast.ai exposes specific ports. TensorBoard defaults to port 6006. To make it accessible:

**Option A: Expose port 6006 via the Instance Portal**

1. Start TensorBoard on the instance:
   ```bash
   uv run tensorboard --logdir=/workspace/checkpoints/ --bind_all
   ```
2. In the Vast.ai Instance Portal (port 1111), add an application that forwards port 6006
3. Access via the portal URL

**Option B: Use SSH port forwarding** (more secure)

On your local machine:
```bash
ssh -L 6006:localhost:6006 root@<instance_ip> -p <ssh_port>
```
Then open `http://localhost:6006` in your local browser.

### Transferring Files To/From the Instance

**Copy ONNX model to your local machine** (to then deploy on the robot):

```bash
# From your local machine
scp -P <ssh_port> root@<instance_ip>:/workspace/checkpoints/<timestamp>_<steps>.onnx ~/Downloads/
```

**Or use Jupyter** (port 8080) for a browser-based file manager with upload/download.

### Stopping the Instance

When training is done, **stop or destroy the instance** to avoid ongoing charges. Download your ONNX models first — once destroyed, data in `/workspace` is deleted unless you've set up persistent storage.

---

## 4. Install CUDA Toolkit

> **If you're using Vast.ai:** CUDA is already pre-installed. Skip this section. Verify with `nvidia-smi`.

Check if you already have CUDA:

```bash
nvidia-smi
```

If this shows your GPU and driver version, you have NVIDIA drivers. You still need the CUDA toolkit.

### Ubuntu

```bash
# Add NVIDIA's package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12.x   # use whatever version is current
```

After installation, add to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then:

```bash
source ~/.bashrc
nvcc --version   # verify CUDA compiler is available
```

---

## 5. Clone and Set Up the Playground Repo

### Install uv (Python package manager)

**If you're using Vast.ai:** you already installed uv in Section 3. Skip this.

**Otherwise:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### Clone and install

The Playground's `pyproject.toml` needs three fixes before running `uv sync`:

1. **Pin `playground==0.0.5`** — the latest version (v0.1.0+) has breaking API changes (`mjx_env.init()` renamed, `collision.py` removed). v0.0.5 is the last compatible version.
2. **Pin JAX to <0.5.0** — `playground==0.0.5` pulls in an older brax that uses `jax.device_put_replicated`, which was removed in JAX 0.5.0.
3. **Remove bare `jaxlib>=0.5.0`** — it pulls in the CPU-only jaxlib, overriding the CUDA version that `jax[cuda12]` provides.

```bash
cd /workspace

# Clone the Playground repo
git clone https://github.com/apirrone/Open_Duck_Playground.git
cd Open_Duck_Playground

# Pin playground to v0.0.5 (the last version compatible with this codebase)
sed -i 's|"playground>=0.0.3"|"playground==0.0.5"|' pyproject.toml

# Pin JAX to 0.4.x — playground==0.0.5's brax uses device_put_replicated,
# which was removed in JAX 0.5.0
sed -i 's|"jax\[cuda12\]>=0.5.0"|"jax[cuda12]<0.5.0"|' pyproject.toml

# Remove the bare jaxlib dependency — it pulls in the CPU-only version and
# conflicts with the CUDA-enabled jaxlib that jax[cuda12] provides.
sed -i '/"jaxlib>=0.5.0",/d' pyproject.toml

# Install everything
uv sync
```

### Verify

```bash
uv run python -c "import mujoco; import jax; print('MuJoCo:', mujoco.__version__); print('JAX devices:', jax.devices())"
```

You should see `MuJoCo: 3.x.x` and `JAX devices: [CudaDevice(id=0)]`.

If JAX shows `[CpuDevice(id=0)]` instead, the bare `jaxlib` dependency pulled in the CPU version. Make sure you ran both `sed` commands above, then re-run `uv sync`.

---

## 6. Verify JAX GPU Support

JAX is installed automatically by `uv sync` in Section 5 (pinned to `<0.5.0` for compatibility with playground==0.0.5's brax). This section verifies it's working correctly.

```bash
uv run python -c "import jax; print(jax.devices())"
```

You should see something like `[cuda(id=0)]`. If you see only `[CpuDevice]`, JAX is not finding your GPU — check CUDA installation (Section 4) and make sure you removed the bare `jaxlib` line from `pyproject.toml` (Section 5).

---

## 7. Reference: Key Dependencies

This section lists what `uv sync` installed in Section 5 for reference.

| Dependency | Purpose |
|---|---|
| `jax[cuda12]`, `jaxlib` | GPU-accelerated computation |
| `tensorflow`, `tf2onnx` | ONNX export (JAX weights -> TF -> ONNX) |
| `onnxruntime` | ONNX inference for evaluation |
| `framesviewer` | Visualization |
| `pygame` | Keyboard control during inference |
| `brax` | PPO training algorithm (transitive via `playground`) |

---

## 8. Understand the Codebase

Before training, understand the key files in the Playground:

```
Open_Duck_Playground/
├── pyproject.toml
├── README.md
├── playground/
│   ├── common/                        # Shared code across all robots
│   │   ├── export_onnx.py             # JAX -> TensorFlow -> ONNX export
│   │   ├── onnx_infer.py              # ONNX inference helper
│   │   ├── poly_reference_motion.py   # Loads polynomial_coefficients.pkl
│   │   ├── poly_reference_motion_numpy.py  # NumPy version
│   │   ├── randomize.py               # Domain randomization
│   │   ├── rewards.py                 # Shared reward functions
│   │   ├── rewards_numpy.py           # NumPy versions of rewards
│   │   ├── runner.py                  # Base training runner (PPO via Brax)
│   │   ├── plot_saved_obs.py          # Plot saved observations
│   │   └── utils.py                   # Shared utilities
│   │
│   └── open_duck_mini_v2/             # YOUR ROBOT
│       ├── runner.py                  # Training launcher (this is what you run)
│       ├── joystick.py                # Joystick walking environment
│       ├── standing.py                # Standing environment
│       ├── base.py                    # Base environment class (handles backlash, joints)
│       ├── constants.py               # Joint/sensor/geometry names, task-to-XML mapping
│       ├── custom_rewards.py          # Robot-specific rewards (imitation)
│       ├── custom_rewards_numpy.py    # NumPy version
│       ├── mujoco_infer.py            # Run ONNX policy in MuJoCo viewer
│       ├── mujoco_infer_base.py       # Base class for inference
│       ├── ref_motion_viewer.py       # Visualize reference motions
│       ├── data/
│       │   └── polynomial_coefficients.pkl  # Reference gait for imitation reward
│       └── xmls/
│           ├── open_duck_mini_v2.xml          # Robot MJCF model
│           ├── open_duck_mini_v2_backlash.xml # Robot with gear backlash simulation
│           ├── scene_flat_terrain.xml          # Flat ground scene
│           ├── scene_flat_terrain_backlash.xml # Flat ground + backlash
│           ├── scene_rough_terrain_backlash.xml # Rough ground + backlash
│           ├── sensors.xml                     # Sensor definitions
│           ├── joints_properties.xml           # Joint properties
│           ├── config.json                     # Onshape-to-robot export config
│           └── assets/                         # Meshes and visual assets
```

### Observation Space

The observation vector is constructed in `joystick.py:_get_obs()`. The policy sees a **101-dim state** vector (the comments in the source code say `# 10` and `# 3` but are stale — the XML model has 14 actuators and the command is 7 dims):

| Group | Dims | Source | Details |
|---|---|---|---|
| Gyroscope | 3 | IMU sensor | Angular velocity (rad/s), with noise |
| Accelerometer | 3 | IMU sensor | Linear acceleration (m/s^2), with noise |
| Command velocity | 7 | Sampled randomly | lin_vel_x, lin_vel_y, ang_vel_yaw, neck_pitch, head_pitch, head_yaw, head_roll |
| Joint angles | 14 | Motor encoders | Current angle - default angle (rad), with noise. All 14 actuated joints |
| Joint velocities | 14 | Motor encoders | Scaled by 0.05, with noise |
| Previous action (t-1) | 14 | Action history | Last step's action |
| Previous action (t-2) | 14 | Action history | Two steps ago |
| Previous action (t-3) | 14 | Action history | Three steps ago |
| Motor targets | 14 | Current targets | Position commands sent to motors this step |
| Foot contacts | 2 | Contact sensors | Binary: left and right foot on ground |
| Imitation phase | 2 | Phase encoder | cos/sin of current gait phase |
| **Total** | **101** | | |

There is also a **privileged state** vector (used only during training, not deployed to the robot) that includes the full state plus ground-truth velocities, gravity, root height, actuator forces, and the full reference motion.

### Action Space (14 dimensions)

The policy outputs **14 target joint position offsets** (not torques). These are added to the default pose and scaled:

```
motor_targets = default_actuator + action * action_scale(0.25)
```

All 14 actuated joints (legs + head):

| Index | Joint |
|---|---|
| 0 | left_hip_yaw |
| 1 | left_hip_roll |
| 2 | left_hip_pitch |
| 3 | left_knee |
| 4 | left_ankle |
| 5 | neck_pitch |
| 6 | head_pitch |
| 7 | head_yaw |
| 8 | head_roll |
| 9 | right_hip_yaw |
| 10 | right_hip_roll |
| 11 | right_hip_pitch |
| 12 | right_knee |
| 13 | right_ankle |

Motor speed is limited to **5.24 rad/s** per control step (configurable via `max_motor_velocity`).

### Reward Function

The total reward is a weighted sum, computed in `joystick.py:_get_reward()`. Each component is multiplied by its scale, then summed and clipped:

| Component | Scale | Type | What it encourages |
|---|---|---|---|
| `alive` | **20.0** | Reward (+) | Staying alive (not falling) |
| `tracking_lin_vel` | **2.5** | Reward (+) | Moving at commanded linear velocity |
| `tracking_ang_vel` | **6.0** | Reward (+) | Turning at commanded yaw rate |
| `imitation` | **1.0** | Reward (+) | Matching reference gait from polynomial_coefficients.pkl |
| `action_rate` | **-0.5** | Cost (-) | Penalizing jerky actions (change between steps) |
| `stand_still` | **-0.2** | Cost (-) | Penalizing joint motion when command is zero |
| `torques` | **-0.001** | Cost (-) | Penalizing high actuator forces |

The imitation reward (based on the Disney BDX paper) is what made the biggest difference in gait quality.

### Domain Randomization and Noise

The environment applies several forms of noise and randomization during training:

- **Sensor noise:** Gaussian noise on joint positions, velocities, gyroscope, accelerometer
- **Action delay:** 0-3 control steps of delay on actions
- **IMU delay:** 0-3 control steps of delay on IMU readings
- **Random pushes:** Horizontal forces every 5-10 seconds with magnitude 0.1-1.0 m/s
- **Initial position randomization:** Joint positions multiplied by uniform [0.5, 1.5]
- **Initial velocity randomization:** Base velocity uniform [-0.05, 0.05]
- **Command randomization:** Velocity commands resampled every 500 steps; 10% chance of zero command

### Available Tasks

From `constants.py`, four scene variants are available:

| Task Name | Scene File | Description |
|---|---|---|
| `flat_terrain` | `scene_flat_terrain.xml` | Flat ground |
| `rough_terrain` | `scene_rough_terrain.xml` | Rough ground |
| `flat_terrain_backlash` | `scene_flat_terrain_backlash.xml` | Flat ground + gear backlash simulation |
| `rough_terrain_backlash` | `scene_rough_terrain_backlash.xml` | Rough ground + backlash |

The `backlash` variants simulate gear backlash in the servos, which is critical for sim-to-real transfer with the Feetech STS3215 motors.

### Control Frequency

| Parameter | Value | Meaning |
|---|---|---|
| `ctrl_dt` | 0.02s | Policy runs at **50Hz** |
| `sim_dt` | 0.002s | Physics simulates at **500Hz** |
| `action_repeat` | 1 | Action applied once per control step |

---

## 9. Configure the Environment and Rewards

Open `playground/open_duck_mini_v2/joystick.py` in a text editor.

### Enable the imitation reward

At the top of the file (line 45):

```python
USE_IMITATION_REWARD = True  # Should be True
USE_MOTOR_SPEED_LIMITS = True  # Should be True
```

The imitation reward uses the reference gait from `data/polynomial_coefficients.pkl`. This file is already included in the repo. You do not need to generate it unless you want a different gait style (use the [reference motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) for that).

### Reward weights

The weights are defined in `default_config()` (around line 49). The defaults are a good starting point. Do not change them on your first training run. Once you see results, you can tune:

- Increase `tracking_lin_vel` (2.5) if the robot doesn't follow forward/backward commands
- Increase `tracking_ang_vel` (6.0) if the robot doesn't turn well
- Increase `alive` (20.0) if the robot falls too often
- Increase `imitation` (1.0) if the gait looks unnatural
- Decrease `action_rate` (-0.5 to -1.0) if movements are too jerky
- Increase `torques` (-0.001 to -0.01) if the robot is too aggressive

### Command velocity ranges

Defined in `default_config()`:

```python
lin_vel_x=[-0.15, 0.15],   # forward/backward (m/s)
lin_vel_y=[-0.2, 0.2],     # lateral (m/s)
ang_vel_yaw=[-1.0, 1.0],   # turning (rad/s)
```

If you want the robot to walk faster, widen the `lin_vel_x` range. If the robot can't track the velocities, narrow the ranges.

---

## 10. Run Training

### Quick test run (5-10 minutes)

Verify everything works before committing to a long training run:

```bash
cd Open_Duck_Playground
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 1000000
```

Watch the console output. You should see:
- Observation size printed: `Observation size: 101`
- Step count and reward being printed periodically
- No errors about missing files, CUDA, or JAX
- GPU utilization (check with `nvidia-smi` in another terminal)

The first run will be slow due to **JAX JIT compilation**. Subsequent steps will be faster because the runner caches compiled kernels in `.tmp/jax_cache/`.

### Full training run

The default configuration uses 150 million timesteps:

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain --num_timesteps 150000000
```

The command that produced the best walking policy (with backlash simulation):

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 150000000
```

On an RTX 3090, 150M timesteps takes roughly **8-16 hours**. On an RTX 3060, expect **1-2 days**.

You can start smaller — try **15 million** first and evaluate:

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain --num_timesteps 15000000
```

### All command-line arguments

```
--output_dir           Where to save checkpoints and ONNX files (default: checkpoints/)
--num_timesteps        Total training steps (default: 150000000)
--env                  Environment: "joystick" or "standing" (default: joystick)
--task                 Scene: flat_terrain, rough_terrain, flat_terrain_backlash,
                       rough_terrain_backlash (default: flat_terrain)
--restore_checkpoint_path   Resume from a saved checkpoint
```

### What to expect during training

- **Steps 0-1M:** Robot thrashes on the ground, learning not to fall over.
- **Steps 1M-10M:** Robot starts standing and occasionally shuffling. Gait is ugly.
- **Steps 10M-50M:** A walking pattern emerges. Robot tracks velocity commands imperfectly.
- **Steps 50M+:** Gait stabilizes. Robot walks reliably and follows commands.

Training is an iterative process. Do not expect a perfect policy on the first run.

---

## 11. Monitor Training with TensorBoard

The runner uses `tensorboardX` to log metrics. Logs are saved to the `checkpoints/` directory (or whatever `--output_dir` you set).

In a separate terminal:

```bash
cd Open_Duck_Playground
uv run tensorboard --logdir=checkpoints/
```

Open `http://localhost:6006` in a browser. Key metrics to watch:

| Metric | What to look for |
|---|---|
| `eval/episode_reward` | Should trend upward over time |
| `reward/tracking_lin_vel` | Should increase toward 1.0 |
| `reward/tracking_ang_vel` | Should increase toward 1.0 |
| `reward/alive` | Should stay consistently high |
| `reward/imitation` | Should increase as the gait improves |
| `cost/action_rate` | Should decrease over time (smoother) |
| `cost/torques` | Should stabilize |

The progress is printed to console every evaluation cycle:

```
-----------
STEP: 1000000 reward: 0.123 reward_std: 0.045
-----------
```

---

## 12. Evaluate in Simulation

Before deploying on the real robot, test your trained policy in the MuJoCo viewer:

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_your_trained_policy.onnx>
```

This opens a 3D MuJoCo window showing the simulated duck walking.

### What to check

- Does the duck walk forward?
- Does it recover from small perturbations?
- Does it follow velocity commands? (keyboard/joystick may be usable)
- Does it fall over after a few steps?

If the policy fails in simulation, it will certainly fail on the real robot. Iterate on reward weights and retrain.

---

## 13. Find Your Exported ONNX Models

**ONNX export is automatic.** You do NOT need to manually run an export script.

Every time the runner saves a checkpoint, it also exports an ONNX file. The `policy_params_fn` in `runner.py` handles this automatically:

1. Saves an Orbax checkpoint (for resuming training)
2. Converts JAX weights -> TensorFlow -> ONNX via `export_onnx.py`

Both files are saved to your output directory with timestamped names:

```
checkpoints/
├── 2026_05_18_143025_1000000/       # Orbax checkpoint (for resume)
├── 2026_05_18_143025_1000000.onnx   # ONNX model (for deployment)
├── 2026_05_18_150812_5000000/
├── 2026_05_18_150812_5000000.onnx
└── events.out.tfevents.*            # TensorBoard logs
```

The ONNX model includes **observation normalization baked in** (mean/std subtraction). The model takes a 101-dim observation and outputs a 10-dim action.

### Resume training from a checkpoint

```bash
uv run playground/open_duck_mini_v2/runner.py \
    --restore_checkpoint_path checkpoints/2026_05_18_143025_1000000 \
    --num_timesteps 150000000
```

---

## 14. Deploy on the Real Robot

### Copy the ONNX model to the robot

From your training machine (or from a Vast.ai instance, download first then scp):

```bash
# If training on local machine:
scp checkpoints/<timestamp>_<steps>.onnx pi@<robot_ip>:~/BEST_WALK_ONNX_custom.onnx

# If training on Vast.ai, first download to your local machine:
scp -P <ssh_port> root@<instance_ip>:/workspace/checkpoints/<timestamp>_<steps>.onnx ~/Downloads/
# Then from your local machine:
scp ~/Downloads/<timestamp>_<steps>.onnx pi@<robot_ip>:~/BEST_WALK_ONNX_custom.onnx
```

### Run on the robot

SSH into the Raspberry Pi:

```bash
ssh pi@<robot_ip>
cd Open_Duck_Mini_Runtime
python scripts/v2_rl_walk_mujoco.py --onnx_model_path ~/BEST_WALK_ONNX_custom.onnx
```

### Safety tips for first deployment

1. **Hold the robot** when starting the policy. Let it run for a few seconds in the air to verify joint movements look reasonable.
2. **Place it on a soft surface** (carpet, mat) for the first ground test.
3. **Keep your hand near the power switch** or battery connector. If the robot thrashes violently, cut power immediately to avoid stripping servo gears.
4. **Verify action scaling.** The training applies `motor_targets = default_actuator + action * 0.25`. Make sure the runtime's `action_scale` matches this value (0.25).

---

## 15. Debug Sim-to-Real

Your first policy will probably not walk perfectly. This is normal. Here's a checklist:

### The robot falls immediately
- **Action scaling mismatch:** The runtime applies `action * action_scale + init_pos`. Check that the action scale (0.25) matches between training and runtime.
- **Joint order mismatch:** The training uses the order from `constants.JOINTS_ORDER_NO_HEAD`. The runtime may order joints differently. Check `rl_utils.py` in the runtime for joint reordering logic (`isaac_to_mujoco` / `mujoco_to_isaac`).
- **Observation size mismatch:** The training produces 101-dim observations. The runtime must construct the same 101-dim vector in the same order. If the dimensions don't match, the policy will output garbage.
- **Default pose mismatch:** The training uses `keyframe("home").ctrl` as the default actuator position. The runtime's `init_pos` must match.

### The robot walks but is unstable
- **Train with backlash:** Use `--task flat_terrain_backlash` instead of `flat_terrain`. The backlash simulation models gear play in the servos.
- **Increase training timesteps** — the policy may not have converged.
- **Check push randomization** — the default push magnitude (0.1-1.0) trains robustness. If your real robot is pushed around more, increase `magnitude_range`.

### The robot walks but ignores commands
- **Increase `tracking_lin_vel` and `tracking_ang_vel` reward scales.**
- **Widen the velocity command ranges** in `default_config()` so the policy sees a wider distribution during training.

### The robot is too aggressive / jerky
- **Increase `action_rate` cost** (make it more negative, e.g., -1.0 instead of -0.5).
- **Increase `torques` cost** (make it more negative, e.g., -0.01 instead of -0.001).
- **Decrease `action_scale`** in the training config (e.g., 0.15 instead of 0.25).

### The robot is too sluggish
- **Decrease `action_rate` cost** (e.g., -0.2 instead of -0.5).
- **Increase `action_scale`** in training (e.g., 0.35 instead of 0.25).
- **Check that motor PID gains** on the real servos match the simulated motor model.

### Motor model accuracy

The sim models the Feetech STS3215 servos using parameters from [BAM actuator identification](https://github.com/Rhoban/bam/). The identified parameters are [here](https://github.com/Rhoban/bam/tree/main/params/feetech_sts3215_7_4V). If your servos behave differently (different firmware, voltage, wear), you may need to re-identify them.

---

## 16. Fallback: CPU-Only Training (Old Pipeline)

If you absolutely cannot access an NVIDIA GPU, you can use the old training pipeline in your local `Open_Duck_Mini` repo. This runs on CPU with regular MuJoCo and stable-baselines3. **Warning:** the code has hardcoded paths to the original author's machine and will require fixes.

### Install dependencies

```bash
cd /path/to/Open_Duck_Mini
pip install -e .[all]
```

This installs from `setup.cfg`:

```
mujoco==3.1.5
gymnasium[mujoco]==0.29.1
stable-baselines3[extra]==2.3.2
sb3_contrib==2.3.0
placo==0.5.0
```

### Fix hardcoded paths

Open `experiments/RL/env.py` and `experiments/RL/env_humanoid.py` and update the hardcoded XML path:

```python
# BEFORE (hardcoded to author's machine):
MujocoEnv.__init__(
    self,
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/scene.xml",
    5,
    ...
)

# AFTER (use your actual path):
MujocoEnv.__init__(
    self,
    "/your/path/to/Open_Duck_Mini/mini_bdx/robots/bdx/scene.xml",
    5,
    ...
)
```

### Fix the reward function

In `env.py`, most reward components are commented out (lines 298-306). Uncomment them:

```python
reward = (
    0.005                                           # time reward
    + 0.1 * self.walking_height_reward()
    + 0.1 * self.upright_reward()
    + 0.1 * self.velocity_tracking_reward()
    + 0.1 * self.smoothness_reward()
    + 0.1 * self.feet_contact_reward()
)
```

Also uncomment the termination check:

```python
if self.is_terminated():
    reward = -10
```

### Run training

```bash
cd experiments/RL/
python train.py -a SAC -n my_first_walk -d cpu
```

Or use the newer version in `experiments/RL/new/`:

```bash
cd experiments/RL/new/
python train.py -a SAC -n my_first_walk -d cpu
```

### Monitoring

```bash
tensorboard --logdir=logs/
```

### Limitations

- **Much slower** than GPU training (10x-50x)
- **Less maintained** — the author has moved to the Playground pipeline
- **Different observation/action space** than the Playground pipeline — the exported policy may not be directly compatible with the current runtime
- **No imitation reward** — this pipeline predates the reference motion integration
- **No backlash simulation** — less accurate motor modeling

---

## Quick Reference: Key URLs

| Resource | URL |
|---|---|
| Training repo (Playground) | https://github.com/apirrone/Open_Duck_Playground |
| Main robot repo | https://github.com/apirrone/Open_Duck_Mini |
| Runtime repo (your fork) | https://github.com/Jac0bXu/Open_Duck_Mini_Runtime |
| Reference motion generator | https://github.com/apirrone/Open_Duck_reference_motion_generator |
| Actuator identification (BAM) | https://github.com/Rhoban/bam/ |
| Community Discord | https://discord.gg/UtJZsgfQGe |
| MuJoCo documentation | https://mujoco.readthedocs.io/ |
| MuJoCo Playground (DeepMind) | https://github.com/google-deepmind/mujoco_playground |
| JAX installation guide | https://github.com/google/jax#installation |
| CAD model (Onshape) | https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05 |
