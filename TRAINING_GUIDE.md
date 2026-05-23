# Training Guide — Open Duck Mini V2 Walking Policy

Step-by-step guide for training on a Vast.ai remote GPU instance. The repo is modified locally and scp'd to the remote.

---

## 1. Setup

### 1A. Transfer codebase to remote

From your local machine:

```bash
# Replace <ssh_port> and <instance_ip> with your Vast.ai instance details
rsync -avz --exclude='.git' --exclude='.tmp' --exclude='checkpoints' \
  /path/to/Open_Duck_Playground/ \
  root@<instance_ip> -p <ssh_port>:/workspace/Open_Duck_Playground/
```

Or with scp:

```bash
scp -P <ssh_port> -r /path/to/Open_Duck_Playground root@<instance_ip>:/workspace/
```

### 1B. SSH in and verify environment

```bash
ssh root@<instance_ip> -p <ssh_port>
cd /workspace/Open_Duck_Playground
```

Check uv is available (pre-installed on Vast.ai):

```bash
uv --version
```

Verify GPU:

```bash
nvidia-smi
```

### 1C. Install dependencies

```bash
uv sync
```

### 1D. Verify GPU support

```bash
uv run python -c "import jax; print(jax.devices())"
```

Must show `[CudaDevice(id=0)]`. If it shows `[CpuDevice(id=0)]`, check that `pyproject.toml` has `jax[cuda12]==0.8.0` and no bare `jaxlib` line.

### 1E. Start tmux

```bash
tmux new -s training
# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

---

## 2. Baseline Training

Train with default config to establish a baseline.

```bash
cd /workspace/Open_Duck_Playground
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --task flat_terrain \
  --output_dir /workspace/checkpoints/00_baseline
```

**Time:** ~30 min on RTX 3090.

**Verify:** Console should print `Observation size: 101` and periodic STEP logs with eval reward.

### Monitor with TensorBoard (separate tmux pane)

```bash
uv run tensorboard --logdir=/workspace/checkpoints --bind_all
```

Key metrics to watch:
- `eval/episode_reward` — should trend upward
- `reward/tracking_lin_vel` — should increase toward 1.0
- `reward/tracking_ang_vel` — should increase toward 1.0
- `reward/imitation` — should increase as gait improves
- `cost/action_rate` — should decrease over time

---

## 3. Evaluate Baseline

After training completes, evaluate all checkpoints:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/00_baseline/*.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml \
  --num_episodes 5
```

This runs each ONNX through 8 command presets (stand, forward, backward, left, right, turn_left, turn_right) for 5 episodes each and reports:
- **Avg Steps**: how long the robot survives (out of 1000 = 20 seconds)
- **Fall Rate**: percentage of episodes where the robot falls
- **Track Lin/Ang**: how well it follows velocity commands (1.0 = perfect)
- **Action Rate**: smoothness of actions (lower = smoother)

Record the results. The best checkpoint from this run becomes the baseline for comparison.

### Optional: Interactive evaluation (if desktop is available)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py \
  -o /workspace/checkpoints/00_baseline/<best_checkpoint>.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml
```

Controls: Arrow keys = velocity, A/E = turn, H = toggle head mode, P/M = speed up/slow down phase.

---

## 4. Backlash Fine-tune

Fine-tune the best baseline checkpoint on the backlash task. This adapts the policy to gear play in the Feetech STS3215 servos — critical for sim-to-real transfer.

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 75000000 \
  --task flat_terrain_backlash \
  --restore_checkpoint_path /workspace/checkpoints/00_baseline/<best_checkpoint_dir> \
  --output_dir /workspace/checkpoints/01_backlash_finetune
```

**Time:** ~15 min on RTX 3090.

### Evaluate:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/01_backlash_finetune/*.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml
```

Note: use the backlash XML for evaluation here.

**Success criteria:** Robot walks with backlash active, fall rate comparable to baseline. If fall rate jumps significantly, proceed to Cycle A (wider randomization).

---

## 5. Iterative Tuning Cycles

Each cycle: edit code on remote -> train -> evaluate -> decide. With 30-min runs we can iterate fast.

### How to edit files on the remote

```bash
# Using sed for quick changes:
sed -i 's/imitation: 1.0/imitation: 2.0/' playground/open_duck_mini_v2/joystick.py

# Or use nano/vim:
nano playground/open_duck_mini_v2/joystick.py
```

Or edit locally and re-rsync the changed file.

---

### Cycle A: Wider Domain Randomization

**Goal:** Better sim-to-real transfer by training with wider randomization ranges.

**Edit `playground/common/randomize.py`:**

```bash
# Floor friction: 0.5,1.0 -> 0.4,1.2
sed -i 's/minval=0.5, maxval=1.0/minval=0.4, maxval=1.2/' playground/common/randomize.py

# KP scale: 0.9,1.1 -> 0.8,1.2
sed -i 's/minval=0.9, maxval=1.1/minval=0.8, maxval=1.2/' playground/common/randomize.py

# COM jitter: -0.05,0.05 -> -0.08,0.08
sed -i 's/(3,), minval=-0.05, maxval=0.05/(3,), minval=-0.08, maxval=0.08/' playground/common/randomize.py

# Mass scale: 0.9,1.1 -> 0.85,1.15
sed -i "s/shape=(model.nbody,), minval=0.9, maxval=1.1/shape=(model.nbody,), minval=0.85, maxval=1.15/" playground/common/randomize.py

# Torso mass: -0.1,0.1 -> -0.15,0.15
sed -i 's/minval=-0.1, maxval=0.1)  # was -0.2, 0.2/minval=-0.15, maxval=0.15)/' playground/common/randomize.py
```

**Train from scratch** (wider randomization changes the training distribution):

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --task flat_terrain_backlash \
  --output_dir /workspace/checkpoints/02_wider_randomization
```

**Evaluate:**

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/02_wider_randomization/*.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml
```

**Keep if:** overall avg_steps is within 10% of the backlash finetune, and fall rate didn't increase significantly. Wider randomization may slightly reduce peak performance but should improve sim-to-real transfer.

---

### Cycle B: Increase Imitation Weight

**Goal:** More natural gait quality.

**Edit `playground/open_duck_mini_v2/joystick.py`:**

```bash
# Line 85: imitation: 1.0 -> 2.0
sed -i 's/imitation=1.0/imitation=2.0/' playground/open_duck_mini_v2/joystick.py
```

**Fine-tune from best checkpoint so far:**

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 75000000 \
  --task flat_terrain_backlash \
  --restore_checkpoint_path /workspace/checkpoints/<best_so_far>/<checkpoint_dir> \
  --output_dir /workspace/checkpoints/03_imitation_x2
```

**Evaluate and compare. Keep if gait looks more natural without losing stability.**

---

### Cycle C: Tune Tracking and Smoothness

Based on evaluation results, pick one issue to fix:

**If jerky (high action_rate):**
```bash
sed -i 's/action_rate=-0.5/action_rate=-1.0/' playground/open_duck_mini_v2/joystick.py
```

**If poor command following (low tracking metrics):**
```bash
sed -i 's/tracking_lin_vel=2.5/tracking_lin_vel=4.0/' playground/open_duck_mini_v2/joystick.py
sed -i 's/tracking_ang_vel=6.0/tracking_ang_vel=8.0/' playground/open_duck_mini_v2/joystick.py
```

**If falls too much:**
```bash
sed -i 's/alive=20.0/alive=25.0/' playground/open_duck_mini_v2/joystick.py
```

**Fine-tune from best checkpoint, 75M steps. Evaluate. Keep if it improves the targeted metric without regressing others.**

---

### Cycle D: Fix Imitation Reward Structure

**Goal:** The current joint position imitation term uses negative squared error (always negative), while velocity terms use exponential (bounded 0-1). This makes imitation act more as a cost than a reward.

**Edit `playground/open_duck_mini_v2/custom_rewards.py` line 127:**

```bash
# Replace the joint_pos_rew line
sed -i 's/joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) \* w_joint_pos/joint_pos_rew = jp.exp(-5.0 * jp.sum(jp.square(joint_pos - ref_joint_pos))) * w_joint_pos/' playground/open_duck_mini_v2/custom_rewards.py
```

Also change joint_vel to match:
```bash
sed -i 's/joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) \* w_joint_vel/joint_vel_rew = jp.exp(-1.0 * jp.sum(jp.square(joint_vel - ref_joint_vels))) * w_joint_vel/' playground/open_duck_mini_v2/custom_rewards.py
```

**Train from scratch** (significant reward landscape change):

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --task flat_terrain_backlash \
  --output_dir /workspace/checkpoints/04_exp_imitation
```

**This is a high-risk, high-reward change. Evaluate carefully. If it helps, it's likely the single biggest improvement.**

---

### Cycle E: Tracking Sigma (if needed)

If tracking rewards plateau below 0.5, the tracking sigma may be too tight:

```bash
sed -i 's/tracking_sigma=0.01/tracking_sigma=0.025/' playground/open_duck_mini_v2/joystick.py
```

Fine-tune from best checkpoint, 75M steps. Wider sigma gives partial credit for getting "close" to commanded velocity.

---

## 6. Final Training

Take the best configuration discovered during tuning and run a longer training:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 200000000 \
  --task flat_terrain_backlash \
  --output_dir /workspace/checkpoints/final_policy
```

**Time:** ~40 min on RTX 3090.

---

## 7. Checkpoint Selection

Compare the last 5 ONNX checkpoints:

```bash
# List all ONNX files sorted by time
ls -lt /workspace/checkpoints/final_policy/*.onnx | head -5

# Evaluate them all at once
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o $(ls -t /workspace/checkpoints/final_policy/*.onnx | head -5) \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml \
  --num_episodes 10
```

Pick the one with the best balance of:
- Highest avg steps (survival)
- Lowest fall rate
- Best tracking scores
- Lowest action rate (smoothest)

### Robustness test on rough terrain:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/final_policy/<best>.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_rough_terrain_backlash.xml \
  --num_episodes 10
```

Policy should handle rough terrain reasonably well due to domain randomization.

---

## 8. Deploy to Robot

### Copy ONNX from remote to local:

```bash
scp -P <ssh_port> root@<instance_ip>:/workspace/checkpoints/final_policy/<best>.onnx ~/Downloads/BEST_WALK_ONNX_custom.onnx
```

### Copy to Raspberry Pi:

```bash
scp ~/Downloads/BEST_WALK_ONNX_custom.onnx pi@<robot_ip>:~/BEST_WALK_ONNX_custom.onnx
```

### Run on robot:

```bash
ssh pi@<robot_ip>
cd Open_Duck_Mini_Runtime
python scripts/v2_rl_walk_mujoco.py --onnx_model_path ~/BEST_WALK_ONNX_custom.onnx
```

### Safety checklist for first deployment:

1. **Hold the robot** when starting — let it run in the air first to verify joint movements
2. **Soft surface** for first ground test (carpet, mat)
3. **Hand near power switch** — cut power if the robot thrashes violently
4. **Verify action scaling** matches between training (0.25) and runtime
5. **Verify observation vector** is 101 dims in the same order
6. **Verify default pose** matches `keyframe("home").ctrl`

---

## Quick Reference: File Paths on Remote

| Path | Purpose |
|------|---------|
| `/workspace/Open_Duck_Playground/` | Codebase root |
| `/workspace/checkpoints/` | All training runs |
| `/workspace/checkpoints/00_baseline/` | Baseline training |
| `/workspace/checkpoints/01_backlash_finetune/` | Backlash adaptation |
| `/workspace/checkpoints/02_wider_randomization/` | Wider domain randomization |
| `/workspace/checkpoints/03_imitation_x2/` | Doubled imitation weight |
| `/workspace/checkpoints/final_policy/` | Final training run |
| `.tmp/jax_cache/` | JAX compilation cache (auto-created) |

## Quick Reference: Commands Cheat Sheet

```bash
# Training (baseline)
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 150000000 --task flat_terrain --output_dir /workspace/checkpoints/00_baseline

# Training (backlash finetune from checkpoint)
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 75000000 --task flat_terrain_backlash --restore_checkpoint_path <checkpoint_path> --output_dir /workspace/checkpoints/01_backlash_finetune

# Headless evaluation
uv run playground/open_duck_mini_v2/eval_headless.py -o /workspace/checkpoints/<run>/*.onnx --num_episodes 5

# TensorBoard
uv run tensorboard --logdir=/workspace/checkpoints --bind_all

# GPU utilization check
nvidia-smi
```
