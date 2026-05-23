# Agent Prompt: Train Open Duck Mini V2 Walking Policy

You are an RL training engineer. Your job is to SSH into a remote GPU machine, set up the training environment, train walking policies for the Open Duck Mini v2 robot, evaluate them, and iteratively tune the reward function until you produce the best possible policy.

## Remote Machine

```
Host: <INSTANCE_IP>
Port: <SSH_PORT>
User: root
GPU: NVIDIA RTX 3090
Training time: ~30 min for 150M steps
```

SSH in with: `ssh root@<INSTANCE_IP> -p <SSH_PORT>`

## Step 1: Transfer Codebase and Set Up

From the LOCAL machine, rsync the repo (already modified with pinned deps) to the remote:

```bash
rsync -avz --exclude='.git' --exclude='.tmp' --exclude='checkpoints' --exclude='.claude' \
  /Users/zhenghao/Documents/GitHub/Open_Duck_Playground/ \
  -e "ssh -p <SSH_PORT>" \
  root@<INSTANCE_IP>:/workspace/Open_Duck_Playground/
```

Then SSH in and run setup:

```bash
ssh root@<INSTANCE_IP> -p <SSH_PORT>
cd /workspace/Open_Duck_Playground
uv --version        # should be pre-installed on Vast.ai
nvidia-smi          # verify GPU
uv sync             # install all deps (takes 2-5 min)
```

Verify JAX GPU:

```bash
uv run python -c "import jax; print(jax.devices())"
# MUST show [CudaDevice(id=0)]. If CpuDevice, stop and debug before continuing.
```

Smoke test (1M steps, ~1 min):

```bash
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 1000000 --task flat_terrain --output_dir /workspace/checkpoints/smoke_test
# Should print "Observation size: 101" and STEP logs. No errors.
```

## Step 2: Baseline Training

Use tmux for long-running training so it survives disconnects.

```bash
tmux new -s train
cd /workspace/Open_Duck_Playground
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --task flat_terrain \
  --output_dir /workspace/checkpoints/00_baseline
# Detach: Ctrl+B then D. Reattach: tmux attach -t train
```

This takes ~30 min. While waiting, you can monitor in another tmux pane:

```bash
uv run tensorboard --logdir=/workspace/checkpoints --bind_all
```

After training completes, evaluate ALL checkpoints with the headless eval script:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/00_baseline/*.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml \
  --num_episodes 5
```

**Record the results.** Note: the eval script runs 8 command presets (stand, forward, backward, left, right, turn_left, turn_right, forward_fast) × 5 episodes each. It reports avg survival steps (out of 1000), fall rate, velocity tracking scores (0-1, higher is better), and action rate (lower = smoother).

Pick the best checkpoint from this run (highest avg steps, lowest fall rate, best tracking). Record its path.

## Step 3: Backlash Fine-tune

Fine-tune the best baseline checkpoint on the backlash task (simulates gear play in Feetech STS3215 servos — critical for sim-to-real transfer):

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 75000000 \
  --task flat_terrain_backlash \
  --restore_checkpoint_path /workspace/checkpoints/00_baseline/<BEST_CHECKPOINT_DIR> \
  --output_dir /workspace/checkpoints/01_backlash_finetune
```

Evaluate:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/01_backlash_finetune/*.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml \
  --num_episodes 5
```

Record results. If fall rate jumped way up compared to baseline, note this — the backlash adaptation may need more steps or wider randomization.

## Step 4: Iterative Tuning

This is the core loop. For EACH cycle below:
1. Edit the specified file on the remote (use `sed` or `nano`)
2. Train (from scratch or fine-tune, as specified)
3. Evaluate
4. Compare to the BEST result so far
5. **Keep the change if it improved metrics. Revert if it made things worse.**
6. Record results in a comparison table

### Important: Always use backlash task for training and evaluation after Step 3.

---

### Cycle A: Wider Domain Randomization

**Why:** The current randomization ranges are narrow (recently tightened from wider values). Wider ranges = better sim-to-real transfer. The policy sees more variety during training and becomes more robust.

Edit `playground/common/randomize.py`:

```bash
# Floor friction: wider range
sed -i 's/minval=0.5, maxval=1.0)/minval=0.4, maxval=1.2)/' playground/common/randomize.py

# KP scale: wider range
sed -i 's/minval=0.9, maxval=1.1/minval=0.8, maxval=1.2/g' playground/common/randomize.py

# COM jitter: wider
sed -i 's/(3,), minval=-0.05, maxval=0.05/(3,), minval=-0.08, maxval=0.08/' playground/common/randomize.py

# Mass scale: wider
sed -i "s/shape=(model.nbody,), minval=0.9, maxval=1.1/shape=(model.nbody,), minval=0.85, maxval=1.15/" playground/common/randomize.py

# Torso mass: wider
sed -i 's/minval=-0.1, maxval=0.1)  # was -0.2, 0.2/minval=-0.15, maxval=0.15)/' playground/common/randomize.py
```

**Train from scratch** (distribution changed significantly):

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --task flat_terrain_backlash \
  --output_dir /workspace/checkpoints/02_wider_randomization
```

**Keep if:** avg_steps is within 15% of previous best AND fall rate didn't increase by more than 10 percentage points. Slightly lower peak performance is acceptable for better sim-to-real robustness.

---

### Cycle B: Increase Imitation Weight

**Why:** The imitation reward encourages the policy to match a natural reference gait. Doubling its weight should produce more natural-looking locomotion.

Edit `playground/open_duck_mini_v2/joystick.py`:

```bash
sed -i 's/imitation=1.0/imitation=2.0/' playground/open_duck_mini_v2/joystick.py
```

**Fine-tune from best checkpoint:**

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 75000000 \
  --task flat_terrain_backlash \
  --restore_checkpoint_path /workspace/checkpoints/<BEST_SO_FAR>/<checkpoint_dir> \
  --output_dir /workspace/checkpoints/03_imitation_x2
```

**Keep if:** tracking scores improved or stayed the same, AND fall rate didn't increase significantly.

---

### Cycle C: Fix Imitation Reward Structure

**Why:** In `custom_rewards.py`, the joint position imitation term uses negative squared error (always negative = always a cost), while velocity terms use exponential (bounded 0-1 = proper reward). This asymmetry means the imitation reward acts more like a penalty than a reward. Changing to exponential provides better gradient signal near the reference motion.

Edit `playground/open_duck_mini_v2/custom_rewards.py`:

```bash
# Line 127: change joint_pos from negative squared to exponential
sed -i 's/joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) \* w_joint_pos/joint_pos_rew = jp.exp(-5.0 * jp.sum(jp.square(joint_pos - ref_joint_pos))) * w_joint_pos/' playground/open_duck_mini_v2/custom_rewards.py

# Line 128: change joint_vel to match
sed -i 's/joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) \* w_joint_vel/joint_vel_rew = jp.exp(-1.0 * jp.sum(jp.square(joint_vel - ref_joint_vels))) * w_joint_vel/' playground/open_duck_mini_v2/custom_rewards.py
```

**Train from scratch** (significant reward landscape change):

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --task flat_terrain_backlash \
  --output_dir /workspace/checkpoints/04_exp_imitation
```

**This is a high-risk, high-reward change.** If it helps, it's likely the single biggest improvement. If metrics drop significantly, revert:

```bash
# Revert joint_pos
sed -i 's/joint_pos_rew = jp.exp(-5.0 \* jp.sum(jp.square(joint_pos - ref_joint_pos))) \* w_joint_pos/joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) * w_joint_pos/' playground/open_duck_mini_v2/custom_rewards.py
# Revert joint_vel
sed -i 's/joint_vel_rew = jp.exp(-1.0 \* jp.sum(jp.square(joint_vel - ref_joint_vels))) \* w_joint_vel/joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) * w_joint_vel/' playground/open_duck_mini_v2/custom_rewards.py
```

---

### Cycle D: Tune Tracking / Smoothness / Stability

Based on evaluation results so far, pick ONE issue to fix. Apply the corresponding edit, fine-tune from best checkpoint for 75M steps, evaluate, and keep/revert.

**Option 1 — If jerky (high action_rate in eval):**
```bash
sed -i 's/action_rate=-0.5/action_rate=-1.0/' playground/open_duck_mini_v2/joystick.py
```

**Option 2 — If poor velocity tracking (low Track LinVel/AngVel in eval):**
```bash
sed -i 's/tracking_lin_vel=2.5/tracking_lin_vel=4.0/' playground/open_duck_mini_v2/joystick.py
sed -i 's/tracking_ang_vel=6.0/tracking_ang_vel=8.0/' playground/open_duck_mini_v2/joystick.py
```

**Option 3 — If falls too much (high fall rate in eval):**
```bash
sed -i 's/alive=20.0/alive=25.0/' playground/open_duck_mini_v2/joystick.py
```

**Option 4 — If tracking rewards plateau below 0.5 (sigma too tight):**
```bash
sed -i 's/tracking_sigma=0.01/tracking_sigma=0.025/' playground/open_duck_mini_v2/joystick.py
```

After each change, train and evaluate:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 75000000 \
  --task flat_terrain_backlash \
  --restore_checkpoint_path /workspace/checkpoints/<BEST_SO_FAR>/<checkpoint_dir> \
  --output_dir /workspace/checkpoints/05_tuning_<descriptive_name>

uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/05_tuning_<descriptive_name>/*.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml \
  --num_episodes 5
```

You may repeat Cycle D with different options if multiple issues need fixing. Always apply ONE change at a time.

---

## Step 5: Final Training

After all tuning cycles, take the BEST configuration (all changes that helped, none that hurt) and run a full 200M step training:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --num_timesteps 200000000 \
  --task flat_terrain_backlash \
  --output_dir /workspace/checkpoints/final_policy
```

## Step 6: Final Evaluation and Checkpoint Selection

Compare the last 5 ONNX checkpoints to pick the absolute best:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o $(ls -t /workspace/checkpoints/final_policy/*.onnx | head -5 | tr '\n' ' ') \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml \
  --num_episodes 10
```

Pick the one with the best balance of: highest avg steps, lowest fall rate, best tracking scores.

Also test robustness on rough terrain:

```bash
uv run playground/open_duck_mini_v2/eval_headless.py \
  -o /workspace/checkpoints/final_policy/<BEST>.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_rough_terrain_backlash.xml \
  --num_episodes 10
```

## Step 7: Report Results

Create a final summary with:
1. Comparison table of all training runs (baseline through final)
2. Which tuning changes helped vs hurt
3. Path to the best ONNX file on the remote
4. The best ONNX file should be copied to local machine at `~/Downloads/BEST_WALK_ONNX_custom.onnx`

```bash
scp -P <SSH_PORT> root@<INSTANCE_IP>:/workspace/checkpoints/final_policy/<BEST>.onnx ~/Downloads/BEST_WALK_ONNX_custom.onnx
```

## Key Reference

### Observation space: 101 dimensions
- gyro (3) + accelerometer (3) + command (7) + joint_angles (14) + joint_vel (14) + last_3_actions (42) + motor_targets (14) + contacts (2) + imitation_phase (2)

### Action space: 14 dimensions
- 10 leg joints + 4 head joints. action_scale=0.25. motor_targets = default_actuator + action * 0.25

### Known Bugs (FIXED)
1. **`mujoco_infer_base.py:get_gravity()` — CRITICAL:** Used sensor ID as sensordata index instead of sensor address. `mj_name2id()` returns the sensor array index (0-based), NOT the byte offset into `sensordata`. Must use `model.sensor_adr[sensor_id]` to get the correct address. Without this fix, `is_fallen()` reads the velocity sensor instead of the upvector, causing ALL policies to appear to fall immediately. **Fix:** Added `self.gravity_addr = self.model.sensor_adr[self.gravity_id]` and changed `get_gravity()` to use `self.gravity_addr`.
2. **`eval_headless.py:imitation_phase` — Minor:** Initial value was `[1.0, 0.0]` but training initializes at `[0.0, 0.0]`. Fixed to match training.
3. **`joystick.py:accelerometer` — Non-issue for training:** The line `accelerometer.at[0].set(accelerometer[0] + 1.3)` is a JAX no-op (returns new array but doesn't assign it). Training does NOT add 1.3 to accelerometer. The eval script adds 1.3 — this is a mismatch but doesn't cause catastrophic failure since ONNX normalization handles it.

### Lessons Learned
- **SSH environment:** On Vast.ai, always use `bash -l -c '...'` to get the conda/uv environment. Non-login shells miss the env activation.
- **Output buffering:** Python buffers stdout when piped through `tee`. Always use `PYTHONUNBUFFERED=1 uv run python -u ...` for training and eval.
- **Timing:** RTX 3060 does ~2.16M steps/min. 150M steps ≈ 70 min. JIT compilation takes ~3-5 min before first step appears.
- **Goal:** Beat `BEST_WALK_ONNX_2.onnx` from the Open_Duck_Mini repo. This policy works on the real robot.

### Files you will edit on the remote:
- `playground/common/randomize.py` — domain randomization ranges
- `playground/open_duck_mini_v2/joystick.py` — reward weights, tracking_sigma
- `playground/open_duck_mini_v2/custom_rewards.py` — imitation reward structure

### Eval output format:
```
================================================================================
Policy: 2026_05_21_143025_150000000.onnx
================================================================================
Command          Avg Steps  Fall Rate  Track Lin  Track Ang   Act Rate
--------------------------------------------------------------------------
stand                980/1000        0%      0.952      0.987     0.0012
forward              850/1000       10%      0.834      0.912     0.0023
backward             920/1000        4%      0.891      0.945     0.0018
...

Overall: avg_steps=890, fall_rate=4%
```

### Decision criteria for keeping/reverting changes:
- **Keep** if: overall avg_steps increased OR fall rate decreased, without regressing the other metric by more than 10%
- **Revert** if: fall rate increased by more than 15 percentage points, OR avg_steps dropped by more than 20%
- **Judgment call**: if metrics are roughly equal, prefer the change that makes the codebase cleaner or more robust (e.g., wider randomization)
