# Agent Prompt: Continue Training Open Duck Mini V2 (Phase 2)

You are continuing an RL training pipeline for the Open Duck Mini v2 robot. Steps 1-3 are COMPLETE. Step 4 Cycle A was interrupted (partial run). You need to evaluate the partial Cycle A, then continue with the remaining tuning cycles and final training.

## Remote Machine

```
Host: 120.238.149.205
Port: 30897
User: root
GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB
SSH: ssh -p 30897 -o StrictHostKeyChecking=no root@120.238.149.205
```

**IMPORTANT:**
- Always use `bash -l -c '...'` for SSH commands to activate the conda/uv environment
- Always prefix training/eval commands with `PYTHONUNBUFFERED=1` when piping through `tee`
- On first connect, verify:
  1. `nvidia-smi` — GPU visible
  2. `cat /sys/fs/cgroup/pids.max` — MUST be > 1024 (512 is too low for JAX/CUDA)
  3. `OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4 uv run python -c "import jax; print(jax.devices())"` — JAX sees GPU
  4. Run a 1M step smoke test before long training
- Always set these env vars for training/eval: `OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4`

## Goal

Beat `BEST_WALK_ONNX_2.onnx` (on remote at `/workspace/Open_Duck_Playground/BEST_WALK_ONNX_2.onnx`). Its eval metrics on scene_flat_terrain:

| Command | TrackLin | TrackAng | Act Rate |
|---------|----------|----------|----------|
| stand | 0.755 | 0.883 | 0.051 |
| forward | 0.095 | 0.520 | 0.221 |
| backward | 0.073 | 0.357 | 0.223 |
| left | 0.044 | 0.606 | 0.206 |
| right | 0.052 | 0.553 | 0.167 |
| turn_left | 0.001 | 0.322 | 0.229 |
| turn_right | 0.000 | 0.286 | 0.232 |

## Current State

### Completed
- **Step 1:** Environment setup, JAX GPU verified, smoke test passed
- **Step 2:** Baseline training (150M steps, flat_terrain) — COMPLETE
  - Best baseline checkpoint: `/workspace/checkpoints/00_baseline/2026_05_21_151946_140574720/`
  - 0% fall rate, all survive 1000 steps
- **Step 3:** Backlash fine-tune (75M steps) — COMPLETE
  - 15 checkpoints in `/workspace/checkpoints/01_backlash_finetune/`
  - **Best backlash checkpoint: `2026_05_22_022811_64880640.onnx`** (0% fall rate)
  - Best backlash eval metrics (on scene_flat_terrain_backlash.xml, 5 episodes):

    | Command | TrackLin | TrackAng | Act Rate |
    |---------|----------|----------|----------|
    | stand | 0.708 | 0.829 | 0.034 |
    | forward | 0.151 | 0.503 | 0.185 |
    | backward | 0.058 | 0.384 | 0.214 |
    | left | 0.039 | 0.409 | 0.194 |
    | right | 0.045 | 0.437 | 0.135 |
    | turn_left | 0.001 | 0.340 | 0.255 |
    | turn_right | 0.004 | 0.332 | 0.195 |

  - Beats BEST_WALK_ONNX_2 on: forward Lin (+59%), turn_left Ang (+6%), turn_right Ang (+16%), backward Ang (+8%), action_rate (smoother)
  - Worse than BEST_WALK_ONNX_2 on: left Ang (-33%), right Ang (-21%)

### Interrupted
- **Step 4 Cycle A: Wider Domain Randomization** — PARTIAL (killed at ~54M/150M steps)
  - 6 checkpoints in `/workspace/checkpoints/02_wider_randomization/`
  - Reward plateaued at ~251 (vs backlash peak ~277)
  - `randomize.py` already modified with wider ranges (backup at `randomize.py.bak`)
  - **Decision needed:** evaluate the 6 partial checkpoints, or restart the full 150M run

### Not Started
- Step 4 Cycles B-D
- Step 5: Final 200M training
- Step 6: Final evaluation
- Step 7: Report and copy best ONNX to ~/Downloads/

## Already Fixed Bugs (do NOT re-fix)
- `mujoco_infer_base.py:get_gravity()` — Fixed: uses `sensor_adr[sensor_id]` instead of `sensor_id`
- `eval_headless.py` — Fixed: imitation_phase init changed to `[0.0, 0.0]`, accelerometer 1.3 offset removed
- These fixes are already on the remote machine

## Step 4: Iterative Tuning

### Where to Start

**First:** Decide whether to continue or skip Cycle A:
```bash
# Option 1: Evaluate the 6 partial wider-randomization checkpoints
ssh ... "bash -l -c 'cd /workspace/Open_Duck_Playground && PYTHONUNBUFFERED=1 uv run python -u playground/open_duck_mini_v2/eval_headless.py -o /workspace/checkpoints/02_wider_randomization/*.onnx --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml --num_episodes 5 2>&1 | tee /workspace/eval_wider_random.log'"

# Option 2: Revert wider randomization and skip Cycle A
ssh ... "bash -l -c 'cd /workspace/Open_Duck_Playground && cp playground/common/randomize.py.bak playground/common/randomize.py'"
```

**If skipping Cycle A**, use best backlash checkpoint (`/workspace/checkpoints/01_backlash_finetune/2026_05_22_022811_64880640.onnx`) as the starting point for fine-tunes. Its checkpoint dir is: `/workspace/checkpoints/01_backlash_finetune/2026_05_22_022811_64880640/`

### For EACH cycle:
1. Edit the specified file on the remote (use `sed`)
2. Train (from scratch or fine-tune, as specified)
3. Evaluate on backlash XML
4. Compare to BEST result so far (include BEST_WALK_ONNX_2 comparison)
5. **Keep the change if metrics improved. Revert if worse.**
6. Record results in `/workspace/Open_Duck_Playground/TRAINING_LOG.md` on the remote

**Always use backlash task for training and evaluation.**

Run training in tmux session `train`:
```bash
ssh ... "bash -l -c 'tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train \"cd /workspace/Open_Duck_Playground && OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4 PYTHONUNBUFFERED=1 uv run python -u playground/open_duck_mini_v2/runner.py ... 2>&1 | tee /workspace/train_<name>.log\"'"
```

Check progress periodically:
```bash
ssh ... "bash -l -c 'grep STEP /workspace/train_<name>.log | tail -3; echo \"---\"; ls /workspace/checkpoints/<dir>/*.onnx 2>/dev/null | wc -l'"
```

### Cycle A: Wider Domain Randomization (may be skipped)

Current state: `randomize.py` already modified. Backup at `randomize.py.bak`.

Changes already applied:
- `minval=0.5, maxval=1.0` → `minval=0.4, maxval=1.2`
- `minval=0.9, maxval=1.1` → `minval=0.8, maxval=1.2` (all instances)
- `minval=-0.05, maxval=0.05` → `minval=-0.08, maxval=0.08`
- `minval=0.9, maxval=1.1` (nbody) → `minval=0.85, maxval=1.15`
- `minval=-0.1, maxval=0.1` → `minval=-0.15, maxval=0.15`

**Train from scratch** (150M steps):
```bash
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 150000000 --task flat_terrain_backlash --output_dir /workspace/checkpoints/02_wider_randomization
```

Keep if: fall rate stayed at 0% and tracking is within 15% of previous best.

Revert if needed:
```bash
cp playground/common/randomize.py.bak playground/common/randomize.py
```

### Cycle B: Increase Imitation Weight

```bash
sed -i 's/imitation=1.0/imitation=2.0/' playground/open_duck_mini_v2/joystick.py
```

**Fine-tune from best checkpoint** (75M steps):
```bash
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 75000000 --task flat_terrain_backlash --restore_checkpoint_path /workspace/checkpoints/<BEST_SO_FAR>/<dir> --output_dir /workspace/checkpoints/03_imitation_x2
```

Revert if needed:
```bash
sed -i 's/imitation=2.0/imitation=1.0/' playground/open_duck_mini_v2/joystick.py
```

### Cycle C: Fix Imitation Reward Structure

```bash
sed -i 's/joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) \* w_joint_pos/joint_pos_rew = jp.exp(-5.0 * jp.sum(jp.square(joint_pos - ref_joint_pos))) * w_joint_pos/' playground/open_duck_mini_v2/custom_rewards.py
sed -i 's/joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) \* w_joint_vel/joint_vel_rew = jp.exp(-1.0 * jp.sum(jp.square(joint_vel - ref_joint_vels))) * w_joint_vel/' playground/open_duck_mini_v2/custom_rewards.py
```

**Train from scratch** (150M steps). High-risk, high-reward. If metrics drop, revert with:
```bash
sed -i 's/joint_pos_rew = jp.exp(-5.0 \* jp.sum(jp.square(joint_pos - ref_joint_pos))) \* w_joint_pos/joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) * w_joint_pos/' playground/open_duck_mini_v2/custom_rewards.py
sed -i 's/joint_vel_rew = jp.exp(-1.0 \* jp.sum(jp.square(joint_vel - ref_joint_vels))) \* w_joint_vel/joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) * w_joint_vel/' playground/open_duck_mini_v2/custom_rewards.py
```

### Cycle D: Tune Tracking / Smoothness / Stability

Based on eval results, pick ONE issue:
- **Option 1 — Jerky (high action_rate):** `sed -i 's/action_rate=-0.5/action_rate=-1.0/' playground/open_duck_mini_v2/joystick.py`
- **Option 2 — Poor velocity tracking:** `sed -i 's/tracking_lin_vel=2.5/tracking_lin_vel=4.0/' playground/open_duck_mini_v2/joystick.py` AND `sed -i 's/tracking_ang_vel=6.0/tracking_ang_vel=8.0/' playground/open_duck_mini_v2/joystick.py`
- **Option 3 — Falls too much:** `sed -i 's/alive=20.0/alive=25.0/' playground/open_duck_mini_v2/joystick.py`
- **Option 4 — Tracking plateau (sigma too tight):** `sed -i 's/tracking_sigma=0.01/tracking_sigma=0.025/' playground/open_duck_mini_v2/joystick.py`

**Fine-tune from best** (75M steps). May repeat with different options.

## Step 5: Final Training

Take the BEST configuration (all helpful changes combined) and run 200M steps:
```bash
uv run playground/open_duck_mini_v2/runner.py --num_timesteps 200000000 --task flat_terrain_backlash --output_dir /workspace/checkpoints/final_policy
```

## Step 6: Final Evaluation

Compare last 5 checkpoints:
```bash
uv run playground/open_duck_mini_v2/eval_headless.py -o $(ls -t /workspace/checkpoints/final_policy/*.onnx | head -5 | tr '\n' ' ') --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml --num_episodes 10
```

Also test on rough terrain:
```bash
uv run playground/open_duck_mini_v2/eval_headless.py -o /workspace/checkpoints/final_policy/<BEST>.onnx --model_path playground/open_duck_mini_v2/xmls/scene_rough_terrain_backlash.xml --num_episodes 10
```

## Step 7: Report

1. Update TRAINING_LOG.md with all results
2. Compare final policy vs BEST_WALK_ONNX_2 in a table
3. Copy best ONNX to local: `scp -P 30897 root@120.238.149.205:/workspace/checkpoints/final_policy/<BEST>.onnx ~/Downloads/BEST_WALK_ONNX_custom.onnx`

## Timing Reference
- RTX 4060 Ti 16GB: estimate speed on first long run (~236K steps/min observed in short smoke test with heavy checkpoint overhead; should be faster in long runs)
- For reference, RTX 3060 did ~2.16M steps/min on long runs
- 75M steps ≈ 35 min (at 3060 speed)
- 150M steps ≈ 70 min (at 3060 speed)
- 200M steps ≈ 93 min (at 3060 speed)
- Eval (1 checkpoint, 5 episodes): ~1 min
- Measure actual speed from first long run and update estimates

## Decision Criteria
- **Keep** if: tracking improved without fall rate increasing >10 percentage points
- **Revert** if: fall rate increased >15 percentage points OR tracking dropped >20%
- **Judgment call**: if equal, prefer changes that improve sim-to-real robustness (wider randomization)
