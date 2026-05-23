# Agent Prompt: Continue Training Open Duck Mini V2 (Phase 3)

You are continuing an RL training pipeline for the Open Duck Mini v2 robot. The baseline, backlash fine-tune, iterative tuning cycles, and a first final training are COMPLETE. You now need to squeeze out further improvements by exploring promising leads identified from the Phase 2 experiments.

## Remote Machine

```
Host: 120.238.149.205
Port: 30897
User: root
GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB
SSH: ssh -p 30897 -o StrictHostKeyChecking=no root@120.238.149.205
```

**IMPORTANT operational lessons from Phase 2:**
- Always use `bash -l -c '...'` for SSH commands to activate the conda/uv environment
- Always prefix training/eval commands with `PYTHONUNBUFFERED=1` when piping through `tee`
- **CRITICAL: OOM fix** — You MUST set these env vars for ALL training/eval commands or the RTX 4060 Ti will OOM during checkpoint saving:
  ```
  TF_GPU_ALLOCATOR=cuda_malloc_async OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4
  ```
- Without `TF_GPU_ALLOCATOR=cuda_malloc_async` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, training crashes with OOM after the 2nd checkpoint (ONNX export via TensorFlow doesn't free GPU memory)
- JIT compilation takes 3-5 min before the first STEP output appears — don't panic
- Training speed: ~2.16M steps/min on this GPU
- **Checkpoint filenames change between runs** — always `ls -lt /workspace/checkpoints/<dir>/*.onnx | head -5` to get actual filenames before running eval. The timestamp+step format is NOT predictable.
- `--restore_checkpoint_path` takes a DIRECTORY (e.g., `2026_05_22_094525_201850880/`), NOT the `.onnx` file. Step counter resets to 0 when restoring, but weights are loaded correctly.
- Use tmux session `train` for all training runs
- Each 75M fine-tune takes ~35 min, 150M from scratch ~70 min, 200M from scratch ~93 min

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

## Current Best Policy

**Best policy so far:** `/workspace/checkpoints/10_tracking_alive/2026_05_22_134750_70287360.onnx`
(This is a fine-tune from the 200M final policy, 70.3M additional steps)

**Its metrics (10 episodes, backlash terrain):**

| Command | TrackLin | TrackAng | Act Rate |
|---------|----------|----------|----------|
| stand | 0.690 | 0.876 | 0.043 |
| forward | 0.162 | 0.446 | 0.211 |
| forward_fast | 0.165 | 0.442 | 0.212 |
| backward | 0.161 | 0.572 | 0.245 |
| left | 0.126 | 0.486 | 0.254 |
| right | 0.033 | 0.599 | 0.198 |
| turn_left | 0.003 | 0.438 | 0.273 |
| turn_right | 0.000 | 0.525 | 0.258 |

Fall rate: 0%

**Already beats BEST_WALK_2 on:** forward Lin (+71%), backward Lin (+121%), backward Ang (+60%), left Lin (+186%), right Ang (+8%), turn_left Ang (+36%), turn_right Ang (+84%)

**Still worse than BEST_WALK_2 on:** forward Ang (0.446 vs 0.520, -14%), left Ang (0.486 vs 0.606, -20%), stand Lin (0.690 vs 0.755, -9%), right Lin (0.033 vs 0.052)

## Winning Configuration (currently active on remote)

The joystick.py currently has these values (from the last experiment):
```
imitation=2.0          (was 1.0)
tracking_sigma=0.025   (was 0.01)
tracking_lin_vel=4.0   (was 2.5)
tracking_ang_vel=8.0   (was 6.0)
alive=25.0             (was 20.0)
action_rate=-0.5       (unchanged)
torques=-1.0e-3        (unchanged)
stand_still=-0.2       (unchanged)
```

## All Checkpoint Directories on Remote

| Dir | Description | Best checkpoint |
|-----|-------------|----------------|
| `/workspace/checkpoints/01_backlash_finetune/` | Backlash fine-tune (75M) | `2026_05_22_022811_64880640` |
| `/workspace/checkpoints/03_imitation_x2/` | imitation=2.0 fine-tune | `2026_05_22_062032_37847040` (0% fall) |
| `/workspace/checkpoints/05_tracking_sigma/` | tracking_sigma=0.025 fine-tune | `2026_05_22_081835_54067200` (0% fall) |
| `/workspace/checkpoints/final_policy/` | 200M from scratch, imit=2 + sigma=0.025 | `2026_05_22_094525_201850880` (0% fall) |
| `/workspace/checkpoints/06_tracking_weights/` | tracking_lin=4 + ang=8 fine-tune | `2026_05_22_105821_59473920` (0% fall, forward Ang 0.502!) |
| `/workspace/checkpoints/07_action_rate/` | action_rate=-1.0 fine-tune | `2026_05_22_113947_59473920` (0% fall, turn_right 0.457) |
| `/workspace/checkpoints/08_alive_25/` | alive=25.0 fine-tune | `2026_05_22_122510_70287360` (0% fall, turn_left 0.468, turn_right 0.450) |
| `/workspace/checkpoints/09_tracking_actionrate/` | tracking + action_rate fine-tune | `2026_05_22_130629_70287360` (0% fall, forward Lin 0.162) |
| `/workspace/checkpoints/10_tracking_alive/` | **BEST** — tracking + alive=25 fine-tune | `2026_05_22_134750_70287360` (0% fall) |
| `/workspace/checkpoints/ultimate_policy/` | 200M from scratch with full winning config | Underperformed fine-tunes |
| `/workspace/checkpoints/04_exp_imitation/` | Exponential reward — FAILED, ignore |

## Already Fixed Bugs (do NOT re-fix)
- `mujoco_infer_base.py:get_gravity()` — Fixed: uses `sensor_adr[sensor_id]` instead of `sensor_id`
- `eval_headless.py` — Fixed: imitation_phase init changed to `[0.0, 0.0]`, accelerometer 1.3 offset removed
- These fixes are already on the remote machine

## Lessons Learned — What Worked and What Didn't

### What worked (KEEP):
1. **imitation=2.0** — Better backward/right tracking, more natural gait
2. **tracking_sigma=0.025** — Massive turn improvement (+47% turn_left, +20% turn_right). Wider sigma gives more gradient signal for velocity tracking
3. **tracking_lin_vel=4.0 + tracking_ang_vel=8.0** — Better forward Lin and turn tracking. Forward Ang reached 0.502 in isolation (Exp A)
4. **alive=25.0** — Better turn tracking (turn_left 0.468, turn_right 0.450). Makes robot more conservative during turns, fewer falls

### What didn't work (AVOID):
1. **Wider domain randomization** — Killed turning completely (turn_left Ang dropped from 0.34 to 0.05). The wider ranges made it too hard to learn turning
2. **Exponential reward structure** (exp instead of negative squared) — Catastrophic. Policy learned to stand still instead of tracking commands. The exponential form makes imitation too easy to "satisfice"
3. **Training from scratch with new config** — The 200M "ultimate" run from scratch with the winning config underperformed all fine-tunes. Fine-tuning from a well-converged checkpoint is much more effective
4. **action_rate=-1.0** — Made movements smoother but forward Ang dropped from 0.502 (Exp A) to 0.459 (A+B combo). The stronger smoothness penalty hurt forward tracking

### Key insight:
**Fine-tuning > from-scratch training** when changing reward weights. The base policy is already strong — reward weight changes just need to steer it. A 75M fine-tune from a converged checkpoint consistently beats a 200M from-scratch run.

### The forward Ang puzzle:
- Exp A (tracking weights only, alive=20): forward Ang = **0.502** (best!)
- Exp A+C (tracking weights + alive=25): forward Ang = **0.446** (worse!)
- The alive=25 reward hurts forward Ang. Higher alive = more conservative = robot doesn't push forward as aggressively.
- But alive=25 is essential for turn tracking (turn_right went from 0.408 to 0.525)

## Leads to Explore

### Lead 1: Sequential fine-tune — stack another fine-tune on top of the best A+C checkpoint
**Hypothesis:** Take the A+C checkpoint (best turns) and fine-tune it AGAIN with alive reverted to 20.0 (to recover forward Ang). The turn improvements may persist while forward Ang recovers.

Config: Revert alive to 20.0, keep everything else (tracking weights still high).

```bash
# On remote:
sed -i 's/alive=25.0/alive=20.0/' playground/open_duck_mini_v2/joystick.py

# Fine-tune from A+C best checkpoint:
tmux new-session -d -s train "cd /workspace/Open_Duck_Playground && TF_GPU_ALLOCATOR=cuda_malloc_async OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4 PYTHONUNBUFFERED=1 uv run python -u playground/open_duck_mini_v2/runner.py --num_timesteps 75000000 --task flat_terrain_backlash --restore_checkpoint_path /workspace/checkpoints/10_tracking_alive/2026_05_22_134750_70287360 --output_dir /workspace/checkpoints/11_alive_revert 2>&1 | tee /workspace/train_alive_revert.log"
```

### Lead 2: Push tracking_lin_vel even higher (6.0)
**Hypothesis:** We went from 2.5→4.0 with good results. Try 6.0 to push forward Ang closer to BEST_WALK_2's 0.520. Keep alive=25 for turn stability.

```bash
sed -i 's/tracking_lin_vel=4.0/tracking_lin_vel=6.0/' playground/open_duck_mini_v2/joystick.py
# Also restore alive=25 if changed:
sed -i 's/alive=20.0/alive=25.0/' playground/open_duck_mini_v2/joystick.py

# Fine-tune from final_policy (the 200M base):
--restore_checkpoint_path /workspace/checkpoints/final_policy/2026_05_22_094525_201850880 --output_dir /workspace/checkpoints/12_lin_vel_6
```

### Lead 3: Push alive even higher (30.0)
**Hypothesis:** Going from 20→25 gave a huge turn improvement (+37% turn_left, +16% turn_right). Try 30 to see if the trend continues.

```bash
sed -i 's/alive=25.0/alive=30.0/' playground/open_duck_mini_v2/joystick.py
# Restore tracking_lin_vel=4.0:
sed -i 's/tracking_lin_vel=6.0/tracking_lin_vel=4.0/' playground/open_duck_mini_v2/joystick.py

# Fine-tune from final_policy:
--restore_checkpoint_path /workspace/checkpoints/final_policy/2026_05_22_094525_201850880 --output_dir /workspace/checkpoints/13_alive_30
```

### Lead 4: Continue fine-tuning the A+C policy for another 75M
**Hypothesis:** The A+C policy was only trained for 70.3M steps. It might not have fully converged. Fine-tune it more with the SAME config.

```bash
# Keep current config (tracking weights + alive=25, already set)
# Restore alive=25.0 if changed:
sed -i 's/alive=30.0/alive=25.0/' playground/open_duck_mini_v2/joystick.py

--restore_checkpoint_path /workspace/checkpoints/10_tracking_alive/2026_05_22_134750_70287360 --output_dir /workspace/checkpoints/14_ac_extended
```

### Lead 5: Combine tracking_lin_vel=6.0 + action_rate=-0.75 (moderate smoothness)
**Hypothesis:** action_rate=-1.0 was too aggressive and hurt forward Ang. Try -0.75 as a middle ground. Combined with higher tracking_lin_vel=6.0, this might give both forward Ang and smoothness.

```bash
sed -i 's/tracking_lin_vel=4.0/tracking_lin_vel=6.0/' playground/open_duck_mini_v2/joystick.py
sed -i 's/action_rate=-0.5/action_rate=-0.75/' playground/open_duck_mini_v2/joystick.py

--restore_checkpoint_path /workspace/checkpoints/final_policy/2026_05_22_094525_201850880 --output_dir /workspace/checkpoints/15_lin6_act075
```

## Execution Strategy

Run these experiments **sequentially** (one GPU). For each:
1. Apply the config change with `sed`
2. Launch 75M fine-tune in tmux
3. Wait ~35 min, check completion
4. Evaluate top 3 checkpoints (5 episodes each) on backlash terrain
5. Compare to current best (10_tracking_alive 70.3M) and BEST_WALK_2
6. Record results
7. Proceed to next experiment

**Priority order:** Lead 1 → Lead 2 → Lead 4 → Lead 3 → Lead 5

Lead 1 is highest priority because it directly addresses the biggest weakness (forward Ang 0.446 vs 0.520) while potentially preserving the turn gains.

**After all experiments:** If any single experiment beats the current best, take that checkpoint and do a final evaluation with 10 episodes. Then copy the best `.onnx` to `~/Downloads/BEST_WALK_ONNX_custom_v3.onnx`.

## Evaluation Command Template

```bash
# Always get actual filenames first:
ls -lt /workspace/checkpoints/<dir>/*.onnx | head -5

# Then eval (get filenames from the ls output):
cd /workspace/Open_Duck_Playground && OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4 PYTHONUNBUFFERED=1 uv run python -u playground/open_duck_mini_v2/eval_headless.py -o <checkpoint1.onnx> <checkpoint2.onnx> <checkpoint3.onnx> --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml --num_episodes 5
```

## Training Command Template

```bash
tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train "cd /workspace/Open_Duck_Playground && TF_GPU_ALLOCATOR=cuda_malloc_async OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 TF_CPP_MIN_THREAD_POOL_SIZE=1 TF_CPP_MAX_THREAD_POOL_SIZE=4 PYTHONUNBUFFERED=1 uv run python -u playground/open_duck_mini_v2/runner.py --num_timesteps 75000000 --task flat_terrain_backlash --restore_checkpoint_path <checkpoint_dir> --output_dir /workspace/checkpoints/<output_dir> 2>&1 | tee /workspace/train_<name>.log"
```

## Progress Monitoring

```bash
# Check training progress:
ssh ... "bash -l -c 'grep STEP /workspace/train_<name>.log | tail -3'"

# Check if training is still running:
ssh ... "nvidia-smi | grep python"

# Count checkpoints:
ssh ... "ls /workspace/checkpoints/<dir>/*.onnx | wc -l"
```

## Decision Criteria
- **Keep** if: any tracking metric improved without others dropping >15%
- **New best** if: overall beats the current best policy (10_tracking_alive 70.3M)
- Fall rate must stay at 0% to be considered

## Final Step
After finding the best policy across all experiments:
```bash
scp -P 30897 -o StrictHostKeyChecking=no root@120.238.149.205:/workspace/checkpoints/<best_dir>/<best>.onnx ~/Downloads/BEST_WALK_ONNX_custom_v3.onnx
```
