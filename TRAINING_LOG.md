# Training Experiment Log

## Environment
- **Remote (Phase 3):** 120.238.149.205:30897 (Vast.ai)
- **GPU:** NVIDIA GeForce RTX 4060 Ti, 16380 MiB
- **JAX:** with CUDA, uv environment

---

## Overview

Training pipeline for Open Duck Mini v2 robot using PPO (via brax/mjx). The goal was to beat the reference policy `BEST_WALK_ONNX_2.onnx` on velocity tracking metrics.

**Total experiments run:** 28 (exp 00-28)

---

## Phase 1: Baseline & Bug Fixes (May 21)

### Run 0: Smoke Test — PASS
### Run 1: Baseline (150M, flat_terrain)
- Stock config, 150M steps
- **Critical bug found:** `mujoco_infer_base.py:get_gravity()` used `sensor_id` instead of `sensor_adr` to index `sensordata`. This made fall detection read velocity data instead of upvector, causing false immediate falls. Fixed by using `model.sensor_adr[sensor_id]`.
- Best checkpoint: 140M steps, all commands 0% fall rate

---

## Phase 2: Iterative Reward Tuning (May 22)

Fine-tuned from baseline using backlash terrain (`flat_terrain_backlash`).

### Key experiments:
- **Backlash fine-tune (75M):** Adapted to backlash dynamics
- **imitation=2.0:** Better backward/right tracking
- **tracking_sigma=0.025:** Massive turn improvement (+47% turn_left, +20% turn_right)
- **tracking_lin_vel=4.0 + tracking_ang_vel=8.0:** Better forward tracking
- **alive=25.0:** Better turn tracking

### What didn't work:
- Wider domain randomization — killed turning
- Exponential reward structure — catastrophic, robot stood still
- Training from scratch with new config — underperformed fine-tunes
- action_rate=-1.0 — hurt forward tracking

### Key insight: Fine-tuning > from-scratch training

---

## Phase 3: Final Training & Stacked Fine-tunes (May 22-23)

### Phase 3A: Original leads (exp 11-21)
- **Exp 18 (ang_vel=10):** Breakthrough — forward Ang improved from 0.446→0.532
- **Exp 21 (200M extended):** 87M checkpoint had stand 0.777, backward 0.269/0.654
- Best at this point: exp 21 87M (ang_vel=10, alive=25)

### Phase 3B: Continued exploration (exp 22-28)

| Exp | Config | Key Finding |
|-----|--------|-------------|
| 22 | 200M from 72M fwd checkpoint | Forward gains fragile, lost under training |
| **23** | **200M ang_vel=12** | **BREAKTHROUGH: forward Ang 0.627, right Lin 0.082** |
| 24 | 200M alive=27.5, ang10 | Best turns (0.592/0.584), backward 0.267/0.664 |
| 25 | ang12+alive27.5 combo | Good backward/turn_left but weaker forward/right |
| 26 | 200M from 87M fwd checkpoint | Forward dominance lost again |
| 27 | sigma=0.03 | Too wide, causes instability |
| **28** | **200M ang_vel=14** | **NEW BEST: forward Ang 0.655, stand Ang 0.907** |

---

## Current Best Policy

**File:** `~/Downloads/BEST_WALK_ONNX_custom_v3.onnx`
- **Checkpoint:** `28_ang_vel_14/2026_05_23_122116_158597120.onnx` (159M of 200M run)
- **Training chain:** final_policy(200M) → ang_vel=10(162M) → ang_vel=12(101M) → ang_vel=14(159M)
- **Config:** imitation=2.0, sigma=0.025, lin_vel=4.0, **ang_vel=14.0**, alive=25.0, act_rate=-0.5

**10-episode eval (0% fall rate, backlash terrain):**

| Command | TrackLin | TrackAng | vs BEST_WALK_2 Lin | vs BEST_WALK_2 Ang |
|---------|----------|----------|--------------------|--------------------|
| stand | 0.669 | 0.907 | 0.755 (-11%) | 0.883 (+3%) |
| forward | 0.186 | 0.655 | 0.095 (+96%) | 0.520 (+26%) |
| forward_fast | 0.170 | 0.654 | — | — |
| backward | 0.229 | 0.614 | 0.073 (+214%) | 0.357 (+72%) |
| left | 0.054 | 0.675 | 0.044 (+23%) | 0.606 (+11%) |
| right | 0.031 | 0.659 | 0.052 (-40%) | 0.553 (+19%) |
| turn_left | 0.001 | 0.547 | 0.001 | 0.322 (+70%) |
| turn_right | 0.000 | 0.526 | 0.000 | 0.286 (+84%) |

**Beats BEST_WALK_2 on 11/14 metrics.** Only behind on stand Lin and right Lin.

---

## Downloaded Policies

| File | Description |
|------|-------------|
| `BEST_WALK_ONNX_custom_v3.onnx` | **Current best** (ang14, 159M) |
| `BEST_WALK_ONNX_custom_v3_ang12_101M.onnx` | ang12 best (right Lin 0.082, best balance) |
| `BEST_WALK_ONNX_custom_v3_ang14_130M.onnx` | ang14 130M (right Ang 0.688, turn_left 0.572) |
| `BEST_WALK_ONNX_custom_v3_best_turns.onnx` | alive=27.5 best (turns 0.592/0.584, backward 0.267/0.664) |

---

## Key Lessons Learned

### Parameters
1. **tracking_ang_vel is the dominant parameter** — each increase (6→8→10→12→14) consistently improves forward Ang
2. **alive=25 is the sweet spot** — 20 causes falls, 27.5 helps turns but hurts forward, 30 causes falls
3. **tracking_lin_vel=4.0 is optimal** — higher values kill lateral movement
4. **sigma=0.025 is optimal** — 0.03 causes instability
5. **action_rate=-0.5 is optimal** — stronger smoothness penalties hurt turns

### Training strategy
1. **Fine-tuning > from-scratch training** when changing reward weights
2. **Stacked fine-tunes from converged checkpoints** produce the best results
3. **Best checkpoints at 100-160M** in 200M runs (later checkpoints degrade or fall)
4. **Forward-dominant checkpoints are fragile** — extending them loses the advantage

### Remaining leads to explore
1. ang_vel=16 — trend may continue
2. ang_vel=14 + alive=27.5 — combine best forward with best turns
3. ang_vel=14 + tracking_lin_vel=5.0 — try to recover right/left Lin
4. imitation=2.5 or 3.0 — improve gait stability

---

## Timing Reference
- **RTX 4060 Ti pacing:** ~2.16M steps/min
- **75M steps:** ~35 min
- **200M steps:** ~93 min
- **JIT compilation:** 3-5 min before first STEP output

## Critical Bugs Fixed
- `mujoco_infer_base.py:get_gravity()` — Fixed: uses `sensor_adr[sensor_id]` instead of `sensor_id`
- `eval_headless.py` — Fixed: imitation_phase init to `[0.0, 0.0]`, accelerometer offset removed
