# Phase 3B Training Results

## Current Best Policy

**Best policy:** `28_ang_vel_14/2026_05_23_122116_158597120.onnx` (159M step from 200M ang_vel=14 run)
- Training chain: final_policy(200M) → ang_vel=10(75M+87M) → ang_vel=12(101M) → ang_vel=14(159M of 200M)
- Config: imitation=2.0, tracking_sigma=0.025, tracking_lin_vel=4.0, **tracking_ang_vel=14.0**, alive=25.0, action_rate=-0.5
- Saved locally as: `~/Downloads/BEST_WALK_ONNX_custom_v3.onnx`

**10-episode eval on backlash terrain (0% fall rate):**
| Command | TrackLin | TrackAng |
|---------|----------|----------|
| stand | 0.669 | 0.907 |
| forward | 0.186 | 0.655 |
| forward_fast | 0.170 | 0.654 |
| backward | 0.229 | 0.614 |
| left | 0.054 | 0.675 |
| right | 0.031 | 0.659 |
| turn_left | 0.001 | 0.547 |
| turn_right | 0.000 | 0.526 |

**Runner-up:** `28_ang_vel_14/2026_05_23_121140_129761280.onnx` (130M)
- right Ang **0.688**, turn_left **0.572**, forward Lin 0.195
- stand Lin 0.603 (lower), but best right/turn metrics

**vs BEST_WALK_2:**
| Command | New Lin | New Ang | Best2 Lin | Best2 Ang |
|---------|---------|---------|-----------|-----------|
| stand | 0.669 | 0.907 | 0.755 | 0.883 |
| forward | 0.186 | 0.655 | 0.095 | 0.520 |
| backward | 0.229 | 0.614 | 0.073 | 0.357 |
| left | 0.054 | 0.675 | 0.044 | 0.606 |
| right | 0.031 | 0.659 | 0.052 | 0.553 |
| turn_left | 0.001 | 0.547 | 0.001 | 0.322 |
| turn_right | 0.000 | 0.526 | 0.000 | 0.286 |

**Beats BEST_WALK_2 on 11/14 metrics!** Only behind on stand Lin (0.669 vs 0.755) and right Lin (0.031 vs 0.052).

---

## All Experiments (Phase 3B, exp 22-28)

### Exp 22: Lead A — 200M from 72M checkpoint (ang_vel=10, alive=25)
**Result:** Forward performance did NOT persist. Best 130M (fwd 0.167/0.506).
**Verdict:** Not better.

### Exp 23: Lead B — 200M ang_vel=12 (**BREAKTHROUGH**)
**Result:** 101M became the best — forward Ang 0.627, right Lin 0.082. 115M had 12% falls.
**Verdict:** ang_vel=12 was the breakthrough.

### Exp 24: Lead C — 200M alive=27.5, ang_vel=10
**Result:** 130M had best turns (0.592/0.584), backward 0.267/0.664. But forward Ang limited to ~0.56.
**Verdict:** alive=27.5 best for turns but needs ang_vel=12+ to match forward.

### Exp 25: Lead D — 200M ang_vel=12 + alive=27.5 combo
**Result:** 101M backward 0.261/0.669, turn_left 0.596. But forward Ang 0.607, right Lin 0.049 (worse than best).
**Verdict:** Good backward/turn_left but weaker forward/right.

### Exp 26: Lead H — 200M from exp 23 87M checkpoint (best fwd Ang 0.638)
**Result:** Forward dominance lost again. 101M had good backward (0.268/0.693) but weak turns. 115M+130M had 12% falls.
**Verdict:** Forward-dominant checkpoints lose their advantage under extended training.

### Exp 27: Lead E — 200M sigma=0.03 (was 0.025)
**Result:** 130M had 12% falls. 144M forward Lin 0.183 but forward Ang 0.550. Wider sigma made training less stable.
**Verdict:** sigma=0.03 too wide. sigma=0.025 remains optimal.

### Exp 28: Lead F — 200M ang_vel=14 (**NEW BREAKTHROUGH**)
**Result:**
- **159M (NEW BEST)**: forward Ang **0.655**, forward_fast Ang **0.654**, stand Ang **0.907**, left Ang **0.675**, turn_right **0.526**
- 130M: right Ang **0.688**, turn_left **0.572**, forward Lin 0.195, forward Ang 0.612
- 144M: 12% falls — disqualified
**Verdict:** ang_vel=14 produces the highest forward Ang (0.655) and stand Ang (0.907) ever with 0% falls. The ang_vel trend (6→8→10→12→14) continues to improve forward/turn metrics.

---

## Key Patterns & Meta-Analysis

1. **tracking_ang_vel is the dominant parameter** — each increase improves forward Ang:
   - ang_vel=6: ~0.45, ang_vel=8: ~0.50, ang_vel=10: ~0.52, ang_vel=12: 0.627, ang_vel=14: 0.655

2. **Tension between angular and lateral tracking:**
   - Higher ang_vel → better forward/turn Ang but WORSE right/left Lin
   - ang_vel=12: right Lin 0.082, ang_vel=14: right Lin 0.031 (-62%)

3. **alive=25 is the sweet spot** — 20 causes falls, 27.5 helps turns but hurts forward, 30 causes falls

4. **sigma=0.025 is optimal** — 0.03 causes instability and falls

5. **Forward-dominant checkpoints are fragile** — extending them loses the advantage (exp 22, 26)

6. **Best checkpoints at 100-160M in 200M runs** — later checkpoints degrade or fall

---

## Remaining Leads for Next Agent

1. **Lead J: ang_vel=16** — The trend keeps going. May push forward Ang past 0.655 but right Lin will likely drop further.

2. **Lead K: ang_vel=14 + alive=27.5** — Combine the new best with alive=27.5 to potentially recover turn_right (exp 25 showed this combo helps turns).

3. **Lead L: ang_vel=14 + tracking_lin_vel=5.0** — Try higher lin_vel with higher ang_vel to balance the right/left Lin loss. The hope is lin_vel=5.0 compensates for the lateral loss.

4. **Lead M: Fine-tune exp 28 130M checkpoint (best right/turn)** — This had right Ang 0.688, turn_left 0.572. Extend to see if forward catches up.

5. **Lead N: Imitation=2.5 or 3.0** — Higher imitation may improve gait stability, helping with the right Lin issue.

## All Key Checkpoints

| Dir | Best Checkpoint | Key Strengths | Status |
|-----|-----------------|---------------|--------|
| **28_ang_vel_14** | **159M (122116)** | **fwd Ang 0.655, stand Ang 0.907, left Ang 0.675** | **CURRENT BEST** |
| 28_ang_vel_14 | 130M (121140) | right Ang 0.688, turn_left 0.572 | Runner-up |
| 23_ang_vel_12 | 101M (051941) | right Lin 0.082, fwd 0.177/0.627 | Previous best |
| 23_ang_vel_12 | 87M (051451) | fwd Ang 0.638, right Ang 0.704 | Best fwd Ang |
| 24_alive_275 | 130M (065222) | backward 0.267/0.664, turns 0.592/0.584 | Best turns |
| 25_ang12_alive275 | 101M (080211) | backward 0.261/0.669, turn_left 0.596 | Best backward+turn |
| 21_v3_200m | 87M (023823) | stand 0.777, backward 0.269/0.654 | Original v3 best |

## Remote Machine State
- Config on remote: tracking_ang_vel=14.0, alive=25.0, sigma=0.025 (from exp 28)
- No training currently running
- All checkpoints in /workspace/checkpoints/21-28/
