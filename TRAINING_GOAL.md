# Training Goal

**Target:** Produce a walking policy that outperforms `BEST_WALK_ONNX_2.onnx` (from the Open_Duck_Mini repo).

This policy is known to work on the real robot (sim2real transfer verified by the project authors).

## Baseline for Comparison

`BEST_WALK_ONNX_2.onnx` is a known-good policy from the project authors. It was trained with the same playground codebase (101-dim observation, 14-dim action). We need to beat it on the eval_headless.py metrics.

**BEST_WALK_ONNX_2.onnx metrics (scene_flat_terrain, 5 episodes):**
| Command | TrackLin | TrackAng | Act Rate |
|---------|----------|----------|----------|
| stand | 0.755 | 0.883 | 0.051 |
| forward | 0.095 | 0.520 | 0.221 |
| forward_fast | 0.093 | 0.520 | 0.221 |
| backward | 0.073 | 0.357 | 0.223 |
| left | 0.044 | 0.606 | 0.206 |
| right | 0.052 | 0.553 | 0.167 |
| turn_left | 0.001 | 0.322 | 0.229 |
| turn_right | 0.000 | 0.286 | 0.232 |

Fall rate: 0%

## Our Best So Far

**Run 1 Baseline checkpoint `2026_05_21_151946_140574720.onnx`:**
| Command | TrackLin | TrackAng | Act Rate | vs BEST_WALK_2 |
|---------|----------|----------|----------|----------------|
| stand | 0.747 | **0.920** | 0.049 | Ang better |
| forward | **0.158** | 0.472 | 0.220 | Lin better |
| backward | **0.238** | **0.852** | 0.058 | **Both much better** |
| left | 0.019 | **0.906** | 0.045 | **Ang much better** |
| right | 0.018 | **0.860** | 0.063 | **Ang much better** |
| turn_left | **0.021** | 0.238 | 0.201 | Mixed |
| turn_right | 0.004 | 0.142 | 0.207 | Mixed |

**Already beating BEST_WALK_2 on backward and lateral tracking. Close on forward. Need to improve forward/turn tracking to win overall.**

## Key Metrics to Beat
- **Velocity tracking** (TrackLin, TrackAng): Higher = better follows commands
- **Fall rate**: 0% is the target
- **Action rate**: Lower = smoother movements (better for real robot servos)

## Strategy
1. Backlash fine-tune (simulates real servo gear play)
2. Iterative reward tuning (wider randomization, imitation weight, reward structure)
3. Final long training (200M steps)
