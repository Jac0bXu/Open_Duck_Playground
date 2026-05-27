# Week 2: Training + Sim Evaluation Results

## Training

**Hardware:** Vast.ai RTX 3090 (24GB VRAM)
**Config:** 100 epochs, batch_size=256, lr=1e-4, CosineAnnealing, AdamW (weight_decay=1e-4)
**Dataset:** 789,700 valid samples from 4,755 episodes (865,780 total steps)
**Model:** ConditionalUnet1D, 16.4M parameters
**Time:** ~85s/epoch, ~142 min total

### Loss Curve

| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|----------|----|
| 1 | 0.0853 | 0.0556 | 1.00e-04 |
| 10 | 0.0345 | 0.0337 | 9.76e-05 |
| 25 | 0.0288 | 0.0289 | 8.54e-05 |
| 50 | 0.0238 | 0.0242 | 3.85e-05 |
| 75 | 0.0223 | 0.0231 | 1.31e-05 |
| 90 | 0.0216 | 0.0222 | 2.45e-06 |
| 100 | 0.0215 | 0.0221 | 0.00e+00 |

Best checkpoint: val=0.0220 at epoch 98.

Loss converged around epoch 60-70, with marginal improvement thereafter. No overfitting observed (train ≈ val throughout).

## Sim Evaluation

**Setup:** 8 command presets, 5 episodes each, 1000 physics steps (100 control steps at decimation=10) per episode. Same MuJoCo scene (`scene_flat_terrain_backlash.xml`), same metrics.

### PPO Baseline (BEST_WALK_ONNX_2.onnx)

| Command | Avg Steps | Fall Rate | Track Lin | Track Ang | Act Rate |
|---------|-----------|-----------|-----------|-----------|----------|
| stand | 100/1000 | 0% | 0.809 | 0.872 | 0.0532 |
| forward | 100/1000 | 0% | 0.043 | 0.417 | 0.2053 |
| forward_fast | 100/1000 | 0% | 0.049 | 0.441 | 0.2080 |
| backward | 100/1000 | 0% | 0.107 | 0.351 | 0.2627 |
| left | 100/1000 | 0% | 0.044 | 0.559 | 0.1969 |
| right | 100/1000 | 0% | 0.033 | 0.496 | 0.1774 |
| turn_left | 100/1000 | 0% | 0.002 | 0.157 | 0.2323 |
| turn_right | 100/1000 | 0% | 0.002 | 0.263 | 0.2379 |
| **Overall** | **100** | **0%** | | | |

### Diffusion Policy (best.pt, 100 epochs)

| Command | Avg Steps | Fall Rate | Track Lin | Track Ang | Act Rate |
|---------|-----------|-----------|-----------|-----------|----------|
| stand | 42/1000 | 80% | 0.125 | 0.234 | 0.3571 |
| forward | 9/1000 | 100% | 0.000 | 0.000 | 0.4656 |
| forward_fast | 5/1000 | 100% | 0.000 | 0.000 | 1.5420 |
| backward | 3/1000 | 100% | 0.000 | 0.000 | 2.9003 |
| left | 3/1000 | 100% | 0.000 | 0.000 | 4.6738 |
| right | 4/1000 | 100% | 0.000 | 0.000 | 3.9144 |
| turn_left | 4/1000 | 100% | 0.000 | 0.000 | 2.5945 |
| turn_right | 4/1000 | 100% | 0.000 | 0.000 | 2.7959 |
| **Overall** | **9** | **98%** | | | |

### Head-to-Head Comparison

| Metric | PPO Baseline | Diffusion BC | Delta |
|--------|-------------|-------------|-------|
| Avg survival steps | 100/1000 | 9/1000 | -91% |
| Fall rate | 0% | 98% | +98pp |
| Stand tracking (lin) | 0.809 | 0.125 | -85% |
| Stand tracking (ang) | 0.872 | 0.234 | -73% |
| Stand action rate | 0.053 | 0.357 | +6.7x |

## Analysis

The BC diffusion policy fails catastrophically at locomotion. Only the `stand` command shows any survival (20% of episodes reach max steps), and even there tracking quality is poor. Motion commands cause immediate falls.

### Root causes (likely)

1. **Compounding error in BC.** The diffusion policy learns a mapping from obs→action chunks, but any small deviation from the training distribution causes the observations to drift. Once the obs leaves the training manifold, the predicted actions degrade, causing further drift. This is the well-known distribution shift problem in behavior cloning.

2. **Action smoothness.** The diffusion policy's action rate (0.36–4.67) is 5-50x higher than PPO's (0.05–0.26). Even with DDIM's deterministic sampling, the denoised action chunks have high inter-step variance, producing jerky motions that destabilize the biped.

3. **Unimodal teacher limitation.** As noted in the project plan, distilling from a single PPO policy produces a unimodal action distribution. The diffusion model cannot capture the diversity of recovery behaviors that PPO implicitly uses to stay upright.

### What this demonstrates

This result is not a failure — it is an honest characterization of where BC-from-distillation breaks down on bipedal locomotion:

- **PPO survives** because it was trained online with RL, optimizing directly for balance and tracking. It has implicit recovery behaviors shaped by the reward signal.
- **BC diffusion fails** because it only mimics successful trajectories. Without the corrective feedback loop of RL, it cannot recover from the inevitable small errors that accumulate during deployment.

This gap is precisely what DPPO (Ren et al., 2024) addresses: by combining diffusion policy's expressive action representation with RL's corrective feedback loop. The natural next step from these results is to apply DPPO's on-policy refinement to the BC-initialized diffusion policy.

## Artifacts

- Checkpoint: `checkpoints/diffusion/best.pt` (187MB, val=0.0220 at epoch 98)
- Training code: `diffusion/train.py`, `diffusion/model.py`, `diffusion/dataset.py`
- Eval code: `diffusion/eval_diffusion.py`, `playground/open_duck_mini_v2/eval_headless.py`
- Dataset: `data/ppo_rollouts_5k.hdf5` (316MB, 4755 episodes)
