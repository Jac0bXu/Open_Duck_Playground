# Open Duck Mini V2: Lessons from Sim-to-Real on a 15-DOF Bipedal Robot

## Abstract

This document recounts a hands-on project working with the Open Duck Mini V2, a 15-DOF bipedal robot. The work spanned hardware assembly, PPO policy training in MuJoCo MJX, sim-to-real transfer, and an attempt to apply Diffusion Policy via behavior cloning from PPO rollouts. The central outcome is negative: the trained PPO policy that dominates in simulation fails to transfer to hardware, and the BC diffusion policy fails even in simulation. Rather than presenting these as successes, I use them to analyze where each stage broke down and what I would do differently. The reflections below are organized around the mistakes I made and the lessons they produced.

---

## 1. The Platform

The Open Duck Mini V2 is a small bipedal robot with 15 degrees of freedom: 10 leg joints (hip yaw/roll/pitch, knee, ankle per leg) and 5 head/neck joints. It runs on a Raspberry Pi 5 with Feetech STS3215 servos and an onboard IMU. The upstream repository provides a complete PPO training pipeline using Brax in MuJoCo MJX, with automatic ONNX export and a runtime for the Pi.

The robot's appeal is that it's cheap (~$200), open-source, and has a working RL training pipeline. The challenge is that it's a biped — inherently less stable than a quadruped, and far more sensitive to sim-to-real gaps in joint dynamics, mass distribution, and sensor calibration.

---

## 2. Hardware Assembly Mistakes

### 2.1 Missing Loctite on Actuator Screws

The assembly guide I followed did not emphasize that Loctite (threadlocker) is essential for all screws on the actuator joints. I assembled the robot without it. After several sessions of testing and walking, screws on multiple actuators loosened progressively. The loosened screws introduced mechanical play — backlash and friction that the simulated model did not account for.

This is a critical lesson for anyone building a physical robot from a kit: **always use threadlocker on any joint that transmits force**. The Feetech servos produce enough torque to vibrate screws loose over hundreds of gait cycles. The resulting backlash creates a sim-to-real gap that no amount of domain randomization in the training pipeline can close, because the backlash is asymmetric, velocity-dependent, and differs from joint to joint.

The backlash simulation in MuJoCo (`scene_flat_terrain_backlash.xml`) models generic gear play, but it cannot capture the specific pattern of screw-loosening-induced play on a particular robot. This is a hardware problem that demands a hardware fix, not a software one.

### 2.2 Incorrect Neck Joint Initialization

The assembly guide did not provide clear instructions on how to initialize the neck joints (neck_pitch, head_pitch, head_yaw, head_roll) to their zero positions. I suspect I initialized them in the wrong position, which placed the head at an offset from where the simulated model expects it.

For a bipedal robot, this matters more than it might seem. The head assembly has non-trivial mass, and its position affects the center of mass of the entire robot. With the neck joints mis-initialized, the real robot's center of mass is shifted relative to what the policy expects. On a quadruped, a small COM offset is a nuisance. On a biped, it can be the difference between balancing and falling.

**Lesson:** Before any sim-to-real deployment, verify every joint's zero position against the simulated model's home keyframe. The runtime's `init_pos` array must exactly match `keyframe("home").ctrl` from the MJCF model. A misaligned neck is not just a cosmetic issue — it shifts the dynamics.

---

## 3. PPO Policy Training

### 3.1 The Training Pipeline

I trained PPO policies using the MuJoCo MJX pipeline on rented cloud GPUs (Vast.ai, RTX 4060 Ti). Over 28 experiments, I iterated on reward weights through stacked fine-tuning: starting from a baseline policy, progressively adjusting reward scales and continuing training from the best checkpoint.

The key parameter discovery was that `tracking_ang_vel` dominates forward walking quality. Increasing it from the default 6.0 to 14.0 (through a chain of 6→8→10→12→14) steadily improved angular velocity tracking and forward gait, at the cost of lateral tracking. The best policy (experiment 28, 159M steps) achieved:

| Metric | Custom Policy | Original Author Policy |
|--------|--------------|----------------------|
| Stand tracking (angular) | 0.907 | 0.883 |
| Forward tracking (angular) | 0.655 | 0.520 |
| Backward tracking (linear) | 0.229 | 0.073 |
| Turn right (angular) | 0.526 | 0.286 |
| **Metrics improved** | **11/14** | — |

This policy beat the original author's best policy on 11 out of 14 evaluation metrics in simulation.

### 3.2 The Sim-to-Real Gap

Despite dominating in simulation, the custom policy performed poorly on the real robot. The robot could stand and hold position, but walking was unstable — it would fall after a few steps, especially during turns and lateral movement.

Looking back, this is a classic case of **reward overfitting in simulation**. By aggressively optimizing `tracking_ang_vel` through 28 experiments, I produced a policy that exploits the simulated dynamics to achieve high reward scores. But the simulated dynamics — even with backlash modeling — do not match the real robot closely enough. The policy has learned behaviors that are brittle: they work in the narrow band of dynamics the simulator provides but fail when the real dynamics diverge.

The original author's policy, while scoring lower in simulation, was likely validated on real hardware during development. It learned a more conservative gait that is robust to the sim-to-real gap. My policy learned a more aggressive gait that scores higher in sim but has no such robustness.

**Lesson:** Beating an author-provided baseline in simulation does not mean you have a better policy for the real robot. If your goal is sim-to-real transfer, you must evaluate on hardware early and often. Simulation metrics are necessary but not sufficient.

### 3.3 Domain Randomization Alone Was Not Enough

My primary strategy for sim-to-real transfer was domain randomization during training: varying floor friction, joint PID gains, mass distribution, initial poses, and applying random pushes. This is a standard approach, and it is necessary — but it was not sufficient for this robot.

The problem is that domain randomization treats the sim-to-real gap as *zero-mean noise* around the simulated parameters. It assumes the real robot's dynamics are somewhere within the randomization envelope. But if the real robot has systematic biases — loosened screws creating asymmetric backlash, a shifted center of mass from the mis-initialized neck, servo responses that differ from the identified BAM model — then no amount of randomization centered on the wrong parameters will help. You cannot randomize your way out of a systematic modeling error.

---

## 4. The Diffusion Policy Attempt

### 4.1 Motivation

I wanted to explore Diffusion Policy (Chi et al., 2023) on this platform, both as a learning exercise and to understand its strengths and weaknesses compared to PPO for bipedal locomotion. The specific approach was behavior cloning from PPO rollouts: generate a dataset of (observation, action) pairs by rolling out the PPO policy in simulation, then train a Conditional U-Net to predict action chunks via denoising.

### 4.2 Setup

- **Dataset:** 4,755 trajectories from the original author's PPO policy, collected at 50Hz in MuJoCo with randomized velocity commands, friction, initial poses, and impulse perturbations. Total: 865,780 steps.
- **Model:** ConditionalUnet1D (16.4M parameters) with DDIM sampling (16 denoising steps). Action chunk size of 16, observation horizon of 2.
- **Training:** 100 epochs on Vast.ai RTX 3090, batch size 256, AdamW with cosine learning rate schedule. Final validation loss: 0.0221.

### 4.3 Results

The diffusion policy failed catastrophically:

| Metric | PPO Baseline | Diffusion BC | Delta |
|--------|-------------|-------------|-------|
| Average survival steps | 100/1000 | 9/1000 | -91% |
| Fall rate | 0% | 98% | +98pp |
| Stand tracking (linear) | 0.809 | 0.125 | -85% |
| Action smoothness (rate) | 0.053 | 0.357 | +6.7x noisier |

Only 20% of episodes survived even the `stand` command. All motion commands caused immediate falls.

### 4.4 Why It Failed

**Compounding error (distribution shift).** Behavior cloning trains a policy to map observations to actions using data from an expert. At test time, any small prediction error causes the robot's state to deviate from the training distribution. The next observation is now slightly out-of-distribution, producing a slightly worse action, which causes further deviation. This feedback loop compounds until the state is so far from anything the policy saw during training that the actions become meaningless and the robot falls.

This is the well-known "DAgger problem" (Ross et al., 2011): BC policies are only correct on the states visited by the expert, but they must act on the states produced by their own (imperfect) predictions. Without online corrective feedback — which BC by definition does not have — the errors compound.

**Action smoothness.** The diffusion policy's action rate was 5-50x higher than PPO's. Even with DDIM's deterministic sampling, the denoised action chunks had high inter-step variance. For a bipedal robot that requires smooth, coordinated joint trajectories to maintain balance, jerky actions are immediately destabilizing.

**Unimodal teacher.** Distilling from a single PPO policy produces a unimodal action distribution. The diffusion model — despite being capable of representing multi-modal distributions — only sees one way to respond to each situation. It cannot learn the diverse recovery behaviors that PPO implicitly uses to stay upright after perturbations. PPO developed these recovery behaviors because the RL training loop rewarded survival; the diffusion policy never experienced this pressure.

**Why this matters beyond "it didn't work."** The failure of BC-from-distillation on bipedal locomotion is not surprising — it is a predictable consequence of known BC limitations applied to an inherently unstable system. But it highlights a real distinction between locomotion and manipulation. Diffusion Policy was demonstrated on manipulation tasks (Chi et al., 2023) where the consequences of small action errors are minor: a slightly misaligned gripper position can be corrected on the next step without catastrophic failure. In locomotion, especially bipedal, the state space is far less forgiving — a small error in foot placement can cause an unrecoverable fall within one or two steps. This is precisely why approaches like DPPO (Ren et al., 2024), which combine diffusion policy's expressive action representation with RL's corrective feedback loop, are the natural next step for applying diffusion methods to locomotion.

---

## 5. What I Would Do Differently

### 5.1 System Identification Before Training

The single most impactful change would be to perform system identification (SysID) on the real robot before training. SysID means measuring the actual dynamics of the physical robot — joint friction curves, backlash profiles, servo latency, mass distribution, COM position — and updating the simulated model to match.

Instead of broad domain randomization centered on generic parameters, I would:
1. Measure each joint's actual response to commanded positions at various speeds and loads
2. Identify the actual backlash profile per joint (likely asymmetric and different per joint after screw loosening)
3. Measure the real robot's mass and COM by physical weighing
4. Update the MuJoCo model's joint parameters, friction coefficients, and body masses to match

This directly addresses the root cause of the sim-to-real gap: the simulated model does not match the real robot. Domain randomization is a workaround for an inaccurate model; SysID fixes the model.

### 5.2 Real-World Data Collection and Fine-Tuning

If I had more time, I would set up a data collection pipeline on the real robot:
1. Run the author's best policy on the real robot while recording observations and actions at 50Hz
2. Use this real-world data to either:
   - Fine-tune the PPO policy via offline RL (conservative Q-learning, decision transformers)
   - Identify systematic observation-action mismatches that reveal modeling errors
3. Iterate: collect data → update model → retrain → evaluate → repeat

This is essentially the "real2sim2real" loop that has become standard in modern robot learning. The key insight is that even a small amount of real-world data is more valuable for closing the sim-to-real gap than any amount of simulated data.

### 5.3 DPPO Instead of BC Diffusion

For the diffusion policy work, the correct approach would be DPPO (Diffusion Policy Policy Optimization, Ren et al., 2024), which uses a diffusion model as the policy class within an on-policy RL objective. This addresses both the compounding error problem (RL provides corrective feedback) and the unimodal teacher limitation (the policy explores diverse recovery behaviors through the RL objective). BC-from-distillation is a reasonable starting point for manipulation where errors are non-catastrophic, but it is the wrong tool for bipedal locomotion.

---

## 6. Conclusion

This project produced two negative results: a PPO policy that excels in simulation but fails on hardware, and a BC diffusion policy that fails even in simulation. Both failures are informative.

The sim-to-real gap on this bipedal robot was dominated by hardware issues (loose screws, mis-initialized joints) that domain randomization cannot fix. The correct response to a systematic sim-to-real gap is not more randomization but better modeling — system identification to bring the simulated dynamics closer to reality.

The diffusion policy failure is a textbook illustration of behavior cloning's limitations on unstable systems. It reinforces the intuition that locomotion requires closed-loop corrective feedback, whether from RL (PPO, DPPO) or from real-world data. BC is well-suited for tasks where small errors are recoverable; bipedal walking is not such a task.

The most valuable outcome of this project is not any single result but the understanding of where each method breaks down and why. For future work, the path forward is clear: SysID to close the sim-to-real gap, and DPPO to combine diffusion's expressive action representation with RL's corrective feedback for bipedal locomotion.

---

## References

- Chi, C., et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* arXiv:2303.04137.
- Ren, A., et al. (2024). *Diffusion Policy Policy Optimization.* arXiv:2409.00588.
- Ross, S., et al. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS.
- Open Duck Mini V2: https://github.com/apirrone/Open_Duck_Mini
- Training Pipeline: https://github.com/apirrone/Open_Duck_Playground
