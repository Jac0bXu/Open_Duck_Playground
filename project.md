# Diffusion Policy on Open Duck Mini V2 (Sim-Only)

## Role in the Portfolio

This is a **secondary CV project** for Fall 2026 PhD applications, sitting alongside a primary publication push:

1. **Minecraft multi-agent collaboration paper (summer 2026, headline)** — workshop / symposium paper, August deadline. Demonstrates research ability and publication track record. This is the primary effort.
2. **Duck sim-only diffusion (this project, ~3 weeks part-time)** — sim-only diffusion-vs-PPO comparison on a 15-DOF bipedal robot, writeup framed around DPPO. Demonstrates robot-learning literacy and ability to ship implementations end-to-end.
3. **ALOHA (fall semester 2026, at school)** — the real Diffusion Policy work on the canonical bimanual manipulation platform. Where deeper methodology lives.
4. **Robot dog (spring 2027, optional)** — breadth across embodiments.

The duck's job here is small: produce a credible writeup that shows you understand modern diffusion-policy methods and can implement one. It does not need a hardware demo, does not need to beat PPO, and does not need to involve any real-robot work. The 2-week hardware window has been **explicitly skipped** so attention stays on the Minecraft publication.

## Goal

Sim-only diffusion policy on Open Duck Mini V2, compared head-to-head against the original repo PPO policy in MJX, with a 3-5 page writeup framed around DPPO (Ren et al., 2024) as the broader research direction.

**Success criteria:**
- Working diffusion policy that walks in MJX sim under matched conditions to PPO baseline
- Comparison table vs. PPO baseline in sim (tracking error, smoothness, perturbation recovery)
- 3-5 page technical writeup discussing the work and positioning it relative to DPPO
- Clean GitHub repo with training code, eval scripts, results

**Non-goals:**
- Real-robot deployment (sim-only by design)
- Publishable novel contribution
- Beating PPO in sim (PPO is likely to win; document honestly)
- Implementing DPPO from scratch (discussed in writeup, but BC diffusion is the actual method used — see below)

**Hard constraint:** Minecraft paper deadline is August. This project must not consume more than ~3 weeks of part-time effort total, and must not be active during weeks when Minecraft is in crunch. If a conflict arises, the duck pauses.

## Background

The Open Duck Mini V2 is a 15-DOF bipedal robot (10 leg + 5 head joints) with a working PPO + sim-to-real pipeline in this repo. The original repo PPO policy will serve as both the *teacher* for distillation and the *baseline* for comparison. (Note: an extensive Phase 3B PPO fine-tuning effort produced a checkpoint that won sim metrics but underperformed the original on real hardware — a classic reward-over-tuning case. The original repo policy is used here because it's the more honest reference point.)

Diffusion Policy (Chi et al., 2023) is the canonical "diffusion for robotics" paper. DPPO (Ren et al., 2024) extends it by using a diffusion policy as the policy class within an RL objective. This project implements the simpler BC-from-distillation variant and writes about DPPO as the broader research direction and natural next step.

## Approach

**Method (what is actually implemented):** Behavior cloning of a Diffusion Policy from PPO rollouts in MJX. U-Net backbone, DDIM sampling with 16 denoising steps, action chunking (chunk size 16). Reference implementation: [diffusion_policy](https://github.com/real-stanford/diffusion_policy) — port to Open Duck obs/action shapes.

**Method (what the writeup is framed around):** DPPO (diffusion-policy + PPO objective). The writeup will:
- Motivate diffusion policies for robotics in the introduction
- Position DPPO as the broader research direction in related work
- Describe the BC-from-distillation experiment honestly as the actual method used (framed as a tractable proxy for studying diffusion-policy behavior on bipedal locomotion)
- Discuss DPPO and on-hardware deployment as future work, with the UDP pipeline (below) as foundation infrastructure already built

This is a standard and honest workshop-paper framing: methods discussed are broader than methods implemented, as long as the actual experiments are clearly described.

**Action space:** Match the original PPO action space exactly — 14-dim (10 leg joints + 4 head joints: neck_pitch, head_pitch, head_yaw, head_roll). Antennae excluded. Drop-in replacement for the PPO ONNX policy.

**Training data:** PPO sim-rollout distillation. Roll the original repo PPO out in MJX across randomized velocity commands, terrain friction, initial poses, and light perturbations. Record (obs_t, action_t) at 30Hz. Target **5k-10k trajectories** of 5-10 seconds each. Caveat (for the writeup): unimodal teacher → unimodal student, partially mitigated by rollout diversity.

**Evaluation:** Sim-only in MJX. Comparison metrics — success rate, velocity tracking error (linear and angular), joint smoothness, recovery from perturbations (lateral and forward pushes injected during eval).

**Foundation infrastructure (built, not deployed):** A Mac↔Pi UDP off-board inference pipeline was designed for real-robot deployment but is not executed in this work (no hardware window taken). The pipeline is described in the writeup as foundation infrastructure for the natural extension to hardware. Action chunking sized so that inference can run at ~2Hz on a laptop while the robot consumes one action per 30Hz control tick.

## Hardware (Compute)

- **Dev machine:** M2 MacBook Air — training (small sweeps) + sim eval
- **Cloud:** rent a GPU (~$20) for larger training runs if needed
- **No physical robot.** All work is in MJX.

## Plan (~3 weeks part-time, target finish ~September 2026)

This plan assumes maybe ~10-15 hours per week, fitted around the Minecraft work. If Minecraft hits crunch, the duck pauses without guilt.

**Week 1 — Dataset generation + repo port**
- Generate PPO distillation dataset in MJX: 5k-10k trajectories with randomized commands, friction, perturbations. Save as HDF5 matching the existing recording format.
- Port [diffusion_policy](https://github.com/real-stanford/diffusion_policy) to Open Duck obs/action shapes. Get one training run started end-to-end on a tiny subset just to validate plumbing.
- Pick one config and commit to it. Don't sweep early.

**Week 2 — Train + sim eval**
- Full training run on the 5k-10k dataset. One or two hyperparameter variants at most (action horizon, denoising steps). **Do not sweep exhaustively** — one working config beats a grid search.
- Sim eval vs. PPO baseline under matched conditions. Build the comparison table: tracking error, smoothness, perturbation recovery.
- If results look reasonable, freeze and move to writeup. If not, one more training iteration with adjusted dataset (more perturbations / wider initial conditions).

**Week 3 — Writeup + repo polish**
- 3-5 page writeup: introduction (motivate diffusion for robotics), related work (DPPO discussion is generous here), methods (honest description of BC-from-distillation), experiments (sim comparison), discussion (where BC underperforms RL, why DPPO is the natural next step), future work (DPPO + hardware via the UDP pipeline).
- Polished GitHub README with reproduction instructions.
- Short eval video (sim recording of both policies side by side under matched commands).
- Done.

**Explicitly out of scope:** real-robot deployment, consistency distillation, actually implementing DPPO, exhaustive hyperparameter sweeps, multi-teacher distillation experiments, ALOHA work (that's a separate project), anything that takes attention from Minecraft.

## Stretch (only if Minecraft is in great shape)

If Minecraft is submitted comfortably ahead of August deadline *and* the BC diffusion experiment finishes ahead of schedule, the natural stretch is **actually implementing DPPO** by adapting the [DPPO repo](https://github.com/irom-princeton/dppo) to Open Duck. This would turn the writeup from "BC diffusion experiments, DPPO in related work" into "DPPO on bipedal locomotion." Much stronger story, much more work — only attempt this if both prior conditions hold.

## Risk Register

- **Minecraft conflict.** The single biggest risk: the duck consuming attention during Minecraft crunch and damaging the primary publication. Mitigation: explicit pause rule — if Minecraft needs the next two weeks, the duck doesn't get touched until after submission. Period.
- **Scope creep into DPPO implementation.** RL with diffusion sampling has annoying gradient-flow and sampling-step debugging that can swallow weeks. Mitigation: BC is the committed method; DPPO is *only* a stretch and only if Minecraft is already submitted. Resist the urge to "just try DPPO real quick" mid-project.
- **Unimodal teacher → unimodal student.** PPO distillation will produce a diffusion student without interesting multi-modality. Acceptance: this is a discussion-section caveat, not a blocker. Workshop writeups regularly acknowledge such limitations.
- **Diffusion underperforms PPO in sim.** Likely, even in sim — BC has known weaknesses (compounding error, undersampled state-space tails) that show up under perturbations. Acceptance: report honestly. "Characterizing where BC underperforms RL on bipedal locomotion" is itself a defensible writeup angle.
- **Sweep paralysis.** Easy to spend weeks tuning prediction horizon / action horizon / denoising steps. Mitigation: one config, committed Week 1. Move on.

## CV / SoP Framing

One bullet in a portfolio, supporting the Minecraft publication and setting up the ALOHA work.

**Standard:**
> Implemented a Diffusion Policy on a 15-DOF bipedal robot in simulation, trained via policy distillation from the repo's PPO policy. Compared against the PPO baseline on locomotion and perturbation-recovery metrics in MJX. Designed (but did not deploy) an off-board UDP inference pipeline as foundation infrastructure for hardware extension. Writeup positions the work relative to DPPO (Ren et al., 2024) as the natural research direction; ongoing follow-on work applies diffusion policies to bimanual manipulation on ALOHA.

**If stretch lands (actual DPPO implementation):**
> ...additionally adapted the DPPO framework (Ren et al., 2024) to bipedal locomotion, comparing diffusion-policy RL against vanilla PPO in simulation.

## References

- Chi et al., 2023. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Ren et al., 2024. *Diffusion Policy Policy Optimization (DPPO).* [arXiv:2409.00588](https://arxiv.org/abs/2409.00588)
- Song et al., 2023. *Consistency Models.* [arXiv:2303.01469](https://arxiv.org/abs/2303.01469) — for future-work discussion
- [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) — this repo, PPO training infra
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy) — reference implementation to port
- [dppo](https://github.com/irom-princeton/dppo) — DPPO reference, for stretch goal and related work
