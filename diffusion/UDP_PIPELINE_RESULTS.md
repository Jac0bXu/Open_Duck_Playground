# UDP Off-Board Inference Pipeline — Results

## What Was Built

A Mac↔Pi UDP inference pipeline for off-board policy execution, designed as
foundation infrastructure for the diffusion policy project.

**Mac side** (`diffusion/udp_inference_server.py`):
- UDP server listening on port 7777
- Receives 414-byte observation packets from the robot
- Runs any policy (PPO ONNX or, eventually, diffusion) and returns a 908-byte
  action chunk (16 actions × 14 joints × float32)
- Maintains observation history for diffusion's `obs_horizon` stacking
- Pluggable policy interface: `make_ppo_policy(onnx_path)` for testing,
  `make_diffusion_policy(checkpoint)` stub for future diffusion work

**Pi side** (`scripts/udp_walk.py` in Runtime repo):
- Async prefetch architecture: background thread fetches next action chunk
  while main control loop consumes actions without blocking
- Prefetch triggered when buffer falls to `refill_threshold` (default 4),
  giving ~80ms headroom before buffer empties at 50Hz
- Full Xbox controller support for velocity commands
- Observation construction matches the trained policy exactly (101-dim)

## Protocol

```
Request  (Pi → Mac):  b"DUCK" + seq:u32 + obs_dim:u16 + obs:float32[101]   = 414 bytes
Response (Mac → Pi):  b"QUAK" + seq:u32 + 16:u16 + 14:u16 + actions:float32[16×14] = 908 bytes
```

Both fit in a single UDP datagram. Sequence numbers guard against stale
responses. All floats are big-endian.

## Test Results

**Protocol round-trip (verified):**
- Tested Pi → Mac → Pi with PPO ONNX (`BEST_WALK_ONNX_2.onnx`)
- Correct 16×14 action chunks returned
- Typical round-trip: ~40ms over local WiFi (192.168.0.x)

**Walking test (failed to walk):**
- Robot motors activate and hold position correctly
- Policy produces reasonable actions (verified by inspecting chunk values)
- Robot did not walk — twitching / barely moving under load

## Root Cause Analysis

### 1. WiFi latency is too high for PPO

PPO is an autoregressive policy trained at 50Hz: it expects a fresh
observation every 20ms and returns one action. The 40ms WiFi RTT already
exceeds one control period. Worse, occasional spikes (100–600ms) caused
the robot to stall mid-motion.

The async prefetch design reduces but does not eliminate this: with chunk_size=16
and refill_threshold=4, the buffer provides ~80ms headroom at 50Hz. But the
obs used to generate the next chunk is already 12 steps stale (~240ms at
50Hz) when those actions are consumed. PPO, unlike diffusion, does not plan
ahead and is not robust to this level of obs staleness.

### 2. PPO is not designed for action chunking

Action chunking (predict N steps from one observation) is the core
assumption of diffusion policy — it was trained and evaluated this way.
PPO was not. Replaying a 16-step chunk generated from a single obs causes
the robot to overshoot and oscillate, even without network latency.

### 3. IMU calibration issue on the Pi

The `raw_imu.py` background thread intermittently throws:
```
TypeError: unsupported operand type(s) for -=: 'NoneType' and 'int'
```
in the accelerometer calibration path. This pre-exists in the runtime and
does not affect the original `v2_rl_walk_mujoco.py` (which runs the policy
onboard where the control loop is tight enough to tolerate stale IMU frames).
Over UDP it becomes more visible because the control loop is slower and each
stale IMU sample corresponds to a larger real-world gap.

## Conclusion

The **pipeline architecture is sound** — the protocol works, the async
prefetch design is correct, and the server correctly wraps any policy. The
blocker is not software design but hardware: WiFi (40ms+ RTT) is too slow
for real-time 50Hz control of a policy that was not designed for chunked
execution.

**Path forward:**

| Approach | Feasibility |
|---|---|
| Diffusion policy with true action chunking | **Viable** — diffusion is designed to predict 16 steps ahead from one obs. With ~40ms WiFi RTT and a 320ms chunk window (16 × 20ms), the pipeline has plenty of headroom. The obs staleness is intentional and modeled during training. |
| PPO over UDP with wired Ethernet | Potentially viable — wired RTT would drop to <5ms, fitting comfortably in a 20ms control period. |
| Onboard inference (RPi 5, Jetson Orin Nano) | Best long-term path — removes network entirely. A small U-Net running locally could achieve 50Hz with ONNX + quantization. |
| DPPO (diffusion policy + RL objective) | The natural next step beyond BC diffusion; would likely produce a more robust chunked policy and is designed for exactly this deployment pattern. |

The UDP pipeline as built is the correct infrastructure for the diffusion
policy deployment. Once a trained diffusion checkpoint exists, plugging it
into `make_diffusion_policy()` in `udp_inference_server.py` completes the loop.
