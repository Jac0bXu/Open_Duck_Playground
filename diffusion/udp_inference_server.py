"""UDP off-board inference server (Mac side).

The robot (Pi) sends one observation vector per request; this server runs the
policy and returns an action chunk.  Both PPO-ONNX and diffusion policies are
supported via a pluggable callable.

Protocol (all fields big-endian):

  Request  (Pi → Mac):
    magic    : 4 bytes  b"DUCK"
    seq_num  : uint32
    obs_dim  : uint16
    obs      : float32[obs_dim]

  Response (Mac → Pi):
    magic      : 4 bytes  b"QUAK"
    seq_num    : uint32   (echoed from request)
    chunk_size : uint16
    action_dim : uint16
    actions    : float32[chunk_size * action_dim]

Usage:
  # PPO ONNX (for integration testing before diffusion is ready)
  python -m diffusion.udp_inference_server --onnx path/to/policy.onnx

  # Diffusion (once implemented)
  python -m diffusion.udp_inference_server --diffusion path/to/checkpoint.pt
"""

import argparse
import socket
import struct
import time
from collections import deque
from typing import Callable

import numpy as np

# ── protocol constants ────────────────────────────────────────────────────────
MAGIC_REQ = b"DUCK"
MAGIC_RES = b"QUAK"

OBS_DIM = 101
ACTION_DIM = 14
CHUNK_SIZE = 16
DEFAULT_PORT = 7777

# ── packet helpers ────────────────────────────────────────────────────────────

def _parse_request(data: bytes) -> tuple[int, np.ndarray] | None:
    if len(data) < 10 or data[:4] != MAGIC_REQ:
        return None
    seq_num = struct.unpack_from("!I", data, 4)[0]
    obs_dim = struct.unpack_from("!H", data, 8)[0]
    if len(data) < 10 + obs_dim * 4:
        return None
    obs = np.frombuffer(data[10 : 10 + obs_dim * 4], dtype=np.float32).copy()
    return seq_num, obs


def _build_response(seq_num: int, actions: np.ndarray) -> bytes:
    chunk_size, action_dim = actions.shape
    header = struct.pack("!4sIHH", MAGIC_RES, seq_num, chunk_size, action_dim)
    return header + actions.astype(np.float32).tobytes()


# ── policy wrappers ───────────────────────────────────────────────────────────

def make_ppo_policy(onnx_path: str) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a PPO ONNX model.  Repeats the single-step action across the chunk."""
    from playground.common.onnx_infer import OnnxInfer

    policy = OnnxInfer(onnx_path, awd=True)

    def _infer(obs_stack: np.ndarray) -> np.ndarray:
        # obs_stack: (obs_horizon, obs_dim) — use most recent obs
        action = policy.infer(obs_stack[-1])          # (action_dim,)
        return np.tile(action, (CHUNK_SIZE, 1))        # (chunk_size, action_dim)

    return _infer


def make_diffusion_policy(checkpoint_path: str, obs_horizon: int = 2) -> Callable:
    """Load a trained diffusion policy checkpoint and return an infer callable."""
    import torch
    from diffusion.model import DiffusionPolicy

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DiffusionPolicy(obs_horizon=obs_horizon)
    model.load_state_dict(ckpt["model_state"])
    model.set_normalizer(ckpt["action_min"], ckpt["action_max"])
    model.eval()

    device = torch.device("cpu")

    def _infer(obs_stack: np.ndarray) -> np.ndarray:
        # obs_stack: (obs_horizon, obs_dim)
        obs_t = torch.from_numpy(obs_stack).float().unsqueeze(0)  # (1, obs_h, obs_dim)
        obs_flat = obs_t.view(1, -1)
        with torch.no_grad():
            action_chunk = model.predict_action(obs_flat)          # (1, chunk_size, action_dim)
        return action_chunk.squeeze(0).numpy()                     # (chunk_size, action_dim)

    return _infer


# ── server ────────────────────────────────────────────────────────────────────

class UDPInferenceServer:
    """Listens for observation requests, runs policy, sends action chunks back."""

    def __init__(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        obs_horizon: int = 1,
    ):
        self.policy_fn = policy_fn
        self.obs_horizon = obs_horizon
        self._obs_history: deque[np.ndarray] = deque(maxlen=obs_horizon)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(1.0)

        self._running = False
        self._n_req = 0
        self._n_resp = 0
        self._n_err = 0
        self._latency_ema = 0.0   # exponential moving avg, seconds

    def _update_obs_history(self, obs: np.ndarray) -> np.ndarray:
        self._obs_history.append(obs)
        # Pad left with earliest frame until history is full
        while len(self._obs_history) < self.obs_horizon:
            self._obs_history.appendleft(obs)
        return np.stack(list(self._obs_history))   # (obs_horizon, obs_dim)

    def run(self) -> None:
        self._running = True
        host, port = self._sock.getsockname()
        print(
            f"UDP inference server on {host}:{port}  "
            f"obs_dim={OBS_DIM} action_dim={ACTION_DIM} chunk_size={CHUNK_SIZE} "
            f"obs_horizon={self.obs_horizon}"
        )

        while self._running:
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            t0 = time.perf_counter()
            parsed = _parse_request(data)
            if parsed is None:
                self._n_err += 1
                continue

            seq_num, obs = parsed
            self._n_req += 1

            obs_stack = self._update_obs_history(obs)

            try:
                action_chunk = self.policy_fn(obs_stack)   # (chunk_size, action_dim)
            except Exception as exc:
                print(f"[server] policy error: {exc}")
                self._n_err += 1
                continue

            assert action_chunk.shape == (CHUNK_SIZE, ACTION_DIM), (
                f"Policy must return ({CHUNK_SIZE}, {ACTION_DIM}), got {action_chunk.shape}"
            )

            try:
                self._sock.sendto(_build_response(seq_num, action_chunk), addr)
                self._n_resp += 1
            except OSError as exc:
                print(f"[server] send error: {exc}")
                continue

            dt = time.perf_counter() - t0
            alpha = 0.1
            self._latency_ema = (1 - alpha) * self._latency_ema + alpha * dt

            if self._n_req % 20 == 0:
                print(
                    f"  req={self._n_req} resp={self._n_resp} err={self._n_err} "
                    f"latency_ema={self._latency_ema * 1000:.1f}ms"
                )

    def stop(self) -> None:
        self._running = False
        self._sock.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UDP off-board inference server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", metavar="PATH", help="PPO ONNX policy path")
    group.add_argument("--diffusion", metavar="PATH", help="Diffusion checkpoint path")

    parser.add_argument(
        "--obs_horizon",
        type=int,
        default=1,
        help="Number of past obs steps to stack (1 for PPO, 2 for diffusion)",
    )
    args = parser.parse_args()

    if args.onnx:
        print(f"Loading PPO policy from {args.onnx}")
        policy_fn = make_ppo_policy(args.onnx)
        obs_horizon = args.obs_horizon
    else:
        print(f"Loading diffusion policy from {args.diffusion}")
        policy_fn = make_diffusion_policy(args.diffusion, obs_horizon=args.obs_horizon)
        obs_horizon = args.obs_horizon

    server = UDPInferenceServer(policy_fn, port=args.port, obs_horizon=obs_horizon)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
