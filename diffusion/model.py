"""Diffusion policy model for Open Duck Mini V2.

ConditionalUnet1D backbone + DDIM sampler, ported from:
  Chi et al. 2023 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

Fixed config (committed per project plan):
  obs_horizon   = 2   (stack last 2 observations)
  pred_horizon  = 16  (action chunk size)
  T_train       = 100 (DDPM diffusion steps)
  T_infer       = 16  (DDIM inference steps)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture constants ────────────────────────────────────────────────────

OBS_DIM = 101
ACTION_DIM = 14
OBS_HORIZON = 2
PRED_HORIZON = 16
T_TRAIN = 100       # training diffusion steps
T_INFER = 16        # DDIM inference steps


# ── Noise schedule ────────────────────────────────────────────────────────────

def _cosine_alphas_cumprod(T: int) -> torch.Tensor:
    """Cosine noise schedule — returns alpha_bar for t in [0, T]."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas = torch.cos(((steps / T + 0.008) / 1.008) * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    return alphas.float()


class NoiseScheduler:
    """DDPM forward process + DDIM reverse process."""

    def __init__(self, T: int = T_TRAIN, T_infer: int = T_INFER):
        self.T = T
        self.T_infer = T_infer

        alphas_cumprod = _cosine_alphas_cumprod(T)
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = betas.clamp(0, 0.999)

        self.register = {}
        self.register["betas"] = betas
        self.register["alphas"] = 1.0 - betas
        self.register["alphas_cumprod"] = alphas_cumprod[1:]          # ᾱ_t, t=1..T
        self.register["alphas_cumprod_prev"] = alphas_cumprod[:-1]    # ᾱ_{t-1}

        # Evenly-spaced inference timesteps: T, T-K, ..., K, 0 (descending)
        self.infer_timesteps = torch.linspace(T - 1, 0, T_infer, dtype=torch.long)

    def _get(self, key: str, t: torch.Tensor, device) -> torch.Tensor:
        # returns (B, 1, 1) for broadcasting over (B, T, C) action tensors
        tensor = self.register[key].to(device)
        return tensor[t].view(-1, 1, 1)

    # ── forward (add noise) ───────────────────────────────────────────────

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε"""
        device = x0.device
        sqrt_ab = self._get("alphas_cumprod", t, device).sqrt()
        sqrt_1_ab = (1.0 - self._get("alphas_cumprod", t, device)).sqrt()
        return sqrt_ab * x0 + sqrt_1_ab * noise

    # ── reverse (DDIM step) ───────────────────────────────────────────────

    def ddim_step(
        self,
        model_fn,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """One DDIM denoising step from x_t → x_{t_prev} (deterministic η=0)."""
        B, T, C = x_t.shape
        device = x_t.device

        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        pred_noise = model_fn(x_t, t_tensor, global_cond)   # (B, T, C)

        ab_t = self.register["alphas_cumprod"][t].to(device)
        ab_prev = (
            self.register["alphas_cumprod"][t_prev].to(device)
            if t_prev >= 0
            else torch.tensor(1.0, device=device)
        )

        # pred_x0 from epsilon prediction
        pred_x0 = (x_t - (1.0 - ab_t).sqrt() * pred_noise) / ab_t.sqrt()
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # DDIM update (η=0 → deterministic)
        x_prev = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * pred_noise
        return x_prev

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,
        shape: tuple,
        global_cond: torch.Tensor,
        device,
    ) -> torch.Tensor:
        """Full DDIM denoising from x_T ~ N(0,I) → x_0."""
        x = torch.randn(shape, device=device)
        timesteps = self.infer_timesteps.tolist()

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.ddim_step(model_fn, x, t, t_prev, global_cond)

        return x


# ── U-Net building blocks ─────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)   # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


class ConditionalResidualBlock1D(nn.Module):
    """1D conv residual block with FiLM conditioning (scale+shift from global_cond)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
            nn.GroupNorm(n_groups, out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad),
            nn.GroupNorm(n_groups, out_channels),
        )
        self.cond_proj = nn.Linear(global_cond_dim, out_channels * 2)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)  cond: (B, cond_dim)
        scale_shift = self.cond_proj(cond)             # (B, C_out*2)
        scale, shift = scale_shift.unsqueeze(-1).chunk(2, dim=1)   # each (B, C_out, 1)

        h = self.act(self.conv1[1](self.conv1[0](x)) * (1 + scale) + shift)
        h = self.act(self.conv2[1](self.conv2[0](h)))
        return h + self.residual_conv(x)


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)   # (B, dim*2, T//2)


class Upsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)   # (B, dim//2, T*2)


# ── Main model ────────────────────────────────────────────────────────────────

class ConditionalUnet1D(nn.Module):
    """1D conditional U-Net for diffusion policy.

    Input:
        noisy_actions : (B, T_pred, action_dim)   — noisy action sequence
        timestep      : (B,)                       — diffusion timestep
        global_cond   : (B, obs_horizon * obs_dim) — stacked observations

    Output:
        predicted noise (B, T_pred, action_dim)
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        obs_cond_dim: int = OBS_HORIZON * OBS_DIM,
        base_dim: int = 128,
        n_groups: int = 8,
    ):
        super().__init__()
        d0, d1, d2, d3 = base_dim, base_dim * 2, base_dim * 4, base_dim * 8

        # ── global conditioning: time + obs ──────────────────────────────
        time_emb_dim = base_dim
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_cond_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        global_cond_dim = time_emb_dim * 8   # cat(time, obs)

        # ── initial projection ────────────────────────────────────────────
        self.init_conv = nn.Conv1d(action_dim, d0, kernel_size=1)

        # ── encoder (T=16 → 8 → 4) ───────────────────────────────────────
        def rb(ic, oc): return ConditionalResidualBlock1D(ic, oc, global_cond_dim, n_groups=n_groups)

        self.down1_res = nn.ModuleList([rb(d0, d0), rb(d0, d0)])
        self.down1_ds = Downsample1D(d0)      # d0 → d1, T/2

        self.down2_res = nn.ModuleList([rb(d1, d1), rb(d1, d1)])
        self.down2_ds = Downsample1D(d1)      # d1 → d2, T/4

        # ── bottleneck (T=4) ──────────────────────────────────────────────
        self.mid_res = nn.ModuleList([rb(d2, d2), rb(d2, d2)])

        # ── decoder (T=4 → 8 → 16) ───────────────────────────────────────
        self.up1_us = Upsample1D(d2)          # d2 → d1, T*2
        self.up1_res = nn.ModuleList([rb(d1 + d1, d1), rb(d1, d1)])

        self.up2_us = Upsample1D(d1)          # d1 → d0, T*4
        self.up2_res = nn.ModuleList([rb(d0 + d0, d0), rb(d0, d0)])

        # ── output ───────────────────────────────────────────────────────
        self.final_conv = nn.Conv1d(d0, action_dim, kernel_size=1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        # noisy_actions: (B, T, action_dim) → transpose to (B, action_dim, T)
        x = noisy_actions.permute(0, 2, 1)      # (B, action_dim, T)

        # global conditioning
        te = self.time_emb(timestep)             # (B, dim*4)
        oe = self.obs_emb(global_cond)           # (B, dim*4)
        cond = torch.cat([te, oe], dim=-1)       # (B, dim*8)

        x = self.init_conv(x)                   # (B, d0, T)

        # encoder
        for blk in self.down1_res:
            x = blk(x, cond)
        h1 = x                                  # skip: (B, d0, T)
        x = self.down1_ds(x)                    # (B, d1, T//2)

        for blk in self.down2_res:
            x = blk(x, cond)
        h2 = x                                  # skip: (B, d1, T//2)
        x = self.down2_ds(x)                    # (B, d2, T//4)

        # bottleneck
        for blk in self.mid_res:
            x = blk(x, cond)

        # decoder
        x = self.up1_us(x)                      # (B, d1, T//2)
        x = torch.cat([x, h2], dim=1)           # (B, d1+d1, T//2)
        for blk in self.up1_res:
            x = blk(x, cond)

        x = self.up2_us(x)                      # (B, d0, T)
        x = torch.cat([x, h1], dim=1)           # (B, d0+d0, T)
        for blk in self.up2_res:
            x = blk(x, cond)

        x = self.final_conv(x)                  # (B, action_dim, T)
        return x.permute(0, 2, 1)               # (B, T, action_dim)


# ── Diffusion Policy (wrapper for train + infer) ──────────────────────────────

class DiffusionPolicy(nn.Module):
    """Combines ConditionalUnet1D + NoiseScheduler into one module."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        obs_horizon: int = OBS_HORIZON,
        pred_horizon: int = PRED_HORIZON,
        T_train: int = T_TRAIN,
        T_infer: int = T_INFER,
        base_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        obs_cond_dim = obs_horizon * obs_dim
        self.net = ConditionalUnet1D(
            action_dim=action_dim,
            obs_cond_dim=obs_cond_dim,
            base_dim=base_dim,
        )
        self.scheduler = NoiseScheduler(T=T_train, T_infer=T_infer)

        # Normalization stats (set after dataset loading)
        self.register_buffer("action_min", torch.zeros(action_dim))
        self.register_buffer("action_max", torch.ones(action_dim))

    def set_normalizer(self, action_min: np.ndarray, action_max: np.ndarray) -> None:
        self.action_min = torch.from_numpy(action_min).float()
        self.action_max = torch.from_numpy(action_max).float()

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        lo = self.action_min.to(action.device)
        hi = self.action_max.to(action.device)
        return 2.0 * (action - lo) / (hi - lo + 1e-8) - 1.0

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        lo = self.action_min.to(action_norm.device)
        hi = self.action_max.to(action_norm.device)
        return (action_norm + 1.0) / 2.0 * (hi - lo + 1e-8) + lo

    def compute_loss(
        self, obs_cond: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Training loss: MSE between predicted and true noise."""
        B = obs_cond.shape[0]
        device = obs_cond.device

        action_norm = self.normalize_action(action)

        t = torch.randint(0, self.scheduler.T, (B,), device=device)
        noise = torch.randn_like(action_norm)
        x_t = self.scheduler.add_noise(action_norm, noise, t)

        pred_noise = self.net(x_t, t, obs_cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def predict_action(
        self, obs_cond: torch.Tensor
    ) -> torch.Tensor:
        """Run DDIM to sample an action chunk. Returns denormalized actions."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        shape = (B, self.pred_horizon, self.action_dim)

        action_norm = self.scheduler.ddim_sample(
            model_fn=lambda x, t, c: self.net(x, t, c),
            shape=shape,
            global_cond=obs_cond,
            device=device,
        )
        return self.denormalize_action(action_norm)
