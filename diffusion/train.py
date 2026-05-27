"""Train diffusion policy from PPO rollout dataset.

Usage (tiny smoke test):
    .venv/bin/python -m diffusion.train \\
        --dataset data/ppo_rollouts_100.hdf5 \\
        --output checkpoints/diffusion/ \\
        --epochs 5 --batch_size 64

Usage (full training run):
    .venv/bin/python -m diffusion.train \\
        --dataset data/ppo_rollouts_5k.hdf5 \\
        --output checkpoints/diffusion/ \\
        --epochs 100 --batch_size 256
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from diffusion.model import DiffusionPolicy, OBS_DIM, ACTION_DIM, OBS_HORIZON, PRED_HORIZON
from diffusion.dataset import DiffusionDataset


def train(
    dataset_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    num_workers: int,
    device_str: str,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── dataset ────────────────────────────────────────────────────────────
    full_dataset = DiffusionDataset(dataset_path, OBS_HORIZON, PRED_HORIZON)

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    print(f"Train samples: {n_train}  Val samples: {n_val}")

    # ── model ──────────────────────────────────────────────────────────────
    model = DiffusionPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
    ).to(device)

    model.set_normalizer(full_dataset.action_min, full_dataset.action_max)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── optimiser ──────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt = os.path.join(output_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            obs_cond = batch["obs_cond"].to(device)       # (B, obs_horizon, obs_dim)
            actions = batch["actions"].to(device)         # (B, pred_horizon, action_dim)

            # Flatten obs_cond to (B, obs_horizon * obs_dim)
            obs_cond_flat = obs_cond.view(obs_cond.shape[0], -1)

            loss = model.compute_loss(obs_cond_flat, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs_cond = batch["obs_cond"].to(device)
                actions = batch["actions"].to(device)
                obs_cond_flat = obs_cond.view(obs_cond.shape[0], -1)
                loss = model.compute_loss(obs_cond_flat, actions)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr_now:.2e}  t={elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "action_min": full_dataset.action_min,
                "action_max": full_dataset.action_max,
            }, best_ckpt)
            print(f"  → saved best checkpoint (val={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            ckpt = os.path.join(output_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "action_min": full_dataset.action_min,
                "action_max": full_dataset.action_max,
            }, ckpt)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion policy")
    parser.add_argument("--dataset", required=True, help="HDF5 dataset path")
    parser.add_argument("--output", default="checkpoints/diffusion/",
                        help="Checkpoint output dir")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu', 'cuda', or 'mps'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        device_str=args.device,
        seed=args.seed,
    )
