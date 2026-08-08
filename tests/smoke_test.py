"""
Smoke test: minimal end-to-end loop exercising every env dep.

What it does:
  1. Load a DMC env (mujoco + dm_control)
  2. Collect random-action pixel rollouts
  3. Train a small pixel autoencoder for a few hundred steps
  4. Save a real-vs-reconstruction video via imageio
  5. Save a loss curve via matplotlib
  6. Log scalars + video + loss curve to wandb (disabled by default)

Run: python tests/smoke_test.py
"""
import os
import sys
import tempfile

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ if present

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v3 as iio
import wandb

import mujoco  # noqa: F401  (imported to verify install; used transitively by dm_control)
from dm_control import suite


DEFAULT_CFG = """
env:
  domain: cartpole
  task: swingup
  image_size: 64
data:
  num_episodes: 2
  episode_length: 50
train:
  num_steps: 100
  batch_size: 32
  lr: 1e-3
  latent_dim: 64
"""


class TinyAutoencoder(nn.Module):
    def __init__(self, image_size: int = 64, latent_dim: int = 64) -> None:
        super().__init__()
        assert image_size == 64, "encoder/decoder strides assume 64x64"
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # 64x64
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        return self.dec(z), z


def pick_device() -> str:
    if torch.cuda.is_available():
        print(f"  device: cuda ({torch.cuda.get_device_name(0)}, "
              f"cc={torch.cuda.get_device_capability(0)})")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  device: mps")
        return "mps"
    print("  device: cpu (no accelerator detected)")
    return "cpu"


def collect_rollouts(cfg) -> np.ndarray:
    env = suite.load(cfg.env.domain, cfg.env.task)
    spec = env.action_spec()
    frames: list[np.ndarray] = []
    for _ in range(cfg.data.num_episodes):
        env.reset()
        for _ in range(cfg.data.episode_length):
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape).astype(np.float32)
            env.step(action)
            pixels = env.physics.render(
                height=cfg.env.image_size, width=cfg.env.image_size, camera_id=0,
            )
            frames.append(pixels)
    return np.stack(frames)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}, Platform: {sys.platform}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', '(unset)')}")

    cfg = OmegaConf.create(DEFAULT_CFG)
    print("\n--- config ---")
    print(OmegaConf.to_yaml(cfg).rstrip())

    print("\n--- device ---")
    device = pick_device()

    print("\n--- wandb ---")
    mode = os.environ.get("WANDB_MODE", "disabled")
    wandb.init(
        entity=None,
        project="11685-rl-guided-project",
        name="smoke",
        tags=["smoke"],
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"  mode: {mode}")

    print("\n--- env rollout ---")
    frames = collect_rollouts(cfg)
    print(f"  collected frames: {frames.shape} ({frames.dtype})")
    expected_shape = (
        cfg.data.num_episodes * cfg.data.episode_length,
        cfg.env.image_size, cfg.env.image_size, 3,
    )
    assert frames.shape == expected_shape, f"got {frames.shape}, expected {expected_shape}"

    x = torch.from_numpy(frames).float().to(device) / 255.0
    x = rearrange(x, "n h w c -> n c h w")
    print(f"  tensor: {tuple(x.shape)} on {x.device}")

    print("\n--- model ---")
    model = TinyAutoencoder(cfg.env.image_size, cfg.train.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TinyAutoencoder: {n_params:,} params")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    print("\n--- train ---")
    losses: list[float] = []
    for step in tqdm(range(cfg.train.num_steps), desc="  step"):
        idx = torch.randint(0, x.shape[0], (cfg.train.batch_size,), device=device)
        recon, _ = model(x[idx])
        loss = F.mse_loss(recon, x[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        wandb.log({"train/loss": loss.item(), "train/step": step})

    early = float(np.mean(losses[:10]))
    late = float(np.mean(losses[-10:]))
    print(f"  loss: mean first 10 = {early:.4f} -> mean last 10 = {late:.4f}")
    assert late < early, "loss should decrease across training; backprop may be broken"

    print("\n--- artifacts ---")
    with torch.no_grad():
        recon, _ = model(x[: cfg.data.episode_length])

    real_u8 = (x[: cfg.data.episode_length] * 255).clamp(0, 255).to(torch.uint8)
    recon_u8 = (recon * 255).clamp(0, 255).to(torch.uint8)
    side = torch.cat([real_u8, recon_u8], dim=-1)  # cat along width
    side_np = rearrange(side, "n c h w -> n h w c").cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    iio.imwrite(video_path, side_np, fps=15, codec="libx264")
    print(f"  video: {video_path} ({os.path.getsize(video_path):,} bytes)")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title("Smoke-test autoencoder loss")
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plot_path = f.name
    fig.savefig(plot_path, dpi=80)
    plt.close(fig)
    print(f"  loss curve: {plot_path} ({os.path.getsize(plot_path):,} bytes)")

    wandb.log({
        "video/real_vs_recon": wandb.Video(video_path, format="mp4"),
        "plot/loss_curve": wandb.Image(plot_path),
    })
    wandb.finish()

    os.remove(video_path)
    os.remove(plot_path)

    print("\nALL CHECKS PASSED. Env is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
