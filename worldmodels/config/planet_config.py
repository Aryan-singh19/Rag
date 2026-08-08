from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PlaNetTrainConfig:
    # path
    checkpoint_dir:str = f"checkpoints/planet"

    # Runtime
    seed: int = 0

    # Environment
    domain: str = "ball_in_cup"
    task: str = "catch"
    image_size: int = 64
    image_bits: int = 5
    camera_id: int = 0
    action_repeat: int = 4
    episode_length: int = 1000

    # Model
    s_dim: int = 30
    h_dim: int = 200
    encoded_dim: int = 1024
    mlp_hidden_dim: int = 200
    mlp_layers: int = 2
    mlp_activation: str = "relu"
    min_std: float = 0.1

    # Optimization and loss
    learning_rate: float = 1e-3
    adam_epsilon: float = 1e-4
    gradient_clip_norm: float = 1000.0
    free_nats: float = 3.0
    kl_loss_scale: float = 1.0
    reconstruction_loss_scale: float = 1.0
    reward_loss_scale: float = 10.0

    # CEM planning
    planning_horizon: int = 12
    num_candidates: int = 1000
    num_elites: int = 100
    refine_iterations: int = 10
    cem_min_std: float = 1e-6

    # Replay and training
    replay_capacity: int = 500_000
    batch_size: int = 50
    sequence_length: int = 50
    seed_episodes: int = 5
    train_steps: int = 99_500
    collect_interval: int = 100
    episodes_per_collection: int = 1
    exploration_noise: float = 0.3
    log_interval: int = 100
    eval_interval: int = 1000
    eval_episodes:int = 10

    # Checkpointing and resume
    # checkpoint_interval counts training steps. resume_latest.pt contains the
    # model, optimizer, replay, and random-number states and is replaced on each
    # save. Each milestone_env_steps value creates a model-only checkpoint.
    checkpoint_interval: int = 1000
    milestone_env_steps: tuple[int, ...] = (500_000, 1_000_000)

    # wandb
    entity: str | None = None
    project: str = "11685-rl-guided-project"
    run_name: str = "planet"
    wandb_tags: tuple[str, ...] = ("baseline", "PlaNet",)
    wandb_mode: Literal['online', 'offline', 'disabled', 'shared'] = "disabled"

    # video rendering
    video_dir:str = f"video/planet"
    video_fps:int = 15

@dataclass(frozen=True)
class PlaNetDebugConfig(PlaNetTrainConfig):
    domain: str = "reacher"
    task: str = "easy"
    episode_length: int = 64
    action_repeat: int = 4

    s_dim: int = 5
    h_dim: int = 7
    mlp_hidden_dim: int = 16
    mlp_layers: int = 1

    planning_horizon: int = 3
    num_candidates: int = 16
    num_elites: int = 4
    refine_iterations: int = 2

    replay_capacity: int = 100
    batch_size: int = 4
    sequence_length: int = 8
    seed_episodes: int = 2
    train_steps: int = 25
    collect_interval: int = 10
    episodes_per_collection: int = 1
    exploration_noise: float = 0.0
    log_interval: int = 5

    eval_interval: int = 10
    eval_episodes: int = 2

    checkpoint_interval: int = 10
    milestone_env_steps: tuple[int, ...] = ()

    wandb_mode:Literal['online', 'offline', 'disabled', 'shared'] = "disabled"
