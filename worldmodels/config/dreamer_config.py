from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class DreamerTrainConfig:
    '''
    Dreamer V1 on DMC from pixels. World-model fields mirror
    PlaNetTrainConfig; actor-critic fields and the optimization split follow
    the paper and danijar/dreamer's TF2 defaults (dense nets 400 units ELU:
    actor 4 layers, value 3, reward 2; model/actor/value lrs 6e-4/8e-5/8e-5,
    grad clip 100, horizon 15, gamma 0.99, lambda 0.95, action repeat 2).

    This implementation keeps PlaNet's 5-bit images; official Dreamer V1 uses
    8-bit images. This makes PlaNet and Dreamer use the same preprocessing,
    but results may differ from published Dreamer curves.
    '''
    # path
    checkpoint_dir: str = "checkpoints/dreamer"

    # Runtime
    seed: int = 0

    # Environment
    domain: str = "walker"
    task: str = "walk"
    image_size: int = 64
    image_bits: int = 5
    camera_id: int = 0
    action_repeat: int = 2
    episode_length: int = 1000

    # Model (RSSM identical to PlaNet; dense activation is ELU here)
    s_dim: int = 30
    h_dim: int = 200
    encoded_dim: int = 1024
    mlp_hidden_dim: int = 200
    mlp_layers: int = 2
    mlp_activation: str = "elu"
    min_std: float = 0.1

    # Heads
    reward_hidden_dim: int = 400
    reward_layers: int = 2
    value_hidden_dim: int = 400
    value_layers: int = 3
    actor_hidden_dim: int = 400
    actor_layers: int = 4
    actor_init_std: float = 5.0
    actor_mean_scale: float = 5.0
    actor_min_std: float = 1e-4

    # Imagination and returns
    imagination_horizon: int = 15
    discount_gamma: float = 0.99
    return_lambda: float = 0.95

    # Optimization and loss
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    adam_epsilon: float = 1e-7
    gradient_clip_norm: float = 100.0
    free_nats: float = 3.0
    kl_loss_scale: float = 1.0
    reconstruction_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0

    # Replay and training (schedule matches the PlaNet trainer: one episode
    # collected per collect_interval train steps, which at repeat 2 equals
    # Dreamer's 100 train steps per 1000 env steps)
    replay_capacity: int = 500_000
    batch_size: int = 50
    sequence_length: int = 50
    seed_episodes: int = 5
    train_steps: int = 100_000
    collect_interval: int = 100
    episodes_per_collection: int = 1
    exploration_noise: float = 0.3
    log_interval: int = 100
    eval_interval: int = 1000
    eval_episodes: int = 10

    # Checkpointing and resume
    checkpoint_interval: int = 1000
    milestone_env_steps: tuple[int, ...] = (500_000, 1_000_000)

    # wandb
    entity: str | None = None
    project: str = "11685-rl-guided-project"
    run_name: str = "dreamer"
    wandb_tags: tuple[str, ...] = ("baseline", "DreamerV1",)
    wandb_mode: Literal['online', 'offline', 'disabled', 'shared'] = "disabled"

    # video rendering
    video_dir: str = "video/dreamer"
    video_fps: int = 15

@dataclass(frozen=True)
class DreamerDebugConfig(DreamerTrainConfig):
    episode_length: int = 64
    action_repeat: int = 4

    s_dim: int = 5
    h_dim: int = 7
    mlp_hidden_dim: int = 16
    mlp_layers: int = 1

    reward_hidden_dim: int = 16
    reward_layers: int = 1
    value_hidden_dim: int = 16
    value_layers: int = 1
    actor_hidden_dim: int = 16
    actor_layers: int = 1

    imagination_horizon: int = 4

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

    wandb_mode: Literal['online', 'offline', 'disabled', 'shared'] = "disabled"
