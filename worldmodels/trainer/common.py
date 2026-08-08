import numpy as np
import torch
import einops as op

from pathlib import Path
from dm_env import Environment, TimeStep, specs
from worldmodels.utils import preprocess_obs
from worldmodels.config import PlaNetTrainConfig

def get_reward(time_step:TimeStep):
    return float(time_step.reward) if time_step.reward is not None else 0.0

def render_pixels(env, image_size=64, camera_id=0) -> np.ndarray:
    '''
    render environment states into images
    '''
    return env.physics.render(height=image_size, width=image_size, camera_id=camera_id)

def get_physics_state(env) -> np.ndarray:
    '''
    Return the current MuJoCo state vector (qpos, qvel, act) as float64.
    '''
    return np.asarray(env.physics.get_state(), dtype=np.float64).copy()

def obs_to_agent_input(
    pixels: np.ndarray,
    device,
    image_bits: int,
) -> torch.Tensor:
    '''
    process raw pixel into (1, 1, C, H, W) torch tensor for agent.act
    '''
    obs = torch.from_numpy(np.ascontiguousarray(pixels))
    obs = op.rearrange(obs, "H W C -> C H W")
    obs = obs.unsqueeze(0).unsqueeze(0)
    obs = preprocess_obs(
        obs,
        train=False,
        image_bits=image_bits,
    )
    obs = obs.to(device)
    return obs

def sample_random_action(action_spec:specs.BoundedArray) -> np.ndarray:
    '''
    randomly sample an action in action space
    '''
    action = np.random.uniform(
        low = action_spec.minimum,
        high = action_spec.maximum,
        size = action_spec.shape
    ).astype(action_spec.dtype)

    return action

def step_with_action_repeat(env:Environment, action:np.ndarray, 
                            action_repeat:int) -> tuple[TimeStep, float]:
    '''
    step through the environment multiple times with same action
    '''
    assert action_repeat >= 1
    
    total_reward = 0.0
    time_step = None

    for _ in range(action_repeat):
        time_step = env.step(action)
        total_reward += get_reward(time_step)
        if time_step.last():
            break

    assert time_step is not None
    
    return time_step, total_reward

def collect_random_episode(env, cfg:PlaNetTrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    collect a single episode under a random policy

    Inputs:
    env: a DeepMind Control Suite environment
    cfg: training config

    Returns:
    obs (T+1, H, W, 3): raw observation images
    actions (T, action_dim): actions for transitions obs[:-1] -> obs[1:]
    rewards (T): rewards emitted AFTER each action
    physics (T+1, state_dim): MuJoCo state per frame, aligned with obs
    '''
    time_step = env.reset()
    action_spec:specs.BoundedArray = env.action_spec()

    obs = []
    actions = []
    rewards = []
    physics = []

    obs.append(render_pixels(env, cfg.image_size, cfg.camera_id))
    physics.append(get_physics_state(env))

    max_decisions = cfg.episode_length // cfg.action_repeat

    for _ in range(max_decisions):
        action = sample_random_action(action_spec)
        time_step, reward = step_with_action_repeat(env, action, cfg.action_repeat)
        obs.append(render_pixels(env, cfg.image_size, cfg.camera_id))
        physics.append(get_physics_state(env))
        actions.append(action)
        rewards.append(reward)
        if time_step.last():
            break


    obs = np.stack(obs, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0)
    physics = np.stack(physics, axis=0)

    return obs, actions, rewards, physics

def select_device() -> torch.device:
    if torch.cuda.is_available():
      return torch.device("cuda")
    if torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")

def write_side_by_side_video(
    actual_frames: np.ndarray,
    imagined_frames: np.ndarray | torch.Tensor,
    output_path: str | Path,
    fps: int = 15,
) -> Path:
    '''
    Write a side-by-side video comparing real rendered frames against decoded imagination.

    actual_frames: (T, H, W, 3), uint8 pixels from the environment
    imagined_frames: (T, C, H, W) or (T, H, W, C), decoder outputs in [-0.5, 0.5]
    '''
    import imageio.v3 as iio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual = np.asarray(actual_frames)
    if actual.ndim != 4 or actual.shape[-1] != 3:
        raise ValueError("actual_frames must have shape (T, H, W, 3)")

    if isinstance(imagined_frames, torch.Tensor):
        imagined = imagined_frames.detach().cpu()
        if imagined.ndim != 4:
            raise ValueError("imagined_frames must have shape (T, C, H, W) or (T, H, W, C)")
        if imagined.shape[1] == 3:
            imagined = op.rearrange(imagined, "T C H W -> T H W C")
        elif imagined.shape[-1] != 3:
            raise ValueError("imagined_frames must have 3 color channels")
        imagined = ((imagined + 0.5) * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    else:
        imagined = np.asarray(imagined_frames)
        if imagined.ndim != 4:
            raise ValueError("imagined_frames must have shape (T, C, H, W) or (T, H, W, C)")
        if imagined.shape[1] == 3:
            imagined = op.rearrange(imagined, "T C H W -> T H W C")
        elif imagined.shape[-1] != 3:
            raise ValueError("imagined_frames must have 3 color channels")
        if imagined.dtype != np.uint8:
            imagined = np.clip((imagined + 0.5) * 255.0, 0, 255).astype(np.uint8)

    if actual.dtype != np.uint8:
        actual = np.clip(actual, 0, 255).astype(np.uint8)

    T = min(actual.shape[0], imagined.shape[0])
    if T == 0:
        raise ValueError("cannot write video with zero frames")

    actual = actual[:T]
    imagined = imagined[:T]
    if actual.shape[1:3] != imagined.shape[1:3]:
        raise ValueError("actual_frames and imagined_frames must have the same height and width")

    side_by_side = np.concatenate([actual, imagined], axis=2)
    iio.imwrite(output_path, side_by_side, fps=fps, codec="libx264")
    return output_path
