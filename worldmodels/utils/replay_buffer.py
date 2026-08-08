import numpy as np
import torch
import einops as op

from collections import deque
from typing import Tuple

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def preprocess_obs(
    obs: torch.Tensor,
    train: bool,
    image_bits: int,
) -> torch.Tensor:
    '''
    Bit-depth reduction + dequantization, centered to [-0.5, 0.5).
    '''
    if not 1 <= image_bits <= 8:
        raise ValueError("image_bits must be between 1 and 8")

    obs = obs.float()
    quantization_factor = 2 ** (8 - image_bits)
    num_bins = 2 ** image_bits
    obs = torch.floor(obs / quantization_factor)
    if train:
        obs = (obs + torch.rand_like(obs))
    else:
        obs = obs + 0.5
    obs = (obs / num_bins) - 0.5
    return obs


class ReplayBuffer:
    '''
    Store complete episodes and sample aligned sequences.
    '''
    def __init__(self, capacity:int, image_bits:int) -> None:
        self.capacity = capacity
        self.image_bits = image_bits
        self.buffer = deque()
        self.num_steps = 0

    def add_episode(self, obs:np.ndarray, actions:np.ndarray, rewards:np.ndarray,
                    physics:np.ndarray|None=None) -> None:
        '''
        Add one complete episode and evict old episodes when over capacity.

        Inputs:
        obs (T+1, H, W, 3): raw image input from DMC, pixel value 0-255, from t=0 to t=T
        actions (T, action_dim): actions for transitions obs[:-1] -> obs[1:]
        rewards (T): rewards emitted AFTER each action
        physics (T+1, state_dim), optional: MuJoCo state per frame, aligned
            with obs. Store it when provided; it is not sampled into training
            batches.

        Store observations as uint8 tensors in [T+1, 3, 64, 64] layout.
        Store actions and rewards as float32 tensors. ``num_steps`` counts
        transitions, not images. Capacity is enforced by removing complete
        episodes from the left (oldest first).
        '''
        obs_t = torch.as_tensor(obs, dtype=torch.uint8)
        obs_t = op.rearrange(obs_t, "t h w c -> t c h w")
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)

        episode = {
            "obs": obs_t,
            "actions": actions_t,
            "rewards": rewards_t,
        }
        if physics is not None:
            episode["physics"] = torch.as_tensor(physics, dtype=torch.float32)

        self.buffer.append(episode)
        self.num_steps += actions_t.shape[0]

        while self.num_steps > self.capacity and len(self.buffer) > 1:
            evicted = self.buffer.popleft()
            self.num_steps -= evicted["actions"].shape[0]

    def state_dict(self) -> dict:
        '''
        Return the replay data stored in a resume checkpoint.

        Store pixels directly because MuJoCo state alone cannot reproduce
        randomized visual details such as Reacher's target position.
        '''
        return {
            "episodes": list(self.buffer),
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state:dict) -> None:
        '''
        Replace the current replay data with state. Every saved episode must
        include pixels because training batches sample those pixels directly.
        '''
        episodes = state["episodes"]
        missing_pixels = [i for i, episode in enumerate(episodes) if "obs" not in episode]
        if missing_pixels:
            raise ValueError(
                "resume checkpoint omits replay pixels; physics-only reconstruction "
                "is unsafe for task-randomized scenes"
            )
        self.buffer = deque(episodes)
        self.num_steps = state["num_steps"]

    def sample_batch(self, batch_size:int, seq_len:int, device=None) -> Batch:
        '''
        Sample aligned fixed-length chunks from complete episodes.

        Inputs:
        batch_size: number of chunks in a batch (=B)
        seq_len: number of transitions in one chunk (=L)
        device: the device sampled batch would be on

        Output:
        obs [B, L, 3, 64, 64], actions [B, L, A], rewards [B, L]

        Every observation must align with the action and reward that led into
        it. Sample with replacement and never cross episode boundaries.
        '''
        eligible = [ep for ep in self.buffer if ep["actions"].shape[0] >= seq_len]
        if not eligible:
            raise ValueError(
                f"No episodes long enough to sample a chunk of seq_len={seq_len}"
            )

        obs_chunks = []
        action_chunks = []
        reward_chunks = []

        for _ in range(batch_size):
            episode = eligible[np.random.randint(len(eligible))]
            episode_len = episode["actions"].shape[0]
            start = np.random.randint(0, episode_len - seq_len + 1)
            end = start + seq_len

            # obs is stored with an extra leading frame (t=0..T); destination
            # observations for transitions [start, end) are obs[start+1:end+1].
            obs_chunks.append(episode["obs"][start + 1:end + 1])
            action_chunks.append(episode["actions"][start:end])
            reward_chunks.append(episode["rewards"][start:end])

        obs = torch.stack(obs_chunks, dim=0)
        actions = torch.stack(action_chunks, dim=0)
        rewards = torch.stack(reward_chunks, dim=0)

        obs = preprocess_obs(obs, train=True, image_bits=self.image_bits)
        actions = actions.to(torch.float32)
        rewards = rewards.to(torch.float32)

        if device is not None:
            obs = obs.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)

        return obs, actions, rewards
