import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from worldmodels.utils import ReplayBuffer, preprocess_obs


def make_episode(offset, num_steps, action_dim=2):
    obs = np.zeros((num_steps + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.zeros((num_steps, action_dim), dtype=np.float32)
    rewards = np.zeros((num_steps,), dtype=np.float32)

    for t in range(num_steps + 1):
        obs[t].fill(offset + t)

    for t in range(num_steps):
        action_id = offset + t
        actions[t, 0] = action_id
        actions[t, 1] = -action_id
        rewards[t] = action_id + 0.5

    return obs, actions, rewards


def test_replay_buffer_adds_episode_and_evicts_oldest():
    buffer = ReplayBuffer(capacity=6, image_bits=8)
    buffer.add_episode(*make_episode(offset=10, num_steps=4))

    assert buffer.num_steps == 4
    assert len(buffer.buffer) == 1

    episode = buffer.buffer[0]
    assert episode["obs"].shape == (5, 3, 64, 64)
    assert episode["actions"].shape == (4, 2)
    assert episode["rewards"].shape == (4,)
    assert episode["obs"].dtype == torch.uint8
    assert episode["actions"].dtype == torch.float32
    assert episode["rewards"].dtype == torch.float32

    buffer.add_episode(*make_episode(offset=30, num_steps=3))

    assert buffer.num_steps == 3
    assert len(buffer.buffer) == 1
    assert buffer.buffer[0]["obs"][0, 0, 0, 0].item() == 30


def test_replay_buffer_samples_aligned_chunks():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    buffer = ReplayBuffer(capacity=100, image_bits=8)
    buffer.add_episode(*make_episode(offset=10, num_steps=5))
    buffer.add_episode(*make_episode(offset=30, num_steps=6))
    buffer.add_episode(*make_episode(offset=200, num_steps=2))

    obs, actions, rewards = buffer.sample_batch(batch_size=64, seq_len=3, device="cpu")

    assert obs.shape == (64, 3, 3, 64, 64)
    assert actions.shape == (64, 3, 2)
    assert rewards.shape == (64, 3)
    assert obs.dtype == torch.float32
    assert actions.dtype == torch.float32
    assert rewards.dtype == torch.float32
    assert obs.device.type == "cpu"

    obs_ids = torch.floor(
        (obs[:, :, 0, 0, 0] + 0.5) * 256.0
    )
    action_ids = obs_ids - 1

    assert torch.allclose(actions[:, :, 0], action_ids)
    assert torch.allclose(actions[:, :, 1], -action_ids)
    assert torch.allclose(rewards, action_ids + 0.5)
    assert obs_ids.max() < 200

    starts = obs_ids[:, 0]
    first_episode_starts = starts[starts < 30]
    second_episode_starts = starts[(starts >= 30) & (starts < 200)]
    assert torch.unique(first_episode_starts).numel() >= 2
    assert torch.unique(second_episode_starts).numel() >= 2


def test_replay_buffer_requires_long_enough_episode():
    buffer = ReplayBuffer(capacity=100, image_bits=8)
    buffer.add_episode(*make_episode(offset=50, num_steps=2))

    try:
        buffer.sample_batch(batch_size=1, seq_len=3)
    except ValueError as error:
        assert "No episodes long enough" in str(error)
    else:
        raise AssertionError("Expected ValueError for too-short replay episodes")


def test_preprocess_obs_respects_image_bits():
    pixels = torch.tensor([0, 7, 8, 255], dtype=torch.uint8)

    processed = preprocess_obs(
        pixels,
        train=False,
        image_bits=5,
    )

    expected = torch.tensor([
        0.5 / 32.0 - 0.5,
        0.5 / 32.0 - 0.5,
        1.5 / 32.0 - 0.5,
        31.5 / 32.0 - 0.5,
    ])
    assert torch.allclose(processed, expected)


if __name__ == "__main__":
    test_replay_buffer_adds_episode_and_evicts_oldest()
    test_replay_buffer_samples_aligned_chunks()
    test_replay_buffer_requires_long_enough_episode()
    test_preprocess_obs_respects_image_bits()
