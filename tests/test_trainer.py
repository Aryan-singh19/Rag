import os
import sys
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch
from dm_control import suite

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from worldmodels.agents import PlaNet
from worldmodels.config import PlaNetDebugConfig, PlaNetTrainConfig
from worldmodels.trainer import collect_random_episode
from worldmodels.trainer.planet_trainer import (
    eval_agent,
    get_planet_episode,
    train_planet,
)


TEST_CONFIG = PlaNetTrainConfig(
    episode_length=8,
    action_repeat=4,
    s_dim=5,
    h_dim=7,
    mlp_hidden_dim=16,
    mlp_layers=1,
    planning_horizon=2,
    num_candidates=4,
    num_elites=2,
    refine_iterations=1,
    exploration_noise=0.0,
    eval_episodes=2,
)


def make_env():
    return suite.load(
        domain_name=TEST_CONFIG.domain,
        task_name=TEST_CONFIG.task,
        task_kwargs={"random": TEST_CONFIG.seed},
    )


def make_agent(action_dim: int) -> PlaNet:
    return PlaNet(
        s_dim=TEST_CONFIG.s_dim,
        h_dim=TEST_CONFIG.h_dim,
        mlp_hidden_dim=TEST_CONFIG.mlp_hidden_dim,
        mlp_layers=TEST_CONFIG.mlp_layers,
        mlp_activation=TEST_CONFIG.mlp_activation,
        action_dim=action_dim,
        device="cpu",
        planning_horizon=TEST_CONFIG.planning_horizon,
        num_candidates=TEST_CONFIG.num_candidates,
        num_elites=TEST_CONFIG.num_elites,
        refine_iterations=TEST_CONFIG.refine_iterations,
        cem_min_std=TEST_CONFIG.cem_min_std,
        encoded_dim=TEST_CONFIG.encoded_dim,
        min_std=TEST_CONFIG.min_std,
        learning_rate=TEST_CONFIG.learning_rate,
        adam_epsilon=TEST_CONFIG.adam_epsilon,
        gradient_clip_norm=TEST_CONFIG.gradient_clip_norm,
        free_nats=TEST_CONFIG.free_nats,
        kl_loss_scale=TEST_CONFIG.kl_loss_scale,
        reconstruction_loss_scale=TEST_CONFIG.reconstruction_loss_scale,
        reward_loss_scale=TEST_CONFIG.reward_loss_scale,
    )


def assert_episode(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    physics: np.ndarray,
    action_spec,
) -> None:
    num_decisions = TEST_CONFIG.episode_length // TEST_CONFIG.action_repeat

    assert obs.shape == (
        num_decisions + 1,
        TEST_CONFIG.image_size,
        TEST_CONFIG.image_size,
        3,
    )
    assert actions.shape == (num_decisions, action_spec.shape[0])
    assert rewards.shape == (num_decisions,)
    assert obs.dtype == np.uint8
    assert actions.dtype == action_spec.dtype
    assert np.isfinite(rewards).all()
    assert (actions >= action_spec.minimum).all()
    assert (actions <= action_spec.maximum).all()

    # physics states align one to one with rendered frames
    assert physics.shape[0] == obs.shape[0]
    assert physics.ndim == 2
    assert physics.dtype == np.float64
    assert np.isfinite(physics).all()


def test_collect_random_episode_real_env():
    np.random.seed(TEST_CONFIG.seed)
    env = make_env()
    try:
        episode = collect_random_episode(env, TEST_CONFIG)
        assert_episode(*episode, env.action_spec())
    finally:
        env.close()


def test_get_planet_episode_real_env_with_cem():
    np.random.seed(TEST_CONFIG.seed)
    torch.manual_seed(TEST_CONFIG.seed)
    env = make_env()
    try:
        action_dim = env.action_spec().shape[0]
        agent = make_agent(action_dim)

        episode = get_planet_episode(env, agent, TEST_CONFIG, training=True)
        assert_episode(*episode, env.action_spec())
    finally:
        env.close()


def test_eval_agent_real_env_with_cem():
    np.random.seed(TEST_CONFIG.seed)
    torch.manual_seed(TEST_CONFIG.seed)
    env = make_env()
    try:
        agent = make_agent(env.action_spec().shape[0])
        agent.train()

        metrics, formatted = eval_agent(TEST_CONFIG, env, agent)

        assert agent.training
        assert set(metrics) == {
            "eval/return_mean",
            "eval/return_std",
            "eval/return_min",
            "eval/return_max",
        }
        assert all(isinstance(value, float) for value in metrics.values())
        assert all(np.isfinite(value) for value in metrics.values())
        assert "eval/return_mean=" in formatted
    finally:
        env.close()


def test_train_planet_debug_config():
    output = StringIO()

    with redirect_stdout(output):
        train_planet(PlaNetDebugConfig())

    captured = output.getvalue()
    assert "step=25\n" in captured
    assert "eval/return_mean=" in captured


if __name__ == "__main__":
    test_collect_random_episode_real_env()
    test_get_planet_episode_real_env_with_cem()
    test_eval_agent_real_env_with_cem()
    test_train_planet_debug_config()
