import os
import sys

import numpy as np
import torch
from dm_env import specs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from worldmodels.agents.planet import PlaNet
from worldmodels.config import PlaNetTrainConfig
from worldmodels.models import RSSMState


TEST_CONFIG = PlaNetTrainConfig(
    min_std=0.2,
    adam_epsilon=2e-4,
    gradient_clip_norm=7.0,
    free_nats=1.5,
    kl_loss_scale=0.5,
    reconstruction_loss_scale=2.0,
    reward_loss_scale=3.0,
    planning_horizon=3,
    num_candidates=8,
    num_elites=4,
    refine_iterations=2,
)


def make_planet(
    s_dim: int,
    h_dim: int,
    mlp_hidden_dim: int,
    mlp_layers: int,
    action_dim: int,
) -> PlaNet:
    return PlaNet(
        s_dim=s_dim,
        h_dim=h_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
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


def test_planet_forward_shapes():
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 4
    s_dim = 30
    h_dim = 200
    action_dim = 6

    model = make_planet(
        s_dim=s_dim,
        h_dim=h_dim,
        mlp_hidden_dim=200,
        mlp_layers=2,
        action_dim=action_dim,
    )

    obs = torch.randn(batch_size, sequence_length, 3, 64, 64)
    actions = torch.randn(batch_size, sequence_length, action_dim)

    torch.manual_seed(17)
    posterior, prior, reward, recon = model(obs, actions)

    assert posterior.s_t.shape == (batch_size, sequence_length, s_dim)
    assert posterior.h_t.shape == (batch_size, sequence_length, h_dim)
    assert prior.s_t.shape == (batch_size, sequence_length, s_dim)
    assert prior.h_t.shape == (batch_size, sequence_length, h_dim)
    assert reward.shape == (batch_size, sequence_length, 1)
    assert recon.shape == (batch_size, sequence_length, 3, 64, 64)
    assert model.optimizer.defaults["lr"] == TEST_CONFIG.learning_rate
    assert model.optimizer.defaults["eps"] == TEST_CONFIG.adam_epsilon
    assert model.gradient_clip_norm == TEST_CONFIG.gradient_clip_norm
    assert model.free_nats == TEST_CONFIG.free_nats
    assert model.rssm.prior.min_std == TEST_CONFIG.min_std
    assert model.rssm.posterior.min_std == TEST_CONFIG.min_std

    target_rewards = torch.zeros(batch_size, sequence_length)
    torch.manual_seed(17)
    losses = model.loss(obs, actions, target_rewards)
    assert set(losses) == {"kl_loss", "recon_loss", "reward_loss", "total"}
    assert all(value.ndim == 0 for value in losses.values())
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(losses[name] >= 0 for name in ("kl_loss", "recon_loss", "reward_loss"))
    expected_total = (
        TEST_CONFIG.kl_loss_scale * losses["kl_loss"]
        + TEST_CONFIG.reconstruction_loss_scale * losses["recon_loss"]
        + TEST_CONFIG.reward_loss_scale * losses["reward_loss"]
    )
    assert torch.allclose(losses["total"], expected_total)


def test_planet_loss_known_values():
    model = make_planet(s_dim=1, h_dim=2, mlp_hidden_dim=8, mlp_layers=1, action_dim=1)
    zeros_s = torch.zeros(1, 2, 1)
    zeros_h = torch.zeros(1, 2, 2)
    posterior = RSSMState(
        zeros_s,
        zeros_h,
        torch.tensor([[[0.0], [2.0]]]),
        torch.tensor([[[1.0], [2.0]]]),
    )
    prior = RSSMState(zeros_s, zeros_h, torch.zeros_like(zeros_s), torch.ones_like(zeros_s))
    obs = torch.zeros(1, 2, 1, 2, 2)
    recon = torch.ones_like(obs)
    recon[:, 1] = 2.0
    reward_hat = torch.tensor([[[3.0], [2.0]]])
    rewards = torch.tensor([[1.0, -1.0]])
    model.forward = lambda _obs, _actions: (posterior, prior, reward_hat, recon)

    losses = model.loss(obs, torch.zeros(1, 2, 1), rewards)
    expected = {
        "kl_loss": 0.6534264,
        "recon_loss": 5.0,
        "reward_loss": 3.25,
        "total": 20.0767136,
    }
    for name, value in expected.items():
        assert torch.allclose(losses[name], torch.tensor(value), atol=1e-5), name


def test_plan_cem_shape_bounds_and_no_grad():
    torch.manual_seed(0)

    action_dim = 2
    model = make_planet(
        s_dim=5,
        h_dim=7,
        mlp_hidden_dim=16,
        mlp_layers=1,
        action_dim=action_dim,
    )

    state = model.rssm.initialize_RSSM_state(batch_size=1)
    action_spec = specs.BoundedArray(
        shape=(action_dim,),
        dtype=np.float32,
        minimum=-0.25,
        maximum=0.25,
    )

    action = model.plan_cem(state, action_spec)

    assert action.shape == (action_dim,)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()
    assert (action >= -0.25).all()
    assert (action <= 0.25).all()


def test_plan_cem_moves_toward_a_known_multistep_optimum():
    torch.manual_seed(0)
    action_dim = 2
    model = make_planet(
        s_dim=action_dim,
        h_dim=3,
        mlp_hidden_dim=16,
        mlp_layers=1,
        action_dim=action_dim,
    )
    model.planning_horizon = 3
    model.num_candidates = 1024
    model.num_elites = 64
    model.refine_iterations = 5

    step_targets = torch.tensor([
        [-0.6, 0.6],
        [0.8, -0.4],
        [0.7, -0.5],
    ])
    expected = step_targets.mean(dim=0)
    calls = []

    def fake_imagine(_start, candidate_actions):
        calls.append(candidate_actions.shape)
        feats = (
            candidate_actions[:, :1]
            - step_targets.to(candidate_actions).unsqueeze(0)
        )
        h = torch.zeros(
            (*feats.shape[:-1], model.rssm.h_dim),
            device=feats.device,
            dtype=feats.dtype,
        )
        return RSSMState(feats, h, feats, torch.ones_like(feats))

    class QuadraticReward(torch.nn.Module):
        def forward(self, feats):
            return -torch.square(feats).sum(dim=-1, keepdim=True)

    model.rssm.imagine = fake_imagine
    model.rssm.get_feat = lambda state: state.s_t
    model.reward_model = QuadraticReward()

    action_spec = specs.BoundedArray(
        shape=(action_dim,),
        dtype=np.float32,
        minimum=-1.0,
        maximum=1.0,
    )
    action = model.plan_cem(model.rssm.initialize_RSSM_state(1), action_spec)

    assert calls == [
        (model.num_candidates, model.planning_horizon, action_dim)
    ] * model.refine_iterations
    assert torch.allclose(action, expected, atol=0.08)


def test_act_strict_tensor_api():
    torch.manual_seed(0)

    action_dim = 2
    s_dim = 5
    h_dim = 7
    model = make_planet(
        s_dim=s_dim,
        h_dim=h_dim,
        mlp_hidden_dim=16,
        mlp_layers=1,
        action_dim=action_dim,
    )

    obs = torch.randn(1, 1, 3, 64, 64)
    prev_action = torch.zeros(1, action_dim)
    prev_state = model.rssm.initialize_RSSM_state(batch_size=1)
    action_spec = specs.BoundedArray(
        shape=(action_dim,),
        dtype=np.float32,
        minimum=-1.0,
        maximum=1.0,
    )
    calls = {}

    def fake_plan_cem(state, action_spec):
        calls["state_batch"] = state.s_t.shape[0]
        calls["action_shape"] = action_spec.shape
        return torch.full((action_dim,), 0.5)

    original_plan_cem = model.plan_cem
    model.plan_cem = fake_plan_cem
    try:
        action, state = model.act(obs, prev_action, prev_state, action_spec)
    finally:
        model.plan_cem = original_plan_cem

    assert action.shape == (1, action_dim)
    assert torch.equal(action, torch.full((1, action_dim), 0.5))
    assert state.s_t.shape == (1, s_dim)
    assert state.h_t.shape == (1, h_dim)
    assert calls == {
        "state_batch": 1,
        "action_shape": (action_dim,),
    }


if __name__ == "__main__":
    test_planet_forward_shapes()
    test_plan_cem_shape_bounds_and_no_grad()
    test_plan_cem_moves_toward_a_known_multistep_optimum()
    test_act_strict_tensor_api()
