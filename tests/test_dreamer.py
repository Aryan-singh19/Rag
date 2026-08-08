'''
Dreamer V1 unit tests: lambda-return math, actor distribution behavior,
imagination shapes, optimizer separation in the update path, and
checkpoint round trips with multiple optimizers.
'''
import numpy as np
import pytest
import torch

from dataclasses import asdict

from worldmodels.agents import Dreamer
from worldmodels.agents.dreamer import lambda_return
from worldmodels.config import DreamerDebugConfig
from worldmodels.models import RSSMState
from worldmodels.networks import TanhGaussianActor
from worldmodels.networks.actor import tanh_dist_mode
from worldmodels.trainer import checkpointing
from worldmodels.utils import ReplayBuffer

ACTION_DIM = 3


def make_agent(cfg=None) -> Dreamer:
    cfg = cfg or DreamerDebugConfig()
    return Dreamer(
        s_dim=cfg.s_dim,
        h_dim=cfg.h_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        mlp_layers=cfg.mlp_layers,
        mlp_activation=cfg.mlp_activation,
        action_dim=ACTION_DIM,
        device=torch.device("cpu"),
        encoded_dim=cfg.encoded_dim,
        min_std=cfg.min_std,
        free_nats=cfg.free_nats,
        kl_loss_scale=cfg.kl_loss_scale,
        reconstruction_loss_scale=cfg.reconstruction_loss_scale,
        reward_loss_scale=cfg.reward_loss_scale,
        reward_hidden_dim=cfg.reward_hidden_dim,
        reward_layers=cfg.reward_layers,
        value_hidden_dim=cfg.value_hidden_dim,
        value_layers=cfg.value_layers,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_layers=cfg.actor_layers,
        actor_init_std=cfg.actor_init_std,
        actor_mean_scale=cfg.actor_mean_scale,
        actor_min_std=cfg.actor_min_std,
        imagination_horizon=cfg.imagination_horizon,
        discount_gamma=cfg.discount_gamma,
        return_lambda=cfg.return_lambda,
        model_lr=cfg.model_lr,
        actor_lr=cfg.actor_lr,
        value_lr=cfg.value_lr,
        adam_epsilon=cfg.adam_epsilon,
        gradient_clip_norm=cfg.gradient_clip_norm,
    )


def snapshot(params):
    return [p.detach().clone() for p in params]


def max_param_delta(before, params) -> float:
    return max((b - p.detach()).abs().max().item() for b, p in zip(before, params))


# ---- lambda_return ---------------------------------------------------------

def test_lambda_return_known_values():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    bootstrap = torch.tensor([40.0])
    returns = lambda_return(rewards, values, bootstrap, gamma=0.9, lam=0.8)
    expected = torch.tensor([[30.1456], [35.48], [39.0]])
    assert returns.shape == (3, 1)
    assert torch.allclose(returns, expected, atol=1e-5)


def test_lambda_return_lam_zero_is_one_step_td():
    '''lam=0 collapses to r_t + gamma * v_{t+1}'''
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    values = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    bootstrap = torch.tensor([70.0, 80.0])
    returns = lambda_return(rewards, values, bootstrap, gamma=0.5, lam=0.0)
    expected = torch.tensor([[16.0, 22.0], [28.0, 34.0], [40.0, 46.0]])
    assert torch.allclose(returns, expected)


def test_lambda_return_lam_one_is_monte_carlo():
    '''lam=1 collapses to the discounted sum of rewards plus bootstrap'''
    rewards = torch.rand(3, 2)
    values = torch.rand(3, 2)  # must not matter except the bootstrap
    bootstrap = torch.rand(2)
    returns = lambda_return(rewards, values, bootstrap, gamma=0.9, lam=1.0)
    expected_first = rewards[0] + 0.9 * (rewards[1] + 0.9 * (rewards[2] + 0.9 * bootstrap))
    assert torch.allclose(returns[0], expected_first, atol=1e-6)


# ---- actor -----------------------------------------------------------------

def test_actor_sample_shape_and_bounds():
    actor = TanhGaussianActor(10, ACTION_DIM, 16, 2, "elu")
    feat = torch.randn(7, 10)
    dist = actor(feat)
    action = dist.sample()
    assert action.shape == (7, ACTION_DIM)
    # float32 tanh saturates to exactly +/-1 for wide base Gaussians; DMC
    # bounds are inclusive, and log_prob stays finite via the transform cache
    assert (action >= -1.0).all() and (action <= 1.0).all()
    logp = dist.log_prob(action)
    assert logp.shape == (7,)
    assert torch.isfinite(logp).all()


def test_actor_rsample_carries_gradient():
    actor = TanhGaussianActor(10, ACTION_DIM, 16, 2, "elu")
    feat = torch.randn(4, 10)
    action = actor(feat).rsample()
    action.sum().backward()
    grads = [p.grad for p in actor.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_tanh_dist_mode_shape_and_bounds():
    actor = TanhGaussianActor(10, ACTION_DIM, 16, 2, "elu")
    feat = torch.randn(5, 10)
    mode = tanh_dist_mode(actor(feat), num_samples=32)
    assert mode.shape == (5, ACTION_DIM)
    assert (mode >= -1.0).all() and (mode <= 1.0).all()


# ---- imagination -----------------------------------------------------------

def test_imagine_rollout_shapes():
    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)
    n = 6
    start = agent.rssm.initialize_RSSM_state(n)
    feats = agent.imagine_rollout(start)
    assert feats.shape == (cfg.imagination_horizon, n, cfg.s_dim + cfg.h_dim)


def test_imagine_rollout_uses_canonical_feature_order():
    cfg = DreamerDebugConfig(s_dim=2, h_dim=3, imagination_horizon=1)
    agent = make_agent(cfg)
    start = RSSMState(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[10.0, 20.0, 30.0]]),
        torch.zeros(1, 2),
        torch.ones(1, 2),
    )

    def fake_imagine_step(_action, state):
        return RSSMState(state.s_t + 1.0, state.h_t + 10.0, state.latent_mean, state.latent_std)

    agent.rssm.imagine_step = fake_imagine_step
    actor_inputs = []
    hook = agent.actor.register_forward_pre_hook(
        lambda _module, args: actor_inputs.append(args[0])
    )
    try:
        feats = agent.imagine_rollout(start)
    finally:
        hook.remove()

    assert torch.equal(actor_inputs[0], agent.rssm.get_feat(start))
    assert torch.equal(feats, torch.tensor([[[2.0, 3.0, 20.0, 30.0, 40.0]]]))


def test_imagine_rollout_carries_actor_and_dynamics_gradients():
    torch.manual_seed(0)
    agent = make_agent()
    actor_inputs_require_grad = []
    sampled_actions = []

    def record_actor_sample(_module, args, distribution):
        actor_inputs_require_grad.append(args[0].requires_grad)
        original_rsample = distribution.rsample

        def rsample(*sample_args, **sample_kwargs):
            action = original_rsample(*sample_args, **sample_kwargs)
            action.retain_grad()
            sampled_actions.append(action)
            return action

        distribution.rsample = rsample

    hook = agent.actor.register_forward_hook(record_actor_sample)
    try:
        feats = agent.imagine_rollout(agent.rssm.initialize_RSSM_state(5))
    finally:
        hook.remove()
    feats[-1].sum().backward()

    assert actor_inputs_require_grad and not any(actor_inputs_require_grad)
    assert len(sampled_actions) == agent.imagination_horizon
    assert all(
        action.grad is not None and action.grad.abs().sum() > 0
        for action in sampled_actions
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.actor.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.rssm.parameters()
    )


# ---- update path -----------------------------------------------------------

def random_batch(cfg, batch=3, seq=4):
    obs = torch.rand(batch, seq, 3, 64, 64) - 0.5
    actions = torch.rand(batch, seq, ACTION_DIM) * 2 - 1
    rewards = torch.rand(batch, seq)
    return obs, actions, rewards


def test_model_loss_interface_and_weights():
    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)
    obs, actions, rewards = random_batch(cfg)
    posterior, prior, reward_hat, recon = agent.forward(obs, actions)
    losses = agent._model_loss(obs, rewards, posterior, prior, reward_hat, recon)

    assert set(losses) == {"kl_loss", "recon_loss", "reward_loss", "total"}
    assert all(value.ndim == 0 for value in losses.values())
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(losses[name] >= 0 for name in ("kl_loss", "recon_loss", "reward_loss"))
    expected_total = (
        cfg.kl_loss_scale * losses["kl_loss"]
        + cfg.reconstruction_loss_scale * losses["recon_loss"]
        + cfg.reward_loss_scale * losses["reward_loss"]
    )
    assert torch.allclose(losses["total"], expected_total)


def test_model_loss_known_values():
    cfg = DreamerDebugConfig(
        free_nats=1.5,
        kl_loss_scale=0.5,
        reconstruction_loss_scale=2.0,
        reward_loss_scale=3.0,
    )
    agent = make_agent(cfg)
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

    losses = agent._model_loss(obs, rewards, posterior, prior, reward_hat, recon)
    expected = {
        "kl_loss": 0.6534264,
        "recon_loss": 5.0,
        "reward_loss": 3.25,
        "total": 20.0767136,
    }
    for name, value in expected.items():
        assert torch.allclose(losses[name], torch.tensor(value), atol=1e-5), name


def test_actor_critic_objective_known_values(monkeypatch):
    cfg = DreamerDebugConfig(imagination_horizon=4, discount_gamma=0.5)
    agent = make_agent(cfg)
    agent.return_lambda = 0.5

    class ActorProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class ValueProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, feat):
            return feat[..., :1] * self.weight

    class RewardProbe(torch.nn.Module):
        def forward(self, feat):
            return feat[..., :1]

    agent.actor = ActorProbe()
    agent.value_model = ValueProbe()
    agent.reward_model = RewardProbe()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.1)
    agent.value_optimizer = torch.optim.SGD(agent.value_model.parameters(), lr=0.1)

    base = torch.zeros(4, 1, agent.feat_dim)
    base[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(
        agent,
        "imagine_rollout",
        lambda _start: base * agent.actor.weight,
    )

    metrics = agent._update_actor_critic(agent.rssm.initialize_RSSM_state(1))
    actual = torch.stack([metrics[name] for name in (
        "actor_loss", "value_loss", "imag_return", "imag_value", "imag_reward",
    )])
    assert torch.allclose(
        actual,
        torch.tensor([-2.5625, 0.458984375, 5.1875, 5.0, 2.5]),
        atol=1e-6,
    )
    assert torch.allclose(agent.actor.weight.detach(), torch.tensor(1.25625), atol=1e-6)
    assert torch.allclose(agent.value_model.weight.detach(), torch.tensor(2.1104167), atol=1e-6)


def test_update_returns_finite_losses():
    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)
    losses = agent.update(*random_batch(cfg))
    for key in ("kl_loss", "recon_loss", "reward_loss", "total",
                "actor_loss", "value_loss", "imag_return"):
        assert key in losses
        assert np.isfinite(losses[key]), f"{key} is not finite"


def test_actor_critic_update_touches_only_its_params():
    '''
    the actor step must move actor params only; the value step value params
    only; world-model params must be untouched by both
    '''
    agent = make_agent()
    model_before = snapshot(agent.model_parameters())
    actor_before = snapshot(agent.actor.parameters())
    value_before = snapshot(agent.value_model.parameters())

    start = agent.rssm.initialize_RSSM_state(8)
    metrics = agent._update_actor_critic(start)

    assert np.isfinite(metrics["actor_loss"].item())
    assert max_param_delta(model_before, agent.model_parameters()) == 0.0
    assert max_param_delta(actor_before, agent.actor.parameters()) > 0.0
    assert max_param_delta(value_before, agent.value_model.parameters()) > 0.0


def test_update_moves_model_params_and_leaves_no_stale_grads():
    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)
    model_before = snapshot(agent.model_parameters())
    agent.update(*random_batch(cfg))
    assert max_param_delta(model_before, agent.model_parameters()) > 0.0
    # the actor backward deposits grads on model params; update must leave
    # them cleared so nothing leaks into the next model step
    for p in agent.model_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0


def test_act_signature_and_modes():
    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)

    class FakeSpec:
        shape = (ACTION_DIM,)
        dtype = np.float32
        minimum = -np.ones(ACTION_DIM, dtype=np.float32)
        maximum = np.ones(ACTION_DIM, dtype=np.float32)

    obs = torch.rand(1, 1, 3, 64, 64) - 0.5
    prev_action = torch.zeros(1, ACTION_DIM)
    prev_state = agent.rssm.initialize_RSSM_state(1)

    agent.train()
    action, state = agent.act(obs, prev_action, prev_state, FakeSpec())
    assert action.shape == (1, ACTION_DIM)
    assert isinstance(state, RSSMState)

    agent.eval()
    action_a, _ = agent.act(obs, prev_action, prev_state, FakeSpec())
    assert action_a.shape == (1, ACTION_DIM)
    assert (action_a.abs() <= 1.0).all()


# ---- checkpointing ---------------------------------------------------------

def test_resume_roundtrip_restores_all_three_optimizers(tmp_path):
    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)

    replay = ReplayBuffer(capacity=cfg.replay_capacity, image_bits=cfg.image_bits)
    T = 10
    replay.add_episode(
        np.random.randint(0, 255, size=(T + 1, 64, 64, 3), dtype=np.uint8),
        np.random.uniform(-1, 1, size=(T, ACTION_DIM)).astype(np.float32),
        np.random.rand(T).astype(np.float32),
    )

    # a few updates so every optimizer holds real state
    for _ in range(2):
        agent.update(*random_batch(cfg))

    path = checkpointing.save_resume_checkpoint(
        tmp_path, cfg, agent, replay, env=None, eval_env=None,
        step=2, episodes_collected=1, transitions_collected=T,
        env_steps_collected=T * cfg.action_repeat,
        best_eval_return=1.0, wandb_run_id=None,
    )

    fresh_agent = make_agent(cfg)
    fresh_replay = ReplayBuffer(capacity=cfg.replay_capacity, image_bits=cfg.image_bits)
    payload = checkpointing.load_resume_checkpoint(
        path, fresh_agent, fresh_replay, env=None, eval_env=None,
    )

    assert payload["agent_type"] == "dreamer"
    assert payload["step"] == 2
    for a, b in zip(agent.state_dict().values(), fresh_agent.state_dict().values()):
        assert torch.equal(a, b)
    for name, opt in fresh_agent.named_optimizers().items():
        original = agent.named_optimizers()[name].state_dict()
        restored = opt.state_dict()
        assert len(restored["state"]) == len(original["state"]), name
        # Verify that Adam step counts were restored.
        for key, entry in restored["state"].items():
            assert entry["step"] == original["state"][key]["step"]


def test_eval_rebuilds_dreamer_from_checkpoint():
    from worldmodels.eval import config_from_payload, build_agent

    cfg = DreamerDebugConfig()
    agent = make_agent(cfg)
    payload = {
        "config": asdict(cfg),
        "agent_type": "dreamer",
        "model": agent.state_dict(),
        "step": 1,
        "env_steps": 0,
    }
    rebuilt_cfg, has_config = config_from_payload(payload)
    assert has_config
    assert type(rebuilt_cfg).__name__ == "DreamerDebugConfig" or rebuilt_cfg.s_dim == cfg.s_dim

    rebuilt = build_agent(rebuilt_cfg, ACTION_DIM, torch.device("cpu"), "dreamer")
    rebuilt.load_state_dict(payload["model"])
    for a, b in zip(agent.state_dict().values(), rebuilt.state_dict().values()):
        assert torch.equal(a, b)


def test_checkpoint_without_agent_type_uses_planet_config():
    from worldmodels.eval import config_from_payload
    cfg, has_config = config_from_payload({"model": {}})
    assert type(cfg).__name__ == "PlaNetTrainConfig"
    assert not has_config
