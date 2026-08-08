import os
import sys

import torch
import torch.distributions as td

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from worldmodels.models.rssm import RSSM

BATCH_SIZE = 3
SEQUENCE_LENGTH = 5
IMAGINE_LENGTH = 4
S_DIM = 30
H_DIM = 200
ACTION_DIM = 6
ENCODED_DIM = 1024


def make_rssm():
    return RSSM(
        s_dim=S_DIM,
        h_dim=H_DIM,
        mlp_hidden_dim=200,
        mlp_layers=2,
        mlp_activation="relu",
        action_dim=ACTION_DIM,
        encoded_dim=ENCODED_DIM,
        min_std=0.1,
    )


def assert_state_shapes(state, batch_size, sequence_length, s_dim, h_dim):
    assert state.s_t.shape == (batch_size, sequence_length, s_dim)
    assert state.h_t.shape == (batch_size, sequence_length, h_dim)
    assert state.latent_mean.shape == (batch_size, sequence_length, s_dim)
    assert state.latent_std.shape == (batch_size, sequence_length, s_dim)


def test_rssm_observe_shapes():
    torch.manual_seed(0)
    rssm = make_rssm()
    embeds = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, ENCODED_DIM)
    actions = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, ACTION_DIM)

    posterior, prior = rssm.observe(embeds, actions)

    assert_state_shapes(posterior, BATCH_SIZE, SEQUENCE_LENGTH, S_DIM, H_DIM)
    assert_state_shapes(prior, BATCH_SIZE, SEQUENCE_LENGTH, S_DIM, H_DIM)

    feat = rssm.get_feat(posterior)
    assert feat.shape == (BATCH_SIZE, SEQUENCE_LENGTH, S_DIM + H_DIM)

    posterior_dist = rssm.get_distribution(posterior)
    prior_dist = rssm.get_distribution(prior)
    assert posterior_dist.batch_shape == torch.Size([BATCH_SIZE, SEQUENCE_LENGTH])
    assert posterior_dist.event_shape == torch.Size([S_DIM])
    assert prior_dist.batch_shape == torch.Size([BATCH_SIZE, SEQUENCE_LENGTH])
    assert prior_dist.event_shape == torch.Size([S_DIM])

    kl = td.kl_divergence(posterior_dist, prior_dist)
    assert kl.shape == (BATCH_SIZE, SEQUENCE_LENGTH)
    assert torch.isfinite(kl).all()


def test_observe_step_uses_embedding_only_for_posterior_and_is_reparameterized():
    rssm = make_rssm()
    prev = rssm.initialize_RSSM_state(BATCH_SIZE)
    actions = torch.zeros(BATCH_SIZE, ACTION_DIM)
    embeds_a = torch.zeros(BATCH_SIZE, ENCODED_DIM)
    embeds_b = torch.full((BATCH_SIZE, ENCODED_DIM), 5.0)

    torch.manual_seed(7)
    posterior_a, prior_a = rssm.observe_step(embeds_a, actions, prev)
    torch.manual_seed(7)
    posterior_b, prior_b = rssm.observe_step(embeds_b, actions, prev)

    assert torch.equal(posterior_a.h_t, prior_a.h_t)
    assert torch.allclose(prior_a.latent_mean, prior_b.latent_mean)
    assert not torch.allclose(posterior_a.latent_mean, posterior_b.latent_mean)

    rssm.zero_grad()
    torch.manual_seed(11)
    posterior, _ = rssm.observe_step(embeds_a, actions, prev)
    posterior.s_t.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in rssm.posterior.parameters()
    )


def test_rssm_imagine_shapes():
    torch.manual_seed(0)
    rssm = make_rssm()
    start_state = rssm.initialize_RSSM_state(BATCH_SIZE)
    imagined_actions = torch.randn(BATCH_SIZE, IMAGINE_LENGTH, ACTION_DIM)
    imagined = rssm.imagine(start_state, imagined_actions)

    assert_state_shapes(imagined, BATCH_SIZE, IMAGINE_LENGTH, S_DIM, H_DIM)


def test_imagine_step_uses_the_action():
    rssm = make_rssm()
    prev = rssm.initialize_RSSM_state(BATCH_SIZE)
    actions_a = torch.zeros(BATCH_SIZE, ACTION_DIM)
    actions_b = torch.ones(BATCH_SIZE, ACTION_DIM)

    torch.manual_seed(13)
    state_a = rssm.imagine_step(actions_a, prev)
    torch.manual_seed(13)
    state_b = rssm.imagine_step(actions_b, prev)

    assert not torch.allclose(state_a.h_t, state_b.h_t)
    assert not torch.allclose(state_a.latent_mean, state_b.latent_mean)


def test_imagine_step_prior_sample_is_reparameterized():
    rssm = make_rssm()
    prev = rssm.initialize_RSSM_state(BATCH_SIZE)

    rssm.zero_grad(set_to_none=True)
    torch.manual_seed(23)
    state = rssm.imagine_step(torch.zeros(BATCH_SIZE, ACTION_DIM), prev)
    assert state.s_t.requires_grad
    state.s_t.sum().backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in rssm.prior.parameters()
    )


if __name__ == "__main__":
    test_rssm_observe_shapes()
    test_observe_step_uses_embedding_only_for_posterior_and_is_reparameterized()
    test_rssm_imagine_shapes()
    test_imagine_step_uses_the_action()
    test_imagine_step_prior_sample_is_reparameterized()
