import torch
import torch.nn as nn
import torch.distributions as td

from dm_env import specs
from itertools import chain
from typing import Tuple

from worldmodels.networks import ScalarMLP, ConvDecoder, ConvEncoder, TanhGaussianActor
from worldmodels.networks.actor import tanh_dist_mode
from worldmodels.models import RSSMState, RSSM


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    '''
    TD(lambda) returns over an imagined trajectory (eq. 6 in Dreamer V1).

    Inputs:
    rewards (H, N): predicted rewards, time-major
    values (H, N): predicted values at the same steps
    bootstrap (N,): value estimate one step past the last reward
    gamma: discount factor
    lam: lambda for the exponentially-weighted mix of n-step returns

    Returns (H, N): V_lambda at every step, computed by the standard
    backward recursion V_t = r_t + gamma * ((1-lam) * v_{t+1} + lam * V_{t+1})
    '''
    H = rewards.shape[0]
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)  # v_{t+1}, (H, N)

    returns = torch.empty_like(rewards)
    acc = bootstrap
    for t in reversed(range(H)):
        acc = rewards[t] + gamma * ((1.0 - lam) * next_values[t] + lam * acc)
        returns[t] = acc

    return returns


class Dreamer(nn.Module):
    '''
    Dream to Control: Learning Behaviors by Latent Imagination
    https://arxiv.org/abs/1912.01603

    Dreamer replaces PlaNet's CEM planner with an actor and value model trained
    on imagined latent trajectories. The model, actor, and value parameters
    use separate optimizers.
    '''
    agent_type = "dreamer"

    def __init__(
        self,
        s_dim: int,
        h_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        mlp_activation: str,
        action_dim: int,
        device,
        encoded_dim: int,
        min_std: float,
        # world model loss
        free_nats: float,
        kl_loss_scale: float,
        reconstruction_loss_scale: float,
        reward_loss_scale: float,
        # heads
        reward_hidden_dim: int,
        reward_layers: int,
        value_hidden_dim: int,
        value_layers: int,
        actor_hidden_dim: int,
        actor_layers: int,
        actor_init_std: float,
        actor_mean_scale: float,
        actor_min_std: float,
        # imagination and returns
        imagination_horizon: int,
        discount_gamma: float,
        return_lambda: float,
        # optimization
        model_lr: float,
        actor_lr: float,
        value_lr: float,
        adam_epsilon: float,
        gradient_clip_norm: float,
    ) -> None:
        super().__init__()
        self.feat_dim = s_dim + h_dim
        self.gradient_clip_norm = gradient_clip_norm
        self.free_nats = free_nats
        self.kl_loss_scale = kl_loss_scale
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self.reward_loss_scale = reward_loss_scale
        self.imagination_horizon = imagination_horizon
        self.discount_gamma = discount_gamma
        self.return_lambda = return_lambda

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder(self.feat_dim)
        self.rssm = RSSM(
            s_dim,
            h_dim,
            mlp_hidden_dim,
            mlp_layers,
            mlp_activation,
            action_dim,
            encoded_dim,
            min_std,
        )
        self.reward_model = ScalarMLP(self.feat_dim, reward_hidden_dim, reward_layers, mlp_activation)

        self.value_model = ScalarMLP(self.feat_dim, value_hidden_dim, value_layers, mlp_activation)
        self.actor = TanhGaussianActor(
            self.feat_dim,
            action_dim,
            actor_hidden_dim,
            actor_layers,
            mlp_activation,
            init_std=actor_init_std,
            mean_scale=actor_mean_scale,
            min_std=actor_min_std,
        )

        self.to(device)
        self.model_optimizer = torch.optim.Adam(
            self.model_parameters(), lr=model_lr, eps=adam_epsilon,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=adam_epsilon,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=value_lr, eps=adam_epsilon,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def model_parameters(self):
        '''
        Parameters updated by model_optimizer. The actor and value model use
        separate optimizers.
        '''
        return chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.rssm.parameters(),
            self.reward_model.parameters(),
        )

    def named_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        '''
        Return the optimizers saved and restored in resume checkpoints.
        '''
        return {
            "model": self.model_optimizer,
            "actor": self.actor_optimizer,
            "value": self.value_optimizer,
        }

    def forward(self, obs, actions) -> Tuple[RSSMState, RSSMState, torch.Tensor, torch.Tensor]:
        '''
        Encode a replay sequence and return posterior and prior states,
        predicted rewards, and reconstructed observations.
        '''
        embeds = self.encoder(obs)
        posterior, prior = self.rssm.observe(embeds, actions)
        feats = self.rssm.get_feat(posterior)
        recon = self.decoder(feats)
        reward = self.reward_model(feats)
        return posterior, prior, reward, recon

    def _model_loss(self, obs, rewards, posterior, prior, reward_hat, recon) -> dict:
        '''
        Compute KL with free nats plus Gaussian NLL (MSE) for reconstruction
        and reward. Return ``kl_loss``, ``recon_loss``, ``reward_loss``, and
        ``total``.
        '''
        posterior_dist = self.rssm.get_distribution(posterior)
        prior_dist = self.rssm.get_distribution(prior)
        kl = td.kl_divergence(posterior_dist, prior_dist)  # (B, T)
        kl_loss = torch.clamp(kl - self.free_nats, min=0.0).mean()

        recon_sq_err = 0.5 * (recon - obs).pow(2).sum(dim=[-3, -2, -1])  # (B, T)
        recon_loss = recon_sq_err.mean()

        reward_hat = reward_hat.squeeze(-1)  # (B, T)
        reward_sq_err = 0.5 * (reward_hat - rewards).pow(2)
        reward_loss = reward_sq_err.mean()

        total = (
            self.kl_loss_scale * kl_loss
            + self.reconstruction_loss_scale * recon_loss
            + self.reward_loss_scale * reward_loss
        )

        return {
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "total": total,
        }

    def loss(self, obs, actions, rewards) -> dict:
        '''
        Compute the same world-model loss as PlaNet.loss.
        '''
        posterior, prior, reward_hat, recon = self.forward(obs, actions)
        return self._model_loss(obs, rewards, posterior, prior, reward_hat, recon)

    def imagine_rollout(self, start: RSSMState) -> torch.Tensor:
        '''
        roll the actor through the learned dynamics for imagination_horizon
        steps from a batch of (detached) posterior states

        The actor sees a stop-gradient of its input features: each action's
        gradient reaches the actor
        only through the future dynamics and rewards it causes, not through
        the state that produced it.

        start: RSSMState with (N, dim) tensors
        Returns imag_feats (H, N, feat_dim); the start state is not included
        '''
        prev_state = start
        feats = []
        for _ in range(self.imagination_horizon):
            # Stop-gradient boundary: the actor must not backprop into the
            # state that produced its action, only forward through its own
            # parameters and the resulting future dynamics/rewards.
            feat = self.rssm.get_feat(prev_state).detach()
            dist = self.actor(feat)
            action = dist.rsample()

            prev_state = self.rssm.imagine_step(action, prev_state)
            feats.append(self.rssm.get_feat(prev_state))

        return torch.stack(feats, dim=0)

    def _update_actor_critic(self, start: RSSMState) -> dict:
        '''
        one actor update and one value update on imagined rollouts

        Uses lambda-returns over H-1 imagined steps with
        the H-th value as bootstrap; the actor maximizes discount-weighted
        returns by backprop through dynamics (only actor params step); the
        value regresses onto stop-gradient returns from detached features.
        Returns detached ``actor_loss``, ``value_loss``, ``imag_return``,
        ``imag_value``, and ``imag_reward`` metrics.
        '''
        imag_feats = self.imagine_rollout(start)  # (H, N, feat_dim)
        imag_reward = self.reward_model(imag_feats).squeeze(-1)  # (H, N)
        imag_value = self.value_model(imag_feats).squeeze(-1)  # (H, N)

        H = imag_feats.shape[0]
        gamma = self.discount_gamma
        lam = self.return_lambda

        # Targets for t = 1..H-1; the final feature only supplies the
        # bootstrap V_H = vD(f_H).
        returns = lambda_return(
            imag_reward[:-1], imag_value[:-1], imag_value[-1],
            gamma=gamma, lam=lam,
        )  # (H-1, N)

        discount = gamma ** torch.arange(H - 1, device=imag_feats.device, dtype=imag_feats.dtype)
        discount = discount.unsqueeze(-1)  # (H-1, 1)

        # ---- actor: maximize discount-weighted lambda returns ----
        # This graph is rooted at imag_feats (attached to actor + dynamics
        # + value_model), so the actor's gradient signal legitimately flows
        # through the critic's predicted future values too.
        actor_loss = -(discount * returns).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_clip_norm)
        self.actor_optimizer.step()

        # ---- critic: regress onto stop-gradient lambda returns ----
        # Re-run the value head on a DETACHED copy of imag_feats. Numerically
        # identical to imag_value, but this graph has no path back into the
        # actor or dynamics params, so (a) the critic's backward can't leak
        # gradient onto the actor's parameters, and (b) it shares no nodes
        # with actor_loss's (already-freed) graph, so stepping the actor
        # first is safe.
        critic_targets = returns.detach()
        critic_value_pred = self.value_model(imag_feats.detach()).squeeze(-1)[:-1]
        value_loss = (0.5 * discount * (critic_value_pred - critic_targets).pow(2)).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=self.gradient_clip_norm)
        self.value_optimizer.step()

        # World-model parameters must remain unchanged and retain no
        # residual gradient from the actor/critic backward passes.
        self.model_optimizer.zero_grad(set_to_none=True)

        return {
            "actor_loss": actor_loss.detach(),
            "value_loss": value_loss.detach(),
            "imag_return": returns.mean().detach(),
            "imag_value": imag_value.mean().detach(),
            "imag_reward": imag_reward.mean().detach(),
        }

    def update(self, obs, actions, rewards) -> dict:
        '''
        one full Dreamer training step on a replay batch:
        world model first, then actor and value on imagination from the
        batch's posterior states
        '''
        # ---- world model ----
        self.model_optimizer.zero_grad()
        posterior, prior, reward_hat, recon = self.forward(obs, actions)
        model_losses = self._model_loss(obs, rewards, posterior, prior, reward_hat, recon)
        model_losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model_parameters(), max_norm=self.gradient_clip_norm)
        self.model_optimizer.step()

        # ---- actor and value on imagined rollouts ----
        # flatten (B, T, dim) posterior states into N = B*T starts, detached:
        # the model graph is already freed and imagination must not backprop
        # into the observation encoding
        start = RSSMState(
            posterior.s_t.detach().reshape(-1, posterior.s_t.shape[-1]),
            posterior.h_t.detach().reshape(-1, posterior.h_t.shape[-1]),
            posterior.latent_mean.detach().reshape(-1, posterior.latent_mean.shape[-1]),
            posterior.latent_std.detach().reshape(-1, posterior.latent_std.shape[-1]),
        )
        ac_metrics = self._update_actor_critic(start)

        losses = {**model_losses, **ac_metrics}
        return {key: val.item() for key, val in losses.items()}

    @torch.no_grad()
    def act(self, obs, prev_action, prev_state: RSSMState, action_spec) -> Tuple[torch.Tensor, RSSMState]:
        '''
        Update the posterior state from the observation, then select an action.
        Training samples from the actor; evaluation uses its approximate mode.
        Return the action and updated posterior state.
        '''
        e_t = self.encoder(obs)[:, 0, :]
        state, _ = self.rssm.observe_step(e_t, prev_action, prev_state)

        feat = self.rssm.get_feat(state)
        dist = self.actor(feat)
        if self.training:
            action = dist.sample()
        else:
            action = tanh_dist_mode(dist)

        return action, state
