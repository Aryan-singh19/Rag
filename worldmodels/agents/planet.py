import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from dm_env import specs

from worldmodels.networks import ScalarMLP, ConvDecoder, ConvEncoder
from worldmodels.models import RSSMState, RSSM
from typing import Tuple

class PlaNet(nn.Module):
    '''
    Learning Latent Dynamics for Planning from Pixels 
    https://arxiv.org/abs/1811.04551
    '''
    def __init__(
        self,
        s_dim: int,
        h_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        mlp_activation: str,
        action_dim: int,
        device,
        planning_horizon: int,
        num_candidates: int,
        num_elites: int,
        refine_iterations: int,
        cem_min_std: float,
        encoded_dim: int,
        min_std: float,
        learning_rate: float,
        adam_epsilon: float,
        gradient_clip_norm: float,
        free_nats: float,
        kl_loss_scale: float,
        reconstruction_loss_scale: float,
        reward_loss_scale: float,
    ) -> None:
        super().__init__()
        self.feat_dim = s_dim + h_dim
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.refine_iterations = refine_iterations
        self.cem_min_std = cem_min_std
        self.gradient_clip_norm = gradient_clip_norm
        self.free_nats = free_nats
        self.kl_loss_scale = kl_loss_scale
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self.reward_loss_scale = reward_loss_scale

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
        
        self.reward_model = ScalarMLP(self.feat_dim, mlp_hidden_dim, mlp_layers, mlp_activation)

        self.to(device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=adam_epsilon,
        )
        
    @property
    def device(self) -> torch.device:
        '''
        Infer the device from the model parameters.
        '''
        return next(self.parameters()).device

    def forward(self, obs, actions) -> Tuple[RSSMState, RSSMState, torch.Tensor, torch.Tensor]:
        '''
        Takes a sequence of observations and actions and returns posterior and
        prior states, predicted rewards, and reconstructed observations.

        Inputs:
        obs (B, T, C, H, W): 64*64*3 images (o_t)
        actions (B, T, action_dim): shifted actions into the obs (a_{t-1})
        '''
        embeds = self.encoder(obs)
        posterior, prior = self.rssm.observe(embeds, actions)
        feats = self.rssm.get_feat(posterior)
        recon = self.decoder(feats)
        reward = self.reward_model(feats)
        return posterior, prior, reward, recon
    
    def loss(self, obs, actions, rewards) -> dict:
        '''
        Compute the model loss corresponding to Equation 3 of the paper.
        
        Inputs:
        obs (B, T, C, H, W): 64*64*3 images (o_t)
        actions (B, T, action_dim): shifted actions into the obs (a_{t-1})
        rewards: (B, T): rewards

        Returns a dict with scalar tensors under the exact keys
        ``kl_loss``, ``recon_loss``, ``reward_loss``, and ``total``.
        '''

        posterior, prior, reward_hat, recon = self.forward(obs, actions)

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
    
    def update(self, obs, actions, rewards) -> dict:
        '''
        perform one gradient update step to the model
        '''
        self.optimizer.zero_grad()
        
        losses = self.loss(obs, actions, rewards)
        loss = losses["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.gradient_clip_norm,
        )
        
        self.optimizer.step()

        losses = {
            key: val.item() for key, val in losses.items()
        }
        return losses
    
    @torch.no_grad()
    def plan_cem(self, state: RSSMState, action_spec: specs.BoundedArray) -> torch.Tensor:
        '''
        Plan using Cross Entropy Method

        Inputs:
        state: current RSSM latent state
        action_spec: result of env.action_spec(), with shape, dtype, minimum,
        and maximum fields
        planning_horizon (H): number of future timesteps looking ahead
        num_candidates (J): number of candidate action sequences sampled
        num_elites (K): number of best/elite candidates action sequences kept
        refine_iter (I): number of optimization iterations

        Returns: 
        a point estimate for next step actions selected by CEM
        '''
        device = self.device
        action_dim = action_spec.shape[0]
        H = self.planning_horizon
        N = self.num_candidates
        K = self.num_elites

        low = torch.as_tensor(
            np.broadcast_to(action_spec.minimum, (action_dim,)).astype(np.float32).copy(),
            device=device,
        )
        high = torch.as_tensor(
            np.broadcast_to(action_spec.maximum, (action_dim,)).astype(np.float32).copy(),
            device=device,
        )

        mu = torch.zeros(H, action_dim, device=device)
        sigma = torch.ones(H, action_dim, device=device)

        batch_size = state.s_t.shape[0]
        repeat_factor = N // batch_size
        start_state = RSSMState(
            s_t=state.s_t.repeat(repeat_factor, 1),
            h_t=state.h_t.repeat(repeat_factor, 1),
            latent_mean=state.latent_mean.repeat(repeat_factor, 1),
            latent_std=state.latent_std.repeat(repeat_factor, 1),
        )

        for _ in range(self.refine_iterations):
            noise = torch.randn(N, H, action_dim, device=device)
            candidates = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            candidates = torch.clamp(candidates, min=low, max=high)

            imagined = self.rssm.imagine(start_state, candidates)
            feats = self.rssm.get_feat(imagined)
            reward_hat = self.reward_model(feats)  # (N, H, 1)
            scores = reward_hat.squeeze(-1).sum(dim=1)  # (N,)

            elite_idx = torch.topk(scores, K, dim=0).indices
            elites = candidates[elite_idx]  # (K, H, action_dim)

            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0, unbiased=False).clamp_min(self.cem_min_std)

        action = torch.clamp(mu[0], min=low, max=high)
        return action
    
    @torch.no_grad()
    def act(self, obs, prev_action, prev_state:RSSMState, action_spec) -> Tuple[torch.Tensor, RSSMState]:
        '''
        provide next action and the updated RSSMState

        Inputs:
        obs (1, 1, 3, 64, 64): the preprocessed observation
        prev_action (1, action_dim): action from the previous time step
        prev_state: RSSM state from previous time step
        action_spec: the action spec from env.action_spec().

        Returns:
        action: the proposed action for this time step
        state: the updated RSSM state for this time step
        '''
        e_t = self.encoder(obs)[:, 0, :]

        state, _ = self.rssm.observe_step(e_t, prev_action, prev_state)
        action = self.plan_cem(state, action_spec)
        action = action.unsqueeze(0)

        return action, state
