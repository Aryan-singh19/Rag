import torch
import torch.nn as nn
import torch.distributions as td

from worldmodels.networks import GaussianMLP, GRUTransition
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RSSMState:
    '''
    Tensors that make up one RSSM state (or a stacked sequence of states).
    '''
    # Stochastic state sampled from the latent Gaussian.
    s_t:torch.Tensor
    
    # Deterministic recurrent state.
    h_t:torch.Tensor 

    # These two tensors parameterize the latent Gaussian used for the KL.
    latent_mean:torch.Tensor
    latent_std:torch.Tensor


class RSSM(nn.Module):
    '''
    The Recurrent State Space Model
    '''
    def __init__(
        self,
        s_dim: int,
        h_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        mlp_activation: str,
        action_dim: int,
        encoded_dim: int,
        min_std: float,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.encoded_dim = encoded_dim
        self.action_dim = action_dim

        self.combined_dim = s_dim + h_dim    

        self.prior = GaussianMLP(
            h_dim,
            mlp_hidden_dim,
            s_dim,
            mlp_layers,
            mlp_activation,
            min_std,
        )
        self.posterior = GaussianMLP(
            h_dim + self.encoded_dim,
            mlp_hidden_dim,
            s_dim,
            mlp_layers,
            mlp_activation,
            min_std,
        )
        self.transition = GRUTransition(s_dim+action_dim, h_dim, mlp_activation)

    def initialize_RSSM_state(self, batch_size) -> RSSMState:
        device = next(self.parameters()).device

        s_t = torch.zeros((batch_size, self.s_dim), device=device)
        h_t = self.transition.initialize_hidden_state(batch_size)
        latent_mean = torch.zeros((batch_size, self.s_dim), device=device)
        latent_std = torch.ones((batch_size, self.s_dim), device=device)
        
        return RSSMState(s_t, h_t, latent_mean, latent_std)
    
    @staticmethod
    def get_feat(state:RSSMState) -> torch.Tensor:
        '''
        Concatenate s_t and h_t into one feature tensor.
        '''
        s_t = state.s_t
        h_t = state.h_t
        feat = torch.cat([s_t, h_t], dim=-1)
        return feat
    
    @staticmethod
    def get_distribution(state:RSSMState) -> td.Distribution:
        '''
        reconstruct the latent distribution from latent mean and latent std
        '''
        latent_mean = state.latent_mean
        latent_std = state.latent_std

        dist = td.Normal(latent_mean, latent_std)
        dist = td.Independent(dist, 1)
        
        return dist
    
    @staticmethod
    def stack_states(states:list[RSSMState]) -> RSSMState:
        '''
        Stacks a list of RSSMState into a single RSSMState with a time dimension. 
        Concretely, [(B, event_dim), ..., (B, event_dim)] to (B, T, event_dim)
        Where event_dim can be s_dim or h_dim
        '''
        s_all_t = []
        h_all_t = []
        latent_means = []
        latent_stds = []

        for state in states:
            s_all_t.append(state.s_t)
            h_all_t.append(state.h_t)
            latent_means.append(state.latent_mean)
            latent_stds.append(state.latent_std)

        s_all_t = torch.stack(s_all_t, 1)
        h_all_t = torch.stack(h_all_t, 1)
        latent_means = torch.stack(latent_means, 1)
        latent_stds = torch.stack(latent_stds, 1)
        stacked_state = RSSMState(s_all_t, h_all_t, latent_means, latent_stds)
        return stacked_state

    def observe_step(self, e_t, a_last_t, prev_state:RSSMState) -> Tuple[RSSMState, RSSMState]:
        '''
        perform one step of RSSM transition

        input:
        e_t (B, encoding_dim): encoding from encoder
        a_last_t (B, action_dim): action that results in e_t
        prev_state: previous RSSM state

        output:
        (posterior, prior): the posterior and prior states at time t
        '''
        h_t = self.transition(prev_state.s_t, a_last_t, prev_state.h_t)

        prior_dist = self.prior(h_t)
        prior_s_t = prior_dist.rsample()
        prior_state = RSSMState(prior_s_t, h_t, prior_dist.mean, prior_dist.stddev)

        posterior_input = torch.cat([h_t, e_t], dim=-1)
        posterior_dist = self.posterior(posterior_input)
        posterior_s_t = posterior_dist.rsample()
        posterior_state = RSSMState(posterior_s_t, h_t, posterior_dist.mean, posterior_dist.stddev)

        return posterior_state, prior_state

    def observe(self, embeds, actions) -> Tuple[RSSMState, RSSMState]:
        '''
        observe a chunk of embedding and actions:

        input:
        embeds (B, T, encoding_dim): embedding chunk from encoder
        actions (B, T, action_dim): actions are shifted so actions[:, t] is the action into embeds[:, t]

        output:
        (posterior, prior): the posterior and prior states, stacked along the time axis
        '''
        B, T = embeds.shape[:2]
        prev_state = self.initialize_RSSM_state(B)

        posterior_states = []
        prior_states = []

        for t in range(T):
            e_t = embeds[:, t, :]
            a_last_t = actions[:, t, :]
            
            posterior_state, prior_state = self.observe_step(e_t, a_last_t, prev_state)
            
            posterior_states.append(posterior_state)
            prior_states.append(prior_state)

            prev_state = posterior_state

        posterior_states = self.stack_states(posterior_states)
        prior_states = self.stack_states(prior_states)

        return posterior_states, prior_states
    
    def imagine_step(self, a_last_t, prev_state:RSSMState) -> RSSMState:
        '''
        Latent imagination: predict next state given prev state and action without observation

        a_last_t (B, action_dim): action leading into time step t
        prev_state: previous RSSM state
        '''
        h_t = self.transition(prev_state.s_t, a_last_t, prev_state.h_t)

        prior_dist = self.prior(h_t)
        s_t = prior_dist.rsample()

        return RSSMState(s_t, h_t, prior_dist.mean, prior_dist.stddev)

    def imagine(self, start_state:RSSMState, actions) -> RSSMState:
        '''
        Imagine across a trajectory

        input:
        start_state: initial state
        actions: (B, T, action_dim)
        
        output:
        prior states stacked along the time axis
        '''
        T = actions.shape[1]
        prev_state = start_state

        prior_states = []

        for t in range(T):
            a_last_t = actions[:, t, :]
            prior_state = self.imagine_step(a_last_t, prev_state)
            prior_states.append(prior_state)
            prev_state = prior_state

        prior_states = self.stack_states(prior_states)
        return prior_states
    
