import math
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as f


class TanhGaussianActor(nn.Module):
    '''
    Dreamer V1 action model: a tanh-transformed Gaussian over actions,
    conditioned on the latent feature (s_t, h_t).

    Follows danijar/dreamer's ActionDecoder (tanh_normal): the mean is
    soft-clamped through mean_scale * tanh(x / mean_scale), the std gets a
    softplus with a large positive init so early policies stay near-uniform
    across the tanh, and min_std keeps the base Gaussian from collapsing.
    '''
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        init_std: float = 5.0,
        mean_scale: float = 5.0,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.mean_scale = mean_scale
        self.min_std = min_std
        # inverse-softplus so std starts at init_std
        self.raw_init_std = math.log(math.exp(init_std) - 1.0)

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "elu":
            activation_fn = nn.ELU()
        else:
            raise ValueError("Unknown activation function")

        layers = [nn.Linear(in_dim, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, feat: torch.Tensor) -> td.Distribution:
        '''
        feat (..., feat_dim) -> Independent tanh-Normal over (..., action_dim)

        rsample() is reparameterized end to end (Normal rsample + tanh), which
        is what lets the actor loss backprop through imagined dynamics.
        '''
        x = self.backbone(feat)
        mean = self.mean_scale * torch.tanh(self.mean_head(x) / self.mean_scale)
        std = f.softplus(self.std_head(x) + self.raw_init_std) + self.min_std

        dist = td.Normal(mean, std)
        # cache_size=1 keeps atanh off our own samples in log_prob (tanh can
        # saturate to exactly +/-1 in float32, where atanh is inf)
        dist = td.TransformedDistribution(dist, td.TanhTransform(cache_size=1))
        dist = td.Independent(dist, 1)
        return dist


@torch.no_grad()
def tanh_dist_mode(dist: td.Distribution, num_samples: int = 100) -> torch.Tensor:
    '''
    approximate mode of a tanh-Normal by sampling (danijar's SampleDist):
    draw num_samples candidates, keep the one with the highest log_prob.

    Input dist has batch shape (B,) and event shape (action_dim,);
    returns (B, action_dim).
    '''
    samples = dist.sample((num_samples,))          # (N, B, A)
    logp = dist.log_prob(samples)                  # (N, B)
    idx = logp.argmax(dim=0)                       # (B,)
    batch = torch.arange(samples.shape[1], device=samples.device)
    return samples[idx, batch]
