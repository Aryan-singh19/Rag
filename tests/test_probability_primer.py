'''A tiny executable example of the PyTorch probability API used by the RSSM.'''
import torch
import torch.distributions as td
import torch.nn.functional as F


def test_diagonal_gaussian_shapes_and_gradient():
    batch_size, latent_dim = 3, 5
    mean = torch.zeros(batch_size, latent_dim, requires_grad=True)
    raw_std = torch.zeros(batch_size, latent_dim, requires_grad=True)
    std = F.softplus(raw_std) + 0.1

    scalar_normals = td.Normal(mean, std)
    latent_gaussian = td.Independent(scalar_normals, 1)
    sample = latent_gaussian.rsample()

    assert latent_gaussian.batch_shape == torch.Size([batch_size])
    assert latent_gaussian.event_shape == torch.Size([latent_dim])
    assert sample.shape == (batch_size, latent_dim)

    sample.sum().backward()
    assert mean.grad is not None
    assert raw_std.grad is not None
