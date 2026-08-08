import torch
import torch.nn as nn
import einops as op
import torch.distributions as td
import torch.nn.functional as f

class ScalarMLP(nn.Module):
    '''
    outputs a scalar value
    '''
    def __init__(self, in_dim:int, hidden_dim:int, num_layers:int, activation:str) -> None:
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "elu":
            activation_fn = nn.ELU()
        else:
            raise ValueError("Unknown activation function")

        for layer in range(num_layers - 1):
            layers.append(activation_fn)
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
class GaussianMLP(nn.Module):
    '''
    output a Gaussian distribution object.
    '''
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int, 
                 num_layers:int, activation:str, min_std:float) -> None:
        super().__init__()
        self.min_std = min_std
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "elu":
            activation_fn = nn.ELU()
        else:
            raise ValueError("Unknown activation function")

        for layer in range(num_layers - 1):
            layers.append(activation_fn)
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(activation_fn)

        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.std_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        mean = self.mean_head(x)
        std = f.softplus(self.std_head(x)) + self.min_std
        dist = td.Normal(mean, std)
        
        # so that it works like B independent D-dim Gaussians instead of B*D 1d Gaussians
        dist = td.Independent(dist, 1)
        return dist
