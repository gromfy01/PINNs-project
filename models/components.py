import numpy as np
import torch
import torch.nn as nn


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "softplus": nn.Softplus,
}


class FourierEmbedding(nn.Module):
    def __init__(self, n_freq=8, sigma=1.0, fixed=True):
        super().__init__()
        B = torch.randn(n_freq) * sigma
        if fixed:
            self.register_buffer("B", B)
        else:
            self.B = nn.Parameter(B)
        self.n_freq = n_freq

    def forward(self, r):
        proj = r * self.B.unsqueeze(0) * 2 * np.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class OutputScaler(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))
        self.shift = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x * self.scale + self.shift


def build_mlp(in_dim, hidden, out_dim, activation):
    Act = ACTIVATIONS[activation.lower()]
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), Act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
