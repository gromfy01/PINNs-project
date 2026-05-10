import torch
import torch.nn as nn

from .components import build_mlp, FourierEmbedding, OutputScaler, ACTIVATIONS


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, activation):
        super().__init__()
        self.net = build_mlp(in_dim, hidden, out_dim, activation)

    def forward(self, x):
        return self.net(x)


class DDENet(nn.Module):
    def __init__(self, layer_sizes, activation):
        super().__init__()
        Act = ACTIVATIONS[activation.lower()]
        self.linears = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1])
             for i in range(len(layer_sizes) - 1)]
        )
        self.act = Act()

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = self.act(x)
        return x


class VPINN(nn.Module):
    def __init__(self, hidden, out_dim, activation, n_freq, n_params=5):
        super().__init__()
        self.r_embed = FourierEmbedding(n_freq=n_freq, fixed=True)
        in_dim = 2 * n_freq + n_params
        self.net = build_mlp(in_dim, hidden, out_dim, activation)
        self.scaler_out = OutputScaler(out_dim)
        self.n_params = n_params

    def forward(self, x):
        params = x[:, :self.n_params]
        r = x[:, self.n_params:self.n_params + 1]
        r_feat = self.r_embed(r)
        inp = torch.cat([params, r_feat], dim=-1)
        return self.scaler_out(self.net(inp))
