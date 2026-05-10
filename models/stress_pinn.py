import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch
import torch.nn as nn

from .components import build_mlp, FourierEmbedding, OutputScaler


class StressPINN(nn.Module):
    def __init__(self, hidden, n_freq=16, n_process_params=5):
        super().__init__()
        self.r_embed = FourierEmbedding(n_freq=n_freq, fixed=True)
        in_dim = 2 * n_freq + 1 + n_process_params
        self.backbone = build_mlp(in_dim, hidden, hidden[-1], "tanh")
        self.heads = nn.ModuleList([nn.Linear(hidden[-1], 1) for _ in range(4)])
        self.scaler = OutputScaler(4)
        self.n_process_params = n_process_params

    def forward(self, x):
        r = x[:, 0:1]
        z = x[:, 1:2]
        params = x[:, 2:2 + self.n_process_params]
        r_feat = self.r_embed(r)
        h = torch.cat([r_feat, z, params], dim=-1)
        h = self.backbone(h)
        out = torch.cat([head(h) for head in self.heads], dim=-1)
        return self.scaler(out)


def equilibrium_residuals_torch(model, x):
    x = x.requires_grad_(True)
    sigma = model(x)
    srr, stt, szz, trz = (sigma[:, i:i + 1] for i in range(4))
    r = x[:, 0:1]

    def grad(y, x_in):
        return torch.autograd.grad(
            y, x_in, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True,
        )[0]

    dsrr_dr = grad(srr, x)[:, 0:1]
    dtrz_dz = grad(trz, x)[:, 1:2]
    dtrz_dr = grad(trz, x)[:, 0:1]
    dszz_dz = grad(szz, x)[:, 1:2]

    eps = 1e-6
    res_r = dsrr_dr + dtrz_dz + (srr - stt) / (r + eps)
    res_z = dtrz_dr + dszz_dz + trz / (r + eps)
    return res_r, res_z


def traction_free_loss_torch(model, x_surface):
    sigma = model(x_surface)
    return sigma[:, 0].pow(2).mean() + sigma[:, 3].pow(2).mean()
