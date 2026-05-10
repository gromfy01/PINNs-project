import torch
import torch.nn as nn

from .components import build_mlp, FourierEmbedding, OutputScaler


class TotalStrainSystem(nn.Module):
    def __init__(self, hidden, n_freq=16, use_z=False, n_process_params=5):
        super().__init__()
        self.use_z = use_z
        self.n_process_params = n_process_params
        self.r_embed = FourierEmbedding(n_freq=n_freq, fixed=True)
        in_dim = 2 * n_freq + (1 if use_z else 0) + n_process_params
        self.net = build_mlp(in_dim, hidden, 4, "tanh")
        self.scaler = OutputScaler(4)

    def forward(self, x):
        r = x[:, 0:1]
        r_feat = self.r_embed(r)
        if self.use_z:
            z = x[:, 1:2]
            params = x[:, 2:]
            inp = torch.cat([r_feat, z, params], dim=-1)
        else:
            params = x[:, 1:]
            inp = torch.cat([r_feat, params], dim=-1)
        return self.scaler(self.net(inp))


def saint_venant_residual(model, x, eps=1e-6):
    if not model.use_z:
        return torch.tensor(0.0, device=x.device)

    x = x.detach().requires_grad_(True)
    eps_pred = model(x)
    err, ett, ezz, grz = (eps_pred[:, i:i + 1] for i in range(4))
    r = x[:, 0:1]

    def d(y, dim):
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True,
        )[0][:, dim:dim + 1]

    d2err_dz2 = d(d(err, 1), 1)
    d2ezz_dr2 = d(d(ezz, 0), 0)
    dezz_dr = d(ezz, 0)
    dgrz_dz = d(grz, 1)
    dgrz_drdz = d(dgrz_dz, 0)

    rhs1 = dgrz_drdz + dgrz_dz / (r + eps)
    res1 = (d2err_dz2 + d2ezz_dr2 - dezz_dr / (r + eps) - rhs1).pow(2).mean()

    d2ett_dz2 = d(d(ett, 1), 1)
    dett_dz = d(ett, 1)
    res2 = (d2ett_dz2 + d(dett_dz, 0) / (r + eps) + dett_dz / (r + eps) ** 2).pow(2).mean()

    return res1 + res2
