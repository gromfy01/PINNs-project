"""
trainer2d.py — PINN с координатой z во входе и ПОЛНОЙ формой равновесия.

Расширенный вариант пункта 2 плана. Вход сети — [Q, k, α, μ, v, r, z], поэтому
осевые производные существуют и обе невязки записываются целиком:

    R_r = ∂σ_rr/∂r + ∂τ_rz/∂z + (σ_rr − σ_θθ)/r
    R_z = ∂τ_rz/∂r + ∂σ_zz/∂z + τ_rz/r

Семейства (ablation, отвечающий на замечание R1-c прямо):
    mlp2d           только data-loss
    pinn2d_reduced  только R_r без ∂τ_rz/∂z — ровно то, что считает 1D-код
    pinn2d_full     обе невязки целиком

Нормировка. r нормирован на радиус поверхности В ТОМ ЖЕ осевом сечении
(r = 1 — поверхность при любом z), z — на длину окна. Цепное правило:
∂/∂r_phys = (1/R) ∂/∂r_norm, ∂/∂z_phys = (1/L) ∂/∂z_norm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from trainer import ACTS, FourierR, IDX_SRR, IDX_STT, IDX_SZZ, IDX_TRZ, Scalers, metrics

FAMILIES_2D = ("mlp2d", "pinn2d_reduced", "pinn2d_full")


@dataclass
class Config2D:
    family: str = "pinn2d_full"
    n_layers: int = 4
    n_units: int = 128
    activation: str = "tanh"
    n_fourier: int = 4
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_sets: int = 32
    max_epochs: int = 300
    patience: int = 30
    val_frac: float = 0.15
    lambda_physics: float = 0.02
    lambda_bc: float = 0.5
    r_clip: float = 0.05
    seed: int = 0
    device: str = "cpu"

    def as_dict(self): return asdict(self)


class Net2D(nn.Module):
    def __init__(self, cfg: Config2D, n_in: int = 7, n_out: int = 4):
        super().__init__()
        self.emb = FourierR(cfg.n_fourier)          # эмбеддинг по последней колонке
        d = n_in + 2 * cfg.n_fourier
        act = ACTS[cfg.activation]
        layers: List[nn.Module] = []
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.n_units), act()]
            d = cfg.n_units
        layers += [nn.Linear(d, n_out)]
        self.body = nn.Sequential(*layers)
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x): return self.body(self.emb(x))


@dataclass
class Result2D:
    cfg: Dict
    epochs_run: int
    best_val: float
    history: List[float] = field(default_factory=list)


class Trainer2D:
    """
    Данные: proc (n,5), y (n,nz,nr,4), r (n,nz,nr), z (n,nz),
            r_phys (n,nz,nr), z_phys (n,nz).
    Колонки входа: [Q,k,alpha,mu,v, z, r] — r последняя, чтобы Fourier-эмбеддинг
    (общий с 1D-версией) считался именно по радиусу.
    """

    def __init__(self, cfg: Config2D):
        self.cfg = cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.dtype = torch.float64

    def _t(self, a): return torch.tensor(a, dtype=self.dtype, device=self.cfg.device)

    def _flat_X(self, proc, r, z):
        n, nz, nr = r.shape
        p = np.repeat(proc, nz * nr, axis=0)
        zz = np.repeat(z.reshape(n, nz, 1), nr, axis=2).reshape(-1, 1)
        rr = r.reshape(-1, 1)
        return np.concatenate([p, zz, rr], axis=1)

    def fit(self, proc, y, r, z, r_phys, z_phys, verbose=False) -> Result2D:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n, nz, nr, _ = y.shape
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * cfg.val_frac)))
        val_sets, tr_sets = perm[:n_val], perm[n_val:]

        Xtr = self._flat_X(proc[tr_sets], r[tr_sets], z[tr_sets])
        Ytr = y[tr_sets].reshape(-1, 4)
        self.sc = Scalers.fit(Xtr, Ytr)
        self.xm = self._t(self.sc.x_mean.reshape(1, -1)); self.xs = self._t(self.sc.x_std.reshape(1, -1))
        self.ym = self._t(self.sc.y_mean.reshape(1, -1)); self.ys = self._t(self.sc.y_std.reshape(1, -1))

        R_char = float(np.mean(r_phys[tr_sets][:, :, -1]))
        L_char = float(np.mean(z_phys[tr_sets][:, -1] - z_phys[tr_sets][:, 0]))
        sd = self.sc.y_std
        self.res_r_scale = self._t(np.array((sd[IDX_SRR] + sd[IDX_STT]) / R_char))
        self.res_z_scale = self._t(np.array((sd[IDX_TRZ] + sd[IDX_SZZ]) / R_char))
        self.L_char = L_char

        self.net = Net2D(cfg).to(dtype=self.dtype)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)

        P = self._t(proc[tr_sets]); Rn = self._t(r[tr_sets]); Zn = self._t(z[tr_sets])
        RP = self._t(r_phys[tr_sets]); ZP = self._t(z_phys[tr_sets])
        Ysc = (self._t(y[tr_sets]) - self.ym.reshape(1, 1, 1, 4)) / self.ys.reshape(1, 1, 1, 4)

        best, bad, hist, best_state = float("inf"), 0, [], None
        n_tr = len(tr_sets)
        g = torch.Generator().manual_seed(cfg.seed)
        use_phys = cfg.family != "mlp2d"
        full = cfg.family == "pinn2d_full"

        for epoch in range(cfg.max_epochs):
            self.net.train()
            order = torch.randperm(n_tr, generator=g)
            for s in range(0, n_tr, cfg.batch_sets):
                idx = order[s:s + cfg.batch_sets]
                B = len(idx)
                pb, rb, zb = P[idx], Rn[idx], Zn[idx]
                rpb, zpb, yb = RP[idx], ZP[idx], Ysc[idx]

                if use_phys:
                    loss = self._loss_with_physics(pb, rb, zb, rpb, zpb, yb, B, nz, nr, full)
                else:
                    X = self._pack(pb, rb, zb, B, nz, nr)
                    pred = self.net((X - self.xm) / self.xs).reshape(B, nz, nr, 4)
                    loss = ((pred - yb) ** 2).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                opt.step()

            v = self.evaluate(proc[val_sets], y[val_sets], r[val_sets], z[val_sets])["macro_rmse"]
            hist.append(v)
            if v < best - 1e-9:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return Result2D(cfg.as_dict(), len(hist), best, hist)

    def _pack(self, pb, rb, zb, B, nz, nr):
        p = pb.unsqueeze(1).unsqueeze(1).expand(B, nz, nr, 5).reshape(-1, 5)
        zz = zb.unsqueeze(-1).expand(B, nz, nr).reshape(-1, 1)
        rr = rb.reshape(-1, 1)
        return torch.cat([p, zz, rr], 1)

    def _loss_with_physics(self, pb, rb, zb, rpb, zpb, yb, B, nz, nr, full):
        cfg = self.cfg
        p = pb.unsqueeze(1).unsqueeze(1).expand(B, nz, nr, 5).reshape(-1, 5)
        rr = rb.reshape(-1, 1).clone().requires_grad_(True)
        zz = zb.unsqueeze(-1).expand(B, nz, nr).reshape(-1, 1).clone().requires_grad_(True)
        X = torch.cat([p, zz, rr], 1)
        y_sc = self.net((X - self.xm) / self.xs)
        y_ph = y_sc * self.ys + self.ym

        l_data = ((y_sc.reshape(B, nz, nr, 4) - yb) ** 2).mean()

        Rm = rpb[:, :, -1:].expand(B, nz, nr).reshape(-1)
        Lm = (zpb[:, -1] - zpb[:, 0]).reshape(B, 1, 1).expand(B, nz, nr).reshape(-1)
        r_ph = rpb.reshape(-1)
        r_safe = torch.maximum(r_ph, Rm * cfg.r_clip)

        d_srr_dr = torch.autograd.grad(y_ph[:, IDX_SRR].sum(), rr, create_graph=True)[0][:, 0] / Rm
        d_tau_dr = torch.autograd.grad(y_ph[:, IDX_TRZ].sum(), rr, create_graph=True)[0][:, 0] / Rm
        if full:
            d_tau_dz = torch.autograd.grad(y_ph[:, IDX_TRZ].sum(), zz, create_graph=True)[0][:, 0] / Lm
            d_szz_dz = torch.autograd.grad(y_ph[:, IDX_SZZ].sum(), zz, create_graph=True)[0][:, 0] / Lm
        else:
            d_tau_dz = torch.zeros_like(d_srr_dr)
            d_szz_dz = torch.zeros_like(d_srr_dr)

        res_r = d_srr_dr + d_tau_dz + (y_ph[:, IDX_SRR] - y_ph[:, IDX_STT]) / r_safe
        l_phys = torch.log1p((res_r / self.res_r_scale) ** 2).mean()
        if full:
            res_z = d_tau_dr + d_szz_dz + y_ph[:, IDX_TRZ] / r_safe
            l_phys = l_phys + torch.log1p((res_z / self.res_z_scale) ** 2).mean()

        # ГУ: σ_rr = τ_rz = 0 на r = 1 при каждом z
        r1 = torch.ones((B * nz, 1), dtype=self.dtype)
        pz = pb.unsqueeze(1).expand(B, nz, 5).reshape(-1, 5)
        zc = zb.reshape(-1, 1)
        Xb = torch.cat([pz, zc, r1], 1)
        pb_ = self.net((Xb - self.xm) / self.xs) * self.ys + self.ym
        l_bc = ((pb_[:, IDX_SRR] / self.ys[0, IDX_SRR]) ** 2
                + (pb_[:, IDX_TRZ] / self.ys[0, IDX_TRZ]) ** 2).mean()

        return l_data + cfg.lambda_physics * l_phys + cfg.lambda_bc * l_bc

    @torch.no_grad()
    def predict(self, proc, r, z) -> np.ndarray:
        self.net.eval()
        n, nz, nr = r.shape
        X = self._t(self._flat_X(proc, r, z))
        p = self.net((X - self.xm) / self.xs) * self.ys + self.ym
        return p.cpu().numpy().reshape(n, nz, nr, 4)

    @torch.no_grad()
    def predict_surface(self, proc, z) -> np.ndarray:
        self.net.eval()
        n, nz = z.shape
        p = np.repeat(proc, nz, axis=0)
        X = np.concatenate([p, z.reshape(-1, 1), np.ones((n * nz, 1))], 1)
        return (self.net((self._t(X) - self.xm) / self.xs) * self.ys
                + self.ym).cpu().numpy()

    def evaluate(self, proc, y, r, z) -> Dict[str, float]:
        return metrics(y.reshape(-1, 1, 4), self.predict(proc, r, z).reshape(-1, 1, 4))


def equilibrium_audit_2d(y: np.ndarray, r_phys: np.ndarray, z_phys: np.ndarray,
                         r_clip: float = 0.05) -> Dict[str, float]:
    """
    Невязки полной и редуцированной формы на сетке (nz, nr).
    Отношение медиан показывает, сколько теряет редукция — та же величина,
    что оценена по сырым данным в audit/RAW_AUDIT.md §6, но уже на предсказаниях.
    """
    n, nz, nr, _ = y.shape
    full_r, red_r, full_z, red_z = [], [], [], []
    for i in range(n):
        srr, stt, szz, tau = (y[i, :, :, k] for k in range(4))
        rp, zp = r_phys[i], z_phys[i]
        d_srr_dr = np.gradient(srr, rp[nz // 2], axis=1)
        d_tau_dr = np.gradient(tau, rp[nz // 2], axis=1)
        d_tau_dz = np.gradient(tau, zp, axis=0)
        d_szz_dz = np.gradient(szz, zp, axis=0)
        r_safe = np.maximum(rp, rp[:, -1:] * r_clip)
        hoop = (srr - stt) / r_safe
        sh = tau / r_safe
        full_r.append(np.median(np.abs(d_srr_dr + d_tau_dz + hoop)))
        red_r.append(np.median(np.abs(d_srr_dr + hoop)))
        full_z.append(np.median(np.abs(d_tau_dr + d_szz_dz + sh)))
        red_z.append(np.median(np.abs(d_tau_dr + sh)))
    return {"eq_r_full": float(np.median(full_r)), "eq_r_reduced": float(np.median(red_r)),
            "eq_z_full": float(np.median(full_z)), "eq_z_reduced": float(np.median(red_z))}
