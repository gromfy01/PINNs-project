"""
pinn_model_dde.py — PINN[DeepXDE / PyTorch backend] surrogate for residual
stress in axisymmetric wire drawing.

Структурный клон pinn_model_torch.py с заменой torch.autograd.grad на
dde.grad.jacobian (auto-cached partial derivatives для multi-output PDE).

PDE strong form (axisymmetric equilibrium, single-z plane):
  R₁(r) = ∂σ_rr/∂r + (σ_rr − σ_θθ)/r
  R₂(r) = ∂τ_rz/∂r + τ_rz/r
Производные в нормализованной системе → физические через 1/std_r.
log1p-bounded loss; r.clip(min=0.05) защита от 1/r blow-up.

BC: экстраполяционная traction-free на r_norm = R_EXTRAP_FREE_SURFACE (=1.5,
физическая свободная поверхность r_phys=R_phys, ВНЕ data range [0,1] =
25–75% R_phys). Физический prior, не противоречащий данным внутри зоны.

Architecture: dde.nn.FNN([6, ...units, 4], smooth-only activation, Glorot
normal), one network, 4 outputs (σ_rr, σ_θθ, σ_zz, τ_rz).

Public API (mirrors VPINN/JAX/Torch):
    smape, build_stress_dataset, von_mises_stress, METRIC_NAMES,
    per_component_report, cuml_scorer_pinn_dde,
    do_optuna_pinn_dde, test_best_model_pinn_dde,
    kfold_test_best_model_pinn_dde, predict_with_pinn_dde
"""
from __future__ import annotations

import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

import torch
import torch.nn as nn

import deepxde as dde

from sklearn.metrics import (
    explained_variance_score, max_error,
    mean_absolute_error, mean_squared_error,
    median_absolute_error, r2_score,
)
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

STRESS_COMPONENTS: List[str] = ["sigma_rr", "sigma_tt", "sigma_zz", "tau_rz"]
N_R: int = 20
PLANE_ORDER: List[int] = [2, 0, 1, 3]
R_EXTRAP_FREE_SURFACE: float = 1.5

METRIC_NAMES: List[str] = [
    "Explained Variance", "Median AE", "MSE", "MAE", "R²",
    "Max Error", "AIC", "BIC", "RMSE (4 comp macro)", "Von Mises R²",
    "SMAPE [%]",
]

ACTIVATIONS = ["tanh", "selu", "softplus"]


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(100.0 * np.mean(num / den))


def build_stress_dataset(X_stress: np.ndarray, y_stress: np.ndarray,
                          n_r: int = 20, verbose: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert y_stress.shape[0] == 4, "y_stress dim 0 must be 4 (stress planes)"
    n_sets = X_stress.shape[1]
    r_grid = np.linspace(0.0, 1.0, n_r, dtype=np.float32)
    proc   = X_stress[0].astype(np.float32)

    proc_rep = np.repeat(proc, n_r, axis=0)
    r_rep    = np.tile(r_grid, n_sets)[:, None]
    X_pinn   = np.hstack([proc_rep, r_rep]).astype(np.float32)

    y_pinn = np.stack([
        y_stress[PLANE_ORDER[i]].reshape(-1) for i in range(4)
    ], axis=1).astype(np.float32)

    if verbose:
        print(f"X_pinn : {X_pinn.shape}   [Q, k, α, μ, v, r]")
        print(f"y_pinn : {y_pinn.shape}   [σ_rr, σ_θθ, σ_zz, τ_rz]")
        print("Mapping: stress_33→σ_rr | stress_11→σ_θθ | "
              "stress_22→σ_zz | stress_12→τ_rz")
        print("\nДиапазоны y (физические FEM значения, МПа):")
        for i, lbl in enumerate(STRESS_COMPONENTS):
            print(f"  {lbl:10s}: [{y_pinn[:,i].min():10.4g},  "
                  f"{y_pinn[:,i].max():10.4g}]"
                  f"   mean_abs={np.abs(y_pinn[:,i]).mean():.4g}")
        print(f"\nBC контроль на границе data range (r_norm=1 = 75% R_phys):")
        for col, lbl in [(0, "σ_rr"), (3, "τ_rz")]:
            at_r1 = y_pinn[n_r-1::n_r, col]
            print(f"  {lbl} at r_norm=1: mean={at_r1.mean():.4g}  "
                  f"max|.|={np.max(np.abs(at_r1)):.4g}  (НЕ ноль — это не "
                  f"свободная поверхность)")
        print(f"  ⇒ BC σ_rr=τ_rz=0 применяется при r_norm={R_EXTRAP_FREE_SURFACE} "
              f"(экстраполяция к r_phys=R_phys, истинная свободная поверхность)")
    return X_pinn, y_pinn, r_grid, proc


def _split_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 3,
              test_size: float = 0.15, random_state: int = 42):
    n_sets = len(X) // N_R
    rng = np.random.default_rng(random_state)
    idx = np.arange(n_sets)
    n_test = max(1, int(n_sets * test_size))
    test_sets = rng.choice(idx, size=n_test, replace=False)
    train_sets = np.setdiff1d(idx, test_sets)

    def rows(sets):
        r = np.concatenate([np.arange(s * N_R, (s + 1) * N_R) for s in sets])
        return X[r], y[r]

    X_test, y_test = rows(test_sets)
    if n_splits == 1:
        return X_test, y_test, [X_test], [y_test], [rows(train_sets)[0]], [rows(train_sets)[1]]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    tr_X, tr_y, vl_X, vl_y = [], [], [], []
    for ti, vi in kf.split(train_sets):
        Xtr, ytr = rows(train_sets[ti])
        Xvl, yvl = rows(train_sets[vi])
        tr_X.append(Xtr); tr_y.append(ytr)
        vl_X.append(Xvl); vl_y.append(yvl)
    return X_test, y_test, vl_X, vl_y, tr_X, tr_y


def _clean(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(X).all(1) & np.isfinite(y).all(1)
    return X[mask], y[mask]


def _choose_worst(errors: np.ndarray) -> np.ndarray:
    return errors[np.argmax(errors[:, -1])]


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    s_rr, s_tt, s_zz, tau = (sigma[:, i] for i in range(4))
    dev = 0.5 * ((s_rr - s_tt) ** 2 + (s_tt - s_zz) ** 2 + (s_zz - s_rr) ** 2)
    return np.sqrt(np.maximum(dev + 3.0 * tau ** 2, 0.0))


def cuml_scorer_pinn_dde(y_true: np.ndarray, y_pred: np.ndarray,
                          model: nn.Module, X_train: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return np.array([0, 1e6, 1e6, 1e6, 0, 1e6, 1e6, 1e6, 1e6, 0, 200.0],
                        dtype=float)

    rmse_macro = float(np.mean([
        np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        for i in range(4)]))
    yf, pf = y_true.reshape(-1), y_pred.reshape(-1)
    evs = explained_variance_score(yf, pf)
    medae = median_absolute_error(yf, pf)
    mse = mean_squared_error(yf, pf)
    mae = mean_absolute_error(yf, pf)
    r2 = r2_score(yf, pf)
    me = max_error(yf, pf)
    n = max(len(X_train), 1)
    n_par = sum(p.numel() for p in model.parameters())
    aic = n * np.log(mse + 1e-12) + 2 * n_par
    bic = n * np.log(mse + 1e-12) + n_par * np.log(n + 1e-12)
    vm_r2 = r2_score(von_mises_stress(y_true), von_mises_stress(y_pred))
    sm = smape(yf, pf)
    return np.array([evs, medae, mse, mae, r2, me, aic, bic,
                     rmse_macro, vm_r2, sm])


def per_component_report(y_true: np.ndarray, y_pred: np.ndarray,
                          label: str = "Test") -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"rmse": [], "r2": [], "smape": []}
    print(f"\n── Per-component RMSE / R² / SMAPE ({label}) ──")
    for i, c in enumerate(STRESS_COMPONENTS):
        ri = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        sm = smape(y_true[:, i], y_pred[:, i])
        out["rmse"].append(ri); out["r2"].append(r2); out["smape"].append(sm)
        print(f"  {c:12s}:  RMSE={ri:10.4g}   R²={r2:+.4f}   SMAPE={sm:7.2f}%")
    vm_t = von_mises_stress(y_true); vm_p = von_mises_stress(y_pred)
    vm_rmse = float(np.sqrt(mean_squared_error(vm_t, vm_p)))
    vm_r2 = float(r2_score(vm_t, vm_p)); vm_sm = smape(vm_t, vm_p)
    print(f"  {'Von Mises':12s}:  RMSE={vm_rmse:10.4g}   R²={vm_r2:+.4f}   "
          f"SMAPE={vm_sm:7.2f}%   ← validation only")
    out["vm_rmse"] = [vm_rmse]; out["vm_r2"] = [vm_r2]; out["vm_smape"] = [vm_sm]
    return out


class OutputScaler:
    def __init__(self): self.mean_ = self.std_ = None

    def fit(self, y: np.ndarray):
        self.mean_ = y.mean(0, keepdims=True).astype(np.float32)
        self.std_  = y.std(0, keepdims=True).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, y):  return (y - self.mean_) / self.std_
    def inverse(self, y_sc): return y_sc * self.std_ + self.mean_

    def to_torch(self, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        m = torch.from_numpy(self.mean_).float()
        s = torch.from_numpy(self.std_).float()
        if device is not None:
            m = m.to(device); s = s.to(device)
        return m, s


def make_pinn_network_dde(layer_sizes: List[int], activation: str = "tanh"
                           ) -> nn.Module:
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation}")
    return dde.nn.FNN(layer_sizes, activation, "Glorot normal")


def _strong_eq_loss_dde(net: nn.Module, X_std: torch.Tensor,
                         r_phys: torch.Tensor, std_r: float,
                         y_mean: torch.Tensor, y_std: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    y_sc = net(X_std)
    d_srr_dr_std = dde.grad.jacobian(y_sc, X_std, i=0, j=5).squeeze(-1)
    d_tau_dr_std = dde.grad.jacobian(y_sc, X_std, i=3, j=5).squeeze(-1)

    y_mean_v = y_mean.reshape(-1); y_std_v = y_std.reshape(-1)
    sig_phys = y_sc * y_std_v + y_mean_v
    s_rr = sig_phys[:, 0]; s_tt = sig_phys[:, 1]; tau = sig_phys[:, 3]

    d_srr_dr = d_srr_dr_std * y_std_v[0] / (std_r + eps)
    d_tau_dr = d_tau_dr_std * y_std_v[3] / (std_r + eps)

    r_safe = r_phys.clamp(min=0.05)
    res1 = d_srr_dr + (s_rr - s_tt) / r_safe
    res2 = d_tau_dr + tau / r_safe

    scale1 = (y_std_v[0] ** 2 + y_std_v[1] ** 2) + eps
    scale2 = (y_std_v[3] ** 2) + eps

    l1 = torch.log1p(res1 ** 2 / scale1).mean()
    l2 = torch.log1p(res2 ** 2 / scale2).mean()
    return l1 + l2


def _bc_loss_dde(net: nn.Module, params_batch: torch.Tensor,
                  mx: torch.Tensor, sx: torch.Tensor,
                  y_mean: torch.Tensor, y_std: torch.Tensor,
                  r_extrap: float = R_EXTRAP_FREE_SURFACE,
                  eps: float = 1e-6) -> torch.Tensor:
    """Экстраполяционная traction-free BC при r_norm = r_extrap (=1.5 =
    физ. свободная поверхность r_phys=R_phys, ВНЕ data range [0, 1])."""
    B = params_batch.shape[0]
    r_col = torch.full((B, 1), float(r_extrap),
                       dtype=params_batch.dtype, device=params_batch.device)
    X = torch.cat([params_batch, r_col], dim=1)
    X_std = (X - mx) / sx
    pred = net(X_std)
    y_mean_v = y_mean.reshape(-1); y_std_v = y_std.reshape(-1)
    bc_rr  = -y_mean_v[0] / (y_std_v[0] + eps)
    bc_tau = -y_mean_v[3] / (y_std_v[3] + eps)
    return ((pred[:, 0] - bc_rr) ** 2).mean() + ((pred[:, 3] - bc_tau) ** 2).mean()


def _profile_residual_loss(pred_sc: torch.Tensor,
                            target_sc: torch.Tensor) -> torch.Tensor:
    p_c = pred_sc - pred_sc.mean(dim=1, keepdim=True)
    t_c = target_sc - target_sc.mean(dim=1, keepdim=True)
    l_centred = ((p_c - t_c) ** 2).mean()
    l_grad = ((torch.diff(pred_sc,   dim=1)
               - torch.diff(target_sc, dim=1)) ** 2).mean()
    return l_centred + l_grad


def _flat_to_samples(X: np.ndarray, y: np.ndarray, n_r: int = N_R):
    N = len(X)
    n_sets = N // n_r
    params = X[::n_r, :5].copy()
    r_grid = X[:n_r, 5].copy()
    y_s    = y.reshape(n_sets, n_r, -1).copy()
    return params, r_grid, y_s


def _make_opt(name: str, params, lr: float, weight_decay: float = 1e-4):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _train_single(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray,
                  best_params: Dict[str, Any], r_col_idx: int,
                  verbose_every: int = 20, label: str = ""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_tr, r_grid_np, y_tr_s = _flat_to_samples(X_tr, y_tr)
    params_te, _,         y_te_s = _flat_to_samples(X_te, y_te)

    scaler = OutputScaler().fit(y_tr)
    y_tr_s_sc = scaler.transform(y_tr_s.reshape(-1, 4)).reshape(y_tr_s.shape)
    y_te_s_sc = scaler.transform(y_te_s.reshape(-1, 4)).reshape(y_te_s.shape)

    Xtr_mx_np = X_tr.mean(0, keepdims=True).astype(np.float32)
    Xtr_sx_np = X_tr.std(0, keepdims=True).astype(np.float32)
    Xtr_sx_np[Xtr_sx_np == 0] = 1.0
    std_r_phys = float(Xtr_sx_np[0, r_col_idx])

    mx       = torch.from_numpy(Xtr_mx_np).float().to(device)
    sx       = torch.from_numpy(Xtr_sx_np).float().to(device)
    y_mean_t, y_std_t = scaler.to_torch(device=device)

    params_tr_t  = torch.from_numpy(params_tr).float().to(device)
    params_te_t  = torch.from_numpy(params_te).float().to(device)
    y_tr_s_sc_t  = torch.from_numpy(y_tr_s_sc).float().to(device)
    y_te_s_sc_t  = torch.from_numpy(y_te_s_sc).float().to(device)
    r_grid_t     = torch.from_numpy(r_grid_np).float().to(device)

    k_lay   = best_params["n_layers"]
    units   = [best_params[f"n_units_{i}"] for i in range(k_lay)]
    act     = best_params["activation"]
    bs      = best_params["batch_size"]
    lr      = best_params["learning_rate"]
    epochs  = best_params["max_epochs"]
    opt_nm  = best_params["optimizer"]
    l_phys  = best_params["lambda_physics"]
    l_bc    = best_params["lambda_bc"]
    l_prof  = best_params.get("lambda_profile", 0.1)
    clip    = best_params.get("grad_clip", 1.0)
    wd      = best_params.get("weight_decay", 1e-4)
    r_extr  = float(best_params.get("r_extrap", R_EXTRAP_FREE_SURFACE))

    layer_sizes = [6] + units + [4]
    seed = int(best_params.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)

    net = make_pinn_network_dde(layer_sizes, activation=act).to(device)
    opt = _make_opt(opt_nm, net.parameters(), lr, wd)

    n_train_sets = params_tr.shape[0]
    rng_np = np.random.default_rng(seed)

    best_val = float("inf"); pat = 0; patience = 30
    best_state: Optional[Dict[str, torch.Tensor]] = None
    train_losses, test_losses, phys_losses, prof_losses, bc_losses, vm_rmse_test = \
        [], [], [], [], [], []

    def _eval_test():
        net.eval()
        with torch.no_grad():
            B = params_te_t.shape[0]
            r_rep = r_grid_t.view(1, -1, 1).expand(B, -1, 1)
            p_rep = params_te_t.unsqueeze(1).expand(-1, N_R, -1)
            X_data = torch.cat([p_rep, r_rep], dim=2).reshape(-1, 6)
            X_data_std = (X_data - mx) / sx
            pred_sc_flat = net(X_data_std)
            pred_sc = pred_sc_flat.reshape(B, N_R, 4)
            tel = ((pred_sc - y_te_s_sc_t) ** 2).mean().item()
            pl_prof = _profile_residual_loss(pred_sc, y_te_s_sc_t).item()
            pred_np = scaler.inverse(pred_sc.reshape(-1, 4).cpu().numpy())
            y_true_flat = y_te_s.reshape(-1, 4)
            vm = float(np.sqrt(mean_squared_error(
                von_mises_stress(y_true_flat), von_mises_stress(pred_np))))
        net.train()
        return tel, pl_prof, pred_np, vm

    def _eval_phys_on_train_sample():
        net.eval()
        sub = min(256, n_train_sets)
        sel = rng_np.choice(n_train_sets, size=sub, replace=False)
        params_sub = params_tr_t[sel]
        r_rep = r_grid_t.view(1, -1, 1).expand(sub, -1, 1)
        p_rep = params_sub.unsqueeze(1).expand(-1, N_R, -1)
        X_data = torch.cat([p_rep, r_rep], dim=2).reshape(-1, 6)
        X_data_std = ((X_data - mx) / sx).detach().requires_grad_(True)
        r_phys = X_data[:, 5]
        l = _strong_eq_loss_dde(net, X_data_std, r_phys, std_r_phys,
                                 y_mean_t, y_std_t)
        dde.grad.clear()
        net.train()
        return float(l.detach())

    def _eval_bc_on_train_sample():
        net.eval()
        with torch.no_grad():
            sub = min(256, n_train_sets)
            sel = rng_np.choice(n_train_sets, size=sub, replace=False)
            params_sub = params_tr_t[sel]
            B = params_sub.shape[0]
            r_col = torch.full((B, 1), r_extr,
                               dtype=params_sub.dtype, device=device)
            X = torch.cat([params_sub, r_col], dim=1)
            X_std = (X - mx) / sx
            pred_sc = net(X_std)
            pred_phys = pred_sc * y_std_t.reshape(-1) + y_mean_t.reshape(-1)
            mse_phys = (pred_phys[:, 0] ** 2 + pred_phys[:, 3] ** 2).mean().item()
        net.train()
        return float(np.sqrt(mse_phys))

    for epoch in range(epochs):
        order = rng_np.permutation(n_train_sets)
        epoch_train_loss = 0.0; n_batches = 0
        for i in range(0, n_train_sets - bs + 1, bs):
            sel = order[i:i + bs]
            batch_p   = params_tr_t[sel]
            batch_y_sc = y_tr_s_sc_t[sel]

            B = batch_p.shape[0]
            r_rep = r_grid_t.view(1, -1, 1).expand(B, -1, 1)
            p_rep = batch_p.unsqueeze(1).expand(-1, N_R, -1)
            X_data = torch.cat([p_rep, r_rep], dim=2).reshape(-1, 6)
            X_data_std = ((X_data - mx) / sx).detach().requires_grad_(True)

            pred_sc_flat = net(X_data_std)
            pred_sc = pred_sc_flat.reshape(B, N_R, 4)

            l_data = ((pred_sc - batch_y_sc) ** 2).mean()

            r_phys = X_data[:, 5]
            l_phys_v = _strong_eq_loss_dde(net, X_data_std, r_phys, std_r_phys,
                                            y_mean_t, y_std_t)

            l_bc_v = _bc_loss_dde(net, batch_p, mx, sx, y_mean_t, y_std_t,
                                   r_extrap=r_extr)
            l_prof_v = _profile_residual_loss(pred_sc, batch_y_sc)

            loss = l_data + l_phys * l_phys_v + l_bc * l_bc_v + l_prof * l_prof_v

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            dde.grad.clear()

            epoch_train_loss += loss.item(); n_batches += 1

        trl = epoch_train_loss / max(n_batches, 1)
        tel, pl_prof, pred_np, vm = _eval_test()
        pl = _eval_phys_on_train_sample()
        bc_rms = _eval_bc_on_train_sample()

        train_losses.append(trl); test_losses.append(tel)
        phys_losses.append(pl);   prof_losses.append(pl_prof)
        bc_losses.append(bc_rms); vm_rmse_test.append(vm)

        if verbose_every and epoch % verbose_every == 0:
            y_true_flat = y_te_s.reshape(-1, 4)
            r2s = [r2_score(y_true_flat[:, i], pred_np[:, i]) for i in range(4)]
            tag = f"[{label}] " if label else ""
            print(f"{tag}Epoch {epoch:4d}/{epochs} | "
                  f"Train {trl:.4f}  Test {tel:.4f}  "
                  f"Strong {pl:.3e}  BC {bc_rms:.2f}МПа  "
                  f"Profile {pl_prof:.4f} | "
                  f"R²=[{r2s[0]:+.3f},{r2s[1]:+.3f},{r2s[2]:+.3f},{r2s[3]:+.3f}]")

        if tel < best_val:
            best_val = tel; pat = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                if verbose_every:
                    print(f"{('['+label+'] ') if label else ''}"
                          f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _, _, pred, _ = _eval_test()
    y_te_flat = y_te_s.reshape(-1, 4)
    metrics = cuml_scorer_pinn_dde(y_te_flat, pred, net, X_tr)

    history = {"train_losses": train_losses, "test_losses":  test_losses,
               "phys_losses":  phys_losses,  "prof_losses":  prof_losses,
               "bc_losses":    bc_losses,    "vm_rmse_test": vm_rmse_test}
    return net, metrics, mx, sx, scaler, history, y_te_flat, pred


def do_optuna_pinn_dde(X: np.ndarray, y: np.ndarray, n_trials: int = 50,
                        r_col_idx: int = -1, **kwargs):
    n_splits     = kwargs.get("n_splits", 3)
    n_layers_max = kwargs.get("n_layers", 8)
    n_neurons    = kwargs.get("n_neurons", 256)
    r_extrap_fix = float(kwargs.get("r_extrap", R_EXTRAP_FREE_SURFACE))

    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}   (PINN[DeepXDE] strong-form, "
          f"extrap-BC at r_norm={r_extrap_fix})")
    print(f"DeepXDE backend: {dde.backend.backend_name}, version: {dde.__version__}")

    (cur_X_test, cur_y_test,
     val_list_X, val_list_y,
     train_list_X, train_list_y) = _split_cv(X, y, n_splits=n_splits)

    def objective(trial: optuna.Trial):
        k_lay = trial.suggest_int("n_layers", 3, n_layers_max)
        units = [trial.suggest_int(f"n_units_{i}", 32, n_neurons)
                 for i in range(k_lay)]
        act   = trial.suggest_categorical("activation", ACTIVATIONS)
        lr    = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
        bs    = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        opt_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        epochs = trial.suggest_int("max_epochs", 200, 600)
        l_phys = trial.suggest_float("lambda_physics", 1e-4, 0.5,  log=True)
        l_bc   = trial.suggest_float("lambda_bc",      1e-3, 5.0,  log=True)
        l_prof = trial.suggest_float("lambda_profile", 1e-3, 1.0,  log=True)
        clip   = trial.suggest_float("grad_clip", 0.5, 5.0)
        wd     = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        errors = np.zeros((n_splits, 11))
        for si in range(n_splits):
            X_tr, y_tr = _clean(train_list_X[si], train_list_y[si])
            X_vl, y_vl = _clean(val_list_X[si],   val_list_y[si])

            trial_params = dict(
                n_layers=k_lay, activation=act, batch_size=bs,
                learning_rate=lr, max_epochs=epochs, optimizer=opt_nm,
                lambda_physics=l_phys, lambda_bc=l_bc,
                lambda_profile=l_prof, grad_clip=clip, weight_decay=wd,
                r_extrap=r_extrap_fix,
                seed=trial.number,
                **{f"n_units_{i}": units[i] for i in range(k_lay)},
            )
            try:
                _, metrics, _, _, _, _, _, _ = _train_single(
                    X_tr, y_tr, X_vl, y_vl, trial_params, r_col_idx,
                    verbose_every=0, label="")
                errors[si] = metrics
            except Exception as e:
                print(f"  [trial {trial.number} skipped fold {si}: "
                      f"{type(e).__name__}: {e}]")
                errors[si] = np.array(
                    [0, 1e6, 1e6, 1e6, 0, 1e6, 1e6, 1e6, 1e6, 0, 200.0])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        w = _choose_worst(errors)
        return float(w[8]) if pd.notnull(w[8]) else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = dict(study.best_params, r_extrap=r_extrap_fix)
    print("Best params:", best_params)
    return best_params, cur_X_test, cur_y_test, study.best_value


def test_best_model_pinn_dde(X: np.ndarray, y: np.ndarray,
                                best_params: Dict[str, Any],
                                r_col_idx: int = -1, **kwargs):
    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_extr = float(best_params.get("r_extrap", R_EXTRAP_FREE_SURFACE))
    print(f"Using device: {device}   (PINN[DeepXDE] strong-form, "
          f"extrap-BC at r_norm={r_extr})")

    (X_test, y_test, _, _, tr_X, tr_y) = _split_cv(X, y, n_splits=1)
    X_tr, y_tr = _clean(tr_X[0], tr_y[0])
    X_te, y_te = _clean(X_test, y_test)

    net, metrics, mx, sx, scaler, history, y_te_ret, pred_te = \
        _train_single(X_tr, y_tr, X_te, y_te, best_params, r_col_idx,
                       verbose_every=20, label="single")

    print("\n── Test Metrics ─────────────────────────────────────")
    for n, v in zip(METRIC_NAMES, metrics):
        print(f"  {n:<26}: {v:.4f}")
    per_component_report(y_te_ret, pred_te, label="Test (single fold)")

    return net, metrics, mx, sx, scaler, history


def kfold_test_best_model_pinn_dde(X: np.ndarray, y: np.ndarray,
                                      best_params: Dict[str, Any],
                                      r_col_idx: int = -1,
                                      n_folds: int = 5, random_state: int = 42,
                                      **kwargs) -> Dict[str, Any]:
    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_extr = float(best_params.get("r_extrap", R_EXTRAP_FREE_SURFACE))
    print(f"Using device: {device}   (PINN[DeepXDE] strong-form, "
          f"extrap-BC at r_norm={r_extr})")

    n_sets = len(X) // N_R
    rng = np.random.default_rng(random_state)
    set_idx = np.arange(n_sets); rng.shuffle(set_idx)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def rows(sets):
        r = np.concatenate([np.arange(s * N_R, (s + 1) * N_R) for s in sets])
        return X[r], y[r]

    per_fold = np.zeros((n_folds, 11))
    all_nets, all_histories = [], []
    all_mx, all_sx, all_scalers = [], [], []
    fold_test_arrays = []

    print("\n" + "=" * 72)
    print(f"PINN[DeepXDE] K-FOLD FINAL EVALUATION   (K={n_folds}, group-aware)")
    print("=" * 72)

    for k, (tr_i, te_i) in enumerate(kf.split(set_idx)):
        tr_sets, te_sets = set_idx[tr_i], set_idx[te_i]
        X_tr, y_tr = _clean(*rows(tr_sets))
        X_te, y_te = _clean(*rows(te_sets))
        print(f"\n─── Fold {k+1}/{n_folds} ──  train_sets={len(tr_sets)}  "
              f"test_sets={len(te_sets)}  train_rows={len(X_tr)}  "
              f"test_rows={len(X_te)} ───")

        fold_params = dict(best_params, seed=k)

        net, metrics, mx, sx, scaler, history, y_te_ret, pred_te = \
            _train_single(X_tr, y_tr, X_te, y_te, fold_params, r_col_idx,
                           verbose_every=50, label=f"fold{k+1}")
        per_fold[k] = metrics
        all_nets.append(net); all_histories.append(history)
        all_mx.append(mx); all_sx.append(sx); all_scalers.append(scaler)
        fold_test_arrays.append((y_te_ret, pred_te))

        print(f"\n[Fold {k+1}]  R²={metrics[4]:+.4f}   "
              f"RMSE_macro={metrics[8]:.4g}   "
              f"Von Mises R²={metrics[9]:+.4f}   SMAPE={metrics[10]:.2f}%")
        per_component_report(y_te_ret, pred_te, label=f"Fold {k+1}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_m = per_fold.mean(axis=0)
    std_m  = per_fold.std(axis=0)

    print("\n" + "=" * 72)
    print("PINN[DeepXDE] K-FOLD AGGREGATE  (mean ± std across folds)")
    print("=" * 72)
    for n, m, s in zip(METRIC_NAMES, mean_m, std_m):
        print(f"  {n:<26}: {m:10.4g}  ±  {s:8.3g}")

    best_k = int(np.argmin(per_fold[:, 8]))
    print(f"\n► Best fold (by macro RMSE): {best_k+1}\n")

    return {
        "per_fold_metrics": per_fold,
        "mean": mean_m, "std": std_m,
        "best_fold": best_k,
        "best_model":  all_nets[best_k],
        "best_mean_X": all_mx[best_k],
        "best_std_X":  all_sx[best_k],
        "best_scaler_y": all_scalers[best_k],
        "best_history":  all_histories[best_k],
        "all_nets": all_nets,
        "all_histories": all_histories,
        "fold_test_arrays": fold_test_arrays,
    }


def predict_with_pinn_dde(model: nn.Module, X: np.ndarray,
                           mean_X: torch.Tensor, std_X: torch.Tensor,
                           scaler_y: OutputScaler) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    X_std = (X_t - mean_X) / std_X
    model.eval()
    with torch.no_grad():
        pred_sc = model(X_std).detach().cpu().numpy()
    sigma = scaler_y.inverse(pred_sc)
    return {"sigma": sigma, "von_mises": von_mises_stress(sigma)}
