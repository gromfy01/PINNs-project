"""
VPINN surrogate for residual stress in axisymmetric wire drawing.

─── Mirror of pinn_model.py with WEAK-FORM equilibrium ────────────────────
Data format, input/output shapes, Optuna pipeline, K-fold logic, external
validation, SMAPE — all identical to pinn_model.py.  The ONLY difference
is how the equilibrium PDEs are enforced.

STRONG form (classical PINN):
    Eq1:  ∂σ_rr/∂r + (σ_rr − σ_θθ)/r = 0   pointwise
    Eq2:  ∂τ_rz/∂r + τ_rz / r         = 0   pointwise

WEAK form (VPINN).  Multiply by r·v(r) with v(0)=v(1)=0, integrate by parts:

    ∫₀¹ [∂σ_rr/∂r + (σ_rr − σ_θθ)/r] · r · v(r) dr  =  0
      ⇒   R_k^(1) = − ∫₀¹ [ r · σ_rr · v_k'(r) + σ_θθ · v_k(r) ] dr

    ∫₀¹ [∂τ_rz/∂r + τ_rz / r] · r · v(r) dr  =  0
      ⇒   R_k^(2) = − ∫₀¹ r · τ_rz · v_k'(r) dr

No autograd on r — the derivative lives on the known test function.  The
r·dr weight is the natural cylindrical volume element, which regularises
the 1/r singularity at the axis:  "r-симметрия зашита в слабой форме".

Test-function basis (Kharazmi et al., hp-VPINNs):
    v_k(r) = sin(kπr),   v_k'(r) = kπ · cos(kπr),    k = 1..K

Quadrature:
    Gauss–Legendre on [0,1] with N_q nodes (default 32).

Everything else — dataset builder, SMAPE, scoring, k-fold, external
validation, plotting — is identical in structure to pinn_model.py so the
runner can be switched with a single import line.

Public API (mirrors pinn_model.py):
    smape                                         METRIC_NAMES
    build_stress_dataset                          per_component_report
    von_mises_stress                              cuml_scorer_vpinn
    do_optuna_vpinn                               test_best_model_vpinn
    kfold_test_best_model_vpinn                   predict_with_vpinn
    evaluate_on_external_set
"""

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    explained_variance_score, mean_squared_error,
    mean_absolute_error, median_absolute_error,
    r2_score, max_error,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
STRESS_COMPONENTS = ["sigma_rr", "sigma_tt", "sigma_zz", "tau_rz"]
N_R = 20                         # radial points per parameter set
PLANE_ORDER = [2, 0, 1, 3]       # y_stress plane → PINN output column

# r_norm=1.5 соответствует физической свободной поверхности проволоки
# (r_norm ∈ [0,1] = 25–75% R_phys; r_phys = R_phys * (0.25 + 0.5*r_norm)).
R_EXTRAP_FREE_SURFACE: float = 1.5

METRIC_NAMES = [
    "Explained Variance", "Median AE", "MSE", "MAE", "R²",
    "Max Error", "AIC", "BIC", "RMSE (4 comp macro)", "Von Mises R²",
    "SMAPE [%]",
]


# ─────────────────────────────────────────────────────────────────────────────
# SMAPE
# ─────────────────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(100.0 * np.mean(num / den))


# ─────────────────────────────────────────────────────────────────────────────
# Data builder (unchanged vs. PINN)
# ─────────────────────────────────────────────────────────────────────────────

def build_stress_dataset(X_stress: np.ndarray, y_stress: np.ndarray,
                          n_r: int = 20, verbose: bool = True):
    """Convert (4, N_sets, 5) / (4, N_sets, 20) to flat VPINN arrays."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Group-aware split
# ─────────────────────────────────────────────────────────────────────────────

def _split_cv(X, y, n_splits=3, test_size=0.15, random_state=42):
    n_sets = len(X) // N_R
    rng    = np.random.default_rng(random_state)
    idx    = np.arange(n_sets)
    n_test = max(1, int(n_sets * test_size))
    test_sets  = rng.choice(idx, size=n_test, replace=False)
    train_sets = np.setdiff1d(idx, test_sets)

    def rows(sets):
        r = np.concatenate([np.arange(s*N_R, (s+1)*N_R) for s in sets])
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


def _clean(X, y):
    mask = np.isfinite(X).all(1) & np.isfinite(y).all(1)
    # Clean must keep (set × n_r) blocks intact for sample-level access.
    # Assume there are no NaN/Inf in well-formed Abaqus output → mask all True.
    return X[mask], y[mask]


def _choose_worst(errors):
    return errors[np.argmax(errors[:, -1])]


# ─────────────────────────────────────────────────────────────────────────────
# Von Mises  (validation only)
# ─────────────────────────────────────────────────────────────────────────────

def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    s_rr, s_tt, s_zz, tau = (sigma[:, i] for i in range(4))
    dev = 0.5 * ((s_rr-s_tt)**2 + (s_tt-s_zz)**2 + (s_zz-s_rr)**2)
    return np.sqrt(np.maximum(dev + 3.0*tau**2, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (11 metrics, identical to PINN module)
# ─────────────────────────────────────────────────────────────────────────────

def cuml_scorer_vpinn(y_true, y_pred, model, X_train):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return np.array([0, 1e6, 1e6, 1e6, 0, 1e6, 1e6, 1e6, 1e6, 0, 200.0],
                        dtype=float)

    rmse_macro = float(np.mean([
        np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        for i in range(4)]))

    yf, pf = y_true.reshape(-1), y_pred.reshape(-1)
    evs   = explained_variance_score(yf, pf)
    medae = median_absolute_error(yf, pf)
    mse   = mean_squared_error(yf, pf)
    mae   = mean_absolute_error(yf, pf)
    r2    = r2_score(yf, pf)
    me    = max_error(yf, pf)
    n_par = sum(p.numel() for p in model.parameters())
    n     = max(len(X_train), 1)
    aic   = n * np.log(mse + 1e-12) + 2 * n_par
    bic   = n * np.log(mse + 1e-12) + n_par * np.log(n + 1e-12)
    vm_r2 = r2_score(von_mises_stress(y_true), von_mises_stress(y_pred))
    sm    = smape(yf, pf)
    return np.array([evs, medae, mse, mae, r2, me, aic, bic,
                     rmse_macro, vm_r2, sm])


def per_component_report(y_true, y_pred, label: str = "Test"):
    out = {"rmse": [], "r2": [], "smape": []}
    print(f"\n── Per-component RMSE / R² / SMAPE ({label}) ──")
    for i, c in enumerate(STRESS_COMPONENTS):
        ri = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        sm = smape(y_true[:, i], y_pred[:, i])
        out["rmse"].append(ri); out["r2"].append(r2); out["smape"].append(sm)
        print(f"  {c:12s}:  RMSE={ri:10.4g}   R²={r2:+.4f}   SMAPE={sm:7.2f}%")
    vm_t = von_mises_stress(y_true); vm_p = von_mises_stress(y_pred)
    vm_rmse = np.sqrt(mean_squared_error(vm_t, vm_p))
    vm_r2 = r2_score(vm_t, vm_p); vm_sm = smape(vm_t, vm_p)
    print(f"  {'Von Mises':12s}:  RMSE={vm_rmse:10.4g}   R²={vm_r2:+.4f}   "
          f"SMAPE={vm_sm:7.2f}%   ← validation only")
    out["vm_rmse"] = vm_rmse; out["vm_r2"] = vm_r2; out["vm_smape"] = vm_sm
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Output scaler
# ─────────────────────────────────────────────────────────────────────────────

class OutputScaler:
    def __init__(self): self.mean_ = self.std_ = None

    def fit(self, y: np.ndarray):
        self.mean_ = y.mean(0, keepdims=True).astype(np.float32)
        self.std_  = y.std(0, keepdims=True).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, y):   return (y - self.mean_) / self.std_
    def inverse(self, y_sc):  return y_sc * self.std_ + self.mean_

    def to_torch(self, device):
        return (torch.FloatTensor(self.mean_).to(device),
                torch.FloatTensor(self.std_).to(device))


# ─────────────────────────────────────────────────────────────────────────────
# Network (identical architecture to PINN — only loss changes)
# ─────────────────────────────────────────────────────────────────────────────

SMOOTH_ACTS = {"tanh": nn.Tanh, "selu": nn.SELU, "softplus": nn.Softplus}


class VPINNNetwork(nn.Module):
    N_OUTPUTS = 4

    def __init__(self, layer_sizes, activation="tanh"):
        super().__init__()
        Act = SMOOTH_ACTS.get(activation, nn.Tanh)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(Act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Quadrature & test functions  (pre-computed once per fold)
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_legendre_01(n_q: int):
    """Gauss–Legendre nodes and weights on [0, 1]."""
    # numpy roots on [-1, 1] → shift to [0, 1]
    x, w = np.polynomial.legendre.leggauss(n_q)
    r_q = 0.5 * (x + 1.0)
    w_q = 0.5 * w
    return r_q.astype(np.float32), w_q.astype(np.float32)


def _build_test_basis(r_q: np.ndarray, K: int):
    """
    sin-basis test functions v_k(r) = sin(kπr), zero at r=0 and r=1.
    Returns
    -------
    phi    : (K, N_q)  v_k(r_q)
    phi_d  : (K, N_q)  v_k'(r_q) = kπ cos(kπr_q)
    """
    k   = np.arange(1, K + 1, dtype=np.float32)[:, None]         # (K, 1)
    rq  = r_q[None, :]                                           # (1, N_q)
    phi   = np.sin(np.pi * k * rq).astype(np.float32)            # (K, N_q)
    phi_d = (np.pi * k * np.cos(np.pi * k * rq)).astype(np.float32)
    return phi, phi_d


class _VPINNContext:
    """Bundles quadrature and test-function tensors, living on device."""
    def __init__(self, n_q: int, K: int, device):
        r_q, w_q = _gauss_legendre_01(n_q)
        phi, phi_d = _build_test_basis(r_q, K)
        self.r_q   = torch.from_numpy(r_q).to(device)            # (N_q,)
        self.w_q   = torch.from_numpy(w_q).to(device)            # (N_q,)
        self.phi   = torch.from_numpy(phi).to(device)            # (K, N_q)
        self.phi_d = torch.from_numpy(phi_d).to(device)          # (K, N_q)
        self.n_q   = n_q
        self.K     = K


# ─────────────────────────────────────────────────────────────────────────────
# Weak-form equilibrium loss (the heart of VPINN)
# ─────────────────────────────────────────────────────────────────────────────

def _weak_equilibrium_loss(model, params_batch, ctx: _VPINNContext,
                           mx, sx, y_mean_t, y_std_t, r_col_idx, eps=1e-6):
    """
    params_batch : (B, 5)   UNscaled process parameters
    ctx          : quadrature + test-function bundle

    For each sample in the batch and each test function v_k, compute
        R_k^(1) = − ∫₀¹ [ r · σ_rr · v_k'(r) + σ_θθ · v_k(r) ] dr
        R_k^(2) = − ∫₀¹ r · τ_rz · v_k'(r) dr
    via Gauss–Legendre quadrature, and return their MSE normalised by
    the physical variance of the stress components involved.

    Implementation notes
    --------------------
    *  No autograd on r — the derivative is baked into v_k'.
    *  The 1/r singularity is handled by the cylindrical weight r·dr,
       which is implicitly included (see derivation in module docstring).
    *  Per-equation normalisation uses σ_rr and τ_rz stds from y_std_t.
    """
    B = params_batch.shape[0]
    N_q, K = ctx.n_q, ctx.K

    # ───── Build input (B, N_q, 6) = params ⊗ r_q ─────
    p_rep = params_batch.unsqueeze(1).expand(B, N_q, 5)           # (B, N_q, 5)
    r_rep = ctx.r_q.unsqueeze(0).unsqueeze(2).expand(B, N_q, 1)   # (B, N_q, 1)
    X     = torch.cat([p_rep, r_rep], dim=2).reshape(B * N_q, 6)  # (B*N_q, 6)

    # Same standardisation as the rest of the pipeline
    X_std = (X - mx) / sx

    # Forward → scaled stresses → physical stresses
    sig_sc   = model(X_std).reshape(B, N_q, 4)                    # scaled
    sig_phys = sig_sc * y_std_t + y_mean_t                        # physical

    s_rr = sig_phys[..., 0]     # (B, N_q)
    s_tt = sig_phys[..., 1]
    tau  = sig_phys[..., 3]

    # ───── Weak residuals (B, K) via quadrature ─────
    # integrand_1(r) = r · σ_rr · v_k'(r) + σ_θθ · v_k(r)
    # integrand_2(r) = r · τ_rz  · v_k'(r)
    # Integrate with weights w_q.  R_k = − Σ_q w_q · integrand(r_q)

    r_w_q = ctx.r_q * ctx.w_q                        # (N_q,)  — r · w_q

    # term_a[b, k] = Σ_q w_q · r_q · σ_rr(b, q) · φ_k'(q)
    # term_b[b, k] = Σ_q w_q · σ_θθ(b, q) · φ_k(q)
    # term_c[b, k] = Σ_q w_q · r_q · τ_rz(b, q) · φ_k'(q)
    # Build (B, N_q) integrand matrices first, then contract with (K, N_q).
    f_rr = s_rr * r_w_q                              # (B, N_q): r·w·σ_rr
    f_tt = s_tt * ctx.w_q                            # (B, N_q): w·σ_θθ
    f_tz = tau  * r_w_q                              # (B, N_q): r·w·τ_rz

    # Einsum over N_q: R[b,k] = Σ_q f[b,q] · phi_something[k,q]
    term_a = torch.einsum("bq,kq->bk", f_rr, ctx.phi_d)    # (B, K)
    term_b = torch.einsum("bq,kq->bk", f_tt, ctx.phi)      # (B, K)
    term_c = torch.einsum("bq,kq->bk", f_tz, ctx.phi_d)    # (B, K)

    R1 = -(term_a + term_b)     # Eq1 residual   (B, K)
    R2 = -term_c                # Eq2 residual   (B, K)

    # ───── Per-equation normalisation (physical variance) ─────
    scale1 = (y_std_t[0, 0] ** 2 + y_std_t[0, 1] ** 2) + eps   # σ_rr, σ_θθ
    scale2 = (y_std_t[0, 3] ** 2) + eps                         # τ_rz

    # log1p keeps the loss bounded when a few outlier sets blow up
    loss1 = torch.mean(torch.log1p(R1 ** 2 / scale1))
    loss2 = torch.mean(torch.log1p(R2 ** 2 / scale2))
    return loss1 + loss2


# ─────────────────────────────────────────────────────────────────────────────
# BC loss — экстраполяционная traction-free на свободной поверхности
# ─────────────────────────────────────────────────────────────────────────────

def _bc_loss(model, params_batch, mx, sx, y_mean_t, y_std_t,
             r_extrap=R_EXTRAP_FREE_SURFACE, eps=1e-6):
    """Экстраполяционная BC: σ_rr(r_extrap) = τ_rz(r_extrap) = 0.

    Коллокационные точки генерируются при r_norm = r_extrap (по умолчанию
    1.5 = физическая свободная поверхность r_phys=R_phys, ВНЕ обучающего
    диапазона r_norm ∈ [0, 1]). Физический prior, не противоречащий данным.
    """
    B = params_batch.shape[0]
    r_col = torch.full((B, 1), float(r_extrap),
                       device=params_batch.device, dtype=params_batch.dtype)
    X = torch.cat([params_batch, r_col], dim=1)
    X_std = (X - mx) / sx
    pred  = model(X_std)
    bc_rr  = -y_mean_t[0, 0] / (y_std_t[0, 0] + eps)
    bc_tau = -y_mean_t[0, 3] / (y_std_t[0, 3] + eps)
    return (torch.mean((pred[:, 0] - bc_rr)  ** 2) +
            torch.mean((pred[:, 3] - bc_tau) ** 2))


def _bc_rms_phys(model, params_batch, mx, sx, y_mean_t, y_std_t,
                  r_extrap=R_EXTRAP_FREE_SURFACE):
    """RMS(σ_rr² + τ_rz²) в физических МПа при r_norm=r_extrap — мониторинг."""
    model.eval()
    with torch.no_grad():
        B = params_batch.shape[0]
        r_col = torch.full((B, 1), float(r_extrap),
                           device=params_batch.device, dtype=params_batch.dtype)
        X = torch.cat([params_batch, r_col], dim=1)
        X_std = (X - mx) / sx
        pred_sc = model(X_std)
        pred_phys = pred_sc * y_std_t + y_mean_t
        mse_phys = (pred_phys[:, 0] ** 2 + pred_phys[:, 3] ** 2).mean().item()
    model.train()
    return float(np.sqrt(mse_phys))


# ─────────────────────────────────────────────────────────────────────────────
# Profile residual loss — sample-level (B, n_r, C)
# ─────────────────────────────────────────────────────────────────────────────

def _profile_residual_loss_samples(pred_sc: torch.Tensor,
                                   target_sc: torch.Tensor) -> torch.Tensor:
    """
    pred_sc, target_sc : (B, n_r, C)   scaled

    Centred-profile + radial-gradient residual, computed correctly per sample.
    """
    p_c = pred_sc   - pred_sc.mean(dim=1, keepdim=True)
    t_c = target_sc - target_sc.mean(dim=1, keepdim=True)
    l_centred = torch.mean((p_c - t_c) ** 2)
    l_grad    = torch.mean((torch.diff(pred_sc,   dim=1)
                            - torch.diff(target_sc, dim=1)) ** 2)
    return l_centred + l_grad


# ─────────────────────────────────────────────────────────────────────────────
# Sample-level data reshaping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_to_samples(X, y, n_r=N_R):
    """
    X : (N_sets*n_r, 6)  →  params (N_sets, 5),  r_grid (n_r,),  y (N_sets, n_r, 4)
    Assumes rows are grouped by parameter set in order.
    """
    N = len(X)
    n_sets = N // n_r
    params = X[::n_r, :5].copy()                # (N_sets, 5)
    r_grid = X[:n_r,  5].copy()                 # (n_r,)
    y_s    = y.reshape(n_sets, n_r, -1).copy()  # (N_sets, n_r, 4)
    return params, r_grid, y_s


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_opt(name, model, lr):
    if name == "Adam":  return optim.Adam(model.parameters(), lr=lr)
    if name == "LBFGS": return optim.LBFGS(model.parameters(), lr=lr,
                                            max_iter=20, history_size=10)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Sample-level training epoch
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, optimizer, opt_name, loader,
                 params_full_t, targets_full_t, r_grid_t,
                 ctx: _VPINNContext,
                 mx, sx, y_mean_t, y_std_t,
                 lambda_phys, lambda_bc, lambda_profile,
                 crit, grad_clip=1.0,
                 r_extrap=R_EXTRAP_FREE_SURFACE):
    model.train()

    def _compute_batch_loss(pb_params, pb_targets_sc):
        """
        pb_params     : (B, 5) unscaled process params
        pb_targets_sc : (B, n_r, 4) scaled target profiles
        """
        B = pb_params.shape[0]
        n_r = pb_targets_sc.shape[1]

        # (a) Data forward at the r_grid points of every sample
        r_grid_rep = r_grid_t.unsqueeze(0).unsqueeze(2).expand(B, n_r, 1)
        p_rep      = pb_params.unsqueeze(1).expand(B, n_r, 5)
        X_data     = torch.cat([p_rep, r_grid_rep], dim=2).reshape(B*n_r, 6)
        X_data_std = (X_data - mx) / sx
        pred_sc    = model(X_data_std).reshape(B, n_r, 4)

        # (b) Data MSE
        l_data = crit(pred_sc, pb_targets_sc)

        # (c) Weak-form equilibrium
        l_phys = lambda_phys * _weak_equilibrium_loss(
            model, pb_params, ctx, mx, sx, y_mean_t, y_std_t, r_col_idx=5)

        # (d) BC — extrapolation at r_norm=r_extrap (free surface)
        l_bc = lambda_bc * _bc_loss(
            model, pb_params, mx, sx, y_mean_t, y_std_t, r_extrap=r_extrap)

        # (e) Profile residual (centred + ∂/∂r via finite differences)
        l_prof = lambda_profile * _profile_residual_loss_samples(
            pred_sc, pb_targets_sc)

        return l_data + l_phys + l_bc + l_prof

    if opt_name == "LBFGS":
        pb_params   = params_full_t
        pb_targets  = targets_full_t
        def closure():
            optimizer.zero_grad()
            loss = _compute_batch_loss(pb_params, pb_targets)
            loss.backward()
            return loss
        optimizer.step(closure)
    else:
        for pb_params, pb_targets in loader:
            optimizer.zero_grad()
            loss = _compute_batch_loss(pb_params, pb_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Core single-fold trainer — mirrors pinn_model._train_single
# ─────────────────────────────────────────────────────────────────────────────

def _train_single(X_tr, y_tr, X_te, y_te, best_params, r_col_idx, device,
                   n_quad, n_test_funcs,
                   verbose_every=20, label=""):
    # ── Reshape flat arrays to sample-level ──
    params_tr, r_grid_np, y_tr_s = _flat_to_samples(X_tr, y_tr)
    params_te, _,          y_te_s = _flat_to_samples(X_te, y_te)

    # ── Output scaler (fit on training flat y — unchanged semantics) ──
    scaler = OutputScaler().fit(y_tr)
    y_tr_s_sc = scaler.transform(y_tr_s.reshape(-1, 4)).reshape(y_tr_s.shape)
    y_te_s_sc = scaler.transform(y_te_s.reshape(-1, 4)).reshape(y_te_s.shape)

    # ── Standardisation reference from flat (X_tr) ──
    Xtr_flat = torch.FloatTensor(X_tr).to(device)
    mx = Xtr_flat.mean(0, keepdim=True)
    sx = Xtr_flat.std(0, unbiased=False, keepdim=True); sx[sx == 0] = 1.0

    # ── Move everything to device ──
    params_tr_t  = torch.FloatTensor(params_tr).to(device)
    params_te_t  = torch.FloatTensor(params_te).to(device)
    y_tr_s_sc_t  = torch.FloatTensor(y_tr_s_sc).to(device)
    y_te_s_sc_t  = torch.FloatTensor(y_te_s_sc).to(device)
    r_grid_t     = torch.FloatTensor(r_grid_np).to(device)
    y_mean_t, y_std_t = scaler.to_torch(device)

    ctx = _VPINNContext(n_quad, n_test_funcs, device)

    # ── Unpack best_params ──
    k_lay   = best_params["n_layers"]
    units   = [best_params[f"n_units_{i}"] for i in range(k_lay)]
    act     = best_params["activation"]
    bs      = best_params["batch_size"]          # sample-level batch
    lr      = best_params["learning_rate"]
    epochs  = best_params["max_epochs"]
    opt_nm  = best_params["optimizer"]
    l_phys  = best_params["lambda_physics"]
    l_bc    = best_params["lambda_bc"]
    l_prof  = best_params.get("lambda_profile", 0.1)
    clip    = best_params.get("grad_clip", 1.0)
    r_extr  = float(best_params.get("r_extrap", R_EXTRAP_FREE_SURFACE))

    layer_sizes = [6] + units + [VPINNNetwork.N_OUTPUTS]
    model     = VPINNNetwork(layer_sizes, act).to(device)
    crit      = nn.MSELoss()
    optimizer = _make_opt(opt_nm, model, lr)
    loader    = DataLoader(TensorDataset(params_tr_t, y_tr_s_sc_t),
                            batch_size=bs, shuffle=True)

    # ── Training loop with early stopping ──
    best_val = 1e18; pat = 0; patience = 30; best_state = None
    train_losses, test_losses, phys_losses, prof_losses, bc_losses, vm_rmse_test = \
        [], [], [], [], [], []

    def _eval_test():
        """Return dict with per-epoch test metrics."""
        model.eval()
        with torch.no_grad():
            # Data MSE on test samples
            B = params_te_t.shape[0]
            n_r = y_te_s_sc_t.shape[1]
            r_rep = r_grid_t.unsqueeze(0).unsqueeze(2).expand(B, n_r, 1)
            p_rep = params_te_t.unsqueeze(1).expand(B, n_r, 5)
            X_te_data = torch.cat([p_rep, r_rep], dim=2).reshape(-1, 6)
            pred_sc_t = model((X_te_data - mx) / sx).reshape(B, n_r, 4)
            tel = crit(pred_sc_t, y_te_s_sc_t).item()
            pl_prof = _profile_residual_loss_samples(pred_sc_t, y_te_s_sc_t).item()
            pred_np = scaler.inverse(pred_sc_t.reshape(-1, 4).cpu().numpy())
            y_true_flat = y_te_s.reshape(-1, 4)
            vm = np.sqrt(mean_squared_error(
                von_mises_stress(y_true_flat), von_mises_stress(pred_np)))
        return tel, pl_prof, pred_np, vm

    def _eval_train_data():
        model.eval()
        with torch.no_grad():
            B = params_tr_t.shape[0]
            n_r = y_tr_s_sc_t.shape[1]
            r_rep = r_grid_t.unsqueeze(0).unsqueeze(2).expand(B, n_r, 1)
            p_rep = params_tr_t.unsqueeze(1).expand(B, n_r, 5)
            X_tr_data = torch.cat([p_rep, r_rep], dim=2).reshape(-1, 6)
            pred_sc_r = model((X_tr_data - mx) / sx).reshape(B, n_r, 4)
            return crit(pred_sc_r, y_tr_s_sc_t).item()

    def _eval_phys_on_train():
        model.train()
        with torch.enable_grad():
            val = _weak_equilibrium_loss(
                model, params_tr_t, ctx, mx, sx, y_mean_t, y_std_t, 5).item()
        model.eval()
        return val

    def _eval_bc_on_train():
        return _bc_rms_phys(model, params_tr_t, mx, sx, y_mean_t, y_std_t,
                             r_extrap=r_extr)

    for epoch in range(epochs):
        _train_epoch(model, optimizer, opt_nm, loader,
                     params_tr_t, y_tr_s_sc_t, r_grid_t,
                     ctx, mx, sx, y_mean_t, y_std_t,
                     l_phys, l_bc, l_prof, crit, clip,
                     r_extrap=r_extr)

        trl        = _eval_train_data()
        tel, pl_prof, pred_np, vm = _eval_test()
        pl         = _eval_phys_on_train()
        bc_rms     = _eval_bc_on_train()

        train_losses.append(trl); test_losses.append(tel)
        phys_losses.append(pl);   prof_losses.append(pl_prof)
        bc_losses.append(bc_rms); vm_rmse_test.append(vm)

        if verbose_every and epoch % verbose_every == 0:
            y_true_flat = y_te_s.reshape(-1, 4)
            r2s = [r2_score(y_true_flat[:, i], pred_np[:, i]) for i in range(4)]
            tag = f"[{label}] " if label else ""
            print(f"{tag}Epoch {epoch:4d}/{epochs} | "
                  f"Train {trl:.4f}  Test {tel:.4f}  "
                  f"Weak {pl:.3e}  BC {bc_rms:.2f}МПа  "
                  f"Profile {pl_prof:.4f} | "
                  f"R²=[{r2s[0]:+.3f},{r2s[1]:+.3f},{r2s[2]:+.3f},{r2s[3]:+.3f}]")

        if tel < best_val:
            best_val = tel; pat = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                if verbose_every:
                    print(f"{('['+label+'] ') if label else ''}"
                          f"Early stopping at epoch {epoch}")
                break

    if best_state: model.load_state_dict(best_state)

    # ── Final test metrics ──
    model.eval()
    with torch.no_grad():
        B = params_te_t.shape[0]; n_r = y_te_s_sc_t.shape[1]
        r_rep = r_grid_t.unsqueeze(0).unsqueeze(2).expand(B, n_r, 1)
        p_rep = params_te_t.unsqueeze(1).expand(B, n_r, 5)
        X_te_data = torch.cat([p_rep, r_rep], dim=2).reshape(-1, 6)
        pred_sc = model((X_te_data - mx) / sx).cpu().numpy()
    pred = scaler.inverse(pred_sc)
    y_te_flat = y_te_s.reshape(-1, 4)
    metrics = cuml_scorer_vpinn(y_te_flat, pred, model, X_tr)

    history = {"train_losses": train_losses, "test_losses":  test_losses,
               "phys_losses":  phys_losses,  "prof_losses":  prof_losses,
               "bc_losses":    bc_losses,    "vm_rmse_test": vm_rmse_test}
    return model, metrics, mx, sx, scaler, history, y_te_flat, pred


# ─────────────────────────────────────────────────────────────────────────────
# Optuna
# ─────────────────────────────────────────────────────────────────────────────

def do_optuna_vpinn(X, y, n_trials=50, r_col_idx=-1, **kwargs):
    n_splits     = kwargs.get("n_splits", 3)
    n_layers_max = kwargs.get("n_layers", 8)
    n_neurons    = kwargs.get("n_neurons", 256)
    n_quad       = kwargs.get("n_quad", 32)
    n_test_funcs = kwargs.get("n_test_funcs", 10)
    r_extrap_fix = float(kwargs.get("r_extrap", R_EXTRAP_FREE_SURFACE))
    n_feat       = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}   (VPINN  N_q={n_quad}  K={n_test_funcs}, "
          f"extrap-BC at r_norm={r_extrap_fix})")

    (cur_X_test, cur_y_test,
     val_list_X, val_list_y,
     train_list_X, train_list_y) = _split_cv(X, y, n_splits=n_splits)

    def objective(trial):
        k_lay   = trial.suggest_int("n_layers", 3, n_layers_max)
        units   = [trial.suggest_int(f"n_units_{i}", 32, n_neurons)
                   for i in range(k_lay)]
        act     = trial.suggest_categorical("activation", list(SMOOTH_ACTS))
        lr      = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
        bs      = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        opt_nm  = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        epochs  = trial.suggest_int("max_epochs", 200, 600)
        l_phys  = trial.suggest_float("lambda_physics", 1e-4, 0.5,  log=True)
        l_bc    = trial.suggest_float("lambda_bc",      1e-3, 5.0,  log=True)
        l_prof  = trial.suggest_float("lambda_profile", 1e-3, 1.0,  log=True)
        clip    = trial.suggest_float("grad_clip", 0.5, 5.0)

        errors = np.zeros((n_splits, 11))
        for si in range(n_splits):
            X_tr, y_tr = _clean(train_list_X[si], train_list_y[si])
            X_vl, y_vl = _clean(val_list_X[si],   val_list_y[si])

            trial_params = dict(
                n_layers=k_lay, activation=act, batch_size=bs,
                learning_rate=lr, max_epochs=epochs, optimizer=opt_nm,
                lambda_physics=l_phys, lambda_bc=l_bc,
                lambda_profile=l_prof, grad_clip=clip,
                r_extrap=r_extrap_fix,
                **{f"n_units_{i}": units[i] for i in range(k_lay)},
            )

            # Short training with early stopping inside _train_single
            try:
                model, metrics, _mx, _sx, _sc, _hist, y_vl_flat, pred = \
                    _train_single(X_tr, y_tr, X_vl, y_vl, trial_params,
                                   r_col_idx, device,
                                   n_quad=n_quad, n_test_funcs=n_test_funcs,
                                   verbose_every=0, label="")
                errors[si] = metrics
            except Exception as e:
                # Guard against rare numerical instabilities in early trials
                print(f"  [trial skipped fold {si}: {type(e).__name__}: {e}]")
                errors[si] = np.array(
                    [0, 1e6, 1e6, 1e6, 0, 1e6, 1e6, 1e6, 1e6, 0, 200.0])

        w = _choose_worst(errors)
        return w[8] if pd.notnull(w[8]) else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = dict(study.best_params, r_extrap=r_extrap_fix)
    print("Best params:", best_params)
    return best_params, cur_X_test, cur_y_test, study.best_value


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold final training (backward-compatible with PINN module)
# ─────────────────────────────────────────────────────────────────────────────

def test_best_model_vpinn(X, y, best_params, r_col_idx=-1,
                           n_quad=32, n_test_funcs=10, **kwargs):
    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_extr = float(best_params.get("r_extrap", R_EXTRAP_FREE_SURFACE))
    print(f"Using device: {device}   (VPINN  N_q={n_quad}  K={n_test_funcs}, "
          f"extrap-BC at r_norm={r_extr})")

    (X_test, y_test, _, _, tr_X, tr_y) = _split_cv(X, y, n_splits=1)
    X_tr, y_tr = _clean(tr_X[0], tr_y[0])
    X_te, y_te = _clean(X_test,  y_test)

    model, metrics, mx, sx, scaler, history, y_te_ret, pred_te = \
        _train_single(X_tr, y_tr, X_te, y_te, best_params, r_col_idx, device,
                       n_quad=n_quad, n_test_funcs=n_test_funcs,
                       verbose_every=20, label="single")

    print("\n── Test Metrics ─────────────────────────────────────")
    for n, v in zip(METRIC_NAMES, metrics):
        print(f"  {n:<26}: {v:.4f}")
    per_component_report(y_te_ret, pred_te, label="Test (single fold)")

    return model, metrics, mx, sx, scaler, history


# ─────────────────────────────────────────────────────────────────────────────
# K-fold final training
# ─────────────────────────────────────────────────────────────────────────────

def kfold_test_best_model_vpinn(X, y, best_params, r_col_idx=-1,
                                 n_folds=5, random_state=42,
                                 n_quad=32, n_test_funcs=10, **kwargs):
    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_extr = float(best_params.get("r_extrap", R_EXTRAP_FREE_SURFACE))
    print(f"Using device: {device}   (VPINN  N_q={n_quad}  K={n_test_funcs}, "
          f"extrap-BC at r_norm={r_extr})")

    n_sets = len(X) // N_R
    rng     = np.random.default_rng(random_state)
    set_idx = np.arange(n_sets); rng.shuffle(set_idx)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def rows(sets):
        r = np.concatenate([np.arange(s * N_R, (s + 1) * N_R) for s in sets])
        return X[r], y[r]

    per_fold = np.zeros((n_folds, 11))
    all_models, all_histories, all_mx, all_sx, all_scalers = [], [], [], [], []
    fold_test_arrays = []

    print("\n" + "=" * 72)
    print(f"VPINN K-FOLD FINAL EVALUATION   (K={n_folds}, group-aware)")
    print("=" * 72)

    for k, (tr_i, te_i) in enumerate(kf.split(set_idx)):
        tr_sets, te_sets = set_idx[tr_i], set_idx[te_i]
        X_tr, y_tr = _clean(*rows(tr_sets))
        X_te, y_te = _clean(*rows(te_sets))
        print(f"\n─── Fold {k+1}/{n_folds} ──  train_sets={len(tr_sets)}  "
              f"test_sets={len(te_sets)}  train_rows={len(X_tr)}  "
              f"test_rows={len(X_te)} ───")

        model, metrics, mx, sx, scaler, history, y_te_ret, pred_te = \
            _train_single(X_tr, y_tr, X_te, y_te, best_params, r_col_idx, device,
                           n_quad=n_quad, n_test_funcs=n_test_funcs,
                           verbose_every=50, label=f"fold{k+1}")
        per_fold[k] = metrics
        all_models.append(model); all_histories.append(history)
        all_mx.append(mx); all_sx.append(sx); all_scalers.append(scaler)
        fold_test_arrays.append((y_te_ret, pred_te))

        print(f"\n[Fold {k+1}]  R²={metrics[4]:+.4f}   "
              f"RMSE_macro={metrics[8]:.4g}   "
              f"Von Mises R²={metrics[9]:+.4f}   SMAPE={metrics[10]:.2f}%")
        per_component_report(y_te_ret, pred_te, label=f"Fold {k+1}")

    mean_m = per_fold.mean(axis=0)
    std_m  = per_fold.std(axis=0)

    print("\n" + "=" * 72)
    print("VPINN K-FOLD AGGREGATE  (mean ± std across folds)")
    print("=" * 72)
    for n, m, s in zip(METRIC_NAMES, mean_m, std_m):
        print(f"  {n:<26}: {m:10.4g}  ±  {s:8.3g}")

    best_k = int(np.argmin(per_fold[:, 8]))
    print(f"\n► Best fold (by macro RMSE): {best_k+1}\n")

    return {
        "per_fold_metrics": per_fold,
        "mean": mean_m, "std": std_m,
        "best_fold": best_k,
        "best_model": all_models[best_k],
        "best_mean_X": all_mx[best_k],
        "best_std_X":  all_sx[best_k],
        "best_scaler_y": all_scalers[best_k],
        "best_history":  all_histories[best_k],
        "all_models": all_models,
        "all_histories": all_histories,
        "fold_test_arrays": fold_test_arrays,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference (identical to PINN — networks share the same input/output layout)
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_vpinn(model, X, mean_X, std_X, scaler_y):
    device = next(model.parameters()).device
    X_std  = (torch.FloatTensor(X).to(device) - mean_X) / std_X
    model.eval()
    with torch.no_grad():
        pred_sc = model(X_std).cpu().numpy()
    sigma = scaler_y.inverse(pred_sc)
    return {"sigma": sigma, "von_mises": von_mises_stress(sigma)}


# ─────────────────────────────────────────────────────────────────────────────
# External validation evaluator (identical interface)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_external_set(model, mean_X, std_X, scaler_y,
                              X_stress_ext: np.ndarray,
                              y_stress_ext: np.ndarray,
                              label: str = "External validation",
                              verbose: bool = True):
    if verbose:
        print(f"\n── Building dataset from external pkl ({label}) ──")
    X_ext, y_ext, r_grid, proc = build_stress_dataset(
        X_stress_ext, y_stress_ext, n_r=N_R, verbose=verbose)

    X_ext, y_ext = _clean(X_ext, y_ext)
    pred_dict = predict_with_vpinn(model, X_ext, mean_X, std_X, scaler_y)
    y_pred = pred_dict["sigma"]

    metrics = cuml_scorer_vpinn(y_ext, y_pred, model, X_ext)

    if verbose:
        print(f"\n── {label} — Metrics ───────────────────────────")
        for n, v in zip(METRIC_NAMES, metrics):
            print(f"  {n:<26}: {v:.4f}")
        per_component_report(y_ext, y_pred, label=label)

    n_sets = y_stress_ext.shape[1]
    pinn_by_set = y_pred.reshape(n_sets, N_R, 4)
    fem_by_set  = y_ext .reshape(n_sets, N_R, 4)

    return {
        "X":      X_ext,
        "y_true": y_ext,
        "y_pred": y_pred,
        "metrics": metrics,
        "per_set": {"pinn": pinn_by_set, "fem": fem_by_set},
        "proc":   proc,
        "r_grid": r_grid,
        "label":  label,
    }
