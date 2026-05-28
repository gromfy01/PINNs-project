"""VPINN деформаций (weak-form Saint-Venant): ∫ R(r)·v_k(r) dr, R = ε_rr − ε_θθ − r·∂ε_θθ/∂r."""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error, max_error,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


STRAIN_COMPONENTS = ["eps_rr", "eps_tt", "eps_zz", "eps_rz"]
PLANE_ORDER       = [2, 0, 1, 3]
N_R               = 20
N_OUTPUTS         = 4

METRIC_NAMES = [
    "R²",
    "RMSE (4 comp macro)",
    "Von Mises R²",
    "SMAPE [%]",
    "MAE",
    "Max Error",
    "Median AE",
    "MSE",
]


N_QUAD       = 8     # узлы Гаусса-Лежандра на [0,1]
N_TEST_FUNCS = 3     # тест-функции v_k(r) = sin(kπr), k=1..K

ACTIVATIONS = {
    "tanh":     nn.Tanh,
    "relu":     nn.ReLU,
    "selu":     nn.SELU,
    "gelu":     nn.GELU,
    "softplus": nn.Softplus,
}


def build_strain_dataset(X_strain: np.ndarray,
                         y_strain: np.ndarray,
                         n_r: int = N_R) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    assert y_strain.shape[0] == 4, (
        f"y_strain expected shape (4, N_sets, {n_r}), got {y_strain.shape}")
    assert y_strain.shape[2] == n_r, (
        f"y_strain expected last dim = {n_r}, got {y_strain.shape[2]}")

    n_sets = X_strain.shape[1]
    r_grid = np.linspace(0.0, 1.0, n_r, dtype=np.float32)
    proc   = X_strain[0].astype(np.float32)

    proc_rep = np.repeat(proc, n_r, axis=0)
    r_rep    = np.tile(r_grid, n_sets)[:, None]
    X = np.hstack([proc_rep, r_rep]).astype(np.float32)

    y = np.stack([
        y_strain[PLANE_ORDER[i]].reshape(-1) for i in range(4)
    ], axis=1).astype(np.float32)

    print(f" X : {X.shape}   [p1..p5, r]")
    print(f" y : {y.shape}   [ε_rr, ε_θθ, ε_zz, ε_rz]")
    print(f" Mapping: LE33→ε_rr | LE11→ε_θθ | LE22→ε_zz | LE12→ε_rz")
    print(f"\nДиапазоны y:")
    for i, lbl in enumerate(STRAIN_COMPONENTS):
        print(f"  {lbl:10s}: [{y[:,i].min():+10.6f}, {y[:,i].max():+10.6f}]"
              f"  mean_abs={np.abs(y[:,i]).mean():.6f}")
    return X, y, r_grid, proc


class ZScoreScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        self.mean_ = X.mean(0, keepdims=True).astype(np.float32)
        self.std_  = X.std(0, keepdims=True).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, X_sc: np.ndarray) -> np.ndarray:
        return (X_sc * self.std_ + self.mean_).astype(np.float32)


class FourierEmbedding(nn.Module):
    def __init__(self, n_freq: int = 8, sigma: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(n_freq, generator=g) * sigma
        self.register_buffer("B", B)

    def forward(self, r: torch.Tensor) -> torch.Tensor:

        proj = 2.0 * math.pi * r * self.B.unsqueeze(0)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class OutputScaler(nn.Module):
    def __init__(self, n_components: int = 4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_components))
        self.shift = nn.Parameter(torch.zeros(n_components))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class StrainVPINN(nn.Module):
    def __init__(self,
                 layer_widths: List[int],
                 activation: str = "tanh",
                 n_freq: int = 8,
                 use_fourier: bool = True,
                 r_idx: int = -1,
                 mean_X: torch.Tensor = None,
                 std_X:  torch.Tensor = None):
        super().__init__()
        self.r_idx       = r_idx
        self.use_fourier = use_fourier
        self.r_embed     = FourierEmbedding(n_freq=n_freq) if use_fourier else None


        in_dim = 5 + (2 * n_freq if use_fourier else 1)

        Act = ACTIVATIONS.get(activation, nn.Tanh)
        layers = []
        prev = in_dim
        for w in layer_widths:
            layers.append(nn.Linear(prev, w))
            layers.append(Act())
            prev = w
        layers.append(nn.Linear(prev, N_OUTPUTS))
        self.net = nn.Sequential(*layers)
        self.scaler_out = OutputScaler(n_components=N_OUTPUTS)


        if mean_X is not None and std_X is not None:
            self.register_buffer("mean_X_buf", mean_X.float())
            self.register_buffer("std_X_buf",  std_X.float())
        else:
            self.mean_X_buf = None
            self.std_X_buf  = None

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


        proc = x[:, :5]
        r    = x[:, 5:6]
        return proc, r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proc, r = self._split(x)
        if self.use_fourier:
            r_feat = self.r_embed(r)
            inp = torch.cat([proc, r_feat], dim=-1)
        else:
            inp = torch.cat([proc, r], dim=-1)
        out = self.net(inp)
        return self.scaler_out(out)


def _gl_nodes_weights_01(n: int, device: torch.device,
                          dtype: torch.dtype = torch.float32
                          ) -> Tuple[torch.Tensor, torch.Tensor]:

    nodes_m11, w_m11 = np.polynomial.legendre.leggauss(n)
    nodes_01 = 0.5 * (nodes_m11 + 1.0)
    w_01     = 0.5 * w_m11
    return (torch.as_tensor(nodes_01, dtype=dtype, device=device),
            torch.as_tensor(w_01,     dtype=dtype, device=device))


def weak_compat_loss(model: StrainVPINN,
                     proc_zscored: torch.Tensor,
                     scaler_y: ZScoreScaler,
                     gl_r: torch.Tensor,
                     gl_w: torch.Tensor,
                     n_test: int) -> torch.Tensor:
    device = proc_zscored.device
    B      = proc_zscored.shape[0]
    Q      = gl_r.shape[0]


    r_grid = gl_r.view(1, Q, 1).expand(B, Q, 1).clone()
    r_grid.requires_grad_(True)
    proc_grid = proc_zscored.view(B, 1, 5).expand(B, Q, 5)
    x_grid = torch.cat([proc_grid, r_grid], dim=-1)


    pred_sc = model(x_grid.view(B * Q, 6)).view(B, Q, 4)


    y_mean = torch.as_tensor(scaler_y.mean_, dtype=pred_sc.dtype, device=device)
    y_std  = torch.as_tensor(scaler_y.std_,  dtype=pred_sc.dtype, device=device)
    pred_phys = pred_sc * y_std + y_mean

    eps_rr = pred_phys[..., 0]
    eps_tt = pred_phys[..., 1]


    grad_outputs = torch.ones_like(eps_tt)
    deps_tt_dr = torch.autograd.grad(
        outputs=eps_tt, inputs=r_grid,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
    )[0].squeeze(-1)

    r_q = gl_r.view(1, Q)
    # strong-form Saint-Venant residual в каждом узле квадратуры
    R   = eps_rr - eps_tt - r_q * deps_tt_dr


    pi    = math.pi
    ks    = torch.arange(1, n_test + 1, dtype=R.dtype, device=device)
    # слабая форма: проекция R на тест-функции v_k(r)=sin(kπr) через GL-квадратуру
    test  = torch.sin(ks.view(n_test, 1) * pi * r_q)
    w     = gl_w.view(1, Q)
    Rk    = (R.unsqueeze(1) * test.unsqueeze(0) * w.unsqueeze(0)).sum(-1)

    return (Rk ** 2).sum(-1).mean()


def profile_loss(pred_sc: torch.Tensor,
                  targ_sc: torch.Tensor,
                  n_r: int = N_R) -> torch.Tensor:
    B = pred_sc.shape[0] // n_r
    p = pred_sc.view(B, n_r, -1)
    t = targ_sc.view(B, n_r, -1)

    pc = p - p.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    L_shape = ((pc - tc) ** 2).mean()

    dp = p[:, 1:, :] - p[:, :-1, :]
    dt = t[:, 1:, :] - t[:, :-1, :]
    L_grad = ((dp - dt) ** 2).mean()
    return L_shape + L_grad


def von_mises_strain(eps: np.ndarray) -> np.ndarray:
    e_rr, e_tt, e_zz, e_rz = (eps[:, i] for i in range(4))
    dev = (2.0 / 9.0) * (
        (e_rr - e_tt) ** 2 + (e_tt - e_zz) ** 2 + (e_zz - e_rr) ** 2
    )
    return np.sqrt(np.maximum(dev + (4.0 / 3.0) * e_rz ** 2, 0.0))


def smape(y_true, y_pred, eps: float = 1e-12) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.abs(yt) + np.abs(yp) + eps
    return float(200.0 * np.mean(np.abs(yp - yt) / denom))


def _metric_vector(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    flat_t = yt.reshape(-1)
    flat_p = yp.reshape(-1)

    rmse_macro = float(np.mean([
        np.sqrt(mean_squared_error(yt[:, i], yp[:, i])) for i in range(4)
    ]))
    vm_t = von_mises_strain(yt)
    vm_p = von_mises_strain(yp)

    return np.array([
        r2_score(flat_t, flat_p),
        rmse_macro,
        r2_score(vm_t, vm_p),
        smape(flat_t, flat_p),
        mean_absolute_error(flat_t, flat_p),
        max_error(flat_t, flat_p),
        median_absolute_error(flat_t, flat_p),
        mean_squared_error(flat_t, flat_p),
    ])


def _group_kfold_indices(n_sets: int, n_folds: int = 5,
                          random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [(tr, te) for tr, te in kf.split(np.arange(n_sets))]


def _rows_for_sets(X: np.ndarray, y: np.ndarray, sets: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate([np.arange(s * N_R, (s + 1) * N_R) for s in sets])
    return X[rows], y[rows]


def _clean(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(X).all(1) & np.isfinite(y).all(1)
    return X[mask], y[mask]


def _group_search_split(X: np.ndarray, y: np.ndarray,
                         n_inner_splits: int = 2,
                         test_frac: float = 0.15,
                         random_state: int = 42) -> Tuple:
    n_sets = len(X) // N_R
    rng    = np.random.default_rng(random_state)
    all_sets    = np.arange(n_sets)
    n_test_sets = max(1, int(round(n_sets * test_frac)))
    test_sets   = rng.choice(all_sets, size=n_test_sets, replace=False)
    train_sets  = np.setdiff1d(all_sets, test_sets)

    X_test, y_test = _rows_for_sets(X, y, test_sets)

    if n_inner_splits == 1:
        n_val = max(1, int(round(len(train_sets) * 0.15)))
        val_sets = rng.choice(train_sets, size=n_val, replace=False)
        tr_sets  = np.setdiff1d(train_sets, val_sets)
        Xtr, ytr = _rows_for_sets(X, y, tr_sets)
        Xvl, yvl = _rows_for_sets(X, y, val_sets)
        return X_test, y_test, [Xvl], [yvl], [Xtr], [ytr]

    kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
    tr_X, tr_y, vl_X, vl_y = [], [], [], []
    for ti, vi in kf.split(train_sets):
        Xtr, ytr = _rows_for_sets(X, y, train_sets[ti])
        Xvl, yvl = _rows_for_sets(X, y, train_sets[vi])
        tr_X.append(Xtr); tr_y.append(ytr)
        vl_X.append(Xvl); vl_y.append(yvl)
    return X_test, y_test, vl_X, vl_y, tr_X, tr_y


class XStrainScaler:
    def __init__(self):
        self.mean_p = None
        self.std_p  = None

    def fit(self, X: np.ndarray) -> "XStrainScaler":
        self.mean_p = X[:, :5].mean(0, keepdims=True).astype(np.float32)
        self.std_p  = X[:, :5].std(0, keepdims=True).astype(np.float32)
        self.std_p[self.std_p < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X.copy().astype(np.float32)
        out[:, :5] = (out[:, :5] - self.mean_p) / self.std_p
        return out

    @property
    def mean_(self) -> np.ndarray:

        m = np.zeros((1, 6), dtype=np.float32)
        m[:, :5] = self.mean_p
        return m

    @property
    def std_(self) -> np.ndarray:
        s = np.ones((1, 6), dtype=np.float32)
        s[:, :5] = self.std_p
        return s


def _train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    best_params: Dict,
                    device: torch.device,
                    n_quad: int = N_QUAD,
                    n_test_funcs: int = N_TEST_FUNCS,
                    early_stop_patience: int = 50,
                    verbose: bool = False,
                    log_every: int = 40) -> Dict:
    n_lay  = best_params["n_layers"]
    units  = [best_params[f"n_units_{i}"] for i in range(n_lay)]
    act    = best_params["activation"]
    n_freq = best_params.get("n_freq", 8)
    lr     = best_params["learning_rate"]
    bs_sets = best_params["batch_size_sets"]
    epochs = best_params["max_epochs"]
    clip   = best_params.get("grad_clip", 1.0)
    wd     = best_params.get("weight_decay", 1e-4)
    opt_nm = best_params["optimizer"]
    lam_p  = best_params.get("lambda_pde",  1e-3)
    lam_pr = best_params.get("lambda_prof", 1e-2)

    sc_x = XStrainScaler().fit(X_tr)
    sc_y = ZScoreScaler().fit(y_tr)


    n_sets_tr = X_tr.shape[0] // N_R
    n_sets_te = X_te.shape[0] // N_R
    Xtr = torch.FloatTensor(sc_x.transform(X_tr)).view(n_sets_tr, N_R, -1).to(device)
    ytr = torch.FloatTensor(sc_y.transform(y_tr)).view(n_sets_tr, N_R, -1).to(device)
    Xte = torch.FloatTensor(sc_x.transform(X_te)).view(n_sets_te, N_R, -1).to(device)
    yte_full = torch.FloatTensor(sc_y.transform(y_te)).to(device)
    Xte_flat = Xte.view(-1, Xte.shape[-1])
    yte_flat = yte_full

    model = StrainVPINN(units, activation=act, n_freq=n_freq,
                         use_fourier=True).to(device)

    if opt_nm == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    crit  = nn.MSELoss()

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=bs_sets, shuffle=True)


    gl_r, gl_w = _gl_nodes_weights_01(n_quad, device)

    train_losses, test_losses = [], []
    phys_losses, prof_losses, vm_rmse_hist = [], [], []
    best_test  = float("inf")
    best_state = None
    bad        = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(epochs):

        model.train()
        sum_d, sum_p, sum_pr, n_seen = 0.0, 0.0, 0.0, 0
        for Xb_sets, yb_sets in loader:

            B_sets = Xb_sets.shape[0]
            Xb = Xb_sets.reshape(B_sets * N_R, -1)
            yb = yb_sets.reshape(B_sets * N_R, -1)

            opt.zero_grad()
            pred_sc = model(Xb)
            L_data  = crit(pred_sc, yb)
            L_prof  = profile_loss(pred_sc, yb, n_r=N_R)


            proc_b = Xb_sets[:, 0, :5]
            L_pde  = weak_compat_loss(model, proc_b, sc_y,
                                        gl_r, gl_w, n_test_funcs)

            loss = L_data + lam_p * L_pde + lam_pr * L_prof
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            B = Xb.shape[0]
            sum_d  += L_data.item() * B
            sum_p  += L_pde.item()  * B
            sum_pr += L_prof.item() * B
            n_seen += B
        train_data_loss = sum_d  / max(n_seen, 1)
        train_pde_loss  = sum_p  / max(n_seen, 1)
        train_prof_loss = sum_pr / max(n_seen, 1)


        model.eval()
        with torch.no_grad():
            pred_sc_te = model(Xte_flat)
            test_loss  = float(crit(pred_sc_te, yte_flat).item())
            pred_phys  = sc_y.inverse(pred_sc_te.cpu().numpy())
        vm_rmse = float(np.sqrt(mean_squared_error(
            von_mises_strain(y_te), von_mises_strain(pred_phys))))

        train_losses.append(train_data_loss)
        test_losses.append(test_loss)
        phys_losses.append(max(train_pde_loss, 1e-30))
        prof_losses.append(max(train_prof_loss, 1e-30))
        vm_rmse_hist.append(vm_rmse)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            r2s = [r2_score(y_te[:, i], pred_phys[:, i]) for i in range(4)]
            print(f"    Epoch {epoch:4d}/{epochs} | "
                  f"Data {train_data_loss:.5f}  PDE {train_pde_loss:.3e}  "
                  f"Prof {train_prof_loss:.3e}  Test {test_loss:.5f}  "
                  f"VM_RMSE {vm_rmse:.5f} | "
                  f"R²=[{r2s[0]:+.3f},{r2s[1]:+.3f},"
                  f"{r2s[2]:+.3f},{r2s[3]:+.3f}]")

        if test_loss < best_test:
            best_test  = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= early_stop_patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}")
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = sc_y.inverse(model(Xte_flat).cpu().numpy())

    return {
        "model":    model,
        "mean_X":   sc_x.mean_,
        "std_X":    sc_x.std_,
        "scaler_y": sc_y,
        "history": {
            "train_losses": train_losses,
            "test_losses":  test_losses,
            "phys_losses":  phys_losses,
            "prof_losses":  prof_losses,
            "vm_rmse_test": vm_rmse_hist,
        },
        "y_true":      y_te,
        "y_pred":      pred,
        "metrics":     _metric_vector(y_te, pred),
        "elapsed_sec": elapsed,
        "n_epochs":    len(train_losses),
    }


def do_optuna_vpinn(X: np.ndarray, y: np.ndarray,
                    n_trials: int = 60,
                    n_inner_splits: int = 2,
                    test_frac: float = 0.15,
                    max_layers: int = 6,
                    min_neurons: int = 32,
                    max_neurons: int = 256,
                    n_quad: int = N_QUAD,
                    n_test_funcs: int = N_TEST_FUNCS,
                    random_state: int = 42,
                    verbose: bool = True) -> Tuple[Dict, np.ndarray, np.ndarray, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Weak-form params: N_quad={n_quad}, K_test={n_test_funcs}")

    (X_test, y_test,
     vl_X, vl_y, tr_X, tr_y) = _group_search_split(
        X, y, n_inner_splits=n_inner_splits, test_frac=test_frac,
        random_state=random_state,
    )

    trial_times: List[float] = []

    def objective(trial: optuna.Trial) -> float:
        n_lay  = trial.suggest_int("n_layers", 2, max_layers)
        units  = [trial.suggest_int(f"n_units_{i}", min_neurons, max_neurons)
                  for i in range(n_lay)]
        act    = trial.suggest_categorical(
            "activation", ["tanh", "relu", "selu", "gelu", "softplus"])
        n_freq = trial.suggest_categorical("n_freq", [4, 8, 16])
        lr     = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
        bs_sets = trial.suggest_categorical("batch_size_sets", [16, 32, 64, 128])
        opt_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        epochs = trial.suggest_int("max_epochs", 200, 600)
        clip   = trial.suggest_float("grad_clip", 0.5, 5.0)
        wd     = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        lam_p  = trial.suggest_float("lambda_pde",  1e-5, 1e-1, log=True)
        lam_pr = trial.suggest_float("lambda_prof", 1e-4, 1.0,  log=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        fold_rmse: List[float] = []
        for si in range(n_inner_splits):
            X_tr, y_tr = _clean(tr_X[si], tr_y[si])
            X_vl, y_vl = _clean(vl_X[si], vl_y[si])

            params = dict(
                n_layers=n_lay, activation=act, n_freq=n_freq,
                learning_rate=lr, batch_size_sets=bs_sets, max_epochs=epochs,
                optimizer=opt_nm, grad_clip=clip, weight_decay=wd,
                lambda_pde=lam_p, lambda_prof=lam_pr,
            )
            for i, u in enumerate(units):
                params[f"n_units_{i}"] = u

            af = _train_one_fold(
                X_tr, y_tr, X_vl, y_vl, params, device,
                n_quad=n_quad, n_test_funcs=n_test_funcs,
                early_stop_patience=30, verbose=False,
            )
            fold_rmse.append(af["metrics"][1])

        if device.type == "cuda":
            torch.cuda.synchronize()
        trial_times.append(time.perf_counter() - t0)
        return max(fold_rmse)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1,
                    show_progress_bar=verbose)

    print(f"\nOptuna завершено за {sum(trial_times):.1f} с  "
          f"({np.mean(trial_times):.1f} ± {np.std(trial_times):.1f} с / trial)")
    print(f"best_value (inner macro RMSE) : {study.best_value:.6f}")
    print(f"best_params:")
    for k, v in study.best_params.items():
        print(f"   {k:18s} = {v}")
    return study.best_params, X_test, y_test, study.best_value


def kfold_test_best_model_vpinn(X: np.ndarray,
                                 y: np.ndarray,
                                 best_params: Dict,
                                 n_folds: int = 5,
                                 random_state: int = 42,
                                 n_quad: int = N_QUAD,
                                 n_test_funcs: int = N_TEST_FUNCS,
                                 early_stop_patience: int = 50,
                                 verbose: bool = True) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_lay = best_params["n_layers"]
    units = [best_params[f"n_units_{i}"] for i in range(n_lay)]
    print(f"Architecture : {n_lay} × {units} ({best_params['activation']}, "
          f"n_freq={best_params.get('n_freq', 8)})")
    print(f"Optimizer    : {best_params['optimizer']}(lr={best_params['learning_rate']:.4g}"
          f", wd={best_params.get('weight_decay', 1e-4):.1e})")
    print(f"Schedule     : up to {best_params['max_epochs']} epochs · "
          f"batch_size_sets={best_params['batch_size_sets']} (×{N_R} rows) · "
          f"early stopping (patience={early_stop_patience})")
    print(f"Physics loss : λ_pde={best_params.get('lambda_pde', 1e-3):.4g}  "
          f"λ_prof={best_params.get('lambda_prof', 1e-2):.4g}  "
          f"(weak-form Saint-Venant, GL N_quad={n_quad}, K_test={n_test_funcs})")

    n_sets = len(X) // N_R
    folds  = _group_kfold_indices(n_sets, n_folds, random_state)

    per_fold_metrics: List[np.ndarray] = []
    per_fold_times:   List[float]      = []
    per_fold_epochs:  List[int]        = []
    fold_test_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_artifacts:   List[Dict]       = []

    for k, (train_sets, test_sets) in enumerate(folds):
        print(f"\n─── Fold {k+1}/{n_folds} ───  "
              f"(train: {len(train_sets)} sets, test: {len(test_sets)} sets)")
        X_tr, y_tr = _clean(*_rows_for_sets(X, y, train_sets))
        X_te, y_te = _clean(*_rows_for_sets(X, y, test_sets))

        af = _train_one_fold(
            X_tr, y_tr, X_te, y_te, best_params, device,
            n_quad=n_quad, n_test_funcs=n_test_funcs,
            early_stop_patience=early_stop_patience,
            verbose=verbose, log_every=max(1, best_params["max_epochs"] // 5),
        )
        per_fold_metrics.append(af["metrics"])
        per_fold_times.append(af["elapsed_sec"])
        per_fold_epochs.append(af["n_epochs"])
        fold_test_arrays.append((af["y_true"], af["y_pred"]))
        fold_artifacts.append(af)

        m = af["metrics"]
        print(f"  → Fold {k+1}: R²={m[0]:+.4f}  RMSE={m[1]:.6f}  "
              f"VM R²={m[2]:+.4f}  SMAPE={m[3]:.2f}%  "
              f"epochs={af['n_epochs']}  time={af['elapsed_sec']:.1f}s")

    pf  = np.array(per_fold_metrics)
    pt  = np.array(per_fold_times)
    pe  = np.array(per_fold_epochs)
    bk  = int(np.argmax(pf[:, 0]))
    best = fold_artifacts[bk]

    return {
        "best_model":       best["model"],
        "best_mean_X":      best["mean_X"],
        "best_std_X":       best["std_X"],
        "best_scaler_y":    best["scaler_y"],
        "best_history":     best["history"],
        "best_fold":        bk,
        "best_elapsed_sec": best["elapsed_sec"],
        "best_n_epochs":    best["n_epochs"],
        "per_fold_metrics": pf,
        "per_fold_times":   pt,
        "per_fold_epochs":  pe,
        "fold_test_arrays": fold_test_arrays,
        "mean":             pf.mean(axis=0),
        "std":              pf.std(axis=0),
    }


def predict_with_vpinn(model: nn.Module,
                       X: np.ndarray,
                       mean_X: np.ndarray,
                       std_X:  np.ndarray,
                       scaler_y: ZScoreScaler) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    Xs = X.copy().astype(np.float32)

    Xs[:, :5] = (Xs[:, :5] - mean_X[:, :5]) / std_X[:, :5]

    Xs_t = torch.FloatTensor(Xs).to(device)
    model.eval()
    with torch.no_grad():
        pred_sc = model(Xs_t).cpu().numpy()
    strain = scaler_y.inverse(pred_sc)
    return {"strain": strain, "von_mises": von_mises_strain(strain)}


def evaluate_on_external_set(model: nn.Module,
                              mean_X: np.ndarray,
                              std_X:  np.ndarray,
                              scaler_y: ZScoreScaler,
                              X_strain_val: np.ndarray,
                              y_strain_val: np.ndarray,
                              n_r: int = N_R,
                              label: str = "External",
                              plane_order: List[int] = None,
                              verbose: bool = True) -> Dict:
    if plane_order is None:
        plane_order = PLANE_ORDER

    n_sets  = X_strain_val.shape[1]
    r_grid  = np.linspace(0.0, 1.0, n_r, dtype=np.float32)
    proc    = X_strain_val[0].astype(np.float32)

    proc_rep   = np.repeat(proc, n_r, axis=0)
    r_rep      = np.tile(r_grid, n_sets)[:, None]
    X_val_flat = np.hstack([proc_rep, r_rep]).astype(np.float32)

    y_val_flat = np.stack([
        y_strain_val[plane_order[i]].reshape(-1) for i in range(4)
    ], axis=1).astype(np.float32)

    res          = predict_with_vpinn(model, X_val_flat, mean_X, std_X, scaler_y)
    y_pred_flat  = res["strain"]
    metrics      = _metric_vector(y_val_flat, y_pred_flat)

    per_set_pred = y_pred_flat.reshape(n_sets, n_r, 4)
    per_set_true = y_val_flat.reshape(n_sets, n_r, 4)

    if verbose:
        print(f"\n── {label} metrics ─────────────────────────")
        for n, v in zip(METRIC_NAMES, metrics):
            print(f"  {n:<24}: {v:.6f}")
        print(f"\n── {label} per-component ──────────────────")
        for i, c in enumerate(STRAIN_COMPONENTS):
            yi  = y_val_flat[:, i]
            pi  = y_pred_flat[:, i]
            ri  = float(np.sqrt(mean_squared_error(yi, pi)))
            r2i = r2_score(yi, pi)
            si  = smape(yi, pi)
            print(f"  {c:10s}:  R²={r2i:+.4f}   RMSE={ri:.6f}   SMAPE={si:.2f}%")

    return {
        "y_true":  y_val_flat,
        "y_pred":  y_pred_flat,
        "proc":    proc,
        "r_grid":  r_grid,
        "per_set": {"true": per_set_true, "vpinn": per_set_pred},
        "metrics": metrics,
    }
