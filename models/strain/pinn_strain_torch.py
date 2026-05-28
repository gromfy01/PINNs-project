"""PINN деформаций (PyTorch, strong-form): R(r) = ε_rr − ε_θθ − r·∂ε_θθ/∂r."""
from __future__ import annotations

import math
import time
import dataclasses
from typing import Any, Dict, List, Tuple

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
N_INPUTS          = 6        # [p1..p5, r]
R_COL_IDX         = 5        # r — последний столбец входа

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

ACTIVATIONS = {
    "tanh":     nn.Tanh,
    "selu":     nn.SELU,
    "softplus": nn.Softplus,
    "gelu":     nn.GELU,
}


def build_strain_dataset(X_strain: np.ndarray,
                         y_strain: np.ndarray,
                         n_r: int = N_R) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    assert y_strain.shape[0] == 4
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


def _standardise_X(X_tr: np.ndarray, X_other: np.ndarray = None
                    ) -> Tuple[np.ndarray, np.ndarray | None,
                                 np.ndarray, np.ndarray, float, float]:
    mx = X_tr.mean(0, keepdims=True).astype(np.float32)
    sx = X_tr.std(0, keepdims=True).astype(np.float32)
    sx[sx < 1e-8] = 1.0
    std_r  = float(sx[0, R_COL_IDX])
    mean_r = float(mx[0, R_COL_IDX])
    Xtr_s = ((X_tr - mx) / sx).astype(np.float32)
    Xo_s  = ((X_other - mx) / sx).astype(np.float32) if X_other is not None else None
    return Xtr_s, Xo_s, mx, sx, std_r, mean_r


class PINNStrainNetwork(nn.Module):
    def __init__(self, hidden_sizes: tuple, activation: str = "tanh"):
        super().__init__()
        Act = ACTIVATIONS.get(activation, nn.Tanh)
        layers = []
        prev = N_INPUTS
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            prev = h
        layers.append(nn.Linear(prev, N_OUTPUTS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _compatibility_loss(model: nn.Module,
                         X_std: torch.Tensor,
                         std_r: float, mean_r: float,
                         y_mean: torch.Tensor, y_std: torch.Tensor
                         ) -> torch.Tensor:
    eps = 1e-6


    X_phys = X_std.clone().detach().requires_grad_(True)

    eps_sc   = model(X_phys)
    eps_phys = eps_sc * y_std + y_mean
    e_rr = eps_phys[:, 0:1]
    e_tt = eps_phys[:, 1:2]


    grad_outputs = torch.ones_like(e_tt)
    de_tt_dx = torch.autograd.grad(
        outputs=eps_sc[:, 1:2],
        inputs=X_phys,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    de_tt_dr_std = de_tt_dx[:, R_COL_IDX:R_COL_IDX + 1]


    de_tt_dr_phys = de_tt_dr_std * (y_std[:, 1:2] / (std_r + eps))

    r_phys = torch.clamp(
        X_phys[:, R_COL_IDX:R_COL_IDX + 1] * std_r + mean_r,
        min=eps,
    )

    # strong-form Saint-Venant (single-z): R(r) = ε_rr − ε_θθ − r·∂ε_θθ/∂r
    R = e_rr - e_tt - r_phys * de_tt_dr_phys


    scale = (y_std[0, 0] ** 2 + y_std[0, 1] ** 2) + eps
    return torch.mean(torch.log1p(R ** 2 / scale))


def _profile_residual_loss(pred_sc: torch.Tensor,
                            target_sc: torch.Tensor,
                            n_pts: int = N_R) -> torch.Tensor:
    N, C = pred_sc.shape
    n_sets = N // n_pts
    if n_sets == 0:
        return torch.zeros((), device=pred_sc.device)

    N_use = n_sets * n_pts
    p = pred_sc[:N_use].reshape(n_sets, n_pts, C)
    t = target_sc[:N_use].reshape(n_sets, n_pts, C)

    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    l_centred = ((p_c - t_c) ** 2).mean()

    dp = p[:, 1:, :] - p[:, :-1, :]
    dt = t[:, 1:, :] - t[:, :-1, :]
    l_grad = ((dp - dt) ** 2).mean()
    return l_centred + l_grad


@dataclasses.dataclass
class TorchPINNHandle:
    model:        Any
    hidden_sizes: tuple
    activation:   str

    def predict_z(self, x: np.ndarray) -> np.ndarray:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            out = self.model(x_t)
        return out.detach().cpu().numpy().astype(np.float32)


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
    flat_t = yt.reshape(-1); flat_p = yp.reshape(-1)
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


def _group_kfold_indices(n_sets: int, n_folds: int = 5, random_state: int = 42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [(tr, te) for tr, te in kf.split(np.arange(n_sets))]


def _rows_for_sets(X: np.ndarray, y: np.ndarray, sets: np.ndarray):
    rows = np.concatenate([np.arange(s * N_R, (s + 1) * N_R) for s in sets])
    return X[rows], y[rows]


def _clean(X: np.ndarray, y: np.ndarray):
    mask = np.isfinite(X).all(1) & np.isfinite(y).all(1)
    return X[mask], y[mask]


def _group_search_split(X: np.ndarray, y: np.ndarray,
                         n_inner_splits: int = 2,
                         test_frac: float = 0.15,
                         random_state: int = 42):
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


def _train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    best_params: Dict,
                    early_stop_patience: int = 50,
                    rng_seed: int = 42,
                    verbose: bool = False,
                    log_every: int = 40) -> Dict:
    n_lay   = best_params["n_layers"]
    units   = tuple(best_params[f"n_units_{i}"] for i in range(n_lay))
    act     = best_params["activation"]
    lr      = best_params["learning_rate"]
    bs_sets = best_params["batch_size_sets"]
    epochs  = best_params["max_epochs"]
    clip    = best_params.get("grad_clip", 1.0)
    wd      = best_params.get("weight_decay", 1e-4)
    opt_nm  = best_params["optimizer"]
    lam_p   = best_params.get("lambda_physics", 1e-3)
    lam_pr  = best_params.get("lambda_profile", 1e-2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)


    scaler_y = ZScoreScaler().fit(y_tr)
    Xtr_s, Xte_s, mx, sx, std_r, mean_r = _standardise_X(X_tr, X_te)
    y_mean = torch.as_tensor(scaler_y.mean_, dtype=torch.float32, device=device)
    y_std  = torch.as_tensor(scaler_y.std_,  dtype=torch.float32, device=device)


    n_sets_tr = X_tr.shape[0] // N_R
    n_sets_te = X_te.shape[0] // N_R
    Xtr_sets = torch.as_tensor(Xtr_s.reshape(n_sets_tr, N_R, N_INPUTS),
                                 dtype=torch.float32, device=device)
    ytr_sets = torch.as_tensor(scaler_y.transform(y_tr).reshape(
                                 n_sets_tr, N_R, N_OUTPUTS),
                                 dtype=torch.float32, device=device)
    Xte_flat = torch.as_tensor(Xte_s, dtype=torch.float32, device=device)
    yte_flat = torch.as_tensor(scaler_y.transform(y_te),
                                 dtype=torch.float32, device=device)


    model = PINNStrainNetwork(hidden_sizes=units, activation=act).to(device)

    if opt_nm == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    crit = nn.MSELoss()


    loader = DataLoader(
        TensorDataset(Xtr_sets, ytr_sets),
        batch_size=bs_sets, shuffle=True,
        generator=torch.Generator(device=device),
    )

    train_losses, test_losses = [], []
    phys_losses, prof_losses, vm_rmse_hist = [], [], []
    best_test = float("inf")
    best_state = None
    bad = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        sum_d, sum_p, sum_pr, n_seen = 0.0, 0.0, 0.0, 0

        for Xb_sets, yb_sets in loader:
            B_sets = Xb_sets.shape[0]
            Xb = Xb_sets.reshape(B_sets * N_R, N_INPUTS)
            yb = yb_sets.reshape(B_sets * N_R, N_OUTPUTS)

            opt.zero_grad()
            pred_sc = model(Xb)
            L_data = crit(pred_sc, yb)
            L_prof = _profile_residual_loss(pred_sc, yb, n_pts=N_R)
            L_pde  = _compatibility_loss(model, Xb, std_r, mean_r,
                                            y_mean, y_std)

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
            pred_phys  = scaler_y.inverse(pred_sc_te.cpu().numpy())
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
            best_test = test_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
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
        pred = scaler_y.inverse(model(Xte_flat).cpu().numpy())

    handle = TorchPINNHandle(
        model=model, hidden_sizes=units, activation=act,
    )

    return {
        "model":    handle,
        "mean_X":   mx,
        "std_X":    sx,
        "scaler_y": scaler_y,
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


def do_optuna_pinn(X: np.ndarray, y: np.ndarray,
                    n_trials: int = 60,
                    n_inner_splits: int = 2,
                    test_frac: float = 0.15,
                    max_layers: int = 6,
                    min_neurons: int = 32,
                    max_neurons: int = 256,
                    random_state: int = 42,
                    verbose: bool = True) -> Tuple[Dict, np.ndarray, np.ndarray, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
            "activation", list(ACTIVATIONS.keys()))
        lr     = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
        bs_sets = trial.suggest_categorical("batch_size_sets", [16, 32, 64, 128])
        opt_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        epochs = trial.suggest_int("max_epochs", 200, 600)
        clip   = trial.suggest_float("grad_clip", 0.5, 5.0)
        wd     = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        lam_p  = trial.suggest_float("lambda_physics", 1e-5, 1e-1, log=True)
        lam_pr = trial.suggest_float("lambda_profile", 1e-4, 1.0,  log=True)

        t0 = time.perf_counter()

        fold_rmse: List[float] = []
        for si in range(n_inner_splits):
            X_tr, y_tr = _clean(tr_X[si], tr_y[si])
            X_vl, y_vl = _clean(vl_X[si], vl_y[si])

            params = dict(
                n_layers=n_lay, activation=act,
                learning_rate=lr, batch_size_sets=bs_sets, max_epochs=epochs,
                optimizer=opt_nm, grad_clip=clip, weight_decay=wd,
                lambda_physics=lam_p, lambda_profile=lam_pr,
            )
            for i, u in enumerate(units):
                params[f"n_units_{i}"] = u

            af = _train_one_fold(
                X_tr, y_tr, X_vl, y_vl, params,
                early_stop_patience=30, verbose=False,
            )
            fold_rmse.append(af["metrics"][1])

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


def kfold_test_best_model_pinn(X: np.ndarray, y: np.ndarray,
                                 best_params: Dict,
                                 n_folds: int = 5,
                                 random_state: int = 42,
                                 early_stop_patience: int = 50,
                                 verbose: bool = True) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_lay = best_params["n_layers"]
    units = [best_params[f"n_units_{i}"] for i in range(n_lay)]
    print(f"Architecture : {n_lay} × {units} ({best_params['activation']})")
    print(f"Optimizer    : {best_params['optimizer']}(lr={best_params['learning_rate']:.4g}"
          f", wd={best_params.get('weight_decay', 1e-4):.1e})")
    print(f"Schedule     : up to {best_params['max_epochs']} epochs · "
          f"batch_size_sets={best_params['batch_size_sets']} (×{N_R} rows) · "
          f"early stopping (patience={early_stop_patience})")
    print(f"Physics loss : λ_phys={best_params.get('lambda_physics', 1e-3):.4g}  "
          f"λ_prof={best_params.get('lambda_profile', 1e-2):.4g}  "
          f"(strong-form Saint-Venant: R = ε_rr − ε_θθ − r·∂ε_θθ/∂r)")

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
            X_tr, y_tr, X_te, y_te, best_params,
            early_stop_patience=early_stop_patience,
            rng_seed=42 + k,
            verbose=verbose,
            log_every=max(1, best_params["max_epochs"] // 5),
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


def predict_with_pinn(handle: TorchPINNHandle,
                      X: np.ndarray,
                      mean_X: np.ndarray,
                      std_X:  np.ndarray,
                      scaler_y: ZScoreScaler) -> Dict[str, np.ndarray]:
    Xs = ((X - mean_X) / std_X).astype(np.float32)
    pred_sc = handle.predict_z(Xs)
    strain = scaler_y.inverse(pred_sc)
    return {"strain": strain, "von_mises": von_mises_strain(strain)}


def evaluate_on_external_set(handle: TorchPINNHandle,
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

    res          = predict_with_pinn(handle, X_val_flat, mean_X, std_X, scaler_y)
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
        "per_set": {"true": per_set_true, "pinn": per_set_pred},
        "metrics": metrics,
    }


def handle_state_dict_to_numpy(handle: TorchPINNHandle) -> Dict:
    return {k: v.detach().cpu().numpy()
            for k, v in handle.model.state_dict().items()}


def handle_from_numpy(state_dict_np: Dict, hidden_sizes: tuple,
                       activation: str) -> TorchPINNHandle:
    model = PINNStrainNetwork(hidden_sizes=tuple(hidden_sizes),
                               activation=activation)
    sd_torch = {k: torch.as_tensor(v) for k, v in state_dict_np.items()}
    model.load_state_dict(sd_torch)
    model.eval()
    return TorchPINNHandle(
        model=model, hidden_sizes=tuple(hidden_sizes), activation=activation,
    )
