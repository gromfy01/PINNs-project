"""MLP+Optuna деформаций: подбор гиперпараметров Optuna, без физики (data-only)."""
from __future__ import annotations

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

N_R       = 20
N_OUTPUTS = 4

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


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 layer_widths: List[int],
                 activation: str = "tanh",
                 n_outputs: int = N_OUTPUTS):
        super().__init__()
        Act = ACTIVATIONS.get(activation, nn.Tanh)
        layers = []
        prev = input_dim
        for w in layer_widths:
            layers.append(nn.Linear(prev, w))
            layers.append(Act())
            prev = w
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_opt(name: str, model: nn.Module, lr: float,
              weight_decay: float = 1e-4) -> optim.Optimizer:
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def von_mises_strain(eps: np.ndarray) -> np.ndarray:
    e_rr, e_tt, e_zz, e_rz = (eps[:, i] for i in range(4))
    dev = (2.0 / 9.0) * (
        (e_rr - e_tt)**2 + (e_tt - e_zz)**2 + (e_zz - e_rr)**2
    )
    return np.sqrt(np.maximum(dev + (4.0 / 3.0) * e_rz**2, 0.0))


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


def _group_kfold_indices(n_sets: int,
                         n_folds: int = 5,
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


def _train_with_early_stopping(model: nn.Module,
                                opt: optim.Optimizer,
                                Xtr: torch.Tensor, ytr: torch.Tensor,
                                Xvl: torch.Tensor, yvl: torch.Tensor,
                                batch_size: int,
                                max_epochs: int,
                                grad_clip: float = 1.0,
                                patience: int = 30,
                                track_history: bool = False
                                ) -> Tuple[float, Dict]:
    crit   = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=batch_size, shuffle=True)

    history = {"train_losses": [], "val_losses": []} if track_history else None
    best_vl    = float("inf")
    best_state = None
    bad        = 0

    for _ in range(max_epochs):

        model.train()
        total = 0.0; n_seen = 0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item() * Xb.size(0)
            n_seen += Xb.size(0)
        train_loss = total / max(n_seen, 1)


        model.eval()
        with torch.no_grad():
            vl_loss = float(crit(model(Xvl), yvl).item())

        if track_history:
            history["train_losses"].append(train_loss)
            history["val_losses"].append(vl_loss)

        if vl_loss < best_vl:
            best_vl    = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_vl, history


def do_optuna_mlp(X: np.ndarray, y: np.ndarray,
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
            "activation", ["tanh", "relu", "selu", "gelu", "softplus"])
        lr     = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
        bs     = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        opt_nm = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        epochs = trial.suggest_int("max_epochs", 200, 600)
        clip   = trial.suggest_float("grad_clip", 0.5, 5.0)
        wd     = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        fold_rmse: List[float] = []
        for si in range(n_inner_splits):
            X_tr, y_tr = _clean(tr_X[si], tr_y[si])
            X_vl, y_vl = _clean(vl_X[si], vl_y[si])

            sc_x = ZScoreScaler().fit(X_tr)
            sc_y = ZScoreScaler().fit(y_tr)

            Xtr = torch.FloatTensor(sc_x.transform(X_tr)).to(device)
            ytr = torch.FloatTensor(sc_y.transform(y_tr)).to(device)
            Xvl = torch.FloatTensor(sc_x.transform(X_vl)).to(device)
            yvl = torch.FloatTensor(sc_y.transform(y_vl)).to(device)

            model = MLP(X_tr.shape[1], units, act).to(device)
            opt   = _make_opt(opt_nm, model, lr, weight_decay=wd)

            _train_with_early_stopping(
                model, opt, Xtr, ytr, Xvl, yvl,
                batch_size=bs, max_epochs=epochs,
                grad_clip=clip, patience=30, track_history=False,
            )

            model.eval()
            with torch.no_grad():
                pred_sc = model(Xvl).cpu().numpy()
            pred = sc_y.inverse(pred_sc)
            fold_rmse.append(float(np.mean([
                np.sqrt(mean_squared_error(y_vl[:, i], pred[:, i]))
                for i in range(4)
            ])))

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


def _train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    best_params: Dict,
                    device: torch.device,
                    early_stop_patience: int = 50,
                    verbose: bool = False,
                    log_every: int = 40) -> Dict:
    n_lay  = best_params["n_layers"]
    units  = [best_params[f"n_units_{i}"] for i in range(n_lay)]
    act    = best_params["activation"]
    lr     = best_params["learning_rate"]
    bs     = best_params["batch_size"]
    opt_nm = best_params["optimizer"]
    epochs = best_params["max_epochs"]
    clip   = best_params.get("grad_clip", 1.0)
    wd     = best_params.get("weight_decay", 1e-4)

    sc_x = ZScoreScaler().fit(X_tr)
    sc_y = ZScoreScaler().fit(y_tr)

    Xtr = torch.FloatTensor(sc_x.transform(X_tr)).to(device)
    ytr = torch.FloatTensor(sc_y.transform(y_tr)).to(device)
    Xte = torch.FloatTensor(sc_x.transform(X_te)).to(device)
    yte = torch.FloatTensor(sc_y.transform(y_te)).to(device)

    model = MLP(X_tr.shape[1], units, act).to(device)
    opt   = _make_opt(opt_nm, model, lr, weight_decay=wd)
    crit  = nn.MSELoss()

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=bs, shuffle=True)

    train_losses, test_losses, vm_rmse_hist = [], [], []
    best_test = float("inf")
    best_state = None
    bad = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(epochs):

        model.train()
        total = 0.0; n_seen = 0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item() * Xb.size(0)
            n_seen += Xb.size(0)
        train_loss = total / max(n_seen, 1)


        model.eval()
        with torch.no_grad():
            test_loss = float(crit(model(Xte), yte).item())
            pred_sc   = model(Xte).cpu().numpy()
        pred = sc_y.inverse(pred_sc)
        vm_rmse = float(np.sqrt(mean_squared_error(
            von_mises_strain(y_te), von_mises_strain(pred))))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        vm_rmse_hist.append(vm_rmse)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            r2s = [r2_score(y_te[:, i], pred[:, i]) for i in range(4)]
            print(f"    Epoch {epoch:4d}/{epochs} | "
                  f"Train {train_loss:.5f}  Test {test_loss:.5f}  "
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
        pred = sc_y.inverse(model(Xte).cpu().numpy())

    return {
        "model":    model,
        "mean_X":   sc_x.mean_,
        "std_X":    sc_x.std_,
        "scaler_y": sc_y,
        "history": {
            "train_losses": train_losses,
            "test_losses":  test_losses,
            "vm_rmse_test": vm_rmse_hist,
        },
        "y_true":      y_te,
        "y_pred":      pred,
        "metrics":     _metric_vector(y_te, pred),
        "elapsed_sec": elapsed,
        "n_epochs":    len(train_losses),
    }


def kfold_test_best_model_mlp(X: np.ndarray,
                               y: np.ndarray,
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
          f"batch_size={best_params['batch_size']} · "
          f"early stopping (patience={early_stop_patience})")

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

    per_fold_metrics_arr = np.array(per_fold_metrics)
    per_fold_times_arr   = np.array(per_fold_times)
    per_fold_epochs_arr  = np.array(per_fold_epochs)

    best_k = int(np.argmax(per_fold_metrics_arr[:, 0]))
    best   = fold_artifacts[best_k]

    return {

        "best_model":       best["model"],
        "best_mean_X":      best["mean_X"],
        "best_std_X":       best["std_X"],
        "best_scaler_y":    best["scaler_y"],
        "best_history":     best["history"],
        "best_fold":        best_k,
        "best_elapsed_sec": best["elapsed_sec"],
        "best_n_epochs":    best["n_epochs"],

        "per_fold_metrics": per_fold_metrics_arr,
        "per_fold_times":   per_fold_times_arr,
        "per_fold_epochs":  per_fold_epochs_arr,
        "fold_test_arrays": fold_test_arrays,

        "mean":             per_fold_metrics_arr.mean(axis=0),
        "std":              per_fold_metrics_arr.std(axis=0),
    }


def predict_with_mlp(model: nn.Module,
                     X: np.ndarray,
                     mean_X: np.ndarray,
                     std_X:  np.ndarray,
                     scaler_y: ZScoreScaler) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    Xs = ((X - mean_X) / std_X).astype(np.float32)
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

    res          = predict_with_mlp(model, X_val_flat, mean_X, std_X, scaler_y)
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
        "per_set": {"true": per_set_true, "mlp": per_set_pred},
        "metrics": metrics,
    }
