"""MLP-baseline деформаций: фиксированная архитектура, vanilla Adam, без физики (data-only)."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error, max_error,
)


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

HIDDEN_LAYERS = [64, 64, 64]   # 3 скрытых слоя × 64, без подбора
ACTIVATION    = "tanh"
LEARNING_RATE = 1e-3            # vanilla Adam
EPOCHS        = 200
BATCH_SIZE    = 256


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


ACTIVATIONS = {
    "tanh":     nn.Tanh,
    "relu":     nn.ReLU,
    "selu":     nn.SELU,
    "gelu":     nn.GELU,
    "softplus": nn.Softplus,
}


class VanillaMLP(nn.Module):
    def __init__(self,
                 input_dim: int = 6,
                 layer_widths: List[int] = HIDDEN_LAYERS,
                 activation: str = ACTIVATION,
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


def _train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    device: torch.device,
                    epochs: int = EPOCHS,
                    batch_size: int = BATCH_SIZE,
                    learning_rate: float = LEARNING_RATE,
                    layer_widths: List[int] = HIDDEN_LAYERS,
                    activation: str = ACTIVATION,
                    verbose: bool = False,
                    log_every: int = 40) -> Dict:
    sc_x = ZScoreScaler().fit(X_tr)
    sc_y = ZScoreScaler().fit(y_tr)

    Xtr = torch.FloatTensor(sc_x.transform(X_tr)).to(device)
    ytr = torch.FloatTensor(sc_y.transform(y_tr)).to(device)
    Xte = torch.FloatTensor(sc_x.transform(X_te)).to(device)
    yte = torch.FloatTensor(sc_y.transform(y_te)).to(device)

    model = VanillaMLP(input_dim=X_tr.shape[1],
                       layer_widths=layer_widths,
                       activation=activation).to(device)

    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=learning_rate)

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=batch_size, shuffle=True)

    train_losses, test_losses, vm_rmse_hist = [], [], []


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
            opt.step()
            total += loss.item() * Xb.size(0)
            n_seen += Xb.size(0)
        train_loss = total / max(n_seen, 1)


        model.eval()
        with torch.no_grad():
            pred_sc = model(Xte).cpu().numpy()
            test_loss = crit(model(Xte), yte).item()
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

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0


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
    }


def kfold_test_best_model_mlp(X: np.ndarray,
                               y: np.ndarray,
                               n_folds: int = 5,
                               random_state: int = 42,
                               epochs: int = EPOCHS,
                               batch_size: int = BATCH_SIZE,
                               learning_rate: float = LEARNING_RATE,
                               layer_widths: List[int] = HIDDEN_LAYERS,
                               activation: str = ACTIVATION,
                               verbose: bool = True) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Architecture : {len(layer_widths)} × {layer_widths[0]} ({activation})")
    print(f"Optimizer    : vanilla Adam (lr={learning_rate})  — no schedule")
    print(f"Schedule     : {epochs} epochs · batch_size={batch_size} · "
          f"no early stopping")

    n_sets = len(X) // N_R
    folds  = _group_kfold_indices(n_sets, n_folds, random_state)

    per_fold_metrics: List[np.ndarray] = []
    per_fold_times:   List[float]      = []
    fold_test_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_artifacts:   List[Dict]       = []

    for k, (train_sets, test_sets) in enumerate(folds):
        print(f"\n─── Fold {k+1}/{n_folds} ───  "
              f"(train: {len(train_sets)} sets, test: {len(test_sets)} sets)")
        X_tr, y_tr = _clean(*_rows_for_sets(X, y, train_sets))
        X_te, y_te = _clean(*_rows_for_sets(X, y, test_sets))

        af = _train_one_fold(
            X_tr, y_tr, X_te, y_te, device,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate,
            layer_widths=layer_widths, activation=activation,
            verbose=verbose, log_every=max(1, epochs // 5),
        )
        per_fold_metrics.append(af["metrics"])
        per_fold_times.append(af["elapsed_sec"])
        fold_test_arrays.append((af["y_true"], af["y_pred"]))
        fold_artifacts.append(af)

        m = af["metrics"]
        print(f"  → Fold {k+1}: R²={m[0]:+.4f}  RMSE={m[1]:.6f}  "
              f"VM R²={m[2]:+.4f}  SMAPE={m[3]:.2f}%  "
              f"time={af['elapsed_sec']:.1f}s")

    per_fold_metrics_arr = np.array(per_fold_metrics)
    per_fold_times_arr   = np.array(per_fold_times)


    best_k  = int(np.argmax(per_fold_metrics_arr[:, 0]))
    best    = fold_artifacts[best_k]

    return {

        "best_model":       best["model"],
        "best_mean_X":      best["mean_X"],
        "best_std_X":       best["std_X"],
        "best_scaler_y":    best["scaler_y"],
        "best_history":     best["history"],
        "best_fold":        best_k,
        "best_elapsed_sec": best["elapsed_sec"],

        "per_fold_metrics": per_fold_metrics_arr,
        "per_fold_times":   per_fold_times_arr,
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
    # маппинг raw-plane тестового pkl -> позиция выхода модели; None = тренировочный PLANE_ORDER
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
