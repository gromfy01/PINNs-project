"""PINN (DeepXDE) для накопленных деформаций: strong-form Saint-Venant
compatibility R(r) = ε_rr − ε_θθ − r·∂ε_θθ/∂r, осесимметрия, single z-plane."""
from __future__ import annotations


import os


# backend выставляем ДО импорта deepxde
os.environ.setdefault("DDE_BACKEND", "pytorch")


import time
import dataclasses
from typing import Any, Dict, List, Tuple


import numpy as np
import optuna
import torch
import deepxde as dde
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
N_INPUTS          = 6
R_COL_IDX         = 5


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


ACT_CHOICES = ["tanh", "selu", "swish", "silu"]


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


@dataclasses.dataclass
class DDEPINNHandle:
    net:          Any
    layer_sizes:  list
    activation:   str
    initializer:  str

    def predict_z(self, x: np.ndarray) -> np.ndarray:
        device = next(self.net.parameters()).device
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            out = self.net(x_t)
        return out.detach().cpu().numpy().astype(np.float32)


def _train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    best_params: Dict,
                    early_stop_patience: int = 50,
                    rng_seed: int = 42,
                    verbose: bool = False,
                    log_every: int = 1000) -> Dict:
    n_lay  = best_params["n_layers"]
    units  = [best_params[f"n_units_{i}"] for i in range(n_lay)]
    act    = best_params["activation"]
    lr     = best_params["learning_rate"]
    iters  = best_params["adam_iters"]
    use_lbfgs = bool(best_params.get("use_lbfgs", False))
    lbfgs_iters = int(best_params.get("lbfgs_iters", 0)) if use_lbfgs else 0
    lam_p  = float(best_params.get("lambda_physics", 1e-3))
    lam_d  = float(best_params.get("lambda_data",    1.0))
    n_anchors = int(best_params.get("n_anchors", 4096))
    init_n = best_params.get("initializer", "Glorot normal")

    dde.config.set_random_seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    scaler_y = ZScoreScaler().fit(y_tr)
    Xtr_s, Xte_s, mx, sx, std_r, mean_r = _standardise_X(X_tr, X_te)

    y_mean = torch.as_tensor(scaler_y.mean_.astype(np.float32))
    y_std  = torch.as_tensor(scaler_y.std_.astype(np.float32))

    lo = Xtr_s.min(axis=0) - 0.05
    hi = Xtr_s.max(axis=0) + 0.05
    geom = dde.geometry.Hypercube(lo.tolist(), hi.tolist())

    rng = np.random.default_rng(rng_seed)
    n_anchors_eff = min(n_anchors, Xtr_s.shape[0])
    anchor_idx = rng.choice(Xtr_s.shape[0], size=n_anchors_eff, replace=False)
    anchors = Xtr_s[anchor_idx].astype(np.float32)

    eps = 1e-6

    def pde(x, y):
        # residual в физических единицах: R = ε_rr − ε_θθ − r·∂ε_θθ/∂r
        de_tt_dr_std = dde.grad.jacobian(y, x, i=1, j=R_COL_IDX)

        y_mean_t = y_mean.to(y.device)
        y_std_t  = y_std.to(y.device)

        e_rr_phys   = y[:, 0:1] * y_std_t[:, 0:1] + y_mean_t[:, 0:1]
        e_tt_phys   = y[:, 1:2] * y_std_t[:, 1:2] + y_mean_t[:, 1:2]
        de_tt_dr_phys = de_tt_dr_std * (y_std_t[:, 1:2] / (std_r + eps))

        r_z   = x[:, R_COL_IDX:R_COL_IDX + 1]
        r_phys = torch.clamp(r_z * std_r + mean_r, min=eps)

        R = e_rr_phys - e_tt_phys - r_phys * de_tt_dr_phys
        return [R]

    y_tr_sc = scaler_y.transform(y_tr)
    bcs = [
        dde.icbc.PointSetBC(Xtr_s, y_tr_sc[:, i:i + 1], component=i)
        for i in range(N_OUTPUTS)
    ]

    y_te_sc = scaler_y.transform(y_te)

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=0,
        anchors=anchors,
        num_test=None,
    )

    layer_sizes = [N_INPUTS] + units + [N_OUTPUTS]
    net = dde.nn.FNN(layer_sizes, act, init_n)

    model = dde.Model(data, net)

    loss_weights = [lam_p] + [lam_d] * N_OUTPUTS

    model.compile("adam", lr=lr, loss="MSE", loss_weights=loss_weights)

    history = {
        "train_losses": [],
        "test_losses":  [],
        "phys_losses":  [],
        "data_losses":  [],
        "vm_rmse_test": [],
    }

    state = {"best_test": float("inf"), "best_state": None,
             "bad": 0, "stopped_at": None}

    Xte_t = torch.as_tensor(Xte_s, dtype=torch.float32)

    class _MonitorCallback(dde.callbacks.Callback):
        def __init__(self, every: int):
            super().__init__()
            self.every = every

        def on_epoch_end(self):
            it = self.model.train_state.step
            if it % self.every != 0 and it != iters:
                return

            tr_losses = self.model.train_state.loss_train
            te_losses = self.model.train_state.loss_test
            phys = float(tr_losses[0]) if len(tr_losses) > 0 else 0.0
            data = float(np.sum(tr_losses[1:]))
            history["phys_losses"].append(phys)
            history["data_losses"].append(data)
            history["train_losses"].append(phys + data)

            with torch.no_grad():
                device = next(self.model.net.parameters()).device
                pred_sc = self.model.net(Xte_t.to(device)).cpu().numpy()
            test_mse = float(np.mean((pred_sc - y_te_sc) ** 2))
            history["test_losses"].append(test_mse)

            pred_phys = scaler_y.inverse(pred_sc)
            vm_rmse = float(np.sqrt(mean_squared_error(
                von_mises_strain(y_te), von_mises_strain(pred_phys))))
            history["vm_rmse_test"].append(vm_rmse)

            if verbose:
                r2s = [r2_score(y_te[:, i], pred_phys[:, i]) for i in range(4)]
                print(f"    Iter {it:6d}/{iters} | "
                      f"PDE {phys:.3e}  Data {data:.5f}  Test {test_mse:.5f}  "
                      f"VM_RMSE {vm_rmse:.5f} | "
                      f"R²=[{r2s[0]:+.3f},{r2s[1]:+.3f},"
                      f"{r2s[2]:+.3f},{r2s[3]:+.3f}]")

            if test_mse < state["best_test"]:
                state["best_test"] = test_mse

                state["best_state"] = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.net.state_dict().items()
                }
                state["bad"] = 0
            else:
                state["bad"] += 1
                if state["bad"] >= early_stop_patience:
                    if verbose:
                        print(f"    Early stop at iter {it}")
                    state["stopped_at"] = it
                    self.model.stop_training = True

    monitor = _MonitorCallback(every=log_every)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.train(iterations=iters, display_every=10**12,
                callbacks=[monitor], verbose=0)

    if use_lbfgs and lbfgs_iters > 0 and not state.get("stopped_at"):
        try:
            dde.optimizers.config.set_LBFGS_options(
                maxcor=50, ftol=1e-9, gtol=1e-8,
                maxiter=lbfgs_iters, maxfun=lbfgs_iters,
                maxls=50,
            )
            model.compile("L-BFGS", loss="MSE", loss_weights=loss_weights)
            model.train(callbacks=[monitor], verbose=0)
        except Exception as e:
            if verbose:
                print(f"    L-BFGS phase failed ({e}); falling back to Adam-only.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if state["best_state"] is not None:
        net.load_state_dict(state["best_state"])

    net.eval()
    with torch.no_grad():
        device = next(net.parameters()).device
        pred_sc = net(Xte_t.to(device)).cpu().numpy()
    pred = scaler_y.inverse(pred_sc)

    handle = DDEPINNHandle(
        net=net,
        layer_sizes=list(layer_sizes),
        activation=act,
        initializer=init_n,
    )

    n_log_iters = len(history["train_losses"])

    return {
        "model":       handle,
        "mean_X":      mx,
        "std_X":       sx,
        "scaler_y":    scaler_y,
        "history":     history,
        "y_true":      y_te,
        "y_pred":      pred,
        "metrics":     _metric_vector(y_te, pred),
        "elapsed_sec": elapsed,
        "n_epochs":    n_log_iters,
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
    print(f"DeepXDE backend  : {dde.backend.backend_name}")
    print(f"PyTorch device   : {'cuda' if torch.cuda.is_available() else 'cpu'}")

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
        act    = trial.suggest_categorical("activation", ACT_CHOICES)
        init_n = trial.suggest_categorical(
            "initializer", ["Glorot normal", "Glorot uniform", "He normal"])
        lr     = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
        iters  = trial.suggest_int("adam_iters", 4_000, 20_000, step=1000)
        use_lbfgs = trial.suggest_categorical("use_lbfgs", [False, True])
        lbfgs_iters = (trial.suggest_int("lbfgs_iters", 200, 1500, step=100)
                        if use_lbfgs else 0)
        lam_p  = trial.suggest_float("lambda_physics", 1e-5, 1e-1, log=True)
        lam_d  = trial.suggest_float("lambda_data",    0.5,  10.0, log=True)
        n_anch = trial.suggest_categorical("n_anchors", [1024, 2048, 4096, 8192])

        t0 = time.perf_counter()

        fold_rmse: List[float] = []
        for si in range(n_inner_splits):
            X_tr, y_tr = _clean(tr_X[si], tr_y[si])
            X_vl, y_vl = _clean(vl_X[si], vl_y[si])

            params = dict(
                n_layers=n_lay, activation=act, initializer=init_n,
                learning_rate=lr, adam_iters=iters,
                use_lbfgs=use_lbfgs, lbfgs_iters=lbfgs_iters,
                lambda_physics=lam_p, lambda_data=lam_d,
                n_anchors=n_anch,
            )
            for i, u in enumerate(units):
                params[f"n_units_{i}"] = u

            af = _train_one_fold(
                X_tr, y_tr, X_vl, y_vl, params,
                early_stop_patience=10,
                rng_seed=42 + si,
                verbose=False,
                log_every=max(1, iters // 20),
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
    print(f"DeepXDE backend  : {dde.backend.backend_name}")
    print(f"PyTorch device   : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    n_lay = best_params["n_layers"]
    units = [best_params[f"n_units_{i}"] for i in range(n_lay)]
    print(f"Architecture : {n_lay} × {units} ({best_params['activation']}, "
          f"init={best_params.get('initializer', 'Glorot normal')})")
    print(f"Optimizer    : Adam(lr={best_params['learning_rate']:.4g})  "
          f"+ L-BFGS({best_params.get('lbfgs_iters', 0)} iters) "
          f"if use_lbfgs={best_params.get('use_lbfgs', False)}")
    print(f"Schedule     : {best_params['adam_iters']} Adam iterations · "
          f"early stopping (patience={early_stop_patience})")
    print(f"Physics loss : λ_phys={best_params.get('lambda_physics', 1e-3):.4g}  "
          f"λ_data={best_params.get('lambda_data', 1.0):.4g}  "
          f"n_anchors={best_params.get('n_anchors', 4096)}")
    print(f"               (strong-form Saint-Venant: R = ε_rr − ε_θθ − r·∂ε_θθ/∂r,")
    print(f"                ∂/∂r через dde.grad.jacobian)")

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
            log_every=max(1, best_params["adam_iters"] // 10),
        )
        per_fold_metrics.append(af["metrics"])
        per_fold_times.append(af["elapsed_sec"])
        per_fold_epochs.append(af["n_epochs"])
        fold_test_arrays.append((af["y_true"], af["y_pred"]))
        fold_artifacts.append(af)

        m = af["metrics"]
        print(f"  → Fold {k+1}: R²={m[0]:+.4f}  RMSE={m[1]:.6f}  "
              f"VM R²={m[2]:+.4f}  SMAPE={m[3]:.2f}%  "
              f"log-points={af['n_epochs']}  time={af['elapsed_sec']:.1f}s")

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


def predict_with_pinn(handle: DDEPINNHandle,
                      X: np.ndarray,
                      mean_X: np.ndarray,
                      std_X:  np.ndarray,
                      scaler_y: ZScoreScaler) -> Dict[str, np.ndarray]:
    Xs = ((X - mean_X) / std_X).astype(np.float32)
    pred_sc = handle.predict_z(Xs)
    strain = scaler_y.inverse(pred_sc)
    return {"strain": strain, "von_mises": von_mises_strain(strain)}


def evaluate_on_external_set(handle: DDEPINNHandle,
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


def handle_state_dict_to_numpy(handle: DDEPINNHandle) -> Dict:
    return {k: v.detach().cpu().numpy()
            for k, v in handle.net.state_dict().items()}


def handle_from_numpy(state_dict_np: Dict, layer_sizes: list,
                       activation: str, initializer: str = "Glorot normal"
                       ) -> DDEPINNHandle:
    net = dde.nn.FNN(layer_sizes, activation, initializer)
    sd_torch = {k: torch.as_tensor(v) for k, v in state_dict_np.items()}
    net.load_state_dict(sd_torch)
    net.eval()
    return DDEPINNHandle(
        net=net, layer_sizes=list(layer_sizes),
        activation=activation, initializer=initializer,
    )
