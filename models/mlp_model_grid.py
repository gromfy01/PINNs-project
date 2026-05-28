from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import (
    explained_variance_score, max_error,
    mean_absolute_error, mean_squared_error,
    median_absolute_error, r2_score,
)
from sklearn.model_selection import KFold

STRESS_COMPONENTS: List[str] = ["sigma_rr", "sigma_tt", "sigma_zz", "tau_rz"]
N_R: int = 20
PLANE_ORDER: List[int] = [2, 0, 1, 3]

METRIC_NAMES: List[str] = [
    "Explained Variance", "Median AE", "MSE", "MAE", "R²",
    "Max Error", "AIC", "BIC", "RMSE (4 comp macro)", "Von Mises R²",
    "SMAPE [%]",
]

ACTIVATIONS = ["tanh", "selu", "softplus"]
_ACT_MAP = {
    "tanh":     nn.Tanh,
    "selu":     nn.SELU,
    "softplus": nn.Softplus,
}

DEFAULT_GRID: Dict[str, List[Any]] = {
    "n_layers":      [3, 4, 5],
    "n_neurons":     [128, 256],
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [64, 128],
}


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
        print(f"X_mlp  : {X_pinn.shape}   [p1..p5, r]")
        print(f"y_mlp  : {y_pinn.shape}   [σ_rr, σ_θθ, σ_zz, τ_rz]")
        print("Mapping: stress_33→σ_rr | stress_11→σ_θθ | "
              "stress_22→σ_zz | stress_12→τ_rz")
        print("\nДиапазоны y (все реальные FEM данные):")
        for i, lbl in enumerate(STRESS_COMPONENTS):
            print(f"  {lbl:10s}: [{y_pinn[:,i].min():10.4g},  "
                  f"{y_pinn[:,i].max():10.4g}]"
                  f"   mean_abs={np.abs(y_pinn[:,i]).mean():.4g}")
        print("\nИнформативно — BC при r=1 (FEM соблюдает её, MLP не учитывает):")
        for col, lbl in [(0, "σ_rr"), (3, "τ_rz")]:
            at_r1 = y_pinn[n_r-1::n_r, col]
            print(f"  {lbl} at r=1: mean={at_r1.mean():.4g}  "
                  f"max|.|={np.max(np.abs(at_r1)):.4g}  (FEM ≈ 0)")
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


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    s_rr, s_tt, s_zz, tau = (sigma[:, i] for i in range(4))
    dev = 0.5 * ((s_rr - s_tt) ** 2 + (s_tt - s_zz) ** 2 + (s_zz - s_rr) ** 2)
    return np.sqrt(np.maximum(dev + 3.0 * tau ** 2, 0.0))


def cuml_scorer_mlp(y_true: np.ndarray, y_pred: np.ndarray,
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


class MLPNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: str = "tanh"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        act_cls = _ACT_MAP[activation]
        layers: List[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            lin = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(layer_sizes) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_mlp_network(layer_sizes: List[int], activation: str = "tanh"
                      ) -> nn.Module:
    return MLPNetwork(layer_sizes, activation)


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


def _expand_uniform_to_per_layer(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    n_lay = cfg["n_layers"]
    n_neur = cfg["n_neurons"]
    for i in range(n_lay):
        out[f"n_units_{i}"] = n_neur
    return out


def _train_single(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray,
                  best_params: Dict[str, Any], r_col_idx: int,
                  verbose_every: int = 20, label: str = "",
                  patience: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_tr, r_grid_np, y_tr_s = _flat_to_samples(X_tr, y_tr)
    params_te, _,         y_te_s = _flat_to_samples(X_te, y_te)

    scaler = OutputScaler().fit(y_tr)
    y_tr_s_sc = scaler.transform(y_tr_s.reshape(-1, 4)).reshape(y_tr_s.shape)
    y_te_s_sc = scaler.transform(y_te_s.reshape(-1, 4)).reshape(y_te_s.shape)

    Xtr_mx_np = X_tr.mean(0, keepdims=True).astype(np.float32)
    Xtr_sx_np = X_tr.std(0, keepdims=True).astype(np.float32)
    Xtr_sx_np[Xtr_sx_np == 0] = 1.0

    mx       = torch.from_numpy(Xtr_mx_np).float().to(device)
    sx       = torch.from_numpy(Xtr_sx_np).float().to(device)

    params_tr_t  = torch.from_numpy(params_tr).float().to(device)
    params_te_t  = torch.from_numpy(params_te).float().to(device)
    y_tr_s_sc_t  = torch.from_numpy(y_tr_s_sc).float().to(device)
    y_te_s_sc_t  = torch.from_numpy(y_te_s_sc).float().to(device)
    r_grid_t     = torch.from_numpy(r_grid_np).float().to(device)

    k_lay   = best_params["n_layers"]
    units   = [best_params[f"n_units_{i}"] for i in range(k_lay)]
    act     = best_params.get("activation", "tanh")
    bs      = best_params["batch_size"]
    lr      = best_params["learning_rate"]
    epochs  = best_params.get("max_epochs", best_params.get("epochs", 300))
    opt_nm  = best_params.get("optimizer", "AdamW")
    clip    = best_params.get("grad_clip", 1.0)
    wd      = best_params.get("weight_decay", 1e-4)

    layer_sizes = [6] + units + [4]
    seed = int(best_params.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)

    net = make_mlp_network(layer_sizes, activation=act).to(device)
    opt = _make_opt(opt_nm, net.parameters(), lr, wd)

    n_train_sets = params_tr.shape[0]
    rng_np = np.random.default_rng(seed)

    best_val = float("inf"); pat = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    train_losses, test_losses, vm_rmse_test = [], [], []

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
            pred_np = scaler.inverse(pred_sc.reshape(-1, 4).cpu().numpy())
            y_true_flat = y_te_s.reshape(-1, 4)
            vm = float(np.sqrt(mean_squared_error(
                von_mises_stress(y_true_flat), von_mises_stress(pred_np))))
        net.train()
        return tel, pred_np, vm

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
            X_data_std = (X_data - mx) / sx

            pred_sc_flat = net(X_data_std)
            pred_sc = pred_sc_flat.reshape(B, N_R, 4)

            loss = ((pred_sc - batch_y_sc) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            epoch_train_loss += loss.item(); n_batches += 1

        trl = epoch_train_loss / max(n_batches, 1)
        tel, pred_np, vm = _eval_test()

        train_losses.append(trl); test_losses.append(tel)
        vm_rmse_test.append(vm)

        if verbose_every and epoch % verbose_every == 0:
            y_true_flat = y_te_s.reshape(-1, 4)
            r2s = [r2_score(y_true_flat[:, i], pred_np[:, i]) for i in range(4)]
            tag = f"[{label}] " if label else ""
            print(f"{tag}Epoch {epoch:4d}/{epochs} | "
                  f"Train {trl:.4f}  Test {tel:.4f} | "
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

    _, pred, _ = _eval_test()
    y_te_flat = y_te_s.reshape(-1, 4)
    metrics = cuml_scorer_mlp(y_te_flat, pred, net, X_tr)

    history = {"train_losses": train_losses, "test_losses": test_losses,
               "vm_rmse_test": vm_rmse_test}
    return net, metrics, mx, sx, scaler, history, y_te_flat, pred


def do_manual_tuning_mlp(X: np.ndarray, y: np.ndarray,
                          grid: Optional[Dict[str, List[Any]]] = None,
                          epochs: int = 300,
                          patience: int = 25,
                          test_size: float = 0.15,
                          random_state: int = 42,
                          activation: str = "tanh",
                          optimizer: str = "AdamW",
                          weight_decay: float = 1e-4,
                          grad_clip: float = 1.0,
                          r_col_idx: int = -1):
    if grid is None:
        grid = DEFAULT_GRID

    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}   (MLP baseline, manual grid search)")
    print(f"torch version: {torch.__version__}")

    (cur_X_test, cur_y_test,
     val_list_X, val_list_y,
     train_list_X, train_list_y) = _split_cv(X, y, n_splits=3,
                                              test_size=test_size,
                                              random_state=random_state)
    X_tr, y_tr = _clean(train_list_X[0], train_list_y[0])
    X_vl, y_vl = _clean(val_list_X[0],   val_list_y[0])

    print(f"\nGrid-search split (group-aware, no leakage):")
    print(f"  test  (held-out)   : {len(cur_X_test) // N_R:4d} sets, "
          f"{len(cur_X_test):5d} rows")
    print(f"  train (grid-tuning): {len(X_tr) // N_R:4d} sets, "
          f"{len(X_tr):5d} rows")
    print(f"  val   (grid-eval)  : {len(X_vl) // N_R:4d} sets, "
          f"{len(X_vl):5d} rows")

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n_total = len(combos)
    print(f"\nGrid search: {n_total} configurations  (keys: {keys})")
    print(f"{'#':>3}  {'Layers':>6}  {'Neurons':>7}  "
          f"{'LR':>9}  {'BS':>5}  {'Val RMSE':>10}  {'Time':>6}")
    print("─" * 60)

    results: List[Dict[str, Any]] = []
    best_val   = float("inf")
    best_cfg: Optional[Dict[str, Any]] = None

    for ci, combo in enumerate(combos, start=1):
        cfg = dict(zip(keys, combo))
        cfg.setdefault("activation",   activation)
        cfg.setdefault("optimizer",    optimizer)
        cfg.setdefault("weight_decay", weight_decay)
        cfg.setdefault("grad_clip",    grad_clip)
        cfg.setdefault("max_epochs",   epochs)
        cfg["seed"] = ci

        full_cfg = _expand_uniform_to_per_layer(cfg)

        t0 = time.perf_counter()
        try:
            _, metrics, _, _, _, _, _, _ = _train_single(
                X_tr, y_tr, X_vl, y_vl, full_cfg, r_col_idx,
                verbose_every=0, label="", patience=patience)
            val_rmse = float(metrics[8])
            val_r2   = float(metrics[4])
        except Exception as e:
            print(f"  [config #{ci} skipped: {type(e).__name__}: {e}]")
            val_rmse = 1e6; val_r2 = 0.0; metrics = None

        elapsed = time.perf_counter() - t0

        marker = " ◀ best" if val_rmse < best_val else ""
        print(f"{ci:>3}  {cfg['n_layers']:>6}  {cfg['n_neurons']:>7}  "
              f"{cfg['learning_rate']:>9.0e}  {cfg['batch_size']:>5}  "
              f"{val_rmse:>10.4f}  {elapsed:>5.1f}s{marker}")

        results.append({
            **cfg, "val_rmse": val_rmse, "val_r2": val_r2,
            "time_s": elapsed,
        })

        if val_rmse < best_val:
            best_val = val_rmse
            best_cfg = full_cfg

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)

    print(f"\n{'═'*60}")
    print(f"Best configuration  (val RMSE = {best_val:.4f}):")
    summary_keys = ["n_layers", "n_neurons", "activation", "learning_rate",
                    "batch_size", "optimizer", "max_epochs"]
    for k in summary_keys:
        if k in best_cfg:
            v = best_cfg[k]
            if isinstance(v, float):
                print(f"  {k:<18}: {v:.6g}")
            else:
                print(f"  {k:<18}: {v}")

    return best_cfg, cur_X_test, cur_y_test, best_val, results_df


def plot_grid_results(results_df: pd.DataFrame, top_n: Optional[int] = None
                       ) -> plt.Figure:
    df = results_df.copy()
    if top_n is not None and top_n < len(df):
        df = df.head(top_n).reset_index(drop=True)

    labels = []
    for row in df.itertuples():
        bs = int(row.batch_size) if hasattr(row, "batch_size") else 0
        nl = int(row.n_layers)
        nn = int(row.n_neurons)
        lr = float(row.learning_rate)
        labels.append(f"L{nl} N{nn}\nlr={lr:.0e} bs={bs}")

    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.4), 5))
    bars = ax.bar(range(len(df)), df["val_rmse"], color=colors)
    try:
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    except Exception:
        pass
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Val RMSE (macro, 4 components) [МПа]")
    ax.set_title("Manual grid search — validation RMSE per configuration\n"
                 "(sorted ascending; синим — лучший)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def test_best_model_mlp(X: np.ndarray, y: np.ndarray,
                          best_params: Dict[str, Any],
                          r_col_idx: int = -1, patience: int = 30,
                          **kwargs):
    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}   (MLP baseline, single-fold)")

    (X_test, y_test, _, _, tr_X, tr_y) = _split_cv(X, y, n_splits=1)
    X_tr, y_tr = _clean(tr_X[0], tr_y[0])
    X_te, y_te = _clean(X_test, y_test)

    net, metrics, mx, sx, scaler, history, y_te_ret, pred_te = \
        _train_single(X_tr, y_tr, X_te, y_te, best_params, r_col_idx,
                       verbose_every=20, label="single", patience=patience)

    print("\n── Test Metrics ─────────────────────────────────────")
    for n, v in zip(METRIC_NAMES, metrics):
        print(f"  {n:<26}: {v:.4f}")
    per_component_report(y_te_ret, pred_te, label="Test (single fold)")

    return net, metrics, mx, sx, scaler, history


def kfold_test_best_model_mlp(X: np.ndarray, y: np.ndarray,
                                best_params: Dict[str, Any],
                                r_col_idx: int = -1,
                                n_folds: int = 5, random_state: int = 42,
                                patience: int = 30,
                                **kwargs) -> Dict[str, Any]:
    n_feat = X.shape[1]
    if r_col_idx < 0: r_col_idx = n_feat + r_col_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}   (MLP baseline, K-fold)")

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
    print(f"MLP K-FOLD FINAL EVALUATION   (K={n_folds}, group-aware)")
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
                           verbose_every=50, label=f"fold{k+1}",
                           patience=patience)
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
    print("MLP K-FOLD AGGREGATE  (mean ± std across folds)")
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


def predict_with_mlp(model: nn.Module, X: np.ndarray,
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
