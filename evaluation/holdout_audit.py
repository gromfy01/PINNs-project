import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from training.data_pipeline import (
    PLANE_ORDER_STRESS, build_stress_dataset, N_R, STRESS_NAMES,
)
from models.loader import load_bundle, forward_predict
from physics.invariants import von_mises_stress
from .metrics import smape, rmse


def sanity_check(bundles, X_hold, y_hold, hold_sets):
    rows = []
    for name, b in bundles.items():
        saved = np.asarray(b["holdout_set_indices"])
        if not np.array_equal(np.sort(saved), hold_sets):
            rows.append({"Model": name, "ok": False, "reason": "holdout mismatch"})
            continue
        y_pred = forward_predict(b, X_hold)
        r2_recomp = r2_score(y_hold.reshape(-1), y_pred.reshape(-1))
        r2_saved = float(b["holdout_metrics"][4])
        drift = abs(r2_recomp - r2_saved)
        rows.append({
            "Model": name,
            "recomputed_R2": r2_recomp,
            "saved_R2": r2_saved,
            "drift": drift,
            "ok": drift < 1e-3,
        })
    return pd.DataFrame(rows)


def rank_models(bundles, X_hold, y_hold):
    rows = []
    for name, b in bundles.items():
        y_pred = forward_predict(b, X_hold)
        yt_flat = y_hold.reshape(-1)
        yp_flat = y_pred.reshape(-1)

        r2 = r2_score(yt_flat, yp_flat)
        rmse_macro = float(np.mean([
            rmse(y_hold[:, i], y_pred[:, i]) for i in range(4)
        ]))
        mae = float(np.mean(np.abs(y_hold - y_pred)))
        sm = smape(yt_flat, yp_flat)

        vm_t = von_mises_stress(y_hold)
        vm_p = von_mises_stress(y_pred)
        vm_r2 = r2_score(vm_t, vm_p)
        vm_rmse = float(np.sqrt(mean_squared_error(vm_t, vm_p)))

        pc_r2 = [r2_score(y_hold[:, i], y_pred[:, i]) for i in range(4)]

        rows.append({
            "Model": name,
            "R2": r2, "RMSE": rmse_macro,
            "VM_R2": vm_r2, "VM_RMSE": vm_rmse,
            "SMAPE": sm, "MAE": mae,
            **{f"R2_{n}": pc_r2[i] for i, n in enumerate(STRESS_NAMES)},
        })

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def predict_all(bundles, X_input):
    return {name: forward_predict(b, X_input) for name, b in bundles.items()}


def reshape_to_per_set(y_flat, n_sets, n_r=N_R):
    n_comp = y_flat.shape[1]
    return y_flat.reshape(n_sets, n_r, n_comp)


def load_holdout_split(X_stress, y_stress):
    return build_stress_dataset(X_stress, y_stress)
