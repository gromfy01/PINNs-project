import numpy as np

from physics.equilibrium import equilibrium_residuals_1d, bc_violation
from .outlier_diagnosis import classify_holdout, split_by_category


def audit_per_set(predictions_per_set, r_grid, exclude_boundary=True):
    n_sets = predictions_per_set.shape[0]
    out = {"eq_r": np.zeros(n_sets), "eq_z": np.zeros(n_sets),
           "bc_srr": np.zeros(n_sets), "bc_trz": np.zeros(n_sets)}
    for i in range(n_sets):
        res_r, res_z = equilibrium_residuals_1d(
            predictions_per_set[i], r_grid, exclude_boundary=exclude_boundary,
        )
        out["eq_r"][i] = np.mean(np.abs(res_r))
        out["eq_z"][i] = np.mean(np.abs(res_z))
        out["bc_srr"][i], out["bc_trz"][i] = bc_violation(predictions_per_set[i])
    return out


def audit_models_by_category(predictions_dict_per_set, y_true_per_set, r_grid):
    labels = classify_holdout(y_true_per_set)
    summary = {}
    for model_name, pred_per_set in predictions_dict_per_set.items():
        per_set = audit_per_set(pred_per_set, r_grid)
        cat_idx = {
            "normal": np.where(labels == "normal")[0],
            "low_signal": np.where(labels == "low_signal")[0],
            "fem_artifact": np.where(labels == "fem_artifact")[0],
        }
        cat_summary = {}
        for cat, idx in cat_idx.items():
            if len(idx) == 0:
                cat_summary[cat] = None
                continue
            cat_summary[cat] = {
                "eq_r_mean": float(np.mean(per_set["eq_r"][idx])),
                "eq_z_mean": float(np.mean(per_set["eq_z"][idx])),
                "bc_srr_mean": float(np.mean(per_set["bc_srr"][idx])),
                "bc_trz_mean": float(np.mean(per_set["bc_trz"][idx])),
                "n_sets": int(len(idx)),
            }
        summary[model_name] = cat_summary
    return summary, labels


def fem_baseline_audit(y_true_per_set, r_grid):
    return audit_per_set(y_true_per_set, r_grid)
