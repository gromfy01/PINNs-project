import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    den = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(100 * np.mean(np.abs(y_true - y_pred) / den))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def per_component_metrics(y_true, y_pred, names=None):
    n_comp = y_true.shape[1]
    if names is None:
        names = [f"comp_{i}" for i in range(n_comp)]
    rows = []
    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        rows.append({
            "component": name,
            "R2": r2_score(yt, yp),
            "RMSE": rmse(yt, yp),
            "SMAPE": smape(yt, yp),
            "MAE": float(np.mean(np.abs(yt - yp))),
            "MaxAE": float(np.max(np.abs(yt - yp))),
        })
    return rows


def overall_metrics(y_true, y_pred):
    yt_flat = y_true.reshape(-1)
    yp_flat = y_pred.reshape(-1)
    return {
        "R2": r2_score(yt_flat, yp_flat),
        "RMSE_macro": float(np.mean([
            rmse(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
        ])),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "SMAPE": smape(yt_flat, yp_flat),
    }
