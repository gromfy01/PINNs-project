"""
evaluate.py — метрики ТЗ, п. 7, на тестовых режимах (МПа).

Невязка равновесия и ГУ считаются одним и тем же оператором (physics.residuals, 4-й порядок)
для МКЭ и для всех моделей. Рядом с метриками модели печатаются те же величины для данных МКЭ
и RMSE базовой модели «среднее поле обучающей выборки», чтобы пороги п. 11 было с чем сравнить.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from physics import Grid, residuals, section_force, stress_intensity

COMP = ("sigma_r", "sigma_theta", "sigma_z", "tau_rz")


def _rms(a) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def physics_metrics(y: np.ndarray, g: Grid) -> Dict[str, float]:
    """Невязки F2, F3 и нарушения ГУ, RMS по узлам и режимам. y: (n, 4, n_r, n_z)."""
    R1, R2 = residuals(y, g)
    return {
        "res_R1": _rms(R1), "res_R2": _rms(R2),
        "res_R1_interior": _rms(R1[:, 1:-1, :]), "res_R2_interior": _rms(R2[:, 1:-1, :]),
        "bc_sr_surface": _rms(y[:, 0, -1, :]), "bc_trz_surface": _rms(y[:, 3, -1, :]),
        "bc_trz_axis": _rms(y[:, 3, 0, :]), "bc_sr_minus_st_axis": _rms(y[:, 0, 0, :] - y[:, 1, 0, :]),
        "axial_force": _rms(section_force(y, g)),
    }


def evaluate(pred: np.ndarray, y: np.ndarray, g: Grid, y_train_mean: np.ndarray,
             flags: Dict[str, np.ndarray] | None = None) -> Dict[str, object]:
    e = pred - y
    out: Dict[str, object] = {}
    for c, nm in enumerate(COMP):
        out[f"rmse_{nm}"] = _rms(e[:, c])
        out[f"maxerr_{nm}_interior"] = float(np.abs(e[:, c, 1:-1, :]).max())
        out[f"maxerr_{nm}_boundary"] = float(np.abs(e[:, c, [0, -1], :]).max())
        out[f"rmse_meanfield_{nm}"] = _rms(y_train_mean[None, c] - y[:, c])
    out["rmse_macro"] = float(np.mean([out[f"rmse_{n}"] for n in COMP]))
    out["rmse_sigma_i"] = _rms(stress_intensity(pred) - stress_intensity(y))

    # σ_z на поверхности в установившейся зоне: среднее по окну, ошибка по режиму
    s_fem = y[:, 2, -1, :].mean(-1)
    s_err = np.abs(pred[:, 2, -1, :].mean(-1) - s_fem)
    tol = np.maximum(0.1 * np.abs(s_fem), 15.0)
    out["surf_sz_err_median"] = float(np.median(s_err))
    out["surf_sz_err_p90"] = float(np.percentile(s_err, 90))
    out["surf_sz_pass_frac"] = float(np.mean(s_err <= tol))

    # знак σ_z: растяжение на поверхности, сжатие на оси — там, где |σ_z МКЭ| > 15 МПа
    a_fem = y[:, 2, 0, :].mean(-1)
    a_pred = pred[:, 2, 0, :].mean(-1)
    ms, ma = np.abs(s_fem) > 15, np.abs(a_fem) > 15
    out["sign_match_surface"] = float(np.mean(np.sign(pred[:, 2, -1, :].mean(-1)[ms]) == np.sign(s_fem[ms])))
    out["sign_match_axis"] = float(np.mean(np.sign(a_pred[ma]) == np.sign(a_fem[ma])))

    # пороги п. 11 в том виде, как они записаны в ТЗ (от размаха по тестовой выборке)
    rng = {nm: float(np.ptp(y[:, c])) for c, nm in enumerate(COMP)}
    out["thr_spec_sigma_z"] = 0.05 * rng["sigma_z"]
    out["thr_spec_sigma_theta"] = 0.05 * rng["sigma_theta"]
    out["thr_spec_sigma_r_tau"] = 0.05 * rng["sigma_z"]

    out.update({f"model_{k}": v for k, v in physics_metrics(pred, g).items()})
    out.update({f"fem_{k}": v for k, v in physics_metrics(y, g).items()})
    out["max_abs_sigma_median"] = float(np.median(np.abs(y).reshape(len(y), -1).max(1)))

    if flags:
        macro = lambda s: float(np.mean([_rms(e[s][:, c]) for c in range(4)]))  # noqa: E731
        for nm, m in flags.items():
            if m.any() and (~m).any():
                out[f"rmse_macro_{nm}"] = macro(m)
                out[f"rmse_macro_not_{nm}"] = macro(~m)
                out[f"n_{nm}"] = int(m.sum())
    return out
