"""
Уравнения совместности Сен-Венана для осесимметричного случая.

Применимы только к ПОЛНЫМ деформациям (упругие + пластические).
В текущих данных доступны только пластические деформации в одной z-плоскости,
поэтому в strain-моделях используется как мягкий регуляризатор с весом ~1e-4.
Полная активация — после получения 2D total strain данных из Abaqus.
"""

import numpy as np


def compatibility_residual_1d(eps_profile, r_grid):
    e_rr = eps_profile[:, 0]
    e_tt = eps_profile[:, 1]
    e_zz = eps_profile[:, 2]

    de_zz_dr = np.gradient(e_zz, r_grid)
    d2e_zz_dr2 = np.gradient(de_zz_dr, r_grid)

    r_safe = np.maximum(r_grid, 0.05)
    return d2e_zz_dr2 - de_zz_dr / r_safe
