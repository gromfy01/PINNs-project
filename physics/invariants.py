import numpy as np


def von_mises_stress(sigma):
    s_rr, s_tt, s_zz, tau = (sigma[..., i] for i in range(4))
    dev = 0.5 * ((s_rr - s_tt) ** 2 + (s_tt - s_zz) ** 2 + (s_zz - s_rr) ** 2)
    return np.sqrt(np.maximum(dev + 3.0 * tau ** 2, 0.0))


def von_mises_strain(eps):
    e_rr, e_tt, e_zz, gamma = (eps[..., i] for i in range(4))
    dev = (2.0 / 3.0) * (
        (e_rr - e_tt) ** 2 + (e_tt - e_zz) ** 2 + (e_zz - e_rr) ** 2
    )
    return np.sqrt(np.maximum(dev + (4.0 / 3.0) * gamma ** 2, 0.0))


def hydrostatic_pressure(sigma):
    return -(sigma[..., 0] + sigma[..., 1] + sigma[..., 2]) / 3.0


def deviatoric_stress(sigma):
    p = (sigma[..., 0] + sigma[..., 1] + sigma[..., 2]) / 3.0
    s = sigma.copy()
    for i in range(3):
        s[..., i] = sigma[..., i] - p
    return s
