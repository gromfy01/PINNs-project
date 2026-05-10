import numpy as np


def hooke_thermoelastic_strain(sigma, E, nu, alpha=0.0, T=0.0):
    s_rr, s_tt, s_zz, tau_rz = (sigma[..., i] for i in range(4))
    e_rr = (s_rr - nu * (s_tt + s_zz)) / E + alpha * T
    e_tt = (s_tt - nu * (s_rr + s_zz)) / E + alpha * T
    e_zz = (s_zz - nu * (s_rr + s_tt)) / E + alpha * T
    G = E / (2.0 * (1.0 + nu))
    gamma_rz = tau_rz / G
    return np.stack([e_rr, e_tt, e_zz, gamma_rz], axis=-1)


def johnson_cook_yield(eps_p, eps_dot_p, T, A, B, n, C, m,
                       eps_dot_0=1.0, T_0=300.0, T_m=1800.0):
    T_star = (T - T_0) / (T_m - T_0)
    T_star = np.clip(T_star, 0.0, 1.0)
    log_term = np.log(np.maximum(eps_dot_p / eps_dot_0, 1e-12))
    return (A + B * np.power(eps_p, n)) * (1.0 + C * log_term) * (1.0 - T_star ** m)


JOHNSON_COOK_STEEL_DEFAULTS = dict(
    A=500e6, B=300e6, n=0.4, C=0.05, m=1.0,
    eps_dot_0=1.0, T_0=300.0, T_m=1800.0,
)
