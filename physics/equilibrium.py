import numpy as np


R_CLIP_FOR_DIVISION = 0.05


def equilibrium_residuals_1d(sigma_profile, r_grid, exclude_boundary=True):
    s_rr = sigma_profile[:, 0]
    s_tt = sigma_profile[:, 1]
    tau_rz = sigma_profile[:, 3]

    dsrr_dr = np.gradient(s_rr, r_grid)
    dtau_dr = np.gradient(tau_rz, r_grid)

    r_safe = np.maximum(r_grid, R_CLIP_FOR_DIVISION)
    res_r = dsrr_dr + (s_rr - s_tt) / r_safe
    res_z = dtau_dr + tau_rz / r_safe

    if exclude_boundary:
        res_r = res_r[1:-1]
        res_z = res_z[1:-1]
    return res_r, res_z


def bc_violation(sigma_profile):
    return abs(sigma_profile[-1, 0]), abs(sigma_profile[-1, 3])


def physics_metrics(predictions, r_grid, exclude_boundary=True):
    n_sets = predictions.shape[0]
    eq_r, eq_z, bc_srr, bc_trz = [], [], [], []
    for i in range(n_sets):
        res_r, res_z = equilibrium_residuals_1d(
            predictions[i], r_grid, exclude_boundary=exclude_boundary,
        )
        bs, bt = bc_violation(predictions[i])
        eq_r.append(np.mean(np.abs(res_r)))
        eq_z.append(np.mean(np.abs(res_z)))
        bc_srr.append(bs)
        bc_trz.append(bt)
    return {
        "eq_r_mean": float(np.mean(eq_r)),
        "eq_z_mean": float(np.mean(eq_z)),
        "bc_srr_mean": float(np.mean(bc_srr)),
        "bc_trz_mean": float(np.mean(bc_trz)),
        "eq_r_per_set": np.array(eq_r),
        "eq_z_per_set": np.array(eq_z),
        "bc_srr_per_set": np.array(bc_srr),
        "bc_trz_per_set": np.array(bc_trz),
    }
