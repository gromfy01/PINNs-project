"""
run_raw_audit.py — один проход по всем сырым .rpt, который отвечает на
вопросы, вынесенные в ERRATA. Результат: audit_raw.npz + печатный отчёт.

Запуск:
    python publication/audit/run_raw_audit.py --root <dir с Vel_*> --out <npz>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from raw_probe import (                      # noqa: E402
    C_R, C_SRR, C_SRZ, C_STT, C_SZZ, C_Z,
    axial_window, list_jobs, parse_job_name, profile_from_rpt, read_rpt,
)

N_R = 20
NZ_GRID = 20


def grid_field(arr, mask, n_r=N_R, n_z=NZ_GRID):
    """
    Неструктурированные узлы окна → регулярная (n_r × n_z) сетка медиан.
    Возвращает (r_axis, z_axis, fields) где fields[c] — (n_r, n_z).
    Пустые ячейки заполняются линейной интерполяцией вдоль r.
    """
    r, z = arr[mask, C_R], arr[mask, C_Z]
    comps = np.stack([arr[mask, C_SRR], arr[mask, C_STT],
                      arr[mask, C_SZZ], arr[mask, C_SRZ]], axis=0) / 1e6

    r_edges = np.linspace(r.min(), r.max(), n_r + 1)
    z_edges = np.linspace(z.min(), z.max(), n_z + 1)
    ri = np.clip(np.digitize(r, r_edges) - 1, 0, n_r - 1)
    zi = np.clip(np.digitize(z, z_edges) - 1, 0, n_z - 1)

    out = np.full((4, n_r, n_z), np.nan)
    flat = ri * n_z + zi
    order = np.argsort(flat)
    flat_s = flat[order]
    bounds = np.searchsorted(flat_s, np.arange(n_r * n_z + 1))
    for cell in range(n_r * n_z):
        lo, hi = bounds[cell], bounds[cell + 1]
        if hi > lo:
            idx = order[lo:hi]
            out[:, cell // n_z, cell % n_z] = np.median(comps[:, idx], axis=1)

    # заполнить дыры вдоль r (по каждому z-столбцу)
    r_axis = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_axis = 0.5 * (z_edges[:-1] + z_edges[1:])
    for c in range(4):
        for j in range(n_z):
            col = out[c, :, j]
            good = ~np.isnan(col)
            if good.sum() >= 2:
                out[c, :, j] = np.interp(r_axis, r_axis[good], col[good])
    return r_axis, z_axis, out


def dropped_term_stats(arr, window="repo"):
    """
    Численная оценка отброшенных осевых членов уравнений равновесия.

    Полная форма:
        R_r = ∂σ_rr/∂r + ∂τ_rz/∂z + (σ_rr − σ_θθ)/r
        R_z = ∂τ_rz/∂r + ∂σ_zz/∂z + τ_rz/r
    Реализовано в коде (редуцированная форма) — без ∂/∂z.

    Возвращает медианы |каждого члена| по узлам окна, МПа/м.
    """
    z = arr[:, C_Z]
    lo, hi = axial_window(z, window)
    mask = (z >= lo) & (z <= hi)
    if mask.sum() < 4 * N_R:
        return None

    r_axis, z_axis, F = grid_field(arr, mask)
    if np.isnan(F).any():
        F = np.nan_to_num(F, nan=0.0)
    s_rr, s_tt, s_zz, tau = F

    d_srr_dr = np.gradient(s_rr, r_axis, axis=0)
    d_tau_dr = np.gradient(tau,  r_axis, axis=0)
    d_tau_dz = np.gradient(tau,  z_axis, axis=1)
    d_szz_dz = np.gradient(s_zz, z_axis, axis=1)

    r_safe = np.maximum(r_axis, 0.05 * r_axis.max())[:, None]
    hoop = (s_rr - s_tt) / r_safe
    shear_over_r = tau / r_safe

    med = lambda a: float(np.median(np.abs(a)))
    return {
        "d_srr_dr": med(d_srr_dr), "d_tau_dz": med(d_tau_dz), "hoop": med(hoop),
        "d_tau_dr": med(d_tau_dr), "d_szz_dz": med(d_szz_dz), "shear_over_r": med(shear_over_r),
        "R1_kept": med(d_srr_dr + hoop), "R1_full": med(d_srr_dr + d_tau_dz + hoop),
        "R2_kept": med(d_tau_dr + shear_over_r), "R2_full": med(d_tau_dr + d_szz_dz + shear_over_r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    jobs = list_jobs(args.root)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"файлов найдено: {len(jobs)}")

    names, params, prof_r, prof_s, metas, drops = [], [], [], [], [], []
    bad = []
    for i, (name, path) in enumerate(jobs):
        try:
            arr = read_rpt(path)
            r_p, s_p, _e_p, meta = profile_from_rpt(arr)
            d = dropped_term_stats(arr)
        except Exception as exc:                       # noqa: BLE001
            bad.append((name, repr(exc)))
            continue
        pj = parse_job_name(name)
        names.append(name)
        params.append([pj.get("Q", np.nan), pj.get("k", np.nan), pj.get("alpha", np.nan),
                       pj.get("mu_repo", np.nan), pj.get("v", np.nan)])
        prof_r.append(r_p)
        prof_s.append(s_p)
        metas.append([meta[k] for k in META_KEYS])
        drops.append([d[k] for k in DROP_KEYS] if d else [np.nan] * len(DROP_KEYS))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(jobs)}")

    np.savez_compressed(
        args.out,
        names=np.array(names), params=np.array(params, dtype=np.float64),
        prof_r=np.array(prof_r), prof_s=np.array(prof_s),
        metas=np.array(metas, dtype=np.float64), meta_keys=np.array(META_KEYS),
        drops=np.array(drops, dtype=np.float64), drop_keys=np.array(DROP_KEYS),
    )
    print(f"сохранено: {args.out}  ({len(names)} наборов, ошибок {len(bad)})")
    for n, e in bad[:10]:
        print("   ✗", n, e)


META_KEYS = ["r_min_all", "r_max_all", "z_min_all", "z_max_all",
             "r_min_win", "r_max_win", "z_min_win", "z_max_win",
             "n_nodes_all", "n_nodes_win", "win_lo", "win_hi"]
DROP_KEYS = ["d_srr_dr", "d_tau_dz", "hoop", "d_tau_dr", "d_szz_dz",
             "shear_over_r", "R1_kept", "R1_full", "R2_kept", "R2_full"]

if __name__ == "__main__":
    main()
