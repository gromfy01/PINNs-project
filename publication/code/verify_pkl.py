"""
verify_pkl.py — «что на самом деле лежит в этом .pkl».

Инструмент существует потому, что проверка, которой пользовались раньше
(«σ_rr на поверхности близок к нулю»), под-определена: на свободной поверхности
к нулю близки ДВЕ величины, и неверная раскладка её проходит. Здесь опознание
делается сопоставлением с сырыми `.rpt` по параметрам задания и сравнением
со ВСЕМИ величинами, которые есть в отчёте Abaqus, а не с четырьмя ожидаемыми.

Запуск:
    python publication/code/verify_pkl.py --pkl data/processed/y_stress.pkl \\
        --xpkl data/processed/X_stress.pkl --raw <каталог с Vel_*>

Печатает таблицу «плоскость → величина» с mean|Δ|, корреляцией и долей
побитовых совпадений, и отдельно — присутствует ли в массиве каждая из четырёх
компонент тензора.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_probe import (C_R, C_Z, axial_window, list_jobs, median_bins,  # noqa: E402
                       parse_job_name, read_rpt)

CANDIDATES = ["sigma_rr", "sigma_tt", "sigma_zz", "tau_rz", "mises",
              "sigma_1_max", "sigma_2_mid", "sigma_3_min",
              "eps_rr", "eps_tt", "eps_zz", "eps_rz"]
TENSOR_STRESS = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")


def raw_table(root: str, n_r: int = 20):
    jobs = list_jobs(root)
    data = np.zeros((len(jobs), n_r, len(CANDIDATES)))
    params = np.zeros((len(jobs), 5))
    for i, (nm, path) in enumerate(jobs):
        a = read_rpt(path)
        z = a[:, C_Z]
        lo, hi = axial_window(z)
        m = (z >= lo) & (z <= hi)
        r = a[m, C_R]
        cols = [a[m, 11] / 1e6, a[m, 13] / 1e6, a[m, 12] / 1e6, a[m, 14] / 1e6,
                a[m, 7] / 1e6, a[m, 8] / 1e6, a[m, 9] / 1e6, a[m, 10] / 1e6,
                a[m, 3], a[m, 5], a[m, 4], 0.5 * a[m, 6]]
        o = np.argsort(r, kind="stable")
        for c, v in enumerate(cols):
            data[i, :, c] = median_bins(v[o], n_r)
        pj = parse_job_name(nm)
        params[i] = [pj.get("Q", np.nan), pj.get("k", np.nan), pj.get("alpha", np.nan),
                     pj.get("mu_repo", np.nan), pj.get("v", np.nan)]
    return data, params


def identify(y: np.ndarray, X0: np.ndarray, data: np.ndarray, params: np.ndarray
             ) -> Tuple[List[Dict], List[str]]:
    key = lambda v: tuple(np.round(v, 9))
    ri: Dict[tuple, int] = {}
    for i in range(len(params)):
        ri.setdefault(key(params[i]), i)
    miss = [j for j in range(X0.shape[0]) if key(X0[j]) not in ri]
    if miss:
        raise SystemExit(f"{len(miss)} наборов .pkl не нашлись в сырых данных — "
                         "проверьте, что каталог тот же")
    pi = np.array([ri[key(X0[j])] for j in range(X0.shape[0])])

    rows = []
    for p in range(y.shape[0]):
        A = y[p]
        best, second = None, None
        for c, name in enumerate(CANDIDATES):
            B = data[pi][:, :, c]
            d = float(np.abs(A - B).mean())
            corr = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
            hit = float(np.mean(np.abs(A - B) < 1e-6))
            rec = {"plane": p, "quantity": name, "mean_abs_diff": d,
                   "corr": corr, "bitwise": hit}
            if best is None or d < best["mean_abs_diff"]:
                second, best = best, rec
            elif second is None or d < second["mean_abs_diff"]:
                second = rec
        best["runner_up"] = second["quantity"]
        best["runner_up_diff"] = second["mean_abs_diff"]
        rows.append(best)
    found = [r["quantity"] for r in rows]
    missing = [c for c in TENSOR_STRESS if c not in found]
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="y_*.pkl")
    ap.add_argument("--xpkl", required=True, help="X_*.pkl")
    ap.add_argument("--raw", required=True, help="каталог с Vel_*/*.rpt")
    a = ap.parse_args()

    y = np.asarray(pickle.load(open(a.pkl, "rb")))
    X = np.asarray(pickle.load(open(a.xpkl, "rb")))
    print(f"{a.pkl}: {y.shape};  {a.xpkl}: {X.shape}")
    data, params = raw_table(a.raw, n_r=y.shape[-1])
    rows, missing = identify(y, X[0], data, params)

    print(f"\n{'плоскость':<10}{'что это':<14}{'mean|Δ|':>10}{'corr':>9}"
          f"{'побитово':>10}   второе место")
    print("─" * 78)
    for r in rows:
        print(f"{r['plane']:<10}{r['quantity']:<14}{r['mean_abs_diff']:10.4g}"
              f"{r['corr']:9.4f}{100*r['bitwise']:9.1f}%   "
              f"{r['runner_up']} ({r['runner_up_diff']:.4g})")
    if missing:
        print(f"\n⚠ В массиве ОТСУТСТВУЮТ компоненты тензора: {', '.join(missing)}")
        print("   Метрики и физические лоссы, ссылающиеся на них, некорректны.")
    else:
        print("\n✓ Все четыре компоненты тензора присутствуют.")
    order = [r["quantity"] for r in rows]
    want = list(TENSOR_STRESS)
    if sorted(order) == sorted(want):
        print(f"   Корректная перестановка: PLANE_ORDER = "
              f"{[order.index(w) for w in want]}")


if __name__ == "__main__":
    main()
