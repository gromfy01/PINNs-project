# -*- coding: utf-8 -*-
"""
flow_stress.py — представительный предел текучести для сверки с аналитикой.

Предел текучести нельзя брать одним измеренным числом: он следует из уравнения
состояния и определён только при конкретных деформации, скорости деформации и
температуре.

Для энергетических (верхнеграничных) решений правильная представительная
величина — предел текучести, осреднённый ВДОЛЬ ПУТИ ДЕФОРМАЦИИ и взвешенный по
пластической работе:

    sigma_f = ( интеграл sigma(eps, eps_dot, T) d(eps_p) ) / ( интеграл d(eps_p) )

по истории каждой материальной точки, затем осреднение по сечению с весом 2*pi*r*dr.
Именно эта величина, умноженная на полную деформацию, даёт работу, которую и
оценивают решения Зибеля и Авитцура.

Все три аргумента берутся ИЗ РАСЧЁТА, а не назначаются:
    eps      — история PEEQ в узле,
    eps_dot  — производная PEEQ по времени,
    T        — история TEMP в том же узле.

Карта материала (из .inp расчёта):
    *Plastic, hardening=JOHNSON COOK   A = 213 МПа, B = 53 МПа, n = 0.345,
                                       m = 0.81, T_melt = 1386 C, T_transition = 20 C
    *Rate Dependent, type=JOHNSON COOK C = 0.055, eps_dot_0 = 0.004 1/с

ВАЖНО про температуру. Расчёт идёт в градусах Цельсия, стартовая температура
ровно 20.0 во всех узлах, и температура перехода равна ей же. Поэтому термический
множитель работает с самого начала. Показатель m = 0.81 меньше единицы, из-за
чего приведённая температура в степени m много больше самой приведённой
температуры: нагрев на 75 K даёт theta = 0.055, но theta^m = 0.095, то есть
разупрочнение 9.5 %, а не 5 %. Пренебрегать этим членом нельзя.

    python publication/reexport/flow_stress.py publication/reexport/history/*.rpt.gz
"""
from __future__ import annotations

import argparse
import numpy as np

from read_history import parse, job_params

A_JC, B_JC, N_JC = 2.13e8, 5.3e7, 0.345
M_JC, T_MELT, T_TRANS = 0.81, 1386.0, 20.0   # градусы Цельсия, уточнено 13.09.2026
C_JC, EDOT0 = 0.055, 0.004


def johnson_cook(eps: np.ndarray, edot: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Предел текучести по карте из .inp. Температура в единицах модели."""
    theta = np.clip((temp - T_TRANS) / (T_MELT - T_TRANS), 0.0, 1.0)
    return ((A_JC + B_JC * np.maximum(eps, 0.0) ** N_JC)
            * (1.0 + C_JC * np.log(np.maximum(edot, EDOT0) / EDOT0))
            * (1.0 - theta ** M_JC))


def representative(path: str) -> dict:
    names, M = parse(path)
    t = M[:, 0]
    i_peeq = [i for i, n in enumerate(names) if n.startswith("PEEQ")]
    i_temp = [i for i, n in enumerate(names) if n.startswith("TEMP")]
    i_r = [i for i, n in enumerate(names) if "COORD:COOR1" in n.replace(" ", "")][1:]

    peeq, temp = M[:, i_peeq], M[:, i_temp]
    radius = np.array([np.nanmax(M[:, j]) for j in i_r])[:peeq.shape[1]]

    sig_node, w_node, eps_f, ed95, t_max = [], [], [], [], []
    for k in range(peeq.shape[1]):
        ok = ~np.isnan(peeq[:, k])
        tt, pe = t[ok], peeq[ok, k]
        if pe.size < 20:
            continue
        col = temp[:, k] if k < temp.shape[1] else np.full(t.size, 20.0)
        ok_t = ~np.isnan(col)
        tk = np.interp(tt, t[ok_t], col[ok_t]) if ok_t.any() else np.full(pe.size, 20.0)

        d_eps = np.diff(pe)
        d_t = np.diff(tt)
        edot = np.where(d_t > 0, d_eps / np.maximum(d_t, 1e-12), 0.0)
        flowing = d_eps > 1e-6            # только шаги, на которых идёт течение
        if flowing.sum() < 5:
            continue
        sig = johnson_cook(pe[:-1][flowing], edot[flowing], tk[:-1][flowing])
        sig_node.append(float(np.sum(sig * d_eps[flowing]) / np.sum(d_eps[flowing])))
        w_node.append(max(float(radius[k]), 1e-4))    # вес 2*pi*r*dr
        eps_f.append(float(pe[-1]))
        ed95.append(float(np.percentile(edot[flowing], 95)))
        t_max.append(float(np.nanmax(tk)))

    s = np.array(sig_node)
    w = np.array(w_node) / np.sum(w_node)
    return dict(job=path.split("/")[-1].replace(".rpt.gz", "").replace(".rpt", ""),
                **job_params(path.split("/")[-1].replace(".rpt.gz", "").replace(".rpt", "")),
                sigma_f_MPa=round(float(np.sum(s * w)) / 1e6, 1),
                sigma_f_min_MPa=round(float(s.min()) / 1e6, 1),
                sigma_f_max_MPa=round(float(s.max()) / 1e6, 1),
                eps_axis=round(min(eps_f), 3), eps_surface=round(max(eps_f), 3),
                edot_min=round(min(ed95), 1), edot_max=round(max(ed95), 1),
                T_max=round(max(t_max), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    hdr = f"{'задание':52s} {'mu':>6} {'sigma_f':>8} {'разброс':>13} {'eps пов.':>9} {'eps_dot':>12} {'T max':>6}"
    print(hdr)
    print("-" * len(hdr))
    for f in a.files:
        r = representative(f)
        print(f"{r['job']:52s} {r['mu']:6.3f} {r['sigma_f_MPa']:8.1f} "
              f"{r['sigma_f_min_MPa']:5.0f}-{r['sigma_f_max_MPa']:<7.0f} {r['eps_surface']:9.3f} "
              f"{r['edot_min']:5.1f}-{r['edot_max']:<6.1f} {r['T_max']:6.1f}")


if __name__ == "__main__":
    main()
