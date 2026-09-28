# -*- coding: utf-8 -*-
"""
verification_recalc.py — сверка напряжения волочения по выгрузке 28.09.2026.

Выгрузка: 112 прогонов, истории RF, COORD, PEEQ и TEMP в том же формате, что и
прежние четыре. Обжатие 10 % по диаметру (106 прогонов) и 15 % (6 прогонов при
alpha = 8, k = 0), полуугол 8, 12 и 16 градусов, k от 0 до 1, скорость 10 и 20 м/мин,
трение 0.025, 0.05 и 0.1. Модель считает в кельвинах (старт 293.15 К); разогрев от
этого не зависит, а в карту материала температура подаётся в градусах Цельсия
(flow_stress.py).

Для каждого прогона считается то же, что в verification_table.py для прежних
четырёх: напряжение волочения из расчёта (среднее |RF2| по установившемуся окну,
найденному по самому сигналу, на реально достигнутое сечение), три замкнутых решения
с одним и тем же представительным пределом текучести из уравнения состояния, давление
на пояске, коэффициент трения из данных, разогрев и замкнутая оценка разогрева.

    python publication/reexport/verification_recalc.py DIR [--json out.json] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

from analytical import avitzur, delta_param, siebel                     # noqa: E402
from vega_avitzur import bearing_pressure, vega_measured_land, vega_terms  # noqa: E402

from analytic_sigma_f import solve as closed_estimate                   # noqa: E402
from drawing_force import measure                                        # noqa: E402
from flow_stress import representative                                   # noqa: E402

R0 = 0.018
S3 = math.sqrt(3.0)


def one(path: str) -> dict:
    fem = measure(path, window="auto")
    flow = representative(path)
    Rf = fem["Rf_mm"] / 1e3
    lnR = math.log(R0 / Rf)
    r_area = 1.0 - (Rf / R0) ** 2
    a = math.radians(fem["alpha"])
    mu, k = fem["mu_label"], fem["k"]
    s0 = flow["sigma_f_MPa"] * 1e6
    t = vega_terms(lnR, a, mu, s0, k)
    p_meas = fem["p_contact_MPa"] * 1e6
    Q_real = 1.0 - Rf / R0
    ce = closed_estimate(Q_real, fem["alpha"], mu, fem["v"])
    return dict(
        job=fem["job"], Q=fem["Q"], Q_real=round(Q_real, 4), alpha=fem["alpha"], k=k,
        v=fem["v"], mu=mu, mu_from_data=fem["mu_from_data"],
        window=fem["window_mm"], window_how=fem["window_how"],
        travel_total_mm=fem["travel_total_mm"], spread_pct=fem["spread_pct"],
        Rf_mm=fem["Rf_mm"], delta=round(float(delta_param(r_area, a)), 2),
        sigma_f=flow["sigma_f_MPa"],
        fem=fem["sigma_d_MPa"],
        siebel=round(float(siebel(r_area, a, mu, s0)) / 1e6, 1),
        avitzur=round(float(avitzur(r_area, a, S3 * mu, s0, L_over_Rf=k)[0]) / 1e6, 1),
        vega=round(t["sigma_d"] / 1e6, 1),
        vega_pmeas=round(vega_measured_land(lnR, a, mu, s0, k, p_meas) / 1e6, 1),
        share_ideal=round(t["ideal"] / t["sigma_d"], 3),
        share_shear=round(t["shear"] / t["sigma_d"], 3),
        share_cone=round(t["cone"] / t["sigma_d"], 3),
        share_land=round(t["land"] / t["sigma_d"], 3),
        p_assumed=round(bearing_pressure(lnR, a, mu, s0, k) / 1e6, 1) if k > 0 else None,
        p_measured=fem["p_contact_MPa"],
        dT_mean=fem["dT_mean_C"], dT_peak=fem["dT_peak_C"],
        dT_closed=ce["dT"], sigma_f_closed=ce["sigma_f_MPa"],
        sigma_f_heat=(fem["sigma_f_heat_min_MPa"], fem["sigma_f_heat_max_MPa"]),
        T_init=fem["T_init"],
    )


def dev(rows, key):
    return 100.0 * float(np.mean([abs(r[key] - r["fem"]) / r["fem"] for r in rows]))


def bias(rows, key):
    return 100.0 * float(np.mean([(r[key] - r["fem"]) / r["fem"] for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.rpt")) + glob.glob(os.path.join(a.dir, "*.rpt.gz")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = [one(f) for f in files]

    keys = ["siebel", "avitzur", "vega"]
    print(f"прогонов: {len(rows)}; окно по сигналу у {sum(r['window_how'] == 'auto' for r in rows)}")
    print(f"{'группа':34s} {'n':>4} " + " ".join(f"{k:>16}" for k in keys) + "   (среднее |откл.| / смещение, %)")

    def line(name, sub):
        if not sub:
            return
        print(f"{name:34s} {len(sub):4d} " + " ".join(
            f"{dev(sub, k):7.1f} /{bias(sub, k):+6.1f}" for k in keys))

    line("все", rows)
    for Q in sorted({r["Q"] for r in rows}):
        line(f"Q = {Q:.0%}", [r for r in rows if r["Q"] == Q])
    q10 = [r for r in rows if abs(r["Q"] - 0.10) < 1e-9]
    for al in sorted({r["alpha"] for r in q10}):
        line(f"Q 10 %, alpha = {al:.0f}", [r for r in q10 if r["alpha"] == al])
    for k in sorted({r["k"] for r in q10}):
        line(f"Q 10 %, k = {k}", [r for r in q10 if r["k"] == k])
    for mu in sorted({r["mu"] for r in q10}):
        line(f"Q 10 %, mu = {mu}", [r for r in q10 if r["mu"] == mu])
    for v in sorted({r["v"] for r in q10}):
        line(f"Q 10 %, v = {v}", [r for r in q10 if r["v"] == v])

    if a.json:
        json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
