# -*- coding: utf-8 -*-
"""
sec31_numbers.py — все числа раздела 3.1 по таблице прогонов verification_recalc.py.

Читает results/recalc_2026-09-28.json (по прогону на строку) и, для величин по
историям узлов (деформация, скорость деформации, температура, разброс предела
текучести по сечению), сами отчёты выгрузки. Печатает каждую величину, которую
цитирует текст, с тем же округлением, — так текст сверяется с данными целиком.

    python publication/reexport/sec31_numbers.py DIR [--json results/recalc_2026-09-28.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

from vega_avitzur import vega, vega_exact_land, vega_terms   # noqa: E402

import analytic_sigma_f as A                                 # noqa: E402
from flow_stress import representative                       # noqa: E402

R0 = 0.018
HERE = os.path.dirname(os.path.abspath(__file__))


def _r(x, nd):
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def rng(xs, nd=1):
    return f"{_r(min(xs), nd)}–{_r(max(xs), nd)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=os.path.join(HERE, "results", "recalc_2026-09-28.json"))
    a = ap.parse_args()
    R = json.load(open(a.json))
    q10 = [r for r in R if abs(r["Q"] - 0.10) < 1e-9]
    q15 = [r for r in R if abs(r["Q"] - 0.15) < 1e-9]
    dev = lambda rows, key: 100.0 * float(np.mean([r[key] / r["fem"] - 1.0 for r in rows]))  # noqa: E731
    P = print

    P(f"прогонов: всего {len(R)}, при 10 % {len(q10)}, при 15 % {len(q15)}")
    P(f"разброс реакции на окне, %: {rng([r['spread_pct'] for r in R])}")
    t = [r for r in R if r['alpha'] == 12 and r['k'] == 0.5 and r['v'] == 10 and r['mu'] == 0.05 and r in q10][0]
    P(f"типичный прогон (10 %, 12°, k 0.5, 10 м/мин, mu 0.05): {t['fem']:.1f} МПа")
    P(f"обжатие достигнутое, %: 10 → {rng([100*r['Q_real'] for r in q10])}, 15 → {rng([100*r['Q_real'] for r in q15])}")
    P(f"L/Rf / k при k > 0: {rng([r['L_over_Rf']/r['k'] for r in R if r['k'] > 0], 2)}")

    P("\nпредел текучести")
    P(f"  по историям, МПа: {rng([r['sigma_f'] for r in R], 0)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fl = [representative(f) for f in sorted(glob.glob(os.path.join(a.dir, "*.rpt")))]
    P(f"  деформация на оси {rng([x['eps_axis'] for x in fl], 2)}, на поверхности (узел 151) {rng([x['eps_surface'] for x in fl], 2)},"
      f" максимум по сечению {rng([x['eps_max'] for x in fl], 2)}")
    P(f"  скорость деформации, 95-й процентиль по узлу, наибольший в прогоне, 1/с: {rng([x['edot_max'] for x in fl], 1)}")
    P(f"  температура: максимум по сечению {rng([x['T_max'] for x in fl], 0)} °C, на поверхности {rng([x['T_surface'] for x in fl], 0)} °C")
    P(f"  осреднённый предел текучести узлов: от {_r(min(x['sigma_f_min_MPa'] for x in fl), 0)} до {_r(max(x['sigma_f_max_MPa'] for x in fl), 0)} МПа по всем прогонам,"
      f" внутри прогона разброс не более {_r(max(x['sigma_f_max_MPa'] - x['sigma_f_min_MPa'] for x in fl), 0)} МПа")

    P("\nзамкнутая оценка")
    sol = [A.solve(r["Q_real"], r["alpha"], r["mu"], r["v"]) for r in R]
    P(f"  sigma_f, МПа: {rng([s['sigma_f_MPa'] for s in sol], 0)}; итераций не более {max(s['iterations'] for s in sol)}")
    P(f"  Delta {rng([s['delta'] for s in sol])}, eps_eff {rng([s['eps_eff'] for s in sol], 2)}, edot {rng([s['edot'] for s in sol])} 1/с")
    alt = [A.solve(r["Q_real"], r["alpha"], r["mu"], r["v"], phi_alt=True) for r in R]
    P(f"  Phi 0.8+D/4.4 против 0.88+0.12D: отношение {rng([s['phi']/t['phi'] for s, t in zip(sol, alt)])},"
      f" сошедшийся sigma_f {rng([100*abs(s['sigma_f_MPa']/t['sigma_f_MPa']-1) for s, t in zip(sol, alt)])} %,"
      f" разогрев {rng([100*abs(s['dT']/t['dT']-1) for s, t in zip(sol, alt)])} %")

    P("\nпоясок")
    lin = []
    for r in R:
        if r["k"] == 0:
            continue
        Rf = r["Rf_mm"] / 1e3
        lnR, al, s0 = math.log(R0 / Rf), math.radians(r["alpha"]), r["sigma_f"] * 1e6
        lin.append(vega_exact_land(lnR, al, r["mu"], s0, r["L_over_Rf"]) / vega(lnR, al, r["mu"], s0, r["L_over_Rf"]) - 1)
    P(f"  точный интеграл выше знаменателя не более чем на {100*max(lin):.1f} %")
    sp, inc_a, inc_v = [], [], []
    for key in {(r["alpha"], r["mu"], r["v"]) for r in q10}:
        g = [r for r in q10 if (r["alpha"], r["mu"], r["v"]) == key]
        f = [r["fem"] for r in g]
        sp.append(100 * (max(f) / min(f) - 1))
        r0 = [r for r in g if r["k"] == 0]
        r1 = [r for r in g if r["k"] == 1.0]
        if r0 and r1:
            inc_a.append(100 * (r1[0]["avitzur"] / r0[0]["avitzur"] - 1))
            inc_v.append(100 * (r1[0]["vega"] / r0[0]["vega"] - 1))
    P(f"  МКЭ по k: размах не более {max(sp):.2f} %")
    P(f"  члены пояска от k = 0 к 1: Авитцур +{rng(inc_a, 0)} %, Вега +{rng(inc_v, 0)} %")
    P(f"  доля осевой силы на пояске при k > 0: не более {100*max(r['land_share_axial'] for r in R if r['k'] > 0):.2f} %")
    k1 = [r for r in R if r["k"] == 1.0]
    P(f"  с членами пояска: Авитцур {dev(R,'avitzur'):.1f} %, Вега {dev(R,'vega'):.1f} %; при k = 1: {dev(k1,'avitzur'):.1f} / {dev(k1,'vega'):.1f} %")
    P(f"  mu из данных: прогонов с поясом {sum(1 for r in R if r['k'] > 0)}, отношение к заданному {rng([r['mu_from_data']/r['mu'] for r in R if r['k'] > 0], 3)}")

    P("\nсверка без членов пояска")
    for key, name in (("siebel", "Зибель"), ("vega_noland", "Coulomb"), ("avitzur_noland", "friction-factor")):
        mins = min(100 * (r[key] / r["fem"] - 1) for r in R)
        P(f"  {name:16s} среднее {dev(R,key):.1f} %, все выше МКЭ: {mins > 0} (мин. {mins:.1f} %);"
          f" 15 %: {dev(q15,key):.1f} %; по mu (10 %): "
          + " / ".join(f"{dev([r for r in q10 if r['mu']==m],key):.1f}" for m in (0.025, 0.05, 0.1))
          + "; по alpha: " + " / ".join(f"{dev([r for r in q10 if r['alpha']==al],key):.1f}" for al in (8, 12, 16))
          + "; по v: " + " / ".join(f"{dev([r for r in q10 if r['v']==v],key):.2f}" for v in (10, 20)))
    for al in (8, 12, 16):
        g = [r for r in q10 if r["alpha"] == al and r["v"] == 10]
        mus = sorted({r["mu"] for r in g})
        y = {key: [np.mean([r[key] for r in g if r["mu"] == m]) for m in mus] for key in ("fem", "siebel")}
        e0 = {key: np.polyfit(mus, y[key], 1)[1] for key in y}
        P(f"  alpha {al}: при mu = 0 МКЭ {e0['fem']:.0f}, Зибель {e0['siebel']:.0f} МПа")
    for al in (8, 12, 16):
        sl = {key: [] for key in ("fem", "siebel", "avitzur_noland", "vega_noland")}
        for v in (10, 20):
            for k in (0, 0.1, 0.3, 0.5, 0.75, 1.0):
                g = {r["mu"]: r for r in q10 if r["alpha"] == al and r["v"] == v and r["k"] == k}
                if 0.025 in g and 0.1 in g:
                    for key in sl:
                        sl[key].append((g[0.1][key] - g[0.025][key]) / 0.075)
        P(f"  alpha {al}: d sigma/d mu, МПа: " + ", ".join(f"{key} {np.mean(v):.0f}" for key, v in sl.items())
          + "; ниже МКЭ на " + " / ".join(f"{100*(1-np.mean(sl[key])/np.mean(sl['fem'])):.0f}" for key in ("siebel", "avitzur_noland", "vega_noland")) + " %")
    for al in (8, 12, 16):
        P(f"  alpha {al}: давление на конусе / sigma_f {rng([r['p_cone']/r['sigma_f'] for r in q10 if r['alpha']==al], 2)}")
    P(f"  15 %: давление на конусе / sigma_f {rng([r['p_cone']/r['sigma_f'] for r in q15], 2)}")
    for name, rows in (("10 %", q10), ("15 %", q15), ("все", R)):
        vv = [r2["fem"] / r1["fem"] - 1 for r1 in rows for r2 in rows if r1["v"] == 10 and r2["v"] == 20
              and (r1["Q"], r1["alpha"], r1["k"], r1["mu"]) == (r2["Q"], r2["alpha"], r2["k"], r2["mu"])]
        P(f"  v 10 → 20 ({name}): МКЭ +{rng([100*x for x in vv])} %")
    jc = [100 * A.C_JC * math.log(2.0) / (1 + A.C_JC * math.log(max(e, A.EDOT0) / A.EDOT0)) for x in fl for e in (x['edot_min'], x['edot_max'])]
    P(f"  член скорости Джонсона-Кука при удвоении скорости деформации: +{rng(jc)} %")
    P(f"  15 %: Зибель по mu: " + " / ".join(f"{dev([r for r in q15 if r['mu']==m],'siebel'):.1f}" for m in (0.025, 0.05, 0.1)))
    le = [r for r in R if r["vega_noland"] <= r["siebel"]]
    tags = sorted({"%.0f%%/%.0f/%s" % (100 * r["Q"], r["alpha"], r["mu"]) for r in le})
    P(f"  Coulomb не выше Зибеля в {len(le)} прогонах: " + ", ".join(tags))
    P(f"  friction-factor выше Зибеля во всех: {all(r['avitzur_noland'] > r['siebel'] for r in R)}")

    P("\nразогрев")
    P(f"  средний по сечению {rng([r['dT_mean'] for r in R], 0)} °C, у поверхности {rng([r['dT_peak'] for r in R], 0)} °C")
    P(f"  энергетический баланс / МКЭ: +{rng([100*(r['dT_energy']/r['dT_mean']-1) for r in R], 0)} %")
    P(f"  к концу деформации: средний {rng([r['dT_def'] for r in R], 0)} °C, баланс выше на {rng([100*(r['dT_energy']/r['dT_def']-1) for r in R], 0)} %")
    P(f"  замкнутая оценка / МКЭ: +{rng([100*(r['dT_closed']/r['dT_mean']-1) for r in R], 0)} %")
    sd = [100 * (s["sigma_d_MPa"] / r["fem"] - 1) for s, r in zip(sol, R)]
    P(f"  sigma_d замкнутой оценки выше МКЭ на {rng(sd, 0)} %")
    Tp = [20 + r["dT_mean"] / 2 for r in R]
    P(f"  средняя за проход {rng(Tp, 0)} °C; (T/20)^2.68 {rng([(t/20)**2.68 for t in Tp])}, (T/15.6)^2.68 {rng([(t/15.6)**2.68 for t in Tp])}")

    P("\nобласть применимости")
    sh = []
    for r in R:
        Rf = r["Rf_mm"] / 1e3
        t0 = vega_terms(math.log(R0 / Rf), math.radians(r["alpha"]), r["mu"], r["sigma_f"] * 1e6, 0.0)
        sh.append(100 * t0["ideal"] / t0["sigma_d"])
    P(f"  Delta {rng([r['delta'] for r in R])}; по углу при 10 %: " + ", ".join(f"{al}: {rng([r['delta'] for r in q10 if r['alpha']==al])}" for al in (8, 12, 16))
      + f"; при 15 %: {rng([r['delta'] for r in q15])}; Delta > 3 в {sum(r['delta'] > 3 for r in R)} прогонах")
    P(f"  идеальная работа {rng(sh, 0)} % предсказания (Coulomb, без пояска)")
    allv = [100 * (r[k] / r["fem"] - 1) for r in R for k in ("siebel", "avitzur_noland", "vega_noland")]
    P(f"  отклонения трёх соотношений без пояска: {rng(allv, 1)} %")

    P("\nтаблица 6 (v = 10, среднее по k)")
    for Q, als in ((0.10, (8, 12, 16)), (0.15, (8,))):
        for al in als:
            for m in (0.025, 0.05, 0.1):
                g = [r for r in R if abs(r["Q"] - Q) < 1e-9 and r["alpha"] == al and r["mu"] == m and r["v"] == 10]
                f = lambda key: np.mean([r[key] for r in g])  # noqa: E731
                P(f"  {Q*100:.0f} & {al} & {m:.3f} & {f('siebel'):.1f} & {f('avitzur_noland'):.1f} & {f('vega_noland'):.1f} & {f('fem'):.1f}"
                  f"   (n {len(g)}, размах МКЭ {max(r['fem'] for r in g)-min(r['fem'] for r in g):.1f})")
    P(f"  итог: {dev(R,'siebel'):.1f} / {dev(R,'avitzur_noland'):.1f} / {dev(R,'vega_noland'):.1f} %")

    # в статье строки таблицы 6 — по одному прогону: k = 0.5 при 10 % и k = 0 при 15 %, v = 10
    P("\nтаблица 6 (v = 10, k = 0.5 при 10 %, k = 0 при 15 %)")
    for Q, k6, als in ((0.10, 0.5, (8, 12, 16)), (0.15, 0.0, (8,))):
        for al in als:
            for m in (0.025, 0.05, 0.1):
                g = [r for r in R if abs(r["Q"] - Q) < 1e-9 and r["alpha"] == al and r["mu"] == m
                     and r["v"] == 10 and r["k"] == k6]
                assert len(g) == 1, (Q, al, m, len(g))
                r = g[0]
                P(f"  {Q*100:.0f} & {al} & {k6:g} & {m:.3f} & " + " & ".join(_r(r[key], 1) for key in
                  ("siebel", "avitzur_noland", "vega_noland", "fem")) + f"   ({r['job']})")
    sl = {key: [] for key in ("fem", "siebel", "avitzur_noland", "vega_noland")}
    g = {r["mu"]: r for r in q10 if r["alpha"] == 12 and r["v"] == 10 and r["k"] == 0.5}
    for key in sl:
        sl[key] = (g[0.1][key] - g[0.025][key]) / 0.075
    P("  d sigma/d mu при 12°, k = 0.5, v = 10, МПа: " + ", ".join(f"{key} {_r(v, 0)}" for key, v in sl.items()))

    P("\nтаблица 7 (v = 10, среднее по k)")
    for Q, als in ((0.10, (8, 12, 16)), (0.15, (8,))):
        for al in als:
            for m in (0.025, 0.05, 0.1):
                g = [r for r in R if abs(r["Q"] - Q) < 1e-9 and r["alpha"] == al and r["mu"] == m and r["v"] == 10]
                f = lambda key: np.mean([r[key] for r in g])  # noqa: E731
                P(f"  {Q*100:.0f} & {al} & {m:.3f} & {f('dT_closed'):.1f} & {f('dT_energy'):.1f} & {f('dT_mean'):.1f} & {20 + f('dT_mean')/2:.1f}")


if __name__ == "__main__":
    main()
