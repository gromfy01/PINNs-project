# -*- coding: utf-8 -*-
"""
verification_table.py — таблица сверки напряжения волочения для раздела верификации.

Собирает в одном месте всё, что сравнивается: напряжение волочения из расчёта,
три замкнутых решения и диагностику давления на пояске.

СТОЛБЦЫ

  МКЭ          drawing_force.py: среднее |RF2| по установившемуся окну, делённое на
               реально достигнутое сечение.
  Зибель       analytical.siebel, вариант A. Члена пояска в этом решении нет вообще.
  Авитцур      analytical.avitzur через фактор трения m = sqrt(3) mu, с пояском
               L/Rf = k. Перевод m = sqrt(3) mu верен, только если нормальное
               давление на контакте равно пределу текучести.
  Авитцур-Вега vega_avitzur.vega: то же решение в кулоновской записи, mu входит
               напрямую. Поясок замкнут условием текучести p = sigma_0 - sigma_x.
  то же при p  vega_avitzur.vega_measured_land: поясок при давлении, измеренном в
  из расчёта   расчёте, остальное без изменений. Это уже не предсказание, а
               диагностика: показывает, сколько из расхождения приходится на
               принятое давление на пояске.

Предел текучести — представительный, из уравнения состояния по историям расчёта
(flow_stress.py), один и тот же для всех столбцов: он входит множителем и сдвигает
все предсказания вместе.

    python publication/reexport/verification_table.py
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

from analytical import avitzur, delta_param, siebel                     # noqa: E402
from vega_avitzur import (bearing_pressure, mu_effective, mu_of_temperature,  # noqa: E402
                          sticking_limit, vega, vega_exact_land, vega_measured_land,
                          vega_terms)

from drawing_force import measure                                        # noqa: E402
from flow_stress import representative                                   # noqa: E402

R0 = 0.018
HISTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")

# четыре различных прогона: три при alpha = 12 и один при alpha = 8.
# Файлы семейства 2a_16 побайтово совпадают (ERRATA E-23), поэтому берётся один,
# а его коэффициент трения восстанавливается из данных, а не из имени.
RUNS = [
    "aisi_1020_2a_24_rd_150_cal_100_v_40_fric_025",
    "aisi_1020_2a_24_rd_150_cal_100_v_40_fric_050",
    "aisi_1020_2a_24_rd_150_cal_100_v_40_fric_0100",
    "aisi_1020_2a_16_red_150_cal_100_vel_40_fric_0100",
]


def build() -> list:
    rows = []
    for job in RUNS:
        path = os.path.join(HISTORY, job + ".rpt.gz")
        fem = measure(path)
        flow = representative(path)

        Rf = fem["Rf_mm"] / 1e3
        lnR = math.log(R0 / Rf)
        r_area = 1.0 - (Rf / R0) ** 2
        a = math.radians(fem["alpha"])
        mu = fem["mu_from_data"]
        k = fem["k"]
        s0 = flow["sigma_f_MPa"] * 1e6
        p_meas = fem["p_contact_MPa"] * 1e6

        t = vega_terms(lnR, a, mu, s0, k)
        rows.append(dict(
            alpha=fem["alpha"], mu=mu, k=k, Rf_mm=fem["Rf_mm"], lnR=lnR,
            delta=float(delta_param(r_area, a)),
            sigma_f=s0 / 1e6,
            fem=fem["sigma_d_MPa"],
            siebel=siebel(r_area, a, mu, s0) / 1e6,
            avitzur=avitzur(r_area, a, math.sqrt(3.0) * mu, s0, L_over_Rf=k)[0] / 1e6,
            avitzur_noland=avitzur(r_area, a, math.sqrt(3.0) * mu, s0)[0] / 1e6,
            vega=t["sigma_d"] / 1e6,
            vega_exact=vega_exact_land(lnR, a, mu, s0, k) / 1e6,
            vega_pmeas=vega_measured_land(lnR, a, mu, s0, k, p_meas) / 1e6,
            p_assumed=bearing_pressure(lnR, a, mu, s0, k) / 1e6,
            p_measured=fem["p_contact_MPa"],
            mu_eff=mu_effective(fem["sigma_d_MPa"] * 1e6, lnR, a, s0, k),
            share_ideal=t["ideal"] / t["sigma_d"],
            share_shear=t["shear"] / t["sigma_d"],
            share_cone=t["cone"] / t["sigma_d"],
            share_land=t["land"] / t["sigma_d"],
            # давление на пояске падает по его длине от p на входе до p на выходе;
            # уравнение (1) линеаризовано по входному значению
            p_along=(bearing_pressure(lnR, a, mu, s0, k)
                     + (s0 - t["sigma_d"])) / 2.0 / 1e6,
            tau_assumed=mu * bearing_pressure(lnR, a, mu, s0, k) / 1e6,
            tau_measured=mu * p_meas / 1e6,
            k_shear=s0 / math.sqrt(3.0) / 1e6,
            sticks=not sticking_limit(mu, bearing_pressure(lnR, a, mu, s0, k), s0),
            T_pass=20.0 + fem["dT_mean_C"] / 2.0,
            dT_mean=fem["dT_mean_C"], dT_peak=fem["dT_peak_C"],
            sigma_f_heat=(fem["sigma_f_heat_min_MPa"], fem["sigma_f_heat_max_MPa"]),
        ))
    return rows


def mape(rows, key):
    return 100.0 * sum(abs(r[key] - r["fem"]) / r["fem"] for r in rows) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="выдать тело таблицы LaTeX")
    a = ap.parse_args()
    rows = build()

    cols = [("МКЭ", "fem"), ("Зибель", "siebel"), ("Авитцур", "avitzur"),
            ("Авитцур-Вега", "vega"), ("то же при p из расчёта", "vega_pmeas")]
    hdr = f"{'alpha':>5} {'mu':>6} {'sigma_f':>8} " + " ".join(f"{n:>22}" for n, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['alpha']:5.0f} {r['mu']:6.3f} {r['sigma_f']:8.1f} "
              + " ".join(f"{r[k]:22.1f}" for _, k in cols))
    print("-" * len(hdr))
    print(f"{'среднее отклонение':>21} "
          + " ".join(f"{'':>22}" if k == "fem" else f"{mape(rows, k):21.1f}%" for _, k in cols))

    print("\nдавление на пояске, МПа: принято решением (на входе / в среднем по длине) "
          "против измеренного")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  {r['p_assumed']:6.1f} / "
              f"{r['p_along']:6.1f}  против {r['p_measured']:6.1f}   отношение по входу "
              f"{r['p_assumed'] / r['p_measured']:.2f}, по средней {r['p_along'] / r['p_measured']:.2f}")

    print("\nнаклон d sigma_d / d mu при alpha = 12, МПа на единицу mu")
    lo = next(r for r in rows if r["alpha"] == 12 and r["mu"] < 0.03)
    hi = next(r for r in rows if r["alpha"] == 12 and r["mu"] > 0.09)
    dmu = hi["mu"] - lo["mu"]
    for name, key in cols:
        print(f"  {name:24s} {(hi[key] - lo[key]) / dmu:7.0f}")

    print("\nкасательное напряжение на пояске против предела текучести на сдвиг, МПа")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  при принятом давлении "
              f"{r['tau_assumed']:5.1f}, при измеренном {r['tau_measured']:5.1f}, "
              f"k = {r['k_shear']:5.1f}" + ("  ПРИЛИПАНИЕ" if r["sticks"] else ""))

    print("\nтемпературная поправка уравнения (2) при средней за проход температуре")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  T = {r['T_pass']:4.1f} C, "
              f"(T/20)^2.68 = {(r['T_pass'] / 20.0) ** 2.68:4.2f}, "
              f"(T/15.6)^2.68 = {(r['T_pass'] / 15.6) ** 2.68:4.2f}")

    print("\nпредел текучести: из уравнения состояния по путям и независимо из тепла по ядру")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  {r['sigma_f']:5.1f} МПа против "
              f"{r['sigma_f_heat'][0]:.0f}-{r['sigma_f_heat'][1]:.0f} МПа из тепла")

    print("\nдоли слагаемых уравнения (1) и обращение по трению")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  идеальная {r['share_ideal']:.0%}, "
              f"сдвиг {r['share_shear']:.0%}, конус {r['share_cone']:.0%}, "
              f"поясок {r['share_land']:.0%};  mu по МКЭ {r['mu_eff']:.4f} "
              f"({r['mu_eff'] / r['mu']:.2f} от заданного)")

    print("\nпараметр неоднородности и разогрев")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  Delta {r['delta']:5.1f}, "
              f"dT средний {r['dT_mean']:5.1f} C, пиковый {r['dT_peak']:5.1f} C")

    print("\nлинеаризация пояска против точного интеграла, МПа")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  {r['vega']:6.1f} / {r['vega_exact']:6.1f}")

    if a.latex:
        print("\n% --- тело таблицы ---")
        for r in rows:
            print(f"${r['alpha']:.0f}^\\circ$ & {r['mu']:.3f} & {r['fem']:.1f} & {r['siebel']:.1f} & "
                  f"{r['avitzur']:.1f} & {r['vega']:.1f} & {r['vega_pmeas']:.1f} \\\\")
        print("\\midrule")
        print("\\multicolumn{3}{@{}l}{mean absolute deviation} & "
              + " & ".join(f"{mape(rows, k):.1f}\\,\\%" for _, k in cols[1:]) + " \\\\")


if __name__ == "__main__":
    main()
