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
                          vega, vega_exact_land, vega_measured_land, vega_terms)

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
            dT_mean=fem["dT_mean_C"], dT_peak=fem["dT_peak_C"],
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

    print("\nдавление на пояске, МПа: принято решением / измерено в расчёте")
    for r in rows:
        print(f"  alpha {r['alpha']:2.0f}, mu {r['mu']:5.3f}:  {r['p_assumed']:6.1f} / "
              f"{r['p_measured']:6.1f}   отношение {r['p_assumed'] / r['p_measured']:.2f}")

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
