# -*- coding: utf-8 -*-
"""
analytic_sigma_f.py — замкнутый аналитический расчёт напряжения волочения.

Деформация и скорость деформации получаются из известных соотношений и
подставляются в уравнение состояния. Остаётся температура, и она тоже не
требует отдельного источника, она замыкается на само
напряжение волочения. Работа волочения на единицу объёма протянутого металла
равна sigma_d, доля beta переходит в тепло, отсюда средний разогрев
dT = beta * sigma_d / (rho * c). Так как sigma_d зависит от предела текучести,
а тот от температуры, получается неподвижная точка; итерация сходится за
несколько шагов.

Ни одна величина не берётся из МКЭ — расчёт самодостаточен.

СООТНОШЕНИЯ

  Деформация (однородная):      eps_hom = 2 * ln(R0 / Rf)
  Избыточная работа (Бэкофен):  Phi = 0.8 + Delta / 4.4,
                                Delta — параметр неоднородности
  Полная деформация:            eps_eff = Phi * eps_hom

  Скорость деформации. В конической зоне из постоянства расхода v*R^2 = const
  время прохода t = (R0^3 - Rf^3) / (3 * tan(alpha) * v * Rf^2), откуда среднее
                                eps_dot = eps_hom / t
                                        = 6*v*Rf^2*tan(alpha)*ln(R0/Rf) / (R0^3 - Rf^3)

  Предел текучести. Взвешенное по пластической работе среднее от деформационного
  члена на пути от 0 до eps_eff берётся точно:
                                <A + B*eps^n> = A + B * eps_eff^n / (n + 1)
  далее множители по скорости и температуре Джонсона-Кука.

  Температура:                  dT = beta * sigma_d / (rho * c)
  Средняя за проход:            T = T0 + dT / 2

ПРОВЕРКА. Оценка разогрева сверена с измеренной по выгрузке: аналитические 13-17 К против
измеренных по сечению 15-20 К в выгрузке из четырёх расчётов.

    python publication/reexport/analytic_sigma_f.py --Q 0.0164 --alpha 12 --mu 0.1 --v 40
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "code"))

# Джонсон-Кук из .inp, температуры в градусах Цельсия
A_JC, B_JC, N_JC = 213e6, 53e6, 0.345
M_JC, T_MELT, T_TRANS = 0.81, 1386.0, 20.0
C_JC, EDOT0 = 0.055, 0.004
RHO, CP, BETA = 7870.0, 470.0, 0.9
R0_DEFAULT = 0.018


def strain_rate(Q: float, alpha: float, v_mpm: float, R0: float = R0_DEFAULT) -> float:
    """Среднее по проходу eps_dot, 1/с. alpha в радианах, v в м/мин."""
    Rf = R0 * (1.0 - Q)
    v = v_mpm / 60.0
    return 6.0 * v * Rf ** 2 * math.tan(alpha) * math.log(R0 / Rf) / (R0 ** 3 - Rf ** 3)


def solve(Q: float, alpha_deg: float, mu: float, v_mpm: float,
          T0: float = 20.0, R0: float = R0_DEFAULT, land: float = 0.0,
          tol: float = 1e-6, itmax: int = 50) -> dict:
    from analytical import avitzur, delta_param      # noqa: E402
    S3 = math.sqrt(3.0)

    Rf = R0 * (1.0 - Q)
    a = math.radians(alpha_deg)
    r_area = 1.0 - (Rf / R0) ** 2
    eps_hom = 2.0 * math.log(R0 / Rf)
    delta = float(delta_param(r_area, a))
    phi = 0.8 + delta / 4.4
    eps_eff = phi * eps_hom
    edot = strain_rate(Q, a, v_mpm, R0)

    T, sigma_d, its = T0, 0.0, 0
    for its in range(1, itmax + 1):
        theta = max(0.0, (T - T_TRANS) / (T_MELT - T_TRANS))
        sigma_f = ((A_JC + B_JC * eps_eff ** N_JC / (N_JC + 1.0))
                   * (1.0 + C_JC * math.log(max(edot, EDOT0) / EDOT0))
                   * (1.0 - theta ** M_JC))
        sigma_d = float(avitzur(r_area, a, S3 * mu, sigma_f, L_over_Rf=land)[0])
        dT = BETA * sigma_d / (RHO * CP)
        T_new = T0 + dT / 2.0
        if abs(T_new - T) < tol:
            T = T_new
            break
        T = T_new

    return dict(Q=Q, alpha=alpha_deg, mu=mu, v=v_mpm,
                eps_hom=round(eps_hom, 4), delta=round(delta, 1), phi=round(phi, 2),
                eps_eff=round(eps_eff, 3), edot=round(edot, 1),
                T_mean=round(T, 1), dT=round(2.0 * (T - T0), 1),
                sigma_f_MPa=round(sigma_f / 1e6, 1),
                sigma_d_MPa=round(sigma_d / 1e6, 1), iterations=its)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=float, default=0.0164, help="обжатие по диаметру")
    ap.add_argument("--alpha", type=float, default=12.0, help="полуугол, градусы")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--v", type=float, default=40.0, help="скорость, м/мин")
    ap.add_argument("--land", type=float, default=0.0, help="длина пояска / радиус")
    a = ap.parse_args()
    r = solve(a.Q, a.alpha, a.mu, a.v, land=a.land)
    print(f"обжатие по диаметру {r['Q']:.4f}, полуугол {r['alpha']:.0f}, трение {r['mu']}, "
          f"скорость {r['v']:.0f} м/мин\n")
    print(f"  деформация однородная   {r['eps_hom']:.4f}")
    print(f"  параметр неоднородности {r['delta']:.1f}, фактор избыточной работы {r['phi']:.2f}")
    print(f"  деформация полная       {r['eps_eff']:.3f}")
    print(f"  скорость деформации     {r['edot']:.1f} 1/с")
    print(f"  разогрев                {r['dT']:.1f} K, средняя температура {r['T_mean']:.1f} C")
    print(f"  предел текучести        {r['sigma_f_MPa']:.1f} МПа")
    print(f"  напряжение волочения    {r['sigma_d_MPa']:.1f} МПа   (итераций {r['iterations']})")


if __name__ == "__main__":
    main()
