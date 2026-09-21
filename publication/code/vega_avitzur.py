# -*- coding: utf-8 -*-
"""
vega_avitzur.py — решение Авитцура в кулоновской записи (форма Веги) и
температурная поправка на трение.

Зачем отдельный модуль. В analytical.py решение Авитцура записано через ФАКТОР
трения m (касательное напряжение на контакте tau = m * sigma_0 / sqrt(3)). В МКЭ
задан КОЭФФИЦИЕНТ трения по Кулону mu, и перевод m = sqrt(3) * mu верен только
если нормальное давление на контакте равно пределу текучести. Ниже — запись, в
которой mu входит напрямую, поэтому никакого соглашения о переводе не нужно.

УРАВНЕНИЕ (1) [Vega, Haddi, Imad, 2009]

    sigma_d           sigma_b/sigma_0 + 2 f(a) ln(Ri/Rf)
    ------- = -------------------------------------------------------------
    sigma_0                      1 + 2 mu P/Rf

              + (2/sqrt3) ( a/sin^2 a - ctg a )
              + 2 mu [ ctg a ( 1 - sigma_b/sigma_0 - ln(Ri/Rf) ) ln(Ri/Rf) + P/Rf ]
                                     -- всё это в том же числителе --

  sigma_b — противонатяжение (у нас 0), P — длина пояска, f(a) — функция угла из
  сферического поля скоростей Авитцура (в самой статье Веги не определена, берётся
  из analytical.f_avitzur).

  Знаменатель 1 + 2 mu P/Rf — не «поправка», а следствие принятого на пояске
  давления. Осевой баланс на цилиндрическом пояске даёт d sigma_x / dz = 2 mu p / Rf,
  а условие текучести при таком напряжённом состоянии — p = sigma_0 - sigma_x,
  откуда точное решение

      sigma_x(P) = sigma_0 - ( sigma_0 - sigma_x(0) ) exp( -2 mu P/Rf ),

  а его линеаризация по 2 mu P/Rf — ровно числитель со знаменателем уравнения (1).
  Значит уравнение (1) закладывает на пояске давление sigma_0 - sigma_x. Это
  проверяемое утверждение, и bearing_pressure() его считает.

УРАВНЕНИЕ (2), собственно модификация

      mu = mu_0 ( T / T_0 ) ** m,   mu_0 = 0.0125, m = 2.68, T_0 = 15.6 C

  Отношение температур — в градусах ЦЕЛЬСИЯ: у Веги T меняется от 26.5 до 66 C
  при T_0 = 15.6 C, и абсцисса его рис. 4 идёт от 1.7 до 4.25. В Кельвинах
  отношение было бы 1.04-1.17 и калибровка mu_0, m потеряла бы смысл.

    python publication/code/vega_avitzur.py
"""
from __future__ import annotations

import math

from analytical import f_avitzur

# константы температурной зависимости трения из статьи (медь, масло, 1-7 м/с)
MU0_VEGA, M_VEGA, T0_VEGA = 0.0125, 2.68, 15.6


def vega_terms(lnR: float, alpha: float, mu: float, sigma_0: float,
               P_over_Rf: float, sigma_b_over_s0: float = 0.0) -> dict:
    """Уравнение (1) по слагаемым. alpha в радианах, sigma_0 в Па."""
    fa = float(f_avitzur(alpha))
    cot = 1.0 / math.tan(alpha)

    ideal = 2.0 * fa * lnR
    shear = (2.0 / math.sqrt(3.0)) * (alpha / math.sin(alpha) ** 2 - cot)
    cone = 2.0 * mu * cot * (1.0 - sigma_b_over_s0 - lnR) * lnR
    land = 2.0 * mu * P_over_Rf
    den = 1.0 + 2.0 * mu * P_over_Rf

    num = sigma_b_over_s0 + ideal + shear + cone + land
    return dict(
        sigma_d=sigma_0 * num / den, f_alpha=fa, denominator=den,
        ideal=sigma_0 * ideal / den, shear=sigma_0 * shear / den,
        cone=sigma_0 * cone / den, land=sigma_0 * land / den,
    )


def vega(lnR: float, alpha: float, mu: float, sigma_0: float,
         P_over_Rf: float, sigma_b_over_s0: float = 0.0) -> float:
    return vega_terms(lnR, alpha, mu, sigma_0, P_over_Rf, sigma_b_over_s0)["sigma_d"]


def vega_exact_land(lnR: float, alpha: float, mu: float, sigma_0: float,
                    P_over_Rf: float) -> float:
    """То же, но поясок проинтегрирован точно, без линеаризации экспоненты."""
    t = vega_terms(lnR, alpha, mu, sigma_0, P_over_Rf)
    # напряжение на входе в поясок: числитель без члена пояска и без знаменателя
    entry = (t["ideal"] + t["shear"] + t["cone"]) * t["denominator"]
    return sigma_0 - (sigma_0 - entry) * math.exp(-2.0 * mu * P_over_Rf)


def vega_measured_land(lnR: float, alpha: float, mu: float, sigma_0: float,
                       P_over_Rf: float, p_land: float) -> float:
    """Уравнение (1) с давлением на пояске, взятым из расчёта, а не из условия
    текучести. Знаменатель при этом уходит: он и есть замыкание p = sigma_0 - sigma_x."""
    t = vega_terms(lnR, alpha, mu, sigma_0, P_over_Rf)
    deformation = (t["ideal"] + t["shear"] + t["cone"]) * t["denominator"]
    return deformation + 2.0 * mu * p_land * P_over_Rf


def bearing_pressure(lnR: float, alpha: float, mu: float, sigma_0: float,
                     P_over_Rf: float) -> float:
    """Давление на пояске, которое закладывает уравнение (1): p = sigma_0 - sigma_x
    на входе в поясок."""
    t = vega_terms(lnR, alpha, mu, sigma_0, P_over_Rf)
    entry = (t["ideal"] + t["shear"] + t["cone"]) * t["denominator"]
    return sigma_0 - entry


def mu_effective(sigma_d: float, lnR: float, alpha: float, sigma_0: float,
                 P_over_Rf: float, lo: float = 0.0, hi: float = 5.0) -> float:
    """Обращение уравнения (1): какой коэффициент трения нужен, чтобы получить
    заданное напряжение волочения. Так Вега и извлекает mu из своих измерений."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if vega(lnR, alpha, mid, sigma_0, P_over_Rf) < sigma_d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mu_of_temperature(T: float, mu_0: float = MU0_VEGA, T_0: float = T0_VEGA,
                      m: float = M_VEGA) -> float:
    """Уравнение (2). T и T_0 — в градусах Цельсия."""
    return mu_0 * (T / T_0) ** m


def sticking_limit(mu: float, p: float, sigma_f: float) -> bool:
    """Кулоновская ветвь применима, пока mu*p не превысило предел текучести на
    сдвиг k = sigma_f/sqrt(3); дальше контакт переходит в прилипание и
    уравнение (1) перестаёт описывать его вовсе."""
    return mu * p <= sigma_f / math.sqrt(3.0)


def _demo() -> None:
    """Самопроверка на собственном случае Веги (его табл. 1)."""
    Ri, Rf, alpha, s0, P = 2.62e-4, 2.25e-4, math.radians(7.9), 300e6, 0.76
    lnR = math.log(Ri / Rf)
    print(f"случай Веги: ln(Ri/Rf) = {lnR:.5f}, обжатие по площади "
          f"{1 - (Rf / Ri) ** 2:.4f}, f(a) = {float(f_avitzur(alpha)):.5f}")
    print(f"{'mu':>6} {'sigma_d, МПа':>13}")
    for mu in (0.02, 0.0518, 0.1, 0.2, 0.3, 0.4, 0.5, 0.604):
        print(f"{mu:6.4f} {vega(lnR, alpha, mu, s0, P) / 1e6:13.1f}")
    print("\nего же уравнение (2) на концах диапазона скоростей:")
    for T in (26.5, 66.0):
        mu = mu_of_temperature(T)
        print(f"  T = {T:4.1f} C -> mu = {mu:.4f} -> sigma_d = "
              f"{vega(lnR, alpha, mu, s0, P) / 1e6:6.1f} МПа")


if __name__ == "__main__":
    _demo()
