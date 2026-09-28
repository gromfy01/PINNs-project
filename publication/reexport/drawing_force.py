# -*- coding: utf-8 -*-
"""
drawing_force.py — усилие волочения, контактное давление и трение из выгрузки.

read_history.py читает отчёты и даёт быструю сводку, но его оценка усилия —
медиана верхнего дециля — смещена вверх: на установившемся участке сигнал не
полка, а пила прилипания-проскальзывания, и дециль ловит её верх. Здесь усилие
берётся так, как его оценивает любой энергетический метод, — средним по времени
на установившемся окне.

ЧТО СЧИТАЕТСЯ

  Окно. Установившийся участок задаётся ходом волоки s(t) = |z_die(t) - z_die(0)|
  от 20 до 100 мм: после захвата и до выхода конуса из заготовки. Полный ход
  191-194 мм, то есть окно — примерно средние 40 % прохода.

  Усилие. F = < |RF2| > по окну, RF2 — осевая реакция в опорном узле волоки.
  Соглашение осесимметричных элементов Abaqus: 1 = r, 2 = z, реакции отнесены ко
  всей окружности, а не к радиану.

  Напряжение. sigma_d = F / (pi Rf^2) по РЕАЛЬНО достигнутому радиусу Rf, который
  измеряется по той же выгрузке (минимум радиальной координаты поверхностного
  узла внутри пояска), а не по номинальному R0 (1 - Q).

  Контактное давление. p = < |RF1| > / A_contact, A_contact — поясок 2 pi Rf L
  плюс конус pi (R0 + Rf)(R0 - Rf)/sin(alpha). Это частное полной силы на полную
  номинальную площадь, то есть среднее по контакту, а не пик CPRESS.

  Трение. Отношение |RF2|/|RF1| на фазе, когда конус уже вышел из заготовки и в
  контакте остался только цилиндрический поясок (ход волоки 115-140 мм): там
  нормаль строго радиальна, и отношение равно коэффициенту трения. Это
  восстанавливает mu из данных, а не из имени файла, — важно, потому что часть
  выгрузки продублирована (ERRATA E-23).

  Разогрев. Средний по сечению подъём температуры: узловые максимумы за всю
  историю, осреднённые по радиусу с весом r (то есть по площади).

  Предел текучести из тепла. По ядру сечения (r <= 0.5 R0), куда трение тепла не
  доносит, весь разогрев идёт от пластической работы, поэтому
  sigma_f = rho c dT / (beta eps_p) по каждому узлу. Это независимая от силового
  баланса проверка уравнения состояния. Ближе к поверхности оценка ломается —
  туда добавляется тепло трения, и значения уходят за 500 МПа.

    python publication/reexport/drawing_force.py publication/reexport/history/*.rpt.gz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import numpy as np

from read_history import job_params, parse

R0 = 0.018          # радиус заготовки, м
T_INIT = 20.0       # начальная температура, градусы Цельсия
TRAVEL_LO, TRAVEL_HI = 0.020, 0.100     # окно установившегося волочения, м
LAND_LO, LAND_HI = 0.115, 0.140         # окно «в контакте только поясок», м
RHO, CP, BETA = 7870.0, 470.0, 0.9      # плотность, теплоёмкость, доля работы в тепло
CORE = 0.5                              # ядро сечения: r <= CORE * R0


def Frad_plateau(rf1: np.ndarray, win: np.ndarray) -> float:
    return float(np.nanmean(rf1[win]))


def _cols(names, needle):
    flat = [n.replace(" ", "") for n in names]
    return [i for i, n in enumerate(flat) if needle in n]


def _by_node(names, needle):
    """{номер узла: индекс колонки} — сопоставлять величины можно только по узлу:
    наборы узлов у TEMP и у координат в выгрузке не совпадают."""
    out = {}
    for i, n in enumerate(names):
        flat = n.replace(" ", "")
        if needle not in flat:
            continue
        m = re.search(r"N:(\d+)", flat)
        if m:
            out.setdefault(int(m.group(1)), i)
    return out


def steady_window(travel: np.ndarray, rf2: np.ndarray, tol: float = 0.05,
                  smooth: float = 0.010, margin: float = 0.005):
    """Установившийся участок, найденный по самому сигналу.

    Уровень полки — медиана |RF2| на ходе от 35 до 55 % полного, где волочение
    заведомо установилось. От середины прохода окно расширяется в обе стороны, пока
    скользящее среднее |RF2| по 10 мм хода остаётся в пределах 5 % от уровня полки,
    и затем сужается на 5 мм с каждой стороны. Возвращает (начало, конец) в метрах
    или None, если окно короче 30 мм: так бывает на пиле прилипания-проскальзывания
    при малом трении, и тогда берётся фиксированное окно.
    """
    ok = np.isfinite(travel) & np.isfinite(rf2)
    tr, f = travel[ok], rf2[ok]
    order = np.argsort(tr)
    tr, f = tr[order], f[order]
    T = float(tr.max())
    mid = (tr >= 0.35 * T) & (tr <= 0.55 * T)
    if mid.sum() < 5:
        return None
    level = float(np.median(f[mid]))
    grid = np.arange(0.0, T, 0.001)
    sm = np.full(grid.size, np.nan)
    for i, g in enumerate(grid):
        m = (tr >= g - smooth / 2) & (tr < g + smooth / 2)
        if m.any():
            sm[i] = f[m].mean()
    good = np.abs(sm / level - 1.0) <= tol
    ic = int(np.argmin(np.abs(grid - 0.45 * T)))
    if not good[ic]:
        return None
    lo = hi = ic
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi < grid.size - 1 and good[hi + 1]:
        hi += 1
    a, b = grid[lo] + margin, grid[hi] - margin
    return (float(a), float(b)) if b - a >= 0.030 else None


def measure(path: str, window="fixed") -> dict:
    """window: "fixed" — окно 20-100 мм, по которому считались четыре прогона при
    обжатии 1.5 %; "auto" — окно по самому сигналу (steady_window), с откатом на
    фиксированное, если полка не найдена."""
    job = os.path.basename(path).replace(".rpt.gz", "").replace(".rpt", "")
    p = job_params(job)
    names, M = parse(path)
    t = M[:, 0]

    i_rf1, i_rf2 = _cols(names, "RF:RF1"), _cols(names, "RF:RF2")
    i_z_die = _cols(names, "COORD:COOR2")[:1]
    i_r_surf = _cols(names, "COORD:COOR1")[1:2]
    temp_by_node = _by_node(names, "TEMP")
    r_by_node = _by_node(names, "COORD:COOR1")

    rf1 = np.nanmax(np.abs(M[:, i_rf1]), axis=1)
    rf2 = np.nanmax(np.abs(M[:, i_rf2]), axis=1)
    z_die = M[:, i_z_die[0]]
    travel = np.abs(z_die - z_die[np.isfinite(z_die)][0])

    lo_w, hi_w, how = TRAVEL_LO, TRAVEL_HI, "fixed"
    if window == "auto":
        found = steady_window(travel, rf2)
        if found is not None:
            (lo_w, hi_w), how = found, "auto"
    win = np.isfinite(rf2) & (travel >= lo_w) & (travel <= hi_w)
    F = float(np.nanmean(rf2[win]))
    spread = float(np.nanstd(rf2[win]) / F)

    # реально достигнутый радиус: минимум радиальной координаты поверхностного узла
    r_surf = M[:, i_r_surf[0]]
    Rf = float(np.nanmin(r_surf[np.isfinite(r_surf)]))
    area = math.pi * Rf ** 2

    # контактная площадь: поясок (L = k * Rf) плюс конус
    a = math.radians(p["alpha"])
    A_land = 2.0 * math.pi * Rf * p["k"] * Rf
    A_cone = math.pi * (R0 + Rf) * (R0 - Rf) / math.sin(a)
    Frad = float(np.nanmean(rf1[win]))
    p_contact = Frad / (A_land + A_cone)

    # трение по фазе «только поясок»: конус вышел, в контакте цилиндр
    if how == "auto":
        # после полки: радиальная реакция ещё не меньше четверти установившейся,
        # последние 60 % таких точек — конус уже вышел, в контакте один поясок
        after = (np.isfinite(rf1) & np.isfinite(rf2) & (travel > hi_w)
                 & (rf1 > 0.25 * Frad_plateau(rf1, win)))
        idx = np.flatnonzero(after)
        late = np.zeros_like(after)
        if idx.size:
            late[idx[int(0.4 * idx.size):]] = True
        if p["k"] == 0:
            late[:] = False
    else:
        late = (np.isfinite(rf1) & np.isfinite(rf2) & (rf1 > 1e3)
                & (travel >= LAND_LO) & (travel <= LAND_HI))
    mu_data = (float(np.nanmedian(rf2[late] / rf1[late]))
               if late.sum() > 5 else float("nan"))

    # средний по сечению разогрев: узловые максимумы по радиусу с весом площади
    # начальная температура: в выгрузке 28.09.2026 модель считает в кельвинах
    # (293.15), в прежней — в градусах Цельсия (20.0); разогрев от неё не зависит
    t0 = np.array([c[np.isfinite(c)][0] for c in M[:, sorted(temp_by_node.values())].T
                   if np.isfinite(c).any()])
    T_init = float(np.median(t0)) if t0.size else T_INIT
    shared = sorted(set(temp_by_node) & set(r_by_node) - {1})
    rr = np.array([np.nanmax(M[:, r_by_node[n]]) for n in shared])
    dd = np.array([np.nanmax(M[:, temp_by_node[n]]) for n in shared]) - T_init
    order = np.argsort(rr)
    rr, dd = rr[order], dd[order]
    dT_mean = float(np.trapezoid(dd * rr, rr) * 2.0 / rr[-1] ** 2)
    temp = M[:, sorted(temp_by_node.values())]

    # предел текучести из тепла по ядру сечения
    peeq_by_node = _by_node(names, "PEEQ")
    core = []
    for n in sorted(set(temp_by_node) & set(r_by_node) & set(peeq_by_node) - {1}):
        rad = np.nanmax(M[:, r_by_node[n]])
        eps = np.nanmax(M[:, peeq_by_node[n]])
        if rad <= CORE * R0 and eps > 1e-3:
            core.append(RHO * CP * (np.nanmax(M[:, temp_by_node[n]]) - T_init) / (BETA * eps))

    return dict(
        job=job, alpha=p["alpha"], mu_label=p["mu"], k=p["k"], v=p["v"], Q=p["Q"],
        window_mm=(round(lo_w * 1e3, 1), round(hi_w * 1e3, 1)), window_how=how,
        n_window=int(win.sum()), travel_total_mm=round(float(np.nanmax(travel)) * 1e3, 1),
        Rf_mm=round(Rf * 1e3, 4), area_mm2=round(area * 1e6, 2),
        F_kN=round(F / 1e3, 2), sigma_d_MPa=round(F / area / 1e6, 1),
        spread_pct=round(spread * 100.0, 1),
        p_contact_MPa=round(p_contact / 1e6, 1),
        mu_from_data=round(mu_data, 4),
        dT_mean_C=round(dT_mean, 1),
        dT_peak_C=round(float(np.nanmax(temp) - T_init), 1),
        T_init=round(T_init, 2), kelvin=bool(T_init > 200.0),
        sigma_f_heat_min_MPa=round(min(core) / 1e6, 0) if core else float("nan"),
        sigma_f_heat_max_MPa=round(max(core) / 1e6, 0) if core else float("nan"),
        n_core=len(core),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [measure(f) for f in a.files]
    hdr = (f"{'задание':50s} {'alpha':>5} {'mu файл':>7} {'mu данные':>9} {'Rf, мм':>7} "
           f"{'F, кН':>7} {'sigma_d':>8} {'разброс':>8} {'p, МПа':>7} {'dT ср':>6} {'dT пик':>7} "
           f"{'sigma_f из тепла':>17}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (-x["alpha"], x["mu_label"])):
        print(f"{r['job']:50s} {r['alpha']:5.0f} {r['mu_label']:7.3f} {r['mu_from_data']:9.4f} "
              f"{r['Rf_mm']:7.3f} {r['F_kN']:7.1f} {r['sigma_d_MPa']:8.1f} {r['spread_pct']:7.1f}% "
              f"{r['p_contact_MPa']:7.1f} {r['dT_mean_C']:6.1f} {r['dT_peak_C']:7.1f} "
              f"{r['sigma_f_heat_min_MPa']:8.0f}-{r['sigma_f_heat_max_MPa']:<8.0f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\nзаписано: {a.json}")


if __name__ == "__main__":
    main()
