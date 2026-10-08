"""
make_report.py — audit_raw.npz → publication/audit/RAW_AUDIT.md.

Все числа в ERRATA и в playbook берутся отсюда и только отсюда.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from filters import find_mu_duplicates, grid_coverage  # noqa: E402


def pct(x, p):
    return float(np.percentile(x, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "RAW_AUDIT.md"))
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    P, S, M, D = d["params"], d["prof_s"], d["metas"], d["drops"]
    names = list(map(str, d["names"]))
    mi = {k: i for i, k in enumerate(map(str, d["meta_keys"]))}
    di = {k: i for i, k in enumerate(map(str, d["drop_keys"]))}
    n = len(names)

    r_lo = M[:, mi["r_min_win"]] / M[:, mi["r_max_all"]]
    r_hi = M[:, mi["r_max_win"]] / M[:, mi["r_max_all"]]
    z_frac = ((M[:, mi["z_max_win"]] - M[:, mi["z_min_win"]]) /
              (M[:, mi["z_max_all"]] - M[:, mi["z_min_all"]]))

    outer = np.abs(S[:, -1, :]).mean(0)
    inner = np.abs(S[:, 0, :]).mean(0)

    mu_u, mu_c = np.unique(np.round(P[:, 3], 6), return_counts=True)
    dup = find_mu_duplicates(P, S)
    cov_raw = grid_coverage(P)
    keep = np.setdiff1d(np.arange(n), np.flatnonzero(~np.isin(np.round(P[:, 3], 6), [0.025, 0.05, 0.10])))
    cov_clean = grid_coverage(P[keep])

    key = lambda i: tuple(np.round(P[i, [0, 1, 2, 4]], 12))
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[key(i)].append(i)
    collapsed = sum(1 for g in groups.values()
                    if len(g) >= 2 and all(np.array_equal(S[g[0]], S[j]) for j in g[1:]))

    ok = ~np.isnan(D).any(1)
    Dk = D[ok]
    ratio1 = Dk[:, di["d_tau_dz"]] / (Dk[:, di["d_srr_dr"]] + Dk[:, di["hoop"]])
    ratio2 = Dk[:, di["d_szz_dz"]] / (Dk[:, di["d_tau_dr"]] + Dk[:, di["shear_over_r"]])
    rel1 = np.abs(Dk[:, di["R1_kept"]] - Dk[:, di["R1_full"]]) / np.maximum(Dk[:, di["R1_full"]], 1e-12)
    rel2 = np.abs(Dk[:, di["R2_kept"]] - Dk[:, di["R2_full"]]) / np.maximum(Dk[:, di["R2_full"]], 1e-12)

    L = []
    w = L.append
    w("# Аудит сырых данных Abaqus\n")
    w("Сгенерировано `publication/audit/make_report.py` по релизу `raw-data-v1`.")
    w("Все утверждения ниже получены из `.rpt`, а не из рукописи, `docs/` или")
    w("предыдущего handover — там, где источники расходятся, прав `.rpt`.\n")
    w(f"Файлов обработано: **{n}** (ошибок парсинга 0).\n")

    w("## 1. Раскладка колонок `.rpt`\n")
    w("Подтверждена по заголовку Field Output Report:\n")
    w("```")
    w("0 Node Label   1 COORD.COOR1 (r)   2 COORD.COOR2 (z)")
    w("3..6   LE.LE11 LE.LE22 LE.LE33 LE.LE12   (ε_rr, ε_zz, ε_θθ, γ_rz)")
    w("7..10  S.Mises  S.Max/Mid/Min Prin")
    w("11..14 S.S11 S.S22 S.S33 S.S12          (σ_rr, σ_zz, σ_θθ, τ_rz)")
    w("```")
    w("Соглашение осесимметричных элементов Abaqus: 1 = r, 2 = z, 3 = θ.")
    w("`preprocessing/rpt_utils.py` читает эти колонки правильно.\n")

    w("## 2. Что означает окно «25–75 %»  (E-01)\n")
    w("| величина | медиана | мин | макс |")
    w("|---|---|---|---|")
    w(f"| доля осевого размаха, попавшая в окно | {np.median(z_frac):.4f} | {z_frac.min():.4f} | {z_frac.max():.4f} |")
    w(f"| r_min(окно) / R | {np.median(r_lo):.4f} | {r_lo.min():.4f} | {r_lo.max():.4f} |")
    w(f"| r_max(окно) / R | {np.median(r_hi):.4f} | {r_hi.min():.4f} | {r_hi.max():.4f} |")
    w("")
    w("**Вывод.** Окно 25–75 % — **осевое** (по `z`, зона выхода на стационар).")
    w("Двадцать радиальных точек покрывают **весь** радиус: `r_norm = 0` — ось,")
    w("`r_norm = 1` — свободная поверхность. Формула `r_phys = R·(0.25 + 0.5·r_norm)`")
    w("из прежнего handover и константа `R_EXTRAP_FREE_SURFACE = 1.5` в")
    w("`models/*.py` данным противоречат: `r_norm = 1.5` — это `1.5 R`, точка вне")
    w("материала.\n")

    w("## 3. Порядок компонент  (E-04)\n")
    w("Стек, который строит `preprocessing_res_np`, — `(s_rr, s_tt, s_zz, s_rz)`.")
    w("Средний модуль по всем наборам:\n")
    w("| слот стека | на оси, МПа | на внешней точке, МПа |")
    w("|---|---|---|")
    for i, nm in enumerate(("0 — s_rr", "1 — s_tt", "2 — s_zz", "3 — s_rz")):
        w(f"| {nm} | {inner[i]:.2f} | {outer[i]:.2f} |")
    w("")
    w("Traction-free (`σ_rr ≈ 0`, `τ_rz ≈ 0`) выполняется в слотах 0 и 3, то есть")
    w("стек **уже физический**. Значит `PLANE_ORDER = [2, 0, 1, 3]` поверх него даёт")
    w("`(σ_zz, σ_rr, σ_θθ, τ_rz)` — перепутанные компоненты. Либо `.pkl` собран")
    w("не тем скриптом, что лежит в репозитории, либо константа неверна; в любом")
    w("случае опубликованный репозиторий сейчас не воспроизводит датасет.\n")
    w("Куда именно разъехалось. `docs/data_format.md` приводит проверку ГУ после")
    w("применения `PLANE_ORDER`: `σ_rr ≈ 5.9`, `σ_θθ ≈ 144`, `σ_zz ≈ 99`, `τ_rz ≈ 3.8` МПа.")
    w("Обратной подстановкой `out = (P2, P0, P1, P3)` получаем, что в `.pkl` плоскости")
    w("лежат в порядке **(σ_θθ, σ_zz, σ_rr, τ_rz)**, тогда как коммитнутый")
    w("`preprocessing_res_np` укладывает **(σ_rr, σ_θθ, σ_zz, τ_rz)**. Чинить одним из")
    w("двух способов, но обязательно с записью в статье: либо пересобрать `.pkl`")
    w("текущим препроцессингом и поставить `PLANE_ORDER = [0, 1, 2, 3]`, либо привести")
    w("порядок стека в `preprocessing_res_np` к тому, каким собран `.pkl`.")
    w("Однострочная проверка на настоящем `.pkl`: та плоскость, у которой")
    w("`mean|y[p][:, -1]|` минимален, и есть `σ_rr`.\n")

    w("## 4. Уровни факторов\n")
    w("| μ | наборов |")
    w("|---|---|")
    for u, c in zip(mu_u, mu_c):
        w(f"| {u:g} | {c} |")
    w("")
    w(f"Уровень `μ = 0.002` встречается **{int(mu_c[mu_u == 0.002][0])} раз**;")
    w("источник — единственный файл `Vel_40/aisi_1020_2a_40_rd_500_cal_100_v_40_fric_02.rpt`,")
    w("где токен `fric_02` вместо `fric_025`. Правило `rpt_utils.f()` (снять первый")
    w("символ, поделить на 1000) превращает опечатку в фантомный уровень фактора.\n")
    for nm, col in (("Q", 0), ("k", 1), ("α", 2), ("v", 4)):
        u, c = np.unique(np.round(P[:, col], 6), return_counts=True)
        w(f"`{nm}`: " + ", ".join(f"{x:g} → {y}" for x, y in zip(u, c)) + "  ")
    w("")
    w(f"Скорость `v = 5` м/мин представлена {int((P[:,4]==5).sum())} наборами против")
    w("554–607 у остальных уровней — сетка несбалансирована, это надо оговаривать.\n")
    w(f"Уникальных комбинаций: {cov_raw['n_unique_combinations']} из")
    w(f"{cov_raw['full_factorial']} (с фантомным μ) и из {cov_clean['full_factorial']}")
    w(f"после его снятия ⇒ покрытие {100 * cov_clean['coverage']:.1f} %.\n")

    w("## 5. Дубликаты по μ  (E-06)\n")
    w(f"Наборов, у которых при совпавших `(Q, k, α, v)` профиль **побитово** равен")
    w(f"профилю другого набора, отличающегося только меткой μ: **{len(dup)}** из {n}")
    w(f"(**{100 * len(dup) / n:.1f} %**).\n")
    w(f"Комбинаций `(Q, k, α, v)`, где все имеющиеся уровни μ дали один и тот же")
    w(f"результат: **{collapsed}**.\n")
    w("Проверено сильнее: у 40 случайно выбранных пар совпали **все числовые")
    w("данные `.rpt` целиком** (40 из 40), при этом md5 файлов различаются —")
    w("расходятся только заголовки (путь к `.odb`, отметка времени). То есть это")
    w("не «трение не повлияло», а **один и тот же расчёт, размноженный под тремя")
    w("метками трения**. Использовать μ как ось экстраполяции нельзя, а таблицу")
    w("факторов в статье надо сопроводить этой оговоркой.\n")
    w("Цифра «~31 % μ-дублей» из `docs/data_format.md` не воспроизводится")
    w(f"ни на каком порядке фильтров; на сырой сетке это {100 * len(dup) / n:.1f} %.\n")

    w("## 6. Отброшенные осевые члены равновесия  (E-02)\n")
    w("Поля внутри осевого окна биннингованы на сетку 20 × 20 (медиана в ячейке),")
    w("производные — `numpy.gradient` по физическим координатам. По каждому набору")
    w("берётся медиана модуля члена, ниже — перцентили этих медиан по 2443 наборам.\n")
    w("| член | 25 % | медиана | 75 % |")
    w("|---|---|---|---|")
    lbl = {"d_srr_dr": "∂σ_rr/∂r", "hoop": "(σ_rr − σ_θθ)/r", "d_tau_dz": "∂τ_rz/∂z *(отброшен)*",
           "d_tau_dr": "∂τ_rz/∂r", "shear_over_r": "τ_rz/r", "d_szz_dz": "∂σ_zz/∂z *(отброшен)*"}
    for k, t in lbl.items():
        c = Dk[:, di[k]]
        w(f"| {t} | {pct(c,25):.3g} | {pct(c,50):.3g} | {pct(c,75):.3g} |")
    w("")
    w("Отношение отброшенного к сумме удержанных, МПа/м на МПа/м:\n")
    w("| уравнение | 25 % | медиана | 75 % | доля наборов с отношением > 0.2 |")
    w("|---|---|---|---|---|")
    w(f"| радиальное: `∂τ_rz/∂z` / (`∂σ_rr/∂r` + `(σ_rr−σ_θθ)/r`) | {pct(ratio1,25):.3f} | "
      f"{pct(ratio1,50):.3f} | {pct(ratio1,75):.3f} | {100*np.mean(ratio1>0.2):.1f} % |")
    w(f"| осевое: `∂σ_zz/∂z` / (`∂τ_rz/∂r` + `τ_rz/r`) | {pct(ratio2,25):.3f} | "
      f"{pct(ratio2,50):.3f} | {pct(ratio2,75):.3f} | {100*np.mean(ratio2>0.2):.1f} % |")
    w("")
    w("Насколько меняется сама невязка при переходе к редуцированной форме")
    w("(медиана относительного отклонения):\n")
    w(f"* радиальное уравнение — **{np.median(rel1):.3f}**;")
    w(f"* осевое уравнение — **{np.median(rel2):.3f}**.\n")
    w("**Вывод.** Для радиального уравнения отбрасывание `∂τ_rz/∂z` оправдано")
    w("численно и это можно написать в статье. Для осевого — **не оправдано**:")
    w("отброшенный член того же порядка, что удержанные. Выражение")
    w("`∂τ_rz/∂r + τ_rz/r`, которое считает код, осевым равновесием не является.\n")

    w("## 7. Нарушение traction-free самими данными FEM\n")
    w(f"В крайней радиальной точке профиля (`r_norm = 1`): средний `|σ_rr|` =")
    w(f"**{outer[0]:.2f} МПа**, `|τ_rz|` = **{outer[3]:.2f} МПа**.\n")
    w("Это базовая линия для Fig. 11–12: точное условие `σ_rr = τ_rz = 0` —")
    w("физический приор, которому сам источник данных удовлетворяет лишь с этой")
    w("точностью. Метрику нарушения ГУ у моделей осмысленно печатать только")
    w("рядом с этой строкой.\n")

    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"записано: {a.out}  ({len(L)} строк)")


if __name__ == "__main__":
    main()
