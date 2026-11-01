"""
make_report.py — audit.npz → raw_audit_report.md (отчёт аудита сырых данных).

Читает выход audit/run_raw_audit.py (audit.npz; code/build_datasets.py
кладёт его в <out>/audit.npz) и записывает в Markdown измеренные по сырым
.rpt числа: число наборов, статистику осевого окна, порядок компонент, уровни
факторов и покрытие сетки, опечатку в метке μ, дубликаты по μ, долю отброшенных
членов уравнений равновесия и нарушение условия свободной поверхности в самих
данных МКЭ. По умолчанию отчёт пишется рядом со скриптом.

    python audit/make_report.py --npz data/audit.npz [--out raw_audit_report.md]
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
    ap.add_argument("--out", default=os.path.join(HERE, "raw_audit_report.md"))
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
    w(f"Сгенерировано `audit/make_report.py` по `{os.path.basename(a.npz)}` — выходу")
    w("`audit/run_raw_audit.py` по сырым отчётам `.rpt` (Field Output Report).\n")
    w(f"Наборов в файле: **{n}**.\n")

    w("## 1. Раскладка колонок `.rpt`\n")
    w("Колонки Field Output Report, по которым читаются данные (`code/raw_probe.py`):\n")
    w("```")
    w("0 Node Label   1 COORD.COOR1 (r)   2 COORD.COOR2 (z)")
    w("3..6   LE.LE11 LE.LE22 LE.LE33 LE.LE12   (ε_rr, ε_zz, ε_θθ, γ_rz)")
    w("7..10  S.Mises  S.Max/Mid/Min Prin")
    w("11..14 S.S11 S.S22 S.S33 S.S12          (σ_rr, σ_zz, σ_θθ, τ_rz)")
    w("```")
    w("Соглашение осесимметричных элементов Abaqus: 1 = r, 2 = z, 3 = θ.\n")

    w("## 2. Осевое окно «25–75 %»\n")
    w("| величина | медиана | мин | макс |")
    w("|---|---|---|---|")
    w(f"| доля осевого размаха, попавшая в окно | {np.median(z_frac):.4f} | {z_frac.min():.4f} | {z_frac.max():.4f} |")
    w(f"| r_min(окно) / R | {np.median(r_lo):.4f} | {r_lo.min():.4f} | {r_lo.max():.4f} |")
    w(f"| r_max(окно) / R | {np.median(r_hi):.4f} | {r_hi.min():.4f} | {r_hi.max():.4f} |")
    w("")
    w("Окно 25–75 % — **осевое** (по `z`, зона выхода на стационар); по радиусу в")
    w(f"него попадает практически всё сечение (медиана `r_max(окно) / R` = {np.median(r_hi):.4f}).")
    w(f"Радиальный профиль из {S.shape[1]} точек идёт от оси (`r_norm = 0`) до свободной")
    w("поверхности (`r_norm = 1`).\n")

    w("## 3. Порядок компонент\n")
    w("Профили напряжений `prof_s` хранят компоненты в порядке `(σ_rr, σ_θθ, σ_zz, τ_rz)`.")
    w("Средний модуль по всем наборам:\n")
    w("| слот | на оси, МПа | на внешней точке, МПа |")
    w("|---|---|---|")
    for i, nm in enumerate(("0 — s_rr", "1 — s_tt", "2 — s_zz", "3 — s_rz")):
        w(f"| {nm} | {inner[i]:.2f} | {outer[i]:.2f} |")
    w("")
    w("На внешней точке малы слоты 0 и 3, то есть `σ_rr` и `τ_rz`, для которых на")
    w("свободной поверхности `σ_rr = τ_rz = 0` (traction-free): порядок компонент")
    w("в профилях физический.\n")

    w("## 4. Уровни факторов\n")
    w("| μ | наборов |")
    w("|---|---|")
    for u, c in zip(mu_u, mu_c):
        w(f"| {u:g} | {c} |")
    w("")
    odd = np.flatnonzero(np.round(P[:, 3], 6) == 0.002)
    if odd.size:
        w(f"Уровень `μ = 0.002` встречается **{odd.size} раз**: "
          + ", ".join(f"`{names[i]}.rpt`" for i in odd) + ".")
        w("В имени задания стоит токен `fric_02`; по правилу разбора имени (первый")
        w("символ снимается, остаток делится на 1000) он даёт `μ = 0.002`, тогда как")
        w("`fric_025` даёт 0.025. Это опечатка в имени задания, а не отдельный уровень")
        w("фактора.\n")
    for nm, col in (("Q", 0), ("k", 1), ("α", 2), ("v", 4)):
        u, c = np.unique(np.round(P[:, col], 6), return_counts=True)
        w(f"`{nm}`: " + ", ".join(f"{x:g} → {y}" for x, y in zip(u, c)) + "  ")
    w("")
    v_u, v_c = np.unique(np.round(P[:, 4], 6), return_counts=True)
    v_rest = v_c[v_u != 5]
    if v_rest.size:
        w(f"Скорость `v = 5` м/мин представлена {int((P[:,4]==5).sum())} наборами против")
        w(f"{v_rest.min()}–{v_rest.max()} у остальных уровней — сетка несбалансирована.\n")
    w(f"Уникальных комбинаций `(Q, k, α, μ, v)`: {cov_raw['n_unique_combinations']} из")
    w(f"{cov_raw['full_factorial']} в полной сетке с уровнем `μ = 0.002`; без него —")
    w(f"{cov_clean['n_unique_combinations']} из {cov_clean['full_factorial']}, "
      f"покрытие {100 * cov_clean['coverage']:.1f} %.\n")

    w("## 5. Дубликаты по μ\n")
    w(f"Наборов, у которых при совпавших `(Q, k, α, v)` профиль **побитово** равен")
    w(f"профилю другого набора, отличающегося только меткой μ: **{len(dup)}** из {n}")
    w(f"(**{100 * len(dup) / n:.1f} %**).\n")
    w(f"Комбинаций `(Q, k, α, v)`, где все имеющиеся уровни μ дали один и тот же")
    w(f"результат: **{collapsed}**.\n")
    w("У наборов-дубликатов метка μ на результат расчёта не влияет, и как отдельные")
    w("уровни трения они информации не несут.\n")

    w("## 6. Отброшенные осевые члены равновесия\n")
    w("Поля внутри осевого окна биннингованы на сетку 20 × 20 (медиана в ячейке),")
    w("производные — `numpy.gradient` по физическим координатам. По каждому набору")
    w(f"берётся медиана модуля члена, ниже — перцентили этих медиан по {len(Dk)} наборам.\n")
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
    w("В радиальном уравнении отброшенный член `∂τ_rz/∂z` мал по сравнению с")
    w("удержанными. В осевом отброшенный `∂σ_zz/∂z` того же порядка, что удержанные,")
    w("и редуцированное выражение `∂τ_rz/∂r + τ_rz/r` осевое равновесие не описывает.\n")

    w("## 7. Нарушение traction-free самими данными МКЭ\n")
    w(f"В крайней радиальной точке профиля (`r_norm = 1`): средний `|σ_rr|` =")
    w(f"**{outer[0]:.2f} МПа**, `|τ_rz|` = **{outer[3]:.2f} МПа**.\n")
    w("Это базовая линия: точное условие `σ_rr = τ_rz = 0` сам источник данных")
    w("выполняет лишь с этой точностью, и нарушение граничных условий у моделей")
    w("имеет смысл сравнивать с этим уровнем.\n")

    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"записано: {a.out}  ({len(L)} строк)")


if __name__ == "__main__":
    main()
