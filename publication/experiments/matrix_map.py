"""
matrix_map.py — как поле меняется по плану эксперимента.

Одиночная карта показывает один режим и не говорит, ЧТО именно делает
параметр. Раскладка «строки = один фактор, столбцы = другой» показывает это
напрямую: по строке видно действие первого фактора, по столбцу — второго.

Образец — Zhang et al., Cryogenics 158 (2026) 104364, Fig. 9(d–f): строки
температура, столбцы деформация. Их исполнение мы не копируем (радужная шкала
на знакопеременных полях, 90 карт на страницу); берём только раскладку.

Шкала ОБЩАЯ на всю сетку, иначе панели несравнимы, а именно ради сравнения
рисунок и строится. Значения берутся из МКЭ: рисунок про физику процесса, а
не про качество модели.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from contours import DIVERGING, INK, MUTED, _sym  # noqa: E402
from dataset2d import Field2D  # noqa: E402

COMP = {"sigma_rr": (0, r"$\sigma_{rr}$"), "sigma_tt": (1, r"$\sigma_{\theta\theta}$"),
        "sigma_zz": (2, r"$\sigma_{zz}$"), "tau_rz": (3, r"$\tau_{rz}$")}
COL = {"Q": 0, "k": 1, "alpha": 2, "mu": 3, "v": 4}
UNIT = {"Q": "", "k": "", "alpha": "°", "mu": "", "v": " м/мин"}


def fig_matrix(field2d: str, out: str, comp: str = "sigma_zz",
               row: str = "alpha", col: str = "Q",
               fixed: Optional[dict] = None, max_n: int = 4):
    """
    Сетка карт: строки — уровни `row`, столбцы — уровни `col`.

    Прочие факторы фиксируются (`fixed`); если для клетки точного совпадения
    нет, берётся БЛИЖАЙШИЙ набор по нормированному расстоянию в пространстве
    оставшихся факторов, и его параметры печатаются в заголовке клетки —
    иначе читатель решит, что сравниваются строго сопоставимые режимы.
    """
    if comp not in COMP:
        raise SystemExit(f"компонента: {sorted(COMP)}")
    ci, tex = COMP[comp]
    f = Field2D.load(field2d)
    proc = f.proc
    fixed = fixed or {}

    rows = sorted(set(proc[:, COL[row]]))
    cols = sorted(set(proc[:, COL[col]]))
    if len(rows) > max_n:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, max_n).round().astype(int)]
    if len(cols) > max_n:
        cols = [cols[i] for i in np.linspace(0, len(cols) - 1, max_n).round().astype(int)]

    free = [k for k in COL if k not in (row, col)]
    scale = {k: (proc[:, COL[k]].std() or 1.0) for k in free}

    picks, exact = {}, 0
    for r in rows:
        for c in cols:
            m = np.isclose(proc[:, COL[row]], r) & np.isclose(proc[:, COL[col]], c)
            for k, v in fixed.items():
                mm = m & np.isclose(proc[:, COL[k]], v)
                if mm.any():
                    m = mm
            idx = np.flatnonzero(m)
            if not len(idx):
                picks[(r, c)] = None
                continue
            d = np.zeros(len(idx))
            for k, v in fixed.items():
                d += ((proc[idx, COL[k]] - v) / scale[k]) ** 2
            j = int(idx[np.argmin(d)])
            picks[(r, c)] = j
            if all(np.isclose(proc[j, COL[k]], v) for k, v in fixed.items()):
                exact += 1

    vals = [f.y[j][:, :, ci] for j in picks.values() if j is not None]
    if not vals:
        raise SystemExit("ни одной клетки не заполнено")
    lim = _sym(np.concatenate([v.ravel() for v in vals]))

    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.15 * nc + 1.5, 2.0 * nr + 1.5),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_2d(axes)
    cf = None
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            k = picks[(r, c)]
            if k is None:
                ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color=MUTED)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            lev = np.linspace(-lim, lim, 21)
            cf = ax.contourf(f.r[k], np.repeat(f.z[k][:, None], f.r.shape[2], axis=1),
                             f.y[k][:, :, ci], levels=lev, cmap=DIVERGING, extend="both")
            ax.contour(f.r[k], np.repeat(f.z[k][:, None], f.r.shape[2], axis=1),
                       f.y[k][:, :, ci], levels=lev[::5], colors="k",
                       linewidths=0.25, alpha=0.35)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
            other = ", ".join(f"{k_}={proc[k, COL[k_]]:g}" for k_ in free)
            ax.set_title(other, fontsize=6.5, color=MUTED)
            if i == 0:
                ax.text(0.5, 1.28, f"{col} = {c:g}{UNIT[col]}", fontsize=8.5,
                        ha="center", transform=ax.transAxes, color=INK)
            if j == 0:
                ax.set_ylabel(f"{row} = {r:g}{UNIT[row]}\n" + r"$z/L$", fontsize=8)
            if i == nr - 1:
                ax.set_xlabel(r"$r/R$")
    cb = fig.colorbar(cf, ax=axes, fraction=0.035, pad=0.015,
                      ticks=np.linspace(-lim, lim, 7))
    cb.set_label(f"{tex}, МПа", fontsize=8)
    cb.ax.tick_params(labelsize=6.5, color=MUTED)

    fig.suptitle(f"{tex} по плану эксперимента: строки — {row}, столбцы — {col}",
                 fontsize=9)
    note = ("Данные МКЭ. Шкала общая на всю сетку — панели сравнимы между собой.\n"
            "Мелким шрифтом над каждой панелью — прочие факторы фактически "
            f"выбранного набора")
    if fixed:
        want = ", ".join(f"{k}={v:g}" for k, v in fixed.items())
        note += (f"; целевые значения ({want}) достигнуты точно в "
                 f"{exact} клетках из {nr * nc}, в остальных взят ближайший набор")
    note += ".\n" + r"$r/R = 1$ — свободная поверхность; $z/L$ — доля осевого окна."
    if comp == "sigma_zz" and row == "alpha" and col == "Q":
        # измеренная зависимость, а не «как принято считать»: размах по
        # радиусу НЕМОНОТОНЕН по обжатию и монотонно падает по полууглу
        note += ("\nПо всему датасету (1756 наборов) размах $\sigma_{zz}$ по радиусу "
                 "НЕмонотонен по Q: 410 → 452 → 461 (пик при Q = 0.10) → 422 → 291 МПа;\n"
                 "по α падает монотонно: 441 → 433 → 422 → 404 → 367 МПа. "
                 "Растяжение на поверхности убывает и с Q (211 → 71), и с α (162 → 130).")
    fig.text(0.5, -0.012, note, ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    print(f"клеток заполнено {sum(v is not None for v in picks.values())} из {nr * nc}, "
          f"точных совпадений по фиксированным факторам {exact}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field2d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comp", default="sigma_zz")
    ap.add_argument("--row", default="alpha")
    ap.add_argument("--col", default="Q")
    ap.add_argument("--fixed", default="", help="например 'mu=0.05,v=20,k=0.5'")
    ap.add_argument("--max-n", dest="max_n", type=int, default=4)
    a = ap.parse_args()
    fixed = {}
    for part in filter(None, a.fixed.split(",")):
        k, v = part.split("=")
        fixed[k.strip()] = float(v)
    print("создан", fig_matrix(a.field2d, a.out, a.comp, a.row, a.col,
                               fixed or None, a.max_n))


if __name__ == "__main__":
    main()
