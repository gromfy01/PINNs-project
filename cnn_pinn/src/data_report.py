"""
data_report.py — короткий отчёт о подготовленных данных (ТЗ, п. 3, результат этапа 1).

    python src/data_report.py --data data/trial_w2.npz --out runs/data_report.md

Число режимов и причины отбраковки, покрытие уровней, пометки, невязка равновесия
и граничные условия самих данных МКЭ на сетке, осевая сила по сечениям.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from physics import Grid, residuals, section_force


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(a, qs=(10, 50, 90)):
    return " / ".join(f"{v:.1f}" for v in np.percentile(a, qs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.data)
    meta = json.loads(str(d["meta"]))
    rep = json.load(open(os.path.splitext(a.data)[0] + "_report.json", encoding="utf-8"))
    Y, P = d["Y"].astype(np.float64), d["P"]
    g = Grid(*meta["grid"], meta["window_R"])
    L = [f"# Данные пробного прогона\n",
         f"Сырых отчётов {meta['n_raw']}, в наборе {meta['n_kept']} режимов. "
         f"Окно {meta['window_R']}R в установившейся зоне, сетка {meta['grid'][0]} × {meta['grid'][1]}.\n",
         "## Отбраковка\n",
         table(sorted(meta["rejected"].items(), key=lambda t: -t[1]), ["причина", "режимов"]) + "\n",
         "## Покрытие\n"]
    names = ("Q_d", "k", "α, °", "μ", "v, м/мин")
    for j, nm in enumerate(names):
        u, c = np.unique(P[:, j], return_counts=True)
        L.append(f"- {nm}: " + ", ".join(f"{x:g} — {n}" for x, n in zip(u, c)))
    qs, als = np.unique(P[:, 0]), np.unique(P[:, 2])
    L.append("\nРежимов в ячейках (Q_d × α):\n")
    L.append(table([[f"{q:g}"] + [int(((P[:, 0] == q) & (P[:, 2] == al)).sum()) for al in als] for q in qs],
                   ["Q_d \\ α"] + [f"{al:g}°" for al in als]) + "\n")
    vs = np.unique(P[:, 4])
    L.append("Режимов в ячейках (Q_d × v):\n")
    L.append(table([[f"{q:g}"] + [int(((P[:, 0] == q) & (P[:, 4] == v)).sum()) for v in vs] for q in qs],
                   ["Q_d \\ v"] + [f"{v:g}" for v in vs]) + "\n")

    R1, R2 = residuals(Y, g)
    mx = np.abs(Y).reshape(len(Y), -1).max(1)
    r1 = np.sqrt(np.mean(R1 ** 2, axis=(1, 2))) / mx * 100
    r2 = np.sqrt(np.mean(R2 ** 2, axis=(1, 2))) / mx * 100
    F = section_force(Y, g)
    Frms = np.sqrt(np.mean(F ** 2, axis=1))
    L += ["## Данные МКЭ на сетке (10-й / 50-й / 90-й перцентили по режимам)\n",
          f"- max|σ| по четырём компонентам: {pct(mx)} МПа",
          f"- RMS невязки радиального уравнения: {pct(r1)} % от max|σ|",
          f"- RMS невязки осевого уравнения: {pct(r2)} % от max|σ|",
          f"- доля режимов с невязкой ≤ 1 % max|σ| по обоим уравнениям: "
          f"{np.mean((r1 <= 1) & (r2 <= 1)) * 100:.1f} %",
          f"- RMS |σ_r| на поверхности: {pct(np.sqrt(np.mean(Y[:, 0, -1, :] ** 2, axis=1)))} МПа",
          f"- RMS τ_rz на поверхности: {pct(np.sqrt(np.mean(Y[:, 3, -1, :] ** 2, axis=1)))} МПа",
          f"- RMS средней по сечению σ_z (2∫σ_zρdρ): {pct(Frms)} МПа, больше 10 МПа — у "
          f"{np.mean(Frms > 10) * 100:.0f} % режимов",
          f"- σ_z на поверхности: {pct(Y[:, 2, -1, :].mean(1))} МПа, на оси: {pct(Y[:, 2, 0, :].mean(1))} МПа",
          f"- утонение после волоки: {pct(d['thinning'] * 1000)} мкм, больше 0.1 мм — у "
          f"{np.mean(d['thinning'] > 0.1) * 100:.0f} % режимов",
          f"- длина установившейся зоны: {pct(d['zone_len_R'])} R\n"]
    by_v = [[f"{v:g}", int((P[:, 4] == v).sum()), f"{np.mean(Frms[P[:, 4] == v] > 10) * 100:.0f} %"] for v in vs]
    L.append("Осевая сила по скоростям:\n")
    L.append(table(by_v, ["v, м/мин", "режимов", "RMS средней σ_z > 10 МПа"]) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
