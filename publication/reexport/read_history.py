# -*- coding: utf-8 -*-
"""
read_history.py — чтение X-Y отчётов Abaqus (история по времени) и извлечение
усилия волочения, температуры и накопленной деформации.

Формат отличается от прежних .rpt: там был Field Output Report — значения по узлам
одного кадра; здесь история во времени для набора узлов. Первая колонка — время,
остальные — величины в конкретных узлах. Пропуски помечены NoValue: величины
пишутся с разной частотой, и в строках, где величина не бралась, стоит эта метка.
Заголовок перенесён на три строки и склеивается по позициям колонок.

Что извлекается:
  RF2 на опорном узле волоки — осевая реакция, то есть усилие волочения.
     Напряжение волочения sigma_d = F / (pi * Rf^2), Rf = R0 * (1 - Q), потому что Q —
     обжатие по ДИАМЕТРУ (ERRATA E-22).
  TEMP в узлах проволоки — температура. Отвечает на вопрос о термосвязке.
  PEEQ в узлах проволоки — накопленная пластическая деформация.

Соглашение осесимметричных элементов Abaqus: 1 = r, 2 = z, поэтому осевая
компонента реакции — RF2.

    python publication/reexport/read_history.py publication/reexport/history/*.rpt.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import statistics

import numpy as np

R0 = 0.018  # радиус заготовки, м


def _open(path: str):
    return gzip.open(path, "rt", errors="ignore") if path.endswith(".gz") else open(path, errors="ignore")


def parse(path: str):
    """-> (имена колонок, матрица значений со столбцом времени первым)."""
    raw = _open(path).read().split("\n")
    width = statistics.mode(len(l.split()) for l in raw if len(l.split()) > 5)
    data = [l for l in raw if len(l.split()) == width]
    ends = [[] for _ in range(width)]
    for line in data[:300]:
        for k, m in enumerate(re.finditer(r"\S+", line)):
            if k < width:
                ends[k].append(m.end())
    col_end = [int(statistics.median(e)) for e in ends]
    head = raw[1:4]
    pad = max(len(x) for x in head) + 5
    head = [x.ljust(pad) for x in head]
    names, prev = [], 0
    for e in col_end:
        names.append(" ".join("".join(h[prev:e + 2] for h in head).split()))
        prev = e
    M = np.full((len(data), width), np.nan)
    for i, line in enumerate(data):
        for j, tok in enumerate(line.split()[:width]):
            if tok != "NoValue":
                try:
                    M[i, j] = float(tok)
                except ValueError:
                    pass
    return names, M


def job_params(job: str) -> dict:
    """Параметры процесса из имени задания."""
    return dict(
        alpha=int(re.search(r"2a_(\d+)", job).group(1)) / 2,
        Q=int(re.search(r"re?d_(\d+)", job).group(1)) / 10000,
        k=int(re.search(r"cal_(\d+)", job).group(1)) / 100,
        v=int(re.search(r"ve?l?_(\d+)_f", job).group(1)),
        mu=int(re.search(r"fric_(\d+)", job).group(1)) / 1000,
    )


def extract(path: str) -> dict:
    job = os.path.basename(path).replace(".rpt.gz", "").replace(".rpt", "")
    p = job_params(job)
    names, M = parse(path)
    flat = [n.replace(" ", "") for n in names]
    i_rf2 = [i for i, n in enumerate(flat) if "RF:RF2" in n]
    i_temp = [i for i, n in enumerate(names) if n.startswith("TEMP")]
    i_peeq = [i for i, n in enumerate(names) if n.startswith("PEEQ")]

    rf2 = np.nanmax(np.abs(M[:, i_rf2]), axis=1) if i_rf2 else np.array([np.nan])
    rf2 = rf2[~np.isnan(rf2)]
    # полка: медиана верхнего дециля по времени — установившееся волочение
    plateau = float(np.median(rf2[rf2 >= np.percentile(rf2, 90)])) if rf2.size else float("nan")

    Rf = R0 * (1.0 - p["Q"])          # Q — обжатие по диаметру (E-22)
    area = math.pi * Rf ** 2
    temp = M[:, i_temp] if i_temp else np.array([[np.nan]])
    peeq = M[:, i_peeq] if i_peeq else np.array([[np.nan]])
    return dict(
        job=job, **p,
        Rf_mm=round(Rf * 1e3, 2), area_mm2=round(area * 1e6, 1),
        F_plateau_N=round(plateau, 1),
        sigma_d_MPa=round(plateau / area / 1e6, 1),
        T_start=round(float(np.nanmin(temp)), 1),
        T_max=round(float(np.nanmax(temp)), 1),
        dT_K=round(float(np.nanmax(temp) - np.nanmin(temp)), 1),
        PEEQ_max=round(float(np.nanmax(peeq)), 4),
        n_samples=int(M.shape[0]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", help="куда сложить результат")
    a = ap.parse_args()
    rows = [extract(f) for f in a.files]
    hdr = f"{'задание':52s} {'alpha':>5} {'mu':>6} {'F, кН':>7} {'sigma_d':>8} {'T_max':>6} {'dT':>5} {'PEEQ':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["alpha"], x["mu"])):
        print(f"{r['job']:52s} {r['alpha']:5.0f} {r['mu']:6.3f} {r['F_plateau_N']/1e3:7.1f} "
              f"{r['sigma_d_MPa']:8.1f} {r['T_max']:6.1f} {r['dT_K']:5.1f} {r['PEEQ_max']:6.3f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\nзаписано: {a.json}")


if __name__ == "__main__":
    main()
