"""
compare.py — попарное сравнение вариантов на одних и тех же тестовых режимах.

У всех прогонов одно разбиение (split_seed), поэтому для каждого тестового режима берётся
macro-RMSE, усреднённая по сидам варианта, и разности сравниваются знаковым критерием
Уилкоксона по режимам. Отдельно — разность macro-RMSE по сидам (среднее ± СКО).

    python src/compare.py --data data/trial_w2.npz --runs runs --out runs/compare.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Y = np.load(a.data)["Y"]
    per, macro, idx = {}, {}, None
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "test_pred.npz"))):
        j = json.load(open(os.path.join(os.path.dirname(f), "metrics.json"), encoding="utf-8"))
        t = np.load(f)
        idx = t["idx"] if idx is None else idx
        assert np.array_equal(idx, t["idx"]), "у прогонов разные тестовые выборки"
        e = t["pred"] - Y[idx]
        r = np.sqrt(np.mean(e ** 2, axis=(2, 3))).mean(1)          # macro-RMSE по режиму
        nm = j["config"]["name"]
        per.setdefault(nm, []).append(r)
        macro.setdefault(nm, []).append(j["metrics"]["rmse_macro"])
    L = [f"# Попарное сравнение вариантов\n", f"Тестовых режимов {len(idx)}. "
         f"Сидов: " + ", ".join(f"{k} — {len(v)}" for k, v in per.items()) + ".\n",
         "| пара | Δ macro-RMSE по сидам, МПа | медиана Δ по режимам, МПа | доля режимов, где первый лучше | p (Уилкоксон) |",
         "|---|---|---|---|---|"]
    order = [n for n in ("hard", "profile1d", "data_only") if n in per] + [n for n in per if n not in ("hard", "profile1d", "data_only")]
    for x, y in combinations(order, 2):
        rx, ry = np.mean(per[x], 0), np.mean(per[y], 0)
        d = rx - ry
        dm = np.array(macro[x][:min(len(macro[x]), len(macro[y]))]) - np.array(macro[y][:len(macro[x])])
        L.append(f"| {x} − {y} | {dm.mean():+.2f} ± {dm.std(ddof=1) if len(dm) > 1 else 0:.2f} | "
                 f"{np.median(d):+.2f} | {np.mean(d < 0):.2f} | {wilcoxon(d).pvalue:.1e} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
