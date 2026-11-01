"""
summarize.py — сводная таблица прогонов: среднее ± СКО по сидам для каждого варианта (ТЗ, п. 7),
рядом — те же величины для данных МКЭ и RMSE модели «среднее поле».

    python src/summarize.py --runs runs --out runs/summary.md [--csv runs/summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

ROWS = [
    ("rmse_sigma_r", "RMSE σ_r, МПа"), ("rmse_sigma_theta", "RMSE σ_θ, МПа"),
    ("rmse_sigma_z", "RMSE σ_z, МПа"), ("rmse_tau_rz", "RMSE τ_rz, МПа"),
    ("rmse_macro", "macro-RMSE, МПа"), ("rmse_sigma_i", "RMSE σ_i, МПа"),
    ("maxerr_sigma_z_interior", "max ошибка σ_z, внутр. узлы, МПа"),
    ("maxerr_sigma_r_boundary", "max ошибка σ_r, ось и поверхность, МПа"),
    ("surf_sz_err_median", "ошибка σ_z на поверхности, медиана, МПа"),
    ("surf_sz_err_p90", "ошибка σ_z на поверхности, p90, МПа"),
    ("surf_sz_pass_frac", "доля режимов с ошибкой σ_z пов. ≤ max(10 %, 15 МПа)"),
    ("model_res_R1", "RMS невязки радиального ур., МПа"),
    ("model_res_R2", "RMS невязки осевого ур., МПа"),
    ("model_bc_sr_surface", "RMS σ_r на поверхности, МПа"),
    ("model_bc_trz_surface", "RMS τ_rz на поверхности, МПа"),
    ("model_bc_sr_minus_st_axis", "RMS σ_r − σ_θ на оси, МПа"),
    ("model_axial_force", "RMS средней σ_z по сечению, МПа"),
    ("rmse_macro_axial_rms_gt10", "macro-RMSE, режимы с осевой силой > 10 МПа"),
    ("rmse_macro_not_axial_rms_gt10", "macro-RMSE, остальные режимы"),
    ("rmse_macro_thinning_gt0.1", "macro-RMSE, утонение > 0.1 мм"),
    ("rmse_macro_not_thinning_gt0.1", "macro-RMSE, без утонения"),
    ("n_params", "параметров"), ("best_epoch", "лучшая эпоха"),
    ("train_seconds", "время обучения, с"), ("predict_ms_per_regime", "предсказание, мс/режим"),
]
FEM = {"model_res_R1": "fem_res_R1", "model_res_R2": "fem_res_R2",
       "model_bc_sr_surface": "fem_bc_sr_surface", "model_bc_trz_surface": "fem_bc_trz_surface",
       "model_bc_sr_minus_st_axis": "fem_bc_sr_minus_st_axis", "model_axial_force": "fem_axial_force"}
BASE = {f"rmse_{c}": f"rmse_meanfield_{c}" for c in ("sigma_r", "sigma_theta", "sigma_z", "tau_rz")}


def fmt(v):
    if len(v) == 0:
        return "—"
    m, s = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    if abs(m) >= 1000:
        return f"{m:.0f}"
    return f"{m:.2f} ± {s:.2f}" if len(v) > 1 else f"{m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "metrics.json"))):
        j = json.load(open(f, encoding="utf-8"))
        runs.setdefault(j["config"]["name"], []).append(j["metrics"])
    order = [n for n in ("hard", "data_only", "profile1d") if n in runs] + \
            [n for n in runs if n not in ("hard", "data_only", "profile1d")]
    first = runs[order[0]][0]
    head = ["величина"] + [f"{n} (сидов: {len(runs[n])})" for n in order] + ["данные МКЭ", "«среднее поле»"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    table = []
    for key, label in ROWS:
        vals = [fmt([m[key] for m in runs[n] if key in m]) for n in order]
        fem = f"{first[FEM[key]]:.2f}" if key in FEM else ""
        base = f"{first[BASE[key]]:.2f}" if key in BASE else ""
        lines.append("| " + " | ".join([label] + vals + [fem, base]) + " |")
        table.append([key] + vals + [fem, base])
    thr = (f"\nПороги п. 11 в нынешней формулировке на этой тестовой выборке: RMSE σ_z ≤ "
           f"{first['thr_spec_sigma_z']:.1f}, σ_θ ≤ {first['thr_spec_sigma_theta']:.1f}, "
           f"σ_r и τ_rz ≤ {first['thr_spec_sigma_r_tau']:.1f} МПа. Медиана max|σ| по режимам "
           f"{first['max_abs_sigma_median']:.0f} МПа, то есть 1 % — около "
           f"{first['max_abs_sigma_median'] / 100:.1f} МПа. Тестовых режимов {first['n_test']}, "
           f"обучающих {first['n_train']}, валидационных {first['n_val']}.\n")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n" + thr)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metric"] + order + ["fem", "meanfield"])
            w.writerows(table)
    print("\n".join(lines) + "\n" + thr)


if __name__ == "__main__":
    main()
