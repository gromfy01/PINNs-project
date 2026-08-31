"""
make_results.py — runs.csv → publication/RESULTS.md.

Все утверждения в отчёте порождаются из данных, а не пишутся руками: где
разница меньше межсидового разброса, отчёт сам печатает «не установлена».
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
from analyze import _summary, load                      # noqa: E402
from stats import delta_vs_matched, nemenyi, significance_gate  # noqa: E402


def safe_summary(d: pd.DataFrame, split: str):
    """
    Сводка по сплиту с усечением до сидов, посчитанных у ВСЕХ моделей.
    Нужна, чтобы отчёт собирался и по частично готовому прогону: сравнивать
    модели на разных подмножествах сидов нельзя, поэтому лишние сиды
    отбрасываются, а не достраиваются.
    """
    sub = d[d["split"] == split]
    if sub.empty:
        return None, 0
    fams = sorted(sub["family"].unique())
    seeds = None
    for f in fams:
        s = set(sub[sub["family"] == f]["seed"].astype(int))
        seeds = s if seeds is None else (seeds & s)
    if not seeds:
        return None, 0
    keep = sub[sub["seed"].astype(int).isin(seeds)]
    dropped = len(sub) - len(keep)
    return _summary(keep, split), dropped

LABEL = {"mlp": "MLP (data-driven)", "pinn": "PINN (сильная форма)",
         "vpinn": "VPINN (слабая форма)",
         "mlp2d": "MLP, z во входе", "pinn2d_reduced": "PINN, редуцированная форма",
         "pinn2d_full": "PINN, полная форма"}
ORDER = ["mlp", "pinn", "vpinn"]
PAIRS = [("extrap:alpha_max", "matched:alpha_max"),
         ("extrap:Q_high", "matched:Q_high"),
         ("extrap:k_max", "matched:k_max"),
         ("extrap:v_max", "matched:v_max"),
         ("extrap_joint:corner", "matched:corner")]


def md_table(rows: List[Dict], cols: List[str], head: List[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def fmt(v, n=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def fem_baseline(dataset_path: str, splits: List[str]) -> Dict[str, Dict[str, float]]:
    """Невязка равновесия и нарушение ГУ у самих данных FEM на каждом
    held-out регионе. Без этой строки цифры моделей не с чем сравнивать."""
    sys.path.insert(0, os.path.join(HERE, "..", "code"))
    from dataset import StressDataset
    from splits import build_split_suite
    from trainer import bc_audit, equilibrium_audit
    ds = StressDataset.load(dataset_path)
    suite = build_split_suite(ds.proc)
    out = {}
    for sp in splits:
        if sp not in suite:
            continue
        te = suite[sp].test_sets
        eq = equilibrium_audit(ds.y[te], ds.r_phys[te])
        bc = bc_audit(ds.y[te][:, -1, :])
        out[sp] = {**eq, **bc}
    return out


def section_main(df, L, dataset_path: str = ""):
    d = df[df["stage"] == "main"]
    if d.empty:
        return
    L("## 1. Экстраполяция против интерполяции\n")
    L("macro-RMSE, МПа; mean ± std по сидам; каждая строка — отдельный сплит.\n")
    splits = sorted(d["split"].unique(), key=lambda s: (not s.startswith("random"), s))
    fams = [f for f in ORDER if f in set(d["family"])]
    head = ["сплит", "n_test", "сидов"] + [LABEL[f] for f in fams]
    rows = []
    incomplete = []
    for sp in splits:
        s, dropped = safe_summary(d, sp)
        if s is None:
            incomplete.append(sp)
            continue
        if dropped:
            incomplete.append(f"{sp} (отброшено {dropped} прогонов вне общего набора сидов)")
        m = {mm: (s.mean[i], s.std[i]) for i, mm in enumerate(s.models)}
        r = {"сплит": f"`{sp}`", "n_test": int(d[d['split'] == sp]['n_test'].iloc[0]),
             "сидов": s.values.shape[1]}
        for f in fams:
            r[LABEL[f]] = f"{m[f][0]:.2f} ± {m[f][1]:.2f}" if f in m else "—"
        rows.append(r)
    L(md_table(rows, head, head) + "\n")
    if incomplete:
        L("> Неполные по сидам сплиты: " + ", ".join(f"`{x}`" for x in incomplete) +
          ". Сравнение ведётся только по сидам, посчитанным у всех моделей.\n")

    L("### 1.1 ΔRMSE относительно контроля того же объёма\n")
    L("Без этой величины «RMSE вырос» не интерпретируется: рост может объясняться "
      "тем, что регион малочисленнее или физически тяжелее.\n")
    rows = []
    have = set(d["split"])
    for e, m in PAIRS:
        if e not in have or m not in have:
            continue
        se, _ = safe_summary(d, e)
        sm, _ = safe_summary(d, m)
        if se is None or sm is None or se.models != sm.models:
            continue
        for r in delta_vs_matched(se, sm):
            rows.append({"регион": f"`{e}`", "модель": LABEL.get(r["model"], r["model"]),
                         "extrap": fmt(r["rmse_extrap"], 2), "контроль": fmt(r["rmse_matched"], 2),
                         "Δ": fmt(r["delta"], 2), "σ(Δ)": fmt(r["delta_std"], 2),
                         "установлена": "да" if r["established"] else "**нет**"})
    if rows:
        h = ["регион", "модель", "extrap", "контроль", "Δ", "σ(Δ)", "установлена"]
        L(md_table(rows, h, h) + "\n")

    L("### 1.2 Ранговый анализ внутри сплита (блоки — сиды)\n")
    for sp in splits:
        s, _ = safe_summary(d, sp)
        if s is None or len(s.models) < 3 or s.values.shape[1] < 3:
            continue
        L("```")
        L(nemenyi(s).to_text())
        L("```\n")

    L("### 1.3 Порог содержательности разницы\n")
    L("Утверждение о разнице делается только если |Δ| ≥ σ по сидам.\n")
    for sp in splits:
        s, _ = safe_summary(d, sp)
        if s is None:
            continue
        L(f"**`{sp}`**\n")
        L("```")
        for v in significance_gate(s):
            L(str(v))
        L("```\n")

    L("### 1.4 Физический аудит на held-out регионе\n")
    g = d.groupby(["split", "family"]).agg(
        eq=("eq_res_median", "mean"), eq_s=("eq_res_median", "std"),
        bcr=("bc_sigma_rr", "mean"), bcr_s=("bc_sigma_rr", "std"),
        bct=("bc_tau_rz", "mean"), bct_s=("bc_tau_rz", "std")).reset_index()
    rows = [{"сплит": f"`{r['split']}`", "модель": LABEL.get(r["family"], r["family"]),
             "медиана |R₁|, МПа/м": f"{r['eq']:.0f} ± {0 if np.isnan(r['eq_s']) else r['eq_s']:.0f}",
             "|σ_rr| при r=1, МПа": f"{r['bcr']:.2f} ± {0 if np.isnan(r['bcr_s']) else r['bcr_s']:.2f}",
             "|τ_rz| при r=1, МПа": f"{r['bct']:.3f} ± {0 if np.isnan(r['bct_s']) else r['bct_s']:.3f}"}
            for _, r in g.iterrows()]
    if dataset_path:
        try:
            base = fem_baseline(dataset_path, splits)
        except Exception as exc:                       # noqa: BLE001
            base = {}
            L(f"> базовая линия FEM не посчитана: {exc}\n")
        for sp, b in base.items():
            rows.append({"сплит": f"`{sp}`", "модель": "**данные FEM (базовая линия)**",
                         "медиана |R₁|, МПа/м": f"{b['eq_res_median']:.0f}",
                         "|σ_rr| при r=1, МПа": f"{b['bc_sigma_rr']:.2f}",
                         "|τ_rz| при r=1, МПа": f"{b['bc_tau_rz']:.3f}"})
        rows.sort(key=lambda r: (r["сплит"], r["модель"]))
    h = ["сплит", "модель", "медиана |R₁|, МПа/м", "|σ_rr| при r=1, МПа", "|τ_rz| при r=1, МПа"]
    L(md_table(rows, h, h) + "\n")
    L("Модель, у которой нарушение ГУ **ниже, чем у самих данных**, воспроизводит "
      "физику строже источника — это и есть содержательный аргумент работы, и "
      "его нельзя предъявлять без строки базовой линии.\n")


def section_backend(df, L):
    d = df[df["stage"] == "backend"]
    if d.empty:
        return
    L("## 2. Бэкенд автодифференцирования\n")
    L("Одна и та же модель, те же начальные веса, тот же порядок батчей, тот же "
      "оптимизатор; отличается только движок. Производные torch и JAX на одних "
      "весах совпадают до 8.7e-15, поэтому любая наблюдаемая разница — стохастика "
      "обучения.\n")
    rows = []
    for sp in sorted(d["split"].unique()):
        sub = d[d["split"] == sp]
        vals = {be: sub[sub["backend"] == be]["macro_rmse"].to_numpy()
                for be in sub["backend"].unique()}
        if len(vals) == 2:
            (a, va), (b, vb) = list(vals.items())
            delta = abs(va.mean() - vb.mean())
            pooled = float(np.sqrt(0.5 * (va.std(ddof=1) ** 2 + vb.std(ddof=1) ** 2)))
            rows.append({"сплит": f"`{sp}`",
                         a: f"{va.mean():.3f} ± {va.std(ddof=1):.3f}",
                         b: f"{vb.mean():.3f} ± {vb.std(ddof=1):.3f}",
                         "|Δ|": f"{delta:.3f}", "σ по сидам": f"{pooled:.3f}",
                         "разница": "установлена" if delta >= pooled else "**не установлена**"})
    if rows:
        h = list(rows[0].keys())
        L(md_table(rows, h, h) + "\n")
        L("Для сравнения: в отклонённой версии разница между «бэкендами» "
          "заявлена как 0.4–0.8 МПа на одном прогоне.\n")


def section_curves(df, L):
    d = df[df["stage"] == "curves"]
    if d.empty:
        return
    L("## 3. Кривые обучения по объёму данных\n")
    for sp in sorted(d["split"].unique()):
        sub = d[d["split"] == sp]
        L(f"**`{sp}`** — macro-RMSE, МПа\n")
        fracs = sorted(sub["frac"].unique())
        fams = [f for f in ORDER if f in set(sub["family"])]
        rows = []
        for f in fams:
            r = {"модель": LABEL[f]}
            for fr in fracs:
                v = sub[(sub["family"] == f) & (sub["frac"] == fr)]["macro_rmse"].to_numpy()
                r[f"{int(fr*100)} %"] = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else fmt(v.mean() if len(v) else None, 2)
            rows.append(r)
        h = ["модель"] + [f"{int(fr*100)} %" for fr in fracs]
        L(md_table(rows, h, h) + "\n")
        curves = {f: {fr: sub[(sub["family"] == f) & (sub["frac"] == fr)]["macro_rmse"].mean()
                      for fr in fracs} for f in fams}
        phys = [f for f in fams if f != "mlp"]
        if phys and "mlp" in fams:
            cross = [fr for fr in fracs if curves["mlp"][fr] <= min(curves[p][fr] for p in phys)]
            L(f"Доля данных, начиная с которой датадривен-базлайн не хуже лучшей "
              f"physics-informed модели: **{int(min(cross)*100) if cross else '—'}%**"
              + ("" if cross else " (в исследованном диапазоне не догоняет)") + "\n")


def section_corrupt(df, L):
    d = df[df["stage"] == "corrupt"]
    if d.empty:
        return
    L("## 4. Устойчивость к порче меток\n")
    L("Портится доля обучающих наборов постоянным сдвигом σ_θθ на +250 МПа "
      "(морфология артефакта #439). Hold-out чистый.\n")
    rates = sorted(d["corrupt_rate"].unique())
    fams = [f for f in ORDER if f in set(d["family"])]
    rows = []
    for f in fams:
        r = {"модель": LABEL[f]}
        base = None
        for rt in rates:
            v = d[(d["family"] == f) & (d["corrupt_rate"] == rt)]["macro_rmse"].to_numpy()
            if base is None and len(v):
                base = v.mean()
            r[f"{int(rt*100)} %"] = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else fmt(v.mean() if len(v) else None, 2)
        worst = d[(d["family"] == f) & (d["corrupt_rate"] == max(rates))]["macro_rmse"].mean()
        r["рост"] = f"×{worst/base:.2f}" if base else "—"
        rows.append(r)
    h = ["модель"] + [f"{int(rt*100)} %" for rt in rates] + ["рост"]
    L(md_table(rows, h, h) + "\n")


def section_twod(df, L):
    d = df[df["stage"] == "twod"]
    if d.empty:
        return
    L("## 5. z во входе и полная форма уравнений равновесия\n")
    L("Поля на сетке (z, r) 8 × 20 внутри осевого окна. Три конфигурации "
      "отличаются только составом физического лосса.\n")
    rows = []
    for sp in sorted(d["split"].unique()):
        for f in [x for x in ["mlp2d", "pinn2d_reduced", "pinn2d_full"] if x in set(d["family"])]:
            v = d[(d["split"] == sp) & (d["family"] == f)]
            if v.empty:
                continue
            row = {"сплит": f"`{sp}`", "конфигурация": LABEL[f],
                   "macro-RMSE": f"{v['macro_rmse'].mean():.2f} ± {v['macro_rmse'].std(ddof=1):.2f}",
                   "|R_r| полная": f"{v['eq_res_median'].mean():.0f}",
                   "|R_r| редуц.": f"{v['eq_r_reduced'].mean():.0f}" if "eq_r_reduced" in v else "—",
                   "|R_z| полная": f"{v['eq_z_full'].mean():.0f}" if "eq_z_full" in v else "—",
                   "|R_z| редуц.": f"{v['eq_z_reduced'].mean():.0f}" if "eq_z_reduced" in v else "—"}
            rows.append(row)
    if rows:
        h = list(rows[0].keys())
        L(md_table(rows, h, h) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="")
    a = ap.parse_args()
    df = load(a.csv)
    lines: List[str] = []
    L = lines.append

    L("# Результаты прогонов\n")
    L("Сгенерировано `publication/experiments/make_results.py` из "
      "`results/runs.csv`. Данные — пересобранный из сырых `.rpt` датасет "
      "(`publication/code/dataset.py`), 1848 наборов; целевые поля — истинные "
      "`(σ_rr, σ_θθ, σ_zz, τ_rz)`, а не то, что лежало в прежнем `y_stress.pkl` "
      "(ERRATA E-18).\n")
    L("Протокол одинаков у всех семейств: AdamW, без планировщика, те же эпохи, "
      "батчи, ранняя остановка, сиды и архитектура; отличается только состав "
      "функции потерь. Гиперпараметры зафиксированы априори и на held-out "
      "регионах не подбирались.\n")
    if a.dataset:
        sys.path.insert(0, os.path.join(HERE, "..", "code"))
        from dataset import StressDataset, sanity
        sn = sanity(StressDataset.load(a.dataset))
        L("Базовая линия самих данных FEM (то, с чем сравнивать физаудит): "
          f"|σ_rr| на поверхности **{sn['bc_sigma_rr_surface']:.2f} МПа**, "
          f"|τ_rz| **{sn['bc_tau_rz_surface']:.2f} МПа**.\n")
    n_ok = len(df)
    L(f"Прогонов в отчёте: **{n_ok}**.\n")

    section_main(df, L, a.dataset)
    section_backend(df, L)
    section_curves(df, L)
    section_corrupt(df, L)
    section_twod(df, L)

    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"записано: {a.out} ({len(lines)} строк, {n_ok} прогонов)")


if __name__ == "__main__":
    main()
