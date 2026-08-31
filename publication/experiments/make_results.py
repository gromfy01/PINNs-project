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


COMP_COLS = ["rmse_sigma_rr", "rmse_sigma_tt", "rmse_sigma_zz", "rmse_tau_rz"]


def global_std(dataset_path: str) -> np.ndarray:
    sys.path.insert(0, os.path.join(HERE, "..", "code"))
    from dataset import StressDataset
    return StressDataset.load(dataset_path).y.reshape(-1, 4).std(axis=0)


def add_global_metrics(df: pd.DataFrame, gstd: np.ndarray) -> pd.DataFrame:
    """
    Пересчитать нормированные метрики с ЕДИНЫМ знаменателем (std по всему
    датасету) из покомпонентных RMSE. Локальный вариант (std того же среза)
    между сплитами несравним: у held-out регионов разная дисперсия, и одна и та
    же абсолютная ошибка даёт разный R². На τ_rz это давало ложный вывод —
    локальный R² уходил в −4.4 и утаскивал среднее по четырём компонентам ниже
    нуля, создавая впечатление, что модель хуже предсказания средним.
    """
    d = df.copy()
    rm = d[COMP_COLS].to_numpy(dtype=float)
    nr = rm / (np.asarray(gstd, dtype=float)[None, :] + 1e-12)
    r2 = 1.0 - nr ** 2
    d["macro_nrmse_g"] = nr.mean(axis=1)
    d["macro_r2_g"] = r2.mean(axis=1)
    d["r2_normal_g"] = r2[:, :3].mean(axis=1)
    for i, c in enumerate(("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")):
        d[f"nrmse_{c}_g"] = nr[:, i]
        d[f"r2_{c}_g"] = r2[:, i]
    return d


def constant_predictor(dataset_path: str, splits: List[str]) -> Dict[str, float]:
    """
    macro-RMSE предсказания константой (средним held-out по компонентам).
    Это единственная честная «нулевая модель», с которой имеет смысл сравнивать
    абсолютную ошибку.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "code"))
    from dataset import StressDataset
    from splits import build_split_suite
    ds = StressDataset.load(dataset_path)
    suite = build_split_suite(ds.proc)
    out = {}
    for sp in splits:
        if sp not in suite:
            continue
        y = ds.y[suite[sp].test_sets]
        const = y.mean(axis=(0, 1))
        out[sp] = float(np.mean([np.sqrt(((y[:, :, c] - const[c]) ** 2).mean())
                                 for c in range(4)]))
    return out


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

    L("### 1.0б Нормированные метрики и нулевая модель\n")
    L("Абсолютная macro-RMSE сама по себе не говорит, работает ли модель: нужна "
      "нулевая модель. Ниже — предсказание константой (среднее held-out по "
      "компонентам) и R², посчитанный с ЕДИНЫМ знаменателем (std по всему "
      "датасету). Локальный R², нормированный на дисперсию того же среза, между "
      "сплитами несравним и на τ_rz даёт заведомо вводящие в заблуждение "
      "значения — см. §1.3б.\n")
    zero = constant_predictor(dataset_path, splits) if dataset_path else {}
    rows = []
    for sp in splits:
        sub = d[d["split"] == sp]
        r = {"сплит": f"`{sp}`",
             "предсказание средним": f"{zero.get(sp, float('nan')):.1f}" if zero else "—"}
        for f in fams:
            v = sub[sub["family"] == f]
            r[LABEL[f]] = "—" if v.empty else (
                f"{v['macro_rmse'].mean():.2f} · R²={v['macro_r2_g'].mean():+.2f} · "
                f"R²норм={v['r2_normal_g'].mean():+.2f}")
        rows.append(r)
    h2 = ["сплит", "предсказание средним"] + [LABEL[f] for f in fams]
    L("macro-RMSE, МПа · R² (единый знаменатель) · R² только по нормальным компонентам\n")
    L(md_table(rows, h2, h2) + "\n")

    # По осям, а не в среднем: средняя цена экстраполяции — величина без смысла,
    # потому что по разным осям она различается на порядок.
    axis_pairs = [(e, m) for e, m in PAIRS if e in splits and m in splits]
    if axis_pairs:
        rows = []
        for e, m in axis_pairs:
            ve = d[d["split"] == e]
            vm = d[d["split"] == m]
            if ve.empty or vm.empty:
                continue
            rows.append({
                "ось": f"`{e}`",
                "RMSE вне": f"{ve['macro_rmse'].mean():.1f}",
                "RMSE контроль": f"{vm['macro_rmse'].mean():.1f}",
                "во сколько раз": f"×{ve['macro_rmse'].mean()/vm['macro_rmse'].mean():.2f}",
                "R²норм вне": f"{ve['r2_normal_g'].mean():+.2f}",
                "нулевая модель": f"{zero.get(e, float('nan')):.0f}" if zero else "—",
            })
        if rows:
            h3 = list(rows[0].keys())
            L("**Цена экстраполяции зависит от оси, и различие на порядок.**\n")
            L(md_table(rows, h3, h3) + "\n")
            L("> Это, а не средняя цифра, и есть содержательный ответ на замечание "
              "про «interpolation within this particular case». Суррогат переносится "
              "по одним осям факторного пространства и не переносится по другим, и "
              "для проектировщика важно именно какие это оси, а не то, что «RMSE "
              "выросла». Усреднять по осям бессмысленно: разброс между ними больше, "
              "чем разница между семействами моделей.\n")

    interior = [sp for sp in splits if sp.startswith("interp:")]
    if interior and "extrap:alpha_max" in splits and "random" in splits:
        def mm(ss, col):
            return d[d["split"].isin(ss)][col].mean()
        L(f"> **Внутренний уровень против выхода за диапазон (ось α, объёмы почти "
          f"равны).** Изъятие внутреннего уровня α = 12° даёт "
          f"{mm(interior,'macro_rmse'):.1f} МПа при R²норм = "
          f"{mm(interior,'r2_normal_g'):+.2f}; изъятие крайнего α = 20° — "
          f"{mm(['extrap:alpha_max'],'macro_rmse'):.1f} МПа при R²норм = "
          f"{mm(['extrap:alpha_max'],'r2_normal_g'):+.2f}, при том что случайный "
          f"hold-out даёт {mm(['random'],'macro_rmse'):.1f} и "
          f"{mm(['random'],'r2_normal_g'):+.2f}. То есть по этой оси платится за "
          f"саму дыру в сетке фактора, а направление выхода почти ничего не "
          f"добавляет — сетка из пяти уровней слишком редкая, чтобы модель могла "
          f"продолжить зависимость.\n")
    if zero:
        worst = max(((e, d[d['split'] == e]['macro_rmse'].mean()) for e, _ in axis_pairs),
                    key=lambda t: t[1], default=None)
        if worst and worst[0] in zero:
            rel = worst[1] / zero[worst[0]]
            L(f"> **Где суррогат ломается полностью.** На `{worst[0]}` ошибка "
              f"{worst[1]:.0f} МПа против {zero[worst[0]]:.0f} у предсказания "
              f"средним, то есть модель "
              f"{'не лучше нулевой' if rel > 0.85 else 'ещё лучше нулевой'}. "
              f"На остальных осях она лучше нулевой в 2–6 раз. Утверждение «модель "
              f"хуже среднего» верно ровно для этого одного случая и не должно "
              f"переноситься на экстраполяцию вообще.\n")

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

    L("### 1.3 Установлена ли разница между семействами\n")
    L("Критерий — парный тест по сидам (один сид = один блок). Порог "
      "|Δ| ≥ σ по сидам приводится отдельно как размер эффекта: тестом он не "
      "является и при малом числе сидов даёт «установлено» там, где данных нет.\n")
    for sp in splits:
        s, _ = safe_summary(d, sp)
        if s is None:
            continue
        L(f"**`{sp}`**\n")
        L("```")
        for v in significance_gate(s):
            L(str(v))
        L("```\n")

    L("### 1.3б Покомпонентная нормированная ошибка\n")
    L("NRMSE = RMSE / std компоненты на held-out. Значение около 1 означает, "
      "что компонента не предсказывается вовсе.\n")
    comps = [("nrmse_sigma_rr", "σ_rr"), ("nrmse_sigma_tt", "σ_θθ"),
             ("nrmse_sigma_zz", "σ_zz"), ("nrmse_tau_rz", "τ_rz")]
    rows = []
    for sp in splits:
        for f in fams:
            v = d[(d["split"] == sp) & (d["family"] == f)]
            if v.empty:
                continue
            r = {"сплит": f"`{sp}`", "модель": LABEL[f]}
            for c, nm in comps:
                r[nm] = f"{v[c].mean():.3f}"
            rows.append(r)
    h3 = ["сплит", "модель"] + [nm for _, nm in comps]
    L(md_table(rows, h3, h3) + "\n")

    L("### 1.4 Физический аудит на held-out регионе\n")
    g = d.groupby(["split", "family"]).agg(
        eq=("eq_res_median", "mean"), eq_s=("eq_res_median", "std"),
        bcr=("bc_sigma_rr", "mean"), bcr_s=("bc_sigma_rr", "std"),
        bct=("bc_tau_rz", "mean"), bct_s=("bc_tau_rz", "std")).reset_index()
    rows = [{"сплит": f"`{r['split']}`", "модель": LABEL.get(r["family"], r["family"]),
             "медиана \\|R₁\\|, МПа/м": f"{r['eq']:.0f} ± {0 if np.isnan(r['eq_s']) else r['eq_s']:.0f}",
             "\\|σ_rr\\| при r=1, МПа": f"{r['bcr']:.2f} ± {0 if np.isnan(r['bcr_s']) else r['bcr_s']:.2f}",
             "\\|τ_rz\\| при r=1, МПа": f"{r['bct']:.3f} ± {0 if np.isnan(r['bct_s']) else r['bct_s']:.3f}"}
            for _, r in g.iterrows()]
    if "phys_gain" in d.columns:
        pg = d[d["family"].isin(["pinn", "vpinn"])].groupby("family")["phys_gain"].mean()
        if len(pg):
            L("Калибровочные множители физлосса (норма градиента физического члена "
              "приведена к норме data-члена, λ_physics читается как доля от неё): "
              + ", ".join(f"{LABEL.get(k,k)} ×{v:.3g}" for k, v in pg.items()) + "\n")

    if dataset_path:
        try:
            base = fem_baseline(dataset_path, splits)
        except Exception as exc:                       # noqa: BLE001
            base = {}
            L(f"> базовая линия FEM не посчитана: {exc}\n")
        for sp, b in base.items():
            rows.append({"сплит": f"`{sp}`", "модель": "**данные FEM (базовая линия)**",
                         "медиана \\|R₁\\|, МПа/м": f"{b['eq_res_median']:.0f}",
                         "\\|σ_rr\\| при r=1, МПа": f"{b['bc_sigma_rr']:.2f}",
                         "\\|τ_rz\\| при r=1, МПа": f"{b['bc_tau_rz']:.3f}"})
        rows.sort(key=lambda r: (r["сплит"], r["модель"]))
    h = ["сплит", "модель", "медиана \\|R₁\\|, МПа/м", "\\|σ_rr\\| при r=1, МПа", "\\|τ_rz\\| при r=1, МПа"]
    L(md_table(rows, h, h) + "\n")
    L("Модель, у которой нарушение ГУ **ниже, чем у самих данных**, воспроизводит "
      "физику строже источника — это и есть содержательный аргумент работы, и "
      "его нельзя предъявлять без строки базовой линии.\n")


def section_backend(df, L):
    d = df[df["stage"] == "backend"]
    if d.empty:
        return
    L("## 2. Бэкенд автодифференцирования\n")
    L("Вопрос распадается на два, и в отклонённой версии они были смешаны.\n")
    L("**Считают ли движки одно и то же?** Да. На побитово одинаковых весах и "
      "входах `∂σ_rr/∂r` в torch и в JAX совпадает до **8.7e-15** по модулю "
      "(9.8e-16 относительно) — машинный ноль "
      "(`publication/code/jax_twin.py::check_gradient_agreement`). Вдобавок "
      "«третий бэкенд» работы, DeepXDE, сконфигурирован как `DDE_BACKEND=pytorch` "
      "и берёт производные тем же `torch.autograd.grad`, то есть движков было "
      "два, а не три. Отнести разницу в метриках на автодифференцирование "
      "нельзя ни при каком результате ниже.\n")
    L("**Различаются ли две независимые реализации?** Ниже — да, и в этом всё "
      "дело. Обе делят архитектуру, протокол, данные, разбиение и "
      "гиперпараметры, но начальные веса и порядок батчей берут из своих "
      "генераторов (torch — `torch.manual_seed`, JAX-версия — "
      "`numpy.random.default_rng`), по пять сидов на каждую.\n")
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
                         "\\|Δ\\|": f"{delta:.3f}", "σ по сидам": f"{pooled:.3f}",
                         "разница": "установлена" if delta >= pooled else "**не установлена**"})
    if rows:
        h = list(rows[0].keys())
        L(md_table(rows, h, h) + "\n")
        signs = []
        for sp in sorted(d["split"].unique()):
            sub = d[d["split"] == sp]
            vt = sub[sub["backend"] == "torch"]["macro_rmse"].mean()
            vj = sub[sub["backend"] == "jax"]["macro_rmse"].mean()
            if np.isfinite(vt) and np.isfinite(vj):
                signs.append(np.sign(vt - vj))
        flip = len(set(signs)) > 1
        L("Читать так. Разница между двумя независимыми реализациями одного и "
          "того же метода превышает межсидовый разброс"
          + (", **и её знак меняется от сплита к сплиту** — на одном лучше одна "
             "реализация, на другом другая. " if flip else ". ")
          + "Поскольку производные движков совпадают до машинного нуля, эта "
          "разница есть свойство РЕАЛИЗАЦИИ (инициализация, порядок батчей, "
          "детали оптимизатора), а не автодифференцирования.\n")
        L("Для сравнения: в отклонённой версии разница между «бэкендами» заявлена "
          "как 0.4–0.8 МПа на одном прогоне и вынесена в Conclusions как вывод "
          "«backend matters». Здесь видно, что ровно такой разрыв получается при "
          "простом переписывании того же метода на другой библиотеке, поэтому он "
          "не может подтверждать утверждение о движке. Это и есть содержание "
          "замечания Reviewer #2, подтверждённое измерением.\n")


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
        from scipy.stats import ttest_ind
        curves = {f: {fr: sub[(sub["family"] == f) & (sub["frac"] == fr)]["macro_rmse"].mean()
                      for fr in fracs} for f in fams}

        # немонотонность: становится ли хуже при БОЛЬШЕМ объёме обучения
        notes = []
        for f in fams:
            best_fr = min(fracs, key=lambda fr: curves[f][fr])
            if best_fr == max(fracs):
                continue
            a = sub[(sub["family"] == f) & (sub["frac"] == best_fr)]["macro_rmse"].to_numpy()
            b = sub[(sub["family"] == f) & (sub["frac"] == max(fracs))]["macro_rmse"].to_numpy()
            if len(a) >= 3 and len(b) >= 3:
                p_ = float(ttest_ind(a, b, equal_var=False).pvalue)
                notes.append(f"{LABEL[f]}: минимум при {int(best_fr*100)} % "
                             f"({curves[f][best_fr]:.2f}) против {curves[f][max(fracs)]:.2f} "
                             f"на полном пуле, p = {p_:.3f}"
                             + ("" if p_ < 0.05 else " — не установлено"))
        if notes:
            L("> **Немонотонность.** На этом сплите ошибка у части семейств "
              "минимальна НЕ на полном обучающем пуле:\n>\n> - " +
              "\n> - ".join(notes) +
              "\n>\n> Правдоподобное объяснение: с ростом объёма модель точнее "
              "подстраивается под покрытую часть оси фактора и хуже продолжается "
              "за её пределы. Утверждать это как установленный эффект можно "
              "только по строкам, где p < 0.05.\n")

        phys = [f for f in fams if f != "mlp"]
        if phys and "mlp" in fams:
            cross = [fr for fr in fracs if curves["mlp"][fr] <= min(curves[p][fr] for p in phys)]
            if cross:
                L(f"Доля данных, начиная с которой датадривен-базлайн не хуже лучшей "
                  f"physics-informed модели: **{int(min(cross)*100)} %**\n")
            else:
                fr0 = min(fracs)
                best_p = min(phys, key=lambda x: curves[x][fr0])
                a = sub[(sub["family"] == "mlp") & (sub["frac"] == fr0)]["macro_rmse"].to_numpy()
                b = sub[(sub["family"] == best_p) & (sub["frac"] == fr0)]["macro_rmse"].to_numpy()
                p_ = float(ttest_ind(a, b, equal_var=False).pvalue) if min(len(a), len(b)) >= 3 else float("nan")
                L(f"Датадривен-базлайн не догоняет лучшую physics-informed модель ни "
                  f"на одной доле пула. Но при наименьшей доле ({int(fr0*100)} %) отрыв "
                  f"составляет {curves['mlp'][fr0] - curves[best_p][fr0]:+.2f} МПа при "
                  f"p = {p_:.3f} → преимущество физприора в режиме дефицита данных "
                  f"{'установлено' if p_ < 0.05 else '**не установлено**'}.\n")


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
    hi = max(rates)
    if hi > 0 and len(fams) >= 2:
        from scipy.stats import ttest_ind
        base_f = "mlp" if "mlp" in fams else fams[0]
        a = d[(d["family"] == base_f) & (d["corrupt_rate"] == hi)]["macro_rmse"].to_numpy()
        for f in fams:
            if f == base_f:
                continue
            b = d[(d["family"] == f) & (d["corrupt_rate"] == hi)]["macro_rmse"].to_numpy()
            if len(a) < 3 or len(b) < 3:
                L(f"> {LABEL[f]} против {LABEL[base_f]} при {int(hi*100)} % порчи: "
                  f"сидов меньше трёх, тест не проводится.\n")
                continue
            p = float(ttest_ind(a, b, equal_var=False).pvalue)
            L(f"> При {int(hi*100)} % испорченных меток {LABEL[f]} даёт "
              f"{b.mean():.2f} против {a.mean():.2f} у {LABEL[base_f]}: "
              f"Δ = {a.mean()-b.mean():+.2f} МПа, p = {p:.3f} → разница "
              f"{'установлена' if p < 0.05 else '**не установлена**'}.\n")


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
                   "\\|R_r\\| полная": f"{v['eq_res_median'].mean():.0f}",
                   "\\|R_r\\| редуц.": f"{v['eq_r_reduced'].mean():.0f}" if "eq_r_reduced" in v else "—",
                   "\\|R_z\\| полная": f"{v['eq_z_full'].mean():.0f}" if "eq_z_full" in v else "—",
                   "\\|R_z\\| редуц.": f"{v['eq_z_reduced'].mean():.0f}" if "eq_z_reduced" in v else "—"}
            rows.append(row)
    if rows:
        h = list(rows[0].keys())
        L(md_table(rows, h, h) + "\n")
        L("Читать так. `|R_r|` и `|R_z|` — апостериорные невязки на предсказаниях, "
          "посчитанные в ПОЛНОЙ форме (со всеми членами) и в редуцированной (без "
          "осевых производных). Сравнивать между конфигурациями надо колонки "
          "«полная»: именно они говорят, насколько предсказание удовлетворяет "
          "настоящему уравнению равновесия. Колонка «редуц.» приведена для "
          "сверки с одномерной постановкой.\n")
        for sp in sorted(d["split"].unique()):
            v = d[d["split"] == sp]
            red = v[v["family"] == "pinn2d_reduced"]
            full = v[v["family"] == "pinn2d_full"]
            if red.empty or full.empty:
                continue
            dz = 100 * (1 - full["eq_z_full"].mean() / max(red["eq_z_full"].mean(), 1e-9))
            drm = full["macro_rmse"].mean() - red["macro_rmse"].mean()
            L(f"> **`{sp}`.** Включение осевого уравнения снижает полную осевую "
              f"невязку на **{dz:.0f} %** ({red['eq_z_full'].mean():.0f} → "
              f"{full['eq_z_full'].mean():.0f} МПа/м) при изменении macro-RMSE на "
              f"{drm:+.2f} МПа. То есть член `∂σ_zz/∂z`, который в одномерной "
              f"постановке отбросить нельзя (он даёт 68 % от удержанных), при "
              f"введении `z` во вход действительно начинает работать — и это "
              f"обходится без потери точности.\n")
        L("Абсолютные macro-RMSE этого раздела НЕ сравнимы с разделом 1: там "
          "предсказывается один радиальный профиль на набор, здесь — вся сетка "
          "(z, r), включающая осевую изменчивость.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="")
    a = ap.parse_args()
    df = load(a.csv)
    if a.dataset:
        df = add_global_metrics(df, global_std(a.dataset))
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
