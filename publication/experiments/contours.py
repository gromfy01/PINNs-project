"""
contours.py — контурные карты напряжений в плоскости (r, z).

Зачем. Одномерный профиль σ(r) — это срез поля, усреднённый по осевому окну;
по нему не видно ни того, как поле меняется вдоль оси, ни того, какую полосу
из расчёта берёт датасет. Контуры показывают и то и другое.

Что рисуется:
  --mode raw    поля прямо по узлам .rpt (неструктурированная триангуляция),
                весь осевой размах, с отметкой окна 25–75 %, из которого
                собирается обучающая выборка;
  --mode model  МКЭ против предсказания модели с z во входе и их разность,
                на сетке (z, r) 8 × 20, для σ_rr, σ_zz и τ_rz; по умолчанию
                берётся набор с МЕДИАННОЙ ошибкой, а не лучший;
  --mode alpha  одна компонента при разных полууглах волоки.

Цвет. Напряжения знакопеременные, поэтому шкала расходящаяся, симметричная
относительно нуля, с нейтральной серединой: ноль всегда читается как «нет
напряжения», а не как произвольный цвет. Для разности — та же шкала.
Радужные шкалы не используются: они создают ложные границы там, где поле
гладкое.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from raw_probe import (C_R, C_Z, axial_window, billet_radius, list_jobs,  # noqa: E402
                       parse_job_name, read_rpt)

PROC_COL = {"Q": 0, "k": 1, "alpha": 2, "mu": 3, "v": 4}

COMPS = [("sigma_rr", r"$\sigma_{rr}$", 11),
         ("sigma_tt", r"$\sigma_{\theta\theta}$", 13),
         ("sigma_zz", r"$\sigma_{zz}$", 12),
         ("tau_rz",   r"$\tau_{rz}$",   14)]
INK, MUTED = "#1a1a1a", "#8a8a8a"
DIVERGING = "RdBu_r"          # два тона + нейтральная середина, без радуги

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 170, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _sym(v: np.ndarray) -> float:
    """Симметричный предел шкалы: ноль оказывается ровно в середине."""
    a = float(np.nanpercentile(np.abs(v), 99.5))
    return a if a > 0 else 1.0


def _panel(ax, tri, vals, title, lim, cmap=DIVERGING, nlev=21):
    """
    Оси НЕ в одном масштабе: проволока длинная и тонкая (радиус ~16 мм при
    осевом размахе ~130 мм), при равных масштабах панель вырождается в полоску
    и поле не читается. Радиальная ось растянута; об этом сказано в подписи.
    """
    levels = np.linspace(-lim, lim, nlev)
    cf = ax.tricontourf(tri, vals, levels=levels, cmap=cmap, extend="both")
    ax.tricontour(tri, vals, levels=levels[::4], colors="k", linewidths=0.25, alpha=0.35)
    ax.set_title(title)
    ax.set_aspect("auto")
    return cf


def fig_raw(path_rpt: str, out: str, job: str = ""):
    """Поля по узлам одного расчёта, весь осевой размах."""
    a = read_rpt(path_rpt)
    # безразмерные координаты: r/R₀ по радиусу ЗАГОТОВКИ (не по текущей
    # поверхности — она меняется вдоль оси, в том и смысл этого рисунка),
    # z/L по полной длине модели
    R0 = billet_radius(path_rpt)
    L = float(a[:, C_Z].max() - a[:, C_Z].min())
    r = a[:, C_R] / R0
    z = (a[:, C_Z] - a[:, C_Z].min()) / L
    lo, hi = axial_window(a[:, C_Z])
    lo = (lo - a[:, C_Z].min()) / L
    hi = (hi - a[:, C_Z].min()) / L
    tri = mtri.Triangulation(r, z)

    fig, axes = plt.subplots(1, 4, figsize=(9.4, 4.0), sharey=True,
                             layout="constrained")
    for ax, (key, tex, col) in zip(axes, COMPS):
        v = a[:, col] / 1e6
        lim = _sym(v)
        cf = _panel(ax, tri, v, f"{tex}, МПа", lim)
        ax.axhline(lo, color=INK, lw=1.0, ls="--")
        ax.axhline(hi, color=INK, lw=1.0, ls="--")
        ax.set_xlabel(r"$r/R_0$")
        cb = fig.colorbar(cf, ax=ax, fraction=0.06, pad=0.03,
                          location="bottom", orientation="horizontal")
        cb.ax.tick_params(labelsize=6, color=MUTED)
    axes[0].set_ylabel(r"$z/L$")
    fig.suptitle(f"Остаточные напряжения по узлам МКЭ · {job}", fontsize=9)
    fig.text(0.5, -0.02,
             f"Пунктир — осевое окно 25–75 %: из этой полосы собирается обучающая "
             f"выборка (20 радиальных точек на набор).\n"
             f"Координаты безразмерные: $R_0$ = {R0 * 1e3:.1f} мм (заготовка), "
             f"$L$ = {L * 1e3:.1f} мм (длина модели).\n"
             f"Оси не в одном масштабе: радиальная растянута, иначе панель "
             f"вырождается в полоску ($L/R_0$ = {L / R0:.1f}).",
             ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out


def _where(spec: str) -> Optional[Dict[str, float]]:
    """'Q=0.10,v=20' → {'Q': 0.1, 'v': 20.0}."""
    out = {}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out or None


def _pick_set(d, err, which, where):
    """Общий выбор набора для карт модели: медиана внутри отобранного режима."""
    if which >= 0:
        return which
    pool = np.ones(len(err), dtype=bool)
    if where:
        for key, val in where.items():
            pool &= np.isclose(d["proc"][:, PROC_COL[key]], val)
        if not pool.any():
            raise SystemExit(f"нет тестовых наборов с {where}")
    idx = np.flatnonzero(pool)
    w = int(idx[np.argsort(err[idx])[len(idx) // 2]])
    tag = (" при " + ", ".join(f"{k} = {v:g}" for k, v in where.items())) if where else ""
    print(f"выбран набор {w}{tag}: медианная ошибка {err[w]:.2f} МПа "
          f"(из {len(idx)} наборов, диапазон {err[idx].min():.2f}–{err[idx].max():.2f})")
    return w


def _regime(proc):
    return (f"Q = {proc[0]:g}, k = {proc[1]:g}, α = {proc[2]:g}°, "
            f"μ = {proc[3]:g}, v = {proc[4]:g} м/мин")


ROWS4 = [(r"$\sigma_{rr}$", 0), (r"$\sigma_{\theta\theta}$", 1),
         (r"$\sigma_{zz}$", 2), (r"$\tau_{rz}$", 3)]


def fig_fields(npz: str, out: str, which: int = -1,
               where: Optional[Dict[str, float]] = None):
    """
    Только поля: МКЭ (верхний ряд) против модели (нижний), четыре компоненты.

    Невязки сюда НЕ включены намеренно — у них своя шкала и свой вопрос;
    смешивание в одной сетке заставляло читать разными глазами соседние
    панели. Ошибки — в fig_errors.
    """
    d = np.load(npz, allow_pickle=True)
    yt_all, yp_all = d["y_true"], d["y_pred"]
    err = np.sqrt(((yp_all - yt_all) ** 2).mean(axis=(1, 2, 3)))
    which = _pick_set(d, err, which, where)
    yt, yp = yt_all[which], yp_all[which]
    R = d["r"][which]
    Z = np.repeat(d["z"][which][:, None], yt.shape[1], axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4), sharex=True, sharey=True,
                             layout="constrained")
    for j, (tex, c) in enumerate(ROWS4):
        lim = _sym(np.concatenate([yt[:, :, c].ravel(), yp[:, :, c].ravel()]))
        lev = np.linspace(-lim, lim, 21)
        for i, (v, who) in enumerate(((yt[:, :, c], "МКЭ"), (yp[:, :, c], "модель"))):
            ax = axes[i, j]
            cf = ax.contourf(R, Z, v, levels=lev, cmap=DIVERGING, extend="both")
            ax.contour(R, Z, v, levels=lev[::5], colors="k",
                       linewidths=0.25, alpha=0.35)
            ax.set_title(f"{tex}  {who}" if i == 0 else who)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
            if j == 0:
                ax.set_ylabel(r"$z/L$")
        cb = fig.colorbar(cf, ax=axes[:, j], fraction=0.07, pad=0.02,
                          location="bottom", orientation="horizontal",
                          ticks=np.linspace(-lim, lim, 5))
        cb.set_label("МПа", fontsize=7)
        cb.ax.tick_params(labelsize=6, color=MUTED)
        axes[1, j].set_xlabel(r"$r/R$")
    fig.suptitle(f"Поля: МКЭ против модели.  Набор с медианной ошибкой:  "
                 f"{_regime(d['proc'][which])}", fontsize=8.5)
    fig.text(0.5, -0.02,
             r"Шкала общая для МКЭ и модели внутри каждой компоненты. "
             r"$r/R = 1$ — свободная поверхность; $z/L$ — доля осевого окна."
             "\nОси не в одном масштабе: радиальная растянута.",
             ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_errors(npz: str, out: str, which: int = -1,
               where: Optional[Dict[str, float]] = None,
               diff_mode: str = "rel"):
    """
    Только невязки (модель − МКЭ) по четырём компонентам.

    diff_mode='rel' — в % от масштаба своей компоненты, шкала ОБЩАЯ на все
    четыре панели, поэтому компоненты сравнимы между собой напрямую.
    diff_mode='abs' — в МПа, своя шкала у каждой панели.
    """
    if diff_mode not in ("rel", "abs"):
        raise ValueError(f"diff_mode: 'rel' или 'abs', получено {diff_mode!r}")
    d = np.load(npz, allow_pickle=True)
    yt_all, yp_all = d["y_true"], d["y_pred"]
    err = np.sqrt(((yp_all - yt_all) ** 2).mean(axis=(1, 2, 3)))
    which = _pick_set(d, err, which, where)
    yt, yp = yt_all[which], yp_all[which]
    R = d["r"][which]
    Z = np.repeat(d["z"][which][:, None], yt.shape[1], axis=1)
    clean = d["clean_mask"] if "clean_mask" in d.files else None

    scales = [_sym(np.concatenate([yt[:, :, c].ravel(), yp[:, :, c].ravel()]))
              for _, c in ROWS4]
    rel_lim = max(100.0 * _sym((yp[:, :, c] - yt[:, :, c]).ravel()) / sc
                  for (_, c), sc in zip(ROWS4, scales))

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.4), sharey=True,
                             layout="constrained")
    for j, (tex, c) in enumerate(ROWS4):
        diff = yp[:, :, c] - yt[:, :, c]
        rmse = float(np.sqrt((diff ** 2).mean()))
        if diff_mode == "rel":
            v, L = 100.0 * diff / scales[j], rel_lim
        else:
            v, L = diff, _sym(diff.ravel())
        ax = axes[j]
        lev = np.linspace(-L, L, 21)
        cf = ax.contourf(R, Z, v, levels=lev, cmap=DIVERGING, extend="both")
        ax.contour(R, Z, v, levels=lev[::5], colors="k",
                   linewidths=0.25, alpha=0.35)
        ax.set_title(f"{tex}\nRMSE {rmse:.1f} МПа = "
                     f"{100 * rmse / scales[j]:.1f} % масштаба", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel(r"$r/R$")
        if j == 0:
            ax.set_ylabel(r"$z/L$")
        if diff_mode == "abs":
            cb = fig.colorbar(cf, ax=ax, fraction=0.07, pad=0.02,
                              location="bottom", orientation="horizontal")
            cb.set_label("МПа", fontsize=7)
            cb.ax.tick_params(labelsize=6, color=MUTED)
    if diff_mode == "rel":
        cb = fig.colorbar(cf, ax=axes, fraction=0.05, pad=0.02,
                          location="bottom", orientation="horizontal",
                          ticks=np.linspace(-rel_lim, rel_lim, 7))
        cb.set_label("невязка, % от масштаба своей компоненты", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, color=MUTED)
        cb.ax.set_xticklabels([f"{t:.0f}" for t in
                               np.linspace(-rel_lim, rel_lim, 7)])

    summ = []
    for tex, c in ROWS4:
        t, q = yt_all[..., c].ravel(), yp_all[..., c].ravel()
        r2 = 1.0 - ((q - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        part = f"{tex}: {np.sqrt(((q - t) ** 2).mean()):.1f} МПа, $R^2$ {r2:.2f}"
        if clean is not None and not clean.all():
            tc = yt_all[clean][..., c].ravel(); qc = yp_all[clean][..., c].ravel()
            r2c = 1.0 - ((qc - tc) ** 2).sum() / ((tc - tc.mean()) ** 2).sum()
            part += f" (чистые {np.sqrt(((qc - tc) ** 2).mean()):.1f} / {r2c:.2f})"
        summ.append(part)
    fig.suptitle(f"Невязка модель − МКЭ.  Набор с медианной ошибкой:  "
                 f"{_regime(d['proc'][which])}", fontsize=8.5)
    note = ("Заголовок панели — RMSE ЭТОГО набора. "
            + ("Шкала общая на все четыре панели: компоненты сравнимы напрямую."
               if diff_mode == "rel" else
               "Шкала своя у каждой панели: сравнивать между собой нельзя.")
            + f"\nПо всему тесту (n = {len(yt_all)}): " + ";  ".join(summ) + ".")
    if clean is not None and not clean.all():
        note += (f"\n«Чистые» — {int(clean.sum())} наборов после снятия "
                 f"{int((~clean).sum())} незавершённых расчётов (E-24).")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out



def _align(npzs: List[str]):
    """
    Несколько прогонов → общий набор тестовых наборов и выровненные предсказания.

    Прогоны могут иметь разный порядок и разный состав теста; сравнивать можно
    только на пересечении, иначе разница между семействами смешается с разницей
    в тестовой выборке.
    """
    ds = [np.load(p, allow_pickle=True) for p in npzs]
    common = set(ds[0]["test_sets"].tolist())
    for d in ds[1:]:
        common &= set(d["test_sets"].tolist())
    common = np.array(sorted(common))
    if not len(common):
        raise SystemExit("у прогонов нет общих тестовых наборов")
    perms = [np.array([{s: i for i, s in enumerate(d["test_sets"])}[c]
                       for c in common]) for d in ds]
    y_true = ds[0]["y_true"][perms[0]]
    for d, pm in zip(ds[1:], perms[1:]):
        if not np.allclose(d["y_true"][pm], y_true):
            raise SystemExit("истина не совпадает между прогонами")
    preds = [d["y_pred"][pm] for d, pm in zip(ds, perms)]
    ref = ds[0]
    geo = {k: ref[k][perms[0]] for k in ("proc", "r", "z")}
    return common, y_true, preds, geo


def fig_compare(npzs: List[str], labels: List[str], out: str,
                which: int = -1, where: Optional[Dict[str, float]] = None,
                errors: bool = False):
    """
    Сравнение СЕМЕЙСТВ на одном наборе.

    errors=False — поля: верхняя строка МКЭ, дальше по строке на семейство.
    errors=True  — невязки каждого семейства, общая шкала в % от масштаба
                   компоненты, поэтому сравнимы и семейства, и компоненты.

    Набор выбирается по МЕДИАННОЙ ошибке ПОСЛЕДНЕГО прогона в списке (обычно
    это полная модель), чтобы выбор не подыгрывал ни одному из сравниваемых.
    """
    _, yt_all, preds, geo = _align(npzs)
    err = np.sqrt(((preds[-1] - yt_all) ** 2).mean(axis=(1, 2, 3)))
    which = _pick_set(geo, err, which, where)
    yt = yt_all[which]
    R = geo["r"][which]
    Z = np.repeat(geo["z"][which][:, None], yt.shape[1], axis=1)

    scales = [_sym(np.concatenate([yt[:, :, c].ravel()]
                                  + [p[which][:, :, c].ravel() for p in preds]))
              for _, c in ROWS4]
    if errors:
        rows = [(lbl, p) for lbl, p in zip(labels, preds)]
        rel_lim = max(100.0 * _sym((p[which][:, :, c] - yt[:, :, c]).ravel()) / sc
                      for _, p in rows for (_, c), sc in zip(ROWS4, scales))
    else:
        rows = [("МКЭ", None)] + list(zip(labels, preds))

    nr_ = len(rows)
    fig, axes = plt.subplots(nr_, 4, figsize=(9.6, 1.75 * nr_ + 1.6),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_2d(axes)
    cf_last = None
    for i, (lbl, pred) in enumerate(rows):
        for j, (tex, c) in enumerate(ROWS4):
            field = yt[:, :, c] if pred is None else pred[which][:, :, c]
            if errors:
                v, L = 100.0 * (field - yt[:, :, c]) / scales[j], rel_lim
            else:
                v, L = field, scales[j]
            ax = axes[i, j]
            lev = np.linspace(-L, L, 21)
            cf = ax.contourf(R, Z, v, levels=lev, cmap=DIVERGING, extend="both")
            ax.contour(R, Z, v, levels=lev[::5], colors="k",
                       linewidths=0.25, alpha=0.35)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
            if i == 0:
                ax.set_title(tex, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{lbl}\n" + r"$z/L$", fontsize=8)
            if i == nr_ - 1:
                ax.set_xlabel(r"$r/R$")
            cf_last = cf
        if not errors:
            pass
    if errors:
        cb = fig.colorbar(cf_last, ax=axes, fraction=0.04, pad=0.02,
                          location="bottom", orientation="horizontal",
                          ticks=np.linspace(-rel_lim, rel_lim, 7))
        cb.set_label("невязка, % от масштаба своей компоненты", fontsize=7)
        cb.ax.set_xticklabels([f"{t:.0f}" for t in
                               np.linspace(-rel_lim, rel_lim, 7)])
        cb.ax.tick_params(labelsize=6.5, color=MUTED)
    else:
        for j, (tex, c) in enumerate(ROWS4):
            sm = plt.cm.ScalarMappable(cmap=DIVERGING,
                                       norm=plt.Normalize(-scales[j], scales[j]))
            cb = fig.colorbar(sm, ax=axes[:, j], fraction=0.05, pad=0.02,
                              location="bottom", orientation="horizontal",
                              ticks=np.linspace(-scales[j], scales[j], 5))
            cb.set_label("МПа", fontsize=7)
            cb.ax.tick_params(labelsize=6, color=MUTED)

    # сводка по всему общему тесту
    summ = []
    for lbl, p in zip(labels, preds):
        parts = []
        for tex, c in ROWS4:
            t, q = yt_all[..., c].ravel(), p[..., c].ravel()
            r2 = 1.0 - ((q - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            parts.append(f"{tex} {r2:.3f}")
        summ.append(f"{lbl} — $R^2$: " + ", ".join(parts))
    fig.suptitle(("Невязки семейств" if errors else "Поля: МКЭ и семейства") +
                 f".  Набор с медианной ошибкой:  {_regime(geo['proc'][which])}",
                 fontsize=8.5)
    fig.text(0.5, -0.015,
             (f"Общий тест: {len(yt_all)} наборов.  " + ";  ".join(summ) + ".\n"
              "Семейства отличаются ТОЛЬКО составом лосса: архитектура, сплит, "
              "сид, число эпох и оптимизатор одинаковы.\n"
              + ("Шкала общая на все панели: сравнимы и семейства, и компоненты."
                 if errors else
                 "Шкала общая внутри компоненты для всех строк.")),
             ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_model(npz: str, out: str, which: int = -1,
              where: Optional[Dict[str, float]] = None,
              diff_mode: str = "rel"):
    """
    МКЭ против предсказания модели с z во входе, и разность.

    Координаты БЕЗРАЗМЕРНЫЕ: r/R — радиус, отнесённый к радиусу поверхности в
    том же осевом сечении (1 = свободная поверхность при любом z), z/L — доля
    осевого окна (0 — вход, 1 — выход). Так карты сопоставимы между наборами
    с разным обжатием, у которых радиус поверхности разный.

    which = -1 (по умолчанию) — набор с МЕДИАННОЙ ошибкой, а не первый
    попавшийся и не лучший: показывать надо типичное поведение.

    where — ограничить выбор режимом, например {"v": 20}. Медиана берётся
    внутри отобранного подмножества, так что набор остаётся типичным ДЛЯ
    ЭТОГО режима, а не для теста целиком.

    diff_mode — как показывать третий столбец:
      'rel' (по умолчанию) — (модель − МКЭ) / масштаб компоненты, в процентах.
            Шкала ОБЩАЯ для всех трёх строк, поэтому строки сравнимы между
            собой: сразу видно, по какой компоненте модель хуже.
      'abs' — в МПа, шкала своя у каждой строки. Структура ошибки видна
            лучше, но строки несравнимы, а панель всегда закрашена на полную
            насыщенность независимо от того, 3 % это или 30 %.

    Один набор — это одна точка выборки, поэтому в подписи печатаются метрики
    ПО ВСЕМУ тесту: по одной панели о качестве модели судить нельзя, особенно
    для τ_rz, величина которого сильно меняется от режима к режиму.
    """
    if diff_mode not in ("rel", "abs"):
        raise ValueError(f"diff_mode: 'rel' или 'abs', получено {diff_mode!r}")
    d = np.load(npz, allow_pickle=True)
    y_true_all, y_pred_all = d["y_true"], d["y_pred"]
    err = np.sqrt(((y_pred_all - y_true_all) ** 2).mean(axis=(1, 2, 3)))
    if which < 0:
        pool = np.ones(len(err), dtype=bool)
        if where:
            for key, val in where.items():
                pool &= np.isclose(d["proc"][:, PROC_COL[key]], val)
            if not pool.any():
                raise SystemExit(f"нет тестовых наборов с {where}")
        idx = np.flatnonzero(pool)
        which = int(idx[np.argsort(err[idx])[len(idx) // 2]])
        tag = (" при " + ", ".join(f"{k} = {v:g}" for k, v in where.items())
               ) if where else ""
        print(f"выбран набор {which}{tag}: медианная ошибка {err[which]:.2f} МПа "
              f"(из {len(idx)} наборов, диапазон {err[idx].min():.2f}–"
              f"{err[idx].max():.2f})")

    rows = [(r"$\sigma_{rr}$", 0), (r"$\sigma_{zz}$", 2), (r"$\tau_{rz}$", 3)]
    clean = d["clean_mask"] if "clean_mask" in d.files else None
    summ = []
    for tex, c in rows:
        t, q = y_true_all[..., c].ravel(), y_pred_all[..., c].ravel()
        r2 = 1.0 - ((q - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        part = f"{tex}: RMSE {np.sqrt(((q - t) ** 2).mean()):.1f}, $R^2$ {r2:.2f}"
        if clean is not None and not clean.all():
            tc = y_true_all[clean][..., c].ravel()
            qc = y_pred_all[clean][..., c].ravel()
            r2c = 1.0 - ((qc - tc) ** 2).sum() / ((tc - tc.mean()) ** 2).sum()
            part += (f" (без конуса {np.sqrt(((qc - tc) ** 2).mean()):.1f} / "
                     f"{r2c:.2f})")
        summ.append(part)

    yt, yp = y_true_all[which], y_pred_all[which]
    # безразмерные координаты: r/R (1 = поверхность), z/L (0..1 по окну)
    R = d["r"][which]
    Z = np.repeat(d["z"][which][:, None], yt.shape[1], axis=1)
    proc = d["proc"][which]

    # масштаб каждой компоненты и общий предел относительной шкалы
    scales = [_sym(np.concatenate([yt[:, :, c].ravel(), yp[:, :, c].ravel()]))
              for _, c in rows]
    rel_lim = max(100.0 * _sym((yp[:, :, c] - yt[:, :, c]).ravel()) / sc
                  for (_, c), sc in zip(rows, scales))

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.6, 7.2),
                             layout="constrained")
    cf_rel = None
    for i, (tex, c) in enumerate(rows):
        t, q = yt[:, :, c], yp[:, :, c]
        lim = scales[i]
        if diff_mode == "rel":
            dv, dlim = 100.0 * (q - t) / lim, rel_lim
            dttl = "невязка, % от масштаба"
        else:
            dv, dlim = q - t, _sym((q - t).ravel())
            dttl = f"невязка, МПа  ({100 * dlim / lim:.0f} % от масштаба)"
        cf_pair = None
        for j, (v, ttl, L) in enumerate(((t, "МКЭ", lim),
                                         (q, "модель", lim),
                                         (dv, dttl, dlim))):
            ax = axes[i, j]
            lev = np.linspace(-L, L, 21)
            cf = ax.contourf(R, Z, v, levels=lev, cmap=DIVERGING, extend="both")
            ax.contour(R, Z, v, levels=lev[::5], colors="k",
                       linewidths=0.25, alpha=0.35)
            ax.set_title(f"{tex}  {ttl}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
            if j < 2:
                cf_pair = cf
            else:
                cf_rel = cf
            if j > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r"$z/L$")
            if i < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$r/R$")
        # одна шкала на пару МКЭ/модель — они в одних пределах
        cb = fig.colorbar(cf_pair, ax=axes[i, :2], fraction=0.045, pad=0.02)
        cb.set_label("МПа", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, color=MUTED)
        if diff_mode == "abs":
            cbd = fig.colorbar(cf, ax=axes[i, 2], fraction=0.09, pad=0.02)
            cbd.set_label("МПа", fontsize=7)
            cbd.ax.tick_params(labelsize=6.5, color=MUTED)
    if diff_mode == "rel":
        # ОДНА шкала на весь столбец невязок: строки становятся сравнимы
        cbd = fig.colorbar(cf_rel, ax=axes[:, 2], fraction=0.055, pad=0.02)
        cbd.set_label("% от масштаба компоненты", fontsize=7)
        cbd.ax.tick_params(labelsize=6.5, color=MUTED)

    fig.suptitle(f"Набор с медианной ошибкой:  Q = {proc[0]:g}, k = {proc[1]:g}, "
                 f"α = {proc[2]:g}°, μ = {proc[3]:g}, v = {proc[4]:g} м/мин",
                 fontsize=8.5)
    note = (r"$r/R$ — радиус, отнесённый к поверхности в том же сечении "
            r"($r/R = 1$ — свободная поверхность); $z/L$ — доля осевого окна."
            "\n")
    if diff_mode == "rel":
        note += ("Невязка отнесена к масштабу своей компоненты (99.5-й "
                 "перцентиль |поля|), шкала общая на столбец — строки сравнимы "
                 "между собой.\n")
    else:
        note += ("Шкала МКЭ и модели общая; у невязки своя, её доля от "
                 "масштаба указана в заголовке панели.\n")
    note += f"По всему тесту (n = {len(y_true_all)}):  " + ";  ".join(summ) + "."
    if clean is not None and not clean.all():
        note += (f"\n«Без конуса» — {int(clean.sum())} наборов после снятия "
                 f"{int((~clean).sum())} с конусом волоки в окне (E-24).")
    fig.text(0.5, -0.015, note, ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_alpha_series(root: str, out: str, comp: int = 12, tex: str = r"$\sigma_{zz}$"):
    """Одна компонента при разных полууглах волоки — что меняет параметр."""
    jobs = list_jobs(root)
    want = []
    for alpha in (4.0, 12.0, 20.0):
        for nm, p in jobs:
            pj = parse_job_name(nm)
            if (pj.get("alpha") == alpha and abs(pj.get("Q", 0) - 0.10) < 1e-9
                    and abs(pj.get("k", -1) - 0.5) < 1e-9 and pj.get("v") == 20):
                want.append((alpha, nm, p)); break
    if len(want) < 2:
        want = []
        for alpha in (4.0, 12.0, 20.0):
            for nm, p in jobs:
                if parse_job_name(nm).get("alpha") == alpha:
                    want.append((alpha, nm, p)); break
    fig, axes = plt.subplots(1, len(want), figsize=(2.8 * len(want), 4.4),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    arrs = [read_rpt(p) for _, _, p in want]
    lim = max(_sym(a[:, comp] / 1e6) for a in arrs)
    others = []
    for ax, (alpha, nm, pth), a in zip(axes, want, arrs):
        R0 = billet_radius(pth)
        L = float(a[:, C_Z].max() - a[:, C_Z].min())
        tri = mtri.Triangulation(a[:, C_R] / R0,
                                 (a[:, C_Z] - a[:, C_Z].min()) / L)
        pj = parse_job_name(nm)
        cf = _panel(ax, tri, a[:, comp] / 1e6, f"α = {alpha:g}°", lim)
        ax.set_xlabel(r"$r/R_0$")
        others.append((pj.get("Q"), pj.get("k"), pj.get("mu_repo"), pj.get("v")))
    axes[0].set_ylabel(r"$z/L$")
    same = len(set(others)) == 1
    o = others[0]
    fig.text(0.5, -0.06,
             (f"Остальные параметры одинаковы: Q = {o[0]:g}, k = {o[1]:g}, "
              f"μ = {o[2]:g}, v = {o[3]:g} м/мин — различие только в полуугле."
              if same else
              "⚠ Расчёты с полностью совпадающими прочими параметрами не нашлись; "
              "панели отличаются не только полууглом: "
              + "; ".join(f"α={w[0]:g}°: Q={x[0]:g}, k={x[1]:g}, μ={x[2]:g}, v={x[3]:g}"
                          for w, x in zip(want, others))),
             ha="center", va="top", fontsize=7, color=INK)
    cb = fig.colorbar(cf, ax=axes.tolist(), fraction=0.045, pad=0.02)
    cb.set_label("МПа", fontsize=7); cb.ax.tick_params(labelsize=6.5, color=MUTED)
    fig.suptitle(f"{tex} при разном полуугле волоки", y=0.99, fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["raw", "model", "fields", "errors", "compare",
                             "compare-errors", "alpha"],
                    required=True)
    ap.add_argument("--npzs", default="",
                    help="для compare: список npz через запятую")
    ap.add_argument("--labels", default="",
                    help="для compare: подписи строк через запятую")
    ap.add_argument("--root", default="")
    ap.add_argument("--rpt", default="")
    ap.add_argument("--npz", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", type=int, default=-1,
                    help="индекс тестового набора; -1 = набор с медианной ошибкой")
    ap.add_argument("--diff-mode", dest="diff_mode", default="rel",
                    choices=["rel", "abs"],
                    help="третий столбец: 'rel' — %% от масштаба, общая шкала "
                         "на столбец; 'abs' — МПа, своя шкала у каждой строки")
    ap.add_argument("--where", default="",
                    help="ограничить режимом, например 'v=20' или 'v=40,alpha=8'")
    a = ap.parse_args()
    if a.mode == "raw":
        job = os.path.basename(a.rpt)[:-4]
        print("создан", fig_raw(a.rpt, a.out, job))
    elif a.mode == "model":
        where = _where(a.where)
        print("создан", fig_model(a.npz, a.out, a.which, where or None,
                                  a.diff_mode))
    elif a.mode in ("compare", "compare-errors"):
        npzs = [x.strip() for x in a.npzs.split(",") if x.strip()]
        labels = [x.strip() for x in a.labels.split(",") if x.strip()]
        if len(labels) != len(npzs):
            raise SystemExit(f"--labels: {len(labels)} против {len(npzs)} npz")
        print("создан", fig_compare(npzs, labels, a.out, a.which,
                                    _where(a.where),
                                    errors=a.mode == "compare-errors"))
    elif a.mode == "fields":
        print("создан", fig_fields(a.npz, a.out, a.which, _where(a.where)))
    elif a.mode == "errors":
        print("создан", fig_errors(a.npz, a.out, a.which, _where(a.where),
                                   a.diff_mode))
    else:
        print("создан", fig_alpha_series(a.root, a.out))


if __name__ == "__main__":
    main()
