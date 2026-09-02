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

from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from raw_probe import C_R, C_Z, axial_window, list_jobs, parse_job_name, read_rpt  # noqa: E402

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
    r = a[:, C_R] * 1e3          # мм
    z = a[:, C_Z] * 1e3
    lo, hi = axial_window(a[:, C_Z])
    tri = mtri.Triangulation(r, z)

    fig, axes = plt.subplots(1, 4, figsize=(8.6, 3.6), sharey=True)
    for ax, (key, tex, col) in zip(axes, COMPS):
        v = a[:, col] / 1e6
        lim = _sym(v)
        cf = _panel(ax, tri, v, f"{tex}, МПа", lim)
        ax.axhline(lo * 1e3, color=INK, lw=1.0, ls="--")
        ax.axhline(hi * 1e3, color=INK, lw=1.0, ls="--")
        ax.set_xlabel("r, мм")
        cb = fig.colorbar(cf, ax=ax, fraction=0.055, pad=0.04)
        cb.ax.tick_params(labelsize=6.5, color=MUTED)
    axes[0].set_ylabel("z, мм")
    fig.suptitle(f"Остаточные напряжения по узлам МКЭ · {job}", y=1.05, fontsize=9)
    fig.text(0.5, -0.10,
             "Пунктир — осевое окно 25–75 %: из этой полосы собирается обучающая "
             "выборка (20 радиальных точек на набор).\n"
             "Оси не в одном масштабе: радиальная растянута, иначе панель "
             "вырождается в полоску (радиус ~16 мм при длине ~130 мм).",
             ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_model(npz: str, out: str, which: int = -1,
              where: Optional[Dict[str, float]] = None):
    """
    МКЭ против предсказания модели с z во входе, и разность.

    which = -1 (по умолчанию) — набор с МЕДИАННОЙ ошибкой, а не первый
    попавшийся и не лучший: показывать надо типичное поведение.

    where — ограничить выбор режимом, например {"v": 20}. Медиана берётся
    внутри отобранного подмножества, так что набор остаётся типичным ДЛЯ
    ЭТОГО режима, а не для теста целиком.

    Один набор — это одна точка выборки, поэтому в подписи печатаются метрики
    ПО ВСЕМУ тесту: по одной панели о качестве модели судить нельзя, особенно
    для τ_rz, величина которого сильно меняется от режима к режиму.
    """
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
        if clean is not None:
            tc = y_true_all[clean][..., c].ravel()
            qc = y_pred_all[clean][..., c].ravel()
            r2c = 1.0 - ((qc - tc) ** 2).sum() / ((tc - tc.mean()) ** 2).sum()
            part += (f" (без конуса {np.sqrt(((qc - tc) ** 2).mean()):.1f} / "
                     f"{r2c:.2f})")
        summ.append(part)

    yt, yp = y_true_all[which], y_pred_all[which]
    R = d["r_phys"][which] * 1e3
    Z = np.repeat(d["z_phys"][which][:, None] * 1e3, yt.shape[1], axis=1)
    proc = d["proc"][which]

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.6, 7.2),
                             layout="constrained")
    for i, (tex, c) in enumerate(rows):
        t, q = yt[:, :, c], yp[:, :, c]
        lim = _sym(np.concatenate([t.ravel(), q.ravel()]))
        dlim = _sym((q - t).ravel())
        cf_pair = None
        # панель разности масштабируется по СЕБЕ, иначе структура не видна;
        # поэтому её долю от поля пишем прямо в заголовок — иначе ошибка в
        # 11 % закрашивается на полную насыщенность и читается как провал
        diff_ttl = f"модель − МКЭ  ({100 * dlim / lim:.0f} % от поля)"
        for j, (v, ttl, L) in enumerate(((t, "МКЭ", lim),
                                         (q, "модель", lim),
                                         (q - t, diff_ttl, dlim))):
            ax = axes[i, j]
            lev = np.linspace(-L, L, 21)
            cf = ax.contourf(R, Z, v, levels=lev, cmap=DIVERGING, extend="both")
            ax.contour(R, Z, v, levels=lev[::5], colors="k",
                       linewidths=0.25, alpha=0.35)
            ax.set_title(f"{tex}  {ttl}")
            if j < 2:
                cf_pair = cf
            if j > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("z, мм")
            if i < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("r, мм")
        # одна шкала на пару МКЭ/модель — они в одних пределах, дублировать незачем
        cb = fig.colorbar(cf_pair, ax=axes[i, :2], fraction=0.045, pad=0.02)
        cb.set_label("МПа", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, color=MUTED)
        cbd = fig.colorbar(cf, ax=axes[i, 2], fraction=0.09, pad=0.02)
        cbd.set_label("МПа", fontsize=7)
        cbd.ax.tick_params(labelsize=6.5, color=MUTED)

    fig.suptitle(f"Набор с медианной ошибкой:  Q = {proc[0]:g}, k = {proc[1]:g}, "
                 f"α = {proc[2]:g}°, μ = {proc[3]:g}, v = {proc[4]:g} м/мин",
                 fontsize=8.5)
    note = ("Шкала МКЭ и модели общая; у разности СВОЯ, её доля от поля "
            "указана в заголовке панели.\n"
            f"По всему тесту (n = {len(y_true_all)}):  " + ";  ".join(summ) + ".")
    if clean is not None:
        note += (f"\n«Без конуса» — {int(clean.sum())} наборов после снятия "
                 f"{int((~clean).sum())} с конусом волоки в окне (E-24): они "
                 "несут 51 % всей квадратичной ошибки.")
    note += "\nОси не в одном масштабе: радиальная растянута."
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
    fig, axes = plt.subplots(1, len(want), figsize=(2.6 * len(want), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    arrs = [read_rpt(p) for _, _, p in want]
    lim = max(_sym(a[:, comp] / 1e6) for a in arrs)
    others = []
    for ax, (alpha, nm, _), a in zip(axes, want, arrs):
        tri = mtri.Triangulation(a[:, C_R] * 1e3, a[:, C_Z] * 1e3)
        pj = parse_job_name(nm)
        cf = _panel(ax, tri, a[:, comp] / 1e6, f"α = {alpha:g}°", lim)
        ax.set_xlabel("r, мм")
        others.append((pj.get("Q"), pj.get("k"), pj.get("mu_repo"), pj.get("v")))
    axes[0].set_ylabel("z, мм")
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
    ap.add_argument("--mode", choices=["raw", "model", "alpha"], required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--rpt", default="")
    ap.add_argument("--npz", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", type=int, default=-1,
                    help="индекс тестового набора; -1 = набор с медианной ошибкой")
    ap.add_argument("--where", default="",
                    help="ограничить режимом, например 'v=20' или 'v=40,alpha=8'")
    a = ap.parse_args()
    if a.mode == "raw":
        job = os.path.basename(a.rpt)[:-4]
        print("создан", fig_raw(a.rpt, a.out, job))
    elif a.mode == "model":
        where = {}
        for part in filter(None, a.where.split(",")):
            k, v = part.split("=")
            where[k.strip()] = float(v)
        print("создан", fig_model(a.npz, a.out, a.which, where or None))
    else:
        print("создан", fig_alpha_series(a.root, a.out))


if __name__ == "__main__":
    main()
