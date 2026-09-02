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
                на сетке (z, r) 8 × 20.

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from raw_probe import C_R, C_Z, axial_window, list_jobs, parse_job_name, read_rpt  # noqa: E402

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


def fig_model(npz: str, out: str, which: int = 0):
    """МКЭ против предсказания модели с z во входе, и разность."""
    d = np.load(npz, allow_pickle=True)
    yt, yp = d["y_true"][which], d["y_pred"][which]
    rp, zp = d["r_phys"][which] * 1e3, d["z_phys"][which] * 1e3
    proc = d["proc"][which]
    nz, nr, _ = yt.shape
    R = np.repeat(rp[:, :, None], 1, axis=2)[:, :, 0]
    Z = np.repeat(zp[:, None], nr, axis=1)

    rows = [("sigma_rr", r"$\sigma_{rr}$", 0), ("sigma_zz", r"$\sigma_{zz}$", 2)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.4, 4.6))
    for i, (key, tex, c) in enumerate(rows):
        t, p = yt[:, :, c], yp[:, :, c]
        lim = _sym(np.concatenate([t.ravel(), p.ravel()]))
        dlim = _sym((p - t).ravel())
        for j, (v, ttl, L) in enumerate(((t, f"{tex}  МКЭ", lim),
                                          (p, f"{tex}  модель", lim),
                                          (p - t, f"{tex}  модель − МКЭ", dlim))):
            ax = axes[i, j]
            cf = ax.contourf(R, Z, v, levels=np.linspace(-L, L, 21),
                             cmap=DIVERGING, extend="both")
            ax.contour(R, Z, v, levels=np.linspace(-L, L, 21)[::4],
                       colors="k", linewidths=0.25, alpha=0.35)
            ax.set_title(ttl + ", МПа", fontsize=8)
            if j == 0:
                ax.set_ylabel("z, мм")
            if i == len(rows) - 1:
                ax.set_xlabel("r, мм")
            cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.04)
            cb.ax.tick_params(labelsize=6, color=MUTED)
    fig.suptitle(f"Q = {proc[0]:g}, k = {proc[1]:g}, α = {proc[2]:g}°, "
                 f"μ = {proc[3]:g}, v = {proc[4]:g} м/мин", y=1.01, fontsize=8.5)
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
    ap.add_argument("--which", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "raw":
        job = os.path.basename(a.rpt)[:-4]
        print("создан", fig_raw(a.rpt, a.out, job))
    elif a.mode == "model":
        print("создан", fig_model(a.npz, a.out, a.which))
    else:
        print("создан", fig_alpha_series(a.root, a.out))


if __name__ == "__main__":
    main()
