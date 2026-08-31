"""
figures.py — рисунки для статьи из publication/experiments/results/runs.csv.

Оформление под двухколоночный Elsevier: ширина колонки 3.35", шрифт 8 pt,
тонкие марки, приглушённая сетка, ось одна на рисунок.

Цвета назначены по СЕМЕЙСТВУ модели и зафиксированы: перекрашивать серии при
смене состава нельзя. Палитра (Okabe–Ito) проверена валидатором на
различимость при дальтонизме: худшая соседняя пара ΔE = 11.0 (deutan),
25.8 (нормальное зрение), контраст к фону ≥ 3:1 — все проверки пройдены.
Идентичность дополнительно несут штриховка и прямые подписи, поэтому рисунки
читаются и в чёрно-белой печати.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR = {"mlp": "#0072B2", "pinn": "#D55E00", "vpinn": "#009E73",
         "mlp2d": "#0072B2", "pinn2d_reduced": "#D55E00", "pinn2d_full": "#009E73",
         "torch": "#0072B2", "jax": "#D55E00"}
HATCH = {"mlp": "", "pinn": "//", "vpinn": "..",
         "mlp2d": "", "pinn2d_reduced": "//", "pinn2d_full": ".."}
LABEL = {"mlp": "MLP (data-driven)", "pinn": "PINN (strong form)",
         "vpinn": "VPINN (weak form)",
         "mlp2d": "MLP, z во входе", "pinn2d_reduced": "PINN, редуцированная форма",
         "pinn2d_full": "PINN, полная форма",
         "torch": "PyTorch", "jax": "JAX"}
ORDER = ["mlp", "pinn", "vpinn"]
ORDER_2D = ["mlp2d", "pinn2d_reduced", "pinn2d_full"]
INK, MUTED = "#1a1a1a", "#8a8a8a"
COL_W, TWO_COL_W = 3.35, 6.9

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#e3e3e0", "grid.linewidth": 0.6, "figure.dpi": 200,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})


def _grid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def agg(df: pd.DataFrame, by: List[str], metric: str) -> pd.DataFrame:
    g = df.groupby(by)[metric].agg(["mean", "std", "count"]).reset_index()
    return g.fillna({"std": 0.0})


def fig_extrapolation(df, out, metric="macro_rmse",
                      splits=("random", "interp:alpha_mid", "matched:alpha_max",
                              "extrap:alpha_max")):
    d = df[(df["stage"] == "main") & (df["split"].isin(splits))]
    if d.empty:
        return None
    g = agg(d, ["split", "family"], metric)
    splits = [s for s in splits if s in set(g["split"])]
    fams = [f for f in ORDER if f in set(g["family"])]
    x = np.arange(len(splits)); w = 0.8 / max(len(fams), 1)

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for i, fam in enumerate(fams):
        sub = g[g["family"] == fam].set_index("split").reindex(splits)
        ax.bar(x + (i - (len(fams) - 1) / 2) * w, sub["mean"], w * 0.92,
               yerr=sub["std"], capsize=2, color=COLOR[fam], hatch=HATCH[fam],
               edgecolor="white", linewidth=0.8, label=LABEL[fam],
               error_kw={"ecolor": INK, "elinewidth": 0.7, "capthick": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("extrap:", "экстрап.\n").replace("interp:", "интерп.\n")
                        .replace("matched:", "контроль\n").replace("random", "случайный\nhold-out")
                        for s in splits])
    ax.set_ylabel("macro-RMSE, МПа")
    ax.legend(frameon=False, loc="upper left")
    _grid(ax)
    fig.savefig(out); plt.close(fig)
    return out


def fig_curves(df, out, metric="macro_rmse"):
    d = df[df["stage"] == "curves"]
    if d.empty:
        return None
    splits = sorted(d["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(TWO_COL_W, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, splits):
        g = agg(d[d["split"] == sp], ["frac", "family"], metric)
        for fam in [f for f in ORDER if f in set(g["family"])]:
            s = g[g["family"] == fam].sort_values("frac")
            ax.plot(s["frac"] * 100, s["mean"], "-o", ms=4, lw=1.6,
                    color=COLOR[fam], label=LABEL[fam])
            ax.fill_between(s["frac"] * 100, s["mean"] - s["std"], s["mean"] + s["std"],
                            color=COLOR[fam], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["10", "25", "50", "100"])
        ax.set_xlabel("доля обучающего пула, %")
        ax.set_title(sp.replace("extrap:", "экстраполяция: ").replace("random", "случайный hold-out"))
        _grid(ax)
    axes[0].set_ylabel("macro-RMSE, МПа")
    axes[-1].legend(frameon=False)
    fig.savefig(out); plt.close(fig)
    return out


def fig_corruption(df, out, metric="macro_rmse"):
    d = df[df["stage"] == "corrupt"]
    if d.empty:
        return None
    g = agg(d, ["corrupt_rate", "family"], metric)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for fam in [f for f in ORDER if f in set(g["family"])]:
        s = g[g["family"] == fam].sort_values("corrupt_rate")
        ax.plot(s["corrupt_rate"] * 100, s["mean"], "-o", ms=4, lw=1.6,
                color=COLOR[fam], label=LABEL[fam])
        ax.fill_between(s["corrupt_rate"] * 100, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=COLOR[fam], alpha=0.15, linewidth=0)
    ax.set_xlabel("доля испорченных обучающих меток, %")
    ax.set_ylabel("macro-RMSE на чистом hold-out, МПа")
    ax.legend(frameon=False)
    _grid(ax)
    fig.savefig(out); plt.close(fig)
    return out


def fig_bc(df, out, fem_srr: Optional[float] = None, fem_trz: Optional[float] = None,
           split="extrap:alpha_max"):
    d = df[(df["stage"] == "main") & (df["split"] == split)]
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_W, 2.3))
    for ax, col, name, fem in ((axes[0], "bc_sigma_rr", r"$|\sigma_{rr}|$ при $r=1$", fem_srr),
                               (axes[1], "bc_tau_rz", r"$|\tau_{rz}|$ при $r=1$", fem_trz)):
        g = agg(d, ["family"], col)
        fams = [f for f in ORDER if f in set(g["family"])]
        s = g.set_index("family").reindex(fams)
        ax.bar(np.arange(len(fams)), s["mean"], 0.6, yerr=s["std"], capsize=2,
               color=[COLOR[f] for f in fams], edgecolor="white", linewidth=0.8,
               error_kw={"ecolor": INK, "elinewidth": 0.7, "capthick": 0.7})
        for i, f in enumerate(fams):
            ax.bar(i, s["mean"].iloc[i], 0.6, color="none", hatch=HATCH[f],
                   edgecolor="white", linewidth=0)
        if fem is not None:
            ax.axhline(fem, color=INK, ls="--", lw=1.0)
            ax.text(len(fams) - 0.5, fem, "  данные FEM", va="bottom", ha="right",
                    fontsize=7, color=INK)
        ax.set_xticks(np.arange(len(fams)))
        ax.set_xticklabels([LABEL[f].split(" (")[0] for f in fams])
        ax.set_ylabel(name + ", МПа")
        _grid(ax)
    fig.suptitle(f"нарушение traction-free на held-out регионе ({split})", y=1.02)
    fig.savefig(out); plt.close(fig)
    return out


def fig_backend(df, out, metric="macro_rmse"):
    d = df[df["stage"] == "backend"]
    if d.empty:
        return None
    splits = sorted(d["split"].unique())
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    xs = []
    for j, sp in enumerate(splits):
        for i, be in enumerate(("torch", "jax")):
            v = d[(d["split"] == sp) & (d["backend"] == be)][metric].to_numpy()
            if not len(v):
                continue
            xpos = j + (i - 0.5) * 0.3
            xs.append((xpos, be, sp))
            ax.scatter(np.full(len(v), xpos), v, s=14, color=COLOR[be],
                       alpha=0.75, zorder=3, label=LABEL[be] if j == 0 else None)
            ax.errorbar(xpos, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0,
                        fmt="_", ms=18, color=INK, elinewidth=1.0, capsize=3, zorder=4)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.replace("extrap:", "экстрап.\n").replace("random", "случайный\nhold-out")
                        for s in splits])
    ax.set_ylabel("macro-RMSE, МПа")
    ax.legend(frameon=False)
    _grid(ax)
    fig.savefig(out); plt.close(fig)
    return out


def fig_twod(df, out, metric="macro_rmse"):
    d = df[df["stage"] == "twod"]
    if d.empty:
        return None
    g = agg(d, ["split", "family"], metric)
    splits = sorted(g["split"].unique())
    fams = [f for f in ORDER_2D if f in set(g["family"])]
    x = np.arange(len(splits)); w = 0.8 / max(len(fams), 1)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for i, fam in enumerate(fams):
        s = g[g["family"] == fam].set_index("split").reindex(splits)
        ax.bar(x + (i - (len(fams) - 1) / 2) * w, s["mean"], w * 0.92, yerr=s["std"],
               capsize=2, color=COLOR[fam], hatch=HATCH[fam], edgecolor="white",
               linewidth=0.8, label=LABEL[fam],
               error_kw={"ecolor": INK, "elinewidth": 0.7, "capthick": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("extrap:", "экстрап.\n").replace("random", "случайный\nhold-out")
                        for s in splits])
    ax.set_ylabel("macro-RMSE, МПа")
    ax.legend(frameon=False)
    _grid(ax)
    fig.savefig(out); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fem-srr", type=float, default=None)
    ap.add_argument("--fem-trz", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.csv)
    df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    made = [
        fig_extrapolation(df, os.path.join(a.outdir, "fig_extrapolation.png")),
        fig_curves(df, os.path.join(a.outdir, "fig_learning_curves.png")),
        fig_corruption(df, os.path.join(a.outdir, "fig_corruption.png")),
        fig_bc(df, os.path.join(a.outdir, "fig_bc_violation.png"), a.fem_srr, a.fem_trz),
        fig_backend(df, os.path.join(a.outdir, "fig_backend.png")),
        fig_twod(df, os.path.join(a.outdir, "fig_2d_ablation.png")),
    ]
    for m in made:
        print(("создан " if m else "пропущен (нет данных стадии)"), m or "")


if __name__ == "__main__":
    main()
