"""Render distribution PNGs for ВКР appendix.

For each of 4 stress components (σ_rr, σ_θθ, σ_zz, τ_rz) and 4 strain
components (ε_rr, ε_θθ, ε_zz, ε_rz), produce three PNGs:

  hist.png     — histogram of all (N_sets × 20) values
  profile.png  — radial profile: median ± IQR, with 5/95-percentile band
  hexbin.png   — 2D density (r_norm, value), log-scaled colorbar

Saves into figures/distributions/<comp>/.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "figures" / "distributions"

PLANE_ORDER = [2, 0, 1, 3]
N_R = 20

STRESS = [
    ("sigma_rr", r"$\sigma_{rr}$",       "МПа"),
    ("sigma_tt", r"$\sigma_{\theta\theta}$", "МПа"),
    ("sigma_zz", r"$\sigma_{zz}$",       "МПа"),
    ("tau_rz",   r"$\tau_{rz}$",         "МПа"),
]
STRAIN = [
    ("eps_rr", r"$\varepsilon_{rr}$",       "—"),
    ("eps_tt", r"$\varepsilon_{\theta\theta}$", "—"),
    ("eps_zz", r"$\varepsilon_{zz}$",       "—"),
    ("eps_rz", r"$\varepsilon_{rz}$",       "—"),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load_components(pkl: Path) -> np.ndarray:
    """Return array (N_sets, N_R, 4) in physical PLANE_ORDER."""
    with open(pkl, "rb") as fh:
        y = np.asarray(pickle.load(fh))
    # y: (4 planes, N_sets, N_R) -> reorder planes to physical
    return np.stack([y[PLANE_ORDER[i]] for i in range(4)], axis=-1)


def plot_hist(values: np.ndarray, *, label: str, unit: str, out: Path):
    vals = values.reshape(-1)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(vals, bins=80, color="#3a7ca5", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"{label}  [{unit}]")
    ax.set_ylabel("Число точек")
    ax.set_title(f"Распределение {label} по полному датасету "
                 f"(N = {vals.size:,})")
    q05, q50, q95 = np.percentile(vals, [5, 50, 95])
    for q, c, lbl in [(q05, "#aa3939", "P05"),
                       (q50, "#222222", "медиана"),
                       (q95, "#aa3939", "P95")]:
        ax.axvline(q, color=c, lw=1.0, ls="--", alpha=0.7)
    ax.text(0.98, 0.97,
            f"медиана = {q50:.3g}\nIQR-полоса = [{q05:.3g}, {q95:.3g}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_profile(values: np.ndarray, *, label: str, unit: str, out: Path):
    # values: (N_sets, N_R)
    r = np.linspace(0.0, 1.0, N_R)
    p05 = np.percentile(values, 5,  axis=0)
    p25 = np.percentile(values, 25, axis=0)
    p50 = np.percentile(values, 50, axis=0)
    p75 = np.percentile(values, 75, axis=0)
    p95 = np.percentile(values, 95, axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.fill_between(r, p05, p95, color="#b8d4e3", alpha=0.55,
                    label="P05–P95")
    ax.fill_between(r, p25, p75, color="#3a7ca5", alpha=0.55,
                    label="IQR (P25–P75)")
    ax.plot(r, p50, color="#16425b", lw=2.0, label="медиана")
    ax.axhline(0.0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel(r"$r$ (нормированный)")
    ax.set_ylabel(f"{label}  [{unit}]")
    ax.set_title(f"Радиальный профиль распределения {label}")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_hexbin(values: np.ndarray, *, label: str, unit: str, out: Path):
    n_sets, n_r = values.shape
    r = np.tile(np.linspace(0.0, 1.0, n_r), n_sets)
    v = values.reshape(-1)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hb = ax.hexbin(r, v, gridsize=50, cmap="viridis", mincnt=1,
                   norm=LogNorm())
    ax.set_xlabel(r"$r$ (нормированный)")
    ax.set_ylabel(f"{label}  [{unit}]")
    ax.set_title(f"2D-плотность $(r, \\,${label}$)$  "
                 f"(N = {v.size:,})")
    cb = fig.colorbar(hb, ax=ax, label="число точек (log)")
    cb.outline.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def render(y: np.ndarray, comps, kind: str):
    """y: (N_sets, N_R, 4)."""
    for i, (slug, label, unit) in enumerate(comps):
        out_dir = OUT / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        comp = y[..., i]
        plot_hist   (comp, label=label, unit=unit, out=out_dir / "hist.png")
        plot_profile(comp, label=label, unit=unit, out=out_dir / "profile.png")
        plot_hexbin (comp, label=label, unit=unit, out=out_dir / "hexbin.png")
        print(f"  [{kind}] {slug:9s} → {out_dir.relative_to(ROOT)}")


def main():
    print(f"data dir : {DATA}")
    print(f"out dir  : {OUT}")
    y_stress = load_components(DATA / "y_stress.pkl")
    y_strain = load_components(DATA / "y_strain.pkl")
    print(f"y_stress : (N_sets={y_stress.shape[0]}, N_R={y_stress.shape[1]})")
    print(f"y_strain : (N_sets={y_strain.shape[0]}, N_R={y_strain.shape[1]})")
    render(y_stress, STRESS, "stress")
    render(y_strain, STRAIN, "strain")
    print("done.")


if __name__ == "__main__":
    main()
