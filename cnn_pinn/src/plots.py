"""
plots.py — графики п. 8 для тестовых режимов: карты полей (МКЭ, сеть, ошибка) и радиальные профили.

    python src/plots.py --data data/trial_w2.npz --runs runs/hard_s0 runs/data_only_s0 --out runs/figs

Шкала расходящаяся и симметричная относительно нуля (синий — сжатие, красный — растяжение),
у МКЭ и сети общая шкала по каждой компоненте, у ошибки — своя. Продольное сечение показано
от −R до R: на отражённой половине τ_rz меняет знак (τ_xz = sign(x)·τ_rz).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = ("σ_r", "σ_θ", "σ_z", "τ_rz")
SIGN_ON_MIRROR = (1, 1, 1, -1)


def mirrored(f: np.ndarray, c: int) -> np.ndarray:
    """(n_r, n_z) → (2n_r − 1, n_z): от −R до R по радиусу."""
    return np.concatenate([SIGN_ON_MIRROR[c] * f[:0:-1], f], axis=0)


def plot_fields(y_fem, preds, names, W, title, path):
    """4 строки (компоненты) × (МКЭ, модели..., ошибки моделей)."""
    n = len(preds)
    fig, axes = plt.subplots(4, 1 + 2 * n, figsize=(2.6 * (1 + 2 * n), 7.5), squeeze=False)
    ext = (0, W, -1, 1)
    for c in range(4):
        vmax = max(np.abs(y_fem[c]).max(), *[np.abs(p[c]).max() for p in preds])
        emax = max(np.abs(p[c] - y_fem[c]).max() for p in preds) or 1.0
        panels = [(y_fem[c], "МКЭ", vmax)] + [(p[c], nm, vmax) for p, nm in zip(preds, names)] \
            + [(p[c] - y_fem[c], f"ошибка: {nm}", emax) for p, nm in zip(preds, names)]
        for j, (f, nm, vm) in enumerate(panels):
            ax = axes[c, j]
            im = ax.imshow(mirrored(f, c), origin="lower", extent=ext, aspect="auto",
                           cmap="RdBu_r", vmin=-vm, vmax=vm)
            if c == 0:
                ax.set_title(nm, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{LABELS[c]}\nr/R", fontsize=9)
            if c == 3:
                ax.set_xlabel("z/R", fontsize=9)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.05).ax.tick_params(labelsize=7)
    fig.suptitle(title + "  (МПа)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_profiles(y_fem, preds, names, nz, title, path):
    """Радиальные профили 4 компонент в трёх сечениях: линии — модели, маркеры — МКЭ."""
    rho = np.linspace(0, 1, y_fem.shape[1])
    cols = (0, nz // 2, nz - 1)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for c, ax in enumerate(axes):
        for k, j in enumerate(cols):
            ax.plot(rho, y_fem[c, :, j], "o", ms=3, color=f"C{k}", label=f"МКЭ, сечение {k + 1}")
            for m, (p, nm) in enumerate(zip(preds, names)):
                ax.plot(rho, p[c, :, j], ("-", "--", ":")[m % 3], color=f"C{k}", lw=1,
                        label=f"{nm}, сечение {k + 1}")
        ax.set_title(LABELS[c])
        ax.set_xlabel("r/R")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("МПа")
    axes[-1].legend(fontsize=6, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.data)
    meta = json.loads(str(d["meta"]))
    W, nz = meta["window_R"], meta["grid"][1]
    preds, names, idx = [], [], None
    for r in a.runs:
        t = np.load(os.path.join(r, "test_pred.npz"))
        idx = t["idx"] if idx is None else idx
        assert np.array_equal(idx, t["idx"]), "у прогонов разные тестовые выборки"
        preds.append(t["pred"])
        names.append(json.load(open(os.path.join(r, "metrics.json")))["config"]["name"])
    Y = d["Y"][idx]
    err = np.sqrt(np.mean((preds[0] - Y) ** 2, axis=(1, 2, 3)))
    order = np.argsort(err)
    picks = {"медиана": order[len(order) // 2], "90-й перцентиль": order[int(0.9 * len(order))]}
    for tag, i in picks.items():
        nm = str(d["names"][idx[i]])
        title = f"{nm} — ошибка первой модели: {tag}"
        safe = tag.replace(" ", "_").replace("-", "")
        plot_fields(Y[i], [p[i] for p in preds], names, W, title, os.path.join(a.out, f"fields_{safe}.png"))
        plot_profiles(Y[i], [p[i] for p in preds], names, nz, title, os.path.join(a.out, f"profiles_{safe}.png"))
    print("графики:", a.out)


if __name__ == "__main__":
    main()
