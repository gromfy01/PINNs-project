"""
residual_maps.py — где именно физика меняет решение.

Карта ошибки предсказания на этот вопрос не отвечает: по точности семейства
почти неразличимы (R² 0.98–0.99 у всех трёх). Расходятся они в другом — в том,
насколько решение УДОВЛЕТВОРЯЕТ уравнениям равновесия и граничному условию.
Эти величины считаются поточечно, поэтому их можно нарисовать полем.

Идея заимствована из Zhang et al., Cryogenics 158 (2026) 104364, Fig. 8(d):
там рисуют пространственное распределение каждой компоненты лосса. Здесь —
три величины:

    R_r = ∂σ_rr/∂r + ∂τ_rz/∂z + (σ_rr − σ_θθ)/r     радиальное равновесие
    R_z = ∂τ_rz/∂r + ∂σ_zz/∂z + τ_rz/r              осевое равновесие
    σ_rr(r → 1)                                      traction-free на поверхности

Базовая линия — САМ МКЭ. Без неё цифра модели не интерпретируется: источник
тоже не удовлетворяет уравнениям точно (узловое осреднение, дискретизация),
и модель, которая нарушает равновесие слабее источника, лучше него по физике.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from contours import DIVERGING, INK, MUTED, _align, _pick_set, _regime, _sym, _where  # noqa: E402
from dataset2d import Field2D  # noqa: E402

R_CLIP = 0.05          # тот же клип 1/r у оси, что в тренере


def residuals(y: np.ndarray, r_phys: np.ndarray, z_phys: np.ndarray):
    """
    Невязки равновесия, БЕЗРАЗМЕРНЫЕ.

    Сама невязка имеет размерность МПа/м и величину порядка 10³, что ничего не
    говорит читателю: непонятно, много это или мало. Поэтому каждая невязка
    делится на масштаб СВОИХ слагаемых — ровно на тот, по которому она
    нормирована в функции потерь (`trainer2d.res_r_scale`, `res_z_scale`):

        scale_r = (sd[σ_rr] + sd[σ_θθ]) / R_char
        scale_z =  sd[σ_zz] / L_char + sd[τ_rz] / R_char

    Результат читается прямо: 0.1 означает, что уравнение не сходится на 10 %
    от характерной величины входящих в него членов, 1.0 — что невязка одного
    порядка с ними, то есть уравнение не выполняется вовсе.
    """
    srr, stt, szz, trz = (y[..., i] for i in range(4))
    dr = np.gradient(r_phys, axis=2)
    dz = np.gradient(z_phys, axis=1)[:, :, None]
    r_safe = np.maximum(r_phys, r_phys[:, :, -1:] * R_CLIP)
    R_r = np.gradient(srr, axis=2) / dr + np.gradient(trz, axis=1) / dz \
        + (srr - stt) / r_safe
    R_z = np.gradient(trz, axis=2) / dr + np.gradient(szz, axis=1) / dz \
        + trz / r_safe
    # характерные масштабы — те же, что в trainer2d
    R_char = float(np.mean(r_phys[:, :, -1]))
    L_char = float(np.mean(z_phys[:, -1] - z_phys[:, 0]))
    sd = y.reshape(-1, 4).std(axis=0)
    scale_r = (sd[0] + sd[1]) / R_char
    scale_z = sd[2] / L_char + sd[3] / R_char
    return R_r / scale_r, R_z / scale_z


def fig_residuals(npzs: List[str], labels: List[str], field2d: str, out: str,
                  which: int = -1, where: Optional[Dict[str, float]] = None):
    common, yt_all, preds, geo = _align(npzs)
    f = Field2D.load(field2d)
    rp, zp = f.r_phys[common], f.z_phys[common]
    if not np.allclose(f.y[common], yt_all):
        raise SystemExit("выравнивание не сошлось: индексы npz не от этого датасета")

    err = np.sqrt(((preds[-1] - yt_all) ** 2).mean(axis=(1, 2, 3)))
    which = _pick_set(geo, err, which, where)

    rows = [("МКЭ", yt_all)] + list(zip(labels, preds))
    # ВАЖНО: нормировочные масштабы берутся из данных МКЭ и одни на все
    # семейства, иначе каждое нормировалось бы на себя и панели стали бы
    # несравнимы — а рисунок строится именно ради сравнения
    R_char = float(np.mean(rp[:, :, -1]))
    L_char = float(np.mean(zp[:, -1] - zp[:, 0]))
    sd = yt_all.reshape(-1, 4).std(axis=0)
    sc_r = (sd[0] + sd[1]) / R_char
    sc_z = sd[2] / L_char + sd[3] / R_char
    def _res(y):
        srr, stt, szz, trz = (y[..., i] for i in range(4))
        dr = np.gradient(rp, axis=2); dz = np.gradient(zp, axis=1)[:, :, None]
        rs = np.maximum(rp, rp[:, :, -1:] * R_CLIP)
        a = np.gradient(srr, axis=2) / dr + np.gradient(trz, axis=1) / dz + (srr - stt) / rs
        b = np.gradient(trz, axis=2) / dr + np.gradient(szz, axis=1) / dz + trz / rs
        return a / sc_r, b / sc_z
    Rr = {lbl: _res(y)[0] for lbl, y in rows}
    Rz = {lbl: _res(y)[1] for lbl, y in rows}
    # внутренние узлы: на краях центральная разность вырождается в одностороннюю
    inner = (slice(None), slice(1, -1), slice(1, -1))

    R = geo["r"][which]
    Z = np.repeat(geo["z"][which][:, None], yt_all.shape[2], axis=1)
    lim_r = _sym(np.concatenate([v[which][1:-1, 1:-1].ravel() for v in Rr.values()]))
    lim_z = _sym(np.concatenate([v[which][1:-1, 1:-1].ravel() for v in Rz.values()]))

    n = len(rows)
    # sharex НЕ ставим: верхние два блока по r/R (0..1), нижний — по МПа
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n + 1.6, 8.0),
                             layout="constrained")
    blocks = [(r"(a)  $R_r$ — радиальное равновесие, безразм.", Rr, lim_r),
              (r"(b)  $R_z$ — осевое равновесие, безразм.", Rz, lim_z)]
    for bi, (title, data, lim) in enumerate(blocks):
        for j, (lbl, _) in enumerate(rows):
            ax = axes[bi, j]
            v = data[lbl][which]
            lev = np.linspace(-lim, lim, 21)
            cf = ax.contourf(R, Z, v, levels=lev, cmap=DIVERGING, extend="both")
            med = np.median(np.abs(data[lbl][inner]))
            ax.set_title(f"{lbl}\nмедиана |{'$R_r$' if bi == 0 else '$R_z$'}| = "
                         f"{med:.3f}", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
            ax.set_xlabel(r"$r/R$")
            if j == 0:
                ax.set_ylabel(r"$z/L$")
            else:
                ax.tick_params(labelleft=False)
        cb = fig.colorbar(cf, ax=axes[bi, :], fraction=0.035, pad=0.015,
                          ticks=np.linspace(-lim, lim, 5))
        cb.set_label("невязка / масштаб членов уравнения", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, color=MUTED)
        axes[bi, 0].text(-0.42, 0.5, title, transform=axes[bi, 0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=8.5)

    # (c) нарушение traction-free — распределение по наборам, а не поле
    hi_bc = max(np.abs(y[:, :, -1, 0]).mean(axis=1).max() for _, y in rows)
    bins = np.linspace(0, hi_bc, 30)
    for j, (lbl, y) in enumerate(rows):
        ax = axes[2, j]
        prof = np.abs(y[:, :, -1, 0]).mean(axis=1)      # |σ_rr(r=1)| по наборам
        ax.hist(prof, bins=bins, color="#0072B2" if j else "#8a8a8a",
                edgecolor="white", linewidth=0.3)
        m = float(prof.mean())
        ax.axvline(m, color="#D55E00", lw=1.2)
        ax.set_title(f"{lbl}\nсреднее = {m:.2f} МПа", fontsize=8)
        ax.set_xlim(0, hi_bc)
        ax.set_xlabel(r"$|\sigma_{rr}(r=1)|$, МПа")
        if j == 0:
            ax.set_ylabel("наборов")
        else:
            ax.tick_params(labelleft=False)
    axes[2, 0].text(-0.42, 0.5, "(c)  нарушение traction-free",
                    transform=axes[2, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=8.5)

    fig.suptitle("Где физика меняет решение.  Поля — набор с медианной ошибкой: "
                 f"{_regime(geo['proc'][which])};  числа — по всему тесту "
                 f"({len(common)} наборов)", fontsize=8.5)
    fig.text(0.5, -0.015,
             "Невязка ОБЕЗРАЗМЕРЕНА: поделена на масштаб входящих в уравнение "
             r"членов ($\mathrm{scale}_r=(\mathrm{sd}[\sigma_{rr}]+"
             r"\mathrm{sd}[\sigma_{\theta\theta}])/R$, "
             r"$\mathrm{scale}_z=\mathrm{sd}[\sigma_{zz}]/L+"
             r"\mathrm{sd}[\tau_{rz}]/R$), " "\n"
             "то же обезразмеривание, что в функции потерь. Значение 0.1 — "
             "уравнение не сходится на 10 % от характерной величины своих "
             "членов; 1.0 — не выполняется вовсе.\n"
             "Масштабы взяты из данных МКЭ и ОДНИ на все семейства, иначе "
             "панели были бы несравнимы. Базовая линия — сам МКЭ: источник тоже "
             "не удовлетворяет уравнениям точно.\n"
             "Медианы по ВНУТРЕННИМ узлам: на краях центральная разность "
             "вырождается в одностороннюю. Семейства отличаются только составом "
             "лосса.",
             ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npzs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--field2d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", type=int, default=-1)
    ap.add_argument("--where", default="")
    a = ap.parse_args()
    npzs = [x.strip() for x in a.npzs.split(",") if x.strip()]
    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    if len(labels) != len(npzs):
        raise SystemExit(f"--labels: {len(labels)} против {len(npzs)} npz")
    print("создан", fig_residuals(npzs, labels, a.field2d, a.out,
                                  a.which, _where(a.where)))


if __name__ == "__main__":
    main()
