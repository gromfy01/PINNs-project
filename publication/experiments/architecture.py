"""
architecture.py — схема входа, сети и состава функции потерь.

Заменяет схему из отклонённой версии статьи, где вход показан как
[Q, k, α, μ, v, r] без осевой координаты. Reviewer #1 указал ровно на это: в
тексте напечатана полная форма уравнений равновесия с ∂/∂z, а во входе сети
z нет, поэтому осевые производные не существуют и осевое уравнение не может
быть записано.

Рисунок показывает оба трека рядом и явно помечает, что именно добавляется, а
также какие члены функции потерь при этом становятся вычислимыми.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED = "#1a1a1a", "#8a8a8a"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
GREY = "#e8e8e8"

plt.rcParams.update({
    "font.size": 8, "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 170, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _box(ax, x, y, w, h, text, fc=GREY, ec=MUTED, fs=8, lw=0.8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, color=INK, fontweight=weight)


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=9, lw=lw, color=color, zorder=1))


def _banner(ax, y, text, color):
    ax.add_patch(FancyBboxPatch((0.005, y), 0.99, 0.032,
                                boxstyle="round,pad=0.004",
                                fc=color, ec="none", alpha=0.16, zorder=1))
    ax.text(0.5, y + 0.016, text, ha="center", va="center",
            fontsize=9, fontweight="bold", color=INK, zorder=3)


def build(out: str):
    fig, ax = plt.subplots(figsize=(9.4, 9.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ─────────────── (a) вход и сеть ───────────────
    _banner(ax, 0.955, "(a)  Вход и архитектура: два трека", BLUE)

    for row, (tag, feats, hi, ycent) in enumerate((
            ("основной трек",
             ["Q", "k", r"$\alpha$", r"$\mu$", "v", "r"], None, 0.855),
            ("расширенный трек",
             ["Q", "k", r"$\alpha$", r"$\mu$", "v", "z", "r"], 5, 0.700))):
        ax.text(0.012, ycent + 0.045, tag, fontsize=8.5, color=INK,
                fontweight="bold")
        n = len(feats)
        bw, gap = 0.036, 0.007
        x0 = 0.045
        for i, f in enumerate(feats):
            fc = ORANGE if i == hi else GREY
            al = "bold" if i == hi else "normal"
            _box(ax, x0 + i * (bw + gap), ycent - 0.022, bw, 0.044, f,
                 fc=fc if i != hi else "#ffd9c2",
                 ec=ORANGE if i == hi else MUTED, weight=al)
        xin = x0 + n * (bw + gap)
        ax.text(x0 + n * (bw + gap) / 2 - 0.02, ycent - 0.052,
                f"{n} признаков", fontsize=7, color=MUTED, ha="center")

        _arrow(ax, xin, ycent, xin + 0.028, ycent)
        _box(ax, xin + 0.030, ycent - 0.030, 0.115, 0.060,
             "Фурье по r\n" + r"$\sin/\cos(2\pi k r)$" + "\nk = 1…4", fs=7)
        _arrow(ax, xin + 0.147, ycent, xin + 0.175, ycent)
        _box(ax, xin + 0.177, ycent - 0.030, 0.130, 0.060,
             "MLP\n3 × 96, tanh\nfloat64", fs=7.5)
        _arrow(ax, xin + 0.309, ycent, xin + 0.337, ycent)
        _box(ax, xin + 0.339, ycent - 0.030, 0.150, 0.060,
             r"$(\sigma_{rr},\ \sigma_{\theta\theta},$" + "\n" +
             r"$\sigma_{zz},\ \tau_{rz})$", fs=8)

    ax.annotate("добавлена осевая\nкоордината",
                xy=(0.045 + 5 * 0.043 + 0.018, 0.678), xytext=(0.30, 0.612),
                fontsize=7.5, color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0))

    # ─────────────── (b) что становится вычислимым ───────────────
    _banner(ax, 0.545, "(b)  Что даёт z: осевые производные и полная форма равновесия",
            ORANGE)

    ax.text(0.03, 0.492, "основной трек — z нет во входе:", fontsize=8,
            color=INK, fontweight="bold")
    ax.text(0.05, 0.455,
            r"$R_r = \dfrac{\partial\sigma_{rr}}{\partial r}"
            r" + \dfrac{\sigma_{rr}-\sigma_{\theta\theta}}{r}$"
            "        (отброшено " r"$\partial\tau_{rz}/\partial z$" ")",
            fontsize=9, color=INK)
    ax.text(0.05, 0.412,
            r"$R_z$" " — записать нельзя:\nбез z осевых производных не существует",
            fontsize=8.5, color=ORANGE, va="top", linespacing=1.35)

    ax.text(0.03, 0.345, "расширенный трек — z во входе:", fontsize=8,
            color=INK, fontweight="bold")
    ax.text(0.05, 0.305,
            r"$R_r = \dfrac{\partial\sigma_{rr}}{\partial r}"
            r" + \dfrac{\partial\tau_{rz}}{\partial z}"
            r" + \dfrac{\sigma_{rr}-\sigma_{\theta\theta}}{r}$",
            fontsize=9, color=INK)
    ax.text(0.05, 0.258,
            r"$R_z = \dfrac{\partial\tau_{rz}}{\partial r}"
            r" + \dfrac{\partial\sigma_{zz}}{\partial z}"
            r" + \dfrac{\tau_{rz}}{r}$",
            fontsize=9, color=INK)
    ax.text(0.615, 0.480,
            "Производные — autograd по входным\n"
            "координатам, с цепным правилом:\n"
            r"$\partial/\partial r_{\rm физ} = (1/R)\,\partial/\partial r_{\rm норм}$" "\n"
            r"$\partial/\partial z_{\rm физ} = (1/L)\,\partial/\partial z_{\rm норм}$",
            fontsize=7.5, color=INK, va="top")
    ax.text(0.615, 0.345,
            "Отброшенное измерено на данных:\n"
            r"радиальное — 1.3 % от суммы удержанных," "\n"
            r"осевое — 68 %. Редукция законна только" "\n"
            "для первого уравнения.",
            fontsize=7.5, color=MUTED, va="top")

    # ─────────────── (c) состав лосса по семействам ───────────────
    _banner(ax, 0.200, "(c)  Семейства отличаются ТОЛЬКО составом функции потерь",
            GREEN)

    cols = ["семейство", r"$L_{\rm data}$", r"$R_r$", r"$R_z$", r"$L_{\rm BC}$"]
    rows = [("MLP", "✓", "—", "—", "—"),
            ("PINN, редуц. форма", "✓", "без " + r"$\partial\tau_{rz}/\partial z$", "—", "✓"),
            ("PINN, полная форма", "✓", "✓", "✓", "✓")]
    xs = [0.045, 0.300, 0.420, 0.580, 0.720]
    for j, c in enumerate(cols):
        ax.text(xs[j], 0.158, c, fontsize=8, fontweight="bold", color=INK)
    for i, r in enumerate(rows):
        y = 0.124 - i * 0.034
        for j, cell in enumerate(r):
            col = GREEN if cell == "✓" else (MUTED if cell == "—" else ORANGE)
            ax.text(xs[j], y, cell, fontsize=8,
                    color=INK if j == 0 else col,
                    fontweight="bold" if j == 0 else "normal")
    ax.plot([0.04, 0.96], [0.148, 0.148], color=MUTED, lw=0.6)

    ax.text(0.045, 0.008,
            r"$L = L_{\rm data} + \lambda_{\rm phys}\cdot g\cdot L_{\rm phys}"
            r" + \lambda_{\rm BC}\cdot L_{\rm BC}$,      "
            r"$L_{\rm phys} = \overline{\log(1+(R_r/s_r)^2)}"
            r" + \overline{\log(1+(R_z/s_z)^2)}$",
            fontsize=8.5, color=INK)
    ax.text(0.045, -0.028,
            "Множитель g калибруется по норме градиента: физический член "
            r"приводится к доле $\lambda_{\rm phys}$ от нормы градиента "
            "data-члена.\n"
            r"$L_{\rm BC}$ — traction-free при $r/R = 1$: "
            r"$\sigma_{rr} = \tau_{rz} = 0$ при каждом z. "
            "Архитектура, сплит, сид, число эпох и оптимизатор у всех "
            "семейств одинаковы.",
            fontsize=7.5, color=MUTED, va="top")

    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    print("создан", build(ap.parse_args().out))
