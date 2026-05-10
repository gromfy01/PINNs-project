import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from .style import MODEL_COLORS, COMPONENT_TEX_STRESS, COMPONENT_TEX_STRAIN, model_marker_style
from evaluation.metrics import smape


def plot_profile_single_component(r_grid, preds_dict, y_true, comp_idx,
                                   params, save_path=None,
                                   component_labels=None):
    if component_labels is None:
        component_labels = COMPONENT_TEX_STRESS

    fig, ax = plt.subplots(figsize=(9, 5.2))

    if y_true is not None:
        marker, ls, lw, ms = model_marker_style("FEM")
        ax.plot(r_grid, y_true[:, comp_idx],
                color=MODEL_COLORS["FEM"], marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, label="FEM")

    for name, ypred in preds_dict.items():
        marker, ls, lw, ms = model_marker_style(name)
        ax.plot(r_grid, ypred[:, comp_idx],
                color=MODEL_COLORS.get(name, "#444"),
                marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, alpha=0.9, label=name)

    ax.set_xlabel(r"$r,\,-$", fontsize=12)
    ax.set_ylabel("Прогноз", fontsize=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    cap = (f"{component_labels[comp_idx]}: "
           f"Q = {params['Q']*100:.1f}%, k = {params['k']:.2f}, "
           f"α = {params['alpha']:.0f}°, μ = {params['mu']:.3f}, "
           f"v = {params['v']:.0f} м/мин")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=11)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_profile_all_components(r_grid, preds_dict, y_true, params,
                                 save_path=None, component_labels=None):
    if component_labels is None:
        component_labels = COMPONENT_TEX_STRESS

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()

    for ci in range(4):
        ax = axes[ci]
        if y_true is not None:
            marker, ls, lw, ms = model_marker_style("FEM")
            ax.plot(r_grid, y_true[:, ci],
                    color=MODEL_COLORS["FEM"], marker=marker, linestyle=ls,
                    linewidth=lw, markersize=ms, label="FEM")
        for name, ypred in preds_dict.items():
            marker, ls, lw, ms = model_marker_style(name)
            ax.plot(r_grid, ypred[:, ci],
                    color=MODEL_COLORS.get(name, "#444"),
                    marker=marker, linestyle=ls,
                    linewidth=lw, markersize=ms - 1, alpha=0.85, label=name)
        ax.set_title(component_labels[ci], fontsize=13)
        ax.set_xlabel(r"$r,\,-$", fontsize=11)
        ax.grid(True, alpha=0.3)
        if ci == 0:
            ax.legend(loc="best", fontsize=9, framealpha=0.85)

    cap = (f"Q = {params['Q']*100:.1f}%, k = {params['k']:.2f}, "
           f"α = {params['alpha']:.0f}°, μ = {params['mu']:.3f}, "
           f"v = {params['v']:.0f} м/мин")
    fig.text(0.5, -0.005, cap, ha="center", fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scatter_grid(predictions, y_true, ordered_names, save_path=None,
                      component_labels=None):
    if component_labels is None:
        component_labels = COMPONENT_TEX_STRESS

    n_models = len(ordered_names)
    fig, axes = plt.subplots(n_models, 5, figsize=(26, 4.5 * n_models),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.30})
    if n_models == 1:
        axes = axes[np.newaxis, :]

    comp_colors = ["steelblue", "tomato", "seagreen", "mediumpurple"]

    for row, name in enumerate(ordered_names):
        y_pred = predictions[name]

        for col in range(4):
            ax = axes[row, col]
            yt, yp = y_true[:, col], y_pred[:, col]
            r2 = r2_score(yt, yp)
            rm = np.sqrt(mean_squared_error(yt, yp))
            sm = smape(yt, yp)

            ax.scatter(yt, yp, s=4, alpha=0.35, color=comp_colors[col])
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.3)

            if col == 0:
                ax.set_ylabel(f"#{row + 1}  {name}\n{component_labels[col]}",
                              fontsize=11)
            else:
                ax.set_ylabel(component_labels[col], fontsize=11)
            ax.set_xlabel("FEM", fontsize=10)
            ax.set_title(f"R²={r2:.3f}  RMSE={rm:.1f}  SMAPE={sm:.1f}%",
                         fontsize=10)
            ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
