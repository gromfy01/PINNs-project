import matplotlib.pyplot as plt


MODEL_COLORS = {
    "MLP-grid":   "#ff7f0e",
    "MLP-optuna": "#ffbb78",
    "PINN-torch": "#1f77b4",
    "PINN-dde":   "#17becf",
    "PINN-jax":   "#aec7e8",
    "VPINN":      "#2ca02c",
    "FEM":        "#888888",
}


COMPONENT_TEX_STRESS = [
    r"$\sigma_{rr}$",
    r"$\sigma_{\theta\theta}$",
    r"$\sigma_{zz}$",
    r"$\tau_{rz}$",
]

COMPONENT_TEX_STRAIN = [
    r"$\varepsilon_{rr}^{p}$",
    r"$\varepsilon_{\theta\theta}^{p}$",
    r"$\varepsilon_{zz}^{p}$",
    r"$\gamma_{rz}^{p}$",
]


def apply_thesis_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def model_marker_style(name):
    base = {
        "FEM":        ("o", "-",  2.0, 6),
        "MLP-grid":   ("v", "--", 1.4, 5),
        "MLP-optuna": ("D", "--", 1.4, 5),
        "PINN-torch": ("s", "--", 1.4, 5),
        "PINN-dde":   ("^", "--", 1.4, 5),
        "PINN-jax":   ("P", "--", 1.4, 5),
        "VPINN":      ("*", "--", 1.4, 6),
    }
    return base.get(name, ("x", "--", 1.2, 4))
