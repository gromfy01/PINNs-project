"""
physics.py — сетка r–z, разностные производные, hard-потенциалы и невязки (ТЗ, п. 5, прил. А).

Поле хранится как тензор (..., n_r, n_z): ось −2 — ρ = r/R от оси (0) до поверхности (1),
ось −1 — ζ = z/R. Компоненты в порядке (σ_r, σ_θ, σ_z, τ_rz).

Два набора разностных операторов, намеренно разных:
  * d1, d2 (torch, 2-й порядок, односторонние шаблоны 2-го порядка на краях) — ими
    строятся компоненты hard-варианта из N, M;
  * d1_eval (numpy, 4-й порядок) — им считается невязка в метриках, одинаково для МКЭ
    и всех моделей. Если считать невязку тем же оператором, которым строились поля,
    критерий становится тавтологией.
"""
from __future__ import annotations

import numpy as np
import torch


def d1(f: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    """Первая производная, 2-й порядок: центральная внутри, односторонняя на краях."""
    f = f.movedim(dim, -1)
    g = torch.empty_like(f)
    g[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2 * h)
    g[..., 0] = (-3 * f[..., 0] + 4 * f[..., 1] - f[..., 2]) / (2 * h)
    g[..., -1] = (3 * f[..., -1] - 4 * f[..., -2] + f[..., -3]) / (2 * h)
    return g.movedim(-1, dim)


def d2(f: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    """Вторая производная, 2-й порядок: трёхточечная внутри, четырёхточечная на краях."""
    f = f.movedim(dim, -1)
    g = torch.empty_like(f)
    g[..., 1:-1] = (f[..., 2:] - 2 * f[..., 1:-1] + f[..., :-2]) / h ** 2
    g[..., 0] = (2 * f[..., 0] - 5 * f[..., 1] + 4 * f[..., 2] - f[..., 3]) / h ** 2
    g[..., -1] = (2 * f[..., -1] - 5 * f[..., -2] + 4 * f[..., -3] - f[..., -4]) / h ** 2
    return g.movedim(-1, dim)


class Grid:
    """Регулярная сетка ρ ∈ [0, 1] (n_r узлов), ζ ∈ [0, W] (n_z узлов)."""

    def __init__(self, nr: int, nz: int, W: float):
        self.nr, self.nz, self.W = nr, nz, W
        self.hr = 1.0 / (nr - 1)
        self.hz = W / (nz - 1)
        self.rho = np.linspace(0.0, 1.0, nr)
        self.zeta = np.linspace(0.0, W, nz)

    def rho_t(self, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.rho, dtype=like.dtype, device=like.device)[:, None]


def hard_fields(N: torch.Tensor, M: torch.Tensor, g: Grid) -> torch.Tensor:
    """
    Компоненты из потенциалов N, M (прил. А.5): равновесие, σ_r = τ_rz = 0 на поверхности,
    τ_rz = 0 и σ_r = σ_θ на оси, нулевая осевая сила — по построению.
    N, M: (B, n_r, n_z) → (B, 4, n_r, n_z).
    """
    rho = g.rho_t(N)
    q = rho * (1 - rho)
    N_r = d1(N, g.hr, -2)
    M_r = d1(M, g.hr, -2)
    M_z = d1(M, g.hz, -1)
    M_zz = d2(M, g.hz, -1)
    s_r = (1 - rho) * N
    s_z = (2 - 3 * rho) * M + q * M_r
    t_rz = -q * M_z
    s_t = (1 - 2 * rho) * N + q * N_r - rho * q * M_zz
    return torch.stack([s_r, s_t, s_z, t_rz], dim=1)


def profile_fields(N: torch.Tensor, S: torch.Tensor, g: Grid) -> torch.Tensor:
    """
    Одномерный hard-вариант (прил. А.8): φ = ρ(1 − ρ)N, σ_r = φ/ρ, σ_θ = dφ/dρ,
    из σ_z вычитается взвешенное среднее 2∫σ_zρdρ, τ_rz = 0. Профиль одинаков по ζ.
    N, S: (B, n_r) → (B, 4, n_r, n_z).
    """
    rho = torch.as_tensor(g.rho, dtype=N.dtype, device=N.device)
    q = rho * (1 - rho)
    s_r = (1 - rho) * N
    s_t = (1 - 2 * rho) * N + q * d1(N, g.hr, -1)
    w = rho * g.hr
    w = torch.cat([w[:1] * 0.5, w[1:-1], w[-1:] * 0.5])        # трапеции
    s_z = S - 2 * (S * w).sum(-1, keepdim=True)
    t = torch.zeros_like(N)
    prof = torch.stack([s_r, s_t, s_z, t], dim=1)                # (B, 4, n_r)
    return prof[..., None].expand(-1, -1, -1, g.nz)


# ───────────────────────────── метрики (numpy) ─────────────────────────────

def d1_eval(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Первая производная 4-го порядка (центральная внутри, односторонняя у краёв)."""
    f = np.moveaxis(f, axis, -1)
    g = np.empty_like(f)
    g[..., 2:-2] = (-f[..., 4:] + 8 * f[..., 3:-1] - 8 * f[..., 1:-3] + f[..., :-4]) / (12 * h)
    g[..., 0] = (-25 * f[..., 0] + 48 * f[..., 1] - 36 * f[..., 2] + 16 * f[..., 3] - 3 * f[..., 4]) / (12 * h)
    g[..., 1] = (-3 * f[..., 0] - 10 * f[..., 1] + 18 * f[..., 2] - 6 * f[..., 3] + f[..., 4]) / (12 * h)
    g[..., -1] = -(-25 * f[..., -1] + 48 * f[..., -2] - 36 * f[..., -3] + 16 * f[..., -4] - 3 * f[..., -5]) / (12 * h)
    g[..., -2] = -(-3 * f[..., -1] - 10 * f[..., -2] + 18 * f[..., -3] - 6 * f[..., -4] + f[..., -5]) / (12 * h)
    return np.moveaxis(g, -1, axis)


def residuals(y: np.ndarray, g: Grid):
    """Невязки уравнений, умноженных на ρ (ТЗ, F2–F3), МПа. y: (..., 4, n_r, n_z)."""
    rho = g.rho[:, None]
    s_r, s_t, s_z, t = (y[..., c, :, :] for c in range(4))
    R1 = d1_eval(rho * s_r, g.hr, -2) + d1_eval(rho * t, g.hz, -1) - s_t
    R2 = d1_eval(rho * t, g.hr, -2) + d1_eval(rho * s_z, g.hz, -1)
    return R1, R2


def section_force(y: np.ndarray, g: Grid) -> np.ndarray:
    """Средняя по сечению σ_z = 2∫₀¹σ_zρdρ для каждого ζ, МПа. y: (..., 4, n_r, n_z)."""
    return 2.0 * np.trapezoid(y[..., 2, :, :] * g.rho[:, None], g.rho, axis=-2)


def stress_intensity(y: np.ndarray) -> np.ndarray:
    """σ_i (ТЗ, F1) из четырёх компонент."""
    s_r, s_t, s_z, t = (y[..., c, :, :] for c in range(4))
    return np.sqrt(0.5 * ((s_r - s_t) ** 2 + (s_t - s_z) ** 2 + (s_z - s_r) ** 2) + 3 * t ** 2)
