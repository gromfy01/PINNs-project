"""TotalStrainSystem — strain PINN с уравнениями совместности Сен-Венана.

Архитектура, описанная в §5.4.5 ВКР: «отдельная архитектура TotalStrainSystem,
в которой роль физического ограничения выполняют уравнения совместности
Сен-Венана; вес соответствующего слагаемого подбирается Optuna с учётом
дисперсионной структуры данных».

Сеть предсказывает четыре компоненты тензора деформаций
(ε_rr, ε_θθ, ε_zz, ε_rz) от шестимерного входа (Q, k, α, μ, v, r). Физическое
ограничение реализовано через осесимметричные уравнения совместности
Сен-Венана:

    ∂²ε_rr/∂z² + ∂²ε_zz/∂r² − (1/r)·∂ε_zz/∂r = (1/r)·∂/∂r(r·∂ε_rz/∂z),   (27)
    ∂²ε_θθ/∂z² + (1/r)·∂/∂r(r·∂ε_θθ/∂z) = 0.                              (28)

Поскольку доступная МКЭ-выборка содержит профили вдоль одной z-плоскости
(установившаяся зона, см. §5.1.2 ВКР), при включении в loss применяется
single-z редукция: ∂/∂z обнуляется и уравнения сводятся к одному скалярному
ограничению на радиальное направление

    R(r) = ε_rr − ε_θθ − r·∂ε_θθ/∂r.                                       (R)

L_compat = mean[ log1p(R² / scale²) ]   ← log1p ограничивает выбросы,
                                          scale = y_std для нормализации.

Окончательный вес λ_compat подбирается Optuna в логарифмической шкале
1e-5..1e-1, как и в остальных strain-PINN модулях.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

N_INPUTS  = 6        # [Q, k, α, μ, v, r]
N_OUTPUTS = 4        # [ε_rr, ε_θθ, ε_zz, ε_rz]
R_COL_IDX = 5        # последний столбец входа

ACTIVATIONS = {
    "tanh":     nn.Tanh,
    "selu":     nn.SELU,
    "softplus": nn.Softplus,
    "gelu":     nn.GELU,
}


class TotalStrainSystem(nn.Module):
    """Полносвязная сеть, выдающая (ε_rr, ε_θθ, ε_zz, ε_rz) от 6-мерного входа.

    Структура идентична другим strain-PINN модулям (pinn_strain_torch.py
    и др.) — отличается только составом функции потерь, который содержит
    L_compat вместо/в дополнение к equilibrium-residuals.
    """

    def __init__(self, hidden_sizes: Tuple[int, ...], activation: str = "tanh"):
        super().__init__()
        Act = ACTIVATIONS.get(activation, nn.Tanh)
        layers: List[nn.Module] = []
        prev = N_INPUTS
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            prev = h
        layers.append(nn.Linear(prev, N_OUTPUTS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compatibility_loss(model: nn.Module,
                       X_std: torch.Tensor,
                       std_r: float, mean_r: float,
                       y_mean: torch.Tensor, y_std: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    """L_compat по single-z редукции уравнений Сен-Венана (27)–(28).

    Возвращает скаляр для добавления в общий лосс с весом λ_compat.

    Parameters
    ----------
    X_std    : (B, 6) — стандартизованный вход; столбец 5 = z-нормированный r.
    std_r    : std r в физических единицах (нужно для chain-rule).
    mean_r   : mean r в физических единицах.
    y_mean   : (1, 4) — среднее по 4 компонентам деформаций (для denorm).
    y_std    : (1, 4) — std по 4 компонентам.
    """
    X_phys = X_std.clone().detach().requires_grad_(True)
    eps_sc = model(X_phys)
    eps_phys = eps_sc * y_std + y_mean
    e_rr = eps_phys[:, 0:1]
    e_tt = eps_phys[:, 1:2]

    grad_outputs = torch.ones_like(eps_sc[:, 1:2])
    de_tt_dx = torch.autograd.grad(
        outputs=eps_sc[:, 1:2],
        inputs=X_phys,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    de_tt_dr_std = de_tt_dx[:, R_COL_IDX:R_COL_IDX + 1]
    de_tt_dr_phys = de_tt_dr_std * (y_std[:, 1:2] / (std_r + eps))

    r_phys = torch.clamp(
        X_phys[:, R_COL_IDX:R_COL_IDX + 1] * std_r + mean_r, min=eps)

    R = e_rr - e_tt - r_phys * de_tt_dr_phys

    scale = (y_std[0, 0] ** 2 + y_std[0, 1] ** 2) + eps
    return torch.mean(torch.log1p(R ** 2 / scale))


@dataclass
class TotalStrainConfig:
    """Гиперпараметры, фигурирующие в Optuna-поиске (§5.4.5 ВКР).

    `lambda_compat` подбирается в логарифмическом диапазоне; типовой ответ
    Optuna лежит в пределах 1e-5..1e-1 (см. главу 3 ВКР).
    """
    hidden_sizes:    Tuple[int, ...] = (128, 128, 128)
    activation:      str             = "tanh"
    learning_rate:   float           = 1e-3
    batch_size:      int             = 64
    max_epochs:      int             = 400
    optimizer:       str             = "AdamW"
    weight_decay:    float           = 1e-4
    grad_clip:       float           = 1.0
    lambda_data:     float           = 1.0
    lambda_compat:   float           = 1e-3   # ← Optuna log-tuned in 1e-5..1e-1
    lambda_profile:  float           = 1e-2
    patience:        int             = 30


def total_loss(model: nn.Module,
               X_std: torch.Tensor,
               y_sc:  torch.Tensor,
               std_r: float, mean_r: float,
               y_mean: torch.Tensor, y_std: torch.Tensor,
               lambda_data:    float = 1.0,
               lambda_compat:  float = 1e-3,
               lambda_profile: float = 0.0,
               n_pts: int = 20) -> Dict[str, torch.Tensor]:
    """Совокупная функция потерь TotalStrainSystem.

    Возвращает словарь с полным лоссом и компонентами для логирования.
    """
    pred_sc = model(X_std)
    L_data  = ((pred_sc - y_sc) ** 2).mean()
    L_comp  = compatibility_loss(
        model, X_std, std_r, mean_r, y_mean, y_std)

    L_prof = torch.zeros((), device=X_std.device)
    if lambda_profile > 0.0:
        N, C = pred_sc.shape
        n_sets = N // n_pts
        if n_sets > 0:
            p = pred_sc[:n_sets * n_pts].reshape(n_sets, n_pts, C)
            t = y_sc[:n_sets * n_pts].reshape(n_sets, n_pts, C)
            L_prof = (
                ((p - p.mean(dim=1, keepdim=True))
                 - (t - t.mean(dim=1, keepdim=True))) ** 2
            ).mean()

    L = lambda_data * L_data + lambda_compat * L_comp + lambda_profile * L_prof
    return {"loss": L, "data": L_data, "compat": L_comp, "profile": L_prof}


__all__ = [
    "TotalStrainSystem",
    "TotalStrainConfig",
    "compatibility_loss",
    "total_loss",
]
