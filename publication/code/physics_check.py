"""
physics_check.py — физика, приведённая в соответствие с данными и с кодом.

Три исправления против отклонённой версии.

E-01. Положение свободной поверхности. В коде моделей стоит
      R_EXTRAP_FREE_SURFACE = 1.5 и комментарий «r_norm ∈ [0,1] покрывает
      25–75 % радиуса, истинная поверхность при r_norm = 1.5». Проверка по
      сырым .rpt (publication/audit/) это опровергает:

          окно 25–75 % — ОСЕВОЕ (по координате z, зона выхода на стационар);
          20 радиальных точек покрывают ВЕСЬ радиус, r_norm = 0 → ось,
          r_norm = 1 → свободная поверхность.

      Медиана r_max_окна / R = 0.996; средний |σ_rr| в крайней точке = 3.8 МПа
      против |σ_θθ| = 114 и |σ_zz| = 129 МПа. Traction-free выполняется именно
      при r_norm = 1. Значит r_norm = 1.5 — это 1.5 R, точка ВНЕ материала,
      и условие σ_rr = τ_rz = 0 там физического смысла не имеет.

      Здесь R_FREE_SURFACE = 1.0.

E-02. Форма уравнений равновесия. Полная осесимметричная форма

          R_r = ∂σ_rr/∂r + ∂τ_rz/∂z + (σ_rr − σ_θθ)/r
          R_z = ∂τ_rz/∂r + ∂σ_zz/∂z + τ_rz/r

      в коде реализована без осевых членов — z не входит во вход сети
      X = [Q, k, α, μ, v, r]. Численная оценка отброшенного по сырым полям
      (2443 набора, медианы по наборам):

          |∂τ_rz/∂z| / (|∂σ_rr/∂r| + |(σ_rr−σ_θθ)/r|) = 0.013   ⇒ отбрасывать МОЖНО
          |∂σ_zz/∂z| / (|∂τ_rz/∂r| + |τ_rz/r|)        = 0.68    ⇒ отбрасывать НЕЛЬЗЯ

      Поэтому редуцированной формой пользуемся ТОЛЬКО для первого уравнения,
      а «второе уравнение» без ∂σ_zz/∂z не является приближением осевого
      равновесия и называть его так нельзя. См. EQUILIBRIUM_R2_STATUS.

E-03. Нарушение ГУ надо мерить там же, где оно налагается. В отклонённой
      версии BC налагалось при r_norm = 1.5, а измерялось в крайней точке
      профиля (r_norm = 1). Теперь и то и другое при r_norm = 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: свободная поверхность в нормированных радиальных координатах.
#: Подтверждено по сырым .rpt, см. publication/audit/RAW_AUDIT.md.
R_FREE_SURFACE: float = 1.0

#: клип знаменателя 1/r у оси симметрии (в долях радиуса)
R_CLIP: float = 0.05

#: (σ_rr, σ_θθ, σ_zz, τ_rz)
IDX_SRR, IDX_STT, IDX_SZZ, IDX_TRZ = 0, 1, 2, 3

EQUILIBRIUM_R1_STATUS = (
    "редуцированная форма оправдана: отброшенный ∂τ_rz/∂z составляет "
    "1.3 % (медиана) от удержанных членов, невязка меняется на 6 %")
EQUILIBRIUM_R2_STATUS = (
    "редуцированная форма НЕ оправдана. Мало того что отброшенный ∂σ_zz/∂z "
    "составляет 68 % (медиана) от удержанных членов: оставшееся выражение "
    "∂τ_rz/∂r + τ_rz/r тождественно равно (1/r)·∂(r·τ_rz)/∂r, поэтому его "
    "обнуление означает r·τ_rz = const, а с регулярностью на оси (τ_rz(0) = 0) "
    "— τ_rz ≡ 0 по всему радиусу. Это не приближение осевого равновесия, а "
    "условие «сдвига нет вовсе», данными не подтверждаемое")


def equilibrium_residual_radial(sigma_profile: np.ndarray, r_grid: np.ndarray,
                                exclude_boundary: bool = True) -> np.ndarray:
    """
    R₁(r) = ∂σ_rr/∂r + (σ_rr − σ_θθ)/r — радиальное равновесие в
    редуцированной форме, ровно как в _strong_eq_loss_torch.

    sigma_profile : (n_r, 4) МПа, физический порядок компонент
    r_grid        : (n_r,)   нормированный радиус
    """
    s_rr = sigma_profile[:, IDX_SRR]
    s_tt = sigma_profile[:, IDX_STT]
    d = np.gradient(s_rr, r_grid)
    r_safe = np.maximum(r_grid, R_CLIP)
    res = d + (s_rr - s_tt) / r_safe
    return res[1:-1] if exclude_boundary else res


def zero_shear_residual(sigma_profile: np.ndarray, r_grid: np.ndarray,
                        exclude_boundary: bool = True) -> np.ndarray:
    """
    ∂τ_rz/∂r + τ_rz/r — то, что остаётся от осевого равновесия, если выбросить
    ∂σ_zz/∂z.

    Названо по тому, что оно на самом деле утверждает. Тождественно

        ∂τ_rz/∂r + τ_rz/r ≡ (1/r) · ∂(r·τ_rz)/∂r,

    поэтому равенство нулю означает r·τ_rz = const, а с условием регулярности
    на оси (τ_rz(0) = 0) — τ_rz ≡ 0 на всём радиусе. То есть это не ослабленное
    равновесие и не регуляризатор гладкости, а жёсткое «сдвига нет».

    Данными не подтверждается: в датасете r·τ_rz вдоль радиуса непостоянно,
    медианный относительный размах 3.15. Использовать как физический лосс
    нельзя; оставлено как метрика для сопоставления с отклонённой версией,
    где это выражение стояло в L_pde.

    ⚠ В отклонённой версии из-за подмены величин (ERRATA E-18) оператор
    применялся к слоту с меткой τ_rz, в котором лежал настоящий σ_rr
    (mean|σ_rr| = 48 МПа). То есть лосс загонял в ноль по всему радиусу
    РАДИАЛЬНОЕ напряжение, которое обязано обращаться в ноль только на
    свободной поверхности.
    """
    tau = sigma_profile[:, IDX_TRZ]
    d = np.gradient(tau, r_grid)
    r_safe = np.maximum(r_grid, R_CLIP)
    res = d + tau / r_safe
    return res[1:-1] if exclude_boundary else res


def bc_violation(sigma_profile: np.ndarray, r_grid: np.ndarray,
                 r_surface: float = R_FREE_SURFACE) -> Tuple[float, float]:
    """
    |σ_rr| и |τ_rz| на свободной поверхности.

    Если r_surface попадает внутрь сетки профиля — линейная интерполяция;
    если он за её пределами (например, унаследованное 1.5) — функция
    отказывается экстраполировать и явно об этом сообщает, потому что
    сравнивать величину, полученную экстраполяцией, с величиной, полученной
    интерполяцией, нельзя.
    """
    if r_surface > r_grid.max() + 1e-12:
        raise ValueError(
            f"r_surface = {r_surface} вне сетки профиля (max r = {r_grid.max()}). "
            "Свободная поверхность соответствует r_norm = 1.0, см. E-01.")
    srr = float(np.interp(r_surface, r_grid, sigma_profile[:, IDX_SRR]))
    trz = float(np.interp(r_surface, r_grid, sigma_profile[:, IDX_TRZ]))
    return abs(srr), abs(trz)


@dataclass
class PhysicsAudit:
    n_sets: int
    bc_srr_mean: float
    bc_trz_mean: float
    r1_median: float
    shear_median: float
    r_surface: float

    def to_text(self) -> str:
        return (f"наборов: {self.n_sets}\n"
                f"  |σ_rr| на r_norm={self.r_surface:g} : {self.bc_srr_mean:8.3f} МПа\n"
                f"  |τ_rz| на r_norm={self.r_surface:g} : {self.bc_trz_mean:8.3f} МПа\n"
                f"  медиана |R₁|                : {self.r1_median:8.3f}\n"
                f"  медиана |∂τ/∂r + τ/r|       : {self.shear_median:8.3f}")


def audit(profiles: np.ndarray, r_grid: np.ndarray,
          r_surface: float = R_FREE_SURFACE) -> PhysicsAudit:
    """
    profiles : (n_sets, n_r, 4) МПа. Годится и для предсказаний модели, и для
    самих данных FEM — базовая линия «насколько источник сам нарушает ГУ»
    обязана печататься рядом, иначе цифра модели не интерпретируется.
    """
    n = profiles.shape[0]
    bc = np.array([bc_violation(profiles[i], r_grid, r_surface) for i in range(n)])
    r1 = np.array([np.median(np.abs(equilibrium_residual_radial(profiles[i], r_grid)))
                   for i in range(n)])
    sh = np.array([np.median(np.abs(zero_shear_residual(profiles[i], r_grid)))
                   for i in range(n)])
    return PhysicsAudit(n, float(bc[:, 0].mean()), float(bc[:, 1].mean()),
                        float(np.median(r1)), float(np.median(sh)), r_surface)
