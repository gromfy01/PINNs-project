"""
corruption.py — устойчивость к дефектам источника данных (пункт 6 плана).

Наблюдение из Discussion отклонённой версии: на наборах с нефизичным
решением солвера MLP воспроизводит аномалию, а PINN — нет. Здесь оно
превращается в контролируемый эксперимент.

Как портим. НЕ гауссовым шумом: шум усредняется по 20 точкам профиля и по
батчу, и любая модель его подавит — эксперимент выйдет пустым. Портим так,
как ломается сам источник: постоянным сдвигом одной компоненты на целом
наборе. Это ровно морфология артефакта #439 (σ_θθ ≈ +255 МПа константой при
σ_θθ_mean ∈ [+1.6, +7.9] у соседей по группе (Q, k, α)).

Обязательные условия:
  * портятся только TRAIN-наборы; hold-out остаётся чистым, иначе
    измеряется не устойчивость, а согласие с порчей;
  * набор портится целиком (все 20 радиальных точек) — построчная порча
    физически невозможна и слишком легко детектируется;
  * список испорченных наборов вложен по долям и общий для всех шести
    моделей.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RATES: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)

#: (σ_rr, σ_θθ, σ_zz, τ_rz)
COMPONENT_NAMES = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")

#: сдвиг, воспроизводящий морфологию артефакта #439
DEFAULT_OFFSET_MPA: float = 250.0
DEFAULT_COMPONENT: int = 1          # σ_θθ


@dataclass
class CorruptionPlan:
    rate: float
    component: int
    offset_mpa: float
    corrupted_sets: np.ndarray
    seed: int
    mode: str = "constant_offset"

    def describe(self) -> str:
        return (f"{self.mode}: {len(self.corrupted_sets)} наборов "
                f"({100 * self.rate:.0f} %), компонента "
                f"{COMPONENT_NAMES[self.component]}, сдвиг {self.offset_mpa:+.0f} МПа, "
                f"seed={self.seed}")


def make_corruption_plans(train_sets: Sequence[int],
                          rates: Sequence[float] = DEFAULT_RATES,
                          component: int = DEFAULT_COMPONENT,
                          offset_mpa: float = DEFAULT_OFFSET_MPA,
                          seed: int = 42,
                          mode: str = "constant_offset") -> Dict[float, CorruptionPlan]:
    """Вложенные планы порчи: 1 % ⊂ 5 % ⊂ 10 %, общие для всех моделей."""
    sets = np.asarray(sorted(train_sets), dtype=int)
    perm = np.random.default_rng(seed).permutation(sets)
    plans: Dict[float, CorruptionPlan] = {}
    for r in sorted(rates):
        n = int(round(len(perm) * r))
        plans[float(r)] = CorruptionPlan(float(r), component, offset_mpa,
                                         np.sort(perm[:n]), seed, mode)
    for a, b in zip(sorted(plans)[:-1], sorted(plans)[1:]):
        assert np.isin(plans[a].corrupted_sets, plans[b].corrupted_sets).all()
    return plans


def apply_corruption(y_sets: np.ndarray, plan: CorruptionPlan,
                     set_index: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    y_sets : (n_sets, n_r, 4) — выходы, разложенные по наборам, МПа.
    set_index : нумерация наборов в y_sets, если она не 0..n-1.

    Возвращает КОПИЮ с внесённой порчей. Исходный массив не трогается.
    """
    y = np.array(y_sets, copy=True)
    idx = np.arange(y.shape[0]) if set_index is None else np.asarray(set_index)
    pos = np.flatnonzero(np.isin(idx, plan.corrupted_sets))
    if plan.mode == "constant_offset":
        y[pos, :, plan.component] += plan.offset_mpa
    elif plan.mode == "constant_replace":
        y[pos, :, plan.component] = plan.offset_mpa
    elif plan.mode == "gaussian":
        rng = np.random.default_rng(plan.seed + 1)
        y[pos, :, plan.component] += rng.normal(0.0, plan.offset_mpa, size=y[pos].shape[:2])
    else:
        raise ValueError(plan.mode)
    return y


def degradation_table(results: Dict[str, Dict[float, float]]) -> List[Dict[str, object]]:
    """
    results: {model: {rate: rmse_на_чистом_holdout}}
    Возвращает наклон деградации — во сколько раз выросла RMSE при 10 %
    порчи относительно 0 %. Это и есть сравниваемая величина между семействами.
    """
    out = []
    for m, curve in results.items():
        base = curve[min(curve)]
        worst = curve[max(curve)]
        out.append({"model": m, "rmse_clean": base, "rmse_worst": worst,
                    "ratio": worst / base if base else float("nan"),
                    "delta": worst - base})
    return sorted(out, key=lambda r: r["ratio"])
