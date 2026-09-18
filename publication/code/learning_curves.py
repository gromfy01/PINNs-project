"""
learning_curves.py — кривые обучения по объёму данных (пункт 5 плана).

Тезис, ради которого всё делается: физприор выигрывает в режиме дефицита
данных. Проверяется обучением на 10 / 25 / 50 / 100 % train-пула.

Два условия, без которых сравнение разваливается:
  1. подвыборки берутся ЦЕЛЫМИ наборами (group-aware) — иначе 20 радиальных
     точек одного набора расползаются, и «10 % данных» на деле остаются
     полным покрытием пространства параметров;
  2. подвыборки ВЛОЖЕНЫ и ОДИНАКОВЫ для всех шести моделей (общий seed) —
     иначе разница между моделями смешивается с разницей выборок.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

DEFAULT_FRACTIONS: tuple = (0.10, 0.25, 0.50, 1.00)


def nested_subsamples(train_sets: Sequence[int],
                      fractions: Sequence[float] = DEFAULT_FRACTIONS,
                      seed: int = 42) -> Dict[float, np.ndarray]:
    """
    Вложенные group-aware подвыборки train-пула.

    Вложенность (10 % ⊂ 25 % ⊂ 50 % ⊂ 100 %) обязательна: иначе немонотонность
    кривой нельзя отличить от смены состава выборки.
    """
    sets = np.asarray(sorted(train_sets), dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sets)                # один порядок для всех долей
    out: Dict[float, np.ndarray] = {}
    for f in sorted(fractions):
        n = max(1, int(round(len(perm) * f)))
        out[float(f)] = np.sort(perm[:n])
    for a, b in zip(sorted(out)[:-1], sorted(out)[1:]):
        assert np.isin(out[a], out[b]).all(), "подвыборки должны быть вложенными"
    return out


def stratified_nested_subsamples(train_sets: Sequence[int], proc: np.ndarray,
                                 axis_col: int,
                                 fractions: Sequence[float] = DEFAULT_FRACTIONS,
                                 seed: int = 42) -> Dict[float, np.ndarray]:
    """
    То же, но с сохранением пропорций уровней одного фактора. Нужно для оси v:
    уровень v = 5 м/мин представлен втрое реже остальных, при случайной
    подвыборке 10 % он может исчезнуть целиком.
    """
    sets = np.asarray(sorted(train_sets), dtype=int)
    rng = np.random.default_rng(seed)
    levels = np.unique(proc[sets, axis_col])
    per_level = {lv: rng.permutation(sets[np.isclose(proc[sets, axis_col], lv)])
                 for lv in levels}
    out: Dict[float, np.ndarray] = {}
    for f in sorted(fractions):
        take = [per_level[lv][: max(1, int(round(len(per_level[lv]) * f)))]
                for lv in levels]
        out[float(f)] = np.sort(np.concatenate(take))
    for a, b in zip(sorted(out)[:-1], sorted(out)[1:]):
        assert np.isin(out[a], out[b]).all(), "подвыборки должны быть вложенными"
    return out


def crossover_fraction(curves: Dict[str, Dict[float, float]],
                       physics_models: Sequence[str],
                       data_models: Sequence[str]) -> float:
    """
    Доля данных, начиная с которой лучшая датадривен-модель догоняет лучшую
    physics-informed. Возвращает nan, если пересечения нет в исследованном
    диапазоне. Это число — прямой ответ на вопрос «где именно физприор
    перестаёт помогать», и его надо назвать в статье явно.
    """
    fracs = sorted(next(iter(curves.values())).keys())
    for f in fracs:
        best_phys = min(curves[m][f] for m in physics_models if m in curves)
        best_data = min(curves[m][f] for m in data_models if m in curves)
        if best_data <= best_phys:
            return float(f)
    return float("nan")
