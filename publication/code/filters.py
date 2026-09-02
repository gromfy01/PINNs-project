"""
filters.py — отбраковка наборов ДО сплита (конвенция §6.3.6 старого handover).

Все фильтры работают на уровне НАБОРОВ (наборов параметров), а не строк:
один набор — это 20 радиальных точек, они всегда живут и умирают вместе.

Порядок фильтров зафиксирован и важен, потому что счётчики в отчёте
последовательные (каждый следующий фильтр считает только то, что уцелело):

    1. явные FEM-артефакты по списку индексов
    2. режим, выходящий за постановку задачи (Q = 0.25 — упругая разгрузка)
    3. значения фактора вне проектной сетки (μ = 0.002 — опечатка имени задания)
    4. дубликаты по μ (побитово совпавший выход при совпавших (Q, k, α, v))

Ничего не «чинится» молча: filter_dataset возвращает и маску, и отчёт,
который печатается в статью.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: проектная сетка коэффициента трения (три уровня, см. ERRATA E-03)
MU_GRID: Tuple[float, ...] = (0.025, 0.05, 0.10)

#: режим обжатия, исключаемый из постановки (упругая разгрузка)
Q_EXCLUDED: Tuple[float, ...] = (0.25,)

#: наборы, забракованные вручную в launch/stress/*.ipynb
#: (артефакт сходимости #439 + шесть «billet adhered to the die»).
#: ⚠ Это индексы в нумерации `.pkl` (2035 наборов). К сырой сетке 2443 они
#: неприменимы: там другой порядок. При работе от сырых `.rpt` передавать
#: drop_removed_indices=False. Самопроверка ниже (все семь должны иметь
#: Q = 0.25) ловит применение списка не к тому массиву.
REMOVED_SET_INDICES: Tuple[int, ...] = (78, 198, 204, 207, 210, 439, 449)

FEATURES = ("Q", "k", "alpha", "mu", "v")
COL = {n: i for i, n in enumerate(FEATURES)}


@dataclass
class FilterReport:
    n_in: int
    n_out: int
    removed: Dict[str, int] = field(default_factory=dict)
    kept_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    notes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        w = max(len(k) for k in self.removed) if self.removed else 1
        lines = [f"наборов на входе : {self.n_in}"]
        for k, v in self.removed.items():
            lines.append(f"  −{v:<5d} {k:<{w}s}")
        lines.append(f"наборов на выходе: {self.n_out}")
        lines.extend("  ⚠ " + n for n in self.notes)
        return "\n".join(lines)


def _mu_on_grid(mu: np.ndarray, grid: Sequence[float], atol: float = 1e-6) -> np.ndarray:
    g = np.asarray(grid, dtype=float)[None, :]
    return np.any(np.abs(np.asarray(mu, dtype=float)[:, None] - g) <= atol, axis=1)


def find_mu_duplicates(proc: np.ndarray, y_sets: np.ndarray,
                       rtol: float = 0.0, atol: float = 0.0) -> np.ndarray:
    """
    Индексы наборов-дубликатов по μ.

    Дубликат — набор, у которого нашёлся другой набор с теми же (Q, k, α, v),
    другим μ и совпавшим (с точностью rtol/atol; по умолчанию — побитово)
    выходом. Из каждой группы совпавших оставляется набор с наименьшим
    индексом, остальные помечаются на удаление.

    proc   : (n_sets, 5)          параметры процесса
    y_sets : (n_sets, n_r, n_c)   выходы, разложенные по наборам
    """
    n = proc.shape[0]
    key_cols = [COL["Q"], COL["k"], COL["alpha"], COL["v"]]
    keys = {}
    for i in range(n):
        keys.setdefault(tuple(np.round(proc[i, key_cols], 12)), []).append(i)

    drop: List[int] = []
    flat = y_sets.reshape(n, -1)
    for _k, idx in keys.items():
        if len(idx) < 2:
            continue
        kept: List[int] = []
        for i in idx:
            dup_of = None
            for j in kept:
                if np.allclose(flat[i], flat[j], rtol=rtol, atol=atol, equal_nan=True):
                    dup_of = j
                    break
            if dup_of is None:
                kept.append(i)
            else:
                drop.append(i)
    return np.array(sorted(drop), dtype=int)


def filter_dataset(proc: np.ndarray,
                   y_sets: np.ndarray,
                   *,
                   drop_removed_indices: bool = True,
                   removed_indices: Sequence[int] = REMOVED_SET_INDICES,
                   drop_q: Sequence[float] = Q_EXCLUDED,
                   drop_mu_offgrid: bool = True,
                   mu_grid: Sequence[float] = MU_GRID,
                   drop_mu_duplicates: bool = True,
                   ) -> Tuple[np.ndarray, FilterReport]:
    """
    Единая отбраковка. Возвращает (kept_index, report).

    kept_index — индексы уцелевших НАБОРОВ в исходной нумерации; их надо
    разворачивать в строки через splits.sets_to_rows.
    """
    n = proc.shape[0]
    alive = np.ones(n, dtype=bool)
    rep = FilterReport(n_in=n, n_out=0)

    if drop_removed_indices:
        m = np.zeros(n, dtype=bool)
        for i in removed_indices:
            if 0 <= i < n:
                m[i] = True
        m &= alive
        rep.removed["FEM-артефакты (список launch/stress)"] = int(m.sum())
        if m.any() and not np.all(np.isclose(proc[m, COL["Q"]], 0.25)):
            rep.notes.append(
                "список REMOVED_SET_INDICES применён к массиву, где не все "
                "семь наборов имеют Q = 0.25 — почти наверняка это не та "
                "нумерация (список задан в порядке .pkl); проверить перед "
                "тем, как доверять результату")
        alive &= ~m

    if drop_q:
        m = np.zeros(n, dtype=bool)
        for q in drop_q:
            m |= np.isclose(proc[:, COL["Q"]], q)
        m &= alive
        rep.removed[f"Q ∈ {tuple(drop_q)} (упругий режим)"] = int(m.sum())
        alive &= ~m

    if drop_mu_offgrid:
        m = ~_mu_on_grid(proc[:, COL["mu"]], mu_grid)
        m &= alive
        rep.removed[f"μ вне сетки {tuple(mu_grid)}"] = int(m.sum())
        if m.sum():
            bad = np.unique(proc[m, COL["mu"]])
            rep.notes.append(f"μ вне сетки: {bad.tolist()} — опечатка в имени задания, не уровень фактора")
        alive &= ~m

    if drop_mu_duplicates:
        idx_alive = np.flatnonzero(alive)
        dup_local = find_mu_duplicates(proc[idx_alive], y_sets[idx_alive])
        m = np.zeros(n, dtype=bool)
        m[idx_alive[dup_local]] = True
        rep.removed["μ-дубликаты (побитово совпавший выход)"] = int(m.sum())
        if alive.sum():
            rep.notes.append(
                f"доля μ-дубликатов среди уцелевших до этого шага: "
                f"{100.0 * m.sum() / alive.sum():.1f} %")
        alive &= ~m

    rep.kept_index = np.flatnonzero(alive)
    rep.n_out = int(alive.sum())
    return rep.kept_index, rep


def grid_coverage(proc: np.ndarray, levels: Optional[Dict[str, Sequence[float]]] = None
                  ) -> Dict[str, object]:
    """
    Покрытие полного декартова произведения уровней факторов.
    Возвращает число уникальных комбинаций, размер полной сетки и долю.
    """
    if levels is None:
        levels = {n: np.unique(proc[:, COL[n]]) for n in FEATURES}
    full = 1
    for n in FEATURES:
        full *= len(levels[n])
    combos = {tuple(np.round(row, 12)) for row in proc}
    return {"n_unique_combinations": len(combos),
            "full_factorial": full,
            "coverage": len(combos) / full if full else float("nan"),
            "levels": {n: np.asarray(levels[n]).tolist() for n in FEATURES}}


def cone_in_window(r_phys_2d: np.ndarray, tol_m: float = 1e-4) -> np.ndarray:
    """
    Наборы, у которых осевое окно 25–75 % захватывает конус волоки.

    Признак: радиус свободной поверхности МЕНЯЕТСЯ вдоль окна. У протянутого
    по всей длине участка он постоянен; если внутри окна остался переходный
    конус, поверхностные узлы там лежат на контакте с волокой, и то, что
    датасет подаёт как «σ_rr на свободной поверхности», на деле является
    контактным давлением — величина другого физического смысла и другого
    порядка.

    Измерено на stress2d.npz (1848 наборов, сетка 8 × 20):

        разброс R по окну > 0.1 мм   →   89 наборов (4.8 %)
        разброс R по окну > 0.5 мм   →   51 набора (2.8 %)
        разброс R по окну > 1.0 мм   →   27 наборов (1.5 %)
        медиана разброса 0.0026 мм, максимум 4.003 мм

    Разделение чистое, а не градиентное:

        σ_rr на поверхности, наборы с конусом (n = 89):  минимум −571.3 МПа,
                                             ячеек < −50 МПа: 65 из 712
        σ_rr на поверхности, остальные (n = 1759):       минимум  −27.3 МПа,
                                             ячеек < −50 МПа:  0 из 14072

    То есть у 1759 «чистых» наборов traction-free не нарушается нигде, а все
    патологические значения сосредоточены в 89 помеченных. В одномерном треке
    осреднение вдоль z это размывает (набор с |σ_rr(пов.)| > 20 МПа всего один
    из 1848), поэтому маска актуальна прежде всего для 2D.

    Параметры
    ---------
    r_phys_2d : (n_sets, n_z, n_r), м — физический радиус узлов сетки
    tol_m     : порог разброса радиуса поверхности вдоль окна, м

    Возвращает булеву маску (n_sets,): True — конус внутри окна.
    """
    r_surface = np.asarray(r_phys_2d)[:, :, -1]
    spread = r_surface.max(axis=1) - r_surface.min(axis=1)
    return spread > tol_m
