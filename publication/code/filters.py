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
                   cone_mask: Optional[np.ndarray] = None,
                   ) -> Tuple[np.ndarray, FilterReport]:
    """
    Единая отбраковка. Возвращает (kept_index, report).

    kept_index — индексы уцелевших НАБОРОВ в исходной нумерации; их надо
    разворачивать в строки через splits.sets_to_rows.

    cone_mask — булева маска наборов, у которых осевое окно захватывает конус
    волоки (см. `cone_in_window`). Отсев обязан идти ДО сплита, а не при
    подсчёте метрик: у этих наборов «свободная поверхность» лежит на контакте
    с волокой, там σ_rr доходит до −571 МПа против −27 МПа у остальных, а
    |τ_rz| до 166 против 47. Попадая в ОБУЧАЮЩУЮ выборку, они ставят data-лосс
    в прямое противоречие с граничным условием traction-free, которое требует
    σ_rr(r=1) = 0. Маску считает вызывающий код: детекция требует радиуса
    поверхности по каждому осевому срезу, которого нет в одномерном датасете.
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

    if cone_mask is not None:
        m = np.asarray(cone_mask, dtype=bool)
        if m.shape[0] != n:
            raise ValueError(f"cone_mask длины {m.shape[0]}, ожидалось {n}")
        m = m & alive
        rep.removed["конус волоки внутри осевого окна"] = int(m.sum())
        if m.any():
            rep.notes.append(
                "наборы с конусом сняты ДО сплита: при отсеве только на этапе "
                "оценки модель всё равно обучается на контактном давлении под "
                "меткой «σ_rr на свободной поверхности»")
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


def unfinished_draw_in_window(r_phys_2d: np.ndarray, tol_m: float = 1e-4) -> np.ndarray:
    """
    Наборы, у которых внутри осевого окна проволока ещё не протянута до конца.

    ПРИЧИНА. Замысел расчётов — снимать кадр, когда проволока уже вышла из
    волоки, и тогда на свободной поверхности не должно быть ничего, кроме
    остаточных напряжений. Часть расчётов оборвалась раньше: на проволоке
    остался непротянутый участок исходного радиуса R₀, а опорная точка волоки
    лежит внутри осевого размаха проволоки. Прямая проверка по всем 2443
    сырым `.rpt`:

        расчётов с непротянутым хвостом (R_max ≈ R₀,
        а протянутая часть заметно меньше)          159 из 2443  (6.5 %)
        из них опорная точка волоки внутри
        размаха проволоки                           153 из 159   (96 %)

    По обжатию хвост распределён неравномерно: при rd = 0.25 таких 17.5 %
    (48 из 275) против 4.3–6.0 % на остальных уровнях. Для сравнения, у
    завершённых расчётов волока стоит за концом проволоки с зазором ~52 мм.

    ЧТО ИМЕННО ПОРТИТ ОБУЧЕНИЕ. Существен не сам факт незавершённости, а то,
    попал ли недотянутый участок в осевое окно 25–75 %, из которого берутся
    данные. Признак — радиус свободной поверхности МЕНЯЕТСЯ вдоль окна; у
    полностью протянутого участка он постоянен. На датасете из 1848 наборов:

        | группа                        |   n  | min σ_rr(пов.) | max \|τ_rz\| |
        |-------------------------------|------|----------------|-------------|
        | хвост И внутри окна           |   63 |   −571.3 МПа   |  165.8 МПа  |
        | внутри окна (хвост вне)       |   26 |   −102.9 МПа   |  132.5 МПа  |
        | хвост, но окно чистое         |    3 |    −27.3 МПа   |   36.3 МПа  |
        | чистые                        | 1756 |    −19.7 МПа   |   47.5 МПа  |

    Три набора с хвостом вне окна безвредны: данные в окне у них нормальные.
    Поэтому маска строится по окну (89 наборов), а не по факту хвоста — но для
    публикации защитимее снимать объединение (92): незавершённый расчёт не
    является корректной точкой выборки независимо от положения окна.

    Пороги разброса радиуса поверхности вдоль окна:

        > 0.1 мм → 89 наборов (4.8 %)
        > 0.5 мм → 51 набор  (2.8 %)
        > 1.0 мм → 27 наборов (1.5 %)
        медиана разброса 0.0026 мм, максимум 4.003 мм

    Отсев обязан идти ДО сплита: из 89 загрязнённых 73 попадают в ОБУЧЕНИЕ и
    лишь 16 в тест, а в обучении они ставят data-лосс в прямое противоречие с
    граничным условием traction-free, требующим σ_rr(r = 1) = 0.

    Параметры
    ---------
    r_phys_2d : (n_sets, n_z, n_r), м — физический радиус узлов сетки
    tol_m     : порог разброса радиуса поверхности вдоль окна, м

    Возвращает булеву маску (n_sets,): True — набор непригоден.
    """
    r_surface = np.asarray(r_phys_2d)[:, :, -1]
    spread = r_surface.max(axis=1) - r_surface.min(axis=1)
    return spread > tol_m


#: прежнее имя. Оно называло признак («конус волоки»), а не причину:
#: волока в этих расчётах не обязана касаться окна — проволока просто не
#: успела пройти её целиком. Оставлено, чтобы не ломать вызовы.
cone_in_window = unfinished_draw_in_window
