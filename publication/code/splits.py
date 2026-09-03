"""
splits.py — экстраполяционные сплиты и контроли к ним.

Зачем. Случайный hold-out из того же пространства параметров показывает
интерполяцию, а не обобщение (Reviewer #1). Здесь из обучения выносится
целая ГРАНЬ факторного пространства, и качество меряется вне диапазона
обучения.

Чем это отличается от «просто взять край». Голая цифра «RMSE вырос вдвое»
не интерпретируется: рост может объясняться тем, что регион малочисленнее
или физически тяжелее. Поэтому у каждого экстраполяционного сплита есть два
контроля:

    matched:<region>  — случайный набор ТОГО ЖЕ размера из всего пула;
                        отчётная величина ΔRMSE = RMSE(extrap) − RMSE(matched)
    <axis>_mid        — внутренний срез по той же оси того же порядка размера

Защита от самообмана — coverage_report(): по каждой оси печатает диапазоны
train и test и колонку «вне диапазона?». Заявлять обобщение можно только по
осям, где стоит ДА.

μ как ось экстраполяции не используется: 13.2 % наборов сырой сетки — точные
копии другого набора, отличающегося только меткой μ (publication/audit/),
обобщать не по чему.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_R_DEFAULT = 20
FEATURES = ("Q", "k", "alpha", "mu", "v")
COL = {n: i for i, n in enumerate(FEATURES)}


def sets_to_rows(set_indices: Sequence[int], n_r: int = N_R_DEFAULT) -> np.ndarray:
    """Индексы наборов → индексы строк (раскладка set-major, r подряд)."""
    s = np.asarray(set_indices, dtype=int)
    if s.size == 0:
        return np.array([], dtype=int)
    return (s[:, None] * n_r + np.arange(n_r)[None, :]).reshape(-1)


# ────────────────────────── реестр регионов ─────────────────────────────

@dataclass(frozen=True)
class Region:
    name: str
    predicate: Callable[[np.ndarray], np.ndarray]
    axes: Tuple[str, ...]
    kind: str          # 'extrap' | 'extrap_joint' | 'interp' | 'random'
    comment: str = ""


def _eq(col: str, value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda proc: np.isclose(proc[:, COL[col]], value)


REGIONS: Dict[str, Region] = {
    "alpha_max": Region("alpha_max", _eq("alpha", 20.0), ("alpha",), "extrap",
                        "максимальный полуугол волоки — основная ось статьи"),
    "alpha_mid": Region("alpha_mid", _eq("alpha", 12.0), ("alpha",), "interp",
                        "контроль: срез внутри диапазона обучения"),
    "Q_high":    Region("Q_high", _eq("Q", 0.20), ("Q",), "extrap",
                        "максимальное обжатие после снятия Q = 0.25"),
    "Q_mid":     Region("Q_mid", _eq("Q", 0.10), ("Q",), "interp", "контроль по оси Q"),
    "k_max":     Region("k_max", _eq("k", 1.0), ("k",), "extrap", "длиннейший калибрующий поясок"),
    "v_max":     Region("v_max", _eq("v", 250.0), ("v",), "extrap", "максимальная скорость"),
    "corner":    Region("corner",
                        lambda proc: np.isclose(proc[:, COL["alpha"]], 20.0)
                        & (proc[:, COL["Q"]] >= 0.15 - 1e-12),
                        ("alpha", "Q"), "extrap_joint",
                        "дыра в СОВМЕСТНОМ пространстве: по каждой оси в "
                        "отдельности test остаётся внутри диапазона train"),
}


@dataclass
class Split:
    name: str
    kind: str
    train_sets: np.ndarray
    test_sets: np.ndarray
    axes: Tuple[str, ...]
    seed: int
    comment: str = ""

    def rows(self, n_r: int = N_R_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
        return sets_to_rows(self.train_sets, n_r), sets_to_rows(self.test_sets, n_r)

    def check(self) -> None:
        assert np.intersect1d(self.train_sets, self.test_sets).size == 0, \
            f"{self.name}: train и test пересекаются — утечка"
        assert self.test_sets.size > 0, f"{self.name}: пустой test"
        assert self.train_sets.size > 0, f"{self.name}: пустой train"


def make_region_split(proc: np.ndarray, region: str, seed: int = 42) -> Split:
    """
    Вынести регион целиком в test, всё остальное — train.

    Сплит с kind='extrap' обязан выходить за диапазон train хотя бы по одной
    оси — иначе он экстраполяционный только по названию, и check() это ловит.
    Для `corner` это заведомо не так (по каждой оси в отдельности test внутри
    диапазона), поэтому у него отдельный kind='extrap_joint': это дыра в
    совместном пространстве, и заявлять по ней «выход за диапазон обучения»
    нельзя.
    """
    reg = REGIONS[region]
    m = reg.predicate(proc)
    test = np.flatnonzero(m)
    train = np.flatnonzero(~m)
    sp = Split(f"{reg.kind}:{region}", reg.kind, train, test, reg.axes, seed, reg.comment)
    sp.check()
    if reg.kind == "extrap":
        cov = coverage_report(proc, sp)
        outside = [c["axis"] for c in cov if c["outside_range"]]
        assert outside, (
            f"{sp.name}: объявлен экстраполяционным, но test не выходит за "
            "диапазон train ни по одной оси — это интерполяция")
    return sp


def make_matched_split(proc: np.ndarray, n_test: int, seed: int = 42,
                       name: str = "matched") -> Split:
    """Контроль: случайные наборы того же объёма из всего пула."""
    n = proc.shape[0]
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(n, size=min(n_test, n - 1), replace=False))
    train = np.setdiff1d(np.arange(n), test)
    sp = Split(name, "random", train, test, (), seed, "контроль того же объёма")
    sp.check()
    return sp


def make_random_split(proc: np.ndarray, frac: float = 0.15, seed: int = 42) -> Split:
    """Исходный сплит статьи — случайный hold-out (интерполяция)."""
    n = proc.shape[0]
    return make_matched_split(proc, int(round(n * frac)), seed=seed, name="random")


def build_split_suite(proc: np.ndarray, seed: int = 42,
                      regions: Optional[Sequence[str]] = None) -> Dict[str, Split]:
    """
    Полный набор сплитов статьи: регионы + matched-контроль к каждому
    экстраполяционному региону + исходный случайный сплит.
    """
    regions = list(REGIONS) if regions is None else list(regions)
    out: Dict[str, Split] = {}
    for r in regions:
        sp = make_region_split(proc, r, seed=seed)
        out[sp.name] = sp
        if REGIONS[r].kind in ("extrap", "extrap_joint"):
            m = make_matched_split(proc, sp.test_sets.size, seed=seed,
                                   name=f"matched:{r}")
            out[m.name] = m
    rs = make_random_split(proc, seed=seed)
    out[rs.name] = rs
    return out


# ───────────────────────── отчёт о покрытии ─────────────────────────────

def coverage_report(proc: np.ndarray, split: Split, atol: float = 1e-9):
    """
    По каждой оси: диапазон и уровни в train, в test, и вышел ли test
    за диапазон train. Возвращает список словарей (годится для DataFrame).

    Заявлять «обобщение за пределы диапазона обучения» можно ТОЛЬКО по осям,
    где outside=True. Если outside=False везде — это интерполяционный сплит,
    как его ни назови.
    """
    tr, te = proc[split.train_sets], proc[split.test_sets]
    rows = []
    for name in FEATURES:
        c = COL[name]
        tr_lo, tr_hi = float(tr[:, c].min()), float(tr[:, c].max())
        te_lo, te_hi = float(te[:, c].min()), float(te[:, c].max())
        tr_lv = np.unique(np.round(tr[:, c], 12))
        te_lv = np.unique(np.round(te[:, c], 12))
        new_levels = np.setdiff1d(te_lv, tr_lv)
        rows.append({
            "axis": name,
            "train_min": tr_lo, "train_max": tr_hi, "train_levels": len(tr_lv),
            "test_min": te_lo, "test_max": te_hi, "test_levels": len(te_lv),
            "outside_range": bool(te_lo < tr_lo - atol or te_hi > tr_hi + atol),
            "unseen_levels": new_levels.tolist(),
        })
    return rows


def format_coverage(rows) -> str:
    head = f"{'ось':<7}{'train':<22}{'test':<22}{'вне диапазона?':<16}{'новые уровни'}"
    out = [head, "─" * len(head)]
    for r in rows:
        tr = f"[{r['train_min']:g}, {r['train_max']:g}] ({r['train_levels']})"
        te = f"[{r['test_min']:g}, {r['test_max']:g}] ({r['test_levels']})"
        flag = "ДА" if r["outside_range"] else "нет"
        out.append(f"{r['axis']:<7}{tr:<22}{te:<22}{flag:<16}"
                   f"{','.join(f'{x:g}' for x in r['unseen_levels']) or '—'}")
    return "\n".join(out)


def summarize_suite(proc: np.ndarray, suite: Dict[str, Split]) -> List[Dict[str, object]]:
    rows = []
    for name, sp in suite.items():
        cov = coverage_report(proc, sp)
        outside = [c["axis"] for c in cov if c["outside_range"]]
        if outside:
            label = ",".join(outside)
        elif sp.kind == "extrap_joint":
            label = "— (дыра в совместном пространстве)"
        else:
            label = "—"
        rows.append({"split": name, "kind": sp.kind,
                     "n_train": int(sp.train_sets.size), "n_test": int(sp.test_sets.size),
                     "extrapolated_axes": label})
    return rows
