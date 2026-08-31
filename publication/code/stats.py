"""
stats.py — статистика сравнения моделей: по сидам, внутри сплита.

Две ошибки отклонённой версии, которые здесь чинятся.

1. Friedman–Nemenyi считался по 304 наборам hold-out. Это разброс ВЫБОРКИ,
   а не разброс обучения: он отвечает на вопрос «на скольких наборах модель A
   лучше B», а не «воспроизводится ли преимущество A при перезапуске». Именно
   это уловил Reviewer #2. Здесь блоками теста служат СИДЫ (и, отдельно,
   сплиты — но никогда не смешанные пулы).

2. Разница 0.4–0.8 МПа объявлялась содержательной без оценки межсидового
   разброса. significance_gate() не даёт этого сделать: любая пара, у которой
   |Δ| меньше объединённого σ по сидам, помечается «не установлено».

Отдельно: агрегировать сплиты в один пул запрещено — pool_guard() падает.
Ранги внутри `extrap:alpha_max` и внутри `random` описывают разные задачи,
их среднее не значит ничего.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: критические значения студентизированного размаха / sqrt(2) для теста
#: Немени, k = число сравниваемых моделей (индекс = k)
_Q_ALPHA: Dict[float, Dict[int, float]] = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
           7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
           7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
}


def pool_guard(split_names: Iterable[str]) -> None:
    """Запрет на объединение сплитов в один пул для рангового теста."""
    names = list(dict.fromkeys(split_names))
    if len(names) > 1:
        raise ValueError(
            "Friedman/Nemenyi считается ВНУТРИ сплита. Получено несколько: "
            f"{names}. Ранги в разных сплитах описывают разные задачи, "
            "объединять их нельзя (см. §10 playbook, п. 4).")


# ───────────────────────── сводка по сидам ──────────────────────────────

@dataclass
class SeedSummary:
    models: List[str]
    seeds: List[int]
    values: np.ndarray          # (n_models, n_seeds), метрика (меньше — лучше)
    metric: str = "macro RMSE, МПа"
    split: str = ""

    @property
    def mean(self) -> np.ndarray:
        return self.values.mean(axis=1)

    @property
    def std(self) -> np.ndarray:
        return self.values.std(axis=1, ddof=1) if self.values.shape[1] > 1 \
            else np.zeros(self.values.shape[0])

    def table(self) -> List[Dict[str, object]]:
        order = np.argsort(self.mean)
        return [{"model": self.models[i], "mean": float(self.mean[i]),
                 "std": float(self.std[i]), "n_seeds": self.values.shape[1],
                 "split": self.split} for i in order]

    def to_text(self) -> str:
        head = f"{'модель':<14}{'mean ± std':<20}{'сидов':<7}"
        out = [f"сплит: {self.split}   метрика: {self.metric}", head, "─" * len(head)]
        for r in self.table():
            out.append(f"{r['model']:<14}{r['mean']:8.3f} ± {r['std']:<8.3f}{r['n_seeds']:<7}")
        return "\n".join(out)


def build_seed_summary(records: Sequence[Dict[str, object]],
                       metric_key: str = "macro_rmse") -> SeedSummary:
    """
    records: [{'model':..., 'seed':..., 'split':..., metric_key: float}, ...]
    Все записи должны относиться к одному сплиту (pool_guard).
    """
    pool_guard(str(r["split"]) for r in records)
    split = str(records[0]["split"]) if records else ""
    models = sorted({str(r["model"]) for r in records})
    seeds = sorted({int(r["seed"]) for r in records})
    vals = np.full((len(models), len(seeds)), np.nan)
    for r in records:
        vals[models.index(str(r["model"])), seeds.index(int(r["seed"]))] = float(r[metric_key])
    if np.isnan(vals).any():
        missing = [(models[i], seeds[j]) for i, j in zip(*np.where(np.isnan(vals)))]
        raise ValueError(f"неполная матрица модель×сид, не хватает: {missing[:8]}")
    return SeedSummary(models, seeds, vals, split=split)


# ───────────────────── порог содержательности разницы ───────────────────

MIN_SEEDS_FOR_TEST = 3


@dataclass
class PairVerdict:
    a: str
    b: str
    delta: float            # mean(a) − mean(b)
    pooled_std: float
    n_seeds: int
    p_value: float
    established: bool       # статистически: парный тест по сидам
    practical: bool         # практически: |Δ| ≥ k·σ по сидам

    def __str__(self) -> str:
        if self.n_seeds < MIN_SEEDS_FOR_TEST:
            return (f"{self.a} − {self.b}: Δ = {self.delta:+.3f} МПа, сидов "
                    f"{self.n_seeds} — тест невозможен, разница НЕ установлена")
        mark = "установлена" if self.established else "НЕ установлена"
        return (f"{self.a} − {self.b}: Δ = {self.delta:+.3f} МПа, p = {self.p_value:.3f}, "
                f"σ по сидам = {self.pooled_std:.3f} → разница {mark}"
                + ("" if self.practical else "; и |Δ| меньше σ по сидам"))


def significance_gate(summary: SeedSummary, k_sigma: float = 1.0,
                      alpha: float = 0.05) -> List[PairVerdict]:
    """
    Разница между моделями считается установленной, только если её
    подтверждает ПАРНЫЙ тест по сидам (один и тот же сид — один блок).

    Порог |Δ| ≥ k_sigma·σ_pooled оставлен отдельной величиной: он говорит о
    практическом размере эффекта, но тестом не является. В прежней версии он
    служил единственным критерием, и это давало «разница установлена» при
    p ≈ 0.12, а при единственном сиде (σ = 0) — вообще всегда, включая Δ = 0.
    Ниже трёх сидов вердикт всегда «не установлена»: это согласовано с
    friedman_test, который при том же условии отказывается считать.
    """
    from scipy.stats import ttest_rel
    out: List[PairVerdict] = []
    m, s = summary.mean, summary.std
    n = summary.values.shape[1]
    for i, j in combinations(range(len(summary.models)), 2):
        pooled = float(np.sqrt(0.5 * (s[i] ** 2 + s[j] ** 2)))
        delta = float(m[i] - m[j])
        if n >= MIN_SEEDS_FOR_TEST:
            res = ttest_rel(summary.values[i], summary.values[j])
            p = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
            est = p < alpha
        else:
            p, est = float("nan"), False
        out.append(PairVerdict(summary.models[i], summary.models[j], delta, pooled,
                               n, p, est, abs(delta) >= k_sigma * pooled))
    return out


# ──────────────────────── Friedman + Nemenyi ────────────────────────────

def ranks_per_block(values: np.ndarray) -> np.ndarray:
    """
    values: (n_treatments, n_blocks), меньше — лучше.
    Возвращает ранги (n_treatments, n_blocks), 1 = лучший, средние при связях.
    """
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 0, values)


def friedman_test(values: np.ndarray) -> Dict[str, float]:
    """
    Тест Фридмана. values: (n_treatments, n_blocks). Блоки — СИДЫ.
    """
    from scipy.stats import friedmanchisquare
    k, n = values.shape
    if k < 3:
        raise ValueError("тест Фридмана требует ≥3 моделей")
    if n < 3:
        raise ValueError(f"блоков (сидов) всего {n}: тест бессмысленен, нужно ≥3; "
                         "это и есть претензия Reviewer #2 — один прогон не тест")
    stat, p = friedmanchisquare(*[values[i] for i in range(k)])
    return {"statistic": float(stat), "p_value": float(p), "k": k, "n_blocks": n}


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Критическая разность средних рангов (post-hoc Немени)."""
    if alpha not in _Q_ALPHA:
        raise ValueError(f"alpha должна быть из {sorted(_Q_ALPHA)}")
    if k not in _Q_ALPHA[alpha]:
        raise ValueError(f"k = {k} вне таблицы (2..10)")
    q = _Q_ALPHA[alpha][k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


@dataclass
class NemenyiResult:
    models: List[str]
    mean_ranks: np.ndarray
    cd: float
    alpha: float
    n_blocks: int
    friedman: Dict[str, float]
    split: str = ""

    def cliques(self) -> List[List[str]]:
        """
        Максимальные группы моделей, ПОПАРНО неразличимых на уровне CD.

        Считается скользящим окном по отсортированным средним рангам: группа
        {i..j} допустима, только если разность рангов её крайних членов не
        превышает CD — тогда неразличимы все пары внутри. Прежняя версия
        собирала CD-окрестности каждой модели, и в них попадали пары с
        разностью рангов больше CD: например при рангах 1.2 / 2.0 / 2.8 и
        CD = 1.48 печаталась одна группа из трёх, хотя крайняя пара (Δ = 1.6)
        различима. Это ровно та ошибка, которая помечена в ERRATA E-15 как
        дефект аннотации отклонённой версии.
        """
        order = list(np.argsort(self.mean_ranks))
        groups: List[List[int]] = []
        for a in range(len(order)):
            b = a
            while (b + 1 < len(order) and
                   self.mean_ranks[order[b + 1]] - self.mean_ranks[order[a]] <= self.cd + 1e-12):
                b += 1
            groups.append(order[a:b + 1])
        maximal = [g for g in groups
                   if not any(set(g) < set(h) for h in groups)]
        seen, out = set(), []
        for g in maximal:
            key = tuple(sorted(g))
            if key not in seen:
                seen.add(key)
                out.append([self.models[i] for i in g])
        return out

    def to_text(self) -> str:
        order = np.argsort(self.mean_ranks)
        out = [f"сплит: {self.split}   блоков (сидов): {self.n_blocks}",
               f"Фридман: χ² = {self.friedman['statistic']:.3f}, "
               f"p = {self.friedman['p_value']:.4g}",
               f"CD(α={self.alpha}) = {self.cd:.3f}", "средние ранги:"]
        for i in order:
            out.append(f"   {self.models[i]:<14}{self.mean_ranks[i]:.3f}")
        out.append("неразличимые группы: " +
                   " | ".join("{" + ", ".join(g) + "}" for g in self.cliques()))
        pairs = []
        for i, j in combinations(range(len(self.models)), 2):
            if abs(self.mean_ranks[i] - self.mean_ranks[j]) > self.cd:
                pairs.append(f"{self.models[i]} ↔ {self.models[j]}")
        out.append("различимые пары: " + (", ".join(pairs) if pairs else "нет"))
        out.append("⚠ Неразличимость не транзитивна: принадлежность одной группе "
                   "не означает, что различима или неразличима любая пара внутри "
                   "неё — смотреть строку выше (ERRATA E-15).")
        return "\n".join(out)


def nemenyi(summary: SeedSummary, alpha: float = 0.05) -> NemenyiResult:
    """Полный ранговый анализ ВНУТРИ одного сплита, блоки — сиды."""
    R = ranks_per_block(summary.values)
    return NemenyiResult(
        models=list(summary.models),
        mean_ranks=R.mean(axis=1),
        cd=nemenyi_cd(len(summary.models), summary.values.shape[1], alpha),
        alpha=alpha,
        n_blocks=summary.values.shape[1],
        friedman=friedman_test(summary.values),
        split=summary.split,
    )


def delta_vs_matched(extrap: SeedSummary, matched: SeedSummary) -> List[Dict[str, object]]:
    """
    ΔRMSE = RMSE(extrap) − RMSE(matched) — отчётная величина пункта 1.
    Без неё «RMSE вырос вдвое» не интерпретируется.
    """
    if extrap.models != matched.models:
        raise ValueError("наборы моделей не совпадают")
    from scipy.stats import ttest_ind
    out = []
    n = min(extrap.values.shape[1], matched.values.shape[1])
    for i, m in enumerate(extrap.models):
        d = float(extrap.mean[i] - matched.mean[i])
        s = float(np.sqrt(extrap.std[i] ** 2 + matched.std[i] ** 2))
        if n >= MIN_SEEDS_FOR_TEST:
            # сиды у двух сплитов независимы (разные выборки), поэтому тест
            # непарный — парный здесь был бы подлогом
            res = ttest_ind(extrap.values[i], matched.values[i], equal_var=False)
            p = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
            est = p < 0.05
        else:
            p, est = float("nan"), False
        out.append({"model": m, "rmse_extrap": float(extrap.mean[i]),
                    "rmse_matched": float(matched.mean[i]),
                    "delta": d, "delta_std": s, "p_value": p,
                    "n_seeds": n, "established": est})
    return out
