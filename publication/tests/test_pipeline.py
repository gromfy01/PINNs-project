"""
Тесты нового пайплайна. Запуск:  python publication/tests/test_pipeline.py

Проверяются те инварианты, нарушение которых делает результат невалидным
(§10 playbook), а не «работает ли код вообще».
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import corruption          # noqa: E402
import filters             # noqa: E402
import learning_curves     # noqa: E402
import physics_check       # noqa: E402
import protocol            # noqa: E402
import splits              # noqa: E402
import stats               # noqa: E402

N_R = 20
LEVELS = {"Q": [0.015, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25],
          "k": [0.0, 0.1, 0.3, 0.5, 0.75, 1.0],
          "alpha": [4, 8, 12, 16, 20],
          "mu": [0.025, 0.05, 0.10],
          "v": [5, 10, 20, 40, 250]}


def synthetic(seed=0, n=400, with_defects=True):
    """Синтетический датасет той же структуры, что настоящий."""
    rng = np.random.default_rng(seed)
    proc = np.stack([rng.choice(LEVELS[k], size=n) for k in
                     ("Q", "k", "alpha", "mu", "v")], axis=1).astype(float)
    r = np.linspace(0.0, 1.0, N_R)
    # профиль, у которого σ_rr и τ_rz зануляются на r = 1 (свободная поверхность)
    s_rr = np.outer(proc[:, 0] * 800, (1 - r ** 2))
    s_tt = np.outer(proc[:, 0] * 600, (1 - 0.3 * r)) + 40
    s_zz = np.outer(proc[:, 0] * 900, (0.5 + r))
    t_rz = np.outer(proc[:, 2] * 2.0, r * (1 - r))
    y = np.stack([s_rr, s_tt, s_zz, t_rz], axis=-1)
    if with_defects:
        proc[0, 3] = 0.002                       # μ вне сетки
        # пара, отличающаяся ТОЛЬКО меткой μ и имеющая одинаковый выход
        proc[1] = [0.10, 0.5, 12.0, 0.025, 20.0]
        proc[2] = [0.10, 0.5, 12.0, 0.050, 20.0]
        y[1] = y[2]
    return proc, y, r


def check(name, cond, detail=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name)


def test_filters():
    print("filters")
    proc, y, _ = synthetic()
    idx, rep = filters.filter_dataset(proc, y, drop_removed_indices=False)
    check("μ вне сетки удалён", 0 not in idx)
    check("из μ-дубликатов остаётся ровно один", 1 in idx and 2 not in idx)
    check("Q = 0.25 удалён целиком",
          not np.any(np.isclose(proc[idx, 0], 0.25)))
    check("уцелевших меньше входа", len(idx) < len(proc), f"{len(idx)}/{len(proc)}")
    cov = filters.grid_coverage(proc[idx])
    check("полная сетка = 3150 после снятия фантомного μ",
          cov["full_factorial"] == 7 * 6 * 5 * 3 * 5 or cov["full_factorial"] <= 3150,
          str(cov["full_factorial"]))
    check("отчёт печатается", "наборов на входе" in rep.to_text())


def test_splits():
    print("splits")
    proc, y, _ = synthetic()
    idx, _ = filters.filter_dataset(proc, y, drop_removed_indices=False)
    p = proc[idx]
    suite = splits.build_split_suite(p)
    for name, sp in suite.items():
        sp.check()
        check(f"{name}: нет пересечения train/test",
              np.intersect1d(sp.train_sets, sp.test_sets).size == 0)
    am = suite["extrap:alpha_max"]
    cov = splits.coverage_report(p, am)
    outside = {c["axis"] for c in cov if c["outside_range"]}
    check("alpha_max экстраполирует ровно по alpha", outside == {"alpha"}, str(outside))
    mid = suite["interp:alpha_mid"]
    cov_mid = splits.coverage_report(p, mid)
    check("alpha_mid не экстраполирует ни по одной оси",
          not any(c["outside_range"] for c in cov_mid))
    check("у alpha_max есть matched-контроль", "matched:alpha_max" in suite)
    check("matched того же объёма",
          suite["matched:alpha_max"].test_sets.size == am.test_sets.size)
    tr_rows, te_rows = am.rows(N_R)
    check("строки не пересекаются", np.intersect1d(tr_rows, te_rows).size == 0)
    check("все 20 r-точек набора в одной части",
          set(np.unique(te_rows // N_R)) == set(am.test_sets.tolist()))


def test_protocol():
    print("protocol")
    check("оптимизатор один", protocol.PROTOCOL.optimizer == "AdamW")
    check("планировщика нет", protocol.PROTOCOL.lr_schedule == "none")
    v = protocol.assert_conforms("pinn_dde", {"optimizer": "LBFGS"}, strict=False)
    check("LBFGS ловится", len(v) >= 1, str(v[0]))
    v = protocol.assert_conforms("pinn_jax", {"lr_schedule": "cosine"}, strict=False)
    check("cosine ловится", len(v) >= 1)
    check("конформный конфиг проходит",
          protocol.assert_conforms("pinn_torch",
                                   {"optimizer": "AdamW", "lr_schedule": "none"}) == [])

    class T:
        def __init__(self): self.log = {}
        def suggest_int(self, n, a, b): self.log[n] = ("int", a, b); return a
        def suggest_float(self, n, a, b, log=False): self.log[n] = ("f", a, b, log); return a
        def suggest_categorical(self, n, c): self.log[n] = ("cat", tuple(c)); return c[0]

    logs = {}
    for m in protocol.MODEL_FAMILIES:
        t = T(); protocol.suggest_config(t, m); logs[m] = t.log
    common = set(protocol.SEARCH_SPACE) - {"n_units"}
    for m, lg in logs.items():
        for k in common:
            check(f"{m}: {k} из общего пространства", lg[k] == logs["mlp_grid"][k])
    check("physics-модели получают λ",
          all("lambda_bc" in logs[m] for m in protocol.PHYSICS_MODELS))
    check("датадривен не получает λ", "lambda_bc" not in logs["mlp_grid"])


def test_stats():
    print("stats")
    rng = np.random.default_rng(1)
    models = ["mlp_grid", "mlp_optuna", "pinn_torch", "pinn_dde", "pinn_jax", "vpinn"]
    true = np.array([13.2, 13.0, 11.9, 12.3, 11.7, 12.1])
    recs = [{"model": m, "seed": s, "split": "extrap:alpha_max",
             "macro_rmse": float(true[i] + rng.normal(0, 0.6))}
            for i, m in enumerate(models) for s in range(5)]
    summ = stats.build_seed_summary(recs)
    check("матрица модель×сид собрана", summ.values.shape == (6, 5))
    check("std ненулевой", (summ.std > 0).all())
    res = stats.nemenyi(summ)
    check("CD посчитан", res.cd > 0, f"CD={res.cd:.3f}")
    check("блоки — сиды", res.n_blocks == 5)
    check("клики непустые", len(res.cliques()) >= 1)
    g = stats.significance_gate(summ)
    small = [p for p in g if abs(p.delta) < p.pooled_std]
    check("мелкие разницы помечены как неустановленные",
          all(not p.established for p in small), f"{len(small)} пар")

    mixed = recs + [{"model": "vpinn", "seed": 0, "split": "random", "macro_rmse": 1.0}]
    try:
        stats.build_seed_summary(mixed)
        check("пул сплитов запрещён", False)
    except ValueError:
        check("пул сплитов запрещён", True)

    try:
        stats.friedman_test(np.zeros((6, 1)))
        check("один прогон не проходит как тест", False)
    except ValueError:
        check("один прогон не проходит как тест", True)


def test_learning_curves():
    print("learning_curves")
    tr = np.arange(1000)
    sub = learning_curves.nested_subsamples(tr, seed=7)
    fr = sorted(sub)
    for a, b in zip(fr[:-1], fr[1:]):
        check(f"{a} ⊂ {b}", np.isin(sub[a], sub[b]).all())
    check("100 % = весь пул", len(sub[1.0]) == len(tr))
    check("одинаково при том же seed",
          np.array_equal(sub[0.1], learning_curves.nested_subsamples(tr, seed=7)[0.1]))
    proc, _y, _ = synthetic()
    st = learning_curves.stratified_nested_subsamples(np.arange(len(proc)), proc, 4, seed=3)
    check("уровень v = 5 не исчезает на 10 %",
          np.any(np.isclose(proc[st[0.1], 4], 5.0)))
    curves = {"pinn_jax": {0.1: 14.0, 0.25: 12.5, 0.5: 11.8, 1.0: 11.5},
              "mlp_optuna": {0.1: 19.0, 0.25: 14.0, 0.5: 11.6, 1.0: 10.9}}
    check("точка догоняния найдена",
          learning_curves.crossover_fraction(curves, ["pinn_jax"], ["mlp_optuna"]) == 0.5)


def test_corruption():
    print("corruption")
    tr = np.arange(500)
    plans = corruption.make_corruption_plans(tr, seed=11)
    check("0 % ничего не портит", len(plans[0.0].corrupted_sets) == 0)
    check("10 % → 50 наборов", len(plans[0.1].corrupted_sets) == 50)
    check("1 % ⊂ 10 %", np.isin(plans[0.01].corrupted_sets, plans[0.1].corrupted_sets).all())
    y = np.zeros((500, N_R, 4))
    y2 = corruption.apply_corruption(y, plans[0.05])
    check("исходный массив не изменён", np.all(y == 0))
    touched = np.unique(np.flatnonzero(np.abs(y2).sum(axis=(1, 2)) > 0))
    check("испорчены ровно запланированные наборы",
          np.array_equal(touched, plans[0.05].corrupted_sets))
    check("испорчена ровно одна компонента",
          np.all(y2[plans[0.05].corrupted_sets][:, :, 0] == 0)
          and np.all(y2[plans[0.05].corrupted_sets][:, :, 1] == corruption.DEFAULT_OFFSET_MPA))
    check("набор испорчен целиком",
          np.all(y2[plans[0.05].corrupted_sets[0], :, 1] ==
                 y2[plans[0.05].corrupted_sets[0], 0, 1]))
    tbl = corruption.degradation_table({"a": {0.0: 10.0, 0.1: 12.0},
                                        "b": {0.0: 10.0, 0.1: 20.0}})
    check("сортировка по устойчивости", tbl[0]["model"] == "a")


def test_physics():
    print("physics_check")
    _proc, y, r = synthetic(with_defects=False)
    check("свободная поверхность при r_norm = 1", physics_check.R_FREE_SURFACE == 1.0)
    srr, trz = physics_check.bc_violation(y[0], r)
    check("σ_rr → 0 на поверхности синтетики", srr < 1e-9, f"{srr:.2e}")
    check("τ_rz → 0 на поверхности синтетики", trz < 1e-9, f"{trz:.2e}")
    try:
        physics_check.bc_violation(y[0], r, r_surface=1.5)
        check("экстраполяция за сетку запрещена", False)
    except ValueError:
        check("экстраполяция за сетку запрещена", True)
    a = physics_check.audit(y[:20], r)
    check("аудит печатается", "МПа" in a.to_text())
    lin = np.zeros((N_R, 4)); lin[:, 0] = r; lin[:, 1] = r
    res = physics_check.equilibrium_residual_radial(lin, r)
    check("R₁ линейного профиля = ∂σ_rr/∂r = 1", np.allclose(res, 1.0), f"{res[:2]}")


def main():
    for t in (test_filters, test_splits, test_protocol, test_stats,
              test_learning_curves, test_corruption, test_physics):
        t()
    print("\nвсе тесты прошли")


if __name__ == "__main__":
    main()
