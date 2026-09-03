"""
Тесты тренеров и физики. Запуск: python publication/tests/test_training.py

Проверяются свойства, ошибка в которых даёт правдоподобный, но неверный
результат — то есть ровно тот класс дефектов, из-за которого статью отклонили.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import torch                                            # noqa: E402
torch.set_num_threads(1)

from dataset import COMPONENTS, StressDataset, sanity   # noqa: E402
from trainer import (Config, Trainer, bc_audit, equilibrium_audit,  # noqa: E402
                     metrics)
from trainer2d import Config2D, Trainer2D, equilibrium_audit_2d  # noqa: E402

N_R = 20


def check(name, cond, detail=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name)


def toy(n=120, seed=0):
    """
    Синтетика с ТОЧНО известной физикой: σ_rr = A·r², σ_θθ = 3A·r² даёт

        R₁ = ∂σ_rr/∂r + (σ_rr − σ_θθ)/r = 2Ar − 2Ar = 0

    то есть радиальное равновесие выполняется тождественно, а σ_rr(1) ≠ 0,
    поэтому ГУ и равновесие проверяются независимо друг от друга.
    """
    rng = np.random.default_rng(seed)
    proc = np.stack([rng.choice([0.05, 0.10, 0.15], n), rng.choice([0.0, 0.5, 1.0], n),
                     rng.choice([4.0, 12.0, 20.0], n), rng.choice([0.025, 0.05], n),
                     rng.choice([10.0, 40.0], n)], axis=1)
    r = np.tile(np.linspace(0.0, 1.0, N_R), (n, 1))
    R = 0.016
    A = (proc[:, 0] * 1000.0)[:, None]
    y = np.stack([A * r ** 2, 3 * A * r ** 2, A * (1 + r), 0.01 * A * r * (1 - r)], -1)
    return proc, y, r, r * R


def test_metrics():
    print("metrics")
    t = np.zeros((5, N_R, 4)); p = np.zeros_like(t)
    t[:, :, 0] = 1.0
    m = metrics(t, p)
    check("RMSE постоянного смещения", abs(m["rmse_sigma_rr"] - 1.0) < 1e-12)
    check("R² нулевой дисперсии не ломает отчёт", np.isfinite(m["macro_r2"]))
    p2 = t.copy()
    m2 = metrics(t, p2)
    check("идеальный прогноз → RMSE 0", m2["macro_rmse"] < 1e-12)


def test_equilibrium_analytic():
    print("equilibrium (аналитика)")
    _proc, y, _r, r_phys = toy(8)
    a = equilibrium_audit(y, r_phys)
    scale = np.abs(np.gradient(y[0, :, 0], r_phys[0])).max()
    check("R₁ ≈ 0 на профиле, где равновесие выполняется точно",
          a["eq_res_median"] < 0.02 * scale,
          f"{a['eq_res_median']:.3g} против масштаба члена {scale:.3g}")


def test_bc_measured_where_imposed():
    print("BC")
    _proc, y, _r, _rp = toy(6)
    b = bc_audit(y[:, -1, :])
    check("σ_rr на r=1 у синтетики не ноль (иначе тест пустой)", b["bc_sigma_rr"] > 1.0)
    from physics_check import R_FREE_SURFACE
    check("свободная поверхность = 1.0", R_FREE_SURFACE == 1.0)


def test_trainer_learns():
    print("trainer")
    proc, y, r, rp = toy(120)
    tr, te = np.arange(90), np.arange(90, 120)
    out = {}
    for fam, lp in (("mlp", 0.0), ("pinn", 0.05), ("vpinn", 0.05)):
        cfg = Config(family=fam, seed=0, max_epochs=60, patience=60, n_units=48,
                     n_layers=3, batch_sets=32, learning_rate=3e-3,
                     lambda_physics=lp, lambda_bc=0.5 if lp else 0.0)
        T = Trainer(cfg)
        res = T.fit(proc[tr], y[tr], r[tr], rp[tr])
        m = T.evaluate(proc[te], y[te], r[te])
        base = float(np.sqrt(np.mean((y[te] - y[tr].mean(axis=0)) ** 2)))
        out[fam] = m["macro_rmse"]
        check(f"{fam}: лучше предсказания средним", m["macro_rmse"] < base,
              f"{m['macro_rmse']:.3f} против {base:.3f}")
        check(f"{fam}: история валидации не пуста", len(res.history) > 0)
        check(f"{fam}: валидация улучшилась", res.history[-1] <= res.history[0] or res.best_val < res.history[0])
    check("физика не ломает обучение на порядок",
          max(out.values()) < 5 * min(out.values()), str({k: round(v, 2) for k, v in out.items()}))


def test_physics_actually_binds():
    print("физлосс действительно влияет")
    proc, y, r, rp = toy(120)
    tr = np.arange(100)
    bc = {}
    for lam in (0.0, 5.0):
        cfg = Config(family="pinn", seed=1, max_epochs=60, patience=60, n_units=48,
                     n_layers=3, batch_sets=32, learning_rate=3e-3,
                     lambda_physics=0.0, lambda_bc=lam)
        T = Trainer(cfg)
        T.fit(proc[tr], y[tr], r[tr], rp[tr])
        bc[lam] = float(np.abs(T.predict_at(proc[tr], 1.0)[:, 0]).mean())
    check("λ_bc = 5 сильнее зануляет σ_rr на поверхности, чем λ_bc = 0",
          bc[5.0] < bc[0.0], f"{bc[5.0]:.2f} против {bc[0.0]:.2f}")


def test_backend_agreement():
    print("бэкенды")
    from jax_twin import check_gradient_agreement
    d = check_gradient_agreement(Config(seed=0, n_layers=3, n_units=48))
    check("прямой проход torch == jax", d["max_abs_diff_forward"] < 1e-10,
          f"{d['max_abs_diff_forward']:.2e}")
    check("производная torch == jax", d["max_rel_diff_grad"] < 1e-10,
          f"{d['max_rel_diff_grad']:.2e}")


def test_2d_reduces_to_1d():
    print("2D физика")
    n, nz, nr = 6, 5, N_R
    rng = np.random.default_rng(0)
    r = np.tile(np.linspace(0.02, 1.0, nr), (n, nz, 1))
    r_phys = r * 0.016
    z_phys = np.tile(np.linspace(0.0, 0.06, nz), (n, 1))
    A = rng.uniform(50, 150, size=(n, 1, 1))
    # поле, не зависящее от z ⇒ осевые производные равны нулю
    y = np.stack([A * r ** 2, 3 * A * r ** 2, A * (1 + 0 * r), 0.0 * r], -1)
    a = equilibrium_audit_2d(y, r_phys, z_phys)
    check("при полях, не зависящих от z, полная и редуцированная формы совпадают",
          abs(a["eq_r_full"] - a["eq_r_reduced"]) < 1e-6 * max(a["eq_r_reduced"], 1.0)
          and abs(a["eq_z_full"] - a["eq_z_reduced"]) < 1e-6 * max(a["eq_z_reduced"], 1.0),
          f"R_r {a['eq_r_full']:.4g}/{a['eq_r_reduced']:.4g}, "
          f"R_z {a['eq_z_full']:.4g}/{a['eq_z_reduced']:.4g}")
    # и наоборот: при зависимости от z формы обязаны разойтись
    yz = y.copy()
    yz[:, :, :, 3] += np.linspace(0, 20, nz).reshape(1, nz, 1)
    b = equilibrium_audit_2d(yz, r_phys, z_phys)
    check("при зависимости τ_rz от z формы расходятся",
          abs(b["eq_r_full"] - b["eq_r_reduced"]) > 1.0,
          f"{b['eq_r_full']:.4g} против {b['eq_r_reduced']:.4g}")


def test_2d_trainer_runs():
    print("trainer2d")
    n, nz, nr = 40, 4, N_R
    rng = np.random.default_rng(0)
    proc = np.stack([rng.choice([0.05, 0.15], n), rng.choice([0.0, 1.0], n),
                     rng.choice([4.0, 20.0], n), rng.choice([0.025, 0.05], n),
                     rng.choice([10.0, 40.0], n)], axis=1)
    r = np.tile(np.linspace(0.0, 1.0, nr), (n, nz, 1))
    z = np.tile(np.linspace(0.0, 1.0, nz), (n, 1))
    A = (proc[:, 0] * 1000)[:, None, None]
    y = np.stack([A * r ** 2, 3 * A * r ** 2, A * (1 + r), 0.01 * A * r * (1 - r)], -1)
    for fam in ("mlp2d", "pinn2d_reduced", "pinn2d_full"):
        cfg = Config2D(family=fam, seed=0, max_epochs=8, patience=8, n_units=32,
                       n_layers=2, batch_sets=16)
        T = Trainer2D(cfg)
        res = T.fit(proc, y, r, z, r * 0.016, np.tile(np.linspace(0, 0.06, nz), (n, 1)))
        m = T.evaluate(proc, y, r, z)
        check(f"{fam}: обучение проходит и метрики конечны",
              np.isfinite(m["macro_rmse"]) and res.epochs_run > 0)


def test_dataset_roundtrip(tmp="/tmp/_ds_test.npz"):
    print("dataset")
    proc, y, r, rp = toy(10)
    ds = StressDataset(proc, y, r, rp, np.array([f"job{i}" for i in range(10)]))
    ds.save(tmp)
    back = StressDataset.load(tmp)
    check("сохранение/загрузка не теряет данные",
          np.allclose(back.y, ds.y) and np.array_equal(back.names, ds.names))
    X, Y = ds.rows()
    check("раскладка строк set-major", X.shape == (10 * N_R, 6) and Y.shape == (10 * N_R, 4))
    check("r — последняя колонка", np.allclose(X[:N_R, 5], r[0]))
    check("set_index = row // 20", np.allclose(X[N_R:2 * N_R, 0], proc[1, 0]))
    s = sanity(ds)
    check("sanity считает все ключи", "bc_sigma_rr_surface" in s and s["n_sets"] == 10)
    os.remove(tmp)


def main():
    for t in (test_metrics, test_equilibrium_analytic, test_bc_measured_where_imposed,
              test_dataset_roundtrip, test_2d_reduces_to_1d, test_backend_agreement,
              test_trainer_learns, test_physics_actually_binds, test_2d_trainer_runs):
        t()
    print("\nвсе тесты прошли")


if __name__ == "__main__":
    main()
