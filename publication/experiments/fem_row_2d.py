"""
fem_row_2d.py — строка «FEM» таблицы допустимости двумерного трека (табл. 13).

Те же величины, что run_matrix.one_run_2d считает на предсказаниях, но на самих
данных МКЭ тех же тестовых наборов: медианные невязки равновесия |R_r|, |R_z|
(полная форма, МПа/м) и средние |σ_rr|, |τ_rz| во внешнем радиальном узле.

Набор из 1756 случаев — двумерная сетка 8 × 20 (dataset2d.build) из сырых `.rpt`
релиза raw-data-v1 минус 92 набора из audit/removed_1848_to_1756.csv: 89 по маске
окна (filters.unfinished_draw_in_window) и 3 завершённых расчёта, снятых
вместе с ними по прежней проверке хвоста. Код, которым это делалось 2026-09-02,
не сохранился; состав восстановлен по отпечатку — стандартным отклонениям
компонент, которые runs_v2.csv хранит неявно как rmse_c / nrmse_c_global.

Запуск из publication/:
    python experiments/fem_row_2d.py --raw <распакованный Raw_Data>
    python experiments/fem_row_2d.py --field2d field2d.npz   # уже собранная сетка 1848
"""
import argparse
import ast
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)
from dataset2d import Field2D, build  # noqa: E402
from splits import build_split_suite  # noqa: E402

COMPONENTS = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")


def load_audit():
    """equilibrium_audit_2d из trainer2d. Сам аудит — numpy, а модуль тянет torch,
    поэтому без torch функция берётся прямо из исходника."""
    try:
        from trainer2d import equilibrium_audit_2d
        return equilibrium_audit_2d
    except ImportError:
        src = open(os.path.join(CODE, "trainer2d.py"), encoding="utf-8").read()
        fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "equilibrium_audit_2d")
        ns = {"np": np, "Dict": dict}
        exec(compile(ast.Module([fn], []), "trainer2d.py", "exec"), ns)
        return ns["equilibrium_audit_2d"]


def clean_1756(f: Field2D, removed_csv: str) -> Field2D:
    drop = {r["job"] for r in csv.DictReader(open(removed_csv, encoding="utf-8"))}
    names = np.array([str(n) for n in f.names])
    missing = drop - set(names)
    if missing:
        raise ValueError(f"в наборе нет {len(missing)} снимаемых заданий: {sorted(missing)[:3]}")
    return f.subset(np.flatnonzero(~np.isin(names, list(drop))))


def fingerprint(f: Field2D, runs_csv: str) -> None:
    """Стандартные отклонения компонент против rmse/nrmse_global стадии twod."""
    rows = [r for r in csv.DictReader(open(runs_csv, encoding="utf-8")) if r["stage"] == "twod"]
    gstd = f.y.reshape(-1, 4).std(axis=0)
    for k, c in enumerate(COMPONENTS):
        ref = np.mean([float(r[f"rmse_{c}"]) / float(r[f"nrmse_{c}_global"]) for r in rows])
        print(f"  {c:9s} std {gstd[k]:.10f}  runs_v2 {ref:.10f}  отн. расхождение {abs(gstd[k] / ref - 1):.1e}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", help="распакованный Raw_Data (2443 .rpt)")
    src.add_argument("--field2d", help="Field2D .npz на 1848 наборах (dataset2d.py)")
    ap.add_argument("--removed", default=os.path.join(HERE, "..", "audit", "removed_1848_to_1756.csv"))
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "results", "runs_v2.csv"))
    a = ap.parse_args()

    f = build(a.raw, 8, 20) if a.raw else Field2D.load(a.field2d)
    g = clean_1756(f, a.removed)
    print(f"наборов: {f.n_sets} → {g.n_sets}")
    fingerprint(g, a.runs)

    audit = load_audit()
    suite = build_split_suite(g.proc)
    print(f"{'split':18s} {'train':>5s} {'test':>5s} {'|R_r|':>8s} {'|R_z|':>8s} "
          f"{'|σrr(1)|':>9s} {'|τrz(1)|':>9s}")
    for name in ("random", "extrap:alpha_max"):
        tr, te = suite[name].train_sets, suite[name].test_sets
        eq = audit(g.y[te], g.r_phys[te], g.z_phys[te])
        srr = float(np.abs(g.y[te][:, :, -1, 0]).mean())
        trz = float(np.abs(g.y[te][:, :, -1, 3]).mean())
        print(f"{name:18s} {len(tr):5d} {len(te):5d} {eq['eq_r_full']:8.1f} {eq['eq_z_full']:8.1f} "
              f"{srr:9.3f} {trz:9.4f}")


if __name__ == "__main__":
    main()
