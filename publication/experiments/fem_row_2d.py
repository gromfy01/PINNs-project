"""
fem_row_2d.py — строка «FEM» таблицы допустимости двумерного трека (табл. 13).

Те же величины, что run_matrix.one_run_2d считает на предсказаниях, но на самих
данных МКЭ тех же тестовых наборов: медианные невязки равновесия |R_r|, |R_z|
(полная форма, МПа/м) и средние |σ_rr|, |τ_rz| во внешнем радиальном узле.

Набор из 1756 случаев — двумерная сетка 8 × 20 (dataset2d.build) из сырых `.rpt`
релиза raw-data-v1 минус 92 набора из audit/removed_1848_to_1756.csv: 89 по маске
окна (filters.unfinished_draw_in_window) и 3 завершённых расчёта, снятых
вместе с ними по прежней проверке хвоста. Состав набора сверяется по
отпечатку — стандартным отклонениям компонент, которые runs_v2.csv хранит
неявно как rmse_c / nrmse_c_global.

Запуск из корня пакета:
    python experiments/fem_row_2d.py --raw <распакованный Raw_Data>
    python experiments/fem_row_2d.py --field2d field2d.npz   # уже собранная сетка 1848
С --json строка пишется ещё и в json, который читает
experiments/paper_tables.py --fem-json (таблица tab:admissibility).
С --save-field2d сохраняется сам набор из 1756 случаев — его берут
run_matrix.py --stage twod, dump_predictions.py и residual_maps.py.
"""
import argparse
import ast
import csv
import json
import os
import sys
from typing import Dict, Sequence

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


SPLITS = ("random", "extrap:alpha_max")


def fem_row(g: Field2D, splits: Sequence[str] = SPLITS) -> Dict[str, Dict[str, float]]:
    """
    Строка FEM по сплитам: {сплит: {n_train, n_test, eq_r_full, eq_z_full,
    bc_sigma_rr, bc_tau_rz}}. Невязки — equilibrium_audit_2d на данных МКЭ
    тестовых наборов, нарушения ГУ — средние |σ_rr| и |τ_rz| во внешнем
    радиальном узле по осевым узлам и наборам.
    """
    audit = load_audit()
    suite = build_split_suite(g.proc)
    out: Dict[str, Dict[str, float]] = {}
    for name in splits:
        tr, te = suite[name].train_sets, suite[name].test_sets
        eq = audit(g.y[te], g.r_phys[te], g.z_phys[te])
        out[name] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                     "eq_r_full": eq["eq_r_full"], "eq_z_full": eq["eq_z_full"],
                     "bc_sigma_rr": float(np.abs(g.y[te][:, :, -1, 0]).mean()),
                     "bc_tau_rz": float(np.abs(g.y[te][:, :, -1, 3]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", help="распакованный Raw_Data (2443 .rpt)")
    src.add_argument("--field2d", help="Field2D .npz на 1848 наборах (dataset2d.py)")
    ap.add_argument("--removed", default=os.path.join(HERE, "..", "audit", "removed_1848_to_1756.csv"))
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "results", "runs_v2.csv"))
    ap.add_argument("--json", default="",
                    help="записать строку FEM в json (для paper_tables.py --fem-json)")
    ap.add_argument("--save-field2d", default="",
                    help="сохранить очищенный набор из 1756 случаев (Field2D .npz)")
    a = ap.parse_args()

    f = build(a.raw, 8, 20) if a.raw else Field2D.load(a.field2d)
    g = clean_1756(f, a.removed)
    print(f"наборов: {f.n_sets} → {g.n_sets}")
    fingerprint(g, a.runs)
    if a.save_field2d:
        g.save(a.save_field2d)
        print(f"сохранено: {a.save_field2d}")

    row = fem_row(g)
    print(f"{'split':18s} {'train':>5s} {'test':>5s} {'|R_r|':>8s} {'|R_z|':>8s} "
          f"{'|σrr(1)|':>9s} {'|τrz(1)|':>9s}")
    for name, v in row.items():
        print(f"{name:18s} {v['n_train']:5d} {v['n_test']:5d} {v['eq_r_full']:8.1f} "
              f"{v['eq_z_full']:8.1f} {v['bc_sigma_rr']:9.3f} {v['bc_tau_rz']:9.4f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"n_sets": int(g.n_sets), "grid": [int(g.y.shape[1]), int(g.y.shape[2])],
                       "splits": row}, fh, ensure_ascii=False, indent=1)
        print(f"записано: {a.json}")


if __name__ == "__main__":
    main()
