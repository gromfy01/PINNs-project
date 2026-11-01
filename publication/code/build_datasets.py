"""
build_datasets.py — наборы данных статьи из сырых .rpt за один запуск.

Шаги:
  1. проход аудита по сырым отчётам — main() из audit/run_raw_audit.py:
     медианные профили prof_s (n, 20, 4) в порядке raw_probe.STACK_ORDER и
     физические радиусы бинов prof_r (n, 20); результат пишется в
     <out>/audit.npz, тот же файл читает audit/make_report.py;
  2. prof_s переупаковывается в раскладку allcols, которую принимает
     dataset.build_from_allcols: data = prof_s, cols = ('s_rr', 's_tt',
     's_zz', 's_rz'), params и names без изменений;
  3. одномерный набор: build_from_allcols (с filter_dataset) — 1848 наборов;
  4. двумерная сетка 8 × 20: dataset2d.build(raw, 8, 20) — тоже 1848 наборов;
  5. из обоих снимаются 92 задания audit/removed_1848_to_1756.csv
     (89 незавершённых расчётов по маске окна и 3 завершённых расчёта при
     Q = 1.5 %, v = 250 м/мин, которые пометила более ранняя проверка хвоста),
     остаётся по 1756 наборов; сохраняются stress_1756.npz и field2d_1756.npz;
  6. отпечаток: стандартные отклонения компонент по всему набору против
     значений, которые results/runs_v2.csv хранит неявно. Там
     nrmse_c_global = rmse_c / (std_c + 1e-12), где std_c — по всему набору,
     на котором шли прогоны (trainer.metrics), поэтому
     rmse_c / nrmse_c_global − 1e-12 и есть этот std. Для stress_1756.npz
     берутся одномерные стадии main, curves, corrupt, backend, для
     field2d_1756.npz — стадия twod, как в experiments/fem_row_2d.py.
     Совпадение до ошибки округления означает тот же состав наборов и те же
     значения напряжений. Радиальные координаты r отпечаток не проверяет.

Нужен только numpy.

Запуск из корня пакета:
    python code/build_datasets.py --raw <распакованный Raw_Data> --out data

Результат в --out: stress_1756.npz (StressDataset), field2d_1756.npz (Field2D),
audit.npz; с --keep-1848 ещё stress_1848.npz и field2d_1848.npz — набор из 1848
случаев принимает experiments/fem_row_2d.py --field2d. --audit берёт готовый
audit.npz вместо нового прохода. --strict завершает работу с кодом 1, если
число наборов или отпечаток не совпали с ожидаемыми.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import tempfile
import time
from typing import Iterable, Sequence, Set, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from dataset import build_from_allcols  # noqa: E402
from dataset import sanity as sanity_1d  # noqa: E402
from dataset2d import build as build_2d  # noqa: E402
from dataset2d import sanity as sanity_2d  # noqa: E402
from raw_probe import STACK_ORDER  # noqa: E402

AUDIT_SCRIPT = os.path.join(ROOT, "audit", "run_raw_audit.py")
REMOVED_CSV = os.path.join(ROOT, "audit", "removed_1848_to_1756.csv")
RUNS_CSV = os.path.join(ROOT, "results", "runs_v2.csv")

COMPONENTS = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")
ALLCOLS_COLS = ("s_rr", "s_tt", "s_zz", "s_rz")   # порядок, который ждёт build_from_allcols
STAGES_1D = ("main", "curves", "corrupt", "backend")
STAGES_2D = ("twod",)
N_Z, N_R = 8, 20
N_FILTERED, N_FINAL, N_REMOVED = 1848, 1756, 92
GLOBAL_STD_EPS = 1e-12      # добавка к знаменателю nrmse_*_global в trainer.metrics
FP_TOL = 1e-9               # допустимое относительное расхождение отпечатка


# ─────────────────────────────── шаги ───────────────────────────────────

def run_audit(raw: str, out_npz: str) -> None:
    """Проход audit/run_raw_audit.py по сырому дереву: его main() с --root/--out."""
    spec = importlib.util.spec_from_file_location("run_raw_audit", AUDIT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = sys.argv
    sys.argv = [AUDIT_SCRIPT, "--root", raw, "--out", out_npz]
    try:
        mod.main()
    finally:
        sys.argv = argv


def write_allcols(audit_npz: str, path: str) -> None:
    """Выход аудита → раскладка allcols для dataset.build_from_allcols."""
    if tuple(STACK_ORDER) != ALLCOLS_COLS:
        raise RuntimeError(f"raw_probe.STACK_ORDER = {STACK_ORDER}, ожидалось {ALLCOLS_COLS}")
    A = np.load(audit_npz, allow_pickle=True)
    np.savez_compressed(path, data=A["prof_s"], params=A["params"], names=A["names"],
                        cols=np.array(STACK_ORDER))


def read_removed(path: str) -> Set[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["job"] for r in csv.DictReader(fh)}


def drop_removed(ds, drop: Set[str]):
    """
    Снять задания из списка по имени. Работает и для StressDataset, и для
    Field2D (нужны только names и subset). Если какого-то задания из списка в
    наборе нет, это ошибка: значит, набор собран не из того сырья или не теми
    фильтрами (так же поступает experiments/fem_row_2d.clean_1756).
    """
    names = np.array([str(n) for n in ds.names])
    missing = drop - set(names)
    if missing:
        raise ValueError(f"в наборе нет {len(missing)} снимаемых заданий: {sorted(missing)[:3]}")
    return ds.subset(np.flatnonzero(~np.isin(names, list(drop))))


# ───────────────────────────── отпечаток ────────────────────────────────

def implied_std(runs_csv: str, stages: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    std компонент, восстановленный из runs_v2.csv по строкам стадий stages:
    (среднее по прогонам, относительный разброс по прогонам, число прогонов).
    Строки с ошибкой или без глобальных метрик пропускаются.
    """
    stages = set(stages)
    with open(runs_csv, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("stage") in stages and not (r.get("error") or "").strip()]
    ref = np.full(len(COMPONENTS), np.nan)
    spread = np.full(len(COMPONENTS), np.nan)
    used = 0
    for k, c in enumerate(COMPONENTS):
        vals = []
        for r in rows:
            try:
                rmse, nrmse = float(r[f"rmse_{c}"]), float(r[f"nrmse_{c}_global"])
            except (KeyError, TypeError, ValueError):
                continue
            if nrmse > 0:
                vals.append(rmse / nrmse - GLOBAL_STD_EPS)
        if vals:
            v = np.asarray(vals)
            ref[k] = v.mean()
            spread[k] = (v.max() - v.min()) / abs(v.mean())
            used = max(used, len(v))
    return ref, spread, used


def fingerprint(y: np.ndarray, runs_csv: str, stages: Sequence[str], title: str) -> float:
    """Печать std компонент набора рядом с runs_v2.csv; возвращает макс. отн. расхождение."""
    gstd = y.reshape(-1, len(COMPONENTS)).std(axis=0)
    ref, spread, n = implied_std(runs_csv, stages)
    print(f"{title}: стадии {', '.join(stages)}, прогонов {n}")
    if n == 0:
        print("  в runs_v2.csv нет подходящих строк — сверять не с чем")
        return float("nan")
    rels = np.abs(gstd / ref - 1.0)
    for k, c in enumerate(COMPONENTS):
        print(f"  {c:9s} std {gstd[k]:.10f}  runs_v2 {ref[k]:.10f}  "
              f"отн. расхождение {rels[k]:.1e}  (разброс по прогонам {spread[k]:.1e})")
    # компонента без опорного значения (nan) — это расхождение, а не совпадение
    return float(rels.max()) if np.isfinite(rels).all() else float("inf")


# ─────────────────────────────── main ───────────────────────────────────

def _check(cond: bool, msg: str, problems: list) -> None:
    print(f"  {'ok' if cond else 'НЕ СОВПАДАЕТ'}: {msg}")
    if not cond:
        problems.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="распакованный Raw_Data (каталоги Vel_* с .rpt)")
    ap.add_argument("--out", required=True, help="каталог для результатов")
    ap.add_argument("--audit", default="",
                    help="готовый выход audit/run_raw_audit.py; без него проход делается заново "
                         "и пишется в <out>/audit.npz")
    ap.add_argument("--removed", default=REMOVED_CSV)
    ap.add_argument("--runs", default=RUNS_CSV,
                    help="журнал прогонов для отпечатка (пустая строка — не сверять)")
    ap.add_argument("--keep-1848", action="store_true",
                    help="сохранить также stress_1848.npz и field2d_1848.npz")
    ap.add_argument("--strict", action="store_true",
                    help="код возврата 1, если число наборов или отпечаток не совпали")
    a = ap.parse_args()

    if not os.path.isdir(a.raw):
        raise SystemExit(f"нет каталога {a.raw}")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    problems: list = []

    # 1. проход аудита
    audit_npz = a.audit or os.path.join(a.out, "audit.npz")
    if a.audit:
        print(f"[1/5] аудит: беру готовый {audit_npz}")
    else:
        print(f"[1/5] аудит сырых отчётов → {audit_npz}")
        run_audit(a.raw, audit_npz)

    # 2–3. одномерный набор
    print(f"\n[2/5] одномерный набор (20 радиальных бинов)  [{time.time() - t0:.0f} с]")
    with tempfile.TemporaryDirectory() as tmp:
        allcols = os.path.join(tmp, "allcols.npz")
        write_allcols(audit_npz, allcols)
        ds = build_from_allcols(allcols, audit_npz)
    print(ds.report)
    for k, v in sanity_1d(ds).items():
        print(f"  {k:24s} {v:12.5g}")

    # 4. двумерная сетка
    print(f"\n[3/5] двумерная сетка {N_Z} × {N_R}  [{time.time() - t0:.0f} с]")
    f2 = build_2d(a.raw, N_Z, N_R)
    print(f2.report or "")
    for k, v in sanity_2d(f2).items():
        print(f"  {k:22s} {v:12.5g}")

    # 5. снять 92 задания: 89 незавершённых и 3 по более ранней проверке хвоста
    print(f"\n[4/5] снятие заданий из {os.path.basename(a.removed)}  [{time.time() - t0:.0f} с]")
    drop = read_removed(a.removed)
    ds_c = drop_removed(ds, drop)
    f2_c = drop_removed(f2, drop)
    print(f"  в списке {len(drop)} заданий")
    print(f"  1D: {ds.n_sets} → {ds_c.n_sets}")
    print(f"  2D: {f2.n_sets} → {f2_c.n_sets}")

    paths = []
    if a.keep_1848:
        paths += [(ds, "stress_1848.npz"), (f2, "field2d_1848.npz")]
    paths += [(ds_c, "stress_1756.npz"), (f2_c, "field2d_1756.npz")]
    for obj, name in paths:
        p = os.path.join(a.out, name)
        obj.save(p)
        print(f"  сохранено: {p}")

    # проверки состава
    print("\nпроверки состава:")
    _check(len(drop) == N_REMOVED, f"в списке снимаемых {N_REMOVED} заданий ({len(drop)})", problems)
    _check(ds.n_sets == N_FILTERED, f"1D после фильтров {N_FILTERED} ({ds.n_sets})", problems)
    _check(f2.n_sets == N_FILTERED, f"2D после фильтров {N_FILTERED} ({f2.n_sets})", problems)
    _check(ds_c.n_sets == N_FINAL, f"1D итог {N_FINAL} ({ds_c.n_sets})", problems)
    _check(f2_c.n_sets == N_FINAL, f"2D итог {N_FINAL} ({f2_c.n_sets})", problems)
    same = (ds_c.n_sets == f2_c.n_sets
            and np.array_equal(np.array([str(n) for n in ds_c.names]),
                               np.array([str(n) for n in f2_c.names]))
            and np.array_equal(ds_c.proc, f2_c.proc))
    _check(same, "1D и 2D: те же задания в том же порядке, те же параметры", problems)

    # 6. отпечаток
    print(f"\n[5/5] отпечаток против {a.runs or '—'}  [{time.time() - t0:.0f} с]")
    if a.runs and os.path.exists(a.runs):
        w1 = fingerprint(ds_c.y, a.runs, STAGES_1D, "1D stress_1756.npz")
        w2 = fingerprint(f2_c.y, a.runs, STAGES_2D, "2D field2d_1756.npz")
        for w, what in ((w1, "1D"), (w2, "2D")):
            ok = bool(np.isfinite(w) and w < FP_TOL)
            _check(ok, f"отпечаток {what}: макс. отн. расхождение {w:.1e} < {FP_TOL:.0e}", problems)
    else:
        print("  журнал прогонов не найден — отпечаток не проверяется")

    print(f"\nготово за {time.time() - t0:.0f} с; "
          + ("всё совпало" if not problems else f"расхождений: {len(problems)}"))
    return 1 if (problems and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
