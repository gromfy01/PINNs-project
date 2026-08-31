"""
dataset.py — сборка ФИЗИЧЕСКИ КОРРЕКТНОГО датасета напряжений из сырых .rpt.

Зачем не использовать y_stress.pkl. Проверка по сырым данным показала, что в
нём по плоскостям лежат (S.Max.Prin, S.Mid.Prin, S.Min.Prin, σ_rr) — три
главных напряжения и σ_rr, а не четыре компоненты тензора. τ_rz, σ_θθ и σ_zz
в нём отсутствуют вовсе (ERRATA E-18). Поэтому целевые поля собираются заново
из колонок S.S11 / S.S33 / S.S22 / S.S12.

Что здесь строится:

    proc    (n_sets, 5)        (Q, k, alpha, mu, v)
    y       (n_sets, 20, 4)    (σ_rr, σ_θθ, σ_zz, τ_rz), МПа
    r       (n_sets, 20)       нормированный радиус, r[.., -1] = 1 = поверхность
    r_phys  (n_sets, 20)       физический радиус, м — нужен для равновесия
    names   (n_sets,)          имена заданий Abaqus

Радиальная сетка. Двадцать точек — медианы равнонаселённых бинов по r внутри
осевого окна 25–75 %; сетка почти равномерна, но НЕ точно, поэтому радиусы
хранятся поштучно, а не подразумеваются `linspace(0, 1, 20)`. Нормировка — на
радиус крайнего бина того же набора, так что r = 1 есть свободная поверхность
по построению.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import COL, FilterReport, filter_dataset  # noqa: E402

N_R = 20
COMPONENTS = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")
IDX_SRR, IDX_STT, IDX_SZZ, IDX_TRZ = 0, 1, 2, 3


@dataclass
class StressDataset:
    proc: np.ndarray        # (n, 5)
    y: np.ndarray           # (n, 20, 4)  МПа
    r: np.ndarray           # (n, 20)     нормированный
    r_phys: np.ndarray      # (n, 20)     м
    names: np.ndarray       # (n,)
    report: Optional[str] = None

    @property
    def n_sets(self) -> int:
        return self.proc.shape[0]

    def subset(self, idx: Sequence[int]) -> "StressDataset":
        i = np.asarray(idx, dtype=int)
        return StressDataset(self.proc[i], self.y[i], self.r[i], self.r_phys[i],
                             self.names[i], self.report)

    def rows(self) -> Tuple[np.ndarray, np.ndarray]:
        """(X, Y) в раскладке строк set-major: X — (n·20, 6), Y — (n·20, 4)."""
        n = self.n_sets
        X = np.concatenate([np.repeat(self.proc, N_R, axis=0),
                            self.r.reshape(-1, 1)], axis=1)
        return X.astype(np.float64), self.y.reshape(n * N_R, 4).astype(np.float64)

    def save(self, path: str) -> None:
        np.savez_compressed(path, proc=self.proc, y=self.y, r=self.r,
                            r_phys=self.r_phys, names=self.names,
                            report=np.array(self.report or ""))

    @staticmethod
    def load(path: str) -> "StressDataset":
        d = np.load(path, allow_pickle=True)
        return StressDataset(d["proc"], d["y"], d["r"], d["r_phys"], d["names"],
                             str(d["report"]))


def build_from_allcols(allcols_npz: str, prof_r_npz: str,
                       apply_filters: bool = True) -> StressDataset:
    """
    allcols_npz : выход publication/audit/probe над сырыми .rpt —
                  data (n, 20, 14), params (n, 5), names, cols
    prof_r_npz  : выход run_raw_audit.py — prof_r (n, 20) физические радиусы бинов
    """
    A = np.load(allcols_npz, allow_pickle=True)
    B = np.load(prof_r_npz, allow_pickle=True)
    cols = list(map(str, A["cols"]))
    data, params = A["data"], A["params"]
    names_a = np.array(list(map(str, A["names"])))
    names_b = np.array(list(map(str, B["names"])))
    if not np.array_equal(names_a, names_b):
        order = {n: i for i, n in enumerate(names_b)}
        sel = np.array([order[n] for n in names_a])
        r_phys = B["prof_r"][sel]
    else:
        r_phys = B["prof_r"]

    y = np.stack([data[:, :, cols.index(c)]
                  for c in ("s_rr", "s_tt", "s_zz", "s_rz")], axis=-1)

    r_max = r_phys[:, -1:].copy()
    bad = r_max[:, 0] <= 0
    if bad.any():
        raise ValueError(f"{int(bad.sum())} наборов с нулевым внешним радиусом")
    r = r_phys / r_max

    ds = StressDataset(params.astype(np.float64), y.astype(np.float64),
                       r.astype(np.float64), r_phys.astype(np.float64), names_a)
    if apply_filters:
        keep, rep = filter_dataset(ds.proc, ds.y, drop_removed_indices=False)
        ds = ds.subset(keep)
        ds.report = rep.to_text()
    return ds


def sanity(ds: StressDataset) -> Dict[str, float]:
    """
    Проверки, которые обязаны выполняться на физически корректном датасете.
    Печатать перед каждым прогоном: если traction-free вдруг перестал
    выполняться, значит компоненты снова разъехались.
    """
    outer = np.abs(ds.y[:, -1, :]).mean(axis=0)
    axis_ = np.abs(ds.y[:, 0, :]).mean(axis=0)
    return {
        "n_sets": float(ds.n_sets),
        "bc_sigma_rr_surface": float(outer[IDX_SRR]),
        "bc_tau_rz_surface": float(outer[IDX_TRZ]),
        "sigma_tt_surface": float(outer[IDX_STT]),
        "sigma_zz_surface": float(outer[IDX_SZZ]),
        "tau_rz_axis": float(axis_[IDX_TRZ]),
        "r_last_is_one": float(np.abs(ds.r[:, -1] - 1.0).max()),
        "r_first": float(np.abs(ds.r[:, 0]).max()),
        "tau_scale": float(np.abs(ds.y[:, :, IDX_TRZ]).mean()),
        "normal_scale": float(np.abs(ds.y[:, :, :3]).mean()),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allcols", required=True)
    ap.add_argument("--profr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = build_from_allcols(a.allcols, a.profr)
    print(ds.report)
    print()
    for k, v in sanity(ds).items():
        print(f"  {k:24s} {v:12.5g}")
    ds.save(a.out)
    print(f"\nсохранено: {a.out}  ({ds.n_sets} наборов)")
