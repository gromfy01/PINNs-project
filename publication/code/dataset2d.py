"""
dataset2d.py — двумерная (r, z) выборка полей внутри осевого окна.

Нужна для расширенного варианта пункта 2 плана: ввести z во вход сети и считать
ПОЛНУЮ осесимметричную форму уравнений равновесия

    ∂σ_rr/∂r + ∂τ_rz/∂z + (σ_rr − σ_θθ)/r = 0
    ∂τ_rz/∂r + ∂σ_zz/∂z + τ_rz/r         = 0

вместо редуцированной. Аудит показал, что для первого уравнения редукция
законна (отброшено 1.3 %), а для второго — нет (68 %), поэтому осевое уравнение
без z вообще нельзя записать. Здесь z есть.

Отличие от dataset.py: узлы окна сворачиваются не в 20 радиальных бинов, а в
сетку n_z × n_r (медиана в ячейке). Радиус нормируется на радиус поверхности
В ТОМ ЖЕ осевом сечении, поэтому r = 1 — поверхность при любом z. z нормируется
на границы окна: z = 0 — вход окна, z = 1 — выход.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import filter_dataset                          # noqa: E402
from raw_probe import C_R, C_Z, axial_window, list_jobs, parse_job_name, read_rpt  # noqa: E402

COMPONENTS = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")


@dataclass
class Field2D:
    proc: np.ndarray        # (n, 5)
    y: np.ndarray           # (n, n_z, n_r, 4)  МПа
    r: np.ndarray           # (n, n_z, n_r)     нормированный, 1 = поверхность
    z: np.ndarray           # (n, n_z)          нормированный, [0,1] по окну
    r_phys: np.ndarray      # (n, n_z, n_r)     м
    z_phys: np.ndarray      # (n, n_z)          м
    names: np.ndarray
    report: Optional[str] = None

    @property
    def n_sets(self) -> int: return self.proc.shape[0]

    def subset(self, idx: Sequence[int]) -> "Field2D":
        i = np.asarray(idx, dtype=int)
        return Field2D(self.proc[i], self.y[i], self.r[i], self.z[i],
                       self.r_phys[i], self.z_phys[i], self.names[i], self.report)

    def save(self, path): np.savez_compressed(
        path, proc=self.proc, y=self.y, r=self.r, z=self.z,
        r_phys=self.r_phys, z_phys=self.z_phys, names=self.names,
        report=np.array(self.report or ""))

    @staticmethod
    def load(path) -> "Field2D":
        d = np.load(path, allow_pickle=True)
        return Field2D(d["proc"], d["y"], d["r"], d["z"], d["r_phys"], d["z_phys"],
                       d["names"], str(d["report"]))


def grid_one(arr: np.ndarray, n_z: int, n_r: int):
    """Узлы окна → медианы в ячейках сетки n_z × n_r."""
    z = arr[:, C_Z]
    lo, hi = axial_window(z)
    m = (z >= lo) & (z <= hi)
    r_w, z_w = arr[m, C_R], arr[m, C_Z]
    comps = np.stack([arr[m, 11], arr[m, 13], arr[m, 12], arr[m, 14]], -1) / 1e6

    z_edges = np.linspace(z_w.min(), z_w.max(), n_z + 1)
    zi = np.clip(np.digitize(z_w, z_edges) - 1, 0, n_z - 1)
    y = np.full((n_z, n_r, 4), np.nan)
    rp = np.full((n_z, n_r), np.nan)
    zp = 0.5 * (z_edges[:-1] + z_edges[1:])
    for j in range(n_z):
        sel = zi == j
        if sel.sum() < n_r:
            return None
        rr, cc = r_w[sel], comps[sel]
        o = np.argsort(rr)
        chunks_r = np.array_split(rr[o], n_r)
        chunks_c = np.array_split(cc[o], n_r)
        for b in range(n_r):
            rp[j, b] = np.median(chunks_r[b])
            y[j, b] = np.median(chunks_c[b], axis=0)
    if not np.isfinite(y).all() or not np.isfinite(rp).all():
        return None
    return y, rp, zp


def build(root: str, n_z: int = 8, n_r: int = 20, apply_filters: bool = True) -> Field2D:
    jobs = list_jobs(root)
    Y, RP, ZP, PR, NM = [], [], [], [], []
    skipped = 0
    for i, (nm, path) in enumerate(jobs):
        try:
            g = grid_one(read_rpt(path), n_z, n_r)
        except Exception:                                    # noqa: BLE001
            g = None
        if g is None:
            skipped += 1
            continue
        y, rp, zp = g
        pj = parse_job_name(nm)
        Y.append(y); RP.append(rp); ZP.append(zp); NM.append(nm)
        PR.append([pj.get("Q", np.nan), pj.get("k", np.nan), pj.get("alpha", np.nan),
                   pj.get("mu_repo", np.nan), pj.get("v", np.nan)])
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(jobs)}", flush=True)
    Y = np.array(Y); RP = np.array(RP); ZP = np.array(ZP); PR = np.array(PR)
    r = RP / RP[:, :, -1:]
    zspan = (ZP[:, -1] - ZP[:, 0]).reshape(-1, 1)
    z = (ZP - ZP[:, :1]) / np.where(zspan == 0, 1.0, zspan)

    f = Field2D(PR, Y, r, z, RP, ZP, np.array(NM))
    print(f"наборов собрано: {f.n_sets}, пропущено (мало узлов): {skipped}")
    if apply_filters:
        # фильтры по профилю среднего осевого сечения — тот же критерий, что в 1D
        mid = Y[:, Y.shape[1] // 2]
        keep, rep = filter_dataset(PR, mid, drop_removed_indices=False)
        f = f.subset(keep); f.report = rep.to_text()
    return f


def sanity(f: Field2D) -> Dict[str, float]:
    return {"n_sets": float(f.n_sets), "n_z": float(f.y.shape[1]), "n_r": float(f.y.shape[2]),
            "bc_sigma_rr_surface": float(np.abs(f.y[:, :, -1, 0]).mean()),
            "bc_tau_rz_surface": float(np.abs(f.y[:, :, -1, 3]).mean()),
            "tau_rz_axis": float(np.abs(f.y[:, :, 0, 3]).mean()),
            "r_last": float(np.abs(f.r[:, :, -1] - 1).max()),
            "z_span_m": float(np.mean(f.z_phys[:, -1] - f.z_phys[:, 0]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--nr", type=int, default=20)
    a = ap.parse_args()
    f = build(a.root, a.nz, a.nr)
    print(f.report or "")
    for k, v in sanity(f).items():
        print(f"  {k:22s} {v:12.5g}")
    f.save(a.out)
    print(f"сохранено: {a.out}")
