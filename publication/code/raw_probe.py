"""
raw_probe.py — независимая перепроверка препроцессинга по сырым Abaqus .rpt.

Задача модуля: не доверять ни рукописи, ни docs/, ни старому handover,
а восстановить факты прямо из источника:

  * какая колонка .rpt — радиус, какая — ось (COOR1 / COOR2);
  * что на самом деле означает окно «25–75 %» (радиальное или осевое);
  * какой радиальный диапазон покрывают 20 точек профиля
    (⇒ где физически находится свободная поверхность в нормированных r);
  * какая перестановка плоскостей даёт физический порядок компонент;
  * уровни μ по именам заданий;
  * отношение |∂/∂z| к |∂/∂r| — численное обоснование редуцированной
    формы уравнений равновесия (пункт 2 плана).

Раскладка колонок .rpt (подтверждена по заголовку Field Output Report):

    0  Node Label
    1  COORD.COOR1     r, м
    2  COORD.COOR2     z, м
    3  LE.LE11         ε_rr        (CAX: 1=r, 2=z, 3=θ)
    4  LE.LE22         ε_zz
    5  LE.LE33         ε_θθ
    6  LE.LE12         γ_rz  (инженерный)
    7  S.Mises
    8  S.Max. Prin
    9  S.Mid. Prin
   10  S.Min. Prin
   11  S.S11           σ_rr, Па
   12  S.S22           σ_zz, Па
   13  S.S33           σ_θθ, Па
   14  S.S12           τ_rz, Па
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

EXPECTED_COLS = 15

C_R, C_Z = 1, 2
C_SRR, C_SZZ, C_STT, C_SRZ = 11, 12, 13, 14
C_ERR, C_EZZ, C_ETT, C_ERZ = 3, 4, 5, 6

#: порядок, в котором компоненты укладываются в ось `plane`
#: ровно так, как это делает preprocessing/rpt_utils.py::preprocessing_res_np
STACK_ORDER = ("s_rr", "s_tt", "s_zz", "s_rz")


def read_rpt(path: str) -> np.ndarray:
    """Числовые строки .rpt → (n_nodes, 15). Копия read_rpt_np без исключений."""
    rows: List[List[float]] = []
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            toks = ln.split()
            if not toks:
                continue
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: пусто")
    ncols, _ = Counter(len(r) for r in rows).most_common(1)[0]
    rows = [r for r in rows if len(r) == ncols]
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != EXPECTED_COLS:
        raise ValueError(f"{path}: ожидал {EXPECTED_COLS} колонок, получил {arr.shape[1]}")
    return arr


# ─────────────────────────── имена заданий ──────────────────────────────

def parse_job_name(name: str) -> Dict[str, float]:
    """Параметры процесса из имени задания (аналог r/c/f/v/h в rpt_utils)."""
    s = name.split("_")
    out: Dict[str, float] = {}
    for i, t in enumerate(s):
        if t in ("red", "rd"):
            out["Q"] = float(s[i + 1]) / 10000
        elif t == "cal":
            out["k"] = float(s[i + 1]) / 100
        elif t in ("vel", "v"):
            out["v"] = float(s[i + 1])
        elif t == "2a":
            out["alpha"] = float(s[i + 1]) / 2
        elif t in ("fric", "f"):
            tok = s[i + 1].split(".")[0]
            out["mu_repo"] = float(tok[1:]) / 1000     # правило rpt_utils.f()
            out["fric_token"] = tok
    return out


def mu_from_token(token: str) -> float:
    """
    Правило rpt_utils.f(): снять первый символ, поделить на 1000.
      '025'  → 25/1000  = 0.025
      '050'  → 50/1000  = 0.050
      '0100' → 100/1000 = 0.100
      '02'   → 2/1000   = 0.002   ← опечатка в имени задания, не уровень фактора
    """
    return float(token[1:]) / 1000


# ─────────────────────── окно и радиальный профиль ──────────────────────

def axial_window(z: np.ndarray, mode: str = "repo") -> Tuple[float, float]:
    """
    Осевое окно 25–75 %.

    mode='repo'   — ровно то, что делает preprocessing/rpt_utils.py:
                      lo = 0.25 * (zmax - zmin);  hi = 0.75 * (zmax - zmin)
                    порог берётся от РАЗМАХА, а не от zmin — если zmin != 0,
                    окно съезжает (см. publication/audit/).
    mode='fixed'  — корректная форма: zmin + 0.25/0.75 * (zmax - zmin).
    """
    zmin, zmax = float(np.min(z)), float(np.max(z))
    span = zmax - zmin
    if mode == "repo":
        return 0.25 * span, 0.75 * span
    if mode == "fixed":
        return zmin + 0.25 * span, zmin + 0.75 * span
    raise ValueError(mode)


def median_bins(values: np.ndarray, nodes: int = 20) -> np.ndarray:
    """Копия average_val_np: разбить на `nodes` последовательных чанков, медиана."""
    return np.array([np.median(ch) for ch in np.array_split(values, nodes)])


def profile_from_rpt(arr: np.ndarray, nodes: int = 20, window: str = "repo"):
    """
    Воспроизводит preprocessing_res_np + average_val_np и возвращает

        r_prof   (nodes,)              медианный радиус каждого бина, м
        stress   (nodes, 4)            в порядке STACK_ORDER, МПа
        strain   (nodes, 4)            (ε_rr, ε_θθ, ε_zz, ε_rz)
        meta     dict                  диапазоны координат и размер окна
    """
    r = arr[:, C_R]
    z = arr[:, C_Z]
    lo, hi = axial_window(z, window)
    mask = (z >= lo) & (z <= hi)

    r_sel, z_sel = r[mask], z[mask]
    s = np.stack([arr[mask, C_SRR], arr[mask, C_STT],
                  arr[mask, C_SZZ], arr[mask, C_SRZ]], axis=-1) / 1e6
    e = np.stack([arr[mask, C_ERR], arr[mask, C_ETT],
                  arr[mask, C_EZZ], 0.5 * arr[mask, C_ERZ]], axis=-1)

    order = np.argsort(r_sel)
    r_sel, z_sel, s, e = r_sel[order], z_sel[order], s[order], e[order]

    r_prof = median_bins(r_sel, nodes)
    stress = np.stack([median_bins(s[:, c], nodes) for c in range(4)], axis=-1)
    strain = np.stack([median_bins(e[:, c], nodes) for c in range(4)], axis=-1)

    meta = {
        "r_min_all": float(r.min()), "r_max_all": float(r.max()),
        "z_min_all": float(z.min()), "z_max_all": float(z.max()),
        "r_min_win": float(r_sel.min()), "r_max_win": float(r_sel.max()),
        "z_min_win": float(z_sel.min()), "z_max_win": float(z_sel.max()),
        "n_nodes_all": int(arr.shape[0]), "n_nodes_win": int(mask.sum()),
        "win_lo": lo, "win_hi": hi,
    }
    return r_prof, stress, strain, meta


def list_jobs(root: str) -> List[Tuple[str, str]]:
    """[(job_name, abs_path)] по всем .rpt под root, отсортировано."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.endswith(".rpt"):
                out.append((fn[:-4], os.path.join(dirpath, fn)))
    out.sort(key=lambda t: t[0])
    return out
