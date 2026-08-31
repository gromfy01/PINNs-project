import os
import pickle
from collections import Counter

import numpy as np


def saver(obj, name, path_import):
    out = os.path.join(path_import, name + ".pkl")
    with open(out, "wb") as f:
        pickle.dump(obj, f)
    print("saved in " + out)


def read_rpt_np(path, expected_cols=15):
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            toks = ln.strip().split()
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
    arr = np.asarray(rows, dtype="double")
    if arr.shape[1] != expected_cols:
        raise ValueError(f"{path}: ожидал {expected_cols} колонок, получил {arr.shape[1]}")
    return arr


def preprocessing_res_np(file):
    x_1 = file[:, 1] * 1e3
    y_1 = file[:, 2] * 1e3

    e_rr = file[:, 3]
    e_zz = file[:, 4]
    e_tt = file[:, 5]
    e_rz = 0.5 * file[:, 6]

    s_rr = file[:, 11] / 1e6
    s_zz = file[:, 12] / 1e6
    s_tt = file[:, 13] / 1e6
    s_rz = file[:, 14] / 1e6

    lo = 0.25 * (np.max(y_1) - np.min(y_1))
    hi = 0.75 * (np.max(y_1) - np.min(y_1))
    mask = (y_1 >= lo) & (y_1 <= hi)

    x_sel = x_1[mask]
    y_sel = y_1[mask]

    Stress = np.stack([s_rr[mask], s_tt[mask], s_zz[mask], s_rz[mask]], axis=-1)
    Strain = np.stack([e_rr[mask], e_tt[mask], e_zz[mask], e_rz[mask]], axis=-1)

    # kind="stable": в осевом окне много узлов приходится на совпадающие
    # радиусы. При неустойчивой сортировке их порядок не определён, и если
    # совпадающий радиус попадает на границу чанка, медиана берётся с другого
    # узла. Измерено: бины без совпадающих радиусов воспроизводятся на 100.00 %,
    # бины с ними — на 74.7 %, что и даёт ~7 % расхождения по массиву.
    # См. publication/ERRATA.md, E-21.
    order = np.argsort(x_sel, kind="stable")
    return x_sel[order], y_sel[order], Stress[order], Strain[order]


def average_val_np(arr, nodes=20):
    b = np.zeros(nodes)
    chunks = np.array_split(arr, nodes)
    for i in range(nodes):
        b[i] = np.median(chunks[i])
    return b


def do_rpt_list(path_import):
    names = []
    for fn in os.listdir(path_import):
        if fn.endswith(".rpt"):
            names.append(fn[:-4])
    names.sort()
    return names


def do_preprocessing_rpt(names, path_import):
    out = []
    for nm in names:
        arr = read_rpt_np(os.path.join(path_import, nm + ".rpt"))
        out.append(preprocessing_res_np(arr))
    print(f"  {len(out)} files processed")
    return out


def _split(j):
    return j.split("_")


def r(j):
    s = _split(j)
    for i, t in enumerate(s):
        if t in ("red", "rd"):
            return float(s[i + 1]) / 10000


def c(j):
    s = _split(j)
    for i, t in enumerate(s):
        if t == "cal":
            return float(s[i + 1]) / 100


def f(j):
    s = _split(j)
    for i, t in enumerate(s):
        if t in ("fric", "f"):
            token = s[i + 1].split(".")[0]
            return float(token[1:]) / 1000


def v(j):
    s = _split(j)
    for i, t in enumerate(s):
        if t in ("vel", "v"):
            return int(s[i + 1])


def h(j):
    s = _split(j)
    for i, t in enumerate(s):
        if t == "2a":
            return int(s[i + 1]) / 2


def get_param(cur_job_name, job_list, all_arrays, char_1=2, char_2=0):
    idx = job_list.index(cur_job_name)
    char = average_val_np(all_arrays[idx][char_1][:, char_2], 20)
    return [r(cur_job_name), c(cur_job_name), h(cur_job_name),
            v(cur_job_name), f(cur_job_name), char]


def data_preparer(train_list, train_arrays):
    X_stress, X_strain = [], []
    y_stress, y_strain = [], []
    for char in range(2):
        for comp in range(4):
            X = np.zeros((len(train_list), 5))
            y = np.zeros((len(train_list), 20))
            for i, jn in enumerate(train_list):
                red, cal, ha, vel, fric, val = get_param(
                    jn, train_list, train_arrays, char_1=2 + char, char_2=comp)
                X[i] = [red, cal, ha, fric, vel]
                y[i] = val
            (X_stress if char == 0 else X_strain).append(X)
            (y_stress if char == 0 else y_strain).append(y)
    return (np.array(X_stress), np.array(X_strain),
            np.array(y_stress), np.array(y_strain))
