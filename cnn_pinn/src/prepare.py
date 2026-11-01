"""
prepare.py — подготовка данных для пробного прогона (ТЗ, п. 3) из сырых .rpt.

Из каждого отчёта Abaqus берутся узлы проволоки (COORD, S11..S12), по ним
находится установившаяся зона, в ней вырезается окно длиной W·R по оси, и поля
интерполируются на регулярную сетку ρ = r/R ∈ [0, 1], ζ = (z − z₀)/R ∈ [0, W]:
линейно, у краёв — ближайший узел. R — измеренный радиус поверхности в зоне.

Отбраковка (порядок как в предложении к ТЗ, п. Т13):
  * метка μ вне плана (опечатка в имени задания);
  * незавершённый проход: на проволоке остался непротянутый участок (r_max > R₀ − 0.1 мм);
  * Q_d > Q_max (по умолчанию уровень 0.25 исключён);
  * утонение после волоки больше 0.5 мм: R_ном − R, R_ном = R₀(1 − Q_d);
  * установившаяся зона короче окна;
  * группы с побитово одинаковым полем при разных μ — целиком (истинное μ неизвестно).
Помечаются, но остаются: утонение больше 0.1 мм и динамическая осевая сила
(RMS средней по сечению σ_z = 2∫σ_zρdρ больше 10 МПа).

Запуск:
    python src/prepare.py --raw <Raw_Data> --out data/trial.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "publication", "code"))
from raw_probe import billet_radius, list_jobs, parse_job_name, read_rpt  # noqa: E402

MU_LEVELS = (0.025, 0.05, 0.1)
COMPONENTS = ("sigma_r", "sigma_theta", "sigma_z", "tau_rz")


def steady_zone(r, z, szz, tol_r=0.05, tol_abs=15.0, tol_rel=0.05, bin_mm=3.0):
    """
    Наибольший непрерывный участок по z, где радиус поверхности постоянен
    (в пределах tol_r, мм), а σ_z на поверхности и на оси отклоняется от своей
    медианы не больше чем на max(tol_abs, tol_rel·|медиана|). Медианы берутся по
    средней части проволоки [0.2, 0.65]·L, где поле обычно установилось.
    Возвращает (z_a, z_b, R) в мм или None.
    """
    z0, z1 = z.min(), z.max()
    L = z1 - z0
    ax = np.abs(r) < 0.05
    za, sa = z[ax], szz[ax]
    o = np.argsort(za)
    za, sa = za[o], sa[o]
    # узел поверхности в каждом бине по z — узел с наибольшим r
    edges = np.arange(z0, z1 + bin_mm, bin_mm)
    idx = np.digitize(z, edges)
    zs, rs, ss = [], [], []
    for b in np.unique(idx):
        m = np.flatnonzero(idx == b)
        k = m[np.argmax(r[m])]
        zs.append(z[k]); rs.append(r[k]); ss.append(szz[k])
    zs, rs, ss = map(np.asarray, (zs, rs, ss))
    mid = lambda zz: (zz >= z0 + 0.2 * L) & (zz <= z0 + 0.65 * L)  # noqa: E731
    if mid(zs).sum() < 3 or mid(za).sum() < 3:
        return None
    R_med = np.median(rs[mid(zs)])
    s_med = np.median(ss[mid(zs)])
    a_med = np.median(sa[mid(za)])
    rs_a = np.interp(za, zs, rs)
    ss_a = np.interp(za, zs, ss)
    ok = ((np.abs(rs_a - R_med) <= tol_r)
          & (np.abs(ss_a - s_med) <= max(tol_abs, tol_rel * abs(s_med)))
          & (np.abs(sa - a_med) <= max(tol_abs, tol_rel * abs(a_med))))
    best, cur = (0, -1), None
    for i, f in enumerate(ok):
        if f and cur is None:
            cur = i
        if (not f or i == len(ok) - 1) and cur is not None:
            j = i if f else i - 1
            if za[j] - za[cur] > za[best[1]] - za[best[0]] if best[1] >= 0 else True:
                best = (cur, j)
            cur = None
    if best[1] < 0:
        return None
    a, b = za[best[0]], za[best[1]]
    inz = (zs >= a) & (zs <= b)
    R = np.median(rs[inz]) if inz.any() else R_med
    return float(a), float(b), float(R)


def to_grid(r, z, fields, R, z_start, W, nr, nz):
    """Узловые поля → регулярная сетка (nr × nz) в координатах ρ = r/R, ζ = (z − z_start)/R."""
    m = (z >= z_start - 3.0) & (z <= z_start + W * R + 3.0)
    pts = np.column_stack([r[m] / R, (z[m] - z_start) / R])
    rho = np.linspace(0.0, 1.0, nr)
    zeta = np.linspace(0.0, W, nz)
    G = np.stack(np.meshgrid(rho, zeta, indexing="ij"), axis=-1).reshape(-1, 2)
    out = np.empty((len(fields), nr, nz))
    for c, f in enumerate(fields):
        lin = LinearNDInterpolator(pts, f[m])(G)
        bad = np.isnan(lin)
        if bad.any():
            lin[bad] = NearestNDInterpolator(pts, f[m])(G[bad])
        out[c] = lin.reshape(nr, nz)
    return out


def section_force(y, nr):
    """Средняя по сечению σ_z = 2∫₀¹σ_zρdρ для каждого столбца ζ, МПа."""
    rho = np.linspace(0.0, 1.0, nr)
    return 2.0 * np.trapezoid(y[2] * rho[:, None], rho, axis=0)


def process(args):
    name, path, W, nr, nz = args
    rec = {"name": name}
    p = parse_job_name(name)
    rec.update(Q_d=p["Q"], k=p["k"], alpha=p["alpha"], mu=p["mu_repo"], v=p["v"])
    arr = read_rpt(path)
    rec["md5"] = hashlib.md5(arr.tobytes()).hexdigest()
    r, z = arr[:, 1] * 1e3, arr[:, 2] * 1e3
    srr, szz, stt, srz = (arr[:, c] / 1e6 for c in (11, 12, 13, 14))
    R0 = billet_radius(path) * 1e3
    rec.update(R0=R0, r_max=float(r.max()), L=float(z.max() - z.min()))
    sz = steady_zone(r, z, szz)
    if sz is None:
        rec["zone"] = None
        return rec, None
    a, b, R = sz
    rec.update(zone=[a, b], R=R, zone_len_R=(b - a) / R, thinning=R0 * (1 - p["Q"]) - R)
    if (b - a) < W * R:
        return rec, None
    z_start = 0.5 * (a + b) - 0.5 * W * R
    rec["z_start"] = z_start
    y = to_grid(r, z, (srr, stt, szz, srz), R, z_start, W, nr, nz)
    f = section_force(y, nr)
    rec["axial_rms"] = float(np.sqrt(np.mean(f ** 2)))
    return rec, y.astype(np.float32)


def features(Q_d, k, alpha_deg, mu, v):
    """Признаки режима (ТЗ, п. 2.1 и п. 3 шаг 5, с пересчётами из вопроса Б4)."""
    Q = 1.0 - (1.0 - Q_d) ** 2                 # обжатие по площади
    k_spec = 2.0 * k / (1.0 - Q_d)             # l_к / R_вых при поясе k·D₀
    a = np.deg2rad(alpha_deg)
    delta = a / Q * (1.0 + np.sqrt(1.0 - Q)) ** 2
    return np.column_stack([Q, k_spec, a, mu, np.log(v), np.log(delta)])


FEATURE_NAMES = ("Q_area", "l_k/R_exit", "alpha_rad", "mu", "log_v", "log_Delta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=2.0, help="длина окна W в радиусах")
    ap.add_argument("--nr", type=int, default=32)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--q-max", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    jobs = [(n, p, a.window, a.nr, a.nz) for n, p in list_jobs(a.raw)]
    with Pool(a.workers) as pool:
        res = pool.map(process, jobs, chunksize=8)
    recs = [r for r, _ in res]

    # группы с побитово одинаковым полем при разных метках μ
    by_md5 = {}
    for r in recs:
        by_md5.setdefault(r["md5"], []).append(r["name"])
    mu_dup = {n for g in by_md5.values() if len(g) > 1 for n in g}

    reasons, keep = {}, []
    for (rec, y) in res:
        n = rec["name"]
        if not any(abs(rec["mu"] - m) < 1e-9 for m in MU_LEVELS):
            reasons[n] = "mu_typo"
        elif rec["r_max"] > rec["R0"] - 0.1:
            reasons[n] = "unfinished"
        elif rec["Q_d"] > a.q_max + 1e-9:
            reasons[n] = "q_out_of_scope"
        elif n in mu_dup:
            reasons[n] = "mu_duplicate_group"
        elif rec.get("zone") is None:
            reasons[n] = "no_steady_zone"
        elif rec["thinning"] > 0.5:
            reasons[n] = "thinning_gt_0.5mm"
        elif y is None:
            reasons[n] = "zone_shorter_than_window"
        else:
            keep.append((rec, y))

    counts = {}
    for v in reasons.values():
        counts[v] = counts.get(v, 0) + 1
    P = np.array([[r["Q_d"], r["k"], r["alpha"], r["mu"], r["v"]] for r, _ in keep])
    Y = np.stack([y for _, y in keep])
    meta = {
        "n_raw": len(recs), "n_kept": len(keep), "rejected": counts,
        "window_R": a.window, "grid": [a.nr, a.nz], "components": COMPONENTS,
        "feature_names": FEATURE_NAMES, "flags": {
            "thinning_gt_0.1mm": int(sum(r["thinning"] > 0.1 for r, _ in keep)),
            "axial_rms_gt_10MPa": int(sum(r["axial_rms"] > 10 for r, _ in keep))},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, Y=Y, P=P, X=features(*P.T),
        names=np.array([r["name"] for r, _ in keep]),
        R=np.array([r["R"] for r, _ in keep]),
        thinning=np.array([r["thinning"] for r, _ in keep]),
        axial_rms=np.array([r["axial_rms"] for r, _ in keep]),
        zone_len_R=np.array([r["zone_len_R"] for r, _ in keep]),
        meta=json.dumps(meta, ensure_ascii=False))
    with open(os.path.splitext(a.out)[0] + "_report.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rejected": reasons,
                   "zone_len_R_all": {r["name"]: r.get("zone_len_R") for r in recs}},
                  fh, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
