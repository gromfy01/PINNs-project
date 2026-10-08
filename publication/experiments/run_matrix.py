"""
run_matrix.py — прогон экспериментальной матрицы статьи.

Все стадии пишут ОДИН csv построчно, по мере готовности: прогон долгий, и
частичный результат должен оставаться пригодным. Повторный запуск с тем же
--out пропускает уже посчитанные строки (ключ: stage+family+split+seed+frac+rate).

Стадии:
  tune      подбор гиперпараметров, один бюджет на все семейства (пункт 4)
  main      6 регионов × контроли × сиды (пункты 1 и 3)
  curves    кривые обучения по объёму данных (пункт 5)
  corrupt   устойчивость к порче меток (пункт 6)
  backend   torch против jax при идентичном протоколе (замечание Reviewer #2)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

FIELDS = ["stage", "family", "backend", "split", "seed", "frac", "corrupt_rate",
          "n_train", "n_test", "epochs", "seconds",
          "macro_rmse", "macro_nrmse", "macro_r2", "rmse_normal_only",
          "rmse_sigma_rr", "rmse_sigma_tt", "rmse_sigma_zz", "rmse_tau_rz",
          "nrmse_sigma_rr", "nrmse_sigma_tt", "nrmse_sigma_zz", "nrmse_tau_rz",
          "r2_sigma_rr", "r2_sigma_tt", "r2_sigma_zz", "r2_tau_rz",
          "eq_res_median", "bc_sigma_rr", "bc_tau_rz",
          "lambda_physics", "lambda_bc", "learning_rate", "n_units", "n_layers",
          "eq_r_reduced", "eq_z_full", "eq_z_reduced", "phys_gain",
          "macro_nrmse_global", "macro_r2_global", "r2_normal_only",
          "r2_normal_only_global",
          "nrmse_sigma_rr_global", "nrmse_sigma_tt_global",
          "nrmse_sigma_zz_global", "nrmse_tau_rz_global",
          "error"]


def key_of(row: Dict) -> Tuple:
    return (row.get("stage"), row.get("family"), row.get("backend"), row.get("split"),
            str(row.get("seed")), str(row.get("frac")), str(row.get("corrupt_rate")))


def load_done(path: str) -> set:
    """
    Ключи уже посчитанных прогонов.

    Читает файл терпимо к РАЗНЫМ схемам строк: набор колонок расширялся между
    стадиями, а файл дописывается, и заголовок остаётся от первой записи. При
    наивном DictReader поле `error` у строк новой схемы попадало на чужую
    колонку, оказывалось непустым, и такие прогоны считались неудачными — то
    есть возобновление пересчитывало бы уже посчитанное.
    """
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return set()
    header = rows[0]
    schemas = {len(header): header, len(FIELDS): list(FIELDS)}
    done = set()
    for r in rows[1:]:
        cols = schemas.get(len(r))
        if cols is None:
            continue
        rec = dict(zip(cols, r))
        if not rec.get("error"):
            done.add(key_of(rec))
    return done


# ───────────────────────────── один прогон ──────────────────────────────

def one_run_2d(job: Dict) -> Dict:
    """Стадия twod: ablation по форме уравнений равновесия с z во входе."""
    import torch
    torch.set_num_threads(1)
    from dataset2d import Field2D
    from trainer2d import Config2D, Trainer2D, equilibrium_audit_2d

    row = {k: "" for k in FIELDS}
    row.update({k: job.get(k, "") for k in
                ("stage", "family", "backend", "split", "seed", "frac", "corrupt_rate")})
    try:
        f = Field2D.load(job["dataset"])
        tr = np.asarray(job["train_sets"], dtype=int)
        te = np.asarray(job["test_sets"], dtype=int)
        cfg = Config2D(family=job["family"], seed=int(job["seed"]),
                       max_epochs=int(job.get("max_epochs", 300)),
                       patience=int(job.get("patience", 30)), **job.get("hp", {}))
        t0 = time.time()
        T = Trainer2D(cfg)
        res = T.fit(f.proc[tr], f.y[tr], f.r[tr], f.z[tr], f.r_phys[tr], f.z_phys[tr])
        secs = time.time() - t0
        gstd = f.y.reshape(-1, 4).std(axis=0)
        m = T.evaluate(f.proc[te], f.y[te], f.r[te], f.z[te], global_std=gstd)
        pred = T.predict(f.proc[te], f.r[te], f.z[te])
        eq = equilibrium_audit_2d(pred, f.r_phys[te], f.z_phys[te])
        surf = T.predict_surface(f.proc[te], f.z[te])
        row.update(m)
        row.update({"phys_gain": round(getattr(T, "phys_gain", 1.0), 5),
                    "eq_res_median": eq["eq_r_full"],
                    "eq_r_reduced": eq["eq_r_reduced"],
                    "eq_z_full": eq["eq_z_full"], "eq_z_reduced": eq["eq_z_reduced"],
                    "bc_sigma_rr": float(np.abs(surf[:, 0]).mean()),
                    "bc_tau_rz": float(np.abs(surf[:, 3]).mean()),
                    "n_train": len(tr), "n_test": len(te),
                    "epochs": res.epochs_run, "seconds": round(secs, 1),
                    "lambda_physics": cfg.lambda_physics, "lambda_bc": cfg.lambda_bc,
                    "learning_rate": cfg.learning_rate, "n_units": cfg.n_units,
                    "n_layers": cfg.n_layers})
    except Exception:                                   # noqa: BLE001
        row["error"] = traceback.format_exc()[-500:]
    return row


def one_run(job: Dict) -> Dict:
    import torch
    torch.set_num_threads(1)
    from dataset import StressDataset
    from trainer import Config, Trainer, bc_audit, equilibrium_audit

    row = {k: "" for k in FIELDS}
    row.update({k: job.get(k, "") for k in
                ("stage", "family", "backend", "split", "seed", "frac", "corrupt_rate")})
    try:
        ds = StressDataset.load(job["dataset"])
        tr = np.asarray(job["train_sets"], dtype=int)
        te = np.asarray(job["test_sets"], dtype=int)

        y_tr = ds.y[tr]
        if job.get("corrupt_sets") is not None and len(job["corrupt_sets"]):
            from corruption import CorruptionPlan, apply_corruption
            plan = CorruptionPlan(float(job["corrupt_rate"]), int(job["corrupt_component"]),
                                  float(job["corrupt_offset"]),
                                  np.asarray(job["corrupt_sets"], dtype=int),
                                  int(job["seed"]))
            y_tr = apply_corruption(y_tr, plan, set_index=tr)

        hp = job.get("hp", {})
        cfg = Config(family=job["family"], seed=int(job["seed"]),
                     max_epochs=int(job.get("max_epochs", 400)),
                     patience=int(job.get("patience", 40)), **hp)

        t0 = time.time()
        if job.get("backend", "torch") == "jax":
            from jax_twin import JaxTrainer
            T = JaxTrainer(cfg)
        else:
            T = Trainer(cfg)
        res = T.fit(ds.proc[tr], y_tr, ds.r[tr], ds.r_phys[tr])
        secs = time.time() - t0

        gstd = ds.y.reshape(-1, 4).std(axis=0)
        m = T.evaluate(ds.proc[te], ds.y[te], ds.r[te], global_std=gstd)
        pred = T.predict(ds.proc[te], ds.r[te])
        eq = equilibrium_audit(pred, ds.r_phys[te])
        bc = bc_audit(T.predict_at(ds.proc[te], 1.0))

        row.update(m); row.update(eq); row.update(bc)
        row.update({"phys_gain": round(getattr(res, "phys_gain", 1.0), 5),
                    "n_train": len(tr), "n_test": len(te),
                    "epochs": res.epochs_run, "seconds": round(secs, 1),
                    "lambda_physics": cfg.lambda_physics, "lambda_bc": cfg.lambda_bc,
                    "learning_rate": cfg.learning_rate, "n_units": cfg.n_units,
                    "n_layers": cfg.n_layers})
    except Exception:                                   # noqa: BLE001
        row["error"] = traceback.format_exc()[-500:]
    return row


# ─────────────────────────── построение заданий ─────────────────────────

def build_jobs(stage: str, ds, suite, hp_by_family: Dict[str, Dict],
               families: List[str], seeds: List[int], dataset_path: str,
               splits: Optional[List[str]] = None) -> List[Dict]:
    from corruption import DEFAULT_COMPONENT, DEFAULT_OFFSET_MPA, make_corruption_plans
    from learning_curves import nested_subsamples

    jobs: List[Dict] = []
    if stage == "main":
        names = splits or list(suite)
        for name, fam, sd in itertools.product(names, families, seeds):
            sp = suite[name]
            jobs.append(dict(stage="main", family=fam, backend="torch", split=name, seed=sd,
                             frac="", corrupt_rate="", dataset=dataset_path,
                             train_sets=sp.train_sets.tolist(), test_sets=sp.test_sets.tolist(),
                             hp=hp_by_family[fam]))
    elif stage == "curves":
        for name in (splits or ["random", "extrap:alpha_max"]):
            sp = suite[name]
            subs = nested_subsamples(sp.train_sets, seed=42)
            for f, fam, sd in itertools.product(sorted(subs), families, seeds):
                jobs.append(dict(stage="curves", family=fam, backend="torch", split=name,
                                 seed=sd, frac=f, corrupt_rate="", dataset=dataset_path,
                                 train_sets=subs[f].tolist(), test_sets=sp.test_sets.tolist(),
                                 hp=hp_by_family[fam]))
    elif stage == "corrupt":
        for name in (splits or ["random"]):
            sp = suite[name]
            plans = make_corruption_plans(sp.train_sets, seed=42)
            for rate, fam, sd in itertools.product(sorted(plans), families, seeds):
                p = plans[rate]
                jobs.append(dict(stage="corrupt", family=fam, backend="torch", split=name,
                                 seed=sd, frac="", corrupt_rate=rate, dataset=dataset_path,
                                 train_sets=sp.train_sets.tolist(),
                                 test_sets=sp.test_sets.tolist(),
                                 corrupt_sets=p.corrupted_sets.tolist(),
                                 corrupt_component=DEFAULT_COMPONENT,
                                 corrupt_offset=DEFAULT_OFFSET_MPA,
                                 hp=hp_by_family[fam]))
    elif stage == "twod":
        for name in (splits or ["random", "extrap:alpha_max"]):
            sp = suite[name]
            for fam, sd in itertools.product(families, seeds):
                jobs.append(dict(stage="twod", family=fam, backend="torch", split=name,
                                 seed=sd, frac="", corrupt_rate="", dataset=dataset_path,
                                 train_sets=sp.train_sets.tolist(),
                                 test_sets=sp.test_sets.tolist(),
                                 hp=hp_by_family.get(fam, {})))
    elif stage == "backend":
        for name in (splits or ["random", "extrap:alpha_max"]):
            sp = suite[name]
            for be, sd in itertools.product(("torch", "jax"), seeds):
                jobs.append(dict(stage="backend", family="pinn", backend=be, split=name,
                                 seed=sd, frac="", corrupt_rate="", dataset=dataset_path,
                                 train_sets=sp.train_sets.tolist(),
                                 test_sets=sp.test_sets.tolist(),
                                 hp=hp_by_family["pinn"]))
    else:
        raise ValueError(stage)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["main", "curves", "corrupt", "backend", "twod"])
    ap.add_argument("--hp", default="", help="json с гиперпараметрами по семействам")
    ap.add_argument("--families", default="mlp,pinn,vpinn")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--splits", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=400)
    a = ap.parse_args()

    import json
    from dataset import StressDataset
    from splits import build_split_suite

    if a.stage == "twod":
        from dataset2d import Field2D
        ds = Field2D.load(a.dataset)
    else:
        ds = StressDataset.load(a.dataset)
    suite = build_split_suite(ds.proc)
    families = a.families.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    splits = [s for s in a.splits.split(",") if s] or None
    hp = json.load(open(a.hp)) if a.hp else {f: {} for f in families}
    for f in families:
        hp.setdefault(f, {})

    jobs = build_jobs(a.stage, ds, suite, hp, families, seeds, a.dataset, splits)
    for j in jobs:
        j["max_epochs"] = a.max_epochs
    done = load_done(a.out)
    n_all = len(jobs)
    jobs = [j for j in jobs if key_of(j) not in done]
    print(f"стадия {a.stage}: заданий к запуску {len(jobs)}, "
          f"пропущено уже посчитанных {n_all - len(jobs)} "
          f"(в файле готовых прогонов: {len(done)})")
    if not jobs:
        return

    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        t0 = time.time()
        fn = one_run_2d if a.stage == "twod" else one_run
        with Pool(a.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(fn, jobs), 1):
                w.writerow(row); fh.flush()
                el = time.time() - t0
                tag = row.get("error", "")[:60].replace("\n", " ")
                print(f"[{i}/{len(jobs)}] {row['stage']}/{row['family']}/{row['backend']}/"
                      f"{row['split']}/s{row['seed']} "
                      f"rmse={row['macro_rmse'] if not tag else 'ОШИБКА ' + tag} "
                      f"({el/60:.1f} мин, осталось ~{el/i*(len(jobs)-i)/60:.0f} мин)",
                      flush=True)


if __name__ == "__main__":
    main()
