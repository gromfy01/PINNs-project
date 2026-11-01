"""
dump_predictions.py — предсказания одной двумерной конфигурации на тестовых
наборах сплита, в npz для contours.py и residual_maps.py.

Обучает одну конфигурацию (семейство, сплит, сид) тем же порядком, что
run_matrix.one_run_2d на стадии twod: Field2D, build_split_suite(proc),
Config2D(family, seed, max_epochs, patience, **hp), Trainer2D.fit на обучающих
наборах, Trainer2D.predict на тестовых. Пишет npz с ключами

    y_true     (n, n_z, n_r, 4)  поля МКЭ тестовых наборов, МПа
    y_pred     (n, n_z, n_r, 4)  предсказание модели, МПа
    test_sets  (n,)              индексы тестовых наборов в Field2D (--dataset)
    proc       (n, 5)            [Q, k, α, μ, v]
    r          (n, n_z, n_r)     r/R, 1 = свободная поверхность
    z          (n, n_z)          z/L, доля осевого окна
    family     ()                mlp2d, pinn2d_reduced или pinn2d_full

и справочными split, seed, n_train, epochs, macro_rmse, hp (json-строка).
Компоненты по последней оси — (σ_rr, σ_θθ, σ_zz, τ_rz).

Этот файл читают contours.py (--mode model, fields, errors, compare,
compare-errors) и residual_maps.py. residual_maps.py получает через --field2d
тот же Field2D, что здесь --dataset, и по test_sets сверяет, что истина в npz
совпадает с его данными; поэтому оба скрипта должны получать один и тот же
файл набора.

Происхождение рисунков статьи. Карты полей и невязок на рисунках 5 и 6
построены по более ранней односидовой тренировке трёх семейств на 1498
обучающих наборах с общим тестом из 261 набора (набор из 1848 случаев,
очищенный только маской окна, и его случайное разбиение); предсказания той
тренировки не сохранились. Этот скрипт позволяет построить такие же рисунки
заново по свежей тренировке одного сида на наборе из 1756 случаев, на котором
посчитаны все прогоны стадии twod в results/runs_v2.csv. Числа на картах будут
близки к опубликованным, но совпадать с ними не обязаны.

Набор из 1756 случаев: python experiments/fem_row_2d.py --field2d
field2d_1848.npz --save-field2d field2d_1756.npz (или --raw вместо --field2d).

Гиперпараметры по умолчанию — те, с которыми получена стадия twod в
results/runs_v2.csv (колонки n_layers, n_units, learning_rate, lambda_physics,
lambda_bc): n_layers = 3, n_units = 96, learning_rate = 2e-3, у mlp2d
lambda_physics = lambda_bc = 0; max_epochs = 300, patience = 30, один поток
torch на прогон. --hp заменяет их json в формате run_matrix.py
({семейство: {поле Config2D: значение}}). Если в runs_v2.csv есть прогон с той
же конфигурацией, его macro-RMSE печатается рядом со свежим для сверки.

Пример (из корня пакета; рисунки 5 и 6):
    for fam in mlp2d pinn2d_reduced pinn2d_full; do
        python experiments/dump_predictions.py --dataset field2d_1756.npz \\
            --family $fam --split random --seed 0 --out pred_$fam.npz
    done
    python experiments/contours.py --mode compare --paper \\
        --npzs pred_mlp2d.npz,pred_pinn2d_reduced.npz,pred_pinn2d_full.npz \\
        --labels "MLP,PINN reduced,PINN full" --out fig5_compare_fields.png
    python experiments/residual_maps.py --paper --field2d field2d_1756.npz \\
        --npzs pred_mlp2d.npz,pred_pinn2d_reduced.npz,pred_pinn2d_full.npz \\
        --labels "MLP,PINN reduced,PINN full" --out fig6_residual_maps.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

FAMILIES_2D = ("mlp2d", "pinn2d_reduced", "pinn2d_full")

# Гиперпараметры стадии twod в results/runs_v2.csv (колонки журнала).
HP_TWOD = {"mlp2d": {"n_layers": 3, "n_units": 96, "learning_rate": 2e-3,
                     "lambda_physics": 0.0, "lambda_bc": 0.0},
           "pinn2d_reduced": {"n_layers": 3, "n_units": 96, "learning_rate": 2e-3},
           "pinn2d_full": {"n_layers": 3, "n_units": 96, "learning_rate": 2e-3}}


def recorded_run(runs_csv: str, family: str, split: str, seed: int) -> Optional[Dict[str, str]]:
    """Строка стадии twod журнала с той же конфигурацией, если она есть."""
    if not runs_csv or not os.path.exists(runs_csv):
        return None
    with open(runs_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("stage") == "twod" and row.get("family") == family
                    and row.get("split") == split and str(row.get("seed")) == str(seed)
                    and not row.get("error")):
                return row
    return None


def removed_present(names, removed_csv: str) -> int:
    """
    Сколько из 92 снятых заданий removed_1848_to_1756.csv есть в наборе: 89
    незавершённых расчётов, выявленных маской окна (разброс радиуса поверхности
    > 0.1 мм), и 3 завершённых расчёта при Q = 1.5 %, v = 250 м/мин, снятых
    вместе с ними более ранней проверкой хвоста (выпучивание головной части
    проволоки 17.82–17.88 мм).
    """
    if not removed_csv or not os.path.exists(removed_csv):
        return 0
    with open(removed_csv, encoding="utf-8") as fh:
        drop = {r["job"] for r in csv.DictReader(fh)}
    return len(drop & {str(n) for n in names})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Field2D .npz (dataset2d.py)")
    ap.add_argument("--family", required=True, choices=FAMILIES_2D)
    ap.add_argument("--split", default="random", help="имя сплита build_split_suite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="выходной .npz")
    ap.add_argument("--hp", default="",
                    help="json {семейство: {поле Config2D: значение}}; "
                         "по умолчанию — гиперпараметры стадии twod runs_v2.csv")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1, help="потоков torch")
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "results", "runs_v2.csv"),
                    help="журнал прогонов для сверки macro-RMSE")
    ap.add_argument("--removed", default=os.path.join(HERE, "..", "audit",
                                                      "removed_1848_to_1756.csv"))
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from dataset2d import Field2D
    from splits import build_split_suite
    from trainer2d import Config2D, Trainer2D

    f = Field2D.load(a.dataset)
    n_bad = removed_present(f.names, a.removed)
    if n_bad:
        print(f"⚠ в наборе {n_bad} снятых заданий из {os.path.basename(a.removed)} "
              f"(незавершённые расчёты и три завершённых при Q = 1.5 %, v = 250 м/мин); "
              f"стадия twod runs_v2.csv посчитана на наборе без них "
              f"(fem_row_2d.py --save-field2d)")
    suite = build_split_suite(f.proc)
    if a.split not in suite:
        raise SystemExit(f"нет сплита {a.split!r}; есть: {', '.join(sorted(suite))}")
    sp = suite[a.split]
    tr = np.asarray(sp.train_sets, dtype=int)
    te = np.asarray(sp.test_sets, dtype=int)

    if a.hp:
        with open(a.hp, encoding="utf-8") as fh:
            hp = json.load(fh).get(a.family, {})
    else:
        hp = dict(HP_TWOD[a.family])
    cfg = Config2D(family=a.family, seed=a.seed, max_epochs=a.max_epochs,
                   patience=a.patience, **hp)
    print(f"{a.family} / {a.split} / сид {a.seed}: обучение {len(tr)}, тест {len(te)}, "
          f"hp {hp}")

    t0 = time.time()
    T = Trainer2D(cfg)
    res = T.fit(f.proc[tr], f.y[tr], f.r[tr], f.z[tr], f.r_phys[tr], f.z_phys[tr])
    secs = time.time() - t0
    gstd = f.y.reshape(-1, 4).std(axis=0)
    m = T.evaluate(f.proc[te], f.y[te], f.r[te], f.z[te], global_std=gstd)
    pred = T.predict(f.proc[te], f.r[te], f.z[te])

    np.savez_compressed(
        a.out, y_true=f.y[te], y_pred=pred, test_sets=te, proc=f.proc[te],
        r=f.r[te], z=f.z[te], family=np.array(a.family), split=np.array(a.split),
        seed=np.array(a.seed), n_train=np.array(len(tr)), epochs=np.array(res.epochs_run),
        macro_rmse=np.array(m["macro_rmse"]), hp=np.array(json.dumps(hp)))
    print(f"эпох {res.epochs_run}, {secs / 60:.1f} мин; macro-RMSE на тесте "
          f"{m['macro_rmse']:.3f} МПа")
    rec = recorded_run(a.runs, a.family, a.split, a.seed)
    if rec is not None and int(float(rec["n_test"])) == len(te):
        print(f"в {os.path.basename(a.runs)} для той же конфигурации: macro-RMSE "
              f"{float(rec['macro_rmse']):.3f} МПа, эпох {rec['epochs']}")
    print(f"записано: {a.out}")


if __name__ == "__main__":
    main()
